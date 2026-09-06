//! Incremental decode over a plan: operand residency plus a KV cache.
//!
//! A [`DecodeSession`] loads every operand **once** — in the format the
//! backend declares — and then advances one token per [`step`], feeding
//! the backend's [`attention_step`] against a per-layer K/V
//! continuation state ([`KvState`], default [`RowKvState`], or a
//! caller-owned provider via [`with_kv_state`](DecodeSession::with_kv_state)).
//! Each step therefore computes exactly one position through the
//! whole stack instead of re-running the forward over the grown
//! sequence, and every weight keeps a stable address for the session's
//! lifetime, which is what lets a pointer-keyed device buffer cache
//! hold the model resident.
//!
//! **This is the second traversal in the executor, and it is pinned to
//! the first.** [`execute_plan_streaming`](super::execute_plan_streaming)
//! remains the batch traversal (parallel positions, streamed planes,
//! resume); this session realises the same program one position at a
//! time. The two share the operand loaders and call construction
//! ([`AttentionOperands`]), and the decode-vs-batch parity tests assert
//! their outputs agree per backend — a change that moves one without
//! the other is a bug by definition.
//!
//! [`step`]: DecodeSession::step
//! [`attention_step`]: super::backend::PlanBackend::attention_step

use std::borrow::Cow;

use super::attention_residual::{self, BoundaryPhase};
use super::backend::{AttentionStepCall, NormCall, PlanBackend};
use super::hyper_connection::{self, Bundle, Mutation, SiteReduction};
use super::kv::{KvState, RowKvState};
use super::observe::{
    AttnResBoundaryRecord, AttnResSiteRecord, HcSite, HcSiteRecord, InputSite, NoopObserver,
    StepEvent, StepObserver,
};
use super::operands::OperandSource;
use super::prepared::{
    ExecutionSlice, PreparedAttentionResidual, PreparedAttnResSite, PreparedHcSite,
    PreparedOperands,
};
use crate::error::VindexError;
use larql_models::config::HyperConnection;

use super::super::ComponentOpPlan;

/// Who holds the session's operands: an image the caller prepared —
/// typically at model lifetime, shared by every request on that model —
/// or one this session lowered for itself.
///
/// The borrowed arm is the point of operand residency; the owned arm
/// keeps the `new(plan, source, backend)` constructors working for
/// callers that legitimately want a one-shot session.
enum OperandsSlot<'a> {
    /// Boxed so the slot stays pointer-sized: the borrowed arm is the
    /// hot one (a request over a model-lifetime image), and the owned
    /// arm is a one-shot session that can afford the indirection.
    Owned(Box<PreparedOperands>),
    Borrowed(&'a PreparedOperands),
}

impl OperandsSlot<'_> {
    fn get(&self) -> &PreparedOperands {
        match self {
            OperandsSlot::Owned(ops) => ops,
            OperandsSlot::Borrowed(ops) => ops,
        }
    }
}

/// Who holds the session's continuation state (VI3-INF-2): the default
/// [`RowKvState`] owned in place, or a caller's provider borrowed for
/// the session's lifetime so the state outlives the session.
enum KvSlot<'a> {
    Owned(RowKvState),
    Borrowed(&'a mut dyn KvState),
}

impl KvSlot<'_> {
    fn state(&self) -> &dyn KvState {
        match self {
            KvSlot::Owned(state) => state,
            KvSlot::Borrowed(state) => &**state,
        }
    }

    fn state_mut(&mut self) -> &mut dyn KvState {
        match self {
            KvSlot::Owned(state) => state,
            KvSlot::Borrowed(state) => &mut **state,
        }
    }
}

/// What one decode step produces.
pub struct StepOutput {
    /// Logits for the position just consumed, when the plan carries an
    /// output head.
    pub logits: Option<Vec<f32>>,
}

/// How a step enters the stack: an ordinary token, or — for the wave-19a
/// witness — a bundle handed straight to the first executed layer, the
/// decode form of the layer-range contract.
enum Entry {
    Token(u32),
    #[cfg(test)]
    Bundle(Bundle),
}

/// The residual carrier through the layer loop (wave 19a).
///
/// One `[hidden]` vector on every component before hyper-connections,
/// and a typed [`Bundle`] on a hyper-connected one. The two arms share
/// every operator: a site REDUCES the bundle to the `[hidden]` vector the
/// operator consumes and UPDATES the bundle from the operator's output,
/// which on the single stream is the residual add. `hidden` means the
/// same thing on both arms; the stream count is a different dimension
/// and never widens it.
enum Carrier {
    Single(Vec<f32>),
    Bundle(Bundle),
    /// One prefix vector plus an ordered history of block-boundary
    /// snapshots (K3-ATTNRES-1, 2a). Not a `Bundle` with a growing
    /// stream count: a bundle's streams are interchangeable parallel
    /// residuals whose count is DECLARED once, and these are ordered
    /// historical states of the prefix produced by boundary EVENTS,
    /// whose count is a function of depth.
    History(attention_residual::History),
}

/// What entering one site produced: the `[hidden]` vector the branch
/// sees, and the reduction the update needs afterwards — `None` on the
/// single stream, where the update is the residual add.
struct SiteEntry {
    branch_input: Vec<f32>,
    reduction: Option<SiteReduction>,
    /// The attention-residual site's entry state, `None` on every other
    /// topology. Distinct from `reduction` above because the two
    /// topologies' reductions produce different objects — a Sinkhorn
    /// split against a distribution over candidates — and a shared field
    /// would have been a union with two empty halves.
    attn_res: Option<AttnResEntry>,
}

/// What entering one attention-residual site produced.
///
/// `reduction` is `None` exactly where the reference does not reduce:
/// layer 0's attention site, whose snapshot set is empty. That absence
/// travels to `leave_site`, which then emits NO record — and the missing
/// record is the observation, because the oracle measured this defect at
/// a divergence of exactly zero.
struct AttnResEntry {
    reduction: Option<attention_residual::Reduction>,
    prefix_before: Vec<f32>,
    snapshot_count_before: usize,
    candidate_count: usize,
}

/// One step's result before it is narrowed to [`StepOutput`]. The
/// witness fields exist only under test, so the production step keeps
/// nothing it does not return.
pub(super) struct StepRun {
    pub(super) logits: Option<Vec<f32>>,
    /// The `[hidden]` vector the exit reduced to, before the final norm.
    #[cfg(test)]
    pub(super) exit: Option<Vec<f32>>,
    /// The bundle after the last executed layer, on a hyper-connected
    /// component.
    #[cfg(test)]
    pub(super) bundle: Option<Bundle>,
}

/// Incremental executor over one component plan (see module docs).
pub struct DecodeSession<'a, B: PlanBackend> {
    plan: &'a ComponentOpPlan,
    backend: &'a B,
    ops: OperandsSlot<'a>,
    kv: KvSlot<'a>,
}

impl<'a, B: PlanBackend> DecodeSession<'a, B> {
    /// Load every operand the plan consumes, once, in the backend's
    /// declared weight format. The embedding table stays f32 — it is a
    /// row lookup, not matrix traffic. Continuation state is the
    /// default in-place [`RowKvState`].
    pub fn new<'s>(
        plan: &'a ComponentOpPlan,
        store: impl Into<OperandSource<'s>>,
        backend: &'a B,
    ) -> Result<Self, VindexError> {
        Self::build(
            plan,
            store.into(),
            backend,
            KvSlot::Owned(RowKvState::default()),
        )
    }

    /// Like [`new`](Self::new), but the caller provides — and keeps
    /// owning — the continuation state, so K/V policy composes outside
    /// the executor and the state outlives the session. The session
    /// continues from `kv.position()`: an empty provider starts a
    /// fresh sequence, and one populated by
    /// [`prefill_plan`](super::prefill_plan) — or by an earlier
    /// session — resumes exactly where it left off. The provider is
    /// the *only* position authority; no separate start argument
    /// exists to disagree with it.
    pub fn with_kv_state<'s>(
        plan: &'a ComponentOpPlan,
        store: impl Into<OperandSource<'s>>,
        backend: &'a B,
        kv: &'a mut dyn KvState,
    ) -> Result<Self, VindexError> {
        Self::build(plan, store.into(), backend, KvSlot::Borrowed(kv))
    }

    /// Open a session over operands the caller already prepared —
    /// typically once, at model lifetime, and shared by every request
    /// on that model. Nothing is loaded here.
    pub fn over_prepared(
        plan: &'a ComponentOpPlan,
        ops: &'a PreparedOperands,
        backend: &'a B,
        kv: &'a mut dyn KvState,
    ) -> Result<Self, VindexError> {
        Self::assemble(
            plan,
            OperandsSlot::Borrowed(ops),
            backend,
            KvSlot::Borrowed(kv),
        )
    }

    fn build(
        plan: &'a ComponentOpPlan,
        store: OperandSource<'_>,
        backend: &'a B,
        kv: KvSlot<'a>,
    ) -> Result<Self, VindexError> {
        let ops = PreparedOperands::load(plan, store, backend, ExecutionSlice::Full)?;
        Self::assemble(plan, OperandsSlot::Owned(Box::new(ops)), backend, kv)
    }

    fn assemble(
        plan: &'a ComponentOpPlan,
        ops: OperandsSlot<'a>,
        backend: &'a B,
        mut kv: KvSlot<'a>,
    ) -> Result<Self, VindexError> {
        ops.get().ensure_providers_in(ops.get().registry())?;
        // The FULL continuation geometry, KV and recurrent alike.
        // `plan_kv_geometry` is the KV-only adapter and refuses a hybrid
        // plan; a session over one needs both forms announced.
        kv.state_mut().prepare_continuation(
            &super::continuation::plan_continuation_geometry(plan).map_err(VindexError::Parse)?,
        )?;
        Ok(Self {
            plan,
            backend,
            ops,
            kv,
        })
    }

    /// What this session's operands actually occupy, by site and
    /// representation — see [`PreparedOperands::residency_census`].
    pub fn residency_census(&self) -> super::prepared::ResidencyCensus {
        self.ops.get().residency_census()
    }

    /// The realization pinned for every operand this session executes,
    /// with its candidates, reason and declared residency.
    pub fn realizations(&self) -> &[super::realization::RealizationRecord] {
        self.ops.get().realizations()
    }

    /// Where this session's operand allocations landed.
    /// The prepared image's mappings and their resident pages, now.
    pub fn mapped_residency(&self) -> super::prepared::MappedResidency {
        self.ops.get().mapped_residency()
    }

    pub fn allocation_census(&self) -> super::prepared::AllocationCensus {
        self.ops.get().allocation_census()
    }

    /// Positions consumed so far — read from the continuation state,
    /// which is the single position authority (VI3-INF-3).
    pub fn position(&self) -> usize {
        self.kv.state().position()
    }

    /// Advance one token: embed it, run it through every layer against
    /// the cached K/V, and return the head's logits for this position.
    ///
    /// Operation ordering mirrors the batch traversal exactly — the
    /// decode-vs-batch parity tests are the guarantee.
    pub fn step(&mut self, token: u32) -> Result<StepOutput, VindexError> {
        self.step_observed(token, &mut NoopObserver)
    }

    /// **CPU-7C.** Advance this continuation through exactly `tokens`, in
    /// order, letting eligible projections execute across those positions
    /// together.
    ///
    /// This is a VERIFICATION primitive, not a generator. It consumes
    /// token ids the caller already has and never samples position `t+1`
    /// from position `t`'s logits — doing so would make the traversal
    /// autoregressive again and destroy the very parallelism it exists to
    /// expose. Given proposed tokens, it evaluates the continuation
    /// through all of them; deciding which to accept is the caller's.
    ///
    /// Semantically it must be indistinguishable from calling
    /// [`step`](Self::step) once per token: the same logits, and the same
    /// continuation state left behind — recurrent buffers, convolution
    /// history, K/V rows and position alike. "Same logits" alone is not
    /// the property, because a wrong recurrent state produces correct
    /// logits for these positions and diverges only on the NEXT one,
    /// which is what the follow-on-step parity gate is for.
    ///
    /// Returns the last position's logits, matching [`step`]. Per-position
    /// logits need the streaming sink and are owed to CPU-7D, which is the
    /// first thing that actually needs them.
    pub fn step_many(&mut self, tokens: &[u32]) -> Result<StepOutput, VindexError> {
        if tokens.is_empty() {
            return Err(VindexError::Parse(
                "step_many advances a continuation through supplied tokens and was given none;                  an empty advance is a caller bug, not a no-op"
                    .to_string(),
            ));
        }
        let Self {
            plan,
            backend,
            ops,
            kv,
        } = self;
        let ops = ops.get();
        let hidden = ops.hidden();
        let embed_table = ops.embed_table().ok_or_else(|| {
            VindexError::Parse(
                "this prepared image carries no embedding table — a layer-range slice consumes                  hidden states, not token ids"
                    .to_string(),
            )
        })?;
        // Checked for EVERY token before any state moves. A bad id found
        // half way through would leave the continuation advanced by some
        // of the batch, which is a corrupted session rather than a failed
        // call.
        for token in tokens {
            if (*token as usize + 1) * hidden > embed_table.len() {
                return Err(VindexError::Parse(format!(
                    "token id {token} is outside the embedding table",
                )));
            }
        }
        let state = kv.state_mut();
        let base = state.position();
        let out = super::traverse(
            plan,
            ops,
            tokens,
            *backend,
            None,
            &mut |_| Ok(()),
            Some(state),
            Mutation::None,
        )?;
        // The provider is the position authority, and the traversal does
        // not move it — the same contract `prefill_prepared` keeps.
        state.set_position(base + tokens.len());
        Ok(StepOutput { logits: out.logits })
    }

    /// [`step`](Self::step) with a subscriber on the step's operation
    /// boundaries (LQL-2 TRACE). This IS the step — `step()` calls it
    /// with [`NoopObserver`] — so observation can never fork the
    /// semantics; the observed-vs-unobserved parity gate pins it.
    pub fn step_observed(
        &mut self,
        token: u32,
        observer: &mut dyn StepObserver,
    ) -> Result<StepOutput, VindexError> {
        let run = self.run(Entry::Token(token), observer, Mutation::None)?;
        Ok(StepOutput { logits: run.logits })
    }

    /// The step under a deliberate defect — the wave-19a negative
    /// controls. Test-only: production has exactly one way in, and it
    /// passes [`Mutation::None`].
    #[cfg(test)]
    pub(super) fn step_mutated(
        &mut self,
        token: u32,
        observer: &mut dyn StepObserver,
        mutation: Mutation,
    ) -> Result<StepRun, VindexError> {
        self.run(Entry::Token(token), observer, mutation)
    }

    /// The step entered with a bundle at the first executed layer instead
    /// of a token — how the witness hands the oracle's own state to a
    /// site. Test-only, and hyper-connected components only.
    #[cfg(test)]
    pub(super) fn step_from_bundle(
        &mut self,
        bundle: Bundle,
        observer: &mut dyn StepObserver,
        mutation: Mutation,
    ) -> Result<StepRun, VindexError> {
        self.run(Entry::Bundle(bundle), observer, mutation)
    }

    /// The one decode step. Every public and test entry above is this.
    fn run(
        &mut self,
        entry: Entry,
        observer: &mut dyn StepObserver,
        mutation: Mutation,
    ) -> Result<StepRun, VindexError> {
        let ops = self.ops.get();
        let hidden = ops.hidden();
        let topology = ops.hyper_connection();
        // The ONE declared fact the attention-residual schedule needs:
        // which layers carry the boundary event. `None` on every other
        // topology.
        let block_size = ops.attention_residual_block_size();
        let position = self.kv.state().position();
        let mut carrier = match entry {
            Entry::Token(token) => {
                let embedding = self
                    .plan
                    .embedding
                    .as_ref()
                    .expect("session construction required an embedding op");
                let embed_table = ops.embed_table().ok_or_else(|| {
                    VindexError::Parse(
                        "this prepared image carries no embedding table — a layer-range slice \
                         consumes hidden states, not token ids"
                            .to_string(),
                    )
                })?;
                if (token as usize + 1) * hidden > embed_table.len() {
                    return Err(VindexError::Parse(format!(
                        "token id {token} is outside the embedding table",
                    )));
                }
                let mut h = self
                    .backend
                    .embed(embed_table, hidden, token, embedding.scale);
                if let Some(norm) = embedding.norm {
                    h = self.backend.norm(NormCall {
                        kind: norm.kind,
                        x: &h,
                        weight: &[],
                        weight_offset: 0.0,
                        eps: norm.eps,
                    });
                }
                observer.event(StepEvent::Embedded { position });
                // The embedding enters a hyper-connected stack replicated
                // into every stream (`Transformer.forward`'s repeat) —
                // after its scale and norm, which belong to the lookup.
                match (topology, block_size) {
                    (Some(hc), _) => Carrier::Bundle(Bundle::replicate(&h, hc.streams)),
                    // The embedding enters an attention-residual stack as
                    // the FIRST PREFIX, with an EMPTY history — the
                    // reference starts `block_residual` at
                    // `new_zeros(tokens, 0, hidden)` and nothing is
                    // replicated. That emptiness is what gives layer 0's
                    // attention site nothing to read.
                    (None, Some(_)) => Carrier::History(attention_residual::History::new(h)),
                    (None, None) => Carrier::Single(h),
                }
            }
            #[cfg(test)]
            Entry::Bundle(bundle) => {
                let Some(hc) = topology else {
                    return Err(VindexError::Parse(
                        "a bundle entered a single-stream component".to_string(),
                    ));
                };
                if bundle.streams() != hc.streams || bundle.hidden() != hidden {
                    return Err(VindexError::Parse(format!(
                        "the entering bundle is {} x {}; the component is {} x {}",
                        bundle.streams(),
                        bundle.hidden(),
                        hc.streams,
                        hidden
                    )));
                }
                Carrier::Bundle(bundle)
            }
        };

        let first = ops.first_layer();
        for (offset, state) in ops.layers().iter().enumerate() {
            let index = first + offset;
            let layer = &self.plan.layers[index];
            // ── Attention site ──
            //
            // On a bundle the site reduces first: the ordinary operator
            // consumes the reduced `[hidden]` vector, and so does the
            // pre-attention norm — the norm normalises what the branch
            // sees, never the bundle. On the single stream the "reduced"
            // vector is the residual itself, as it always was.
            // The entering prefix state, captured BEFORE the site reads
            // anything: it is what the boundary event snapshots, and
            // capturing it later would snapshot whatever the site had
            // already done to the carrier.
            let entering_prefix: Option<Vec<f32>> = match &carrier {
                Carrier::History(history) => history.prefix().map(<[f32]>::to_vec),
                _ => None,
            };
            let boundary = match (block_size, &carrier) {
                (Some(size), Carrier::History(_)) => {
                    attention_residual::is_block_boundary(index, size)
                }
                _ => false,
            };
            let boundary_context = SiteContext {
                layer: index,
                site: HcSite::Attention,
                position,
                mutation,
            };
            // Phase one of three: before the attention site reads. The
            // reference does NOTHING here; one control moves the
            // snapshot to this point so the reduction below sees the new
            // set instead of the old one.
            if boundary {
                let entering = entering_prefix.as_deref().unwrap_or_default();
                boundary_event(
                    &mut carrier,
                    BoundaryPhase::BeforeAttentionReduce,
                    entering,
                    entering,
                    boundary_context,
                    observer,
                );
            }
            let attention_site = enter(
                &carrier,
                state.hyper_connection.as_ref().map(|hc| &hc.attention),
                state.attention_residual.as_ref(),
                HcSite::Attention,
                topology,
                layer.declared_norm_eps,
                index,
                mutation,
            )?;
            // Phase two: between the attention site's reduction and the
            // attention branch. **Where the reference puts it** — the
            // point wave 19's two-point seam cannot express.
            if boundary {
                let entering = entering_prefix.as_deref().unwrap_or_default();
                boundary_event(
                    &mut carrier,
                    BoundaryPhase::AfterAttentionReduce,
                    entering,
                    &attention_site.branch_input,
                    boundary_context,
                    observer,
                );
            }
            // Under post-norm placement there is no pre-attention norm and
            // the sublayer reads the RAW vector — the norm applies to its
            // output, below, before the update. Cloning rather than
            // normalising by an identity keeps the absent op absent.
            let norm_input: Cow<'_, [f32]> = match (mutation, &carrier) {
                (Mutation::PreNormOnStreamZero, Carrier::Bundle(x)) => Cow::Borrowed(x.stream(0)),
                (Mutation::PreNormOnStreamMean, Carrier::Bundle(x)) => Cow::Owned(x.stream_mean()),
                _ => Cow::Borrowed(&attention_site.branch_input),
            };
            let inputs = [match &state.pre_attention {
                Some(norm) => norm.apply(self.backend, &norm_input),
                None => norm_input.into_owned(),
            }];
            // The sensitivity tap from main, kept ahead of the operator
            // dispatch: it observes the attention INPUT, which both
            // operators read, so it belongs to neither branch.
            observer.operand_input(index, InputSite::Attention, inputs[0].as_slice());
            // One position, either operator. A recurrence appends no KV
            // row — it rewrites its own buffers in place, which is only
            // correct because those buffers are DURABLE: the convolution
            // history spans the step boundary, and a single-position call
            // that reconstructed it from the batch would see a window of
            // one. That is the whole point of QW-3.6a.
            let _attention_stage = super::stages::stage(super::stages::Stage::Attention);
            let raw_attn = match &state.attention {
                super::prepared::PreparedAttention::GatedDelta(delta) => {
                    let recurrent = self.kv.state_mut().recurrent_state(index)?;
                    let projector = self.backend.dense_projector();
                    let mut planes = super::gated_delta::layer_forward_with(
                        &delta.op,
                        &delta.weights()?,
                        &inputs,
                        recurrent,
                        super::gated_delta::Mutation::None,
                        projector,
                    );
                    planes.output.remove(0)
                }
                super::prepared::PreparedAttention::Mamba2(mixer) => {
                    let recurrent = self.kv.state_mut().recurrent_state(index)?;
                    let projector = self.backend.dense_projector();
                    let mut planes = super::mamba2::layer_forward_with(
                        &mixer.op,
                        &mixer.weights()?,
                        &inputs,
                        recurrent,
                        projector,
                    );
                    planes.output.remove(0)
                }
                super::prepared::PreparedAttention::Kda(ops) => {
                    let recurrent = self.kv.state_mut().recurrent_state(index)?;
                    let projector = self.backend.dense_projector();
                    let mut planes = super::kda::layer_forward_with(
                        &super::kda::BackendKdaProjections(projector),
                        &inputs[0],
                        inputs[0].len(),
                        ops.weights()?,
                        ops.op.geometry(),
                        recurrent,
                        super::kda::Mutation::None,
                    );
                    // One position in, one position out: the planes are
                    // flat and this call contributed exactly `hidden`.
                    planes.output.truncate(inputs[0].len());
                    std::mem::take(&mut planes.output)
                }
                super::prepared::PreparedAttention::Mla(ops) => {
                    // MLA appends its own position to the latent cache
                    // and reads the whole prefix back — the cache IS the
                    // continuation, and the provider owns it across the
                    // step boundary exactly as it owns KV rows.
                    let projector = self.backend.dense_projector();
                    let hidden = inputs[0].len();
                    let weights = ops.weights()?;
                    let geometry = ops.op.geometry();
                    let latent = self.kv.state_mut().latent_state(index)?;
                    super::mla::mla_forward_with(
                        projector,
                        &inputs[0],
                        hidden,
                        weights,
                        geometry,
                        latent,
                        super::mla::Mutation::None,
                    )
                    .output
                }
                super::prepared::PreparedAttention::ConvQkv(ops) => {
                    // Two regions, borrowed in phases: past rows copied
                    // out, conv history advanced by the forward, the
                    // step's row appended after — same choreography as
                    // the batch path, at one position.
                    let past_keys: Vec<Vec<f32>> = self.kv.state().keys(index).to_vec();
                    let past_values: Vec<Vec<f32>> = self.kv.state().values(index).to_vec();
                    let base = position;
                    let recurrent = self.kv.state_mut().recurrent_state(index)?;
                    let projector = self.backend.dense_projector();
                    let mut planes = super::conv_qkv::layer_forward_with(
                        &ops.op,
                        &ops.weights()?,
                        &inputs,
                        recurrent,
                        &past_keys,
                        &past_values,
                        base,
                        projector,
                    );
                    let key = planes.keys.remove(0);
                    let value = planes.values.remove(0);
                    self.kv.state_mut().append(index, key, value);
                    planes.output.remove(0)
                }
                super::prepared::PreparedAttention::Softmax(sops) => {
                    let call = sops.call(
                        layer
                            .attention
                            .softmax()
                            .expect("prepared softmax operands imply a softmax op"),
                        &inputs,
                        layer.declared_norm_eps,
                        hidden,
                    );
                    let _site = super::cpu::ledger::in_site(super::cpu::ledger::Site::Attention);
                    let out = self.backend.attention_step(AttentionStepCall {
                        op: call,
                        position,
                        keys: self.kv.state().keys(index),
                        values: self.kv.state().values(index),
                    })?;
                    self.kv.state_mut().append(index, out.key, out.value);
                    out.output
                }
            };
            drop(_attention_stage);
            let mut attn_out = match &state.post_attention {
                Some(norm) => norm.apply(self.backend, &raw_attn),
                None => raw_attn,
            };
            super::scale_residual_delta(layer.residual_scale, &mut attn_out);
            leave_site(
                self.backend,
                &mut carrier,
                attn_out,
                attention_site.reduction,
                attention_site.attn_res,
                SiteContext {
                    layer: index,
                    site: HcSite::Attention,
                    position,
                    mutation,
                },
                observer,
            )?;
            // Phase three: after the attention branch. The reference does
            // NOTHING here; one control moves the snapshot to this point
            // so it carries the post-attention prefix instead of the
            // entering state.
            if boundary {
                let entering = entering_prefix.as_deref().unwrap_or_default();
                boundary_event(
                    &mut carrier,
                    BoundaryPhase::AfterAttentionBranch,
                    entering,
                    entering,
                    boundary_context,
                    observer,
                );
            }
            observer.event(StepEvent::AttentionDone { layer: index });

            // A mixer-only (Mamba2) layer carries no FFN program: its one
            // residual update happened above, and running a fabricated FFN
            // stage here would be the schema-6 fabrication re-enacted at
            // execution time. Presence follows the program here too.
            //
            // The discriminator is the FFN OP, never its pre-norm. Under
            // post-norm placement the FFN reads the raw residual and has
            // no pre-norm at all, and gating the whole sublayer on that
            // norm's presence silently ran OLMo-2 as attention-only —
            // caught by real-checkpoint parity, not by any synthetic
            // fixture, because a stack missing every FFN still produces
            // fluent-looking planes.
            if let (Some(ffn), Some(ffn_op)) = (&state.ffn, &layer.ffn) {
                let ffn_site = enter(
                    &carrier,
                    state.hyper_connection.as_ref().map(|hc| &hc.ffn),
                    state.attention_residual.as_ref(),
                    HcSite::Ffn,
                    topology,
                    layer.declared_norm_eps,
                    index,
                    mutation,
                )?;
                let normed = match &state.pre_ffn {
                    Some(pre_ffn) => pre_ffn.apply(self.backend, &ffn_site.branch_input),
                    None => ffn_site.branch_input.clone(),
                };
                observer.operand_input(index, InputSite::Ffn, normed.as_slice());
                // The FFN's raw residual — what a hybrid's router and
                // expert pre-norm read — is the vector the branch sees:
                // the reduced one on a bundle, never a stream of it.
                let residual: Cow<'_, [f32]> = match (mutation, &carrier) {
                    (Mutation::HybridResidualFromStreamZero, Carrier::Bundle(x)) => {
                        Cow::Borrowed(x.stream(0))
                    }
                    (Mutation::HybridResidualFromStreamMean, Carrier::Bundle(x)) => {
                        Cow::Owned(x.stream_mean())
                    }
                    _ => Cow::Borrowed(&ffn_site.branch_input),
                };
                let _site = super::cpu::ledger::in_site(super::cpu::ledger::Site::Ffn);
                let ffn_out =
                    ffn.apply_from_residual(ffn_op, self.backend, &residual, &normed, hidden)?;
                drop(residual);
                observer.operand_input(index, InputSite::FfnOutput, ffn_out.as_slice());
                let mut ffn_out = match &state.post_ffn {
                    Some(norm) => norm.apply(self.backend, &ffn_out),
                    None => ffn_out,
                };
                super::scale_residual_delta(layer.residual_scale, &mut ffn_out);
                leave_site(
                    self.backend,
                    &mut carrier,
                    ffn_out,
                    ffn_site.reduction,
                    ffn_site.attn_res,
                    SiteContext {
                        layer: index,
                        site: HcSite::Ffn,
                        position,
                        mutation,
                    },
                    observer,
                )?;
                if let Some(scale) = state.layer_scale {
                    match &mut carrier {
                        Carrier::Single(h) => self.backend.scale_row(h, scale),
                        // Preparation refuses this combination; reaching
                        // it is an executor bug, not a model.
                        // Preparation refuses both of these; reaching
                        // one is an executor bug, not a model.
                        Carrier::Bundle(_) | Carrier::History(_) => {
                            return Err(VindexError::Parse(format!(
                                "layer {index} carries a layer scale on a component whose \
                                 residual is not one vector; preparation should have refused it"
                            )))
                        }
                    }
                }
            }
            observer.event(StepEvent::FfnDone { layer: index });
        }

        // ── The exit ──
        //
        // A bundle leaves the stack through the head's OWN reduction
        // (a different operation from a site's — no Sinkhorn) when the
        // image carries one, and a whole-stack image of a hyper-connected
        // component always does (preparation refuses otherwise). A
        // layer-range image has no exit: the bundle after its last layer
        // IS its output, and it produces no logits.
        let exit = match carrier {
            Carrier::Single(h) => Exit {
                hidden: Some(h),
                #[cfg(test)]
                bundle: None,
            },
            // The attention-residual exit: the same reduction a site
            // runs, once, over the WHOLE snapshot history plus the
            // prefix, before the final norm. Required by the
            // declaration — preparation refuses a whole-stack image
            // without one — so a layer-range image is the only way to
            // reach the `None` arm, and its output is the history it
            // hands on rather than a `[hidden]` vector.
            Carrier::History(history) => {
                let reduced = match ops.attention_residual_exit() {
                    Some(exit) if mutation != Mutation::AttnResExitSkipped => {
                        let pair = match mutation {
                            // The control that makes "the SHIPPED pair" a
                            // claim: reduce with a layer's pair instead.
                            Mutation::AttnResExitUsesALayerPair => ops.layers()[0]
                                .attention_residual
                                .as_ref()
                                .map(|sites| sites.ffn.pair())
                                .unwrap_or_else(|| exit.pair()),
                            _ => exit.pair(),
                        };
                        let reduction =
                            attention_residual::reduce(&history, pair, exit.norm_eps(), mutation)?;
                        observer.attention_residual_site(AttnResSiteRecord {
                            layer: self.plan.layers.len(),
                            site: HcSite::Ffn,
                            position,
                            candidate_count: history.candidate_count(),
                            snapshot_count_before: history.snapshot_count(),
                            probs: &reduction.probs,
                            mixed_vector: &reduction.mixed,
                            prefix_before: history.prefix().unwrap_or(&reduction.mixed),
                            prefix_after: &reduction.mixed,
                        });
                        Some(reduction.mixed)
                    }
                    Some(_) => Some(history.clone().into_prefix()?),
                    None => None,
                };
                Exit {
                    hidden: reduced,
                    #[cfg(test)]
                    bundle: None,
                }
            }
            Carrier::Bundle(x) => {
                let reduced = match (ops.hyper_connection_head(), topology) {
                    (Some(head), Some(hc)) => Some(hyper_connection::head_reduce(
                        x.as_flat(),
                        x.streams(),
                        hidden,
                        &head.weights(),
                        head.norm_eps(),
                        hc.sinkhorn_eps,
                    )),
                    _ => None,
                };
                Exit {
                    hidden: reduced,
                    #[cfg(test)]
                    bundle: Some(x),
                }
            }
        };
        #[cfg(test)]
        let witness_exit = exit.hidden.clone();
        let logits = match exit.hidden {
            Some(exit_hidden) => {
                let final_hidden = match ops.final_norm() {
                    Some(norm) => norm.apply(self.backend, &exit_hidden),
                    None => exit_hidden,
                };
                match ops.output() {
                    Some((op, weight)) => Some(self.backend.output_head(
                        weight.slice(),
                        op.projection.shape[0],
                        hidden,
                        &final_hidden,
                        op.multiplier,
                        op.softcapping,
                    )?),
                    None => None,
                }
            }
            None => {
                if ops.final_norm().is_some() || ops.output().is_some() {
                    return Err(VindexError::Parse(
                        "a hyper-connected bundle reached a whole-stack exit with no head \
                         reduction; preparation should have refused the image"
                            .to_string(),
                    ));
                }
                None
            }
        };
        if let Some(logits) = &logits {
            observer.event(StepEvent::Logits {
                vocab: logits.len(),
            });
        }
        self.kv.state_mut().set_position(position + 1);
        Ok(StepRun {
            logits,
            #[cfg(test)]
            exit: witness_exit,
            #[cfg(test)]
            bundle: exit.bundle,
        })
    }
}

/// What the layer loop leaves for the exit.
struct Exit {
    hidden: Option<Vec<f32>>,
    #[cfg(test)]
    bundle: Option<Bundle>,
}

/// Which site a record belongs to, and under which control.
#[derive(Clone, Copy)]
struct SiteContext {
    layer: usize,
    site: HcSite,
    position: usize,
    mutation: Mutation,
}

/// Enter one site: produce the `[hidden]` vector the branch consumes.
///
/// The carrier, the layer's site operands and the topology must agree
/// three ways — preparation guarantees it, and a disagreement here is
/// an executor bug rather than a model to run anyway.
fn enter_site(
    carrier: &Carrier,
    site: Option<&PreparedHcSite>,
    topology: Option<HyperConnection>,
    norm_eps: f64,
    mutation: Mutation,
) -> Result<SiteEntry, VindexError> {
    match (carrier, site, topology) {
        (Carrier::Single(h), None, None) => Ok(SiteEntry {
            branch_input: h.clone(),
            reduction: None,
            attn_res: None,
        }),
        (Carrier::Bundle(x), Some(site), Some(hc)) => {
            if mutation == Mutation::BypassComposition {
                // The control: stream 0 stands in for the whole bundle
                // and no split exists to report.
                return Ok(SiteEntry {
                    branch_input: x.stream(0).to_vec(),
                    reduction: None,
                    attn_res: None,
                });
            }
            let reduction = hyper_connection::reduce(x, &site.weights(), hc, norm_eps, mutation);
            Ok(SiteEntry {
                branch_input: reduction.reduced.clone(),
                reduction: Some(reduction),
                attn_res: None,
            })
        }
        _ => Err(VindexError::Parse(
            "the residual carrier, the layer's hyper-connection sites and the declared topology \
             disagree; preparation should have refused the image"
                .to_string(),
        )),
    }
}

/// Enter one site, on whichever residual programme the carrier runs.
///
/// One call site per sublayer in the layer loop, so the loop reads the
/// same on every topology and the difference lives here.
#[allow(clippy::too_many_arguments)]
fn enter(
    carrier: &Carrier,
    hc_site: Option<&PreparedHcSite>,
    attn_res: Option<&PreparedAttentionResidual>,
    which: HcSite,
    topology: Option<HyperConnection>,
    norm_eps: f64,
    layer: usize,
    mutation: Mutation,
) -> Result<SiteEntry, VindexError> {
    match (carrier, attn_res) {
        (Carrier::History(history), Some(sites)) => {
            let site = match which {
                HcSite::Attention => &sites.attention,
                HcSite::Ffn => &sites.ffn,
            };
            enter_attention_residual_site(history, site, which, norm_eps, layer, mutation)
        }
        (Carrier::History(_), None) => Err(VindexError::Parse(format!(
            "layer {layer} carries a residual history and no attention-residual sites; \
             preparation should have refused the image"
        ))),
        _ => enter_site(carrier, hc_site, topology, norm_eps, mutation),
    }
}

/// Enter one attention-residual site.
///
/// The reduction is GUARDED at the attention site and UNCONDITIONAL at
/// the mlp site, which is the reference's own asymmetry
/// (`if block_residual is not None and block_residual.shape[1] > 0`
/// around the first, nothing around the second). Where it does not
/// reduce, the branch input is the prefix itself and no record will be
/// emitted — the reference's guard and a regularised always-run site
/// compute the same vector, so only the absence distinguishes them.
fn enter_attention_residual_site(
    history: &attention_residual::History,
    site: &PreparedAttnResSite,
    which: HcSite,
    norm_eps: f64,
    layer: usize,
    mutation: Mutation,
) -> Result<SiteEntry, VindexError> {
    let prefix = history.prefix().ok_or_else(|| {
        VindexError::Parse(format!(
            "layer {layer} entered a site with no prefix; a boundary reset it and no branch \
             has supplied one"
        ))
    })?;
    let guarded = match which {
        // The reference's guard, and the one control that drops it.
        HcSite::Attention => {
            history.snapshot_count() > 0 || mutation == Mutation::AttnResLayer0AttentionSiteRuns
        }
        // Unconditional in the reference; the control gives it the
        // attention site's guard, and a second skips it at layer 0.
        HcSite::Ffn => match mutation {
            Mutation::AttnResMlpSiteGuardedOnNonEmpty => history.snapshot_count() > 0,
            Mutation::AttnResMlpSiteSkippedAtLayer0 if layer == 0 => false,
            _ => true,
        },
    };
    let entry = AttnResEntry {
        prefix_before: prefix.to_vec(),
        snapshot_count_before: history.snapshot_count(),
        candidate_count: history.candidate_count(),
        reduction: None,
    };
    if !guarded {
        return Ok(SiteEntry {
            branch_input: prefix.to_vec(),
            reduction: None,
            attn_res: Some(entry),
        });
    }
    let reduction = attention_residual::reduce(history, site.pair(), norm_eps, mutation)?;
    Ok(SiteEntry {
        branch_input: reduction.mixed.clone(),
        reduction: None,
        attn_res: Some(AttnResEntry {
            reduction: Some(reduction),
            ..entry
        }),
    })
}

/// The block-boundary event, offered at each of the three points a
/// snapshot could be taken. Does something only at the point this run's
/// control selects; the reference's point is
/// [`BoundaryPhase::AfterAttentionReduce`].
///
/// The prefix RESET always happens at the reference's point, whatever
/// the snapshot's phase — the reference resets in the same statement it
/// appends, and a control that moved the append must not silently move
/// the reset with it.
fn boundary_event(
    carrier: &mut Carrier,
    phase: BoundaryPhase,
    entering_prefix: &[f32],
    mixed_vector: &[f32],
    context: SiteContext,
    observer: &mut dyn StepObserver,
) {
    let Carrier::History(history) = carrier else {
        return;
    };
    let snapshot_phase = match context.mutation {
        Mutation::AttnResSiteOverNewSnapshots => BoundaryPhase::BeforeAttentionReduce,
        Mutation::AttnResSnapshotAfterAttention => BoundaryPhase::AfterAttentionBranch,
        _ => BoundaryPhase::AfterAttentionReduce,
    };
    if phase == snapshot_phase {
        // WHICH vector is snapshotted is the other thing a control
        // perturbs, and the reference's answer is the ENTERING prefix
        // state — not the attention site's output, and not the
        // post-attention prefix.
        let value: Vec<f32> = match (context.mutation, phase) {
            (Mutation::AttnResSnapshotIsMixedVector, _) => mixed_vector.to_vec(),
            (_, BoundaryPhase::AfterAttentionBranch) => history
                .prefix()
                .map(<[f32]>::to_vec)
                .unwrap_or_else(|| entering_prefix.to_vec()),
            _ => entering_prefix.to_vec(),
        };
        let before = history.snapshot_count();
        history.push_snapshot(value.clone());
        observer.attention_residual_boundary(AttnResBoundaryRecord {
            layer: context.layer,
            position: context.position,
            snapshots_before: before,
            snapshots_after: history.snapshot_count(),
            value: &value,
            entering_prefix,
        });
    }
    if phase == BoundaryPhase::AfterAttentionReduce {
        history.reset_prefix();
    }
}

/// Leave one site: fold the branch's `[hidden]` delta back into the
/// carrier. On the single stream that is the residual add; on a bundle
/// it is stage five, which carries every stream forward through `comb`
/// and scatters the delta by `post` — one operation, not an add — and
/// the observer sees the site's state the moment it exists.
fn leave_site<B: PlanBackend + ?Sized>(
    backend: &B,
    carrier: &mut Carrier,
    delta: Vec<f32>,
    reduction: Option<SiteReduction>,
    attn_res: Option<AttnResEntry>,
    context: SiteContext,
    observer: &mut dyn StepObserver,
) -> Result<(), VindexError> {
    // The attention-residual write, and it is NOT always an add: a
    // boundary reset this prefix, and then the branch's output BECOMES
    // the prefix rather than being added to one. `History::write` is one
    // method for that reason — the reference is one expression, and
    // splitting it would let a caller forget the second arm.
    if let (Carrier::History(history), Some(entry)) = (&mut *carrier, attn_res) {
        history.write(&delta);
        // Emitted only where the reference REDUCED. Layer 0's attention
        // site emits nothing, and that absence is the observation.
        if let Some(reduction) = &entry.reduction {
            let prefix_after = history.prefix().unwrap_or(&delta);
            observer.attention_residual_site(AttnResSiteRecord {
                layer: context.layer,
                site: context.site,
                position: context.position,
                candidate_count: entry.candidate_count,
                snapshot_count_before: entry.snapshot_count_before,
                probs: &reduction.probs,
                mixed_vector: &reduction.mixed,
                prefix_before: &entry.prefix_before,
                prefix_after,
            });
        }
        return Ok(());
    }
    match (carrier, reduction) {
        (Carrier::Single(h), None) => {
            backend.residual_add(h, &delta);
            Ok(())
        }
        (Carrier::Bundle(x), Some(reduction)) => {
            let next = hyper_connection::update(x, &delta, &reduction.split, context.mutation);
            observer.hyper_connection_site(HcSiteRecord {
                layer: context.layer,
                site: context.site,
                position: context.position,
                split: &reduction.split,
                reduced: &reduction.reduced,
                branch_output: &delta,
                bundle_out: &next,
            });
            *x = next;
            Ok(())
        }
        (Carrier::Bundle(x), None) => {
            // The bypass control: the delta lands in stream 0 alone and
            // no record is emitted, exactly what a traversal that never
            // ran the topology would do.
            debug_assert_eq!(context.mutation, Mutation::BypassComposition);
            backend.residual_add(x.stream_mut(0), &delta);
            Ok(())
        }
        // A history carrier reaches this match only if its entry was
        // lost on the way in; the arm above returns before it otherwise.
        (Carrier::History(_), _) => Err(VindexError::Parse(
            "an attention-residual carrier left a site with no site entry; the traversal built \
             one at every site it entered"
                .to_string(),
        )),
        (Carrier::Single(_), Some(_)) => Err(VindexError::Parse(
            "a single-stream carrier received a site reduction; preparation should have refused \
             the image"
                .to_string(),
        )),
    }
}
