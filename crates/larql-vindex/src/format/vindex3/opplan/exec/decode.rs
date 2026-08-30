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

use super::backend::{AttentionStepCall, NormCall, PlanBackend};
use super::kv::{KvState, RowKvState};
use super::observe::{NoopObserver, StepEvent, StepObserver};
use super::operands::OperandSource;
use super::prepared::{ExecutionSlice, PreparedOperands};
use crate::error::VindexError;

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

    /// Where this session's operand allocations landed.
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
        let ops = self.ops.get();
        let hidden = ops.hidden();
        let embedding = self
            .plan
            .embedding
            .as_ref()
            .expect("session construction required an embedding op");
        let embed_table = ops.embed_table().ok_or_else(|| {
            VindexError::Parse(
                "this prepared image carries no embedding table — a layer-range slice consumes \
                 hidden states, not token ids"
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

        let position = self.kv.state().position();
        observer.event(StepEvent::Embedded { position });
        let first = ops.first_layer();
        for (offset, state) in ops.layers().iter().enumerate() {
            let index = first + offset;
            let layer = &self.plan.layers[index];
            // Attention input is normalised once and handed over; the
            // judged gate reads the same vector (same as the batch path).
            let inputs = [state.pre_attention.apply(self.backend, &h)];
            // The sensitivity tap from main, kept ahead of the operator
            // dispatch: it observes the attention INPUT, which both
            // operators read, so it belongs to neither branch.
            observer.operand_input(
                index,
                super::observe::InputSite::Attention,
                inputs[0].as_slice(),
            );
            // One position, either operator. A recurrence appends no KV
            // row — it rewrites its own buffers in place, which is only
            // correct because those buffers are DURABLE: the convolution
            // history spans the step boundary, and a single-position call
            // that reconstructed it from the batch would see a window of
            // one. That is the whole point of QW-3.6a.
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
                super::prepared::PreparedAttention::Softmax(sops) => {
                    let call = sops.call(
                        layer
                            .attention
                            .softmax()
                            .expect("prepared softmax operands imply a softmax op"),
                        &inputs,
                        layer.pre_attention_norm.eps,
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
            let mut attn_out = match &state.post_attention {
                Some(norm) => norm.apply(self.backend, &raw_attn),
                None => raw_attn,
            };
            super::scale_residual_delta(layer.residual_scale, &mut attn_out);
            self.backend.residual_add(&mut h, &attn_out);
            observer.event(StepEvent::AttentionDone { layer: index });

            // A mixer-only (Mamba2) layer carries no FFN program: its one
            // residual add happened above, and running a fabricated FFN
            // stage here would be the schema-6 fabrication re-enacted at
            // execution time. Presence follows the program here too.
            if let (Some(pre_ffn), Some(ffn), Some(ffn_op)) =
                (&state.pre_ffn, &state.ffn, &layer.ffn)
            {
                let normed = pre_ffn.apply(self.backend, &h);
                observer.operand_input(index, super::observe::InputSite::Ffn, normed.as_slice());
                let _site = super::cpu::ledger::in_site(super::cpu::ledger::Site::Ffn);
                let ffn_out = ffn.apply_from_residual(ffn_op, self.backend, &h, &normed, hidden)?;
                observer.operand_input(
                    index,
                    super::observe::InputSite::FfnOutput,
                    ffn_out.as_slice(),
                );
                let mut ffn_out = match &state.post_ffn {
                    Some(norm) => norm.apply(self.backend, &ffn_out),
                    None => ffn_out,
                };
                super::scale_residual_delta(layer.residual_scale, &mut ffn_out);
                self.backend.residual_add(&mut h, &ffn_out);
                if let Some(scale) = state.layer_scale {
                    self.backend.scale_row(&mut h, scale);
                }
            }
            observer.event(StepEvent::FfnDone { layer: index });
        }

        let final_hidden = match ops.final_norm() {
            Some(norm) => norm.apply(self.backend, &h),
            None => h,
        };
        let logits = match ops.output() {
            Some((op, weight)) => Some(self.backend.output_head(
                weight.slice(),
                op.projection.shape[0],
                hidden,
                &final_hidden,
                op.multiplier,
                op.softcapping,
            )?),
            None => None,
        };
        if let Some(logits) = &logits {
            observer.event(StepEvent::Logits {
                vocab: logits.len(),
            });
        }
        self.kv.state_mut().set_position(position + 1);
        Ok(StepOutput { logits })
    }
}
