//! The decode session constructors and per-token step entry points.

use super::super::super::ComponentOpPlan;
use super::super::backend::PlanBackend;
use super::super::continuation_registry::BoxedContinuation;
use super::super::hyper_connection::Mutation;
use super::super::intervene::{InterventionPlan, InterventionStepOutput};
use super::super::intervene_heads::HeadInterventionPlan;
use super::super::kv::KvState;
use super::super::observe::{NoopObserver, StepObserver};
use super::super::operands::OperandSource;
use super::super::prepared::{ExecutionSlice, PreparedOperands};
use crate::error::VindexError;

#[allow(unused_imports)]
use super::*;

impl<'a, B: PlanBackend> DecodeSession<'a, B> {
    /// Load every operand the plan consumes, once, in the backend's
    /// declared weight format. The embedding table stays f32 — it is a
    /// row lookup, not matrix traffic. The session owns `state` — a fresh
    /// provider the caller built, normally from a
    /// [`SelectedContinuation`](super::super::continuation_registry::SelectedContinuation).
    pub fn new<'s>(
        plan: &'a ComponentOpPlan,
        store: impl Into<OperandSource<'s>>,
        backend: &'a B,
        state: BoxedContinuation,
    ) -> Result<Self, VindexError> {
        Self::build(plan, store.into(), backend, KvSlot::Owned(state))
    }

    /// Like [`new`](Self::new), but the caller provides — and keeps
    /// owning — the continuation state, so K/V policy composes outside
    /// the executor and the state outlives the session. The session
    /// continues from `kv.position()`: an empty provider starts a
    /// fresh sequence, and one populated by
    /// [`prefill_plan`](super::super::prefill_plan) — or by an earlier
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

    pub(super) fn build(
        plan: &'a ComponentOpPlan,
        store: OperandSource<'_>,
        backend: &'a B,
        kv: KvSlot<'a>,
    ) -> Result<Self, VindexError> {
        let ops = PreparedOperands::load(plan, store, backend, ExecutionSlice::Full)?;
        Self::assemble(plan, OperandsSlot::Owned(Box::new(ops)), backend, kv)
    }

    pub(super) fn assemble(
        plan: &'a ComponentOpPlan,
        ops: OperandsSlot<'a>,
        backend: &'a B,
        mut kv: KvSlot<'a>,
    ) -> Result<Self, VindexError> {
        if matches!(ops.get().slice(), ExecutionSlice::Endpoints) {
            return Err(VindexError::Parse(
                "endpoints-only operands require a distributed coordinator".into(),
            ));
        }
        ops.get().ensure_stack_ready()?;
        ops.get().ensure_providers_in(ops.get().registry())?;
        ops.get().ensure_lowered_by(backend)?;
        // The FULL continuation geometry, KV and recurrent alike.
        // `plan_kv_geometry` is the KV-only adapter and refuses a hybrid
        // plan; a session over one needs both forms announced.
        kv.state_mut().prepare_continuation(
            &super::super::continuation::plan_continuation_geometry(plan)
                .map_err(VindexError::Parse)?,
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
    pub fn residency_census(&self) -> super::super::prepared::ResidencyCensus {
        self.ops.get().residency_census()
    }

    /// The realization pinned for every operand this session executes,
    /// with its candidates, reason and declared residency.
    pub fn realizations(&self) -> &[super::super::realization::RealizationRecord] {
        self.ops.get().realizations()
    }

    /// Where this session's operand allocations landed.
    /// The prepared image's mappings and their resident pages, now.
    pub fn mapped_residency(&self) -> super::super::prepared::MappedResidency {
        self.ops.get().mapped_residency()
    }

    /// One residency observation of the prepared image, every mapped
    /// region read once with its outcome kept.
    pub fn observe_residency(&self) -> super::super::prepared::ResidencyObservation {
        self.ops.get().observe_residency()
    }

    pub fn allocation_census(&self) -> super::super::prepared::AllocationCensus {
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

    /// Consume one ready-to-concatenate embedding row at the next absolute
    /// position. Scaling/projector work belongs to the input producer. This
    /// enters the same layer loop as a token lookup; it does not edit weights.
    pub fn step_embedding(&mut self, row: &[f32]) -> Result<StepOutput, VindexError> {
        let ops = self.ops.get();
        if row.len() != ops.hidden() || row.iter().any(|x| !x.is_finite()) {
            return Err(VindexError::Parse(format!(
                "external embedding must contain {} finite values",
                ops.hidden()
            )));
        }
        if !ops.slice().is_whole_stack() || matches!(ops.slice(), ExecutionSlice::Endpoints) {
            return Err(VindexError::Parse(
                "external embeddings require a whole-stack image".into(),
            ));
        }
        let run = self.run(
            Entry::Hidden(row.to_vec()),
            &mut NoopObserver,
            Mutation::None,
            &InterventionPlan::none(),
            &HeadInterventionPlan::none(),
        )?;
        Ok(StepOutput { logits: run.logits })
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
        let out = super::super::traverse(
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
        // V3-HEAD-OBS-1, property A8: a request for heads against a
        // backend that cannot serve them refuses before the token
        // executes, never after a partial capture.
        if observer.wants_attention_heads() && !self.backend.serves_attention_heads() {
            return Err(VindexError::Parse(format!(
                "per-head attention observation is not served by the {} backend; observe \
                 without heads or run on a backend that declares them",
                self.backend.name()
            )));
        }
        self.admit_ffn_down_input(observer)?;
        let run = self.run(
            Entry::Token(token),
            observer,
            Mutation::None,
            &InterventionPlan::none(),
            &HeadInterventionPlan::none(),
        )?;
        Ok(StepOutput { logits: run.logits })
    }

    /// The step with interventions armed (V3-INTERVENE-1 carrier
    /// addresses, V3-INTERVENE-2 head addresses — declared beside each
    /// other, since one declaration file may name both). Both plans are
    /// admitted against this image before the token executes: an address
    /// off the executed layers, a site or head the layer never writes, a
    /// carrier that is not `Single`, or a vector of the wrong width
    /// refuses here. The returned firings say which declared
    /// interventions applied on this step; a declared address the run
    /// never reaches is the caller's to report from
    /// `InterventionPlan::unreached` / `HeadInterventionPlan::unreached`.
    pub fn step_intervened(
        &mut self,
        token: u32,
        observer: &mut dyn StepObserver,
        interventions: &InterventionPlan,
        head_interventions: &HeadInterventionPlan,
    ) -> Result<InterventionStepOutput, VindexError> {
        self.step_entered(
            Entry::Token(token),
            observer,
            interventions,
            head_interventions,
        )
    }

    /// Resume a single-stream layer-range image from its entering carrier
    /// (the GW programme's carrier entry). The caller supplies the
    /// canonical continuation state at this position; the layer loop,
    /// backend refusals and intervention admission are exactly
    /// [`Self::step_intervened`]'s. A full image, a multi-stream
    /// topology, or a carrier of the wrong width or with a non-finite
    /// value refuses before anything executes.
    pub fn step_from_carrier_intervened(
        &mut self,
        carrier: &[f32],
        observer: &mut dyn StepObserver,
        interventions: &InterventionPlan,
        head_interventions: &HeadInterventionPlan,
    ) -> Result<InterventionStepOutput, VindexError> {
        let ops = self.ops.get();
        if !matches!(ops.slice(), ExecutionSlice::LayerRange { .. })
            || !self.plan.residual_topology.is_single_stream()
            || carrier.len() != ops.hidden()
            || carrier.iter().any(|v| !v.is_finite())
        {
            return Err(VindexError::Parse(
                "carrier entry requires a finite, correctly sized single-stream layer-range input"
                    .into(),
            ));
        }
        self.step_entered(
            Entry::Single(carrier.to_vec()),
            observer,
            interventions,
            head_interventions,
        )
    }

    /// The one admitted entry both public intervened steps share: backend
    /// capability refusals, then both plans' admission, then the run.
    pub(super) fn step_entered(
        &mut self,
        entry: Entry,
        observer: &mut dyn StepObserver,
        interventions: &InterventionPlan,
        head_interventions: &HeadInterventionPlan,
    ) -> Result<InterventionStepOutput, VindexError> {
        if observer.wants_attention_heads() && !self.backend.serves_attention_heads() {
            return Err(VindexError::Parse(format!(
                "per-head attention observation is not served by the {} backend; observe \
                 without heads or run on a backend that declares them",
                self.backend.name()
            )));
        }
        self.admit_ffn_down_input(observer)?;
        if !head_interventions.is_none() && !self.backend.serves_head_intervention() {
            return Err(VindexError::Parse(format!(
                "per-head attention intervention is not served by the {} backend",
                self.backend.name()
            )));
        }
        interventions.admit(self.plan, self.ops.get())?;
        head_interventions.admit(self.plan, self.ops.get())?;
        let run = self.run(
            entry,
            observer,
            Mutation::None,
            interventions,
            head_interventions,
        )?;
        Ok(InterventionStepOutput {
            logits: run.logits,
            firings: run.firings,
            head_firings: run.head_firings,
        })
    }

    /// The step under a deliberate defect — the wave-19a negative
    /// controls. Test-only: production has exactly one way in, and it
    /// passes [`Mutation::None`].
    #[cfg(test)]
    pub(in super::super) fn step_mutated(
        &mut self,
        token: u32,
        observer: &mut dyn StepObserver,
        mutation: Mutation,
    ) -> Result<StepRun, VindexError> {
        self.run(
            Entry::Token(token),
            observer,
            mutation,
            &InterventionPlan::none(),
            &HeadInterventionPlan::none(),
        )
    }

    /// The step entered with a bundle at the first executed layer instead
    /// of a token — how the witness hands the oracle's own state to a
    /// site. Test-only, and hyper-connected components only.
    #[cfg(test)]
    pub(in super::super) fn step_from_bundle(
        &mut self,
        bundle: Bundle,
        observer: &mut dyn StepObserver,
        mutation: Mutation,
    ) -> Result<StepRun, VindexError> {
        self.run(
            Entry::Bundle(bundle),
            observer,
            mutation,
            &InterventionPlan::none(),
            &HeadInterventionPlan::none(),
        )
    }

    /// The one decode step. Every public and test entry above is this.
    /// CAL-1.1: a down-input request the step cannot serve refuses
    /// before the token executes, as the per-head request does — never
    /// mid-step, after earlier layers have written KV and emitted
    /// observations. Judged over exactly the layers the loop runs.
    pub(super) fn admit_ffn_down_input(
        &self,
        observer: &dyn StepObserver,
    ) -> Result<(), VindexError> {
        let ops = self.ops.get();
        let first = ops.first_layer();
        for (offset, state) in ops.layers().iter().enumerate() {
            let index = first + offset;
            let (Some(ffn), Some(ffn_op)) = (&state.ffn, &self.plan.layers[index].ffn) else {
                continue;
            };
            if !observer.wants_ffn_down_input(index) {
                continue;
            }
            if !self.backend.serves_ffn_down_input() {
                return Err(VindexError::Parse(format!(
                    "FFN down-input capture is not served by the {} backend",
                    self.backend.name()
                )));
            }
            if !ffn.serves_down_input(ffn_op) {
                return Err(VindexError::Parse(format!(
                    "FFN down-input capture at layer {index} requires a dense FFN"
                )));
            }
        }
        Ok(())
    }
}
