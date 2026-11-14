//! The `PlanBackend` trait every executor backend implements.

use super::super::lowering::LoweringIdentity;
use super::super::realization::{RepresentationFacts, Selection, SelectionRefusal};
use crate::error::VindexError;
use crate::format::vindex3::opplan::planned::PlannedOperand;

#[allow(unused_imports)]
use super::*;

pub trait PlanBackend: Sync {
    /// A name for diagnostics and parity reports. Not dispatched on, and
    /// not an identity: two instances of one provider may carry different
    /// names (a device backend names its device and realisation), and the
    /// name says nothing a registry or a pin can rely on.
    fn name(&self) -> &str;

    /// **The provider's identity** — family and revision — which is the
    /// authority a pin will record and a registry will key. Required, so
    /// a provider that says nothing does not exist; never derived from
    /// [`Self::name`]. See [`super::super::lowering`] for what the two fields
    /// mean and when the revision moves.
    fn identity(&self) -> LoweringIdentity;

    /// Cumulative device-dispatch accounting, when the backend keeps it.
    /// `None` for backends with no device to account for.
    fn dispatch_stats(&self) -> Option<DispatchStats> {
        None
    }

    /// The realization this backend takes for one planned operand, given
    /// what the registry declares for its stored representation.
    ///
    /// Chosen from candidates the backend derives from those declarations,
    /// or refused naming every candidate considered. Asked per operand at
    /// preparation, BEFORE any byte is read, and pinned in the prepared
    /// plan: the executor runs what was pinned or refuses. The default is
    /// the reference oracle — the literal f32 transcription — so a backend
    /// that says nothing inherits the arithmetic that defines correctness.
    fn select(
        &self,
        operand: &PlannedOperand,
        facts: &RepresentationFacts,
    ) -> Result<Selection, Box<SelectionRefusal>> {
        super::super::realization::reference_selection(operand, facts)
    }

    /// How this backend performs a Gated DeltaNet layer's dense
    /// projections.
    ///
    /// Defaults to the literal scalar transcription, so a backend that
    /// says nothing gets the reference arithmetic rather than inheriting
    /// somebody else's. The recurrence itself is not selectable — only
    /// the five matrix products around it.
    fn dense_projector(&self) -> &dyn super::super::gated_delta::DenseProjections {
        &super::super::gated_delta::ScalarProjections
    }

    /// Residency hint before a decode run: every matrix operand the
    /// session will read, already loaded. Computes nothing and must
    /// change no number — a backend may warm caches or wire device
    /// memory, or ignore it entirely. The default does nothing.
    fn prepare(&self, _weights: &[WeightSlice<'_>]) {}

    /// Look up one embedding row, applying the scale operation when the
    /// plan carries one. `scale` `None` = no such operation, so the row
    /// is returned unscaled rather than multiplied by an identity.
    fn embed(&self, table: &[f32], hidden: usize, token: u32, scale: Option<f32>) -> Vec<f32>;

    fn norm(&self, call: NormCall<'_>) -> Vec<f32>;

    /// Fallible for the same reason as [`Self::attention`]: a device
    /// backend may be unable to perform the work, and it must say so
    /// rather than borrow another backend's arithmetic.
    fn project(&self, call: ProjectCall<'_>) -> Result<Vec<f32>, VindexError>;

    /// Attention over the whole sequence.
    ///
    /// Returns the conditioned K/V rows alongside the outputs because
    /// the realisation already computes them: it must, to attend at
    /// all. Discarding them is what forced a caller that wanted a
    /// populated K/V cache down [`Self::attention_step`] instead —
    /// coupling "I want KV" to "run attention one position at a time"
    /// (V3-SERVE-2).
    ///
    /// The rows must be the same rows [`Self::attention_step`] would
    /// produce for the same position and input; both realisations of a
    /// backend answer for one program, and the attention-parity gates
    /// pin them together.
    fn attention(&self, call: AttentionCall<'_>) -> Result<AttentionOut, VindexError>;

    /// One position's attention against cached K/V — the decode step.
    ///
    /// Must realise exactly the arithmetic its own [`Self::attention`]
    /// applies to a single position: the decode-vs-batch parity tests
    /// pin the two paths together per backend, and a backend may not
    /// borrow another backend's step to fill the gap.
    fn attention_step(&self, call: AttentionStepCall<'_>) -> Result<AttentionStepOut, VindexError>;

    /// V3-HEAD-OBS-1: whether this backend's softmax attention core can
    /// hand an observer one [`super::super::observe::AttentionHeadRecord`] per
    /// query head. `false` by default, and a request against a backend
    /// that says so is refused before the first token executes.
    fn serves_attention_heads(&self) -> bool {
        false
    }

    /// [`Self::attention_step`] with the per-head tap armed: the same
    /// arithmetic, with each query head's distribution and mixed value
    /// handed to `tap` between aggregation and the output gate. The
    /// default refuses, matching [`Self::serves_attention_heads`].
    fn attention_step_observed(
        &self,
        _call: AttentionStepCall<'_>,
        _tap: &mut dyn FnMut(super::super::observe::AttentionHeadRecord<'_>),
    ) -> Result<AttentionStepOut, VindexError> {
        Err(VindexError::Parse(format!(
            "per-head attention observation is not served by the {} backend",
            self.name()
        )))
    }

    /// V3-INTERVENE-2: whether this backend's softmax attention core can
    /// take a per-head intervention — mutate `ctx_h` in place, after the
    /// (uninintervened) head record fires and before the gate multiply.
    /// `false` by default, matching [`Self::serves_attention_heads`].
    fn serves_head_intervention(&self) -> bool {
        false
    }

    /// [`Self::attention_step`] with a per-head intervention armed, and
    /// optionally the per-head tap too (records still see the
    /// uninintervened `ctx_h`, J3). The default refuses, matching
    /// [`Self::serves_head_intervention`].
    fn attention_step_intervened(
        &self,
        _call: AttentionStepCall<'_>,
        _tap: Option<&mut dyn FnMut(super::super::observe::AttentionHeadRecord<'_>)>,
        _head_intervene: &mut HeadIntervene<'_>,
    ) -> Result<AttentionStepOut, VindexError> {
        Err(VindexError::Parse(format!(
            "per-head attention intervention is not served by the {} backend",
            self.name()
        )))
    }

    /// Fallible for the same reason as [`Self::attention`]: a backend
    /// with no kernel for a judged variant must say so, not borrow
    /// another backend's arithmetic to fill the gap.
    fn ffn(&self, call: FfnCall<'_>) -> Result<Vec<f32>, VindexError>;

    /// CAL-1.1: whether [`Self::ffn_observed`] can hand an observer the
    /// dense down input. `false` by default, and a request against a
    /// backend that says so is refused before the token executes.
    fn serves_ffn_down_input(&self) -> bool {
        false
    }

    /// Borrow the actual intermediate immediately before down projection.
    /// The default refuses rather than reconstructing another backend's input.
    fn ffn_observed(
        &self,
        _call: FfnCall<'_>,
        _tap: &mut dyn FnMut(&[f32]),
    ) -> Result<Vec<f32>, VindexError> {
        Err(VindexError::Parse(
            "FFN down-input capture unavailable on this backend".into(),
        ))
    }

    /// The dense FFN over several positions at once.
    ///
    /// Default is the loop it replaces, so every backend keeps working
    /// untouched. Overriding it is a claim about SCHEDULE only: each
    /// position keeps its own activation and its own arithmetic, and the
    /// results must be indistinguishable from calling [`Self::ffn`] once
    /// per position.
    fn ffn_many(&self, call: FfnManyCall<'_>) -> Result<Vec<Vec<f32>>, VindexError> {
        call.xs
            .iter()
            .map(|x| {
                self.ffn(FfnCall {
                    x,
                    hidden: call.hidden,
                    intermediate: call.intermediate,
                    gate: call.gate,
                    up: call.up,
                    down: call.down,
                    activation: call.activation,
                    gate_policy: call.gate_policy,
                })
            })
            .collect()
    }

    /// The routed FFN — a mixture of experts. Required of every backend
    /// for the same reason as [`Self::ffn`]: a backend without the
    /// arithmetic must refuse, never borrow it.
    /// Multiply one hidden row by a scalar in place — Gemma 4's
    /// `layer_scalar` on the whole layer output. Elementwise glue like
    /// [`Self::residual_add`]; a backend overrides only to keep the row on
    /// its device.
    fn scale_row(&self, row: &mut [f32], scale: f32) {
        for value in row {
            *value *= scale;
        }
    }

    fn routed_ffn(&self, call: RoutedFfnCall<'_>) -> Result<Vec<f32>, VindexError>;

    /// One unweighted expert, with its own biases. No routing or reduction.
    fn expert_transform(
        &self,
        _call: super::super::routed_experts::ExpertTransformCall<'_>,
    ) -> Result<Vec<f32>, VindexError> {
        Err(VindexError::Parse(format!(
            "{} does not support selected expert transforms",
            self.name()
        )))
    }

    /// Route and reduce locally while a bound provider executes selected IDs.
    fn routed_ffn_placed(
        &self,
        _call: RoutedFfnCall<'_>,
        _layer: usize,
        _provider: &dyn super::super::routed_experts::RoutedExpertProvider,
    ) -> Result<Vec<f32>, VindexError> {
        Err(VindexError::Parse(format!(
            "{} does not support routed expert placement",
            self.name()
        )))
    }

    /// [`Self::routed_ffn_placed`] for several rows of one layer: route each
    /// row locally, run every row's selected experts through ONE provider
    /// call, reduce each row in its own selection order. Returns one output
    /// per call, in call order, or fails as a whole.
    fn routed_ffn_placed_rows(
        &self,
        _calls: &[RoutedFfnCall<'_>],
        _layer: usize,
        _provider: &dyn super::super::routed_experts::RoutedExpertProvider,
    ) -> Result<Vec<Vec<f32>>, VindexError> {
        Err(VindexError::Parse(format!(
            "{} does not support multi-row routed expert placement",
            self.name()
        )))
    }

    /// Vocabulary projection plus the head's optional multiplier and
    /// softcap, in that order.
    fn output_head(
        &self,
        projection: WeightSlice<'_>,
        vocab: usize,
        hidden: usize,
        x: &[f32],
        multiplier: Option<f64>,
        softcapping: Option<f32>,
    ) -> Result<Vec<f32>, VindexError>;

    /// Add `delta` into `acc` elementwise — the residual write.
    ///
    /// A method rather than a loop in the interpreter because residual
    /// accumulation order is exactly the kind of thing a fused production
    /// kernel wants to own, and because a backend that reassociates it
    /// should have to say so.
    fn residual_add(&self, acc: &mut [f32], delta: &[f32]);
}
