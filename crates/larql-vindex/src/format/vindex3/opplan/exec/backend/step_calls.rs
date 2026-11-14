//! Routed-FFN, expert-slice and attention-step calls.

use super::super::lowering::LoweringIdentity;
use super::super::realization::{RepresentationFacts, Selection, SelectionRefusal};
use crate::error::VindexError;
use crate::format::vindex3::opplan::planned::PlannedOperand;
use larql_models::config::{Activation, ExpertRoutingPolicy, GateUpLayout, MoeRouterKind};

#[allow(unused_imports)]
use super::*;

/// One routed feed-forward operation over one vector, fully resolved:
/// the router in f32 (glue-sized), every expert's projections in the
/// backend's declared FFN format, and every judged semantic as an
/// argument. The backend routes, runs the selected experts and combines
/// — nothing here is re-derived from the plan.
pub struct RoutedFfnCall<'a> {
    pub x: &'a [f32],
    pub hidden: usize,
    /// Per-expert intermediate width.
    pub intermediate: usize,
    pub experts: usize,
    pub top_k: usize,
    pub router_kind: MoeRouterKind,
    pub routing_policy: ExpertRoutingPolicy,
    /// Multiplier on every selected expert's weight — the plan's declared
    /// `branch_scale`, 1 when it declares none. The shared expert is not
    /// under it.
    pub branch_scale: f32,
    pub activation: Activation,
    pub gate_policy: larql_models::ExpertGatePolicy,
    /// Router logits matrix `[experts, hidden]`, row-major.
    pub router: &'a [f32],
    /// Additive router bias `[experts]`.
    pub router_bias: Option<&'a [f32]>,
    /// The experts' matrices, in the shape their bank stores them.
    pub weights: ExpertSlices<'a>,
    /// Fused gate/up bias, `[experts · 2·intermediate]` flat, in the
    /// operand's own row layout. Packed banks only.
    pub gate_up_bias: Option<&'a [f32]>,
    /// Down bias, `[experts · hidden]` flat. Packed banks only.
    pub down_bias: Option<&'a [f32]>,
    /// What the router reads. Every family but Gemma 4 routes on the same
    /// vector the experts consume (`x`); Gemma 4's router reads the RAW
    /// post-attention residual and conditions it itself. `None` = `x`.
    pub router_input: Option<&'a [f32]>,
    /// `MoeRouterKind::TopKRenormScaled` conditioning, present iff the plan
    /// carries it: `router_input` is RMS-normalised without a weight
    /// (`router_norm_eps`), multiplied by `router_scale` `[hidden]` and by
    /// `hidden^-0.5` before the projection; the renormalised top-k weights
    /// are multiplied by `router_per_expert_scale[selected]`.
    pub router_scale: Option<&'a [f32]>,
    pub router_per_expert_scale: Option<&'a [f32]>,
    pub router_norm_eps: Option<f64>,
}

/// How a routed layer's expert matrices are handed to a backend: the
/// shape the bank STORES them in, never converted to the other.
pub enum ExpertSlices<'a> {
    /// A packed bank: one fused `[2·intermediate, hidden]` gate/up per
    /// expert, split by the call's `gate_up_layout`, and one
    /// `[hidden, intermediate]` down.
    Fused {
        gate_up: &'a [WeightSlice<'a>],
        down: &'a [WeightSlice<'a>],
        /// How each expert's fused rows split into gate and up — a
        /// property of the fused operand, so it travels with it.
        layout: GateUpLayout,
    },
    /// A per-expert bank: gate `[intermediate, hidden]`, up
    /// `[intermediate, hidden]` and down `[hidden, intermediate]` as three
    /// whole matrices per expert — each one the stored operand it is.
    Separate {
        gate: &'a [WeightSlice<'a>],
        up: &'a [WeightSlice<'a>],
        down: &'a [WeightSlice<'a>],
        /// How the selected experts' pages are brought in for this call.
        access: super::super::realization::MappedAccess,
    },
}

/// One position's attention against interpreter-owned K/V state — the
/// decode step.
///
/// `op.inputs` holds exactly one row: this position's already-normalised
/// attention input. `keys`/`values` are the post-norm, post-rope K and V
/// rows of every earlier position, exactly as this backend returned them
/// from earlier steps — the interpreter owns the cache; the backend owns
/// only the arithmetic of one step.
pub struct AttentionStepCall<'a> {
    /// The resolved attention operation, identical in meaning to the
    /// batch call — one struct so the two paths cannot drift apart in
    /// what they carry.
    pub op: AttentionCall<'a>,
    /// Absolute position of the row in `op.inputs`.
    pub position: usize,
    /// Earlier positions' K and V rows. Private: a step is built only by
    /// [`new`](Self::new), which checks the plan's required range against
    /// it, so no backend receives a step whose provider dropped a row the
    /// step needs.
    pub(super) rows: super::super::kv_view::KvView<'a>,
}

impl<'a> AttentionStepCall<'a> {
    /// A step at `position` reading `rows`. Refused, by name, unless the
    /// view ends at `position` and holds every earlier position the plan's
    /// retention authority says this step may read.
    pub fn new(
        op: AttentionCall<'a>,
        position: usize,
        rows: super::super::kv_view::KvView<'a>,
    ) -> Result<Self, super::super::kv_view::ViewRefusal> {
        let required = op.history().required_range(position);
        // The step's own row is fresh, not held: it needs [start, position).
        let history = required.start..position;
        if rows.end() != position {
            return Err(super::super::kv_view::ViewRefusal {
                needed: history,
                base: rows.base(),
                end: rows.end(),
            });
        }
        rows.covers(history)?;
        Ok(Self { op, position, rows })
    }

    /// Earlier positions' K and V rows, already checked against the plan.
    pub fn rows(&self) -> super::super::kv_view::KvView<'a> {
        self.rows
    }
}

/// One position's projected, conditioned (Q, K, V) — the intermediate
/// every backend's projection helper produces.
pub type ProjectedQkv = (Vec<f32>, Vec<f32>, Vec<f32>);

/// What one decode step returns: this position's K and V rows (for the
/// interpreter to append to its cache) and the attention output
/// (post gate, post output-projection).
pub struct AttentionStepOut {
    pub key: Vec<f32>,
    pub value: Vec<f32>,
    pub output: Vec<f32>,
}

/// The numerical realisation of a plan's operations.
///
/// Every method is total over its arguments: the caller has already
/// decided the operation happens. A backend may fail on work it cannot
/// perform (an unimplemented QK-norm scope, a device error), but it may
/// not decline work on semantic grounds — that judgment was made before
/// the call.
///
/// `Sync` because the interpreter issues per-position calls from
/// worker threads. Positions are independent through every operation
/// (attention reads other positions' K/V but never writes them), so
/// this parallelism reorders nothing within any one position's
/// arithmetic — results stay bit-identical to a serial execution.
/// What a backend spent inside its own dispatch calls, for attributing a
/// token's latency between device work and the interpreter's glue.
///
/// Exists because "the part that does not scale with weight bytes" is not
/// automatically submission overhead: the elementwise glue (norms, RoPE,
/// softmax over the KV cache, activations, residuals) is also a fixed
/// per-token cost, and optimising the wrong one of the two is free to
/// look like progress on a fit that cannot tell them apart.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct DispatchStats {
    /// Wall nanoseconds inside device dispatch calls — submission,
    /// device execution, and the wait, together.
    pub device_nanos: u64,
    /// Device submissions made (one per command buffer).
    pub submissions: u64,
    /// The device's own account of the buffers it waited on: commit to
    /// completion, and GPU execution. `None` when the device does not
    /// measure. With `device_nanos`, it splits a device call into host
    /// work, queue latency and GPU time.
    pub device_clock: Option<larql_compute::SubmissionClock>,
}

/// A shared handle IS the provider it holds: every method, the provided
/// ones included, goes to the provider underneath. Spelled out rather
/// than left to defaults on purpose — a `select` or `dense_projector`
/// that fell back to the trait's default here would quietly hand a
/// device provider the reference oracle's selection.
impl<T: PlanBackend + Send + ?Sized> PlanBackend for std::sync::Arc<T> {
    fn name(&self) -> &str {
        (**self).name()
    }

    fn identity(&self) -> LoweringIdentity {
        (**self).identity()
    }

    fn dispatch_stats(&self) -> Option<DispatchStats> {
        (**self).dispatch_stats()
    }

    fn select(
        &self,
        operand: &PlannedOperand,
        facts: &RepresentationFacts,
    ) -> Result<Selection, Box<SelectionRefusal>> {
        (**self).select(operand, facts)
    }

    fn dense_projector(&self) -> &dyn super::super::gated_delta::DenseProjections {
        (**self).dense_projector()
    }

    fn prepare(&self, weights: &[WeightSlice<'_>]) {
        (**self).prepare(weights)
    }

    fn embed(&self, table: &[f32], hidden: usize, token: u32, scale: Option<f32>) -> Vec<f32> {
        (**self).embed(table, hidden, token, scale)
    }

    fn norm(&self, call: NormCall<'_>) -> Vec<f32> {
        (**self).norm(call)
    }

    fn project(&self, call: ProjectCall<'_>) -> Result<Vec<f32>, VindexError> {
        (**self).project(call)
    }

    fn attention(&self, call: AttentionCall<'_>) -> Result<AttentionOut, VindexError> {
        (**self).attention(call)
    }

    fn attention_step(&self, call: AttentionStepCall<'_>) -> Result<AttentionStepOut, VindexError> {
        (**self).attention_step(call)
    }

    fn serves_attention_heads(&self) -> bool {
        (**self).serves_attention_heads()
    }

    fn attention_step_observed(
        &self,
        call: AttentionStepCall<'_>,
        tap: &mut dyn FnMut(super::super::observe::AttentionHeadRecord<'_>),
    ) -> Result<AttentionStepOut, VindexError> {
        (**self).attention_step_observed(call, tap)
    }

    fn serves_head_intervention(&self) -> bool {
        (**self).serves_head_intervention()
    }

    fn attention_step_intervened(
        &self,
        call: AttentionStepCall<'_>,
        tap: Option<&mut dyn FnMut(super::super::observe::AttentionHeadRecord<'_>)>,
        head_intervene: &mut HeadIntervene<'_>,
    ) -> Result<AttentionStepOut, VindexError> {
        (**self).attention_step_intervened(call, tap, head_intervene)
    }

    fn ffn(&self, call: FfnCall<'_>) -> Result<Vec<f32>, VindexError> {
        (**self).ffn(call)
    }

    fn serves_ffn_down_input(&self) -> bool {
        (**self).serves_ffn_down_input()
    }

    fn ffn_observed(
        &self,
        call: FfnCall<'_>,
        tap: &mut dyn FnMut(&[f32]),
    ) -> Result<Vec<f32>, VindexError> {
        (**self).ffn_observed(call, tap)
    }

    fn ffn_many(&self, call: FfnManyCall<'_>) -> Result<Vec<Vec<f32>>, VindexError> {
        (**self).ffn_many(call)
    }

    fn scale_row(&self, row: &mut [f32], scale: f32) {
        (**self).scale_row(row, scale)
    }

    fn routed_ffn(&self, call: RoutedFfnCall<'_>) -> Result<Vec<f32>, VindexError> {
        (**self).routed_ffn(call)
    }
    fn expert_transform(
        &self,
        call: super::super::routed_experts::ExpertTransformCall<'_>,
    ) -> Result<Vec<f32>, VindexError> {
        (**self).expert_transform(call)
    }
    fn routed_ffn_placed(
        &self,
        call: RoutedFfnCall<'_>,
        layer: usize,
        provider: &dyn super::super::routed_experts::RoutedExpertProvider,
    ) -> Result<Vec<f32>, VindexError> {
        (**self).routed_ffn_placed(call, layer, provider)
    }
    fn routed_ffn_placed_rows(
        &self,
        calls: &[RoutedFfnCall<'_>],
        layer: usize,
        provider: &dyn super::super::routed_experts::RoutedExpertProvider,
    ) -> Result<Vec<Vec<f32>>, VindexError> {
        (**self).routed_ffn_placed_rows(calls, layer, provider)
    }

    fn output_head(
        &self,
        projection: WeightSlice<'_>,
        vocab: usize,
        hidden: usize,
        x: &[f32],
        multiplier: Option<f64>,
        softcapping: Option<f32>,
    ) -> Result<Vec<f32>, VindexError> {
        (**self).output_head(projection, vocab, hidden, x, multiplier, softcapping)
    }

    fn residual_add(&self, acc: &mut [f32], delta: &[f32]) {
        (**self).residual_add(acc, delta)
    }
}

/// A provider names itself by presentation name and identity, so a
/// refusal or a test can say which one it was talking about.
impl std::fmt::Debug for dyn PlanBackend + '_ {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "provider {} ({})", self.name(), self.identity())
    }
}
