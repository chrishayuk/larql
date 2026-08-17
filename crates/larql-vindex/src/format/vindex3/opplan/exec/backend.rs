//! The backend seam (V3-G5b-3b): what *executes* a plan, versus what the
//! plan *means*.
//!
//! One [`ComponentOpPlan`](super::super::ComponentOpPlan), one interpreter,
//! many backends. The interpreter in [`super`] owns every decision that is
//! semantics — operation ordering, residual ordering, layer traversal,
//! whether an optional operation exists at all, and how position and span
//! policy dispatch. A [`PlanBackend`] owns only the numerical realisation
//! of work it is handed.
//!
//! **Nothing in this file mentions a model family, and nothing in it takes
//! a plan type.** Backends receive primitives, judged enums, and *already
//! loaded* weight slices — never a `LayerPlan`, an `OperandRef`, or the
//! `OperandStore`. That is deliberate and load-bearing: a backend that
//! could resolve its own operands by name, or read the layer structure,
//! could quietly grow into a second implementation of the model and
//! disagree with the IR while still passing. It cannot reach the bytes,
//! so it cannot reinterpret them.
//!
//! The corollary for anyone adding a method: if a backend needs to ask
//! *whether* to do something, the seam is in the wrong place. It should
//! only ever be told what to compute.

use larql_models::config::{
    Activation, AttentionGateSpec, AttentionSinkSpec, ExpertRoutingPolicy, GateUpLayout,
    MoeRouterKind, NormType, ParameterFreeQkNorm, PositionPolicy, QkNormScope,
};

use super::super::super::graph::policy::AttentionSpan;
use crate::error::VindexError;

/// The numerical representation a backend wants matrix operands in.
///
/// Asked once by the interpreter (a capability, like [`PlanBackend::name`],
/// not a per-call decision): the interpreter loads every matrix operand in
/// the declared format and the backend receives what it asked for. Norm
/// and QK-norm weights and the embedding table are always f32 — they are
/// elementwise glue, not matrix traffic, and narrowing them buys nothing.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum WeightFormat {
    /// Widened f32 — the constitutional representation.
    F32,
    /// IEEE 754 half, little-endian. Exactly representable from stored
    /// bf16 for all normal-range values (bf16's 7 mantissa bits fit in
    /// f16's 10); conversion fails closed on overflow. A device backend
    /// declares this so weights can stay resident in half the bytes.
    F16,
    /// OCP microscaling 4-bit float: e2m1 codes two-per-byte plus one
    /// e8m0 scale per 32-element group, in separate streams. A lossy
    /// realisation — quantised at load, judged by the parity gates —
    /// that quarters the bytes every decoded token must read.
    Mxfp4,
    /// The same e2m1 elements under a different scale geometry: 16-element
    /// groups with **E4M3** scales, plus one f32 per matrix. 4.5 bpw
    /// against MXFP4's 4.25.
    ///
    /// Present as its own format rather than a parameter of [`Self::Mxfp4`]
    /// because the difference is the point: E8M0 forces a group's scale to
    /// a power of two, and a weight-reconstruction sweep over Muse-Glimmer
    /// with an equal-bit-budget control (E8M0 at group 16) found the group
    /// size worth nothing and the scale format worth 1.27x in relative RMS
    /// and 1.7x in worst-element error.
    Nvfp4,
}

/// Which matrix a format question is about. Formats are declared per
/// class because the classes have different numerical stakes: the
/// output head feeds logits directly, attention feeds the softmax, and
/// the FFN is the bulk of the bytes.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum MatrixClass {
    AttentionProjection,
    FfnProjection,
    OutputHead,
}

/// A backend's declared format per matrix class.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct WeightFormats {
    pub attention: WeightFormat,
    pub ffn: WeightFormat,
    pub head: WeightFormat,
}

impl WeightFormats {
    /// The same format everywhere.
    pub fn uniform(format: WeightFormat) -> Self {
        Self {
            attention: format,
            ffn: format,
            head: format,
        }
    }

    pub fn for_class(&self, class: MatrixClass) -> WeightFormat {
        match class {
            MatrixClass::AttentionProjection => self.attention,
            MatrixClass::FfnProjection => self.ffn,
            MatrixClass::OutputHead => self.head,
        }
    }
}

/// One matrix operand, in the representation the backend declared.
///
/// An `F16` slice may be longer than the matrix needs (page-padded for
/// zero-copy device wrapping); geometry always travels separately.
#[derive(Clone, Copy)]
pub enum WeightSlice<'a> {
    F32(&'a [f32]),
    /// Little-endian IEEE f16 bytes.
    F16(&'a [u8]),
    /// MXFP4: packed e2m1 codes (`[n, k/32, 16]`, lo nibble first) and
    /// e8m0 scales (`[n, k/32]`) as two streams.
    Mxfp4 {
        packed: &'a [u8],
        scales: &'a [u8],
    },
    /// NVFP4: packed e2m1 codes (`[n, k/16, 8]`, lo nibble first), E4M3
    /// group scales (`[n, k/16]`), and the single f32 both scale levels
    /// are expressed relative to.
    Nvfp4 {
        packed: &'a [u8],
        scales: &'a [u8],
        tensor_scale: f32,
    },
}

impl<'a> WeightSlice<'a> {
    /// The f32 view a CPU backend computes with. A backend that declared
    /// `F32` can never legitimately receive `F16`, so this is fail-closed
    /// evidence of an interpreter bug, not a conversion point.
    pub fn as_f32(&self) -> Result<&'a [f32], VindexError> {
        match self {
            WeightSlice::F32(w) => Ok(w),
            WeightSlice::F16(_) | WeightSlice::Mxfp4 { .. } | WeightSlice::Nvfp4 { .. } => {
                Err(VindexError::Parse(
                    "backend declared f32 weights but was handed another format — interpreter \
                 loaded the wrong representation"
                        .to_string(),
                ))
            }
        }
    }
}

/// One normalisation, fully resolved.
///
/// `weight` empty means a parameter-free application (statistic only) —
/// the interpreter decides that from the plan, never the backend.
pub struct NormCall<'a> {
    pub kind: NormType,
    pub x: &'a [f32],
    pub weight: &'a [f32],
    pub weight_offset: f32,
    pub eps: f64,
}

/// One `[out, in]` row-major projection applied to one vector.
pub struct ProjectCall<'a> {
    pub weight: WeightSlice<'a>,
    pub out_dim: usize,
    pub in_dim: usize,
    pub x: &'a [f32],
}

/// QK normalisation weights and scope, when the plan binds them.
pub struct QkNormCall<'a> {
    pub scope: QkNormScope,
    pub weight_offset: f32,
    pub q_weight: &'a [f32],
    pub k_weight: &'a [f32],
}

/// The attention output gate, when the surface judged one.
pub struct GateCall<'a> {
    pub spec: AttentionGateSpec,
    pub weight: WeightSlice<'a>,
}

/// The judged attention-sink semantics plus the per-query-head logits,
/// f32 like every other elementwise operand.
pub struct SinkCall<'a> {
    pub spec: AttentionSinkSpec,
    /// `num_q_heads` logits.
    pub logits: &'a [f32],
}

/// The additive projection biases, all four present or none — closure
/// guarantees the pairing with the surface's `attention_bias`. Each is
/// one value per output row of its projection.
pub struct BiasCall<'a> {
    pub q: &'a [f32],
    pub k: &'a [f32],
    pub v: &'a [f32],
    pub o: &'a [f32],
}

/// One attention operation over a whole sequence, fully resolved.
///
/// `inputs` are the attention *inputs* — already normalised by the
/// interpreter — because the judged gate reads that same vector, and
/// handing the backend one operand for both uses removes any chance of
/// the two drifting apart.
pub struct AttentionCall<'a> {
    pub inputs: &'a [Vec<f32>],
    pub hidden: usize,
    pub num_q_heads: usize,
    pub num_kv_heads: usize,
    pub head_dim: usize,
    pub w_q: WeightSlice<'a>,
    pub w_k: WeightSlice<'a>,
    pub w_v: WeightSlice<'a>,
    pub w_o: WeightSlice<'a>,
    pub qk_norm: Option<QkNormCall<'a>>,
    pub parameter_free_qk_norm: ParameterFreeQkNorm,
    /// Epsilon for both QK-norm forms; rides with the layer's norm
    /// surface because neither form declares its own.
    pub qk_norm_eps: f64,
    /// `None` = no query-scale operation, never an invented 1.0.
    pub query_scale: Option<f64>,
    pub score_scale: f64,
    pub logit_softcapping: Option<f32>,
    pub position: PositionPolicy,
    pub span: AttentionSpan,
    pub window: Option<usize>,
    pub gate: Option<GateCall<'a>>,
    /// Q/K/V/O biases: Q and K added right after projection (before
    /// QK-norm and rope), V before caching, O after the output
    /// projection. `None` = the op has no biases.
    pub bias: Option<BiasCall<'a>>,
    /// Attention sinks; `None` = ordinary softmax.
    pub sinks: Option<SinkCall<'a>>,
}

/// One feed-forward operation over one vector, fully resolved.
///
/// `gate` present means gated; absent means standard. Again the
/// interpreter reads that from the plan.
pub struct FfnCall<'a> {
    pub x: &'a [f32],
    pub hidden: usize,
    pub intermediate: usize,
    pub gate: Option<WeightSlice<'a>>,
    pub up: WeightSlice<'a>,
    pub down: WeightSlice<'a>,
    pub activation: Activation,
    /// How `gate` combines with `up`. Every backend must honour it or
    /// refuse: computing `activation(gate) * up` for a `ClampedGlu` plan
    /// runs a different model.
    pub gate_policy: larql_models::ExpertGatePolicy,
}

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
    pub activation: Activation,
    pub gate_policy: larql_models::ExpertGatePolicy,
    /// How each expert's fused `gate_up` rows split into gate and up.
    pub gate_up_layout: GateUpLayout,
    /// Router logits matrix `[experts, hidden]`, row-major.
    pub router: &'a [f32],
    /// Additive router bias `[experts]`.
    pub router_bias: Option<&'a [f32]>,
    /// One `[2·intermediate, hidden]` matrix per expert.
    pub gate_up: &'a [WeightSlice<'a>],
    /// Fused gate/up bias, `[experts · 2·intermediate]` flat, in the
    /// operand's own row layout.
    pub gate_up_bias: Option<&'a [f32]>,
    /// One `[hidden, intermediate]` matrix per expert.
    pub down: &'a [WeightSlice<'a>],
    /// Down bias, `[experts · hidden]` flat.
    pub down_bias: Option<&'a [f32]>,
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
    /// Cached K rows for positions `0..position`.
    pub keys: &'a [Vec<f32>],
    /// Cached V rows for positions `0..position`.
    pub values: &'a [Vec<f32>],
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
}

pub trait PlanBackend: Sync {
    /// A name for diagnostics and parity reports. Not dispatched on.
    fn name(&self) -> &str;

    /// Cumulative device-dispatch accounting, when the backend keeps it.
    /// `None` for backends with no device to account for.
    fn dispatch_stats(&self) -> Option<DispatchStats> {
        None
    }

    /// The representation this backend wants matrix operands of `class`
    /// loaded in. A capability, asked per site at load time — not a
    /// per-call decision.
    fn weight_format(&self, _class: MatrixClass) -> WeightFormat {
        WeightFormat::F32
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

    /// Attention over the whole sequence, returning one output vector per
    /// position (post output-projection).
    fn attention(&self, call: AttentionCall<'_>) -> Result<Vec<Vec<f32>>, VindexError>;

    /// One position's attention against cached K/V — the decode step.
    ///
    /// Must realise exactly the arithmetic its own [`Self::attention`]
    /// applies to a single position: the decode-vs-batch parity tests
    /// pin the two paths together per backend, and a backend may not
    /// borrow another backend's step to fill the gap.
    fn attention_step(&self, call: AttentionStepCall<'_>) -> Result<AttentionStepOut, VindexError>;

    /// Fallible for the same reason as [`Self::attention`]: a backend
    /// with no kernel for a judged variant must say so, not borrow
    /// another backend's arithmetic to fill the gap.
    fn ffn(&self, call: FfnCall<'_>) -> Result<Vec<f32>, VindexError>;

    /// The routed FFN — a mixture of experts. Required of every backend
    /// for the same reason as [`Self::ffn`]: a backend without the
    /// arithmetic must refuse, never borrow it.
    fn routed_ffn(&self, call: RoutedFfnCall<'_>) -> Result<Vec<f32>, VindexError>;

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
