//! `larql vindex3 ops` — the generic operation plan (V3-G5b-1).
//!
//! Given only a container, answer: **what exact generic program does this
//! component mean?** Every argument comes from the persisted graph — the
//! execution surface, the per-layer attention policy, the operand roles —
//! and every operand is a logical-object reference plus a segment-relative
//! tensor. No family name, no layer-pattern arithmetic, no HF tensor name
//! appears anywhere in a plan.
//!
//! **Operand closure** is the hard gate this rung adds (the invariant G4
//! cannot state): four-authority equivalence proves *consistency*; closure
//! proves *sufficiency*.
//!
//! ```text
//! for every tensor of an executable object:
//!     tensor → classified operand role → consumed by a generic op
//! and for every op the surface implies:
//!     its operands exist, with the geometry the surface states
//! ```
//!
//! A tensor the roles cannot classify, an operand implying an op the
//! surface does not carry (the attention-gate discovery), a missing
//! operand, or a wrong shape each block the plan — itemised, before a
//! single matmul.

pub mod build;
pub mod exec;

#[cfg(test)]
mod tests;

use larql_models::config::{
    Activation, AttentionGateSpec, AttentionSinkSpec, ExpertFormat, ExpertRoutingPolicy,
    GateUpLayout, MoeRouterKind, NormType, ParameterFreeQkNorm, PositionPolicy, QkNormScope,
};
use serde::Serialize;

use super::graph::policy::AttentionSpan;
use super::graph::{ObjectKind, OperandRole};

pub use build::plan_component_ops;

/// One kernel argument: a logical object plus its segment-relative tensor.
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct OperandRef {
    /// Logical object id (`target.decoder_stack`).
    pub object: String,
    /// Segment-relative tensor name (`3.self_attn.q_proj.weight`).
    pub tensor: String,
    pub dtype: String,
    pub shape: Vec<usize>,
}

/// A normalisation op, fully parameterised.
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct NormOp {
    pub kind: NormType,
    pub eps: f64,
    pub weight_offset: f32,
    pub weight: OperandRef,
}

/// QK normalisation inside attention.
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct QkNormOp {
    pub scope: QkNormScope,
    pub weight_offset: f32,
    pub q: OperandRef,
    pub k: OperandRef,
}

/// The optional gate on attention output: the fully judged semantics
/// plus the operand implementing it.
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct GateOp {
    pub spec: AttentionGateSpec,
    pub projection: OperandRef,
}

/// The optional attention sinks: the judged semantics plus the operand
/// holding the per-query-head logits.
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct SinkOp {
    pub spec: AttentionSinkSpec,
    pub logits: OperandRef,
}

/// One layer's attention op: geometry and scaling from the surface,
/// span/window/position from the per-layer policy table — never from an
/// index pattern.
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct AttentionOp {
    pub num_q_heads: usize,
    pub num_kv_heads: usize,
    pub head_dim: usize,
    /// The query-scale operation, applied to the (normalised) query
    /// states before position encoding. `None` = the op is absent, which
    /// the executor must skip rather than multiply by an identity it
    /// invented.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub query_scale: Option<f64>,
    /// The canonical score-time multiply — deliberately not folded into
    /// [`Self::query_scale`] (algebra-equivalent, not fp-equivalent).
    pub score_scale: f64,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub logit_softcapping: Option<f32>,
    pub span: AttentionSpan,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub window: Option<usize>,
    pub position: PositionPolicy,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub qk_norm: Option<QkNormOp>,
    /// Weightless Q/K RMS normalisation, when the judged semantics say so.
    pub parameter_free_qk_norm: ParameterFreeQkNorm,
    pub q: OperandRef,
    pub k: OperandRef,
    /// The value projection; the SAME operand as `k` when `v_from_k`.
    pub v: OperandRef,
    /// V is the raw K projection (Gemma 4 `attention_k_eq_v` on full
    /// layers): `v` names the K operand, and the executor must take V from
    /// that projection BEFORE the key's norm and rotation. Untagged
    /// default so plans without it serialise byte-identically.
    #[serde(default, skip_serializing_if = "std::ops::Not::not")]
    pub v_from_k: bool,
    pub o: OperandRef,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub output_gate: Option<GateOp>,
    /// Additive projection biases; all four present iff the surface
    /// declares `attention_bias`. Absent from the serialised op otherwise,
    /// so a bias-free plan serialises exactly as before.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub q_bias: Option<OperandRef>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub k_bias: Option<OperandRef>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub v_bias: Option<OperandRef>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub o_bias: Option<OperandRef>,
    /// Attention sinks, present iff the surface carries the judgment.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub sinks: Option<SinkOp>,
}

/// One layer's FFN op.
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct FfnOp {
    pub intermediate_size: usize,
    pub activation: Activation,
    /// How the gate combines with the up branch (plain gated, or GPT-OSS's
    /// clamped GLU). Transcribed from `FfnSurface.gate_policy`.
    pub gate_policy: larql_models::ExpertGatePolicy,
    /// Present iff the surface says the FFN is gated.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub gate: Option<OperandRef>,
    pub up: OperandRef,
    pub down: OperandRef,
}

/// One packed expert projection: the bytes for every expert in one
/// operand (`[experts, rows, …]`), plus the companion streams its
/// representation needs. `scales` is present iff the expert format keeps
/// its dequantisation scales in a separate stream (MXFP4); `bias` iff the
/// checkpoint carries per-expert biases.
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct PackedProjection {
    pub weights: OperandRef,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub scales: Option<OperandRef>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub bias: Option<OperandRef>,
}

/// One layer's routed FFN op — a mixture of experts, entirely inside the
/// generic graph: the router operands live in the decoder stack, the
/// expert operands in the component's expert-bank object, and every
/// semantic the executor needs is transcribed here from `FfnSurface.moe`.
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct RoutedFfnOp {
    pub experts: usize,
    pub top_k: usize,
    pub expert_intermediate_size: usize,
    pub router_kind: MoeRouterKind,
    pub routing_policy: ExpertRoutingPolicy,
    pub activation: Activation,
    /// How each expert's gate combines with its up branch.
    pub gate_policy: larql_models::ExpertGatePolicy,
    pub expert_format: ExpertFormat,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub gate_up_layout: Option<GateUpLayout>,
    /// Router logits: `[experts, hidden]`.
    pub router: OperandRef,
    /// Additive router bias `[experts]`, iff the surface declares one.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub router_bias: Option<OperandRef>,
    /// Gemma 4 (`MoeRouterKind::Gemma4Hybrid`) router conditioning: the
    /// residual is RMS-normalised WITHOUT a weight (eps
    /// `router_norm_eps`), multiplied by `router_scale` `[hidden]` and by
    /// `hidden^-0.5`, then projected; the renormalised top-k weights are
    /// multiplied by `router_per_expert_scale[selected]`. Present iff the
    /// router kind is `Gemma4Hybrid` (closure-paired). Absent from every
    /// other plan, so those serialise byte-identically.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub router_scale: Option<OperandRef>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub router_per_expert_scale: Option<OperandRef>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub router_norm_eps: Option<f64>,
    /// Fused gate+up per expert: `[experts, 2·inter, hidden]` in the
    /// declared layout (packed as the format dictates).
    pub gate_up: PackedProjection,
    /// Down per expert: `[experts, hidden, inter]`.
    pub down: PackedProjection,
}

/// Gemma 4's hybrid FFN: a dense MLP and a routed expert block in ONE
/// layer, both fed from the post-attention residual `r`, outputs summed
/// before the layer's post-FFN norm. Transcribed from
/// `Gemma4TextDecoderLayer`:
///
/// ```text
/// h  = pre_ffn_norm(r)                     (the layer's PreFfnNorm)
/// d  = post_dense_norm(mlp(h))             (post_feedforward_layernorm_1)
/// e  = post_experts_norm(experts(pre_experts_norm(r)))
///                                          (…_2 pre, …_2 post; router reads r)
/// out = r + post_ffn_norm(d + e)           (the layer's PostFfnNorm)
/// ```
///
/// The router's own conditioning rides on [`RoutedFfnOp`]. Neither branch
/// is a fallback for the other: closure requires every operand of both.
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct HybridFfnOp {
    pub dense: FfnOp,
    pub routed: RoutedFfnOp,
    pub pre_experts_norm: NormOp,
    pub post_dense_norm: NormOp,
    pub post_experts_norm: NormOp,
}

/// One layer's FFN: dense, routed, or both. Untagged, so a dense layer
/// serialises exactly as its [`FfnOp`] always has — a dense plan is
/// byte-identical before and after routed and hybrid FFNs existed.
#[derive(Debug, Clone, PartialEq, Serialize)]
#[serde(untagged)]
pub enum LayerFfn {
    /// All boxed: the ops differ several-fold in size and a plan holds one
    /// per layer; the untagged serialisation is unaffected.
    Dense(Box<FfnOp>),
    Routed(Box<RoutedFfnOp>),
    Hybrid(Box<HybridFfnOp>),
}

impl LayerFfn {
    /// The dense op, when this layer's FFN is dense ONLY. A hybrid layer's
    /// dense half is reached through [`Self::hybrid`] — an executor that
    /// ran only the dense half of a hybrid layer would run a different
    /// model, so this does not hand it out.
    pub fn dense(&self) -> Option<&FfnOp> {
        match self {
            Self::Dense(op) => Some(op.as_ref()),
            Self::Routed(_) | Self::Hybrid(_) => None,
        }
    }

    /// The routed op, when this layer's FFN is a mixture of experts ONLY
    /// (same reasoning as [`Self::dense`]).
    pub fn routed(&self) -> Option<&RoutedFfnOp> {
        match self {
            Self::Routed(op) => Some(op.as_ref()),
            Self::Dense(_) | Self::Hybrid(_) => None,
        }
    }

    /// The hybrid op, when this layer runs both branches.
    pub fn hybrid(&self) -> Option<&HybridFfnOp> {
        match self {
            Self::Hybrid(op) => Some(op.as_ref()),
            Self::Dense(_) | Self::Routed(_) => None,
        }
    }
}

/// The generic program of one decoder layer. Norm placement is explicit
/// op positions, not a count: under two-norm placement the
/// `post_attention_layernorm` operand *is* [`Self::pre_ffn_norm`] and the
/// post-positions are absent; under four-norm placement attention and FFN
/// are each wrapped pre + post.
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct LayerPlan {
    pub layer: usize,
    pub pre_attention_norm: NormOp,
    pub attention: AttentionOp,
    /// Normalises attention output before its residual add (four-norm
    /// placement only).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub post_attention_norm: Option<NormOp>,
    pub pre_ffn_norm: NormOp,
    pub ffn: LayerFfn,
    /// Normalises FFN output before its residual add (four-norm only).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub post_ffn_norm: Option<NormOp>,
    /// A learned scalar `[1]` the whole layer output is multiplied by,
    /// after the FFN residual add (Gemma 4 `layer_scalar`). Present iff
    /// the layer ships the operand — an absent scalar is no multiply, not
    /// a multiply by one that a reader may assume.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub layer_scale: Option<OperandRef>,
    /// Operand accounting: consumed by the ops above / present in the
    /// segment for this layer. Closure requires equality.
    pub operands_accounted: usize,
    pub operands_present: usize,
}

/// Embedding lookup.
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct EmbeddingOp {
    pub table: OperandRef,
    /// Weightless normalisation of the looked-up row. `None` = absent.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub norm: Option<larql_models::config::EmbeddingNorm>,
    /// The embedding-scale operation. `None` = the op is absent.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub scale: Option<f32>,
    pub vocab_size: usize,
}

/// The output head.
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct OutputOp {
    pub projection: OperandRef,
    /// The output-multiplier operation. `None` = the op is absent.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub multiplier: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub softcapping: Option<f32>,
}

/// The complete generic program of one component.
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct ComponentOpPlan {
    pub component: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub embedding: Option<EmbeddingOp>,
    pub layers: Vec<LayerPlan>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub final_norm: Option<NormOp>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub output: Option<OutputOp>,
}

/// Why a plan could not be built — each variant names the exact operand
/// or fact, so a refusal is a work item, not a mystery.
#[derive(Debug, Clone, PartialEq, Serialize)]
pub enum ClosureDefect {
    /// The component has no (complete) execution surface.
    MissingSurface { component: String },
    /// The component has no per-layer attention policy table.
    MissingAttentionTable { component: String },
    /// A stack tensor no operand role classifies.
    UnclassifiedOperand { object: String, tensor: String },
    /// An operand classified into a role that lives in another object
    /// kind — an expert operand in the stack, a router in the bank.
    MisplacedOperand {
        object: String,
        tensor: String,
        belongs_in: ObjectKind,
    },
    /// An operand exists whose op the surface does not carry — the
    /// container physically requires a primitive its semantics lack.
    OperandImpliesAbsentOp {
        object: String,
        tensor: String,
        required_primitive: String,
    },
    /// An op the surface/placement implies has no operand.
    MissingOperand { layer: usize, role: OperandRole },
    /// Two tensors classified into the same role of the same layer.
    DuplicateOperand { layer: usize, role: OperandRole },
    /// An operand's stored shape contradicts the surface's geometry.
    GeometryMismatch {
        tensor: String,
        expected: Vec<usize>,
        actual: Vec<usize>,
    },
    /// A non-stack executable object with an unexpected tensor estate.
    ObjectShape { object: String, detail: String },
    /// The structure requires a semantic fact nothing has established.
    ///
    /// Distinct from [`Self::MissingOperand`]: no tensor is absent, and
    /// the plan would build. It would simply execute a value nobody
    /// judged — the failure mode where an identity or inherited default
    /// is numerically plausible but semantically unfounded, so the
    /// program looks executable and is quietly wrong.
    UnjudgedSemantic {
        component: String,
        /// The fact, named as the surface names it.
        fact: String,
        /// The structure that makes it load-bearing.
        required_by: String,
    },
}

impl std::fmt::Display for ClosureDefect {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::MissingSurface { component } => {
                write!(f, "component `{component}` has no complete execution surface")
            }
            Self::MissingAttentionTable { component } => {
                write!(f, "component `{component}` has no per-layer attention policy table")
            }
            Self::UnclassifiedOperand { object, tensor } => {
                write!(f, "unclassified executable operand: {object}/{tensor}")
            }
            Self::MisplacedOperand {
                object,
                tensor,
                belongs_in,
            } => write!(
                f,
                "misplaced operand: {object}/{tensor} belongs in the {} object",
                belongs_in.name()
            ),
            Self::OperandImpliesAbsentOp {
                object,
                tensor,
                required_primitive,
            } => write!(
                f,
                "unrepresented executable operand: {object}/{tensor} — required primitive: {required_primitive}"
            ),
            Self::MissingOperand { layer, role } => {
                write!(f, "layer {layer}: no operand for role {role:?}")
            }
            Self::DuplicateOperand { layer, role } => {
                write!(f, "layer {layer}: two operands claim role {role:?}")
            }
            Self::GeometryMismatch {
                tensor,
                expected,
                actual,
            } => write!(
                f,
                "geometry mismatch: `{tensor}` is {actual:?}, surface implies {expected:?}"
            ),
            Self::ObjectShape { object, detail } => write!(f, "object `{object}`: {detail}"),
            Self::UnjudgedSemantic {
                component,
                fact,
                required_by,
            } => write!(
                f,
                "component `{component}`: {fact} is not judged, and {required_by} requires it"
            ),
        }
    }
}

/// The planning outcome: the plan exists **only** when closure holds.
#[derive(Debug, Serialize)]
pub struct OpPlanOutcome {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub plan: Option<ComponentOpPlan>,
    pub defects: Vec<ClosureDefect>,
}

impl OpPlanOutcome {
    pub fn closed(&self) -> bool {
        self.defects.is_empty()
    }
}
