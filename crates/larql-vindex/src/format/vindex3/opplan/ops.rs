//! The per-operation plan records.

use super::super::graph::policy::AttentionSpan;
use larql_models::config::{
    Activation, AttentionGateSpec, AttentionSinkSpec, ExpertFormat, ExpertRoutingPolicy,
    GateUpLayout, MoeRouterKind, NormType, ParameterFreeQkNorm, PositionPolicy, QkNormScope,
};
use serde::Serialize;

#[allow(unused_imports)]
use super::*;

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
    /// The additive bias after the gain (LayerNorm's affine `bias`),
    /// present iff the surface declares `norm.bias`. Absent from the
    /// serialised op otherwise, so a bias-free plan serialises unchanged.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub bias: Option<OperandRef>,
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
    /// Additive projection biases: all four iff the surface declares
    /// `attention_bias`, Q/K/V alone iff it declares `qkv_bias`. Absent
    /// from the serialised op otherwise, so a bias-free plan serialises
    /// exactly as before.
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
    /// Additive biases after the up projection (before the activation) and
    /// after the down projection, present iff the surface declares
    /// `ffn.bias`. Absent from the serialised op otherwise.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub up_bias: Option<OperandRef>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub down_bias: Option<OperandRef>,
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

/// How one layer's expert weights are bound.
///
/// `Packed` when the checkpoint already fuses every expert's gate+up into
/// one operand (and down into another) — MXFP4 blocks or unquantised BF16.
/// `PerExpert` when it ships three wholly separate tensors PER expert
/// (`ExpertFormat::PerExpert` — Kimi Linear, Mixtral, DeepSeek): there is
/// no fused gate+up operand to name, so the variant carries gate, up and
/// down as three independent lists rather than reusing [`PackedProjection`]
/// with a placeholder — `PackedProjection::weights` is not optional, and a
/// per-expert layer inventing one to fill it is exactly the
/// silently-wrong shape this crate refuses everywhere else.
///
/// `PerExpert`'s lists are ordered by expert index (`gate[i]` is expert
/// `i`'s gate projection). No executor reads this arm yet; [`super::super::exec`]
/// refuses it explicitly rather than misinterpret it as a packed bank of
/// one expert.
#[derive(Debug, Clone, PartialEq, Serialize)]
#[serde(tag = "storage", rename_all = "snake_case")]
pub enum ExpertBank {
    // Boxed: `PerExpert`'s three empty `Vec`s are 24 bytes each, and an
    // unboxed pair of `PackedProjection`s (each an `OperandRef` — a
    // `String` shape/tensor/dtype plus two `Option<OperandRef>`s — nearly
    // 300 bytes) would size every `PerExpert` value at the packed
    // variant's width for no reason.
    Packed {
        gate_up: Box<PackedProjection>,
        down: Box<PackedProjection>,
    },
    PerExpert {
        gate: Vec<OperandRef>,
        up: Vec<OperandRef>,
        down: Vec<OperandRef>,
    },
}

/// The always-active shared expert(s) beside the routed selection —
/// DeepSeek-lineage and Kimi Linear's `shared_experts`. Every token reads
/// this branch; the router's top-k never gates it. Distinct from Gemma 4's
/// hybrid dense MLP ([`HybridFfnOp::dense`]): that branch is summed with
/// its OWN norm pair, this one is summed with the routed branch unscaled
/// (`routed_scaling_factor` multiplies only the routed weights — Kimi's
/// `KimiSparseMoeBlock.forward`: `y = moe(...); y = y +
/// shared_experts(identity)`).
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct SharedExpertOp {
    /// The branch's intermediate width, as the judgment declares it —
    /// `FfnSurface::moe`'s `shared_expert_intermediate_size`. Two
    /// lineages size it differently (Qwen from its own key, DeepSeek and
    /// Kimi as one wider FFN at `moe_intermediate_size * shared_experts`)
    /// and the architecture already chose between them, so this is
    /// transcribed rather than recomputed.
    pub intermediate_size: usize,
    pub activation: Activation,
    pub gate_policy: larql_models::ExpertGatePolicy,
    /// The branch's own SwiGLU gate projection.
    pub gate: OperandRef,
    pub up: OperandRef,
    pub down: OperandRef,
    /// The scalar gate on the branch's OUTPUT, where the family runs one:
    /// `out = routed + sigmoid(branch_gate(x)) * shared(x)`. `None` sums
    /// the branch unscaled, which is the DeepSeek/Kimi form — the two are
    /// different models, not a present-or-defaulted operand, so closure
    /// requires this operand iff the surface declares the gate.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub branch_gate: Option<SharedExpertBranchGateOp>,
}

/// The judged semantics of the shared branch's output gate, beside the
/// operand it reads.
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct SharedExpertBranchGateOp {
    pub spec: larql_models::config::SharedExpertGateSpec,
    pub weight: OperandRef,
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
    /// The declared multiplier on the routed branch's summed output
    /// (`routed_scaling_factor`): applied to the routed sum alone, never
    /// to the shared expert. Absent when the checkpoint declares none,
    /// which executes as 1 — and serialises to nothing, so every other
    /// plan is byte-identical.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub branch_scale: Option<f64>,
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
    /// Gemma 4 (`MoeRouterKind::TopKRenormScaled`) router conditioning: the
    /// residual is RMS-normalised WITHOUT a weight (eps
    /// `router_norm_eps`), multiplied by `router_scale` `[hidden]` and by
    /// `hidden^-0.5`, then projected; the renormalised top-k weights are
    /// multiplied by `router_per_expert_scale[selected]`. Present iff the
    /// router kind is `TopKRenormScaled` (closure-paired). Absent from every
    /// other plan, so those serialise byte-identically.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub router_scale: Option<OperandRef>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub router_per_expert_scale: Option<OperandRef>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub router_norm_eps: Option<f64>,
    /// Every expert's gate/up/down projections, packed or per-expert per
    /// [`Self::expert_format`].
    pub bank: ExpertBank,
    /// The always-active shared expert(s), when the judgment declares any.
    /// `None` and `shared_experts == 0` in [`FfnSurface::moe`]'s judgment
    /// agree by construction — see [`build::plan_component_ops`]'s closure
    /// pass, which requires the operand set iff this would be `Some`.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub shared: Option<SharedExpertOp>,
    /// The bottleneck the ROUTED experts run behind (Kimi-K3). `None` =
    /// the experts consume the block input at `hidden`.
    ///
    /// Deliberately INSIDE the routed op rather than a projection the
    /// caller applies first. Two of this operator's three placement facts
    /// are then structural rather than maintained: the router reads the
    /// op's own input because it never sees anything else, and the shared
    /// branch is summed by the caller after this op returns, so it cannot
    /// enter the bottleneck. Only the norm's placement is left to get
    /// wrong, which is exactly what the oracle found — the other two are
    /// shape-protected there for the same reason they are structural
    /// here.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub latent: Option<LatentBranchOp>,
}

/// The latent routed branch's three operands and its width.
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct LatentBranchOp {
    /// The width the experts run at — the authority their bank is loaded
    /// and shaped against, not `hidden`.
    pub width: usize,
    /// `[width, hidden]`: the block input down to the bottleneck.
    pub down: OperandRef,
    /// `[hidden, width]`: the aggregate back to the residual stream.
    pub up: OperandRef,
    /// The RMSNorm on the WEIGHTED AGGREGATE, between summation and the
    /// up-projection. `None` when the family declares none.
    ///
    /// Its epsilon rides with it rather than being read from the layer,
    /// because "the layer's eps" is a claim about this family that the
    /// two neighbouring MLA norms falsify — they run at a class default
    /// ten times smaller.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub norm: Option<LatentNormOp>,
}

/// The routed-aggregate norm: its weight and its own epsilon.
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct LatentNormOp {
    pub weight: OperandRef,
    pub eps: f64,
}

impl RoutedFfnOp {
    /// The multiplier the executor applies to every routed weight: the
    /// declared branch scale, or exactly 1 when the checkpoint declares
    /// none — a missing declaration is not a zero.
    pub fn executed_branch_scale(&self) -> f32 {
        self.branch_scale.map_or(1.0, |scale| scale as f32)
    }
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

/// One layer's attention-class operator: softmax attention, or a Gated
/// DeltaNet recurrence.
///
/// Untagged for the same reason [`LayerFfn`] is: a softmax layer
/// serialises exactly as its [`AttentionOp`] always has, so every plan
/// written before linear attention existed is byte-identical afterwards.
///
/// Boxed on both arms because the two ops differ several-fold in size and
/// a plan holds one per layer.
#[derive(Debug, Clone, PartialEq, Serialize)]
#[serde(untagged)]
pub enum LayerAttention {
    Softmax(Box<AttentionOp>),
    GatedDelta(Box<GatedDeltaOp>),
    /// Mamba2/SSD — its own variant, a third recurrence family. See
    /// [`mamba2`] for why it shares nothing with the other two.
    Mamba2(Box<Mamba2Op>),
    /// Conv-QKV attention — the hybrid Mamba2Attn stack's attention
    /// block: its own variant, not a `Softmax` with a conv bolted on.
    /// See [`conv_qkv`] for the transcribed forward and the two-region
    /// continuation (KV cache AND conv history).
    ConvQkv(Box<conv_qkv::ConvQkvOp>),
    /// Kimi Delta Attention — its own variant, not a `GatedDelta` with
    /// different numbers. See [`kda`] for why the two cannot share one.
    Kda(Box<KdaOp>),
    /// Multi-Latent Attention — its own variant, not a `Softmax` with
    /// different operand names. See [`mla`] for why the two cannot share
    /// one: the compressed-KV operands have no softmax counterpart, and
    /// the shared `q_proj`/`o_proj` SUFFIXES are a different WIDTH.
    Mla(Box<MlaOp>),
}
