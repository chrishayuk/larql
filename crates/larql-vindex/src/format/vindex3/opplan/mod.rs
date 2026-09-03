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
pub mod conv_qkv;
pub mod exec;
pub mod gated_delta;
pub mod kda;
pub mod mamba2;
pub mod mla;

#[cfg(test)]
pub(crate) mod tests;

use larql_models::config::{
    Activation, AttentionGateSpec, AttentionSinkSpec, ExpertFormat, ExpertRoutingPolicy,
    GateUpLayout, MoeRouterKind, NormType, ParameterFreeQkNorm, PositionPolicy, QkNormScope,
};
use serde::Serialize;

use super::graph::policy::AttentionSpan;
use super::graph::{ObjectKind, OperandRole};

pub use build::plan_component_ops;
pub use gated_delta::GatedDeltaOp;
pub use kda::KdaOp;
pub use mamba2::Mamba2Op;
pub use mla::MlaOp;

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
/// `i`'s gate projection). No executor reads this arm yet; [`super::exec`]
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
    /// Every expert's gate/up/down projections, packed or per-expert per
    /// [`Self::expert_format`].
    pub bank: ExpertBank,
    /// The always-active shared expert(s), when the judgment declares any.
    /// `None` and `shared_experts == 0` in [`FfnSurface::moe`]'s judgment
    /// agree by construction — see [`build::plan_component_ops`]'s closure
    /// pass, which requires the operand set iff this would be `Some`.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub shared: Option<SharedExpertOp>,
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

impl LayerAttention {
    /// The softmax op, when this layer attends by softmax. `None` on a
    /// DeltaNet or MLA layer — which is the point: a consumer that needs
    /// a span, a KV shape or a head geometry must handle the absence
    /// rather than receive a fabricated one.
    pub fn softmax(&self) -> Option<&AttentionOp> {
        match self {
            Self::Softmax(op) => Some(op.as_ref()),
            Self::GatedDelta(_)
            | Self::Kda(_)
            | Self::Mla(_)
            | Self::Mamba2(_)
            | Self::ConvQkv(_) => None,
        }
    }

    /// [`Self::softmax`], mutably.
    pub fn softmax_mut(&mut self) -> Option<&mut AttentionOp> {
        match self {
            Self::Softmax(op) => Some(op.as_mut()),
            Self::GatedDelta(_)
            | Self::Kda(_)
            | Self::Mla(_)
            | Self::Mamba2(_)
            | Self::ConvQkv(_) => None,
        }
    }

    /// The Gated DeltaNet op, when this layer runs that recurrence.
    ///
    /// `None` on a KDA layer — the two are different operators, and a
    /// consumer that wants "whichever recurrence this is" must ask for
    /// each rather than receive one standing in for the other.
    pub fn gated_delta(&self) -> Option<&GatedDeltaOp> {
        match self {
            Self::GatedDelta(op) => Some(op.as_ref()),
            Self::Softmax(_) | Self::Kda(_) | Self::Mla(_) | Self::Mamba2(_) | Self::ConvQkv(_) => {
                None
            }
        }
    }

    /// The KDA op, when this layer runs Kimi Delta Attention.
    pub fn kda(&self) -> Option<&KdaOp> {
        match self {
            Self::Kda(op) => Some(op.as_ref()),
            Self::Softmax(_)
            | Self::GatedDelta(_)
            | Self::Mla(_)
            | Self::Mamba2(_)
            | Self::ConvQkv(_) => None,
        }
    }

    /// The MLA op, when this layer runs Multi-Latent Attention.
    pub fn mla(&self) -> Option<&MlaOp> {
        match self {
            Self::Mla(op) => Some(op.as_ref()),
            Self::Softmax(_)
            | Self::GatedDelta(_)
            | Self::Kda(_)
            | Self::Mamba2(_)
            | Self::ConvQkv(_) => None,
        }
    }

    /// The Mamba2 op, when this layer runs the SSD mixer.
    pub fn mamba2(&self) -> Option<&Mamba2Op> {
        match self {
            Self::Mamba2(op) => Some(op.as_ref()),
            Self::Softmax(_)
            | Self::GatedDelta(_)
            | Self::Kda(_)
            | Self::Mla(_)
            | Self::ConvQkv(_) => None,
        }
    }

    /// Elements of recurrent state this layer retains, or `None` when it
    /// keeps a per-position cache instead — a softmax OR an MLA layer:
    /// MLA compresses the cache, it does not remove it, so it answers
    /// `None` here exactly as plain softmax does, not a state-elements
    /// count.
    ///
    /// The question a KV planner asks once a stack may hold either
    /// recurrence: both answer in state elements, and neither answers in
    /// positions.
    pub fn recurrent_state_elements(&self) -> Option<usize> {
        match self {
            // Conv-QKV attention keeps a per-position cache (its conv
            // history is a fixed-size extra region, not a recurrence),
            // so it answers as the cache-keeping operators do.
            Self::Softmax(_) | Self::Mla(_) | Self::ConvQkv(_) => None,
            Self::GatedDelta(op) => Some(op.state_elements()),
            Self::Kda(op) => Some(op.state_elements()),
            Self::Mamba2(op) => Some(op.state_elements()),
        }
    }

    /// The `layer_types` spelling this operator corresponds to, for
    /// comparing what the plan carries against what the checkpoint
    /// declared.
    pub fn declared_name(&self) -> &'static str {
        match self {
            Self::Softmax(op) => op.span.declared_name(),
            Self::GatedDelta(_) | Self::Kda(_) | Self::Mamba2(_) => {
                larql_models::config::LAYER_TYPE_LINEAR_ATTENTION
            }
            // The hybrid's index-set spelling names WHICH layers attend;
            // the span vocabulary is its round-trip, as in the graph's
            // identical judgment.
            Self::ConvQkv(_) => AttentionSpan::Full.declared_name(),
            // MLA has no `layer_types` spelling of its own — see
            // `graph::policy::AttentionLayerPolicy::declared_name`'s
            // identical judgment.
            Self::Mla(_) => AttentionSpan::Full.declared_name(),
        }
    }
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
    /// The pre-block norm. On an attention-class layer this is
    /// `input_layernorm`; on a mixer-only layer it is the layer's single
    /// pre-mixer norm — same position in the program, one field.
    ///
    /// `None` under [`NormPlacement::PostOnly`](crate::format::vindex3::graph::NormPlacement):
    /// the sublayer reads the RAW residual there and the norm applies to
    /// its output instead. Absence is the program, not a missing operand
    /// — closure requires this norm's operand exactly when the placement
    /// says the site exists.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub pre_attention_norm: Option<NormOp>,
    /// The component's DECLARED norm epsilon (`rms_norm_eps`).
    ///
    /// Carried on the layer in its own right rather than read back off
    /// [`Self::pre_attention_norm`], because the operators that need it
    /// are not all norms at that site: QK norm, DeltaNet's gated RMSNorm
    /// and KDA's all run at this epsilon. Reading it from one norm SITE
    /// was a coupling between an epsilon and a placement — two unrelated
    /// facts — and it is what made a post-norm stack, which has no
    /// pre-attention norm to borrow from, unrepresentable in the op plan.
    ///
    /// The same reasoning `MlaOp::kv_a_norm_eps` already applies in the
    /// other direction: a norm whose epsilon differs from the layer's
    /// carries its own, and one that shares it reads this.
    pub declared_norm_eps: f64,
    /// This layer's attention-class operator. Not every layer attends by
    /// softmax: a hybrid checkpoint interleaves DeltaNet recurrences with
    /// full-attention layers, so the op is a choice, not a shape.
    pub attention: LayerAttention,
    /// Normalises attention output before its residual add (four-norm
    /// placement only).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub post_attention_norm: Option<NormOp>,
    /// Absent on a mixer-only (Mamba2) layer, which has no FFN to
    /// normalise into — schema 6's presence-means-presence, in the plan.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub pre_ffn_norm: Option<NormOp>,
    /// This layer's FFN, absent on a mixer-only layer (the mixer is the
    /// whole block; no `intermediate_size` exists in the checkpoint).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub ffn: Option<LayerFfn>,
    /// Normalises FFN output before its residual add (four-norm only).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub post_ffn_norm: Option<NormOp>,
    /// A learned scalar `[1]` the whole layer output is multiplied by,
    /// after the FFN residual add (Gemma 4 `layer_scalar`). Present iff
    /// the layer ships the operand — an absent scalar is no multiply, not
    /// a multiply by one that a reader may assume.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub layer_scale: Option<OperandRef>,
    /// Residual-stream scaling: the attention/FFN sublayer's own output
    /// (after any post-norm above) is multiplied by this immediately
    /// before its residual add, at both sites with the same value.
    /// `None` = the op is absent.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub residual_scale: Option<f32>,
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

/// Which FFN a layer runs, as a declaration or as operand evidence.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
pub enum FfnIdentity {
    /// A dense MLP.
    Dense,
    /// A routed expert block (routed alone, or routed-plus-dense hybrid).
    Routed,
}

impl FfnIdentity {
    /// Lower-case name for messages.
    pub const fn name(self) -> &'static str {
        match self {
            Self::Dense => "dense",
            Self::Routed => "routed",
        }
    }
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
    /// The declared FFN schedule and the operand evidence disagree for a
    /// layer. The surface declares which layers are routed
    /// (`dense_prefix_layers`, else every layer of a MoE surface); the
    /// expert bank and router operands are evidence that the declaration
    /// is honoured, never the authority. A routed layer whose bank is
    /// absent used to plan as dense with no defect — a missing expert
    /// bank must not quietly turn a MoE layer into an MLP — and a
    /// declared-dense prefix layer with routed operands is the same
    /// disagreement the other way round.
    FfnIdentityMismatch {
        layer: usize,
        declared: FfnIdentity,
        evidence: FfnIdentity,
    },
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
    /// The surface states the fact exactly, and this build has no
    /// lowering for it.
    ///
    /// The opposite of [`Self::UnjudgedSemantic`] and the distinction
    /// matters: there, nothing established the value and executing would
    /// be guessing; here the value is established and the OP SET is what
    /// is missing. Recognising a shape is not implementing it, and the
    /// two must not read alike — a reader who cannot tell them apart
    /// cannot tell "we do not know what this model does" from "we know
    /// exactly what it does and cannot yet do it".
    ///
    /// Refusing is the point. Every alternative lowering of a shape this
    /// build does not implement is numerically plausible and
    /// semantically wrong, and produces fluent output rather than an
    /// error — the same reason `MoeRouterKind::Sigmoid` and
    /// `PositionPolicy::Relative` refuse at their own match arms.
    UnimplementedSemantic {
        component: String,
        /// The fact, named as the surface names it.
        fact: String,
        /// What the container can already state about it, so a reader
        /// knows the gap is in execution and not in representation.
        representable_as: String,
    },
}

impl std::fmt::Display for ClosureDefect {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::MissingSurface { component } => {
                write!(f, "component `{component}` has no complete execution surface")
            }
            Self::UnimplementedSemantic {
                component,
                fact,
                representable_as,
            } => write!(
                f,
                "component `{component}`: {fact} is representable ({representable_as}) and this                  build has no lowering for it — refused rather than lowered as a shape it is not"
            ),
            Self::MissingAttentionTable { component } => {
                write!(f, "component `{component}` has no per-layer attention policy table")
            }
            Self::UnclassifiedOperand { object, tensor } => {
                write!(f, "unclassified executable operand: {object}/{tensor}")
            }
            Self::FfnIdentityMismatch {
                layer,
                declared,
                evidence,
            } => write!(
                f,
                "layer {layer}: the surface declares a {} FFN but the operands are those of a {} \
                 one — a missing expert bank does not make a routed layer dense, and stray \
                 routed operands do not make a dense-prefix layer routed",
                declared.name(),
                evidence.name()
            ),
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
