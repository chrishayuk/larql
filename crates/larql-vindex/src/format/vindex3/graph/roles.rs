//! Operand roles: the typed vocabulary of what each tensor inside a
//! decoder stack *is* to the generic operations (V3-G5).
//!
//! One definition, three consumers: the surface builder derives
//! norm-placement evidence from it, operand-closure accounting classifies
//! every stack tensor through it, and the operation planner binds kernel
//! arguments by it. A tensor no row classifies is an **unclassified
//! executable operand** — a blocking fact, never a silently skipped file.
//!
//! Placement rule (judged here, once): `post_attention_layernorm` is an
//! overloaded upstream name. In a two-norm layer it normalises the
//! residual stream *before the FFN*; in a four-norm layer (where
//! `pre_feedforward_layernorm` exists) it normalises the *attention
//! output*. Count is not semantics — placement is — so the role table
//! keeps the raw role and [`NormPlacement`] resolves what it means.

use serde::{Deserialize, Serialize};

/// What one decoder-stack tensor is to the generic ops.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum OperandRole {
    AttnQ,
    AttnK,
    AttnV,
    AttnO,
    /// Elementwise gate on attention output — the primitive the
    /// `self_attn.gate_proj` operand implies.
    AttnOutputGate,
    /// Additive bias on the Q/K/V/O projections — present iff the surface
    /// declares `attention_bias`, all four together (GPT-OSS).
    AttnQBias,
    AttnKBias,
    AttnVBias,
    AttnOBias,
    /// Per-query-head attention-sink logits — the operand the judged
    /// [`AttentionSinkSpec`](larql_models::config::AttentionSinkSpec)
    /// consumes.
    AttnSinks,
    AttnQNorm,
    AttnKNorm,
    /// `input_layernorm` — normalises the stream before attention.
    PreAttentionNorm,
    /// `post_attention_layernorm` — before-FFN in a two-norm layer,
    /// attention-output in a four-norm layer (see module docs).
    PostAttentionNorm,
    PreFfnNorm,
    PostFfnNorm,
    FfnGate,
    FfnUp,
    FfnDown,
    /// Router logits `[experts, hidden]` of a routed FFN — lives in the
    /// decoder stack (it is dense).
    MoeRouterWeight,
    /// Additive router bias `[experts]`.
    MoeRouterBias,
    /// Packed expert operands, living in the component's expert-bank
    /// object: the fused gate+up projection of every expert, its
    /// dequantisation scales (formats that keep them apart) and its
    /// per-expert bias; likewise the down projection.
    ExpertGateUp,
    ExpertGateUpScales,
    ExpertGateUpBias,
    ExpertDown,
    ExpertDownScales,
    ExpertDownBias,
    /// Gemma 4's hybrid block (a dense MLP AND a routed expert block in
    /// one layer, outputs summed). The router's learned input scale
    /// `[hidden]` (applied after a scale-less RMS norm of the residual)
    /// and its per-expert scale `[experts]` (applied to the renormalised
    /// top-k weights) live in the decoder stack.
    MoeRouterScale,
    MoeRouterPerExpertScale,
    /// The three FFN-branch norms beyond the pre/post pair: the expert
    /// branch's own pre-norm over the residual, and the post-norms on
    /// each branch's output before they are summed
    /// (`pre_feedforward_layernorm_2`, `post_feedforward_layernorm_1`,
    /// `post_feedforward_layernorm_2`).
    PreExpertsNorm,
    PostDenseFfnNorm,
    PostExpertsNorm,
    /// A per-layer scalar `[1]` the whole layer output is multiplied by
    /// (Gemma 4 `layer_scalar`).
    LayerScalar,
}

impl OperandRole {
    /// Whether this operand lives in the expert-bank object rather than
    /// the decoder stack.
    pub fn is_expert_bank(self) -> bool {
        matches!(
            self,
            Self::ExpertGateUp
                | Self::ExpertGateUpScales
                | Self::ExpertGateUpBias
                | Self::ExpertDown
                | Self::ExpertDownScales
                | Self::ExpertDownBias
        )
    }
}

/// How norms are placed around attention and FFN in every layer of a
/// stack — judged from operand evidence, never from a family default.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum NormPlacement {
    /// Two norms: pre-attention + pre-FFN (`post_attention_layernorm`).
    PreOnly,
    /// Four norms: attention and FFN each wrapped pre + post.
    PrePost,
}

/// Suffix → role. Exact matches on the layer-relative suffix (after
/// `{layer}.`), so a new upstream spelling classifies as *nothing* and
/// blocks, rather than fuzzy-matching into the wrong op.
const ROLE_TABLE: &[(&str, OperandRole)] = &[
    ("self_attn.q_proj.weight", OperandRole::AttnQ),
    ("self_attn.k_proj.weight", OperandRole::AttnK),
    ("self_attn.v_proj.weight", OperandRole::AttnV),
    ("self_attn.o_proj.weight", OperandRole::AttnO),
    ("self_attn.gate_proj.weight", OperandRole::AttnOutputGate),
    ("self_attn.q_proj.bias", OperandRole::AttnQBias),
    ("self_attn.k_proj.bias", OperandRole::AttnKBias),
    ("self_attn.v_proj.bias", OperandRole::AttnVBias),
    ("self_attn.o_proj.bias", OperandRole::AttnOBias),
    ("self_attn.sinks", OperandRole::AttnSinks),
    ("self_attn.q_norm.weight", OperandRole::AttnQNorm),
    ("self_attn.k_norm.weight", OperandRole::AttnKNorm),
    ("input_layernorm.weight", OperandRole::PreAttentionNorm),
    (
        "post_attention_layernorm.weight",
        OperandRole::PostAttentionNorm,
    ),
    ("pre_feedforward_layernorm.weight", OperandRole::PreFfnNorm),
    (
        "post_feedforward_layernorm.weight",
        OperandRole::PostFfnNorm,
    ),
    ("mlp.gate_proj.weight", OperandRole::FfnGate),
    ("mlp.up_proj.weight", OperandRole::FfnUp),
    ("mlp.down_proj.weight", OperandRole::FfnDown),
    ("mlp.router.weight", OperandRole::MoeRouterWeight),
    ("mlp.router.bias", OperandRole::MoeRouterBias),
    // Packed MXFP4 (GPT-OSS): blocks + scales + bias per projection.
    ("mlp.experts.gate_up_proj_blocks", OperandRole::ExpertGateUp),
    (
        "mlp.experts.gate_up_proj_scales",
        OperandRole::ExpertGateUpScales,
    ),
    (
        "mlp.experts.gate_up_proj_bias",
        OperandRole::ExpertGateUpBias,
    ),
    ("mlp.experts.down_proj_blocks", OperandRole::ExpertDown),
    (
        "mlp.experts.down_proj_scales",
        OperandRole::ExpertDownScales,
    ),
    ("mlp.experts.down_proj_bias", OperandRole::ExpertDownBias),
    // Packed BF16 (Gemma 4 A4B): one unquantised operand per projection,
    // in both spellings seen — the checkpoint's own (`experts.…`, no
    // `mlp.` — the experts sit beside the dense `mlp`, not inside it) and
    // the `mlp.experts.…` form.
    ("mlp.experts.gate_up_proj", OperandRole::ExpertGateUp),
    ("mlp.experts.down_proj", OperandRole::ExpertDown),
    ("experts.gate_up_proj", OperandRole::ExpertGateUp),
    ("experts.down_proj", OperandRole::ExpertDown),
    // Gemma 4 hybrid block: router beside the dense mlp, its two scales,
    // the three extra branch norms, and the layer scalar.
    ("router.proj.weight", OperandRole::MoeRouterWeight),
    ("router.scale", OperandRole::MoeRouterScale),
    (
        "router.per_expert_scale",
        OperandRole::MoeRouterPerExpertScale,
    ),
    (
        "pre_feedforward_layernorm_2.weight",
        OperandRole::PreExpertsNorm,
    ),
    (
        "post_feedforward_layernorm_1.weight",
        OperandRole::PostDenseFfnNorm,
    ),
    (
        "post_feedforward_layernorm_2.weight",
        OperandRole::PostExpertsNorm,
    ),
    ("layer_scalar", OperandRole::LayerScalar),
];

/// Classify one stack tensor by its object-relative name
/// (`{layer}.{suffix}`). `None` when the name is not layer-shaped or the
/// suffix matches no judged role — callers treat that as a blocking fact.
pub fn classify_stack_tensor(relative_name: &str) -> Option<(usize, OperandRole)> {
    let (layer, suffix) = relative_name.split_once('.')?;
    let layer: usize = layer.parse().ok()?;
    ROLE_TABLE
        .iter()
        .find(|(name, _)| *name == suffix)
        .map(|(_, role)| (layer, *role))
}

/// Norm placement for a stack, from the roles present across its layers.
///
/// Fail-closed: both FFN-wrap norms or neither; per-layer norms must
/// exist at all. The error names what the evidence actually shows.
pub fn norm_placement_evidence<'a>(
    relative_names: impl Iterator<Item = &'a str>,
) -> Result<NormPlacement, String> {
    let mut pre_attention = false;
    let mut post_attention = false;
    let mut pre_ffn = false;
    let mut post_ffn = false;
    for name in relative_names {
        match classify_stack_tensor(name).map(|(_, role)| role) {
            Some(OperandRole::PreAttentionNorm) => pre_attention = true,
            Some(OperandRole::PostAttentionNorm) => post_attention = true,
            Some(OperandRole::PreFfnNorm) => pre_ffn = true,
            Some(OperandRole::PostFfnNorm) => post_ffn = true,
            _ => {}
        }
    }
    match (pre_attention, post_attention, pre_ffn, post_ffn) {
        (true, true, true, true) => Ok(NormPlacement::PrePost),
        (true, true, false, false) => Ok(NormPlacement::PreOnly),
        (false, false, false, false) => Err("stack carries no per-layer norm operands".to_string()),
        _ => Err(format!(
            "norm operand set is neither two-norm nor four-norm \
             (pre_attn {pre_attention}, post_attn {post_attention}, \
             pre_ffn {pre_ffn}, post_ffn {post_ffn})"
        )),
    }
}
