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

use super::policy::LayerOperator;

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

    /// Gated DeltaNet operands. A `linear_attention` layer owns all nine
    /// and none of the `Attn*` roles: there is no query/key/value to
    /// retain, no output gate projection separate from `InProjZ`, and no
    /// span to mask. Closure requires the complete set — a DeltaNet layer
    /// missing one is not a partially-specified attention layer, it is an
    /// operator that cannot run.
    /// Fused query|key|value, `[2·Hk·Dk + Hv·Dv, hidden]`.
    LinearAttnInProjQkv,
    /// Per-value-head decay projection, `[Hv, hidden]`.
    LinearAttnInProjA,
    /// Per-value-head write-strength projection, `[Hv, hidden]`.
    LinearAttnInProjB,
    /// Output-gate projection, `[Hv·Dv, hidden]`.
    LinearAttnInProjZ,
    /// Depthwise causal convolution over the fused q|k|v channels.
    LinearAttnConv1d,
    /// Per-value-head log decay, `[Hv]`.
    LinearAttnALog,
    /// Per-value-head timestep bias, `[Hv]`.
    LinearAttnDtBias,
    /// Gated RMSNorm weight over one value head's width, `[Dv]`.
    LinearAttnNorm,
    /// Output projection, `[hidden, Hv·Dv]`.
    LinearAttnOutProj,
    /// Kimi Delta Attention operands. Fifteen, sharing **no** role with
    /// the Gated DeltaNet set above and only their *spelling* with the
    /// softmax set: on a KDA layer `self_attn.q_proj.weight` is the
    /// recurrence's query projection at `[Hv·Dv, hidden]`, and on a
    /// full-attention layer of the same checkpoint it is the softmax
    /// query at a different width. `self_attn.o_proj.weight` is worse —
    /// on Kimi Linear it is byte-identical in shape, `[2304, 4096]`, on
    /// both. Neither the name nor the shape separates them; only the
    /// layer's operator does, which is why role classification is
    /// layer-aware (see [`classify_stack_tensor_on`]).
    /// Query projection, `[Hv·Dv, hidden]`.
    KdaQProj,
    /// Key projection, `[Hv·Dv, hidden]`.
    KdaKProj,
    /// Value projection, `[Hv·Dv, hidden]`.
    KdaVProj,
    /// Depthwise causal conv over the query channels, `[Hv·Dv, 1, kernel]`.
    KdaQConv1d,
    /// Depthwise causal conv over the key channels.
    KdaKConv1d,
    /// Depthwise causal conv over the value channels.
    KdaVConv1d,
    /// Decay-gate down-projection, `[rank, hidden]` — the f gate's first
    /// factor. Low-rank by construction; Gated DeltaNet has no analogue.
    KdaFAProj,
    /// Decay-gate up-projection, `[Hv·Dv, rank]`.
    KdaFBProj,
    /// Output-gate down-projection, `[rank, hidden]`.
    KdaGAProj,
    /// Output-gate up-projection, `[Hv·Dv, rank]`.
    KdaGBProj,
    /// Per-head write-strength projection, `[Hv, hidden]`.
    KdaBProj,
    /// Per-head log decay, `[Hv]`.
    KdaALog,
    /// Per-**channel** timestep bias, `[Hv·Dv]`. The single operand whose
    /// geometry most sharply separates KDA from Gated DeltaNet, whose
    /// `dt_bias` is `[Hv]`.
    KdaDtBias,
    /// Gated RMSNorm weight over one head's width, `[Dv]`.
    KdaONorm,
    /// Output projection, `[hidden, Hv·Dv]`.
    KdaOutProj,
    /// Multi-Latent Attention operands. Five, sharing only their
    /// *spelling* — never their shape — with the softmax set: on an MLA
    /// layer `self_attn.q_proj.weight` is `[heads·(nope+rope), hidden]`,
    /// wider than a softmax layer's `[heads·head_dim, hidden]`, and
    /// `self_attn.o_proj.weight` collides the same way `[hidden,
    /// heads·v_head_dim]` at Kimi's own asymmetric v_head_dim. Neither
    /// name nor a single per-head width separates them; only the layer's
    /// operator does — see [`classify_stack_tensor_on`].
    ///
    /// Query projection, fused nope+rope per head, `[Hq·(nope+rope), hidden]`.
    MlaQProj,
    /// Shared (MQA-style) compressed KV projection: latent + one rope-K,
    /// `[kv_lora_rank + rope, hidden]`.
    MlaKvAProj,
    /// KV decompression: nope-K and V per head, fused,
    /// `[Hq·(nope+v_head_dim), kv_lora_rank]`.
    MlaKvBProj,
    /// RMSNorm weight over the compressed KV latent, applied before
    /// decompression, `[kv_lora_rank]`.
    MlaKvANorm,
    /// Output projection, `[hidden, Hq·v_head_dim]`.
    MlaOutProj,
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
    /// One `ExpertFormat::PerExpert` expert's own gate/up/down projection —
    /// the checkpoint ships `experts` separate tensors per role rather than
    /// one fused bank tensor, so the role carries the index that separates
    /// expert 3's `w1` from expert 200's. Named `PerExpert*` rather than
    /// reusing [`Self::ExpertGateUp`]/[`Self::ExpertDown`] because those
    /// names are already taken by the packed-format unit roles they sit
    /// beside; lives in the expert-bank object like they do ([`Self::
    /// is_expert_bank`]), but as `experts` distinct bindings per role
    /// instead of one — [`super::super::opplan::ExpertBank`] is the
    /// typed choice between the two storage shapes downstream.
    ///
    /// Kimi Linear's `w1`/`w3`/`w2` respectively — checked against
    /// `KimiBlockSparseMLP.forward` (`modeling_kimi.py`), not guessed from
    /// the names: `w1`/`w3` feed the gated product, `w2` reads it.
    PerExpertGate(u16),
    PerExpertUp(u16),
    PerExpertDown(u16),
    /// Always-active expert(s) alongside the routed ones — DeepSeek-lineage
    /// and Kimi's `shared_experts`. A distinct branch from both the routed
    /// bank (every token reads it, not just the router's top-k) and Gemma
    /// 4's hybrid dense MLP ([`Self::FfnGate`]/[`Self::FfnUp`]/
    /// [`Self::FfnDown`], summed with the ROUTED branch unscaled while
    /// Gemma 4's dense branch is summed with a separately-normed one) —
    /// conflating the two under one role vocabulary would silently pick
    /// the wrong combination rule. Lives in the decoder stack: it runs on
    /// every token, so it is not part of the per-token-selected bank.
    SharedExpertGate,
    SharedExpertUp,
    SharedExpertDown,
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
                | Self::PerExpertGate(_)
                | Self::PerExpertUp(_)
                | Self::PerExpertDown(_)
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
    // Gated DeltaNet (Qwen3.8 `linear_attention` layers). Nine operands,
    // sharing nothing with the softmax set above: the recurrence has no
    // per-position key or value to retain, so none of the Attn* roles
    // apply. Exact suffixes, like every entry here — a DeltaNet layer's
    // `linear_attn.norm.weight` must never be mistaken for a decoder norm.
    (
        "linear_attn.in_proj_qkv.weight",
        OperandRole::LinearAttnInProjQkv,
    ),
    (
        "linear_attn.in_proj_a.weight",
        OperandRole::LinearAttnInProjA,
    ),
    (
        "linear_attn.in_proj_b.weight",
        OperandRole::LinearAttnInProjB,
    ),
    (
        "linear_attn.in_proj_z.weight",
        OperandRole::LinearAttnInProjZ,
    ),
    ("linear_attn.conv1d.weight", OperandRole::LinearAttnConv1d),
    ("linear_attn.A_log", OperandRole::LinearAttnALog),
    ("linear_attn.dt_bias", OperandRole::LinearAttnDtBias),
    ("linear_attn.norm.weight", OperandRole::LinearAttnNorm),
    (
        "linear_attn.out_proj.weight",
        OperandRole::LinearAttnOutProj,
    ),
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
    // Kimi Linear: router beside its own component name (not `mlp.`), its
    // bias-corrected-selection tensor kept apart from the router weight
    // (see module docs on `MoeRouterBias`), and the always-active shared
    // expert under the same component.
    ("block_sparse_moe.gate.weight", OperandRole::MoeRouterWeight),
    (
        "block_sparse_moe.gate.e_score_correction_bias",
        OperandRole::MoeRouterBias,
    ),
    (
        "block_sparse_moe.shared_experts.gate_proj.weight",
        OperandRole::SharedExpertGate,
    ),
    (
        "block_sparse_moe.shared_experts.up_proj.weight",
        OperandRole::SharedExpertUp,
    ),
    (
        "block_sparse_moe.shared_experts.down_proj.weight",
        OperandRole::SharedExpertDown,
    ),
];

/// One leaf spelling after the expert index (`"w1.weight"`) and the role
/// constructor it binds.
type IndexedExpertLeaf = (&'static str, fn(u16) -> OperandRole);

/// One `ExpertFormat::PerExpert` family's indexed-operand spelling: the
/// fixed text surrounding the expert index, and which role each of the
/// family's per-expert leaf names maps to.
///
/// A second vocabulary beside [`ROLE_TABLE`] rather than an attempt to fold
/// indexed suffixes into it, because [`ROLE_TABLE`] matches by exact
/// string equality — the whole point of it never fuzzy-matching a new
/// spelling into the wrong role — and an expert index is exactly the kind
/// of value no static string can stand in for.
struct IndexedExpertFamily {
    /// Text before the expert index, including its trailing `.`
    /// (`"block_sparse_moe.experts."`).
    prefix: &'static str,
    /// This family's leaf spellings.
    leaves: &'static [IndexedExpertLeaf],
}

/// Every `ExpertFormat::PerExpert` family this build recognises.
///
/// One entry today — Kimi Linear's `w1`/`w2`/`w3`, checked against
/// `KimiBlockSparseMLP.forward` in the checkpoint's `modeling_kimi.py`
/// (`w1`/`w3` feed the gated product, `w2` reads it — NOT alphabetic
/// gate/up/down order). A second `PerExpert` family (Mixtral's
/// `experts.{id}.w1/w2/w3` is the SAME leaf spelling under a different
/// prefix; DeepSeek's `experts.{id}.gate_proj/up_proj/down_proj` is not)
/// adds its own entry here, never a guess from this one.
const INDEXED_EXPERT_FAMILIES: &[IndexedExpertFamily] = &[IndexedExpertFamily {
    prefix: "block_sparse_moe.experts.",
    leaves: &[
        ("w1.weight", OperandRole::PerExpertGate),
        ("w3.weight", OperandRole::PerExpertUp),
        ("w2.weight", OperandRole::PerExpertDown),
    ],
}];

/// Classify a layer-relative suffix as one `PerExpert`-format family's
/// indexed operand, or `None` — never a guess: the prefix must match
/// exactly, the segment between prefix and leaf must be *only* decimal
/// digits (so `experts.10.w1.weight` cannot be mistaken for `experts.1`'s
/// `0.w1.weight`, and a non-numeric segment refuses rather than silently
/// classifying), and the leaf must match one of the family's declared
/// spellings exactly.
fn classify_indexed_expert(suffix: &str) -> Option<OperandRole> {
    for family in INDEXED_EXPERT_FAMILIES {
        let Some(rest) = suffix.strip_prefix(family.prefix) else {
            continue;
        };
        let Some((index, leaf)) = rest.split_once('.') else {
            continue;
        };
        if index.is_empty() || !index.bytes().all(|b| b.is_ascii_digit()) {
            continue;
        }
        let Ok(expert_id) = index.parse::<u16>() else {
            continue;
        };
        if let Some((_, role)) = family.leaves.iter().find(|(name, _)| *name == leaf) {
            return Some(role(expert_id));
        }
    }
    None
}

/// Classify one stack tensor by its object-relative name
/// (`{layer}.{suffix}`). `None` when the name is not layer-shaped or the
/// suffix matches no judged role — callers treat that as a blocking fact.
/// Suffix → role **on a KDA layer**, consulted before [`ROLE_TABLE`].
///
/// Only the suffixes KDA claims are listed; everything else (norms, FFN,
/// router) falls through, because those mean the same thing whatever the
/// attention operator is. Five of these collide with softmax spellings and
/// are the reason this table exists.
const KDA_ROLE_TABLE: &[(&str, OperandRole)] = &[
    // Collides with the softmax set — same suffix, different operator.
    ("self_attn.q_proj.weight", OperandRole::KdaQProj),
    ("self_attn.k_proj.weight", OperandRole::KdaKProj),
    ("self_attn.v_proj.weight", OperandRole::KdaVProj),
    ("self_attn.o_proj.weight", OperandRole::KdaOutProj),
    // KDA-only spellings.
    ("self_attn.q_conv1d.weight", OperandRole::KdaQConv1d),
    ("self_attn.k_conv1d.weight", OperandRole::KdaKConv1d),
    ("self_attn.v_conv1d.weight", OperandRole::KdaVConv1d),
    ("self_attn.f_a_proj.weight", OperandRole::KdaFAProj),
    ("self_attn.f_b_proj.weight", OperandRole::KdaFBProj),
    ("self_attn.g_a_proj.weight", OperandRole::KdaGAProj),
    ("self_attn.g_b_proj.weight", OperandRole::KdaGBProj),
    ("self_attn.b_proj.weight", OperandRole::KdaBProj),
    ("self_attn.A_log", OperandRole::KdaALog),
    ("self_attn.dt_bias", OperandRole::KdaDtBias),
    ("self_attn.o_norm.weight", OperandRole::KdaONorm),
];

/// Suffix → role **on an MLA layer**, consulted before [`ROLE_TABLE`] —
/// the same reason [`KDA_ROLE_TABLE`] exists: `q_proj`/`o_proj` collide in
/// SPELLING with the softmax set at a DIFFERENT shape, so only the
/// layer's operator can tell them apart.
const MLA_ROLE_TABLE: &[(&str, OperandRole)] = &[
    // Collides with the softmax set — same suffix, different geometry.
    ("self_attn.q_proj.weight", OperandRole::MlaQProj),
    ("self_attn.o_proj.weight", OperandRole::MlaOutProj),
    // MLA-only spellings.
    (
        "self_attn.kv_a_proj_with_mqa.weight",
        OperandRole::MlaKvAProj,
    ),
    ("self_attn.kv_b_proj.weight", OperandRole::MlaKvBProj),
    ("self_attn.kv_a_layernorm.weight", OperandRole::MlaKvANorm),
];

/// Classify one stack tensor, given the operator its layer runs.
///
/// The operator is required, not optional, because a name alone cannot
/// answer for every checkpoint: Kimi Linear's `self_attn.o_proj.weight` is
/// `[2304, 4096]` on its KDA layers, `[2304, 4096]` on its MLA layers
/// (heads·v_head_dim = 32·128 = 4096, coincidentally equal to KDA's width
/// on THIS checkpoint — a shape check alone would not even prove the
/// two apart here), and would answer a THIRD width on a genuine softmax
/// layer. The graph's per-layer table is the only authority that can
/// separate them, which is what makes the interleave carriage (P3a) a
/// precondition for KDA/MLA operand binding rather than a nicety.
pub fn classify_stack_tensor_on(
    relative_name: &str,
    operator: LayerOperator,
) -> Option<(usize, OperandRole)> {
    let (layer, suffix) = relative_name.split_once('.')?;
    let layer: usize = layer.parse().ok()?;
    if operator.is_kda() {
        if let Some((_, role)) = KDA_ROLE_TABLE.iter().find(|(name, _)| *name == suffix) {
            return Some((layer, *role));
        }
    }
    if operator.is_mla() {
        if let Some((_, role)) = MLA_ROLE_TABLE.iter().find(|(name, _)| *name == suffix) {
            return Some((layer, *role));
        }
    }
    if let Some(role) = ROLE_TABLE
        .iter()
        .find(|(name, _)| *name == suffix)
        .map(|(_, role)| *role)
    {
        return Some((layer, role));
    }
    // Indexed `PerExpert` operands last: every fixed spelling above is
    // tried first, so an exact-match family entry always wins over the
    // pattern-matcher on the same suffix.
    classify_indexed_expert(suffix).map(|role| (layer, role))
}

/// [`classify_stack_tensor_on`] for a layer running softmax attention.
///
/// Correct only for roles that cannot collide across operators — the norms
/// this crate's norm-placement evidence reads. Do **not** reach for it to
/// classify an attention operand: on a KDA layer it answers `AttnQ` for
/// the recurrence's query projection.
pub fn classify_stack_tensor(relative_name: &str) -> Option<(usize, OperandRole)> {
    classify_stack_tensor_on(relative_name, LayerOperator::Softmax)
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
