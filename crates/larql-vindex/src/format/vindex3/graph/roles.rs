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
    /// The output gate's FULL-RANK form, `[Hv·Dv, hidden]` — one
    /// projection where [`Self::KdaGAProj`]/[`Self::KdaGBProj`] are two.
    /// Kimi-K3 declares it (`linear_attn_config.use_full_rank_gate`) and
    /// ships no pair. Only the gate's projection differs between the
    /// forms; the sigmoid and the gated norm do not. Spelled
    /// `self_attn.g_proj.weight` — the SAME spelling and, on K3, the same
    /// shape as [`Self::MlaOutputGate`]; the layer's operator is what
    /// separates them, exactly as it separates `o_proj`.
    KdaGProj,
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
    ///
    /// Present only under [`MlaQueryForm::Direct`]; a layer that
    /// factorises its query has no `q_proj` at all, and closure refuses
    /// one that ships both.
    ///
    /// [`MlaQueryForm::Direct`]: larql_models::config::MlaQueryForm::Direct
    MlaQProj,
    /// Query DOWN-projection, `[q_lora_rank, hidden]` — Kimi-K3's
    /// factorised query (`q_lora_rank: 1536`).
    MlaQAProj,
    /// RMSNorm weight over the compressed query latent, applied between
    /// the down- and up-projections, `[q_lora_rank]`.
    ///
    /// Its epsilon is the family's own and is NOT the layer's
    /// `rms_norm_eps` — nor, though the numbers agree today, is it
    /// derived from [`Self::MlaKvANorm`]'s.
    MlaQANorm,
    /// Query UP-projection, `[Hq·(nope+rope), q_lora_rank]`.
    ///
    /// **Same ROW count as [`Self::MlaQProj`]** — `Hq·(nope+rope)` is
    /// 18432 on K3 either way — and a different COLUMN count: the rank
    /// against `hidden`. Anything discriminating these two by rows
    /// accepts either for the other, so the column count is what
    /// closure checks and the declared form is what says which to
    /// expect.
    MlaQBProj,
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
    /// The output gate's projection, `[Hq·v_head_dim, hidden]`, on a
    /// family that declares `mla_use_output_gate` (Kimi-K3):
    /// `sigmoid(g_proj(x)) ⊙ attn_value` before `o_proj`, the same
    /// generic operation [`Self::AttnOutputGate`] is for the softmax
    /// family. Spelled `self_attn.g_proj.weight`, colliding with
    /// [`Self::KdaGProj`] in name and (on K3) in shape.
    MlaOutputGate,
    /// Mamba2/SSD mixer operands. Nine, sharing nothing with any set
    /// above: one fused five-way input projection where DeltaNet splits
    /// qkv|a|b|z and KDA splits q|k|v entirely; a conv that runs over the
    /// x|B|C channels ONLY (the gate channels are deliberately excluded,
    /// where DeltaNet convolves its full fused projection); per-**head**
    /// scalar decay/skip/timestep against KDA's per-channel `dt_bias`.
    ///
    /// Fused input projection z|x|B|C|dt,
    /// `[2·d_inner + 2·groups·state + heads, hidden]`.
    Mamba2InProj,
    /// Depthwise causal conv over x|B|C, `[conv_dim, 1, kernel]`.
    Mamba2Conv1d,
    /// Conv bias `[conv_dim]` — required iff `use_conv_bias`.
    Mamba2Conv1dBias,
    /// Per-head log decay, `[heads]`.
    Mamba2ALog,
    /// Per-head skip weight, `[heads]`.
    Mamba2D,
    /// Per-head timestep bias, `[heads]` — the geometry that separates
    /// this family from KDA's per-channel `[Hv·Dv]`.
    Mamba2DtBias,
    /// Gated RMSNorm over the full inner width between state read-out and
    /// the output projection, `[d_inner]` — present iff `rms_norm`.
    Mamba2GatedNorm,
    /// Output projection, `[hidden, d_inner]`.
    Mamba2OutProj,
    /// Conv-QKV attention operands (the hybrid Mamba2Attn stack's
    /// attention block). Four, colliding with the MAMBA2 set in
    /// SPELLING at different shapes — `mixer.in_proj.weight` is
    /// `[(Hq+2·Hkv)·Dh, hidden]` here against the mixer's five-way
    /// fusion — so only the layer's operator can tell them apart.
    ///
    /// Fused QKV projection q|k|v, `[(Hq + 2·Hkv)·Dh, hidden]`.
    ConvQkvInProj,
    /// Depthwise causal conv over the FULL fused QKV (no activation),
    /// `[(Hq + 2·Hkv)·Dh, 1, kernel]`.
    ConvQkvConv1d,
    /// Conv bias `[(Hq + 2·Hkv)·Dh]` — required iff `use_conv_bias`.
    ConvQkvConv1dBias,
    /// Output projection, `[hidden, Hq·Dh]`.
    ConvQkvOutProj,
    /// The single pre-mixer norm of a mixer-only layer
    /// (`backbone.layers.N.norm.weight`), `[hidden]`. Its own role rather
    /// than [`Self::PreAttentionNorm`]: a mixer-only stack has ONE norm
    /// per layer, and folding it into the attention vocabulary would let
    /// a transformer stack missing its FFN norms read as a valid
    /// mixer placement.
    Mamba2PreMixerNorm,
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
    /// The scalar gate on the shared branch's OUTPUT — Qwen MoE's
    /// `shared_expert_gate`, a `[1, hidden]` projection whose sigmoid
    /// scales the whole branch before it is summed with the routed one.
    ///
    /// Not [`Self::SharedExpertGate`], which is the branch's own SwiGLU
    /// gate projection at `[shared_expert_intermediate_size, hidden]`.
    /// The two live one name apart in the checkpoint
    /// (`mlp.shared_expert_gate.weight` against
    /// `mlp.shared_expert.gate_proj.weight`) and differ in every
    /// dimension; binding either to the other's role would load a
    /// 5632-row projection where one row is read, and produce plausible
    /// output while gating nothing.
    SharedExpertBranchGate,
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
    /// Sinkhorn hyper-connection SITE operands (wave 18). A layer whose
    /// component declares `ResidualTopology::HyperConnection` wraps each
    /// of its two sublayers in one site, and a site owns three operands
    /// that the five wave-17 stages read:
    ///
    /// ```text
    /// mix_fn   [(2 + hc)·hc, hc·hidden]   stage 1, the dynamic mix
    ///                                     projection over the FLATTENED
    ///                                     bundle
    /// base     [(2 + hc)·hc]              stage 2, the logits' additive
    ///                                     offset before the split
    /// scale    [3]                        stage 2, one scalar each for
    ///                                     the pre, post and combination
    ///                                     logits
    /// ```
    ///
    /// The role carries **no dtype**. DeepSeek-V4 stores `hc_attn_fn` as
    /// F32 and GLM-5.3-Flash as BF16, and both are the same operand; a
    /// role that pinned a dtype would have confused the semantic with the
    /// physical, and REPRESENT would inherit the mistake. The geometry IS
    /// the role's: `hc` comes from the component's declared topology, so
    /// Hy4-preview's `[2·hc, hc·hidden]` Sinkhorn-free form cannot bind
    /// here even if its spelling ever matched.
    ///
    /// The head's own reduction (`hc_head_{fn,base,scale}`) is NOT a
    /// stack operand — it is not layer-shaped and it is a different
    /// operation — see [`HcHeadOperand`].
    HcAttnMixFn,
    HcAttnBase,
    HcAttnScale,
    HcFfnMixFn,
    HcFfnBase,
    HcFfnScale,

    /// Attention-residual SITE operands (K3-ATTNRES-1). A layer whose
    /// component declares
    /// [`ResidualTopology::AttentionResidual`](larql_models::config::ResidualTopology::AttentionResidual)
    /// carries two sites — one before attention, one before the FFN —
    /// and each owns a PAIR:
    ///
    /// ```text
    /// norm   [hidden]        the score vector's first factor
    /// proj   [1, hidden]     its second — ONE row, not a matrix
    /// ```
    ///
    /// `_apply_attn_res` multiplies the two elementwise into a single
    /// learned score vector, dots it against the RMS-normalised
    /// candidates and softmaxes over them. There is no query and no
    /// per-token projection of the state, which is why the pair is two
    /// stored vectors rather than a mix projection.
    ///
    /// **Not aliases of the generic norm role.** A `[hidden]` tensor
    /// classified as `PreAttentionNorm` would be applied to the branch
    /// input by an executor that reads norms; this one is half of a
    /// score and is never applied as a norm at all.
    ///
    /// **Not hyper-connection sites either**, and no stream count makes
    /// them so: a Sinkhorn site's mix is `[(2 + hc)·hc, hc·hidden]`,
    /// which is `[1, hidden]` for no `hc`, and its base is
    /// `[(2 + hc)·hc]`, never `[hidden]`. Pinned in
    /// `opplan::tests::wave18_hc_carriage::k3s_residual_operands_are_not_sinkhorn_sites_under_any_stream_count`.
    ///
    /// These roles are reachable ONLY through
    /// [`classify_stack_tensor_under`], which is given the component's
    /// declared topology. [`classify_stack_tensor_on`] — the
    /// operator-only classifier — answers `None` for their spellings on
    /// every operator, so a checkpoint that ships the operands without
    /// declaring the period never acquires the topology from its tensor
    /// names. Identity is declared, not inferred from operands.
    AttnResAttentionNorm,
    AttnResAttentionProj,
    AttnResMlpNorm,
    AttnResMlpProj,
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
    /// Two norms, both on the sublayer's OUTPUT: the attention and FFN
    /// blocks each read the raw residual and their result is normalised
    /// before the add.
    ///
    /// ```text
    /// residual = h
    /// h = attn(h)                      // no pre-norm
    /// h = post_attention_layernorm(h)  // the norm sees the sublayer OUTPUT
    /// h = residual + h                 // ...before the add
    /// ```
    ///
    /// Transcribed from `Olmo2DecoderLayer.forward`, and identical line
    /// for line in `Olmo3DecoderLayer.forward` and
    /// `Exaone4DecoderLayer.forward`.
    ///
    /// **Not classic post-LN** (`h = norm(residual + attn(h))`, the norm
    /// after the add), and not [`Self::PreOnly`] with the norms renamed.
    /// All three place a norm somewhere around a sublayer; they differ in
    /// which tensor it sees, and reading one as another produces fluent
    /// wrong output rather than a failure. That is why this is a variant
    /// recognised from operand evidence rather than a flag.
    ///
    /// The operand evidence is unambiguous: `post_attention_layernorm`
    /// AND `post_feedforward_layernorm` with NO `input_layernorm`. A
    /// two-norm Llama stack carries `input_layernorm` and overloads
    /// `post_attention_layernorm` as its pre-FFN norm, and it has no
    /// `post_feedforward_layernorm` at all.
    PostOnly,
    /// One norm: the pre-mixer norm of a mixer-only (pure-SSM) layer.
    /// There is no FFN and no attention to wrap, so neither existing
    /// placement describes it — and reading a one-norm layer as a broken
    /// two-norm one is exactly the misreading its own variant prevents.
    PreMixer,
}

impl NormPlacement {
    /// Why this build cannot lower this placement, when it cannot.
    ///
    /// **One authority.** The op plan refuses on it. The plan report read
    /// it too, naming a refused placement as an unsupported component so
    /// the two could never disagree — a report that calls a component
    /// admissible while the op plan refuses to build it is the
    /// looks-supported failure in its purest form — and stopped in wave
    /// 19, when no judged placement (nor topology) refused any more and
    /// the reader had nothing left to read. A variant that returns `Some`
    /// again must bring that reader back beside it.
    ///
    /// `None` is a claim, not an absence: this build lowers the
    /// placement and an executor reads its operands.
    pub fn unimplemented_reason(self) -> Option<&'static str> {
        match self {
            // Every judged placement lowers. `PostOnly` joined them in
            // wave 12: the generic executor already applied the wrap
            // norms to each sublayer's OUTPUT before the residual add,
            // and what it lacked was the ability to run with NO
            // pre-sublayer norm. It has that now, on both the batch and
            // the decode path, and the epsilon its QK norm runs at moved
            // off the pre-norm site onto the layer's own declared field.
            Self::PreOnly | Self::PrePost | Self::PreMixer | Self::PostOnly => None,
        }
    }
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
    // LFM2's spelling of the same two-norm estate. `Lfm2DecoderLayer.
    // forward` is `residual = h; h = mixer(operator_norm(h)); h = h +
    // residual; h = h + feed_forward(ffn_norm(h))` — structurally the
    // two-norm PRE-only stack, with the mixer being attention on the
    // layers named in `full_attn_idxs` and a short convolution
    // elsewhere.
    //
    // `ffn_norm` binds to `PostAttentionNorm` and that reads oddly until
    // you know the rule this module already states: in a TWO-norm layer
    // `post_attention_layernorm` IS the pre-FFN norm, and the role keeps
    // the historical name. Binding LFM2's honestly-named `ffn_norm` to
    // the honestly-named `PreFfnNorm` would instead resolve the estate
    // as a partial FOUR-norm stack and refuse it.
    ("operator_norm.weight", OperandRole::PreAttentionNorm),
    ("ffn_norm.weight", OperandRole::PostAttentionNorm),
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
    // Qwen MoE: the branch is singular (`shared_expert`) where the
    // DeepSeek/Kimi lineage spells it `shared_experts`, and it carries a
    // scalar output gate the other lineage has no operand for.
    (
        "mlp.shared_expert.gate_proj.weight",
        OperandRole::SharedExpertGate,
    ),
    (
        "mlp.shared_expert.up_proj.weight",
        OperandRole::SharedExpertUp,
    ),
    (
        "mlp.shared_expert.down_proj.weight",
        OperandRole::SharedExpertDown,
    ),
    (
        "mlp.shared_expert_gate.weight",
        OperandRole::SharedExpertBranchGate,
    ),
    // Sinkhorn hyper-connection sites, as DeepSeek-V4 and GLM-5.3-Flash
    // both spell them — read from the checkpoints' own safetensors
    // headers, not from a reference implementation. Bare leaves with no
    // `.weight`: `layers.N.hc_attn_fn` on DeepSeek,
    // `model.language_model.layers.N.hc_attn_fn` on GLM, and the stack
    // prefix is what differs between the two, never the suffix. Layer-
    // blind, like the norms: a KDA layer and an MLA layer of the same
    // GLM stack each carry all six.
    //
    // Hy4-preview's `hc_attn_layer.hc_pre.hc_fn` is deliberately NOT
    // here. It spells a Sinkhorn-free topology this build has not
    // judged (HC-PREPOST), and binding it to a Sinkhorn role would be
    // matching on the substring `hc_` rather than on the operation.
    ("hc_attn_fn", OperandRole::HcAttnMixFn),
    ("hc_attn_base", OperandRole::HcAttnBase),
    ("hc_attn_scale", OperandRole::HcAttnScale),
    ("hc_ffn_fn", OperandRole::HcFfnMixFn),
    ("hc_ffn_base", OperandRole::HcFfnBase),
    ("hc_ffn_scale", OperandRole::HcFfnScale),
];

/// Suffix → role **under the attention-residual topology**, consulted
/// before every other table by [`classify_stack_tensor_under`] and by
/// NOTHING else.
///
/// K3 spells all four as `.weight` leaves directly under the layer
/// (`language_model.model.layers.{L}.self_attention_res_norm.weight`),
/// read from the checkpoint's own safetensors headers rather than from a
/// reference implementation. The stack prefix is what differs between
/// checkpoints of this dialect, never the suffix — the same argument the
/// hyper-connection site spellings carry.
///
/// The table is gated on the DECLARATION rather than on the operator,
/// which is the one structural difference from [`KDA_ROLE_TABLE`] and
/// friends. The operator cannot answer here: K3 carries these four on
/// its KDA layers and its MLA layers alike, and a softmax stack of the
/// same dialect would carry them too. What decides whether they are
/// site operands is whether the component declares
/// `attn_res_block_size` — and a build that read the topology off the
/// names instead would let any checkpoint acquire a residual programme
/// by spelling.
const ATTENTION_RESIDUAL_ROLE_TABLE: &[(&str, OperandRole)] = &[
    (
        "self_attention_res_norm.weight",
        OperandRole::AttnResAttentionNorm,
    ),
    (
        "self_attention_res_proj.weight",
        OperandRole::AttnResAttentionProj,
    ),
    ("mlp_res_norm.weight", OperandRole::AttnResMlpNorm),
    ("mlp_res_proj.weight", OperandRole::AttnResMlpProj),
];

/// The attention-residual EXIT's two tensor groups, as K3 spells them
/// (`language_model.model.output_attn_res_{norm,proj}`).
///
/// Held here, beside the roles, because two independent consumers ask
/// about them and must never disagree: the graph builder's placement
/// vocabulary matches these as name fragments (ordered BEFORE its
/// generic `norm` fragment, which otherwise swallows the exit norm into
/// the component's final-norm object), and the op plan classifies the
/// object's tensors through [`ATTENTION_RESIDUAL_EXIT_TABLE`].
pub const ATTENTION_RESIDUAL_EXIT_LEAVES: &[&str] =
    &["output_attn_res_norm", "output_attn_res_proj"];

/// The attention-residual exit's own operands — a component-level
/// object, not a stack operand.
///
/// The stack's last layer leaves a prefix sum and a history of
/// snapshots; `_apply_output_attn_res` reduces them to the ONE vector
/// the final norm and the head read. Its pair has a site's geometry and
/// a site's arithmetic, and is still not a site: it runs once, at the
/// stack's end, over the whole history, and a component declaring the
/// topology must ship it (K3 does). That is why it is its own object
/// with its own closure rather than a third pair on some layer.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum AttentionResidualExitOperand {
    /// `output_attn_res_norm.weight`, `[hidden]`.
    Norm,
    /// `output_attn_res_proj.weight`, `[1, hidden]`.
    Proj,
}

/// The exit's spellings, object-relative. Two groups under one common
/// segment prefix means the container names them `output_attn_res_*.
/// weight`, whatever the checkpoint's stack prefix was.
const ATTENTION_RESIDUAL_EXIT_TABLE: &[(&str, AttentionResidualExitOperand)] = &[
    (
        "output_attn_res_norm.weight",
        AttentionResidualExitOperand::Norm,
    ),
    (
        "output_attn_res_proj.weight",
        AttentionResidualExitOperand::Proj,
    ),
];

/// Classify one tensor of the attention-residual exit object by its
/// object-relative name. Exact, like every classifier in this module: a
/// spelling not in the table is `None` and blocks.
pub fn classify_attention_residual_exit_tensor(
    relative_name: &str,
) -> Option<AttentionResidualExitOperand> {
    ATTENTION_RESIDUAL_EXIT_TABLE
        .iter()
        .find(|(name, _)| *name == relative_name)
        .map(|(_, operand)| *operand)
}

/// Whether `relative_name` is one of the four attention-residual SITE
/// operands, asked WITHOUT the declaration.
///
/// The graph builder never calls this — placement of the per-layer pairs
/// is the decoder stack's, as it already was. The op plan calls it in
/// exactly one place: to tell a stray from an unrecognised spelling when
/// a component that does NOT declare the topology ships these names, so
/// the defect can say what the operand implies instead of only that
/// nothing classified it. Recognition is not ownership — the same
/// separation the hyper-connection head's placement arm makes.
pub fn is_attention_residual_site_operand(relative_name: &str) -> bool {
    layer_and_suffix(relative_name).is_some_and(|(_, suffix)| {
        ATTENTION_RESIDUAL_ROLE_TABLE
            .iter()
            .any(|(name, _)| *name == suffix)
    })
}

/// The hyper-connection HEAD's own operands — a component-level object,
/// not a stack operand, and a different operation from a site's.
///
/// `ParallelHead.hc_head` runs no Sinkhorn: `reduce_fn` has ONE row per
/// stream where a site's `mix_fn` has `(2 + hc)·hc`, and `scale` is a
/// single scalar where a site carries three. Wave 17 recorded the
/// difference in the executor ([`HeadWeights`](crate::format::vindex3::opplan::exec::hyper_connection::HeadWeights));
/// this is the same difference on the addressing side, so a checkpoint
/// that stored a site's operands under the head's names would fail the
/// head's geometry rather than bind.
///
/// ```text
/// reduce_fn   [hc, hc·hidden]
/// base        [hc]
/// scale       [1]
/// ```
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum HcHeadOperand {
    ReduceFn,
    Base,
    Scale,
}

/// The head's spellings, as DeepSeek-V4 writes them: three BARE
/// top-level tensors with no `model.` prefix. GLM-5.3-Flash ships none
/// (its `mhc` flag sits unexplained beside that absence and is not read
/// as meaning anything here). Hy4-preview's `model.hc_head.hc_head_fn`
/// is not listed for the same reason its site spelling is not: a
/// different topology's dialect.
const HC_HEAD_TABLE: &[(&str, HcHeadOperand)] = &[
    ("hc_head_fn", HcHeadOperand::ReduceFn),
    ("hc_head_base", HcHeadOperand::Base),
    ("hc_head_scale", HcHeadOperand::Scale),
];

/// Classify one tensor of the hyper-connection head object by its
/// object-relative name. Exact, like every classifier in this module:
/// a spelling not in [`HC_HEAD_TABLE`] is `None` and blocks.
pub fn classify_hyper_connection_head_tensor(relative_name: &str) -> Option<HcHeadOperand> {
    HC_HEAD_TABLE
        .iter()
        .find(|(name, _)| *name == relative_name)
        .map(|(_, operand)| *operand)
}

/// Whether `group_prefix` is one of the head's bare tensor groups — the
/// graph builder's placement question, answered from the same table the
/// op plan classifies by so the two cannot disagree about which names
/// are the head's.
pub fn is_hyper_connection_head_group(group_prefix: &str) -> bool {
    HC_HEAD_TABLE.iter().any(|(name, _)| *name == group_prefix)
}

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
    // The full-rank form of the same gate. Which form the layer is
    // EXPECTED to ship is the declaration's question, answered at closure;
    // the table only says what the spelling is on this operator.
    ("self_attn.g_proj.weight", OperandRole::KdaGProj),
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
    // Kimi-K3's factorised query. Named unconditionally here — the table
    // answers for the OPERATOR, as its siblings do — and held to the
    // declared form by closure, which is where "declared Direct but
    // shipped a triple" is refused.
    ("self_attn.q_a_proj.weight", OperandRole::MlaQAProj),
    ("self_attn.q_a_layernorm.weight", OperandRole::MlaQANorm),
    ("self_attn.q_b_proj.weight", OperandRole::MlaQBProj),
    ("self_attn.o_proj.weight", OperandRole::MlaOutProj),
    // MLA-only spellings.
    (
        "self_attn.kv_a_proj_with_mqa.weight",
        OperandRole::MlaKvAProj,
    ),
    ("self_attn.kv_b_proj.weight", OperandRole::MlaKvBProj),
    ("self_attn.kv_a_layernorm.weight", OperandRole::MlaKvANorm),
    // Same spelling as the KDA full-rank gate, different operator, different
    // operation. Expected only under `mla_use_output_gate`, which closure
    // checks; the table answers for the operator alone.
    ("self_attn.g_proj.weight", OperandRole::MlaOutputGate),
];

/// Suffix → role **on a Mamba2 layer**, consulted before [`ROLE_TABLE`]
/// for the same reason [`KDA_ROLE_TABLE`] is: the roles are
/// operator-gated, so an unknown stack shipping a bare `norm.weight`
/// still classifies as *nothing* and blocks rather than acquiring the
/// mixer's placement vocabulary.
const MAMBA2_ROLE_TABLE: &[(&str, OperandRole)] = &[
    ("mixer.in_proj.weight", OperandRole::Mamba2InProj),
    ("mixer.conv1d.weight", OperandRole::Mamba2Conv1d),
    ("mixer.conv1d.bias", OperandRole::Mamba2Conv1dBias),
    ("mixer.A_log", OperandRole::Mamba2ALog),
    ("mixer.D", OperandRole::Mamba2D),
    ("mixer.dt_bias", OperandRole::Mamba2DtBias),
    ("mixer.norm.weight", OperandRole::Mamba2GatedNorm),
    ("mixer.out_proj.weight", OperandRole::Mamba2OutProj),
    ("norm.weight", OperandRole::Mamba2PreMixerNorm),
];

/// Suffix → role **on a conv-QKV attention layer**, consulted before
/// [`ROLE_TABLE`]. Every spelling here collides with
/// [`MAMBA2_ROLE_TABLE`] at a different shape — the hybrid stack wraps
/// both block kinds in the same `mixer.`/`norm.` estate — so the layer's
/// operator is the only authority that can separate them, exactly the
/// per-layer-table argument [`classify_stack_tensor_on`] documents.
/// The pre-mixer norm role is shared deliberately: it is the SAME
/// declaration (one bare `norm.weight` wrapping the block) on both
/// layer kinds of this lineage.
const CONV_QKV_ROLE_TABLE: &[(&str, OperandRole)] = &[
    ("mixer.in_proj.weight", OperandRole::ConvQkvInProj),
    ("mixer.conv1d.weight", OperandRole::ConvQkvConv1d),
    ("mixer.conv1d.bias", OperandRole::ConvQkvConv1dBias),
    ("mixer.out_proj.weight", OperandRole::ConvQkvOutProj),
    ("norm.weight", OperandRole::Mamba2PreMixerNorm),
];

/// Split a stack tensor's object-relative name into its layer index and
/// the suffix every role table matches on. `None` when the name is not
/// layer-shaped, which is what makes a bare top-level tensor under a
/// stack binding a blocking fact rather than a layer-0 operand.
fn layer_and_suffix(relative_name: &str) -> Option<(usize, &str)> {
    let (layer, suffix) = relative_name.split_once('.')?;
    Some((layer.parse().ok()?, suffix))
}

/// Classify one stack tensor under the component's declared residual
/// topology as well as its layer's operator.
///
/// **The op plan's entry point.** Two authorities gate a role here and
/// they answer different questions: the OPERATOR separates spellings two
/// attention families share (`self_attn.o_proj.weight` on a KDA layer
/// and on an MLA layer of the same checkpoint), and the TOPOLOGY decides
/// whether a residual programme's operands exist at all.
///
/// The topology cannot be inferred from the operands, and this is the
/// site where that rule is enforced: a checkpoint shipping
/// `mlp_res_norm.weight` without declaring `attn_res_block_size`
/// classifies as NOTHING here, exactly as it did before this rung, and
/// the op plan reports it as an operand implying an absent op. Grading
/// it a site role would let a component acquire a residual topology by
/// spelling — the same failure as reading hyper-connections off an
/// `hc_`-prefixed name.
pub fn classify_stack_tensor_under(
    relative_name: &str,
    operator: LayerOperator,
    topology: larql_models::config::ResidualTopology,
) -> Option<(usize, OperandRole)> {
    if matches!(
        topology,
        larql_models::config::ResidualTopology::AttentionResidual { .. }
    ) {
        if let Some((layer, suffix)) = layer_and_suffix(relative_name) {
            if let Some((_, role)) = ATTENTION_RESIDUAL_ROLE_TABLE
                .iter()
                .find(|(name, _)| *name == suffix)
            {
                return Some((layer, *role));
            }
        }
    }
    classify_stack_tensor_on(relative_name, operator)
}

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
    let (layer, suffix) = layer_and_suffix(relative_name)?;
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
    if operator.is_mamba2() {
        if let Some((_, role)) = MAMBA2_ROLE_TABLE.iter().find(|(name, _)| *name == suffix) {
            return Some((layer, *role));
        }
    }
    if operator.is_conv_qkv() {
        if let Some((_, role)) = CONV_QKV_ROLE_TABLE.iter().find(|(name, _)| *name == suffix) {
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
        // Both wrap norms present, neither pre-norm: the sublayer reads
        // the raw residual. See [`NormPlacement::PostOnly`].
        (false, true, false, true) => Ok(NormPlacement::PostOnly),
        (false, false, false, false) => Err("stack carries no per-layer norm operands".to_string()),
        _ => Err(format!(
            "norm operand set is neither two-norm nor four-norm \
             (pre_attn {pre_attention}, post_attn {post_attention}, \
             pre_ffn {pre_ffn}, post_ffn {post_ffn})"
        )),
    }
}

/// Norm placement for a **mixer-only** (pure-SSM) stack, from its own
/// evidence: the single pre-mixer norm per layer, and no transformer
/// wrap norms beside it.
///
/// A separate function rather than a fourth tuple arm in
/// [`norm_placement_evidence`]: that function is operator-blind and would
/// have to read `(pre_attn, ..)` from a vocabulary the mixer's estate
/// never uses — a transformer stack that lost its FFN norms must keep
/// reading as the defect it is, never as a valid mixer placement. The
/// caller chooses this path only when the component's declared program is
/// mixer-only.
pub fn mixer_norm_placement_evidence<'a>(
    relative_names: impl Iterator<Item = &'a str>,
) -> Result<NormPlacement, String> {
    let mut pre_mixer = false;
    let mut transformer_norms = false;
    for name in relative_names {
        match classify_stack_tensor_on(name, LayerOperator::Mamba2).map(|(_, role)| role) {
            Some(OperandRole::Mamba2PreMixerNorm) => pre_mixer = true,
            Some(
                OperandRole::PreAttentionNorm
                | OperandRole::PostAttentionNorm
                | OperandRole::PreFfnNorm
                | OperandRole::PostFfnNorm,
            ) => transformer_norms = true,
            _ => {}
        }
    }
    match (pre_mixer, transformer_norms) {
        (true, false) => Ok(NormPlacement::PreMixer),
        (true, true) => Err(
            "stack carries a pre-mixer norm AND attention/FFN wrap norms — \
             not a mixer-only placement"
                .to_string(),
        ),
        (false, _) => Err("mixer-only stack carries no per-layer norm operands".to_string()),
    }
}
