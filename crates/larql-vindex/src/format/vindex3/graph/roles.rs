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

mod classify;
mod tables;
pub use classify::*;
pub use tables::*;

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
    /// Additive bias on the attention projections: all four iff the surface
    /// declares `attention_bias` (GPT-OSS), Q/K/V alone iff it declares
    /// `qkv_bias` (Qwen2, whose output projection is unbiased).
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
    /// The additive bias beside each norm weight (LayerNorm's affine
    /// `bias`), one per site. Present iff the surface declares
    /// `norm.bias`, and then beside EVERY weighted norm site; never
    /// evidence of placement — only the weights are.
    PreAttentionNormBias,
    PostAttentionNormBias,
    PreFfnNormBias,
    PostFfnNormBias,
    FfnGate,
    FfnUp,
    FfnDown,
    /// The dense FFN's additive biases: after the up projection (before
    /// the activation) and after the down projection. Present iff the
    /// surface declares `ffn.bias`.
    FfnUpBias,
    FfnDownBias,
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
    /// Kimi-K3's latent routed branch: the bottleneck the ROUTED experts
    /// live behind. `routed_expert_down_proj` `[latent, hidden]` takes the
    /// block input to the routed width, the optional `routed_expert_norm`
    /// `[latent]` normalises the weighted aggregate, and
    /// `routed_expert_up_proj` `[hidden, latent]` returns it.
    ///
    /// These are DENSE per-layer operands, not expert-bank ones: there is
    /// one of each per routed layer regardless of expert count, and they
    /// wrap the bank rather than living inside it.
    ///
    /// Required exactly when the component declares the latent form —
    /// down and up always, the norm iff the form carries one. Shipped
    /// without that declaration they are refused BY NAME: a
    /// `routed_expert_down_proj` may confirm the form, it must never
    /// select it.
    MoeLatentDownProj,
    MoeLatentNorm,
    MoeLatentUpProj,
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
