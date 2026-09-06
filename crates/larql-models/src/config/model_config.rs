//! [`ModelConfig`] — the parsed `config.json`, with no behaviour attached.
//!
//! Behaviour lives on [`ModelArchitecture`](super::ModelArchitecture), which
//! reads this struct. Keeping the two apart is what lets a config fact be read
//! once in a trait default instead of per architecture.

use super::RopeScaling;

/// Model dimensions and architecture parameters, parsed from config.json.
#[derive(Debug, Clone)]
pub struct ModelConfig {
    pub model_type: String,
    /// RMS-norm / LayerNorm epsilon parsed from `rms_norm_eps` (or
    /// `layer_norm_eps` for LN architectures). `None` means the loader
    /// found no value and callers should fall back to their architecture
    /// default. Bug 2 in `docs/diagnoses/shannon-cross-engine-divergence.md`
    /// was the hardcoded 1e-6 in `ModelArchitecture::norm_eps()` ignoring
    /// this field; Mistral / Llama / Gemma all ship `1e-5` and need it.
    pub norm_eps: Option<f64>,
    pub num_layers: usize,
    pub hidden_size: usize,
    pub intermediate_size: usize,
    /// Per-layer dense-FFN width declared by a derived checkpoint
    /// (`larql_ffn_intermediate_size_by_layer`, one entry per layer),
    /// verbatim. `None` means every layer runs at
    /// [`Self::intermediate_size`] — the only meaning an undeclared key
    /// has. The planner checks each layer's gate/up/down shapes against
    /// this and states the width on that layer's op; nothing else may
    /// read a layer's width off a tensor.
    pub ffn_intermediate_size_by_layer: Option<Vec<usize>>,
    pub head_dim: usize,
    pub num_q_heads: usize,
    pub num_kv_heads: usize,
    pub vocab_size: Option<usize>,
    pub rope_base: f64,
    /// RoPE base for local/sliding window layers (Gemma3: 10,000).
    pub rope_local_base: Option<f64>,
    /// Per-layer declared rope theta (`layer_rope_theta` in `config.json`),
    /// verbatim including the upstream `0.0` NoPE sentinel. Interpretation
    /// happens once, in
    /// [`ModelArchitecture::position_policy_for_layer`](super::ModelArchitecture::position_policy_for_layer)
    /// via [`PositionPolicy::from_declared_theta`](super::PositionPolicy::from_declared_theta) —
    /// nothing else may read this array's zeros as numbers.
    pub layer_rope_theta: Option<Vec<f64>>,
    /// Per-layer rotary mask (`no_rope_layers`), verbatim.
    ///
    /// **The key is named for what it disables and its values say the
    /// opposite**: SmolLM3's config documents *"A `1` at an index
    /// position indicates that the corresponding layer will use RoPE,
    /// while a `0` indicates that it's a NoPE layer"*, and both SmolLM3
    /// and Llama 4 read it as `self.use_rope = config.no_rope_layers[i]`.
    /// Reading the name instead of the reference inverts the schedule,
    /// which on SmolLM3-3B is 27 of 36 layers rotated wrongly and a model
    /// that still produces fluent-looking output.
    ///
    /// Held verbatim for the same reason [`Self::layer_rope_theta`] holds
    /// its `0.0` sentinel verbatim: the polarity is honoured exactly once,
    /// in [`PositionPolicy::rope_enabled_by_flag`](super::PositionPolicy::rope_enabled_by_flag),
    /// and nothing else may read these integers as booleans.
    pub no_rope_layers: Option<Vec<i64>>,
    /// The regular-interval FALLBACK for [`Self::no_rope_layers`]
    /// (`no_rope_layer_interval`): every `n`-th layer is NoPE.
    ///
    /// A generator, not a redundant spelling — both references consult it
    /// only `if no_rope_layers is None`, so when the explicit mask is
    /// present this is superseded and must not be allowed to disagree
    /// with it into effect.
    pub no_rope_layer_interval: Option<usize>,
    /// The checkpoint's declared rotary PAIRING (`rope_interleaved`).
    ///
    /// Read by no reference implementation — the exact key appears
    /// nowhere in transformers, and SmolLM2-135M declares it under
    /// `model_type: llama`, which has no such field. It is read here
    /// anyway, because the alternative is an unread declaration that
    /// happens to agree: this build pairs split-half
    /// ([`ROPE_PAIRING_INTERLEAVED`](super::ROPE_PAIRING_INTERLEAVED)),
    /// so `false` agrees and `true` is a different operator that would
    /// otherwise be performed the wrong way round in silence.
    pub rope_interleaved: Option<bool>,
    /// The checkpoint's declared multi-axis rotary flag (`use_mrope`).
    ///
    /// Also read by no reference implementation in transformers 5.5.0 —
    /// and unlike `use_sliding_window`, which HF genuinely consults, this
    /// one has no upstream reader to agree with. Read for the same reason:
    /// Qwen2.5-0.5B declares `false`, which agrees with the text policy
    /// this build resolves, and `true` on a config carrying no
    /// `mrope_section` is a claim nothing can honour.
    ///
    /// A third spelling of the fact [`Self::mrope_section`] and
    /// [`Self::mrope_interleaved`] state jointly; the effective policy is
    /// resolved from those, and this is checked against it.
    pub use_mrope: Option<bool>,
    /// The checkpoint's declared FFN shape in Falcon's one-word dialect
    /// (`activation`: `swiglu`, `geglu`, or a plain nonlinearity name),
    /// verbatim.
    ///
    /// Read by no transformers-5.5.0 loader for a `model_type: llama`
    /// checkpoint (Falcon3 declares `activation: "swiglu"` beside
    /// `hidden_act: "silu"`). Stored so the claim is CHECKED against the
    /// FFN this build runs — `swiglu` is gated SiLU, `geglu` is a different
    /// FFN, and a plain name is the ungated shape — rather than left as an
    /// unread agreement one value away from the wrong arithmetic. The
    /// vocabulary is [`super::activation::ffn_shape_from_hf_name`].
    pub ffn_shape_name: Option<String>,
    /// The checkpoint's declared `is_llama_config` flag (SmolLM2), verbatim.
    ///
    /// Appears nowhere in transformers 5.5.0. A claim about WHICH family
    /// serves the checkpoint, checked against the family the declared
    /// identity actually resolved to — never echoed, so `true` under a
    /// `model_type` no registry entry matches is refused rather than
    /// believed.
    pub is_llama_config: Option<bool>,
    /// The checkpoint's declared positional-encoding scheme
    /// (`position_embedding_type`), verbatim.
    ///
    /// Carried as the declared string rather than a parsed enum because
    /// **the same leaf means different things in different families**:
    /// the BERT lineage spells `absolute` / `relative_key` here, while
    /// `granitemoehybrid` documents exactly `[None, "rope"]`. A shared
    /// enum would have to be the union of every family's vocabulary, and
    /// a value's meaning would still depend on who declared it.
    ///
    /// Nothing may read this directly to decide behaviour — the effective
    /// policy is resolved in
    /// [`ModelArchitecture::position_policy_for_layer`](super::ModelArchitecture::position_policy_for_layer),
    /// where the family that owns the spelling interprets it.
    pub position_embedding_type: Option<String>,
    pub sliding_window: Option<usize>,
    /// The checkpoint's explicit *enable* flag for sliding-window
    /// attention (`use_sliding_window`).
    ///
    /// Separate from [`Self::sliding_window`] because declaring a window
    /// and using one are different facts, and Qwen states both: Qwen2.5
    /// ships `sliding_window: 32768` beside `use_sliding_window: false`,
    /// and Qwen3 ships `sliding_window: null` beside the same flag.
    /// `None` means the checkpoint does not state the flag, which every
    /// family before Qwen leaves to the window's own presence.
    ///
    /// Nothing may read this directly to decide behaviour — the effective
    /// policy is resolved once, in
    /// [`ModelArchitecture::sliding_window_size`](super::ModelArchitecture::sliding_window_size),
    /// so a caller cannot honour the window while ignoring the flag.
    pub use_sliding_window: Option<bool>,
    /// How many layers use the window when it is enabled
    /// (`max_window_layers`): the bottom `n` layers slide, the rest
    /// attend fully. Inert while the window is disabled.
    pub max_window_layers: Option<usize>,
    // MoE fields
    pub num_experts: Option<usize>,
    pub num_experts_per_token: Option<usize>,
    pub num_shared_experts: Option<usize>,
    /// The always-on shared branch's own intermediate width, where the
    /// family declares one (`shared_expert_intermediate_size` on Qwen
    /// MoE, `moe_shared_expert_intermediate_size` on Nemotron-H).
    ///
    /// `None` does NOT mean "no shared expert": the DeepSeek/Kimi lineage
    /// declares a shared-expert COUNT and sizes one wider FFN at
    /// `moe_intermediate_size * count`. Which of the two a family means is
    /// answered once, by
    /// [`ModelArchitecture::shared_expert_intermediate_size`](super::ModelArchitecture::shared_expert_intermediate_size),
    /// so no caller has to know the lineage to size the branch.
    pub shared_expert_intermediate_size: Option<usize>,
    /// Gemma 4 A4B: enables hybrid dense-MLP + MoE-experts block per layer.
    pub enable_moe_block: bool,
    /// Gemma 4 A4B: experts activated per token (stored as `top_k_experts` in config.json).
    pub top_k_experts: Option<usize>,
    /// Gemma 4 A4B: intermediate (hidden) dimension of each expert's FFN.
    pub moe_intermediate_size: Option<usize>,
    /// GPT-OSS: clamp bound applied to both halves of the fused gate/up
    /// projection before the GLU (`swiglu_limit` in `config.json`, 7.0 on
    /// the released checkpoints). `None` for architectures that don't clamp.
    pub swiglu_limit: Option<f64>,
    /// Whether the router renormalises its top-k probabilities to sum to 1
    /// (`norm_topk_prob` in `config.json`). `None` means the field was absent;
    /// architectures that read it treat that as `false`, matching HF's own
    /// default for the OLMoE/Mixtral family.
    pub norm_topk_prob: Option<bool>,
    // MLA fields
    pub kv_lora_rank: Option<usize>,
    pub q_lora_rank: Option<usize>,
    /// DS-V3 MLA: non-RoPE part of head dim (nope). qk_head_dim = qk_nope_head_dim + qk_rope_head_dim.
    pub qk_nope_head_dim: Option<usize>,
    /// DS-V3 MLA: RoPE part of head dim.
    pub qk_rope_head_dim: Option<usize>,
    /// DS-V3 MLA: V head dim (may differ from qk_nope+rope total).
    pub v_head_dim: Option<usize>,
    // RoPE scaling
    pub rope_scaling: Option<RopeScaling>,
    // Softcapping (Gemma2)
    pub attn_logit_softcapping: Option<f64>,
    pub final_logit_softcapping: Option<f64>,
    /// Override attention scale denominator (Gemma: query_pre_attn_scalar).
    pub query_pre_attn_scalar: Option<f64>,
    // Granite-style scaling multipliers
    pub embedding_multiplier: Option<f64>,
    pub residual_multiplier: Option<f64>,
    pub attention_multiplier: Option<f64>,
    pub logits_scaling: Option<f64>,
    // Per-layer attention geometry (Gemma 4 style: different head_dim / KV heads
    // for sliding vs global attention layers).
    /// Head dimension for global (full) attention layers. If None, all layers use head_dim.
    pub global_head_dim: Option<usize>,
    /// Number of KV heads for global attention layers. If None, all layers use num_kv_heads.
    pub num_global_kv_heads: Option<usize>,
    /// Fraction of head_dim dimensions to apply RoPE to (0.0–1.0). If None, full rotation.
    pub partial_rotary_factor: Option<f64>,
    /// Sliding window pattern: every Nth layer is full attention.
    /// E.g., 6 means layers 5, 11, 17, ... are full attention.
    pub sliding_window_pattern: Option<usize>,
    /// Explicit per-layer type array (e.g., ["sliding_attention", "full_attention", ...]).
    /// When present, overrides sliding_window_pattern.
    pub layer_types: Option<Vec<String>>,
    /// Whether value projection shares key projection (K=V) on some layers.
    pub attention_k_eq_v: bool,
    /// Per-layer embedding dimension (PLE). If > 0, each layer adds a gated
    /// per-layer embedding lookup to the hidden state before attention.
    pub per_layer_embed_dim: Option<usize>,
    /// Number of layers at the end of the model that share KV from earlier layers.
    /// E.g., 20 means the last 20 layers reuse KV cache from earlier source layers.
    pub num_kv_shared_layers: Option<usize>,
    /// Whether the model's config.json contains a `vision_config` section.
    pub has_vision_config: bool,
    /// `tie_word_embeddings` — whether the output projection *is* the
    /// embedding matrix. `None` when the config omits it.
    ///
    /// Load-bearing as a **check**, not a shortcut: the loader ties whenever
    /// `lm_head.weight` is absent, so a checkpoint that declares `false` and
    /// then fails to produce the tensor for any reason (a key mismatch, a
    /// skip filter, a bad shard) would silently run with the wrong output
    /// projection. Untied-but-missing is now an error.
    pub tie_word_embeddings: Option<bool>,

    // ── Attention/output scaling and norm shape (declared, per-checkpoint) ──
    /// Extra multiplier on attention scores on top of `1/sqrt(head_dim)`
    /// (`qk_scale_factor`). Distinct from `query_pre_attn_scalar`, which
    /// *replaces* the denominator.
    pub qk_scale_factor: Option<f64>,
    /// Multiplier applied to the model output / hidden state before the
    /// vocabulary projection (`output_multiplier`).
    pub output_multiplier: Option<f64>,
    /// Epsilon for post-norms when it differs from `rms_norm_eps`
    /// (`post_norm_eps`). `None` = post-norms share `norm_eps`.
    pub post_norm_eps: Option<f64>,
    /// Whether attention projections carry bias terms (`attention_bias`).
    /// `None` = the config is silent and the family default answers.
    pub attention_bias: Option<bool>,
    /// Whether FFN/MLP projections carry bias terms (`mlp_bias`). Same
    /// contract as [`attention_bias`](Self::attention_bias): the boolean
    /// itself has no schema field downstream — operand closure over the
    /// checkpoint's actual bias tensors is the real gate, this is here so
    /// `mlp_bias` is `consumed` rather than silently unread.
    pub mlp_bias: Option<bool>,
    /// FFN activation name, verbatim (`hidden_act` / `hidden_activation`).
    /// Mapped to [`Activation`](super::Activation) by
    /// `ModelArchitecture::activation`; an unrecognised spelling must fail
    /// there, not default.
    pub hidden_act: Option<String>,
    /// SiTU-GLU's gate softcap (`activation_situ_beta`), verbatim.
    ///
    /// Read only as a parameter OF the activation `hidden_act` names —
    /// declaring it beside any other activation configures a combine the
    /// checkpoint says it does not use, and the plan reports that rather
    /// than the value being quietly applied. `f64` because the config
    /// declares a JSON number; the resolution to `f32` (and the
    /// reference's `beta or 1.0`) happens once, on the architecture.
    pub activation_situ_beta: Option<f64>,
    /// SiTU-GLU's up-branch softcap (`activation_situ_linear_beta`),
    /// verbatim. Absent means the up branch is untouched — a different
    /// function, not an infinite bound.
    pub activation_situ_linear_beta: Option<f64>,
    /// Declared context bound (`max_position_embeddings`).
    pub max_position_embeddings: Option<usize>,

    // ── Multimodal protocol + adapter geometry (root-level HF fields) ──
    /// Token id standing in for an image patch (`image_token_id`).
    pub image_token_id: Option<u64>,
    /// Token id standing in for a video segment (`video_token_id`).
    pub video_token_id: Option<u64>,
    /// Vision-adapter output width into the language model (`out_hidden_size`).
    pub out_hidden_size: Option<usize>,
    /// Vision-adapter hidden width (`projector_hidden_size`).
    pub projector_hidden_size: Option<usize>,
    /// Vision-adapter activation name (`projector_hidden_act`).
    pub projector_hidden_act: Option<String>,

    // ── Drafter interface declaration (DFlash-style speculative sidecar) ──
    /// Target-model layers whose hidden states this (drafter) checkpoint
    /// consumes (`target_layer_ids`). Presence declares the artifact a
    /// hidden-state consumer; the cross-component edge lives in the vindex3
    /// system graph, this is its source declaration.
    pub target_layer_ids: Option<Vec<usize>>,
    /// Tokens proposed per drafter forward (`block_size`). Only read when
    /// `target_layer_ids` is present — a bare `block_size` on a non-drafter
    /// config is some other concept and must stay unjudged.
    pub draft_block_size: Option<usize>,
    /// Mask token the block-diffusion drafter fills (`mask_token_id`).
    pub mask_token_id: Option<u64>,

    // ── Gemma 3n/4-E per-layer-input family knobs ──
    /// Doubles the MLP width on the KV-shared layers (`use_double_wide_mlp`).
    /// Read verbatim: no executor represents it, so a checkpoint declaring
    /// `true` must block on the value rather than lose it.
    pub use_double_wide_mlp: Option<bool>,
    /// The per-layer-input embedding vocabulary (`vocab_size_per_layer_input`)
    /// — the width of a table that exists only when `per_layer_embed_dim`
    /// is set. Read verbatim, judged against that width.
    pub vocab_size_per_layer_input: Option<u64>,

    // ── Hybrid linear-attention + multi-token-prediction (declared,
    //    R2/Kimi-Linear-rung prep — see `docs/k3-funnel.md`) ──
    //
    // Qwen3.5-style hybrid architectures interleave full-attention layers
    // with linear-attention (gated recurrent / short-conv, SSM-adjacent)
    // layers and add a multi-token-prediction head. None of this is
    // executable yet: there is no `AttentionOp`/component variant for a
    // linear-attention layer and no MTP-head object in the VINDEX3 schema.
    // These fields are read and retained verbatim — not judged, not
    // guessed at — so a future R2 pass has the declared values in hand
    // instead of the parser having silently dropped them.
    /// Convolution kernel width in the linear-attention block's short conv
    /// (`linear_conv_kernel_dim`).
    pub linear_conv_kernel_dim: Option<usize>,
    /// Per-head key dimension in the linear-attention block
    /// (`linear_key_head_dim`) — distinct from the full-attention `head_dim`.
    pub linear_key_head_dim: Option<usize>,
    /// Per-head value dimension in the linear-attention block
    /// (`linear_value_head_dim`).
    pub linear_value_head_dim: Option<usize>,
    /// Number of key heads in the linear-attention block
    /// (`linear_num_key_heads`).
    pub linear_num_key_heads: Option<usize>,
    /// Number of value heads in the linear-attention block
    /// (`linear_num_value_heads`).
    pub linear_num_value_heads: Option<usize>,
    /// The recurrent/full interleave when the checkpoint declares it as
    /// layer-index sets (`linear_attn_config.{kda_layers,
    /// full_attn_layers}`) rather than as a `layer_types` array — Kimi
    /// Linear's spelling, and GLM-5.3-Flash's second one.
    ///
    /// An outcome, not an `Option`: "declared nothing" and "declared
    /// something this build could not read" must stay distinguishable, or
    /// an unreadable declaration silently becomes the caller's default.
    /// See [`DeclaredInterleave`](super::DeclaredInterleave).
    pub linear_attn_interleave: super::DeclaredInterleave,
    /// The MTP sub-stack's own declared interleave. Its own field because
    /// it indexes its own layer space — Inkling-Small declares
    /// `local_layer_ids` for the decoder and again for its 8-layer MTP
    /// stack, and one resolution cannot speak for both.
    pub mtp_interleave: super::DeclaredInterleave,
    /// The KDA block's declared geometry (`linear_attn_config.{num_heads,
    /// head_dim, short_conv_kernel_size}`). `None` when the checkpoint
    /// declares no KDA block, or declares it partially — which is refused
    /// rather than defaulted. See [`KdaGeometry`](super::KdaGeometry).
    pub kda_geometry: Option<super::KdaGeometry>,
    /// KDA's decay-gate lower bound (`linear_attn_config.gate_lower_bound`,
    /// -5.0 on both observed checkpoints). `None` = undeclared; a clamp is
    /// never invented, because a wrong one changes the decay envelope
    /// without changing any shape.
    pub kda_gate_lower_bound: Option<f32>,

    /// `linear_attn_config.safe_gate` — whether the family's clamped gate
    /// branch is enabled when no bound is declared.
    ///
    /// `None` means the checkpoint said nothing, which is NOT the same as
    /// `Some(false)`: GLM-5.3-Flash's reference treats an absent key as
    /// `True`. Carried so [`ModelArchitecture::kda_gate_form`] can apply
    /// the family's own rule to a checked value instead of an assumed one.
    pub kda_safe_gate: Option<bool>,
    /// The FORM of KDA's output gate (`linear_attn_config.use_full_rank_gate`):
    /// `Some(true)` = one full-rank `g_proj` of `[Hv·Dv, hidden]` (Kimi-K3);
    /// `Some(false)` = the low-rank `g_a_proj`/`g_b_proj` pair; `None` =
    /// undeclared, which the reference reads as the pair
    /// (`config.linear_attn_config.get("use_full_rank_gate", False)`) — a
    /// CHECKED default, carried as an option so "undeclared" stays
    /// distinguishable from "declared low rank". Only the gate's
    /// projection changes with the form; its sigmoid and the gated norm
    /// do not. K3-REP-GATE-1.
    pub kda_use_full_rank_gate: Option<bool>,
    /// Whether MLA gates its aggregated value before `o_proj`
    /// (`mla_use_output_gate`): `sigmoid(g_proj(x)) ⊙ attn_value`, the
    /// same generic operation the softmax family's `attn_output_gate`
    /// declares, at width `Hq·v_head_dim`. `None` = undeclared, which the
    /// reference reads as no gate (`getattr(config, "mla_use_output_gate",
    /// False)`). K3-REP-GATE-1.
    pub mla_use_output_gate: Option<bool>,
    /// Width of the learned relative-position term (`d_rel`), and the
    /// bounded distance it spans (`rel_extent`). Declared together or not
    /// at all; a checkpoint declaring them uses a relative scheme and no
    /// rotation. See [`PositionPolicy::Relative`](super::PositionPolicy).
    /// Router scoring function, verbatim (`scoring_func` /
    /// `moe_router_activation_func`). Carried rather than judged: the
    /// typed [`MoeRouterKind`](super::MoeRouterKind) is what dispatch
    /// reads, and a spelling this build has not judged must not silently
    /// take the default softmax rule.
    pub router_activation: Option<String>,
    /// Multiplier applied to the routed-expert branch
    /// (`routed_scaling_factor`). A real rescale of the whole branch, so an
    /// absence is not 1.0 — it is "undeclared", and a consumer must say
    /// which it means.
    pub routed_scaling_factor: Option<f64>,
    /// Expert grouping: how many groups the router partitions experts into
    /// (`n_group` / `num_expert_group`) and how many it selects from
    /// (`topk_group`), plus the flag that turns grouping on.
    pub expert_groups: Option<usize>,
    pub topk_group: Option<usize>,
    pub use_grouped_topk: Option<bool>,
    /// Period of the MoE cadence after the dense prefix (`moe_layer_freq`),
    /// and the number of leading dense layers (`first_k_dense_replace`).
    pub moe_layer_freq: Option<usize>,
    pub first_k_dense_replace: Option<usize>,
    /// Whether the MLA block omits rotary entirely (`mla_use_nope`).
    pub mla_use_nope: Option<bool>,
    /// Kimi Linear's spelling of the serving context bound.
    pub model_max_length: Option<usize>,
    pub d_rel: Option<usize>,
    pub rel_extent: Option<usize>,
    /// Dtype the linear-attention block's SSM/recurrent state is computed
    /// in (`mamba_ssm_dtype`), verbatim.
    pub mamba_ssm_dtype: Option<String>,
    /// The Mamba2 mixer's declared geometry (`state_size`, `num_heads`,
    /// `expand`, …). `None` when the checkpoint declares no Mamba2 mixer,
    /// or declares it partially — which is refused rather than defaulted.
    /// See [`Mamba2Geometry`](super::Mamba2Geometry).
    pub mamba2_geometry: Option<super::Mamba2Geometry>,
    /// How the Mamba2 geometry was read: which key dialect, and every
    /// field that came from a recorded family default rather than the
    /// checkpoint's declaration. Present exactly when `mamba2_geometry`
    /// is. See [`Mamba2Provenance`](super::Mamba2Provenance).
    pub mamba2_provenance: Option<super::Mamba2Provenance>,
    /// The hybrid stack's conv-QKV attention geometry (`attention_*` /
    /// `rope_emb_dim` keys). `None` when the checkpoint declares no such
    /// block, or declares it partially — refused rather than defaulted.
    /// See [`ConvQkvAttnGeometry`](super::ConvQkvAttnGeometry).
    pub conv_qkv_attn: Option<super::ConvQkvAttnGeometry>,
    /// How the conv-QKV geometry was read: dialect and recorded family
    /// defaults. Present exactly when `conv_qkv_attn` is.
    pub conv_qkv_provenance: Option<super::ConvQkvProvenance>,
    /// `attn_cfg.causal` — the hybrid attention block's declared
    /// masking, verbatim. The operator is causal by construction, so a
    /// declared `false` blocks rather than running causal anyway.
    pub attn_causal: Option<bool>,
    /// `pad_vocab_size_multiple` — the embedding rows are the declared
    /// vocab rounded UP to this multiple (mamba_ssm lineage).
    pub pad_vocab_size_multiple: Option<usize>,
    /// `fused_add_norm` — whether the reference runtime fuses the
    /// residual add with the norm. A kernel-schedule fact about the
    /// same operation, carried verbatim.
    pub fused_add_norm: Option<bool>,
    /// The mamba_ssm lineage's declared MLP width (`mlp_intermediate_size`).
    /// `Some(0)` is a declaration — no MLP blocks exist in the stack —
    /// distinct from `None` (the key was never declared).
    pub mlp_intermediate_size: Option<usize>,
    /// Padding multiple the declared MLP width is rounded up to
    /// (`mlp_padding_size`). Parameterises the same MLP, absent or not.
    pub mlp_padding_size: Option<usize>,
    /// Whether the declared MLP's projections carry biases
    /// (`use_mlp_bias`).
    pub use_mlp_bias: Option<bool>,
    /// Whether the residual stream is kept at fp32 against a lower-precision
    /// model (`residual_in_fp32`) — an execution-precision fact, verbatim.
    pub residual_in_fp32: Option<bool>,
    /// `hc_mult` — how many parallel residual streams the component's
    /// state carries. `None` = one, the topology every family judged
    /// before hyper-connections uses.
    ///
    /// A COMPONENT fact, never a layer one: once the residual means
    /// `[..., hc, d]`, the embedding, every branch operator and the head
    /// all have to agree about it.
    pub hc_streams: Option<usize>,
    /// `hc_sinkhorn_iters` — iterations of the normalisation that splits
    /// the projected state statistics into the reduce weights, the
    /// expand weights and the cross-stream combination matrix.
    pub hc_sinkhorn_iters: Option<usize>,
    /// `hc_eps` — the epsilon that split runs at. NOT the component's
    /// `norm_eps`: the reference passes them separately (the mix
    /// projection's RMS uses `norm_eps`, the split uses this), and
    /// merging them would run a different model.
    pub hc_eps: Option<f64>,
    /// `attn_res_block_size` — the layer period at which the ENTERING
    /// residual state is snapshotted into the history every sublayer
    /// then reads (Kimi-K3 declares 12). `None` = the key was never
    /// declared, which is the ordinary residual.
    ///
    /// A COMPONENT fact for the same reason `hc_mult` is: the snapshot
    /// schedule, every layer's read of the history and the stack's own
    /// exit reduction must agree about it. Read as declared or not at
    /// all — a defaulted period silently changes which layers snapshot.
    pub attn_res_block_size: Option<usize>,
    /// Whether attention output is gated before `o_proj` (`attn_output_gate`).
    /// Distinct from the judged [`AttentionGateSpec`](super::AttentionGateSpec)
    /// an architecture returns from `attention_output_gate()` — this is the
    /// checkpoint's raw declaration, carried through even where no family
    /// has judged what the gate computes yet.
    pub attn_output_gate: Option<bool>,
    /// Attention output gate nonlinearity, verbatim (`output_gate_type`,
    /// e.g. `"swish"`).
    pub output_gate_type: Option<String>,
    /// Number of hidden layers in the multi-token-prediction head
    /// (`mtp_num_hidden_layers`). `None`/absent = no MTP head declared.
    pub mtp_num_hidden_layers: Option<usize>,
    /// Whether the MTP head uses its own embedding table rather than
    /// sharing the backbone's (`mtp_use_dedicated_embeddings`).
    pub mtp_use_dedicated_embeddings: Option<bool>,
    /// Whether mRoPE sections interleave across the `mrope_section` split
    /// (`rope_parameters.mrope_interleaved`) — Qwen-VL-style multi-axis
    /// position encoding [`PositionPolicy`](super::PositionPolicy) does
    /// not express.
    pub mrope_interleaved: Option<bool>,
    /// mRoPE per-axis section widths (`rope_parameters.mrope_section`),
    /// verbatim.
    pub mrope_section: Option<Vec<usize>>,
}
