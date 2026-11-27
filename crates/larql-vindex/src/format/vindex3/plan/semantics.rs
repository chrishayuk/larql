//! Semantic classification of config keys the parser does not consume.
//!
//! The registry maps *known* HF config-field names to a semantic class.
//! It is a vocabulary of the HF config format, not of any model family —
//! `qk_scale_factor` means an attention-scale override whichever checkpoint
//! declares it. A name the registry has never seen classifies as
//! [`SemanticClass::Unknown`], which blocks the plan: an unjudged key must
//! not pass silently, because "unconsumed and unjudged" is exactly the
//! silent-default shape the whole instrument exists to catch.

mod clusters;
mod interface_keys;
pub use clusters::*;
pub use interface_keys::*;

/// Keys that change what a forward pass computes: norms, activations,
/// position encoding, attention/output scaling, attention span policy.
pub const EXECUTION_SEMANTIC_KEYS: &[&str] = &[
    // The Sinkhorn hyper-connection topology (wave 19): carried to the
    // component's residual topology and executed by both traversals.
    "hc_mult",
    "hc_sinkhorn_iters",
    "hc_eps",
    // The attention-residual period (K3-ATTNRES-1): it decides which
    // layers snapshot the entering residual into the history every
    // sublayer then reads, so a build that ignored it would compute a
    // different model at every layer past the first block.
    "attn_res_block_size",
    "layer_rope_theta",
    // E30 static shards: the per-layer dense-FFN width a derived checkpoint
    // declares. Execution semantics of the plainest kind — it is the row
    // count every layer's gate/up projections run at — carried to the
    // layer's `FfnOp` and checked against the stored tensors.
    "larql_ffn_intermediate_size_by_layer",
    "qk_scale_factor",
    "output_multiplier",
    "post_norm_eps",
    "hidden_activation",
    "hidden_act",
    "attention_bias",
    "qkv_bias",
    "mlp_bias",
    "layer_norm_eps",
    "rms_norm_eps",
    "norm_epsilon",
    "norm_eps",
    "rope_theta",
    "rope_type",
    "layer_types",
    // The same per-layer topology in the spellings that state it as index
    // sets rather than as an array. Execution-semantic for exactly the
    // reason `layer_types` is: they decide which operator each layer runs
    // and, for a sliding layer, how far back it attends. Inkling-Small
    // states 35 of its 42 layers sliding through `local_layer_ids` alone.
    "local_layer_ids",
    // The window itself, in Inkling-Small's spelling of it.
    "sliding_window_size",
    // How many multi-token-prediction layers the checkpoint carries. No
    // MTP object exists in this schema, so it will grade unrepresented on
    // carriage — which is the honest answer, and a different one from
    // "nobody judged this key".
    "num_nextn_predict_layers",
    // The relative-position scheme. Execution-semantic in the strongest
    // sense: a checkpoint declaring it does not rotate, and a build that
    // ignored it would rotate anyway at a default base.
    "d_rel",
    "rel_extent",
    // The MoE facts Kimi Linear spells its own way, each resolving into
    // the same execution surface the DeepSeek-lineage spellings do.
    "moe_renormalize",
    "num_shared_experts",
    // The same branch's WIDTH, in both declared spellings. Beside the
    // count and not with the operand sizes, because the proof the two
    // buckets offer is different: a `tensor_semantic` key is waved
    // through as "read by a registered parser", which on a component
    // that built no surface proves nothing at all. This one is judged by
    // its carriage rule against the width the branch will actually be
    // built at, so a checkpoint whose declaration and resolution
    // disagree is a mismatch rather than a pass.
    "shared_expert_intermediate_size",
    "moe_shared_expert_intermediate_size",
    "moe_router_activation_func",
    "scoring_func",
    // The two-set interleave and the KDA conv width.
    "kda_layers",
    "full_attn_layers",
    "short_conv_kernel_size",
    // The KDA decay clamp, which changes the decay envelope without
    // changing any shape.
    "gate_lower_bound",
    // The KDA output gate's FORM (Kimi-K3): one full-rank projection or
    // the low-rank pair. Execution-semantic because it decides which
    // operands the gate is computed from; carried to the op and held
    // against the shipped tensors.
    "use_full_rank_gate",
    // MLA's output gate (Kimi-K3): a sigmoid gate on the aggregated value
    // before `o_proj`, absent everywhere else. A build that ignored it
    // would run every MLA layer ungated with every shape still closing.
    "mla_use_output_gate",
    // Expert grouping. Declared by Kimi Linear and GLM-5.3-Flash alike,
    // and at one group it selects over every expert — the same thing an
    // ungrouped router does.
    "num_expert_group",
    "n_group",
    "topk_group",
    "use_grouped_topk",
    // The dense/sparse cadence after the dense prefix, and the prefix.
    "moe_layer_freq",
    "first_k_dense_replace",
    // A real rescale of the whole routed branch.
    "routed_scaling_factor",
    // Whether the MLA block omits rotary entirely.
    "mla_use_nope",
    "sliding_window",
    // The window's ENABLE flag and its layer bound. Execution semantics,
    // not metadata: they decide whether the window applies at all and how
    // far up the stack, and `ModelArchitecture::sliding_window_size`
    // resolves all three into one effective policy. Qwen ships a window
    // beside `use_sliding_window: false`, and honouring the size without
    // the flag is how a declared-inactive feature becomes an active wrong
    // answer.
    "use_sliding_window",
    "max_window_layers",
    // Execution semantics, and on one family the switch itself: HF builds
    // `granitemoehybrid`'s rotary embedding only when this reads `rope`,
    // so a checkpoint omitting it runs with no positional encoding. The
    // same leaf is `absolute` / `relative_key` in the BERT lineage, which
    // is why the value is interpreted by the architecture and not here.
    "position_embedding_type",
    // The per-layer rotary schedule. Execution semantics of the plainest
    // kind — it decides which layers encode position at all — and the
    // mask's polarity is inverted relative to its own name, so an
    // unconsumed declaration here is 27 of SmolLM3-3B's 36 layers rotated
    // the wrong way while still emitting fluent text.
    "no_rope_layers",
    "no_rope_layer_interval",
    // Declared by checkpoints, read by no reference implementation. Still
    // execution semantics: each names an operator this build either
    // performs or does not, and an unread agreement is one value away
    // from a rotation done the wrong way round.
    "rope_interleaved",
    "use_mrope",
    // Falcon's dialect for the FFN SHAPE: `activation: "swiglu"` beside
    // `hidden_act: "silu"` under `model_type: llama` (Falcon3). No
    // transformers-5.5.0 loader reads it for that family; still the word
    // names gated-vs-ungated and the gate nonlinearity together, and the
    // wrong word is a different FFN.
    "activation",
    // SmolLM2's `is_llama_config: true` — read by nothing upstream; a claim
    // about which family serves the checkpoint, checked against the
    // family the identity actually resolved to.
    "is_llama_config",
    "max_position_embeddings",
    // Kimi Linear's spelling of the same serving bound.
    "model_max_length",
    "num_kv_shared_layers",
    "query_pre_attn_scalar",
    "final_logit_softcapping",
    "attn_logit_softcapping",
    "partial_rotary_factor",
    // The YaRN block's leaves. Each changes what attention computes —
    // `factor` sets the frequency blend AND the amplitude on every logit,
    // the betas and `truncate` set the correction band, and
    // `original_max_position_embeddings` is the window those bounds are
    // defined against — so each is judged against `PositionPolicy::Yarn`
    // by its own carriage rule, not credited for being parsed. Under a
    // non-YaRN `rope_type` (llama3, linear) the probes answer `None` and
    // the leaves report unrepresented, which is the truth until those
    // classes have a variant.
    "factor",
    "beta_fast",
    "beta_slow",
    "truncate",
    "original_max_position_embeddings",
    // GPT-OSS clamps both halves of the fused gate/up projection at
    // ±this value before the GLU. It changes what the FFN computes, so
    // it is execution-semantic wherever it is declared.
    "swiglu_limit",
    // Kimi-K3's SiTU-GLU softcaps bound the gate and up branches before
    // they multiply. Like `swiglu_limit` they change what the FFN
    // computes, so they are execution-semantic wherever they are
    // declared, and each is judged against `ExpertGatePolicy::SituGlu` by
    // its own carriage rule rather than credited for being parsed. Beside
    // an activation that is not `situ`, the probes answer `None` and the
    // leaves report unrepresented — which is the truth: the checkpoint
    // configured a combine it says it does not use.
    "activation_situ_beta",
    "activation_situ_linear_beta",
    // GPT-2's spelling of the norm epsilon `rms_norm_eps` etc. already
    // cover — same fact, fourth name; `parser.rs` folds all four into one
    // `norm_eps` read, so this shares `rms_norm_eps`'s carriage rule.
    "layer_norm_epsilon",
    // GPT-2's spelling of `hidden_act` (`gelu_new`), its spelling of the
    // serving bound — which on this lineage is ALSO the learned position
    // table's row count, so it is execution semantics here — and its
    // attention switches: whether scores are scaled at all, a per-layer
    // inverse score scaling, a reordered upcast score path, and a
    // cross-attention block. The last three are represented only as off.
    "activation_function",
    "n_positions",
    "scale_attn_weights",
    "scale_attn_by_inverse_layer_idx",
    "reorder_and_upcast_attn",
    "add_cross_attention",
    // Per-layer attention geometry and behaviour (A-9/A-11 census,
    // 2026-08-18: these were `consumed` but absent from every registry
    // here, so they silently graded `representable` instead of blocking —
    // the exact "parsed but unjudged" shape this module exists to name).
    // Which layers are sliding vs full.
    "sliding_window_pattern",
    // A second rope base for local/sliding layers, alongside `rope_theta`.
    "rope_local_base_freq",
    // Whether router weights are renormalised after top-k selection.
    "norm_topk_prob",
    // Kimi-K3's latent routed branch. `routed_expert_hidden_size` is a
    // width, but it is NOT stored geometry the way `kv_lora_rank` and
    // `moe_intermediate_size` are: those are proven carried by the placed
    // objects whose shapes they describe, and this one governs an
    // operator the checkpoint's own tensors cannot demonstrate — its
    // presence adds two projections and a norm to the forward pass, and
    // K3's expert bank is a compressed dialect that closure never
    // reaches. Execution-semantic, therefore, and judged by a carriage
    // rule with a probe rather than credited to a bank that does not
    // close. `latent_moe_use_norm` is execution-semantic for the plainest
    // reason: it decides whether an operation happens.
    "routed_expert_hidden_size",
    "latent_moe_use_norm",
    // Routing width: how many experts activate per token.
    "num_experts_per_tok",
    "num_experts_per_token",
    // The rope-scaling (YaRN / Llama-3-style) block's own leaves, besides
    // `rope_type` — every one of them is consumed and changes what rope
    // computes, and none has a schema field yet (the A-9.0 YaRN work).
    "type",
    "low_freq_factor",
    "high_freq_factor",
    "mscale",
    "mscale_all_dim",
    // Granite-style scaling multipliers (A-11.1): consumed into
    // `ModelConfig` but not yet carried past it — `embedding_multiplier`
    // is the one exception, wired through `embed_scale()`. See A-11.2/.3
    // in ROADMAP.md for the schema work that gives the other three a
    // canonical home instead of borrowing `qk_scale_factor` /
    // `output_multiplier`'s names.
    "embedding_multiplier",
    "attention_multiplier",
    "residual_multiplier",
    "logits_scaling",
    // Gemma 4 (V3-F0 witness 3). Each changes what a layer computes:
    // V taken from the K projection on the layers a family says so;
    // whether a routed expert block runs beside the dense MLP and how
    // many experts each token routes to; the head geometry the full
    // layers use instead of the component's (`global_head_dim`,
    // `num_global_key_value_heads`); the per-layer-input (PLE) width and
    // the double-wide MLP on shared-KV layers, both of which the graph
    // represents only as ABSENT (`0` / `false`), so any other declaration
    // blocks; and the tower's clipped-linears flag, likewise `false` only.
    "attention_k_eq_v",
    "enable_moe_block",
    "top_k_experts",
    "global_head_dim",
    "num_global_key_value_heads",
    "hidden_size_per_layer_input",
    "use_double_wide_mlp",
    "use_clipped_linears",
    // Hybrid linear-attention block geometry (Qwen3.5/Kimi-Linear-style):
    // changes what the linear-attention layers compute, even though no
    // `AttentionOp` variant executes them yet — see
    // `crate::format::vindex3::graph::policy::AttentionSpan` and
    // `docs/k3-funnel.md`'s R2/Kimi-Linear rung. Deliberately
    // execution-semantic, not tensor-semantic: a consumed tensor-semantic
    // key is reported representable unconditionally (proven by the graph
    // holding the operand), which would be false here — nothing places
    // these tensors.
    "linear_conv_kernel_dim",
    "linear_key_head_dim",
    "linear_value_head_dim",
    "linear_num_key_heads",
    "linear_num_value_heads",
    // Precision the linear-attention block's recurrent/SSM-adjacent state
    // is computed in — an execution-relevant fact distinct from the
    // checkpoint's overall storage `dtype`.
    "mamba_ssm_dtype",
    // The Mamba2/SSD mixer's declared geometry and forward-pass switches
    // (`Mamba2Geometry`, all-or-nothing at the parse boundary). Each
    // changes what a layer computes: the state and conv widths, the head
    // axis the scalar decay runs over, the SSD chunking (an fp
    // accumulation-order fact, not a tuning knob), the forward-time dt
    // clamp, the gated RMSNorm's presence, and the bias estate. The
    // spellings unique to the family live here; `num_heads`/`head_dim`
    // stay tensor-semantic — they describe stored operand shapes under
    // every family that declares them.
    "state_size",
    "expand",
    "conv_kernel",
    "n_groups",
    "chunk_size",
    "time_step_limit",
    "rms_norm",
    "use_bias",
    "use_conv_bias",
    // The mamba_ssm key dialect of the same mixer geometry (OuteAI
    // Mamba2Attn), read into the SAME `Mamba2Geometry` fields by
    // `Mamba2Geometry::read_mamba_ssm` — each probe answers from the
    // same surface site its HF twin does.
    "mamba2_num_heads",
    "mamba2_head_dim",
    "mamba2_conv_kernel",
    "use_mamba2_bias",
    // The hybrid's conv-QKV attention block (`ConvQkvAttnGeometry`):
    // each changes what the four attention layers compute — the head
    // and conv geometry, the partial-rotary width, and the bias estate.
    "attention_head_dim",
    "attention_conv_kernel",
    "rope_emb_dim",
    "use_attention_qkv_bias",
    "use_attention_out_bias",
    // The hybrid interleave's index-set spellings — WHICH layers attend
    // is as execution-semantic as a `layer_types` array.
    "attention_layers_idx",
    "attn_layer_idx",
    // The mamba_ssm lineage's MLP declaration: `mlp_intermediate_size: 0`
    // declares NO MLP blocks anywhere — absence as a stated program
    // fact; the padding multiple and bias flag parameterise that same
    // (possibly absent) MLP.
    "mlp_intermediate_size",
    "d_intermediate",
    "mlp_padding_size",
    "use_mlp_bias",
    // The mamba_ssm-native nested spellings (`ssm_cfg.layer` is the
    // identity-as-layer-class declaration; the attn_cfg leaves are the
    // conv-QKV block's own names for judged facts).
    "layer",
    "d_conv",
    "d_state",
    "headdim",
    "ngroups",
    "rotary_emb_dim",
    "qkv_proj_bias",
    "out_proj_bias",
    "causal",
    // Residual-stream precision, declared against a lower-precision
    // model. Execution-semantic wherever it appears.
    "residual_in_fp32",
    // Attention output gate: whether one exists, and its nonlinearity.
    // Distinct from the judged `AttentionGateSpec` a family may one day
    // return from `attention_output_gate()` — these are the checkpoint's
    // raw declaration.
    "attn_output_gate",
    "output_gate_type",
    // Multi-token-prediction head shape. No MTP-head object exists in the
    // VINDEX3 schema yet (`mtp.fc` has no placement rule either).
    "mtp_num_hidden_layers",
    "mtp_use_dedicated_embeddings",
    // mRoPE sectioning (Qwen-VL-style multi-axis position encoding).
    // `PositionPolicy` expresses unscaled single-axis rope only.
    "mrope_interleaved",
    "mrope_section",
];

/// Keys that describe stored operands: widths, depths, head geometry,
/// patching — the shape of what a container would have to hold.
pub const TENSOR_SEMANTIC_KEYS: &[&str] = &[
    // The perception encoder's declared input geometry: it fixes the
    // patch grid, and so the soft-token count the connector emits.
    "image_size",
    "hidden_size",
    "intermediate_size",
    "num_hidden_layers",
    "num_attention_heads",
    "num_key_value_heads",
    "head_dim",
    "vocab_size",
    "out_hidden_size",
    "projector_hidden_size",
    "projector_hidden_act",
    "merge_size",
    "patch_size",
    "patch_temporal",
    "pos_emb_height",
    "pos_emb_width",
    // GPT-2 aliases of shape fields above (`hidden_size`, `num_hidden_layers`,
    // `intermediate_size`, `num_attention_heads` respectively).
    "n_embd",
    "n_layer",
    "n_inner",
    "n_head",
    // Channel count of the patch embedder's input. Classified here beside
    // the other patch geometry, with its Qwen3-VL spelling.
    "num_channels",
    "in_channels",
    // Qwen3-VL perception-tower aliases of the shape fields above
    // (`num_hidden_layers`, `num_attention_heads`, `merge_size`,
    // `patch_temporal`). The same 27-layer, 16-head, 1152-wide tower under
    // a different vocabulary — the inventory reads them canonical-first, so
    // a checkpoint using the canonical spelling never reaches these.
    "depth",
    "num_heads",
    "spatial_merge_size",
    "temporal_patch_size",
    // The stored representation: what the checkpoint's raw-byte tensors
    // *are*. `quantization_config.quant_method` (`mxfp4` on GPT-OSS) and
    // its `modules_to_not_convert` exclusion list decide the encoding a
    // `U8` blocks/scales pair is placed under. Read by the inventory's
    // representation reader; proven carried by the placed object's
    // `representations[].encoding` (which names MXFP4, not U8), the same
    // way every other tensor semantic is proven by placement.
    "quant_method",
    "modules_to_not_convert",
    // The fine-grained (block-wise) FP8 scheme's two STORAGE facts.
    //
    // `fmt` is load-bearing and enforced: `e4m3` and `e5m2` are different
    // codecs of the same byte width, so decoding one as the other yields
    // plausible numbers from every byte rather than an error.
    // `StoredRepresentation::is_finegrained_fp8_e4m3` requires it, so an
    // `e5m2` checkpoint reaches a refusal instead of a wrong decode.
    //
    // `weight_block_size` is carried as PROVENANCE WITH A CROSS-CHECK,
    // not as the authority. The tile actually applied is derived per
    // tensor from the scale grid (`weight.shape / weight_scale_inv.shape`)
    // because one checkpoint may ship several grids — transformers' own
    // dequantiser does exactly this and cites MoE experts at `[1, 32]`
    // beside dense linears at `[128, 128]`. Reading the declared value
    // instead would be right on GLM-5.3-Flash and wrong by construction.
    //
    // Both are proven carried the same way `quant_method` is: by the
    // placed object's encoding, and — for these two — by
    // `larql_models::quant::fp8_finegrained` reproducing the reference
    // dequantiser BIT-EXACTLY on real checkpoint tensors
    // (`scripts/glm_fp8_dequant_gate.py`).
    "fmt",
    "weight_block_size",
    // MoE operand counts: how many expert tensors exist, not how the
    // forward pass selects among them (that's `num_experts_per_tok` etc.,
    // in `EXECUTION_SEMANTIC_KEYS`) — proven carried by the placed
    // `expert_bank` object and the operand closure over its shapes.
    "n_routed_experts",
    "num_local_experts",
    "num_experts",
    "n_shared_experts",
    "moe_intermediate_size",
    // MLA (DeepSeek-style) head/rank geometry.
    "kv_lora_rank",
    "q_lora_rank",
    "qk_nope_head_dim",
    "qk_rope_head_dim",
    "v_head_dim",
    // Gemma 4's per-layer-input vocabulary: the width of a table that is
    // absent when `hidden_size_per_layer_input` is 0 (that leaf's rule
    // holds the gate); a non-zero PLE width would place the table.
    "vocab_size_per_layer_input",
    // Perception-tower stored geometry: the output projector's pooling
    // kernel, its position-embedding table size, and its declared global
    // head width (equal to `head_dim` on Gemma 4 vision).
    "pooling_kernel_size",
    "position_embedding_size",
    // SigLIP's attention-pooling head (Gemma 3's `vision_config`):
    // `true` places a `head.*` parameter set after the encoder, `false`
    // (what every Gemma 3 checkpoint ships) means the tower's last
    // hidden state is its output and no head tensors exist. A fact about
    // which tensors the tower holds.
    "vision_use_head",
    // Input standardisation: its parameters are the placed `std_scale` /
    // `std_bias` tensors; the flag says they apply.
    "standardize", // mamba_ssm's own spelling of the hidden width, read through the
    // same alias chain `n_embd` is.
    "d_model",
    // The embedding-row padding: declared vocab rounded UP to this
    // multiple is the stored row count — a fact about the tensor the
    // graph holds.
    "pad_vocab_size_multiple",
];
