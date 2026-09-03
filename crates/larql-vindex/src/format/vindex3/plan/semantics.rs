//! Semantic classification of config keys the parser does not consume.
//!
//! The registry maps *known* HF config-field names to a semantic class.
//! It is a vocabulary of the HF config format, not of any model family —
//! `qk_scale_factor` means an attention-scale override whichever checkpoint
//! declares it. A name the registry has never seen classifies as
//! [`SemanticClass::Unknown`], which blocks the plan: an unjudged key must
//! not pass silently, because "unconsumed and unjudged" is exactly the
//! silent-default shape the whole instrument exists to catch.

use serde::{Deserialize, Serialize};

use super::report::SemanticClass;

/// Keys that change what a forward pass computes: norms, activations,
/// position encoding, attention/output scaling, attention span policy.
pub const EXECUTION_SEMANTIC_KEYS: &[&str] = &[
    "layer_rope_theta",
    "qk_scale_factor",
    "output_multiplier",
    "post_norm_eps",
    "hidden_activation",
    "hidden_act",
    "attention_bias",
    "mlp_bias",
    "layer_norm_eps",
    "rms_norm_eps",
    "norm_epsilon",
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
    "moe_router_activation_func",
    "scoring_func",
    // The two-set interleave and the KDA conv width.
    "kda_layers",
    "full_attn_layers",
    "short_conv_kernel_size",
    // The KDA decay clamp, which changes the decay envelope without
    // changing any shape.
    "gate_lower_bound",
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
    // GPT-2's spelling of the norm epsilon `rms_norm_eps` etc. already
    // cover — same fact, fourth name; `parser.rs` folds all four into one
    // `norm_eps` read, so this shares `rms_norm_eps`'s carriage rule.
    "layer_norm_epsilon",
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

/// Keys that declare a cross-component contract: hidden-state taps, block
/// protocols, special-token roles in a multimodal or drafter interface.
pub const INTERFACE_SEMANTIC_KEYS: &[&str] = &[
    "target_layer_ids",
    "block_size",
    "mask_token_id",
    "image_token_id",
    "video_token_id",
    // The rest of a multimodal join (Gemma 4): the tokens that open,
    // close or stand in for an audio / image span, how many soft tokens
    // an image expands to (declared twice — root and tower), the span
    // kind the text model attends bidirectionally over, and a tower the
    // checkpoint declares it does NOT have (`audio_config: null`).
    "audio_token_id",
    "boi_token_id",
    "eoi_token_id",
    "boa_token_id",
    "eoa_token_id",
    "eoa_token_index",
    "vision_soft_tokens_per_image",
    "default_output_length",
    "use_bidirectional_attention",
    "audio_config",
];

/// Identity facts inert for a forward pass wherever they appear.
pub const METADATA_KEYS: &[&str] = &[
    // HF's dynamic-import map: which Python class to load for this
    // `model_type`. Loader plumbing for another runtime entirely — it
    // names code, not a forward-pass fact, and two checkpoints differing
    // only here compute identical logits.
    "AutoConfig",
    "AutoModel",
    "AutoModelForCausalLM",
    "model_type",
    "tie_word_embeddings",
    // The mamba_ssm lineage spellings of the same fact, read by the same
    // parser fallback chain.
    "tie_embedding_weights",
    "tie_embeddings",
    // Whether the reference runtime fuses the residual add with the norm
    // — a kernel-schedule fact about the SAME operation, the exact class
    // `cache_implementation` sits in: two checkpoints differing only
    // here compute the same function.
    "fused_add_norm",
    // `rope_scaling` as a bare leaf (not recursed into) means its value is
    // not an object — in every checkpoint on hand, `null`. A non-null
    // object never reaches this leaf; it flattens into `rope_type`/
    // `factor`/etc. instead, covered above. So a bare `rope_scaling` fact
    // carries no scaling information to lose — the same claim
    // `max_position_embeddings` makes about itself, just true unconditionally
    // here rather than by schema absence.
    "rope_scaling",
    // HF's serving-time KV-cache implementation selector (`"hybrid"`,
    // `"static"`, …) — which cache *class* generation code should
    // instantiate to hold a mix of sliding/full attention layers
    // efficiently. It names a consequence of the per-layer attention
    // topology, not an independent forward-pass fact: the topology itself
    // is declared elsewhere (`sliding_window` + the architecture's layer
    // alternation, e.g. Gemma 2's fixed period-2 pattern) and VINDEX3
    // already carries *that*, per layer, in the attention table. Two
    // checkpoints differing only in `cache_implementation` compute
    // identical logits for any prompt both can hold.
    "cache_implementation",
];

/// Keys that parameterise *training* and are inert at inference. Each
/// entry must name the training-time path it belongs to, because "we
/// don't run that" is the entire justification for dropping it.
pub const TRAINING_ONLY_KEYS: &[&str] = &[
    // MoE load-balancing auxiliary loss: added to the training objective,
    // never read on a forward pass.
    "router_aux_loss_coef",
    // Whether the model *returns* router logits alongside hidden states —
    // a training/analysis output switch. It changes what is returned, not
    // what is computed, and generic execution returns logits only.
    "output_router_logits",
    // Mamba2 `dt_bias` initialisation bounds (`Mamba2Mixer.__init__`
    // samples dt in [time_step_min, time_step_max], floors it at
    // time_step_floor, and stores softplus⁻¹ of it as the initial
    // `dt_bias`). Once the checkpoint ships a trained `dt_bias` tensor,
    // these parameterise nothing a forward pass reads — the forward-time
    // clamp is `time_step_limit`, which is execution-semantic above.
    "time_step_floor",
    "time_step_min",
    "time_step_max",
    // Mamba1's dt-projection rank. Mamba2's dt is a per-head scalar from
    // the fused `in_proj`; transformers carries the field on
    // `Mamba2Config` for lineage and its `__init__` alone reads it.
    "time_step_rank",
    // Weight-init scaling for the residual projections
    // (`_init_weights` divides by √(2·n_layer) when set) — same class as
    // `initializer_range`, inert once training is over.
    "rescale_prenorm_residual",
    // The mamba_ssm lineage's per-tensor init bounds (OuteAI Mamba2Attn):
    // A's log-uniform sampling range, and the conv/embedding init ranges
    // — the same class as `initializer_range`, split per tensor. Inert
    // once the trained tensors ship.
    "A_initializer_range",
    "conv_initializer_range",
    "emb_initializer_range",
    // Dropout on a classification head this causal-LM config does not
    // have — and dropout is identity at inference regardless.
    "classifier_dropout",
];

/// Redundant spellings: `alias → canonical`. An entry claims the same
/// fact is declared under `canonical` *in the same config* and read
/// there, which the gate verifies — so listing a key here cannot silence
/// it if the canonical spelling is missing or disagrees.
pub const ALIAS_KEYS: &[(&str, &str)] = &[
    // GPT-OSS declares both spellings, with the same value; the parser's
    // alias list reads `num_experts_per_tok`.
    ("experts_per_token", "num_experts_per_tok"),
    // The pre-scaling context length, also declared inside the rope
    // scaling block, which is where the parser reads it.
    (
        "initial_context_length",
        "rope_scaling.original_max_position_embeddings",
    ),
    // The regular-interval spelling of a hybrid interleave ("every Nth
    // layer is full attention"): a compressed encoding of exactly the
    // fact the explicit `layer_types` array states per layer. Qwen3.5
    // declares both; the parser reads the array. Benign only while
    // `layer_types` is genuinely present and consumed in the same
    // config — the gate verifies that, same as every other alias.
    ("full_attention_interval", "layer_types"),
];

/// Reviewed-and-safe-to-drop keys. Empty by design until a key has actually
/// been reviewed; every future entry must carry a justification comment.
pub const IGNORED_SAFE_KEYS: &[&str] = &[];

/// Keys that configure a model COMPONENT this build does not implement:
/// `leaf → component`.
///
/// The point is arithmetic. A component is one piece of engineering
/// whatever its key count, so nine keys naming one absent indexer should
/// read as one job and not nine mysteries. `Unknown` cannot say that —
/// it means "nobody has looked" — and a report made mostly of `unknown`
/// tells you how much was unexamined rather than how much is left.
///
/// # The registration rule
///
/// **An entry requires positive evidence of component ownership, never
/// merely plausible adjacency.**
///
/// Concretely, that rules out the shortcuts that would make this table
/// cheap to extend:
///
/// * no prefix-only registration — `index*` is how these keys were
///   *found*, not why they are grouped;
/// * no regex or pattern rule, ever. A regex over `rope` is what filed
///   `indexer_rope_interleave` under general RoPE, where acting on it
///   would have re-paired the whole model's rotary against the wrong
///   partners;
/// * a key that merely sits beside a registered one stays
///   [`SemanticClass::Unknown`], which is the honest answer.
///
/// What counts as positive evidence is a semantic witness: a geometry
/// self-consistent and distinct from the model's own, a value that is a
/// selection count rather than a width, an array whose length equals the
/// layer count and so is a per-layer schedule for *this* component. The
/// discovery tool may be a pattern; the authority may not be.
///
/// The cost of relaxing this is not a wrong label — it is an engineering
/// estimate that reads tidier than the evidence supports, which is worse
/// than no estimate.
pub const UNSUPPORTED_COMPONENT_KEYS: &[(&str, &str)] = &[
    // GLM-5.3-Flash's learned SPARSE ATTENTION INDEXER: a side network
    // that scores keys so attention can read `index_topk` of them
    // instead of the whole context.
    //
    // Grouped on evidence rather than on the `index` prefix alone. The
    // geometry is self-consistent and separate from the model's own
    // (`index_n_heads: 32`, `index_head_dim: 128`, against the text
    // stack's `qk_head_dim: 256`); `index_topk: 2048` is a selection
    // count, not a width; the pooling trio describes one mechanism; and
    // `indexer_types` carries exactly 45 entries against
    // `num_hidden_layers: 45`, so it is a per-layer schedule for this
    // component the way `layer_types` is for attention.
    //
    // No reference implementation exists to check any of this against:
    // `glm5_next` is absent from transformers 5.5.0 and the repo ships
    // no remote modeling code (72 files, zero `.py`). So the component is
    // NAMED and REFUSED, never guessed at — which is the whole difference
    // between an engineering estimate and a compatibility claim.
    ("index_head_dim", GLM_SPARSE_INDEXER),
    ("index_n_heads", GLM_SPARSE_INDEXER),
    ("index_topk", GLM_SPARSE_INDEXER),
    ("index_kpool", GLM_SPARSE_INDEXER),
    ("index_kpool_compress", GLM_SPARSE_INDEXER),
    ("index_kpool_always_select_tail", GLM_SPARSE_INDEXER),
    ("index_share_for_mtp_iteration", GLM_SPARSE_INDEXER),
    ("indexer_types", GLM_SPARSE_INDEXER),
    // The indexer's OWN rotary pairing, and the reason this key is worth
    // singling out: a regex over `rope` swept it into the general RoPE
    // cluster, where "fixing" it would have meant applying an interleaved
    // pairing to the whole model. Wrong component AND wrong operator, and
    // the checkpoint declares `true`, so the mistake would have been live
    // rather than latent.
    ("indexer_rope_interleave", GLM_SPARSE_INDEXER),
    // Hyper-connections: a residual stream widened by `hc_mult` and mixed
    // by a Sinkhorn-normalised map. Grouped on the shared `hc_` prefix
    // plus `hc_sinkhorn_iters`, which names a specific algorithm rather
    // than a generic knob.
    //
    // `mhc: true` sits beside these and is NOT listed. It is a bare
    // boolean whose expansion cannot be checked without a reference, and
    // guessing it into this table is exactly the failure the table's
    // contract forbids. It stays `unknown`, which is what it is.
    ("hc_eps", GLM_HYPER_CONNECTIONS),
    ("hc_mult", GLM_HYPER_CONNECTIONS),
    ("hc_sinkhorn_iters", GLM_HYPER_CONNECTIONS),
];

/// Component label for GLM's learned sparse attention indexer.
const GLM_SPARSE_INDEXER: &str = "sparse attention indexer (GLM-5.x)";

/// Component label for GLM's hyper-connection residual mixing.
const GLM_HYPER_CONNECTIONS: &str = "hyper-connections (GLM-5.x)";

/// The unimplemented component this leaf configures, if any.
pub fn unsupported_component(leaf: &str) -> Option<&'static str> {
    UNSUPPORTED_COMPONENT_KEYS
        .iter()
        .find(|(key, _)| *key == leaf)
        .map(|(_, component)| *component)
}

/// The canonical spelling this leaf aliases, if it is a registered alias.
pub fn alias_canonical(leaf: &str) -> Option<&'static str> {
    ALIAS_KEYS
        .iter()
        .find(|(alias, _)| *alias == leaf)
        .map(|(_, canonical)| *canonical)
}

/// `generate()` defaults that ship inside `config.json`.
///
/// Transformers moved decoding policy to `generation_config.json` years
/// ago and still reads these from the model config for old checkpoints,
/// so they keep appearing. Every entry selects among a forward pass's
/// outputs or says what `generate()` returns; none changes what the
/// forward pass computes, which is what a container represents.
///
/// The contract is the same as every table here: an entry is a claim
/// about a specific key, not a pattern. `top_k` is listed because it is
/// the sampling cutoff; MoE's `num_experts_per_tok` is the routing
/// top-k and is execution-semantic, which is exactly why membership is
/// by exact leaf name and never by a substring.
pub const GENERATION_POLICY_KEYS: &[&str] = &[
    // Sampling policy.
    "do_sample",
    "temperature",
    "top_k",
    "top_p",
    "typical_p",
    "epsilon_cutoff",
    "eta_cutoff",
    "repetition_penalty",
    "encoder_repetition_penalty",
    "length_penalty",
    "no_repeat_ngram_size",
    "encoder_no_repeat_ngram_size",
    "exponential_decay_length_penalty",
    "renormalize_logits",
    "remove_invalid_values",
    // Search policy.
    "num_beams",
    "num_beam_groups",
    "diversity_penalty",
    "early_stopping",
    "num_return_sequences",
    "penalty_alpha",
    // Length policy.
    "max_length",
    "min_length",
    "max_new_tokens",
    "min_new_tokens",
    // Token constraints. Vocabulary *positions*, not model geometry:
    // banning a token changes which of the logits may be selected, never
    // the logits.
    "bad_words_ids",
    "begin_suppress_tokens",
    "suppress_tokens",
    "forced_bos_token_id",
    "forced_eos_token_id",
    "forced_decoder_ids",
    // What `generate()` hands back. `output_hidden_states` and
    // `output_attentions` make the forward pass *retain* intermediates;
    // they do not change the values it computes.
    "output_scores",
    "output_attentions",
    "output_hidden_states",
    "output_logits",
    "return_dict",
    "return_dict_in_generate",
    "num_logits_to_keep",
    "use_cache",
    // Legacy per-task default blocks (`task_specific_params.*`) and the
    // GPT-2-era sequence-classification head knobs, which describe a head
    // no text-generation container builds.
    "summary_type",
    "summary_use_proj",
    "summary_activation",
    "summary_proj_to_labels",
    "summary_first_dropout",
];

/// Regularisation rates. Dropout is the identity at inference — the
/// module is placed but never active outside training — so the RATE
/// parameterises a path a container never runs.
///
/// Value-independent, unlike [`INERT_AT_VALUE`]: a dropout of 0.0 and one
/// of 0.5 compute the same forward pass here.
pub const DROPOUT_KEYS: &[&str] = &[
    "attn_pdrop",
    "embd_pdrop",
    "resid_pdrop",
    "summary_pdrop",
    "attention_dropout",
    "hidden_dropout",
    "embedding_dropout",
    "residual_dropout",
    "classifier_dropout",
    "mlp_dropout",
    "conv_dropout",
    // MoE router noise, added to logits during training only.
    "input_jitter_noise",
    "router_jitter_noise",
];

/// A value a key must hold to be inert.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum InertValue {
    /// Inert when the declared integer equals this.
    Int(i64),
    /// Inert when the declared array is empty — a schedule that selects
    /// no layers is not a schedule.
    EmptyList,
}

impl InertValue {
    fn matches(self, value: &serde_json::Value) -> bool {
        match self {
            Self::Int(want) => value.as_i64() == Some(want),
            Self::EmptyList => value.as_array().is_some_and(|a| a.is_empty()),
        }
    }
}

/// Keys inert **at one value** and execution-semantic at any other.
///
/// `pretraining_tp` is why this table is keyed by value rather than by
/// name. It reads as a training-time knob and is not one: HF Llama's
/// forward pass branches on it, slicing every projection into `tp` shards
/// and summing them, to reproduce the numerics of the tensor-parallel run
/// that trained the weights. At the `1` that every checkpoint in the
/// conformance corpus declares it is exactly a no-op. Listing the NAME
/// would have silenced a key that changes the forward pass the moment a
/// checkpoint ships `2` — a checked default, never an assumed one.
///
/// An entry here must name a value whose inertness is *verified*, not
/// assumed from the key reading like a default.
pub const INERT_AT_VALUE: &[(&str, InertValue)] = &[
    ("pretraining_tp", InertValue::Int(1)),
    // Qwen's MoE layer schedule: which layers route to an expert bank and
    // which run a plain MLP. `decoder_sparse_step` is the stride (HF:
    // `layer_idx % decoder_sparse_step == 0` is a MoE layer) and
    // `mlp_only_layers` names the exceptions.
    //
    // At a stride of 1 with no exceptions, every layer is a MoE layer —
    // which is the uniform stack the graph already builds, so the pair
    // describes what is already represented. Any OTHER value is a real
    // per-layer topology: a stride of 2 makes half the tower dense, and a
    // non-empty exception list carves out named layers. Neither is
    // expressible today, and both keep blocking.
    //
    // Value-keyed rather than name-listed for exactly that reason. This
    // is the same shape as `pretraining_tp` above: a key that reads like
    // a default and is one only at one value.
    ("decoder_sparse_step", InertValue::Int(1)),
    ("mlp_only_layers", InertValue::EmptyList),
];

/// The inert value registered for this leaf, if any.
pub fn inert_at_value(leaf: &str) -> Option<InertValue> {
    INERT_AT_VALUE
        .iter()
        .find(|(key, _)| *key == leaf)
        .map(|(_, v)| *v)
}

/// [`classify_key`], with the declared value available.
///
/// The value settles exactly one question — whether a key registered in
/// [`INERT_AT_VALUE`] is holding the value that makes it inert. Every
/// other classification is by name, so a key not in that table answers
/// identically here and in [`classify_key`].
pub fn classify_key_at(leaf: &str, value: &serde_json::Value) -> SemanticClass {
    match inert_at_value(leaf) {
        // Inert at this value: declared, preserved, and read by nothing a
        // forward pass runs.
        Some(inert) if inert.matches(value) => SemanticClass::TrainingOnly,
        // Registered, but holding some OTHER value: this is the case the
        // table exists to keep blocking.
        Some(_) => SemanticClass::ExecutionSemantic,
        None => classify_key(leaf),
    }
}

/// The model concept a finding is about.
///
/// This is the census's third axis, and it answers a different question
/// from [`SemanticClass`]. The class says how much losing a subject would
/// matter and decides whether it blocks; the cluster says *which idea of
/// the model* it belongs to. Forty `rope_*` spellings across ten
/// organisations are forty subjects, one idea — and the unit of
/// remediation work is the idea, never the spelling.
///
/// It lives here, keyed by exact leaf name, rather than in a script's
/// regex, because a regex over finding text is a fourth authority on a
/// question three already answer. `num_experts_per_tok` and a sampling
/// `top_k` share no substring rule that separates them; an exact table
/// does.
///
/// [`Self::Unclustered`] is honest and expected: a subject no table has
/// judged. It is a *finding about the taxonomy*, not a bucket to grow by
/// pattern-matching, and the conformance report counts it as its own row.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum SemanticCluster {
    PositionRope,
    PositionMrope,
    AttentionSchedule,
    AttentionBias,
    AttentionSparseIndexer,
    AttentionLogitSoftcapping,
    MoeRouting,
    FfnActivation,
    NormGeometry,
    ShapeAndTensorNaming,
    MixerSsmGeometry,
    DecodeMultiTokenPrediction,
    ResidualHyperConnections,
    ResidualWiring,
    ScaleMultipliers,
    RepresentationQuantization,
    ModalityVision,
    ModalityAudio,
    /// The graph could not be completed — a consequence finding, whose
    /// cause is one of the others.
    ExecutionSurface,
    /// Who the checkpoint says it is.
    ArchitectureIdentity,
    InertTrainingOnly,
    InertGenerationPolicy,
    Unclustered,
}

/// Leaf names whose concept is known, grouped by concept.
///
/// Exact names throughout. The contract is the same one every table in
/// this module carries: an entry is a claim about a specific key, and a
/// key nobody has judged stays [`SemanticCluster::Unclustered`] rather
/// than being pattern-matched into a neighbour.
const CLUSTER_KEYS: &[(SemanticCluster, &[&str])] = &[
    (
        SemanticCluster::PositionRope,
        &[
            "rope_theta",
            "rope_type",
            "type",
            "factor",
            "rope_scaling",
            "rope_parameters",
            "low_freq_factor",
            "high_freq_factor",
            "original_max_position_embeddings",
            "attention_factor",
            "beta_fast",
            "beta_slow",
            "mscale",
            "mscale_all_dim",
            "short_factor",
            "long_factor",
            "short_mscale",
            "long_mscale",
            "partial_rotary_factor",
            "rotary_pct",
            "rotary_emb_base",
            "rope_interleave",
            "rope_local_base_freq",
            "no_rope_layers",
            "no_rope_layer_interval",
            "position_embedding_type",
            "n_positions",
            "n_ctx",
            "interpolate_factor",
            "llama_4_scaling_beta",
            "rope_scaling_factor",
            "use_mrope",
        ],
    ),
    (
        SemanticCluster::PositionMrope,
        &["mrope_section", "mrope_interleaved"],
    ),
    (
        SemanticCluster::AttentionSchedule,
        &[
            "sliding_window",
            "use_sliding_window",
            "sliding_window_size",
            "layer_types",
            "local_layer_ids",
            "full_attention_interval",
            "attn_layer_indices",
            "attn_layer_offset",
            "attn_layer_period",
            "attention_chunk_size",
            "full_attn_mod",
            "attention_policy",
            "max_window_layers",
            "num_kv_shared_layers",
        ],
    ),
    (
        SemanticCluster::AttentionBias,
        &[
            "attention_bias",
            "qkv_bias",
            "use_bias",
            "bias",
            "clip_qkv",
            "o_proj_bias",
            "attention_out_bias",
        ],
    ),
    (
        SemanticCluster::AttentionSparseIndexer,
        &[
            "index_topk",
            "index_n_heads",
            "index_head_dim",
            "index_topk_freq",
            "index_topk_pattern",
            "index_skip_topk_offset",
            "index_kpool",
            "index_kpool_compress",
            "index_kpool_always_select_tail",
            "indexer_rope_interleave",
            "compress_ratios",
            "compress_rope_theta",
            "o_lora_rank",
            "o_groups",
        ],
    ),
    (
        SemanticCluster::AttentionLogitSoftcapping,
        &[
            "attn_logit_softcapping",
            "final_logit_softcapping",
            "query_pre_attn_scalar",
        ],
    ),
    (
        SemanticCluster::MoeRouting,
        &[
            "num_experts",
            "num_local_experts",
            "num_experts_per_tok",
            "n_routed_experts",
            "n_shared_experts",
            "shared_expert_intermediate_size",
            "moe_intermediate_size",
            "expert_intermediate_size",
            "top_k_experts",
            "topk_method",
            "topk_group",
            "n_group",
            "scoring_func",
            "norm_topk_prob",
            "routed_scaling_factor",
            "decoder_sparse_step",
            "mlp_only_layers",
            "moe_layer_freq",
            "first_k_dense_replace",
            "num_experts_shared",
            "use_qk_norm",
        ],
    ),
    (
        SemanticCluster::FfnActivation,
        &[
            "hidden_act",
            "hidden_activation",
            "activation",
            "activation_function",
            "swiglu_limit",
            "mlp_type",
            "mlp_expansion_factor",
            "use_double_wide_mlp",
        ],
    ),
    (
        SemanticCluster::NormGeometry,
        &[
            "rms_norm_eps",
            "layer_norm_eps",
            "layer_norm_epsilon",
            "norm_eps",
            "normalization_type",
            "qk_layernorm",
            "use_qk_layernorm",
        ],
    ),
    (
        SemanticCluster::MixerSsmGeometry,
        &[
            "mamba_d_conv",
            "mamba_d_state",
            "mamba_expand",
            "mamba_n_heads",
            "mamba_n_groups",
            "mamba_conv_bias",
            "mamba_proj_bias",
            "mamba_d_head",
            "mamba_chunk_size",
            "conv_kernel",
            "state_size",
            "linear_conv_kernel_dim",
            "linear_attn_config",
            "time_step_limit",
        ],
    ),
    (
        SemanticCluster::DecodeMultiTokenPrediction,
        &[
            "num_nextn_predict_layers",
            "mtp_num_hidden_layers",
            "mtp_use_dedicated_embeddings",
            "layers",
            "fc",
            "norm",
            "pre_fc_norm_embedding",
            "pre_fc_norm_hidden",
        ],
    ),
    (
        SemanticCluster::ResidualHyperConnections,
        &[
            "hc_mult",
            "hc_eps",
            "hc_sinkhorn_iters",
            "hc_count",
            "hc_lowrank",
            "hc_head_base",
            "hc_head_fn",
            "hc_head_scale",
            "mhc",
        ],
    ),
    (
        SemanticCluster::ResidualWiring,
        &[
            "use_parallel_residual",
            "attn_res_block_size",
            "output_attn_res_proj",
        ],
    ),
    (
        SemanticCluster::ScaleMultipliers,
        &[
            "attention_multiplier",
            "embedding_multiplier",
            "logits_scaling",
            "residual_multiplier",
            "attention_in_multiplier",
            "attention_out_multiplier",
            "key_multiplier",
            "lm_head_multiplier",
            "mlp_multipliers",
        ],
    ),
    (SemanticCluster::ArchitectureIdentity, &["is_llama_config"]),
    (
        SemanticCluster::ShapeAndTensorNaming,
        &[
            "hidden_size",
            "intermediate_size",
            "num_hidden_layers",
            "num_attention_heads",
            "num_key_value_heads",
            "head_dim",
            "vocab_size",
            "max_position_embeddings",
            "n_head",
            "n_embd",
            "n_layer",
            "n_inner",
            "qk_nope_head_dim",
            "qk_rope_head_dim",
            "v_head_dim",
            "kv_lora_rank",
            "q_lora_rank",
            "architectures",
        ],
    ),
    (
        SemanticCluster::ModalityAudio,
        &[
            "audio_token_id",
            "boa_token_id",
            "eoa_token_id",
            "eoa_token_index",
            "audio_embed_dim",
            "audio_config",
        ],
    ),
];

/// Clusters carried by a whole flattened path rather than a leaf name.
///
/// A structural finding names a rule, not a config key — `layer_census`
/// is not a leaf anyone declared — and a subject that is a *nested
/// section* (`quantization_config.…`) is owned by the section rather than
/// by whatever its last segment happens to be called.
fn cluster_by_path(subject: &str) -> Option<SemanticCluster> {
    if subject.contains("quantization_config") {
        return Some(SemanticCluster::RepresentationQuantization);
    }
    if subject.contains("vision") || subject.contains("image") || subject.contains("video") {
        return Some(SemanticCluster::ModalityVision);
    }
    if subject.contains("execution_surface") || subject == "layer_census" {
        return Some(SemanticCluster::ExecutionSurface);
    }
    if subject == "architecture_identity" || subject == "architecture_family" {
        return Some(SemanticCluster::ArchitectureIdentity);
    }
    if subject.contains("mtp") {
        return Some(SemanticCluster::DecodeMultiTokenPrediction);
    }
    None
}

/// The concept a finding's subject belongs to.
///
/// One derivation, used by the plan document and by anything reporting
/// over it. Path-scoped ownership is asked first: a key nested under a
/// section belongs to that section even when its leaf name would match a
/// general table.
pub fn cluster_for(subject: &str) -> SemanticCluster {
    if let Some(cluster) = cluster_by_path(subject) {
        return cluster;
    }
    let leaf = leaf_of(subject);
    if let Some((cluster, _)) = CLUSTER_KEYS.iter().find(|(_, keys)| keys.contains(&leaf)) {
        return *cluster;
    }
    // Fall back to what the key IS, when the registry already judged it
    // inert. These are concepts too — "this is decoding policy" is a
    // statement about the model, and reporting them as unclustered would
    // hide the largest cheap win in the corpus behind a shrug.
    match classify_key(leaf) {
        SemanticClass::GenerationPolicy => SemanticCluster::InertGenerationPolicy,
        SemanticClass::TrainingOnly => SemanticCluster::InertTrainingOnly,
        _ => SemanticCluster::Unclustered,
    }
}

/// Classify an unconsumed config key by its leaf name.
pub fn classify_key(leaf: &str) -> SemanticClass {
    if EXECUTION_SEMANTIC_KEYS.contains(&leaf) {
        SemanticClass::ExecutionSemantic
    } else if TENSOR_SEMANTIC_KEYS.contains(&leaf) {
        SemanticClass::TensorSemantic
    } else if INTERFACE_SEMANTIC_KEYS.contains(&leaf) {
        SemanticClass::InterfaceSemantic
    } else if METADATA_KEYS.contains(&leaf) {
        SemanticClass::MetadataOnly
    } else if TRAINING_ONLY_KEYS.contains(&leaf) || DROPOUT_KEYS.contains(&leaf) {
        SemanticClass::TrainingOnly
    } else if GENERATION_POLICY_KEYS.contains(&leaf) {
        SemanticClass::GenerationPolicy
    } else if alias_canonical(leaf).is_some() {
        SemanticClass::Alias
    } else if IGNORED_SAFE_KEYS.contains(&leaf) {
        SemanticClass::IgnoredSafe
    } else if unsupported_component(leaf).is_some() {
        SemanticClass::UnsupportedComponent
    } else {
        SemanticClass::Unknown
    }
}

/// Logical component a flattened config path belongs to.
///
/// `<name>_config.<rest>` attributes to `<name>` (`text_config.x` → `text`);
/// everything else is the artifact root — including a bare leaf that
/// happens to end in `_config`. SmolLM2's `is_llama_config: true` is a
/// boolean at the root, not the section of a component called `is_llama`,
/// and reading it as one sent its probe to a component the graph never
/// builds. A section is a segment with something after it.
pub fn component_of(path: &str) -> String {
    const CONFIG_SUFFIX: &str = "_config";
    const ROOT_COMPONENT: &str = "root";
    match path.split_once('.') {
        // A section that parameterises an operator of the main stack is
        // not a component of its own, so its keys belong to the stack that
        // runs that operator. Naming a component here that the graph never
        // builds sends every probe looking for it and finding nothing —
        // which reads as "not carried" for facts that are carried
        // perfectly well. `linear_attn_config` is the case.
        Some((section, _rest))
            if section.ends_with(CONFIG_SUFFIX)
                && !larql_models::inventory::is_operator_config_section(section) =>
        {
            section[..section.len() - CONFIG_SUFFIX.len()].to_string()
        }
        _ => ROOT_COMPONENT.to_string(),
    }
}

/// Last dot-separated segment of a flattened path.
pub fn leaf_of(path: &str) -> &str {
    path.rsplit('.').next().unwrap_or(path)
}
