//! Semantic classification of config keys the parser does not consume.
//!
//! The registry maps *known* HF config-field names to a semantic class.
//! It is a vocabulary of the HF config format, not of any model family —
//! `qk_scale_factor` means an attention-scale override whichever checkpoint
//! declares it. A name the registry has never seen classifies as
//! [`SemanticClass::Unknown`], which blocks the plan: an unjudged key must
//! not pass silently, because "unconsumed and unjudged" is exactly the
//! silent-default shape the whole instrument exists to catch.

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

/// The canonical spelling this leaf aliases, if it is a registered alias.
pub fn alias_canonical(leaf: &str) -> Option<&'static str> {
    ALIAS_KEYS
        .iter()
        .find(|(alias, _)| *alias == leaf)
        .map(|(_, canonical)| *canonical)
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
    } else if TRAINING_ONLY_KEYS.contains(&leaf) {
        SemanticClass::TrainingOnly
    } else if alias_canonical(leaf).is_some() {
        SemanticClass::Alias
    } else if IGNORED_SAFE_KEYS.contains(&leaf) {
        SemanticClass::IgnoredSafe
    } else {
        SemanticClass::Unknown
    }
}

/// Logical component a flattened config path belongs to.
///
/// `<name>_config.<rest>` attributes to `<name>` (`text_config.x` → `text`);
/// everything else is the artifact root.
pub fn component_of(path: &str) -> String {
    const CONFIG_SUFFIX: &str = "_config";
    const ROOT_COMPONENT: &str = "root";
    match path.split('.').next() {
        // A section that parameterises an operator of the main stack is
        // not a component of its own, so its keys belong to the stack that
        // runs that operator. Naming a component here that the graph never
        // builds sends every probe looking for it and finding nothing —
        // which reads as "not carried" for facts that are carried
        // perfectly well. `linear_attn_config` is the case.
        Some(first)
            if first.ends_with(CONFIG_SUFFIX)
                && !larql_models::inventory::is_operator_config_section(first) =>
        {
            first[..first.len() - CONFIG_SUFFIX.len()].to_string()
        }
        _ => ROOT_COMPONENT.to_string(),
    }
}

/// Last dot-separated segment of a flattened path.
pub fn leaf_of(path: &str) -> &str {
    path.rsplit('.').next().unwrap_or(path)
}
