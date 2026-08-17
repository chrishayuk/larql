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
    "layer_norm_eps",
    "rms_norm_eps",
    "norm_epsilon",
    "rope_theta",
    "rope_type",
    "layer_types",
    "sliding_window",
    "max_position_embeddings",
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
];

/// Keys that describe stored operands: widths, depths, head geometry,
/// patching — the shape of what a container would have to hold.
pub const TENSOR_SEMANTIC_KEYS: &[&str] = &[
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
    // The stored representation: what the checkpoint's raw-byte tensors
    // *are*. `quantization_config.quant_method` (`mxfp4` on GPT-OSS) and
    // its `modules_to_not_convert` exclusion list decide the encoding a
    // `U8` blocks/scales pair is placed under. Read by the inventory's
    // representation reader; proven carried by the placed object's
    // `representations[].encoding` (which names MXFP4, not U8), the same
    // way every other tensor semantic is proven by placement.
    "quant_method",
    "modules_to_not_convert",
];

/// Keys that declare a cross-component contract: hidden-state taps, block
/// protocols, special-token roles in a multimodal or drafter interface.
pub const INTERFACE_SEMANTIC_KEYS: &[&str] = &[
    "target_layer_ids",
    "block_size",
    "mask_token_id",
    "image_token_id",
    "video_token_id",
];

/// Identity facts inert for a forward pass wherever they appear.
pub const METADATA_KEYS: &[&str] = &["model_type", "tie_word_embeddings"];

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
        Some(first) if first.ends_with(CONFIG_SUFFIX) => {
            first[..first.len() - CONFIG_SUFFIX.len()].to_string()
        }
        _ => ROOT_COMPONENT.to_string(),
    }
}

/// Last dot-separated segment of a flattened path.
pub fn leaf_of(path: &str) -> &str {
    path.rsplit('.').next().unwrap_or(path)
}
