//! Flatten `config.json` into classified key facts.
//!
//! The classification registry below mirrors what
//! [`super::super::detect::parser`] actually reads. When the parser learns a
//! new key, add it to [`CONSUMED_LEAF_KEYS`] — the sync test in
//! `tests/config_keys.rs` scans the parser source for string-literal key
//! accesses and fails if one is missing here, so the two cannot drift apart
//! silently.

use serde_json::Value;

use super::report::{ConfigKeyFact, InterfaceFact, KeyStatus};

/// Leaf keys the `ModelConfig` parser reads, directly or via an alias list.
///
/// Sources: `detect/parser.rs` (all `["…"]` / `.get("…")` accesses) and the
/// alias constants in `detect/config_io.rs`.
pub const CONSUMED_LEAF_KEYS: &[&str] = &[
    // identity + required topology (incl. aliases)
    "model_type",
    "hidden_size",
    "n_embd",
    "num_hidden_layers",
    "n_layer",
    "intermediate_size",
    "n_inner",
    "num_attention_heads",
    "n_head",
    // attention shape
    "head_dim",
    "num_key_value_heads",
    "sliding_window",
    "sliding_window_pattern",
    "layer_types",
    "global_head_dim",
    "num_global_key_value_heads",
    "attention_k_eq_v",
    "num_kv_shared_layers",
    // rope
    "rope_theta",
    "rope_local_base_freq",
    "partial_rotary_factor",
    "layer_rope_theta",
    // rope_scaling / rope_parameters leaves
    "type",
    "rope_type",
    "factor",
    "low_freq_factor",
    "high_freq_factor",
    "original_max_position_embeddings",
    "beta_fast",
    "beta_slow",
    "truncate",
    "mscale",
    "mscale_all_dim",
    // vocab / embeddings
    "vocab_size",
    "tie_word_embeddings",
    "hidden_size_per_layer_input",
    "use_double_wide_mlp",
    "vocab_size_per_layer_input",
    // MoE
    "n_routed_experts",
    "num_local_experts",
    "num_experts",
    "num_experts_per_tok",
    "num_experts_per_token",
    "n_shared_experts",
    "enable_moe_block",
    "top_k_experts",
    "moe_intermediate_size",
    "swiglu_limit",
    "norm_topk_prob",
    // MLA
    "kv_lora_rank",
    "q_lora_rank",
    "qk_nope_head_dim",
    "qk_rope_head_dim",
    "v_head_dim",
    // norms
    "rms_norm_eps",
    "layer_norm_eps",
    "layer_norm_epsilon",
    "norm_epsilon",
    // softcapping + scaling multipliers
    "attn_logit_softcapping",
    "final_logit_softcapping",
    "query_pre_attn_scalar",
    "embedding_multiplier",
    "residual_multiplier",
    "attention_multiplier",
    "logits_scaling",
    // attention/output scaling + norm shape (G2b)
    "qk_scale_factor",
    "output_multiplier",
    "post_norm_eps",
    "attention_bias",
    "hidden_act",
    "hidden_activation",
    "max_position_embeddings",
    // multimodal protocol + adapter geometry (root-level)
    "image_token_id",
    "video_token_id",
    "out_hidden_size",
    "projector_hidden_size",
    "projector_hidden_act",
    // drafter interface declaration
    "target_layer_ids",
    "block_size",
    "mask_token_id",
];

/// Containers the parser recurses into: leaves inside keep consumed-leaf
/// credit. A leaf under any *other* container is never `consumed`, whatever
/// its name — `vision_config.hidden_size` shares a name with a consumed key
/// but the parser never reads it, and crediting it would under-report, which
/// is the one failure mode this instrument exists to prevent.
pub const CONSUMED_CONTAINER_KEYS: &[&str] = &[
    "text_config",
    "language_config",
    "rope_scaling",
    "rope_parameters",
    "full_attention",
    "sliding_attention",
];

/// Containers whose *presence* is the only thing the parser reads.
/// `vision_config` sets `has_vision_config`; everything inside it is
/// unread by this parser and classifies accordingly.
pub const PRESENCE_ONLY_CONTAINER_KEYS: &[&str] = &["vision_config"];

/// Identity or training-time facts known to be inert for a text forward
/// pass. Deliberately short: when in doubt a key stays `unconsumed`, because
/// an over-broad benign list is exactly how a silent default hides.
pub const METADATA_LEAF_KEYS: &[&str] = &[
    "architectures",
    "transformers_version",
    "dtype",
    "torch_dtype",
    "_name_or_path",
    "bos_token_id",
    "eos_token_id",
    "pad_token_id",
    "initializer_range",
    "use_cache",
    "attention_dropout",
    // HF `PretrainedConfig` plumbing that changes what a call RETURNS or
    // how a training/eval loop chunks work, never what a forward computes:
    // classification-head labels on a config that has no head, output
    // switches, the encoder-decoder flag, and the feed-forward chunk size
    // (identical numbers at any chunking; Gemma 4's vision tower ships 0).
    "problem_type",
    "return_dict",
    "output_attentions",
    "output_hidden_states",
    "is_encoder_decoder",
    "chunk_size_feed_forward",
];

/// Containers whose every leaf is metadata: HF label maps
/// (`id2label.0 = "LABEL_0"`), whose leaves are numbered and so cannot be
/// named in [`METADATA_LEAF_KEYS`]. Only listed as a whole because nothing
/// under them can move a forward pass.
pub const METADATA_CONTAINER_KEYS: &[&str] = &["id2label", "label2id"];

/// Config fields that declare a cross-component interface. The one known
/// occupant is the DFlash-style drafter contract: `target_layer_ids` names
/// the target-model hidden states the drafter consumes. Data-driven — the
/// field is reported wherever it appears, for any model.
pub const KNOWN_INTERFACE_KEYS: &[&str] = &["target_layer_ids"];

/// Flatten a parsed `config.json` into classified per-leaf facts, sorted by
/// path. Arrays are leaves — `layer_types` is one fact, not fifty-two.
///
/// `extra_consumed` credits full paths read by another registered reader —
/// today the nested-component topology reader
/// ([`super::components`]), which records exactly what it stored. Paths in
/// the set classify `consumed` regardless of container context, because the
/// credit comes from an actual read, not a name match.
pub fn classify_config(
    config: &Value,
    extra_consumed: &std::collections::BTreeSet<String>,
) -> Vec<ConfigKeyFact> {
    let mut facts = Vec::new();
    flatten_into(config, String::new(), true, extra_consumed, &mut facts);
    facts.sort_by(|a, b| a.path.cmp(&b.path));
    facts
}

/// Interface declarations found anywhere in the config, sorted by path.
pub fn find_interfaces(facts: &[ConfigKeyFact]) -> Vec<InterfaceFact> {
    facts
        .iter()
        .filter(|f| KNOWN_INTERFACE_KEYS.contains(&leaf_name(&f.path)))
        .map(|f| InterfaceFact {
            path: f.path.clone(),
            value: f.value.clone(),
        })
        .collect()
}

/// Every leaf under a [`METADATA_CONTAINER_KEYS`] container, as metadata.
fn flatten_metadata_into(value: &Value, prefix: String, out: &mut Vec<ConfigKeyFact>) {
    match value {
        Value::Object(map) => {
            for (key, child) in map {
                flatten_metadata_into(child, format!("{prefix}.{key}"), out);
            }
        }
        leaf => out.push(ConfigKeyFact {
            status: KeyStatus::Metadata,
            path: prefix,
            value: leaf.clone(),
        }),
    }
}

/// Last dot-separated segment of a flattened path.
fn leaf_name(path: &str) -> &str {
    path.rsplit('.').next().unwrap_or(path)
}

/// `in_consumed_context` is true only while every ancestor container is one
/// the parser recurses into. Outside that context no leaf can be `consumed`
/// by name — only by an `extra_consumed` recorded read.
fn flatten_into(
    value: &Value,
    prefix: String,
    in_consumed_context: bool,
    extra_consumed: &std::collections::BTreeSet<String>,
    out: &mut Vec<ConfigKeyFact>,
) {
    match value {
        Value::Object(map) => {
            for (key, child) in map {
                let path = if prefix.is_empty() {
                    key.clone()
                } else {
                    format!("{prefix}.{key}")
                };
                if child.is_object() && METADATA_CONTAINER_KEYS.contains(&key.as_str()) {
                    // A label map: every leaf metadata, whatever it is named.
                    flatten_metadata_into(child, path, out);
                } else if child.is_object() {
                    // Recurse always — every leaf must surface individually —
                    // but consumed-credit survives only through containers the
                    // parser itself recurses into. Presence-only containers
                    // and unknown objects cut the context.
                    let child_context = in_consumed_context
                        && CONSUMED_CONTAINER_KEYS.contains(&key.as_str())
                        && !PRESENCE_ONLY_CONTAINER_KEYS.contains(&key.as_str());
                    flatten_into(child, path, child_context, extra_consumed, out);
                } else {
                    out.push(ConfigKeyFact {
                        status: classify_leaf(key, &path, in_consumed_context, extra_consumed),
                        path,
                        value: child.clone(),
                    });
                }
            }
        }
        // A non-object at the root has no key structure to classify; a real
        // config.json never has this shape.
        _ => {
            out.push(ConfigKeyFact {
                status: KeyStatus::Unconsumed,
                path: prefix,
                value: value.clone(),
            });
        }
    }
}

fn classify_leaf(
    key: &str,
    path: &str,
    in_consumed_context: bool,
    extra_consumed: &std::collections::BTreeSet<String>,
) -> KeyStatus {
    // Credit either a recorded component-reader read (full path) or a
    // name-registry hit inside a parser-recursed container.
    if extra_consumed.contains(path) || (in_consumed_context && CONSUMED_LEAF_KEYS.contains(&key)) {
        KeyStatus::Consumed
    } else if METADATA_LEAF_KEYS.contains(&key) {
        KeyStatus::Metadata
    } else {
        KeyStatus::Unconsumed
    }
}
