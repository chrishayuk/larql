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
    // Read by `ModelArchitecture::sliding_window_size`, which resolves the
    // effective policy from all three declarations at once — the flag is
    // credited because something reads it and acts on what it read.
    "use_sliding_window",
    // Read by `ModelArchitecture::is_sliding_window_layer` as the bound on
    // how far up the stack an enabled window applies.
    "max_window_layers",
    // Read by `ModelArchitecture::position_policy_for_layer`, where the
    // family that owns the spelling interprets it: on `granitemoehybrid`
    // it is the opt-in that turns rotation on at all, and the same leaf
    // means `absolute` / `relative_key` in the BERT lineage.
    "position_embedding_type",
    // The per-layer rotary SCHEDULE (SmolLM3, Llama 4), resolved by
    // `position_policy_for_layer` before the rotary shape. The mask's
    // polarity is inverted relative to its name and is honoured once, in
    // `PositionPolicy::rope_enabled_by_flag`; the interval is the
    // fallback both references consult only when the mask is absent.
    "no_rope_layers",
    "no_rope_layer_interval",
    // Two declarations no reference implementation reads. Consumed here
    // so agreement is CHECKED: this build pairs split-half and resolves
    // a text policy, and a checkpoint claiming otherwise must mismatch
    // rather than be quietly overridden.
    "rope_interleaved",
    "use_mrope",
    // Two more of the same kind (wave 9 of the conformance sweep):
    // Falcon's one-word FFN shape and SmolLM2's family flag. Stored by
    // the parser, judged against the FFN and the family that actually
    // resolve.
    "activation",
    "is_llama_config",
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
    // E30 static shards: a derived checkpoint's per-layer dense-FFN width
    "larql_ffn_intermediate_size_by_layer",
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
    // The shared branch's own width, in both spellings: Qwen MoE writes
    // `shared_expert_intermediate_size`, Nemotron-H prefixes it.
    "shared_expert_intermediate_size",
    "moe_shared_expert_intermediate_size",
    // Hyper-connections (DeepSeek-V4): the residual is a bundle of
    // `hc_mult` streams, reduced and expanded per token through a
    // Sinkhorn-split mixing matrix. All three are read together — a
    // partial declaration refuses rather than being completed.
    "hc_mult",
    "hc_sinkhorn_iters",
    "hc_eps",
    // Attention residuals (Kimi-K3): the residual is one vector plus a
    // history of block-boundary snapshots of it, taken every
    // `attn_res_block_size` layers and read by every sublayer.
    "attn_res_block_size",
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
    // LFM2's spelling. Its separate `block_norm_eps` is a DIFFERENT
    // fact (the FFN blocks' epsilon) and is deliberately not credited
    // here — nothing reads it yet, and it must keep saying so.
    "norm_eps",
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
    "mlp_bias",
    // `rope_scaling` itself, distinct from the leaves inside it: the parser
    // calls `text_config.get("rope_scaling")` unconditionally (below,
    // `rope_type` etc.), so a checkpoint that declares the key `null` — no
    // scaling block, not an omission — is genuinely read, not silently
    // skipped. Declaring it here (as well as in `CONSUMED_CONTAINER_KEYS`,
    // which governs the object case) covers the scalar case too, since a
    // non-object value never recurses and would otherwise flatten to an
    // `Unconsumed` leaf despite the read actually happening.
    "rope_scaling",
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
    // hybrid linear-attention + multi-token-prediction (declared,
    // R2/Kimi-Linear-rung prep — see `docs/k3-funnel.md`)
    // The declared interleave and its window, in the flat spellings.
    // `local_layer_ids` is Inkling-Small's; `sliding_window_size` is the
    // same fact `sliding_window` states for every family before it.
    "local_layer_ids",
    "sliding_window_size",
    // MoE facts in the spellings Kimi Linear uses, each read into the same
    // canonical field its DeepSeek-lineage twin fills.
    "num_shared_experts",
    "moe_renormalize",
    "routed_scaling_factor",
    "n_group",
    "num_expert_group",
    "topk_group",
    "use_grouped_topk",
    "moe_layer_freq",
    "first_k_dense_replace",
    "mla_use_nope",
    "model_max_length",
    "moe_router_activation_func",
    "scoring_func",
    // The relative-position scheme's two parameters, declared together.
    "d_rel",
    "rel_extent",
    "num_nextn_predict_layers",
    "linear_conv_kernel_dim",
    "linear_key_head_dim",
    "linear_value_head_dim",
    "linear_num_key_heads",
    "linear_num_value_heads",
    "mamba_ssm_dtype",
    // The Mamba2 mixer geometry (`config::Mamba2Geometry::read`, all
    // fields or none) plus the residual-precision fact. `num_heads` and
    // `head_dim` are genuinely read at the top level by that same
    // geometry read; `head_dim` was already consumed as an attention key,
    // and on a Mamba2 declaration the one value is the mixer's.
    "state_size",
    "num_heads",
    "expand",
    "conv_kernel",
    "n_groups",
    "chunk_size",
    "time_step_limit",
    "rms_norm",
    "use_bias",
    "use_conv_bias",
    // The mamba_ssm key dialect of the same mixer geometry
    // (`Mamba2Geometry::read_mamba_ssm` — OuteAI Mamba2Attn), and the
    // hybrid's conv-QKV attention block
    // (`config::ConvQkvAttnGeometry::read`, all fields or none).
    // `rope_theta` and the attention head counts were already consumed
    // as attention keys; on a hybrid declaration the one value is the
    // conv-QKV block's.
    "mamba2_num_heads",
    "mamba2_head_dim",
    "mamba2_conv_kernel",
    "use_mamba2_bias",
    "attention_head_dim",
    "attention_conv_kernel",
    "rope_emb_dim",
    "use_attention_qkv_bias",
    "use_attention_out_bias",
    // The hybrid interleave's index-set spellings
    // (`interleave::spellings::read_attention_layer_idx`).
    "attention_layers_idx",
    "attn_layer_idx",
    // The mamba_ssm lineage spelling of `tie_word_embeddings`, read by
    // the same parser fallback chain.
    "tie_embedding_weights",
    // The mamba_ssm lineage's MLP declaration: width (0 = no MLP blocks,
    // a declaration), padding multiple, bias flag. `d_intermediate` is
    // the package's own spelling of the same width.
    "mlp_intermediate_size",
    "d_intermediate",
    "mlp_padding_size",
    "use_mlp_bias",
    // The mamba_ssm-native (state-spaces) top-level keys: the package's
    // own spellings of facts judged under other names — `d_model`
    // (hidden_size alias), `tie_embeddings` (the third tie spelling) —
    // plus the embedding-row padding and the fused add+norm schedule.
    "d_model",
    "tie_embeddings",
    "pad_vocab_size_multiple",
    "fused_add_norm",
    "residual_in_fp32",
    "attn_output_gate",
    "output_gate_type",
    // MLA's output gate (Kimi-K3): declared at the text level, beside the
    // softmax family's `attn_output_gate`, and read into the MLA execution
    // record as the same generic gate spec.
    "mla_use_output_gate",
    "mtp_num_hidden_layers",
    "mtp_use_dedicated_embeddings",
    "mrope_interleaved",
    "mrope_section",
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

/// Leaf names the parser reads out of a [`PATH_READ_CONTAINER_KEYS`]
/// container, credited by **full path** rather than by name.
pub const PATH_READ_LEAF_KEYS: &[&str] = &[
    // The interleave, in each spelling that lives inside a container.
    "kda_layers",
    "full_attn_layers",
    // Inkling-Small's MTP sub-stack states its own interleave and its own
    // layer count inside `mtp_config`.
    "local_layer_ids",
    "num_nextn_predict_layers",
    // The KDA block's geometry and gate clamp. `head_dim` and `num_heads`
    // are the reason this list is by-path: both are consumed leaf NAMES
    // elsewhere, so crediting them here by name would also credit them on
    // every container that happens to spell them.
    "num_heads",
    "head_dim",
    "short_conv_kernel_size",
    "gate_lower_bound",
    // The output gate's FORM (Kimi-K3's `use_full_rank_gate`): full-rank
    // `g_proj` or the low-rank pair. A leaf of `linear_attn_config`, so
    // by-path like the three above.
    "use_full_rank_gate",
    // The mamba_ssm-native nested blocks: `ssm_cfg.layer` is the
    // package's identity declaration; the rest are the geometry keys
    // its dialect reads (`Mamba2Geometry::read_mamba_ssm_native`,
    // `ConvQkvAttnGeometry::read_attn_cfg`) — every absent one a
    // RECORDED family default, never a silent fill.
    "layer",
    "expand",
    "headdim",
    "d_state",
    "d_conv",
    "ngroups",
    "chunk_size",
    "rmsnorm",
    "bias",
    "conv_bias",
    "causal",
    "num_heads_kv",
    "rotary_emb_base",
    "rotary_emb_dim",
    "qkv_proj_bias",
    "out_proj_bias",
];

/// Containers the parser reads specific leaves of, by path, without
/// recursing with name credit.
///
/// Deliberately *not* in [`CONSUMED_CONTAINER_KEYS`]. `linear_attn_config`
/// holds a `head_dim` and a `num_heads`, and `head_dim` is a consumed leaf
/// name — so crediting this container by name would mark
/// `linear_attn_config.head_dim` consumed when the parser reads
/// `linear_key_head_dim` instead and has never looked at it. That is the
/// `vision_config.hidden_size` failure described above, and it under-reports
/// in the direction this instrument exists to prevent.
pub const PATH_READ_CONTAINER_KEYS: &[&str] =
    &["linear_attn_config", "mtp_config", "ssm_cfg", "attn_cfg"];

/// Full paths of the [`PATH_READ_LEAF_KEYS`] this config actually declares,
/// for the recorded-read credit in `build_inventory`.
///
/// Both nestings are checked because the two observed checkpoints differ:
/// GLM-5.3-Flash nests the section under `text_config`, Kimi Linear writes
/// it flat.
pub fn path_read_leaves(config: &serde_json::Value) -> Vec<String> {
    const NESTINGS: &[&str] = &["", "text_config."];
    let mut paths = Vec::new();
    for nesting in NESTINGS {
        for container in PATH_READ_CONTAINER_KEYS {
            for leaf in PATH_READ_LEAF_KEYS {
                let path = format!("{nesting}{container}.{leaf}");
                if path
                    .split('.')
                    .try_fold(config, |node, seg| node.get(seg))
                    .is_some()
                {
                    paths.push(path);
                }
            }
        }
    }
    paths
}

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
    // Weight-init scheme (Granite 4.1 30B: `"mup"`) — describes how the
    // checkpoint's weights were initialised before training, same class as
    // `initializer_range` right above; inert once training is over.
    "init_method",
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
