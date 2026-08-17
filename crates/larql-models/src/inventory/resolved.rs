//! The resolved half of the inventory: what this build's detection would
//! actually run for the checkpoint.
//!
//! Everything here goes through the public detection surface
//! ([`crate::detect_from_json`], the registry) — no re-implementation of any
//! resolution rule. The point is to report the serving path's own answers,
//! so a wrong answer (a generic fallback running full attention everywhere,
//! a defaulted rope base) appears *as the serving path would produce it*,
//! next to the declared facts it disagrees with.

use serde_json::Value;

use crate::detect::{detect_from_json, find_architecture};

use super::report::{
    AttentionSummary, Detection, Identity, LayerPolicy, ResolvedExecution, ResolvedTopology,
};

/// Attention-kind labels for [`Detection::attention_kind`].
const ATTENTION_SLIDING: &str = "sliding";
const ATTENTION_FULL: &str = "full";

/// Read the checkpoint's identity facts straight from the config value.
pub fn read_identity(config: &Value) -> Identity {
    let text_config = config.get("text_config").unwrap_or(config);
    let model_type = text_config["model_type"]
        .as_str()
        .or_else(|| config["model_type"].as_str())
        .unwrap_or("")
        .to_string();
    let architectures = config["architectures"]
        .as_array()
        .map(|arr| {
            arr.iter()
                .filter_map(|v| v.as_str().map(str::to_string))
                .collect()
        })
        .unwrap_or_default();
    let dtype = config["dtype"]
        .as_str()
        .or_else(|| config["torch_dtype"].as_str())
        .map(str::to_string);
    let transformers_version = config["transformers_version"].as_str().map(str::to_string);
    // Nested component configs: any top-level object value whose key ends in
    // `_config` (`text_config`, `vision_config`, `language_config`, …).
    let components = config
        .as_object()
        .map(|map| {
            map.iter()
                .filter(|(k, v)| v.is_object() && k.ends_with("_config"))
                .map(|(k, _)| k.clone())
                .collect()
        })
        .unwrap_or_default();
    Identity {
        model_type,
        architectures,
        dtype,
        transformers_version,
        components,
    }
}

/// Run detection and describe what came back.
pub fn resolve(config: &Value, identity: &Identity) -> (Detection, ResolvedTopology) {
    let arch = detect_from_json(config);
    let registry_entry = find_architecture(&identity.model_type);
    let validation_errors = match arch.validate() {
        Ok(()) => Vec::new(),
        Err(errors) => errors.iter().map(|e| format!("{e:?}")).collect(),
    };
    let detection = Detection {
        family: arch.family().to_string(),
        generic_fallback: registry_entry.is_none(),
        // `AttentionKind` serialises to its lowercase tag; reuse that rather
        // than inventing a second spelling here.
        attention_kind: registry_entry.and_then(|e| {
            serde_json::to_value(e.attention_kind)
                .ok()
                .and_then(|v| v.as_str().map(str::to_string))
        }),
        validation_errors,
    };

    let cfg = arch.config();
    let layers: Vec<LayerPolicy> = (0..cfg.num_layers)
        .map(|layer| {
            let sliding = arch.is_sliding_window_layer(layer);
            LayerPolicy {
                layer,
                attention: if sliding {
                    ATTENTION_SLIDING
                } else {
                    ATTENTION_FULL
                }
                .to_string(),
                window: if sliding {
                    arch.sliding_window_size()
                } else {
                    None
                },
                position: arch.position_policy_for_layer(layer),
                head_dim: arch.head_dim_for_layer(layer),
                num_kv_heads: arch.num_kv_heads_for_layer(layer),
            }
        })
        .collect();
    let sliding_layers = layers
        .iter()
        .filter(|l| l.attention == ATTENTION_SLIDING)
        .count();
    // Every semantic decision the serving path would make, resolved once
    // and recorded — the executor downstream reads, never defaults.
    //
    // Absence stays absence here. An identity default (`query_scale` 1.0,
    // `output_multiplier` 1.0) is numerically plausible but semantically
    // indistinguishable from a real declaration, so an ingestion
    // regression would produce a fully executable *wrong* program rather
    // than a loud one. Only a judgment may turn absence into an operation.
    let execution = ResolvedExecution {
        query_scale: arch.qk_scale_factor(),
        score_scale: arch.attention_scale(),
        attn_logit_softcapping: arch.attn_logit_softcapping(),
        qk_norm_scope: arch.qk_norm_scope(),
        qk_norm_weight_offset: arch.qk_norm_weight_offset(),
        parameter_free_qk_norm: arch.parameter_free_qk_norm(),
        attention_output_gate: arch.attention_output_gate(),
        activation: arch.activation(),
        ffn_type: arch.ffn_type(),
        gate_policy: arch.expert_gate_policy(),
        norm_pre: arch.pre_norm_spec(),
        norm_post: arch.post_norm_spec(),
        norm_final: arch.final_norm_spec(),
        embedding_norm: arch.embedding_norm(),
        post_norms: arch.has_post_norms(),
        embed_scale: arch.embed_scale(),
        output_multiplier: arch.output_multiplier(),
        final_logit_softcapping: arch.final_logit_softcapping(),
    };
    let topology = ResolvedTopology {
        num_layers: cfg.num_layers,
        hidden_size: cfg.hidden_size,
        intermediate_size: cfg.intermediate_size,
        num_q_heads: cfg.num_q_heads,
        num_kv_heads: cfg.num_kv_heads,
        head_dim: cfg.head_dim,
        vocab_size: cfg.vocab_size,
        sliding_window: cfg.sliding_window,
        attention: AttentionSummary {
            sliding_layers,
            full_layers: cfg.num_layers - sliding_layers,
        },
        layers,
        execution: Some(execution),
    };
    (detection, topology)
}
