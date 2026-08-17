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
    AttentionSummary, Detection, Identity, LayerPolicy, MoeExecution, ResolvedExecution,
    ResolvedTopology,
};
use crate::config::ModelArchitecture;

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
                expert_bank: expert_bank_prefix(arch.as_ref(), layer),
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
        attention_sinks: arch.attention_sinks(),
        attention_bias: arch.attention_bias(),
        moe: arch.is_moe().then(|| MoeExecution {
            experts: arch.num_experts(),
            top_k: arch.num_experts_per_token(),
            expert_intermediate_size: arch.moe_intermediate_size(),
            router_kind: arch.moe_router_kind(),
            routing_policy: arch.expert_routing_policy(),
            router_bias: arch.moe_router_bias_key(0).is_some(),
            expert_format: arch.expert_format(),
            gate_up_layout: arch.gate_up_layout(),
            shared_experts: arch.num_shared_experts(),
            hybrid: arch.is_hybrid_moe(),
        }),
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

/// The architecture-relative prefix of a layer's packed expert bank: the
/// parent of the family's fused `gate_up` operand key. Asked of the arch,
/// never inferred from a substring — the family names its own operands.
/// [`bind_expert_banks`] resolves it to the source spelling once the
/// tensor names are known.
fn expert_bank_prefix(arch: &dyn ModelArchitecture, layer: usize) -> Option<String> {
    let key = arch
        .packed_gate_up_blocks_key(layer)
        .or_else(|| arch.packed_experts_gate_up_key(layer))?;
    key.rsplit_once('.').map(|(parent, _)| parent.to_string())
}

/// Resolve each layer's arch-relative expert-bank prefix to the source
/// name the checkpoint actually spells (`layers.3.mlp.experts` →
/// `model.layers.3.mlp.experts`): the tensor whose name ends with the
/// arch prefix at a segment boundary names it. A bank the arch declares
/// but no tensor spells resolves to `None` — the layer is then not
/// routed by evidence, and closure says so.
pub fn bind_expert_banks(topology: &mut ResolvedTopology, tensors: &[super::report::TensorFact]) {
    for layer in &mut topology.layers {
        let Some(relative) = layer.expert_bank.take() else {
            continue;
        };
        let dotted = format!("{relative}.");
        layer.expert_bank = tensors
            .iter()
            .filter_map(|t| {
                // `…{relative}.{leaf}` at a segment boundary: the source
                // prefix is everything before `{relative}.{leaf}` plus
                // `{relative}` itself.
                let at = t.name.find(&dotted)?;
                let boundary_ok = at == 0 || t.name.as_bytes()[at - 1] == b'.';
                boundary_ok.then(|| t.name[..at + relative.len()].to_string())
            })
            .next();
    }
}
