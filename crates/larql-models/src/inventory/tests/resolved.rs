//! Identity + detection + per-layer resolution.

use serde_json::json;

use crate::inventory::resolved::{read_identity, resolve};

/// A Glimmer-shaped config: unknown `model_type`, explicit `layer_types`.
fn glimmer_shaped() -> serde_json::Value {
    let layer_types: Vec<&str> = (0..52)
        .map(|i| {
            if i % 4 == 3 {
                "full_attention"
            } else {
                "sliding_attention"
            }
        })
        .collect();
    json!({
        "architectures": ["MuseGlimmerForConditionalGeneration"],
        "dtype": "bfloat16",
        "model_type": "muse_glimmer",
        "transformers_version": "5.15.0.dev0",
        "text_config": {
            "model_type": "muse_glimmer_text",
            "hidden_size": 6656,
            "num_hidden_layers": 52,
            "intermediate_size": 19968,
            "num_attention_heads": 32,
            "num_key_value_heads": 2,
            "head_dim": 128,
            "sliding_window": 2048,
            "vocab_size": 202048,
            "rms_norm_eps": 1e-5,
            "qk_scale_factor": 3.87,
            "layer_types": layer_types
        },
        "vision_config": { "hidden_size": 1536 }
    })
}

#[test]
fn identity_reads_nested_model_type_and_components() {
    let identity = read_identity(&glimmer_shaped());
    assert_eq!(identity.model_type, "muse_glimmer_text");
    assert_eq!(
        identity.architectures,
        vec!["MuseGlimmerForConditionalGeneration"]
    );
    assert_eq!(identity.dtype.as_deref(), Some("bfloat16"));
    assert_eq!(
        identity.transformers_version.as_deref(),
        Some("5.15.0.dev0")
    );
    assert_eq!(identity.components, vec!["text_config", "vision_config"]);
}

#[test]
fn identity_handles_flat_configs() {
    let config = json!({
        "model_type": "muse_glimmer_assistant",
        "torch_dtype": "bfloat16"
    });
    let identity = read_identity(&config);
    assert_eq!(identity.model_type, "muse_glimmer_assistant");
    assert_eq!(identity.dtype.as_deref(), Some("bfloat16"));
    assert!(identity.components.is_empty());
    assert!(identity.architectures.is_empty());
}

/// The central finding for an unsupported family: detection succeeds via the
/// generic fallback, and the report says so. (Glimmer graduated to a
/// registered family, so the unknown here is genuinely unjudged.)
#[test]
fn unknown_model_type_reports_generic_fallback() {
    let mut config = glimmer_shaped();
    config["model_type"] = serde_json::json!("unjudged_future_model");
    config["text_config"]["model_type"] = serde_json::json!("unjudged_future_model_text");
    let identity = read_identity(&config);
    let (detection, _) = resolve(&config, &identity);
    assert!(detection.generic_fallback);
    assert_eq!(detection.family, "generic");
    assert!(detection.attention_kind.is_none());
}

/// The registered Glimmer target resolves with its judged semantics —
/// the gate spec and parameter-free QK norm — while the assistant stays
/// generic (unjudged).
#[test]
fn glimmer_target_resolves_with_judged_semantics() {
    let config = glimmer_shaped();
    let identity = read_identity(&config);
    let (detection, topology) = resolve(&config, &identity);
    assert!(!detection.generic_fallback);
    assert_eq!(detection.family, "muse_glimmer");
    let execution = topology.execution.unwrap();
    assert!(execution.attention_output_gate.is_some());
    assert!(execution.parameter_free_qk_norm.q);
    assert!(execution.parameter_free_qk_norm.k);
    // Scales stay separate: declared query factor, canonical score scale.
    // `Some` is load-bearing — an absent declaration must not arrive here
    // as a plausible 1.0.
    let query_scale = execution.query_scale.expect("declared qk_scale_factor");
    assert!((query_scale - 3.87).abs() < 1e-12);
    assert!(execution.score_scale < 1.0);
}

/// A known family does not trip the fallback flag.
#[test]
fn known_family_is_not_a_fallback() {
    let config = json!({
        "model_type": "llama",
        "hidden_size": 4096,
        "num_hidden_layers": 2,
        "intermediate_size": 11008,
        "num_attention_heads": 32,
        "num_key_value_heads": 32,
        "vocab_size": 32000
    });
    let identity = read_identity(&config);
    let (detection, _) = resolve(&config, &identity);
    assert!(!detection.generic_fallback);
    assert_eq!(detection.attention_kind.as_deref(), Some("standard"));
}

/// `layer_types` drives the per-layer table even under the generic
/// fallback — the interleave is served from the config, and the table must
/// show exactly what the serving path would run.
#[test]
fn layer_table_reflects_declared_layer_types() {
    let config = glimmer_shaped();
    let identity = read_identity(&config);
    let (_, topology) = resolve(&config, &identity);

    assert_eq!(topology.num_layers, 52);
    assert_eq!(topology.layers.len(), 52);
    assert_eq!(topology.attention.sliding_layers, 39);
    assert_eq!(topology.attention.full_layers, 13);

    // Pattern is [sliding, sliding, sliding, full] from layer 0.
    assert_eq!(topology.layers[0].attention, "sliding");
    assert_eq!(topology.layers[0].window, Some(2048));
    assert_eq!(topology.layers[3].attention, "full");
    assert_eq!(topology.layers[3].window, None);
    assert_eq!(topology.layers[51].attention, "full");

    // GQA topology flows through.
    assert_eq!(topology.num_q_heads, 32);
    assert_eq!(topology.num_kv_heads, 2);
    assert_eq!(topology.head_dim, 128);
    assert_eq!(topology.vocab_size, Some(202048));
}

/// A config with no `layer_types` and no override resolves all-full — that
/// is what the serving path would do, and the table must not pretend
/// otherwise.
#[test]
fn no_layer_types_resolves_all_full() {
    let config = json!({
        "model_type": "some_unknown_arch",
        "hidden_size": 64,
        "num_hidden_layers": 4,
        "intermediate_size": 256,
        "num_attention_heads": 8,
        "num_key_value_heads": 8
    });
    let identity = read_identity(&config);
    let (_, topology) = resolve(&config, &identity);
    assert_eq!(topology.attention.full_layers, 4);
    assert_eq!(topology.attention.sliding_layers, 0);
}

/// Validation findings are carried as data, not raised as errors.
#[test]
fn validation_errors_are_data() {
    // Zero layers is invalid, but the inventory must still describe it.
    let config = json!({ "model_type": "some_unknown_arch" });
    let identity = read_identity(&config);
    let (detection, topology) = resolve(&config, &identity);
    assert!(!detection.validation_errors.is_empty());
    assert_eq!(topology.num_layers, 0);
    assert!(topology.layers.is_empty());
}

/// A gpt-oss-shaped config: routed MoE with router bias, attention sinks and
/// projection biases, clamped GLU, YaRN — every A-9 semantic in one family.
fn gpt_oss_shaped() -> serde_json::Value {
    json!({
        "architectures": ["GptOssForCausalLM"],
        "model_type": "gpt_oss",
        "hidden_size": 2880,
        "intermediate_size": 2880,
        "num_hidden_layers": 2,
        "num_attention_heads": 64,
        "num_key_value_heads": 8,
        "head_dim": 64,
        "attention_bias": true,
        "num_local_experts": 32,
        "experts_per_token": 4,
        "vocab_size": 201088,
        "rope_theta": 150000.0,
        "swiglu_limit": 7.0,
        "layer_types": ["sliding_attention", "full_attention"],
        "sliding_window": 128
    })
}

/// A routed family resolves its MoE facts and names each layer's expert
/// bank in the architecture's own namespace — before any tensor is seen.
#[test]
fn routed_family_resolves_moe_and_names_its_banks_arch_relative() {
    let config = gpt_oss_shaped();
    let identity = read_identity(&config);
    let (detection, topology) = resolve(&config, &identity);
    assert!(!detection.generic_fallback);
    let execution = topology.execution.expect("judged execution");
    let moe = execution.moe.expect("gpt-oss is routed");
    assert_eq!(moe.experts, 32);
    assert_eq!(moe.top_k, 4);
    assert!(moe.router_bias);
    assert!(execution.attention_sinks.is_some());
    assert_eq!(execution.attention_bias, Some(true));
    assert!(matches!(
        execution.gate_policy,
        crate::config::ExpertGatePolicy::ClampedGlu { .. }
    ));
    let banks: Vec<Option<String>> = topology
        .layers
        .iter()
        .map(|l| l.expert_bank.clone())
        .collect();
    assert_eq!(
        banks,
        vec![
            Some("layers.0.mlp.experts".to_string()),
            Some("layers.1.mlp.experts".to_string())
        ]
    );
}

/// A dense family names no bank on any layer.
#[test]
fn dense_family_names_no_expert_bank() {
    let config = glimmer_shaped();
    let identity = read_identity(&config);
    let (_, topology) = resolve(&config, &identity);
    assert!(topology.layers.iter().all(|l| l.expert_bank.is_none()));
    assert!(topology.execution.unwrap().moe.is_none());
}

fn tensor(name: &str) -> crate::inventory::TensorFact {
    crate::inventory::TensorFact {
        name: name.to_string(),
        dtype: "U8".to_string(),
        shape: vec![32, 5760, 90, 16],
        bytes: 0,
        file: "model.safetensors".to_string(),
    }
}

/// Binding resolves the arch-relative prefix to the spelling the checkpoint
/// uses, at a segment boundary; a bank no tensor spells resolves to `None`.
#[test]
fn expert_banks_bind_to_the_source_spelling_or_to_nothing() {
    use crate::inventory::resolved::bind_expert_banks;
    let config = gpt_oss_shaped();
    let identity = read_identity(&config);
    let (_, mut topology) = resolve(&config, &identity);
    let tensors = vec![
        // Layer 0 is spelled by the checkpoint under `model.`.
        tensor("model.layers.0.mlp.experts.gate_up_proj_blocks"),
        // A near-miss for layer 1: `xlayers.1…` is not a segment boundary.
        tensor("model.xlayers.1.mlp.experts.gate_up_proj_blocks"),
    ];
    bind_expert_banks(&mut topology, &tensors);
    assert_eq!(
        topology.layers[0].expert_bank.as_deref(),
        Some("model.layers.0.mlp.experts")
    );
    assert_eq!(topology.layers[1].expert_bank, None);
}

/// A bank spelled with no source prefix at all binds at offset zero.
#[test]
fn expert_bank_at_the_start_of_the_name_binds_too() {
    use crate::inventory::resolved::bind_expert_banks;
    let config = gpt_oss_shaped();
    let identity = read_identity(&config);
    let (_, mut topology) = resolve(&config, &identity);
    let tensors = vec![tensor("layers.1.mlp.experts.down_proj_blocks")];
    bind_expert_banks(&mut topology, &tensors);
    assert_eq!(topology.layers[0].expert_bank, None);
    assert_eq!(
        topology.layers[1].expert_bank.as_deref(),
        Some("layers.1.mlp.experts")
    );
}
