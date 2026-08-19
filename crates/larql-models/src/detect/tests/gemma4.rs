//! Gemma 4 detection: key formats, per-layer geometry, hybrid MoE.

use crate::detect::*;

#[test]
fn test_gemma4_key_formats() {
    let config = serde_json::json!({
        "model_type": "gemma4",
        "text_config": {
            "model_type": "gemma4_text",
            "hidden_size": 1536,
            "intermediate_size": 6144,
            "num_hidden_layers": 8,
            "num_attention_heads": 8,
            "num_key_value_heads": 1,
            "head_dim": 256,
        }
    });
    let arch = detect_from_json(&config);

    // Gemma 4 uses HF-style llama keys (no architecture-specific override in gemma4.rs)
    assert_eq!(arch.attn_q_key(3), "layers.3.self_attn.q_proj.weight");
    assert_eq!(arch.attn_k_key(3), "layers.3.self_attn.k_proj.weight");
    assert_eq!(arch.attn_v_key(3), "layers.3.self_attn.v_proj.weight");
    assert_eq!(arch.attn_o_key(3), "layers.3.self_attn.o_proj.weight");
    assert_eq!(arch.ffn_gate_key(3), "layers.3.mlp.gate_proj.weight");
    assert_eq!(arch.ffn_up_key(3), "layers.3.mlp.up_proj.weight");
    assert_eq!(arch.ffn_down_key(3), "layers.3.mlp.down_proj.weight");

    // Multimodal wrapper prefixes (stripped on load)
    let prefixes = arch.key_prefixes_to_strip();
    assert!(prefixes.contains(&"model.language_model.model."));
    assert!(prefixes.contains(&"model.language_model."));
    assert!(prefixes.contains(&"language_model.model."));
    assert!(prefixes.contains(&"model."));

    // QK norm keys (inherited from Gemma 3)
    assert_eq!(
        arch.attn_q_norm_key(3),
        Some("layers.3.self_attn.q_norm.weight".to_string())
    );
    assert_eq!(
        arch.attn_k_norm_key(3),
        Some("layers.3.self_attn.k_norm.weight".to_string())
    );

    // Gemma 4's shipped tokenizer.json drops BOS from its post-processor
    // `single` template (Gemma 2/3 kept it), so the arch must advertise
    // the BOS id so the inference tokenizer helper can prepend it.
    assert_eq!(arch.bos_token_id(), Some(2));
}

#[test]
fn test_bos_token_id_gemma4_only() {
    // Only Gemma 4 advertises an explicit BOS id — every other
    // architecture's tokenizer.json already includes BOS in its
    // post-processor so callers don't need to prepend it.
    let non_gemma4 = [
        serde_json::json!({"model_type": "llama", "hidden_size": 4096,
            "num_hidden_layers": 32, "intermediate_size": 14336,
            "num_attention_heads": 32, "num_key_value_heads": 8}),
        serde_json::json!({"model_type": "gemma3", "hidden_size": 2560,
            "num_hidden_layers": 34}),
        serde_json::json!({"model_type": "gemma2", "hidden_size": 2304,
            "num_hidden_layers": 26}),
        serde_json::json!({"model_type": "mistral", "hidden_size": 4096,
            "num_hidden_layers": 32}),
        serde_json::json!({"model_type": "qwen2", "hidden_size": 2048,
            "num_hidden_layers": 24, "intermediate_size": 5504,
            "num_attention_heads": 16, "num_key_value_heads": 2}),
        serde_json::json!({"model_type": "tinymodel", "hidden_size": 512,
            "num_hidden_layers": 20, "intermediate_size": 2048,
            "num_attention_heads": 8, "num_key_value_heads": 4}),
    ];
    for cfg in &non_gemma4 {
        let arch = detect_from_json(cfg);
        assert!(
            arch.bos_token_id().is_none(),
            "{} should not advertise a BOS id",
            arch.family()
        );
    }
}

#[test]
fn test_detect_gemma4_31b() {
    // Real Gemma 4 31B config — matches actual HuggingFace config.json
    let config = serde_json::json!({
        "model_type": "gemma4",
        "text_config": {
            "model_type": "gemma4_text",
            "hidden_size": 5376,
            "intermediate_size": 21504,
            "num_hidden_layers": 60,
            "num_attention_heads": 32,
            "num_key_value_heads": 16,
            "head_dim": 256,
            "global_head_dim": 512,
            "num_global_key_value_heads": 4,
            "vocab_size": 262144,
            "attention_k_eq_v": true,
            "sliding_window": 1024,
            "final_logit_softcapping": 30.0,
            "rope_parameters": {
                "full_attention": {
                    "partial_rotary_factor": 0.25,
                    "rope_theta": 1000000.0,
                    "rope_type": "proportional"
                },
                "sliding_attention": {
                    "rope_theta": 10000.0,
                    "rope_type": "default"
                }
            },
            "layer_types": [
                "sliding_attention", "sliding_attention", "sliding_attention",
                "sliding_attention", "sliding_attention", "full_attention",
                "sliding_attention", "sliding_attention", "sliding_attention",
                "sliding_attention", "sliding_attention", "full_attention",
                "sliding_attention", "sliding_attention", "sliding_attention",
                "sliding_attention", "sliding_attention", "full_attention",
                "sliding_attention", "sliding_attention", "sliding_attention",
                "sliding_attention", "sliding_attention", "full_attention",
                "sliding_attention", "sliding_attention", "sliding_attention",
                "sliding_attention", "sliding_attention", "full_attention",
                "sliding_attention", "sliding_attention", "sliding_attention",
                "sliding_attention", "sliding_attention", "full_attention",
                "sliding_attention", "sliding_attention", "sliding_attention",
                "sliding_attention", "sliding_attention", "full_attention",
                "sliding_attention", "sliding_attention", "sliding_attention",
                "sliding_attention", "sliding_attention", "full_attention",
                "sliding_attention", "sliding_attention", "sliding_attention",
                "sliding_attention", "sliding_attention", "full_attention",
                "sliding_attention", "sliding_attention", "sliding_attention",
                "sliding_attention", "sliding_attention", "full_attention"
            ]
        }
    });

    let arch = detect_from_json(&config);
    assert_eq!(arch.family(), "gemma4");
    assert_eq!(arch.config().num_layers, 60);
    assert_eq!(arch.config().hidden_size, 5376);
    assert_eq!(arch.config().head_dim, 256);
    assert_eq!(arch.config().global_head_dim, Some(512));
    assert_eq!(arch.config().num_global_kv_heads, Some(4));

    // Sliding layer (layer 0): uses base head_dim and kv_heads
    assert!(arch.is_sliding_window_layer(0));
    assert_eq!(arch.head_dim_for_layer(0), 256);
    assert_eq!(arch.num_kv_heads_for_layer(0), 16);
    assert_eq!(arch.num_q_heads_for_layer(0), 32);
    assert_eq!(arch.rotary_fraction_for_layer(0), 1.0);

    // Global layer (layer 5): uses global_head_dim and global kv_heads
    assert!(!arch.is_sliding_window_layer(5));
    assert_eq!(arch.head_dim_for_layer(5), 512);
    assert_eq!(arch.num_kv_heads_for_layer(5), 4);
    // Q heads constant across all layers
    assert_eq!(arch.num_q_heads_for_layer(5), 32);
    assert_eq!(arch.rotary_fraction_for_layer(5), 0.25);

    // RoPE bases
    assert_eq!(arch.rope_base_for_layer(0), 10_000.0); // sliding
    assert_eq!(arch.rope_base_for_layer(5), 1_000_000.0); // global

    // Gemma 4 stores norm weights as full multiplier (no +1 offset, unlike Gemma 2/3)
    assert_eq!(arch.norm_weight_offset(), 0.0);
    assert_eq!(arch.embed_scale(), Some((5376.0f32).sqrt()));
    assert!(arch.has_post_norms());
    assert!(arch.attn_q_norm_key(0).is_some());
    assert_eq!(arch.final_logit_softcapping(), Some(30.0));

    // Layer scalar key
    assert_eq!(
        arch.layer_scalar_key(5),
        Some("layers.5.layer_scalar".to_string())
    );

    // Gemma 4 uses QK-norm, so attention scale is 1.0 (no 1/sqrt(head_dim))
    assert_eq!(arch.attention_scale_for_layer(0), 1.0);
    assert_eq!(arch.attention_scale_for_layer(5), 1.0);

    // K=V flag parsed — v_shares_k() exposes it via the trait.
    // On 31B, attention_k_eq_v=true applies only to global (full_attention) layers;
    // sliding layers still ship v_proj in safetensors.
    assert!(arch.config().attention_k_eq_v);
    assert!(!arch.v_shares_k(0)); // sliding
    assert!(arch.v_shares_k(5)); // global

    // V-norm (parameter-free RMSNorm on V states)
    assert!(arch.has_v_norm());

    // 31B has no KV sharing (num_kv_shared_layers absent)
    assert!(arch.kv_shared_source_layer(0).is_none());
    assert!(arch.kv_shared_source_layer(30).is_none());

    // 31B has no PLE
    assert!(!arch.has_per_layer_embeddings());
}

#[test]
fn test_detect_gemma4_e2b() {
    // Real E2B config with PLE, KV sharing, global_head_dim, layer_types
    let config = serde_json::json!({
        "model_type": "gemma4",
        "text_config": {
            "model_type": "gemma4_text",
            "hidden_size": 1536,
            "intermediate_size": 6144,
            "num_hidden_layers": 35,
            "num_attention_heads": 8,
            "num_key_value_heads": 1,
            "head_dim": 256,
            "global_head_dim": 512,
            "vocab_size": 262144,
            "sliding_window": 512,
            "final_logit_softcapping": 30.0,
            "hidden_size_per_layer_input": 256,
            "num_kv_shared_layers": 20,
            "attention_k_eq_v": false,
            "use_double_wide_mlp": true,
            "rope_parameters": {
                "full_attention": {
                    "partial_rotary_factor": 0.25,
                    "rope_theta": 1000000.0,
                    "rope_type": "proportional"
                },
                "sliding_attention": {
                    "rope_theta": 10000.0,
                    "rope_type": "default"
                }
            },
            "layer_types": [
                "sliding_attention", "sliding_attention", "sliding_attention",
                "sliding_attention", "full_attention",
                "sliding_attention", "sliding_attention", "sliding_attention",
                "sliding_attention", "full_attention",
                "sliding_attention", "sliding_attention", "sliding_attention",
                "sliding_attention", "full_attention",
                "sliding_attention", "sliding_attention", "sliding_attention",
                "sliding_attention", "full_attention",
                "sliding_attention", "sliding_attention", "sliding_attention",
                "sliding_attention", "full_attention",
                "sliding_attention", "sliding_attention", "sliding_attention",
                "sliding_attention", "full_attention",
                "sliding_attention", "sliding_attention", "sliding_attention",
                "sliding_attention", "full_attention"
            ]
        }
    });

    let arch = detect_from_json(&config);
    assert_eq!(arch.family(), "gemma4");
    assert_eq!(arch.config().num_layers, 35);

    // Layer types from explicit array
    assert!(arch.is_sliding_window_layer(0));
    assert!(arch.is_sliding_window_layer(3));
    assert!(!arch.is_sliding_window_layer(4)); // global
    assert!(arch.is_sliding_window_layer(5));
    assert!(!arch.is_sliding_window_layer(9)); // global

    // Per-layer head_dim: sliding=256, global=512
    assert_eq!(arch.head_dim_for_layer(0), 256);
    assert_eq!(arch.head_dim_for_layer(4), 512);
    assert_eq!(arch.num_q_heads_for_layer(0), 8);
    assert_eq!(arch.num_q_heads_for_layer(4), 8); // constant across layers

    // Partial rotary on global layers
    assert_eq!(arch.rotary_fraction_for_layer(0), 1.0);
    assert_eq!(arch.rotary_fraction_for_layer(4), 0.25);

    // RoPE bases from rope_parameters
    assert_eq!(arch.rope_base_for_layer(0), 10_000.0);
    assert_eq!(arch.rope_base_for_layer(4), 1_000_000.0);

    // PLE (Per-Layer Embeddings)
    assert!(arch.has_per_layer_embeddings());
    assert_eq!(arch.per_layer_embed_dim(), 256);

    // KV sharing: layers 15-34 share from source layers
    // First 15 layers are non-shared
    assert!(arch.kv_shared_source_layer(0).is_none());
    assert!(arch.kv_shared_source_layer(14).is_none());
    // Layers 15+ are shared: sliding→L13, global→L14
    assert_eq!(arch.kv_shared_source_layer(15), Some(13)); // sliding shared
    assert_eq!(arch.kv_shared_source_layer(19), Some(14)); // global shared
    assert_eq!(arch.kv_shared_source_layer(34), Some(14)); // last layer (global)

    // V-norm, attention scale
    assert!(arch.has_v_norm());
    assert_eq!(arch.attention_scale(), 1.0);
    assert_eq!(arch.norm_weight_offset(), 0.0);

    // No K=V on E2B
    assert!(!arch.config().attention_k_eq_v);
    assert!(!arch.v_shares_k(0));
}

#[test]
fn test_detect_gemma4_real_config() {
    // Test against the actual HuggingFace config.json if available
    let config_path = std::env::var("HOME").ok().map(|h| {
        std::path::PathBuf::from(h).join(".cache/huggingface/hub/models--google--gemma-4-31B-it")
    });
    let config_path = match config_path {
        Some(p) if p.exists() => {
            // Find the snapshot
            let snapshots = p.join("snapshots");
            std::fs::read_dir(&snapshots)
                .ok()
                .and_then(|mut entries| entries.next())
                .and_then(|e| e.ok())
                .map(|e| e.path().join("config.json"))
        }
        _ => None,
    };
    let config_path = match config_path {
        Some(p) if p.exists() => p,
        _ => return, // skip if model not cached
    };

    let text = std::fs::read_to_string(&config_path).unwrap();
    let config: serde_json::Value = serde_json::from_str(&text).unwrap();
    let arch = detect_from_json(&config);

    assert_eq!(arch.family(), "gemma4");
    assert_eq!(arch.config().num_layers, 60);
    assert_eq!(arch.config().hidden_size, 5376);
    assert_eq!(arch.config().head_dim, 256);
    assert_eq!(arch.config().global_head_dim, Some(512));
    assert_eq!(arch.config().num_kv_heads, 16);
    assert_eq!(arch.config().num_global_kv_heads, Some(4));
    assert_eq!(arch.config().partial_rotary_factor, Some(0.25));
    assert!(arch.config().attention_k_eq_v);

    // Verify layer_types parsed correctly (60 layers: 50 sliding + 10 full)
    assert!(arch.config().layer_types.is_some());
    let types = arch.config().layer_types.as_ref().unwrap();
    assert_eq!(types.len(), 60);
    let full_count = types.iter().filter(|t| *t == "full_attention").count();
    assert_eq!(full_count, 10);

    // Layer 5 is full_attention in the real config
    assert!(!arch.is_sliding_window_layer(5));
    assert_eq!(arch.head_dim_for_layer(5), 512);
    assert_eq!(arch.num_kv_heads_for_layer(5), 4);
    assert_eq!(arch.rotary_fraction_for_layer(5), 0.25);

    // RoPE bases from rope_parameters
    assert_eq!(arch.rope_base_for_layer(0), 10_000.0);
    assert_eq!(arch.rope_base_for_layer(5), 1_000_000.0);
}

#[test]
fn test_detect_gemma4_26b_a4b() {
    // Gemma 4 26B A4B — hybrid dense-MLP + MoE per layer.
    // Architecture: 30 layers, hidden=2816, dense_intermediate=9216,
    // 128 experts each with moe_intermediate=704, top_k=8.
    let config = serde_json::json!({
        "model_type": "gemma4",
        "text_config": {
            "model_type": "gemma4_text",
            "hidden_size": 2816,
            "intermediate_size": 9216,
            "num_hidden_layers": 30,
            "num_attention_heads": 16,
            "num_key_value_heads": 8,
            "head_dim": 256,
            "global_head_dim": 512,
            "num_global_key_value_heads": 4,
            "vocab_size": 262144,
            "enable_moe_block": true,
            "num_experts": 128,
            "top_k_experts": 8,
            "moe_intermediate_size": 704,
            "final_logit_softcapping": 30.0,
            "rope_parameters": {
                "full_attention": {
                    "partial_rotary_factor": 0.25,
                    "rope_theta": 1000000.0
                },
                "sliding_attention": {
                    "rope_theta": 10000.0
                }
            }
        }
    });

    let arch = detect_from_json(&config);
    assert_eq!(arch.family(), "gemma4");
    assert_eq!(arch.config().num_layers, 30);
    assert_eq!(arch.config().hidden_size, 2816);
    assert_eq!(arch.config().intermediate_size, 9216);

    // MoE
    assert!(arch.is_moe());
    assert!(arch.is_hybrid_moe());
    assert_eq!(arch.num_experts(), 128);
    assert_eq!(arch.num_experts_per_token(), 8);
    assert_eq!(arch.moe_intermediate_size(), 704);

    // Router keys
    assert_eq!(
        arch.moe_router_key(0),
        Some("layers.0.router.proj.weight".to_string())
    );
    assert_eq!(
        arch.moe_router_scale_key(3),
        Some("layers.3.router.scale".to_string())
    );
    assert_eq!(
        arch.moe_router_per_expert_scale_key(3),
        Some("layers.3.router.per_expert_scale".to_string())
    );

    // Packed expert keys
    assert_eq!(
        arch.packed_experts_gate_up_key(5),
        Some("layers.5.experts.gate_up_proj".to_string())
    );
    assert_eq!(
        arch.packed_experts_down_key(5),
        Some("layers.5.experts.down_proj".to_string())
    );

    // Hybrid MoE norm keys — dense branch gets _1 suffix
    assert_eq!(
        arch.post_feedforward_layernorm_key(0),
        Some("layers.0.post_feedforward_layernorm_1.weight".to_string())
    );
    assert_eq!(
        arch.moe_pre_experts_norm_key(0),
        Some("layers.0.pre_feedforward_layernorm_2.weight".to_string())
    );
    assert_eq!(
        arch.moe_post_experts_norm_key(0),
        Some("layers.0.post_feedforward_layernorm_2.weight".to_string())
    );

    // Dense FFN keys still present (both branches coexist)
    assert_eq!(arch.ffn_gate_key(0), "layers.0.mlp.gate_proj.weight");
    assert_eq!(arch.ffn_up_key(0), "layers.0.mlp.up_proj.weight");
    assert_eq!(arch.ffn_down_key(0), "layers.0.mlp.down_proj.weight");

    // ExpertFormat
    use crate::config::ExpertFormat;
    assert_eq!(arch.expert_format(), ExpertFormat::PackedBF16);

    // Gemma 4 features still work
    assert_eq!(arch.norm_weight_offset(), 0.0);
    assert!(arch.has_v_norm());
    assert!(arch.has_post_norms());
    assert_eq!(arch.bos_token_id(), Some(2));
}

#[test]
fn test_detect_gemma4_dense_returns_none_for_moe_getters() {
    // Non-MoE Gemma 4 must return None / non-MoE-specific values from
    // every MoE-only getter — covers the `else` arms in
    // architectures/gemma4.rs (lines 270-393 None branches).
    let config = serde_json::json!({
        "model_type": "gemma4",
        "text_config": {
            "model_type": "gemma4_text",
            "hidden_size": 2560,
            "intermediate_size": 10240,
            "num_hidden_layers": 30,
            "num_attention_heads": 8,
            "num_key_value_heads": 4,
            "head_dim": 256,
        }
    });
    let arch = detect_from_json(&config);
    assert_eq!(arch.family(), "gemma4");
    assert!(!arch.is_hybrid_moe());
    assert_eq!(arch.moe_router_type(), "top_k_softmax");
    assert!(arch.moe_router_key(0).is_none());
    assert!(arch.moe_router_scale_key(0).is_none());
    assert!(arch.moe_router_per_expert_scale_key(0).is_none());
    assert!(!arch.moe_router_norm_parameter_free());
    assert!(arch.moe_router_input_scalar().is_none());
    assert!(arch.packed_experts_gate_up_key(0).is_none());
    assert!(arch.packed_experts_down_key(0).is_none());
    assert!(arch.moe_pre_experts_norm_key(0).is_none());
    assert!(arch.moe_post_experts_norm_key(0).is_none());
    assert!(arch.moe_post_outer_norm_key(0).is_none());
    assert!(!arch.moe_has_combined_output_norm());
    // Dense Gemma 4 uses the un-suffixed post_feedforward_layernorm key.
    assert_eq!(
        arch.post_feedforward_layernorm_key(0),
        Some("layers.0.post_feedforward_layernorm.weight".to_string())
    );
    // `moe_post_ffn1_norm_key` aliases `post_feedforward_layernorm_key`.
    assert_eq!(
        arch.moe_post_ffn1_norm_key(0),
        arch.post_feedforward_layernorm_key(0)
    );
}

#[test]
fn test_detect_gemma4_moe_uses_gemma4_top_k_softmax_router_type() {
    // The MoE-only `moe_router_type` returns "gemma4_top_k_softmax" when
    // `enable_moe_block` is true — covers the if-branch in gemma4.rs L265.
    let config = serde_json::json!({
        "model_type": "gemma4",
        "text_config": {
            "model_type": "gemma4_text",
            "hidden_size": 2816,
            "intermediate_size": 9216,
            "num_hidden_layers": 30,
            "num_attention_heads": 16,
            "num_key_value_heads": 8,
            "head_dim": 256,
            "enable_moe_block": true,
            "num_experts": 128,
            "top_k_experts": 8,
            "moe_intermediate_size": 704,
        }
    });
    let arch = detect_from_json(&config);
    assert_eq!(arch.moe_router_type(), "gemma4_top_k_softmax");
    assert!(arch.moe_router_norm_parameter_free());
    // input_scalar = hidden_size^-0.5
    let scalar = arch.moe_router_input_scalar().unwrap();
    assert!((scalar - (2816.0f32).powf(-0.5)).abs() < 1e-6);
    // moe_post_outer_norm_key for hybrid MoE points at the un-suffixed key.
    assert_eq!(
        arch.moe_post_outer_norm_key(0),
        Some("layers.0.post_feedforward_layernorm.weight".to_string())
    );
}

/// The PLE-family knobs are read verbatim into the config: a checkpoint
/// declaring the double-wide MLP on or a per-layer-input vocabulary keeps
/// those values, and one declaring neither reads `None` — not a default.
#[test]
fn ple_family_knobs_are_read_verbatim() {
    let mut config = serde_json::json!({
        "model_type": "gemma4",
        "text_config": {
            "hidden_size": 64,
            "num_hidden_layers": 2,
            "intermediate_size": 128,
            "num_attention_heads": 8,
            "num_key_value_heads": 2,
            "head_dim": 8,
            "vocab_size": 128,
            "use_double_wide_mlp": true,
            "vocab_size_per_layer_input": 262144
        }
    });
    let parsed = crate::detect::detect_from_json(&config).config().clone();
    assert_eq!(parsed.use_double_wide_mlp, Some(true));
    assert_eq!(parsed.vocab_size_per_layer_input, Some(262144));
    config["text_config"]
        .as_object_mut()
        .unwrap()
        .remove("use_double_wide_mlp");
    config["text_config"]
        .as_object_mut()
        .unwrap()
        .remove("vocab_size_per_layer_input");
    let parsed = crate::detect::detect_from_json(&config).config().clone();
    assert_eq!(parsed.use_double_wide_mlp, None);
    assert_eq!(parsed.vocab_size_per_layer_input, None);
}
