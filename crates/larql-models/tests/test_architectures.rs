//! Integration tests for model architecture detection and key patterns.

use larql_models::{
    detect_from_json, detect_from_json_validated,
    validation::{
        FIELD_HEAD_DIM, FIELD_HIDDEN_SIZE, FIELD_INTERMEDIATE_SIZE, FIELD_LAYER_TYPES,
        FIELD_MOE_INTERMEDIATE_SIZE, FIELD_NUM_EXPERTS_PER_TOKEN, FIELD_NUM_KV_HEADS,
        FIELD_NUM_KV_SHARED_LAYERS, FIELD_NUM_LAYERS, FIELD_NUM_Q_HEADS,
        FIELD_PARTIAL_ROTARY_FACTOR, FIELD_ROPE_BASE, FIELD_ROPE_SCALING_FACTOR,
        FIELD_ROPE_SCALING_TYPE,
    },
    ExpertFormat, ModelArchitecture,
};

// ═══════════════════════════════════════════════════════════════
// GPT-OSS architecture
// ═══════════════════════════════════════════════════════════════

fn gpt_oss_arch() -> Box<dyn ModelArchitecture> {
    detect_from_json(&serde_json::json!({
        "model_type": "gpt_oss",
        "hidden_size": 2880,
        "num_hidden_layers": 36,
        "intermediate_size": 2880,
        "num_attention_heads": 64,
        "num_key_value_heads": 8,
        "num_local_experts": 128,
        "num_experts_per_tok": 4,
        "head_dim": 64,
        "rope_theta": 150000.0,
    }))
}

#[test]
fn gpt_oss_detection() {
    let arch = gpt_oss_arch();
    assert_eq!(arch.family(), "gpt_oss");
    assert_eq!(arch.config().num_layers, 36);
    assert_eq!(arch.config().hidden_size, 2880);
}

#[test]
fn gpt_oss_is_moe() {
    let arch = gpt_oss_arch();
    assert!(arch.is_moe());
    assert_eq!(arch.num_experts(), 128);
    assert_eq!(arch.num_experts_per_token(), 4);
}

#[test]
fn gpt_oss_expert_format() {
    let arch = gpt_oss_arch();
    assert_eq!(arch.expert_format(), ExpertFormat::PackedMxfp4);
}

#[test]
fn gpt_oss_packed_keys() {
    let arch = gpt_oss_arch();
    assert_eq!(
        arch.packed_gate_up_blocks_key(5).unwrap(),
        "layers.5.mlp.experts.gate_up_proj_blocks"
    );
    assert_eq!(
        arch.packed_gate_up_scales_key(5).unwrap(),
        "layers.5.mlp.experts.gate_up_proj_scales"
    );
    assert_eq!(
        arch.packed_down_blocks_key(5).unwrap(),
        "layers.5.mlp.experts.down_proj_blocks"
    );
    assert_eq!(
        arch.packed_down_scales_key(5).unwrap(),
        "layers.5.mlp.experts.down_proj_scales"
    );
}

#[test]
fn gpt_oss_router_key() {
    let arch = gpt_oss_arch();
    assert_eq!(
        arch.moe_router_key(0).unwrap(),
        "layers.0.mlp.router.weight"
    );
}

#[test]
fn gpt_oss_attn_keys() {
    let arch = gpt_oss_arch();
    assert_eq!(arch.attn_q_key(3), "layers.3.self_attn.q_proj.weight");
    assert_eq!(arch.attn_k_key(3), "layers.3.self_attn.k_proj.weight");
    assert_eq!(arch.attn_v_key(3), "layers.3.self_attn.v_proj.weight");
    assert_eq!(arch.attn_o_key(3), "layers.3.self_attn.o_proj.weight");
}

/// GPT-OSS's per-expert keys describe **loaded** state, not the checkpoint.
///
/// On disk there are no per-expert tensors — everything is packed and fused,
/// which is why this test previously asserted `is_none()`. But the safetensors
/// loader dequantises and de-interleaves the experts at load time and stores
/// each one separately, so a compute backend reading `ModelWeights` does see
/// per-expert weights. Advertising them is what lets a generic per-expert FFN
/// backend serve this model without knowing anything about MXFP4 — and until
/// 2026-07-30 nothing could, which is why `shannon score` could not score it.
/// Callers that read the *checkpoint* (extraction) still want `packed_*`.
/// See `docs/k3-funnel.md` §4.7.
#[test]
fn gpt_oss_exposes_dequantised_per_expert_keys() {
    let arch = gpt_oss_arch();
    assert_eq!(
        arch.expert_ffn_gate_key(0, 5).as_deref(),
        Some("layers.0.block_sparse_moe.experts.5.w1.weight")
    );
    assert_eq!(
        arch.expert_ffn_up_key(0, 5).as_deref(),
        Some("layers.0.block_sparse_moe.experts.5.w3.weight")
    );
    assert_eq!(
        arch.expert_ffn_down_key(0, 5).as_deref(),
        Some("layers.0.block_sparse_moe.experts.5.w2.weight")
    );
}

/// The packed keys remain the checkpoint's own layout, unchanged by the above.
#[test]
fn gpt_oss_still_reports_packed_checkpoint_keys() {
    let arch = gpt_oss_arch();
    assert_eq!(
        arch.packed_gate_up_blocks_key(0).as_deref(),
        Some("layers.0.mlp.experts.gate_up_proj_blocks")
    );
    assert_eq!(
        arch.expert_format(),
        larql_models::ExpertFormat::PackedMxfp4
    );
}

#[test]
fn gpt_oss_prefix_strip() {
    let arch = gpt_oss_arch();
    assert_eq!(arch.key_prefixes_to_strip(), &["model."]);
}

// ═══════════════════════════════════════════════════════════════
// Mixtral — PerExpert format comparison
// ═══════════════════════════════════════════════════════════════

fn mixtral_arch() -> Box<dyn ModelArchitecture> {
    detect_from_json(&serde_json::json!({
        "model_type": "mixtral",
        "hidden_size": 4096,
        "num_hidden_layers": 32,
        "intermediate_size": 14336,
        "num_attention_heads": 32,
        "num_key_value_heads": 8,
        "num_local_experts": 8,
        "num_experts_per_tok": 2,
    }))
}

#[test]
fn mixtral_expert_format() {
    let arch = mixtral_arch();
    assert_eq!(arch.expert_format(), ExpertFormat::PerExpert);
}

#[test]
fn mixtral_per_expert_keys() {
    let arch = mixtral_arch();
    assert_eq!(
        arch.expert_ffn_gate_key(0, 3).unwrap(),
        "layers.0.block_sparse_moe.experts.3.w1.weight"
    );
    assert_eq!(
        arch.expert_ffn_down_key(0, 3).unwrap(),
        "layers.0.block_sparse_moe.experts.3.w2.weight"
    );
}

#[test]
fn mixtral_no_packed_keys() {
    let arch = mixtral_arch();
    assert!(arch.packed_gate_up_blocks_key(0).is_none());
}

#[test]
fn mixtral_config_accessor_returns_loaded_config() {
    let arch = mixtral_arch();
    let config = arch.config();
    assert_eq!(config.hidden_size, 4096);
    assert_eq!(config.num_layers, 32);
    assert_eq!(config.intermediate_size, 14336);
}

// ═══════════════════════════════════════════════════════════════
// Dense model — no MoE
// ═══════════════════════════════════════════════════════════════

#[test]
fn llama_not_moe() {
    let arch = detect_from_json(&serde_json::json!({
        "model_type": "llama",
        "hidden_size": 4096,
        "num_hidden_layers": 32,
        "intermediate_size": 14336,
        "num_attention_heads": 32,
        "num_key_value_heads": 8,
    }));
    assert!(!arch.is_moe());
    assert_eq!(arch.expert_format(), ExpertFormat::PerExpert); // default
    assert_eq!(arch.num_experts(), 0);
}

#[test]
fn generic_architecture_exercises_default_trait_contract() {
    let arch = detect_from_json(&serde_json::json!({
        "model_type": "unknown_model",
        "hidden_size": 16,
        "num_hidden_layers": 2,
        "intermediate_size": 32,
        "num_attention_heads": 4,
        "num_key_value_heads": 2,
        "head_dim": 4,
        "sliding_window": 128,
        "rope_theta": 20000.0,
        "rope_scaling": {"type": "linear", "factor": 2.0}
    }));

    assert_eq!(arch.family(), "generic");
    assert_eq!(arch.layer_prefix(7), "layers.7.");
    assert_eq!(
        arch.key_prefixes_to_strip(),
        &["language_model.model.", "model."]
    );
    assert_eq!(arch.embed_key(), "embed_tokens.weight");
    assert_eq!(arch.final_norm_key(), "norm.weight");
    assert_eq!(arch.attn_q_key(1), "layers.1.self_attn.q_proj.weight");
    assert_eq!(arch.attn_k_key(1), "layers.1.self_attn.k_proj.weight");
    assert_eq!(arch.attn_v_key(1), "layers.1.self_attn.v_proj.weight");
    assert_eq!(arch.attn_o_key(1), "layers.1.self_attn.o_proj.weight");
    assert_eq!(arch.ffn_gate_key(1), "layers.1.mlp.gate_proj.weight");
    assert_eq!(arch.ffn_up_key(1), "layers.1.mlp.up_proj.weight");
    assert_eq!(arch.ffn_down_key(1), "layers.1.mlp.down_proj.weight");
    assert_eq!(
        arch.input_layernorm_key(1),
        "layers.1.input_layernorm.weight"
    );
    assert_eq!(
        arch.post_attention_layernorm_key(1),
        "layers.1.post_attention_layernorm.weight"
    );
    assert_eq!(
        arch.pre_feedforward_layernorm_key(1),
        Some("layers.1.pre_feedforward_layernorm.weight".to_string())
    );
    assert_eq!(
        arch.post_feedforward_layernorm_key(1),
        Some("layers.1.post_feedforward_layernorm.weight".to_string())
    );

    assert_eq!(arch.attn_o_bias_key(1), None);
    assert_eq!(arch.attn_q_bias_key(1), None);
    assert_eq!(arch.attn_k_bias_key(1), None);
    assert_eq!(arch.attn_v_bias_key(1), None);
    assert_eq!(arch.attn_q_norm_key(1), None);
    assert_eq!(arch.attn_k_norm_key(1), None);
    assert_eq!(arch.ffn_up_bias_key(1), None);
    assert_eq!(arch.ffn_down_bias_key(1), None);

    assert_eq!(arch.norm_type(), larql_models::NormType::RmsNorm);
    assert_eq!(arch.norm_weight_offset(), 0.0);
    assert_eq!(arch.qk_norm_weight_offset(), 0.0);
    // No embedding-scale operation declared — `None`, not `Some(1.0)`.
    assert_eq!(arch.embed_scale(), None);
    assert_eq!(arch.bos_token_id(), None);
    assert_eq!(arch.activation(), larql_models::Activation::Silu);
    assert_eq!(arch.ffn_type(), larql_models::FfnType::Gated);
    assert!(!arch.has_post_norms());
    assert!(!arch.is_sliding_window_layer(1));
    assert_eq!(arch.sliding_window_size(), Some(128));
    assert_eq!(arch.rope_base_for_layer(1), 20000.0);
    assert_eq!(arch.head_dim_for_layer(1), 4);
    assert_eq!(arch.num_q_heads_for_layer(1), 4);
    assert_eq!(arch.num_kv_heads_for_layer(1), 2);
    assert_eq!(arch.rotary_fraction_for_layer(1), 1.0);
    assert!(!arch.v_shares_k(1));
    assert!(!arch.has_v_norm());
    assert_eq!(arch.layer_scalar_key(1), None);
    assert_eq!(arch.attention_scale(), 0.5);
    assert_eq!(arch.attention_scale_for_layer(1), 0.5);
    assert_eq!(arch.kv_shared_source_layer(1), None);

    assert!(!arch.has_per_layer_embeddings());
    assert_eq!(arch.per_layer_embed_dim(), 0);
    assert_eq!(arch.per_layer_embed_key(), None);
    assert_eq!(arch.per_layer_input_gate_key(1), None);
    assert_eq!(arch.per_layer_projection_key(1), None);
    assert_eq!(arch.post_per_layer_input_norm_key(1), None);
    assert_eq!(arch.attn_logit_softcapping(), None);
    assert_eq!(arch.final_logit_softcapping(), None);
    assert_eq!(arch.residual_multiplier(), 1.0);
    assert_eq!(arch.attention_multiplier(), 1.0);
    assert_eq!(arch.logits_scaling(), 1.0);

    assert_eq!(arch.expert_format(), ExpertFormat::PerExpert);
    assert!(!arch.is_moe());
    assert_eq!(arch.num_experts(), 0);
    assert_eq!(arch.num_experts_per_token(), 0);
    assert_eq!(arch.num_shared_experts(), 0);
    assert_eq!(arch.moe_router_key(1), None);
    assert_eq!(arch.moe_router_type(), "top_k_softmax");
    assert_eq!(arch.expert_ffn_gate_key(1, 0), None);
    assert_eq!(arch.expert_ffn_up_key(1, 0), None);
    assert_eq!(arch.expert_ffn_down_key(1, 0), None);
    assert_eq!(arch.packed_gate_up_blocks_key(1), None);
    assert_eq!(arch.packed_gate_up_scales_key(1), None);
    assert_eq!(arch.packed_down_blocks_key(1), None);
    assert_eq!(arch.packed_down_scales_key(1), None);
    assert_eq!(arch.shared_expert_gate_key(1), None);
    assert_eq!(arch.shared_expert_up_key(1), None);
    assert_eq!(arch.shared_expert_down_key(1), None);

    assert!(!arch.is_hybrid_moe());
    assert_eq!(arch.moe_intermediate_size(), 0);
    assert_eq!(arch.packed_experts_gate_up_key(1), None);
    assert_eq!(arch.packed_experts_down_key(1), None);
    assert_eq!(arch.moe_router_scale_key(1), None);
    assert_eq!(arch.moe_router_per_expert_scale_key(1), None);
    assert_eq!(arch.moe_router_norm_key(1), None);
    assert!(!arch.moe_router_norm_parameter_free());
    assert_eq!(arch.moe_router_input_scalar(), None);
    assert_eq!(arch.moe_post_outer_norm_key(1), None);
    assert_eq!(arch.moe_post_ffn1_norm_key(1), None);
    assert_eq!(arch.moe_pre_experts_norm_key(1), None);
    assert_eq!(arch.moe_post_experts_norm_key(1), None);
    assert!(!arch.moe_has_combined_output_norm());

    assert!(!arch.uses_mla());
    assert_eq!(arch.kv_lora_rank(), 0);
    assert_eq!(arch.q_lora_rank(), 0);
    assert_eq!(arch.mla_kv_a_key(1), None);
    assert_eq!(arch.mla_kv_b_key(1), None);
    assert_eq!(arch.mla_q_a_key(1), None);
    assert_eq!(arch.mla_q_b_key(1), None);
    assert_eq!(arch.rope_scaling_type(), Some("linear"));
    assert_eq!(arch.rope_scaling_factor(), 2.0);
    assert_eq!(arch.norm_eps(), 1e-6);
}

// ═══════════════════════════════════════════════════════════════
// `output_head_reuses_embedding` — tied-embedding synthesis
// ═══════════════════════════════════════════════════════════════

fn text_lm_arch_with_tie(tie_word_embeddings: Option<bool>) -> Box<dyn ModelArchitecture> {
    let mut config = serde_json::json!({
        "model_type": "unknown_model",
        "hidden_size": 16, "num_hidden_layers": 2, "intermediate_size": 32,
        "num_attention_heads": 4, "num_key_value_heads": 2,
    });
    if let Some(tie) = tie_word_embeddings {
        config["tie_word_embeddings"] = serde_json::json!(tie);
    }
    detect_from_json(&config)
}

#[test]
fn output_head_reuses_embedding_when_tie_word_embeddings_absent() {
    // H5a's own rule: absence is not a claim either way, and every
    // existing tied-embedding checkpoint (Gemma 2/3 included) relies on
    // exactly this — they never bother re-declaring the HF class default.
    let arch = text_lm_arch_with_tie(None);
    assert!(arch.output_head_reuses_embedding());
}

#[test]
fn output_head_reuses_embedding_when_tie_word_embeddings_true() {
    let arch = text_lm_arch_with_tie(Some(true));
    assert!(arch.output_head_reuses_embedding());
}

#[test]
fn output_head_does_not_reuse_embedding_when_explicitly_untied() {
    // GPT-OSS/OLMoE shape: `tie_word_embeddings: false` must block the
    // reuse fallback, so a checkpoint that lost its `lm_head` tensor fails
    // loudly instead of silently serving the embedding as the projection.
    let arch = text_lm_arch_with_tie(Some(false));
    assert!(!arch.output_head_reuses_embedding());
}

#[test]
fn gemma2_output_head_reuses_embedding_by_default() {
    // Gemma 2 checkpoints ship no `lm_head.weight` and never declare
    // `tie_word_embeddings` — the trait default must resolve this to
    // "tied", not "unjudged".
    let arch = gemma2_arch();
    assert!(arch.has_lm_head());
    assert!(arch.output_head_reuses_embedding());
}

// ═══════════════════════════════════════════════════════════════
// Config validation
// ═══════════════════════════════════════════════════════════════

fn validation_fields(arch: &dyn ModelArchitecture) -> Vec<&'static str> {
    arch.validate()
        .expect_err("config should fail validation")
        .into_iter()
        .map(|error| error.field)
        .collect()
}

#[test]
fn validation_accepts_known_architecture_configs() {
    let configs = [
        serde_json::json!({"model_type": "llama", "hidden_size": 4096, "num_hidden_layers": 32, "intermediate_size": 14336, "num_attention_heads": 32, "num_key_value_heads": 8}),
        serde_json::json!({"model_type": "gpt_oss", "hidden_size": 2880, "num_hidden_layers": 36, "intermediate_size": 2880, "num_attention_heads": 64, "num_key_value_heads": 8, "num_local_experts": 128, "num_experts_per_tok": 4, "head_dim": 64}),
        serde_json::json!({"model_type": "qwen3_moe", "hidden_size": 2048, "num_hidden_layers": 48, "intermediate_size": 6144, "moe_intermediate_size": 768, "num_attention_heads": 32, "num_key_value_heads": 4, "num_experts": 128, "num_experts_per_tok": 8}),
        serde_json::json!({"model_type": "gemma4", "text_config": {"model_type": "gemma4_text", "hidden_size": 1536, "intermediate_size": 6144, "num_hidden_layers": 4, "num_attention_heads": 8, "num_key_value_heads": 1, "head_dim": 256, "global_head_dim": 512, "num_global_key_value_heads": 1, "sliding_window_pattern": 2, "layer_types": ["sliding_attention", "full_attention", "sliding_attention", "full_attention"], "num_kv_shared_layers": 1, "rope_parameters": {"full_attention": {"rope_theta": 1000000.0, "partial_rotary_factor": 0.25}, "sliding_attention": {"rope_theta": 10000.0}}}}),
    ];

    for config in &configs {
        let arch = detect_from_json(config);
        assert!(
            arch.validate().is_ok(),
            "{} failed validation: {:?}",
            arch.family(),
            arch.validate().err()
        );
    }
}

#[test]
fn validation_rejects_zero_core_dimensions() {
    let arch = detect_from_json(&serde_json::json!({
        "model_type": "llama",
        "hidden_size": 0,
        "num_hidden_layers": 0,
        "intermediate_size": 0,
        "num_attention_heads": 0,
        "num_key_value_heads": 0,
        "head_dim": 0,
        "rope_theta": 0.0,
    }));
    let fields = validation_fields(arch.as_ref());

    assert!(fields.contains(&FIELD_NUM_LAYERS));
    assert!(fields.contains(&FIELD_HIDDEN_SIZE));
    assert!(fields.contains(&FIELD_INTERMEDIATE_SIZE));
    assert!(fields.contains(&FIELD_NUM_Q_HEADS));
    assert!(fields.contains(&FIELD_NUM_KV_HEADS));
    assert!(fields.contains(&FIELD_HEAD_DIM));
    assert!(fields.contains(&FIELD_ROPE_BASE));
}

#[test]
fn detect_from_json_validated_returns_validation_error() {
    let result = detect_from_json_validated(&serde_json::json!({
        "model_type": "llama",
        "hidden_size": 0,
        "num_hidden_layers": 1,
        "intermediate_size": 16,
        "num_attention_heads": 2,
        "num_key_value_heads": 2,
        "head_dim": 2,
    }));

    assert!(result.is_err());
}

#[test]
fn validation_rejects_invalid_attention_geometry() {
    let arch = detect_from_json(&serde_json::json!({
        "model_type": "llama",
        "hidden_size": 4100,
        "num_hidden_layers": 2,
        "intermediate_size": 8192,
        "num_attention_heads": 10,
        "num_key_value_heads": 3,
        "head_dim": 128,
    }));
    let fields = validation_fields(arch.as_ref());

    assert!(fields.contains(&FIELD_NUM_Q_HEADS));
}

#[test]
fn validation_rejects_invalid_rope_values() {
    let arch = detect_from_json(&serde_json::json!({
        "model_type": "llama",
        "hidden_size": 4096,
        "num_hidden_layers": 2,
        "intermediate_size": 8192,
        "num_attention_heads": 32,
        "num_key_value_heads": 8,
        "head_dim": 128,
        "partial_rotary_factor": 1.5,
        "rope_scaling": {"type": "", "factor": -1.0},
    }));
    let fields = validation_fields(arch.as_ref());

    assert!(fields.contains(&FIELD_PARTIAL_ROTARY_FACTOR));
    assert!(fields.contains(&FIELD_ROPE_SCALING_TYPE));
    assert!(fields.contains(&FIELD_ROPE_SCALING_FACTOR));
}

#[test]
fn validation_rejects_layer_metadata_mismatch() {
    let arch = detect_from_json(&serde_json::json!({
        "model_type": "gemma4",
        "text_config": {
            "model_type": "gemma4_text",
            "hidden_size": 1536,
            "intermediate_size": 6144,
            "num_hidden_layers": 4,
            "num_attention_heads": 8,
            "num_key_value_heads": 1,
            "head_dim": 256,
            "layer_types": ["sliding_attention", ""],
            "num_kv_shared_layers": 4,
        }
    }));
    let fields = validation_fields(arch.as_ref());

    assert!(fields.contains(&FIELD_LAYER_TYPES));
    assert!(fields.contains(&FIELD_NUM_KV_SHARED_LAYERS));
}

#[test]
fn validation_rejects_moe_without_routing_width() {
    let arch = detect_from_json(&serde_json::json!({
        "model_type": "qwen3_moe",
        "hidden_size": 2048,
        "num_hidden_layers": 4,
        "intermediate_size": 6144,
        "num_attention_heads": 32,
        "num_key_value_heads": 4,
        "num_experts": 16,
    }));
    let fields = validation_fields(arch.as_ref());

    assert!(fields.contains(&FIELD_NUM_EXPERTS_PER_TOKEN));
}

#[test]
fn validation_rejects_moe_top_k_greater_than_experts() {
    let arch = detect_from_json(&serde_json::json!({
        "model_type": "qwen3_moe",
        "hidden_size": 2048,
        "num_hidden_layers": 4,
        "intermediate_size": 6144,
        "num_attention_heads": 32,
        "num_key_value_heads": 4,
        "num_experts": 4,
        "num_experts_per_tok": 8,
    }));
    let fields = validation_fields(arch.as_ref());

    assert!(fields.contains(&FIELD_NUM_EXPERTS_PER_TOKEN));
}

#[test]
fn validation_rejects_hybrid_moe_without_expert_hidden_size() {
    let arch = detect_from_json(&serde_json::json!({
        "model_type": "gemma4",
        "text_config": {
            "model_type": "gemma4_text",
            "hidden_size": 1536,
            "intermediate_size": 6144,
            "num_hidden_layers": 4,
            "num_attention_heads": 8,
            "num_key_value_heads": 1,
            "head_dim": 256,
            "enable_moe_block": true,
            "num_experts": 4,
            "top_k_experts": 1,
        }
    }));
    let fields = validation_fields(arch.as_ref());

    assert!(fields.contains(&FIELD_MOE_INTERMEDIATE_SIZE));
}

// ═══════════════════════════════════════════════════════════════
// Cross-architecture key comparison
// ═══════════════════════════════════════════════════════════════

#[test]
fn all_architectures_have_attn_keys() {
    let configs = [
        serde_json::json!({"model_type": "llama", "hidden_size": 4096, "num_hidden_layers": 32, "intermediate_size": 14336, "num_attention_heads": 32, "num_key_value_heads": 8}),
        serde_json::json!({"model_type": "gemma3", "text_config": {"model_type": "gemma3_text", "hidden_size": 2560, "num_hidden_layers": 34, "intermediate_size": 10240, "num_attention_heads": 8, "num_key_value_heads": 4}}),
        serde_json::json!({"model_type": "mistral", "hidden_size": 4096, "num_hidden_layers": 32, "intermediate_size": 14336, "num_attention_heads": 32, "num_key_value_heads": 8}),
        serde_json::json!({"model_type": "qwen2", "hidden_size": 2048, "num_hidden_layers": 24, "intermediate_size": 5504, "num_attention_heads": 16, "num_key_value_heads": 2}),
        serde_json::json!({"model_type": "gpt_oss", "hidden_size": 2880, "num_hidden_layers": 36, "intermediate_size": 2880, "num_attention_heads": 64, "num_key_value_heads": 8, "num_local_experts": 128, "num_experts_per_tok": 4}),
    ];

    for config in &configs {
        let arch = detect_from_json(config);
        // All architectures must produce non-empty attention keys
        assert!(
            !arch.attn_q_key(0).is_empty(),
            "{} has empty Q key",
            arch.family()
        );
        assert!(
            !arch.attn_k_key(0).is_empty(),
            "{} has empty K key",
            arch.family()
        );
        assert!(
            !arch.attn_v_key(0).is_empty(),
            "{} has empty V key",
            arch.family()
        );
        assert!(
            !arch.attn_o_key(0).is_empty(),
            "{} has empty O key",
            arch.family()
        );
    }
}

// ═══════════════════════════════════════════════════════════════
// Norm-epsilon fallback — a per-family fact, pinned per family
// ═══════════════════════════════════════════════════════════════

/// `transformers`' config classes disagree about the `rms_norm_eps` class
/// default, and a checkpoint that omits the field is served at *its own*
/// class default. One crate-wide fallback answered for everyone, and OLMoE —
/// which ships no `rms_norm_eps` at all — ran every norm of every layer an
/// order of magnitude tight. See `docs/k3-funnel.md` §4.8.
///
/// Every family in the support table is listed, so adding an architecture
/// without deciding its value fails here rather than silently inheriting the
/// majority. The values are read off the `transformers` config classes, not
/// guessed.
#[test]
fn norm_eps_fallback_is_pinned_per_family() {
    const EPS_1E6: f32 = 1e-6;
    const EPS_1E5: f32 = 1e-5;
    let cases: [(serde_json::Value, f32); 9] = [
        (
            serde_json::json!({"model_type": "llama", "hidden_size": 64, "num_hidden_layers": 2, "intermediate_size": 128, "num_attention_heads": 4, "num_key_value_heads": 2}),
            EPS_1E6,
        ),
        (
            serde_json::json!({"model_type": "mistral", "hidden_size": 64, "num_hidden_layers": 2, "intermediate_size": 128, "num_attention_heads": 4, "num_key_value_heads": 2}),
            EPS_1E6,
        ),
        (
            serde_json::json!({"model_type": "qwen2", "hidden_size": 64, "num_hidden_layers": 2, "intermediate_size": 128, "num_attention_heads": 4, "num_key_value_heads": 2}),
            EPS_1E6,
        ),
        (
            serde_json::json!({"model_type": "qwen3_moe", "hidden_size": 64, "num_hidden_layers": 2, "intermediate_size": 128, "num_attention_heads": 4, "num_key_value_heads": 2, "num_experts": 8, "num_experts_per_tok": 2}),
            EPS_1E6,
        ),
        (
            serde_json::json!({"model_type": "gemma2", "hidden_size": 64, "num_hidden_layers": 2, "intermediate_size": 128, "num_attention_heads": 4, "num_key_value_heads": 2}),
            EPS_1E6,
        ),
        (
            serde_json::json!({"model_type": "gemma3", "text_config": {"model_type": "gemma3_text", "hidden_size": 64, "num_hidden_layers": 2, "intermediate_size": 128, "num_attention_heads": 4, "num_key_value_heads": 2}}),
            EPS_1E6,
        ),
        // The 1e-5 families. OLMoE is the one that actually fires in practice.
        (
            serde_json::json!({"model_type": "olmoe", "hidden_size": 64, "num_hidden_layers": 2, "intermediate_size": 128, "num_attention_heads": 4, "num_key_value_heads": 4, "num_experts": 8, "num_experts_per_tok": 2, "norm_topk_prob": false}),
            EPS_1E5,
        ),
        (
            serde_json::json!({"model_type": "gpt_oss", "hidden_size": 64, "num_hidden_layers": 2, "intermediate_size": 64, "num_attention_heads": 4, "num_key_value_heads": 2, "num_local_experts": 8, "num_experts_per_tok": 2}),
            EPS_1E5,
        ),
        (
            serde_json::json!({"model_type": "starcoder2", "hidden_size": 64, "num_hidden_layers": 2, "intermediate_size": 128, "num_attention_heads": 4, "num_key_value_heads": 2}),
            EPS_1E5,
        ),
    ];

    for (config, expected) in &cases {
        let arch = detect_from_json(config);
        assert_eq!(
            arch.default_norm_eps(),
            *expected,
            "{}: fallback epsilon must match its own transformers config class \
             default, not the majority's",
            arch.family()
        );
        // With no eps in the config, `norm_eps()` must *be* the fallback —
        // otherwise declaring the fallback changes nothing that runs.
        assert_eq!(arch.norm_eps(), *expected, "{}", arch.family());
    }
}

/// An explicit `rms_norm_eps` always wins over the family fallback: the
/// fallback exists for absence, and must never override a stated value.
#[test]
fn explicit_norm_eps_overrides_the_family_fallback() {
    let arch = detect_from_json(&serde_json::json!({
        "model_type": "olmoe", "hidden_size": 64, "num_hidden_layers": 2,
        "intermediate_size": 128, "num_attention_heads": 4, "num_key_value_heads": 4,
        "num_experts": 8, "num_experts_per_tok": 2, "norm_topk_prob": false,
        "rms_norm_eps": 1e-7,
    }));
    assert_eq!(arch.default_norm_eps(), 1e-5, "the fallback is unchanged");
    assert_eq!(arch.norm_eps(), 1e-7, "but the config is what runs");
}

// ═══════════════════════════════════════════════════════════════
// ModelWeights: drop_ffn_weights
// ═══════════════════════════════════════════════════════════════

#[test]
fn drop_ffn_weights_removes_ffn_tensors() {
    use larql_models::{ModelWeights, WeightArray};
    use std::collections::HashMap;

    let arch = detect_from_json(&serde_json::json!({
        "model_type": "llama",
        "hidden_size": 4,
        "num_hidden_layers": 2,
        "intermediate_size": 8,
        "num_attention_heads": 2,
        "num_key_value_heads": 2
    }));

    let small = WeightArray::zeros((2, 4));

    let mut tensors = HashMap::new();
    // FFN tensors (should be removed)
    tensors.insert("layers.0.mlp.gate_proj.weight".into(), small.clone());
    tensors.insert("layers.0.mlp.up_proj.weight".into(), small.clone());
    tensors.insert("layers.0.mlp.down_proj.weight".into(), small.clone());
    tensors.insert("layers.1.mlp.gate_proj.weight".into(), small.clone());
    tensors.insert("layers.1.mlp.up_proj.weight".into(), small.clone());
    tensors.insert("layers.1.mlp.down_proj.weight".into(), small.clone());
    // Attention tensors (should be kept)
    tensors.insert("layers.0.self_attn.q_proj.weight".into(), small.clone());
    tensors.insert("layers.0.self_attn.k_proj.weight".into(), small.clone());
    // Norm (should be kept)
    tensors.insert("layers.0.input_layernorm.weight".into(), small.clone());

    let mut weights = ModelWeights {
        tensors,
        vectors: HashMap::new(),
        raw_bytes: HashMap::new(),
        skipped_tensors: Vec::new(),
        packed_mmaps: HashMap::new(),
        packed_byte_ranges: HashMap::new(),
        per_layer_ffn_format: Default::default(),
        per_layer_ffn_arrangement: Default::default(),
        embed: small.clone(),
        lm_head: small.clone(),
        position_embed: None,
        arch,
        num_layers: 2,
        hidden_size: 4,
        intermediate_size: 8,
        vocab_size: 100,
        head_dim: 2,
        num_q_heads: 2,
        num_kv_heads: 2,
        rope_base: 10000.0,
    };

    assert_eq!(weights.tensors.len(), 9);
    let freed = weights.drop_ffn_weights();

    // 6 FFN tensors removed (2 layers × gate/up/down)
    assert_eq!(weights.tensors.len(), 3, "should keep attn + norm only");
    assert!(freed > 0, "should report freed bytes");

    // Verify correct tensors remain
    assert!(weights
        .tensors
        .contains_key("layers.0.self_attn.q_proj.weight"));
    assert!(weights
        .tensors
        .contains_key("layers.0.self_attn.k_proj.weight"));
    assert!(weights
        .tensors
        .contains_key("layers.0.input_layernorm.weight"));

    // Verify FFN tensors are gone
    assert!(!weights
        .tensors
        .contains_key("layers.0.mlp.gate_proj.weight"));
    assert!(!weights
        .tensors
        .contains_key("layers.1.mlp.down_proj.weight"));
}

#[test]
fn drop_ffn_weights_removes_moe_experts() {
    use larql_models::{ModelWeights, WeightArray};
    use std::collections::HashMap;

    let arch = detect_from_json(&serde_json::json!({
        "model_type": "mixtral",
        "hidden_size": 4,
        "num_hidden_layers": 1,
        "intermediate_size": 8,
        "num_attention_heads": 2,
        "num_key_value_heads": 2,
        "num_local_experts": 4,
        "num_experts_per_tok": 2
    }));

    let small = WeightArray::zeros((2, 4));
    let mut tensors = HashMap::new();
    // MoE expert tensors
    tensors.insert(
        "layers.0.block_sparse_moe.experts.0.w1.weight".into(),
        small.clone(),
    );
    tensors.insert(
        "layers.0.block_sparse_moe.experts.0.w2.weight".into(),
        small.clone(),
    );
    tensors.insert(
        "layers.0.block_sparse_moe.experts.0.w3.weight".into(),
        small.clone(),
    );
    // Attention (keep)
    tensors.insert("layers.0.self_attn.q_proj.weight".into(), small.clone());

    let mut weights = ModelWeights {
        tensors,
        vectors: HashMap::new(),
        raw_bytes: HashMap::new(),
        skipped_tensors: Vec::new(),
        packed_mmaps: HashMap::new(),
        packed_byte_ranges: HashMap::new(),
        per_layer_ffn_format: Default::default(),
        per_layer_ffn_arrangement: Default::default(),
        embed: small.clone(),
        lm_head: small.clone(),
        position_embed: None,
        arch,
        num_layers: 1,
        hidden_size: 4,
        intermediate_size: 8,
        vocab_size: 100,
        head_dim: 2,
        num_q_heads: 2,
        num_kv_heads: 2,
        rope_base: 10000.0,
    };

    weights.drop_ffn_weights();
    // mlp.experts matches the "mlp.experts" pattern
    assert_eq!(weights.tensors.len(), 1, "should only keep attn");
    assert!(weights
        .tensors
        .contains_key("layers.0.self_attn.q_proj.weight"));
}

#[test]
fn drop_ffn_weights_removes_mmap_backed_packed_experts() {
    let mut weights = minimal_weights();
    weights.packed_byte_ranges.insert(
        "layers.0.experts.gate_up_proj".into(),
        ("experts.safetensors".into(), 128, 16),
    );
    weights.packed_byte_ranges.insert(
        "layers.0.experts.down_proj".into(),
        ("experts.safetensors".into(), 256, 8),
    );

    let freed = weights.drop_ffn_weights();

    assert!(freed >= 24);
    assert!(weights.packed_byte_ranges.is_empty());
}

#[test]
fn drop_ffn_weights_removes_starcoder2_ffn_tensors_and_biases() {
    use larql_models::{ModelWeights, WeightArray};
    use std::collections::HashMap;

    let arch = detect_from_json(&serde_json::json!({
        "model_type": "starcoder2",
        "hidden_size": 4,
        "num_hidden_layers": 1,
        "intermediate_size": 8,
        "num_attention_heads": 2,
        "num_key_value_heads": 2
    }));

    let small = WeightArray::zeros((2, 4));
    let mut tensors = HashMap::new();
    tensors.insert("layers.0.mlp.c_fc.weight".into(), small.clone());
    tensors.insert("layers.0.mlp.c_proj.weight".into(), small.clone());
    tensors.insert("layers.0.self_attn.q_proj.weight".into(), small.clone());

    let mut vectors = HashMap::new();
    vectors.insert("layers.0.mlp.c_fc.bias".into(), vec![0.0; 8]);
    vectors.insert("layers.0.mlp.c_proj.bias".into(), vec![0.0; 4]);
    vectors.insert("layers.0.input_layernorm.weight".into(), vec![1.0; 4]);

    let mut weights = ModelWeights {
        tensors,
        vectors,
        raw_bytes: HashMap::new(),
        skipped_tensors: Vec::new(),
        packed_mmaps: HashMap::new(),
        packed_byte_ranges: HashMap::new(),
        per_layer_ffn_format: Default::default(),
        per_layer_ffn_arrangement: Default::default(),
        embed: small.clone(),
        lm_head: small.clone(),
        position_embed: None,
        arch,
        num_layers: 1,
        hidden_size: 4,
        intermediate_size: 8,
        vocab_size: 100,
        head_dim: 2,
        num_q_heads: 2,
        num_kv_heads: 2,
        rope_base: 10000.0,
    };

    let freed = weights.drop_ffn_weights();
    assert!(freed > 0);
    assert!(!weights.tensors.contains_key("layers.0.mlp.c_fc.weight"));
    assert!(!weights.tensors.contains_key("layers.0.mlp.c_proj.weight"));
    assert!(!weights.vectors.contains_key("layers.0.mlp.c_fc.bias"));
    assert!(!weights.vectors.contains_key("layers.0.mlp.c_proj.bias"));
    assert!(weights
        .tensors
        .contains_key("layers.0.self_attn.q_proj.weight"));
    assert!(weights
        .vectors
        .contains_key("layers.0.input_layernorm.weight"));
}

// ═══════════════════════════════════════════════════════════════
// Gemma 4 — per-layer geometry, partial RoPE, V-norm, KV sharing
// ═══════════════════════════════════════════════════════════════

fn gemma4_e2b_arch() -> Box<dyn ModelArchitecture> {
    detect_from_json(&serde_json::json!({
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
            "hidden_size_per_layer_input": 256,
            "num_kv_shared_layers": 20,
            "rope_parameters": {
                "full_attention": {
                    "partial_rotary_factor": 0.25,
                    "rope_theta": 1000000.0
                },
                "sliding_attention": {
                    "rope_theta": 10000.0
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
    }))
}

#[test]
fn gemma4_per_layer_head_dim() {
    let arch = gemma4_e2b_arch();
    // Sliding: head_dim=256, Global: head_dim=512
    assert_eq!(arch.head_dim_for_layer(0), 256);
    assert_eq!(arch.head_dim_for_layer(3), 256);
    assert_eq!(arch.head_dim_for_layer(4), 512); // first global
    assert_eq!(arch.head_dim_for_layer(9), 512);
    assert_eq!(arch.head_dim_for_layer(34), 512); // last layer (global)

    // Q heads constant across all layers
    assert_eq!(arch.num_q_heads_for_layer(0), 8);
    assert_eq!(arch.num_q_heads_for_layer(4), 8);

    // KV heads: 1 everywhere (E2B has MQA, no num_global_key_value_heads)
    assert_eq!(arch.num_kv_heads_for_layer(0), 1);
    assert_eq!(arch.num_kv_heads_for_layer(4), 1);
}

#[test]
fn gemma4_partial_rotary() {
    let arch = gemma4_e2b_arch();
    // Sliding: full rotation
    assert_eq!(arch.rotary_fraction_for_layer(0), 1.0);
    assert_eq!(arch.rotary_fraction_for_layer(3), 1.0);
    // Global: 25% rotation
    assert_eq!(arch.rotary_fraction_for_layer(4), 0.25);
    assert_eq!(arch.rotary_fraction_for_layer(9), 0.25);
}

#[test]
fn gemma4_rope_bases() {
    let arch = gemma4_e2b_arch();
    // Sliding: 10k, Global: 1M
    assert_eq!(arch.rope_base_for_layer(0), 10_000.0);
    assert_eq!(arch.rope_base_for_layer(4), 1_000_000.0);
}

#[test]
fn gemma4_attention_scale_is_one() {
    let arch = gemma4_e2b_arch();
    // QK-norm makes explicit scaling unnecessary
    assert_eq!(arch.attention_scale(), 1.0);
    assert_eq!(arch.attention_scale_for_layer(0), 1.0);
    assert_eq!(arch.attention_scale_for_layer(4), 1.0);
}

#[test]
fn gemma4_v_norm() {
    let arch = gemma4_e2b_arch();
    assert!(arch.has_v_norm());
}

#[test]
fn gemma4_norm_offset_zero() {
    let arch = gemma4_e2b_arch();
    // Gemma 4 stores weights as full multiplier (no +1 like Gemma 2/3)
    assert_eq!(arch.norm_weight_offset(), 0.0);
}

#[test]
fn gemma4_kv_sharing() {
    let arch = gemma4_e2b_arch();
    // First 15 layers: no sharing
    for l in 0..15 {
        assert!(
            arch.kv_shared_source_layer(l).is_none(),
            "L{l} should not be shared"
        );
    }
    // Layers 15-34: shared
    // Sliding shared layers → last non-shared sliding (L13)
    assert_eq!(arch.kv_shared_source_layer(15), Some(13));
    assert_eq!(arch.kv_shared_source_layer(16), Some(13));
    // Global shared layers → last non-shared global (L14)
    assert_eq!(arch.kv_shared_source_layer(19), Some(14));
    assert_eq!(arch.kv_shared_source_layer(34), Some(14));
}

#[test]
fn gemma4_ple() {
    let arch = gemma4_e2b_arch();
    assert!(arch.has_per_layer_embeddings());
    assert_eq!(arch.per_layer_embed_dim(), 256);
    assert_eq!(
        arch.per_layer_input_gate_key(5),
        Some("layers.5.per_layer_input_gate.weight".to_string())
    );
    assert_eq!(
        arch.per_layer_projection_key(5),
        Some("layers.5.per_layer_projection.weight".to_string())
    );
}

#[test]
fn gemma4_layer_scalar() {
    let arch = gemma4_e2b_arch();
    assert_eq!(
        arch.layer_scalar_key(10),
        Some("layers.10.layer_scalar".to_string())
    );
}

#[test]
fn gemma4_prefix_strip() {
    let arch = gemma4_e2b_arch();
    let prefixes = arch.key_prefixes_to_strip();
    // Must strip model.language_model. for multimodal Gemma 4
    assert!(prefixes.contains(&"model.language_model."));
    assert!(prefixes.contains(&"model.language_model.model."));
}

#[test]
fn gemma4_gemma_family_traits() {
    let arch = gemma4_e2b_arch();
    assert_eq!(arch.activation(), larql_models::Activation::GeluTanh);
    assert!(arch.has_post_norms());
    assert!(arch.attn_q_norm_key(0).is_some());
    assert!(arch.attn_k_norm_key(0).is_some());
    // embed_scale = sqrt(hidden_size)
    assert_eq!(arch.embed_scale(), Some((1536.0f32).sqrt()));
}

// ═══════════════════════════════════════════════════════════════
// Gemma 2 — softcapping, QK norm with +1 offset
// ═══════════════════════════════════════════════════════════════

fn gemma2_arch() -> Box<dyn ModelArchitecture> {
    detect_from_json(&serde_json::json!({
        "model_type": "gemma2",
        "hidden_size": 2304, "num_hidden_layers": 26, "intermediate_size": 9216,
        "num_attention_heads": 8, "num_key_value_heads": 4, "head_dim": 256,
        "query_pre_attn_scalar": 256.0,
        "attn_logit_softcapping": 50.0, "final_logit_softcapping": 30.0
    }))
}

#[test]
fn gemma2_detection() {
    let arch = gemma2_arch();
    assert_eq!(arch.family(), "gemma2");
    assert_eq!(arch.config().num_layers, 26);
}

#[test]
fn gemma2_softcapping() {
    let arch = gemma2_arch();
    assert_eq!(arch.attn_logit_softcapping(), Some(50.0));
    assert_eq!(arch.final_logit_softcapping(), Some(30.0));
}

#[test]
fn gemma2_norm_offsets() {
    let arch = gemma2_arch();
    assert_eq!(arch.norm_weight_offset(), 1.0);
    assert_eq!(arch.qk_norm_weight_offset(), 1.0);
}

#[test]
fn gemma2_qk_norm_keys() {
    let arch = gemma2_arch();
    assert_eq!(
        arch.attn_q_norm_key(5).unwrap(),
        "layers.5.self_attn.q_norm.weight"
    );
    assert_eq!(
        arch.attn_k_norm_key(5).unwrap(),
        "layers.5.self_attn.k_norm.weight"
    );
}

#[test]
fn gemma2_attention_scale() {
    let arch = gemma2_arch();
    // query_pre_attn_scalar = 256 → scale = 256^(-0.5) = 1/16 = 0.0625
    let expected = (256.0f64).powf(-0.5);
    assert_eq!(arch.attention_scale(), expected);
}

#[test]
fn gemma2_gemma_family_traits() {
    let arch = gemma2_arch();
    assert_eq!(arch.activation(), larql_models::Activation::GeluTanh);
    assert!(arch.has_post_norms());
    assert_eq!(arch.embed_scale(), Some((2304.0f32).sqrt()));
}

#[test]
fn gemma2_sliding_window_alternates_every_other_layer() {
    // HF `Gemma2DecoderLayer.is_sliding = not bool(layer_idx % 2)`: even
    // layers slide, odd layers see full attention — a fixed period-2
    // pattern, not a declared `layer_types` interleave.
    let arch = gemma2_arch();
    for layer in 0..26 {
        assert_eq!(
            arch.is_sliding_window_layer(layer),
            layer.is_multiple_of(2),
            "layer {layer} sliding-window mismatch"
        );
    }
    assert_eq!(arch.sliding_window_size(), None); // no `sliding_window` in gemma2_arch()'s fixture
}

// ═══════════════════════════════════════════════════════════════
// Gemma 3 — sliding window, dual RoPE, QK norm offset
// ═══════════════════════════════════════════════════════════════

fn gemma3_arch() -> Box<dyn ModelArchitecture> {
    detect_from_json(&serde_json::json!({
        "model_type": "gemma3",
        "text_config": {
            "model_type": "gemma3_text",
            "hidden_size": 2560, "num_hidden_layers": 34, "intermediate_size": 10240,
            "num_attention_heads": 8, "num_key_value_heads": 4,
            "head_dim": 256, "sliding_window": 1024
        }
    }))
}

#[test]
fn gemma3_detection() {
    let arch = gemma3_arch();
    assert_eq!(arch.family(), "gemma3");
    assert_eq!(arch.config().num_layers, 34);
}

#[test]
fn gemma3_sliding_window_pattern() {
    let arch = gemma3_arch();
    // Every 6th layer (0-indexed: 5, 11, 17, ...) is full attention
    assert!(arch.is_sliding_window_layer(0));
    assert!(arch.is_sliding_window_layer(4));
    assert!(!arch.is_sliding_window_layer(5)); // full
    assert!(arch.is_sliding_window_layer(6));
    assert!(!arch.is_sliding_window_layer(11)); // full
}

#[test]
fn gemma3_dual_rope_bases() {
    let arch = gemma3_arch();
    // Sliding layers: 10k, full layers: 1M
    assert_eq!(arch.rope_base_for_layer(0), 10_000.0);
    assert_eq!(arch.rope_base_for_layer(5), 1_000_000.0);
}

#[test]
fn gemma3_norm_offsets() {
    let arch = gemma3_arch();
    assert_eq!(arch.norm_weight_offset(), 1.0);
    assert_eq!(arch.qk_norm_weight_offset(), 1.0);
}

#[test]
fn gemma3_gemma_family_traits() {
    let arch = gemma3_arch();
    assert_eq!(arch.activation(), larql_models::Activation::GeluTanh);
    assert!(arch.has_post_norms());
    assert_eq!(arch.embed_scale(), Some((2560.0f32).sqrt()));
    assert!(arch.attn_q_norm_key(0).is_some());
    // No softcapping on Gemma 3
    assert!(arch.attn_logit_softcapping().is_none());
    assert!(arch.final_logit_softcapping().is_none());
}

// ═══════════════════════════════════════════════════════════════
// Mistral — sliding window, Llama-compatible keys
// ═══════════════════════════════════════════════════════════════

#[test]
fn mistral_detection_and_keys() {
    let arch = detect_from_json(&serde_json::json!({
        "model_type": "mistral",
        "hidden_size": 4096, "num_hidden_layers": 32, "intermediate_size": 14336,
        "num_attention_heads": 32, "num_key_value_heads": 8, "sliding_window": 4096
    }));
    assert_eq!(arch.family(), "mistral");
    assert_eq!(arch.sliding_window_size(), Some(4096));
    // Mistral uses same keys as Llama
    assert_eq!(arch.attn_q_key(0), "layers.0.self_attn.q_proj.weight");
    assert_eq!(arch.ffn_gate_key(0), "layers.0.mlp.gate_proj.weight");
    // RMSNorm, SiLU, gated FFN
    assert_eq!(arch.norm_type(), larql_models::NormType::RmsNorm);
    assert_eq!(arch.activation(), larql_models::Activation::Silu);
    assert_eq!(arch.ffn_type(), larql_models::FfnType::Gated);
}

// ═══════════════════════════════════════════════════════════════
// Qwen — attention bias, QK norm keys
// ═══════════════════════════════════════════════════════════════

fn qwen_arch() -> Box<dyn ModelArchitecture> {
    detect_from_json(&serde_json::json!({
        "model_type": "qwen2",
        "hidden_size": 2048, "num_hidden_layers": 24, "intermediate_size": 5504,
        "num_attention_heads": 16, "num_key_value_heads": 2
    }))
}

#[test]
fn qwen_detection() {
    let arch = qwen_arch();
    assert_eq!(arch.family(), "qwen2");
    assert_eq!(arch.config().num_layers, 24);
}

#[test]
fn qwen_attention_bias_keys() {
    let arch = qwen_arch();
    assert_eq!(
        arch.attn_q_bias_key(3).unwrap(),
        "layers.3.self_attn.q_proj.bias"
    );
    assert_eq!(
        arch.attn_k_bias_key(3).unwrap(),
        "layers.3.self_attn.k_proj.bias"
    );
    assert_eq!(
        arch.attn_v_bias_key(3).unwrap(),
        "layers.3.self_attn.v_proj.bias"
    );
}

#[test]
fn qwen_qk_norm_keys() {
    let arch = qwen_arch();
    assert_eq!(
        arch.attn_q_norm_key(0).unwrap(),
        "layers.0.self_attn.q_norm.weight"
    );
    assert_eq!(
        arch.attn_k_norm_key(0).unwrap(),
        "layers.0.self_attn.k_norm.weight"
    );
}

#[test]
fn qwen_classic_norm_weights_are_not_offsets() {
    // Only qwen3_5/qwen3_next store norm weights as offsets from one;
    // the rest of the family stores them plain.
    let arch = qwen_arch();
    assert_eq!(arch.norm_weight_offset(), 0.0);
    assert_eq!(arch.qk_norm_weight_offset(), 0.0);
}

#[test]
fn qwen35_declared_output_gate_resolves_to_the_hf_sigmoid_spec() {
    use larql_models::config::{GateActivation, GateCombine, GatePlacement, GateSource};
    let arch = detect_from_json(&serde_json::json!({
        "model_type": "qwen3_5",
        "hidden_size": 2048, "num_hidden_layers": 4, "intermediate_size": 5504,
        "num_attention_heads": 16, "num_key_value_heads": 2,
        "attn_output_gate": true
    }));
    let spec = arch.attention_output_gate().expect("gate declared true");
    assert_eq!(spec.source, GateSource::FusedQueryProjection);
    // Sigmoid, NOT the silu the config's `output_gate_type: "swish"`
    // spelling suggests — HF computes `x · sigmoid(g)`.
    assert_eq!(spec.activation, GateActivation::Sigmoid);
    assert_eq!(spec.combine, GateCombine::ElementwiseMultiply);
    assert_eq!(
        spec.placement,
        GatePlacement::AfterAggregationBeforeOutputProjection
    );

    // Declared false (or absent) is no gate, not a disabled gate.
    let ungated = detect_from_json(&serde_json::json!({
        "model_type": "qwen3_5",
        "hidden_size": 2048, "num_hidden_layers": 4, "intermediate_size": 5504,
        "num_attention_heads": 16, "num_key_value_heads": 2,
        "attn_output_gate": false
    }));
    assert!(ungated.attention_output_gate().is_none());
}

#[test]
fn qwen_dense_returns_none_for_moe_keys() {
    let arch = qwen_arch();
    assert!(arch.moe_router_key(0).is_none());
    assert!(arch.expert_ffn_gate_key(0, 0).is_none());
    assert!(arch.expert_ffn_up_key(0, 0).is_none());
    assert!(arch.expert_ffn_down_key(0, 0).is_none());
}

// ═══════════════════════════════════════════════════════════════
// DeepSeek — MoE + MLA
// ═══════════════════════════════════════════════════════════════

fn deepseek_arch() -> Box<dyn ModelArchitecture> {
    detect_from_json(&serde_json::json!({
        "model_type": "deepseek_v2",
        "hidden_size": 5120, "num_hidden_layers": 60, "intermediate_size": 12288,
        "num_attention_heads": 128, "num_key_value_heads": 128,
        "n_routed_experts": 160, "num_experts_per_tok": 6, "n_shared_experts": 2,
        "kv_lora_rank": 512, "q_lora_rank": 1536,
        "rope_scaling": { "type": "yarn", "factor": 40.0 }
    }))
}

#[test]
fn deepseek_detection() {
    let arch = deepseek_arch();
    assert_eq!(arch.family(), "deepseek");
    assert_eq!(arch.config().num_layers, 60);
}

#[test]
fn deepseek_moe() {
    let arch = deepseek_arch();
    assert!(arch.is_moe());
    assert_eq!(arch.num_experts(), 160);
    assert_eq!(arch.num_experts_per_token(), 6);
    assert_eq!(arch.num_shared_experts(), 2);
    assert_eq!(arch.expert_format(), ExpertFormat::PerExpert);
}

#[test]
fn deepseek_expert_keys() {
    let arch = deepseek_arch();
    assert_eq!(arch.moe_router_key(0).unwrap(), "layers.0.mlp.gate.weight");
    assert_eq!(
        arch.expert_ffn_gate_key(0, 5).unwrap(),
        "layers.0.mlp.experts.5.gate_proj.weight"
    );
    assert_eq!(
        arch.expert_ffn_up_key(0, 5).unwrap(),
        "layers.0.mlp.experts.5.up_proj.weight"
    );
    assert_eq!(
        arch.expert_ffn_down_key(0, 5).unwrap(),
        "layers.0.mlp.experts.5.down_proj.weight"
    );
}

#[test]
fn deepseek_shared_expert_keys() {
    let arch = deepseek_arch();
    assert_eq!(
        arch.shared_expert_gate_key(0).unwrap(),
        "layers.0.mlp.shared_experts.gate_proj.weight"
    );
    assert_eq!(
        arch.shared_expert_up_key(0).unwrap(),
        "layers.0.mlp.shared_experts.up_proj.weight"
    );
    assert_eq!(
        arch.shared_expert_down_key(0).unwrap(),
        "layers.0.mlp.shared_experts.down_proj.weight"
    );
}

#[test]
fn deepseek_mla() {
    let arch = deepseek_arch();
    assert!(arch.uses_mla());
    assert_eq!(arch.kv_lora_rank(), 512);
    assert_eq!(arch.q_lora_rank(), 1536);
    assert_eq!(
        arch.mla_kv_a_key(0).unwrap(),
        "layers.0.self_attn.kv_a_proj_with_mqa.weight"
    );
    assert_eq!(
        arch.mla_kv_b_key(0).unwrap(),
        "layers.0.self_attn.kv_b_proj.weight"
    );
    assert_eq!(
        arch.mla_q_a_key(0).unwrap(),
        "layers.0.self_attn.q_a_proj.weight"
    );
    assert_eq!(
        arch.mla_q_b_key(0).unwrap(),
        "layers.0.self_attn.q_b_proj.weight"
    );
}

#[test]
fn deepseek_rope_scaling() {
    let arch = deepseek_arch();
    assert_eq!(arch.rope_scaling_type(), Some("yarn"));
    assert_eq!(arch.rope_scaling_factor(), 40.0);
}

// ═══════════════════════════════════════════════════════════════
// Granite — scaling multipliers
// ═══════════════════════════════════════════════════════════════

fn granite_arch() -> Box<dyn ModelArchitecture> {
    detect_from_json(&serde_json::json!({
        "model_type": "granite",
        "hidden_size": 2048, "num_hidden_layers": 40, "intermediate_size": 8192,
        "num_attention_heads": 32, "num_key_value_heads": 8,
        "embedding_multiplier": 12.0, "residual_multiplier": 0.22,
        "attention_multiplier": 0.22, "logits_scaling": 0.13
    }))
}

#[test]
fn granite_detection() {
    let arch = granite_arch();
    assert_eq!(arch.family(), "granite");
    assert_eq!(arch.config().num_layers, 40);
}

#[test]
fn granite_scaling_multipliers() {
    let arch = granite_arch();
    assert_eq!(arch.embed_scale(), Some(12.0));
    assert_eq!(arch.residual_multiplier(), 0.22);
    assert_eq!(arch.attention_multiplier(), 0.22);
    assert_eq!(arch.logits_scaling(), 0.13);
}

#[test]
fn granite_uses_llama_defaults() {
    let arch = granite_arch();
    // Same keys, norm, activation as Llama
    assert_eq!(arch.attn_q_key(0), "layers.0.self_attn.q_proj.weight");
    assert_eq!(arch.ffn_gate_key(0), "layers.0.mlp.gate_proj.weight");
    assert_eq!(arch.norm_type(), larql_models::NormType::RmsNorm);
    assert_eq!(arch.activation(), larql_models::Activation::Silu);
    assert!(!arch.is_moe());
}

// ═══════════════════════════════════════════════════════════════
// StarCoder2 — LayerNorm, GELU, bias, non-gated FFN, c_fc/c_proj keys
// ═══════════════════════════════════════════════════════════════

fn starcoder2_arch() -> Box<dyn ModelArchitecture> {
    detect_from_json(&serde_json::json!({
        "model_type": "starcoder2",
        "hidden_size": 3072, "num_hidden_layers": 30, "intermediate_size": 12288,
        "num_attention_heads": 24, "num_key_value_heads": 2
    }))
}

#[test]
fn starcoder2_detection() {
    let arch = starcoder2_arch();
    assert_eq!(arch.family(), "starcoder2");
    assert_eq!(arch.config().num_layers, 30);
}

/// LayerNorm is `γ·x̂ + β`, so a LayerNorm architecture must name its `β`.
///
/// Nothing named a norm bias until 2026-07-31: extraction never wrote one and
/// `build_pipeline_layers` hardcoded `input_norm_bias: None`, so every
/// vindex-backed and Metal path silently dropped the shift term for GPT-2 and
/// StarCoder2 — while the Metal `layer_norm` shader implemented `+ bias` and
/// always selected its no-bias variant.
#[test]
fn starcoder2_names_its_layernorm_biases() {
    let arch = starcoder2_arch();
    assert_eq!(
        arch.input_layernorm_bias_key(3).as_deref(),
        Some("layers.3.input_layernorm.bias")
    );
    assert_eq!(
        arch.post_attention_layernorm_bias_key(3).as_deref(),
        Some("layers.3.post_attention_layernorm.bias")
    );
    assert_eq!(arch.final_norm_bias_key().as_deref(), Some("norm.bias"));
}

/// An RMSNorm architecture has no `β` at all, so it must claim none —
/// otherwise the coverage audit would treat a nonexistent tensor as expected
/// and extraction would ask for a key that can never resolve.
#[test]
fn an_rmsnorm_architecture_claims_no_layernorm_bias() {
    let arch = detect_from_json(&serde_json::json!({
        "model_type": "llama",
        "hidden_size": 64, "num_hidden_layers": 2, "intermediate_size": 128,
        "num_attention_heads": 4, "num_key_value_heads": 4, "vocab_size": 32
    }));
    assert_eq!(arch.norm_type(), larql_models::NormType::RmsNorm);
    assert!(arch.input_layernorm_bias_key(0).is_none());
    assert!(arch.post_attention_layernorm_bias_key(0).is_none());
    assert!(arch.final_norm_bias_key().is_none());
}

/// The bias key is derived from the weight key, so it tracks any architecture
/// that overrides its norm naming instead of drifting from it.
#[test]
fn the_norm_bias_key_tracks_the_weight_key() {
    let arch = starcoder2_arch();
    for layer in [0usize, 7] {
        let weight = arch.input_layernorm_key(layer);
        let bias = arch.input_layernorm_bias_key(layer).unwrap();
        assert_eq!(bias, weight.replace(".weight", ".bias"));
        assert!(bias.ends_with(".bias"));
    }
}

#[test]
fn starcoder2_norm_and_activation() {
    let arch = starcoder2_arch();
    assert_eq!(arch.norm_type(), larql_models::NormType::LayerNorm);
    assert_eq!(arch.activation(), larql_models::Activation::GeluTanh);
    assert_eq!(arch.ffn_type(), larql_models::FfnType::Standard);
}

#[test]
fn starcoder2_ffn_keys() {
    let arch = starcoder2_arch();
    // Uses c_fc/c_proj naming
    assert_eq!(arch.ffn_up_key(0), "layers.0.mlp.c_fc.weight");
    assert_eq!(arch.ffn_down_key(0), "layers.0.mlp.c_proj.weight");
}

#[test]
fn starcoder2_bias_keys() {
    let arch = starcoder2_arch();
    // FFN biases
    assert_eq!(arch.ffn_up_bias_key(0).unwrap(), "layers.0.mlp.c_fc.bias");
    assert_eq!(
        arch.ffn_down_bias_key(0).unwrap(),
        "layers.0.mlp.c_proj.bias"
    );
    // Attention biases (including O)
    assert_eq!(
        arch.attn_q_bias_key(0).unwrap(),
        "layers.0.self_attn.q_proj.bias"
    );
    assert_eq!(
        arch.attn_k_bias_key(0).unwrap(),
        "layers.0.self_attn.k_proj.bias"
    );
    assert_eq!(
        arch.attn_v_bias_key(0).unwrap(),
        "layers.0.self_attn.v_proj.bias"
    );
    assert_eq!(
        arch.attn_o_bias_key(0).unwrap(),
        "layers.0.self_attn.o_proj.bias"
    );
}

// ═══════════════════════════════════════════════════════════════
// Generic fallback — safe defaults for unknown models
// ═══════════════════════════════════════════════════════════════

#[test]
fn generic_fallback() {
    let arch = detect_from_json(&serde_json::json!({
        "model_type": "some_future_model",
        "hidden_size": 4096, "num_hidden_layers": 32, "intermediate_size": 11008,
        "num_attention_heads": 32, "num_key_value_heads": 32
    }));
    assert_eq!(arch.family(), "generic");
    // All safe defaults
    assert_eq!(arch.norm_type(), larql_models::NormType::RmsNorm);
    assert_eq!(arch.activation(), larql_models::Activation::Silu);
    assert_eq!(arch.ffn_type(), larql_models::FfnType::Gated);
    assert_eq!(arch.norm_weight_offset(), 0.0);
    // No embedding-scale operation declared — `None`, not `Some(1.0)`.
    assert_eq!(arch.embed_scale(), None);
    assert!(!arch.has_post_norms());
    assert!(!arch.is_moe());
    assert!(!arch.uses_mla());
    assert!(arch.attn_q_norm_key(0).is_none());
    assert!(arch.attn_logit_softcapping().is_none());
    assert!(arch.attn_q_bias_key(0).is_none());
    assert!(arch.ffn_up_bias_key(0).is_none());
    // Standard keys still work
    assert_eq!(arch.attn_q_key(0), "layers.0.self_attn.q_proj.weight");
    assert_eq!(arch.ffn_gate_key(0), "layers.0.mlp.gate_proj.weight");
}

// ═══════════════════════════════════════════════════════════════
// Cross-architecture: default multipliers are 1.0 for non-Granite
// ═══════════════════════════════════════════════════════════════

#[test]
fn non_granite_multipliers_are_one() {
    let configs = [
        serde_json::json!({"model_type": "llama", "hidden_size": 4096, "num_hidden_layers": 32, "intermediate_size": 14336, "num_attention_heads": 32, "num_key_value_heads": 8}),
        serde_json::json!({"model_type": "mistral", "hidden_size": 4096, "num_hidden_layers": 32, "intermediate_size": 14336, "num_attention_heads": 32, "num_key_value_heads": 8}),
        serde_json::json!({"model_type": "qwen2", "hidden_size": 2048, "num_hidden_layers": 24, "intermediate_size": 5504, "num_attention_heads": 16, "num_key_value_heads": 2}),
    ];
    for config in &configs {
        let arch = detect_from_json(config);
        assert_eq!(
            arch.residual_multiplier(),
            1.0,
            "{} should have residual_multiplier=1.0",
            arch.family()
        );
        assert_eq!(
            arch.attention_multiplier(),
            1.0,
            "{} should have attention_multiplier=1.0",
            arch.family()
        );
        assert_eq!(
            arch.logits_scaling(),
            1.0,
            "{} should have logits_scaling=1.0",
            arch.family()
        );
    }
}

// ═══════════════════════════════════════════════════════════════
// Q4 round-trip: quantize then dequantize
// ═══════════════════════════════════════════════════════════════

#[test]
fn q4_0_round_trip() {
    use larql_models::quant::ggml;

    let data: Vec<f32> = (0..64).map(|i| (i as f32 - 32.0) * 0.25).collect();
    let q4 = ggml::quantize_q4_0(&data);
    let decoded = ggml::dequantize_q4_0(&q4, 64).unwrap();

    assert_eq!(decoded.len(), 64);
    let max_err: f32 = data
        .iter()
        .zip(decoded.iter())
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f32, f32::max);
    // Q4 is lossy but should be within ~2x the quantization step
    assert!(
        max_err < 2.0,
        "Q4 round-trip max error {max_err} exceeds 2.0"
    );
}

#[test]
fn q8_0_round_trip() {
    use larql_models::quant::ggml;

    let data: Vec<f32> = (0..32).map(|i| (i as f32 - 16.0) * 0.1).collect();
    let q8 = ggml::quantize_q8_0(&data);
    let decoded = ggml::dequantize(&q8, ggml::TYPE_Q8_0, 32).unwrap();

    assert_eq!(decoded.len(), 32);
    let max_err: f32 = data
        .iter()
        .zip(decoded.iter())
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f32, f32::max);
    // Q8 should be much more accurate than Q4
    assert!(
        max_err < 0.02,
        "Q8 round-trip max error {max_err} exceeds 0.02"
    );
}

// ═══════════════════════════════════════════════════════════════
// ModelWeights — drop_attn_weights, drop_lm_head, drop_embed, get_packed_bytes
// ═══════════════════════════════════════════════════════════════

fn minimal_weights() -> larql_models::ModelWeights {
    use larql_models::{ModelWeights, WeightArray};
    use std::collections::HashMap;

    let arch = detect_from_json(&serde_json::json!({
        "model_type": "llama",
        "hidden_size": 4,
        "num_hidden_layers": 1,
        "intermediate_size": 8,
        "num_attention_heads": 2,
        "num_key_value_heads": 2,
    }));
    let small = WeightArray::zeros((2, 4));
    let mut tensors = HashMap::new();
    tensors.insert("layers.0.self_attn.q_proj.weight".into(), small.clone());
    tensors.insert("layers.0.self_attn.k_proj.weight".into(), small.clone());
    tensors.insert("layers.0.self_attn.v_proj.weight".into(), small.clone());
    tensors.insert("layers.0.self_attn.o_proj.weight".into(), small.clone());
    tensors.insert("layers.0.self_attn.q_norm.weight".into(), small.clone());
    tensors.insert("layers.0.mlp.gate_proj.weight".into(), small.clone());
    tensors.insert("layers.0.mlp.up_proj.weight".into(), small.clone());
    tensors.insert("layers.0.mlp.down_proj.weight".into(), small.clone());
    tensors.insert("layers.0.input_layernorm.weight".into(), small.clone());
    ModelWeights {
        tensors,
        vectors: HashMap::new(),
        raw_bytes: HashMap::new(),
        skipped_tensors: Vec::new(),
        packed_mmaps: HashMap::new(),
        packed_byte_ranges: HashMap::new(),
        per_layer_ffn_format: Default::default(),
        per_layer_ffn_arrangement: Default::default(),
        embed: small.clone(),
        lm_head: small.clone(),
        position_embed: None,
        arch,
        num_layers: 1,
        hidden_size: 4,
        intermediate_size: 8,
        vocab_size: 100,
        head_dim: 2,
        num_q_heads: 2,
        num_kv_heads: 2,
        rope_base: 10000.0,
    }
}

#[test]
fn drop_attn_weights_removes_qkvo_and_norms() {
    let mut w = minimal_weights();
    assert_eq!(w.tensors.len(), 9);
    let freed = w.drop_attn_weights();
    assert!(freed > 0);
    // q/k/v/o + q_norm removed (5 tensors); FFN + norm remain (4)
    assert_eq!(w.tensors.len(), 4, "expected ffn + layernorm to remain");
    assert!(!w.tensors.contains_key("layers.0.self_attn.q_proj.weight"));
    assert!(!w.tensors.contains_key("layers.0.self_attn.q_norm.weight"));
    assert!(w.tensors.contains_key("layers.0.mlp.gate_proj.weight"));
    assert!(w.tensors.contains_key("layers.0.input_layernorm.weight"));
}

#[test]
fn drop_attn_weights_frees_correct_byte_count() {
    let mut w = minimal_weights();
    // 5 attn tensors × (2×4 elements) × 4 bytes = 160 bytes
    let freed = w.drop_attn_weights();
    assert_eq!(freed, 5 * 2 * 4 * 4);
}

#[test]
fn drop_lm_head_zeroes_matrix_and_reports_freed() {
    let mut w = minimal_weights();
    let freed = w.drop_lm_head();
    assert_eq!(freed, 2 * 4 * 4, "freed should be elem_count × sizeof(f32)");
    assert_eq!(w.lm_head.shape(), &[0, 0]);
}

#[test]
fn drop_embed_zeroes_matrix_and_reports_freed() {
    let mut w = minimal_weights();
    let freed = w.drop_embed();
    assert_eq!(freed, 2 * 4 * 4);
    assert_eq!(w.embed.shape(), &[0, 0]);
}

#[test]
fn get_packed_bytes_from_raw_bytes() {
    let mut w = minimal_weights();
    w.raw_bytes
        .insert("experts.gate_up_proj".into(), vec![1u8, 2, 3, 4]);
    let bytes = w.get_packed_bytes("experts.gate_up_proj").unwrap();
    assert_eq!(bytes, &[1u8, 2, 3, 4]);
}

#[test]
fn get_packed_bytes_from_mmap_range_takes_precedence() {
    use std::io::Write;

    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("packed.bin");
    let mut file = std::fs::File::create(&path).unwrap();
    file.write_all(&[10u8, 11, 12, 13, 14, 15]).unwrap();
    file.flush().unwrap();
    drop(file);

    let file = std::fs::File::open(&path).unwrap();
    let mmap = unsafe { memmap2::Mmap::map(&file).unwrap() };
    let mut w = minimal_weights();
    w.raw_bytes.insert("tensor.key".into(), vec![1u8, 2, 3]);
    w.packed_mmaps.insert("packed.bin".into(), mmap);
    w.packed_byte_ranges
        .insert("tensor.key".into(), ("packed.bin".into(), 2, 3));

    assert_eq!(w.get_packed_bytes("tensor.key").unwrap(), &[12u8, 13, 14]);
}

#[test]
fn get_packed_bytes_out_of_bounds_mmap_range_returns_none() {
    use std::io::Write;

    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("packed.bin");
    let mut file = std::fs::File::create(&path).unwrap();
    file.write_all(&[10u8, 11, 12, 13]).unwrap();
    file.flush().unwrap();
    drop(file);

    let file = std::fs::File::open(&path).unwrap();
    let mmap = unsafe { memmap2::Mmap::map(&file).unwrap() };
    let mut w = minimal_weights();
    w.packed_mmaps.insert("packed.bin".into(), mmap);
    w.packed_byte_ranges
        .insert("tensor.key".into(), ("packed.bin".into(), 3, 4));

    assert!(w.get_packed_bytes("tensor.key").is_none());
}

#[test]
fn per_layer_ffn_bytes_detects_and_loads_entries() {
    let mut w = minimal_weights();
    w.raw_bytes.insert(
        larql_models::weights::per_layer_ffn_key(
            2,
            7,
            larql_models::weights::PER_LAYER_FFN_GATE_UP,
        ),
        vec![1u8, 2, 3],
    );
    w.raw_bytes.insert(
        larql_models::weights::per_layer_ffn_key(2, 7, larql_models::weights::PER_LAYER_FFN_DOWN),
        vec![4u8, 5],
    );
    w.packed_byte_ranges.insert(
        larql_models::weights::per_layer_ffn_key(
            9,
            1,
            larql_models::weights::PER_LAYER_FFN_GATE_UP,
        ),
        ("missing.bin".into(), 0, 1),
    );

    assert!(w.has_per_layer_ffn());
    let (gate_up, down) = w.get_layer_entry_bytes(2, 7).unwrap();
    assert_eq!(gate_up, &[1u8, 2, 3]);
    assert_eq!(down, &[4u8, 5]);
    assert!(w.get_layer_entry_bytes(2, 8).is_none());
    assert_eq!(
        larql_models::weights::per_layer_ffn_key(3, 4, larql_models::weights::PER_LAYER_FFN_DOWN,),
        "layers/3/4/down"
    );
}

#[test]
fn drop_ffn_weights_removes_raw_packed_expert_bytes() {
    let mut w = minimal_weights();
    w.raw_bytes
        .insert("layers.0.experts.gate_up_proj".into(), vec![1u8; 8]);
    w.raw_bytes
        .insert("layers.0.experts.down_proj".into(), vec![2u8; 4]);
    w.raw_bytes.insert("attention.cache".into(), vec![3u8; 2]);

    let freed = w.drop_ffn_weights();

    assert!(freed >= 12);
    assert!(!w.raw_bytes.contains_key("layers.0.experts.gate_up_proj"));
    assert!(!w.raw_bytes.contains_key("layers.0.experts.down_proj"));
    assert!(w.raw_bytes.contains_key("attention.cache"));
}

#[test]
fn drop_ffn_weights_releases_unreferenced_mmaps() {
    use std::io::Write;

    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("packed.bin");
    let mut file = std::fs::File::create(&path).unwrap();
    file.write_all(&[0u8; 16]).unwrap();
    file.flush().unwrap();
    drop(file);

    let file = std::fs::File::open(&path).unwrap();
    let mmap = unsafe { memmap2::Mmap::map(&file).unwrap() };
    let mut w = minimal_weights();
    w.packed_mmaps.insert("packed.bin".into(), mmap);
    w.packed_byte_ranges.insert(
        "layers.0.experts.gate_up_proj".into(),
        ("packed.bin".into(), 0, 8),
    );

    let freed = w.drop_ffn_weights();

    assert!(freed >= 8);
    assert!(w.packed_byte_ranges.is_empty());
    assert!(w.packed_mmaps.is_empty());
}

#[test]
fn get_packed_bytes_missing_key_returns_none() {
    let w = minimal_weights();
    assert!(w.get_packed_bytes("nonexistent.key").is_none());
}

#[test]
fn get_packed_bytes_mmap_range_missing_file_falls_through_to_raw() {
    // packed_byte_ranges points to a file not in packed_mmaps → falls through to raw_bytes.
    let mut w = minimal_weights();
    w.raw_bytes.insert("tensor.key".into(), vec![9u8, 8]);
    w.packed_byte_ranges
        .insert("tensor.key".into(), ("missing_file.bin".into(), 0, 2));
    // mmap file absent → fallback to raw_bytes
    let bytes = w.get_packed_bytes("tensor.key").unwrap();
    assert_eq!(bytes, &[9u8, 8]);
}

// ═══════════════════════════════════════════════════════════════
// Trait-default + OLMoE sweeps, duplicated from the in-crate unit
// tests on purpose.
//
// `cargo llvm-cov --package larql-models` (the CI coverage gate) merges
// this binary's instantiation of the crate with the lib-test binary's.
// A default method exercised only by the unit tests still reports
// uncovered lines from THIS binary's copy, and the per-file floor is
// computed on the merged union — so the public-API surface below has to
// walk the same paths, or `config.rs`/`olmoe.rs` sit under their floors
// with green tests.
// ═══════════════════════════════════════════════════════════════

/// OLMoE-1B-7B's real routing shape (64 experts, 8 per token, expert
/// width in plain `intermediate_size`).
fn olmoe() -> Box<dyn ModelArchitecture> {
    detect_from_json(&serde_json::json!({
        "model_type": "olmoe",
        "hidden_size": 2048,
        "intermediate_size": 1024,
        "num_hidden_layers": 16,
        "num_attention_heads": 16,
        "num_key_value_heads": 16,
        "num_experts": 64,
        "num_experts_per_tok": 8,
        "norm_topk_prob": false,
    }))
}

#[test]
fn olmoe_moe_shape_and_keys() {
    let a = olmoe();
    assert_eq!(a.family(), "olmoe");
    assert!(a.is_moe());
    assert_eq!(a.num_experts(), 64);
    assert_eq!(a.num_experts_per_token(), 8);
    // No `moe_intermediate_size` field: expert width comes from
    // `intermediate_size`, not an `unwrap_or(0)`.
    assert_eq!(a.moe_intermediate_size(), 1024);
    // `norm_topk_prob: false` keeps raw softmax probabilities.
    assert_eq!(
        a.expert_routing_policy(),
        larql_models::ExpertRoutingPolicy::SoftmaxThenSelect
    );
    let p = a.layer_prefix(2);
    assert_eq!(a.moe_router_key(2), Some(format!("{p}mlp.gate.weight")));
    assert_eq!(
        a.expert_ffn_gate_key(2, 5),
        Some(format!("{p}mlp.experts.5.gate_proj.weight"))
    );
    assert_eq!(
        a.expert_ffn_up_key(2, 5),
        Some(format!("{p}mlp.experts.5.up_proj.weight"))
    );
    assert_eq!(
        a.expert_ffn_down_key(2, 5),
        Some(format!("{p}mlp.experts.5.down_proj.weight"))
    );
    assert_eq!(
        a.attn_q_norm_key(2),
        Some(format!("{p}self_attn.q_norm.weight"))
    );
    assert_eq!(
        a.attn_k_norm_key(2),
        Some(format!("{p}self_attn.k_norm.weight"))
    );
}

#[test]
fn olmoe_without_experts_is_dense() {
    let a = detect_from_json(&serde_json::json!({
        "model_type": "olmoe",
        "hidden_size": 64,
        "intermediate_size": 128,
        "num_hidden_layers": 1,
        "num_attention_heads": 4,
    }));
    assert!(!a.is_moe());
    assert_eq!(a.num_experts_per_token(), 0);
    assert_eq!(a.moe_router_key(0), None);
    assert_eq!(a.expert_ffn_gate_key(0, 0), None);
    assert_eq!(a.expert_ffn_up_key(0, 0), None);
    assert_eq!(a.expert_ffn_down_key(0, 0), None);
}

/// Walk the `ModelArchitecture` defaults through a family that overrides
/// almost nothing, so this binary's instantiation of the default bodies
/// executes. Assertions mirror `config::tests` — see the header comment.
#[test]
fn trait_defaults_via_public_api() {
    let a = detect_from_json(&serde_json::json!({
        "model_type": "llama",
        "hidden_size": 64,
        "intermediate_size": 128,
        "num_hidden_layers": 2,
        "num_attention_heads": 4,
        "num_key_value_heads": 2,
        "head_dim": 16,
        "vocab_size": 32,
    }));
    // Dense-model MoE surface.
    assert!(!a.is_moe());
    assert!(!a.is_hybrid_moe());
    assert_eq!(a.num_experts(), 0);
    assert_eq!(a.num_shared_experts(), 0);
    assert_eq!(a.moe_intermediate_size(), 0);
    assert_eq!(a.moe_router_type(), "top_k_softmax");
    assert_eq!(a.expert_format(), ExpertFormat::PerExpert);
    assert_eq!(
        a.expert_gate_policy(),
        larql_models::ExpertGatePolicy::Gated
    );
    assert_eq!(
        a.expert_routing_policy(),
        larql_models::ExpertRoutingPolicy::SoftmaxThenSelect
    );
    for k in [
        a.moe_router_bias_key(0),
        a.moe_router_scale_key(0),
        a.moe_router_per_expert_scale_key(0),
        a.moe_router_norm_key(0),
        a.shared_expert_gate_key(0),
        a.shared_expert_up_key(0),
        a.shared_expert_down_key(0),
        a.packed_gate_up_blocks_key(0),
        a.packed_gate_up_scales_key(0),
        a.packed_gate_up_bias_key(0),
        a.packed_down_blocks_key(0),
        a.packed_down_scales_key(0),
        a.packed_down_bias_key(0),
        a.packed_experts_gate_up_key(0),
        a.packed_experts_down_key(0),
        a.moe_post_outer_norm_key(0),
        a.moe_post_ffn1_norm_key(0),
        a.moe_pre_experts_norm_key(0),
        a.moe_post_experts_norm_key(0),
        a.attn_sinks_key(0),
        a.fused_qkv_key(0),
        a.fused_qkv_bias_key(0),
        a.ffn_up_bias_key(0),
        a.ffn_down_bias_key(0),
        a.layer_scalar_key(0),
        a.mla_kv_a_key(0),
        a.mla_kv_b_key(0),
        a.mla_q_a_key(0),
        a.mla_q_b_key(0),
    ] {
        assert_eq!(k, None);
    }
    assert!(!a.moe_router_norm_parameter_free());
    assert_eq!(a.moe_router_input_scalar(), None);
    assert!(!a.moe_has_combined_output_norm());
    // MLA off means standard GQA.
    assert!(!a.uses_mla());
    assert_eq!(a.kv_lora_rank(), 0);
    assert_eq!(a.q_lora_rank(), 0);
    assert_eq!(a.mla_qk_nope_head_dim(), None);
    assert_eq!(a.mla_qk_rope_head_dim(), None);
    assert_eq!(a.mla_v_head_dim(), None);
    // Identity multipliers, no softcapping, no PLE, no rope scaling.
    assert_eq!(a.residual_multiplier(), 1.0);
    assert_eq!(a.attention_multiplier(), 1.0);
    assert_eq!(a.logits_scaling(), 1.0);
    assert_eq!(a.attn_logit_softcapping(), None);
    assert_eq!(a.final_logit_softcapping(), None);
    assert!(!a.has_per_layer_embeddings());
    assert_eq!(a.per_layer_embed_dim(), 0);
    assert_eq!(a.per_layer_embed_key(), None);
    assert_eq!(a.per_layer_model_projection_key(), None);
    assert_eq!(a.per_layer_projection_norm_key(), None);
    assert_eq!(a.per_layer_input_gate_key(0), None);
    assert_eq!(a.per_layer_projection_key(0), None);
    assert_eq!(a.post_per_layer_input_norm_key(0), None);
    assert_eq!(a.rope_position_divisor_for_layer(0), 1.0);
    // llama overrides llama3_rope_scaling, so only walk it — the default's
    // None is asserted by `config::tests` on an override-free arch.
    let _ = a.llama3_rope_scaling();
    assert!(a.multimodal().is_none());
    assert_eq!(a.kv_shared_source_layer(0), None);
    // Scale falls back to head_dim^-0.5 without query_pre_attn_scalar.
    assert_eq!(a.attention_scale_for_layer(0), (16.0f64).powf(-0.5));
    // The markov-residual precondition holds for the default surface.
    assert!(a.kv_recomputable_from_residuals());
}
