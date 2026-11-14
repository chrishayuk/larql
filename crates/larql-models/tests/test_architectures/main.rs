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

// GPT-OSS architecture

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

// Mixtral — PerExpert format comparison

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

// Dense model — no MoE

// `output_head_reuses_embedding` — tied-embedding synthesis

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

// Config validation

fn validation_fields(arch: &dyn ModelArchitecture) -> Vec<&'static str> {
    arch.validate()
        .expect_err("config should fail validation")
        .into_iter()
        .map(|error| error.field)
        .collect()
}

// Cross-architecture key comparison

// Norm-epsilon fallback — a per-family fact, pinned per family

// ModelWeights: drop_ffn_weights

// Gemma 4 — per-layer geometry, partial RoPE, V-norm, KV sharing

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

// Gemma 2 — softcapping, QK norm with +1 offset

fn gemma2_arch() -> Box<dyn ModelArchitecture> {
    detect_from_json(&serde_json::json!({
        "model_type": "gemma2",
        "hidden_size": 2304, "num_hidden_layers": 26, "intermediate_size": 9216,
        "num_attention_heads": 8, "num_key_value_heads": 4, "head_dim": 256,
        "query_pre_attn_scalar": 256.0,
        "attn_logit_softcapping": 50.0, "final_logit_softcapping": 30.0
    }))
}

// Gemma 3 — sliding window, dual RoPE, QK norm offset

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

// Mistral — sliding window, Llama-compatible keys

// Qwen — attention bias, QK norm keys

fn qwen_arch() -> Box<dyn ModelArchitecture> {
    detect_from_json(&serde_json::json!({
        "model_type": "qwen2",
        "hidden_size": 2048, "num_hidden_layers": 24, "intermediate_size": 5504,
        "num_attention_heads": 16, "num_key_value_heads": 2
    }))
}

// DeepSeek — MoE + MLA

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

// Granite — scaling multipliers

fn granite_arch() -> Box<dyn ModelArchitecture> {
    detect_from_json(&serde_json::json!({
        "model_type": "granite",
        "hidden_size": 2048, "num_hidden_layers": 40, "intermediate_size": 8192,
        "num_attention_heads": 32, "num_key_value_heads": 8,
        "embedding_multiplier": 12.0, "residual_multiplier": 0.22,
        "attention_multiplier": 0.22, "logits_scaling": 0.13
    }))
}

// StarCoder2 — LayerNorm, GELU, bias, non-gated FFN, c_fc/c_proj keys

fn starcoder2_arch() -> Box<dyn ModelArchitecture> {
    detect_from_json(&serde_json::json!({
        "model_type": "starcoder2",
        "hidden_size": 3072, "num_hidden_layers": 30, "intermediate_size": 12288,
        "num_attention_heads": 24, "num_key_value_heads": 2
    }))
}

// Generic fallback — safe defaults for unknown models

// Cross-architecture: default multipliers are 1.0 for non-Granite

// Q4 round-trip: quantize then dequantize

// ModelWeights — drop_attn_weights, drop_lm_head, drop_embed, get_packed_bytes

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

mod config_validation;
mod deepseek_moe_mla;
mod dense_model_no_moe;
mod gemma3_text_flat_interleave;
mod gemma_4_per_layer_geometry_partial_rope;
mod gpt_oss_architecture;
mod modelweights_drop_attn_weights_drop_lm_h;
mod modelweights_drop_ffn_weights;
mod norm_epsilon_fallback_a_per_family_fact;
mod qwen_attention_bias_qk_norm_keys;
mod starcoder2_layernorm_gelu_bias_non_gated;
