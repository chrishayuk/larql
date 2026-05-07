//! Auto-detect model architecture from config.json.

use std::path::Path;

use crate::architectures::deepseek::DeepSeekArch;
use crate::architectures::gemma2::Gemma2Arch;
use crate::architectures::gemma3::Gemma3Arch;
use crate::architectures::gemma4::Gemma4Arch;
use crate::architectures::generic::GenericArch;
use crate::architectures::gpt_oss::GptOssArch;
use crate::architectures::granite::GraniteArch;
use crate::architectures::llama::LlamaArch;
use crate::architectures::mistral::MistralArch;
use crate::architectures::mixtral::MixtralArch;
use crate::architectures::qwen::QwenArch;
use crate::architectures::starcoder2::StarCoder2Arch;
use crate::architectures::tinymodel::TinyModelArch;
use crate::config::{ModelArchitecture, ModelConfig, RopeScaling};
use crate::validation::ConfigValidationError;

/// Error from model detection/config parsing.
#[derive(Debug, thiserror::Error)]
pub enum ModelError {
    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),
    #[error("JSON parse error: {0}")]
    Json(#[from] serde_json::Error),
    #[error("parse error: {0}")]
    Parse(String),
    #[error("unsupported dtype: {0}")]
    UnsupportedDtype(String),
    #[error("missing tensor: {0}")]
    MissingTensor(String),
    #[error("not a directory: {0}")]
    NotADirectory(std::path::PathBuf),
    #[error("no safetensors files in {0}")]
    NoSafetensors(std::path::PathBuf),
    #[error("config validation failed: {0:?}")]
    ConfigValidation(Vec<ConfigValidationError>),
}

/// Read config.json from a model directory and return the architecture.
pub fn detect_architecture(model_dir: &Path) -> Result<Box<dyn ModelArchitecture>, ModelError> {
    let config_json = read_config_json(model_dir)?;
    Ok(detect_from_json(&config_json))
}

/// Read config.json from a model directory, detect the architecture, and validate it.
pub fn detect_architecture_validated(
    model_dir: &Path,
) -> Result<Box<dyn ModelArchitecture>, ModelError> {
    let arch = detect_architecture(model_dir)?;
    validate_detected_architecture(arch)
}

fn read_config_json(model_dir: &Path) -> Result<serde_json::Value, ModelError> {
    let config_path = model_dir.join("config.json");
    if config_path.exists() {
        let text = std::fs::read_to_string(&config_path)?;
        Ok(serde_json::from_str::<serde_json::Value>(&text)?)
    } else {
        Ok(serde_json::json!({}))
    }
}

/// Detect architecture from an already-parsed config.json value.
pub fn detect_from_json(config: &serde_json::Value) -> Box<dyn ModelArchitecture> {
    let model_config = parse_model_config(config);
    let model_type = model_config.model_type.as_str();

    match model_type {
        // Gemma family
        t if t.starts_with("gemma4") => Box::new(Gemma4Arch::from_config(model_config)),
        t if t.starts_with("gemma3") => Box::new(Gemma3Arch::from_config(model_config)),
        t if t.starts_with("gemma2") || t == "gemma" => {
            Box::new(Gemma2Arch::from_config(model_config))
        }
        // Llama family
        t if t.starts_with("llama") => Box::new(LlamaArch::from_config(model_config)),
        // Mistral (dense)
        "mistral" => Box::new(MistralArch::from_config(model_config)),
        // Mixtral (MoE) — block_sparse_moe pattern
        "mixtral" => Box::new(MixtralArch::from_config(model_config)),
        // GPT-OSS (MoE, MXFP4 packed experts)
        "gpt_oss" => Box::new(GptOssArch::from_config(model_config)),
        // Qwen family (dense and MoE share same keys)
        t if t.starts_with("qwen") => Box::new(QwenArch::from_config(model_config)),
        // DeepSeek family (MoE + MLA)
        t if t.starts_with("deepseek") => Box::new(DeepSeekArch::from_config(model_config)),
        // StarCoder 2
        "starcoder2" => Box::new(StarCoder2Arch::from_config(model_config)),
        // Granite family (dense and MoE share same base keys)
        t if t.starts_with("granite") => Box::new(GraniteArch::from_config(model_config)),
        // TinyModel — research-scale decoder used for LARQL compile/walk work
        "tinymodel" => Box::new(TinyModelArch::from_config(model_config)),
        // Unknown — generic fallback
        _ => Box::new(GenericArch::from_config(model_config)),
    }
}

/// Detect architecture from an already-parsed config.json value and validate it.
pub fn detect_from_json_validated(
    config: &serde_json::Value,
) -> Result<Box<dyn ModelArchitecture>, ModelError> {
    let arch = detect_from_json(config);
    validate_detected_architecture(arch)
}

pub(crate) fn validate_detected_architecture(
    arch: Box<dyn ModelArchitecture>,
) -> Result<Box<dyn ModelArchitecture>, ModelError> {
    match arch.validate() {
        Ok(()) => Ok(arch),
        Err(errors) => Err(ModelError::ConfigValidation(errors)),
    }
}

// ── RoPE base defaults ───────────────────────────────────────────────────────
/// Default RoPE theta for Gemma family models.
const ROPE_BASE_GEMMA: f64 = 1_000_000.0;
/// Default RoPE theta for all other model families.
const ROPE_BASE_DEFAULT: f64 = 10_000.0;

// ── Config field name aliases ────────────────────────────────────────────────
// Different model families use different JSON keys for the same concept.
// Ordering is priority: first match wins.

/// Total routed expert count: DeepSeek, Qwen MoE, Mixtral variants.
const NUM_EXPERTS_KEYS: &[&str] = &["n_routed_experts", "num_local_experts", "num_experts"];

/// Experts activated per token: llama.cpp / HF spelling variants.
const NUM_EXPERTS_PER_TOK_KEYS: &[&str] = &["num_experts_per_tok", "num_experts_per_token"];

/// Return the first `u64` found under any of `keys` in `config`.
fn field_u64(config: &serde_json::Value, keys: &[&str]) -> Option<u64> {
    keys.iter().find_map(|k| config[k].as_u64())
}

/// Parse ModelConfig from a config.json value.
/// Handles both top-level and nested text_config (multimodal models).
fn parse_model_config(config: &serde_json::Value) -> ModelConfig {
    let text_config = config.get("text_config").unwrap_or(config);

    // Detect model_type from text_config or top level.
    let model_type = text_config["model_type"]
        .as_str()
        .or_else(|| config["model_type"].as_str())
        .unwrap_or("")
        .to_string();

    // Pick defaults based on model type.
    let is_gemma = model_type.starts_with("gemma");
    let rope_default = if is_gemma {
        ROPE_BASE_GEMMA
    } else {
        ROPE_BASE_DEFAULT
    };

    let num_layers = text_config["num_hidden_layers"].as_u64().unwrap_or(32) as usize;
    let hidden_size = text_config["hidden_size"].as_u64().unwrap_or(2048) as usize;
    let intermediate_size = text_config["intermediate_size"].as_u64().unwrap_or(8192) as usize;
    // Gemma3 HF configs omit num_attention_heads, head_dim, num_key_value_heads
    // from text_config — they're architecture-class defaults in transformers.
    let default_head_dim: usize = if is_gemma { 256 } else { 0 };
    let num_q_heads = text_config["num_attention_heads"].as_u64().unwrap_or(8) as usize;
    // head_dim: explicit config value, Gemma default (256), or compute from hidden/heads.
    let head_dim = text_config["head_dim"]
        .as_u64()
        .map(|v| v as usize)
        .unwrap_or(if default_head_dim > 0 {
            default_head_dim
        } else {
            hidden_size.checked_div(num_q_heads).unwrap_or(0)
        });
    let num_kv_heads = text_config["num_key_value_heads"].as_u64().unwrap_or(4) as usize;
    // RoPE base: check rope_parameters.full_attention.rope_theta (Gemma 4),
    // then top-level rope_theta, then default.
    let rope_params = text_config.get("rope_parameters");
    let rope_base = rope_params
        .and_then(|rp| rp.get("full_attention"))
        .and_then(|fa| fa["rope_theta"].as_f64())
        .or_else(|| text_config["rope_theta"].as_f64())
        .unwrap_or(rope_default);
    // Local RoPE base for sliding window layers: check rope_parameters.sliding_attention,
    // then rope_local_base_freq.
    let rope_local_base = rope_params
        .and_then(|rp| rp.get("sliding_attention"))
        .and_then(|sa| sa["rope_theta"].as_f64())
        .or_else(|| text_config["rope_local_base_freq"].as_f64());
    let vocab_size = text_config["vocab_size"].as_u64().map(|v| v as usize);
    let sliding_window = text_config["sliding_window"].as_u64().map(|v| v as usize);

    // MoE fields
    let num_experts = field_u64(text_config, NUM_EXPERTS_KEYS).map(|v| v as usize);
    let num_experts_per_token =
        field_u64(text_config, NUM_EXPERTS_PER_TOK_KEYS).map(|v| v as usize);
    let num_shared_experts = text_config["n_shared_experts"].as_u64().map(|v| v as usize);
    // Gemma 4 A4B hybrid MoE fields
    let enable_moe_block = text_config["enable_moe_block"].as_bool().unwrap_or(false);
    let top_k_experts = text_config["top_k_experts"].as_u64().map(|v| v as usize);
    let moe_intermediate_size = text_config["moe_intermediate_size"]
        .as_u64()
        .map(|v| v as usize);

    // MLA fields
    let kv_lora_rank = text_config["kv_lora_rank"].as_u64().map(|v| v as usize);
    let q_lora_rank = text_config["q_lora_rank"].as_u64().map(|v| v as usize);

    // RoPE scaling
    let rope_scaling = text_config.get("rope_scaling").and_then(|rs| {
        // HF uses "type" for most models, but Llama 3.1+ uses "rope_type"
        let scaling_type = rs
            .get("type")
            .or_else(|| rs.get("rope_type"))
            .and_then(|v| v.as_str())?
            .to_string();
        let factor = rs.get("factor")?.as_f64()?;
        Some(RopeScaling {
            scaling_type,
            factor,
        })
    });

    // Softcapping and attention scale
    let attn_logit_softcapping = text_config["attn_logit_softcapping"].as_f64();
    let final_logit_softcapping = text_config["final_logit_softcapping"].as_f64();
    let query_pre_attn_scalar = text_config["query_pre_attn_scalar"].as_f64();

    // Granite-style scaling multipliers
    let embedding_multiplier = text_config["embedding_multiplier"].as_f64();
    let residual_multiplier = text_config["residual_multiplier"].as_f64();
    let attention_multiplier = text_config["attention_multiplier"].as_f64();
    let logits_scaling = text_config["logits_scaling"].as_f64();

    // Per-layer attention geometry (Gemma 4 style)
    let global_head_dim = text_config["global_head_dim"].as_u64().map(|v| v as usize);
    let num_global_kv_heads = text_config["num_global_key_value_heads"]
        .as_u64()
        .map(|v| v as usize);
    // Partial rotary factor: check rope_parameters.full_attention first (Gemma 4),
    // then top-level partial_rotary_factor.
    let partial_rotary_factor = rope_params
        .and_then(|rp| rp.get("full_attention"))
        .and_then(|fa| fa["partial_rotary_factor"].as_f64())
        .or_else(|| text_config["partial_rotary_factor"].as_f64());
    // Sliding window pattern: explicit sliding_window_pattern field, or infer later.
    let sliding_window_pattern = text_config["sliding_window_pattern"]
        .as_u64()
        .map(|v| v as usize);
    // Explicit per-layer type array (Gemma 4: ["sliding_attention", "full_attention", ...])
    let layer_types = text_config.get("layer_types").and_then(|lt| {
        lt.as_array().map(|arr| {
            arr.iter()
                .filter_map(|v| v.as_str().map(|s| s.to_string()))
                .collect()
        })
    });
    // K=V sharing flag
    let attention_k_eq_v = text_config["attention_k_eq_v"].as_bool().unwrap_or(false);
    // KV sharing across layers
    let num_kv_shared_layers = text_config["num_kv_shared_layers"]
        .as_u64()
        .map(|v| v as usize)
        .filter(|&v| v > 0);
    // Per-layer embedding dimension (PLE)
    let per_layer_embed_dim = text_config["hidden_size_per_layer_input"]
        .as_u64()
        .map(|v| v as usize)
        .filter(|&v| v > 0);

    ModelConfig {
        model_type,
        num_layers,
        hidden_size,
        intermediate_size,
        head_dim,
        num_q_heads,
        num_kv_heads,
        vocab_size,
        rope_base,
        rope_local_base,
        sliding_window,
        num_experts,
        num_experts_per_token,
        num_shared_experts,
        kv_lora_rank,
        q_lora_rank,
        rope_scaling,
        attn_logit_softcapping,
        final_logit_softcapping,
        query_pre_attn_scalar,
        embedding_multiplier,
        residual_multiplier,
        attention_multiplier,
        logits_scaling,
        global_head_dim,
        num_global_kv_heads,
        partial_rotary_factor,
        sliding_window_pattern,
        layer_types,
        attention_k_eq_v,
        per_layer_embed_dim,
        num_kv_shared_layers,
        enable_moe_block,
        top_k_experts,
        moe_intermediate_size,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_detect_gemma3() {
        let config = serde_json::json!({
            "model_type": "gemma3",
            "text_config": {
                "model_type": "gemma3_text",
                "hidden_size": 2560,
                "head_dim": 256,
                "num_hidden_layers": 34,
                "num_attention_heads": 8,
                "intermediate_size": 10240,
                "sliding_window": 1024
            }
        });

        let arch = detect_from_json(&config);
        assert_eq!(arch.family(), "gemma3");
        assert_eq!(arch.config().num_layers, 34);
        assert_eq!(arch.config().hidden_size, 2560);
        assert_eq!(arch.config().rope_base, 1_000_000.0);
        assert_eq!(arch.norm_weight_offset(), 1.0);
        assert_eq!(arch.embed_scale(), (2560.0f32).sqrt());
        assert!(arch.has_post_norms());
        assert!(arch.attn_q_norm_key(0).is_some());

        // Sliding window: layer 4 is sliding, layer 5 is full
        assert!(arch.is_sliding_window_layer(4));
        assert!(!arch.is_sliding_window_layer(5));
    }

    #[test]
    fn test_detect_llama() {
        let config = serde_json::json!({
            "model_type": "llama",
            "hidden_size": 4096,
            "num_hidden_layers": 32
        });

        let arch = detect_from_json(&config);
        assert_eq!(arch.family(), "llama");
        assert_eq!(arch.config().hidden_size, 4096);
        assert_eq!(arch.config().rope_base, 10_000.0);
        assert_eq!(arch.norm_weight_offset(), 0.0);
        assert_eq!(arch.embed_scale(), 1.0);
        assert!(!arch.has_post_norms());
        assert!(arch.attn_q_norm_key(0).is_none());
    }

    #[test]
    fn test_detect_tinymodel() {
        let config = serde_json::json!({
            "model_type": "tinymodel",
            "hidden_size": 512,
            "num_hidden_layers": 20,
            "intermediate_size": 2048,
            "num_attention_heads": 8,
            "num_key_value_heads": 4,
            "vocab_size": 71261,
            "max_position_embeddings": 256
        });

        let arch = detect_from_json(&config);
        assert_eq!(arch.family(), "tinymodel");
        assert_eq!(arch.config().hidden_size, 512);
        assert_eq!(arch.config().num_layers, 20);
        assert_eq!(arch.config().rope_base, 10_000.0);
        assert_eq!(arch.embed_scale(), (512.0_f32).sqrt());
        assert_eq!(arch.embed_key(), "embed.weight");
        assert_eq!(arch.final_norm_key(), "norm.weight");
        assert_eq!(arch.attn_q_key(5), "layers.5.attn.q_proj.weight");
        assert_eq!(arch.ffn_gate_key(5), "layers.5.ffn.gate.weight");
        assert_eq!(arch.ffn_down_key(5), "layers.5.ffn.down.weight");
        assert_eq!(arch.input_layernorm_key(5), "layers.5.attn_norm.weight");
        assert_eq!(
            arch.post_attention_layernorm_key(5),
            "layers.5.ffn_norm.weight"
        );
        assert_eq!(arch.key_prefixes_to_strip(), &[] as &[&str]);
        assert!(!arch.has_post_norms());
    }

    #[test]
    fn test_tinymodel_full_key_coverage() {
        let config = serde_json::json!({
            "model_type": "tinymodel",
            "hidden_size": 512,
            "num_hidden_layers": 20,
            "intermediate_size": 2048,
            "num_attention_heads": 8,
            "num_key_value_heads": 4,
        });
        let arch = detect_from_json(&config);

        // Complete attention key set
        assert_eq!(arch.attn_q_key(7), "layers.7.attn.q_proj.weight");
        assert_eq!(arch.attn_k_key(7), "layers.7.attn.k_proj.weight");
        assert_eq!(arch.attn_v_key(7), "layers.7.attn.v_proj.weight");
        assert_eq!(arch.attn_o_key(7), "layers.7.attn.o_proj.weight");

        // Complete FFN key set
        assert_eq!(arch.ffn_gate_key(7), "layers.7.ffn.gate.weight");
        assert_eq!(arch.ffn_up_key(7), "layers.7.ffn.up.weight");
        assert_eq!(arch.ffn_down_key(7), "layers.7.ffn.down.weight");

        // Not MoE, not MLA, no QK norm
        assert!(!arch.is_moe());
        assert!(!arch.uses_mla());
        assert!(arch.attn_q_norm_key(0).is_none());
        assert!(arch.attn_k_norm_key(0).is_none());
    }

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
    fn test_detect_mistral() {
        let config = serde_json::json!({
            "model_type": "mistral",
            "hidden_size": 4096,
            "num_hidden_layers": 32
        });

        let arch = detect_from_json(&config);
        assert_eq!(arch.family(), "mistral");
    }

    #[test]
    fn test_detect_qwen2() {
        let config = serde_json::json!({
            "model_type": "qwen2",
            "hidden_size": 4096,
            "num_hidden_layers": 32
        });

        let arch = detect_from_json(&config);
        assert_eq!(arch.family(), "qwen2");
    }

    #[test]
    fn test_detect_qwen3() {
        let config = serde_json::json!({
            "model_type": "qwen3",
            "hidden_size": 2048,
            "num_hidden_layers": 28
        });

        let arch = detect_from_json(&config);
        assert_eq!(arch.family(), "qwen3");
        assert!(!arch.is_moe());
    }

    #[test]
    fn test_detect_qwen3_moe_30b() {
        // Matches Qwen/Qwen3-30B-A3B config.json
        let config = serde_json::json!({
            "model_type": "qwen3_moe",
            "hidden_size": 2048,
            "intermediate_size": 6144,
            "moe_intermediate_size": 768,
            "num_hidden_layers": 48,
            "num_attention_heads": 32,
            "num_key_value_heads": 8,
            "num_experts": 128,
            "num_experts_per_tok": 8
        });

        let arch = detect_from_json(&config);
        assert!(arch.is_moe());
        assert!(!arch.is_hybrid_moe());
        assert_eq!(arch.num_experts(), 128);
        assert_eq!(arch.num_experts_per_token(), 8);
        assert_eq!(arch.moe_intermediate_size(), 768);
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
    fn test_detect_unknown_defaults_to_generic() {
        let config = serde_json::json!({
            "model_type": "some_unknown_model",
            "hidden_size": 2048,
            "num_hidden_layers": 24
        });

        let arch = detect_from_json(&config);
        assert_eq!(arch.family(), "generic");
    }

    #[test]
    fn test_tensor_keys() {
        let config = serde_json::json!({"model_type": "gemma3_text"});
        let arch = detect_from_json(&config);

        assert_eq!(arch.attn_q_key(5), "layers.5.self_attn.q_proj.weight");
        assert_eq!(arch.ffn_gate_key(10), "layers.10.mlp.gate_proj.weight");
        assert_eq!(
            arch.input_layernorm_key(0),
            "layers.0.input_layernorm.weight"
        );
        assert_eq!(arch.final_norm_key(), "norm.weight");
        assert_eq!(arch.embed_key(), "embed_tokens.weight");

        assert_eq!(
            arch.attn_q_norm_key(3),
            Some("layers.3.self_attn.q_norm.weight".to_string())
        );
    }

    #[test]
    fn test_detect_llama2() {
        // Real Llama 2 7B config — no head_dim, no rope_theta, no GQA
        let config = serde_json::json!({
            "model_type": "llama",
            "hidden_size": 4096,
            "intermediate_size": 11008,
            "num_hidden_layers": 32,
            "num_attention_heads": 32,
            "num_key_value_heads": 32,
            "vocab_size": 32000
        });

        let arch = detect_from_json(&config);
        assert_eq!(arch.family(), "llama");
        assert_eq!(arch.config().num_layers, 32);
        assert_eq!(arch.config().hidden_size, 4096);
        assert_eq!(arch.config().num_q_heads, 32);
        assert_eq!(arch.config().num_kv_heads, 32); // no GQA in Llama 2
                                                    // head_dim computed: 4096 / 32 = 128
        assert_eq!(arch.config().head_dim, 128);
        // rope_theta absent → defaults to 10000
        assert_eq!(arch.config().rope_base, 10_000.0);
        assert!(!arch.is_moe());
        assert!(!arch.uses_mla());

        // Standard tensor keys
        assert_eq!(arch.attn_q_key(0), "layers.0.self_attn.q_proj.weight");
        assert_eq!(arch.ffn_gate_key(5), "layers.5.mlp.gate_proj.weight");
        assert_eq!(
            arch.input_layernorm_key(0),
            "layers.0.input_layernorm.weight"
        );
        assert_eq!(
            arch.post_attention_layernorm_key(0),
            "layers.0.post_attention_layernorm.weight"
        );
        assert_eq!(arch.embed_key(), "embed_tokens.weight");
        assert_eq!(arch.final_norm_key(), "norm.weight");
    }

    #[test]
    fn test_detect_llama3() {
        // Real Llama 3 8B config — no head_dim, GQA (8 KV heads), higher rope_theta
        let config = serde_json::json!({
            "model_type": "llama",
            "hidden_size": 4096,
            "intermediate_size": 14336,
            "num_hidden_layers": 32,
            "num_attention_heads": 32,
            "num_key_value_heads": 8,
            "vocab_size": 128256,
            "rope_theta": 500000.0
        });

        let arch = detect_from_json(&config);
        assert_eq!(arch.family(), "llama");
        assert_eq!(arch.config().num_kv_heads, 8); // GQA in Llama 3
        assert_eq!(arch.config().head_dim, 128); // computed: 4096/32
        assert_eq!(arch.config().rope_base, 500_000.0);
        assert_eq!(arch.config().vocab_size, Some(128256));
        assert!(arch.rope_scaling_type().is_none()); // no scaling in base Llama 3
    }

    #[test]
    fn test_detect_llama31() {
        // Real Llama 3.1 8B config — uses "rope_type" instead of "type"
        let config = serde_json::json!({
            "model_type": "llama",
            "hidden_size": 4096,
            "intermediate_size": 14336,
            "num_hidden_layers": 32,
            "num_attention_heads": 32,
            "num_key_value_heads": 8,
            "vocab_size": 128256,
            "rope_theta": 500000.0,
            "rope_scaling": {
                "rope_type": "llama3",
                "factor": 8.0
            }
        });

        let arch = detect_from_json(&config);
        assert_eq!(arch.family(), "llama");
        assert_eq!(arch.rope_scaling_type(), Some("llama3"));
        assert_eq!(arch.rope_scaling_factor(), 8.0);
    }

    #[test]
    fn test_detect_mistral_7b() {
        // Real Mistral 7B config — no head_dim, GQA, sliding window
        let config = serde_json::json!({
            "model_type": "mistral",
            "hidden_size": 4096,
            "intermediate_size": 14336,
            "num_hidden_layers": 32,
            "num_attention_heads": 32,
            "num_key_value_heads": 8,
            "sliding_window": 4096
        });

        let arch = detect_from_json(&config);
        assert_eq!(arch.family(), "mistral");
        assert_eq!(arch.config().num_kv_heads, 8);
        assert_eq!(arch.config().head_dim, 128); // computed: 4096/32
        assert_eq!(arch.sliding_window_size(), Some(4096));
    }

    #[test]
    fn test_detect_deepseek_v2() {
        let config = serde_json::json!({
            "model_type": "deepseek_v2",
            "hidden_size": 5120,
            "intermediate_size": 12288,
            "num_hidden_layers": 60,
            "num_attention_heads": 128,
            "num_key_value_heads": 128,
            "head_dim": 128,
            "n_routed_experts": 160,
            "num_experts_per_tok": 6,
            "n_shared_experts": 2,
            "kv_lora_rank": 512,
            "q_lora_rank": 1536,
            "rope_scaling": {
                "type": "yarn",
                "factor": 40.0
            }
        });

        let arch = detect_from_json(&config);
        assert_eq!(arch.family(), "deepseek");

        // MoE
        assert!(arch.is_moe());
        assert_eq!(arch.num_experts(), 160);
        assert_eq!(arch.num_experts_per_token(), 6);
        assert_eq!(arch.num_shared_experts(), 2);

        // MoE tensor keys
        assert_eq!(
            arch.moe_router_key(0),
            Some("layers.0.mlp.gate.weight".to_string())
        );
        assert_eq!(
            arch.expert_ffn_gate_key(5, 3),
            Some("layers.5.mlp.experts.3.gate_proj.weight".to_string())
        );
        assert_eq!(
            arch.shared_expert_down_key(10),
            Some("layers.10.mlp.shared_experts.down_proj.weight".to_string())
        );

        // MLA
        assert!(arch.uses_mla());
        assert_eq!(arch.kv_lora_rank(), 512);
        assert_eq!(arch.q_lora_rank(), 1536);
        assert_eq!(
            arch.mla_kv_a_key(0),
            Some("layers.0.self_attn.kv_a_proj_with_mqa.weight".to_string())
        );
        assert_eq!(
            arch.mla_q_b_key(5),
            Some("layers.5.self_attn.q_b_proj.weight".to_string())
        );

        // RoPE
        assert_eq!(arch.rope_scaling_type(), Some("yarn"));
        assert_eq!(arch.rope_scaling_factor(), 40.0);
    }

    #[test]
    fn test_detect_deepseek_v3() {
        let config = serde_json::json!({
            "model_type": "deepseek_v3",
            "hidden_size": 7168,
            "num_hidden_layers": 61,
            "n_routed_experts": 256,
            "num_experts_per_tok": 8,
            "n_shared_experts": 1,
            "kv_lora_rank": 512,
            "q_lora_rank": 1536
        });

        let arch = detect_from_json(&config);
        assert_eq!(arch.family(), "deepseek");
        assert!(arch.is_moe());
        assert_eq!(arch.num_experts(), 256);
        assert_eq!(arch.num_experts_per_token(), 8);
        assert_eq!(arch.num_shared_experts(), 1);
    }

    #[test]
    fn test_non_moe_model_defaults() {
        let config = serde_json::json!({
            "model_type": "llama",
            "hidden_size": 4096,
            "num_hidden_layers": 32
        });

        let arch = detect_from_json(&config);
        assert!(!arch.is_moe());
        assert_eq!(arch.num_experts(), 0);
        assert!(!arch.uses_mla());
        assert_eq!(arch.kv_lora_rank(), 0);
        assert!(arch.moe_router_key(0).is_none());
        assert!(arch.mla_kv_a_key(0).is_none());
        assert!(arch.rope_scaling_type().is_none());
        assert_eq!(arch.rope_scaling_factor(), 1.0);
    }

    // ── Tests against real HuggingFace configs ──

    #[test]
    fn test_real_llama32_3b() {
        // Exact config from meta-llama/Llama-3.2-3B-Instruct
        let config = serde_json::json!({
            "model_type": "llama",
            "hidden_size": 3072,
            "intermediate_size": 8192,
            "num_hidden_layers": 28,
            "num_attention_heads": 24,
            "num_key_value_heads": 8,
            "head_dim": 128,
            "vocab_size": 128256,
            "rope_theta": 500000.0,
            "rope_scaling": {
                "factor": 32.0,
                "high_freq_factor": 4.0,
                "low_freq_factor": 1.0,
                "original_max_position_embeddings": 8192,
                "rope_type": "llama3"
            }
        });

        let arch = detect_from_json(&config);
        assert_eq!(arch.family(), "llama");
        assert_eq!(arch.config().hidden_size, 3072);
        assert_eq!(arch.config().head_dim, 128);
        assert_eq!(arch.config().num_q_heads, 24);
        assert_eq!(arch.config().num_kv_heads, 8);
        assert_eq!(arch.config().num_layers, 28);
        assert_eq!(arch.config().rope_base, 500_000.0);
        assert_eq!(arch.rope_scaling_type(), Some("llama3"));
        assert_eq!(arch.rope_scaling_factor(), 32.0);
    }

    #[test]
    fn test_real_llama32_1b() {
        // Exact config from meta-llama/Llama-3.2-1B — head_dim=64 (not 128!)
        let config = serde_json::json!({
            "model_type": "llama",
            "hidden_size": 2048,
            "intermediate_size": 8192,
            "num_hidden_layers": 16,
            "num_attention_heads": 32,
            "num_key_value_heads": 8,
            "head_dim": 64,
            "vocab_size": 128256,
            "rope_theta": 500000.0,
            "rope_scaling": {
                "factor": 32.0,
                "rope_type": "llama3"
            }
        });

        let arch = detect_from_json(&config);
        assert_eq!(arch.family(), "llama");
        assert_eq!(arch.config().head_dim, 64); // explicit, not computed
        assert_eq!(arch.config().num_q_heads, 32);
        // Without explicit head_dim, compute would give 2048/32=64 — same result
        assert_eq!(arch.rope_scaling_type(), Some("llama3"));
    }

    #[test]
    fn test_real_mistral_7b_v03() {
        // Exact config from mistralai/Mistral-7B-Instruct-v0.3 — head_dim null
        let config = serde_json::json!({
            "model_type": "mistral",
            "hidden_size": 4096,
            "intermediate_size": 14336,
            "num_hidden_layers": 32,
            "num_attention_heads": 32,
            "num_key_value_heads": 8,
            "head_dim": null,
            "vocab_size": 32768,
            "rope_theta": 1000000.0,
            "sliding_window": null
        });

        let arch = detect_from_json(&config);
        assert_eq!(arch.family(), "mistral");
        assert_eq!(arch.config().head_dim, 128); // computed: 4096/32
        assert_eq!(arch.config().rope_base, 1_000_000.0);
        assert!(arch.sliding_window_size().is_none());
    }

    #[test]
    fn test_real_tinyllama() {
        // Exact config from TinyLlama/TinyLlama-1.1B-Chat-v1.0
        let config = serde_json::json!({
            "model_type": "llama",
            "hidden_size": 2048,
            "intermediate_size": 5632,
            "num_hidden_layers": 22,
            "num_attention_heads": 32,
            "num_key_value_heads": 4,
            "vocab_size": 32000,
            "rope_theta": 10000.0
        });

        let arch = detect_from_json(&config);
        assert_eq!(arch.family(), "llama");
        assert_eq!(arch.config().head_dim, 64); // computed: 2048/32
        assert_eq!(arch.config().num_kv_heads, 4);
        assert_eq!(arch.config().rope_base, 10_000.0);
    }

    #[test]
    fn test_real_mixtral_8x7b() {
        // Exact config from mistralai/Mixtral-8x7B-Instruct-v0.1
        let config = serde_json::json!({
            "model_type": "mixtral",
            "hidden_size": 4096,
            "intermediate_size": 14336,
            "num_hidden_layers": 32,
            "num_attention_heads": 32,
            "num_key_value_heads": 8,
            "vocab_size": 32000,
            "rope_theta": 1000000.0,
            "num_local_experts": 8,
            "num_experts_per_tok": 2
        });

        let arch = detect_from_json(&config);
        assert_eq!(arch.family(), "mixtral");
        assert!(arch.is_moe());
        assert_eq!(arch.num_experts(), 8);
        assert_eq!(arch.num_experts_per_token(), 2);

        // Mixtral MoE tensor keys — block_sparse_moe + w1/w2/w3
        assert_eq!(
            arch.moe_router_key(0),
            Some("layers.0.block_sparse_moe.gate.weight".to_string())
        );
        assert_eq!(
            arch.expert_ffn_gate_key(5, 3),
            Some("layers.5.block_sparse_moe.experts.3.w1.weight".to_string())
        );
        assert_eq!(
            arch.expert_ffn_down_key(5, 3),
            Some("layers.5.block_sparse_moe.experts.3.w2.weight".to_string())
        );
        assert_eq!(
            arch.expert_ffn_up_key(5, 3),
            Some("layers.5.block_sparse_moe.experts.3.w3.weight".to_string())
        );

        // Attention is standard Llama
        assert_eq!(arch.attn_q_key(0), "layers.0.self_attn.q_proj.weight");
    }

    #[test]
    fn test_real_starcoder2_3b() {
        // Exact config from bigcode/starcoder2-3b
        let config = serde_json::json!({
            "model_type": "starcoder2",
            "hidden_size": 3072,
            "intermediate_size": 12288,
            "num_hidden_layers": 30,
            "num_attention_heads": 24,
            "num_key_value_heads": 2,
            "vocab_size": 49152,
            "rope_theta": 999999.4420358813,
            "sliding_window": 4096
        });

        let arch = detect_from_json(&config);
        assert_eq!(arch.family(), "starcoder2");
        assert_eq!(arch.config().head_dim, 128); // 3072/24
        assert_eq!(arch.config().num_kv_heads, 2);
        assert_eq!(arch.sliding_window_size(), Some(4096));
        assert!(!arch.is_moe());
    }

    #[test]
    fn test_real_granite_2b() {
        // Exact config from ibm-granite/granite-3.1-2b-base
        let config = serde_json::json!({
            "model_type": "granite",
            "hidden_size": 2048,
            "intermediate_size": 8192,
            "num_hidden_layers": 40,
            "num_attention_heads": 32,
            "num_key_value_heads": 8,
            "vocab_size": 49155,
            "rope_theta": 5000000.0
        });

        let arch = detect_from_json(&config);
        assert_eq!(arch.family(), "granite");
        assert_eq!(arch.config().head_dim, 64); // 2048/32
        assert_eq!(arch.config().rope_base, 5_000_000.0);
        assert!(!arch.is_moe());
    }

    #[test]
    fn test_real_granitemoe() {
        // Exact config from ibm-granite/granite-3.0-1b-a400m-instruct
        let config = serde_json::json!({
            "model_type": "granitemoe",
            "hidden_size": 1024,
            "intermediate_size": 512,
            "num_hidden_layers": 24,
            "num_attention_heads": 16,
            "num_key_value_heads": 8,
            "vocab_size": 49155,
            "rope_theta": 10000,
            "num_local_experts": 32,
            "num_experts_per_tok": 8
        });

        let arch = detect_from_json(&config);
        assert_eq!(arch.family(), "granitemoe");
        assert_eq!(arch.config().num_experts, Some(32));
        assert_eq!(arch.config().num_experts_per_token, Some(8));
    }

    #[test]
    fn test_real_qwen2_moe() {
        // Exact config from Qwen/Qwen1.5-MoE-A2.7B-Chat
        let config = serde_json::json!({
            "model_type": "qwen2_moe",
            "hidden_size": 2048,
            "intermediate_size": 5632,
            "num_hidden_layers": 24,
            "num_attention_heads": 16,
            "num_key_value_heads": 16,
            "vocab_size": 151936,
            "rope_theta": 1000000.0,
            "sliding_window": 32768,
            "num_experts_per_tok": 4
        });

        let arch = detect_from_json(&config);
        assert_eq!(arch.family(), "qwen2_moe");
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
        assert_eq!(arch.embed_scale(), (5376.0f32).sqrt());
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
            std::path::PathBuf::from(h)
                .join(".cache/huggingface/hub/models--google--gemma-4-31B-it")
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
    fn test_empty_config() {
        let config = serde_json::json!({});
        let arch = detect_from_json(&config);
        assert_eq!(arch.family(), "generic");
        assert_eq!(arch.config().num_layers, 32);
    }
}
