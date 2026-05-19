//! Gemma 3 architecture — Google's multimodal model family.
//!
//! Key differences from standard Llama:
//! - Embedding scaled by sqrt(hidden_size)
//! - QK normalization per-head (q_norm, k_norm weights)
//! - 4 norms per layer (pre/post attention, pre/post FFN)
//! - Sliding window attention on most layers (every Nth layer is full)
//! - rope_theta defaults to 1,000,000 (not in config.json, HF class default)
//!
//! Note: HuggingFace saves Gemma norm weights with the +1 offset already baked in,
//! so norm_weight_offset is 0.0 (the saved weight IS the final multiplier).

use crate::config::{Activation, ModelArchitecture, ModelConfig};

/// Gemma 3 sliding window pattern: every 6th layer (0-indexed: 5, 11, 17, ...)
/// uses full attention, the rest use sliding window.
const GEMMA3_SLIDING_WINDOW_PATTERN: usize = 6;

pub struct Gemma3Arch {
    config: ModelConfig,
}

impl Gemma3Arch {
    pub fn from_config(config: ModelConfig) -> Self {
        Self { config }
    }
}

impl ModelArchitecture for Gemma3Arch {
    fn family(&self) -> &str {
        "gemma3"
    }

    fn config(&self) -> &ModelConfig {
        &self.config
    }

    // ── Gemma 3 has QK norm ──

    fn attn_q_norm_key(&self, layer: usize) -> Option<String> {
        Some(format!(
            "{}self_attn.q_norm.weight",
            self.layer_prefix(layer)
        ))
    }

    fn attn_k_norm_key(&self, layer: usize) -> Option<String> {
        Some(format!(
            "{}self_attn.k_norm.weight",
            self.layer_prefix(layer)
        ))
    }

    // ── Gemma-specific behavior ──

    // All Gemma 3 norms (layer + QK) use 1.0 + learned_weight at runtime.
    fn norm_weight_offset(&self) -> f32 {
        1.0
    }

    fn qk_norm_weight_offset(&self) -> f32 {
        1.0
    }

    fn activation(&self) -> Activation {
        Activation::GeluTanh
    }

    fn embed_scale(&self) -> f32 {
        (self.config.hidden_size as f32).sqrt()
    }

    fn has_post_norms(&self) -> bool {
        true
    }

    fn is_sliding_window_layer(&self, layer: usize) -> bool {
        // Full attention on every Nth layer, sliding window on the rest.
        // Layer indices 5, 11, 17, 23, 29 are full attention (0-indexed).
        !(layer + 1).is_multiple_of(GEMMA3_SLIDING_WINDOW_PATTERN)
    }

    fn rope_base_for_layer(&self, layer: usize) -> f64 {
        if self.is_sliding_window_layer(layer) {
            // Local layers use a lower RoPE base.
            self.config
                .rope_local_base
                .unwrap_or(crate::defaults::ROPE_BASE_DEFAULT)
        } else {
            // Global layers use the full rope_theta.
            self.config.rope_base
        }
    }

    /// Apply linear `rope_scaling.factor` to global (full-attention)
    /// layers only. HF's `Gemma3TextConfig` expands the flat
    /// `rope_scaling = {rope_type: linear, factor: N}` into the
    /// structured `{full_attention: {rope_type: linear, factor: N},
    /// sliding_attention: {rope_type: default}}` form — sliding layers
    /// stay at standard RoPE.
    ///
    /// The parser sets `gemma3_global_only = true` on the structured
    /// form. For the flat form (older Gemma 3 dumps), we still honour
    /// `scaling_type = linear` as global-only because that matches what
    /// `Gemma3TextConfig` produces from the same input.
    fn rope_position_divisor_for_layer(&self, layer: usize) -> f64 {
        let rs = match self.config.rope_scaling.as_ref() {
            Some(rs) => rs,
            None => return 1.0,
        };
        if !rs.scaling_type.eq_ignore_ascii_case("linear") {
            return 1.0;
        }
        if self.is_sliding_window_layer(layer) {
            1.0
        } else {
            rs.factor
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::RopeScaling;

    fn synth_config(rope_scaling: Option<RopeScaling>) -> ModelConfig {
        ModelConfig {
            model_type: "gemma3".into(),
            norm_eps: Some(1e-6),
            num_layers: 34,
            hidden_size: 2560,
            intermediate_size: 10240,
            head_dim: 256,
            num_q_heads: 8,
            num_kv_heads: 4,
            vocab_size: Some(256_000),
            rope_base: 1_000_000.0,
            rope_local_base: Some(10_000.0),
            sliding_window: Some(1024),
            num_experts: None,
            num_experts_per_token: None,
            num_shared_experts: None,
            enable_moe_block: false,
            top_k_experts: None,
            moe_intermediate_size: None,
            kv_lora_rank: None,
            q_lora_rank: None,
            rope_scaling,
            attn_logit_softcapping: None,
            final_logit_softcapping: None,
            query_pre_attn_scalar: None,
            embedding_multiplier: None,
            residual_multiplier: None,
            attention_multiplier: None,
            logits_scaling: None,
            global_head_dim: None,
            num_global_kv_heads: None,
            partial_rotary_factor: None,
            sliding_window_pattern: None,
            layer_types: None,
            attention_k_eq_v: false,
            per_layer_embed_dim: None,
            num_kv_shared_layers: None,
        }
    }

    #[test]
    fn attn_k_norm_key_renders_per_layer_prefix() {
        let arch = Gemma3Arch::from_config(synth_config(None));
        let key = arch.attn_k_norm_key(7).unwrap();
        assert!(key.ends_with("self_attn.k_norm.weight"));
        assert!(key.contains("7"), "layer index missing from key: {key}");
    }

    #[test]
    fn rope_divisor_returns_one_when_rope_scaling_missing() {
        // `rope_scaling = None` early-returns 1.0 — covers L107-110.
        let arch = Gemma3Arch::from_config(synth_config(None));
        assert_eq!(arch.rope_position_divisor_for_layer(0), 1.0);
    }

    #[test]
    fn rope_divisor_returns_one_when_scaling_type_is_not_linear() {
        // Non-linear scaling_type early-returns 1.0 — covers L111-113.
        let arch = Gemma3Arch::from_config(synth_config(Some(RopeScaling {
            scaling_type: "yarn".into(),
            factor: 8.0,
            llama3_low_freq_factor: None,
            llama3_high_freq_factor: None,
            llama3_original_max_position_embeddings: None,
            gemma3_global_only: false,
        })));
        assert_eq!(arch.rope_position_divisor_for_layer(0), 1.0);
        assert_eq!(arch.rope_position_divisor_for_layer(5), 1.0);
    }

    #[test]
    fn linear_rope_divisor_applies_to_full_attention_layers_only() {
        let arch = Gemma3Arch::from_config(synth_config(Some(RopeScaling {
            scaling_type: "linear".into(),
            factor: 8.0,
            llama3_low_freq_factor: None,
            llama3_high_freq_factor: None,
            llama3_original_max_position_embeddings: None,
            gemma3_global_only: true,
        })));
        // Layers 5, 11, 17, ... are full attention; everyone else sliding.
        assert_eq!(arch.rope_position_divisor_for_layer(5), 8.0);
        assert_eq!(arch.rope_position_divisor_for_layer(4), 1.0);
    }
}
