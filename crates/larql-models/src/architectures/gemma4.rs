//! Gemma 4 architecture — Google's multimodal model family (2025).
//!
//! Key differences from Gemma 3:
//! - Dual head_dim: sliding layers use head_dim (256), global layers use global_head_dim (512)
//! - Fewer KV heads on global layers (num_global_key_value_heads)
//! - Partial rotary: global layers apply RoPE to only 25% of head dims
//! - K=V sharing: later global layers have no v_proj (value = key)
//! - Per-layer scalar multiplier (layer_scalar)
//! - QK-norm (inherited from Gemma 3)
//! - 4 norms per layer (inherited from Gemma 3)
//! - Logit softcapping (inherited from Gemma 2)
//!
//! Layer types are determined from:
//! 1. Explicit `layer_types` array in config.json (["sliding_attention", "full_attention", ...])
//! 2. `sliding_window_pattern` field (every Nth layer is full)
//! 3. Default pattern of 6 (every 6th layer is full)

use crate::config::{Activation, ExpertFormat, ModelArchitecture, ModelConfig};

/// Layer type string used in Gemma 4 `layer_types` config field.
const LAYER_TYPE_FULL: &str = "full_attention";
/// Default sliding-window period when not explicit in config.
const DEFAULT_SLIDING_WINDOW_PATTERN: usize = 6;

pub struct Gemma4Arch {
    config: ModelConfig,
    /// Precomputed: which layer indices are full (global) attention.
    global_layers: Vec<bool>,
    /// Precomputed: KV sharing source for each layer (None = compute own KV).
    kv_sources: Vec<Option<usize>>,
}

impl Gemma4Arch {
    pub fn from_config(config: ModelConfig) -> Self {
        let num_layers = config.num_layers;

        // Determine global layers from explicit layer_types or pattern
        let global_layers: Vec<bool> = if let Some(ref types) = config.layer_types {
            (0..num_layers)
                .map(|layer| types.get(layer).is_some_and(|t| t == LAYER_TYPE_FULL))
                .collect()
        } else {
            let pattern = config
                .sliding_window_pattern
                .filter(|&pattern| pattern > 0)
                .unwrap_or(DEFAULT_SLIDING_WINDOW_PATTERN);
            (0..num_layers)
                .map(|layer| (layer + 1) % pattern == 0)
                .collect()
        };

        // Precompute KV sharing sources.
        // Layers in the shared region reuse KV from the last non-shared layer
        // of the same type (sliding→last sliding source, global→last global source).
        let num_shared = config.num_kv_shared_layers.unwrap_or(0);
        let first_shared = if num_shared > 0 && num_shared < num_layers {
            num_layers - num_shared
        } else {
            num_layers // no sharing
        };
        let kv_sources = if num_shared > 0 {
            // Find the last non-shared sliding and global layers
            let last_sliding = (0..first_shared).rev().find(|&l| !global_layers[l]);
            let last_global = (0..first_shared).rev().find(|&l| global_layers[l]);

            (0..num_layers)
                .map(|layer| {
                    if layer < first_shared {
                        None // non-shared: compute own KV
                    } else if global_layers[layer] {
                        last_global // shared global → last non-shared global
                    } else {
                        last_sliding // shared sliding → last non-shared sliding
                    }
                })
                .collect()
        } else {
            vec![None; num_layers]
        };

        Self {
            config,
            global_layers,
            kv_sources,
        }
    }

    fn is_global_layer(&self, layer: usize) -> bool {
        self.global_layers.get(layer).copied().unwrap_or(false)
    }
}

impl ModelArchitecture for Gemma4Arch {
    fn family(&self) -> &str {
        "gemma4"
    }

    fn config(&self) -> &ModelConfig {
        &self.config
    }

    /// Gemma 4 weights use `model.language_model.` prefix (multimodal wrapper).
    fn key_prefixes_to_strip(&self) -> &[&str] {
        &[
            "model.language_model.model.",
            "model.language_model.",
            "language_model.model.",
            "model.",
        ]
    }

    // ── Per-layer attention geometry ──

    fn head_dim_for_layer(&self, layer: usize) -> usize {
        if self.is_global_layer(layer) {
            self.config.global_head_dim.unwrap_or(self.config.head_dim)
        } else {
            self.config.head_dim
        }
    }

    fn num_kv_heads_for_layer(&self, layer: usize) -> usize {
        if self.is_global_layer(layer) {
            self.config
                .num_global_kv_heads
                .unwrap_or(self.config.num_kv_heads)
        } else {
            self.config.num_kv_heads
        }
    }

    fn num_q_heads_for_layer(&self, _layer: usize) -> usize {
        // Gemma 4 keeps num_q_heads constant across all layers.
        // At global layers, each head uses global_head_dim instead of head_dim,
        // so Q projection output is larger (num_q * global_head_dim).
        self.config.num_q_heads
    }

    fn rotary_fraction_for_layer(&self, layer: usize) -> f64 {
        if self.is_global_layer(layer) {
            self.config.partial_rotary_factor.unwrap_or(1.0)
        } else {
            1.0
        }
    }

    fn v_shares_k(&self, layer: usize) -> bool {
        // On 31B, attention_k_eq_v=true means V reuses K only on global (full_attention)
        // layers — v_proj is still present on sliding layers. On E2B (attention_k_eq_v=false)
        // this is always false. Per-layer gating matches what ships in the safetensors.
        self.config.attention_k_eq_v && self.is_global_layer(layer)
    }

    fn has_v_norm(&self) -> bool {
        true
    }

    fn kv_shared_source_layer(&self, layer: usize) -> Option<usize> {
        self.kv_sources.get(layer).copied().flatten()
    }

    // Gemma 4 uses QK-norm which already normalizes dot products.
    // No additional 1/sqrt(head_dim) scaling is applied (scaling = 1.0).
    fn attention_scale(&self) -> f64 {
        1.0
    }

    fn attention_scale_for_layer(&self, _layer: usize) -> f64 {
        1.0
    }

    fn layer_scalar_key(&self, layer: usize) -> Option<String> {
        Some(format!("{}layer_scalar", self.layer_prefix(layer)))
    }

    // ── QK norm (inherited from Gemma 3) ──

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

    // ── Gemma-family behavior ──

    // Gemma 4 stores norm weights as the full multiplier (no +1 offset).
    // Unlike Gemma 2/3 which used 1+weight, Gemma 4's Gemma4RMSNorm applies weight directly.
    fn norm_weight_offset(&self) -> f32 {
        0.0
    }

    fn activation(&self) -> Activation {
        Activation::GeluTanh
    }

    fn embed_scale(&self) -> f32 {
        (self.config.hidden_size as f32).sqrt()
    }

    // Gemma 4's shipped `tokenizer.json` omits `<bos>` from its
    // `TemplateProcessing.single` template (Gemma 2/3 included it), so
    // `encode(prompt, add_special=true)` returns a sequence without the
    // leading BOS token and the model's attention sees a broken prefix.
    // Callers consult this to prepend token id 2 when missing.
    fn bos_token_id(&self) -> Option<u32> {
        Some(2)
    }

    fn has_post_norms(&self) -> bool {
        true
    }

    fn is_sliding_window_layer(&self, layer: usize) -> bool {
        !self.is_global_layer(layer)
    }

    fn rope_base_for_layer(&self, layer: usize) -> f64 {
        if self.is_sliding_window_layer(layer) {
            self.config
                .rope_local_base
                .unwrap_or(crate::defaults::ROPE_BASE_DEFAULT)
        } else {
            self.config.rope_base
        }
    }

    // ── Hybrid MoE (26B A4B: dense MLP + expert block, outputs summed) ──

    fn is_moe(&self) -> bool {
        self.config.enable_moe_block
    }

    fn is_hybrid_moe(&self) -> bool {
        self.config.enable_moe_block
    }

    fn expert_format(&self) -> ExpertFormat {
        ExpertFormat::PackedBF16
    }

    fn num_experts(&self) -> usize {
        self.config.num_experts.unwrap_or(0)
    }

    fn num_experts_per_token(&self) -> usize {
        self.config
            .top_k_experts
            .or(self.config.num_experts_per_token)
            .unwrap_or(0)
    }

    fn moe_intermediate_size(&self) -> usize {
        self.config.moe_intermediate_size.unwrap_or(0)
    }

    fn moe_router_type(&self) -> &str {
        if self.config.enable_moe_block {
            "gemma4_top_k_softmax"
        } else {
            "top_k_softmax"
        }
    }

    /// Router linear projection: selects top-k experts.
    fn moe_router_key(&self, layer: usize) -> Option<String> {
        if self.config.enable_moe_block {
            Some(format!("{}router.proj.weight", self.layer_prefix(layer)))
        } else {
            None
        }
    }

    fn moe_router_scale_key(&self, layer: usize) -> Option<String> {
        if self.config.enable_moe_block {
            Some(format!("{}router.scale", self.layer_prefix(layer)))
        } else {
            None
        }
    }

    fn moe_router_per_expert_scale_key(&self, layer: usize) -> Option<String> {
        if self.config.enable_moe_block {
            Some(format!(
                "{}router.per_expert_scale",
                self.layer_prefix(layer)
            ))
        } else {
            None
        }
    }

    fn moe_router_norm_parameter_free(&self) -> bool {
        // HF `Gemma4TextRouter` uses `Gemma4RMSNorm(with_scale=False)`, i.e.
        // pure variance normalisation — no learned weight exists on disk.
        self.config.enable_moe_block
    }

    fn moe_router_input_scalar(&self) -> Option<f32> {
        if self.config.enable_moe_block {
            Some((self.config.hidden_size as f32).powf(-0.5))
        } else {
            None
        }
    }

    /// All experts' gate+up weights packed: [num_experts, 2*moe_intermediate, hidden].
    fn packed_experts_gate_up_key(&self, layer: usize) -> Option<String> {
        if self.config.enable_moe_block {
            Some(format!("{}experts.gate_up_proj", self.layer_prefix(layer)))
        } else {
            None
        }
    }

    /// All experts' down weights packed: [num_experts, hidden, moe_intermediate].
    fn packed_experts_down_key(&self, layer: usize) -> Option<String> {
        if self.config.enable_moe_block {
            Some(format!("{}experts.down_proj", self.layer_prefix(layer)))
        } else {
            None
        }
    }

    // In MoE layers, post_feedforward_layernorm becomes _1 (dense branch).
    fn post_feedforward_layernorm_key(&self, layer: usize) -> Option<String> {
        if self.config.enable_moe_block {
            Some(format!(
                "{}post_feedforward_layernorm_1.weight",
                self.layer_prefix(layer)
            ))
        } else {
            Some(format!(
                "{}post_feedforward_layernorm.weight",
                self.layer_prefix(layer)
            ))
        }
    }

    fn moe_pre_experts_norm_key(&self, layer: usize) -> Option<String> {
        if self.config.enable_moe_block {
            Some(format!(
                "{}pre_feedforward_layernorm_2.weight",
                self.layer_prefix(layer)
            ))
        } else {
            None
        }
    }

    fn moe_post_experts_norm_key(&self, layer: usize) -> Option<String> {
        if self.config.enable_moe_block {
            Some(format!(
                "{}post_feedforward_layernorm_2.weight",
                self.layer_prefix(layer)
            ))
        } else {
            None
        }
    }

    fn moe_post_ffn1_norm_key(&self, layer: usize) -> Option<String> {
        // Alias for post_feedforward_layernorm_1 — same key, explicit name for clarity.
        self.post_feedforward_layernorm_key(layer)
    }

    fn moe_post_outer_norm_key(&self, layer: usize) -> Option<String> {
        if self.config.enable_moe_block {
            Some(format!(
                "{}post_feedforward_layernorm.weight",
                self.layer_prefix(layer)
            ))
        } else {
            None
        }
    }

    fn moe_has_combined_output_norm(&self) -> bool {
        // Gemma 4 hybrid MoE: after combining dense + expert outputs, apply
        // post_feedforward_layernorm to the sum before adding to residual.
        // Matches HF Gemma4TextDecoderLayer.forward():
        //   hidden_states = h1 + h2
        //   hidden_states = self.post_feedforward_layernorm(hidden_states)
        //   hidden_states = residual + hidden_states
        self.config.enable_moe_block
    }
}
