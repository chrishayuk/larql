//! Qwen 3.6 hybrid Gated DeltaNet + full-attention architecture.
//!
//! `qwen35` (dense 27B) and `qwen35moe` (MoE 35B-A3B) interleave
//! Gated DeltaNet linear-attention layers with periodic full
//! softmax-attention layers (3:1 ratio, attention at layer
//! `(i+1) % full_attention_interval == 0`).
//!
//! Key per-layer tensor differences from `QwenArch`:
//!
//! - **Linear layers** (48 of 64 in 27B): `attn_qkv` fused
//!   projection, `attn_gate` Z-gate, `ssm_conv1d` depthwise Conv1D,
//!   per-head log-decay `ssm_a`, `ssm_beta`/`ssm_alpha` projections,
//!   `ssm_norm` post-mixer norm, `ssm_out` output projection.
//! - **Full-attention layers** (16 of 64 in 27B): standard
//!   Q/K/V/O but `attn_q.weight` is the Qwen3-Next **fused Q +
//!   per-head gate** projection (output dim `2 * n_head * head_dim`,
//!   split into Q and a per-head sigmoid gate). Per-head Q and K
//!   RMSNorms at shape `[head_dim]`, NOT `[hidden]`.
//!
//! See `openspec/changes/inference-qwen35-deltanet/design.md` for
//! the full architecture brief.

use crate::config::{ModelArchitecture, ModelConfig};

pub struct Qwen35Arch {
    config: ModelConfig,
}

impl Qwen35Arch {
    pub fn from_config(config: ModelConfig) -> Self {
        Self { config }
    }
}

impl ModelArchitecture for Qwen35Arch {
    fn family(&self) -> &str {
        &self.config.model_type
    }

    fn config(&self) -> &ModelConfig {
        &self.config
    }

    fn has_post_norms(&self) -> bool {
        // Qwen 3.6 uses Gemma-2-style pre+post norm sandwich
        // (`attn_norm` AND `attn_post_norm`). The QwenArch base
        // is pre-norm-only; this is the load-bearing difference.
        true
    }

    // ── Hybrid layer routing ──

    fn is_linear_attention_layer(&self, layer: usize) -> bool {
        // recurrent_layer_arr[i] = ((i + 1) % full_attention_interval) != 0
        // i.e. attention-layer is the LAST of each group of `interval`.
        let interval = self.config.full_attention_interval.unwrap_or(0);
        if interval == 0 {
            return false;
        }
        (layer + 1) % interval != 0
    }

    fn full_attention_interval(&self) -> usize {
        self.config.full_attention_interval.unwrap_or(0)
    }

    fn ssm_state_size(&self) -> usize {
        self.config.ssm_state_size.unwrap_or(0)
    }

    fn ssm_inner_size(&self) -> usize {
        self.config.ssm_inner_size.unwrap_or(0)
    }

    fn ssm_group_count(&self) -> usize {
        self.config.ssm_group_count.unwrap_or(0)
    }

    fn ssm_dt_rank(&self) -> usize {
        self.config.ssm_dt_rank.unwrap_or(0)
    }

    fn ssm_conv_kernel(&self) -> usize {
        self.config.ssm_conv_kernel.unwrap_or(0)
    }

    fn rope_dimension_sections(&self) -> Option<&[usize]> {
        self.config.rope_dimension_sections.as_deref()
    }

    // ── Linear-attention layer tensor keys ──
    //
    // GGUF→HF normalisation maps `blk.N.<x>` → `layers.N.<x>` and
    // leaves the DeltaNet-specific suffixes (`attn_qkv`, `ssm_*`)
    // intact. So the per-layer prefix is `layers.{layer}.` and the
    // tensor names below match the GGUF originals exactly.

    fn attn_qkv_key(&self, layer: usize) -> Option<String> {
        if !self.is_linear_attention_layer(layer) {
            return None;
        }
        Some(format!("{}attn_qkv.weight", self.layer_prefix(layer)))
    }

    fn attn_gate_key(&self, layer: usize) -> Option<String> {
        // Z-gate on linear layers. Full-attn layers have their Q+gate
        // fused into `attn_q.weight` and don't expose this key.
        if !self.is_linear_attention_layer(layer) {
            return None;
        }
        Some(format!("{}attn_gate.weight", self.layer_prefix(layer)))
    }

    fn ssm_conv1d_key(&self, layer: usize) -> Option<String> {
        if !self.is_linear_attention_layer(layer) {
            return None;
        }
        Some(format!("{}ssm_conv1d.weight", self.layer_prefix(layer)))
    }

    fn ssm_dt_key(&self, layer: usize) -> Option<String> {
        // NOTE: `ssm_dt` is a BIAS, not a weight (despite the
        // confusing naming heritage from Mamba). GGUF emits it as
        // `<layer>.ssm_dt.bias`. The forward kernel adds it to
        // `alpha` before softplus.
        if !self.is_linear_attention_layer(layer) {
            return None;
        }
        Some(format!("{}ssm_dt.bias", self.layer_prefix(layer)))
    }

    fn ssm_a_key(&self, layer: usize) -> Option<String> {
        // Per-head log-decay scalar; stored as the bare tensor name
        // (no `.weight` suffix), one f32 per V head.
        if !self.is_linear_attention_layer(layer) {
            return None;
        }
        Some(format!("{}ssm_a", self.layer_prefix(layer)))
    }

    fn ssm_beta_key(&self, layer: usize) -> Option<String> {
        if !self.is_linear_attention_layer(layer) {
            return None;
        }
        Some(format!("{}ssm_beta.weight", self.layer_prefix(layer)))
    }

    fn ssm_alpha_key(&self, layer: usize) -> Option<String> {
        if !self.is_linear_attention_layer(layer) {
            return None;
        }
        Some(format!("{}ssm_alpha.weight", self.layer_prefix(layer)))
    }

    fn ssm_norm_key(&self, layer: usize) -> Option<String> {
        if !self.is_linear_attention_layer(layer) {
            return None;
        }
        Some(format!("{}ssm_norm.weight", self.layer_prefix(layer)))
    }

    fn ssm_out_key(&self, layer: usize) -> Option<String> {
        if !self.is_linear_attention_layer(layer) {
            return None;
        }
        Some(format!("{}ssm_out.weight", self.layer_prefix(layer)))
    }

    // ── Full-attention layer tensor keys ──
    //
    // GGUF→HF key replacement maps `attn_q.` → `self_attn.q_proj.`
    // etc. for the projection weights. The per-head Q/K RMSNorm
    // tensors keep their GGUF-native names (`attn_q_norm.weight` /
    // `attn_k_norm.weight`) — the substring `attn_q.` is NOT a
    // match against `attn_q_norm.` (underscore, not dot), so the
    // existing GGUF→HF normalizer leaves them alone.

    fn attn_q_per_head_norm_key(&self, layer: usize) -> Option<String> {
        if self.is_linear_attention_layer(layer) {
            return None;
        }
        Some(format!("{}attn_q_norm.weight", self.layer_prefix(layer)))
    }

    fn attn_k_per_head_norm_key(&self, layer: usize) -> Option<String> {
        if self.is_linear_attention_layer(layer) {
            return None;
        }
        Some(format!("{}attn_k_norm.weight", self.layer_prefix(layer)))
    }

    // ── Post-attention RMSNorm (Gemma-2-style pre+post sandwich) ──
    //
    // Qwen 3.6 GGUFs store the post-attention norm as
    // `blk.N.post_attention_norm.weight` (no `_layernorm` suffix),
    // which the GGUF→HF normalizer doesn't touch — so the key in
    // `ModelWeights.vectors` is `layers.N.post_attention_norm.weight`.
    // The trait default returns `..._layernorm.weight`; override.
    fn post_attention_layernorm_key(&self, layer: usize) -> String {
        format!("{}post_attention_norm.weight", self.layer_prefix(layer))
    }
}

// ────────────────────────────────────────────────────────────────────────
// MoE variant (Qwen 3.6 35B-A3B). Identical DeltaNet + attention; only
// the FFN block differs (top-8-of-256 expert routing). Inherits everything
// from `Qwen35Arch` and overrides the MoE-related methods.
// ────────────────────────────────────────────────────────────────────────

pub struct Qwen35MoeArch {
    inner: Qwen35Arch,
}

impl Qwen35MoeArch {
    pub fn from_config(config: ModelConfig) -> Self {
        Self {
            inner: Qwen35Arch::from_config(config),
        }
    }
}

impl ModelArchitecture for Qwen35MoeArch {
    // Delegate all the non-MoE methods to the dense Qwen35Arch.
    fn family(&self) -> &str {
        self.inner.family()
    }

    fn config(&self) -> &ModelConfig {
        self.inner.config()
    }

    fn has_post_norms(&self) -> bool {
        self.inner.has_post_norms()
    }

    fn is_linear_attention_layer(&self, layer: usize) -> bool {
        self.inner.is_linear_attention_layer(layer)
    }

    fn full_attention_interval(&self) -> usize {
        self.inner.full_attention_interval()
    }

    fn ssm_state_size(&self) -> usize {
        self.inner.ssm_state_size()
    }

    fn ssm_inner_size(&self) -> usize {
        self.inner.ssm_inner_size()
    }

    fn ssm_group_count(&self) -> usize {
        self.inner.ssm_group_count()
    }

    fn ssm_dt_rank(&self) -> usize {
        self.inner.ssm_dt_rank()
    }

    fn ssm_conv_kernel(&self) -> usize {
        self.inner.ssm_conv_kernel()
    }

    fn rope_dimension_sections(&self) -> Option<&[usize]> {
        self.inner.rope_dimension_sections()
    }

    fn attn_qkv_key(&self, layer: usize) -> Option<String> {
        self.inner.attn_qkv_key(layer)
    }
    fn attn_gate_key(&self, layer: usize) -> Option<String> {
        self.inner.attn_gate_key(layer)
    }
    fn ssm_conv1d_key(&self, layer: usize) -> Option<String> {
        self.inner.ssm_conv1d_key(layer)
    }
    fn ssm_dt_key(&self, layer: usize) -> Option<String> {
        self.inner.ssm_dt_key(layer)
    }
    fn ssm_a_key(&self, layer: usize) -> Option<String> {
        self.inner.ssm_a_key(layer)
    }
    fn ssm_beta_key(&self, layer: usize) -> Option<String> {
        self.inner.ssm_beta_key(layer)
    }
    fn ssm_alpha_key(&self, layer: usize) -> Option<String> {
        self.inner.ssm_alpha_key(layer)
    }
    fn ssm_norm_key(&self, layer: usize) -> Option<String> {
        self.inner.ssm_norm_key(layer)
    }
    fn ssm_out_key(&self, layer: usize) -> Option<String> {
        self.inner.ssm_out_key(layer)
    }
    fn attn_q_per_head_norm_key(&self, layer: usize) -> Option<String> {
        self.inner.attn_q_per_head_norm_key(layer)
    }
    fn attn_k_per_head_norm_key(&self, layer: usize) -> Option<String> {
        self.inner.attn_k_per_head_norm_key(layer)
    }
    fn post_attention_layernorm_key(&self, layer: usize) -> String {
        self.inner.post_attention_layernorm_key(layer)
    }

    // ── MoE-specific overrides ──

    fn is_moe(&self) -> bool {
        self.config().num_experts.unwrap_or(0) > 0
    }

    fn num_experts(&self) -> usize {
        self.config().num_experts.unwrap_or(0)
    }

    fn num_experts_per_token(&self) -> usize {
        self.config()
            .num_experts_per_token
            .or(self.config().top_k_experts)
            .unwrap_or(0)
    }

    fn moe_intermediate_size(&self) -> usize {
        self.config().moe_intermediate_size.unwrap_or(0)
    }

    fn moe_router_key(&self, layer: usize) -> Option<String> {
        if !self.is_moe() {
            return None;
        }
        Some(format!("{}mlp.gate.weight", self.layer_prefix(layer)))
    }

    fn expert_ffn_gate_key(&self, layer: usize, expert_id: usize) -> Option<String> {
        if !self.is_moe() {
            return None;
        }
        Some(format!(
            "{}mlp.experts.{expert_id}.gate_proj.weight",
            self.layer_prefix(layer)
        ))
    }

    fn expert_ffn_up_key(&self, layer: usize, expert_id: usize) -> Option<String> {
        if !self.is_moe() {
            return None;
        }
        Some(format!(
            "{}mlp.experts.{expert_id}.up_proj.weight",
            self.layer_prefix(layer)
        ))
    }

    fn expert_ffn_down_key(&self, layer: usize, expert_id: usize) -> Option<String> {
        if !self.is_moe() {
            return None;
        }
        Some(format!(
            "{}mlp.experts.{expert_id}.down_proj.weight",
            self.layer_prefix(layer)
        ))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::ModelConfig;

    fn qwen35_config_27b() -> ModelConfig {
        // Reproduce the GGUF metadata for `Qwen3.6-27B-Q4_K_S`.
        ModelConfig {
            model_type: "qwen35".into(),
            num_layers: 64,
            hidden_size: 5120,
            intermediate_size: 17408,
            head_dim: 256,
            num_q_heads: 24,
            num_kv_heads: 4,
            vocab_size: Some(262144),
            rope_base: 10_000_000.0,
            rope_local_base: None,
            sliding_window: None,
            num_experts: None,
            num_experts_per_token: None,
            num_shared_experts: None,
            kv_lora_rank: None,
            q_lora_rank: None,
            rope_scaling: None,
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
            enable_moe_block: false,
            top_k_experts: None,
            moe_intermediate_size: None,
            full_attention_interval: Some(4),
            ssm_state_size: Some(128),
            ssm_inner_size: Some(6144),
            ssm_dt_rank: Some(48),
            ssm_group_count: Some(16),
            ssm_conv_kernel: Some(4),
            rope_dimension_sections: Some(vec![16, 24, 24, 0]),
        }
    }

    #[test]
    fn qwen35_layer_routing_27b() {
        // 64 layers, full_attention_interval=4 →
        // attention at layers 3, 7, 11, 15, …, 63 (16 attention layers)
        // linear at all other 48 layers.
        let arch = Qwen35Arch::from_config(qwen35_config_27b());
        let attention_layers: Vec<usize> = (0..64)
            .filter(|&l| !arch.is_linear_attention_layer(l))
            .collect();
        let expected: Vec<usize> = (3..64).step_by(4).collect();
        assert_eq!(attention_layers, expected);
        assert_eq!(attention_layers.len(), 16);
        let linear_count = (0..64)
            .filter(|&l| arch.is_linear_attention_layer(l))
            .count();
        assert_eq!(linear_count, 48);
    }

    #[test]
    fn qwen35_linear_layer_keys_are_present() {
        let arch = Qwen35Arch::from_config(qwen35_config_27b());
        assert!(arch.is_linear_attention_layer(0));
        assert_eq!(
            arch.attn_qkv_key(0),
            Some("layers.0.attn_qkv.weight".into())
        );
        assert_eq!(
            arch.attn_gate_key(0),
            Some("layers.0.attn_gate.weight".into())
        );
        assert_eq!(
            arch.ssm_conv1d_key(0),
            Some("layers.0.ssm_conv1d.weight".into())
        );
        assert_eq!(arch.ssm_dt_key(0), Some("layers.0.ssm_dt.bias".into()));
        assert_eq!(arch.ssm_a_key(0), Some("layers.0.ssm_a".into()));
        assert_eq!(
            arch.ssm_beta_key(0),
            Some("layers.0.ssm_beta.weight".into())
        );
        assert_eq!(
            arch.ssm_alpha_key(0),
            Some("layers.0.ssm_alpha.weight".into())
        );
        assert_eq!(
            arch.ssm_norm_key(0),
            Some("layers.0.ssm_norm.weight".into())
        );
        assert_eq!(arch.ssm_out_key(0), Some("layers.0.ssm_out.weight".into()));
        // Per-head Q/K norms only exist on full-attn layers.
        assert_eq!(arch.attn_q_per_head_norm_key(0), None);
        assert_eq!(arch.attn_k_per_head_norm_key(0), None);
    }

    #[test]
    fn qwen35_full_attention_layer_keys_are_present() {
        let arch = Qwen35Arch::from_config(qwen35_config_27b());
        // Layer 3 is full-attention (4 % 4 == 0).
        assert!(!arch.is_linear_attention_layer(3));
        // DeltaNet keys absent on full-attn layers.
        assert_eq!(arch.attn_qkv_key(3), None);
        assert_eq!(arch.attn_gate_key(3), None);
        assert_eq!(arch.ssm_conv1d_key(3), None);
        // Per-head Q/K norms present — GGUF-native names
        // (the existing GGUF→HF normalizer doesn't rewrite
        // `attn_q_norm.` because the rule is `attn_q.` with a
        // trailing dot, not underscore).
        assert_eq!(
            arch.attn_q_per_head_norm_key(3),
            Some("layers.3.attn_q_norm.weight".into())
        );
        assert_eq!(
            arch.attn_k_per_head_norm_key(3),
            Some("layers.3.attn_k_norm.weight".into())
        );
        // Post-attention RMSNorm key — Qwen 3.6 GGUF stores it
        // as `post_attention_norm.weight` (no `_layernorm` suffix).
        assert_eq!(
            arch.post_attention_layernorm_key(3),
            "layers.3.post_attention_norm.weight"
        );
        // Standard Q/K/V/O keys from default trait impls.
        assert_eq!(arch.attn_q_key(3), "layers.3.self_attn.q_proj.weight");
        assert_eq!(arch.attn_k_key(3), "layers.3.self_attn.k_proj.weight");
        assert_eq!(arch.attn_v_key(3), "layers.3.self_attn.v_proj.weight");
        assert_eq!(arch.attn_o_key(3), "layers.3.self_attn.o_proj.weight");
    }

    #[test]
    fn qwen35_exposes_ssm_dimensions() {
        let arch = Qwen35Arch::from_config(qwen35_config_27b());
        assert_eq!(arch.full_attention_interval(), 4);
        assert_eq!(arch.ssm_state_size(), 128);
        assert_eq!(arch.ssm_inner_size(), 6144);
        assert_eq!(arch.ssm_dt_rank(), 48);
        assert_eq!(arch.ssm_group_count(), 16);
        assert_eq!(arch.ssm_conv_kernel(), 4);
        assert_eq!(arch.rope_dimension_sections(), Some(&[16, 24, 24, 0][..]));
    }

    #[test]
    fn qwen35_has_post_norms() {
        // Gemma-2-style sandwich is the load-bearing difference
        // from plain QwenArch.
        let arch = Qwen35Arch::from_config(qwen35_config_27b());
        assert!(arch.has_post_norms());
    }

    #[test]
    fn qwen35moe_inherits_linear_attention_layer_routing() {
        let mut cfg = qwen35_config_27b();
        cfg.model_type = "qwen35moe".into();
        cfg.num_experts = Some(256);
        cfg.num_experts_per_token = Some(8);
        cfg.moe_intermediate_size = Some(512);
        let arch = Qwen35MoeArch::from_config(cfg);
        // Same hybrid routing.
        assert!(arch.is_linear_attention_layer(0));
        assert!(!arch.is_linear_attention_layer(3));
        // MoE on.
        assert!(arch.is_moe());
        assert_eq!(arch.num_experts(), 256);
        assert_eq!(arch.num_experts_per_token(), 8);
        assert_eq!(arch.moe_intermediate_size(), 512);
        // MoE expert keys.
        assert_eq!(
            arch.expert_ffn_gate_key(0, 7),
            Some("layers.0.mlp.experts.7.gate_proj.weight".into())
        );
        assert_eq!(
            arch.moe_router_key(0),
            Some("layers.0.mlp.gate.weight".into())
        );
    }

    #[test]
    fn qwen35_without_interval_is_not_hybrid() {
        // If full_attention_interval is missing, treat every layer
        // as full-attention (no DeltaNet routing). Pure-transformer
        // qwen models should never come through this arch handler
        // anyway, but the predicate must degrade gracefully.
        let mut cfg = qwen35_config_27b();
        cfg.full_attention_interval = None;
        let arch = Qwen35Arch::from_config(cfg);
        for l in 0..64 {
            assert!(!arch.is_linear_attention_layer(l));
        }
    }
}
