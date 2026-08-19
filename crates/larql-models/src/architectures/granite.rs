//! IBM Granite architecture.
//!
//! Llama-compatible: same tensor keys, norms, activation, RoPE.
//! Granite Vision variants additionally declare a multi-modal protocol
//! for SigLIP2 + MLP GELU connector + AnyRes tiling (Phase 2).

use crate::config::{ModelArchitecture, ModelConfig};
use crate::multimodal::{MultiModalProtocol, PlaceholderProtocol, PrecomputedScaling, TokenBudget};

/// Multi-modal contract for Granite Vision models.
pub struct GraniteVisionMultiModal;

pub const GRANITE_VISION_MULTIMODAL: GraniteVisionMultiModal = GraniteVisionMultiModal;

const GRANITE_VALID_TILE_COUNTS: &[usize] = &[1, 2, 3, 4, 5, 6];

impl MultiModalProtocol for GraniteVisionMultiModal {
    fn vision_encoder(&self) -> Option<&str> {
        Some("siglip2")
    }

    fn image_placeholder(&self) -> Option<PlaceholderProtocol> {
        Some(PlaceholderProtocol {
            start: None,
            fill: 49152,
            end: None,
        })
    }

    fn image_token_budget(&self) -> TokenBudget {
        TokenBudget::PerTile {
            tokens_per_tile: 729,
        }
    }

    fn precomputed_scaling(&self) -> PrecomputedScaling {
        PrecomputedScaling::None
    }

    fn valid_tile_counts(&self) -> &[usize] {
        GRANITE_VALID_TILE_COUNTS
    }
}

pub struct GraniteArch {
    config: ModelConfig,
}

impl GraniteArch {
    pub fn from_config(config: ModelConfig) -> Self {
        Self { config }
    }
}

impl ModelArchitecture for GraniteArch {
    fn family(&self) -> &str {
        &self.config.model_type
    }

    fn config(&self) -> &ModelConfig {
        &self.config
    }

    fn multimodal(&self) -> Option<&dyn MultiModalProtocol> {
        if self.config.has_vision_config {
            Some(&GRANITE_VISION_MULTIMODAL)
        } else {
            None
        }
    }

    // ── MoE (granitemoe) ──
    //
    // GraniteMoE stacks its experts into three tensors per layer rather than
    // storing them per-expert, so it uses the PACKED keys — the same shape
    // convention Gemma 4 uses — and emits no `expert_ffn_*` keys at all:
    //
    //   block_sparse_moe.input_linear.weight   [E, 2*inter, hidden]  (gate+up)
    //   block_sparse_moe.output_linear.weight  [E, hidden, inter]    (down)
    //   block_sparse_moe.router.layer.weight   [E, hidden]           (router)
    //
    // Verified against ibm-granite/granite-3.0-1b-a400m-instruct:
    // [32, 1024, 1024] / [32, 1024, 512] / [32, 1024] at hidden=1024,
    // inter=512, E=32.
    //
    // Unlike Gemma 4 this is a pure MoE block, not a hybrid — there is no
    // parallel dense branch — so `is_hybrid_moe()` stays false and the dense
    // Granite path is untouched (every method here gates on `is_moe()`).

    fn is_moe(&self) -> bool {
        self.config.num_experts.unwrap_or(0) > 0
    }

    fn num_experts(&self) -> usize {
        self.config.num_experts.unwrap_or(0)
    }

    fn num_experts_per_token(&self) -> usize {
        self.config
            .num_experts_per_token
            .or(self.config.top_k_experts)
            .unwrap_or(0)
    }

    /// GraniteMoE has no `moe_intermediate_size`; per-expert width is the
    /// model's `intermediate_size` (512 for 1B-A400M). Falling back to 0
    /// would size every expert to nothing.
    fn moe_intermediate_size(&self) -> usize {
        if !self.is_moe() {
            return 0;
        }
        self.config
            .moe_intermediate_size
            .unwrap_or(self.config.intermediate_size)
    }

    /// GraniteMoE stacks all experts into one BF16 tensor per projection —
    /// the same physical layout Gemma 4 26B A4B uses, so the existing
    /// packed-expert quantiser consumes it unchanged.
    fn expert_format(&self) -> crate::config::ExpertFormat {
        if self.is_moe() {
            crate::config::ExpertFormat::PackedBF16
        } else {
            crate::config::ExpertFormat::PerExpert
        }
    }

    /// `GraniteMoeTopKGating.forward` selects *then* normalises:
    ///
    /// ```text
    /// top_k_logits, top_k_indices = logits.topk(self.top_k, dim=1)
    /// top_k_gates = torch.softmax(top_k_logits, dim=1)
    /// ```
    ///
    /// so the chosen experts' weights sum to 1. The inherited default is
    /// softmax-over-all-then-select, whose weights do not — on this model
    /// that is a 49.8% bits/char disagreement with the reference, not a
    /// rounding difference. Selection is unaffected (softmax is monotonic);
    /// only the weights change.
    fn moe_router_kind(&self) -> crate::config::MoeRouterKind {
        crate::config::MoeRouterKind::TopKThenSoftmax
    }

    /// The reference tier's spelling of the same fact. `granitemoe` ships no
    /// `norm_topk_prob`, so the config-reading default answers
    /// `SoftmaxThenSelect` for a model that renormalises.
    fn expert_routing_policy(&self) -> crate::config::ExpertRoutingPolicy {
        crate::config::ExpertRoutingPolicy::NormalisedOverSelected
    }

    /// `GraniteMoeMoE.forward` splits with `hidden_states.chunk(2, dim=-1)`
    /// — the leading half is gate. Not inferable from `PackedBF16`: GPT-OSS
    /// is equally packed and interleaved.
    fn gate_up_layout(&self) -> Option<crate::config::GateUpLayout> {
        self.is_moe()
            .then_some(crate::config::GateUpLayout::ContiguousHalves)
    }

    fn moe_router_key(&self, layer: usize) -> Option<String> {
        if !self.is_moe() {
            return None;
        }
        Some(format!(
            "{}block_sparse_moe.router.layer.weight",
            self.layer_prefix(layer)
        ))
    }

    fn packed_experts_gate_up_key(&self, layer: usize) -> Option<String> {
        if !self.is_moe() {
            return None;
        }
        Some(format!(
            "{}block_sparse_moe.input_linear.weight",
            self.layer_prefix(layer)
        ))
    }

    fn packed_experts_down_key(&self, layer: usize) -> Option<String> {
        if !self.is_moe() {
            return None;
        }
        Some(format!(
            "{}block_sparse_moe.output_linear.weight",
            self.layer_prefix(layer)
        ))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::ModelConfig;

    fn text_only_config() -> ModelConfig {
        ModelConfig {
            model_type: "granite".into(),
            norm_eps: Some(1e-5),
            num_layers: 28,
            hidden_size: 2048,
            intermediate_size: 8192,
            head_dim: 64,
            num_q_heads: 32,
            num_kv_heads: 8,
            vocab_size: Some(49152),
            rope_base: 10_000.0,
            layer_rope_theta: None,
            rope_local_base: None,
            sliding_window: None,
            num_experts: None,
            num_experts_per_token: None,
            num_shared_experts: None,
            enable_moe_block: false,
            top_k_experts: None,
            moe_intermediate_size: None,
            swiglu_limit: None,
            norm_topk_prob: None,
            kv_lora_rank: None,
            q_lora_rank: None,
            qk_nope_head_dim: None,
            qk_rope_head_dim: None,
            v_head_dim: None,
            rope_scaling: None,
            attn_logit_softcapping: None,
            final_logit_softcapping: None,
            query_pre_attn_scalar: None,
            embedding_multiplier: Some(12.0),
            residual_multiplier: Some(0.22),
            attention_multiplier: Some(0.015625),
            logits_scaling: Some(10.0),
            global_head_dim: None,
            num_global_kv_heads: None,
            partial_rotary_factor: None,
            sliding_window_pattern: None,
            layer_types: None,
            attention_k_eq_v: false,
            per_layer_embed_dim: None,
            num_kv_shared_layers: None,
            has_vision_config: false,
            tie_word_embeddings: None,
            qk_scale_factor: None,
            output_multiplier: None,
            post_norm_eps: None,
            attention_bias: None,
            hidden_act: None,
            max_position_embeddings: None,
            image_token_id: None,
            video_token_id: None,
            out_hidden_size: None,
            projector_hidden_size: None,
            projector_hidden_act: None,
            target_layer_ids: None,
            draft_block_size: None,
            mask_token_id: None,
            use_double_wide_mlp: None,
            vocab_size_per_layer_input: None,
        }
    }

    fn vision_config() -> ModelConfig {
        let mut cfg = text_only_config();
        cfg.has_vision_config = true;
        cfg
    }

    #[test]
    fn text_only_granite_has_no_multimodal() {
        let arch = GraniteArch::from_config(text_only_config());
        assert!(arch.multimodal().is_none());
    }

    #[test]
    fn vision_granite_declares_multimodal_protocol() {
        let arch = GraniteArch::from_config(vision_config());
        let mm = arch
            .multimodal()
            .expect("Granite Vision must declare multimodal protocol");
        assert_eq!(mm.vision_encoder(), Some("siglip2"));
        assert!(mm.audio_encoder().is_none());
    }

    #[test]
    fn image_placeholder_has_fill_only() {
        let arch = GraniteArch::from_config(vision_config());
        let ph = arch
            .multimodal()
            .unwrap()
            .image_placeholder()
            .expect("Granite Vision declares image placeholder");
        assert!(ph.start.is_none());
        assert!(ph.end.is_none());
        assert_eq!(ph.fill, 49152);
    }

    #[test]
    fn token_budget_is_per_tile_729() {
        let arch = GraniteArch::from_config(vision_config());
        match arch.multimodal().unwrap().image_token_budget() {
            TokenBudget::PerTile { tokens_per_tile } => assert_eq!(tokens_per_tile, 729),
            other => panic!("expected PerTile, got {other:?}"),
        }
    }

    #[test]
    fn precomputed_scaling_is_none() {
        let arch = GraniteArch::from_config(vision_config());
        assert_eq!(
            arch.multimodal().unwrap().precomputed_scaling(),
            PrecomputedScaling::None
        );
    }

    #[test]
    fn valid_tile_counts_non_empty() {
        let arch = GraniteArch::from_config(vision_config());
        let counts = arch.multimodal().unwrap().valid_tile_counts();
        assert!(!counts.is_empty());
        assert_eq!(counts, &[1, 2, 3, 4, 5, 6]);
    }

    #[test]
    fn no_audio_in_granite_vision() {
        let arch = GraniteArch::from_config(vision_config());
        assert!(arch.multimodal().unwrap().audio_placeholder().is_none());
    }
}
