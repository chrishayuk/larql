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

use crate::config::{Activation, ModelArchitecture, ModelConfig, PostNormEps};
use crate::multimodal::{MultiModalProtocol, PlaceholderProtocol, PrecomputedScaling, TokenBudget};
use crate::tensor_keys::qk_norm;

/// Gemma 3 sliding window pattern: every 6th layer (0-indexed: 5, 11, 17, ...)
/// uses full attention, the rest use sliding window.
const GEMMA3_SLIDING_WINDOW_PATTERN: usize = 6;

/// Multi-modal contract for Gemma 3.
///
/// Verified token IDs against `google/gemma-3-4b-pt/tokenizer.json`
/// (snapshot cc012e0a, multimodal-capable). The protocol describes
/// what the *LM* expects; whether the encoder weights are present in
/// any given checkpoint is a separate concern handled at encoder-load
/// time (Phase 1b+).
///
/// **Phase 1a scope**: protocol declaration only. `multimodal()` on
/// `Gemma3Arch` returns `Some(&GEMMA3_MULTIMODAL)`, but no encoder is
/// loaded and no embedding-path behaviour changes. Text-only forward
/// passes remain bit-identical to pre-Phase-1a because `embed_plan`'s
/// text-only fast path bypasses any multimodal-protocol consultation.
pub struct Gemma3MultiModal;

/// Singleton protocol — Gemma 3's multimodal contract is fixed across
/// model sizes (4B / 12B / 27B), so a single static instance suffices.
pub const GEMMA3_MULTIMODAL: Gemma3MultiModal = Gemma3MultiModal;

impl MultiModalProtocol for Gemma3MultiModal {
    fn vision_encoder(&self) -> Option<&str> {
        Some("siglip")
    }

    fn image_placeholder(&self) -> Option<PlaceholderProtocol> {
        // IDs read from tokenizer.json of google/gemma-3-4b-pt — verified
        // 2026-05-24. The model emits a fixed sandwich:
        //   <start_of_image> + 256 × <image_soft_token> + <end_of_image>
        // Host splices 256 rows of vision-projected embeddings at the
        // <image_soft_token> positions.
        Some(PlaceholderProtocol {
            start: Some(255_999),
            fill: 262_144,
            end: Some(256_000),
        })
    }

    fn image_token_budget(&self) -> TokenBudget {
        // Fixed per-image budget — Gemma 3 always expands one image to
        // exactly 256 soft tokens regardless of input resolution.
        TokenBudget::Fixed(256)
    }

    fn precomputed_scaling(&self) -> PrecomputedScaling {
        // Phase 1a default. The PaliGemma reference impl in
        // HuggingFace transformers does NOT apply embed_scale to vision
        // embeddings — only to text token embeddings. Vision goes in
        // post-projection, already scaled by the multi-modal projector.
        // Re-verify at Phase 1d (CLI integration) against caption output;
        // if Gemma 3 captions come out systematically scaled-wrong vs
        // PaliGemma, this is the first thing to flip.
        PrecomputedScaling::None
    }
}

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
        qk_norm::q(&self.layer_prefix(layer))
    }

    fn attn_k_norm_key(&self, layer: usize) -> Option<String> {
        qk_norm::k(&self.layer_prefix(layer))
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

    fn embed_scale(&self) -> Option<f32> {
        Some((self.config.hidden_size as f32).sqrt())
    }

    fn has_post_norms(&self) -> bool {
        true
    }

    /// Gemma 3's post-norms share `rms_norm_eps` with its pre-norms — see
    /// [`Gemma2Architecture::post_norm_eps`](super::gemma2::Gemma2Architecture).
    /// Declared rather than inherited: a four-norm stack that leaves the
    /// post-norm epsilon unjudged is refused.
    fn post_norm_eps(&self) -> Option<PostNormEps> {
        Some(PostNormEps::Shared)
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
        if !rs
            .scaling_type
            .eq_ignore_ascii_case(crate::ROPE_TYPE_LINEAR)
        {
            return 1.0;
        }
        if self.is_sliding_window_layer(layer) {
            1.0
        } else {
            rs.factor
        }
    }

    fn multimodal(&self) -> Option<&dyn MultiModalProtocol> {
        // Always-on for Gemma 3 — the *protocol* is part of the family
        // contract. Whether a given checkpoint actually ships SigLIP
        // weights is decided at encoder-load time (Phase 1b). Text-only
        // Gemma 3 1B variants will simply never construct an encoder;
        // the protocol's presence does not perturb their forward pass
        // because `embed_plan` only consults `multimodal()` for the
        // mixed-modality path.
        Some(&GEMMA3_MULTIMODAL)
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
            layer_rope_theta: None,
            rope_local_base: Some(10_000.0),
            sliding_window: Some(1024),
            use_sliding_window: None,
            position_embedding_type: None,
            no_rope_layers: None,
            no_rope_layer_interval: None,
            rope_interleaved: None,
            use_mrope: None,
            ffn_shape_name: None,
            is_llama_config: None,
            max_window_layers: None,
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
            has_vision_config: false,
            tie_word_embeddings: None,
            qk_scale_factor: None,
            output_multiplier: None,
            post_norm_eps: None,
            attention_bias: None,
            mlp_bias: None,
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
            linear_conv_kernel_dim: None,
            linear_key_head_dim: None,
            linear_value_head_dim: None,
            linear_num_key_heads: None,
            linear_num_value_heads: None,
            linear_attn_interleave: crate::config::DeclaredInterleave::Absent,
            mtp_interleave: crate::config::DeclaredInterleave::Absent,
            kda_geometry: None,
            kda_gate_lower_bound: None,
            router_activation: None,
            routed_scaling_factor: None,
            expert_groups: None,
            topk_group: None,
            use_grouped_topk: None,
            moe_layer_freq: None,
            first_k_dense_replace: None,
            mla_use_nope: None,
            model_max_length: None,
            d_rel: None,
            rel_extent: None,
            mamba_ssm_dtype: None,
            mamba2_geometry: None,
            mamba2_provenance: None,
            conv_qkv_attn: None,
            conv_qkv_provenance: None,
            attn_causal: None,
            pad_vocab_size_multiple: None,
            fused_add_norm: None,
            mlp_intermediate_size: None,
            mlp_padding_size: None,
            use_mlp_bias: None,
            residual_in_fp32: None,
            attn_output_gate: None,
            output_gate_type: None,
            mtp_num_hidden_layers: None,
            mtp_use_dedicated_embeddings: None,
            mrope_interleaved: None,
            mrope_section: None,
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
            yarn_beta_fast: None,
            yarn_beta_slow: None,
            yarn_truncate: None,
            yarn_mscale: None,
            yarn_mscale_all_dim: None,
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
            yarn_beta_fast: None,
            yarn_beta_slow: None,
            yarn_truncate: None,
            yarn_mscale: None,
            yarn_mscale_all_dim: None,
            gemma3_global_only: true,
        })));
        // Layers 5, 11, 17, ... are full attention; everyone else sliding.
        assert_eq!(arch.rope_position_divisor_for_layer(5), 8.0);
        assert_eq!(arch.rope_position_divisor_for_layer(4), 1.0);
    }

    // ─── Phase 1a: MultiModalProtocol contract ────────────────────────────

    #[test]
    fn multimodal_protocol_is_present_on_gemma3() {
        let arch = Gemma3Arch::from_config(synth_config(None));
        let mm = arch
            .multimodal()
            .expect("Gemma 3 must declare a multimodal protocol");
        assert_eq!(mm.vision_encoder(), Some("siglip"));
        assert_eq!(mm.audio_encoder(), None);
    }

    #[test]
    fn image_placeholder_ids_match_gemma3_tokenizer() {
        // IDs verified against google/gemma-3-4b-pt/tokenizer.json
        // (snapshot cc012e0a) — must not drift without re-verifying
        // against the upstream tokenizer.
        let arch = Gemma3Arch::from_config(synth_config(None));
        let ph = arch
            .multimodal()
            .unwrap()
            .image_placeholder()
            .expect("Gemma 3 declares an image placeholder protocol");
        assert_eq!(ph.start, Some(255_999), "<start_of_image>");
        assert_eq!(ph.fill, 262_144, "<image_soft_token>");
        assert_eq!(ph.end, Some(256_000), "<end_of_image>");
    }

    #[test]
    fn image_token_budget_is_fixed_256() {
        let arch = Gemma3Arch::from_config(synth_config(None));
        match arch.multimodal().unwrap().image_token_budget() {
            TokenBudget::Fixed(n) => assert_eq!(n, 256),
            other => panic!("expected Fixed(256), got {other:?}"),
        }
    }

    #[test]
    fn precomputed_scaling_defaults_to_none() {
        // Phase 1a default — vision rows go in post-connector with no
        // additional embed_scale applied. Phase 1d verifies against
        // PaliGemma; if wrong, flip to SameAsTokens here.
        let arch = Gemma3Arch::from_config(synth_config(None));
        assert_eq!(
            arch.multimodal().unwrap().precomputed_scaling(),
            PrecomputedScaling::None
        );
    }

    #[test]
    fn no_audio_placeholder_in_gemma3() {
        // Audio is a Gemma 4 (Phase 5) concern, not Gemma 3.
        let arch = Gemma3Arch::from_config(synth_config(None));
        assert!(arch.multimodal().unwrap().audio_placeholder().is_none());
    }

    #[test]
    fn valid_tile_counts_empty_on_fixed_budget_model() {
        // Gemma 3 uses TokenBudget::Fixed, not PerTile, so no AnyRes
        // tile-count enumeration is required.
        let arch = Gemma3Arch::from_config(synth_config(None));
        assert!(arch.multimodal().unwrap().valid_tile_counts().is_empty());
    }
}
