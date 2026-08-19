//! Gemma 2 architecture.
//!
//! Key differences from Gemma 3:
//! - attn_logit_softcapping (typically 50.0)
//! - final_logit_softcapping (typically 30.0)
//! - Sliding window on every *even* layer (0, 2, 4, …), full attention on
//!   every odd layer — a fixed period-2 alternation, not a declared
//!   `layer_types` interleave and not Gemma 3's stride-N pattern. Matches
//!   HF `Gemma2DecoderLayer.is_sliding = not bool(layer_idx % 2)`.
//! - No local RoPE base — sliding and full layers share the one
//!   `rope_theta` (unlike Gemma 3, which lowers the RoPE base on sliding
//!   layers)
//! - query_pre_attn_scalar may differ from head_dim

use crate::config::{Activation, ModelArchitecture, ModelConfig, PostNormEps};
use crate::tensor_keys::qk_norm;

pub struct Gemma2Arch {
    config: ModelConfig,
}

impl Gemma2Arch {
    pub fn from_config(config: ModelConfig) -> Self {
        Self { config }
    }
}

impl ModelArchitecture for Gemma2Arch {
    fn family(&self) -> &str {
        "gemma2"
    }

    fn config(&self) -> &ModelConfig {
        &self.config
    }

    fn attn_q_norm_key(&self, layer: usize) -> Option<String> {
        qk_norm::q(&self.layer_prefix(layer))
    }

    fn attn_k_norm_key(&self, layer: usize) -> Option<String> {
        qk_norm::k(&self.layer_prefix(layer))
    }

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

    /// Gemma 2's post-norms use `rms_norm_eps` — the same epsilon as its
    /// pre-norms. The checkpoint declares no separate `post_norm_eps` and
    /// the reference implementation builds all four norms from the one
    /// value, so sharing is established rather than assumed. Stated
    /// explicitly because a four-norm stack that leaves this unjudged is
    /// refused, and silence would otherwise read as "unknown".
    fn post_norm_eps(&self) -> Option<PostNormEps> {
        Some(PostNormEps::Shared)
    }

    /// Fixed period-2 alternation: even layers (0-indexed) slide at
    /// `sliding_window`, odd layers see full attention. Unlike Gemma 3's
    /// stride-N pattern this is not configurable and the checkpoint never
    /// declares it via `layer_types` — it is a property of the
    /// architecture itself, so it is judged here rather than left to the
    /// `layer_types`-or-`false` trait default (which would silently grade
    /// every layer full attention for a checkpoint that, like every real
    /// Gemma 2 release, declares no `layer_types` at all).
    fn is_sliding_window_layer(&self, layer: usize) -> bool {
        layer.is_multiple_of(2)
    }

    // rope_base_for_layer: no override — sliding and full layers share the
    // one `rope_theta` the trait default already returns.
}
