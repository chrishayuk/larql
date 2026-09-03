//! EXAONE-4 (LG AI Research) — a post-norm stack with PER-HEAD QK norm.
//!
//! Registered separately from [`crate::architectures::olmo2::Olmo2Arch`]
//! even though the two share a decoder shape, because the one place they
//! differ is an operator:
//!
//! | | OLMo-2 | EXAONE-4 |
//! |---|---|---|
//! | norm placement | post-norm | post-norm (identical) |
//! | QK norm | `RMSNorm(num_heads · head_dim)`, before the head reshape | `RMSNorm(head_dim)`, **after** the reshape |
//!
//! `Exaone4Attention.forward` projects, reshapes to heads, applies
//! `q_norm`/`k_norm` per head, and only then applies the rotary. That is
//! the Qwen3/Gemma reduction over `head_dim` elements, not OLMo-2's
//! single reduction over the whole projection. Aliasing either onto the
//! other would normalise a different vector and produce a running model
//! with different numbers, which is why the two families are two entries.
//!
//! `Exaone4Config.rms_norm_eps` defaults to **1e-5**, like OLMo-2's and
//! OLMoE's and unlike Llama's 1e-6. Declared from the class so a
//! checkpoint omitting the field runs at what its own reference would.
//!
//! What this entry does NOT resolve: `sliding_window_pattern` is a period
//! string (`"LLLG"`) that this schema has no field for, and it keeps
//! blocking after registration. A family entry resolves a NAME; it does
//! not license filling in what the schema cannot say.

use crate::config::{ModelArchitecture, ModelConfig, QkNormScope};
use crate::tensor_keys::qk_norm;

pub struct Exaone4Arch {
    config: ModelConfig,
}

impl Exaone4Arch {
    pub fn from_config(config: ModelConfig) -> Self {
        Self { config }
    }
}

impl ModelArchitecture for Exaone4Arch {
    fn family(&self) -> &str {
        &self.config.model_type
    }

    fn config(&self) -> &ModelConfig {
        &self.config
    }

    /// `Exaone4Config.rms_norm_eps` class default — see the module docs.
    fn default_norm_eps(&self) -> f32 {
        crate::defaults::DEFAULT_NORM_EPS_1E5
    }

    /// `Exaone4RMSNorm(self.head_dim)`, applied after the head reshape.
    /// Stated rather than left to the trait default: this is the fact
    /// that distinguishes EXAONE-4 from OLMo-2, and a reader must be
    /// able to see it was judged rather than defaulted.
    fn qk_norm_scope(&self) -> QkNormScope {
        QkNormScope::PerHead
    }

    fn attn_q_norm_key(&self, layer: usize) -> Option<String> {
        qk_norm::q(&self.layer_prefix(layer))
    }

    fn attn_k_norm_key(&self, layer: usize) -> Option<String> {
        qk_norm::k(&self.layer_prefix(layer))
    }
}
