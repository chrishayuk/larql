//! Llama-family architecture.
//!
//! Covers Llama, Mistral, Qwen, and other Llama-compatible models.
//! Uses all trait defaults (which are Llama-style).

use crate::config::{Llama3RopeScaling, ModelArchitecture, ModelConfig};
use crate::defaults::{
    LLAMA3_HIGH_FREQ_FACTOR_DEFAULT, LLAMA3_LOW_FREQ_FACTOR_DEFAULT,
    LLAMA3_ORIGINAL_MAX_POSITION_EMBEDDINGS_DEFAULT,
};

pub struct LlamaArch {
    config: ModelConfig,
}

impl LlamaArch {
    pub fn from_config(config: ModelConfig) -> Self {
        Self { config }
    }
}

impl ModelArchitecture for LlamaArch {
    fn family(&self) -> &str {
        "llama"
    }

    fn config(&self) -> &ModelConfig {
        &self.config
    }

    /// Honour `rope_scaling = {rope_type: llama3, ...}` from config.json.
    /// Returns the full wavelength-band parameter set when the loader
    /// parsed all four fields; `None` otherwise (no scaling applied).
    /// Default values fall back to the canonical Llama-3 settings if a
    /// freq factor is omitted — Llama-3.x configs always populate them
    /// in practice, but we keep the fallback so a partial config doesn't
    /// silently degrade to no scaling.
    fn llama3_rope_scaling(&self) -> Option<Llama3RopeScaling> {
        let rs = self.config.rope_scaling.as_ref()?;
        if !rs.scaling_type.eq_ignore_ascii_case("llama3") {
            return None;
        }
        Some(Llama3RopeScaling {
            factor: rs.factor,
            low_freq_factor: rs
                .llama3_low_freq_factor
                .unwrap_or(LLAMA3_LOW_FREQ_FACTOR_DEFAULT),
            high_freq_factor: rs
                .llama3_high_freq_factor
                .unwrap_or(LLAMA3_HIGH_FREQ_FACTOR_DEFAULT),
            original_max_position_embeddings: rs
                .llama3_original_max_position_embeddings
                .unwrap_or(LLAMA3_ORIGINAL_MAX_POSITION_EMBEDDINGS_DEFAULT),
        })
    }
}
