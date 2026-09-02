//! Llama-family architecture.
//!
//! Covers Llama, Mistral, Qwen, and other Llama-compatible models.
//! Uses all trait defaults (which are Llama-style).

use crate::config::{ModelArchitecture, ModelConfig};

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
}
