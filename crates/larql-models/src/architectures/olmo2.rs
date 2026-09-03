//! OLMo-2 (Allen AI) — a post-norm stack with whole-projection QK norm.
//!
//! Registered as its own identity rather than aliased onto Llama, because
//! three execution-sensitive facts differ and every one of them is
//! silent if inherited:
//!
//! 1. **Norm placement is post-norm.** Each sublayer reads the RAW
//!    residual and its OUTPUT is normalised before the add
//!    (`Olmo2DecoderLayer.forward`). Nothing here declares that — it is
//!    read from operand evidence, because the estate says it
//!    unambiguously (`post_attention_layernorm` +
//!    `post_feedforward_layernorm`, no `input_layernorm`) and a config
//!    flag would be a second authority for a fact the tensors already
//!    settle.
//!
//! 2. **QK norm is over the WHOLE projection**, not per head.
//!    `Olmo2Attention.__init__` builds `q_norm = Olmo2RMSNorm(
//!    num_attention_heads · head_dim)` and applies it to the projection
//!    output BEFORE the head reshape and before the rotary. Qwen3 and
//!    Gemma normalise each head independently over `head_dim` elements;
//!    these are different reductions over different vectors and produce
//!    different numbers. OLMoE already judged this exact operator, so
//!    this is a shared semantic named as such — not a new one, and not a
//!    default.
//!
//! 3. **`Olmo2Config.rms_norm_eps` defaults to 1e-5**, not the 1e-6 that
//!    Llama, Qwen3 and Gemma use. OLMo-2-0425-1B ships `1e-06`
//!    explicitly so the class default does not bite there — which is
//!    exactly why it is declared from the CLASS and not observed from a
//!    checkpoint. [`crate::architectures::olmoe::OlmoeArch`] records what
//!    getting this wrong costs: on OLMoE, serving 1e-6 in place of the
//!    1e-5 default moved final-residual cosine from 0.890 to 0.991.
//!
//! What this entry deliberately does NOT do is make anything else true.
//! Every fact it does not state is either declared by the config, judged
//! from operand evidence, or unresolved and therefore blocking — a
//! registry entry is a name resolving, never a licence to fill in the
//! rest.

use crate::config::{ModelArchitecture, ModelConfig, QkNormScope};
use crate::tensor_keys::qk_norm;

pub struct Olmo2Arch {
    config: ModelConfig,
}

impl Olmo2Arch {
    pub fn from_config(config: ModelConfig) -> Self {
        Self { config }
    }
}

impl ModelArchitecture for Olmo2Arch {
    /// The config's own `model_type`, echoed rather than normalised: the
    /// entry covers `olmo2` and `olmo3`, which share this decoder shape
    /// exactly, and collapsing them to one label would lose which one a
    /// report is describing.
    fn family(&self) -> &str {
        &self.config.model_type
    }

    fn config(&self) -> &ModelConfig {
        &self.config
    }

    /// See the module docs, point 3. The class default, not a checkpoint
    /// observation.
    fn default_norm_eps(&self) -> f32 {
        crate::defaults::DEFAULT_NORM_EPS_1E5
    }

    /// See the module docs, point 2 — the same operator OLMoE declares.
    fn qk_norm_scope(&self) -> QkNormScope {
        QkNormScope::FullProjection
    }

    fn attn_q_norm_key(&self, layer: usize) -> Option<String> {
        qk_norm::q(&self.layer_prefix(layer))
    }

    fn attn_k_norm_key(&self, layer: usize) -> Option<String> {
        qk_norm::k(&self.layer_prefix(layer))
    }
}
