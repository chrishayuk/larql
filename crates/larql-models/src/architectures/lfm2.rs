//! LFM2 (Liquid AI) — a two-norm pre-only stack whose every other layer
//! is a short causal convolution rather than attention.
//!
//! This entry carries LFM2's SPELLINGS and nothing else. Its placement is
//! one this build already executes — `Lfm2DecoderLayer.forward` is
//! `residual = h; h = mixer(operator_norm(h)); h = h + residual;
//! h = h + feed_forward(ffn_norm(h))`, structurally the two-norm PRE-only
//! program — so no new execution semantic is declared here and none is
//! implied.
//!
//! **What is deliberately NOT declared:** the conv mixer. LFM2 runs
//! attention only on the layers named in `full_attn_idxs` and a
//! depthwise short convolution elsewhere (`conv.conv`, `conv.in_proj`,
//! `conv.out_proj`), and that operator is not judged. Naming its tensors
//! here would let a container encode with those layers bound to
//! something they are not — the failure real-checkpoint parity caught in
//! wave 13, where an absent operand silently turned a sublayer off. The
//! conv layers keep refusing, and the encode gate keeps refusing with
//! them.
//!
//! The dialect this entry does carry, each read from the reference:
//!
//! | fact | Llama spelling | LFM2 spelling |
//! |---|---|---|
//! | attention output | `self_attn.o_proj` | `self_attn.out_proj` |
//! | QK norm | `self_attn.{q,k}_norm` | `self_attn.{q,k}_layernorm` |
//! | FFN gate/up/down | `mlp.{gate,up,down}_proj` | `feed_forward.{w1,w3,w2}` |
//! | final norm | `model.norm` | `model.embedding_norm` |
//!
//! The QK norm is `Lfm2RMSNorm(head_dim)` applied after the head reshape
//! and before the rotary — per-head, the same reduction Qwen3 and
//! EXAONE-4 use, and NOT OLMo-2's whole-projection norm. Stated rather
//! than defaulted, because it is the fact that would be silently wrong.

use crate::config::{ModelArchitecture, ModelConfig, QkNormScope};

pub struct Lfm2Arch {
    config: ModelConfig,
}

impl Lfm2Arch {
    pub fn from_config(config: ModelConfig) -> Self {
        Self { config }
    }
}

impl ModelArchitecture for Lfm2Arch {
    /// The config's own `model_type`, so a report says whether it is
    /// `lfm2` or `lfm2_moe`.
    fn family(&self) -> &str {
        &self.config.model_type
    }

    fn config(&self) -> &ModelConfig {
        &self.config
    }

    /// `Lfm2RMSNorm(head_dim)` after the head reshape — per head.
    fn qk_norm_scope(&self) -> QkNormScope {
        QkNormScope::PerHead
    }

    fn attn_q_norm_key(&self, layer: usize) -> Option<String> {
        Some(format!(
            "{}self_attn.q_layernorm.weight",
            self.layer_prefix(layer)
        ))
    }

    fn attn_k_norm_key(&self, layer: usize) -> Option<String> {
        Some(format!(
            "{}self_attn.k_layernorm.weight",
            self.layer_prefix(layer)
        ))
    }

    fn attn_o_key(&self, layer: usize) -> String {
        format!("{}self_attn.out_proj.weight", self.layer_prefix(layer))
    }

    // `w1` gates, `w3` is the up branch and `w2` reads their product —
    // the Mixtral/Kimi spelling, checked against `Lfm2MLP.forward`
    // rather than assumed from the numbering.
    fn ffn_gate_key(&self, layer: usize) -> String {
        format!("{}feed_forward.w1.weight", self.layer_prefix(layer))
    }

    fn ffn_up_key(&self, layer: usize) -> String {
        format!("{}feed_forward.w3.weight", self.layer_prefix(layer))
    }

    fn ffn_down_key(&self, layer: usize) -> String {
        format!("{}feed_forward.w2.weight", self.layer_prefix(layer))
    }

    fn final_norm_key(&self) -> &str {
        "model.embedding_norm.weight"
    }
}
