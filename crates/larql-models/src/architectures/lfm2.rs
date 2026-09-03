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

#[cfg(test)]
mod tests {
    use super::*;

    fn arch(model_type: &str) -> Box<dyn ModelArchitecture> {
        crate::detect_from_json(&serde_json::json!({
            "model_type": model_type,
            "hidden_size": 1024,
            "num_hidden_layers": 16,
            "intermediate_size": 4608,
            "num_attention_heads": 16,
            "num_key_value_heads": 8,
            "vocab_size": 65536,
            "norm_eps": 1e-05,
            "full_attn_idxs": [2, 5, 8, 10, 12, 14]
        }))
    }

    /// Every spelling this entry carries, exercised. Each differs from
    /// the Llama default it replaces, and a spelling nothing reads is a
    /// spelling nothing is checking.
    #[test]
    fn every_dialect_spelling_is_readable() {
        let a = arch("lfm2");
        assert_eq!(a.family(), "lfm2");
        assert_eq!(a.config().hidden_size, 1024);
        assert_eq!(a.qk_norm_scope(), QkNormScope::PerHead);
        assert_eq!(
            a.attn_q_norm_key(2).as_deref(),
            Some("layers.2.self_attn.q_layernorm.weight")
        );
        assert_eq!(
            a.attn_k_norm_key(2).as_deref(),
            Some("layers.2.self_attn.k_layernorm.weight")
        );
        assert_eq!(a.attn_o_key(2), "layers.2.self_attn.out_proj.weight");
        assert_eq!(a.ffn_gate_key(2), "layers.2.feed_forward.w1.weight");
        assert_eq!(a.ffn_up_key(2), "layers.2.feed_forward.w3.weight");
        assert_eq!(a.ffn_down_key(2), "layers.2.feed_forward.w2.weight");
        assert_eq!(a.final_norm_key(), "model.embedding_norm.weight");
        // `norm_eps`, not `rms_norm_eps`.
        assert!((a.norm_eps() - 1e-5).abs() < 1e-12);
    }

    /// The MoE generation reaches the same entry and keeps its own label,
    /// so a report says which one it is describing.
    #[test]
    fn the_moe_generation_keeps_its_own_label() {
        assert_eq!(arch("lfm2_moe").family(), "lfm2_moe");
    }
}
