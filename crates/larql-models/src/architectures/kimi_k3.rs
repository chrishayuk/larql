//! Kimi K3 — a container identity whose text component declares the
//! Kimi-Linear lineage, and which this build IDENTIFIES without claiming
//! it can execute.
//!
//! **Recognition is not capability.** K3's checkpoint declares `kimi_k3`
//! at the container level and `kimi_linear` on its text component. Both
//! are true. Before this entry existed, `kimi_k3` resolved to nothing,
//! the two levels disagreed, and the identity gate refused the whole
//! model — correctly, but uninformatively: every downstream capability
//! question was blocked by one answer that named no specific missing
//! semantic.
//!
//! This entry exists so those questions can be asked individually. It is
//! the [`super::lfm2::Lfm2Arch`] posture — carry what is genuinely
//! established, refuse the rest explicitly, and let the encode gate keep
//! refusing alongside.
//!
//! # What is declared, each read from the public config
//!
//! ```text
//! num_hidden_layers          93
//! linear_attn_config         69 KDA + 24 full-attention layers, listed
//! num_experts                896      num_experts_per_token  16
//! num_shared_experts          2       first_k_dense_replace   1
//! hidden_size              7168       moe_intermediate_size 3072
//! kv_lora_rank              512       q_lora_rank           1536
//! routed_expert_hidden_size 3584      vocab_size          163840
//! quantization_config        MXFP4 routed experts; the dense families
//!                            (self_attn, shared_experts, mlp, lm_head,
//!                            vision) are IGNORED and stay BF16
//! ```
//!
//! # What is deliberately NOT declared
//!
//! ```text
//! AttnRes             residual topology, `*_res_norm` / `*_res_proj`
//! SiTU-GLU            `activation_situ_beta`, and its linear variant
//! QB frozen bias      the router's `e_score_correction_bias`
//! LatentMoE wrapping  `routed_expert_{up,down}_proj` around the bank
//! hybrid execution    KDA and MLA interleaved in one stack
//! ```
//!
//! None of those are judged here. K3 is a DELTA from its ancestor, and
//! naming these tensors without an operator behind them is exactly the
//! failure LFM2's conv mixer documents: an absent operand silently turns
//! a sublayer off, and the container encodes with layers bound to
//! something they are not.
//!
//! # The lineage declaration is not inheritance
//!
//! The registry records `KimiK3 declares text = kimi_linear`. That says
//! which architecture occupies the text slot. It does **not** say K3
//! executes as Kimi-Linear, and it grants K3 none of the ancestor's
//! semantics — the LatentMoE wrapper alone is BF16 machinery
//! Kimi-Linear does not have at all.

use crate::config::{ModelArchitecture, ModelConfig};

/// K3's container identity. Identified, not yet executable.
pub struct KimiK3Arch {
    config: ModelConfig,
}

impl KimiK3Arch {
    pub fn from_config(config: ModelConfig) -> Self {
        Self { config }
    }
}

impl ModelArchitecture for KimiK3Arch {
    /// Its own family, never the ancestor's.
    ///
    /// Returning `kimi_linear` here would make every consumer that
    /// branches on family treat K3 as the 48B model — the aliasing this
    /// whole rung exists to refuse, arriving through a getter instead of
    /// through the registry.
    fn family(&self) -> &str {
        "kimi_k3"
    }

    fn config(&self) -> &ModelConfig {
        &self.config
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// K3's config as the public checkpoint declares it, trimmed to what
    /// detection reads. `text_config.model_type` is `kimi_linear` — the
    /// lineage — and the container's own is `kimi_k3`.
    fn k3_config() -> serde_json::Value {
        serde_json::json!({
            "architectures": ["KimiK3ForConditionalGeneration"],
            "model_type": "kimi_k3",
            "text_config": {
                "model_type": "kimi_linear",
                "hidden_size": 7168,
                "num_hidden_layers": 93,
                "intermediate_size": 33792,
                "num_attention_heads": 96,
                "num_key_value_heads": 96,
                "vocab_size": 163840,
                "rms_norm_eps": 1e-5
            }
        })
    }

    /// K3's geometry, parsed the way every other entry's fixture is.
    fn flat_k3_model_config() -> crate::config::ModelConfig {
        crate::detect_from_json(&flat_k3_config()).config().clone()
    }

    /// The container alone, with no `text_config` to prefer.
    fn flat_k3_config() -> serde_json::Value {
        serde_json::json!({
            "model_type": "kimi_k3",
            "hidden_size": 7168,
            "num_hidden_layers": 93,
            "vocab_size": 163840
        })
    }

    /// The getter refuses the aliasing the registry refuses.
    ///
    /// A `kimi_linear` here would make every consumer that branches on
    /// family treat K3 as the 48B ancestor — the same substitution the
    /// component declaration is worded to prevent, arriving through an
    /// accessor instead. Asserted against BOTH strings, because a test
    /// that only checks the wanted value passes on any typo of it.
    #[test]
    fn the_family_is_its_own_and_never_the_ancestors() {
        let arch = KimiK3Arch::from_config(flat_k3_model_config());
        assert_eq!(arch.family(), "kimi_k3");
        assert_ne!(arch.family(), "kimi_linear");
    }

    /// The config it carries is the one it was built from — nothing here
    /// defaults K3's geometry into existence.
    #[test]
    fn the_container_config_is_carried_not_defaulted() {
        let arch = KimiK3Arch::from_config(flat_k3_model_config());
        assert_eq!(arch.config().num_layers, 93);
        assert_eq!(arch.config().hidden_size, 7168);
    }

    /// A flat `kimi_k3` config reaches this entry rather than the generic
    /// fallback — the dispatch arm in `detect::detect_from_json` works.
    #[test]
    fn a_flat_container_config_reaches_this_entry() {
        assert_eq!(
            crate::detect_from_json(&flat_k3_config()).family(),
            "kimi_k3"
        );
    }

    /// **OPEN GAP, pinned deliberately — the next K3 rung.**
    ///
    /// `parse_model_config` prefers `text_config.model_type` over the
    /// container's (parser.rs), which is right for a vision-language
    /// container whose text tower IS what executes. K3 is the case where
    /// it is not: `text_config.model_type` is `kimi_linear`, which is
    /// K3's LINEAGE and not something K3 can execute as. So the real
    /// checkpoint routes through this function to the 48B ancestor —
    /// exactly the aliasing this rung refuses, one layer below the
    /// registry that refuses it.
    ///
    /// Not a regression and not fixed here: the vindex3 identity gate
    /// reads the REGISTRY (`find_architecture(container)` +
    /// `declares_component`), never this path, so K3-ARCH-1's result
    /// stands. But the dispatch arm above is unreachable for the real
    /// config, and a declaration whose fate is "never fires on the model
    /// it was written for" must be recorded, not assumed working.
    ///
    /// When that rung lands, this assertion inverts to `kimi_k3`.
    #[test]
    fn the_real_nested_config_still_routes_to_the_ancestor() {
        assert_eq!(
            crate::detect_from_json(&k3_config()).family(),
            "kimi_linear"
        );
    }

    /// The ancestor still resolves to the ancestor — so the pin above
    /// cannot be read as evidence that dispatch is broken generally.
    #[test]
    fn kimi_linear_still_resolves_to_kimi_linear() {
        let mut config = k3_config();
        config["model_type"] = serde_json::json!("kimi_linear");
        assert_eq!(crate::detect_from_json(&config).family(), "kimi_linear");
    }
}
