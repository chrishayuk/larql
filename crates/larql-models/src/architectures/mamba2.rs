//! Mamba2 — a pure-SSM decoder stack, no attention anywhere.
//!
//! `model_type: "mamba2"` (transformers' HF conversion of the
//! state-spaces SSD checkpoints, e.g. `AntonV/mamba2-780m-hf`). Without
//! this file it falls to [`super::generic::GenericArch`], whose
//! attention-class defaults fabricate a softmax tower for a model that
//! never attends — 48 full-attention RoPE layers with invented 8/4 head
//! geometry, observed live on the schema-6 witness (ontology drill F1/F3).
//!
//! The overrides state three family facts and nothing else:
//!
//! - **every layer runs the Mamba2 mixer** — the `model_type` is the
//!   whole-stack declaration (a pure-SSM config writes no `layer_types`);
//! - **no layer encodes position by rotation** — position lives in the
//!   recurrence, and the checkpoint declares no rope key;
//! - **validation judges SSM geometry**, not attention geometry — head
//!   counts and `intermediate_size` are attention/FFN facts this family
//!   does not have, and requiring them here is how the fabrication
//!   started.

use crate::config::{LayerKind, ModelArchitecture, ModelConfig, PositionPolicy, RecurrenceFamily};
use crate::validation::ConfigValidationResult;

/// The `model_type` this family matches, exactly. Mamba1 (`"mamba"`) is a
/// different operator — per-channel decay, no SSD scan, no head axis —
/// and must not be read as this one.
pub const MAMBA2_MODEL_TYPE: &str = "mamba2";

pub struct Mamba2Arch {
    config: ModelConfig,
}

impl Mamba2Arch {
    pub fn from_config(config: ModelConfig) -> Self {
        Self { config }
    }
}

impl ModelArchitecture for Mamba2Arch {
    fn family(&self) -> &str {
        MAMBA2_MODEL_TYPE
    }

    fn config(&self) -> &ModelConfig {
        &self.config
    }

    /// SSM geometry closure instead of the attention-shaped default —
    /// see [`crate::validation::validate_mamba2`].
    fn validate(&self) -> ConfigValidationResult {
        crate::validation::validate_mamba2(&self.config)
    }

    /// The `model_type` declares every layer a Mamba2 mixer. Which
    /// recurrence actually runs is still identified from the declared
    /// geometry downstream: a mamba2 checkpoint whose geometry did not
    /// fully resolve declares recurrence it cannot name, and blocks as an
    /// unidentified one rather than acquiring an operator from the label.
    fn declared_uniform_layer_kind(&self) -> Option<LayerKind> {
        Some(LayerKind::Recurrent(
            if self.config.mamba2_geometry.is_some() {
                RecurrenceFamily::Mamba2
            } else {
                RecurrenceFamily::Unidentified
            },
        ))
    }

    /// No layer rotates. The trait default resolves an undeclared rope
    /// key to `Rope { theta: 10000 }` — a rotation this checkpoint never
    /// asked for, on a model with nothing to rotate.
    fn position_policy_for_layer(&self, _layer: usize) -> PositionPolicy {
        PositionPolicy::None
    }

    /// The tensor estate lives under `backbone.` (`backbone.layers.N.…`,
    /// `backbone.embeddings.weight`, `backbone.norm_f.weight`).
    fn key_prefixes_to_strip(&self) -> &[&str] {
        &["backbone."]
    }

    /// `backbone.embeddings.weight`, after prefix stripping.
    fn embed_key(&self) -> &str {
        "embeddings.weight"
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::detect_from_json;
    use serde_json::json;

    /// The real checkpoint's declaration, verbatim where it matters.
    fn mamba2_780m_config() -> serde_json::Value {
        json!({
            "model_type": "mamba2",
            "num_hidden_layers": 48,
            "hidden_size": 1536,
            "vocab_size": 50288,
            "state_size": 128,
            "num_heads": 48,
            "head_dim": 64,
            "expand": 2,
            "conv_kernel": 4,
            "n_groups": 1,
            "chunk_size": 256,
            "time_step_limit": [0.0, "Infinity"],
            "rms_norm": true,
            "use_bias": false,
            "use_conv_bias": true,
            "hidden_act": "silu",
            "layer_norm_epsilon": 1e-5,
            "residual_in_fp32": true,
            "tie_word_embeddings": true
        })
    }

    /// **A mamba2 config detects as its own family, not the generic
    /// fallback** — the fallback is where the fabricated softmax tower
    /// came from (drill F1).
    #[test]
    fn mamba2_detects_as_its_own_family_and_validates() {
        let arch = detect_from_json(&mamba2_780m_config());
        assert_eq!(arch.family(), MAMBA2_MODEL_TYPE);
        arch.validate().expect("the real geometry closes");
    }

    /// **No attention heads are fabricated.** Transformers' Mamba2Config
    /// carries no `num_attention_heads`; the parser's 8/4 attention-class
    /// defaults must not answer for a declared SSM.
    #[test]
    fn no_attention_head_geometry_is_fabricated() {
        let arch = detect_from_json(&mamba2_780m_config());
        let cfg = arch.config();
        assert_eq!(cfg.num_q_heads, 0, "no attention-head count exists");
        assert_eq!(cfg.num_kv_heads, 0, "no KV-head count exists");
        assert_eq!(cfg.head_dim, 64, "the declared value is the mixer's");
        assert!(cfg.mamba2_geometry.is_some());
        assert_eq!(cfg.residual_in_fp32, Some(true));
    }

    /// **Every layer is declared a Mamba2 recurrence, and none rotates.**
    #[test]
    fn the_model_type_declares_every_layer_and_no_rotation() {
        let arch = detect_from_json(&mamba2_780m_config());
        assert_eq!(
            arch.declared_uniform_layer_kind(),
            Some(LayerKind::Recurrent(RecurrenceFamily::Mamba2))
        );
        assert_eq!(arch.position_policy_for_layer(0), PositionPolicy::None);
        assert_eq!(arch.position_policy_for_layer(47), PositionPolicy::None);
    }

    /// **A mamba2 config with a partial geometry still declares
    /// recurrence — as one this build cannot name**, so it blocks
    /// downstream instead of acquiring an operator from the label.
    #[test]
    fn partial_geometry_declares_unidentified_recurrence() {
        let mut config = mamba2_780m_config();
        config.as_object_mut().unwrap().remove("state_size");
        let arch = detect_from_json(&config);
        assert_eq!(arch.family(), MAMBA2_MODEL_TYPE);
        assert_eq!(
            arch.declared_uniform_layer_kind(),
            Some(LayerKind::Recurrent(RecurrenceFamily::Unidentified))
        );
        assert!(
            arch.validate().is_err(),
            "a mamba2 stack without its geometry must not validate"
        );
    }

    /// **Mamba1 is not this family.** Per-channel decay with no SSD scan
    /// is a different operator; reading it as Mamba2 would bind wrong
    /// roles at plausible shapes.
    #[test]
    fn mamba1_stays_on_the_generic_path() {
        let mut config = mamba2_780m_config();
        config["model_type"] = json!("mamba");
        let arch = detect_from_json(&config);
        assert_eq!(arch.family(), "generic");
    }

    /// The estate lives under `backbone.`.
    #[test]
    fn tensor_keys_are_backbone_relative() {
        let arch = detect_from_json(&mamba2_780m_config());
        assert_eq!(arch.key_prefixes_to_strip(), &["backbone."]);
        assert_eq!(arch.embed_key(), "embeddings.weight");
    }
}
