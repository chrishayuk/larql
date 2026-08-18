//! Declared text-decoder features the graph represents only as ABSENT.
//!
//! Gemma 3n/4-E style knobs — the double-wide MLP on KV-shared layers and
//! the per-layer-input embedding vocabulary — that the current checkpoints
//! of interest declare OFF (`false` / a vocabulary with a zero-width
//! table). They must be READ so a checkpoint that turns them on blocks on
//! the value rather than vanishing, but they are not (yet) `ModelConfig`
//! fields: adding fields there touches every literal constructor across
//! six crates, which a parallel architecture landing is also doing. This
//! reader stores exactly what it reads and records the paths, like the
//! representation and interface readers; the facts migrate to
//! `ModelConfig` once that landing merges (see ROADMAP, V3-F0 witness 3).

use std::collections::BTreeSet;

use serde::{Deserialize, Serialize};
use serde_json::Value;

/// The text component container these live under.
pub const TEXT_CONFIG_KEY: &str = "text_config";
pub const DOUBLE_WIDE_MLP_KEY: &str = "use_double_wide_mlp";
pub const PER_LAYER_INPUT_VOCAB_KEY: &str = "vocab_size_per_layer_input";

/// What the checkpoint declares, verbatim.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct TextFeatures {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub double_wide_mlp: Option<bool>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub per_layer_input_vocab: Option<u64>,
}

impl TextFeatures {
    pub fn is_empty(&self) -> bool {
        *self == Self::default()
    }
}

/// One reading: the features plus the paths it consumed.
#[derive(Debug, Clone)]
pub struct TextFeaturesReading {
    pub features: TextFeatures,
    pub consumed_paths: BTreeSet<String>,
}

/// Read the declared features. `None` when the checkpoint declares none.
pub fn read_text_features(config: &Value) -> Option<TextFeaturesReading> {
    let text = config.get(TEXT_CONFIG_KEY)?;
    let mut consumed_paths = BTreeSet::new();
    let mut features = TextFeatures::default();
    if let Some(flag) = text.get(DOUBLE_WIDE_MLP_KEY).and_then(Value::as_bool) {
        consumed_paths.insert(format!("{TEXT_CONFIG_KEY}.{DOUBLE_WIDE_MLP_KEY}"));
        features.double_wide_mlp = Some(flag);
    }
    if let Some(vocab) = text.get(PER_LAYER_INPUT_VOCAB_KEY).and_then(Value::as_u64) {
        consumed_paths.insert(format!("{TEXT_CONFIG_KEY}.{PER_LAYER_INPUT_VOCAB_KEY}"));
        features.per_layer_input_vocab = Some(vocab);
    }
    (!features.is_empty()).then_some(TextFeaturesReading {
        features,
        consumed_paths,
    })
}

#[cfg(test)]
mod tests;
