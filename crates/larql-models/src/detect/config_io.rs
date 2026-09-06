//! Loading `config.json` from a model directory and enforcing presence
//! of fields without a defensible architecture-class default.
//!
//! Silently defaulting topology fields (the historical behaviour) makes a
//! "wrong directory" / "incomplete download" surface as a panic deep in
//! the extract pipeline (issue #22 — `could not broadcast array from
//! shape: [2560] to: [2048]`). Failing here keeps the error message
//! attached to the actual cause.

use std::path::{Path, PathBuf};

use super::ModelError;

/// HF-convention config file name read from a model directory.
pub(super) const CONFIG_FILE_NAME: &str = "config.json";

/// Nested-config wrapper used by multimodal models (Gemma 3 IT, Gemma 4).
pub(super) const CONFIG_KEY_TEXT_CONFIG: &str = "text_config";

/// Nested-config wrapper used by speech models whose backbone is a text
/// LM nested under `language_config` (MOSS-TTS-Realtime).
pub(super) const CONFIG_KEY_LANGUAGE_CONFIG: &str = "language_config";

// JSON keys for required topology fields. These have no defensible
// architecture-class default — silently substituting a guess masks real
// "wrong directory" / "incomplete download" failure modes and surfaces
// later as a broadcast/matmul panic.
// GPT-2 / GPT-style configs use legacy names (`n_embd`, `n_layer`, `n_head`,
// `n_inner`); modern HF Llama-style configs use the canonical names below.
// The parser reads from either via the alias lists.

/// Aliases for `hidden_size`. GPT-2 family uses `n_embd`.
pub(super) const CONFIG_KEY_HIDDEN_SIZE_ALIASES: &[&str] = &["hidden_size", "n_embd", "d_model"];

/// Aliases for `num_hidden_layers`. GPT-2 family uses `n_layer`.
pub(super) const CONFIG_KEY_NUM_HIDDEN_LAYERS_ALIASES: &[&str] = &["num_hidden_layers", "n_layer"];

/// Aliases for `intermediate_size`. GPT-2 sometimes sets `n_inner`; when it
/// doesn't, the parser fills in `4 * hidden_size` for `gpt2` model_type
/// (HF's model-side fallback in `GPT2Config.n_inner`).
pub(super) const CONFIG_KEY_INTERMEDIATE_SIZE_ALIASES: &[&str] = &["intermediate_size", "n_inner"];
/// A DERIVED checkpoint's per-layer dense-FFN width (E30 static shards):
/// one entry per layer, each the row count of that layer's `gate_proj` /
/// `up_proj` and the column count of its `down_proj`. LARQL's own key —
/// no upstream config spells a per-layer dense width — read under
/// `text_config` first, then at the top level, like every topology field.
pub(super) const CONFIG_KEY_FFN_INTERMEDIATE_SIZE_BY_LAYER: &str =
    "larql_ffn_intermediate_size_by_layer";

/// Aliases for `num_attention_heads`. GPT-2 family uses `n_head`.
pub(super) const CONFIG_KEY_NUM_ATTENTION_HEADS_ALIASES: &[&str] =
    &["num_attention_heads", "n_head"];

// Canonical (first-of-alias-list) names. Only consumed from the test
// module today; production code reads through the alias lists above.
#[cfg(test)]
pub(super) const CONFIG_KEY_HIDDEN_SIZE: &str = CONFIG_KEY_HIDDEN_SIZE_ALIASES[0];
#[cfg(test)]
pub(super) const CONFIG_KEY_NUM_HIDDEN_LAYERS: &str = CONFIG_KEY_NUM_HIDDEN_LAYERS_ALIASES[0];
#[cfg(test)]
pub(super) const CONFIG_KEY_INTERMEDIATE_SIZE: &str = CONFIG_KEY_INTERMEDIATE_SIZE_ALIASES[0];

/// Fields whose absence makes the config unsuitable for inferring topology.
/// Each entry is an alias list; the field counts as present when *any*
/// alias resolves under top-level or `text_config`. Topology fields have
/// no defensible architecture-class default — silently substituting one
/// masks "wrong directory" / "incomplete download" failures and surfaces
/// later as a broadcast/matmul panic.
pub(super) const REQUIRED_CONFIG_FIELDS: &[&[&str]] = &[
    CONFIG_KEY_HIDDEN_SIZE_ALIASES,
    CONFIG_KEY_NUM_HIDDEN_LAYERS_ALIASES,
    CONFIG_KEY_INTERMEDIATE_SIZE_ALIASES,
];

/// Resolve the conventional `<model_dir>/config.json` path.
pub(super) fn config_path(model_dir: &Path) -> PathBuf {
    model_dir.join(CONFIG_FILE_NAME)
}

/// Read and parse a `config.json` at the given path.
///
/// Returns [`ModelError::ConfigMissing`] when the file does not exist,
/// rather than the prior behavior of synthesising an empty `{}` and
/// letting magic-number defaults pretend the model was successfully
/// described.
pub(super) fn read_config_json(config_path: &Path) -> Result<serde_json::Value, ModelError> {
    if !config_path.exists() {
        return Err(ModelError::ConfigMissing(config_path.to_path_buf()));
    }
    let text = std::fs::read_to_string(config_path)?;
    // The judged non-finite boundary (config::nonfinite_json): bare
    // Python `Infinity`/`NaN` literals parse as the strings they spell.
    Ok(crate::config::nonfinite_json::parse_config_json(&text)?)
}

/// Fail loudly when a parsed config is missing any field whose silent
/// default would diverge from a real model's topology. Top-level, nested
/// `text_config` (multimodal) and nested `language_config` (speech)
/// layouts are all accepted; a field counts as present when *any* of its
/// aliases (e.g. `hidden_size` or `n_embd`) resolves under any layout.
pub(super) fn require_config_fields(
    config: &serde_json::Value,
    config_path: &Path,
) -> Result<(), ModelError> {
    let text_config = config
        .get(CONFIG_KEY_TEXT_CONFIG)
        .or_else(|| config.get(CONFIG_KEY_LANGUAGE_CONFIG))
        .unwrap_or(config);
    let model_type = text_config
        .get("model_type")
        .or_else(|| config.get("model_type"))
        .and_then(|v| v.as_str())
        .unwrap_or("");
    // GPT-2 doesn't ship `n_inner`; HF derives intermediate_size as
    // `4 * n_embd` at the model boundary. Skip the intermediate_size check
    // for that model_type — the parser performs the same derivation when
    // `intermediate_size` and `n_inner` are both absent.
    //
    // Mamba2 has no FFN at all — the mixer is the whole block — so an
    // `intermediate_size` genuinely does not exist to require; the
    // family's own validation judges the SSM geometry instead
    // (`validation::validate_mamba2`).
    let skip_intermediate =
        model_type == "gpt2" || model_type == crate::architectures::mamba2::MAMBA2_MODEL_TYPE;
    let missing: Vec<&'static str> = REQUIRED_CONFIG_FIELDS
        .iter()
        .filter_map(|aliases| {
            if skip_intermediate && aliases == &CONFIG_KEY_INTERMEDIATE_SIZE_ALIASES {
                return None;
            }
            let present = aliases.iter().any(|alias| {
                text_config.get(*alias).and_then(|v| v.as_u64()).is_some()
                    || config.get(*alias).and_then(|v| v.as_u64()).is_some()
            });
            if present {
                None
            } else {
                // Report the canonical (first-listed) name as the missing field.
                Some(aliases[0])
            }
        })
        .collect();
    if !missing.is_empty() {
        return Err(ModelError::ConfigFieldsMissing {
            path: config_path.to_path_buf(),
            missing,
        });
    }
    Ok(())
}

/// Read the first alias under `text_config` or top-level. Returns `None`
/// when no alias resolves to a `u64`.
pub(super) fn read_aliased_u64(
    config: &serde_json::Value,
    text_config: &serde_json::Value,
    aliases: &[&str],
) -> Option<u64> {
    aliases.iter().find_map(|key| {
        text_config
            .get(*key)
            .and_then(|v| v.as_u64())
            .or_else(|| config.get(*key).and_then(|v| v.as_u64()))
    })
}
