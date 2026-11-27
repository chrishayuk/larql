//! GPT-Neo's per-layer attention declaration, read into the `layer_types`
//! vocabulary every downstream reader already speaks.
//!
//! `GPTNeoConfig` takes `attention_types` — a run-length list of
//! `[[pattern], repeat]` pairs, `[[["global", "local"], 4]]` on
//! TinyStories — and expands it into `attention_layers`
//! (`expand_attention_types_params`); both are serialised. A `global`
//! layer attends causally over the whole prefix, a `local` one over the
//! trailing `window_size` positions.

use crate::config::layer_types::{
    expand_attention_types, GPT_NEO_GLOBAL, GPT_NEO_LOCAL, LAYER_TYPE_FULL_ATTENTION,
    LAYER_TYPE_SLIDING_ATTENTION,
};

/// The `layer_types` spelling of one GPT-Neo word. An unrecognised word is
/// kept verbatim, so it reaches the planner as an unexpressed span and
/// refuses there rather than resolving to either kind here.
fn layer_type_of(word: &str) -> String {
    match word {
        GPT_NEO_GLOBAL => LAYER_TYPE_FULL_ATTENTION.to_string(),
        GPT_NEO_LOCAL => LAYER_TYPE_SLIDING_ATTENTION.to_string(),
        other => other.to_string(),
    }
}

/// The per-layer types a GPT-Neo config declares, in the `layer_types`
/// vocabulary: from `attention_types` when it is present and well formed,
/// else from `attention_layers`.
pub(super) fn gpt_neo_layer_types(text_config: &serde_json::Value) -> Option<Vec<String>> {
    let words = match text_config.get("attention_types") {
        Some(types) => expand_attention_types(types)?,
        None => text_config
            .get("attention_layers")?
            .as_array()?
            .iter()
            .map(|w| w.as_str().map(str::to_string))
            .collect::<Option<Vec<_>>>()?,
    };
    Some(words.iter().map(|w| layer_type_of(w)).collect())
}
