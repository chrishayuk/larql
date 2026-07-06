//! Query executor: WALK, INFER, SELECT, DESCRIBE, EXPLAIN.
//!
//! Each verb lives in its own file. Shared helpers (layer-band
//! resolution, prompt tokenization) live here because multiple verbs
//! consume them.

mod describe;
mod explain;
mod infer;
mod infer_trace;
mod select;
mod walk;

use crate::error::LqlError;

/// Tokenize an LQL prompt against a vindex, prepending BOS when the
/// architecture requires it but the tokenizer's post-processor doesn't
/// add it (Gemma 4). Every text-prompt query path must route through
/// this (or [`encode_dense_prompt`]) rather than calling
/// `tokenizer.encode` directly — a silently missing BOS is enough to
/// turn gemma-4 prose into token salad. Legacy vindexes without a
/// recorded `model_config` fall back to the tokenizer's own output.
pub(super) fn encode_vindex_prompt(
    config: &larql_vindex::VindexConfig,
    tokenizer: &larql_inference::tokenizers::Tokenizer,
    prompt: &str,
) -> Result<Vec<u32>, LqlError> {
    match larql_vindex::arch_from_vindex_config(config) {
        Some(arch) => larql_inference::encode_prompt(tokenizer, arch.as_ref(), prompt)
            .map_err(|e| LqlError::exec("tokenize error", e)),
        None => {
            let encoding = tokenizer
                .encode(prompt, true)
                .map_err(|e| LqlError::exec("tokenize error", e))?;
            Ok(encoding.get_ids().to_vec())
        }
    }
}

/// [`encode_vindex_prompt`] for the dense Weight backend, where the
/// loaded `ModelWeights` already carries the detected architecture.
pub(super) fn encode_dense_prompt(
    weights: &larql_inference::ModelWeights,
    tokenizer: &larql_inference::tokenizers::Tokenizer,
    prompt: &str,
) -> Result<Vec<u32>, LqlError> {
    larql_inference::encode_prompt(tokenizer, weights.arch.as_ref(), prompt)
        .map_err(|e| LqlError::exec("tokenize error", e))
}

/// Resolve the layer-band boundaries from the vindex config, with a
/// family-based default and a final whole-range fallback.
pub(super) fn resolve_bands(config: &larql_vindex::VindexConfig) -> larql_vindex::LayerBands {
    let last = config.num_layers.saturating_sub(1);
    config
        .layer_bands
        .clone()
        .or_else(|| larql_vindex::LayerBands::for_family(&config.family, config.num_layers))
        .unwrap_or(larql_vindex::LayerBands {
            syntax: (0, last),
            knowledge: (0, last),
            output: (0, last),
        })
}
