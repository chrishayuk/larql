//! Query executor: WALK, INFER, SELECT, DESCRIBE, EXPLAIN.
//!
//! Each verb lives in its own file. Shared helpers (layer-band
//! resolution, prompt tokenization) live here because multiple verbs
//! consume them.

pub(crate) mod describe;
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
/// turn gemma-4 prose into token salad. Mutation paths that capture or
/// calibrate residuals must use the same helper so their coordinates match
/// query-time inference. Legacy vindexes without a recorded `model_config`
/// fall back to the tokenizer's own output.
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
pub(crate) fn resolve_bands(config: &larql_vindex::VindexConfig) -> larql_vindex::LayerBands {
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

#[cfg(test)]
mod tests {
    use super::*;

    fn tokenizer_without_bos_postprocessor() -> larql_inference::tokenizers::Tokenizer {
        let json = serde_json::json!({
            "version": "1.0",
            "truncation": null,
            "padding": null,
            "added_tokens": [],
            "normalizer": null,
            "pre_tokenizer": {"type": "Whitespace"},
            "post_processor": null,
            "decoder": null,
            "model": {
                "type": "WordLevel",
                "vocab": {"[UNK]": 0, "<bos>": 2, "hello": 3},
                "unk_token": "[UNK]"
            }
        });
        larql_inference::tokenizers::Tokenizer::from_bytes(
            serde_json::to_vec(&json).expect("tokenizer json"),
        )
        .expect("tokenizer")
    }

    fn gemma4_config() -> larql_vindex::VindexConfig {
        larql_vindex::VindexConfig {
            family: "gemma4".into(),
            num_layers: 2,
            hidden_size: 8,
            intermediate_size: 16,
            vocab_size: 32,
            model_config: Some(larql_vindex::VindexModelConfig {
                model_type: "gemma4_text".into(),
                head_dim: 4,
                num_q_heads: 2,
                num_kv_heads: 1,
                rope_base: 10_000.0,
                ..Default::default()
            }),
            ..Default::default()
        }
    }

    #[test]
    fn encode_vindex_prompt_prepends_gemma4_bos() {
        let tokenizer = tokenizer_without_bos_postprocessor();

        let ids = encode_vindex_prompt(&gemma4_config(), &tokenizer, "hello").expect("encode");

        assert_eq!(ids, vec![2, 3]);
    }

    /// **The BOS blind spot.** `encode_vindex_prompt` (V2) consults the
    /// architecture and prepends BOS where the tokenizer's
    /// post-processor does not; the V3 encoder used to do a bare encode.
    /// On a BOS-requiring model whose tokenizer.json carries no BOS
    /// post-processor — Gemma 4 is exactly that — the two surfaces feed
    /// the model DIFFERENT token sequences for the same user prompt.
    ///
    /// The compose/parity fixtures cannot see this: their synthetic
    /// tokenizers and non-BOS architectures make both arms agree by
    /// accident. This pins the difference directly.
    /// **The control for the gate below — keep them together.**
    ///
    /// A parity gate is worth nothing unless its fixture can see the
    /// difference it claims to rule out, and this one's could not: the
    /// V2 and V3 arms diverged for months while every compose/parity
    /// test stayed green, because their synthetic tokenizers and
    /// non-BOS architectures made both arms agree by accident.
    ///
    /// This pins the fixture's discriminating power directly. Feed the
    /// V3 side no declared BOS — the pre-fix behaviour, and what a
    /// container that declares nothing still does — and the two arms
    /// MUST disagree. If a later simplification of the tokenizer or the
    /// config makes this test fail, the agreement gate below has become
    /// decorative and must be repaired, not deleted.
    #[test]
    fn the_bos_fixture_can_actually_see_a_divergence() {
        let tokenizer = tokenizer_without_bos_postprocessor();

        let v2 = encode_vindex_prompt(&gemma4_config(), &tokenizer, "hello").expect("v2 encode");
        let v3_without_the_fact =
            crate::executor::vindex3::encode_raw_prompt(&tokenizer, "hello", None)
                .expect("v3 encode");

        assert_ne!(
            v2, v3_without_the_fact,
            "the fixture must be able to distinguish the arms, or the \
             agreement gate proves nothing"
        );
        assert_eq!(v2, vec![2, 3], "V2 prepends the architecture's BOS");
        assert_eq!(v3_without_the_fact, vec![3], "an undeclared V3 does not");
    }

    #[test]
    fn v2_and_v3_prompt_encoders_agree_on_a_bos_requiring_model() {
        let tokenizer = tokenizer_without_bos_postprocessor();
        let config = gemma4_config();

        // The V3 side reads the fact the container CARRIES — the
        // checkpoint's own generation_config.json, placed beside the
        // segments by the M2 capability snapshot. A real Gemma 4
        // checkpoint declares bos_token_id 2 there, which is the same
        // id gemma4.rs's architecture hardcodes for the V2 side.
        let container = tempfile::tempdir().unwrap();
        std::fs::write(
            container.path().join("generation_config.json"),
            serde_json::json!({"bos_token_id": 2}).to_string(),
        )
        .unwrap();

        let v2 = encode_vindex_prompt(&config, &tokenizer, "hello").expect("v2 encode");
        let bos = larql_inference::vindex3::declared_prompt_bos(container.path());
        let v3 = crate::executor::vindex3::encode_raw_prompt(&tokenizer, "hello", bos)
            .expect("v3 encode");

        assert_eq!(
            v2, v3,
            "the same prompt must reach both generations as the same tokens"
        );
    }

    #[test]
    fn encode_vindex_prompt_preserves_legacy_fallback() {
        let config = larql_vindex::VindexConfig::default();
        let tokenizer = tokenizer_without_bos_postprocessor();

        let ids = encode_vindex_prompt(&config, &tokenizer, "hello").expect("encode");

        assert_eq!(ids, vec![3]);
    }
}
