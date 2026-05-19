//! Tokenizer loading and helpers.

use larql_vindex::format::filenames::*;
use std::path::Path;

use larql_models::ModelArchitecture;

use crate::error::InferenceError;

/// Load a tokenizer from a model directory.
pub fn load_tokenizer(model_dir: &Path) -> Result<tokenizers::Tokenizer, InferenceError> {
    let path = model_dir.join(TOKENIZER_JSON);
    if !path.exists() {
        return Err(InferenceError::MissingTensor(
            "tokenizer.json not found".into(),
        ));
    }
    tokenizers::Tokenizer::from_file(&path).map_err(|e| InferenceError::Parse(e.to_string()))
}

/// Tokenize `prompt` with BOS prepended when the architecture requires
/// it but the tokenizer's post-processor doesn't add it (Gemma 4).
///
/// Acts as a thin wrapper over `tokenizer.encode(prompt, true)` — the
/// prepend only fires when `arch.bos_token_id()` is `Some` AND the
/// resulting encoding doesn't already start with that id. Safe to call
/// on Gemma 2/3/Llama/etc.; they return `None` and the encoding is
/// untouched.
pub fn encode_prompt(
    tokenizer: &tokenizers::Tokenizer,
    arch: &dyn ModelArchitecture,
    prompt: &str,
) -> Result<Vec<u32>, InferenceError> {
    let encoding = tokenizer
        .encode(prompt, true)
        .map_err(|e| InferenceError::Parse(format!("tokenize error: {e}")))?;
    let ids: Vec<u32> = encoding.get_ids().to_vec();
    Ok(maybe_prepend_bos(ids, arch.bos_token_id()))
}

/// Prepend `bos` to `ids` when `bos` is `Some` and the sequence doesn't
/// already start with it. Factored out of [`encode_prompt`] so callers
/// that already have token ids (e.g. from a cached encoding) can reuse
/// the logic, and so the prepend contract can be unit-tested without
/// standing up a real tokenizer.
pub(crate) fn maybe_prepend_bos(mut ids: Vec<u32>, bos: Option<u32>) -> Vec<u32> {
    if let Some(bos) = bos {
        if ids.first().copied() != Some(bos) {
            ids.insert(0, bos);
        }
    }
    ids
}

/// Decode a single token ID to a trimmed string.
pub fn decode_token(tokenizer: &tokenizers::Tokenizer, id: u32) -> Option<String> {
    tokenizer
        .decode(&[id], true)
        .ok()
        .map(|s| s.trim().to_string())
        .filter(|s| !s.is_empty())
}

/// Decode a single token ID, including special tokens (BOS, EOS, etc.).
/// Falls back to the raw vocabulary entry if normal decode produces nothing.
pub fn decode_token_raw(tokenizer: &tokenizers::Tokenizer, id: u32) -> String {
    // Try normal decode first (skip_special_tokens=true)
    if let Some(s) = decode_token(tokenizer, id) {
        return s;
    }
    // Fall back to vocabulary lookup (returns <bos>, <eos>, etc.)
    if let Some(s) = tokenizer.id_to_token(id) {
        return s;
    }
    format!("[{id}]")
}

#[cfg(all(test, not(target_arch = "wasm32")))]
mod tests {
    use super::*;

    #[test]
    fn maybe_prepend_bos_noop_when_arch_has_no_bos() {
        // Llama/Mistral/Qwen tokenizers already prepend BOS via their
        // post-processor; `arch.bos_token_id()` returns None for them and
        // the helper must leave the encoding untouched.
        let ids = vec![818, 5279, 529, 7001, 563];
        assert_eq!(maybe_prepend_bos(ids.clone(), None), ids);
    }

    #[test]
    fn maybe_prepend_bos_fires_on_gemma4_style_missing_bos() {
        // Gemma 4's tokenizer.json drops BOS — `encode(prompt, true)`
        // returns the prompt tokens with no leading id=2. The helper must
        // prepend the arch-declared BOS so attention sees the expected
        // prefix.
        let ids = vec![818, 5279, 529, 7001, 563];
        let out = maybe_prepend_bos(ids, Some(2));
        assert_eq!(out, vec![2, 818, 5279, 529, 7001, 563]);
    }

    #[test]
    fn maybe_prepend_bos_idempotent_when_already_present() {
        // Don't double-prepend when the post-processor already added BOS.
        let ids = vec![2, 818, 5279];
        assert_eq!(maybe_prepend_bos(ids.clone(), Some(2)), ids);
    }

    #[test]
    fn maybe_prepend_bos_empty_input() {
        // Empty encoding (shouldn't happen in practice, but don't panic).
        assert_eq!(maybe_prepend_bos(vec![], Some(2)), vec![2]);
        assert_eq!(maybe_prepend_bos(vec![], None), Vec::<u32>::new());
    }

    #[test]
    fn load_tokenizer_missing_file_errors() {
        let dir = std::env::temp_dir().join(format!(
            "larql_tokenizer_missing_{}_{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .map(|d| d.as_nanos())
                .unwrap_or(0)
        ));
        std::fs::create_dir_all(&dir).unwrap();
        let result = load_tokenizer(&dir);
        assert!(matches!(result, Err(InferenceError::MissingTensor(_))));
        let _ = std::fs::remove_dir(&dir);
    }

    #[test]
    fn load_tokenizer_invalid_json_returns_parse_error() {
        let dir = std::env::temp_dir().join(format!(
            "larql_tokenizer_bad_json_{}_{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .map(|d| d.as_nanos())
                .unwrap_or(0)
        ));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join(TOKENIZER_JSON);
        std::fs::write(&path, b"not valid json").unwrap();
        let result = load_tokenizer(&dir);
        assert!(matches!(result, Err(InferenceError::Parse(_))));
        let _ = std::fs::remove_file(&path);
        let _ = std::fs::remove_dir(&dir);
    }

    #[test]
    fn encode_prompt_returns_token_ids() {
        // Use the synthetic test tokenizer so we don't need a real model on disk.
        let tok = crate::test_utils::make_test_tokenizer(32);
        let arch_json = serde_json::json!({
            "model_type": "tinymodel",
            "hidden_size": 16,
            "num_hidden_layers": 2,
            "intermediate_size": 32,
            "head_dim": 8,
            "num_attention_heads": 2,
            "num_key_value_heads": 1,
            "vocab_size": 32,
        });
        let arch = larql_models::detect_from_json(&arch_json);
        // Synthetic tokenizer's vocabulary words are "[N]" — single-token
        // prompt should encode to exactly one id.
        let result = encode_prompt(&tok, &*arch, "[5]");
        let ids = result.expect("encode must succeed for in-vocab prompt");
        assert!(!ids.is_empty());
    }

    #[test]
    fn decode_token_returns_some_for_valid_id() {
        let tok = crate::test_utils::make_test_tokenizer(32);
        // Token 5 should decode to "[5]" (per the synthetic tokenizer's
        // vocab map). decode_token trims whitespace.
        let s = decode_token(&tok, 5);
        assert!(s.is_some(), "decode_token must succeed for valid id");
    }

    #[test]
    fn decode_token_raw_falls_back_to_vocab_lookup() {
        let tok = crate::test_utils::make_test_tokenizer(32);
        // For an in-vocab id the normal decode succeeds and we get the
        // surface form.
        let s = decode_token_raw(&tok, 7);
        assert!(!s.is_empty());
    }

    #[test]
    fn decode_token_raw_format_fallback_for_out_of_vocab_id() {
        let tok = crate::test_utils::make_test_tokenizer(32);
        // Id 9999 is out of vocab — both decode_token and id_to_token
        // return None → format!("[{id}]") fallback.
        let s = decode_token_raw(&tok, 9999);
        assert_eq!(s, "[9999]");
    }
}
