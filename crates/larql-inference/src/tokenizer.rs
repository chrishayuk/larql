//! Tokenizer loading and helpers.

use larql_vindex::format::filenames::*;
use std::path::Path;

use larql_models::ModelArchitecture;

use crate::error::InferenceError;

/// Classic GPT-2 pre-tokenization regex. Used as the override pattern
/// when a model's `tokenizer_config.json` declares
/// `tokenizer_class: "GPT2Tokenizer"` but the shipped `tokenizer.json`
/// has been packaged with a different (cl100k-style) pre-tokenizer.
/// See [`maybe_patch_gpt2_pretok`] for the why.
const GPT2_PRETOK_REGEX: &str =
    r"'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+";

/// Load a tokenizer from a model directory.
///
/// Honours `tokenizer_config.json`'s `tokenizer_class` declaration: when
/// the config says `GPT2Tokenizer` we patch the shipped `tokenizer.json`'s
/// pre-tokenizer regex to the classic GPT-2 form before handing it to
/// `tokenizers::Tokenizer::from_bytes`. This works around models (IBM
/// Granite 4.1 3B/8B/30B) that ship a cl100k-style pre-tokenizer regex
/// in `tokenizer.json` but were actually trained against the slow
/// `GPT2Tokenizer` regex — without this, LARQL Rust tokenizes the same
/// 1 KB corpus to ~244 tokens while HF/MLX (which honour
/// `tokenizer_class`) get ~264, and the bits/char delta blows the
/// shannon-verify gate by ~40 %. See `detect/tests.rs::test_real_granite_4_1_3b`.
pub fn load_tokenizer(model_dir: &Path) -> Result<tokenizers::Tokenizer, InferenceError> {
    let path = model_dir.join(TOKENIZER_JSON);
    if !path.exists() {
        return Err(InferenceError::MissingTensor(
            "tokenizer.json not found".into(),
        ));
    }
    let raw = std::fs::read(&path).map_err(|e| InferenceError::Parse(e.to_string()))?;
    let bytes = match read_tokenizer_class(model_dir).as_deref() {
        Some("GPT2Tokenizer") => maybe_patch_gpt2_pretok(&raw).unwrap_or(raw),
        _ => raw,
    };
    tokenizers::Tokenizer::from_bytes(&bytes).map_err(|e| InferenceError::Parse(e.to_string()))
}

/// Read `tokenizer_class` from `tokenizer_config.json` if present.
/// Returns `None` when the file is absent, unreadable, or has no
/// `tokenizer_class` key — both are non-fatal: the legacy load path
/// just consumes `tokenizer.json` as shipped.
fn read_tokenizer_class(model_dir: &Path) -> Option<String> {
    let cfg_path = model_dir.join(TOKENIZER_CONFIG_JSON);
    let cfg_raw = std::fs::read(&cfg_path).ok()?;
    let cfg: serde_json::Value = serde_json::from_slice(&cfg_raw).ok()?;
    cfg.get("tokenizer_class")?.as_str().map(str::to_string)
}

/// Patch the first pre-tokenizer's `Split` regex in a `tokenizer.json`
/// payload to the classic GPT-2 pattern. Returns the rewritten JSON as
/// bytes, or `None` if the payload doesn't match the expected shape (in
/// which case the caller falls back to the unpatched file rather than
/// failing the load).
///
/// The cl100k-style regex shipped in Granite 4.1's `tokenizer.json` eats
/// a leading punctuation char into adjacent letter runs (`"re-use"` →
/// `["re", "-use"]`) and merges mixed whitespace into single pretokens
/// (`"\n    \n"` → one token). GPT-2's regex keeps them separated. Both
/// regexes share the same `'s|'t|...` and trailing `\s+(?!\S)|\s+` arms,
/// so the patch is a single-field rewrite — no other branches of the
/// `Sequence` pre-tokenizer (ByteLevel etc.) are touched.
fn maybe_patch_gpt2_pretok(raw: &[u8]) -> Option<Vec<u8>> {
    let mut tj: serde_json::Value = serde_json::from_slice(raw).ok()?;
    let regex_slot = tj
        .get_mut("pre_tokenizer")?
        .get_mut("pretokenizers")?
        .get_mut(0)?
        .get_mut("pattern")?
        .get_mut("Regex")?;
    if !regex_slot.is_string() {
        return None;
    }
    *regex_slot = serde_json::Value::String(GPT2_PRETOK_REGEX.to_string());
    serde_json::to_vec(&tj).ok()
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

#[cfg(test)]
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
    fn maybe_patch_gpt2_pretok_replaces_first_regex() {
        // Minimal tokenizer.json shape that mirrors the Granite 4.1
        // packaging: a Sequence pre_tokenizer whose first leaf is a
        // Split with a Regex pattern. The patched bytes must round-trip
        // through JSON with the GPT-2 regex installed; everything else
        // (ByteLevel sibling, model section, etc.) must round-trip
        // untouched.
        let original = serde_json::json!({
            "pre_tokenizer": {
                "type": "Sequence",
                "pretokenizers": [
                    {
                        "type": "Split",
                        "pattern": {"Regex": "shipped-cl100k-regex-here"},
                        "behavior": "Removed",
                        "invert": true,
                    },
                    {"type": "ByteLevel", "add_prefix_space": false}
                ]
            },
            "model": {"type": "BPE"},
        });
        let raw = serde_json::to_vec(&original).unwrap();
        let patched = maybe_patch_gpt2_pretok(&raw).expect("patch must succeed for known shape");
        let parsed: serde_json::Value = serde_json::from_slice(&patched).unwrap();
        let regex = parsed["pre_tokenizer"]["pretokenizers"][0]["pattern"]["Regex"]
            .as_str()
            .unwrap();
        assert_eq!(regex, GPT2_PRETOK_REGEX);
        // ByteLevel sibling untouched
        assert_eq!(
            parsed["pre_tokenizer"]["pretokenizers"][1]["type"],
            "ByteLevel"
        );
        assert_eq!(parsed["model"]["type"], "BPE");
    }

    #[test]
    fn maybe_patch_gpt2_pretok_returns_none_for_unexpected_shape() {
        // tokenizer.json without a Sequence pre_tokenizer (e.g. a
        // Llama-style Metaspace pre-tokenizer) should not be mutated —
        // we return None and the caller falls through to the unpatched
        // load. Otherwise we'd corrupt every model that happens to
        // declare tokenizer_class: GPT2Tokenizer but ships a non-Split
        // pre-tokenizer.
        let metaspace = serde_json::json!({
            "pre_tokenizer": {"type": "Metaspace", "replacement": "▁"},
            "model": {"type": "BPE"},
        });
        let raw = serde_json::to_vec(&metaspace).unwrap();
        assert!(maybe_patch_gpt2_pretok(&raw).is_none());
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

    /// `load_tokenizer` honours `tokenizer_class: "GPT2Tokenizer"` by
    /// patching the shipped tokenizer.json's first Split regex. Covers
    /// the `Some("GPT2Tokenizer")` arm of the match (line 39).
    #[test]
    fn load_tokenizer_patches_gpt2_pretok_when_tokenizer_class_says_gpt2() {
        let dir = std::env::temp_dir().join(format!(
            "larql_tokenizer_gpt2_{}_{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .map(|d| d.as_nanos())
                .unwrap_or(0)
        ));
        std::fs::create_dir_all(&dir).unwrap();

        // Minimal tokenizer.json with a Sequence pre-tokenizer.
        let tj = serde_json::json!({
            "version": "1.0",
            "truncation": null,
            "padding": null,
            "added_tokens": [],
            "normalizer": null,
            "pre_tokenizer": {
                "type": "Sequence",
                "pretokenizers": [
                    {"type": "Split", "pattern": {"Regex": "garbage-pattern"},
                     "behavior": "Removed", "invert": true},
                    {"type": "ByteLevel", "add_prefix_space": false,
                     "trim_offsets": true, "use_regex": true}
                ]
            },
            "post_processor": null,
            "decoder": null,
            "model": {"type": "WordLevel", "vocab": {"a": 0}, "unk_token": "a"}
        });
        std::fs::write(dir.join(TOKENIZER_JSON), serde_json::to_vec(&tj).unwrap()).unwrap();
        // tokenizer_config.json declares GPT2Tokenizer.
        std::fs::write(
            dir.join(TOKENIZER_CONFIG_JSON),
            br#"{"tokenizer_class":"GPT2Tokenizer"}"#,
        )
        .unwrap();

        // Without the patch, the garbage regex would make
        // `Tokenizer::from_bytes` fail. With the patch, the GPT-2
        // regex installs cleanly and the load succeeds.
        let result = load_tokenizer(&dir);
        assert!(
            result.is_ok(),
            "GPT-2-class tokenizer should patch and load: {result:?}"
        );

        let _ = std::fs::remove_file(dir.join(TOKENIZER_JSON));
        let _ = std::fs::remove_file(dir.join(TOKENIZER_CONFIG_JSON));
        let _ = std::fs::remove_dir(&dir);
    }

    /// `maybe_patch_gpt2_pretok` returns `None` when the Regex pattern
    /// field exists but isn't a string (line 78 — `return None`).
    #[test]
    fn maybe_patch_gpt2_pretok_returns_none_when_regex_is_not_string() {
        let tj = serde_json::json!({
            "pre_tokenizer": {
                "type": "Sequence",
                "pretokenizers": [
                    {"type": "Split",
                     // Regex must be a string; an object value here trips
                     // the `is_string()` check.
                     "pattern": {"Regex": {"nested": "object"}},
                     "behavior": "Removed", "invert": true},
                ]
            },
            "model": {"type": "BPE"},
        });
        let raw = serde_json::to_vec(&tj).unwrap();
        assert!(maybe_patch_gpt2_pretok(&raw).is_none());
    }

    /// `read_tokenizer_class` returns None when the config file exists
    /// but lacks the `tokenizer_class` key.
    #[test]
    fn read_tokenizer_class_returns_none_when_field_absent() {
        let dir = std::env::temp_dir().join(format!(
            "larql_tokenizer_class_absent_{}_{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .map(|d| d.as_nanos())
                .unwrap_or(0)
        ));
        std::fs::create_dir_all(&dir).unwrap();
        std::fs::write(dir.join(TOKENIZER_CONFIG_JSON), br#"{"other":"x"}"#).unwrap();
        assert!(read_tokenizer_class(&dir).is_none());
        let _ = std::fs::remove_file(dir.join(TOKENIZER_CONFIG_JSON));
        let _ = std::fs::remove_dir(&dir);
    }

    /// `read_tokenizer_class` returns None when the config file is
    /// missing (the `?` on the `read` call short-circuits).
    #[test]
    fn read_tokenizer_class_returns_none_when_file_missing() {
        let dir = std::env::temp_dir().join(format!(
            "larql_tokenizer_class_missing_{}_{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .map(|d| d.as_nanos())
                .unwrap_or(0)
        ));
        std::fs::create_dir_all(&dir).unwrap();
        assert!(read_tokenizer_class(&dir).is_none());
        let _ = std::fs::remove_dir(&dir);
    }
}
