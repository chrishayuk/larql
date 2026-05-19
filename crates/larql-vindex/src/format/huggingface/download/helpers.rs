//! Pure helpers carved out of `download.rs` (round-6 split, 2026-05-10).
//!
//! These are network-free utilities — header parsing, repo-id filtering,
//! cache-path resolution — kept here so the network-bound code in
//! `download/mod.rs` is easier to read and so each helper can carry its
//! own dense test coverage without racing against `hf_hub` API mocks.

use std::path::PathBuf;

use super::RepoKind;

/// Normalise an HTTP ETag header to the raw content hash hf-hub uses
/// as blob filenames. Handles:
///   * strong etag: `"abc123"` → `abc123`
///   * weak etag:   `W/"abc123"` → `abc123`
pub(super) fn strip_etag_quoting(raw: &str) -> String {
    let trimmed = raw.trim();
    let no_weak = trimmed.strip_prefix("W/").unwrap_or(trimmed);
    no_weak.trim_matches('"').to_string()
}

/// Filenames that we never want to pull from a model repo even when
/// they're listed in the siblings response. PyTorch `.bin` weights are
/// skipped when safetensors are present (the standard HF mirror has
/// both); image and metadata files are noise; .gguf is a different
/// acquisition path.
pub(super) fn want_model_file(name: &str) -> bool {
    let lower = name.to_lowercase();
    if lower.ends_with(".png")
        || lower.ends_with(".jpg")
        || lower.ends_with(".jpeg")
        || lower.ends_with(".gif")
        || lower.ends_with(".svg")
        || lower.ends_with(".gguf")
        || lower.ends_with(".onnx")
        || lower == ".gitattributes"
        || lower.starts_with("readme")
        || lower.starts_with("license")
    {
        return false;
    }
    if lower.ends_with(".bin") || lower.ends_with(".pt") || lower.ends_with(".pth") {
        return false;
    }
    true
}

/// Resolve the hf-hub cache directory for a repo: the root of
/// `~/.cache/huggingface/hub/{datasets,models}--{owner}--{name}/`. Honours
/// `HF_HOME` and `HUGGINGFACE_HUB_CACHE` env overrides that hf-hub itself
/// respects.
pub(super) fn hf_cache_repo_dir(kind: RepoKind, repo_id: &str) -> Option<PathBuf> {
    let hub_root = if let Ok(hub) = std::env::var("HUGGINGFACE_HUB_CACHE") {
        PathBuf::from(hub)
    } else if let Ok(hf_home) = std::env::var("HF_HOME") {
        PathBuf::from(hf_home).join("hub")
    } else {
        let home = std::env::var("HOME").ok()?;
        PathBuf::from(home)
            .join(".cache")
            .join("huggingface")
            .join("hub")
    };
    let safe = repo_id.replace('/', "--");
    Some(hub_root.join(format!("{}{safe}", kind.cache_prefix())))
}

#[cfg(all(test, not(target_arch = "wasm32")))]
mod tests {
    use super::*;
    use serial_test::serial;

    // ─── strip_etag_quoting ────────────────────────────────────────────

    #[test]
    fn strip_etag_quoting_unquotes_strong_etag() {
        assert_eq!(strip_etag_quoting("\"abc123\""), "abc123");
    }

    #[test]
    fn strip_etag_quoting_handles_weak_etag() {
        assert_eq!(strip_etag_quoting("W/\"abc123\""), "abc123");
    }

    #[test]
    fn strip_etag_quoting_trims_surrounding_whitespace() {
        assert_eq!(strip_etag_quoting("  \"abc\"  "), "abc");
    }

    #[test]
    fn strip_etag_quoting_handles_unquoted_input() {
        assert_eq!(strip_etag_quoting("plainhash"), "plainhash");
    }

    #[test]
    fn strip_etag_quoting_handles_empty_string() {
        assert_eq!(strip_etag_quoting(""), "");
    }

    #[test]
    fn strip_etag_quoting_handles_weak_unquoted() {
        assert_eq!(strip_etag_quoting("W/abc"), "abc");
    }

    // ─── want_model_file ───────────────────────────────────────────────

    #[test]
    fn want_model_file_accepts_safetensors() {
        assert!(want_model_file("model.safetensors"));
        assert!(want_model_file("model-00001-of-00002.safetensors"));
    }

    #[test]
    fn want_model_file_accepts_config_and_tokenizer() {
        assert!(want_model_file("config.json"));
        assert!(want_model_file("tokenizer.json"));
        assert!(want_model_file("tokenizer_config.json"));
        assert!(want_model_file("special_tokens_map.json"));
    }

    #[test]
    fn want_model_file_rejects_pickle_torch_shards() {
        assert!(!want_model_file("pytorch_model.bin"));
        assert!(!want_model_file("pytorch_model-00001-of-00002.bin"));
        assert!(!want_model_file("model.pt"));
        assert!(!want_model_file("model.pth"));
    }

    #[test]
    fn want_model_file_rejects_repo_metadata_and_media() {
        for name in [
            "README.md",
            "README",
            "readme.txt",
            "LICENSE",
            "license.txt",
            ".gitattributes",
            "preview.png",
            "logo.JPG",
            "banner.jpeg",
            "demo.gif",
            "diagram.svg",
            "model.onnx",
            "weights.gguf",
        ] {
            assert!(!want_model_file(name), "should reject {name}");
        }
    }

    #[test]
    fn want_model_file_case_insensitive() {
        assert!(!want_model_file("LOGO.PNG"));
        assert!(!want_model_file("Model.GGUF"));
    }

    #[test]
    fn want_model_file_accepts_unknown_supporting_files() {
        assert!(want_model_file("generation_config.json"));
        assert!(want_model_file("chat_template.jinja"));
    }

    // ─── hf_cache_repo_dir ─────────────────────────────────────────────
    //
    // Env-var driven; serialised to avoid races on HUGGINGFACE_HUB_CACHE /
    // HF_HOME / HOME between parallel test threads.

    /// RAII guard for `(key, value)` pairs in std::env. Restores the
    /// original values on drop so neighbouring tests aren't affected.
    struct EnvSet {
        keys: Vec<(String, Option<String>)>,
    }

    impl EnvSet {
        fn new(pairs: &[(&str, Option<&str>)]) -> Self {
            let mut keys = Vec::new();
            for (k, v) in pairs {
                let prev = std::env::var(*k).ok();
                match v {
                    Some(val) => std::env::set_var(*k, val),
                    None => std::env::remove_var(*k),
                }
                keys.push((k.to_string(), prev));
            }
            Self { keys }
        }
    }

    impl Drop for EnvSet {
        fn drop(&mut self) {
            for (k, prev) in self.keys.drain(..) {
                match prev {
                    Some(v) => std::env::set_var(&k, v),
                    None => std::env::remove_var(&k),
                }
            }
        }
    }

    // Compare with `PathBuf` joins instead of forward-slash strings —
    // on Windows, `Path::join` uses `\`, which breaks raw string
    // equality. Rebuilding the expected path the same way the
    // implementation does keeps the assertion platform-portable.

    #[test]
    #[serial]
    fn hf_cache_repo_dir_uses_huggingface_hub_cache_when_set() {
        let _e = EnvSet::new(&[
            ("HUGGINGFACE_HUB_CACHE", Some("/tmp/test-hub")),
            ("HF_HOME", None),
        ]);
        let dir = hf_cache_repo_dir(RepoKind::Dataset, "owner/name").unwrap();
        let expected = std::path::PathBuf::from("/tmp/test-hub").join("datasets--owner--name");
        assert_eq!(dir, expected);
    }

    #[test]
    #[serial]
    fn hf_cache_repo_dir_uses_hf_home_when_hub_unset() {
        let _e = EnvSet::new(&[
            ("HUGGINGFACE_HUB_CACHE", None),
            ("HF_HOME", Some("/tmp/hf-home")),
        ]);
        let dir = hf_cache_repo_dir(RepoKind::Model, "owner/name").unwrap();
        let expected = std::path::PathBuf::from("/tmp/hf-home")
            .join("hub")
            .join("models--owner--name");
        assert_eq!(dir, expected);
    }

    #[test]
    #[serial]
    fn hf_cache_repo_dir_falls_back_to_home_default() {
        let _e = EnvSet::new(&[
            ("HUGGINGFACE_HUB_CACHE", None),
            ("HF_HOME", None),
            ("HOME", Some("/tmp/fallback-home")),
        ]);
        let dir = hf_cache_repo_dir(RepoKind::Dataset, "owner/repo").unwrap();
        let expected = std::path::PathBuf::from("/tmp/fallback-home")
            .join(".cache")
            .join("huggingface")
            .join("hub")
            .join("datasets--owner--repo");
        assert_eq!(dir, expected);
    }

    #[test]
    #[serial]
    fn hf_cache_repo_dir_returns_none_when_home_missing() {
        let _e = EnvSet::new(&[
            ("HUGGINGFACE_HUB_CACHE", None),
            ("HF_HOME", None),
            ("HOME", None),
        ]);
        assert!(hf_cache_repo_dir(RepoKind::Model, "owner/name").is_none());
    }

    #[test]
    #[serial]
    fn hf_cache_repo_dir_replaces_slash_in_repo_id() {
        let _e = EnvSet::new(&[("HUGGINGFACE_HUB_CACHE", Some("/tmp/x")), ("HF_HOME", None)]);
        let dir = hf_cache_repo_dir(RepoKind::Model, "complex/owner/name").unwrap();
        assert!(dir
            .to_string_lossy()
            .ends_with("models--complex--owner--name"));
    }

    #[test]
    #[serial]
    fn hf_cache_repo_dir_distinguishes_dataset_from_model() {
        let _e = EnvSet::new(&[("HUGGINGFACE_HUB_CACHE", Some("/tmp/y")), ("HF_HOME", None)]);
        let ds = hf_cache_repo_dir(RepoKind::Dataset, "x/y").unwrap();
        let md = hf_cache_repo_dir(RepoKind::Model, "x/y").unwrap();
        assert_ne!(ds, md, "RepoKind must produce distinct cache dirs");
        assert!(ds.to_string_lossy().contains("datasets--"));
        assert!(md.to_string_lossy().contains("models--"));
    }
}
