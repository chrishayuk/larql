//! Snapshot small, useful HF metadata files from a model source dir into a
//! vindex. Keeps them side-by-side with `tokenizer.json` so the runtime
//! doesn't need a second lookup path (HF cache traversal, etc.) to find
//! things like the chat template.
//!
//! Non-fatal: if a file is missing from the source (common for GGUF-only
//! conversions), it's silently skipped. Failing to snapshot shouldn't abort
//! an otherwise-successful vindex build.

use crate::format::filenames::*;

use std::path::Path;

/// Files we opportunistically copy from the HF source directory. Names
/// match the upstream HF layout so a round-trip back to a HF-shaped model
/// dir is possible without renaming.
///
/// - `tokenizer_config.json` holds the Jinja chat template + role tokens.
/// - `special_tokens_map.json` maps logical tokens (`bos_token`, etc.) to
///   strings, used by some templates and by tokenizer diagnostics.
/// - `generation_config.json` supplies default sampling params (temperature,
///   top_p, max_new_tokens). Runtime can read it for sensible defaults.
pub const SNAPSHOT_FILES: &[&str] = &[
    TOKENIZER_CONFIG_JSON,
    "special_tokens_map.json",
    GENERATION_CONFIG_JSON,
    // Newer HF convention (Gemma 4, etc.): the chat template is a
    // standalone `chat_template.jinja` file rather than a field inside
    // `tokenizer_config.json`. Ship it alongside so the runtime can pick
    // up either location.
    "chat_template.jinja",
];

/// Copy each of [`SNAPSHOT_FILES`] from `source_dir` to `output_dir` when
/// present. Returns the list of files actually copied (empty `Vec` is a
/// valid outcome — GGUF sources have none of these). Errors only on I/O
/// failures for files that *did* exist in the source.
pub fn snapshot_hf_metadata(source_dir: &Path, output_dir: &Path) -> std::io::Result<Vec<String>> {
    let mut copied = Vec::new();
    for name in SNAPSHOT_FILES {
        let src = source_dir.join(name);
        if !src.is_file() {
            continue;
        }
        let dst = output_dir.join(name);
        std::fs::copy(&src, &dst)?;
        copied.push((*name).to_string());
    }
    Ok(copied)
}

#[cfg(all(test, not(target_arch = "wasm32")))]
mod tests {
    use super::*;
    use std::fs;

    #[test]
    fn copies_present_files_only() {
        let tmp = tempfile::tempdir().unwrap();
        let src = tmp.path().join("src");
        let dst = tmp.path().join("dst");
        fs::create_dir_all(&src).unwrap();
        fs::create_dir_all(&dst).unwrap();

        fs::write(src.join(TOKENIZER_CONFIG_JSON), r#"{"k":"v"}"#).unwrap();
        // special_tokens_map.json intentionally missing — should be skipped.
        fs::write(src.join("generation_config.json"), r#"{"t":1.0}"#).unwrap();

        let copied = snapshot_hf_metadata(&src, &dst).unwrap();
        assert_eq!(
            copied,
            vec![
                TOKENIZER_CONFIG_JSON.to_string(),
                "generation_config.json".to_string()
            ]
        );
        assert!(dst.join(TOKENIZER_CONFIG_JSON).exists());
        assert!(!dst.join("special_tokens_map.json").exists());
        assert!(dst.join("generation_config.json").exists());
    }

    #[test]
    fn empty_source_is_noop() {
        let tmp = tempfile::tempdir().unwrap();
        let src = tmp.path().join("src");
        let dst = tmp.path().join("dst");
        fs::create_dir_all(&src).unwrap();
        fs::create_dir_all(&dst).unwrap();
        let copied = snapshot_hf_metadata(&src, &dst).unwrap();
        assert!(copied.is_empty());
    }
}
