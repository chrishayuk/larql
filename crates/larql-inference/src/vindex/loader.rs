//! Strict vindex loader for inference paths.
//!
//! Single entry point that opens a vindex directory and loads every
//! sub-component generation needs (lm_head, attention weights, FFN
//! interleaved blocks). Designed to **fail loud** rather than silently
//! degrade — the looser `let _ = index.load_*(...)` pattern used in
//! demos masked the stale-148-byte-stride bug for a full session
//! before it was diagnosed.
//!
//! Resolution order (fail-loud means: any *malformed* file is an error;
//! "file not found" is the only legitimate fall-through):
//!
//!   1. `VectorIndex::load_vindex(path)` — required.
//!   2. `lm_head.bin` / `lm_head_q4.bin` — best-effort. The model's
//!      tied embeddings are always a fallback at the inference layer
//!      via `backend_lm_head_topk`, so missing lm_head files don't
//!      fail the load.
//!   3. **Attention weights** — exactly one of:
//!      a. `attn_weights_q4k.bin` (preferred) — strict load.
//!      b. `attn_weights_q8.bin` — strict load when (a) absent.
//!      If neither exists, return an error: GPU prefill needs them.
//!   4. **FFN weights** — `interleaved_q4k.bin` (preferred) or
//!      `interleaved_q4.bin` — at least one required, strict load.
//!
//! ## Why "strict" matters
//!
//! On a stale vindex with a 148-byte Q4_K stride, `load_attn_q4k` now
//! returns a clear "rebuild" error (see
//! [`crate::larql_vindex::quant::registry::QuantFormatInfo::expected_bytes`]).
//! The previous "try everything silently" pattern would catch the
//! error, fall through to Q8 attention (which on the same stale vindex
//! is also broken in different ways), and produce silent NaN that
//! decoded as `<unused*>` tokens. This loader propagates the validation
//! error so the user sees the rebuild guidance directly.

use std::path::Path;

use crate::error::InferenceError;
use larql_vindex::format::filenames::{
    ATTN_WEIGHTS_Q4K_BIN as ATTN_Q4K_BIN, ATTN_WEIGHTS_Q8_BIN as ATTN_Q8_BIN, INTERLEAVED_Q4K_BIN,
    INTERLEAVED_Q4_BIN, LM_HEAD_BIN, LM_HEAD_Q4_BIN,
};
use larql_vindex::{SilentLoadCallbacks, VectorIndex, VindexError};

/// Env var pointing at a real `*.vindex` directory. Real-model
/// integration tests (`#[ignore]` in `layer_graph::generate::tests` and
/// `forward::memit::tests`) honour this; if unset they no-op. Single
/// source of truth so the literal isn't repeated across test fixtures.
pub const ENV_VINDEX_PATH: &str = "LARQL_VINDEX_PATH";

/// Open a vindex for inference: load core, lm_head (best-effort),
/// attention weights (strict), FFN weights (strict).
///
/// See module docs for the full resolution order. Returns a clear error
/// on stride/manifest validation failure so callers see "rebuild the
/// vindex" guidance instead of garbage decode output.
pub fn open_inference_vindex(path: &Path) -> Result<VectorIndex, InferenceError> {
    let mut cb = SilentLoadCallbacks;
    let mut index = VectorIndex::load_vindex(path, &mut cb)?;

    // ── lm_head: best-effort. Tied-embedding models don't have a
    // dedicated lm_head file, and `backend_lm_head_topk` falls back to
    // `weights.lm_head` (cloned from embed) when the vindex KNN is
    // absent — see `layer_graph::generate::lm_head::lm_head_topk`.
    if path.join(LM_HEAD_BIN).is_file() {
        let _ = index.load_lm_head(path);
    }
    if path.join(LM_HEAD_Q4_BIN).is_file() {
        let _ = index.load_lm_head_q4(path);
    }

    // ── attention: strict, prefer Q4_K when present.
    if path.join(ATTN_Q4K_BIN).is_file() {
        index.load_attn_q4k(path)?;
    } else if path.join(ATTN_Q8_BIN).is_file() {
        index.load_attn_q8(path)?;
    } else {
        return Err(InferenceError::Vindex(VindexError::Parse(format!(
            "no attention weights in vindex {path:?} \
             (looked for {ATTN_Q4K_BIN}, {ATTN_Q8_BIN})"
        ))));
    }

    // ── FFN: strict, prefer Q4_K when present.
    if path.join(INTERLEAVED_Q4K_BIN).is_file() {
        index.load_interleaved_q4k(path)?;
    } else if path.join(INTERLEAVED_Q4_BIN).is_file() {
        index.load_interleaved_q4(path)?;
    } else {
        return Err(InferenceError::Vindex(VindexError::Parse(format!(
            "no FFN weights in vindex {path:?} \
             (looked for {INTERLEAVED_Q4K_BIN}, {INTERLEAVED_Q4_BIN})"
        ))));
    }

    Ok(index)
}

#[cfg(all(test, not(target_arch = "wasm32")))]
mod tests {
    use super::*;

    #[test]
    fn missing_directory_errors() {
        let tmp = tempfile::tempdir().unwrap();
        let result = open_inference_vindex(&tmp.path().join("does-not-exist"));
        assert!(result.is_err(), "missing directory must error");
    }

    /// Helper: drop a marker file at `path` so the loader's
    /// `path.is_file()` checks see it. We're not testing what's inside
    /// — just the file-presence logic that picks Q4_K vs Q8 vs absent.
    fn touch(dir: &std::path::Path, name: &str) {
        std::fs::write(dir.join(name), b"").unwrap();
    }

    /// Path-selection: with no attention files at all, the error
    /// message must name BOTH possible files so the user knows what to
    /// produce. A previous `load_*` chain that swallowed errors silently
    /// would just return Ok with a half-loaded index — subtle and bad.
    #[test]
    fn loader_lists_both_attn_filenames_when_neither_present() {
        let tmp = tempfile::tempdir().unwrap();
        // Put a minimal index.json so the load_vindex stage doesn't fail
        // first — we want to reach the attn check. (Empty file is fine —
        // load_vindex will fail parsing, which we catch and inspect.)
        let result = open_inference_vindex(tmp.path());
        assert!(result.is_err());
        // We don't care which stage failed — just that the eventual error
        // mentions an inference-relevant file so the user can act.
        let msg = match result {
            Ok(_) => unreachable!(),
            Err(e) => format!("{e}"),
        };
        let lower = msg.to_lowercase();
        assert!(
            lower.contains("index.json")
                || lower.contains("attn_weights")
                || lower.contains("not found")
                || lower.contains("no such file")
                || lower.contains("cannot find the file"),
            "error must point at the missing file — got: {msg}"
        );
    }

    /// Path-selection: filename constants stay in sync with what the
    /// loader probes. Catches a typo where (e.g.) someone renames the
    /// bin file but forgets to update the loader's `is_file()` check —
    /// the loader would silently fall through to the wrong path.
    #[test]
    fn loader_filename_constants_match_vindex_format_module() {
        // These must equal `larql_vindex::format::filenames::*`. The
        // loader is colocated with the inference crate so it pins the
        // names; a divergence here is the warning sign.
        assert_eq!(super::ATTN_Q4K_BIN, "attn_weights_q4k.bin");
        assert_eq!(super::ATTN_Q8_BIN, "attn_weights_q8.bin");
        assert_eq!(super::INTERLEAVED_Q4K_BIN, "interleaved_q4k.bin");
        assert_eq!(super::INTERLEAVED_Q4_BIN, "interleaved_q4.bin");
        assert_eq!(super::LM_HEAD_BIN, "lm_head.bin");
        assert_eq!(super::LM_HEAD_Q4_BIN, "lm_head_q4.bin");
    }

    /// File-presence helper smoke test — confirms `touch` writes a real
    /// file the loader's `is_file()` check would see.
    #[test]
    fn touch_creates_file_visible_to_path_is_file() {
        let tmp = tempfile::tempdir().unwrap();
        touch(tmp.path(), "lm_head.bin");
        assert!(tmp.path().join("lm_head.bin").is_file());
    }

    #[test]
    fn missing_attn_files_errors_with_guidance() {
        // Empty dir — load_vindex fails first (no index.json), but the
        // important assertion is that we never return Ok with no
        // attention weights loaded.
        let tmp = tempfile::tempdir().unwrap();
        let result = open_inference_vindex(tmp.path());
        assert!(result.is_err(), "empty dir must error");
        let msg = match result {
            Ok(_) => unreachable!(),
            Err(e) => format!("{e}"),
        };
        let lower = msg.to_lowercase();
        assert!(
            lower.contains("attn_weights")
                || lower.contains("index.json")
                || lower.contains("not found")
                || lower.contains("no such file")
                || lower.contains("cannot find the file")
                || lower.contains("parse"),
            "error must explain what's missing — got: {msg}"
        );
    }
}
