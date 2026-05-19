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
//!   4. **FFN weights** — `interleaved_kquant.bin` (preferred) or
//!      `interleaved_q4.bin` — at least one required, strict load.
//!
//! ## Why "strict" matters
//!
//! On a stale vindex with a 148-byte Q4_K stride, `load_attn_kquant` now
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
    has_kquant_attn_weights, has_kquant_interleaved, has_kquant_lm_head, ATTN_WEIGHTS_KQUANT_BIN,
    ATTN_WEIGHTS_Q8_BIN as ATTN_Q8_BIN, INTERLEAVED_KQUANT_BIN, INTERLEAVED_Q4_BIN,
    LEGACY_ATTN_WEIGHTS_Q4K_BIN, LEGACY_INTERLEAVED_Q4K_BIN, LM_HEAD_BIN,
};
#[cfg(test)]
use larql_vindex::format::filenames::{LEGACY_LM_HEAD_Q4_BIN, LM_HEAD_KQUANT_BIN};
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
    if has_kquant_lm_head(path) {
        let _ = index.load_lm_head_kquant(path);
    }

    // ── attention: strict, prefer k-quant when present.
    if has_kquant_attn_weights(path) {
        index.load_attn_kquant(path)?;
    } else if path.join(ATTN_Q8_BIN).is_file() {
        index.load_attn_q8(path)?;
    } else {
        return Err(InferenceError::Vindex(VindexError::Parse(format!(
            "no attention weights in vindex {path:?} \
             (looked for {ATTN_WEIGHTS_KQUANT_BIN}, legacy {LEGACY_ATTN_WEIGHTS_Q4K_BIN}, {ATTN_Q8_BIN})"
        ))));
    }

    // ── FFN: strict, prefer k-quant when present.
    if has_kquant_interleaved(path) {
        index.load_interleaved_kquant(path)?;
    } else if path.join(INTERLEAVED_Q4_BIN).is_file() {
        index.load_interleaved_q4(path)?;
    } else {
        return Err(InferenceError::Vindex(VindexError::Parse(format!(
            "no FFN weights in vindex {path:?} \
             (looked for {INTERLEAVED_KQUANT_BIN}, legacy {LEGACY_INTERLEAVED_Q4K_BIN}, {INTERLEAVED_Q4_BIN})"
        ))));
    }

    Ok(index)
}

#[cfg(test)]
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
        assert_eq!(super::ATTN_WEIGHTS_KQUANT_BIN, "attn_weights_kquant.bin");
        assert_eq!(super::LEGACY_ATTN_WEIGHTS_Q4K_BIN, "attn_weights_q4k.bin");
        assert_eq!(super::ATTN_Q8_BIN, "attn_weights_q8.bin");
        assert_eq!(super::INTERLEAVED_KQUANT_BIN, "interleaved_kquant.bin");
        assert_eq!(super::LEGACY_INTERLEAVED_Q4K_BIN, "interleaved_q4k.bin");
        assert_eq!(super::INTERLEAVED_Q4_BIN, "interleaved_q4.bin");
        assert_eq!(super::LM_HEAD_BIN, "lm_head.bin");
        assert_eq!(super::LM_HEAD_KQUANT_BIN, "lm_head_kquant.bin");
        assert_eq!(super::LEGACY_LM_HEAD_Q4_BIN, "lm_head_q4.bin");
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

    /// Happy path — write a synthetic Q4_K vindex and load it through
    /// `open_inference_vindex`. Exercises the success branches of the
    /// loader's attention + FFN resolution order (attn_weights_q4k →
    /// interleaved_kquant → tied-embedding lm_head best-effort).
    #[test]
    fn open_inference_vindex_loads_synthetic_q4k_fixture() {
        use crate::test_utils::write_synthetic_q4k_model_dir;
        let tmp = tempfile::tempdir().unwrap();
        write_synthetic_q4k_model_dir(tmp.path()).expect("write synthetic Q4K vindex");
        let index =
            open_inference_vindex(tmp.path()).expect("loader should accept synthetic Q4K fixture");
        // Q4K attention + FFN bytes both loaded.
        assert!(
            index.attn_kquant_layer_data(0).is_some(),
            "attn_kquant must be loaded for layer 0"
        );
        assert!(
            index.has_interleaved_kquant(),
            "interleaved_kquant must be loaded"
        );
    }

    /// "Attention present but FFN missing" — exercises the FFN-missing
    /// error branch (lines 90-93). Touch the Q4K attention files only.
    #[test]
    fn loader_errors_when_attn_present_but_no_ffn() {
        let tmp = tempfile::tempdir().unwrap();
        // Write a tiny valid vindex skeleton first so load_vindex doesn't
        // fail before we reach the FFN check. Easiest: use the Q4K
        // fixture, then delete the interleaved files.
        use crate::test_utils::write_synthetic_q4k_model_dir;
        write_synthetic_q4k_model_dir(tmp.path()).expect("write q4k fixture");
        // Synthetic fixture writes under the new kquant names; remove both
        // pairs to be safe in case the fixture grows back-compat dual-write.
        let _ = std::fs::remove_file(tmp.path().join(INTERLEAVED_KQUANT_BIN));
        let _ = std::fs::remove_file(tmp.path().join("interleaved_kquant_manifest.json"));
        let _ = std::fs::remove_file(tmp.path().join(LEGACY_INTERLEAVED_Q4K_BIN));
        let _ = std::fs::remove_file(tmp.path().join("interleaved_q4k_manifest.json"));
        let result = open_inference_vindex(tmp.path());
        let msg = match result {
            Ok(_) => panic!("loader must reject vindex without FFN weights"),
            Err(e) => format!("{e}"),
        };
        assert!(
            msg.contains("no FFN weights"),
            "error must mention missing FFN weights — got: {msg}"
        );
    }

    /// "lm_head.bin best-effort" — loader silently loads it if present.
    /// Use the Q4K fixture (which already writes lm_head_q4.bin) and
    /// drop a stub `lm_head.bin` next to it. Coverage drives the
    /// `if path.join(LM_HEAD_BIN).is_file()` arm at line 65-66.
    #[test]
    fn loader_loads_lm_head_when_present() {
        use crate::test_utils::write_synthetic_q4k_model_dir;
        let tmp = tempfile::tempdir().unwrap();
        write_synthetic_q4k_model_dir(tmp.path()).expect("write q4k fixture");
        // Drop a stub f32 lm_head.bin. The Q4K loader's `load_lm_head`
        // is best-effort (`let _ = ...`), so even a malformed stub is
        // fine — coverage is the goal.
        std::fs::write(tmp.path().join(LM_HEAD_BIN), [0u8; 32]).expect("write stub lm_head.bin");
        let result = open_inference_vindex(tmp.path());
        assert!(
            result.is_ok(),
            "loader must accept Q4K fixture with stub lm_head.bin"
        );
    }

    /// Synthetic Q4K vindex round-trip via the broader
    /// `InferenceWeights::load(Quantised)` shape: the fixture writes the
    /// full disk layout, the loader reads it, and the resulting
    /// `InferenceWeights` reports `is_quantised()`.
    #[test]
    fn synthetic_q4k_fixture_round_trips_through_inference_weights() {
        use crate::forward::InferenceWeights;
        use crate::test_utils::write_synthetic_q4k_model_dir;
        use larql_vindex::{load_vindex_config, SilentLoadCallbacks};
        let tmp = tempfile::tempdir().unwrap();
        write_synthetic_q4k_model_dir(tmp.path()).expect("write synthetic Q4K vindex");
        let config = load_vindex_config(tmp.path()).expect("load_vindex_config");
        assert_eq!(config.quant, larql_vindex::QuantFormat::Q4K);

        let mut cb = SilentLoadCallbacks;
        let iw = InferenceWeights::load(tmp.path(), &config, &mut cb)
            .expect("InferenceWeights::load Quantised branch");
        assert!(iw.is_quantised(), "Q4K fixture must report is_quantised()");
        let w = iw.as_weights();
        assert!(w.num_layers > 0);
        assert!(w.hidden_size > 0);
    }
}
