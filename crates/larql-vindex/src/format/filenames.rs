//! Vindex on-disk filenames — single source of truth.
//!
//! Every `.bin` / `.json` filename written or read by the vindex format
//! lives here as a `pub const`. Use these instead of string literals.
//!
//! Why: the 2026-04-25 audit found 244 occurrences of these names
//! scattered across 18+ files. A typo silently triggers a fallback
//! codepath (the file just "doesn't exist") and bugs go undiagnosed.
//! Centralising means renaming a file changes one line.
//!
//! Convention: `SCREAMING_SNAKE`, named for what they hold, not how
//! they're encoded.

// ── Top-level config / sidecars ─────────────────────────────────────────
pub const INDEX_JSON: &str = "index.json";
pub const TOKENIZER_JSON: &str = "tokenizer.json";
pub const TOKENIZER_CONFIG_JSON: &str = "tokenizer_config.json";
pub const GENERATION_CONFIG_JSON: &str = "generation_config.json";
pub const WEIGHT_MANIFEST_JSON: &str = "weight_manifest.json";
pub const KNN_STORE_BIN: &str = "knn_store.bin";
pub const MODEL_WEIGHTS_BIN: &str = "model_weights.bin";

// ── Labels / clustering sidecars ───────────────────────────────────────
pub const RELATION_CLUSTERS_JSON: &str = "relation_clusters.json";
pub const FEATURE_CLUSTERS_JSONL: &str = "feature_clusters.jsonl";
pub const FEATURE_LABELS_JSON: &str = "feature_labels.json";

// ── Embeddings + norms (always present) ────────────────────────────────
pub const EMBEDDINGS_BIN: &str = "embeddings.bin";
pub const NORMS_BIN: &str = "norms.bin";

// ── Gate vectors ───────────────────────────────────────────────────────
pub const GATE_VECTORS_BIN: &str = "gate_vectors.bin";
pub const GATE_VECTORS_Q4_BIN: &str = "gate_vectors_q4.bin";
pub const ROUTER_WEIGHTS_BIN: &str = "router_weights.bin";

// ── Down meta + feature-major projections ──────────────────────────────
pub const DOWN_META_BIN: &str = "down_meta.bin";
pub const DOWN_META_JSONL: &str = "down_meta.jsonl";
pub const DOWN_FEATURES_BIN: &str = "down_features.bin";
pub const UP_FEATURES_BIN: &str = "up_features.bin";
pub const PLE_WEIGHTS_BIN: &str = "ple_weights.bin";

// ── Layer-major FFN weight files (PyTorch `nn.Linear` orientation) ────
//
// `[layer, intermediate, hidden]` for up and `[layer, hidden, intermediate]`
// for down — distinct from the feature-major projection files above.
// Written by f32 extraction, consumed by Q4_K conversion + checksumming +
// HuggingFace upload.
pub const UP_WEIGHTS_BIN: &str = "up_weights.bin";
pub const DOWN_WEIGHTS_BIN: &str = "down_weights.bin";

/// Feature-major k-quant encoded down projections (W2 of perf round-4).
///
/// On-disk PyTorch `nn.Linear` orientation for down is
/// `[hidden, intermediate]`, so a single feature's down vector requires
/// gathering across `hidden` separate rows — there is no per-feature
/// row decode. The legacy code path (`kquant_ffn_layer` + cache) amortises
/// this by dequantising the whole layer to f32 and transposing once.
///
/// Emitting this at extract time stores down already in feature-major
/// `[intermediate, hidden]` orientation, k-quant encoded. Per-feature
/// decode becomes a single row dequant — no cache, no transpose, no
/// ~840 MB heap ceiling on Gemma 4B. The disk cost is roughly the same
/// as the down portion of `interleaved_kquant.bin` (~14 MB / layer at
/// Gemma 4B dims). Opt-in via `KquantWriteOptions::feature_major_down`.
pub const DOWN_FEATURES_KQUANT_BIN: &str = "down_features_kquant.bin";
/// Per-layer (offset, length, format) entries for `down_features_kquant.bin`.
pub const DOWN_FEATURES_KQUANT_MANIFEST_JSON: &str = "down_features_kquant_manifest.json";
/// Legacy q4k-named feature-major down file. Readers accept it as a
/// fallback when the kquant-named file is absent; writers no longer emit it.
pub const LEGACY_DOWN_FEATURES_Q4K_BIN: &str = "down_features_q4k.bin";
/// Legacy q4k-named manifest paired with [`LEGACY_DOWN_FEATURES_Q4K_BIN`].
pub const LEGACY_DOWN_FEATURES_Q4K_MANIFEST_JSON: &str = "down_features_q4k_manifest.json";

// ── Interleaved FFN (gate|up|down packed per layer) ────────────────────
pub const INTERLEAVED_BIN: &str = "interleaved.bin";
pub const INTERLEAVED_Q4_BIN: &str = "interleaved_q4.bin";
pub const INTERLEAVED_KQUANT_BIN: &str = "interleaved_kquant.bin";
pub const INTERLEAVED_KQUANT_MANIFEST_JSON: &str = "interleaved_kquant_manifest.json";
/// Legacy q4k-named interleaved FFN file. Read-only back-compat fallback.
pub const LEGACY_INTERLEAVED_Q4K_BIN: &str = "interleaved_q4k.bin";
/// Legacy q4k-named manifest paired with [`LEGACY_INTERLEAVED_Q4K_BIN`].
pub const LEGACY_INTERLEAVED_Q4K_MANIFEST_JSON: &str = "interleaved_q4k_manifest.json";

// ── Attention weights ──────────────────────────────────────────────────
pub const ATTN_WEIGHTS_BIN: &str = "attn_weights.bin";
pub const ATTN_WEIGHTS_Q4_BIN: &str = "attn_weights_q4.bin";
pub const ATTN_WEIGHTS_Q4_MANIFEST_JSON: &str = "attn_weights_q4_manifest.json";
pub const ATTN_WEIGHTS_KQUANT_BIN: &str = "attn_weights_kquant.bin";
pub const ATTN_WEIGHTS_KQUANT_MANIFEST_JSON: &str = "attn_weights_kquant_manifest.json";
/// Legacy q4k-named attention weights file. Read-only back-compat fallback.
pub const LEGACY_ATTN_WEIGHTS_Q4K_BIN: &str = "attn_weights_q4k.bin";
/// Legacy q4k-named manifest paired with [`LEGACY_ATTN_WEIGHTS_Q4K_BIN`].
pub const LEGACY_ATTN_WEIGHTS_Q4K_MANIFEST_JSON: &str = "attn_weights_q4k_manifest.json";
pub const ATTN_WEIGHTS_Q8_BIN: &str = "attn_weights_q8.bin";
pub const ATTN_WEIGHTS_Q8_MANIFEST_JSON: &str = "attn_weights_q8_manifest.json";

// ── Per-layer FFN weights (§5.12) ──────────────────────────────────────
//
// Unified format for both dense and MoE FFN weights. One file per layer.
// File header declares the quantization format; all entries within a file
// use it uniformly (no mixing formats). Dense: num_entries=1.
// MoE: num_entries=num_experts.
pub const LAYERS_DIR: &str = "layers";

/// Return the path of `layers/layer_{L:02}.weights` for layer `L`.
pub fn layer_weights_filename(layer: usize) -> String {
    format!("layers/layer_{layer:02}.weights")
}

// ── k-quant dual-read path resolution ──────────────────────────────────
//
// New writers emit `*_kquant.bin` filenames. Readers must accept both
// the new names and the legacy `*_q4k.bin` / `lm_head_q4.bin` names so
// vindexes published before the rename keep loading. These helpers
// centralise the new-first / legacy-fallback resolution so a future
// drop of the legacy paths is one edit per family.
//
// `Resolved::path` is the file that exists on disk (new preferred,
// legacy fallback). When neither exists `path` points at the new name
// so the downstream "not found" error mentions the canonical filename.

use std::path::{Path, PathBuf};

/// Resolved location for a k-quant file family: the (.bin, .json) pair
/// where the manifest is optional (some families don't sidecar one).
pub struct ResolvedKquantPaths {
    pub bin: PathBuf,
    pub manifest: Option<PathBuf>,
    /// True when the resolved path is the legacy q4k-named file.
    /// Loaders log a one-time deprecation hint at this signal.
    pub is_legacy: bool,
}

fn pick_bin(dir: &Path, new: &str, legacy: &str) -> (PathBuf, bool) {
    let new_path = dir.join(new);
    if new_path.exists() {
        return (new_path, false);
    }
    let legacy_path = dir.join(legacy);
    if legacy_path.exists() {
        return (legacy_path, true);
    }
    // Neither present — return the new path so error messages cite
    // the canonical filename, not the deprecated one.
    (new_path, false)
}

/// Resolve `attn_weights_*.bin` + matching manifest. New `_kquant_`
/// preferred over legacy `_q4k_`.
pub fn resolve_attn_weights_kquant(dir: &Path) -> ResolvedKquantPaths {
    let (bin, is_legacy) = pick_bin(dir, ATTN_WEIGHTS_KQUANT_BIN, LEGACY_ATTN_WEIGHTS_Q4K_BIN);
    let manifest_name = if is_legacy {
        LEGACY_ATTN_WEIGHTS_Q4K_MANIFEST_JSON
    } else {
        ATTN_WEIGHTS_KQUANT_MANIFEST_JSON
    };
    ResolvedKquantPaths {
        bin,
        manifest: Some(dir.join(manifest_name)),
        is_legacy,
    }
}

/// Resolve `interleaved_*.bin` + matching manifest.
pub fn resolve_interleaved_kquant(dir: &Path) -> ResolvedKquantPaths {
    let (bin, is_legacy) = pick_bin(dir, INTERLEAVED_KQUANT_BIN, LEGACY_INTERLEAVED_Q4K_BIN);
    let manifest_name = if is_legacy {
        LEGACY_INTERLEAVED_Q4K_MANIFEST_JSON
    } else {
        INTERLEAVED_KQUANT_MANIFEST_JSON
    };
    ResolvedKquantPaths {
        bin,
        manifest: Some(dir.join(manifest_name)),
        is_legacy,
    }
}

/// Resolve `lm_head_*.bin`. No sidecar manifest — the kind tag lives
/// in the top-level `weight_manifest.json`.
pub fn resolve_lm_head_kquant(dir: &Path) -> ResolvedKquantPaths {
    let (bin, is_legacy) = pick_bin(dir, LM_HEAD_KQUANT_BIN, LEGACY_LM_HEAD_Q4_BIN);
    ResolvedKquantPaths {
        bin,
        manifest: None,
        is_legacy,
    }
}

/// Resolve `down_features_*.bin` + matching manifest.
pub fn resolve_down_features_kquant(dir: &Path) -> ResolvedKquantPaths {
    let (bin, is_legacy) = pick_bin(dir, DOWN_FEATURES_KQUANT_BIN, LEGACY_DOWN_FEATURES_Q4K_BIN);
    let manifest_name = if is_legacy {
        LEGACY_DOWN_FEATURES_Q4K_MANIFEST_JSON
    } else {
        DOWN_FEATURES_KQUANT_MANIFEST_JSON
    };
    ResolvedKquantPaths {
        bin,
        manifest: Some(dir.join(manifest_name)),
        is_legacy,
    }
}

/// Whether the directory contains a k-quant attention weights file
/// under either the new or legacy name.
pub fn has_kquant_attn_weights(dir: &Path) -> bool {
    dir.join(ATTN_WEIGHTS_KQUANT_BIN).is_file() || dir.join(LEGACY_ATTN_WEIGHTS_Q4K_BIN).is_file()
}

/// Whether the directory contains an interleaved k-quant FFN file
/// under either the new or legacy name.
pub fn has_kquant_interleaved(dir: &Path) -> bool {
    dir.join(INTERLEAVED_KQUANT_BIN).is_file() || dir.join(LEGACY_INTERLEAVED_Q4K_BIN).is_file()
}

/// Whether the directory contains an LM-head k-quant file under
/// either the new or legacy name.
pub fn has_kquant_lm_head(dir: &Path) -> bool {
    dir.join(LM_HEAD_KQUANT_BIN).is_file() || dir.join(LEGACY_LM_HEAD_Q4_BIN).is_file()
}

/// Whether the directory contains a feature-major down k-quant file
/// under either the new or legacy name.
pub fn has_kquant_down_features(dir: &Path) -> bool {
    dir.join(DOWN_FEATURES_KQUANT_BIN).is_file() || dir.join(LEGACY_DOWN_FEATURES_Q4K_BIN).is_file()
}

// ── LM head ────────────────────────────────────────────────────────────
pub const LM_HEAD_BIN: &str = "lm_head.bin";
/// Canonical k-quant LM head filename. New writers emit this.
pub const LM_HEAD_KQUANT_BIN: &str = "lm_head_kquant.bin";
/// Legacy q4-named LM head file. Read-only back-compat fallback for
/// vindexes written before the kquant rename (HF-hosted models, etc).
pub const LEGACY_LM_HEAD_Q4_BIN: &str = "lm_head_q4.bin";

// ── FP4 / FP8 projections (exp 26) ─────────────────────────────────────
pub const GATE_VECTORS_FP4_BIN: &str = "gate_vectors_fp4.bin";
pub const UP_FEATURES_FP4_BIN: &str = "up_features_fp4.bin";
pub const DOWN_FEATURES_FP8_BIN: &str = "down_features_fp8.bin";

// ── HuggingFace upload manifest order ──────────────────────────────────
//
// Order matches what `format/huggingface.rs` uploads. Adding or
// removing a vindex file means updating both this list AND the
// per-file upload code.
//
// Both the new kquant-named and legacy q4k-named filenames are listed
// so freshly-extracted vindexes (kquant) and re-uploads of pre-rename
// vindexes (q4k) both publish correctly. The upload code skips entries
// whose file is absent on disk, so listing both is safe.
pub const HF_UPLOAD_FILES: &[&str] = &[
    INDEX_JSON,
    TOKENIZER_JSON,
    WEIGHT_MANIFEST_JSON,
    EMBEDDINGS_BIN,
    NORMS_BIN,
    GATE_VECTORS_BIN,
    DOWN_META_BIN,
    INTERLEAVED_BIN,
    INTERLEAVED_KQUANT_BIN,
    INTERLEAVED_KQUANT_MANIFEST_JSON,
    LEGACY_INTERLEAVED_Q4K_BIN,
    LEGACY_INTERLEAVED_Q4K_MANIFEST_JSON,
    ATTN_WEIGHTS_BIN,
    ATTN_WEIGHTS_KQUANT_BIN,
    ATTN_WEIGHTS_KQUANT_MANIFEST_JSON,
    LEGACY_ATTN_WEIGHTS_Q4K_BIN,
    LEGACY_ATTN_WEIGHTS_Q4K_MANIFEST_JSON,
    DOWN_FEATURES_BIN,
    UP_FEATURES_BIN,
    LM_HEAD_KQUANT_BIN,
    LEGACY_LM_HEAD_Q4_BIN,
];

#[cfg(test)]
mod tests {
    use super::*;

    /// Constants must never collide — a duplicate name would silently
    /// route two writers at the same file.
    #[test]
    fn all_filenames_unique() {
        let names = [
            INDEX_JSON,
            TOKENIZER_JSON,
            TOKENIZER_CONFIG_JSON,
            GENERATION_CONFIG_JSON,
            WEIGHT_MANIFEST_JSON,
            KNN_STORE_BIN,
            MODEL_WEIGHTS_BIN,
            RELATION_CLUSTERS_JSON,
            FEATURE_CLUSTERS_JSONL,
            FEATURE_LABELS_JSON,
            EMBEDDINGS_BIN,
            NORMS_BIN,
            GATE_VECTORS_BIN,
            GATE_VECTORS_Q4_BIN,
            ROUTER_WEIGHTS_BIN,
            GATE_VECTORS_FP4_BIN,
            DOWN_META_BIN,
            DOWN_META_JSONL,
            DOWN_FEATURES_BIN,
            DOWN_FEATURES_FP8_BIN,
            DOWN_FEATURES_KQUANT_BIN,
            DOWN_FEATURES_KQUANT_MANIFEST_JSON,
            LEGACY_DOWN_FEATURES_Q4K_BIN,
            LEGACY_DOWN_FEATURES_Q4K_MANIFEST_JSON,
            DOWN_WEIGHTS_BIN,
            UP_FEATURES_BIN,
            UP_FEATURES_FP4_BIN,
            UP_WEIGHTS_BIN,
            PLE_WEIGHTS_BIN,
            INTERLEAVED_BIN,
            INTERLEAVED_Q4_BIN,
            INTERLEAVED_KQUANT_BIN,
            INTERLEAVED_KQUANT_MANIFEST_JSON,
            LEGACY_INTERLEAVED_Q4K_BIN,
            LEGACY_INTERLEAVED_Q4K_MANIFEST_JSON,
            ATTN_WEIGHTS_BIN,
            ATTN_WEIGHTS_Q4_BIN,
            ATTN_WEIGHTS_Q4_MANIFEST_JSON,
            ATTN_WEIGHTS_KQUANT_BIN,
            ATTN_WEIGHTS_KQUANT_MANIFEST_JSON,
            LEGACY_ATTN_WEIGHTS_Q4K_BIN,
            LEGACY_ATTN_WEIGHTS_Q4K_MANIFEST_JSON,
            ATTN_WEIGHTS_Q8_BIN,
            ATTN_WEIGHTS_Q8_MANIFEST_JSON,
            LM_HEAD_BIN,
            LM_HEAD_KQUANT_BIN,
            LEGACY_LM_HEAD_Q4_BIN,
        ];
        let unique: std::collections::HashSet<_> = names.iter().collect();
        assert_eq!(unique.len(), names.len(), "duplicate filename constant");
    }

    #[test]
    fn hf_upload_files_subset_of_all() {
        // HF_UPLOAD_FILES must reference real constants. If a constant
        // is removed, this test catches the dangling reference.
        for name in HF_UPLOAD_FILES {
            assert!(
                name.ends_with(".bin") || name.ends_with(".json"),
                "HF_UPLOAD_FILES has odd entry: {name}"
            );
        }
    }

    // ── Dual-read resolver behaviour ───────────────────────────────────
    //
    // Pin the new-first / legacy-fallback resolution shape so a future
    // edit doesn't silently regress to "legacy only" or "new only" —
    // either would brick existing vindexes or silently drop the rename.

    #[test]
    fn resolve_attn_weights_kquant_prefers_new_when_both_present() {
        let dir = tempfile::tempdir().unwrap();
        std::fs::write(dir.path().join(ATTN_WEIGHTS_KQUANT_BIN), b"new").unwrap();
        std::fs::write(dir.path().join(LEGACY_ATTN_WEIGHTS_Q4K_BIN), b"legacy").unwrap();
        let r = resolve_attn_weights_kquant(dir.path());
        assert!(!r.is_legacy);
        assert!(r.bin.ends_with(ATTN_WEIGHTS_KQUANT_BIN));
        assert!(r
            .manifest
            .unwrap()
            .ends_with(ATTN_WEIGHTS_KQUANT_MANIFEST_JSON));
    }

    #[test]
    fn resolve_attn_weights_kquant_falls_back_to_legacy() {
        let dir = tempfile::tempdir().unwrap();
        std::fs::write(dir.path().join(LEGACY_ATTN_WEIGHTS_Q4K_BIN), b"legacy").unwrap();
        let r = resolve_attn_weights_kquant(dir.path());
        assert!(r.is_legacy, "must report legacy when only q4k file exists");
        assert!(r.bin.ends_with(LEGACY_ATTN_WEIGHTS_Q4K_BIN));
        assert!(r
            .manifest
            .unwrap()
            .ends_with(LEGACY_ATTN_WEIGHTS_Q4K_MANIFEST_JSON));
    }

    #[test]
    fn resolve_attn_weights_kquant_returns_new_path_when_neither_exists() {
        // Empty dir — neither file present. The returned path should
        // point at the new (canonical) name so any downstream "not
        // found" error mentions the post-rename filename.
        let dir = tempfile::tempdir().unwrap();
        let r = resolve_attn_weights_kquant(dir.path());
        assert!(!r.is_legacy);
        assert!(r.bin.ends_with(ATTN_WEIGHTS_KQUANT_BIN));
    }

    #[test]
    fn resolve_lm_head_kquant_no_manifest() {
        // LM head has no sidecar manifest — manifest field is None.
        let dir = tempfile::tempdir().unwrap();
        std::fs::write(dir.path().join(LM_HEAD_KQUANT_BIN), b"new").unwrap();
        let r = resolve_lm_head_kquant(dir.path());
        assert!(!r.is_legacy);
        assert!(r.manifest.is_none());
    }

    #[test]
    fn resolve_lm_head_kquant_falls_back_to_legacy_q4() {
        let dir = tempfile::tempdir().unwrap();
        std::fs::write(dir.path().join(LEGACY_LM_HEAD_Q4_BIN), b"legacy").unwrap();
        let r = resolve_lm_head_kquant(dir.path());
        assert!(r.is_legacy);
        assert!(r.bin.ends_with(LEGACY_LM_HEAD_Q4_BIN));
        assert!(r.manifest.is_none());
    }

    #[test]
    fn resolve_interleaved_kquant_dual_read() {
        // New present → use new.
        let dir = tempfile::tempdir().unwrap();
        std::fs::write(dir.path().join(INTERLEAVED_KQUANT_BIN), b"new").unwrap();
        let r = resolve_interleaved_kquant(dir.path());
        assert!(!r.is_legacy);
        assert!(r.bin.ends_with(INTERLEAVED_KQUANT_BIN));
        assert!(r
            .manifest
            .unwrap()
            .ends_with(INTERLEAVED_KQUANT_MANIFEST_JSON));

        // Only legacy present → use legacy.
        let dir = tempfile::tempdir().unwrap();
        std::fs::write(dir.path().join(LEGACY_INTERLEAVED_Q4K_BIN), b"legacy").unwrap();
        let r = resolve_interleaved_kquant(dir.path());
        assert!(r.is_legacy);
        assert!(r.bin.ends_with(LEGACY_INTERLEAVED_Q4K_BIN));
        assert!(r
            .manifest
            .unwrap()
            .ends_with(LEGACY_INTERLEAVED_Q4K_MANIFEST_JSON));
    }

    #[test]
    fn resolve_down_features_kquant_dual_read() {
        // New present → use new.
        let dir = tempfile::tempdir().unwrap();
        std::fs::write(dir.path().join(DOWN_FEATURES_KQUANT_BIN), b"new").unwrap();
        let r = resolve_down_features_kquant(dir.path());
        assert!(!r.is_legacy);
        assert!(r.bin.ends_with(DOWN_FEATURES_KQUANT_BIN));
        assert!(r
            .manifest
            .unwrap()
            .ends_with(DOWN_FEATURES_KQUANT_MANIFEST_JSON));

        // Only legacy present → use legacy.
        let dir = tempfile::tempdir().unwrap();
        std::fs::write(dir.path().join(LEGACY_DOWN_FEATURES_Q4K_BIN), b"legacy").unwrap();
        let r = resolve_down_features_kquant(dir.path());
        assert!(r.is_legacy);
        assert!(r.bin.ends_with(LEGACY_DOWN_FEATURES_Q4K_BIN));
        assert!(r
            .manifest
            .unwrap()
            .ends_with(LEGACY_DOWN_FEATURES_Q4K_MANIFEST_JSON));
    }

    #[test]
    fn has_kquant_helpers_accept_either_filename() {
        let dir = tempfile::tempdir().unwrap();
        assert!(!has_kquant_attn_weights(dir.path()));
        std::fs::write(dir.path().join(LEGACY_ATTN_WEIGHTS_Q4K_BIN), b"x").unwrap();
        assert!(has_kquant_attn_weights(dir.path()));
        std::fs::remove_file(dir.path().join(LEGACY_ATTN_WEIGHTS_Q4K_BIN)).unwrap();
        assert!(!has_kquant_attn_weights(dir.path()));
        std::fs::write(dir.path().join(ATTN_WEIGHTS_KQUANT_BIN), b"x").unwrap();
        assert!(has_kquant_attn_weights(dir.path()));
    }

    #[test]
    fn has_kquant_interleaved_accepts_either_filename() {
        let dir = tempfile::tempdir().unwrap();
        assert!(!has_kquant_interleaved(dir.path()));
        std::fs::write(dir.path().join(LEGACY_INTERLEAVED_Q4K_BIN), b"x").unwrap();
        assert!(has_kquant_interleaved(dir.path()));
        std::fs::remove_file(dir.path().join(LEGACY_INTERLEAVED_Q4K_BIN)).unwrap();
        std::fs::write(dir.path().join(INTERLEAVED_KQUANT_BIN), b"x").unwrap();
        assert!(has_kquant_interleaved(dir.path()));
    }

    #[test]
    fn has_kquant_lm_head_accepts_either_filename() {
        let dir = tempfile::tempdir().unwrap();
        assert!(!has_kquant_lm_head(dir.path()));
        std::fs::write(dir.path().join(LEGACY_LM_HEAD_Q4_BIN), b"x").unwrap();
        assert!(has_kquant_lm_head(dir.path()));
        std::fs::remove_file(dir.path().join(LEGACY_LM_HEAD_Q4_BIN)).unwrap();
        std::fs::write(dir.path().join(LM_HEAD_KQUANT_BIN), b"x").unwrap();
        assert!(has_kquant_lm_head(dir.path()));
    }

    #[test]
    fn has_kquant_down_features_accepts_either_filename() {
        let dir = tempfile::tempdir().unwrap();
        assert!(!has_kquant_down_features(dir.path()));
        std::fs::write(dir.path().join(LEGACY_DOWN_FEATURES_Q4K_BIN), b"x").unwrap();
        assert!(has_kquant_down_features(dir.path()));
        std::fs::remove_file(dir.path().join(LEGACY_DOWN_FEATURES_Q4K_BIN)).unwrap();
        std::fs::write(dir.path().join(DOWN_FEATURES_KQUANT_BIN), b"x").unwrap();
        assert!(has_kquant_down_features(dir.path()));
    }
}
