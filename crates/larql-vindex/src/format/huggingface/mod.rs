//! HuggingFace Hub integration — download, publish, and discovery
//! for vindex-shaped dataset repos.
//!
//! ```text
//! # Download a vindex
//! larql> USE "hf://chrishayuk/gemma-3-4b-it-vindex";
//!
//! # Upload a vindex
//! larql publish gemma3-4b.vindex --repo chrishayuk/gemma-3-4b-it-vindex
//! ```
//!
//! Module split (post 2026-04-25 audit):
//! - [`download`]  — `hf://` resolution, snapshot caching, conditional fetch
//! - [`publish`]   — repo creation, file uploads, LFS protocol, callbacks
//! - [`discovery`] — collections, repo existence, item fetch
//!
//! Shared constants live here. Each submodule re-imports them via
//! `use super::{VINDEX_METADATA_FILES, VINDEX_BIN_FILES, vindex_core_files,
//! VINDEX_WEIGHT_FILES}`.

use crate::format::filenames::*;

/// Small metadata files needed to describe a vindex (`larql show`,
/// browse-tier UIs, schema introspection). All of these together stay
/// well under a few MB on a typical vindex, so they're safe to fetch
/// eagerly even on slow links.
pub(crate) const VINDEX_METADATA_FILES: &[&str] = &[
    INDEX_JSON,
    TOKENIZER_JSON,
    DOWN_META_BIN,
    DOWN_META_JSONL,
    RELATION_CLUSTERS_JSON,
    FEATURE_LABELS_JSON,
];

/// Big tensor files lazy-pulled on first walk/run. These can be
/// hundreds of MB to multiple GB; metadata-only commands like
/// `larql show` shouldn't pay for them. Callers that actually need
/// the tensors (run / walk / extract) use the progress-aware
/// entrypoint that pulls METADATA + BIN together.
pub(crate) const VINDEX_BIN_FILES: &[&str] = &[GATE_VECTORS_BIN, EMBEDDINGS_BIN];

/// Union of metadata + bin — preserves prior CORE behavior for the
/// progress-aware entrypoint that is willing to wait on big files.
pub(crate) fn vindex_core_files() -> Vec<&'static str> {
    let mut v: Vec<&'static str> = VINDEX_METADATA_FILES.to_vec();
    v.extend_from_slice(VINDEX_BIN_FILES);
    v
}

pub(crate) const VINDEX_WEIGHT_FILES: &[&str] = &[
    ATTN_WEIGHTS_BIN,
    ATTN_WEIGHTS_KQUANT_BIN,
    ATTN_WEIGHTS_KQUANT_MANIFEST_JSON,
    LEGACY_ATTN_WEIGHTS_Q4K_BIN,
    LEGACY_ATTN_WEIGHTS_Q4K_MANIFEST_JSON,
    INTERLEAVED_KQUANT_BIN,
    INTERLEAVED_KQUANT_MANIFEST_JSON,
    LEGACY_INTERLEAVED_Q4K_BIN,
    LEGACY_INTERLEAVED_Q4K_MANIFEST_JSON,
    NORMS_BIN,
    UP_WEIGHTS_BIN,
    DOWN_WEIGHTS_BIN,
    LM_HEAD_BIN,
    LM_HEAD_KQUANT_BIN,
    LEGACY_LM_HEAD_Q4_BIN,
    WEIGHT_MANIFEST_JSON,
];

pub mod discovery;
pub mod download;
pub mod publish;

// Re-export the previous flat-module surface so callers don't have to
// pick a submodule.
pub use discovery::{
    add_collection_item, dataset_repo_exists, ensure_collection, fetch_collection_items,
    repo_exists, CollectionItem,
};
pub use download::{
    download_hf_weights, resolve_hf_model_with_progress, resolve_hf_vindex,
    resolve_hf_vindex_with_progress, DownloadProgress,
};
pub use publish::{
    publish_vindex, publish_vindex_with_opts, PublishCallbacks, PublishOptions,
    SilentPublishCallbacks,
};

/// Check if a path is an `hf://` reference. Lives here (not under
/// `download`) because callers in `publish` and `discovery` test it
/// too.
pub fn is_hf_path(path: &str) -> bool {
    path.starts_with("hf://")
}

#[cfg(test)]
mod tests {
    use super::*;

    // ── is_hf_path ────────────────────────────────────────────────────────

    #[test]
    fn is_hf_path_accepts_typical_repo_id() {
        assert!(is_hf_path("hf://owner/repo"));
        assert!(is_hf_path("hf://chrishayuk/gemma-3-4b-it-vindex"));
    }

    #[test]
    fn is_hf_path_accepts_revision_suffix() {
        // `@<rev>` is parsed downstream by the resolver; `is_hf_path` only
        // checks the scheme prefix.
        assert!(is_hf_path("hf://owner/repo@v1.0"));
    }

    #[test]
    fn is_hf_path_rejects_non_hf_schemes() {
        assert!(!is_hf_path("https://huggingface.co/owner/repo"));
        assert!(!is_hf_path("file:///tmp/local"));
        assert!(!is_hf_path("/absolute/path"));
        assert!(!is_hf_path("./relative"));
    }

    #[test]
    fn is_hf_path_rejects_empty() {
        assert!(!is_hf_path(""));
    }

    #[test]
    fn is_hf_path_is_prefix_only_check() {
        // Anything starting with `hf://` is accepted — payload validation
        // happens later, in the resolver.
        assert!(is_hf_path("hf://"));
        assert!(is_hf_path("hf:///garbage"));
    }

    // ── VINDEX_METADATA_FILES / VINDEX_BIN_FILES / vindex_core_files ──────
    //
    // These constants are the contract between `download`, `publish`, and
    // `discovery`. Pin the contents so accidental edits show up here.

    #[test]
    fn vindex_metadata_files_list_is_pinned() {
        // Order matters: `index.json` must be first so the resolver can
        // bootstrap from a single GET to learn the rest.
        assert_eq!(VINDEX_METADATA_FILES[0], INDEX_JSON);
        assert!(VINDEX_METADATA_FILES.contains(&TOKENIZER_JSON));
        assert!(VINDEX_METADATA_FILES.contains(&DOWN_META_BIN));
        assert!(VINDEX_METADATA_FILES.contains(&DOWN_META_JSONL));
        assert!(VINDEX_METADATA_FILES.contains(&RELATION_CLUSTERS_JSON));
        assert!(VINDEX_METADATA_FILES.contains(&FEATURE_LABELS_JSON));
        // Big tensor files belong to BIN, not METADATA.
        assert!(!VINDEX_METADATA_FILES.contains(&GATE_VECTORS_BIN));
        assert!(!VINDEX_METADATA_FILES.contains(&EMBEDDINGS_BIN));
    }

    #[test]
    fn vindex_bin_files_list_is_pinned() {
        assert!(VINDEX_BIN_FILES.contains(&GATE_VECTORS_BIN));
        assert!(VINDEX_BIN_FILES.contains(&EMBEDDINGS_BIN));
    }

    #[test]
    fn vindex_core_files_unions_metadata_and_bin() {
        let core = vindex_core_files();
        for f in VINDEX_METADATA_FILES {
            assert!(core.contains(f), "core missing METADATA entry: {f}");
        }
        for f in VINDEX_BIN_FILES {
            assert!(core.contains(f), "core missing BIN entry: {f}");
        }
        assert_eq!(
            core.len(),
            VINDEX_METADATA_FILES.len() + VINDEX_BIN_FILES.len(),
            "core must be the disjoint union of METADATA and BIN"
        );
    }

    #[test]
    fn vindex_weight_files_includes_kquant_artifacts() {
        // Pull-time vindex weights include the f32 names plus both the
        // new kquant-named and legacy q4k-named k-quant companions, so
        // both pre- and post-rename vindexes round-trip through HF.
        assert!(VINDEX_WEIGHT_FILES.contains(&ATTN_WEIGHTS_BIN));
        assert!(VINDEX_WEIGHT_FILES.contains(&ATTN_WEIGHTS_KQUANT_BIN));
        assert!(VINDEX_WEIGHT_FILES.contains(&ATTN_WEIGHTS_KQUANT_MANIFEST_JSON));
        assert!(VINDEX_WEIGHT_FILES.contains(&LEGACY_ATTN_WEIGHTS_Q4K_BIN));
        assert!(VINDEX_WEIGHT_FILES.contains(&LEGACY_ATTN_WEIGHTS_Q4K_MANIFEST_JSON));
        assert!(VINDEX_WEIGHT_FILES.contains(&INTERLEAVED_KQUANT_BIN));
        assert!(VINDEX_WEIGHT_FILES.contains(&INTERLEAVED_KQUANT_MANIFEST_JSON));
        assert!(VINDEX_WEIGHT_FILES.contains(&LEGACY_INTERLEAVED_Q4K_BIN));
        assert!(VINDEX_WEIGHT_FILES.contains(&LEGACY_INTERLEAVED_Q4K_MANIFEST_JSON));
        assert!(VINDEX_WEIGHT_FILES.contains(&LM_HEAD_BIN));
        assert!(VINDEX_WEIGHT_FILES.contains(&LM_HEAD_KQUANT_BIN));
        assert!(VINDEX_WEIGHT_FILES.contains(&LEGACY_LM_HEAD_Q4_BIN));
        assert!(VINDEX_WEIGHT_FILES.contains(&WEIGHT_MANIFEST_JSON));
    }
}
