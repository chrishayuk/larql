#![allow(clippy::doc_overindented_list_items)]
#![allow(clippy::doc_lazy_continuation)]

//! Vindex — the queryable model format.
//!
//! Decompile, browse, edit, and recompile neural networks.
//! This crate owns the complete vindex lifecycle:
//! extract, load, query, mutate, patch, save, compile.
//!
//! ## Module map
//!
//! - `extract`:    build a vindex from `safetensors` / GGUF.
//! - `format`:     on-disk layout, checksums, HF Hub publish/resolve.
//! - `index`:      `VectorIndex`, gate KNN, walk, HNSW, MoE router, residency.
//! - `patch`:      `PatchedVindex` overlay, `KnnStore`, refine pass.
//! - `storage`:    `StorageEngine` lifecycle, MEMIT decomposition (`memit_solve`).
//! - `clustering`: kmeans + offset/cluster labelling.
//! - `describe`:   token-level edge labelling.
//! - `vindexfile`: declarative build pipeline.
//! - `mmap_util`:  `madvise` hints for residency control.
//!
//! All matrix operations route through `larql_compute` (BLAS on CPU,
//! Metal GPU when `--features metal`).

// BLAS provided by larql-compute dependency (no direct blas_src needed)

// ── Module structure ──
pub mod clustering;
pub mod config;
pub mod describe;
pub mod engine;
pub mod error;
pub mod extract;
pub mod format;
pub mod index;
pub mod patch;
pub mod quant;
pub mod trie;
pub mod walker;
// Back-compat alias — the top-level lifecycle dir was renamed
// `storage/` → `engine/` in the 2026-04-25 round-2 cleanup. The name
// `storage` was confusing because `index/storage/` held the actual
// data substores. Drop this alias once external callers migrate.
pub use engine as storage;
pub mod mmap_util;
pub mod vindexfile;

// ── Re-export dependencies ──
pub use ndarray;
pub use tokenizers;

// ── Re-export essentials at crate root ──

// Config
pub use config::dtype::StorageDtype;
pub use config::types::{
    ComplianceGate, DownMetaRecord, DownMetaTopK, ExtractLevel, FfnLayout, Fp4Config, LayerBands,
    MoeConfig, Precision, ProjectionFormat, Projections, QuantFormat, VindexConfig,
    VindexLayerInfo, VindexModelConfig, VindexSource,
};

// Error
pub use error::VindexError;

// Index
pub use index::core::{
    FeatureMeta, FfnRowAccess, Fp4FfnAccess, GateIndex, GateLookup, IndexLoadCallbacks,
    NativeFfnAccess, PatchOverrides, QuantizedFfnAccess, SilentLoadCallbacks, StorageBucket,
    VectorIndex, WalkHit, WalkTrace,
};
pub use index::residency::{LayerState, ResidencyManager};
pub use index::router::{RouteResult, RouterIndex};

// Describe
pub use describe::{DescribeEdge, LabelSource};

// Extract
pub use extract::{
    build_vindex, build_vindex_from_vectors, build_vindex_streaming, snapshot_hf_metadata,
    IndexBuildCallbacks, SilentBuildCallbacks, SNAPSHOT_FILES,
};

// Format
pub use format::checksums;
pub use format::down_meta;
pub use format::load::{
    load_feature_labels, load_vindex_config, load_vindex_embeddings, load_vindex_tokenizer,
};
// Model loading: use larql_models::{load_model_dir, resolve_model_path, load_gguf} directly
pub use format::huggingface::{
    dataset_repo_exists, download_hf_weights, ensure_collection, fetch_collection_items,
    is_hf_path, publish_vindex, publish_vindex_with_opts, repo_exists,
    resolve_hf_model_with_progress, resolve_hf_vindex, resolve_hf_vindex_with_progress,
    CollectionItem, DownloadProgress, PublishCallbacks, PublishOptions, SilentPublishCallbacks,
};
pub use format::weights::{
    load_model_weights, load_model_weights_kquant, load_model_weights_kquant_shard,
    load_model_weights_with_opts, write_model_weights, write_model_weights_kquant,
    write_model_weights_kquant_with_opts, write_model_weights_with_opts, DownProjFormat,
    KquantWriteOptions, LoadWeightsOptions, StreamingWeights, WeightSource, WriteWeightsOptions,
};

// Patch
pub use patch::core::{PatchOp, PatchedVindex, VindexPatch};
pub use patch::knn_store::{KnnEntry, KnnStore};
pub use patch::refine::{refine_gates, RefineInput, RefineResult, RefinedGate};

// Trie probe (route classifier loaded from JSON)
pub use trie::CascadeTrie;

// Walker — build-time graph extraction (FFN + attention) and vector dump.
// Full module surface is reachable via `larql_vindex::walker::*`.
pub use walker::attention_walker::{AttentionLayerResult, AttentionWalker};
pub use walker::vector_extractor::{
    ExtractCallbacks, ExtractConfig, ExtractSummary, VectorExtractor,
};
pub use walker::weight_walker::{
    walk_model, LayerResult, LayerStats, WalkCallbacks, WalkConfig, WeightWalker,
};

// Storage engine — `engine` (preferred); `storage` still available as alias.
pub use engine::{
    memit_solve, CompactStatus, Epoch, MemitCycle, MemitFact, MemitSolveResult, MemitStore,
    StorageEngine,
};

// Vindexfile
pub use vindexfile::{
    build_from_vindexfile, parse_vindexfile, Vindexfile, VindexfileDirective, VindexfileStage,
};
