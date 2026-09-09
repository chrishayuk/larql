//! Backend enum, patch-recording state, and Session accessors that
//! discriminate on the active backend. Split out of `mod.rs` so that
//! module contains only `Session` + `execute()` dispatch.

use std::path::{Path, PathBuf};

use crate::error::LqlError;
use crate::relations::RelationClassifier;

use super::Session;

/// The active backend for the session.
/// The base vindex is always loaded readonly. A PatchedVindex overlay
/// handles all mutations without modifying base files on disk.
//
// The `Vindex` variant is much larger than the other three — it owns
// the full `PatchedVindex` + `MemitStore`. Boxing the payload would
// add an indirection on every backend access (common hot path) to
// save stack space on a single enum value the session holds for its
// lifetime. Not a worthwhile trade.
#[allow(clippy::large_enum_variant)]
pub(crate) enum Backend {
    Vindex {
        path: PathBuf,
        config: larql_vindex::VindexConfig,
        /// Patched overlay on the readonly base. All queries and mutations
        /// go through this. The base files on disk are never modified.
        patched: larql_vindex::PatchedVindex,
        relation_classifier: Option<RelationClassifier>,
        /// MoE router index (if available). Used for MoE-aware DESCRIBE.
        router: Option<larql_vindex::RouterIndex>,
        /// L2 store of MEMIT-decomposed `(key, decomposed_down)` pairs
        /// produced by `COMPACT MAJOR`. Persists across the session so
        /// subsequent COMPACT MAJOR runs accumulate cycles.
        ///
        /// (Eventually subsumed by a `StorageEngine` that wraps
        /// `patched` + `memit_store` + the epoch / mutation counters
        /// currently duplicated on `Session`.)
        memit_store: larql_vindex::MemitStore,
    },
    /// Direct model weight access — no vindex extraction needed.
    /// Supports INFER, EXPLAIN INFER, and STATS. Browse/mutation ops
    /// require extraction to a vindex first.
    Weight {
        model_id: String,
        weights: larql_inference::ModelWeights,
        tokenizer: larql_inference::tokenizers::Tokenizer,
    },
    /// A bound VINDEX3 container: the opened executable program plus
    /// serving glue (LQL-1). Holds NO `ModelWeights`, `VectorIndex`, or
    /// `ModelArchitecture` — statements consume the runtime's declared
    /// facts and capabilities; nothing below the binding re-detects the
    /// format. Supports INFER [GENERATE], STATS, SHOW LAYERS.
    Vindex3 {
        #[allow(dead_code)]
        path: PathBuf,
        runtime: larql_inference::Vindex3Runtime<
            larql_vindex::format::vindex3::opplan::exec::lowering::SharedProvider,
        >,
        /// The container's tokenizer, when it carries one — the text
        /// capability. INFER requires it; USE/STATS/SHOW do not.
        tokenizer: Option<larql_inference::tokenizers::Tokenizer>,
        /// The browse view (V3-LQL-3A): semantic roles bound to the
        /// plan's operands. `Some` iff the tokenizer capability is
        /// present (annotations decode token ids); built at USE.
        knowledge: Option<larql_vindex::format::vindex3::knowledge::KnowledgeView>,
        /// The logical mutation overlay (V3-LQL-3B): entity-keyed KNN
        /// entries and applied patches over the read-only container.
        /// Always present — mutation capability is not gated on the
        /// tokenizer.
        overlay: larql_vindex::format::vindex3::knowledge::KnowledgeOverlay,
        /// The BOS token id the container **declares**, resolved once at
        /// bind. Every prompt this session encodes prepends it when the
        /// tokenizer's own post-processor did not — the V2 surface gets
        /// the same fact from its architecture, and the two must agree
        /// (`v2_and_v3_prompt_encoders_agree_on_a_bos_requiring_model`).
        bos_token: Option<u32>,
    },
    /// Remote server backend — queries forwarded via HTTP.
    /// Local patches can be applied for client-side overlay.
    Remote {
        url: String,
        client: reqwest::blocking::Client,
        local_patches: Vec<larql_vindex::VindexPatch>,
        session_id: String,
    },
    None,
}

/// Metadata for an installed fact. Populated at INSERT time, used by
/// subsequent INSERTs' cross-fact balance check.
#[derive(Debug, Clone)]
#[allow(dead_code)]
pub(crate) struct InstalledEdge {
    pub layer: usize,
    pub feature: usize,
    pub canonical_prompt: String,
    pub target: String,
    pub target_id: u32,
}

/// Active patch recording session (between BEGIN PATCH and SAVE PATCH).
pub(crate) struct PatchRecording {
    pub path: String,
    pub operations: Vec<larql_vindex::PatchOp>,
}

impl Session {
    // ── Backend accessors ──

    /// Get readonly access to the patched vindex (base + overlay).
    pub(crate) fn require_patched(&self) -> Result<&larql_vindex::PatchedVindex, LqlError> {
        match &self.backend {
            Backend::Vindex { patched, .. } => Ok(patched),
            Backend::Vindex3 { .. } => Err(super::vindex3::unsupported("this statement")),
            Backend::Weight { model_id, .. } => Err(LqlError::Execution(format!(
                "this operation requires a vindex. Extract first:\n  \
                 EXTRACT MODEL \"{}\" INTO \"{}.vindex\"",
                model_id,
                model_id.split('/').next_back().unwrap_or(model_id),
            ))),
            _ => Err(LqlError::NoBackend),
        }
    }

    /// Get mutable access to the patched overlay.
    pub(crate) fn require_patched_mut(
        &mut self,
    ) -> Result<
        (
            &Path,
            &larql_vindex::VindexConfig,
            &mut larql_vindex::PatchedVindex,
        ),
        LqlError,
    > {
        match &mut self.backend {
            Backend::Vindex {
                path,
                config,
                patched,
                ..
            } => Ok((path, config, patched)),
            Backend::Vindex3 { .. } => Err(super::vindex3::unsupported("mutation")),
            Backend::Weight { model_id, .. } => Err(LqlError::Execution(format!(
                "mutation requires a vindex. Extract first:\n  \
                 EXTRACT MODEL \"{}\" INTO \"{}.vindex\"",
                model_id,
                model_id.split('/').next_back().unwrap_or(model_id),
            ))),
            _ => Err(LqlError::NoBackend),
        }
    }

    /// Get readonly access to path + config + base index.
    pub(crate) fn require_vindex(
        &self,
    ) -> Result<
        (
            &Path,
            &larql_vindex::VindexConfig,
            &larql_vindex::PatchedVindex,
        ),
        LqlError,
    > {
        match &self.backend {
            Backend::Vindex {
                path,
                config,
                patched,
                ..
            } => Ok((path, config, patched)),
            Backend::Vindex3 { .. } => Err(super::vindex3::unsupported("this statement")),
            Backend::Weight { model_id, .. } => Err(LqlError::Execution(format!(
                "this operation requires a vindex. Extract first:\n  \
                 EXTRACT MODEL \"{}\" INTO \"{}.vindex\"",
                model_id,
                model_id.split('/').next_back().unwrap_or(model_id),
            ))),
            _ => Err(LqlError::NoBackend),
        }
    }

    pub(crate) fn relation_classifier(&self) -> Option<&RelationClassifier> {
        match &self.backend {
            Backend::Vindex {
                relation_classifier,
                ..
            } => relation_classifier.as_ref(),
            _ => None,
        }
    }

    /// Mutable access to the Vindex backend's L2 MEMIT store.
    /// Used by `COMPACT MAJOR` to persist decomposed (k, d) pairs.
    pub(crate) fn memit_store_mut(&mut self) -> Result<&mut larql_vindex::MemitStore, LqlError> {
        match &mut self.backend {
            Backend::Vindex { memit_store, .. } => Ok(memit_store),
            _ => Err(LqlError::NoBackend),
        }
    }

    /// Readonly access to the active backend's MEMIT store. Returns
    /// `None` for non-Vindex backends (Weight, Remote, Empty) so
    /// `SHOW COMPACT STATUS` can render a single zero line without an
    /// error path.
    pub(crate) fn memit_store(&self) -> Option<&larql_vindex::MemitStore> {
        match &self.backend {
            Backend::Vindex { memit_store, .. } => Some(memit_store),
            _ => None,
        }
    }

    /// Mutable access to the patch overlay of the current vindex backend,
    /// for tests and benchmarks that need to inject patches without going
    /// through the full INSERT pipeline (which would require a real
    /// tokenizer + relation classifier the synthetic test fixtures don't
    /// carry). Returns `None` if no vindex is loaded. Production code
    /// should go through `INSERT`/`DELETE`/`UPDATE` statements instead.
    pub fn patched_overlay_mut(&mut self) -> Option<&mut larql_vindex::PatchedVindex> {
        match &mut self.backend {
            Backend::Vindex { patched, .. } => Some(patched),
            _ => None,
        }
    }
}
