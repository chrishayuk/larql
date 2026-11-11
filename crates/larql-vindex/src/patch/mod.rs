//! Patch system — lightweight, shareable knowledge diffs.
//!
//! - `format`:       on-the-wire `.vlp` JSON — `VindexPatch`, `PatchOp`,
//!                    `PatchDownMeta`, base64 helpers.
//! - `overlay`:      `PatchedVindex` runtime overlay over an immutable base.
//! - `gate_overlay`: the shared override-row retrieval structure behind
//!                   both `gate_knn` and the KNN store (unified 2026-07-31).
//! - `knn_store`:    L0 residual-key KNN (architecture-B) — metadata +
//!                   ranking policy over `gate_overlay` rows.
//! - `refine`:       refine pass for compiled gates.
//! - `validate`:     whether a patch fits the index it is applied onto.

pub mod format;
pub(crate) mod gate_overlay;
pub mod knn_store;
pub mod knn_store_io;
pub mod overlay;
pub mod overlay_apply;
pub mod overlay_gate_trait;
pub mod refine;
pub mod validate;

pub use format::*;
pub use knn_store::{KnnEntry, KnnStore};
pub use overlay::*;
pub use refine::{refine_gates, RefineInput, RefineResult, RefinedGate};
pub use validate::PatchGeometry;

/// Compatibility alias — the patch surface used to live in `patch::core`.
/// External callers reach in via `larql_vindex::patch::core::Foo` paths;
/// keep them working by re-exporting both new modules through `core`.
pub mod core {
    pub use super::format::*;
    pub use super::overlay::*;
}
