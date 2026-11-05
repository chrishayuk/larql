//! **Weight patch**: edit stored weights of a VINDEX3 container and write a
//! new, immutable container.
//!
//! ```text
//! source container (untouched)
//!     → validate the edit (one tensor, explicit elements, bounded, finite)
//!     → rewrite ONLY the edited canonical segment; hard-link everything else
//!     → re-derive the derived representations of that object (Q8_0 pack)
//!     → write the ledger (source, every edit, resulting digests)
//!     → inspect the assembly; rename into place, or leave nothing
//! ```
//!
//! This is deliberately restrictive: one source, one named BF16 matrix,
//! explicit coordinates, at most [`MAX_PATCH_ELEMENTS`] elements, finite
//! deltas. It refuses anything it cannot apply exactly as declared, and it
//! refuses rather than guessing which representation an edit was meant for.
//!
//! The edit is defined on the canonical weights: a patched element is
//! `RNE_bf16(old + delta)` ([`bf16`]), rounded once. Derived representations
//! are re-derived from the edited canonical bytes by the same compiler that
//! made them, so a derived pack never describes weights the container does
//! not hold. The model identity is unchanged; the ledger records the
//! derivation, and its result digests are the identity of the edited source.
//!
//! Nothing here verifies that a patch is *good*, or even that this writer
//! implemented it correctly. [`ledger::check_ledger`] checks that the ledger
//! describes the container; an independent verifier is what checks the
//! container against the declared edit.

pub mod bf16;
pub mod error;
pub mod ledger;
pub mod op;

mod apply;

pub use apply::{
    apply_weight_patch, apply_weight_patch_with_origins, plan_weight_patch, PatchReport,
    WeightEdit, WeightPatch,
};
pub use error::WeightPatchError;
pub use ledger::{check_ledger, LedgerEdit, WeightPatchLedger, WEIGHT_PATCHES_SCHEMA};
pub use op::{bind_source, record_op, DeltaEdit, SourceBinding, WeightDeltaOp};

/// The most elements one patch may edit. A patch is a bounded, reviewable
/// change; a rewrite of a tensor is a different operation.
pub const MAX_PATCH_ELEMENTS: usize = 4096;
