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
//! This is deliberately restrictive: one source, named BF16 matrices of one
//! canonical representation, explicit coordinates, at most
//! [`MAX_PATCH_ELEMENTS`] elements in all, finite deltas. It refuses anything it cannot apply exactly as declared, and it
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

pub mod any_op;
pub mod bf16;
pub mod error;
pub mod ledger;
pub mod low_rank_op;
pub mod op;
pub mod op_target;
mod op_target_check;
pub mod structured;
pub mod structured_op;

mod apply;

pub use any_op::{declare_patches, WeightOp, WeightOpRef};
pub use apply::{
    apply_declared_patches, apply_weight_patch, apply_weight_patch_with_origins,
    apply_weight_patches, plan_declared_patch, plan_weight_patch, read_bf16_tensor, tensor_shape,
    DeclaredPatch, PatchReport, ResultEdit, ResultPatch, WeightEdit, WeightPatch,
};
pub use error::WeightPatchError;
pub use ledger::{
    check_ledger, parse_ledger, EditEncoding, LedgerEdit, LedgerTarget, WeightPatchLedger,
    WEIGHT_PATCHES_SCHEMA, WEIGHT_PATCHES_SCHEMA_V1, WEIGHT_PATCHES_SCHEMA_V3,
    WEIGHT_PATCHES_SCHEMA_V4,
};
pub use low_rank_op::LowRankColumnsOp;
pub use op::{bind_source, record_op, DeltaEdit, SourceBinding, WeightDeltaOp};
pub use structured_op::StructuredWeightOp;

/// The most elements one patch may edit, across all its tensors. A patch is
/// a bounded, reviewable change; a rewrite of a tensor is a different
/// operation. Sized (2^18) to carry an enumerated import of a published,
/// structured edit, such as WSP-001's p50c (130,188 BF16 elements over two
/// `[1024, 3584]` matrices), with headroom; one such matrix is 3,670,016.
pub const MAX_PATCH_ELEMENTS: usize = 1 << 18;

// Bounds of the structured-op path (`weight-patches/v4` operation targets, docs/verification/weight-patches-v4.md §5).
// An operation target is recorded by its op, not by its elements, so `MAX_PATCH_ELEMENTS` does not bound it; these do,
// so that a compact record can never be an instruction to do unbounded work. Each is checked before any arithmetic.

/// The largest rank `K` of one `add_low_rank_columns` op. A rank-one edit is the case SG-EDIT-1 needs; this leaves
/// room for low-rank fits without admitting a dense factorisation.
pub const MAX_STRUCTURED_RANK: usize = 256;
/// The most operand elements one structured op may carry (`len(a) + len(b)`, or `len(u) + 1`): 2^24, so an op's
/// operands stay a reviewable file (a rank-one edit of a `[2560, 10240]` matrix carries 12,800).
pub const MAX_STRUCTURED_OPERAND_ELEMENTS: usize = 1 << 24;
/// The most elements (`rows × cols`) of a tensor a structured op may edit: 2^30, two GiB of BF16, above any single
/// matrix of the models in scope (a Gemma 3 4B `down_proj` is 26,214,400).
pub const MAX_STRUCTURED_TENSOR_ELEMENTS: usize = 1 << 30;
/// The most multiply-adds one compile's structured ops may cost together: `rows × |J| × K` per low-rank op and
/// `2 × rows × |S|` per removal. 2^32 admits a rank-one edit of every column of a 2^30-element matrix with headroom.
pub const MAX_STRUCTURED_WORK: u64 = 1 << 32;
