//! Why a weight patch is refused. Every refusal is by name: a patch that
//! cannot be applied exactly as declared is never applied approximately.

use std::path::PathBuf;

use crate::error::VindexError;

#[derive(Debug, thiserror::Error)]
pub enum WeightPatchError {
    #[error("the patch edits {count} elements; the limit is {limit}")]
    TooManyElements { count: usize, limit: usize },
    #[error("the patch edits no elements")]
    Empty,
    #[error("element ({row}, {col}) is edited more than once")]
    DuplicateElement { row: usize, col: usize },
    #[error("the delta for element ({row}, {col}) is not finite")]
    NonFiniteDelta { row: usize, col: usize },
    #[error("tensor `{0}` is not held by any canonical representation of the source")]
    UnknownTensor(String),
    #[error("tensor `{tensor}` is held by more than one canonical representation: {holders:?}")]
    AmbiguousTensor {
        tensor: String,
        holders: Vec<String>,
    },
    #[error("tensor `{tensor}` is stored as {dtype}; only BF16 weights can be patched")]
    UnsupportedDtype { tensor: String, dtype: String },
    #[error("tensor `{tensor}` has shape {shape:?}; a patch needs a matrix")]
    NotAMatrix { tensor: String, shape: Vec<usize> },
    #[error("element ({row}, {col}) is outside the {rows}x{cols} tensor `{tensor}`")]
    OutOfBounds {
        tensor: String,
        row: usize,
        col: usize,
        rows: usize,
        cols: usize,
    },
    #[error("the stored element ({row}, {col}) of `{tensor}` is not finite")]
    NonFiniteSource {
        tensor: String,
        row: usize,
        col: usize,
    },
    #[error("the patched value of element ({row}, {col}) overflows bfloat16")]
    Overflow { row: usize, col: usize },
    #[error("the declared result of element ({row}, {col}) is not a finite bfloat16")]
    NonFiniteResult { row: usize, col: usize },
    #[error("column {col} of `{tensor}` is already written by {by}; two structured ops do not share a column")]
    ColumnConflict {
        tensor: String,
        col: usize,
        by: String,
    },
    #[error("`{tensor}` is edited by a delta op and a structured op ({by}); one tensor takes one form in this version")]
    MixedForms { tensor: String, by: String },
    #[error("structured op: {0}")]
    Structured(String),
    #[error("the source is not a canonical container ({0}); a deployment image cannot be patched")]
    NotCanonical(String),
    #[error("the source already carries a weight-patch ledger; patching a patched container is not supported")]
    AlreadyPatched,
    #[error("the source carries alternate variants, which a patch would leave stale")]
    HasVariants,
    #[error("derived representation `{derived}` does not derive from the edited `{canonical}`; this patch re-derives only what it edits")]
    DerivedFromOther { derived: String, canonical: String },
    #[error("the source contains `{0}`, which this patch does not know how to carry")]
    UnexpectedEntry(String),
    #[error("the source is not coherent: {0}")]
    SourceIncoherent(String),
    #[error("the derived representation cannot be reproduced under the default policy: {0}")]
    NotReproducible(String),
    #[error("output `{}` already exists; containers are immutable", .0.display())]
    OutputExists(PathBuf),
    #[error("the assembled container is not coherent: {0}")]
    Incoherent(String),
    #[error("element ({row}, {col}) of `{tensor}` is already edited by {by}; edits to one element are not composed")]
    Conflict {
        tensor: String,
        row: usize,
        col: usize,
        by: String,
    },
    #[error("this edit was written against another artifact: expected index/weights {expected}, this container is {found}")]
    WrongSource { expected: String, found: String },
    #[error("weight edit in a patch: {0}")]
    PatchOp(String),
    #[error("tensor `{0}` is edited by two parts of one patch; give each tensor once")]
    DuplicateTensor(String),
    #[error("one patch edits one canonical representation: `{}` is in `{}`, `{}` is in `{}`", .first.0, .first.1, .other.0, .other.1)]
    SpansRepresentations {
        /// (tensor, representation) of the first edited tensor.
        first: (String, String),
        /// (tensor, representation) of one held elsewhere.
        other: (String, String),
    },
    #[error(transparent)]
    Vindex(#[from] VindexError),
    #[error(transparent)]
    Io(#[from] std::io::Error),
}

impl From<WeightPatchError> for VindexError {
    fn from(e: WeightPatchError) -> Self {
        match e {
            WeightPatchError::Vindex(v) => v,
            other => VindexError::Parse(format!("weight patch refused: {other}")),
        }
    }
}
