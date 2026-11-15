//! Why a stored tensor cannot be read.

use crate::error::VindexError;
use crate::format::vindex3::weight_patch::WeightPatchError;

#[derive(Debug, thiserror::Error)]
pub enum StoredTensorError {
    /// A bare name no canonical representation holds.
    #[error("tensor `{0}` is not held by any canonical representation; a tensor of a derived representation is named `<representation>::<tensor>`; SHOW TENSORS lists every name")]
    UnknownBare(String),
    /// A qualified name whose representation does not exist or does not hold the tensor.
    #[error("`{0}` names no tensor of this container; SHOW TENSORS lists every name")]
    UnknownQualified(String),
    /// A bare name held by more than one canonical representation. It carries the refusal a weight edit gives, so
    /// the hint on how to qualify the name is the same wherever a name is resolved.
    #[error(transparent)]
    Ambiguous(WeightPatchError),
    /// Stored bytes that are not plain float elements.
    #[error("tensor `{tensor}` is stored as {dtype}, a packed encoding with no element-wise values; only BF16, F16 and F32 tensors can be read. Read the representation it was compiled from")]
    PackedDtype { tensor: String, dtype: String },
    /// A row or column selection that does not fit the tensor.
    #[error("selection of `{tensor}` refused: {why}")]
    Selection { tensor: String, why: String },
    /// A tensor table entry whose byte length does not match its shape and dtype.
    #[error("tensor `{tensor}` is malformed: {why}")]
    Malformed { tensor: String, why: String },
    #[error(transparent)]
    Vindex(#[from] VindexError),
    #[error(transparent)]
    Io(#[from] std::io::Error),
}
