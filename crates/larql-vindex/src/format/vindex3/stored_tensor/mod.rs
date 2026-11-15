//! **Stored tensors**: list every tensor a VINDEX3 container holds, and read one exactly as stored.
//!
//! This is the read side of tensor naming. A tensor is named as a weight edit names it ([`super::weight_patch::
//! tensor_name`]): bare (`0.mlp.down_proj.weight`), which finds the one canonical representation that holds it, or
//! qualified (`target.embedding@BF16::weight`), which finds it in the representation it names, canonical or
//! derived.
//!
//! A read is not an edit, so none of a patch's refusals apply: a tensor of a container whose derived pack depends
//! on another object, or of a container that already carries a ledger, reads like any other. What it refuses is a
//! name that finds no tensor or several, and a tensor whose stored bytes are not plain float elements (a packed
//! quantised encoding such as Q8_0), since those have no element-wise value to give.
//!
//! The bytes come from the container's segments, never from a runtime or an overlay: a read is of what is stored.

mod error;
mod listing;
mod read;

pub use error::StoredTensorError;
pub use listing::{list_tensors, BareName, TensorListing};
pub use read::{locate_stored_tensor, Selection, StoredTensor};
