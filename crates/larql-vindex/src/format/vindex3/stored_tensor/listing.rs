//! Every tensor of every representation, with its qualified name and what its bare name finds.

use std::collections::BTreeMap;
use std::path::Path;

use super::error::StoredTensorError;
use crate::format::filenames::INDEX_JSON;
use crate::format::vindex3::container_path;
use crate::format::vindex3::encode::segment::read_segment_header;
use crate::format::vindex3::index::Vindex3Index;
use crate::format::vindex3::weight_patch::qualified;

/// What a tensor's bare name finds, as a weight edit or an export resolves it.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BareName {
    /// The bare name finds this tensor: it is the one canonical representation that holds the name.
    Resolves,
    /// The bare name is held by more than one canonical representation, and is refused: qualify it.
    Ambiguous,
    /// The bare name does not find this tensor: it is held by a derived representation, which a bare name never
    /// finds.
    Derived,
}

impl BareName {
    /// The word a listing prints.
    pub fn label(self) -> &'static str {
        match self {
            Self::Resolves => "yes",
            Self::Ambiguous => "ambiguous",
            Self::Derived => "no (derived)",
        }
    }
}

/// One tensor of one representation.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TensorListing {
    /// The representation id (`object@encoding`).
    pub representation: String,
    /// The logical object the representation materialises.
    pub object: String,
    /// Whether the representation is canonical (not compiled from another).
    pub canonical: bool,
    /// The name in the segment's tensor table.
    pub tensor: String,
    pub dtype: String,
    pub shape: Vec<usize>,
    pub bare_name: BareName,
}

impl TensorListing {
    /// `<representation>::<tensor>`, which always names exactly this tensor.
    pub fn qualified_name(&self) -> String {
        qualified(&self.representation, &self.tensor)
    }
}

pub(super) fn read_index(root: &Path) -> Result<Vindex3Index, StoredTensorError> {
    let text = std::fs::read_to_string(root.join(INDEX_JSON))?;
    serde_json::from_str(&text)
        .map_err(|e| crate::error::VindexError::Parse(format!("parse {INDEX_JSON}: {e}")).into())
}

/// Every tensor of every representation of the container at `root`, canonical and derived, in directory order and
/// then tensor-table order.
pub fn list_tensors(root: &Path) -> Result<Vec<TensorListing>, StoredTensorError> {
    let index = read_index(root)?;
    let mut listings = Vec::new();
    for (id, entry) in &index.representations {
        let segment = container_path::resolve(root, &entry.segment)?;
        let (header, _) = read_segment_header(&segment)?;
        for tensor in header.tensors {
            listings.push(TensorListing {
                representation: id.clone(),
                object: entry.object.clone(),
                canonical: entry.compiled_from.is_none(),
                tensor: tensor.name,
                dtype: tensor.dtype,
                shape: tensor.shape,
                bare_name: BareName::Derived,
            });
        }
    }
    let mut canonical_holders: BTreeMap<String, usize> = BTreeMap::new();
    for listing in listings.iter().filter(|l| l.canonical) {
        *canonical_holders.entry(listing.tensor.clone()).or_default() += 1;
    }
    for listing in listings.iter_mut().filter(|l| l.canonical) {
        listing.bare_name = match canonical_holders.get(&listing.tensor) {
            Some(1) => BareName::Resolves,
            _ => BareName::Ambiguous,
        };
    }
    Ok(listings)
}
