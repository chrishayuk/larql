//! What a patched tensor stores: the one path from a declared patch to BF16 bytes.
//!
//! [`patch_tensor_bytes`] is the only place a resolved patch is written over a tensor's stored bytes. The segment
//! writer of [`apply_declared_patches`](super::apply_declared_patches) calls it for the container it writes, and
//! [`materialise_declared_patches`] calls it for a caller that wants the same bytes and no container: a session
//! executing its pending edits. Both run [`validate_all`](super::validate_all) and then [`check_source`], in that
//! order, so a patch one accepts is a patch the other accepts up to the point where a container starts being
//! written.
//!
//! What the pure path cannot check is what compile learns only by writing: that the derived representations
//! re-derive under the default policy (`NotReproducible`) and that the assembled container is coherent
//! (`Incoherent`). Neither changes the canonical bytes.

use std::fs::File;
use std::path::Path;

use sha2::{Digest, Sha256};

use super::super::error::WeightPatchError;
use super::locate::{locate_all, Located, Resolved};
use super::{
    hex, require_patchable, validate_all, DeclaredPatch, BF16_BYTES, CANDIDATE_JSON, SEGMENTS_DIR,
};
use crate::format::filenames::{INDEX_JSON, WEIGHT_PATCHES_JSON};
use crate::format::vindex3::encode::segment::read_exact_at;
use crate::format::vindex3::index::Vindex3Index;
use crate::format::vindex3::inspect::inspect_container;

/// A source that passed every check made before a patch writes anything.
pub(super) struct CheckedSource<'a> {
    /// `index.json` exactly as read: the ledger digests these bytes.
    pub(super) raw_index: String,
    pub(super) index: Vindex3Index,
    /// Every patch located and resolved, in the order given.
    pub(super) edited: Vec<(Located, Resolved<'a>)>,
    /// The top-level files a patched container carries over unchanged.
    pub(super) carried: Vec<String>,
}

/// Check `patches` against the container at `src`, reading and writing nothing but the source: its index parses,
/// it is coherent, it is a canonical container with no ledger and no variants, every tensor is a BF16 matrix held
/// by exactly one canonical representation and all by the same one, every element is in bounds and finite, every
/// result fits, and the source holds nothing a patch does not know how to carry.
pub(super) fn check_source<'a>(
    src: &Path,
    patches: &'a [DeclaredPatch],
) -> Result<CheckedSource<'a>, WeightPatchError> {
    let raw_index = std::fs::read_to_string(src.join(INDEX_JSON))?;
    let index: Vindex3Index = serde_json::from_str(&raw_index)
        .map_err(|e| crate::error::VindexError::Parse(format!("parse {INDEX_JSON}: {e}")))?;
    let inspection = inspect_container(src, false)?;
    if !inspection.is_coherent() {
        return Err(WeightPatchError::SourceIncoherent(format!(
            "{:?}",
            inspection.defects
        )));
    }
    require_patchable(&index)?;
    let edited = locate_all(src, &index, patches)?;
    let carried = carried_files(src)?;
    Ok(CheckedSource {
        raw_index,
        index,
        edited,
        carried,
    })
}

/// The top-level files of `src` a patched container carries over: every file but the three a patch writes
/// itself. The only directory a container holds is its segments; any other is refused, since a patch would not
/// know whether it describes the weights being edited.
fn carried_files(src: &Path) -> Result<Vec<String>, WeightPatchError> {
    let mut carried = Vec::new();
    for entry in std::fs::read_dir(src)? {
        let entry = entry?;
        let name = entry.file_name().to_string_lossy().to_string();
        // `Path::is_dir` follows a symlink: a linked directory is a directory here, refused by name like any
        // other, where the entry's own file type would call it a file and the copy would fail on it.
        if entry.path().is_dir() {
            if name != SEGMENTS_DIR {
                return Err(WeightPatchError::UnexpectedEntry(name));
            }
        } else if ![INDEX_JSON, CANDIDATE_JSON, WEIGHT_PATCHES_JSON].contains(&name.as_str()) {
            carried.push(name);
        }
    }
    Ok(carried)
}

/// Write `resolved` over `bytes`, one tensor's stored BF16 (little-endian, row-major, `cols` wide): each listed
/// element's new bits, or the whole tensor an operation target compiled to.
pub(super) fn patch_tensor_bytes(bytes: &mut [u8], cols: u64, resolved: &Resolved<'_>) {
    let width = BF16_BYTES as usize;
    match resolved {
        Resolved::Edits(edits) => {
            for e in edits {
                let at_byte = ((e.row as u64 * cols + e.col as u64) * BF16_BYTES) as usize;
                bytes[at_byte..at_byte + width].copy_from_slice(&e.new_bits.to_le_bytes());
            }
        }
        Resolved::Operations(p) => {
            for (chunk, bits) in bytes.chunks_exact_mut(width).zip(&p.result) {
                chunk.copy_from_slice(&bits.to_le_bytes());
            }
        }
    }
}

/// One patched tensor exactly as a container compiled from the same patches stores it.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MaterialisedTensor {
    /// The graph object whose canonical representation holds the tensor.
    pub object: String,
    /// That canonical representation's id.
    pub representation: String,
    /// The tensor's name in the segment table.
    pub tensor: String,
    pub shape: [usize; 2],
    /// The whole tensor: BF16, little-endian, row-major.
    pub bytes: Vec<u8>,
}

impl MaterialisedTensor {
    /// SHA-256 of the tensor's stored payload: what an operation target's ledger records as its result, and
    /// what hashing the same tensor's bytes in a compiled container gives.
    pub fn sha256(&self) -> String {
        hex(&Sha256::digest(&self.bytes))
    }
}

/// The bytes [`apply_declared_patches`](super::apply_declared_patches) would store for each patched tensor of
/// the container at `src`, in the order given, writing nothing. Refuses whatever that function refuses before
/// it starts writing, by the same checks in the same order.
pub fn materialise_declared_patches(
    src: &Path,
    patches: &[DeclaredPatch],
) -> Result<Vec<MaterialisedTensor>, WeightPatchError> {
    validate_all(patches)?;
    let checked = check_source(src, patches)?;
    let mut out = Vec::with_capacity(checked.edited.len());
    for (at, resolved) in &checked.edited {
        let file = File::open(src.join(&at.entry.segment))?;
        let mut bytes = vec![0u8; at.tensor.len as usize];
        read_exact_at(&file, &mut bytes, at.payload_start + at.tensor.offset)?;
        patch_tensor_bytes(&mut bytes, at.tensor.shape[1] as u64, resolved);
        out.push(MaterialisedTensor {
            object: at.entry.object.clone(),
            representation: at.canonical_id.clone(),
            tensor: at.tensor.name.clone(),
            shape: [at.tensor.shape[0], at.tensor.shape[1]],
            bytes,
        });
    }
    Ok(out)
}
