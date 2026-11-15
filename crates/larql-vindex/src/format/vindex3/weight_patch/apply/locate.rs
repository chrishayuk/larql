//! Finding the tensor a patch edits, and resolving its edits against the stored bits.

use std::fs::File;
use std::path::Path;

use super::super::bf16;
use super::super::error::WeightPatchError;
use super::super::ledger::{EditEncoding, LedgerEdit};
use super::super::op_target::OperationPatch;
use super::{DeclaredPatch, BF16_BYTES, BF16_DTYPE};
use crate::format::vindex3::encode::segment::{
    read_exact_at, read_segment_header, SegmentHeader, SegmentTensor,
};
use crate::format::vindex3::index::{RepresentationEntry, Vindex3Index};

/// What a located patch resolved to: element edits, or a tensor compiled from its operations.
pub(super) enum Resolved<'a> {
    Edits(Vec<LedgerEdit>),
    Operations(&'a OperationPatch),
}

/// Where the edited tensor lives and what derives from it.
pub(super) struct Located {
    pub(super) canonical_id: String,
    pub(super) entry: RepresentationEntry,
    pub(super) header: SegmentHeader,
    pub(super) payload_start: u64,
    pub(super) tensor: SegmentTensor,
    pub(super) derived: Vec<(String, RepresentationEntry)>,
}

/// Locate and resolve every patch; refuse unless one canonical
/// representation holds them all, since the re-derivation and the result
/// digests belong to one representation.
pub(super) fn locate_all<'a>(
    src: &Path,
    index: &Vindex3Index,
    patches: &'a [DeclaredPatch],
) -> Result<Vec<(Located, Resolved<'a>)>, WeightPatchError> {
    let mut out = Vec::with_capacity(patches.len());
    for p in patches {
        let at = locate(src, index, p)?;
        let edits = resolve_edits(src, &at, p)?;
        out.push((at, edits));
    }
    let first = out[0].0.canonical_id.clone();
    if let Some((other, _)) = out.iter().find(|(a, _)| a.canonical_id != first) {
        return Err(WeightPatchError::SpansRepresentations {
            first: (patches[0].tensor().to_string(), first),
            other: (other.tensor.name.clone(), other.canonical_id.clone()),
        });
    }
    Ok(out)
}

pub(super) fn locate(
    src: &Path,
    index: &Vindex3Index,
    patch: &DeclaredPatch,
) -> Result<Located, WeightPatchError> {
    let name = patch.tensor();
    let mut holders = Vec::new();
    for (id, entry) in index
        .representations
        .iter()
        .filter(|(_, e)| e.compiled_from.is_none())
    {
        let (header, payload_start) = read_segment_header(&src.join(&entry.segment))?;
        if let Some(t) = header.tensors.iter().find(|t| t.name == name) {
            holders.push((
                id.clone(),
                entry.clone(),
                header.clone(),
                payload_start,
                t.clone(),
            ));
        }
    }
    let (canonical_id, entry, header, payload_start, tensor) = match holders.len() {
        0 => return Err(WeightPatchError::UnknownTensor(name.to_string())),
        1 => holders.remove(0),
        _ => {
            return Err(WeightPatchError::AmbiguousTensor {
                tensor: name.to_string(),
                holders: holders.into_iter().map(|h| h.0).collect(),
            })
        }
    };
    if tensor.dtype != BF16_DTYPE {
        return Err(WeightPatchError::UnsupportedDtype {
            tensor: name.to_string(),
            dtype: tensor.dtype,
        });
    }
    let (rows, cols) = match tensor.shape.as_slice() {
        [r, c] => (*r, *c),
        _ => {
            return Err(WeightPatchError::NotAMatrix {
                tensor: name.to_string(),
                shape: tensor.shape,
            })
        }
    };
    for (row, col) in patch.coords() {
        if row >= rows || col >= cols {
            return Err(WeightPatchError::OutOfBounds {
                tensor: name.to_string(),
                row,
                col,
                rows,
                cols,
            });
        }
    }
    let mut derived = Vec::new();
    for (id, e) in index
        .representations
        .iter()
        .filter(|(_, e)| e.compiled_from.is_some())
    {
        if e.compiled_from.as_deref() != Some(canonical_id.as_str()) {
            return Err(WeightPatchError::DerivedFromOther {
                derived: id.clone(),
                canonical: canonical_id,
            });
        }
        derived.push((id.clone(), e.clone()));
    }
    Ok(Located {
        canonical_id,
        entry,
        header,
        payload_start,
        tensor,
        derived,
    })
}

/// The ledger's edits: the stored bits read from the source, and the new
/// bits: computed by the declared rule for a delta, taken as declared for a
/// structured op's result.
pub(super) fn resolve_edits<'a>(
    src: &Path,
    at: &Located,
    patch: &'a DeclaredPatch,
) -> Result<Resolved<'a>, WeightPatchError> {
    if let DeclaredPatch::Operations(p) = patch {
        if at.tensor.shape != p.shape {
            return Err(WeightPatchError::Structured(format!(
                "`{}` is {:?} in the container, but was compiled as {:?}",
                p.tensor, at.tensor.shape, p.shape
            )));
        }
        // The ops were resolved against a base; refuse to write them over another.
        if stored_tensor_sha256(src, at)? != p.base_tensor_sha256 {
            return Err(WeightPatchError::Structured(format!(
                "`{}` differs from the base its operations were compiled against",
                p.tensor
            )));
        }
        return Ok(Resolved::Operations(p));
    }
    let file = File::open(src.join(&at.entry.segment))?;
    let cols = at.tensor.shape[1] as u64;
    let name = patch.tensor();
    let read_old = |row: usize, col: usize| -> Result<u16, WeightPatchError> {
        let mut stored = [0u8; BF16_BYTES as usize];
        let offset =
            at.payload_start + at.tensor.offset + (row as u64 * cols + col as u64) * BF16_BYTES;
        read_exact_at(&file, &mut stored, offset)?;
        let old_bits = u16::from_le_bytes(stored);
        if !bf16::is_finite(old_bits) {
            return Err(WeightPatchError::NonFiniteSource {
                tensor: name.to_string(),
                row,
                col,
            });
        }
        Ok(old_bits)
    };
    let mut out = Vec::with_capacity(patch.len());
    match patch {
        DeclaredPatch::Deltas(p) => {
            for e in &p.edits {
                let old_bits = read_old(e.row, e.col)?;
                let new_bits =
                    bf16::patched_bits(old_bits, e.delta).ok_or(WeightPatchError::Overflow {
                        row: e.row,
                        col: e.col,
                    })?;
                out.push(LedgerEdit {
                    row: e.row,
                    col: e.col,
                    delta: e.delta,
                    delta_bits: e.delta.to_bits(),
                    old_bits,
                    new_bits,
                    encoding: EditEncoding::Delta,
                });
            }
        }
        DeclaredPatch::Results(p) => {
            for e in &p.edits {
                let old_bits = read_old(e.row, e.col)?;
                if !bf16::is_finite(e.new_bits) {
                    return Err(WeightPatchError::NonFiniteResult {
                        row: e.row,
                        col: e.col,
                    });
                }
                out.push(LedgerEdit {
                    row: e.row,
                    col: e.col,
                    delta: 0.0,
                    delta_bits: 0,
                    old_bits,
                    new_bits: e.new_bits,
                    encoding: EditEncoding::ResultBits,
                });
            }
        }
        DeclaredPatch::Operations(_) => unreachable!("returned above"),
    }
    Ok(Resolved::Edits(out))
}

/// How much of a stored tensor is held in memory at a time while it is hashed.
const HASH_CHUNK_BYTES: usize = 1 << 20;

/// The digest [`tensor_sha256`](super::super::op_target::tensor_sha256) gives the located tensor, read from its
/// segment a chunk at a time. Stored BF16 is little-endian, which is the byte order that digest is defined over, so
/// the stored bytes are hashed as they are and the tensor is never held whole: an operation target may be gigabytes.
fn stored_tensor_sha256(src: &Path, at: &Located) -> Result<String, WeightPatchError> {
    use sha2::{Digest, Sha256};
    use std::io::{Read, Seek, SeekFrom};
    let mut file = File::open(src.join(&at.entry.segment))?;
    file.seek(SeekFrom::Start(at.payload_start + at.tensor.offset))?;
    let elements: u64 = at.tensor.shape.iter().map(|d| *d as u64).product();
    let mut stored = file.take(elements * BF16_BYTES);
    let mut hash = Sha256::new();
    let mut chunk = vec![0u8; HASH_CHUNK_BYTES];
    // A segment shorter than its tensor table says hashes to another digest, and is refused as another base.
    loop {
        let read = stored.read(&mut chunk)?;
        if read == 0 {
            return Ok(super::hex(&hash.finalize()));
        }
        hash.update(&chunk[..read]);
    }
}
