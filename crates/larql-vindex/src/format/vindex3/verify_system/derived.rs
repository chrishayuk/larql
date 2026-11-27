//! Source ≡ encoded for DERIVED tensors, element by element.
//!
//! A storage-layout derivation (an input-major weight transposed, a fused
//! projection split — `larql_models::inventory::derive`) means the
//! container does not hold the checkpoint's bytes in the checkpoint's
//! order, so the payload hash alone proves only that the encoder and the
//! re-hash ran the same transform. This audit asks the stronger question
//! from the other end, independently of that transform:
//!
//! ```text
//! tiling     the parts derived from one stored tensor cover its output
//!            axis exactly once — no stored element dropped or doubled
//! elements   every element the CONTAINER holds for a derived tensor is
//!            the stored element the declared slice/transpose maps it to,
//!            computed here from the index arithmetic alone
//! ```
//!
//! The container side is read from the segment itself, never from the
//! encoder's output, so a wrong transpose, a swapped split or a stale
//! container fails here even if both hash sides agree with each other.

use std::collections::BTreeMap;
use std::io::{Read, Seek, SeekFrom};
use std::path::Path;

use larql_models::inventory::derive::{element_bytes, TensorDerivation};
use larql_models::inventory::{ArchitectureInventory, TensorFact};

use crate::format::vindex3::encode::binding_owner;
use crate::format::vindex3::encode::segment::{PlannedTensor, SegmentTensor};
use crate::format::vindex3::encode::source::{ArtifactSource, TensorSource};
use crate::format::vindex3::graph::LogicalObject;

/// Audit every derived tensor of one representation. `Ok(n)` is the
/// number of derived tensors checked (0 for an object with none).
pub(super) fn audit_derived_tensors(
    object: &LogicalObject,
    planned: &[PlannedTensor],
    header: &[SegmentTensor],
    segment: &Path,
    payload_start: u64,
    sources: &BTreeMap<&str, ArtifactSource>,
    inventories: &BTreeMap<&str, &ArchitectureInventory>,
) -> Result<usize, String> {
    let owner_of = |name: &str| {
        let artifact = binding_owner(object, name)?;
        Some((sources.get(artifact)?, *inventories.get(artifact)?))
    };
    let mut checked = 0;
    let mut by_source: BTreeMap<(String, String), Vec<&TensorDerivation>> = BTreeMap::new();
    for tensor in planned {
        let Some((source, inventory)) = owner_of(&tensor.source_name) else {
            return Err(format!("`{}` has no owning artifact", tensor.source_name));
        };
        let Some(fact) = fact_of(inventory, &tensor.source_name) else {
            continue;
        };
        let Some(derivation) = &fact.derivation else {
            continue;
        };
        by_source
            .entry((inventory.path.clone(), derivation.source.clone()))
            .or_default()
            .push(derivation);
        let entry = header
            .iter()
            .find(|e| e.name == tensor.relative_name)
            .ok_or_else(|| format!("`{}` is not in the segment table", tensor.relative_name))?;
        let mut stored = Vec::new();
        source
            .stream_payload(&derivation.source, &mut stored, &mut |_| {})
            .map_err(|e| e.to_string())?;
        let encoded = read_range(segment, payload_start + entry.offset, entry.len)?;
        check_elements(fact, derivation, &stored, &encoded)?;
        checked += 1;
    }
    for ((_, source), parts) in &by_source {
        check_tiling(source, parts)?;
    }
    Ok(checked)
}

fn fact_of<'a>(inventory: &'a ArchitectureInventory, name: &str) -> Option<&'a TensorFact> {
    inventory
        .tensors
        .tensors
        .binary_search_by(|t| t.name.as_str().cmp(name))
        .ok()
        .map(|i| &inventory.tensors.tensors[i])
}

fn read_range(path: &Path, offset: u64, len: u64) -> Result<Vec<u8>, String> {
    let mut file = std::fs::File::open(path).map_err(|e| e.to_string())?;
    file.seek(SeekFrom::Start(offset))
        .map_err(|e| e.to_string())?;
    let len = usize::try_from(len).map_err(|e| e.to_string())?;
    let mut out = vec![0u8; len];
    file.read_exact(&mut out).map_err(|e| e.to_string())?;
    Ok(out)
}

/// The stored element index of derived element `k` (row-major over the
/// DERIVED shape), from the declaration's arithmetic alone.
fn stored_index(d: &TensorDerivation, derived_shape: &[usize], k: usize) -> Option<usize> {
    match (d.source_shape.as_slice(), derived_shape) {
        // A vector: an offset into it.
        ([_], [_]) => Some(k + d.slice.map_or(0, |s| s.start)),
        ([_, cols], [_, dc]) => {
            let (r, c) = (k / dc, k % dc);
            // Undo the transpose: derived (r, c) is sliced (c, r).
            let (sr, sc) = if d.transpose { (c, r) } else { (r, c) };
            let (sr, sc) = match d.slice {
                Some(s) if s.axis == 0 => (sr + s.start, sc),
                Some(s) => (sr, sc + s.start),
                None => (sr, sc),
            };
            Some(sr * cols + sc)
        }
        _ => None,
    }
}

fn check_elements(
    fact: &TensorFact,
    d: &TensorDerivation,
    stored: &[u8],
    encoded: &[u8],
) -> Result<(), String> {
    let elem = element_bytes(&fact.dtype)
        .ok_or_else(|| format!("`{}`: dtype {} has no element width", fact.name, fact.dtype))?;
    let count: usize = fact.shape.iter().product();
    if encoded.len() != count * elem {
        return Err(format!(
            "`{}`: container holds {} bytes for {:?} of {}",
            fact.name,
            encoded.len(),
            fact.shape,
            fact.dtype
        ));
    }
    for k in 0..count {
        let s = stored_index(d, &fact.shape, k)
            .ok_or_else(|| format!("`{}`: no index map for {:?}", fact.name, d.source_shape))?;
        let want = stored
            .get(s * elem..(s + 1) * elem)
            .ok_or_else(|| format!("`{}`: element {k} maps outside `{}`", fact.name, d.source))?;
        if &encoded[k * elem..(k + 1) * elem] != want {
            return Err(format!(
                "`{}`: container element {k} is not stored element {s} of `{}`",
                fact.name, d.source
            ));
        }
    }
    Ok(())
}

/// The parts derived from one stored tensor cover its sliced axis exactly
/// once; an unsliced derivation is the only one of its source.
fn check_tiling(source: &str, parts: &[&TensorDerivation]) -> Result<(), String> {
    let slices: Vec<_> = parts.iter().filter_map(|d| d.slice).collect();
    if slices.is_empty() {
        return if parts.len() == 1 {
            Ok(())
        } else {
            Err(format!("`{source}` is derived whole {} times", parts.len()))
        };
    }
    if slices.len() != parts.len() || slices.iter().any(|s| s.axis != slices[0].axis) {
        return Err(format!("`{source}`'s parts do not split one axis"));
    }
    let shape = &parts[0].source_shape;
    let axis_len = if shape.len() == 1 {
        shape[0]
    } else {
        shape[slices[0].axis]
    };
    let mut ranges: Vec<(usize, usize)> = slices.iter().map(|s| (s.start, s.len)).collect();
    ranges.sort_unstable();
    let mut next = 0;
    for (start, len) in ranges {
        if start != next {
            return Err(format!(
                "`{source}`'s parts leave a gap or overlap at {next}"
            ));
        }
        next = start + len;
    }
    if next != axis_len {
        return Err(format!(
            "`{source}`'s parts cover {next} of its {axis_len} output rows"
        ));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use larql_models::inventory::derive::AxisSlice;

    fn d(slice: Option<AxisSlice>, transpose: bool, shape: &[usize]) -> TensorDerivation {
        TensorDerivation {
            source: "w".into(),
            source_shape: shape.to_vec(),
            slice,
            transpose,
        }
    }

    #[test]
    fn the_index_map_undoes_slice_and_transpose() {
        // Stored [2, 6]; part = columns 2..4, transposed to [2, 2].
        let part = d(
            Some(AxisSlice {
                axis: 1,
                start: 2,
                len: 2,
            }),
            true,
            &[2, 6],
        );
        // derived (0,0) = stored (0,2); (0,1) = stored (1,2); (1,0) = (0,3).
        assert_eq!(stored_index(&part, &[2, 2], 0), Some(2));
        assert_eq!(stored_index(&part, &[2, 2], 1), Some(8));
        assert_eq!(stored_index(&part, &[2, 2], 2), Some(3));
    }

    #[test]
    fn tiling_refuses_a_gap_an_overlap_and_a_short_cover() {
        let s = |start, len| {
            Some(AxisSlice {
                axis: 0,
                start,
                len,
            })
        };
        let ok = [
            d(s(0, 2), false, &[6]),
            d(s(2, 2), false, &[6]),
            d(s(4, 2), false, &[6]),
        ];
        assert!(check_tiling("b", &ok.iter().collect::<Vec<_>>()).is_ok());
        let gap = [d(s(0, 2), false, &[6]), d(s(4, 2), false, &[6])];
        assert!(check_tiling("b", &gap.iter().collect::<Vec<_>>()).is_err());
        let overlap = [d(s(0, 4), false, &[6]), d(s(2, 4), false, &[6])];
        assert!(check_tiling("b", &overlap.iter().collect::<Vec<_>>()).is_err());
        let twice = [d(None, true, &[2, 2]), d(None, true, &[2, 2])];
        assert!(check_tiling("w", &twice.iter().collect::<Vec<_>>()).is_err());
    }
}
