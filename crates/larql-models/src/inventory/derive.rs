//! Storage-layout normalisation at the inventory boundary.
//!
//! A family whose checkpoint stores projections input-major (`Conv1D`) or
//! fused (`q | k | v` in one tensor) declares it through
//! [`SourceLayout`]; this module rewrites the scanned tensor list so that
//! everything downstream sees canonical `[out, in]` operands, one per
//! logical projection. Each rewritten tensor records the
//! [`TensorDerivation`] that produces its bytes from the stored tensor, so
//! the rewrite is a declaration the encoder executes, never an
//! interpretation a second reader could make differently.
//!
//! Registered buffers the family names (`attn.bias`, the causal-mask
//! buffer older GPT-2 conversions serialise) leave the operand list and
//! are recorded in [`ExcludedTensor`] with their reason — never silently
//! dropped.

use serde::{Deserialize, Serialize};

use crate::config::SourceLayout;

use super::report::{TensorFact, TensorGroup, TensorInventory};

/// Leaf names of a projection module's two tensors.
const WEIGHT_LEAF: &str = "weight";
const BIAS_LEAF: &str = "bias";

/// A contiguous range of one axis of the stored tensor.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct AxisSlice {
    pub axis: usize,
    pub start: usize,
    pub len: usize,
}

/// How one inventory tensor's bytes are produced from a stored tensor:
/// slice (optional) then transpose (optional). Both are byte
/// permutations, exact for every dtype.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct TensorDerivation {
    /// Full name of the stored tensor the bytes come from.
    pub source: String,
    /// Its stored shape.
    pub source_shape: Vec<usize>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub slice: Option<AxisSlice>,
    /// Swap the two axes of the (sliced) 2-D tensor.
    #[serde(default, skip_serializing_if = "std::ops::Not::not")]
    pub transpose: bool,
}

/// A stored tensor that is not an operand, and why.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ExcludedTensor {
    pub name: String,
    pub dtype: String,
    pub shape: Vec<usize>,
    pub reason: String,
    /// The contract its stored contents must meet; the encoder checks it
    /// against the declared span and refuses on any disagreement.
    pub kind: crate::config::BufferKind,
}

/// Bytes per element of a safetensors dtype this module can permute.
pub fn element_bytes(dtype: &str) -> Option<usize> {
    match dtype {
        "F64" | "I64" | "U64" => Some(8),
        "F32" | "I32" | "U32" => Some(4),
        "F16" | "BF16" | "I16" | "U16" => Some(2),
        "F8_E4M3" | "F8_E5M2" | "I8" | "U8" | "BOOL" => Some(1),
        _ => None,
    }
}

/// Split a tensor name into its layer prefix (through the first numeric
/// segment, with the trailing dot) and the layer-relative remainder.
fn split_layer(name: &str) -> Option<(&str, &str)> {
    let mut offset = 0;
    for segment in name.split('.') {
        let end = offset + segment.len();
        if !segment.is_empty() && segment.bytes().all(|b| b.is_ascii_digit()) {
            let rest = name.get(end + 1..)?;
            return Some((&name[..end + 1], rest));
        }
        offset = end + 1;
    }
    None
}

/// Apply `layout` to a scanned inventory in place: input-major weights
/// are transposed, fused projections split, named buffers excluded.
///
/// A tensor whose shape does not fit the declaration (a fused width not
/// divisible by its parts, a weight that is not 2-D, a buffer that is not
/// a 4-D mask) is left exactly as stored, so it reaches operand closure
/// as itself and refuses there by name rather than being guessed into
/// shape here.
pub fn apply_source_layout(layout: &SourceLayout, inventory: &mut TensorInventory) {
    if layout.is_canonical() {
        return;
    }
    let mut out: Vec<TensorFact> = Vec::with_capacity(inventory.tensors.len());
    for tensor in std::mem::take(&mut inventory.tensors) {
        let Some((prefix, relative)) = split_layer(&tensor.name) else {
            out.push(tensor);
            continue;
        };
        if let Some(buffer) = layout.buffers.iter().find(|b| b.suffix == relative) {
            if buffer_shape_fits(buffer.kind, &tensor.shape) {
                inventory.excluded.push(ExcludedTensor {
                    name: tensor.name.clone(),
                    dtype: tensor.dtype.clone(),
                    shape: tensor.shape.clone(),
                    reason: buffer.reason.to_string(),
                    kind: buffer.kind,
                });
                continue;
            }
        }
        let (module, leaf) = match relative.rsplit_once('.') {
            Some((module, leaf)) if leaf == WEIGHT_LEAF || leaf == BIAS_LEAF => (module, leaf),
            _ => {
                out.push(tensor);
                continue;
            }
        };
        let input_major = layout.input_major.contains(&module);
        match layout.fused.iter().find(|f| f.module == module) {
            Some(fused) => match split_fused(&tensor, prefix, leaf, fused, input_major) {
                Some(parts) => out.extend(parts),
                None => out.push(tensor),
            },
            None if input_major && leaf == WEIGHT_LEAF => match transposed(&tensor) {
                Some(t) => out.push(t),
                None => out.push(tensor),
            },
            None => out.push(tensor),
        }
    }
    out.sort_by(|a, b| a.name.cmp(&b.name));
    inventory.tensors = out;
    inventory.total_tensors = inventory.tensors.len();
    inventory.total_bytes = inventory.tensors.iter().map(|t| t.bytes).sum();
    inventory.groups = regroup(&inventory.tensors);
}

/// A causal mask is `[1, 1, n, n]` and a fill a scalar; anything else
/// under the same name is not the buffer the family declared, and stays
/// an operand so closure refuses it by name.
fn buffer_shape_fits(kind: crate::config::BufferKind, shape: &[usize]) -> bool {
    match kind {
        crate::config::BufferKind::CausalMask => matches!(shape, [1, 1, a, b] if a == b),
        crate::config::BufferKind::MaskFill => shape.is_empty() || shape == [1],
    }
}

fn derived_bytes(shape: &[usize], dtype: &str) -> Option<u64> {
    let elems = shape
        .iter()
        .try_fold(1usize, |acc, d| acc.checked_mul(*d))?;
    Some((elems.checked_mul(element_bytes(dtype)?)?) as u64)
}

fn transposed(tensor: &TensorFact) -> Option<TensorFact> {
    let [rows, cols] = tensor.shape[..] else {
        return None;
    };
    Some(TensorFact {
        name: tensor.name.clone(),
        dtype: tensor.dtype.clone(),
        shape: vec![cols, rows],
        bytes: derived_bytes(&[cols, rows], &tensor.dtype)?,
        file: tensor.file.clone(),
        derivation: Some(TensorDerivation {
            source: tensor.name.clone(),
            source_shape: tensor.shape.clone(),
            slice: None,
            transpose: true,
        }),
    })
}

fn split_fused(
    tensor: &TensorFact,
    prefix: &str,
    leaf: &str,
    fused: &crate::config::FusedProjection,
    input_major: bool,
) -> Option<Vec<TensorFact>> {
    let parts = fused.parts.len();
    if parts == 0 {
        return None;
    }
    // The output axis: the last axis of an input-major weight, the first
    // of a canonical weight, the only axis of a bias.
    let (axis, transpose) = match (leaf, tensor.shape.len()) {
        (WEIGHT_LEAF, 2) if input_major => (1, true),
        (WEIGHT_LEAF, 2) => (0, false),
        (BIAS_LEAF, 1) => (0, false),
        _ => return None,
    };
    let total = tensor.shape[axis];
    if total == 0 || !total.is_multiple_of(parts) {
        return None;
    }
    let width = total / parts;
    fused
        .parts
        .iter()
        .enumerate()
        .map(|(i, part)| {
            let mut shape = tensor.shape.clone();
            shape[axis] = width;
            if transpose {
                shape.swap(0, 1);
            }
            Some(TensorFact {
                name: format!("{prefix}{part}.{leaf}"),
                dtype: tensor.dtype.clone(),
                bytes: derived_bytes(&shape, &tensor.dtype)?,
                shape,
                file: tensor.file.clone(),
                derivation: Some(TensorDerivation {
                    source: tensor.name.clone(),
                    source_shape: tensor.shape.clone(),
                    slice: Some(AxisSlice {
                        axis,
                        start: i * width,
                        len: width,
                    }),
                    transpose,
                }),
            })
        })
        .collect()
}

fn regroup(tensors: &[TensorFact]) -> Vec<TensorGroup> {
    let mut groups: std::collections::BTreeMap<String, (usize, u64)> = Default::default();
    for t in tensors {
        let entry = groups
            .entry(super::tensors::group_prefix(&t.name))
            .or_insert((0, 0));
        entry.0 += 1;
        entry.1 += t.bytes;
    }
    groups
        .into_iter()
        .map(|(prefix, (tensors, bytes))| TensorGroup {
            prefix,
            tensors,
            bytes,
        })
        .collect()
}

/// Produce a derived tensor's bytes from the stored tensor's bytes.
///
/// `source` must be exactly the stored payload (`source_shape` elements of
/// `dtype`). Refuses rather than truncates on any disagreement.
pub fn derive_bytes(
    derivation: &TensorDerivation,
    dtype: &str,
    source: &[u8],
) -> Result<Vec<u8>, String> {
    let elem = element_bytes(dtype)
        .ok_or_else(|| format!("dtype `{dtype}` has no element width to permute"))?;
    let shape = &derivation.source_shape;
    let elems = shape
        .iter()
        .try_fold(1usize, |acc, d| acc.checked_mul(*d))
        .ok_or("stored shape overflows")?;
    if elems.checked_mul(elem) != Some(source.len()) {
        return Err(format!(
            "`{}`: {} stored bytes, expected {:?} of {dtype}",
            derivation.source,
            source.len(),
            shape
        ));
    }
    // Normalise to 2-D (`[1, n]` for a vector) so one routine serves both.
    let (rows, cols) = match shape[..] {
        [n] => (1, n),
        [r, c] => (r, c),
        _ => {
            return Err(format!(
                "`{}`: only 1-D and 2-D tensors derive",
                derivation.source
            ))
        }
    };
    let (r0, nr, c0, nc) = match derivation.slice {
        None => (0, rows, 0, cols),
        Some(AxisSlice { axis, start, len }) => {
            let axis = if shape.len() == 1 { 1 } else { axis };
            let bound = if axis == 0 { rows } else { cols };
            if start.checked_add(len).is_none_or(|end| end > bound) {
                return Err(format!(
                    "`{}`: slice outside axis {axis}",
                    derivation.source
                ));
            }
            if axis == 0 {
                (start, len, 0, cols)
            } else {
                (0, rows, start, len)
            }
        }
    };
    let mut out = Vec::with_capacity(nr * nc * elem);
    let at = |r: usize, c: usize| {
        let i = ((r0 + r) * cols + (c0 + c)) * elem;
        &source[i..i + elem]
    };
    if derivation.transpose {
        for c in 0..nc {
            for r in 0..nr {
                out.extend_from_slice(at(r, c));
            }
        }
    } else {
        for r in 0..nr {
            for c in 0..nc {
                out.extend_from_slice(at(r, c));
            }
        }
    }
    Ok(out)
}

#[cfg(test)]
mod tests;
