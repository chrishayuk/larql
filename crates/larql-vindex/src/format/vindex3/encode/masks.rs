//! Stored attention-mask buffers, checked against the declaration that
//! claims to reproduce them.
//!
//! A family may name registered buffers its checkpoint serialises
//! (`larql_models::config::NonParameterBuffer` — GPT-2's `attn.bias`,
//! GPT-Neo's `attn.attention.bias`); the inventory moves them out of the
//! operand estate. That exclusion is a CLAIM — "this layer's declared
//! span computes exactly this mask" — and this module verifies it from
//! the stored bytes before any container is written, and again at G4:
//!
//! ```text
//! causal mask   [1, 1, n, n], nonzero exactly where key j is visible to
//!               query i under the layer's declared span: j <= i on a full
//!               layer, i - window < j <= i on a sliding one
//! mask fill     a scalar at or below MASK_FILL_CEILING, whose softmax
//!               weight underflows to exactly zero in f32
//! ```
//!
//! A buffer that disagrees refuses the encode by name. Nothing is ever
//! excluded unchecked: a buffer whose layer has no declared span, or
//! whose dtype this module cannot read, refuses too.

use std::collections::BTreeMap;

use larql_models::config::BufferKind;
use larql_models::inventory::derive::ExcludedTensor;
use larql_models::inventory::ArchitectureInventory;

use super::source::TensorSource;
use crate::error::VindexError;
use crate::format::vindex3::graph::policy::{AttentionLayerPolicy, AttentionSpan};
use crate::format::vindex3::graph::{ComponentRole, SystemGraph};

/// A fill at or below this makes a masked score's weight exactly zero:
/// `exp(-1e4)` underflows f32 whatever the visible scores are, and every
/// GPT-2-lineage checkpoint stores `-1e4` or `finfo.min`.
pub const MASK_FILL_CEILING: f32 = -1e4;

/// Check every excluded buffer of every artifact. `Ok(n)` is the number
/// checked; the first disagreement is an error naming the tensor.
pub fn check_excluded_buffers(
    graph: &SystemGraph,
    named: &[(String, ArchitectureInventory)],
    sources: &BTreeMap<&str, &dyn TensorSource>,
) -> Result<usize, VindexError> {
    let mut checked = 0;
    for (artifact, inventory) in named {
        if inventory.tensors.excluded.is_empty() {
            continue;
        }
        let source = sources.get(artifact.as_str()).ok_or_else(|| {
            VindexError::Parse(format!("no payload source for artifact `{artifact}`"))
        })?;
        let table = graph
            .components
            .iter()
            .find(|c| &c.source_artifact == artifact && c.role == ComponentRole::PrimaryText)
            .and_then(|c| c.attention.as_ref());
        for buffer in &inventory.tensors.excluded {
            check_one(buffer, table.map(Vec::as_slice), *source).map_err(|e| {
                VindexError::Parse(format!("refusing to exclude `{}`: {e}", buffer.name))
            })?;
            checked += 1;
        }
    }
    Ok(checked)
}

fn check_one(
    buffer: &ExcludedTensor,
    table: Option<&[AttentionLayerPolicy]>,
    source: &dyn TensorSource,
) -> Result<(), String> {
    let mut bytes = Vec::new();
    source
        .stream_payload(&buffer.name, &mut bytes, &mut |_| {})
        .map_err(|e| e.to_string())?;
    let values = decode(&buffer.dtype, &bytes)?;
    match buffer.kind {
        BufferKind::MaskFill => match values.as_slice() {
            [fill] if *fill <= MASK_FILL_CEILING => Ok(()),
            [fill] => Err(format!(
                "the mask fill is {fill}, above {MASK_FILL_CEILING}: a masked score would keep \
                 weight, which no declared span computes"
            )),
            other => Err(format!(
                "a mask fill holds one value, found {}",
                other.len()
            )),
        },
        BufferKind::CausalMask => {
            let n = match buffer.shape.as_slice() {
                [1, 1, a, b] if a == b => *a,
                other => return Err(format!("a causal mask is [1, 1, n, n], found {other:?}")),
            };
            let layer = layer_of(&buffer.name)
                .ok_or("the buffer names no layer, so no declared span can answer for it")?;
            let policy = table
                .and_then(|t| t.get(layer))
                .ok_or_else(|| format!("layer {layer} has no declared attention span"))?;
            let visible: Box<dyn Fn(usize, usize) -> bool> = match (policy.span, policy.window) {
                (Some(AttentionSpan::Full), _) => Box::new(|i, j| j <= i),
                (Some(AttentionSpan::Sliding), Some(w)) => {
                    Box::new(move |i, j| j <= i && j + w > i)
                }
                (span, window) => {
                    return Err(format!(
                        "layer {layer}'s span {span:?} (window {window:?}) has no mask form to \
                         check against"
                    ))
                }
            };
            for i in 0..n {
                for j in 0..n {
                    let stored = values[i * n + j] != 0.0;
                    if stored != visible(i, j) {
                        return Err(format!(
                            "layer {layer}: stored mask at query {i}, key {j} is {stored}, the \
                             declared span says {}",
                            visible(i, j)
                        ));
                    }
                }
            }
            Ok(())
        }
    }
}

/// The first all-digit segment of a tensor name: its layer.
fn layer_of(name: &str) -> Option<usize> {
    name.split('.').find_map(|s| s.parse().ok())
}

/// Every element as f32 (exact for the 0/1 and fill values a mask holds).
fn decode(dtype: &str, bytes: &[u8]) -> Result<Vec<f32>, String> {
    let f16 = |b: &[u8]| larql_models::quant::half::f16_to_f32(u16::from_le_bytes([b[0], b[1]]));
    let bf16 = |b: &[u8]| larql_models::quant::half::bf16_to_f32(u16::from_le_bytes([b[0], b[1]]));
    Ok(match dtype {
        "F32" => bytes
            .chunks_exact(4)
            .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
            .collect(),
        "F16" => bytes.chunks_exact(2).map(f16).collect(),
        "BF16" => bytes.chunks_exact(2).map(bf16).collect(),
        "BOOL" | "U8" => bytes.iter().map(|b| f32::from(*b)).collect(),
        "I8" => bytes.iter().map(|b| f32::from(*b as i8)).collect(),
        other => {
            return Err(format!(
                "dtype {other} is not a mask dtype this build reads"
            ))
        }
    })
}
