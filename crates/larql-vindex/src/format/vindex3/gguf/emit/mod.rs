//! Execute plans. Nothing else.
//!
//! This module is deliberately hostile to intelligence. It receives
//! already-resolved metadata and [`LoweredTensorPlan`]s and does what
//! they say: no semantic lookup, no representation choice, no family
//! fallback, no role inference, no target-name construction, no shape
//! correction. It does not know a single role name — every decision
//! the file embodies was made above this boundary, where the gates
//! live. If the emitter needs to know something the plan does not
//! carry, that is a missing plan field, never a lookup here.
//!
//! What executing a plan means:
//!
//! ```text
//! layout      permute bytes as instructed — V-head blocks, column
//!             groups, a singleton squeeze. The permutation is the
//!             plan's; the emitter only moves memory.
//! value       f32 arithmetic on an unquantised lattice, stored F32 —
//!             the exact result of the computation performed.
//! encoding    BF16 passes through, F32 re-encodes, NVFP4 repacks
//!             losslessly into GGML's block ABI with the tensor scale
//!             carried out as the sibling the plan named.
//! ```
//!
//! And what checking one means: [`verify_emitted`] parses the finished
//! file back through the INDEPENDENT reader — the one written for
//! foreign GGUFs, sharing no code with the writer path's descriptors —
//! and requires the metadata to be exactly what was resolved and every
//! tensor to be exactly what was planned, in name, dims, and type,
//! with nothing extra and nothing missing.

use std::io::{self, Read, Write};
use std::path::Path;

use larql_models::loading::gguf::{write_streaming, GgufFile, GgufTensorDescriptor, GgufValue};
use larql_models::quant::half::bf16_to_f32;
use larql_models::quant::nvfp4::NVFP4_GROUP_ELEMS;
use larql_models::quant::nvfp4_ggml::{ggml_nvfp4_bytes, repack_nvfp4, TYPE_NVFP4};

use super::metadata::{MetaKey, MetaValue};
use super::plan::{LayoutTransform, LoweredTensorPlan, RepresentationKind, ValueTransform};
use crate::format::vindex3::represent::nvfp4_pack::{split, PackLayout};
use crate::VindexError;

pub const TYPE_F32: u32 = 0;
pub const TYPE_BF16: u32 = 30;

/// Grouped → tiled head order: `new[a·K + b] = old[b·r + a]`.
///
/// The single permutation behind every V-head transform. Returned as
/// "which old head each new position reads", which is the direction a
/// copy wants.
fn head_perm(key_heads: usize, v_per_k: usize) -> Vec<usize> {
    let mut p = Vec::with_capacity(key_heads * v_per_k);
    for a in 0..v_per_k {
        for b in 0..key_heads {
            p.push(b * v_per_k + a);
        }
    }
    p
}

/// Permute `perm.len()` equal blocks starting at `offset`. Bytes before
/// and after the region do not move.
fn permute_blocks(
    bytes: &mut [u8],
    offset: usize,
    block: usize,
    perm: &[usize],
) -> Result<(), VindexError> {
    let end = offset + block * perm.len();
    if bytes.len() < end {
        return Err(VindexError::Parse(format!(
            "emit: permutation reads {end} bytes of a {}-byte payload — the plan and the \
             tensor disagree about its extent",
            bytes.len()
        )));
    }
    let region = bytes[offset..end].to_vec();
    for (new_i, &old_i) in perm.iter().enumerate() {
        bytes[offset + new_i * block..offset + (new_i + 1) * block]
            .copy_from_slice(&region[old_i * block..(old_i + 1) * block]);
    }
    Ok(())
}

/// Execute a plan whose source stores numbers directly.
fn lower_unquantised(plan: &LoweredTensorPlan, mut bytes: Vec<u8>) -> Result<Vec<u8>, VindexError> {
    let src_elem = match plan.source_representation {
        RepresentationKind::F32 => 4usize,
        RepresentationKind::Bf16 => 2,
        RepresentationKind::Nvfp4 => {
            return Err(VindexError::Parse(format!(
                "emit: `{}` reached the unquantised path with an NVFP4 source",
                plan.target_name
            )))
        }
    };
    let mut shape = plan.source_shape.clone();
    for t in &plan.layout {
        match t {
            LayoutTransform::SqueezeSingletonAxis { axis } => {
                // The constructor proved the axis is a singleton, so no
                // byte moves; only the bookkeeping shape changes.
                shape.remove(*axis);
            }
            LayoutTransform::ReorderVRows {
                key_heads,
                v_per_k,
                head_dim,
                v_offset_rows,
            } => {
                let row_bytes = shape[1..].iter().product::<u64>() as usize * src_elem;
                permute_blocks(
                    &mut bytes,
                    v_offset_rows * row_bytes,
                    head_dim * row_bytes,
                    &head_perm(*key_heads, *v_per_k),
                )?;
            }
            LayoutTransform::ReorderVColumnsByGroups {
                key_heads, v_per_k, ..
            } => {
                let heads = key_heads * v_per_k;
                let cols = shape[1] as usize;
                if !cols.is_multiple_of(heads) {
                    return Err(VindexError::Parse(format!(
                        "emit: `{}` has {cols} columns, not divisible into {heads} V heads",
                        plan.target_name
                    )));
                }
                let row_bytes = cols * src_elem;
                let head_block = (cols / heads) * src_elem;
                let perm = head_perm(*key_heads, *v_per_k);
                for r in 0..shape[0] as usize {
                    permute_blocks(&mut bytes, r * row_bytes, head_block, &perm)?;
                }
            }
        }
    }
    match (plan.source_representation, plan.target_type) {
        (RepresentationKind::Bf16, TYPE_BF16) if plan.value.is_empty() => Ok(bytes),
        (RepresentationKind::F32 | RepresentationKind::Bf16, TYPE_F32) => {
            let mut values: Vec<f32> = match plan.source_representation {
                RepresentationKind::Bf16 => bytes
                    .chunks_exact(2)
                    .map(|c| bf16_to_f32(u16::from_le_bytes([c[0], c[1]])))
                    .collect(),
                _ => bytes
                    .chunks_exact(4)
                    .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
                    .collect(),
            };
            for t in &plan.value {
                match t {
                    ValueTransform::MaterializeLogDecay => {
                        for v in &mut values {
                            *v = -v.exp();
                        }
                    }
                    ValueTransform::ApplyWeightOffset(o) => {
                        for v in &mut values {
                            *v += o;
                        }
                    }
                }
            }
            let mut out = Vec::with_capacity(values.len() * 4);
            for v in values {
                out.extend_from_slice(&v.to_le_bytes());
            }
            Ok(out)
        }
        (src, ty) => Err(VindexError::Parse(format!(
            "emit: `{}` plans {src:?} → ggml type {ty} — no such lowering is defined, and \
             inventing one here is exactly what this module refuses to do",
            plan.target_name
        ))),
    }
}

/// Execute a plan whose source is an NVFP4 pack. Codes are moved and
/// re-arranged, never decoded: the representation that was measured is
/// the representation that leaves.
fn lower_quantised(
    plan: &LoweredTensorPlan,
    payload: &[u8],
) -> Result<(Vec<u8>, f32), VindexError> {
    let [rows, k] = plan.source_shape[..] else {
        return Err(VindexError::Parse(format!(
            "emit: `{}` is NVFP4 with shape {:?} — the pack format is defined for matrices",
            plan.target_name, plan.source_shape
        )));
    };
    let (rows, k) = (rows as usize, k as usize);
    let layout = PackLayout::derive(&[rows, k], &plan.source)?;
    let (codes, scales, tensor_scale) = split(payload, &layout, &plan.source)?;
    let mut codes = codes.to_vec();
    let mut scales = scales.to_vec();
    let code_row = k / 2;
    let scale_row = k / NVFP4_GROUP_ELEMS;

    for t in &plan.layout {
        match t {
            LayoutTransform::ReorderVRows {
                key_heads,
                v_per_k,
                head_dim,
                v_offset_rows,
            } => {
                let perm = head_perm(*key_heads, *v_per_k);
                permute_blocks(
                    &mut codes,
                    v_offset_rows * code_row,
                    head_dim * code_row,
                    &perm,
                )?;
                permute_blocks(
                    &mut scales,
                    v_offset_rows * scale_row,
                    head_dim * scale_row,
                    &perm,
                )?;
            }
            LayoutTransform::ReorderVColumnsByGroups {
                key_heads,
                v_per_k,
                groups_per_head,
            } => {
                let heads = key_heads * v_per_k;
                if k != heads * groups_per_head * NVFP4_GROUP_ELEMS {
                    return Err(VindexError::Parse(format!(
                        "emit: `{}` has K={k} but the plan says {heads} heads of \
                         {groups_per_head} groups — the group accounting and the tensor \
                         disagree",
                        plan.target_name
                    )));
                }
                let perm = head_perm(*key_heads, *v_per_k);
                let head_code = groups_per_head * NVFP4_GROUP_ELEMS / 2;
                for r in 0..rows {
                    permute_blocks(&mut codes, r * code_row, head_code, &perm)?;
                    permute_blocks(&mut scales, r * scale_row, *groups_per_head, &perm)?;
                }
            }
            LayoutTransform::SqueezeSingletonAxis { .. } => {
                return Err(VindexError::Parse(format!(
                    "emit: `{}` plans a squeeze on a quantised pack — no such lowering exists",
                    plan.target_name
                )));
            }
        }
    }
    let ggml = repack_nvfp4(&codes, &scales, tensor_scale, rows, k)
        .map_err(|e| VindexError::Parse(format!("emit: `{}`: {e}", plan.target_name)))?;
    Ok((ggml.blocks, tensor_scale))
}

/// The resolved metadata table, in the writer's vocabulary. A plain
/// re-spelling: nothing is added, renamed, or defaulted.
pub fn metadata_to_gguf(table: &[MetaKey]) -> Vec<(String, GgufValue)> {
    table
        .iter()
        .map(|m| {
            let value = match &m.value {
                MetaValue::Str(s) => GgufValue::String(s.clone()),
                MetaValue::U32(v) => GgufValue::U32(*v),
                MetaValue::F32(v) => GgufValue::F32(*v),
                MetaValue::ArrU32(vs) => {
                    GgufValue::Array(vs.iter().map(|v| GgufValue::U32(*v)).collect())
                }
            };
            (m.key.clone(), value)
        })
        .collect()
}

/// One descriptor slot: a planned tensor, or the scale sibling its
/// plan named.
enum Slot {
    Tensor(usize),
    Scale(usize),
}

/// The declared payload length for a plan, from its target alone.
fn target_len(plan: &LoweredTensorPlan) -> Result<u64, VindexError> {
    let elems: u64 = plan.target_shape.iter().product();
    match plan.target_type {
        TYPE_F32 => Ok(elems * 4),
        TYPE_BF16 => Ok(elems * 2),
        TYPE_NVFP4 => ggml_nvfp4_bytes(elems as usize)
            .map(|b| b as u64)
            .map_err(|e| VindexError::Parse(format!("emit: `{}`: {e}", plan.target_name))),
        other => Err(VindexError::Parse(format!(
            "emit: `{}` plans ggml type {other}, which this emitter cannot express",
            plan.target_name
        ))),
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct EmitReport {
    pub tensors: usize,
    pub scale_siblings: usize,
    pub metadata_keys: usize,
}

/// Write the file the plans describe. `open_source` yields each plan's
/// raw source payload by its semantic address; a plan that needs no
/// transformation streams through without buffering.
pub fn emit_gguf(
    metadata: &[(String, GgufValue)],
    plans: &[LoweredTensorPlan],
    open_source: &mut dyn FnMut(&str) -> io::Result<Box<dyn Read>>,
    out: &Path,
) -> Result<EmitReport, VindexError> {
    let mut descriptors = Vec::new();
    let mut slots = Vec::new();
    for (i, plan) in plans.iter().enumerate() {
        // GGUF dims are fastest-varying first; the plan's are row-major.
        let mut dims: Vec<u64> = plan.target_shape.clone();
        dims.reverse();
        descriptors.push(GgufTensorDescriptor {
            name: plan.target_name.clone(),
            dims,
            ggml_type: plan.target_type,
            len: target_len(plan)?,
        });
        slots.push(Slot::Tensor(i));
        if let Some(scale_name) = &plan.scale_tensor {
            descriptors.push(GgufTensorDescriptor {
                name: scale_name.clone(),
                dims: vec![1],
                ggml_type: TYPE_F32,
                len: 4,
            });
            slots.push(Slot::Scale(i));
        }
    }

    let mut scales: Vec<Option<f32>> = vec![None; plans.len()];
    let mut failure: Option<VindexError> = None;
    let result = write_streaming(metadata, &descriptors, out, |slot, w| {
        let mut emit = |slot: &Slot, w: &mut dyn Write, scales: &mut Vec<Option<f32>>| match slot {
            Slot::Tensor(i) => {
                let plan = &plans[*i];
                let mut reader = open_source(&plan.source)?;
                let passthrough = plan.layout.is_empty()
                    && plan.value.is_empty()
                    && matches!(
                        (plan.source_representation, plan.target_type),
                        (RepresentationKind::Bf16, TYPE_BF16) | (RepresentationKind::F32, TYPE_F32)
                    );
                if passthrough {
                    io::copy(&mut reader, w)?;
                    return Ok(());
                }
                let mut payload = Vec::new();
                reader.read_to_end(&mut payload)?;
                if plan.source_representation == RepresentationKind::Nvfp4 {
                    let (blocks, scale) =
                        lower_quantised(plan, &payload).map_err(io::Error::other)?;
                    scales[*i] = Some(scale);
                    w.write_all(&blocks)?;
                } else {
                    let bytes = lower_unquantised(plan, payload).map_err(io::Error::other)?;
                    w.write_all(&bytes)?;
                }
                Ok(())
            }
            Slot::Scale(i) => {
                let scale = scales[*i].ok_or_else(|| {
                    io::Error::other(format!(
                        "emit: `{}` names a scale sibling but its source carried no \
                             tensor scale",
                        plans[*i].target_name
                    ))
                })?;
                w.write_all(&scale.to_le_bytes())
            }
        };
        emit(&slots[slot], w, &mut scales).inspect_err(|e| {
            failure = Some(VindexError::Parse(format!("emit: {e}")));
        })
    });
    if let Err(e) = result {
        return Err(failure.unwrap_or(VindexError::Io(e)));
    }
    Ok(EmitReport {
        tensors: plans.len(),
        scale_siblings: plans.iter().filter(|p| p.scale_tensor.is_some()).count(),
        metadata_keys: metadata.len(),
    })
}

#[derive(Debug, Clone, PartialEq)]
pub struct VerifyReport {
    pub tensors: usize,
    pub scale_siblings: usize,
    pub nvfp4_tensors: usize,
    pub metadata_keys: usize,
}

/// Parse the emitted file back through the independent reader and
/// require it to be exactly what was resolved and planned. Every
/// mismatch is collected, not just the first — a wrong file should
/// name everything wrong with it.
pub fn verify_emitted(
    path: &Path,
    metadata: &[(String, GgufValue)],
    plans: &[LoweredTensorPlan],
    required: &[&str],
) -> Result<VerifyReport, Vec<String>> {
    let gguf = GgufFile::open(path).map_err(|e| vec![format!("parse: {e}")])?;
    let mut wrong = Vec::new();

    // Metadata: exact, both directions.
    for (key, value) in metadata {
        match gguf.metadata.get(key) {
            None => wrong.push(format!("metadata `{key}` is missing from the file")),
            Some(read) if read != value => wrong.push(format!(
                "metadata `{key}` is {read:?} in the file but {value:?} was resolved"
            )),
            Some(_) => {}
        }
    }
    for key in gguf.metadata.keys() {
        if !metadata.iter().any(|(k, _)| k == key) {
            wrong.push(format!(
                "metadata `{key}` appears in the file but was never resolved"
            ));
        }
    }

    // Tensors: exact, both directions — names, dims, types.
    let mut expected: std::collections::BTreeMap<&str, (Vec<u64>, u32)> =
        std::collections::BTreeMap::new();
    for plan in plans {
        let mut dims = plan.target_shape.clone();
        dims.reverse();
        expected.insert(plan.target_name.as_str(), (dims, plan.target_type));
        if let Some(scale) = &plan.scale_tensor {
            expected.insert(scale.as_str(), (vec![1], TYPE_F32));
        }
    }
    let mut seen = std::collections::BTreeSet::new();
    for info in &gguf.tensor_infos {
        seen.insert(info.name().to_string());
        match expected.get(info.name()) {
            None => wrong.push(format!(
                "tensor `{}` appears in the file but no plan produced it",
                info.name()
            )),
            Some((dims, ty)) => {
                if info.dims() != dims.as_slice() {
                    wrong.push(format!(
                        "tensor `{}` has dims {:?} in the file but the plan says {dims:?}",
                        info.name(),
                        info.dims()
                    ));
                }
                if info.tensor_type() != *ty {
                    wrong.push(format!(
                        "tensor `{}` has ggml type {} in the file but the plan says {ty}",
                        info.name(),
                        info.tensor_type()
                    ));
                }
            }
        }
    }
    for name in expected.keys() {
        if !seen.contains(*name) {
            wrong.push(format!("planned tensor `{name}` is missing from the file"));
        }
    }
    for name in required {
        if !seen.contains(*name) {
            wrong.push(format!(
                "required tensor `{name}` is missing — the target runtime would fall back \
                 or fail, and either is a different model than was exported"
            ));
        }
    }
    // NVFP4 always travels with its sibling.
    let mut nvfp4 = 0usize;
    for plan in plans {
        if plan.target_type == TYPE_NVFP4 {
            nvfp4 += 1;
            match &plan.scale_tensor {
                Some(s) if seen.contains(s.as_str()) => {}
                Some(s) => wrong.push(format!(
                    "NVFP4 `{}` lost its sibling `{s}`",
                    plan.target_name
                )),
                None => wrong.push(format!(
                    "NVFP4 `{}` was planned without a scale sibling",
                    plan.target_name
                )),
            }
        }
    }

    if wrong.is_empty() {
        Ok(VerifyReport {
            tensors: gguf.tensor_infos.len(),
            scale_siblings: plans.iter().filter(|p| p.scale_tensor.is_some()).count(),
            nvfp4_tensors: nvfp4,
            metadata_keys: gguf.metadata.len(),
        })
    } else {
        Err(wrong)
    }
}

#[cfg(test)]
mod tests;
