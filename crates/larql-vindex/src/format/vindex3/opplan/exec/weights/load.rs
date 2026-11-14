//! Loading a stored weight into its resident form.

use super::super::backend::{KQuantActivation, Nvfp4Activation, WeightFormat};
use super::super::narrow::{bf16_bytes_to_f16, f32_bytes_to_f16};
use super::super::operands::{OperandSource, OperandStore};
use super::super::quantise::{quantise_q4, quantise_q8};
use crate::error::VindexError;
use crate::format::vindex3::opplan::OperandRef;
use crate::format::vindex3::represent::codec::codecs::fp8_block::SCALES as FP8_SCALES;
use crate::format::vindex3::represent::kquant::{self};
use crate::format::vindex3::represent::nvfp4_pack::DTYPE_NVFP4;
use larql_models::quant::mxfp4::{e8m0_to_f32, MXFP4_GROUP_BYTES, MXFP4_GROUP_ELEMS};
use staged::StagedF32;

#[allow(unused_imports)]
use super::*;

/// Load one matrix operand in `format`, through the closure-verified
/// path only.
pub fn load_weight(
    store: OperandSource<'_>,
    operand: &OperandRef,
    format: WeightFormat,
) -> Result<LoadedWeight, VindexError> {
    match format {
        WeightFormat::F32 => Ok(LoadedWeight::F32(StagedF32::stage(store.load(operand)?)?)),
        // Fine-grained FP8 binds TWO container objects: the E4M3 codes,
        // and the scale grid the codec DECLARES as its dependency and the
        // container's reference table ADDRESSES — never a name this
        // loader spells. The codes are not widened — that is the point of
        // the format on this programme, since a widened GLM-5.3-Flash
        // would be 612 GB of a 306 GB checkpoint. The grid is decoded
        // through its own codec: it is retained beside the codes, and
        // its bytes are its own representation's business.
        WeightFormat::Fp8Block => {
            let raw = store.load_raw(operand)?;
            let owner = crate::format::vindex3::auxiliary_references::OperandAddress::new(
                &operand.object,
                &operand.tensor,
            );
            let grid = store
                .store()
                .references()
                .target(&owner, FP8_SCALES)
                .cloned()
                .ok_or_else(|| {
                    VindexError::Parse(format!(
                        "operand `{}`: the container declares no `{FP8_SCALES}` dependency for \
                         it, and fine-grained FP8 codes mean nothing without their grid",
                        operand.tensor
                    ))
                })?;
            let scale_shape = store.store().stored_shape(&grid).ok_or_else(|| {
                VindexError::Parse(format!(
                    "operand `{}`: its `{FP8_SCALES}` dependency {} is referenced and the \
                     container holds no such tensor",
                    operand.tensor,
                    grid.describe()
                ))
            })?;
            let scales = store.load(&OperandRef {
                object: grid.object.clone(),
                tensor: grid.tensor.clone(),
                dtype: String::new(),
                shape: scale_shape.clone(),
            })?;
            load_fp8_block(operand, raw, &grid.tensor, &scale_shape, scales)
        }
        WeightFormat::Q8 => {
            let in_dim = operand.shape.get(1).copied().ok_or_else(|| {
                VindexError::Parse(format!(
                    "tensor `{}` has shape {:?}; q8 residency blocks along the INPUT axis and \
                     needs a `[out, in]` matrix to know where the blocks are",
                    operand.tensor, operand.shape
                ))
            })?;
            // **CPU5-K1 was FALSIFIED and is opt-in.** Precomputing the
            // weight-code sums removes a second `SDOT` from the
            // asymmetric row, but measured 868 ms against the 757 ms it
            // was meant to beat: the extra ~12% of compact traffic, and a
            // third memory stream the prefetcher has to track, cost more
            // than a `vdotq` over codes already in registers. Kept
            // reachable so the negative result stays reproducible.
            let indexed = super::super::cpu::integer::weight_index_enabled()
                && matches!(
                    super::super::cpu::integer::activation_code(),
                    super::super::cpu::integer::ActivationCode::Asymmetric
                );
            store.store().note_runtime_quantisation(&operand.tensor)?;
            Ok(quantise_q8(&store.load(operand)?, in_dim, indexed))
        }
        WeightFormat::Q4 => {
            let in_dim = operand.shape.get(1).copied().ok_or_else(|| {
                VindexError::Parse(format!(
                    "tensor `{}` has shape {:?}; q4 residency blocks along the INPUT axis and \
                     needs a `[out, in]` matrix to know where the blocks are",
                    operand.tensor, operand.shape
                ))
            })?;
            // Two codes to the byte, so an odd input axis has no packing.
            // Refusing is the only honest answer: the alternative is a
            // silently dropped final weight in every row.
            if in_dim % 2 != 0 {
                return Err(VindexError::Parse(format!(
                    "tensor `{}` has an odd input axis {in_dim}; q4 packs two codes per byte \
                     and cannot represent a ragged half-byte",
                    operand.tensor
                )));
            }
            store.store().note_runtime_quantisation(&operand.tensor)?;
            Ok(quantise_q4(&store.load(operand)?, in_dim))
        }
        WeightFormat::Bf16 => {
            let raw = store.load_raw(operand)?;
            if raw.dtype.as_str() != DTYPE_BF16 {
                // No widening or narrowing here on purpose. This format
                // means "the stored bytes ARE the resident bytes"; a
                // checkpoint holding something else needs a judged
                // conversion, and inventing one silently would make the
                // format a lie.
                return Err(VindexError::Parse(format!(
                    "tensor `{}` is `{}`, not bf16 — the bf16 residency format copies stored \
                     bytes and performs no conversion",
                    operand.tensor, raw.dtype
                )));
            }
            Ok(LoadedWeight::Bf16(AlignedBytes::from_bytes(&raw.bytes)))
        }
        WeightFormat::Mxfp4 => {
            let rows = operand.shape.first().copied().unwrap_or(0);
            let k = operand.shape.get(1).copied().unwrap_or(0);
            store.store().note_runtime_quantisation(&operand.tensor)?;
            let values = store.load(operand)?;
            quantize_mxfp4(&values, rows, k, &operand.tensor)
        }
        WeightFormat::Nvfp4 => {
            let rows = operand.shape.first().copied().unwrap_or(0);
            let k = operand.shape.get(1).copied().unwrap_or(0);
            // A compiled pack is already in the grid the kernel wants, so
            // the whole load is a read: no widening to f32, no requantise,
            // no arithmetic at all. That is the point of persisting it.
            let raw = store.load_raw(operand)?;
            check_pack_conforms(store, operand, &raw.dtype)?;
            if raw.dtype == DTYPE_NVFP4 {
                return nvfp4_from_stored(&raw.bytes, rows, k, &operand.tensor);
            }
            // Whether this request binds at source precision is ONE fact,
            // derived once on the store and read identically by selection
            // — see `OperandStore::nvfp4_request_binds_at_source`. A second
            // derivation here is how the device selector came to pin NVFP4
            // on a head this loader then bound at f16.
            if store
                .store()
                .nvfp4_request_binds_at_source(operand, &raw.dtype)
            {
                store.store().note_stored_precision();
                return narrow_to_f16(&raw, &operand.tensor);
            }
            store.store().note_runtime_quantisation(&operand.tensor)?;
            let values = widen_raw(&raw, &operand.tensor)?;
            quantize_nvfp4(&values, rows, k, &operand.tensor)
        }
        WeightFormat::Nvfp4Q8 => {
            // The Q8-activation arm runs a PERSISTED pack under integer
            // arithmetic, and changes nothing else: a tensor the pack holds
            // at source precision binds there, exactly as under
            // `WeightFormat::Nvfp4`. What it never does is quantise at
            // load — that would measure a different weight under this
            // arm's name.
            let rows = operand.shape.first().copied().unwrap_or(0);
            let k = operand.shape.get(1).copied().unwrap_or(0);
            let raw = store.load_raw(operand)?;
            check_pack_conforms(store, operand, &raw.dtype)?;
            if raw.dtype == DTYPE_NVFP4 {
                return match nvfp4_from_stored(&raw.bytes, rows, k, &operand.tensor)? {
                    LoadedWeight::Nvfp4 {
                        packed,
                        scales,
                        tensor_scale,
                        ..
                    } => Ok(LoadedWeight::Nvfp4 {
                        packed,
                        scales,
                        tensor_scale,
                        activation: Nvfp4Activation::Q8,
                    }),
                    other => Ok(other),
                };
            }
            if store
                .store()
                .nvfp4_request_binds_at_source(operand, &raw.dtype)
            {
                store.store().note_stored_precision();
                return narrow_to_f16(&raw, &operand.tensor);
            }
            Err(VindexError::Parse(format!(
                "{}: NVFP4 x Q8 executes a stored NVFP4 pack; the operand is stored as {} \
                 and would have to be quantised at load",
                operand.tensor, raw.dtype
            )))
        }
        WeightFormat::KQuant => kquant_from_stored(store, operand, KQuantActivation::F32),
        WeightFormat::KQuantQ8k => kquant_from_stored(store, operand, KQuantActivation::Q8k),
        // Generic pass-through: whatever the container recorded as this
        // operand's stored representation, read and kept as bytes plus
        // its own name. No dtype is judged, no conversion attempted —
        // that is the point of the format, and why it is the only arm
        // here with no per-dtype match on `raw.dtype`.
        WeightFormat::CodecOwned => {
            let raw = store.load_raw(operand)?;
            Ok(LoadedWeight::CodecOwned {
                bytes: raw.bytes,
                label: raw.dtype,
            })
        }
        WeightFormat::F16 => {
            let raw = store.load_raw(operand)?;
            // A compiled NVFP4 pack has no source bytes to narrow; it binds
            // as stored — the same fact selection pins by, see
            // `OperandStore::f16_request_binds_compiled_nvfp4`.
            if OperandStore::f16_request_binds_compiled_nvfp4(&raw.dtype) {
                check_pack_conforms(store, operand, &raw.dtype)?;
                let rows = operand.shape.first().copied().unwrap_or(0);
                let k = operand.shape.get(1).copied().unwrap_or(0);
                return nvfp4_from_stored(&raw.bytes, rows, k, &operand.tensor);
            }
            match raw.dtype.as_str() {
                DTYPE_BF16 => Ok(LoadedWeight::F16(bf16_bytes_to_f16(
                    &raw.bytes,
                    &operand.tensor,
                )?)),
                DTYPE_F32 => Ok(LoadedWeight::F16(f32_bytes_to_f16(
                    &raw.bytes,
                    &operand.tensor,
                )?)),
                other => Err(VindexError::Parse(format!(
                    "tensor `{}`: no judged f16 narrowing for dtype `{other}`",
                    operand.tensor
                ))),
            }
        }
    }
}

/// Refuse a stored tensor whose dtype the container's precision map does
/// not permit.
///
/// The map is the authority a pack is supposed to satisfy, so `stored`
/// checks conformance rather than taking the bytes' word for what program
/// they implement. A pack that compiled a tensor the map protects is not
/// a pack for this program, and silently executing it would mean running
/// something other than what the container declares. Shared by every arm
/// that binds stored compiled bytes, so the NVFP4 and K-quant paths
/// cannot drift in what they check.
pub(super) fn check_pack_conforms(
    store: OperandSource<'_>,
    operand: &OperandRef,
    dtype: &str,
) -> Result<(), VindexError> {
    if let (Some(program), true) = (
        store.store().program(),
        store.store().is_stored(&operand.object),
    ) {
        // Resolved through the store, which reads the plan first and the
        // name heuristics second — the same order the compiler used.
        // Calling `classify` directly here was the miss that made a
        // correctly compiled pack fail its own conformance check.
        let role = store
            .store()
            .role_of(&operand.object, &operand.tensor, &operand.shape);
        if !program.conforms(role, &operand.tensor, dtype) {
            return Err(VindexError::Parse(format!(
                "tensor `{}` is stored as `{dtype}`, which the container's precision map \
                 `{}` does not permit — the pack does not implement the program the \
                 container declares",
                operand.tensor, program.name
            )));
        }
    }
    Ok(())
}

/// Bind a compiled K-quant pack: read the stored blocks and keep them.
///
/// No decode happens here and none may: if this path ever widened a
/// block or re-encoded one, the executed bytes would be a derivative of
/// the artifact rather than the artifact, and the v3 gate's claim — the
/// SAME stored bytes, two arithmetics — would be void. The geometry is
/// checked against the codec's own plan of the operand's shape, so a
/// stream of the wrong length for `[out, in]` is refused here by name
/// rather than read at the wrong stride by a kernel.
pub(super) fn kquant_from_stored(
    store: OperandSource<'_>,
    operand: &OperandRef,
    activation: KQuantActivation,
) -> Result<LoadedWeight, VindexError> {
    let raw = store.load_raw(operand)?;
    let Some(codec) = kquant::lookup(&raw.dtype) else {
        return Err(VindexError::Parse(format!(
            "tensor `{}` is `{}`, not a K-quant this executor runs in place ({}) — direct \
             K-quant residency binds stored blocks and performs no conversion",
            operand.tensor,
            raw.dtype,
            kquant::compilable_names()
        )));
    };
    check_pack_conforms(store, operand, &raw.dtype)?;
    let want = codec.plan(&operand.shape, &operand.tensor)?;
    if raw.bytes.len() != want {
        return Err(VindexError::Parse(format!(
            "tensor `{}`: {} bytes of {} do not describe shape {:?}, which needs {want} — \
             refusing rather than reading rows at the wrong stride",
            operand.tensor,
            raw.bytes.len(),
            codec.name,
            operand.shape
        )));
    }
    // A Q8_K binding for a member with no Q8_K kernel would pin a
    // realization that fails at the first token; refuse it at load, by
    // name, as the codec mismatch above is.
    if activation == KQuantActivation::Q8k && !codec.has_q8k_gemv() {
        return Err(VindexError::Parse(format!(
            "tensor `{}` is {}, which has no Q8_K-activation kernel — only Q4_K and Q6_K bind \
             for a Q8_K activation",
            operand.tensor, codec.name
        )));
    }
    super::super::cpu::gemv_witness::note_operand(
        &operand.object,
        &operand.tensor,
        codec.name,
        &raw.bytes,
    );
    Ok(LoadedWeight::KQuant {
        blocks: raw.bytes,
        codec,
        activation,
    })
}

/// Bind a compiled NVFP4 pack: copy each region into a page-aligned
/// buffer the device can take, and read the tensor scale.
///
/// No quantisation happens here and none may: if this path ever needed to
/// compute a scale or round an element, the pack would not have been a
/// compiled representation.
pub(super) fn nvfp4_from_stored(
    bytes: &[u8],
    rows: usize,
    k: usize,
    name: &str,
) -> Result<LoadedWeight, VindexError> {
    use crate::format::vindex3::represent::nvfp4_pack::{split, PackLayout};

    let layout = PackLayout::derive(&[rows, k], name)?;
    let (packed_src, scales_src, tensor_scale) = split(bytes, &layout, name)?;

    let mut packed = AlignedBytes::zeroed(packed_src.len());
    packed.as_mut_slice()[..packed_src.len()].copy_from_slice(packed_src);
    let mut scales = AlignedBytes::zeroed(scales_src.len());
    scales.as_mut_slice()[..scales_src.len()].copy_from_slice(scales_src);

    Ok(LoadedWeight::Nvfp4 {
        packed,
        scales,
        tensor_scale,
        activation: Nvfp4Activation::F32,
    })
}

/// Bind an already-read float operand as f16, the narrowing the F16 arm
/// performs. Shared so the stored-precision path cannot drift from it.
pub(super) fn narrow_to_f16(
    raw: &super::super::operands::RawOperand,
    name: &str,
) -> Result<LoadedWeight, VindexError> {
    match raw.dtype.as_str() {
        DTYPE_BF16 => Ok(LoadedWeight::F16(bf16_bytes_to_f16(&raw.bytes, name)?)),
        DTYPE_F32 => Ok(LoadedWeight::F16(f32_bytes_to_f16(&raw.bytes, name)?)),
        other => Err(VindexError::Parse(format!(
            "tensor `{name}`: no judged f16 narrowing for stored dtype `{other}`"
        ))),
    }
}

/// Widen an already-read raw operand, so the NVFP4 path can inspect the
/// stored dtype without paying for a second read of the same bytes.
pub(super) fn widen_raw(
    raw: &super::super::operands::RawOperand,
    name: &str,
) -> Result<Vec<f32>, VindexError> {
    super::super::operands::widen(&raw.dtype, &raw.bytes, name)
}

/// e2m1's largest magnitude; the shared exponent is chosen so the
/// group's max maps at or below it, saturating the rare overshoot.
pub(super) const MXFP4_MAX_MAG: f32 = 6.0;
/// Exponent of [`MXFP4_MAX_MAG`]'s leading bit: `floor(log2(6)) = 2`.
pub(super) const MXFP4_EMAX: i32 = 2;

/// Quantise one `[rows, k]` f32 matrix to MXFP4 — the OCP microscaling
/// rule: per 32-element group, shared scale `2^(floor(log2(max|x|)) -
/// 2)` as e8m0, elements rounded to the nearest e2m1 grid value
/// (ties to the even code index), saturating at ±6.
///
/// A lossy realisation by construction; the parity gates against the
/// f16/f32 anchors and the upstream trace are its judge. Layout is the
/// kernel's, and the nibble-order control in the tests pins it against
/// the independent `larql-models` decoder.
pub fn quantize_mxfp4(
    values: &[f32],
    rows: usize,
    k: usize,
    name: &str,
) -> Result<LoadedWeight, VindexError> {
    if !k.is_multiple_of(MXFP4_GROUP_ELEMS) {
        return Err(VindexError::Parse(format!(
            "tensor `{name}`: k={k} is not a multiple of the MXFP4 32-element group"
        )));
    }
    if values.len() != rows * k {
        return Err(VindexError::Parse(format!(
            "tensor `{name}`: {} values do not fill [{rows}, {k}]",
            values.len()
        )));
    }
    let groups = k / MXFP4_GROUP_ELEMS;
    let mut packed = AlignedBytes::zeroed(rows * groups * MXFP4_GROUP_BYTES);
    let mut scales = AlignedBytes::zeroed(rows * groups);
    {
        use rayon::prelude::*;
        let packed_dst = packed.as_mut_slice();
        let scales_dst = scales.as_mut_slice();
        packed_dst[..rows * groups * MXFP4_GROUP_BYTES]
            .par_chunks_mut(groups * MXFP4_GROUP_BYTES)
            .zip(scales_dst[..rows * groups].par_chunks_mut(groups))
            .zip(values.par_chunks(k))
            .for_each(|((row_packed, row_scales), row_values)| {
                for g in 0..groups {
                    let group = &row_values[g * MXFP4_GROUP_ELEMS..(g + 1) * MXFP4_GROUP_ELEMS];
                    let max_abs = group.iter().fold(0.0f32, |m, v| m.max(v.abs()));
                    let scale_byte = if max_abs == 0.0 {
                        0u8 // decodes to 0.0; all codes zero
                    } else {
                        let exponent = max_abs.log2().floor() as i32 - MXFP4_EMAX;
                        (exponent + 127).clamp(1, 254) as u8
                    };
                    row_scales[g] = scale_byte;
                    let scale = e8m0_to_f32(scale_byte);
                    let inv = if scale == 0.0 { 0.0 } else { scale.recip() };
                    let bytes = &mut row_packed[g * MXFP4_GROUP_BYTES..(g + 1) * MXFP4_GROUP_BYTES];
                    for (b, pair) in group.chunks_exact(2).enumerate() {
                        let lo = nearest_mxfp4_code(pair[0] * inv);
                        let hi = nearest_mxfp4_code(pair[1] * inv);
                        bytes[b] = lo | (hi << 4);
                    }
                }
            });
    }
    Ok(LoadedWeight::Mxfp4 { packed, scales })
}
