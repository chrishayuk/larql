//! Loaded weights and FP8 block loading.

use super::super::backend::{KQuantActivation, Nvfp4Activation, WeightFormat, WeightSlice};
use super::super::operands::RawOperand;
use super::super::quantise::{Q4_BLOCK, Q8_BLOCK};
use crate::error::VindexError;
use crate::format::vindex3::opplan::OperandRef;

#[allow(unused_imports)]
use super::*;

impl LoadedWeight {
    /// The borrowed view a call struct carries.
    /// Bytes this operand OCCUPIES — the allocation, page padding
    /// included, because that is what the process holds.
    ///
    /// Not the matrix's logical size: a census that reported geometry
    /// would agree with itself no matter how much memory was really in
    /// use, which is the one thing a residency claim must not do.
    pub fn resident_bytes(&self) -> usize {
        match self {
            LoadedWeight::F32(w) => std::mem::size_of_val(&w[..]),
            LoadedWeight::Q8 {
                codes,
                scales,
                sums,
            } => {
                codes.len() + std::mem::size_of_val(&scales[..]) + std::mem::size_of_val(&sums[..])
            }
            LoadedWeight::Q4 { packed, scales } => {
                packed.len() + std::mem::size_of_val(&scales[..])
            }
            LoadedWeight::Bf16(b) | LoadedWeight::F16(b) => b.as_slice().len(),
            // The committed half of a mapping: its pages resident now.
            // Address space is `mapped_bytes`, and the two are never
            // summed into one figure. A failed read counts 0 here, so this
            // is a FOOTPRINT figure and never a residency witness: a
            // witness reads `mapped_region()` and keeps the outcome.
            LoadedWeight::Mapped { region, .. } => region.resident_bytes().unwrap_or(0) as usize,
            LoadedWeight::Mxfp4 { packed, scales } => {
                packed.as_slice().len() + scales.as_slice().len()
            }
            LoadedWeight::Nvfp4 { packed, scales, .. } => {
                packed.as_slice().len() + scales.as_slice().len()
            }
            LoadedWeight::KQuant { blocks, .. } => blocks.len(),
            // Codes and scales both: the scales are read on every
            // projection, so a footprint that omitted them would flatter
            // the format by exactly what its metadata costs.
            LoadedWeight::Fp8Block { codes, scales, .. } => {
                codes.as_slice().len() + scales.len() * 4
            }
            LoadedWeight::CodecOwned { bytes, .. } => bytes.len(),
        }
    }

    /// Every backing allocation this operand holds: `(address, bytes)`.
    ///
    /// Plural because the compact formats are not one buffer. Q8 keeps
    /// codes and scales in separate allocations, so a model resident as
    /// Q8 holds roughly twice as many as the same model as bf16 — and
    /// where those land is invisible to a kernel benchmark that allocates
    /// one matrix and reuses it.
    pub fn allocations(&self) -> Vec<(usize, usize)> {
        let of = |p: *const u8, n: usize| (p as usize, n);
        match self {
            LoadedWeight::F32(w) => vec![of(w.as_ptr().cast(), std::mem::size_of_val(&w[..]))],
            LoadedWeight::Q8 {
                codes,
                scales,
                sums,
            } => {
                let mut v = vec![
                    of(codes.as_ptr().cast(), codes.len()),
                    of(scales.as_ptr().cast(), std::mem::size_of_val(&scales[..])),
                ];
                if !sums.is_empty() {
                    v.push(of(sums.as_ptr().cast(), std::mem::size_of_val(&sums[..])));
                }
                v
            }
            LoadedWeight::Q4 { packed, scales } => vec![
                of(packed.as_ptr().cast(), packed.len()),
                of(scales.as_ptr().cast(), std::mem::size_of_val(&scales[..])),
            ],
            // A mapping is the OS's page cache, not an allocation of this
            // process; `mapped_bytes` and `resident_bytes` account for it.
            LoadedWeight::Mapped { .. } => Vec::new(),
            LoadedWeight::Bf16(b) | LoadedWeight::F16(b) => {
                vec![of(b.as_slice().as_ptr(), b.as_slice().len())]
            }
            LoadedWeight::Mxfp4 { packed, scales } | LoadedWeight::Nvfp4 { packed, scales, .. } => {
                vec![
                    of(packed.as_slice().as_ptr(), packed.as_slice().len()),
                    of(scales.as_slice().as_ptr(), scales.as_slice().len()),
                ]
            }
            LoadedWeight::KQuant { blocks, .. } => vec![of(blocks.as_ptr(), blocks.len())],
            LoadedWeight::Fp8Block { codes, scales, .. } => vec![
                of(codes.as_slice().as_ptr(), codes.as_slice().len()),
                of(scales.as_ptr().cast::<u8>(), scales.len() * 4),
            ],
            LoadedWeight::CodecOwned { bytes, .. } => vec![of(bytes.as_ptr(), bytes.len())],
        }
    }

    /// The mapped region behind this weight, for a mapping; `None` for
    /// every owned form. A residency witness reads through this so the
    /// outcome of each read survives.
    pub fn mapped_region(&self) -> Option<&WeightRegion> {
        match self {
            LoadedWeight::Mapped { region, .. } => Some(region),
            _ => None,
        }
    }

    /// Address space this weight occupies as a mapping of the container's
    /// segment — zero for every owned form. Pages of it become resident
    /// only as touched; `resident_bytes` reports those.
    pub fn mapped_bytes(&self) -> usize {
        match self {
            LoadedWeight::Mapped { region, .. } => region.len() as usize,
            LoadedWeight::F32(_)
            | LoadedWeight::Q8 { .. }
            | LoadedWeight::Q4 { .. }
            | LoadedWeight::Bf16(_)
            | LoadedWeight::F16(_)
            | LoadedWeight::Mxfp4 { .. }
            | LoadedWeight::Nvfp4 { .. }
            | LoadedWeight::KQuant { .. }
            | LoadedWeight::Fp8Block { .. }
            | LoadedWeight::CodecOwned { .. } => 0,
        }
    }

    /// Whether these bytes are the checkpoint's own, or a widened image
    /// of them.
    ///
    /// The distinction the whole rung turns on: `F32` over a bf16
    /// checkpoint means the loader DOUBLED the model, and no total alone
    /// can say where that happened.
    pub fn is_widened_f32(&self) -> bool {
        matches!(self, LoadedWeight::F32(_))
    }

    /// How many of this operand's allocations are page-padded — the
    /// `AlignedBytes` buffers, whose length rounds up to the page. A
    /// reconciliation of declared against resident bytes tolerates one
    /// page of padding per such allocation and nothing for the rest.
    pub fn padded_allocations(&self) -> usize {
        match self {
            LoadedWeight::F32(_)
            | LoadedWeight::Q8 { .. }
            | LoadedWeight::Q4 { .. }
            | LoadedWeight::Mapped { .. }
            | LoadedWeight::CodecOwned { .. } => 0,
            LoadedWeight::Bf16(_) | LoadedWeight::F16(_) => 1,
            LoadedWeight::Mxfp4 { .. } | LoadedWeight::Nvfp4 { .. } => 2,
            LoadedWeight::KQuant { .. } => 0,
            // Only the codes are an `AlignedBytes`; the scales are a
            // plain `Vec<f32>` and are not page-padded.
            LoadedWeight::Fp8Block { .. } => 1,
        }
    }

    /// The representation these bytes are resident in — what a pinned
    /// realization is checked against.
    pub fn format(&self) -> WeightFormat {
        match self {
            LoadedWeight::F32(_) => WeightFormat::F32,
            LoadedWeight::Bf16(_) => WeightFormat::Bf16,
            LoadedWeight::Mapped { form, .. } => form.format(),
            LoadedWeight::F16(_) => WeightFormat::F16,
            LoadedWeight::Q8 { .. } => WeightFormat::Q8,
            LoadedWeight::Q4 { .. } => WeightFormat::Q4,
            LoadedWeight::Mxfp4 { .. } => WeightFormat::Mxfp4,
            LoadedWeight::Nvfp4 { activation, .. } => match activation {
                Nvfp4Activation::F32 => WeightFormat::Nvfp4,
                Nvfp4Activation::Q8 => WeightFormat::Nvfp4Q8,
            },
            LoadedWeight::KQuant { activation, .. } => match activation {
                KQuantActivation::F32 => WeightFormat::KQuant,
                KQuantActivation::Q8k => WeightFormat::KQuantQ8k,
            },
            LoadedWeight::Fp8Block { .. } => WeightFormat::Fp8Block,
            LoadedWeight::CodecOwned { .. } => WeightFormat::CodecOwned,
        }
    }

    pub fn slice(&self) -> WeightSlice<'_> {
        match self {
            LoadedWeight::F32(w) => WeightSlice::F32(w),
            LoadedWeight::Q8 {
                codes,
                scales,
                sums,
            } => WeightSlice::Q8 {
                codes,
                scales,
                sums,
                block: Q8_BLOCK,
            },
            LoadedWeight::Q4 { packed, scales } => WeightSlice::Q4 {
                packed,
                scales,
                block: Q4_BLOCK,
            },
            // SAFETY: `AlignedBytes` is page-aligned, so u16 alignment
            // holds; the length is even because the load arm rejects a
            // byte count that is not.
            LoadedWeight::Bf16(b) => {
                let bytes = b.as_slice();
                WeightSlice::Bf16(unsafe {
                    std::slice::from_raw_parts(bytes.as_ptr().cast::<u16>(), bytes.len() / 2)
                })
            }
            LoadedWeight::Fp8Block {
                codes,
                scales,
                block_rows,
                block_cols,
                scale_cols,
            } => WeightSlice::Fp8Block {
                // `logical_slice`, not `as_slice`: the page padding is
                // allocation, not matrix, and a slab whose length
                // included it would fail the geometry check below for a
                // reason that has nothing to do with the checkpoint.
                codes: &codes.as_slice()[..codes.logical_len()],
                scales,
                block_rows: *block_rows,
                block_cols: *block_cols,
                scale_cols: *scale_cols,
            },
            LoadedWeight::F16(b) => WeightSlice::F16(b.as_slice()),
            // SAFETY: a segment's payload offsets are multiples of
            // `WEIGHT_BINDING_ALIGN` (4) over a page-aligned mapping, and
            // `map_region` refused a length that is not a whole number of
            // elements, so both casts are aligned and exact.
            LoadedWeight::Mapped { region, form } => {
                let bytes = region.bytes();
                match form {
                    MappedForm::Bf16 => WeightSlice::Bf16(unsafe {
                        std::slice::from_raw_parts(bytes.as_ptr().cast::<u16>(), bytes.len() / 2)
                    }),
                    MappedForm::F32 => WeightSlice::F32(unsafe {
                        std::slice::from_raw_parts(bytes.as_ptr().cast::<f32>(), bytes.len() / 4)
                    }),
                }
            }
            LoadedWeight::Mxfp4 { packed, scales } => WeightSlice::Mxfp4 {
                packed: packed.as_slice(),
                scales: scales.as_slice(),
            },
            LoadedWeight::Nvfp4 {
                packed,
                scales,
                tensor_scale,
                activation,
            } => WeightSlice::Nvfp4 {
                packed: packed.as_slice(),
                scales: scales.as_slice(),
                tensor_scale: *tensor_scale,
                activation: *activation,
            },
            LoadedWeight::KQuant {
                blocks,
                codec,
                activation,
            } => WeightSlice::KQuant {
                blocks,
                codec: *codec,
                activation: *activation,
            },
            LoadedWeight::CodecOwned { bytes, label } => WeightSlice::CodecOwned {
                bytes,
                label: label.as_str(),
            },
        }
    }
}

/// Bind a fine-grained FP8 pair: the codes byte-for-byte, the scale grid
/// as its own codec decoded it, and the tile DERIVED from the two shapes.
///
/// `quantization_config.weight_block_size` is not consulted: it is a
/// whole-checkpoint summary, and the scheme permits per-tensor grids that
/// contradict it. Nothing here reads a dtype name either — the grid
/// arrives as values through the registry, so a grid stored under any
/// registered representation binds, and one under none was refused by
/// name before this ran.
pub(super) fn load_fp8_block(
    operand: &OperandRef,
    codes: RawOperand,
    grid_tensor: &str,
    scale_shape: &[usize],
    scales: Vec<f32>,
) -> Result<LoadedWeight, VindexError> {
    use larql_models::quant::fp8_finegrained::Fp8Grid;

    let refuse = |what: String| VindexError::Parse(what);
    let (rows, cols) = match operand.shape.as_slice() {
        [r, c] => (*r, *c),
        other => {
            return Err(refuse(format!(
                "operand `{}` has shape {other:?}; fine-grained FP8 tiles a `[out, in]` \
                 matrix and has no reading of any other rank",
                operand.tensor
            )))
        }
    };
    let (scale_rows, scale_cols) = match scale_shape {
        [r, c] => (*r, *c),
        other => {
            return Err(refuse(format!(
                "operand `{}`: scale grid `{grid_tensor}` has shape {other:?}, which is not \
                 a two-dimensional grid",
                operand.tensor
            )))
        }
    };
    let grid = Fp8Grid {
        rows,
        cols,
        scale_rows,
        scale_cols,
    };
    let (block_rows, block_cols) = grid
        .tile()
        .map_err(|e| refuse(format!("operand `{}`: {e}", operand.tensor)))?;

    if codes.bytes.len() != grid.elements() {
        return Err(refuse(format!(
            "operand `{}` declares {rows}x{cols} but its segment holds {} E4M3 bytes",
            operand.tensor,
            codes.bytes.len()
        )));
    }
    if scales.len() != grid.scales() {
        return Err(refuse(format!(
            "operand `{}`: scale grid `{grid_tensor}` declares {scale_rows}x{scale_cols} but \
             decodes to {} values",
            operand.tensor,
            scales.len()
        )));
    }
    Ok(LoadedWeight::Fp8Block {
        codes: AlignedBytes::from_bytes(&codes.bytes),
        scales,
        block_rows,
        block_cols,
        scale_cols,
    })
}
