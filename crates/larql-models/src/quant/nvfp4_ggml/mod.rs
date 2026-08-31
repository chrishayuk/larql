//! NVFP4 on the GGML wire: lowering a VINDEX3 pack to `GGML_TYPE_NVFP4`.
//!
//! **This is ABI lowering, not quantization.** No floating-point weight
//! passes through this module. The E2M1 codes that leave are the codes
//! that arrived, the E4M3 scale magnitudes are unchanged, and the f32
//! tensor scale is carried out intact as a sibling tensor rather than
//! folded into anything. The representation that measured KL 0.02411
//! across 1,740 positions is the representation in the GGUF, because
//! nothing here is free to choose a different one.
//!
//! That matters more than it sounds. Both formats are called NVFP4 and
//! they do not agree on layout:
//!
//! ```text
//! VINDEX3   three contiguous regions over the whole matrix
//!           [ E2M1 codes ][ E4M3 group scales ][ f32 tensor scale ]
//!           one group scale per 16 elements, adjacent-pair nibbles
//!
//! GGML      interleaved 64-element blocks, 36 bytes each
//!           [ 4 × UE4M3 ][ 32 × E2M1 ]        no per-tensor level
//! ```
//!
//! GGML has no per-tensor scale, which is why the f32 leaves as a
//! separate `.scale` tensor — the arrangement llama.cpp's own converter
//! uses. Folding it into the UE4M3 bytes would have been a
//! re-quantization: UE4M3 is four exponent bits and three mantissa bits,
//! and the product is not generally representable in it. That the two
//! levels cannot be collapsed is the reason the two-level recipe exists.
//!
//! Three transformations, none of them arithmetic on values:
//!
//! 1. **Nibble order.** VINDEX3 packs elements `2j` and `2j+1` in byte
//!    `j`. GGML packs elements `j` and `j+8` of each 16-group in byte
//!    `j` — planar halves rather than adjacent pairs.
//! 2. **Sign bit.** E4M3 group scales are positive by construction, so
//!    UE4M3 is the same byte with the sign stripped.
//! 3. **Grouping.** Four 16-element groups become one 64-element block,
//!    scales first.

use crate::quant::nvfp4::{NVFP4_GROUP_BYTES, NVFP4_GROUP_ELEMS};
use crate::ModelError;

/// `GGML_TYPE_NVFP4`.
pub const TYPE_NVFP4: u32 = 40;

/// `QK_NVFP4` — elements per GGML block.
pub const NVFP4_BLOCK_ELEMS: usize = 64;

/// Sub-groups per block; each carries one UE4M3 scale.
pub const NVFP4_BLOCK_SUBGROUPS: usize = NVFP4_BLOCK_ELEMS / NVFP4_GROUP_ELEMS;

/// `sizeof(block_nvfp4)` — four scale bytes plus thirty-two code bytes.
pub const NVFP4_BLOCK_BYTES: usize = NVFP4_BLOCK_SUBGROUPS + NVFP4_BLOCK_ELEMS / 2;

/// A lowered tensor: the block stream, and the tensor scale that must
/// travel beside it as its own `.scale` tensor.
#[derive(Debug, Clone, PartialEq)]
pub struct GgmlNvfp4 {
    pub blocks: Vec<u8>,
    pub tensor_scale: f32,
}

/// Lower a VINDEX3 NVFP4 pack to the GGML block layout.
///
/// `packed` and `scales` are the pack's first two regions, exactly as
/// `nvfp4_pack::split` borrows them. Refuses rather than pads: GGML's
/// block is 64 elements and a matrix whose row is not a whole number of
/// blocks cannot be expressed without inventing weights.
pub fn repack_nvfp4(
    packed: &[u8],
    scales: &[u8],
    tensor_scale: f32,
    rows: usize,
    cols: usize,
) -> Result<GgmlNvfp4, ModelError> {
    if !cols.is_multiple_of(NVFP4_BLOCK_ELEMS) {
        return Err(ModelError::Parse(format!(
            "NVFP4 GGML lowering needs K a multiple of {NVFP4_BLOCK_ELEMS}, got {cols} — \
             padding would invent weights the source does not have"
        )));
    }
    let groups = cols / NVFP4_GROUP_ELEMS;
    let want_codes = rows * groups * NVFP4_GROUP_BYTES;
    let want_scales = rows * groups;
    if packed.len() != want_codes {
        return Err(ModelError::Parse(format!(
            "NVFP4 code region is {} bytes, layout wants {want_codes}",
            packed.len()
        )));
    }
    if scales.len() != want_scales {
        return Err(ModelError::Parse(format!(
            "NVFP4 scale region is {} bytes, layout wants {want_scales}",
            scales.len()
        )));
    }

    let blocks_per_row = cols / NVFP4_BLOCK_ELEMS;
    let mut out = Vec::with_capacity(rows * blocks_per_row * NVFP4_BLOCK_BYTES);

    for row in 0..rows {
        for block in 0..blocks_per_row {
            let g0 = row * groups + block * NVFP4_BLOCK_SUBGROUPS;

            // Scales first: E4M3 is positive here, so UE4M3 is the same
            // magnitude with the sign bit cleared.
            for g in 0..NVFP4_BLOCK_SUBGROUPS {
                out.push(scales[g0 + g] & 0x7f);
            }

            // Then the codes, re-nibbled from adjacent pairs to planar
            // halves. Element order is preserved; only which byte holds
            // which element changes.
            for g in 0..NVFP4_BLOCK_SUBGROUPS {
                let base = (g0 + g) * NVFP4_GROUP_BYTES;
                let src = &packed[base..base + NVFP4_GROUP_BYTES];
                let mut elems = [0u8; NVFP4_GROUP_ELEMS];
                for (j, byte) in src.iter().enumerate() {
                    elems[2 * j] = byte & 0x0f;
                    elems[2 * j + 1] = byte >> 4;
                }
                let half = NVFP4_GROUP_ELEMS / 2;
                for j in 0..half {
                    out.push(elems[j] | (elems[j + half] << 4));
                }
            }
        }
    }

    Ok(GgmlNvfp4 {
        blocks: out,
        tensor_scale,
    })
}

/// Stored bytes for `n_elements` under `GGML_TYPE_NVFP4`.
pub fn ggml_nvfp4_bytes(n_elements: usize) -> Result<usize, ModelError> {
    if !n_elements.is_multiple_of(NVFP4_BLOCK_ELEMS) {
        return Err(ModelError::Parse(format!(
            "NVFP4: n_elements {n_elements} is not a multiple of {NVFP4_BLOCK_ELEMS}"
        )));
    }
    Ok(n_elements / NVFP4_BLOCK_ELEMS * NVFP4_BLOCK_BYTES)
}

#[cfg(test)]
mod tests;
