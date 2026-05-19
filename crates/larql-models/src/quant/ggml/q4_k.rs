//! Q4_K — 256-element super-block, 144 bytes/block. Most common
//! Ollama-compatible FFN format. NEON-accelerated row dot and
//! scaled-add, with scalar fallbacks.

use crate::ModelError;

use super::check_block_input;
use crate::quant::half::f16_to_f32;

/// Q4_K block layout (144 bytes per super-block of 256 elements), as
/// written by llama.cpp / GGUF files:
///   bytes 0-1:   d    (f16 global scale)
///   bytes 2-3:   dmin (f16 global min)
///   bytes 4-15:  12 bytes of packed 6-bit scales + 6-bit mins (8 each)
///   bytes 16-143: 128 bytes of 4-bit quants (2 nibbles per byte = 256 values)
///
/// The 6-bit scale/min unpacking follows llama.cpp's `get_scale_min_k4`:
///   For j < 4: scales[j] = bytes[j] & 0x3F;       mins[j] = bytes[j+4] & 0x3F
///   For j ≥ 4: scales[j] = (bytes[j+4] & 0x0F) | ((bytes[j-4] >> 6) << 4)
///              mins[j]   = (bytes[j+4] >> 4)    | ((bytes[j]   >> 6) << 4)
///
/// Each (scale, min) pair governs 32 elements within the 256-element super-block.
/// Fused Q4_K decode + dot product — `dot(dequant(data), x)` without
/// materialising the decoded row. Same math as
/// `dequantize_q4_k(data, x.len())` followed by `a.dot(x)`, but skips the
/// Vec<f32> allocation, the intermediate write, and the separate BLAS sdot
/// call. Hot path on very large models where we'd otherwise pay 2 decodes
/// + 2 buffer copies + 2 BLAS dispatches per feature.
#[inline(always)]
pub fn q4k_row_dot(data: &[u8], x: &[f32]) -> Result<f32, ModelError> {
    // Already inline(always) — kept explicit for clarity.
    const BLOCK: usize = 144;
    const SUPER: usize = 256;
    let n = x.len();
    if !n.is_multiple_of(SUPER) {
        return Err(ModelError::Parse(format!(
            "q4k_row_dot: row length {n} not a multiple of {SUPER}"
        )));
    }
    let n_blocks = n / SUPER;
    if data.len() < n_blocks * BLOCK {
        return Err(ModelError::Parse(format!(
            "q4k_row_dot: data short: {} < {}",
            data.len(),
            n_blocks * BLOCK,
        )));
    }

    #[cfg(target_arch = "aarch64")]
    unsafe {
        Ok(q4k_row_dot_neon(data, x, n_blocks))
    }
    #[cfg(not(target_arch = "aarch64"))]
    Ok(q4k_row_dot_scalar(data, x, n_blocks))
}

/// Scalar reference used on non-aarch64 and by tests.
#[inline]
#[allow(dead_code)]
pub(super) fn q4k_row_dot_scalar(data: &[u8], x: &[f32], n_blocks: usize) -> f32 {
    let mut acc = 0.0f32;
    for sb in 0..n_blocks {
        let block = &data[sb * 144..(sb + 1) * 144];
        let d = f16_to_f32(u16::from_le_bytes([block[0], block[1]]));
        let dmin = f16_to_f32(u16::from_le_bytes([block[2], block[3]]));
        let (scales, mins) = unpack_q4k_scales(&block[4..16]);
        let quants = &block[16..144];
        let sb_base = sb * 256;
        for g in 0..4 {
            let sb_lo = 2 * g;
            let sb_hi = 2 * g + 1;
            let sc_lo = d * scales[sb_lo] as f32;
            let sc_hi = d * scales[sb_hi] as f32;
            let mn_lo = dmin * mins[sb_lo] as f32;
            let mn_hi = dmin * mins[sb_hi] as f32;
            let chunk = &quants[g * 32..(g + 1) * 32];
            let base_lo = sb_base + sb_lo * 32;
            let base_hi = sb_base + sb_hi * 32;
            for l in 0..32 {
                let byte = chunk[l];
                let v_lo = sc_lo * (byte & 0x0F) as f32 - mn_lo;
                let v_hi = sc_hi * ((byte >> 4) & 0x0F) as f32 - mn_hi;
                acc += v_lo * x[base_lo + l];
                acc += v_hi * x[base_hi + l];
            }
        }
    }
    acc
}

/// 12 packed bytes → 8 six-bit scales + 8 six-bit mins.
#[inline]
fn unpack_q4k_scales(scales_bytes: &[u8]) -> ([u8; 8], [u8; 8]) {
    let mut scales = [0u8; 8];
    let mut mins = [0u8; 8];
    for j in 0..4 {
        scales[j] = scales_bytes[j] & 0x3F;
        mins[j] = scales_bytes[j + 4] & 0x3F;
    }
    for j in 4..8 {
        scales[j] = (scales_bytes[j + 4] & 0x0F) | ((scales_bytes[j - 4] >> 6) << 4);
        mins[j] = (scales_bytes[j + 4] >> 4) | ((scales_bytes[j] >> 6) << 4);
    }
    (scales, mins)
}

/// NEON-SIMD Q4K dequant + dot. Processes 4 nibbles per iteration into
/// f32x4 lanes, uses two parallel accumulators for ILP, reduces to scalar
/// at the end. Cuts ~50μs Q4K decode to ~12-15μs on M-series silicon.
#[cfg(target_arch = "aarch64")]
#[inline]
unsafe fn q4k_row_dot_neon(data: &[u8], x: &[f32], n_blocks: usize) -> f32 {
    use std::arch::aarch64::*;
    let mut acc0 = vdupq_n_f32(0.0);
    let mut acc1 = vdupq_n_f32(0.0);
    let x_ptr = x.as_ptr();
    for sb in 0..n_blocks {
        let block = data.as_ptr().add(sb * 144);
        let d = f16_to_f32(u16::from_le_bytes([*block, *block.add(1)]));
        let dmin = f16_to_f32(u16::from_le_bytes([*block.add(2), *block.add(3)]));
        let scales_slice = std::slice::from_raw_parts(block.add(4), 12);
        let (scales, mins) = unpack_q4k_scales(scales_slice);
        let quants = block.add(16);
        let sb_base = sb * 256;
        for g in 0..4 {
            let sb_lo = 2 * g;
            let sb_hi = 2 * g + 1;
            let sc_lo = vdupq_n_f32(d * scales[sb_lo] as f32);
            let sc_hi = vdupq_n_f32(d * scales[sb_hi] as f32);
            let mn_lo = vdupq_n_f32(dmin * mins[sb_lo] as f32);
            let mn_hi = vdupq_n_f32(dmin * mins[sb_hi] as f32);
            let chunk = quants.add(g * 32);
            let base_lo = x_ptr.add(sb_base + sb_lo * 32);
            let base_hi = x_ptr.add(sb_base + sb_hi * 32);
            // 32 bytes → 32 low + 32 high = 64 elements. Process 4 bytes at
            // a time (8 elements per inner iter), unrolled ×8.
            for l4 in 0..8 {
                let b0 = *chunk.add(l4 * 4);
                let b1 = *chunk.add(l4 * 4 + 1);
                let b2 = *chunk.add(l4 * 4 + 2);
                let b3 = *chunk.add(l4 * 4 + 3);
                let lo_arr = [
                    (b0 & 0x0F) as f32,
                    (b1 & 0x0F) as f32,
                    (b2 & 0x0F) as f32,
                    (b3 & 0x0F) as f32,
                ];
                let hi_arr = [
                    (b0 >> 4) as f32,
                    (b1 >> 4) as f32,
                    (b2 >> 4) as f32,
                    (b3 >> 4) as f32,
                ];
                let lo = vld1q_f32(lo_arr.as_ptr());
                let hi = vld1q_f32(hi_arr.as_ptr());
                let v_lo = vsubq_f32(vmulq_f32(sc_lo, lo), mn_lo);
                let v_hi = vsubq_f32(vmulq_f32(sc_hi, hi), mn_hi);
                let x_lo = vld1q_f32(base_lo.add(l4 * 4));
                let x_hi = vld1q_f32(base_hi.add(l4 * 4));
                acc0 = vfmaq_f32(acc0, v_lo, x_lo);
                acc1 = vfmaq_f32(acc1, v_hi, x_hi);
            }
        }
    }
    let acc = vaddq_f32(acc0, acc1);
    vaddvq_f32(acc)
}

/// Fused Q4_K decode + scaled add — `out += alpha * dequant(data)` without
/// materialising the decoded row. Counterpart to `q4k_row_dot` for the
/// down-projection leg of the walk.
#[inline]
pub fn q4k_row_scaled_add(data: &[u8], alpha: f32, out: &mut [f32]) -> Result<(), ModelError> {
    const BLOCK: usize = 144;
    const SUPER: usize = 256;
    let n = out.len();
    if !n.is_multiple_of(SUPER) {
        return Err(ModelError::Parse(format!(
            "q4k_row_scaled_add: row length {n} not a multiple of {SUPER}"
        )));
    }
    let n_blocks = n / SUPER;
    if data.len() < n_blocks * BLOCK {
        return Err(ModelError::Parse(format!(
            "q4k_row_scaled_add: data short: {} < {}",
            data.len(),
            n_blocks * BLOCK,
        )));
    }

    #[cfg(target_arch = "aarch64")]
    unsafe {
        q4k_row_scaled_add_neon(data, alpha, out, n_blocks);
    }
    #[cfg(not(target_arch = "aarch64"))]
    q4k_row_scaled_add_scalar(data, alpha, out, n_blocks);
    Ok(())
}

#[inline]
#[allow(dead_code)]
pub(super) fn q4k_row_scaled_add_scalar(data: &[u8], alpha: f32, out: &mut [f32], n_blocks: usize) {
    for sb in 0..n_blocks {
        let block = &data[sb * 144..(sb + 1) * 144];
        let d = f16_to_f32(u16::from_le_bytes([block[0], block[1]]));
        let dmin = f16_to_f32(u16::from_le_bytes([block[2], block[3]]));
        let (scales, mins) = unpack_q4k_scales(&block[4..16]);
        let quants = &block[16..144];
        let sb_base = sb * 256;
        for g in 0..4 {
            let sb_lo = 2 * g;
            let sb_hi = 2 * g + 1;
            let sc_lo = alpha * d * scales[sb_lo] as f32;
            let sc_hi = alpha * d * scales[sb_hi] as f32;
            let mn_lo = alpha * dmin * mins[sb_lo] as f32;
            let mn_hi = alpha * dmin * mins[sb_hi] as f32;
            let chunk = &quants[g * 32..(g + 1) * 32];
            let base_lo = sb_base + sb_lo * 32;
            let base_hi = sb_base + sb_hi * 32;
            for l in 0..32 {
                let byte = chunk[l];
                out[base_lo + l] += sc_lo * (byte & 0x0F) as f32 - mn_lo;
                out[base_hi + l] += sc_hi * ((byte >> 4) & 0x0F) as f32 - mn_hi;
            }
        }
    }
}

/// NEON-SIMD fused Q4K dequant + scaled-add. Folds `alpha` into the scale
/// factors so the inner loop is a single FMA per lane.
#[cfg(target_arch = "aarch64")]
#[inline]
unsafe fn q4k_row_scaled_add_neon(data: &[u8], alpha: f32, out: &mut [f32], n_blocks: usize) {
    use std::arch::aarch64::*;
    let out_ptr = out.as_mut_ptr();
    for sb in 0..n_blocks {
        let block = data.as_ptr().add(sb * 144);
        let d = f16_to_f32(u16::from_le_bytes([*block, *block.add(1)]));
        let dmin = f16_to_f32(u16::from_le_bytes([*block.add(2), *block.add(3)]));
        let scales_slice = std::slice::from_raw_parts(block.add(4), 12);
        let (scales, mins) = unpack_q4k_scales(scales_slice);
        let quants = block.add(16);
        let sb_base = sb * 256;
        for g in 0..4 {
            let sb_lo = 2 * g;
            let sb_hi = 2 * g + 1;
            // Fold alpha into the per-group scales — one FMA per lane.
            let sc_lo = vdupq_n_f32(alpha * d * scales[sb_lo] as f32);
            let sc_hi = vdupq_n_f32(alpha * d * scales[sb_hi] as f32);
            let mn_lo = vdupq_n_f32(alpha * dmin * mins[sb_lo] as f32);
            let mn_hi = vdupq_n_f32(alpha * dmin * mins[sb_hi] as f32);
            let chunk = quants.add(g * 32);
            let base_lo = out_ptr.add(sb_base + sb_lo * 32);
            let base_hi = out_ptr.add(sb_base + sb_hi * 32);
            for l4 in 0..8 {
                let b0 = *chunk.add(l4 * 4);
                let b1 = *chunk.add(l4 * 4 + 1);
                let b2 = *chunk.add(l4 * 4 + 2);
                let b3 = *chunk.add(l4 * 4 + 3);
                let lo_arr = [
                    (b0 & 0x0F) as f32,
                    (b1 & 0x0F) as f32,
                    (b2 & 0x0F) as f32,
                    (b3 & 0x0F) as f32,
                ];
                let hi_arr = [
                    (b0 >> 4) as f32,
                    (b1 >> 4) as f32,
                    (b2 >> 4) as f32,
                    (b3 >> 4) as f32,
                ];
                let lo = vld1q_f32(lo_arr.as_ptr());
                let hi = vld1q_f32(hi_arr.as_ptr());
                // v = sc * nibble - mn, then out += v
                let v_lo = vsubq_f32(vmulq_f32(sc_lo, lo), mn_lo);
                let v_hi = vsubq_f32(vmulq_f32(sc_hi, hi), mn_hi);
                let old_lo = vld1q_f32(base_lo.add(l4 * 4));
                let old_hi = vld1q_f32(base_hi.add(l4 * 4));
                vst1q_f32(base_lo.add(l4 * 4), vaddq_f32(old_lo, v_lo));
                vst1q_f32(base_hi.add(l4 * 4), vaddq_f32(old_hi, v_hi));
            }
        }
    }
}

pub fn dequantize_q4_k(data: &[u8], n_elements: usize) -> Result<Vec<f32>, ModelError> {
    let block_size = 144; // 2 + 2 + 12 + 128, llama.cpp GGUF layout.
    let super_block = 256;
    let n_blocks = check_block_input("Q4_K", data, n_elements, super_block, block_size)?;
    let mut out = vec![0.0f32; n_elements];

    for sb in 0..n_blocks {
        let block = &data[sb * block_size..(sb + 1) * block_size];
        let d = f16_to_f32(u16::from_le_bytes([block[0], block[1]]));
        let dmin = f16_to_f32(u16::from_le_bytes([block[2], block[3]]));

        // 12 bytes of packed scales + mins at bytes 4..16, per
        // llama.cpp's `get_scale_min_k4`.
        let scales_bytes = &block[4..16];
        let mut scales = [0u8; 8];
        let mut mins = [0u8; 8];
        for j in 0..8 {
            if j < 4 {
                scales[j] = scales_bytes[j] & 0x3F;
                mins[j] = scales_bytes[j + 4] & 0x3F;
            } else {
                scales[j] = (scales_bytes[j + 4] & 0x0F) | ((scales_bytes[j - 4] >> 6) << 4);
                mins[j] = (scales_bytes[j + 4] >> 4) | ((scales_bytes[j] >> 6) << 4);
            }
        }

        // Nibble layout (matches llama.cpp `dequantize_row_q4_K`): four
        // groups of 32 bytes, each group spans two adjacent sub-blocks.
        //   byte[g*32 + l].low_nibble  → y[sb*256 + 2g*32     + l]  (sub-block 2g)
        //   byte[g*32 + l].high_nibble → y[sb*256 + (2g+1)*32 + l]  (sub-block 2g+1)
        //   scales[2g]   / mins[2g]   scale the low nibbles
        //   scales[2g+1] / mins[2g+1] scale the high nibbles
        let quants = &block[16..144];
        let sb_base = sb * super_block;
        for g in 0..4 {
            let sb_lo = 2 * g;
            let sb_hi = 2 * g + 1;
            let sc_lo = d * scales[sb_lo] as f32;
            let sc_hi = d * scales[sb_hi] as f32;
            let mn_lo = dmin * mins[sb_lo] as f32;
            let mn_hi = dmin * mins[sb_hi] as f32;
            let chunk = &quants[g * 32..(g + 1) * 32];
            let base_lo = sb_base + sb_lo * 32;
            let base_hi = sb_base + sb_hi * 32;
            for l in 0..32 {
                let byte = chunk[l];
                out[base_lo + l] = sc_lo * (byte & 0x0F) as f32 - mn_lo;
                out[base_hi + l] = sc_hi * ((byte >> 4) & 0x0F) as f32 - mn_hi;
            }
        }
    }
    Ok(out)
}
