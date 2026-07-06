//! Q6_K — 256-element super-block, 210 bytes/block. Highest precision
//! K-quant; typical for the down projection in Ollama-shaped Q4_K_M
//! mixes. NEON row dot + scaled-add with scalar fallbacks.

use crate::ModelError;

use super::check_block_input;
use crate::quant::half::f16_to_f32;

/// Decode the 16 signed 6-bit values of scale sub-block `j` (0..16) from
/// one 210-byte Q6_K super-block, following ggml's planar layout
/// (`dequantize_row_q6_K` in llama.cpp): within each 128-element half,
/// the low nibbles of ql[0..32] / ql[32..64] hold elements 0..31 / 32..63
/// and the high nibbles hold elements 64..95 / 96..127; qh[l] carries the
/// two high bits for elements l, l+32, l+64, l+96 at shifts 0/2/4/6.
/// Scale sub-block j covers elements j*16 .. j*16+16, which always lie in
/// a single nibble plane, so the 16 source bytes are contiguous.
#[inline]
pub fn q6k_subblock_vals(block: &[u8], j: usize) -> [i8; 16] {
    let ql = &block[0..128];
    let qh = &block[128..192];
    let half = j / 8; // which 128-element half
    let g = j % 8; // scale group within the half
    let plane = g / 2; // q1/q2/q3/q4 in ggml's naming
    let lbase = (g % 2) * 16; // byte offset within the 32-byte plane row
    let ql_off = half * 64 + (plane & 1) * 32 + lbase;
    let qh_off = half * 32 + lbase;
    let shift = (plane as u32) * 2;
    let mut out = [0i8; 16];
    for (i, o) in out.iter_mut().enumerate() {
        let lo4 = if plane < 2 {
            ql[ql_off + i] & 0x0F
        } else {
            ql[ql_off + i] >> 4
        };
        let hi2 = (qh[qh_off + i] >> shift) & 0x03;
        *o = ((lo4 as i32 | ((hi2 as i32) << 4)) - 32) as i8;
    }
    out
}

pub fn q6k_row_dot(data: &[u8], x: &[f32]) -> Result<f32, ModelError> {
    const BLOCK: usize = 210;
    const SUPER: usize = 256;
    let n = x.len();
    if !n.is_multiple_of(SUPER) {
        return Err(ModelError::Parse(format!(
            "q6k_row_dot: row length {n} not a multiple of {SUPER}"
        )));
    }
    let n_blocks = n / SUPER;
    if data.len() < n_blocks * BLOCK {
        return Err(ModelError::Parse(format!(
            "q6k_row_dot: data short: {} < {}",
            data.len(),
            n_blocks * BLOCK,
        )));
    }

    #[cfg(target_arch = "aarch64")]
    unsafe {
        Ok(q6k_row_dot_neon(data, x, n_blocks))
    }
    #[cfg(not(target_arch = "aarch64"))]
    Ok(q6k_row_dot_scalar(data, x, n_blocks))
}

/// Scalar reference used on non-aarch64 and by tests.
#[inline]
#[allow(dead_code)]
pub(super) fn q6k_row_dot_scalar(data: &[u8], x: &[f32], n_blocks: usize) -> f32 {
    let mut acc = 0.0f32;
    for sb in 0..n_blocks {
        let block = &data[sb * 210..(sb + 1) * 210];
        let scales = &block[192..208];
        let d = f16_to_f32(u16::from_le_bytes([block[208], block[209]]));
        for (j, &sc_byte) in scales[..16].iter().enumerate() {
            let sc = d * (sc_byte as i8) as f32;
            let vals = q6k_subblock_vals(block, j);
            for (i, &v) in vals.iter().enumerate() {
                acc += sc * (v as f32) * x[sb * 256 + j * 16 + i];
            }
        }
    }
    acc
}

/// NEON-SIMD Q6K dequant + dot. Decodes 16 signed 6-bit values per scale
/// subblock into four f32x4 lanes, uses four parallel accumulators for ILP.
/// Cuts per-layer Q6_K down-projection from ~42ms to ~10-12ms on M-series.
#[cfg(target_arch = "aarch64")]
#[inline]
unsafe fn q6k_row_dot_neon(data: &[u8], x: &[f32], n_blocks: usize) -> f32 {
    use std::arch::aarch64::*;
    const BLOCK: usize = 210;
    let mut acc0 = vdupq_n_f32(0.0);
    let mut acc1 = vdupq_n_f32(0.0);
    let mut acc2 = vdupq_n_f32(0.0);
    let mut acc3 = vdupq_n_f32(0.0);
    let x_ptr = x.as_ptr();
    for sb in 0..n_blocks {
        let block = data.as_ptr().add(sb * BLOCK);
        let scales = block.add(192);
        let d = f16_to_f32(u16::from_le_bytes([*block.add(208), *block.add(209)]));
        let sb_base = x_ptr.add(sb * 256);
        // 16 scale subblocks × 16 elements = 256 super-block elements,
        // decoded through the shared planar-layout helper (ggml's
        // `dequantize_row_q6_K` ordering), then widened and FMA'd.
        let block_slice = std::slice::from_raw_parts(block, 210);
        for j in 0..16 {
            let sc = d * (*(scales.add(j) as *const i8)) as f32;
            let vals = q6k_subblock_vals(block_slice, j);
            // Widen i8×16 → i16×8 × 2 → i32×4 × 4 → f32×4 × 4.
            let vals_i8 = vld1q_s8(vals.as_ptr());
            let lo_i16 = vmovl_s8(vget_low_s8(vals_i8));
            let hi_i16 = vmovl_s8(vget_high_s8(vals_i8));
            let v0 = vcvtq_f32_s32(vmovl_s16(vget_low_s16(lo_i16)));
            let v1 = vcvtq_f32_s32(vmovl_s16(vget_high_s16(lo_i16)));
            let v2 = vcvtq_f32_s32(vmovl_s16(vget_low_s16(hi_i16)));
            let v3 = vcvtq_f32_s32(vmovl_s16(vget_high_s16(hi_i16)));
            let sc_v = vdupq_n_f32(sc);
            let x_j = sb_base.add(j * 16);
            let x0 = vld1q_f32(x_j);
            let x1 = vld1q_f32(x_j.add(4));
            let x2 = vld1q_f32(x_j.add(8));
            let x3 = vld1q_f32(x_j.add(12));
            // acc += (v * sc) * x — pre-scale then FMA.
            acc0 = vfmaq_f32(acc0, vmulq_f32(v0, sc_v), x0);
            acc1 = vfmaq_f32(acc1, vmulq_f32(v1, sc_v), x1);
            acc2 = vfmaq_f32(acc2, vmulq_f32(v2, sc_v), x2);
            acc3 = vfmaq_f32(acc3, vmulq_f32(v3, sc_v), x3);
        }
    }
    let acc01 = vaddq_f32(acc0, acc1);
    let acc23 = vaddq_f32(acc2, acc3);
    vaddvq_f32(vaddq_f32(acc01, acc23))
}

/// Fused Q6_K decode + scaled add.
#[inline]
pub fn q6k_row_scaled_add(data: &[u8], alpha: f32, out: &mut [f32]) -> Result<(), ModelError> {
    let block_size = 210;
    let super_block = 256;
    let n = out.len();
    if !n.is_multiple_of(super_block) {
        return Err(ModelError::Parse(format!(
            "q6k_row_scaled_add: row length {n} not a multiple of {super_block}"
        )));
    }
    let n_blocks = n / super_block;
    if data.len() < n_blocks * block_size {
        return Err(ModelError::Parse(format!(
            "q6k_row_scaled_add: data short: {} < {}",
            data.len(),
            n_blocks * block_size,
        )));
    }
    for sb in 0..n_blocks {
        let block = &data[sb * block_size..(sb + 1) * block_size];
        let scales = &block[192..208];
        let d = f16_to_f32(u16::from_le_bytes([block[208], block[209]]));
        for (j, &sc_byte) in scales[..16].iter().enumerate() {
            let sc = d * (sc_byte as i8) as f32;
            let vals = q6k_subblock_vals(block, j);
            for (i, &v) in vals.iter().enumerate() {
                out[sb * 256 + j * 16 + i] += alpha * sc * (v as f32);
            }
        }
    }
    Ok(())
}

/// Q6_K: super-block of 256 values = 210 bytes.
/// [0..127] lower 4 bits, [128..191] upper 2 bits, [192..207] 16 int8 scales, [208..209] f16 d.
pub fn dequantize_q6_k(data: &[u8], n_elements: usize) -> Result<Vec<f32>, ModelError> {
    let block_size = 210;
    let super_block = 256;
    let n_blocks = check_block_input("Q6_K", data, n_elements, super_block, block_size)?;
    let mut out = Vec::with_capacity(n_elements);

    for sb in 0..n_blocks {
        let block = &data[sb * block_size..(sb + 1) * block_size];
        let scales = &block[192..208]; // 16 int8 scales
        let d = f16_to_f32(u16::from_le_bytes([block[208], block[209]]));

        for (j, &sc_byte) in scales[..16].iter().enumerate() {
            let sc = d * (sc_byte as i8) as f32;
            let vals = q6k_subblock_vals(block, j);
            for &v in vals.iter() {
                out.push(sc * v as f32);
            }
        }
    }
    Ok(out)
}
