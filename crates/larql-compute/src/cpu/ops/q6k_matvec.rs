//! CPU reference implementation for Q6_K matrix-vector multiply.
//!
//! Mirrors the Metal shader `q6k_matvec` exactly for cross-backend testing.
//! Not optimised — scalar code intended as a correctness reference.

use larql_models::quant::ggml::Q6_K_BLOCK_BYTES as Q6K_BLOCK_SIZE;

/// Decode f16 bits to f32.
fn f16_to_f32(bits: u16) -> f32 {
    let sign = ((bits >> 15) & 1) as u32;
    let exp = ((bits >> 10) & 0x1F) as i32;
    let mant = (bits & 0x3FF) as u32;
    if exp == 0 {
        if mant == 0 {
            return if sign == 1 { -0.0 } else { 0.0 };
        }
        let val = mant as f32 / 1024.0 * 2.0f32.powi(-14);
        return if sign == 1 { -val } else { val };
    }
    if exp == 31 {
        return if mant == 0 {
            if sign == 1 {
                f32::NEG_INFINITY
            } else {
                f32::INFINITY
            }
        } else {
            f32::NAN
        };
    }
    let val = (1.0 + mant as f32 / 1024.0) * 2.0f32.powi(exp - 15);
    if sign == 1 {
        -val
    } else {
        val
    }
}

/// CPU Q6_K matvec: out[N] = Q6_K[N, K] @ x[K].
///
/// Mirrors the Metal `q6k_matvec` shader: per-row dot product over super-blocks.
pub fn dispatch(q6k_data: &[u8], x: &[f32], num_rows: usize, hidden: usize) -> Vec<f32> {
    let superblocks = hidden / 256;
    let bytes_per_row = superblocks * Q6K_BLOCK_SIZE;
    let mut out = vec![0.0f32; num_rows];

    // par_chunks_mut(CHUNK_ROWS) — fewer-but-larger work units, less
    // rayon work-stealing overhead. Same rationale as
    // `q4_common::q4k_matvec_into`.
    const CHUNK_ROWS: usize = 32;
    use rayon::prelude::*;
    let q6k_ref = q6k_data;
    let x_ref = x;
    out.par_chunks_mut(CHUNK_ROWS)
        .enumerate()
        .for_each(|(chunk_idx, chunk_slots)| {
            let row_base = chunk_idx * CHUNK_ROWS;
            for (local_r, out_val) in chunk_slots.iter_mut().enumerate() {
                let row = row_base + local_r;
                if row >= num_rows {
                    break;
                }
                let row_start = row * bytes_per_row;
                let mut acc = 0.0f32;

                for sb in 0..superblocks {
                    let block = &q6k_ref[row_start + sb * Q6K_BLOCK_SIZE..];

                    let ql = &block[0..128];
                    let qh = &block[128..192];
                    let scales = &block[192..208];
                    let d_bits = u16::from_le_bytes([block[208], block[209]]);
                    let d = f16_to_f32(d_bits);

                    let x_base = sb * 256;

                    for (j, &scale) in scales.iter().enumerate() {
                        let sc = d * (scale as i8) as f32;
                        // Sub-block of 16 elements: ql[j*8 .. j*8+8] gives 16
                        // 4-bit lo values; qh[j*4 .. j*4+4] gives 16 2-bit hi
                        // values (4 packed into each byte).
                        let ql_sub = &ql[j * 8..j * 8 + 8];
                        let qh_sub = &qh[j * 4..j * 4 + 4];
                        let x_sub = &x_ref[x_base + j * 16..x_base + j * 16 + 16];

                        acc += sc * q6_subblock_dot_16(ql_sub, qh_sub, x_sub);
                    }
                }
                *out_val = acc;
            }
        });
    out
}

/// Decode one 16-element Q6_K sub-block and dot it with 16 f32 inputs.
/// Returns `sum_{i=0..16} ((lo4_i + hi2_i * 16) - 32) * x[i]`.
/// Dispatches to NEON on aarch64; scalar elsewhere.
#[inline]
fn q6_subblock_dot_16(ql_sub: &[u8], qh_sub: &[u8], x_sub: &[f32]) -> f32 {
    debug_assert_eq!(ql_sub.len(), 8);
    debug_assert_eq!(qh_sub.len(), 4);
    debug_assert_eq!(x_sub.len(), 16);
    #[cfg(target_arch = "aarch64")]
    {
        unsafe { q6_subblock_dot_16_neon(ql_sub, qh_sub, x_sub) }
    }
    #[cfg(not(target_arch = "aarch64"))]
    {
        let mut acc = 0.0f32;
        #[allow(clippy::needless_range_loop)]
        for i in 0..8usize {
            let qi = i * 2;
            let lo_byte = ql_sub[i];
            let lo4_0 = (lo_byte & 0x0F) as f32;
            let lo4_1 = ((lo_byte >> 4) & 0x0F) as f32;
            let hi_byte_idx_0 = qi / 4;
            let hi_byte_idx_1 = (qi + 1) / 4;
            let bit_off_0 = (qi % 4) * 2;
            let bit_off_1 = ((qi + 1) % 4) * 2;
            let hi2_0 = ((qh_sub[hi_byte_idx_0] >> bit_off_0) & 0x03) as f32;
            let hi2_1 = ((qh_sub[hi_byte_idx_1] >> bit_off_1) & 0x03) as f32;
            let v0 = (lo4_0 + hi2_0 * 16.0) - 32.0;
            let v1 = (lo4_1 + hi2_1 * 16.0) - 32.0;
            acc += v0 * x_sub[qi] + v1 * x_sub[qi + 1];
        }
        acc
    }
}

#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
unsafe fn q6_subblock_dot_16_neon(ql_sub: &[u8], qh_sub: &[u8], x_sub: &[f32]) -> f32 {
    use core::arch::aarch64::*;

    // ── Low 4 bits: 16 nibbles in output-order ───────────────────────
    // ql_sub is 8 bytes; byte i packs (output 2i, output 2i+1) as
    // (lo nibble, hi nibble). Result: u8x16 where lane k = lo4[k].
    let lo_bytes_u64 = u64::from_le_bytes([
        ql_sub[0], ql_sub[1], ql_sub[2], ql_sub[3], ql_sub[4], ql_sub[5], ql_sub[6], ql_sub[7],
    ]);
    let lo_bytes: uint8x8_t = vcreate_u8(lo_bytes_u64);
    let mask4 = vdup_n_u8(0x0F);
    let even_lo = vand_u8(lo_bytes, mask4); // lo nibble per byte → outputs 0,2,4,...
    let odd_lo = vshr_n_u8::<4>(lo_bytes); //  hi nibble per byte → outputs 1,3,5,...
                                           // Interleave to [even[0], odd[0], even[1], odd[1], ...]
    let lo16: uint8x16_t = vcombine_u8(vzip1_u8(even_lo, odd_lo), vzip2_u8(even_lo, odd_lo));

    // ── Hi 2 bits: 16 values, 4 per byte ─────────────────────────────
    // qh_sub is 4 bytes; byte i holds outputs (4i, 4i+1, 4i+2, 4i+3)
    // at bits (0-1, 2-3, 4-5, 6-7). Broadcast each byte 4× then per-
    // lane right-shift by [0, 2, 4, 6] then mask with 0x03.
    let qh_bytes_u32 = u32::from_le_bytes([qh_sub[0], qh_sub[1], qh_sub[2], qh_sub[3]]);
    // Replicate u32 to fill a u8x16: [b0,b1,b2,b3, b0,b1,b2,b3, ...]
    // — we want [b0,b0,b0,b0, b1,b1,b1,b1, b2,b2,b2,b2, b3,b3,b3,b3].
    // tbl with index pattern [0,0,0,0, 1,1,1,1, 2,2,2,2, 3,3,3,3].
    let qh_lane: uint8x16_t = vreinterpretq_u8_u32(vdupq_n_u32(qh_bytes_u32));
    #[rustfmt::skip]
    let tbl_idx: uint8x16_t = vld1q_u8([
        0u8, 0, 0, 0, 1, 1, 1, 1,
        2,   2, 2, 2, 3, 3, 3, 3,
    ].as_ptr());
    let qh_bcast = vqtbl1q_u8(qh_lane, tbl_idx);
    // Per-lane right-shift by [0,2,4,6, 0,2,4,6, ...] using
    // vshlq_s8 with negative shifts (treats input as signed s8 but
    // we mask immediately after so sign doesn't leak).
    #[rustfmt::skip]
    let shift_idx: int8x16_t = vld1q_s8([
        0i8, -2, -4, -6, 0, -2, -4, -6,
        0,   -2, -4, -6, 0, -2, -4, -6,
    ].as_ptr());
    let hi_shifted = vshlq_u8(qh_bcast, shift_idx);
    let mask2 = vdupq_n_u8(0x03);
    let hi16 = vandq_u8(hi_shifted, mask2);

    // ── Combine: u8 value = lo4 + hi2 * 16, then -32 in f32 ──────────
    // (hi2 << 4) | lo4 fits in u8 (max 63); we widen later.
    let combined = vorrq_u8(lo16, vshlq_n_u8::<4>(hi16));

    // Widen u8x16 → 4× u32x4 → 4× f32x4 and subtract 32.
    let lo16u = vmovl_u8(vget_low_u8(combined));
    let hi16u = vmovl_u8(vget_high_u8(combined));
    let v0u = vmovl_u16(vget_low_u16(lo16u));
    let v1u = vmovl_u16(vget_high_u16(lo16u));
    let v2u = vmovl_u16(vget_low_u16(hi16u));
    let v3u = vmovl_u16(vget_high_u16(hi16u));
    let off = vdupq_n_f32(32.0);
    let v0 = vsubq_f32(vcvtq_f32_u32(v0u), off);
    let v1 = vsubq_f32(vcvtq_f32_u32(v1u), off);
    let v2 = vsubq_f32(vcvtq_f32_u32(v2u), off);
    let v3 = vsubq_f32(vcvtq_f32_u32(v3u), off);

    // FMA against x_sub[0..16] into four independent accumulators so
    // the 4 FMAs pipeline at 1/cycle instead of serialising on a
    // single dst register (M3 FMA: 4-cycle latency, 1/cycle throughput).
    let x0 = vld1q_f32(x_sub.as_ptr());
    let x1 = vld1q_f32(x_sub.as_ptr().add(4));
    let x2 = vld1q_f32(x_sub.as_ptr().add(8));
    let x3 = vld1q_f32(x_sub.as_ptr().add(12));
    let acc0 = vmulq_f32(v0, x0);
    let acc1 = vmulq_f32(v1, x1);
    let acc2 = vmulq_f32(v2, x2);
    let acc3 = vmulq_f32(v3, x3);
    let acc = vaddq_f32(vaddq_f32(acc0, acc1), vaddq_f32(acc2, acc3));
    vaddvq_f32(acc)
}

#[cfg(test)]
mod neon_tests {
    use super::*;

    // Reference scalar oracle for the NEON sub-block dot. Indexed
    // access mirrors the Q6_K layout walk used by the production
    // kernel; switching to enumerate()/iter() obscures the
    // sub-block-offset arithmetic that's the point of the test.
    #[allow(clippy::needless_range_loop)]
    fn scalar_subblock_dot_16(ql_sub: &[u8], qh_sub: &[u8], x_sub: &[f32]) -> f32 {
        let mut acc = 0.0f32;
        for i in 0..8usize {
            let qi = i * 2;
            let lo_byte = ql_sub[i];
            let lo4_0 = (lo_byte & 0x0F) as f32;
            let lo4_1 = ((lo_byte >> 4) & 0x0F) as f32;
            let hi_byte_idx_0 = qi / 4;
            let hi_byte_idx_1 = (qi + 1) / 4;
            let bit_off_0 = (qi % 4) * 2;
            let bit_off_1 = ((qi + 1) % 4) * 2;
            let hi2_0 = ((qh_sub[hi_byte_idx_0] >> bit_off_0) & 0x03) as f32;
            let hi2_1 = ((qh_sub[hi_byte_idx_1] >> bit_off_1) & 0x03) as f32;
            let v0 = (lo4_0 + hi2_0 * 16.0) - 32.0;
            let v1 = (lo4_1 + hi2_1 * 16.0) - 32.0;
            acc += v0 * x_sub[qi] + v1 * x_sub[qi + 1];
        }
        acc
    }

    #[test]
    fn q6_subblock_matches_scalar_full_6bit_range() {
        // Pack lo + hi to cover every 6-bit value 0..63 across 16
        // positions, repeated.
        let ql_sub: Vec<u8> = (0..8u8).map(|i| (i * 2) | ((i * 2 + 1) << 4)).collect();
        let qh_sub: Vec<u8> = vec![0b11_10_01_00, 0b00_01_10_11, 0b10_10_01_01, 0b11_00_11_00];
        let x_sub: Vec<f32> = (0..16).map(|i| (i as f32 - 8.0) * 0.125).collect();

        let s = scalar_subblock_dot_16(&ql_sub, &qh_sub, &x_sub);
        let g = q6_subblock_dot_16(&ql_sub, &qh_sub, &x_sub);
        // Same arithmetic, possibly different summation order — allow
        // small relative drift.
        let rel = ((s - g).abs() / s.abs().max(1e-6)) as f64;
        assert!(rel < 1e-5, "scalar={s} neon={g}");
    }

    #[test]
    fn q6_subblock_zero_input_zero_output() {
        let ql = vec![0xFFu8; 8];
        let qh = vec![0xFFu8; 4];
        let x = vec![0.0f32; 16];
        assert_eq!(q6_subblock_dot_16(&ql, &qh, &x), 0.0);
    }

    #[test]
    fn q6_subblock_zero_weights_zero_output() {
        // 6-bit value 32 = (lo4=0, hi2=2). Pack lo=0, hi=2 everywhere
        // → all 6-bit values = 32 → after -32 offset all coefficients
        // are zero → result zero regardless of x.
        let ql = vec![0x00u8; 8];
        let qh = vec![0b10_10_10_10u8; 4]; // every hi2 = 2
        let x: Vec<f32> = (0..16).map(|i| i as f32 + 1.0).collect();
        let got = q6_subblock_dot_16(&ql, &qh, &x);
        assert!(got.abs() < 1e-5, "expected ~0, got {got}");
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cpu::ops::q4_common::quantize_q6_k;

    #[test]
    fn q6k_produces_nonzero() {
        let hidden = 256;
        let rows = 4;
        let matrix: Vec<f32> = (0..rows * hidden)
            .map(|i| (i as f32 * 0.001).cos())
            .collect();
        let q6k = quantize_q6_k(&matrix);
        let x: Vec<f32> = (0..hidden).map(|i| (i as f32 * 0.01).sin()).collect();
        let out = dispatch(&q6k, &x, rows, hidden);
        assert!(
            out.iter().any(|&v| v.abs() > 0.001),
            "Q6_K matvec should produce nonzero"
        );
    }

    // ── local f16_to_f32 edge cases ──

    #[test]
    fn f16_to_f32_neg_zero() {
        // bits=0x8000: sign=1, exp=0, mant=0 → negative zero
        let v = super::f16_to_f32(0x8000);
        assert!(v == 0.0 && v.is_sign_negative(), "0x8000 should be -0.0");
    }

    #[test]
    fn f16_to_f32_subnormal_positive() {
        // bits=0x0001: sign=0, exp=0, mant=1 → smallest positive subnormal ≈ 5.96e-8
        let v = super::f16_to_f32(0x0001);
        assert!(
            v > 0.0 && v < 1e-6,
            "0x0001 should be a tiny positive subnormal, got {v}"
        );
    }

    #[test]
    fn f16_to_f32_subnormal_negative() {
        // bits=0x8001: sign=1, exp=0, mant=1 → smallest negative subnormal
        let v = super::f16_to_f32(0x8001);
        assert!(
            v < 0.0 && v > -1e-6,
            "0x8001 should be a tiny negative subnormal, got {v}"
        );
    }

    #[test]
    fn f16_to_f32_neg_infinity() {
        // bits=0xFC00: sign=1, exp=31, mant=0 → negative infinity
        let v = super::f16_to_f32(0xFC00);
        assert!(v == f32::NEG_INFINITY, "0xFC00 should be -inf, got {v}");
    }

    #[test]
    fn f16_to_f32_nan() {
        // bits=0x7C01: sign=0, exp=31, mant=1 → NaN
        let v = super::f16_to_f32(0x7C01);
        assert!(v.is_nan(), "0x7C01 should be NaN, got {v}");
    }
}
