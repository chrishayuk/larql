//! CPU reference implementation for Q6_K matrix-vector multiply.
//!
//! Decodes ggml's planar Q6_K super-block layout through the shared
//! `larql_models::quant::ggml::q6_k::q6k_subblock_vals` helper — the
//! single source of truth for Q6_K bit placement. Not optimised —
//! scalar code intended as a correctness reference.

use larql_models::quant::ggml::q6_k::q6k_subblock_vals;
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
/// Per-row dot product over super-blocks, decoded in ggml's planar layout.
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
                    let block =
                        &q6k_ref[row_start + sb * Q6K_BLOCK_SIZE..][..Q6K_BLOCK_SIZE];
                    let scales = &block[192..208];
                    let d_bits = u16::from_le_bytes([block[208], block[209]]);
                    let d = f16_to_f32(d_bits);
                    let x_base = sb * 256;

                    for (j, &scale) in scales.iter().enumerate() {
                        let sc = d * (scale as i8) as f32;
                        let vals = q6k_subblock_vals(block, j);
                        let x_sub = &x_ref[x_base + j * 16..x_base + j * 16 + 16];
                        let mut sub = 0.0f32;
                        for (v, xi) in vals.iter().zip(x_sub) {
                            sub += *v as f32 * xi;
                        }
                        acc += sc * sub;
                    }
                }
                *out_val = acc;
            }
        });
    out
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

    #[test]
    fn q6k_round_trip_approximates_f32_matvec() {
        // quantize_q6_k (writer) and dispatch (reader) must agree on the
        // block layout AND approximate the original f32 matvec — this is
        // the pair that broke when the writer packed a private interleaved
        // layout while GGUF data arrived planar.
        let hidden = 512;
        let rows = 8;
        let matrix: Vec<f32> = (0..rows * hidden)
            .map(|i| ((i as f32 * 0.037).sin()) * 0.05)
            .collect();
        let x: Vec<f32> = (0..hidden).map(|i| ((i as f32 * 0.013).cos()) * 0.5).collect();

        let q6k = quantize_q6_k(&matrix);
        let got = dispatch(&q6k, &x, rows, hidden);

        for r in 0..rows {
            let want: f32 = matrix[r * hidden..(r + 1) * hidden]
                .iter()
                .zip(&x)
                .map(|(w, xi)| w * xi)
                .sum();
            let tol = 0.02 * want.abs().max(0.5);
            assert!(
                (got[r] - want).abs() < tol,
                "row {r}: got {}, want {want} (tol {tol})",
                got[r]
            );
        }
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
