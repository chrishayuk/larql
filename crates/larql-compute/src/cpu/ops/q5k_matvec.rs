//! CPU Q5_K matrix-vector multiply.
//!
//! Unlike `q6k_matvec`, this is not a hand-rolled reference: the row math
//! already exists as `larql_models::quant::ggml::q5k_row_dot` (NEON with a
//! scalar fallback), so this module is only the row-parallel wrapper. There
//! is no Metal `q5k` shader for it to mirror — Q5_K is CPU-only, and
//! `QuantFormat::has_metal_kernel` reports that.

use larql_models::quant::ggml::{q5k_row_dot, K_QUANT_BLOCK_ELEMS, Q5_K_BLOCK_BYTES};

/// Rows per rayon work unit. Matches `q6k_matvec` / `q4k_matvec_into` —
/// fewer, larger units cost less in work-stealing overhead than one task
/// per row.
const CHUNK_ROWS: usize = 32;

/// CPU Q5_K matvec: `out[N] = Q5_K[N, K] @ x[K]`.
///
/// Returns `None` when the geometry doesn't describe a whole number of
/// super-blocks per row, or when `q5k_data` is too short for
/// `num_rows × hidden`. Callers get a loud missing-capability signal
/// instead of a truncated read — the same contract `quant_matvec` uses for
/// formats it can't serve.
pub fn dispatch(q5k_data: &[u8], x: &[f32], num_rows: usize, hidden: usize) -> Option<Vec<f32>> {
    if hidden == 0 || !hidden.is_multiple_of(K_QUANT_BLOCK_ELEMS) || x.len() < hidden {
        return None;
    }
    let bytes_per_row = (hidden / K_QUANT_BLOCK_ELEMS) * Q5_K_BLOCK_BYTES;
    if q5k_data.len() < num_rows.checked_mul(bytes_per_row)? {
        return None;
    }

    let mut out = vec![0.0f32; num_rows];
    let x_row = &x[..hidden];

    use rayon::prelude::*;
    out.par_chunks_mut(CHUNK_ROWS)
        .enumerate()
        .for_each(|(chunk_idx, chunk_slots)| {
            let row_base = chunk_idx * CHUNK_ROWS;
            for (local_r, out_val) in chunk_slots.iter_mut().enumerate() {
                let row = row_base + local_r;
                if row >= num_rows {
                    break;
                }
                let start = row * bytes_per_row;
                let row_bytes = &q5k_data[start..start + bytes_per_row];
                // Geometry was validated above, so the row slice is exactly
                // one whole number of super-blocks and matches `x_row` — the
                // only way this errors is a logic bug in the bounds above,
                // which should surface rather than decode as zero.
                *out_val = q5k_row_dot(row_bytes, x_row)
                    .expect("q5k_row_dot on pre-validated row geometry");
            }
        });
    Some(out)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cpu::ops::q4_common::quantize_q5_k;

    fn seeded(n: usize, mut seed: u64) -> Vec<f32> {
        (0..n)
            .map(|_| {
                seed = seed
                    .wrapping_mul(6364136223846793005)
                    .wrapping_add(1442695040888963407);
                ((seed >> 33) as f32 / (1u64 << 31) as f32) - 0.5
            })
            .collect()
    }

    #[test]
    fn matches_dequantize_then_dot_per_row() {
        let (rows, cols) = (5usize, K_QUANT_BLOCK_ELEMS * 3);
        let weights = seeded(rows * cols, 0xC0FFEE);
        let x = seeded(cols, 0xBEEF);
        let bytes = quantize_q5_k(&weights);

        let decoded = larql_models::quant::ggml::dequantize_q5_k(&bytes, rows * cols).unwrap();
        let expected: Vec<f32> = (0..rows)
            .map(|r| {
                decoded[r * cols..(r + 1) * cols]
                    .iter()
                    .zip(&x)
                    .map(|(w, v)| w * v)
                    .sum()
            })
            .collect();

        let got = dispatch(&bytes, &x, rows, cols).expect("valid geometry must dispatch");
        let denom = expected.iter().map(|v| v.abs()).fold(1e-6, f32::max);
        for (r, (g, e)) in got.iter().zip(&expected).enumerate() {
            assert!((g - e).abs() / denom < 1e-5, "row {r}: {g:e} vs {e:e}");
        }
    }

    /// More rows than one rayon chunk, so the chunk-base arithmetic is
    /// exercised rather than assumed.
    #[test]
    fn spans_multiple_rayon_chunks() {
        let (rows, cols) = (CHUNK_ROWS * 2 + 7, K_QUANT_BLOCK_ELEMS);
        let weights = seeded(rows * cols, 0xD00D);
        let x = vec![1.0f32; cols];
        let bytes = quantize_q5_k(&weights);

        let got = dispatch(&bytes, &x, rows, cols).unwrap();
        assert_eq!(got.len(), rows);

        let decoded = larql_models::quant::ggml::dequantize_q5_k(&bytes, rows * cols).unwrap();
        for r in 0..rows {
            let want: f32 = decoded[r * cols..(r + 1) * cols].iter().sum();
            assert!(
                (got[r] - want).abs() <= 1e-4 * want.abs().max(1.0),
                "row {r}: {} vs {want}",
                got[r]
            );
        }
    }

    #[test]
    fn rejects_bad_geometry() {
        let cols = K_QUANT_BLOCK_ELEMS;
        let bytes = quantize_q5_k(&seeded(cols, 1));
        let x = vec![0.0f32; cols];

        // hidden not a whole number of super-blocks.
        assert!(dispatch(&bytes, &x, 1, 100).is_none());
        // Buffer holds one row; asking for two must not read past it.
        assert!(dispatch(&bytes, &x, 2, cols).is_none());
        // Input vector shorter than `hidden`.
        assert!(dispatch(&bytes, &x[..cols / 2], 1, cols).is_none());
    }
}
