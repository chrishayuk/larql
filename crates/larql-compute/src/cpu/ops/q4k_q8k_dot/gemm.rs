//! Blocked Q4_K × Q8_K GEMM — kept as **recorded negative evidence**,
//! not a production path (the `q4k_matmul` shader precedent).
//!
//! Hypothesis (2026-08-10): the MOSS 343-row prefill's ~67k spin-pool
//! fork/joins (one matvec dispatch per activation per matrix) were a
//! meaningful part of its 1.4 s floor, so restoring the matrix shape —
//! one dispatch per matrix, activations quantised once — should
//! recover real time.
//!
//! **Falsified at the real shapes** (`moss_prefill_bench`, clean-run
//! interleaved): three loop structures all lose to or tie the
//! row-at-a-time path — per-(weight-row × activation) single-row dots
//! (2.1M kernel calls per gate matrix; 2202 ms), weight-chunk-parallel
//! with a `[rows, seq]` scratch (pays a 2.1M-element scattered
//! transpose per matrix, the same defect `q4_common::matmul` carries;
//! 2095 ms), and the activation-parallel form below (1499 ms vs the
//! row path's 1390 ms). The Q4K·Q8K kernel is issue-bound at ~650-700
//! GFLOPS whatever the granularity, and the spin pool's dispatch cost
//! is negligible on a quiet machine. What actually wins at prefill
//! shape is dequant-whole + fp32 AMX BLAS (863 ms) — see
//! [`crate::ffn::q4k_weight::Q4kFfn`]'s multi-row path. A future
//! integer GEMM would need per-super-block scale/min unpack amortised
//! across activations *inside* the kernel to change this verdict.

use super::dispatch::q4k_q8k_matvec_into;
use super::q8k_activation::Q8KActivation;

/// Q4_K super-block geometry (one 144-byte super-block per 256 columns).
const Q4K_BLOCK_BYTES: usize = 144;
const Q4K_BLOCK_ELEMS: usize = 256;
/// Activation rows per parallel chunk. Small enough to load-balance
/// 343-row prefills across the pool, large enough that a chunk's
/// output slab amortises the dispatch.
const CHUNK_SEQ: usize = 8;

/// `out[s * rows + r] = W[r, :] · X[s, :]` — `out` is `[seq, rows]`
/// row-major, `w` is `[rows, cols]` Q4_K row-major, `q8k_rows` holds
/// each activation row quantised to Q8_K.
///
/// Panics on shape mismatches (caller owns validity — the FFN backend
/// checked dimensions at quantisation time).
pub fn q4k_q8k_gemm_into(
    out: &mut [f32],
    q8k_rows: &[Q8KActivation],
    w: &[u8],
    rows: usize,
    cols: usize,
) {
    let seq = q8k_rows.len();
    assert_eq!(out.len(), seq * rows, "output must be [seq, rows]");
    assert_eq!(
        cols % Q4K_BLOCK_ELEMS,
        0,
        "cols must divide into super-blocks"
    );
    let row_bytes = cols / Q4K_BLOCK_ELEMS * Q4K_BLOCK_BYTES;
    assert!(w.len() >= rows * row_bytes, "packed weight too short");
    if seq == 0 || rows == 0 {
        return;
    }

    // Parallelism over ACTIVATION rows, not weight rows: each worker
    // owns a contiguous [CHUNK_SEQ, rows] slab of the final output and
    // runs the whole-matrix single-threaded matvec once per owned
    // activation. No scratch, no transpose, one pool dispatch per
    // matrix. Two earlier cuts of this kernel lost to the row path and
    // are recorded here so they stay dead: per-(weight-row ×
    // activation) single-row dots drowned in call dispatch (2.1M
    // calls/matrix), and weight-chunk-parallel with a [rows, seq]
    // scratch paid a 2.1M-element scattered transpose per matrix —
    // the same defect `q4_common::matmul` carries.
    crate::cpu::spin_pool::par_chunks_mut(out, CHUNK_SEQ * rows, |chunk_index, slots| {
        let seq_base = chunk_index * CHUNK_SEQ;
        for (local, q8k) in q8k_rows[seq_base..seq.min(seq_base + CHUNK_SEQ)]
            .iter()
            .enumerate()
        {
            q4k_q8k_matvec_into(
                &mut slots[local * rows..(local + 1) * rows],
                q8k,
                w,
                rows,
                cols,
            )
            .unwrap_or_else(|e| {
                panic!(
                    "q4k_q8k_gemm_into: row {} refused after the asserts above: {e}",
                    seq_base + local
                )
            });
        }
    });
}

#[cfg(test)]
mod tests {
    use super::super::dispatch::q4k_q8k_matvec_parallel;
    use super::super::q8k_activation::quantize_x_to_q8k;
    use super::*;
    use crate::cpu::ops::q4_common::quantize_q4_k;

    const ROWS: usize = 96;
    const COLS: usize = 512;
    const SEQ: usize = 7;

    fn packed_weights() -> Vec<u8> {
        let data: Vec<f32> = (0..ROWS * COLS)
            .map(|i| ((i * 31 + 7) % 23) as f32 * 0.01 - 0.11)
            .collect();
        quantize_q4_k(&data)
    }

    fn activation_rows() -> Vec<Vec<f32>> {
        (0..SEQ)
            .map(|s| {
                (0..COLS)
                    .map(|c| ((s * 13 + c * 3) % 19) as f32 * 0.01 - 0.09)
                    .collect()
            })
            .collect()
    }

    #[test]
    fn gemm_is_bit_identical_to_row_at_a_time_matvec() {
        let w = packed_weights();
        let xs = activation_rows();
        let q8k: Vec<_> = xs.iter().map(|x| quantize_x_to_q8k(x)).collect();

        let mut gemm_out = vec![0.0f32; SEQ * ROWS];
        q4k_q8k_gemm_into(&mut gemm_out, &q8k, &w, ROWS, COLS);

        for (s, x) in xs.iter().enumerate() {
            let q8 = quantize_x_to_q8k(x);
            let mut row_out = vec![0.0f32; ROWS];
            q4k_q8k_matvec_parallel(&mut row_out, &q8, &w, ROWS, COLS, "Q4_K")
                .expect("valid shape");
            assert_eq!(
                &gemm_out[s * ROWS..(s + 1) * ROWS],
                row_out.as_slice(),
                "activation row {s} must match the decode path bit-for-bit"
            );
        }
    }

    #[test]
    fn gemm_handles_empty_seq() {
        let w = packed_weights();
        let mut out = vec![0.0f32; 0];
        q4k_q8k_gemm_into(&mut out, &[], &w, ROWS, COLS);
    }

    #[test]
    #[should_panic(expected = "output must be")]
    fn gemm_rejects_wrong_output_shape() {
        let w = packed_weights();
        let xs = activation_rows();
        let q8k: Vec<_> = xs.iter().map(|x| quantize_x_to_q8k(x)).collect();
        let mut out = vec![0.0f32; 3];
        q4k_q8k_gemm_into(&mut out, &q8k, &w, ROWS, COLS);
    }
}
