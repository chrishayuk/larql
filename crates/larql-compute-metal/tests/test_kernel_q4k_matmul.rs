//! Parity tests for the Q4_K matmul (gemm) Metal kernel.
//!
//! `q4k_matmul` is a batched companion to `q4k_matvec`: amortises the
//! Q4_K dequant cost across `seq_len` positions in one dispatch. The
//! per-element math MUST match calling `q4k_matvec` once per position
//! and stacking the results — the matmul kernel only saves dequant
//! passes, never changes the answer.
//!
//! Tests run only when the `metal` feature is enabled and a Metal
//! backend is available (no-op skip otherwise so CI on non-macOS
//! workflows doesn't false-fail).

#![cfg(target_os = "macos")]

extern crate blas_src;

use larql_compute::cpu::ops::q4_common::quantize_q4_k;
use larql_compute::prelude::*;
use larql_compute_metal::MetalBackend;

fn synth(len: usize, seed: u64) -> Vec<f32> {
    let mut s = seed;
    (0..len)
        .map(|_| {
            s = s.wrapping_mul(6364136223846793005).wrapping_add(1);
            ((s >> 33) as f32) / (u32::MAX as f32) * 2.0 - 1.0
        })
        .collect()
}

fn metal_or_skip() -> Option<MetalBackend> {
    MetalBackend::new()
}

/// Stack `seq_len` independent matvec calls into a `[seq_len, num_rows]`
/// output. This is the reference behavior that the matmul must match
/// element-by-element (within a tiny f32 reordering tolerance —
/// dequant + accumulation order can differ across kernels).
fn matvec_reference(
    metal: &MetalBackend,
    q4k_data: &[u8],
    x_matrix: &[f32],
    num_rows: usize,
    hidden: usize,
    seq_len: usize,
) -> Vec<f32> {
    let mut out = Vec::with_capacity(seq_len * num_rows);
    for m in 0..seq_len {
        let row = &x_matrix[m * hidden..(m + 1) * hidden];
        let scores = metal
            .q4k_matvec(q4k_data, row, num_rows, hidden)
            .expect("matvec");
        out.extend(scores);
    }
    out
}

#[test]
fn q4k_matmul_matches_stacked_matvec_basic() {
    let metal = match metal_or_skip() {
        Some(m) => m,
        None => return,
    };

    // Smallest viable shape: 1 super-block per row.
    let num_rows = 4usize;
    let hidden = 256usize;
    let seq_len = 4usize;

    let weights = synth(num_rows * hidden, 41);
    let x = synth(seq_len * hidden, 42);
    let q4k = quantize_q4_k(&weights);

    let matmul = metal
        .q4k_matmul(&q4k, &x, num_rows, hidden, seq_len)
        .expect("matmul should be implemented");
    let reference = matvec_reference(&metal, &q4k, &x, num_rows, hidden, seq_len);

    assert_eq!(matmul.len(), reference.len(), "output length mismatch");
    for (i, (a, b)) in matmul.iter().zip(&reference).enumerate() {
        let diff = (a - b).abs();
        assert!(
            diff < 1e-4,
            "matmul vs stacked-matvec drift at idx {i}: matmul={a} reference={b} diff={diff}"
        );
    }
}

#[test]
fn q4k_matmul_matches_stacked_matvec_seq_len_1_decode_shape() {
    // seq_len=1 must still produce identical output to a single matvec —
    // this is the safety net for any future code path that always
    // routes through matmul (e.g. unifying decode + prefill).
    let metal = match metal_or_skip() {
        Some(m) => m,
        None => return,
    };

    let num_rows = 8usize;
    let hidden = 256usize;
    let seq_len = 1usize;

    let weights = synth(num_rows * hidden, 51);
    let x = synth(hidden, 52);
    let q4k = quantize_q4_k(&weights);

    let matmul = metal
        .q4k_matmul(&q4k, &x, num_rows, hidden, seq_len)
        .expect("matmul");
    let matvec = metal
        .q4k_matvec(&q4k, &x, num_rows, hidden)
        .expect("matvec");

    assert_eq!(matmul.len(), num_rows);
    for (i, (a, b)) in matmul.iter().zip(&matvec).enumerate() {
        let diff = (a - b).abs();
        assert!(
            diff < 1e-4,
            "seq_len=1 matmul must equal matvec; idx {i}: matmul={a} matvec={b} diff={diff}"
        );
    }
}

#[test]
fn q4k_matmul_handles_seq_len_not_multiple_of_cols_per_tg() {
    // COLS_PER_TG = 4. Test seq_len = 7 → first TG covers 4 positions,
    // tail TG covers 3. The shader's `cols_in_tg` guard must avoid
    // OOB writes for the unused 4th slot in the tail TG.
    let metal = match metal_or_skip() {
        Some(m) => m,
        None => return,
    };

    let num_rows = 8usize;
    let hidden = 512usize; // 2 super-blocks per row → exercises ix=0/ix=1 interleave
    let seq_len = 7usize;

    let weights = synth(num_rows * hidden, 61);
    let x = synth(seq_len * hidden, 62);
    let q4k = quantize_q4_k(&weights);

    let matmul = metal
        .q4k_matmul(&q4k, &x, num_rows, hidden, seq_len)
        .expect("matmul");
    let reference = matvec_reference(&metal, &q4k, &x, num_rows, hidden, seq_len);

    assert_eq!(matmul.len(), seq_len * num_rows);
    for (i, (a, b)) in matmul.iter().zip(&reference).enumerate() {
        let diff = (a - b).abs();
        assert!(
            diff < 1e-4,
            "tail-TG drift at idx {i} (pos={}, row={}): matmul={a} reference={b} diff={diff}",
            i / num_rows,
            i % num_rows
        );
    }
}

#[test]
fn q4k_matmul_handles_num_rows_not_multiple_of_rows_per_tg() {
    // ROWS_PER_TG = 4 simdgroups. num_rows=5 means the second row TG
    // has sg_id=0..3 but only sg_id=0 produces a valid row; the
    // `if row_idx >= N return` guard at the top of the shader must
    // skip the rest cleanly.
    let metal = match metal_or_skip() {
        Some(m) => m,
        None => return,
    };

    let num_rows = 5usize;
    let hidden = 256usize;
    let seq_len = 4usize;

    let weights = synth(num_rows * hidden, 71);
    let x = synth(seq_len * hidden, 72);
    let q4k = quantize_q4_k(&weights);

    let matmul = metal
        .q4k_matmul(&q4k, &x, num_rows, hidden, seq_len)
        .expect("matmul");
    let reference = matvec_reference(&metal, &q4k, &x, num_rows, hidden, seq_len);

    assert_eq!(matmul.len(), seq_len * num_rows);
    for (i, (a, b)) in matmul.iter().zip(&reference).enumerate() {
        let diff = (a - b).abs();
        assert!(
            diff < 1e-4,
            "ragged-row drift at idx {i}: matmul={a} reference={b} diff={diff}"
        );
    }
}

#[test]
fn q4k_matmul_production_shape_4b_o_proj() {
    // Production shape: Gemma 3 4B O projection. N = hidden = 2560,
    // K = q_dim = 8192 (32 superblocks per row), M = a typical
    // prefill seq_len. Smaller than full 18-token prompt to keep CI
    // cycles tight, but exercises the multi-superblock path.
    let metal = match metal_or_skip() {
        Some(m) => m,
        None => return,
    };

    let num_rows = 64usize; // 2560 is overkill for a unit test
    let hidden = 2560usize; // 10 super-blocks per row — production-ish
    let seq_len = 8usize;

    let weights = synth(num_rows * hidden, 81);
    let x = synth(seq_len * hidden, 82);
    let q4k = quantize_q4_k(&weights);

    let matmul = metal
        .q4k_matmul(&q4k, &x, num_rows, hidden, seq_len)
        .expect("matmul");
    let reference = matvec_reference(&metal, &q4k, &x, num_rows, hidden, seq_len);

    assert_eq!(matmul.len(), seq_len * num_rows);
    let mut max_diff = 0.0f32;
    for (a, b) in matmul.iter().zip(&reference) {
        let diff = (a - b).abs();
        if diff > max_diff {
            max_diff = diff;
        }
    }
    // Looser tolerance for 10-superblock accumulation noise (10×
    // more f32 adds than the 1-superblock test). Still well below
    // the 0.13 nibble-step that would indicate semantic drift.
    assert!(
        max_diff < 1e-3,
        "production-shape max diff {max_diff} exceeds 1e-3 — kernel drift not noise"
    );
}
