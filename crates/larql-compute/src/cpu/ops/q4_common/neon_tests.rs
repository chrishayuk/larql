use super::*;

use super::dual::q4_dual_dot_32;

/// Scalar reference for the dual-nibble dot-product the NEON kernel
/// replaces. Used as the correctness oracle for the NEON path.
fn scalar_dual_dot_32(chunk: &[u8], x_lo: &[f32], x_hi: &[f32]) -> (f32, f32) {
    let mut dot_lo = 0.0f32;
    let mut dot_hi = 0.0f32;
    for l in 0..32 {
        let byte = chunk[l];
        let q_lo = (byte & 0x0F) as f32;
        let q_hi = ((byte >> 4) & 0x0F) as f32;
        dot_lo += q_lo * x_lo[l];
        dot_hi += q_hi * x_hi[l];
    }
    (dot_lo, dot_hi)
}

#[test]
fn q4_dual_dot_32_matches_scalar_on_deterministic_input() {
    // 32 nibble pairs spanning all 16 nibble values both lo and hi.
    let chunk: Vec<u8> = (0..32u8).map(|i| (i & 0x0F) | ((i & 0x0F) << 4)).collect();
    let x_lo: Vec<f32> = (0..32).map(|i| (i as f32) * 0.013).collect();
    let x_hi: Vec<f32> = (0..32).map(|i| (i as f32) * -0.021 + 0.5).collect();

    let (scalar_lo, scalar_hi) = scalar_dual_dot_32(&chunk, &x_lo, &x_hi);
    let (got_lo, got_hi) = q4_dual_dot_32(&chunk, &x_lo, &x_hi);

    // Allow a small relative tolerance — NEON's grouped FMA orders
    // the 32-element sum differently than the scalar sequential
    // sum (4-lane reductions vs left-to-right), so bit-identity
    // isn't guaranteed.
    let rel = |s: f32, g: f32| ((s - g).abs() / (s.abs().max(1e-6))) as f64;
    assert!(
        rel(scalar_lo, got_lo) < 1e-5,
        "lo dot diverges: scalar={scalar_lo} neon={got_lo}"
    );
    assert!(
        rel(scalar_hi, got_hi) < 1e-5,
        "hi dot diverges: scalar={scalar_hi} neon={got_hi}"
    );
}

#[test]
fn q4_dual_dot_32_zero_x_returns_zero() {
    let chunk = vec![0xFFu8; 32];
    let x_lo = vec![0.0f32; 32];
    let x_hi = vec![0.0f32; 32];
    let (lo, hi) = q4_dual_dot_32(&chunk, &x_lo, &x_hi);
    assert_eq!(lo, 0.0);
    assert_eq!(hi, 0.0);
}

#[test]
fn q4_dual_dot_32_max_nibble_high_only() {
    // All hi nibbles = 15, all lo nibbles = 0.
    let chunk = vec![0xF0u8; 32];
    let x_lo = vec![1.0f32; 32];
    let x_hi = vec![1.0f32; 32];
    let (lo, hi) = q4_dual_dot_32(&chunk, &x_lo, &x_hi);
    assert_eq!(lo, 0.0);
    assert_eq!(hi, 15.0 * 32.0);
}

/// q4k_dual_matvec_into must produce the same output as two
/// sequential q4k_matvec_into calls within f32-summation noise.
/// The two paths accumulate per-super-block in slightly different
/// orders (single running acc in the dual path; helper-based
/// per-super-block reduction in the singleton path), so strict
/// bit-equality isn't expected. Tolerance is generous enough to
/// absorb summation-order rounding but tight enough to catch any
/// real divergence.
#[test]
fn q4k_dual_matvec_into_matches_two_sequential_calls() {
    let rows = 8;
    let cols = 512; // 2 super-blocks per row, exercises the multi-block loop
    let n_elem = rows * cols;
    let weights_a: Vec<f32> = (0..n_elem)
        .map(|i| ((i as f32 / n_elem as f32) - 0.5) * 1.0)
        .collect();
    let weights_b: Vec<f32> = (0..n_elem)
        .map(|i| ((i as f32 * 0.003).cos() - 0.3) * 0.7)
        .collect();
    let q4k_a = quantize_q4_k(&weights_a);
    let q4k_b = quantize_q4_k(&weights_b);

    let x: Vec<f32> = (0..cols).map(|j| (j as f32 * 0.011).sin()).collect();

    let mut sep_a = vec![0.0f32; rows];
    let mut sep_b = vec![0.0f32; rows];
    q4k_matvec_into(&mut sep_a, &x, &q4k_a, rows, cols).expect("valid shape");
    q4k_matvec_into(&mut sep_b, &x, &q4k_b, rows, cols).expect("valid shape");

    let mut fused_a = vec![0.0f32; rows];
    let mut fused_b = vec![0.0f32; rows];
    q4k_dual_matvec_into(&mut fused_a, &mut fused_b, &x, &q4k_a, &q4k_b, rows, cols)
        .expect("valid shape");

    for r in 0..rows {
        let rel_a = (sep_a[r] - fused_a[r]).abs() / sep_a[r].abs().max(1e-6);
        let rel_b = (sep_b[r] - fused_b[r]).abs() / sep_b[r].abs().max(1e-6);
        assert!(
            rel_a < 1e-5,
            "fused matvec A row {r} drifts: sep={} fused={} rel={rel_a}",
            sep_a[r],
            fused_a[r]
        );
        assert!(
            rel_b < 1e-5,
            "fused matvec B row {r} drifts: sep={} fused={} rel={rel_b}",
            sep_b[r],
            fused_b[r]
        );
    }
}

#[test]
fn q4k_dual_matvec_into_zero_dims_zero_output() {
    let mut out_a = vec![1.0f32; 4];
    let mut out_b = vec![1.0f32; 4];
    q4k_dual_matvec_into(&mut out_a, &mut out_b, &[], &[], &[], 4, 0).expect("valid shape");
    assert!(out_a.iter().all(|&v| v == 0.0));
    assert!(out_b.iter().all(|&v| v == 0.0));
}

#[test]
fn q4k_dual_matvec_into_non_multiple_cols_is_refused_untouched() {
    // cols = 100 is not a multiple of 256: refused by name, both
    // outputs left as handed in. Matches the single-matvec contract.
    let mut out_a = vec![1.0f32; 2];
    let mut out_b = vec![2.0f32; 2];
    let x = vec![1.0f32; 100];
    let w = vec![0u8; 2 * 144];
    let err = q4k_dual_matvec_into(&mut out_a, &mut out_b, &x, &w, &w, 2, 100)
        .expect_err("cols off a super-block must be refused");
    assert_eq!(out_a, vec![1.0f32; 2]);
    assert_eq!(out_b, vec![2.0f32; 2]);
    assert_eq!(err.kernel, "q4k_dual_matvec_into (a)");
    assert_eq!(err.cols, 100);
}
