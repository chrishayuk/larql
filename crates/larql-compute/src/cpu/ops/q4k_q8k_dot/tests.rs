use super::common::BLOCK_BYTES;
#[cfg(target_arch = "x86_64")]
use super::q4k_avx2::q4k_q8k_matvec_avx2;
use super::q6k::Q6K_BLOCK_BYTES;
use super::*;
use crate::cpu::ops::q4_common::{q4k_matvec_into, quantize_q4_k, quantize_q6_k};

/// Regression for docs/audits/dec-readiness-review-2026-07-22.md §1b:
/// the DEC serving path must be able to log which kernel class it
/// selected. Pins the format (three `key=value` fields) rather than
/// specific values, since those are host-dependent.
#[test]
fn kernel_class_summary_reports_all_three_kernels() {
    let summary = kernel_class_summary();
    assert!(summary.contains("q4k_matvec="), "{summary}");
    assert!(summary.contains("q6k_matvec="), "{summary}");
    assert!(summary.contains("q4k_gate_up="), "{summary}");
}

/// Q8_K round-trip should reconstruct within 0.5% of absmax (1 LSB on
/// the 127-step scale).  Sums must equal the literal i32 sums of the
/// quantised values per sub-block.
#[test]
fn q8k_quantize_round_trip_within_quant_step() {
    let x: Vec<f32> = (0..256).map(|i| (i as f32 / 128.0 - 1.0) * 5.0).collect();
    let q = quantize_x_to_q8k(&x);
    assert_eq!(q.qs.len(), 256);
    assert_eq!(q.d.len(), 1);
    assert_eq!(q.sums.len(), 8);

    let amax = x.iter().fold(0.0f32, |a, &v| a.max(v.abs()));
    let step = amax / 127.0;
    for (xv, qv) in x.iter().zip(q.qs.iter()) {
        let recon = q.d[0] * (*qv as f32);
        assert!(
            (xv - recon).abs() < step.max(1e-6),
            "x={xv} recon={recon} step={step}"
        );
    }
    // Sums match the literal sums per sub-block.
    for s in 0..8 {
        let actual: i32 = q.qs[s * 32..(s + 1) * 32].iter().map(|&v| v as i32).sum();
        assert_eq!(actual as i16, q.sums[s]);
    }
}

/// Q8_K of all-zeros should produce zero scale + all-zero sums.
#[test]
fn q8k_zero_input_clean() {
    let x = vec![0.0f32; 256];
    let q = quantize_x_to_q8k(&x);
    assert_eq!(q.d[0], 0.0);
    assert!(q.qs.iter().all(|&v| v == 0));
    assert!(q.sums.iter().all(|&v| v == 0));
}

/// Scalar Q4_K×Q8_K matches the f32-cached path within Q8 quant noise.
/// Same Q4_K-quantised weights and same f32 activation; one path runs
/// the f32 dot `q4_common::q4k_matvec_into`, the other quantises x to
/// Q8_K and runs the integer-dot reference.  Difference should be on
/// the order of `‖w‖ · ε_q8 · ‖x‖`, well below 1e-3 for typical inputs.
#[test]
fn q8k_matvec_matches_f32_cached_within_q8_noise() {
    // Single super-block, single row matrix.
    let cols = 256;
    let rows = 4;
    let x: Vec<f32> = (0..cols).map(|i| (i as f32 * 0.013).sin()).collect();
    let w_f32: Vec<f32> = (0..rows * cols)
        .map(|i| (i as f32 * 0.007).cos() * 0.5)
        .collect();
    let w_q4 = quantize_q4_k(&w_f32);
    assert_eq!(w_q4.len(), rows * 144);

    let mut out_f32 = vec![0.0f32; rows];
    q4k_matvec_into(&mut out_f32, &x, &w_q4, rows, cols).expect("valid shape");

    let q8 = quantize_x_to_q8k(&x);
    let mut out_q8 = vec![0.0f32; rows];
    q4k_q8k_matvec_scalar(&mut out_q8, &q8, &w_q4, rows, cols).expect("valid shape");

    // Q8 quantisation step on x is amax/127; downstream noise per
    // output element is ~‖w_row‖₁ · step.  For typical sin-ramp inputs
    // that comes out in the 1e-2 range; tolerate 5e-2 to leave headroom
    // for f16 scale conversion error in d/dmin.
    for r in 0..rows {
        let diff = (out_f32[r] - out_q8[r]).abs();
        assert!(
            diff < 5e-2,
            "row {r}: f32={} q8={} diff={diff}",
            out_f32[r],
            out_q8[r]
        );
    }
}

/// Multi-block matrix: hidden=512 = 2 super-blocks per row.  Stresses
/// the per-super-block aggregation (`acc += ...` summed over 2+ blocks).
#[test]
fn q8k_matvec_multi_block_within_noise() {
    let cols = 512; // 2 super-blocks
    let rows = 16;
    let x: Vec<f32> = (0..cols).map(|i| (i as f32 * 0.011).cos() * 2.0).collect();
    let w_f32: Vec<f32> = (0..rows * cols)
        .map(|i| (i as f32 * 0.009).sin() * 0.3)
        .collect();
    let w_q4 = quantize_q4_k(&w_f32);

    let mut out_f32 = vec![0.0f32; rows];
    q4k_matvec_into(&mut out_f32, &x, &w_q4, rows, cols).expect("valid shape");

    let q8 = quantize_x_to_q8k(&x);
    let mut out_q8 = vec![0.0f32; rows];
    q4k_q8k_matvec_scalar(&mut out_q8, &q8, &w_q4, rows, cols).expect("valid shape");

    for r in 0..rows {
        let diff = (out_f32[r] - out_q8[r]).abs();
        assert!(
            diff < 8e-2,
            "row {r}: f32={} q8={} diff={diff}",
            out_f32[r],
            out_q8[r]
        );
    }
}

/// NEON kernel must be bit-identical to the scalar Q8_K reference on
/// aarch64 — both implement the same i32 dot math.  Different inputs
/// from the noise tests above to catch byte-ordering / lane-mapping
/// bugs that happen to vanish on regular ramps.
#[cfg(all(target_arch = "aarch64", target_feature = "neon"))]
#[test]
fn q8k_matvec_neon_matches_scalar_bit_exact() {
    let cols = 1024; // 4 super-blocks — exercises sb-loop + g-loop
    let rows = 7; // odd row count — exercises tail handling
                  // Use a non-symmetric, non-monotonic input so any lane/byte-swap
                  // bug can't accidentally produce the right sum.
    let x: Vec<f32> = (0..cols)
        .map(|i| {
            let f = i as f32;
            ((f * 0.0173).sin() * 1.7 + (f * 0.041).cos() * 0.9) * 1.3
        })
        .collect();
    let w_f32: Vec<f32> = (0..rows * cols)
        .map(|i| {
            let f = i as f32;
            ((f * 0.013).cos() * 0.4 - (f * 0.027).sin() * 0.2) * 0.6
        })
        .collect();
    let w_q4 = quantize_q4_k(&w_f32);
    let q8 = quantize_x_to_q8k(&x);

    let mut out_scalar = vec![0.0f32; rows];
    let mut out_neon = vec![0.0f32; rows];
    q4k_q8k_matvec_scalar(&mut out_scalar, &q8, &w_q4, rows, cols).expect("valid shape");
    q4k_q8k_matvec_neon(&mut out_neon, &q8, &w_q4, rows, cols).expect("valid shape");

    for r in 0..rows {
        assert_eq!(
            out_scalar[r].to_bits(),
            out_neon[r].to_bits(),
            "row {r}: scalar={} neon={} diff={}",
            out_scalar[r],
            out_neon[r],
            (out_scalar[r] - out_neon[r]).abs()
        );
    }
}

/// C12 hand-asm kernel must be bit-identical to the scalar reference —
/// it computes the same i32 `sum1`, same `sum2`, same f32 epilogue.
/// Exercises several shapes: odd rows (tail-free in this kernel), a
/// production attention width (2560), and a multi-super-block FFN width.
#[cfg(all(target_arch = "aarch64", target_feature = "neon"))]
#[test]
fn q8k_matvec_asm_matches_scalar_bit_exact() {
    for &(rows, cols) in &[(7usize, 1024usize), (8, 2560), (3, 2560), (16, 512)] {
        // Non-symmetric, non-monotonic inputs so a lane/byte-swap bug
        // can't accidentally produce the right sum.
        let x: Vec<f32> = (0..cols)
            .map(|i| {
                let f = i as f32;
                ((f * 0.0173).sin() * 1.7 + (f * 0.041).cos() * 0.9) * 1.3
            })
            .collect();
        let w_f32: Vec<f32> = (0..rows * cols)
            .map(|i| {
                let f = i as f32;
                ((f * 0.013).cos() * 0.4 - (f * 0.027).sin() * 0.2) * 0.6
            })
            .collect();
        let w_q4 = quantize_q4_k(&w_f32);
        let q8 = quantize_x_to_q8k(&x);

        let mut out_scalar = vec![0.0f32; rows];
        let mut out_asm = vec![0.0f32; rows];
        q4k_q8k_matvec_scalar(&mut out_scalar, &q8, &w_q4, rows, cols).expect("valid shape");
        q4k_q8k_matvec_asm(&mut out_asm, &q8, &w_q4, rows, cols).expect("valid shape");

        for r in 0..rows {
            assert_eq!(
                out_scalar[r].to_bits(),
                out_asm[r].to_bits(),
                "rows={rows} cols={cols} row {r}: scalar={} asm={} diff={}",
                out_scalar[r],
                out_asm[r],
                (out_scalar[r] - out_asm[r]).abs()
            );
        }
    }
}

/// `q4k_q8k_matvec_asm`: zero dims are the empty product (zeros written), and a
/// weight slab shorter than `rows` packed rows is refused by name with the
/// output left exactly as handed in — never silently zeroed.
#[cfg(all(target_arch = "aarch64", target_feature = "neon"))]
#[test]
fn q8k_matvec_asm_zero_dims_are_empty_and_short_weights_are_refused() {
    let empty = Q8KActivation {
        qs: vec![],
        d: vec![],
        sums: vec![],
    };
    let mut out = vec![1.0f32; 4];
    q4k_q8k_matvec_asm(&mut out, &empty, &[], 4, 0).expect("zero dims are the empty product");
    assert!(out.iter().all(|&v| v == 0.0), "zero-dims must zero output");

    let cols = 256;
    let rows = 2;
    let q = quantize_x_to_q8k(&vec![0.5f32; cols]);
    let w = vec![0u8; BLOCK_BYTES]; // one row's worth, but rows == 2
    let mut out = vec![1.0f32; rows];
    let err = q4k_q8k_matvec_asm(&mut out, &q, &w, rows, cols)
        .expect_err("a short weight slab must be refused");
    assert_eq!(out, vec![1.0f32; rows], "refusal must not touch the output");
    assert_eq!(err.kernel, "q4k_q8k_matvec_asm");
    assert_eq!(
        (err.weight_bytes, err.needed_bytes),
        (BLOCK_BYTES, 2 * BLOCK_BYTES)
    );
    assert_eq!(
        (err.out_len, err.rows, err.x_len, err.cols),
        (rows, rows, cols, cols)
    );
}

/// The fused gate+up hand-asm kernel must be bit-exact with two
/// independent scalar matvecs — same shapes discipline as the
/// single-matrix asm test, with DIFFERENT gate vs up weights so a
/// pointer/register swap between the two matrices can't cancel out.
#[cfg(all(target_arch = "aarch64", target_feature = "neon"))]
#[test]
fn q8k_gate_up_asm_matches_scalar_bit_exact() {
    for &(rows, cols) in &[(7usize, 1024usize), (8, 2560), (3, 2560), (16, 512)] {
        let x: Vec<f32> = (0..cols)
            .map(|i| {
                let f = i as f32;
                ((f * 0.0173).sin() * 1.7 + (f * 0.041).cos() * 0.9) * 1.3
            })
            .collect();
        let g_f32: Vec<f32> = (0..rows * cols)
            .map(|i| {
                let f = i as f32;
                ((f * 0.013).cos() * 0.4 - (f * 0.027).sin() * 0.2) * 0.6
            })
            .collect();
        let u_f32: Vec<f32> = (0..rows * cols)
            .map(|i| {
                let f = i as f32;
                ((f * 0.019).sin() * 0.5 + (f * 0.031).cos() * 0.3) * 0.7
            })
            .collect();
        let g_q4 = quantize_q4_k(&g_f32);
        let u_q4 = quantize_q4_k(&u_f32);
        let q8 = quantize_x_to_q8k(&x);

        let mut g_scalar = vec![0.0f32; rows];
        let mut u_scalar = vec![0.0f32; rows];
        q4k_q8k_matvec_scalar(&mut g_scalar, &q8, &g_q4, rows, cols).expect("valid shape");
        q4k_q8k_matvec_scalar(&mut u_scalar, &q8, &u_q4, rows, cols).expect("valid shape");

        let mut g_asm = vec![0.0f32; rows];
        let mut u_asm = vec![0.0f32; rows];
        q4k_q8k_gate_up_asm(&mut g_asm, &mut u_asm, &q8, &g_q4, &u_q4, rows, cols)
            .expect("valid shape");

        for r in 0..rows {
            assert_eq!(
                g_scalar[r].to_bits(),
                g_asm[r].to_bits(),
                "gate rows={rows} cols={cols} row {r}: scalar={} asm={}",
                g_scalar[r],
                g_asm[r],
            );
            assert_eq!(
                u_scalar[r].to_bits(),
                u_asm[r].to_bits(),
                "up rows={rows} cols={cols} row {r}: scalar={} asm={}",
                u_scalar[r],
                u_asm[r],
            );
        }
    }
}

/// `q4k_q8k_gate_up_asm`: zero dims zero BOTH outputs, and a short gate slab is
/// refused naming the gate half, with both outputs left untouched.
#[cfg(all(target_arch = "aarch64", target_feature = "neon"))]
#[test]
fn q8k_gate_up_asm_zero_dims_are_empty_and_short_weights_are_refused() {
    let empty = Q8KActivation {
        qs: vec![],
        d: vec![],
        sums: vec![],
    };
    let mut g = vec![1.0f32; 4];
    let mut u = vec![1.0f32; 4];
    q4k_q8k_gate_up_asm(&mut g, &mut u, &empty, &[], &[], 4, 0)
        .expect("zero dims are the empty product");
    assert!(g.iter().chain(u.iter()).all(|&v| v == 0.0));

    let cols = 256;
    let rows = 2;
    let q = quantize_x_to_q8k(&vec![0.5f32; cols]);
    let w_short = vec![0u8; BLOCK_BYTES]; // one row's worth, rows == 2
    let w_full = vec![0u8; 2 * BLOCK_BYTES];
    let mut g = vec![1.0f32; rows];
    let mut u = vec![1.0f32; rows];
    let err = q4k_q8k_gate_up_asm(&mut g, &mut u, &q, &w_short, &w_full, rows, cols)
        .expect_err("a short gate slab must be refused");
    assert_eq!(
        (g, u),
        (vec![1.0f32; rows], vec![1.0f32; rows]),
        "refusal must not touch either output"
    );
    assert_eq!(err.kernel, "q4k_q8k_gate_up_asm (gate)");
    assert_eq!(
        (err.weight_bytes, err.needed_bytes),
        (BLOCK_BYTES, 2 * BLOCK_BYTES)
    );

    // The up half is checked independently: a short UP slab names it.
    let mut g = vec![1.0f32; rows];
    let mut u = vec![1.0f32; rows];
    let err = q4k_q8k_gate_up_asm(&mut g, &mut u, &q, &w_full, &w_short, rows, cols)
        .expect_err("a short up slab must be refused");
    assert_eq!(err.kernel, "q4k_q8k_gate_up_asm (up)");
}

/// The v2 (all-glue-in-asm) kernel must be bit-exact with the scalar
/// reference: the vectorised scale/min unpack must reproduce
/// `unpack_scales_mins` exactly, `fcvt`/`scvtf` match the software
/// conversions bit-for-bit, and the epilogue preserves expression order.
#[cfg(all(target_arch = "aarch64", target_feature = "neon"))]
#[test]
fn q8k_matvec_asm_v2_matches_scalar_bit_exact() {
    for &(rows, cols) in &[(7usize, 1024usize), (8, 2560), (3, 2560), (16, 512)] {
        let x: Vec<f32> = (0..cols)
            .map(|i| {
                let f = i as f32;
                ((f * 0.0173).sin() * 1.7 + (f * 0.041).cos() * 0.9) * 1.3
            })
            .collect();
        let w_f32: Vec<f32> = (0..rows * cols)
            .map(|i| {
                let f = i as f32;
                ((f * 0.013).cos() * 0.4 - (f * 0.027).sin() * 0.2) * 0.6
            })
            .collect();
        let w_q4 = quantize_q4_k(&w_f32);
        let q8 = quantize_x_to_q8k(&x);

        let mut out_scalar = vec![0.0f32; rows];
        let mut out_v2 = vec![0.0f32; rows];
        q4k_q8k_matvec_scalar(&mut out_scalar, &q8, &w_q4, rows, cols).expect("valid shape");
        q4k_q8k_matvec_asm_v2(&mut out_v2, &q8, &w_q4, rows, cols).expect("valid shape");

        for r in 0..rows {
            assert_eq!(
                out_scalar[r].to_bits(),
                out_v2[r].to_bits(),
                "rows={rows} cols={cols} row {r}: scalar={} v2={} diff={}",
                out_scalar[r],
                out_v2[r],
                (out_scalar[r] - out_v2[r]).abs()
            );
        }
    }
}

/// The v3 (whole-row-in-asm) kernel must be bit-exact with the scalar
/// reference — the in-asm loop changes only WHERE the iteration happens,
/// not any arithmetic or its order.
#[cfg(all(target_arch = "aarch64", target_feature = "neon"))]
#[test]
fn q8k_matvec_asm_v3_matches_scalar_bit_exact() {
    for &(rows, cols) in &[(7usize, 1024usize), (8, 2560), (3, 2560), (16, 512)] {
        let x: Vec<f32> = (0..cols)
            .map(|i| {
                let f = i as f32;
                ((f * 0.0173).sin() * 1.7 + (f * 0.041).cos() * 0.9) * 1.3
            })
            .collect();
        let w_f32: Vec<f32> = (0..rows * cols)
            .map(|i| {
                let f = i as f32;
                ((f * 0.013).cos() * 0.4 - (f * 0.027).sin() * 0.2) * 0.6
            })
            .collect();
        let w_q4 = quantize_q4_k(&w_f32);
        let q8 = quantize_x_to_q8k(&x);

        let mut out_scalar = vec![0.0f32; rows];
        let mut out_v3 = vec![0.0f32; rows];
        q4k_q8k_matvec_scalar(&mut out_scalar, &q8, &w_q4, rows, cols).expect("valid shape");
        q4k_q8k_matvec_asm_v3(&mut out_v3, &q8, &w_q4, rows, cols).expect("valid shape");

        for r in 0..rows {
            assert_eq!(
                out_scalar[r].to_bits(),
                out_v3[r].to_bits(),
                "rows={rows} cols={cols} row {r}: scalar={} v3={} diff={}",
                out_scalar[r],
                out_v3[r],
                (out_scalar[r] - out_v3[r]).abs()
            );
        }
    }
}

/// The Q6_K hand-asm kernel must be bit-exact with the scalar reference
/// (and therefore the neon form) — the TBL-replicate + vector-lane scale
/// restructure changes only the i32 summation order, which is exact.
#[cfg(all(target_arch = "aarch64", target_feature = "neon"))]
#[test]
#[ignore = "TODO(q6k-planar): the asm kernel still decodes the pre-fix \
            interleaved nibble layout while the scalar reference is now \
            ggml-planar, so parity cannot hold. The dispatcher routes every \
            arch through scalar; re-enable once the asm form is reworked and \
            re-verified on ARM."]
fn q6k_matvec_asm_matches_scalar_bit_exact() {
    for &(rows, cols) in &[(7usize, 1024usize), (8, 2560), (3, 2560), (16, 512)] {
        let x: Vec<f32> = (0..cols)
            .map(|i| {
                let f = i as f32;
                ((f * 0.0173).sin() * 1.7 + (f * 0.041).cos() * 0.9) * 1.3
            })
            .collect();
        let w_f32: Vec<f32> = (0..rows * cols)
            .map(|i| {
                let f = i as f32;
                ((f * 0.013).cos() * 0.4 - (f * 0.027).sin() * 0.2) * 0.6
            })
            .collect();
        let w_q6 = quantize_q6_k(&w_f32);
        let q8 = quantize_x_to_q8k(&x);

        let mut out_scalar = vec![0.0f32; rows];
        let mut out_asm = vec![0.0f32; rows];
        q6k_q8k_matvec_scalar(&mut out_scalar, &q8, &w_q6, rows, cols).expect("valid shape");
        q6k_q8k_matvec_asm(&mut out_asm, &q8, &w_q6, rows, cols).expect("valid shape");

        for r in 0..rows {
            assert_eq!(
                out_scalar[r].to_bits(),
                out_asm[r].to_bits(),
                "rows={rows} cols={cols} row {r}: scalar={} asm={} diff={}",
                out_scalar[r],
                out_asm[r],
                (out_scalar[r] - out_asm[r]).abs()
            );
        }
    }
}

/// `q6k_q8k_matvec_asm`: zero dims are the empty product (zeros written), and a
/// weight slab shorter than `rows` packed rows is refused by name with the
/// output left exactly as handed in — never silently zeroed.
#[cfg(all(target_arch = "aarch64", target_feature = "neon"))]
#[test]
fn q6k_matvec_asm_zero_dims_are_empty_and_short_weights_are_refused() {
    let empty = Q8KActivation {
        qs: vec![],
        d: vec![],
        sums: vec![],
    };
    let mut out = vec![1.0f32; 4];
    q6k_q8k_matvec_asm(&mut out, &empty, &[], 4, 0).expect("zero dims are the empty product");
    assert!(out.iter().all(|&v| v == 0.0), "zero-dims must zero output");

    let cols = 256;
    let rows = 2;
    let q = quantize_x_to_q8k(&vec![0.5f32; cols]);
    let w = vec![0u8; Q6K_BLOCK_BYTES]; // one row's worth, but rows == 2
    let mut out = vec![1.0f32; rows];
    let err = q6k_q8k_matvec_asm(&mut out, &q, &w, rows, cols)
        .expect_err("a short weight slab must be refused");
    assert_eq!(out, vec![1.0f32; rows], "refusal must not touch the output");
    assert_eq!(err.kernel, "q6k_q8k_matvec_asm");
    assert_eq!(
        (err.weight_bytes, err.needed_bytes),
        (Q6K_BLOCK_BYTES, 2 * Q6K_BLOCK_BYTES)
    );
    assert_eq!(
        (err.out_len, err.rows, err.x_len, err.cols),
        (rows, rows, cols, cols)
    );
}

/// `quantize_x_to_q8k_into` must produce the same `qs`, `d`, `sums` as
/// the allocating `quantize_x_to_q8k` for any well-sized input — both
/// also handle resize correctly when reused across different sizes.
#[test]
fn q8k_in_place_matches_alloc_version() {
    let x: Vec<f32> = (0..512).map(|i| (i as f32 * 0.013).sin() * 3.0).collect();
    let alloc_q = quantize_x_to_q8k(&x);

    let mut buf = Q8KActivation::with_capacity(512);
    quantize_x_to_q8k_into(&mut buf, &x);

    assert_eq!(buf.qs, alloc_q.qs);
    assert_eq!(buf.d, alloc_q.d);
    assert_eq!(buf.sums, alloc_q.sums);

    // Resize-on-reuse: quantise smaller input into the same buffer.
    let x2: Vec<f32> = (0..256).map(|i| (i as f32 * 0.021).cos()).collect();
    let alloc_q2 = quantize_x_to_q8k(&x2);
    quantize_x_to_q8k_into(&mut buf, &x2);
    assert_eq!(buf.qs.len(), 256);
    assert_eq!(buf.d.len(), 1);
    assert_eq!(buf.sums.len(), 8);
    assert_eq!(buf.qs, alloc_q2.qs);
    assert_eq!(buf.d, alloc_q2.d);
    assert_eq!(buf.sums, alloc_q2.sums);
}

/// 2-row matvec must produce bit-exact outputs equal to the single-row
/// kernel for the same input — the dot math is identical, only the
/// instruction scheduling differs.  Test on both even and odd row
/// counts so the tail-handling path is exercised.
#[cfg(all(target_arch = "aarch64", target_feature = "neon"))]
#[test]
fn q8k_matvec_2row_matches_single_row_bit_exact() {
    for &rows in &[2usize, 4, 7, 11, 16, 17] {
        let cols = 1024;
        let x: Vec<f32> = (0..cols)
            .map(|i| (i as f32 * 0.0173).sin() * 1.7 + (i as f32 * 0.041).cos() * 0.9)
            .collect();
        let w_f32: Vec<f32> = (0..rows * cols)
            .map(|i| (i as f32 * 0.013).cos() * 0.4 - (i as f32 * 0.027).sin() * 0.2)
            .collect();
        let w_q4 = quantize_q4_k(&w_f32);
        let q8 = quantize_x_to_q8k(&x);

        let mut out_single = vec![0.0f32; rows];
        let mut out_2row = vec![0.0f32; rows];
        q4k_q8k_matvec_neon(&mut out_single, &q8, &w_q4, rows, cols).expect("valid shape");
        q4k_q8k_matvec_neon_2row(&mut out_2row, &q8, &w_q4, rows, cols).expect("valid shape");

        for r in 0..rows {
            assert_eq!(
                out_single[r].to_bits(),
                out_2row[r].to_bits(),
                "rows={rows} r={r}: single={} 2row={} diff={}",
                out_single[r],
                out_2row[r],
                (out_single[r] - out_2row[r]).abs()
            );
        }
    }
}

/// Fused gate+up must produce bit-exact outputs equal to two separate
/// matvec calls — both compile down to the same i32 dot math; only the
/// instruction interleaving differs.
#[test]
fn q8k_gate_up_fused_matches_separate_matvecs() {
    let cols = 1024;
    let rows = 11;
    let x: Vec<f32> = (0..cols)
        .map(|i| (i as f32 * 0.0151).sin() * 1.4 + (i as f32 * 0.029).cos() * 0.7)
        .collect();
    let g_f32: Vec<f32> = (0..rows * cols)
        .map(|i| (i as f32 * 0.011).cos() * 0.4 - (i as f32 * 0.027).sin() * 0.2)
        .collect();
    let u_f32: Vec<f32> = (0..rows * cols)
        .map(|i| (i as f32 * 0.013).sin() * 0.3 + (i as f32 * 0.041).cos() * 0.5)
        .collect();
    let g_w = quantize_q4_k(&g_f32);
    let u_w = quantize_q4_k(&u_f32);
    let q8 = quantize_x_to_q8k(&x);

    let mut g_sep = vec![0.0f32; rows];
    let mut u_sep = vec![0.0f32; rows];
    q4k_q8k_matvec_into(&mut g_sep, &q8, &g_w, rows, cols).expect("valid shape");
    q4k_q8k_matvec_into(&mut u_sep, &q8, &u_w, rows, cols).expect("valid shape");

    let mut g_fused = vec![0.0f32; rows];
    let mut u_fused = vec![0.0f32; rows];
    q4k_q8k_gate_up_into(&mut g_fused, &mut u_fused, &q8, &g_w, &u_w, rows, cols)
        .expect("valid shape");

    for r in 0..rows {
        assert_eq!(
            g_sep[r].to_bits(),
            g_fused[r].to_bits(),
            "gate row {r}: sep={} fused={}",
            g_sep[r],
            g_fused[r]
        );
        assert_eq!(
            u_sep[r].to_bits(),
            u_fused[r].to_bits(),
            "up row {r}: sep={} fused={}",
            u_sep[r],
            u_fused[r]
        );
    }
}

/// Empty / degenerate dims should produce zeros without panic.
#[test]
fn q8k_matvec_zero_dims_returns_zero() {
    let q = Q8KActivation {
        qs: vec![],
        d: vec![],
        sums: vec![],
    };
    let mut out = vec![1.0f32; 4];
    q4k_q8k_matvec_scalar(&mut out, &q, &[], 4, 0).expect("valid shape");
    assert!(out.iter().all(|&v| v == 0.0));
}

/// A weight slab shorter than `rows` packed rows is refused by name, and
/// the output is left exactly as the caller handed it in — never zeros,
/// which a sampler would accept as logits.
#[test]
fn q8k_matvec_short_weight_buffer_is_refused_untouched() {
    let cols = 256;
    let rows = 2;
    let x = vec![0.5f32; cols];
    let q = quantize_x_to_q8k(&x);
    let w = vec![0u8; 144]; // only enough for 1 row, but rows=2
    let mut out = vec![1.0f32; rows];
    let err = q4k_q8k_matvec_scalar(&mut out, &q, &w, rows, cols)
        .expect_err("a short weight slab must be refused");
    assert_eq!(out, vec![1.0f32; rows], "refusal must not touch the output");
    assert_eq!(err.kernel, "q4k_q8k_matvec_scalar");
    assert_eq!((err.weight_bytes, err.needed_bytes), (144, 288));
    assert_eq!(
        (err.out_len, err.rows, err.x_len, err.cols),
        (rows, rows, cols, cols)
    );
}

#[test]
fn q6k_q8k_matvec_matches_q6k_f32_dispatch_within_noise() {
    let cols = 512;
    let rows = 5;
    let x: Vec<f32> = (0..cols).map(|i| (i as f32 * 0.017).sin() * 1.5).collect();
    let w_f32: Vec<f32> = (0..rows * cols)
        .map(|i| (i as f32 * 0.006).cos() * 0.7)
        .collect();
    let w_q6 = quantize_q6_k(&w_f32);

    let f32_path = crate::cpu::ops::q6k_matvec::dispatch(&w_q6, &x, rows, cols);
    let q8 = quantize_x_to_q8k(&x);
    let mut q8_path = vec![0.0f32; rows];
    q6k_q8k_matvec_scalar(&mut q8_path, &q8, &w_q6, rows, cols).expect("valid shape");

    for r in 0..rows {
        let diff = (f32_path[r] - q8_path[r]).abs();
        assert!(
            diff < 1.2e-1,
            "row {r}: f32={} q8={} diff={diff}",
            f32_path[r],
            q8_path[r]
        );
    }
}

#[test]
fn q6k_q8k_public_entrypoint_matches_scalar() {
    let cols = 256;
    let rows = 3;
    let x: Vec<f32> = (0..cols).map(|i| (i as f32 * 0.031).cos()).collect();
    let w_f32: Vec<f32> = (0..rows * cols)
        .map(|i| (i as f32 * 0.011).sin() * 0.4)
        .collect();
    let w_q6 = quantize_q6_k(&w_f32);
    let q8 = quantize_x_to_q8k(&x);
    let mut scalar = vec![0.0f32; rows];
    let mut dispatched = vec![0.0f32; rows];

    q6k_q8k_matvec_scalar(&mut scalar, &q8, &w_q6, rows, cols).expect("valid shape");
    q6k_q8k_matvec_into(&mut dispatched, &q8, &w_q6, rows, cols).expect("valid shape");

    for (a, b) in scalar.iter().zip(dispatched.iter()) {
        assert_eq!(a.to_bits(), b.to_bits());
    }
}

/// `q6k_q8k_matvec_scalar`: zero dims are the empty product (zeros written), and a
/// weight slab shorter than `rows` packed rows is refused by name with the
/// output left exactly as handed in — never silently zeroed.
#[test]
fn q6k_q8k_zero_dims_are_empty_and_short_weights_are_refused() {
    let empty = Q8KActivation {
        qs: vec![],
        d: vec![],
        sums: vec![],
    };
    let mut out = vec![1.0f32; 4];
    q6k_q8k_matvec_scalar(&mut out, &empty, &[], 4, 0).expect("zero dims are the empty product");
    assert!(out.iter().all(|&v| v == 0.0), "zero-dims must zero output");

    let cols = 256;
    let rows = 2;
    let q = quantize_x_to_q8k(&vec![0.5f32; cols]);
    let w = vec![0u8; Q6K_BLOCK_BYTES]; // one row's worth, but rows == 2
    let mut out = vec![1.0f32; rows];
    let err = q6k_q8k_matvec_scalar(&mut out, &q, &w, rows, cols)
        .expect_err("a short weight slab must be refused");
    assert_eq!(out, vec![1.0f32; rows], "refusal must not touch the output");
    assert_eq!(err.kernel, "q6k_q8k_matvec_scalar");
    assert_eq!(
        (err.weight_bytes, err.needed_bytes),
        (Q6K_BLOCK_BYTES, 2 * Q6K_BLOCK_BYTES)
    );
    assert_eq!(
        (err.out_len, err.rows, err.x_len, err.cols),
        (rows, rows, cols, cols)
    );
}

/// AVX2 must produce bit-identical output to the scalar reference.
#[cfg(target_arch = "x86_64")]
#[test]
fn q8k_matvec_avx2_matches_scalar() {
    if !is_x86_feature_detected!("avx2") {
        return; // Skip on hardware without AVX2.
    }
    let cols = 1024;
    let rows = 7;
    let x: Vec<f32> = (0..cols)
        .map(|i| {
            let f = i as f32;
            ((f * 0.0173).sin() * 1.7 + (f * 0.041).cos() * 0.9) * 1.3
        })
        .collect();
    let w_f32: Vec<f32> = (0..rows * cols)
        .map(|i| {
            let f = i as f32;
            ((f * 0.013).cos() * 0.4 - (f * 0.027).sin() * 0.2) * 0.6
        })
        .collect();
    let w_q4 = quantize_q4_k(&w_f32);
    let q8 = quantize_x_to_q8k(&x);

    let mut out_scalar = vec![0.0f32; rows];
    let mut out_avx2 = vec![0.0f32; rows];
    q4k_q8k_matvec_scalar(&mut out_scalar, &q8, &w_q4, rows, cols).expect("valid shape");
    unsafe { q4k_q8k_matvec_avx2(&mut out_avx2, &q8, &w_q4, rows, cols) }.expect("valid shape");

    for r in 0..rows {
        assert_eq!(
            out_scalar[r].to_bits(),
            out_avx2[r].to_bits(),
            "row {r}: scalar={} avx2={} diff={}",
            out_scalar[r],
            out_avx2[r],
            (out_scalar[r] - out_avx2[r]).abs()
        );
    }
}

/// Unknown-format contract (`quant_route`): the kernel entry point must
/// panic on a tag with no route, never leave `out` silently zeroed —
/// a plausible-but-wrong logit vector is the dec-readiness review's §1
/// failure class.
#[test]
#[should_panic(expected = "unknown quant format tag")]
fn matvec_parallel_panics_on_unknown_format_tag_instead_of_zero_filling() {
    let x: Vec<f32> = (0..256).map(|i| i as f32 * 0.01).collect();
    let q8k_x = quantize_x_to_q8k(&x);
    let bytes = quantize_q4_k(&x);
    let mut out = vec![0.0f32; 1];
    q4k_q8k_matvec_parallel(&mut out, &q8k_x, &bytes, 1, 256, "MXFP9").expect("valid shape");
}

/// Same contract for a format that parses but has no Q8K matvec kernel.
#[test]
#[should_panic(expected = "no Q8K matvec kernel")]
fn matvec_parallel_panics_on_format_without_q8k_kernel() {
    let x: Vec<f32> = (0..256).map(|i| i as f32 * 0.01).collect();
    let q8k_x = quantize_x_to_q8k(&x);
    let mut out = vec![0.0f32; 1];
    // BF16 parses via `from_registry_tag` but has no block-stream kernel.
    let bytes = vec![0u8; 512];
    q4k_q8k_matvec_parallel(&mut out, &q8k_x, &bytes, 1, 256, "BF16").expect("valid shape");
}

/// The NEON-intrinsic fused gate+up kernel must be bit-exact with two
/// independent scalar matvecs — same discipline as the asm form. The
/// default `q4k_q8k_gate_up_into` dispatch takes the asm path, so the
/// intrinsic twin needs its own direct exercise (per-file coverage floor).
#[cfg(all(target_arch = "aarch64", target_feature = "neon"))]
#[test]
fn q8k_gate_up_neon_matches_scalar_bit_exact() {
    for &(rows, cols) in &[(7usize, 1024usize), (8, 2560), (3, 2560), (16, 512)] {
        let x: Vec<f32> = (0..cols)
            .map(|i| {
                let f = i as f32;
                ((f * 0.0173).sin() * 1.7 + (f * 0.041).cos() * 0.9) * 1.3
            })
            .collect();
        let g_f32: Vec<f32> = (0..rows * cols)
            .map(|i| {
                let f = i as f32;
                ((f * 0.013).cos() * 0.4 - (f * 0.027).sin() * 0.2) * 0.6
            })
            .collect();
        let u_f32: Vec<f32> = (0..rows * cols)
            .map(|i| {
                let f = i as f32;
                ((f * 0.019).sin() * 0.5 + (f * 0.031).cos() * 0.3) * 0.7
            })
            .collect();
        let g_q4 = quantize_q4_k(&g_f32);
        let u_q4 = quantize_q4_k(&u_f32);
        let q8 = quantize_x_to_q8k(&x);

        let mut g_scalar = vec![0.0f32; rows];
        let mut u_scalar = vec![0.0f32; rows];
        q4k_q8k_matvec_scalar(&mut g_scalar, &q8, &g_q4, rows, cols).expect("valid shape");
        q4k_q8k_matvec_scalar(&mut u_scalar, &q8, &u_q4, rows, cols).expect("valid shape");

        let mut g_neon = vec![0.0f32; rows];
        let mut u_neon = vec![0.0f32; rows];
        q4k_q8k_gate_up_neon(&mut g_neon, &mut u_neon, &q8, &g_q4, &u_q4, rows, cols)
            .expect("valid shape");

        for r in 0..rows {
            assert_eq!(
                g_scalar[r].to_bits(),
                g_neon[r].to_bits(),
                "gate rows={rows} cols={cols} row {r}: scalar={} neon={}",
                g_scalar[r],
                g_neon[r],
            );
            assert_eq!(
                u_scalar[r].to_bits(),
                u_neon[r].to_bits(),
                "up rows={rows} cols={cols} row {r}: scalar={} neon={}",
                u_scalar[r],
                u_neon[r],
            );
        }
    }
}

/// `q4k_q8k_gate_up_neon`: zero dims zero BOTH outputs, and a short gate slab is
/// refused naming the gate half, with both outputs left untouched.
#[cfg(all(target_arch = "aarch64", target_feature = "neon"))]
#[test]
fn q8k_gate_up_neon_zero_dims_are_empty_and_short_weights_are_refused() {
    let empty = Q8KActivation {
        qs: vec![],
        d: vec![],
        sums: vec![],
    };
    let mut g = vec![1.0f32; 4];
    let mut u = vec![1.0f32; 4];
    q4k_q8k_gate_up_neon(&mut g, &mut u, &empty, &[], &[], 4, 0)
        .expect("zero dims are the empty product");
    assert!(g.iter().chain(u.iter()).all(|&v| v == 0.0));

    let cols = 256;
    let rows = 2;
    let q = quantize_x_to_q8k(&vec![0.5f32; cols]);
    let w_short = vec![0u8; BLOCK_BYTES]; // one row's worth, rows == 2
    let w_full = vec![0u8; 2 * BLOCK_BYTES];
    let mut g = vec![1.0f32; rows];
    let mut u = vec![1.0f32; rows];
    let err = q4k_q8k_gate_up_neon(&mut g, &mut u, &q, &w_short, &w_full, rows, cols)
        .expect_err("a short gate slab must be refused");
    assert_eq!(
        (g, u),
        (vec![1.0f32; rows], vec![1.0f32; rows]),
        "refusal must not touch either output"
    );
    assert_eq!(err.kernel, "q4k_q8k_gate_up_neon (gate)");
    assert_eq!(
        (err.weight_bytes, err.needed_bytes),
        (BLOCK_BYTES, 2 * BLOCK_BYTES)
    );

    // The up half is checked independently: a short UP slab names it.
    let mut g = vec![1.0f32; rows];
    let mut u = vec![1.0f32; rows];
    let err = q4k_q8k_gate_up_neon(&mut g, &mut u, &q, &w_full, &w_short, rows, cols)
        .expect_err("a short up slab must be refused");
    assert_eq!(err.kernel, "q4k_q8k_gate_up_neon (up)");
}

/// The Q6_K NEON-intrinsic kernel must be bit-exact with the scalar
/// reference — the default `q6k_q8k_matvec_into` dispatch takes the asm
/// path, so the intrinsic twin needs its own direct exercise.
#[cfg(all(target_arch = "aarch64", target_feature = "neon"))]
#[test]
fn q6k_matvec_neon_matches_scalar_bit_exact() {
    for &(rows, cols) in &[(7usize, 1024usize), (8, 2560), (3, 2560), (16, 512)] {
        let x: Vec<f32> = (0..cols)
            .map(|i| {
                let f = i as f32;
                ((f * 0.0173).sin() * 1.7 + (f * 0.041).cos() * 0.9) * 1.3
            })
            .collect();
        let w_f32: Vec<f32> = (0..rows * cols)
            .map(|i| {
                let f = i as f32;
                ((f * 0.013).cos() * 0.4 - (f * 0.027).sin() * 0.2) * 0.6
            })
            .collect();
        let w_q6 = quantize_q6_k(&w_f32);
        let q8 = quantize_x_to_q8k(&x);

        let mut out_scalar = vec![0.0f32; rows];
        let mut out_neon = vec![0.0f32; rows];
        q6k_q8k_matvec_scalar(&mut out_scalar, &q8, &w_q6, rows, cols).expect("valid shape");
        q6k_q8k_matvec_neon(&mut out_neon, &q8, &w_q6, rows, cols).expect("valid shape");

        for r in 0..rows {
            assert_eq!(
                out_scalar[r].to_bits(),
                out_neon[r].to_bits(),
                "rows={rows} cols={cols} row {r}: scalar={} neon={}",
                out_scalar[r],
                out_neon[r],
            );
        }
    }
}

/// `q6k_q8k_matvec_neon`: zero dims are the empty product (zeros written), and a
/// weight slab shorter than `rows` packed rows is refused by name with the
/// output left exactly as handed in — never silently zeroed.
#[cfg(all(target_arch = "aarch64", target_feature = "neon"))]
#[test]
fn q6k_matvec_neon_zero_dims_are_empty_and_short_weights_are_refused() {
    let empty = Q8KActivation {
        qs: vec![],
        d: vec![],
        sums: vec![],
    };
    let mut out = vec![1.0f32; 4];
    q6k_q8k_matvec_neon(&mut out, &empty, &[], 4, 0).expect("zero dims are the empty product");
    assert!(out.iter().all(|&v| v == 0.0), "zero-dims must zero output");

    let cols = 256;
    let rows = 2;
    let q = quantize_x_to_q8k(&vec![0.5f32; cols]);
    let w = vec![0u8; Q6K_BLOCK_BYTES]; // one row's worth, but rows == 2
    let mut out = vec![1.0f32; rows];
    let err = q6k_q8k_matvec_neon(&mut out, &q, &w, rows, cols)
        .expect_err("a short weight slab must be refused");
    assert_eq!(out, vec![1.0f32; rows], "refusal must not touch the output");
    assert_eq!(err.kernel, "q6k_q8k_matvec_neon");
    assert_eq!(
        (err.weight_bytes, err.needed_bytes),
        (Q6K_BLOCK_BYTES, 2 * Q6K_BLOCK_BYTES)
    );
    assert_eq!(
        (err.out_len, err.rows, err.x_len, err.cols),
        (rows, rows, cols, cols)
    );
}

/// `q4k_q8k_matvec_neon`: zero dims are the empty product (zeros written), and a
/// weight slab shorter than `rows` packed rows is refused by name with the
/// output left exactly as handed in — never silently zeroed.
#[cfg(all(target_arch = "aarch64", target_feature = "neon"))]
#[test]
fn q4k_matvec_neon_zero_dims_are_empty_and_short_weights_are_refused() {
    let empty = Q8KActivation {
        qs: vec![],
        d: vec![],
        sums: vec![],
    };
    let mut out = vec![1.0f32; 4];
    q4k_q8k_matvec_neon(&mut out, &empty, &[], 4, 0).expect("zero dims are the empty product");
    assert!(out.iter().all(|&v| v == 0.0), "zero-dims must zero output");

    let cols = 256;
    let rows = 2;
    let q = quantize_x_to_q8k(&vec![0.5f32; cols]);
    let w = vec![0u8; BLOCK_BYTES]; // one row's worth, but rows == 2
    let mut out = vec![1.0f32; rows];
    let err = q4k_q8k_matvec_neon(&mut out, &q, &w, rows, cols)
        .expect_err("a short weight slab must be refused");
    assert_eq!(out, vec![1.0f32; rows], "refusal must not touch the output");
    assert_eq!(err.kernel, "q4k_q8k_matvec_neon");
    assert_eq!(
        (err.weight_bytes, err.needed_bytes),
        (BLOCK_BYTES, 2 * BLOCK_BYTES)
    );
    assert_eq!(
        (err.out_len, err.rows, err.x_len, err.cols),
        (rows, rows, cols, cols)
    );
}

/// Q4_K NEON 2-row kernel: guards zero the output, and an ODD row count
/// exercises the single-row tail fallback documented on the kernel.
#[cfg(all(target_arch = "aarch64", target_feature = "neon"))]
#[test]
fn q4k_matvec_neon_2row_guards_and_odd_row_tail() {
    let empty = Q8KActivation {
        qs: vec![],
        d: vec![],
        sums: vec![],
    };
    let mut out = vec![1.0f32; 4];
    q4k_q8k_matvec_neon_2row(&mut out, &empty, &[], 4, 0).expect("valid shape");
    assert!(out.iter().all(|&v| v == 0.0));

    let cols = 512;
    let rows = 3; // odd → last row via the single-row tail
    let q8 = quantize_x_to_q8k(
        &(0..cols)
            .map(|i| ((i as f32) * 0.017).sin())
            .collect::<Vec<_>>(),
    );
    let w_f32: Vec<f32> = (0..rows * cols)
        .map(|i| ((i as f32) * 0.011).cos() * 0.5)
        .collect();
    let w = quantize_q4_k(&w_f32);

    // short-weight guard: one row's bytes for rows == 3 is refused,
    // output untouched.
    let mut out = vec![1.0f32; rows];
    let err = q4k_q8k_matvec_neon_2row(&mut out, &q8, &w[..2 * BLOCK_BYTES], rows, cols)
        .expect_err("a short weight slab must be refused");
    assert_eq!(out, vec![1.0f32; rows]);
    assert_eq!(err.kernel, "q4k_q8k_matvec_neon_2row");

    let mut out_single = vec![0.0f32; rows];
    let mut out_2row = vec![0.0f32; rows];
    q4k_q8k_matvec_neon(&mut out_single, &q8, &w, rows, cols).expect("valid shape");
    q4k_q8k_matvec_neon_2row(&mut out_2row, &q8, &w, rows, cols).expect("valid shape");
    for r in 0..rows {
        assert_eq!(
            out_single[r].to_bits(),
            out_2row[r].to_bits(),
            "row {r}: single={} 2row={}",
            out_single[r],
            out_2row[r],
        );
    }
}

// ── Public execution path: `q4k_q8k_matvec_parallel` ─────────────────────
//
// The parallel entry is the single source for every quantised projection
// on the decode path. One fixture serves the refusal cases and the parity
// case beside them, so the same weights prove both halves of the change:
// an invalid shape refuses with the output untouched, and the unchanged
// valid shape still computes exactly what the scalar reference computes.

/// `(packed Q4_K weights, Q8_K activation, rows, cols)` for the public-path
/// tests. Five rows keeps a partial final chunk out of the picture; two
/// super-blocks per row keeps the scale/min unpack exercised twice.
fn public_path_fixture() -> (Vec<u8>, Q8KActivation, usize, usize) {
    let rows = 5;
    let cols = 512;
    let x: Vec<f32> = (0..cols).map(|i| (i as f32 * 0.013).sin() * 1.7).collect();
    let w_f32: Vec<f32> = (0..rows * cols)
        .map(|i| (i as f32 * 0.007).cos() * 0.6)
        .collect();
    (quantize_q4_k(&w_f32), quantize_x_to_q8k(&x), rows, cols)
}

/// Valid shape, unchanged numerics: the parallel entry is bit-identical to
/// the scalar reference on the fixture the refusal tests below reuse.
#[test]
fn q8k_matvec_parallel_computes_exactly_the_scalar_reference_on_the_fixture() {
    let (w, q, rows, cols) = public_path_fixture();
    let mut reference = vec![0.0f32; rows];
    q4k_q8k_matvec_scalar(&mut reference, &q, &w, rows, cols).expect("valid shape");
    let mut out = vec![7.0f32; rows];
    q4k_q8k_matvec_parallel(&mut out, &q, &w, rows, cols, "Q4_K").expect("valid shape");
    assert!(
        reference.iter().any(|&v| v != 0.0),
        "control: the fixture must not be all zeros"
    );
    for r in 0..rows {
        assert_eq!(
            out[r].to_bits(),
            reference[r].to_bits(),
            "row {r}: parallel={} scalar={}",
            out[r],
            reference[r]
        );
    }
}

/// An activation that is not `cols` long is refused by name — the hand-asm
/// kernels read `qs` through a bare pointer, so this used to be an OOB
/// read hidden behind a zero-filled output.
#[test]
fn q8k_matvec_parallel_refuses_an_activation_of_the_wrong_length() {
    let (w, _, rows, cols) = public_path_fixture();
    let wrong = quantize_x_to_q8k(&vec![0.5f32; cols + 256]);
    let mut out = vec![7.0f32; rows];
    let err = q4k_q8k_matvec_parallel(&mut out, &wrong, &w, rows, cols, "Q4_K")
        .expect_err("activation length must match cols");
    assert_eq!(out, vec![7.0f32; rows], "refusal must not touch the output");
    assert_eq!(err.kernel, "q4k_q8k_matvec_parallel");
    assert_eq!((err.x_len, err.cols), (cols + 256, cols));
    assert!(
        err.to_string()
            .contains("activation length 768 for 512 cols"),
        "{err}"
    );
}

/// An output shorter than `rows` is refused before any row is written.
#[test]
fn q8k_matvec_parallel_refuses_an_output_shorter_than_rows() {
    let (w, q, rows, cols) = public_path_fixture();
    let mut out = vec![7.0f32; rows - 1];
    let err = q4k_q8k_matvec_parallel(&mut out, &q, &w, rows, cols, "Q4_K")
        .expect_err("output must hold rows");
    assert_eq!(out, vec![7.0f32; rows - 1]);
    assert_eq!((err.out_len, err.rows), (rows - 1, rows));
}

/// A weight slab shorter than `rows` packed rows is refused by name. This
/// case used to panic at the entry point; it is now the same typed
/// refusal the per-row kernels return, so a caller with a channel can
/// carry it up.
#[test]
fn q8k_matvec_parallel_refuses_a_short_weight_slab() {
    let (w, q, rows, cols) = public_path_fixture();
    let short = &w[..w.len() - BLOCK_BYTES];
    let mut out = vec![7.0f32; rows];
    let err = q4k_q8k_matvec_parallel(&mut out, &q, short, rows, cols, "Q4_K")
        .expect_err("short slab must be refused");
    assert_eq!(out, vec![7.0f32; rows]);
    assert_eq!(
        (err.weight_bytes, err.needed_bytes),
        (w.len() - BLOCK_BYTES, w.len())
    );
}

/// The refusal reaches the `FormatRoute` registry's kernel pointer too:
/// the Q6_K route returns the same typed error.
#[test]
fn q8k_matvec_parallel_refuses_through_the_q6k_route_as_well() {
    let (_, q, rows, cols) = public_path_fixture();
    let short = vec![0u8; Q6K_BLOCK_BYTES];
    let mut out = vec![7.0f32; rows];
    let err = q4k_q8k_matvec_parallel(&mut out, &q, &short, rows, cols, "Q6_K")
        .expect_err("short Q6_K slab must be refused");
    assert_eq!(out, vec![7.0f32; rows]);
    assert_eq!(err.needed_bytes, rows * 2 * Q6K_BLOCK_BYTES);
}
