//! VERIFY-N: `encode_matmul_rows` against the production GEMV, row by row.
//!
//! A multi-position matmul is correct only if every position's output is
//! the single-position GEMV's output for that position — the verify path
//! must predict exactly what `step` would have. Checked at every width
//! 1..=8 (3, 5, 6 and 7 decompose into several arms), with non-zero input
//! and output offsets (a K/V block lands mid-cache), on a row count that
//! is not a multiple of any arm's rows-per-threadgroup (tail rows), and on
//! a row-sliced NVFP4 matrix (the packed-attention layout). Widths 5..=8
//! run the tiled simdgroup-matrix kernels, 1..=4 the multi-RHS arms; the
//! f16 tiled kernel engages only at head width, checked on its own.

#![cfg(target_os = "macos")]

use larql_compute_metal::lowering::{LoweredMatrix, MatmulRowsTarget, MatvecTarget};
use larql_compute_metal::MetalBackend;
use larql_models::quant::half::f32_to_f16;
use larql_models::quant::nvfp4;

/// Output rows: not a multiple of 8 or 16, so every arm has a tail.
const N: usize = 72;
const K: usize = 256;
const MAX_ROWS: usize = 8;
/// Activation rows placed before the block, to exercise `x_offset`.
const X_LEAD: usize = 1;
/// Output rows placed before the block, to exercise `out_offset`.
const OUT_LEAD: usize = 3;
/// Same per-row element order and fold as the GEMV; only the grouping of
/// the in-group sum differs.
const REL_RMS_BOUND: f64 = 1e-5;

fn values(seed: usize, len: usize) -> Vec<f32> {
    (0..len)
        .map(|i| (((i * 31 + seed) % 977) as f32 / 977.0) - 0.5)
        .collect()
}

fn rel_rms(reference: &[f32], got: &[f32]) -> f64 {
    let (mut num, mut den) = (0.0f64, 0.0f64);
    for (a, b) in reference.iter().zip(got) {
        num += ((a - b) as f64).powi(2);
        den += (*a as f64).powi(2);
    }
    (num / den.max(1e-30)).sqrt()
}

fn run(gpu: &MetalBackend, encode: impl Fn(&metal::ComputeCommandEncoderRef)) {
    let cmd = gpu.new_lowering_command_buffer();
    let enc = cmd.new_compute_command_encoder();
    encode(enc);
    enc.end_encoding();
    cmd.commit();
    cmd.wait_until_completed();
}

/// Every width against `reference` (the GEMV on one row), for matrix `w`.
fn check_widths(gpu: &MetalBackend, w: &LoweredMatrix<'_>, reference_w: &LoweredMatrix<'_>) {
    check_widths_n(gpu, w, reference_w, N);
}

/// [`check_widths`] at `n` output rows.
fn check_widths_n(
    gpu: &MetalBackend,
    w: &LoweredMatrix<'_>,
    reference_w: &LoweredMatrix<'_>,
    n: usize,
) {
    let f = std::mem::size_of::<f32>() as u64;
    let x = values(7, (X_LEAD + MAX_ROWS) * K);
    let xb = gpu.lowering_upload(&x).expect("x");
    // Reference: one GEMV per activation row.
    let mut reference = Vec::new();
    for r in 0..MAX_ROWS {
        let xr = gpu
            .lowering_upload(&x[(X_LEAD + r) * K..(X_LEAD + r + 1) * K])
            .expect("x row");
        let out = gpu.lowering_scratch(n);
        run(gpu, |enc| {
            gpu.encode_matvec(
                enc,
                reference_w,
                &MatvecTarget {
                    x: &xr,
                    out: &out,
                    out_offset: 0,
                    n,
                    k: K,
                },
            )
        });
        reference.push(gpu.lowering_readback(&out, n).expect("readback"));
    }
    for rows in 1..=MAX_ROWS {
        let out = gpu.lowering_scratch((OUT_LEAD + rows) * n);
        run(gpu, |enc| {
            gpu.encode_matmul_rows(
                enc,
                w,
                &MatmulRowsTarget {
                    x: &xb,
                    x_offset: (X_LEAD * K) as u64 * f,
                    out: &out,
                    out_offset: (OUT_LEAD * n) as u64 * f,
                    n,
                    k: K,
                    rows,
                },
            )
            .expect("encodes")
        });
        let got = gpu
            .lowering_readback(&out, (OUT_LEAD + rows) * n)
            .expect("readback");
        for r in 0..rows {
            let row = &got[(OUT_LEAD + r) * n..(OUT_LEAD + r + 1) * n];
            let err = rel_rms(&reference[r], row);
            assert!(
                err <= REL_RMS_BOUND,
                "width {rows}, row {r}: rel_rms {err:.2e} against the GEMV"
            );
        }
    }
}

#[test]
fn nvfp4_rows_match_the_gemv_at_every_width() {
    // Fail, never skip: a shader that does not compile makes `new()`
    // return None, and a skipped gate reads as a pass.
    let gpu = MetalBackend::new().expect("Metal backend (shader library must compile)");
    let m = nvfp4::quantize(&values(3, N * K), N, K).expect("quantise");
    let packed = gpu.lowering_weight(&m.packed);
    let scales = gpu.lowering_weight(&m.scales);
    let w = LoweredMatrix::Nvfp4 {
        packed: &packed,
        packed_offset: 0,
        scales: &scales,
        scales_offset: 0,
        tensor_scale: m.tensor_scale,
    };
    check_widths(&gpu, &w, &w);
}

#[test]
fn nvfp4_rows_bind_a_row_sliced_matrix_at_its_offsets() {
    // Fail, never skip: a shader that does not compile makes `new()`
    // return None, and a skipped gate reads as a pass.
    let gpu = MetalBackend::new().expect("Metal backend (shader library must compile)");
    // Two matrices in one allocation; the block must compute the SECOND.
    let first = nvfp4::quantize(&values(11, N * K), N, K).expect("quantise");
    let second = nvfp4::quantize(&values(5, N * K), N, K).expect("quantise");
    let packed = gpu.lowering_weight(&[first.packed.clone(), second.packed.clone()].concat());
    let scales = gpu.lowering_weight(&[first.scales.clone(), second.scales.clone()].concat());
    let sliced = LoweredMatrix::Nvfp4 {
        packed: &packed,
        packed_offset: first.packed.len() as u64,
        scales: &scales,
        scales_offset: first.scales.len() as u64,
        tensor_scale: second.tensor_scale,
    };
    let alone_packed = gpu.lowering_weight(&second.packed);
    let alone_scales = gpu.lowering_weight(&second.scales);
    let alone = LoweredMatrix::Nvfp4 {
        packed: &alone_packed,
        packed_offset: 0,
        scales: &alone_scales,
        scales_offset: 0,
        tensor_scale: second.tensor_scale,
    };
    check_widths(&gpu, &sliced, &alone);
}

#[test]
fn f16_rows_match_the_gemv_at_every_width() {
    // Fail, never skip: a shader that does not compile makes `new()`
    // return None, and a skipped gate reads as a pass.
    let gpu = MetalBackend::new().expect("Metal backend (shader library must compile)");
    let bytes: Vec<u8> = values(9, N * K)
        .iter()
        .flat_map(|v| f32_to_f16(*v).to_le_bytes())
        .collect();
    let buf = gpu.lowering_weight(&bytes);
    let w = LoweredMatrix::F16 { bytes: &buf };
    check_widths(&gpu, &w, &w);
}

#[test]
fn mxfp4_rows_are_refused_by_name() {
    // Fail, never skip: a shader that does not compile makes `new()`
    // return None, and a skipped gate reads as a pass.
    let gpu = MetalBackend::new().expect("Metal backend (shader library must compile)");
    let packed = gpu.lowering_weight(&[0u8; 16]);
    let scales = gpu.lowering_weight(&[0u8; 16]);
    let x = gpu.lowering_scratch(K);
    let out = gpu.lowering_scratch(N);
    let cmd = gpu.new_lowering_command_buffer();
    let enc = cmd.new_compute_command_encoder();
    let err = gpu
        .encode_matmul_rows(
            enc,
            &LoweredMatrix::Mxfp4 {
                packed: &packed,
                scales: &scales,
            },
            &MatmulRowsTarget {
                x: &x,
                x_offset: 0,
                out: &out,
                out_offset: 0,
                n: N,
                k: K,
                rows: 2,
            },
        )
        .expect_err("MXFP4 has no multi-position kernel");
    enc.end_encoding();
    assert!(err.contains("MXFP4"), "{err}");
}

/// The tiled f16 kernel engages only on a matrix at least this wide
/// (`F16_TILED_MIN_OUT_ROWS` in the lowering) — the LM head.
const F16_TILED_N: usize = 65536;

#[test]
fn f16_head_width_rows_match_the_gemv_through_the_tiled_kernel() {
    // Fail, never skip: a shader that does not compile makes `new()`
    // return None, and a skipped gate reads as a pass.
    let gpu = MetalBackend::new().expect("Metal backend (shader library must compile)");
    let bytes: Vec<u8> = values(13, F16_TILED_N * K)
        .iter()
        .flat_map(|v| f32_to_f16(*v).to_le_bytes())
        .collect();
    let buf = gpu.lowering_weight(&bytes);
    let w = LoweredMatrix::F16 { bytes: &buf };
    check_widths_n(&gpu, &w, &w, F16_TILED_N);
}
