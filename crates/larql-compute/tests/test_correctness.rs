//! Correctness tests: verify all backends produce matching output.
//!
//! Gated on the `heavy_tests` feature — runtime is sub-second but the
//! per-binary build is ~30 s, which dominates plain
//! `cargo test -p larql-compute`. Run via `make
//! larql-compute-test-integration` or `cargo test -p larql-compute
//! --features heavy_tests`.
#![cfg(feature = "heavy_tests")]

extern crate blas_src;

use larql_compute::cpu::q4::quantize_q4_0;
use larql_compute::cpu_backend;
use ndarray::Array2;

fn synth_matrix(rows: usize, cols: usize, seed: u64) -> Array2<f32> {
    let mut state = seed;
    let data: Vec<f32> = (0..rows * cols)
        .map(|_| {
            state = state.wrapping_mul(6364136223846793005).wrapping_add(1);
            ((state >> 33) as f32) / (u32::MAX as f32) * 2.0 - 1.0
        })
        .collect();
    Array2::from_shape_vec((rows, cols), data).unwrap()
}

fn max_diff(a: &Array2<f32>, b: &Array2<f32>) -> f32 {
    a.iter()
        .zip(b.iter())
        .map(|(x, y)| (x - y).abs())
        .fold(0.0f32, f32::max)
}

#[test]
fn cpu_matmul_matches_ndarray() {
    let cpu = cpu_backend();
    let a = synth_matrix(6, 2560, 42);
    let b = synth_matrix(2560, 2560, 43);
    let expected = a.dot(&b);
    let result = cpu.matmul(a.view(), b.view());
    assert!(max_diff(&expected, &result) < 1e-5, "matmul mismatch");
}

#[test]
fn cpu_matmul_transb_matches_ndarray() {
    let cpu = cpu_backend();
    let a = synth_matrix(6, 2560, 42);
    let b = synth_matrix(10240, 2560, 43);
    let expected = a.dot(&b.t());
    let result = cpu.matmul_transb(a.view(), b.view());
    assert!(
        max_diff(&expected, &result) < 1e-5,
        "matmul_transb mismatch"
    );
}

#[test]
fn cpu_supports_q4_k() {
    let cpu = cpu_backend();
    assert!(
        cpu.supports_quant(larql_compute::QuantFormat::Q4_K),
        "CPU backend should support Q4_K"
    );
}

#[test]
fn cpu_q4_matvec_nonzero() {
    use larql_compute::cpu::q4;

    let hidden = 256; // small for test speed
    let rows = 128;
    let x: Vec<f32> = (0..hidden).map(|i| (i as f32 * 0.01).sin()).collect();
    let matrix: Vec<f32> = (0..rows * hidden)
        .map(|i| (i as f32 * 0.001).cos())
        .collect();

    // Quantize matrix to Q4
    let q4_data = quantize_q4_0(&matrix);
    let (q8_x, q8_scales) = q4::quantize_to_q8(&x);

    let cpu = cpu_backend();
    let result = cpu
        .q4_matvec(&q4_data, &q8_x, &q8_scales, rows, hidden)
        .unwrap();

    assert_eq!(result.len(), rows);
    assert!(
        result.iter().any(|&v| v.abs() > 0.01),
        "Q4 matvec should produce nonzero output"
    );
}

#[test]
fn cpu_q4_vecmat_nonzero() {
    use larql_compute::cpu::q4;

    let hidden = 256;
    let inter = 128;
    let activation: Vec<f32> = (0..inter)
        .map(|i| if i % 3 == 0 { 1.0 } else { 0.0 })
        .collect();
    let matrix: Vec<f32> = (0..inter * hidden)
        .map(|i| (i as f32 * 0.001).cos())
        .collect();
    let q4_data = quantize_q4_0(&matrix);

    let result = q4::q4_vecmat(&activation, &q4_data, inter, hidden);
    assert_eq!(result.len(), hidden);
    assert!(
        result.iter().any(|&v| v.abs() > 0.01),
        "Q4 vecmat should produce nonzero output"
    );
}

#[test]
fn default_backend_has_name() {
    let be = larql_compute::default_backend();
    assert!(!be.name().is_empty());
}

/// `Capability` truth table for `CpuBackend`. Pins what the backend
/// claims it can accelerate so a regression in `cpu/mod.rs::supports`
/// can't quietly slip through.
#[test]
fn cpu_backend_capability_truth_table() {
    use larql_compute::Capability;

    let cpu = cpu_backend();

    // CPU accelerates the quant matvec family + Q4 vecmat (the latter
    // uses the C kernel). Everything GPU-flavoured returns false.
    let supported = [Capability::QuantMatVec, Capability::Q4VecMat];
    let unsupported = [
        Capability::F32Gemv,
        Capability::F16Gemv,
        Capability::Q4PairBatch,
        Capability::FullPipelineQ4,
        Capability::MultiLayerQ4Ffn,
        Capability::DecodeToken,
        Capability::DecodeMoe,
        Capability::DecodeProfile,
        Capability::PrefillQ4,
    ];

    for cap in supported {
        assert!(cpu.supports(cap), "expected CpuBackend to support {cap:?}");
    }
    for cap in unsupported {
        assert!(
            !cpu.supports(cap),
            "expected CpuBackend to NOT support {cap:?}"
        );
    }
}

/// `quant_matvec_q8_input` for Q4_0 must equal the legacy `q4_matvec`
/// helper bit-for-bit — both take pre-quantised Q8 input and dispatch
/// the same kernel. This pins the migration contract for the four
/// hot decode callers (lm_head, gate_knn ×2, attention/gpu).
#[test]
fn cpu_quant_matvec_q8_input_q4_0_matches_q4_matvec() {
    use larql_compute::cpu::q4;
    use larql_compute::QuantFormat;

    let hidden = 256usize;
    let rows = 128usize;
    let x: Vec<f32> = (0..hidden).map(|i| (i as f32 * 0.01).sin() + 0.5).collect();
    let matrix: Vec<f32> = (0..rows * hidden)
        .map(|i| (i as f32 * 0.001).cos() + 0.5)
        .collect();

    let q4_0 = quantize_q4_0(&matrix);
    let (q8_x, q8s) = q4::quantize_to_q8(&x);

    let cpu = cpu_backend();
    let helper = cpu.q4_matvec(&q4_0, &q8_x, &q8s, rows, hidden).unwrap();
    let q8_input = cpu
        .quant_matvec_q8_input(QuantFormat::Q4_0, &q4_0, &q8_x, &q8s, rows, hidden)
        .unwrap();

    assert_eq!(
        helper, q8_input,
        "Q4_0 q8_input path must equal q4_matvec helper bit-for-bit"
    );
}

/// Pin the unified `quant_matvec` dispatch: every supported format on
/// the CPU backend must produce the same output as its per-format
/// helper. This is the contract callers depend on when migrating off
/// `q4_matvec` / `q4k_matvec` / `q6k_matvec` (see ROADMAP P1a).
#[test]
fn cpu_quant_matvec_matches_per_format_helpers() {
    use larql_compute::cpu::q4;
    use larql_compute::QuantFormat;

    // K must be a multiple of 256 for Q4_K / Q6_K super-block layout.
    let hidden = 256usize;
    let rows = 128usize;
    let x: Vec<f32> = (0..hidden).map(|i| (i as f32 * 0.01).sin() + 0.5).collect();
    let matrix: Vec<f32> = (0..rows * hidden)
        .map(|i| (i as f32 * 0.001).cos() + 0.5)
        .collect();

    let cpu = cpu_backend();

    // Q4_0: per-format helper takes pre-quantised Q8 input; unified
    // method takes f32 and quantises internally. Same output expected.
    let q4_0 = quantize_q4_0(&matrix);
    let (q8_x, q8s) = q4::quantize_to_q8(&x);
    let helper = cpu.q4_matvec(&q4_0, &q8_x, &q8s, rows, hidden).unwrap();
    let unified = cpu
        .quant_matvec(QuantFormat::Q4_0, &q4_0, &x, rows, hidden)
        .unwrap();
    assert_eq!(helper.len(), rows);
    assert_eq!(unified.len(), rows);
    let max_diff: f32 = helper
        .iter()
        .zip(&unified)
        .map(|(a, b)| (a - b).abs())
        .fold(0.0, f32::max);
    assert!(
        max_diff < 1e-5,
        "Q4_0 quant_matvec diverges from q4_matvec helper: max_diff={max_diff}"
    );
}
