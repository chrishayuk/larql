//! Coverage for the backend trait default methods (matmul_batch, gemv stubs)
//! and quant_matvec dispatch for Q4_K / Q6_K / quant_matvec_q8_input.

extern crate blas_src;

use larql_compute::cpu::ops::q4_common::{quantize_q4_k, quantize_q6_k, quantize_to_q8};
use larql_compute::prelude::*;
use larql_compute::{cpu_backend, MatMulOp, QuantFormat};
use ndarray::Array2;

fn synth(rows: usize, cols: usize, seed: u64) -> Array2<f32> {
    let mut s = seed;
    Array2::from_shape_fn((rows, cols), |_| {
        s = s.wrapping_mul(6364136223846793005).wrapping_add(1);
        ((s >> 33) as f32) / (u32::MAX as f32) * 2.0 - 1.0
    })
}

fn synth_vec(len: usize, seed: u64) -> Vec<f32> {
    let mut s = seed;
    (0..len)
        .map(|_| {
            s = s.wrapping_mul(6364136223846793005).wrapping_add(1);
            ((s >> 33) as f32) / (u32::MAX as f32) * 2.0 - 1.0
        })
        .collect()
}

// ── MatMul::matmul_batch ─────────────────────────────────────────────────────

#[test]
fn matmul_batch_no_transpose_serial_dispatch() {
    let cpu = cpu_backend();
    let a1 = synth(3, 4, 1);
    let b1 = synth(4, 5, 2);
    let a2 = synth(2, 4, 3);
    let b2 = synth(4, 6, 4);
    let ops = vec![
        MatMulOp {
            a: a1.clone(),
            b: b1.clone(),
            transpose_b: false,
        },
        MatMulOp {
            a: a2.clone(),
            b: b2.clone(),
            transpose_b: false,
        },
    ];
    let results = cpu.matmul_batch(&ops);
    assert_eq!(results.len(), 2);
    assert_eq!(results[0].shape(), &[3, 5]);
    assert_eq!(results[1].shape(), &[2, 6]);
    // Verify against individual matmul calls
    let expected0 = cpu.matmul(a1.view(), b1.view());
    let expected1 = cpu.matmul(a2.view(), b2.view());
    let diff0: f32 = results[0]
        .iter()
        .zip(&expected0)
        .map(|(a, b)| (a - b).abs())
        .fold(0.0, f32::max);
    let diff1: f32 = results[1]
        .iter()
        .zip(&expected1)
        .map(|(a, b)| (a - b).abs())
        .fold(0.0, f32::max);
    assert!(diff0 < 1e-5);
    assert!(diff1 < 1e-5);
}

#[test]
fn matmul_batch_with_transpose_serial_dispatch() {
    let cpu = cpu_backend();
    let a = synth(3, 8, 5);
    let b = synth(6, 8, 6); // B is [6, 8], transpose → [8, 6]
    let ops = vec![MatMulOp {
        a: a.clone(),
        b: b.clone(),
        transpose_b: true,
    }];
    let results = cpu.matmul_batch(&ops);
    assert_eq!(results[0].shape(), &[3, 6]);
    let expected = cpu.matmul_transb(a.view(), b.view());
    let diff: f32 = results[0]
        .iter()
        .zip(&expected)
        .map(|(a, b)| (a - b).abs())
        .fold(0.0, f32::max);
    assert!(diff < 1e-5);
}

// ── MatMul gemv stubs (CPU returns None) ─────────────────────────────────────

#[test]
fn f32_gemv_returns_none_on_cpu() {
    let cpu = cpu_backend();
    let w = synth(512, 256, 7);
    let x = synth_vec(256, 8);
    assert!(cpu.f32_gemv(w.view(), &x).is_none());
}

#[test]
fn f32_gemv_force_returns_none_on_cpu() {
    let cpu = cpu_backend();
    let w = synth(512, 256, 9);
    let x = synth_vec(256, 10);
    // Default delegates to f32_gemv, so also None.
    assert!(cpu.f32_gemv_force(w.view(), &x).is_none());
}

#[test]
fn f16_gemv_returns_none_on_cpu() {
    let cpu = cpu_backend();
    let n = 512usize;
    let k = 256usize;
    let w_f16 = vec![0u8; n * k * 2];
    let x = synth_vec(k, 11);
    assert!(cpu.f16_gemv(&w_f16, &x, n, k).is_none());
}

#[test]
fn f16_gemv_force_returns_none_on_cpu() {
    let cpu = cpu_backend();
    let n = 512usize;
    let k = 256usize;
    let w_f16 = vec![0u8; n * k * 2];
    let x = synth_vec(k, 12);
    // Default delegates to f16_gemv, so also None.
    assert!(cpu.f16_gemv_force(&w_f16, &x, n, k).is_none());
}

// ── QuantMatVec::quant_matvec for Q4_K and Q6_K ──────────────────────────────

#[test]
fn quant_matvec_q4k_dispatches_to_q4k_kernel() {
    let cpu = cpu_backend();
    let hidden = 256usize;
    let rows = 4usize;
    let weights: Vec<f32> = synth_vec(rows * hidden, 13);
    let x: Vec<f32> = synth_vec(hidden, 14);
    let q4k = quantize_q4_k(&weights);
    let result = cpu
        .quant_matvec(QuantFormat::Q4_K, &q4k, &x, rows, hidden)
        .expect("CPU should support Q4_K via q4k_matvec");
    assert_eq!(result.len(), rows);
    assert!(
        result.iter().any(|v| v.abs() > 1e-4),
        "expected nonzero output"
    );
}

#[test]
fn quant_matvec_q4kf_dispatches_same_as_q4k() {
    // Q4_KF is the pre-baked half-scale layout used by the fast GPU path.
    let cpu = cpu_backend();
    let hidden = 256usize;
    let rows = 4usize;
    let weights: Vec<f32> = synth_vec(rows * hidden, 15);
    let x: Vec<f32> = synth_vec(hidden, 16);
    let q4kf = larql_compute::cpu::ops::q4_common::quantize_q4_kf(&weights);
    let result = cpu
        .quant_matvec(QuantFormat::Q4_KF, &q4kf, &x, rows, hidden)
        .expect("CPU should support Q4_KF via q4kf_matvec");
    assert_eq!(result.len(), rows);
}

#[test]
fn quant_matvec_q6k_dispatches_to_q6k_kernel() {
    let cpu = cpu_backend();
    let hidden = 256usize;
    let rows = 4usize;
    let weights: Vec<f32> = synth_vec(rows * hidden, 17);
    let x: Vec<f32> = synth_vec(hidden, 18);
    let q6k = quantize_q6_k(&weights);
    let result = cpu
        .quant_matvec(QuantFormat::Q6_K, &q6k, &x, rows, hidden)
        .expect("CPU should support Q6_K via q6k_matvec");
    assert_eq!(result.len(), rows);
    assert!(
        result.iter().any(|v| v.abs() > 1e-4),
        "expected nonzero output"
    );
}

// ── QuantMatVec::quant_matvec_q8_input for Q4_K (triggers dequantise_q8) ────

#[test]
fn quant_matvec_q8_input_q4k_dequantises_then_dispatches() {
    // quant_matvec_q8_input with Q4_K hits the dequantise_q8 → f32 → q4k_matvec path.
    let cpu = cpu_backend();
    let hidden = 256usize;
    let rows = 4usize;
    let weights: Vec<f32> = synth_vec(rows * hidden, 19);
    let x: Vec<f32> = synth_vec(hidden, 20);
    let q4k = quantize_q4_k(&weights);
    let (q8_x, q8_scales) = quantize_to_q8(&x);

    let result = cpu
        .quant_matvec_q8_input(QuantFormat::Q4_K, &q4k, &q8_x, &q8_scales, rows, hidden)
        .expect("CPU should support Q4_K via quant_matvec_q8_input");
    assert_eq!(result.len(), rows);
    // Should approximately match quant_matvec (some Q8 round-trip error expected)
    let direct = cpu
        .quant_matvec(QuantFormat::Q4_K, &q4k, &x, rows, hidden)
        .unwrap();
    let max_diff: f32 = result
        .iter()
        .zip(&direct)
        .map(|(a, b)| (a - b).abs())
        .fold(0.0, f32::max);
    let mag: f32 = direct.iter().map(|v| v.abs()).fold(0.0, f32::max);
    // Allow up to 5% relative error from the Q8 round-trip
    assert!(
        max_diff < 0.05 * mag.max(1.0),
        "Q8-input path diverges from f32 path: {max_diff} vs mag {mag}"
    );
}

#[test]
fn quant_matvec_q8_input_q6k_dequantises_then_dispatches() {
    let cpu = cpu_backend();
    let hidden = 256usize;
    let rows = 4usize;
    let weights: Vec<f32> = synth_vec(rows * hidden, 21);
    let x: Vec<f32> = synth_vec(hidden, 22);
    let q6k = quantize_q6_k(&weights);
    let (q8_x, q8_scales) = quantize_to_q8(&x);

    let result = cpu
        .quant_matvec_q8_input(QuantFormat::Q6_K, &q6k, &q8_x, &q8_scales, rows, hidden)
        .expect("CPU should support Q6_K via quant_matvec_q8_input");
    assert_eq!(result.len(), rows);
}

// ── QuantMatVec::q4_vecmat via trait ─────────────────────────────────────────

#[test]
fn q4_vecmat_via_trait_nonzero() {
    use larql_compute::cpu::ops::q4_common::quantize_q4_0;
    let cpu = cpu_backend();
    let inter = 128usize;
    let hidden = 256usize;
    let activation: Vec<f32> = synth_vec(inter, 23);
    let matrix: Vec<f32> = synth_vec(inter * hidden, 24);
    let q4 = quantize_q4_0(&matrix);
    let result = cpu
        .q4_vecmat(&activation, &q4, inter, hidden)
        .expect("CPU should support q4_vecmat");
    assert_eq!(result.len(), hidden);
    assert!(result.iter().any(|v| v.abs() > 1e-4));
}

// ── MinimalBackend — exercises default trait implementations ──────────────────

use larql_compute::backend::DecodeBackend;
use ndarray::ArrayView2;

struct MinimalBackend;

impl MatMul for MinimalBackend {
    fn matmul(&self, a: ArrayView2<f32>, b: ArrayView2<f32>) -> Array2<f32> {
        a.dot(&b)
    }
    fn matmul_transb(&self, a: ArrayView2<f32>, b: ArrayView2<f32>) -> Array2<f32> {
        a.dot(&b.t())
    }
}
impl QuantMatVec for MinimalBackend {} // all methods default to None/false
impl DecodeBackend for MinimalBackend {} // all methods default to None/no-op
impl larql_compute::ComputeBackend for MinimalBackend {
    fn name(&self) -> &str {
        "minimal"
    }
    fn as_any(&self) -> &dyn std::any::Any {
        self
    }
    // device_info: default → self.name().to_string()
    // supports:    default → false
}

#[test]
fn default_device_info_delegates_to_name() {
    let be = MinimalBackend;
    assert_eq!(be.device_info(), "minimal");
}

#[test]
fn default_supports_returns_false() {
    let be = MinimalBackend;
    assert!(!be.supports(larql_compute::Capability::F32Gemv));
    assert!(!be.supports(larql_compute::Capability::FullPipelineQ4));
}

#[test]
fn default_quant_matvec_stubs_return_none() {
    let be = MinimalBackend;
    let dummy = vec![0u8; 18];
    let dummy_i8 = vec![0i8; 32];
    let dummy_f32 = vec![0.0f32; 256];
    let dummy_scales = vec![0.0f32; 1];
    assert!(be
        .q4_matvec(&dummy, &dummy_i8, &dummy_scales, 1, 32)
        .is_none());
    assert!(be.q4_vecmat(&dummy_f32[..32], &dummy, 32, 256).is_none());
    assert!(be.q4k_matvec(&dummy, &dummy_f32[..256], 1, 256).is_none());
    assert!(be.q6k_matvec(&dummy, &dummy_f32[..256], 1, 256).is_none());
    assert!(be
        .q4_matvec_pair_batch(&dummy, &dummy, &dummy_f32[..256], 1, 1, 256)
        .is_none());
    assert!(!be.has_q4());
}

#[test]
fn default_decode_stubs() {
    let be = MinimalBackend;
    assert!(!be.has_kv_cache());
    be.reset_kv_cache(); // default no-op, must not panic
}

#[test]
fn default_speculative_tree_keep_cache_returns_none() {
    // `cuda-spec-branching-tree` T3.2 default: backends that don't
    // implement the tree-mask kernel return None so the spec
    // dispatcher routes to the conservative fallback.
    let be = MinimalBackend;
    let x = vec![vec![0.0_f32; 16]; 3];
    let ancestors = vec![0b001, 0b011, 0b101]; // 3-node tree
    let out = be.decode_tokens_speculative_tree_keep_cache(
        &[],
        &x,
        &ancestors,
        16,
        16,
        16,
        16,
        1,
        1,
        16,
        10_000.0,
    );
    assert!(out.is_none());
}
