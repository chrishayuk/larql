//! Parity tests for the CUDA Q4_0 / Q4_K / Q6_K matvec path
//! (`cuda-q4-matvec`). Same env-gate pattern as `test_cuda_f32`:
//! the tests compile on a CPU host and no-op at runtime; set
//! `LARQL_CUDA_AVAILABLE=1` on a GPU host to actually exercise
//! the dequant + cuBLAS gemv path.

#![cfg(feature = "cuda")]

use larql_compute::cpu::ops::q4_common::{quantize_q4_k, quantize_q6_k};
use larql_compute::cuda::CudaBackend;
use larql_compute::prelude::*;
use larql_compute::CpuBackend;
use larql_models::quant::ggml::quantize::quantize_q4_0;

const TOL_ABS: f32 = 1e-3;
const TOL_COS: f32 = 0.9999;

fn gpu_or_skip() -> Option<CudaBackend> {
    if std::env::var("LARQL_CUDA_AVAILABLE").ok().as_deref() != Some("1") {
        return None;
    }
    CudaBackend::new().ok()
}

/// Deterministic random `f32` vector.
fn synth(n: usize, seed: u64) -> Vec<f32> {
    let mut s = seed.wrapping_mul(0x9E37_79B9_7F4A_7C15);
    (0..n)
        .map(|_| {
            s ^= s << 13;
            s ^= s >> 7;
            s ^= s << 17;
            ((s & 0xFF_FFFF) as f32 / 8_388_608.0) - 1.0
        })
        .collect()
}

fn cosine(a: &[f32], b: &[f32]) -> f32 {
    let dot: f32 = a.iter().zip(b).map(|(x, y)| x * y).sum();
    let na: f32 = a.iter().map(|v| v * v).sum::<f32>().sqrt();
    let nb: f32 = b.iter().map(|v| v * v).sum::<f32>().sqrt();
    dot / (na * nb + 1e-12)
}

fn max_abs_diff(a: &[f32], b: &[f32]) -> f32 {
    a.iter()
        .zip(b)
        .map(|(x, y)| (x - y).abs())
        .fold(0.0_f32, f32::max)
}

fn quantize_to_q8_x(x: &[f32]) -> (Vec<i8>, Vec<f32>) {
    larql_compute::cpu::ops::q4_common::quantize_to_q8(x)
}

#[test]
fn q4_0_matvec_parity() {
    let Some(cuda) = gpu_or_skip() else { return };
    let n = 1024;
    let k = 1024;
    let weights_f32 = synth(n * k, 0xA1);
    let q4_0 = quantize_q4_0(&weights_f32);
    let x = synth(k, 0xA2);
    let (q8_x, q8_scales) = quantize_to_q8_x(&x);
    let cpu = CpuBackend
        .q4_matvec(&q4_0, &q8_x, &q8_scales, n, k)
        .expect("CPU Q4_0 matvec");
    let gpu = cuda
        .q4_matvec(&q4_0, &q8_x, &q8_scales, n, k)
        .expect("CUDA Q4_0 matvec must be Some after q4 baseline");
    let diff = max_abs_diff(&cpu, &gpu);
    let cos = cosine(&cpu, &gpu);
    assert!(diff <= TOL_ABS, "max abs diff {diff} > {TOL_ABS}");
    assert!(cos >= TOL_COS, "cosine {cos} < {TOL_COS}");
}

#[test]
fn q4k_matvec_ffn_gate_parity() {
    let Some(cuda) = gpu_or_skip() else { return };
    let n = 10_240; // Gemma 4B intermediate
    let k = 2_560; // Gemma 4B hidden
    let weights_f32 = synth(n * k, 0xB1);
    let q4k = quantize_q4_k(&weights_f32);
    let x = synth(k, 0xB2);
    let cpu = CpuBackend
        .q4k_matvec(&q4k, &x, n, k)
        .expect("CPU q4k_matvec");
    let gpu = cuda
        .q4k_matvec(&q4k, &x, n, k)
        .expect("CUDA q4k_matvec must be Some");
    let diff = max_abs_diff(&cpu, &gpu);
    let cos = cosine(&cpu, &gpu);
    assert!(diff <= TOL_ABS, "max abs diff {diff} > {TOL_ABS}");
    assert!(cos >= TOL_COS, "cosine {cos} < {TOL_COS}");
}

#[test]
fn q4k_matvec_lm_head_parity() {
    let Some(cuda) = gpu_or_skip() else { return };
    let n = 128_256; // Llama-class vocab
    let k = 4_096;
    let weights_f32 = synth(n * k, 0xC1);
    let q4k = quantize_q4_k(&weights_f32);
    let x = synth(k, 0xC2);
    let cpu = CpuBackend
        .q4k_matvec(&q4k, &x, n, k)
        .expect("CPU q4k_matvec");
    let gpu = cuda
        .q4k_matvec(&q4k, &x, n, k)
        .expect("CUDA q4k_matvec must be Some");
    let diff = max_abs_diff(&cpu, &gpu);
    let cos = cosine(&cpu, &gpu);
    assert!(diff <= TOL_ABS, "LM head max abs diff {diff} > {TOL_ABS}");
    assert!(cos >= TOL_COS, "LM head cosine {cos} < {TOL_COS}");
}

#[test]
fn q6k_matvec_lm_head_parity() {
    let Some(cuda) = gpu_or_skip() else { return };
    let n = 128_256;
    let k = 4_096;
    let weights_f32 = synth(n * k, 0xD1);
    let q6k = quantize_q6_k(&weights_f32);
    let x = synth(k, 0xD2);
    let cpu = CpuBackend
        .q6k_matvec(&q6k, &x, n, k)
        .expect("CPU q6k_matvec");
    let gpu = cuda
        .q6k_matvec(&q6k, &x, n, k)
        .expect("CUDA q6k_matvec must be Some");
    let diff = max_abs_diff(&cpu, &gpu);
    let cos = cosine(&cpu, &gpu);
    assert!(diff <= TOL_ABS, "Q6_K max abs diff {diff} > {TOL_ABS}");
    assert!(cos >= TOL_COS, "Q6_K cosine {cos} < {TOL_COS}");
}

#[test]
fn quant_matvec_dispatches_to_q4k() {
    let Some(cuda) = gpu_or_skip() else { return };
    let n = 256;
    let k = 256;
    let weights_f32 = synth(n * k, 0xE1);
    let q4k = quantize_q4_k(&weights_f32);
    let x = synth(k, 0xE2);
    let direct = cuda.q4k_matvec(&q4k, &x, n, k).expect("direct q4k_matvec");
    let dispatched = cuda
        .quant_matvec(larql_compute::QuantFormat::Q4_K, &q4k, &x, n, k)
        .expect("quant_matvec dispatch");
    let diff = max_abs_diff(&direct, &dispatched);
    assert!(
        diff <= 1e-6,
        "quant_matvec dispatch must byte-match direct: max diff {diff}"
    );
}
