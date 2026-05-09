//! Parity tests for the CUDA Q4_0 / Q4_K / Q6_K matvec path
//! (`cuda-q4-matvec`). Same env-gate pattern as `test_cuda_f32`:
//! the tests compile on a CPU host and no-op at runtime; set
//! `LARQL_CUDA_AVAILABLE=1` on a GPU host to actually exercise
//! the dequant + cuBLAS gemv path.

#![cfg(feature = "cuda")]

use larql_compute::cpu::ops::q4_common::{q4k_to_q4kf, quantize_q4_k, quantize_q6_k};
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

fn dequant_q4kf_for_test(data: &[u8], n_elements: usize) -> Vec<f32> {
    const BLOCK_ELEMS: usize = 256;
    const BLOCK_BYTES: usize = larql_compute::pipeline::Q4_KF_BLOCK_BYTES;
    assert_eq!(n_elements % BLOCK_ELEMS, 0);
    let n_blocks = n_elements / BLOCK_ELEMS;
    assert!(data.len() >= n_blocks * BLOCK_BYTES);
    let mut out = vec![0.0f32; n_elements];
    for sb in 0..n_blocks {
        let block = &data[sb * BLOCK_BYTES..(sb + 1) * BLOCK_BYTES];
        let mut scales = [0.0f32; 8];
        let mut mins = [0.0f32; 8];
        for j in 0..8 {
            let sc = u16::from_le_bytes([block[j * 2], block[j * 2 + 1]]);
            let mn = u16::from_le_bytes([block[16 + j * 2], block[16 + j * 2 + 1]]);
            scales[j] = larql_compute::cpu::ops::q4_common::f16_to_f32(sc);
            mins[j] = larql_compute::cpu::ops::q4_common::f16_to_f32(mn);
        }
        let quants = &block[32..160];
        let base = sb * BLOCK_ELEMS;
        for g in 0..4 {
            let lo = 2 * g;
            let hi = lo + 1;
            for l in 0..32 {
                let byte = quants[g * 32 + l];
                out[base + lo * 32 + l] = scales[lo] * (byte & 0x0F) as f32 - mins[lo];
                out[base + hi * 32 + l] = scales[hi] * ((byte >> 4) & 0x0F) as f32 - mins[hi];
            }
        }
    }
    out
}

fn dense_matvec(weights: &[f32], x: &[f32], n: usize, k: usize) -> Vec<f32> {
    (0..n)
        .map(|row| {
            weights[row * k..(row + 1) * k]
                .iter()
                .zip(x)
                .map(|(w, xv)| w * xv)
                .sum()
        })
        .collect()
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
fn q4k_matvec_small_direct_parity() {
    let Some(cuda) = gpu_or_skip() else { return };
    let n = 32;
    let k = 256;
    let weights_f32 = synth(n * k, 0xB31);
    let q4k = quantize_q4_k(&weights_f32);
    let x = synth(k, 0xB32);
    let cpu = CpuBackend
        .q4k_matvec(&q4k, &x, n, k)
        .expect("CPU q4k_matvec");
    let gpu = cuda
        .q4k_matvec(&q4k, &x, n, k)
        .expect("CUDA direct q4k_matvec must be Some");
    let diff = max_abs_diff(&cpu, &gpu);
    let cos = cosine(&cpu, &gpu);
    assert!(
        diff <= TOL_ABS,
        "small Q4_K max abs diff {diff} > {TOL_ABS}"
    );
    assert!(cos >= TOL_COS, "small Q4_K cosine {cos} < {TOL_COS}");
}

#[test]
fn q4k_matvec_host_dequant_fallback_parity() {
    let Some(cuda) = gpu_or_skip() else { return };
    let n = 64;
    let k = 256;
    let weights_f32 = synth(n * k, 0xB41);
    let q4k = quantize_q4_k(&weights_f32);
    let x = synth(k, 0xB42);
    std::env::set_var("LARQL_CUDA_Q4K_HOST_DEQUANT", "1");
    let gpu = cuda
        .q4k_matvec(&q4k, &x, n, k)
        .expect("CUDA host-dequant fallback q4k_matvec must be Some");
    std::env::remove_var("LARQL_CUDA_Q4K_HOST_DEQUANT");
    let cpu = CpuBackend
        .q4k_matvec(&q4k, &x, n, k)
        .expect("CPU q4k_matvec");
    let diff = max_abs_diff(&cpu, &gpu);
    let cos = cosine(&cpu, &gpu);
    assert!(diff <= TOL_ABS, "fallback max abs diff {diff} > {TOL_ABS}");
    assert!(cos >= TOL_COS, "fallback cosine {cos} < {TOL_COS}");
}

#[test]
fn q4k_matvec_reuses_device_cache() {
    let Some(cuda) = gpu_or_skip() else { return };
    let n = 64;
    let k = 256;
    let weights_f32 = synth(n * k, 0xB51);
    let q4k = quantize_q4_k(&weights_f32);
    let x1 = synth(k, 0xB52);
    let x2 = synth(k, 0xB53);
    assert_eq!(cuda.q4k_device_cache_len(), 0);

    let gpu1 = cuda
        .q4k_matvec(&q4k, &x1, n, k)
        .expect("first CUDA q4k_matvec");
    assert_eq!(cuda.q4k_device_cache_len(), 1);
    let gpu2 = cuda
        .q4k_matvec(&q4k, &x2, n, k)
        .expect("second CUDA q4k_matvec should reuse cache");
    assert_eq!(cuda.q4k_device_cache_len(), 1);

    let cpu1 = CpuBackend
        .q4k_matvec(&q4k, &x1, n, k)
        .expect("CPU q4k_matvec x1");
    let cpu2 = CpuBackend
        .q4k_matvec(&q4k, &x2, n, k)
        .expect("CPU q4k_matvec x2");
    assert!(max_abs_diff(&cpu1, &gpu1) <= TOL_ABS);
    assert!(max_abs_diff(&cpu2, &gpu2) <= TOL_ABS);
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
fn q4kf_matvec_ffn_gate_parity() {
    let Some(cuda) = gpu_or_skip() else { return };
    let n = 10_240; // Gemma 4B intermediate
    let k = 2_560; // Gemma 4B hidden
    let weights_f32 = synth(n * k, 0xF41);
    let q4k = quantize_q4_k(&weights_f32);
    let q4kf = q4k_to_q4kf(&q4k, n, k);
    let x = synth(k, 0xF42);
    let q4kf_f32 = dequant_q4kf_for_test(&q4kf, n * k);
    let cpu = dense_matvec(&q4kf_f32, &x, n, k);
    let gpu = cuda
        .quant_matvec(larql_compute::QuantFormat::Q4_KF, &q4kf, &x, n, k)
        .expect("CUDA Q4_KF matvec must be Some");
    let diff = max_abs_diff(&cpu, &gpu);
    let cos = cosine(&cpu, &gpu);
    assert!(diff <= TOL_ABS, "Q4_KF max abs diff {diff} > {TOL_ABS}");
    assert!(cos >= TOL_COS, "Q4_KF cosine {cos} < {TOL_COS}");
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
fn q6k_matvec_reuses_device_cache() {
    let Some(cuda) = gpu_or_skip() else { return };
    let n = 64;
    let k = 256;
    let weights_f32 = synth(n * k, 0xD51);
    let q6k = quantize_q6_k(&weights_f32);
    let x1 = synth(k, 0xD52);
    let x2 = synth(k, 0xD53);
    assert_eq!(cuda.q6k_f32_device_cache_len(), 0);

    let gpu1 = cuda
        .q6k_matvec(&q6k, &x1, n, k)
        .expect("first CUDA q6k_matvec");
    assert_eq!(cuda.q6k_f32_device_cache_len(), 1);
    let gpu2 = cuda
        .q6k_matvec(&q6k, &x2, n, k)
        .expect("second CUDA q6k_matvec should reuse cache");
    assert_eq!(cuda.q6k_f32_device_cache_len(), 1);

    let cpu1 = CpuBackend
        .q6k_matvec(&q6k, &x1, n, k)
        .expect("CPU q6k_matvec x1");
    let cpu2 = CpuBackend
        .q6k_matvec(&q6k, &x2, n, k)
        .expect("CPU q6k_matvec x2");
    assert!(max_abs_diff(&cpu1, &gpu1) <= TOL_ABS);
    assert!(max_abs_diff(&cpu2, &gpu2) <= TOL_ABS);
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
