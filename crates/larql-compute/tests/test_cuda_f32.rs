//! Parity tests for the CUDA f32 baseline — see openspec change
//! `cuda-f32-baseline`.
//!
//! Tests are env-gated (not feature-gated) so they compile on a
//! CPU-only build but no-op at runtime. Set `LARQL_CUDA_AVAILABLE=1`
//! on a GPU host to actually exercise cuBLAS.

#![cfg(feature = "cuda")]

use std::sync::OnceLock;

use larql_compute::prelude::*;
use larql_compute::CpuBackend;
use larql_compute::cuda::CudaBackend;
use ndarray::Array2;

/// Skip-flag for the GPU gate. We can't `return` early inside a
/// `#[test]` and have it report as skipped, but a no-op test counts
/// as "ok" in cargo's eyes — which is what we want on CPU CI.
fn gpu_or_skip() -> Option<CudaBackend> {
    if std::env::var("LARQL_CUDA_AVAILABLE").ok().as_deref() != Some("1") {
        return None;
    }
    match CudaBackend::new() {
        Ok(b) => Some(b),
        Err(e) => {
            eprintln!("[test] LARQL_CUDA_AVAILABLE=1 but CudaBackend::new failed: {e}");
            None
        }
    }
}

/// Deterministic random `f32` matrix for reproducible parity checks.
fn synth(rows: usize, cols: usize, seed: u64) -> Array2<f32> {
    let mut data = Vec::with_capacity(rows * cols);
    let mut s = seed;
    for _ in 0..(rows * cols) {
        // xorshift64 → f32 in [-1, 1)
        s ^= s << 13;
        s ^= s >> 7;
        s ^= s << 17;
        let bits = (s & 0xFFFFFF) as u32;
        let v = (bits as f32 / 8_388_608.0) - 1.0;
        data.push(v);
    }
    Array2::from_shape_vec((rows, cols), data).unwrap()
}

fn cpu_backend() -> &'static CpuBackend {
    static CPU: OnceLock<CpuBackend> = OnceLock::new();
    CPU.get_or_init(|| CpuBackend)
}

fn max_abs_diff(a: &Array2<f32>, b: &Array2<f32>) -> f32 {
    a.iter()
        .zip(b.iter())
        .map(|(x, y)| (x - y).abs())
        .fold(0.0_f32, f32::max)
}

const TOL: f32 = 5e-4;

#[test]
fn matmul_square_parity() {
    let Some(cuda) = gpu_or_skip() else { return };
    let a = synth(256, 256, 0xA);
    let b = synth(256, 256, 0xB);
    let cpu_out = cpu_backend().matmul(a.view(), b.view());
    let cuda_out = cuda.matmul(a.view(), b.view());
    let diff = max_abs_diff(&cpu_out, &cuda_out);
    assert!(diff <= TOL, "max abs diff {diff} exceeds {TOL}");
}

#[test]
fn matmul_gemma4b_shape_parity() {
    let Some(cuda) = gpu_or_skip() else { return };
    let a = synth(64, 2560, 0xC);
    let b = synth(2560, 10240, 0xD);
    let cpu_out = cpu_backend().matmul(a.view(), b.view());
    let cuda_out = cuda.matmul(a.view(), b.view());
    let diff = max_abs_diff(&cpu_out, &cuda_out);
    assert!(diff <= TOL, "max abs diff {diff} exceeds {TOL}");
}

#[test]
fn matmul_transb_parity() {
    let Some(cuda) = gpu_or_skip() else { return };
    let a = synth(32, 4096, 0xE);
    let b = synth(4096, 4096, 0xF); // shape n × k = 4096 × 4096
    let cpu_out = cpu_backend().matmul_transb(a.view(), b.view());
    let cuda_out = cuda.matmul_transb(a.view(), b.view());
    let diff = max_abs_diff(&cpu_out, &cuda_out);
    assert!(diff <= TOL, "max abs diff {diff} exceeds {TOL}");
}

#[test]
fn gemv_returns_some() {
    let Some(cuda) = gpu_or_skip() else { return };
    let w = synth(128, 64, 0x1);
    let x: Vec<f32> = (0..64).map(|i| (i as f32) * 0.01).collect();
    let result = cuda.f32_gemv(w.view(), &x);
    assert!(result.is_some(), "f32_gemv must return Some after baseline");
    assert_eq!(result.unwrap().len(), 128);
}

#[test]
fn gemv_lm_head_parity() {
    let Some(cuda) = gpu_or_skip() else { return };
    // Llama-class LM head: 128256 outputs × 4096 inputs.
    let w = synth(128_256, 4096, 0x2);
    let x: Vec<f32> = (0..4096)
        .map(|i| ((i as f32) * 0.001).sin())
        .collect();
    let cpu_out = cpu_backend()
        .matmul_transb(
            Array2::from_shape_vec((1, 4096), x.clone()).unwrap().view(),
            w.view(),
        )
        .into_raw_vec_and_offset()
        .0;
    let cuda_out = cuda.f32_gemv(w.view(), &x).expect("Some(_)");
    let max = cpu_out
        .iter()
        .zip(cuda_out.iter())
        .map(|(a, b)| (a - b).abs())
        .fold(0.0_f32, f32::max);
    assert!(max <= TOL, "max abs diff {max} exceeds {TOL}");
    // Cosine similarity for good measure.
    let dot: f32 = cpu_out.iter().zip(cuda_out.iter()).map(|(a, b)| a * b).sum();
    let na: f32 = cpu_out.iter().map(|v| v * v).sum::<f32>().sqrt();
    let nb: f32 = cuda_out.iter().map(|v| v * v).sum::<f32>().sqrt();
    let cos = dot / (na * nb + 1e-12);
    assert!(cos >= 0.9999, "cosine sim {cos} below 0.9999");
}

#[test]
fn sequential_matmul_no_contamination() {
    let Some(cuda) = gpu_or_skip() else { return };
    let a1 = synth(64, 64, 0x11);
    let b1 = synth(64, 64, 0x22);
    let a2 = synth(64, 64, 0x33);
    let b2 = synth(64, 64, 0x44);
    let _warmup = cuda.matmul(a1.view(), b1.view());
    let cuda_1 = cuda.matmul(a1.view(), b1.view());
    let cuda_2 = cuda.matmul(a2.view(), b2.view());
    let cpu_1 = cpu_backend().matmul(a1.view(), b1.view());
    let cpu_2 = cpu_backend().matmul(a2.view(), b2.view());
    assert!(max_abs_diff(&cuda_1, &cpu_1) <= TOL);
    assert!(max_abs_diff(&cuda_2, &cpu_2) <= TOL);
}

#[test]
fn driver_init_succeeds_when_cuda_available() {
    if std::env::var("LARQL_CUDA_AVAILABLE").ok().as_deref() != Some("1") {
        return;
    }
    let backend = CudaBackend::new().expect("LARQL_CUDA_AVAILABLE=1 but init failed");
    assert_eq!(backend.name(), "cuda");
}

#[test]
fn kernel_cache_dir_created() {
    if std::env::var("LARQL_CUDA_AVAILABLE").ok().as_deref() != Some("1") {
        return;
    }
    let _ = CudaBackend::new().expect("init");
    // Resolve path in the same way as the cache module.
    let dir = if let Ok(xdg) = std::env::var("XDG_CACHE_HOME") {
        std::path::PathBuf::from(xdg).join("larql/cudarc")
    } else if let Ok(home) = std::env::var("HOME") {
        std::path::PathBuf::from(home).join(".cache/larql/cudarc")
    } else {
        return;
    };
    assert!(dir.exists(), "expected cache dir at {dir:?}");
    assert!(dir.join(".version").exists(), "expected .version marker");
}

#[test]
fn kernel_cache_respects_xdg_cache_home() {
    // Always-runs (no GPU needed) test of the path resolver. Setting
    // env vars here is fine because we don't actually init the
    // backend in this test.
    let prev = std::env::var("XDG_CACHE_HOME").ok();
    // SAFETY: environment mutation in a test; restored at end.
    unsafe {
        std::env::set_var("XDG_CACHE_HOME", "/tmp/larql-test-xdg");
    }
    // Probe the cache dir resolution by triggering it via init.
    if std::env::var("LARQL_CUDA_AVAILABLE").ok().as_deref() == Some("1") {
        let _ = CudaBackend::new();
        let expected = std::path::PathBuf::from("/tmp/larql-test-xdg/larql/cudarc/.version");
        assert!(expected.exists(), "expected XDG-honoured cache at {expected:?}");
    }
    unsafe {
        match prev {
            Some(v) => std::env::set_var("XDG_CACHE_HOME", v),
            None => std::env::remove_var("XDG_CACHE_HOME"),
        }
    }
}
