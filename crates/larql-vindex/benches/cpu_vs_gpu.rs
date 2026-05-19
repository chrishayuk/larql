//! CPU vs GPU side-by-side — identical operation, both backends, on
//! production-shape gate matrices.
//!
//! What's compared:
//!   1. **f32 gate KNN gemv** — single-position score-all-features.
//!      CPU goes through Accelerate / OpenBLAS via `gemv`; Metal goes
//!      through `f32_gemv_force` (the row-per-simdgroup kernel that
//!      closed lm_head on Gemma 3 4B).
//!   2. **f32 gate batch matmul** — multi-position prefill at seq_len=64.
//!      Both backends through `matmul_transb` (Metal route compiles
//!      to a fused MPS gemm on M-series).
//!   3. **Q4 gate matvec** — production decode path. CPU via
//!      `cpu.q4_matvec`, Metal via `metal.q4_matvec`. Reproduces the
//!      Q4-Metal-vs-f32-BLAS table in `PERFORMANCE.md`.
//!
//! Run:
//!   cargo bench  -p larql-vindex                   --bench cpu_vs_gpu   # CPU only
//!   cargo bench  -p larql-vindex --features metal  --bench cpu_vs_gpu   # CPU + Metal
//!
//! Without `--features metal` the Metal cases compile out and the
//! bench prints CPU-only numbers.

use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion};
use larql_compute::{CpuBackend, MatMul, QuantMatVec};
use ndarray::{Array1, Array2, ArrayView2};

fn random_query(hidden: usize) -> Array1<f32> {
    let mut state = 0xc0ffeeu64;
    Array1::from_shape_fn(hidden, |_| {
        state = state.wrapping_mul(6364136223846793005).wrapping_add(1);
        ((state >> 33) as f32) / (u32::MAX as f32) * 2.0 - 1.0
    })
}

fn synth_matrix(rows: usize, cols: usize) -> Array2<f32> {
    let mut state = 42u64;
    Array2::from_shape_fn((rows, cols), |_| {
        state = state.wrapping_mul(6364136223846793005).wrapping_add(1);
        ((state >> 33) as f32) / (u32::MAX as f32) * 2.0 - 1.0
    })
}

/// Pre-quantise a gate matrix to Q4_0 bytes for the q4_matvec
/// comparison. Layout matches `gate_vectors_q4.bin`.
fn quantise_gate_q4(gate: &ArrayView2<f32>) -> Vec<u8> {
    let (rows, cols) = (gate.shape()[0], gate.shape()[1]);
    let flat: Vec<f32> = gate.iter().copied().collect();
    debug_assert_eq!(flat.len(), rows * cols);
    larql_compute::cpu::ops::q4_common::quantize_q4_0(&flat)
}

/// (label, intermediate, hidden) — production gate-matrix shapes.
fn configs() -> &'static [(&'static str, usize, usize)] {
    &[
        ("gemma-3-4b/10240x2560", 10_240, 2560),
        ("llama-3-8b/14336x4096", 14_336, 4096),
    ]
}

fn bench_f32_gemv(c: &mut Criterion) {
    let mut group = c.benchmark_group("cpu_vs_gpu/f32_gemv_single_position");
    let cpu = CpuBackend;
    #[cfg(all(feature = "metal", target_os = "macos"))]
    let metal = larql_compute_metal::MetalBackend::new();

    for &(name, features, hidden) in configs() {
        let gate = synth_matrix(features, hidden);
        let query = random_query(hidden);
        let q_slice = query.as_slice().unwrap();

        // CPU: matmul_transb against [1, hidden] × [features, hidden]^T.
        let q_2d = query.view().into_shape_with_order((1, hidden)).unwrap();
        group.bench_with_input(
            BenchmarkId::new("cpu", name),
            &(gate.view(), q_2d),
            |b, (g, q)| {
                b.iter(|| cpu.matmul_transb(*q, *g));
            },
        );

        // Metal f32_gemv_force: dedicated row-per-simdgroup kernel.
        #[cfg(all(feature = "metal", target_os = "macos"))]
        if let Some(ref m) = metal {
            group.bench_with_input(
                BenchmarkId::new("metal", name),
                &(gate.view(), q_slice),
                |b, (g, x)| {
                    b.iter(|| m.f32_gemv_force(*g, x));
                },
            );
        }
        // Suppress unused warning when `metal` feature is off.
        let _ = q_slice;
    }
    group.finish();
}

fn bench_f32_batch_matmul(c: &mut Criterion) {
    let mut group = c.benchmark_group("cpu_vs_gpu/f32_batch_matmul_seq64");
    let cpu = CpuBackend;
    #[cfg(all(feature = "metal", target_os = "macos"))]
    let metal = larql_compute_metal::MetalBackend::new();

    let seq_len = 64usize; // typical mid-size prefill batch
    for &(name, features, hidden) in configs() {
        let gate = synth_matrix(features, hidden);
        let x = synth_matrix(seq_len, hidden);

        group.bench_with_input(
            BenchmarkId::new("cpu", name),
            &(gate.view(), x.view()),
            |b, (g, x)| {
                b.iter(|| cpu.matmul_transb(*x, *g));
            },
        );

        #[cfg(all(feature = "metal", target_os = "macos"))]
        if let Some(ref m) = metal {
            group.bench_with_input(
                BenchmarkId::new("metal", name),
                &(gate.view(), x.view()),
                |b, (g, x)| {
                    b.iter(|| m.matmul_transb(*x, *g));
                },
            );
        }
    }
    group.finish();
}

fn bench_q4_matvec(c: &mut Criterion) {
    let mut group = c.benchmark_group("cpu_vs_gpu/q4_matvec_decode");
    let cpu = CpuBackend;
    #[cfg(all(feature = "metal", target_os = "macos"))]
    let metal = larql_compute_metal::MetalBackend::new();

    for &(name, features, hidden) in configs() {
        let gate = synth_matrix(features, hidden);
        let q4_bytes = quantise_gate_q4(&gate.view());
        let query = random_query(hidden);
        let x_slice = query.as_slice().unwrap();
        let (q8_x, q8_scales) = larql_compute::cpu::q4::quantize_to_q8(x_slice);

        group.bench_with_input(
            BenchmarkId::new("cpu", name),
            &(q4_bytes.clone(), q8_x.clone(), q8_scales.clone()),
            |b, (bytes, q8x, q8s)| {
                b.iter(|| cpu.q4_matvec(bytes, q8x, q8s, features, hidden));
            },
        );

        #[cfg(all(feature = "metal", target_os = "macos"))]
        if let Some(ref m) = metal {
            group.bench_with_input(
                BenchmarkId::new("metal", name),
                &(q4_bytes.clone(), q8_x.clone(), q8_scales.clone()),
                |b, (bytes, q8x, q8s)| {
                    b.iter(|| m.q4_matvec(bytes, q8x, q8s, features, hidden));
                },
            );
        }
    }
    group.finish();
}

criterion_group!(
    benches,
    bench_f32_gemv,
    bench_f32_batch_matmul,
    bench_q4_matvec,
);
criterion_main!(benches);
