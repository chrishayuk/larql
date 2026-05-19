//! Cross-backend f32 / f16 matmul + gemv benchmarks.
//!
//! Complements `benches/quant_matvec.rs` — that one covers quantised
//! matvec; this one covers the **dense** f32 / f16 surface
//! (`matmul`, `matmul_transb`, `f32_gemv`, `f16_gemv`) at the shapes
//! the production decode and lm-head paths actually run.
//!
//! Run: `cargo bench -p larql-compute --bench matmul`
//! Or with metal: `cargo bench -p larql-compute --features metal --bench matmul`
//!
//! ## What's covered
//!
//! - **`matmul_transb`** at three shapes: tile (6×2560×2560), FFN
//!   gate/up shape (6×10240×2560), and lm-head vocab projection
//!   (1×262144×2560 — the row-drop regression-detector shape).
//! - **`f32_gemv`** (Metal-only — CPU returns `None`) at the lm-head
//!   shape — the specialised single-row × large-N × large-K kernel.
//! - **`f16_gemv`** (Metal-only) at the same shape but with a `half`
//!   weight matrix — saves a 5.6 GB f32 clone on tied-embedding 31B
//!   models.

extern crate blas_src;

use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use larql_compute::prelude::*;
use larql_compute::CpuBackend;
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

/// Cross-backend `matmul_transb` at three production-relevant shapes.
fn bench_matmul_transb(c: &mut Criterion) {
    let mut group = c.benchmark_group("matmul_transb");
    group.sample_size(20);

    let cpu = CpuBackend;

    #[cfg(target_os = "macos")]
    let metal = larql_compute_metal::MetalBackend::new();
    #[cfg(target_os = "macos")]
    if let Some(ref m) = metal {
        m.set_flop_threshold(1);
    }

    // The 262144 lm-head shape allocates a 2.5 GB weight matrix. On
    // CI runners with shared GPU memory it's enough to push the
    // following bench's Metal alloc into null-pointer territory; cap
    // at 32 K rows there (320 MB) — same kernel coverage, no OOM.
    let lmhead_n = if std::env::var_os("CI").is_some() {
        32_768usize
    } else {
        262_144usize
    };
    for &(m, n, k) in &[
        (6usize, 2_560usize, 2_560usize),
        (6, 10_240, 2_560),
        (1, lmhead_n, 2_560),
    ] {
        let a = synth_matrix(m, k, 42);
        let b = synth_matrix(n, k, 43);
        let label = format!("M{m}_N{n}_K{k}");
        group.throughput(Throughput::Elements((m * n * k) as u64));

        group.bench_with_input(
            BenchmarkId::from_parameter(format!("cpu/{label}")),
            &(&a, &b),
            |bench, (a, b)| {
                bench.iter(|| cpu.matmul_transb(a.view(), b.view()));
            },
        );

        #[cfg(target_os = "macos")]
        if let Some(ref m_be) = metal {
            group.bench_with_input(
                BenchmarkId::from_parameter(format!("metal/{label}")),
                &(&a, &b),
                |bench, (a, b)| {
                    bench.iter(|| m_be.matmul_transb(a.view(), b.view()));
                },
            );
        }
    }
    group.finish();
}

/// Specialised single-row gemv at the lm-head shape (Metal-only —
/// CPU's `f32_gemv` returns `None` and the caller falls back to
/// `matmul_transb`). Bench covers the vocab projection where `M=1`
/// makes the tiled sgemm waste 31/32 threads, and the
/// row-per-simdgroup `f32_gemv` shader's the specialised replacement.
///
/// Default shape is Gemma's full 262144-vocab × 2560-hidden (2.5 GB
/// weight matrix). On CI we drop to N=32768 — same kernel, same
/// arithmetic intensity, but a 320 MB buffer that fits comfortably on
/// shared GitHub mac runners. Override either with
/// `LARQL_BENCH_LMHEAD_N` if you want a specific size.
#[cfg(target_os = "macos")]
fn bench_f32_gemv_lmhead(c: &mut Criterion) {
    let Some(metal) = larql_compute_metal::MetalBackend::new() else {
        return;
    };
    metal.set_flop_threshold(1);

    let default_n = if std::env::var_os("CI").is_some() {
        32_768usize
    } else {
        262_144usize
    };
    let n = std::env::var("LARQL_BENCH_LMHEAD_N")
        .ok()
        .and_then(|v| v.parse::<usize>().ok())
        .unwrap_or(default_n);
    let k = 2_560usize;
    let w = synth_matrix(n, k, 42);
    let x: Vec<f32> = (0..k).map(|i| ((i as f32) * 0.013).sin() * 0.5).collect();

    let mut group = c.benchmark_group("f32_gemv_lmhead");
    group.sample_size(20);
    group.throughput(Throughput::Elements((n * k) as u64));
    let label = format!("metal/N{n}_K{k}");
    group.bench_function(BenchmarkId::from_parameter(label), |bench| {
        bench.iter(|| metal.f32_gemv_force(w.view(), &x));
    });
    group.finish();
}

#[cfg(not(target_os = "macos"))]
fn bench_f32_gemv_lmhead(_c: &mut Criterion) { /* metal-only */
}

criterion_group!(benches, bench_matmul_transb, bench_f32_gemv_lmhead);
criterion_main!(benches);
