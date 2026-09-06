//! Criterion benchmarks for the linalg primitives — Cholesky and the
//! ridge-regression decomposition `ridge_decomposition_solve` (the
//! generic solve underlying `larql_vindex::memit_solve`).
//!
//! Run: `cargo bench -p larql-compute --bench linalg`

extern crate blas_src;

#[path = "support/qual_slowdown.rs"]
mod qual_slowdown;

use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion};
use larql_compute::cpu::ops::linalg::{cholesky, cholesky_solve, ridge_decomposition_solve};
use ndarray::Array2;

fn synth_matrix_f32(rows: usize, cols: usize, seed: u64) -> Array2<f32> {
    let mut state = seed;
    Array2::from_shape_fn((rows, cols), |_| {
        state = state.wrapping_mul(6364136223846793005).wrapping_add(1);
        ((state >> 33) as f32) / (u32::MAX as f32) * 2.0 - 1.0
    })
}

fn synth_spd_f64(n: usize, seed: u64) -> Array2<f64> {
    // X X^T + nI is symmetric positive-definite.
    let x = {
        let mut state = seed;
        Array2::<f64>::from_shape_fn((n, n), |_| {
            state = state.wrapping_mul(6364136223846793005).wrapping_add(1);
            ((state >> 33) as f64) / (u32::MAX as f64) * 2.0 - 1.0
        })
    };
    let mut a = x.dot(&x.t());
    for i in 0..n {
        a[[i, i]] += n as f64;
    }
    a
}

fn bench_cholesky(c: &mut Criterion) {
    let mut group = c.benchmark_group("cholesky_factor");
    for &n in &[16usize, 64, 256] {
        let a = synth_spd_f64(n, 42);
        group.bench_with_input(BenchmarkId::from_parameter(n), &a, |b, a| {
            b.iter(|| qual_slowdown::slowdown().run(|| cholesky(a, 1e-6).unwrap()));
        });
    }
    group.finish();
}

fn bench_cholesky_solve(c: &mut Criterion) {
    let mut group = c.benchmark_group("cholesky_solve");
    for &n in &[16usize, 64, 256] {
        let a = synth_spd_f64(n, 99);
        let l = cholesky(&a, 1e-6).unwrap();
        let rhs = Array2::<f64>::from_elem((n, 64), 0.5);
        group.bench_with_input(
            BenchmarkId::from_parameter(n),
            &(&l, &rhs),
            |b, (l, rhs)| {
                b.iter(|| qual_slowdown::slowdown().run(|| cholesky_solve(l, rhs)));
            },
        );
    }
    group.finish();
}

fn bench_ridge_decomposition(c: &mut Criterion) {
    // Realistic MEMIT shapes: N facts × hidden_dim d.
    // d=2560 is Gemma 3 4B's hidden_dim; d=128 is a small-model proxy.
    let mut group = c.benchmark_group("ridge_decomposition_solve");
    group.sample_size(20); // d=2560, N=120 is multi-second per iter
    for &(n, d) in &[
        (10usize, 128usize),
        (30, 128),
        (10, 2560),
        (30, 2560),
        (60, 2560),
        (120, 2560),
    ] {
        let keys = synth_matrix_f32(n, d, 1);
        let targets = synth_matrix_f32(n, d, 2);
        let label = format!("N={n}_d={d}");
        group.bench_with_input(
            BenchmarkId::from_parameter(label),
            &(&keys, &targets),
            |b, (k, t)| {
                b.iter(|| {
                    qual_slowdown::slowdown().run(|| ridge_decomposition_solve(k, t, 1e-3).unwrap())
                });
            },
        );
    }
    group.finish();
}

criterion_group!(
    benches,
    bench_cholesky,
    bench_cholesky_solve,
    bench_ridge_decomposition
);
criterion_main!(benches);
