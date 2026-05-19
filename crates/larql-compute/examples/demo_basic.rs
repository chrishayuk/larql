//! Demo: auto-detect backend and run basic operations.
//!
//! Usage:
//!   cargo run --release -p larql-compute --example demo_basic
//!   cargo run --release -p larql-compute --features metal --example demo_basic

extern crate blas_src;

use larql_compute::{cpu_backend, default_backend};
use ndarray::Array2;

fn synth_matrix(rows: usize, cols: usize, seed: u64) -> Array2<f32> {
    let mut state = seed;
    Array2::from_shape_fn((rows, cols), |_| {
        state = state.wrapping_mul(6364136223846793005).wrapping_add(1);
        ((state >> 33) as f32) / (u32::MAX as f32) * 2.0 - 1.0
    })
}

fn main() {
    println!("=== larql-compute demo ===\n");

    let default = default_backend();
    let cpu = cpu_backend();

    println!("Default backend: {}", default.name());
    println!("  Device: {}", default.device_info());
    println!(
        "  Q4 support: {}",
        default.supports_quant(::larql_compute::QuantFormat::Q4_K)
    );
    println!();
    println!("CPU backend: {}", cpu.name());
    println!(
        "  Q4 support: {}",
        cpu.supports_quant(::larql_compute::QuantFormat::Q4_K)
    );
    println!();

    // f32 matmul
    let a = synth_matrix(6, 2560, 42);
    let b = synth_matrix(10240, 2560, 43);

    let t0 = std::time::Instant::now();
    let result_cpu = cpu.matmul_transb(a.view(), b.view());
    let cpu_ms = t0.elapsed().as_secs_f64() * 1000.0;

    let t0 = std::time::Instant::now();
    let result_default = default.matmul_transb(a.view(), b.view());
    let default_ms = t0.elapsed().as_secs_f64() * 1000.0;

    let diff: f32 = result_cpu
        .iter()
        .zip(result_default.iter())
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f32, f32::max);

    println!("matmul_transb [6,2560] x [10240,2560]^T:");
    println!("  CPU:     {cpu_ms:.2}ms");
    println!("  Default: {default_ms:.2}ms");
    println!("  Max diff: {diff:.2e}");

    println!("\n=== Done ===");
}
