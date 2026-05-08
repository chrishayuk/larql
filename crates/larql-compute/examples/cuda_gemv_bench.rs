use larql_compute::prelude::*;
use ndarray::Array2;
use std::time::Instant;

#[cfg(all(feature = "cuda", target_os = "linux"))]
fn main() {
    let rows: usize = std::env::var("LARQL_CUDA_BENCH_ROWS")
        .ok()
        .and_then(|raw| raw.parse().ok())
        .unwrap_or(4096);
    let cols: usize = std::env::var("LARQL_CUDA_BENCH_COLS")
        .ok()
        .and_then(|raw| raw.parse().ok())
        .unwrap_or(4096);
    let iters: usize = std::env::var("LARQL_CUDA_BENCH_ITERS")
        .ok()
        .and_then(|raw| raw.parse().ok())
        .unwrap_or(5);

    let w = Array2::from_shape_fn((rows, cols), |(row, col)| {
        ((row * 17 + col * 31) % 251) as f32 / 125.0 - 1.0
    });
    let x: Vec<f32> = (0..cols)
        .map(|idx| ((idx * 13 + 7) % 127) as f32 / 63.0 - 1.0)
        .collect();

    let cpu = larql_compute::CpuBackend;
    let Some(cuda) = larql_compute::CudaBackend::new() else {
        eprintln!("CUDA backend unavailable");
        std::process::exit(2);
    };

    // Warm both paths before timing.
    let cpu_input = Array2::from_shape_vec((1, cols), x.clone()).expect("cpu input");
    let cpu_ref = cpu
        .matmul_transb(cpu_input.view(), w.view())
        .row(0)
        .to_vec();
    let cuda_ref = cuda
        .f32_gemv_force(w.view(), &x)
        .expect("CUDA f32_gemv_force warmup");
    assert_close(&cuda_ref, &cpu_ref, 1e-2);
    let resident = cuda
        .resident_f32_matrix(w.view())
        .expect("CUDA resident matrix warmup");
    let resident_ref = resident.gemv(&cuda, &x).expect("resident CUDA warmup");
    assert_close(&resident_ref, &cpu_ref, 1e-2);

    let cpu_start = Instant::now();
    let mut cpu_last = Vec::new();
    for _ in 0..iters {
        cpu_last = cpu
            .matmul_transb(cpu_input.view(), w.view())
            .row(0)
            .to_vec();
    }
    let cpu_ms = cpu_start.elapsed().as_secs_f64() * 1000.0 / iters as f64;

    let cuda_start = Instant::now();
    let mut cuda_last = Vec::new();
    for _ in 0..iters {
        cuda_last = cuda
            .f32_gemv_force(w.view(), &x)
            .expect("CUDA f32_gemv_force");
    }
    let cuda_ms = cuda_start.elapsed().as_secs_f64() * 1000.0 / iters as f64;

    let resident_start = Instant::now();
    let mut resident_last = Vec::new();
    for _ in 0..iters {
        resident_last = resident.gemv(&cuda, &x).expect("resident CUDA gemv");
    }
    let resident_ms = resident_start.elapsed().as_secs_f64() * 1000.0 / iters as f64;

    assert_close(&cuda_last, &cpu_last, 1e-2);
    assert_close(&resident_last, &cpu_last, 1e-2);
    let max_abs = max_abs_diff(&cuda_last, &cpu_last);
    let resident_max_abs = max_abs_diff(&resident_last, &cpu_last);
    println!(
        "backend={} device={} rows={} cols={} iters={} cpu_ms_per_iter={:.3} cuda_ms_per_iter={:.3} resident_cuda_ms_per_iter={:.3} cuda_speedup={:.3} resident_cuda_speedup={:.3} max_abs_diff={:.6} resident_max_abs_diff={:.6}",
        cuda.name(),
        cuda.device_info(),
        rows,
        cols,
        iters,
        cpu_ms,
        cuda_ms,
        resident_ms,
        cpu_ms / cuda_ms,
        cpu_ms / resident_ms,
        max_abs,
        resident_max_abs,
    );
}

#[cfg(not(all(feature = "cuda", target_os = "linux")))]
fn main() {
    eprintln!("cuda_gemv_bench requires Linux with --features cuda");
    std::process::exit(2);
}

#[cfg(all(feature = "cuda", target_os = "linux"))]
fn assert_close(got: &[f32], want: &[f32], tol: f32) {
    assert_eq!(got.len(), want.len());
    let max_abs = max_abs_diff(got, want);
    assert!(
        max_abs <= tol,
        "max_abs_diff {max_abs} exceeded tolerance {tol}"
    );
}

#[cfg(all(feature = "cuda", target_os = "linux"))]
fn max_abs_diff(a: &[f32], b: &[f32]) -> f32 {
    a.iter()
        .zip(b)
        .map(|(lhs, rhs)| (lhs - rhs).abs())
        .fold(0.0, f32::max)
}
