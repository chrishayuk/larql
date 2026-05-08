#[cfg(feature = "cuda-oxide")]
use std::time::{Duration, Instant};

#[cfg(feature = "cuda-oxide")]
use larql_rotorquant::{cuda_oxide, dequantize_k, quantize_k, KvFormat};

#[cfg(not(feature = "cuda-oxide"))]
fn main() {
    eprintln!("skipping cuda-oxide benchmark: build with --features cuda-oxide");
}

#[cfg(feature = "cuda-oxide")]
fn main() {
    if std::env::var("LARQL_CUDA_AVAILABLE").as_deref() != Ok("1") {
        eprintln!("skipping cuda-oxide benchmark: set LARQL_CUDA_AVAILABLE=1");
        return;
    }

    let n_rows = env_usize("LARQL_CUDA_OXIDE_BENCH_ROWS", 16_384);
    let head_dim = env_usize("LARQL_CUDA_OXIDE_BENCH_HEAD_DIM", 320);
    let warmup = env_usize("LARQL_CUDA_OXIDE_BENCH_WARMUP", 3);
    let iters = env_usize("LARQL_CUDA_OXIDE_BENCH_ITERS", 20);

    let input = synth(n_rows * head_dim, 0xC0DA_BEEF);
    let qkv = quantize_k(KvFormat::Iso3, &input, n_rows, head_dim).expect("quantize_k");
    let cpu = dequantize_k(&qkv).expect("dequantize_k");
    let ctx = cuda_oxide::CudaContext::new(0).expect("cuda context");

    for _ in 0..warmup {
        let _ = cuda_oxide::dequantize_iso3(&ctx, &qkv).expect("warmup cuda-oxide dequantize");
    }

    let gpu_start = Instant::now();
    let mut gpu = Vec::new();
    for _ in 0..iters {
        gpu = cuda_oxide::dequantize_iso3(&ctx, &qkv).expect("cuda-oxide dequantize");
    }
    let gpu_elapsed = gpu_start.elapsed();

    let cpu_start = Instant::now();
    let mut cpu_last = Vec::new();
    for _ in 0..iters {
        cpu_last = dequantize_k(&qkv).expect("dequantize_k");
    }
    let cpu_elapsed = cpu_start.elapsed();

    let max_diff = gpu
        .iter()
        .zip(&cpu)
        .map(|(a, b)| (a - b).abs())
        .fold(0.0_f32, f32::max);
    assert_eq!(cpu_last.len(), gpu.len());

    let elements = n_rows * head_dim;
    let bytes = elements * std::mem::size_of::<f32>();
    println!("rows {n_rows}");
    println!("head_dim {head_dim}");
    println!("elements {elements}");
    println!("iters {iters}");
    println!("max_diff {max_diff:.8}");
    println!("cpu_seconds_total {:.6}", duration_seconds(cpu_elapsed));
    println!("gpu_seconds_total {:.6}", duration_seconds(gpu_elapsed));
    println!(
        "cpu_ms_per_iter {:.4}",
        duration_seconds(cpu_elapsed) * 1000.0 / iters as f64
    );
    println!(
        "gpu_ms_per_iter {:.4}",
        duration_seconds(gpu_elapsed) * 1000.0 / iters as f64
    );
    println!(
        "cpu_gib_per_sec {:.4}",
        gib_per_sec(bytes, iters, cpu_elapsed)
    );
    println!(
        "gpu_gib_per_sec {:.4}",
        gib_per_sec(bytes, iters, gpu_elapsed)
    );
}

#[cfg(feature = "cuda-oxide")]
fn env_usize(name: &str, default: usize) -> usize {
    std::env::var(name)
        .ok()
        .and_then(|value| value.parse().ok())
        .unwrap_or(default)
}

#[cfg(feature = "cuda-oxide")]
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

#[cfg(feature = "cuda-oxide")]
fn duration_seconds(duration: Duration) -> f64 {
    duration.as_secs_f64()
}

#[cfg(feature = "cuda-oxide")]
fn gib_per_sec(bytes_per_iter: usize, iters: usize, duration: Duration) -> f64 {
    let total_gib = (bytes_per_iter * iters) as f64 / 1024.0_f64.powi(3);
    total_gib / duration_seconds(duration)
}
