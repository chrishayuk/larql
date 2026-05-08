#[cfg(feature = "cuda")]
use std::ffi::c_void;
#[cfg(feature = "cuda")]
use std::time::{Duration, Instant};

#[cfg(feature = "cuda")]
use cudarc::driver::{CudaContext, DevicePtr, DevicePtrMut};
#[cfg(feature = "cuda")]
use half::f16;
#[cfg(feature = "cuda")]
use larql_rotorquant::{
    copy_f16_to_quantized_device, dequantize_to_f32_device, quantized_device_len_bytes, CudaStream,
    KvFormat,
};

#[cfg(not(feature = "cuda"))]
fn main() {
    eprintln!("skipping CUDA RotorQuant benchmark: build with --features cuda");
}

#[cfg(feature = "cuda")]
fn main() {
    if std::env::var("LARQL_CUDA_AVAILABLE").as_deref() != Ok("1") {
        eprintln!("skipping CUDA RotorQuant benchmark: set LARQL_CUDA_AVAILABLE=1");
        return;
    }

    let format = env_format();
    let n_rows = env_usize("LARQL_CUDA_ROTORQUANT_BENCH_ROWS", 16_384);
    let head_dim = env_usize("LARQL_CUDA_ROTORQUANT_BENCH_HEAD_DIM", 320);
    let warmup = env_usize("LARQL_CUDA_ROTORQUANT_BENCH_WARMUP", 3);
    let iters = env_usize("LARQL_CUDA_ROTORQUANT_BENCH_ITERS", 20);
    let elements = n_rows * head_dim;
    assert!(
        elements.is_multiple_of(128),
        "benchmark elements must be divisible by 128"
    );

    let ctx = CudaContext::new(0).expect("cuda context");
    let stream = ctx.default_stream();
    let input = synth(elements, 0x51A7_CADA);
    let input_f16: Vec<u16> = input.iter().map(|v| f16::from_f32(*v).to_bits()).collect();
    let packed_bytes = quantized_device_len_bytes(format, elements).expect("packed bytes");
    let src_dev = stream.clone_htod(&input_f16).expect("copy input");
    let mut packed_dev = stream
        .alloc_zeros::<u8>(packed_bytes)
        .expect("alloc packed");
    let mut out_dev = stream.alloc_zeros::<f32>(elements).expect("alloc output");

    for _ in 0..warmup {
        launch_quantize_dequantize(
            format,
            elements,
            &stream,
            &src_dev,
            &mut packed_dev,
            &mut out_dev,
        );
    }
    stream.synchronize().expect("warmup sync");

    let start = Instant::now();
    for _ in 0..iters {
        launch_quantize_dequantize(
            format,
            elements,
            &stream,
            &src_dev,
            &mut packed_dev,
            &mut out_dev,
        );
    }
    stream.synchronize().expect("bench sync");
    let elapsed = start.elapsed();

    let output = stream.clone_dtoh(&out_dev).expect("copy output");
    let cos = cosine(&input, &output);
    let fp16_bytes = elements * std::mem::size_of::<u16>();
    println!("format {format:?}");
    println!("rows {n_rows}");
    println!("head_dim {head_dim}");
    println!("elements {elements}");
    println!("iters {iters}");
    println!("fp16_kv_bytes {fp16_bytes}");
    println!("rotorquant_kv_bytes {packed_bytes}");
    println!(
        "memory_ratio {:.4}",
        packed_bytes as f64 / fp16_bytes as f64
    );
    println!("round_trip_cosine {cos:.6}");
    println!("seconds_total {:.6}", seconds(elapsed));
    println!(
        "ms_per_iter {:.4}",
        seconds(elapsed) * 1000.0 / iters as f64
    );
    println!(
        "logical_fp16_gib_per_sec {:.4}",
        gib_per_sec(fp16_bytes, iters, elapsed)
    );
}

#[cfg(feature = "cuda")]
fn launch_quantize_dequantize(
    format: KvFormat,
    elements: usize,
    stream: &std::sync::Arc<cudarc::driver::CudaStream>,
    src_dev: &cudarc::driver::CudaSlice<u16>,
    packed_dev: &mut cudarc::driver::CudaSlice<u8>,
    out_dev: &mut cudarc::driver::CudaSlice<f32>,
) {
    {
        let (src_ptr, _src_guard) = src_dev.device_ptr(stream);
        let (dst_ptr, _dst_guard) = packed_dev.device_ptr_mut(stream);
        let raw_stream = unsafe { CudaStream::from_raw(stream.cu_stream() as *mut c_void) };
        unsafe {
            copy_f16_to_quantized_device(
                format,
                src_ptr as usize as *const c_void,
                dst_ptr as usize as *mut c_void,
                elements,
                raw_stream,
            )
            .expect("copy_f16_to_quantized_device");
        }
    }
    {
        let (src_ptr, _src_guard) = packed_dev.device_ptr(stream);
        let (dst_ptr, _dst_guard) = out_dev.device_ptr_mut(stream);
        let raw_stream = unsafe { CudaStream::from_raw(stream.cu_stream() as *mut c_void) };
        unsafe {
            dequantize_to_f32_device(
                format,
                src_ptr as usize as *const c_void,
                dst_ptr as usize as *mut c_void,
                elements,
                raw_stream,
            )
            .expect("dequantize_to_f32_device");
        }
    }
}

#[cfg(feature = "cuda")]
fn env_format() -> KvFormat {
    match std::env::var("LARQL_CUDA_ROTORQUANT_BENCH_FORMAT")
        .unwrap_or_else(|_| "iso3".to_string())
        .to_ascii_lowercase()
        .as_str()
    {
        "planar3" => KvFormat::Planar3,
        "planar4" => KvFormat::Planar4,
        "iso3" => KvFormat::Iso3,
        "iso4" => KvFormat::Iso4,
        other => panic!("unknown RotorQuant format {other:?}"),
    }
}

#[cfg(feature = "cuda")]
fn env_usize(name: &str, default: usize) -> usize {
    std::env::var(name)
        .ok()
        .and_then(|value| value.parse().ok())
        .unwrap_or(default)
}

#[cfg(feature = "cuda")]
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

#[cfg(feature = "cuda")]
fn cosine(a: &[f32], b: &[f32]) -> f32 {
    let dot: f32 = a.iter().zip(b).map(|(x, y)| x * y).sum();
    let na: f32 = a.iter().map(|v| v * v).sum::<f32>().sqrt();
    let nb: f32 = b.iter().map(|v| v * v).sum::<f32>().sqrt();
    dot / (na * nb + 1e-12)
}

#[cfg(feature = "cuda")]
fn seconds(duration: Duration) -> f64 {
    duration.as_secs_f64()
}

#[cfg(feature = "cuda")]
fn gib_per_sec(bytes_per_iter: usize, iters: usize, duration: Duration) -> f64 {
    let total_gib = (bytes_per_iter * iters) as f64 / 1024.0_f64.powi(3);
    total_gib / seconds(duration)
}
