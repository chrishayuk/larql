#![cfg(feature = "cuda")]

use std::ffi::c_void;
use std::path::PathBuf;

use cudarc::driver::{CudaContext, DevicePtr, DevicePtrMut};
use half::f16;
use larql_rotorquant::{
    copy_f16_to_quantized_device, dequantize_iso3_quantized_device, dequantize_k,
    dequantize_to_f32_device, quantize_k, quantized_device_len_bytes, CudaStream, KvFormat,
};

fn gpu_or_skip() -> Option<std::sync::Arc<CudaContext>> {
    if std::env::var("LARQL_CUDA_AVAILABLE").as_deref() != Ok("1") {
        eprintln!("skipping CUDA RotorQuant test: set LARQL_CUDA_AVAILABLE=1");
        return None;
    }
    CudaContext::new(0).ok()
}

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

fn f32_bytes(values: &[f32]) -> Vec<u8> {
    let mut out = Vec::with_capacity(std::mem::size_of_val(values));
    for value in values {
        out.extend_from_slice(&value.to_le_bytes());
    }
    out
}

fn parity_dir() -> Option<PathBuf> {
    std::env::var_os("LARQL_ROTORQUANT_PARITY_DIR").map(PathBuf::from)
}

fn cuda_round_trip(format: KvFormat) -> Vec<f32> {
    let ctx = gpu_or_skip().expect("gpu_or_skip returned none");
    let stream = ctx.default_stream();
    let elements = 8 * 128;
    let input = synth(elements, 0xCADA_0000 ^ format as u64);
    let input_f16: Vec<u16> = input.iter().map(|v| f16::from_f32(*v).to_bits()).collect();
    let packed_bytes = quantized_device_len_bytes(format, elements).expect("packed bytes");

    let src_dev = stream.clone_htod(&input_f16).expect("copy input");
    let mut packed_dev = stream
        .alloc_zeros::<u8>(packed_bytes)
        .expect("alloc packed");
    let mut out_dev = stream.alloc_zeros::<f32>(elements).expect("alloc output");

    {
        let (src_ptr, _src_guard) = src_dev.device_ptr(&stream);
        let (dst_ptr, _dst_guard) = packed_dev.device_ptr_mut(&stream);
        let stream = unsafe { CudaStream::from_raw(stream.cu_stream() as *mut c_void) };
        unsafe {
            copy_f16_to_quantized_device(
                format,
                src_ptr as usize as *const c_void,
                dst_ptr as usize as *mut c_void,
                elements,
                stream,
            )
            .expect("copy_f16_to_quantized_device");
        }
    }

    {
        let (src_ptr, _src_guard) = packed_dev.device_ptr(&stream);
        let (dst_ptr, _dst_guard) = out_dev.device_ptr_mut(&stream);
        let stream = unsafe { CudaStream::from_raw(stream.cu_stream() as *mut c_void) };
        unsafe {
            dequantize_to_f32_device(
                format,
                src_ptr as usize as *const c_void,
                dst_ptr as usize as *mut c_void,
                elements,
                stream,
            )
            .expect("dequantize_to_f32_device");
        }
    }

    stream.synchronize().expect("cuda stream sync");
    let output = stream.clone_dtoh(&out_dev).expect("copy output");
    let cos = cosine(&input, &output);
    let min_cos = match format {
        KvFormat::Planar3 | KvFormat::Iso3 => 0.98,
        KvFormat::Planar4 | KvFormat::Iso4 => 0.99,
    };
    assert!(
        cos >= min_cos,
        "{format:?} CUDA RotorQuant round-trip cosine {cos}, expected >= {min_cos}"
    );
    output
}

fn cuda_iso3_quantized_dequant(n_rows: usize, head_dim: usize) -> (Vec<f32>, Vec<f32>, Vec<f32>) {
    let ctx = gpu_or_skip().expect("gpu_or_skip returned none");
    let stream = ctx.default_stream();
    let input = synth(n_rows * head_dim, 0xC0DA);
    let qkv = quantize_k(KvFormat::Iso3, &input, n_rows, head_dim).expect("quantize_k");
    let cpu = dequantize_k(&qkv).expect("dequantize_k");

    let codes_dev = stream.clone_htod(&qkv.codes).expect("copy codes");
    let norms_dev = stream.clone_htod(&qkv.norms).expect("copy norms");
    let rotations_dev = stream
        .clone_htod(&qkv.rotation_indices)
        .expect("copy rotations");
    let mut out_dev = stream
        .alloc_zeros::<f32>(n_rows * head_dim)
        .expect("alloc output");

    {
        let (codes_ptr, _codes_guard) = codes_dev.device_ptr(&stream);
        let (norms_ptr, _norms_guard) = norms_dev.device_ptr(&stream);
        let (rotations_ptr, _rotations_guard) = rotations_dev.device_ptr(&stream);
        let (dst_ptr, _dst_guard) = out_dev.device_ptr_mut(&stream);
        let stream = unsafe { CudaStream::from_raw(stream.cu_stream() as *mut c_void) };
        unsafe {
            dequantize_iso3_quantized_device(
                codes_ptr as usize as *const c_void,
                norms_ptr as usize as *const c_void,
                rotations_ptr as usize as *const c_void,
                dst_ptr as usize as *mut c_void,
                n_rows,
                head_dim,
                stream,
            )
            .expect("dequantize_iso3_quantized_device");
        }
    }

    stream.synchronize().expect("cuda stream sync");
    let gpu = stream.clone_dtoh(&out_dev).expect("copy output");
    (input, cpu, gpu)
}

#[test]
fn planar3_cuda_preserves_direction() {
    if gpu_or_skip().is_none() {
        return;
    }
    let _ = cuda_round_trip(KvFormat::Planar3);
}

#[test]
fn planar4_cuda_preserves_direction() {
    if gpu_or_skip().is_none() {
        return;
    }
    let _ = cuda_round_trip(KvFormat::Planar4);
}

#[test]
fn iso3_cuda_preserves_direction() {
    if gpu_or_skip().is_none() {
        return;
    }
    let _ = cuda_round_trip(KvFormat::Iso3);
}

#[test]
fn iso4_cuda_preserves_direction() {
    if gpu_or_skip().is_none() {
        return;
    }
    let _ = cuda_round_trip(KvFormat::Iso4);
}

#[test]
fn iso3_quantized_layout_dequantizes_like_cpu_reference() {
    if gpu_or_skip().is_none() {
        return;
    }

    let n_rows = 64;
    let head_dim = 320;
    let (input, cpu, gpu) = cuda_iso3_quantized_dequant(n_rows, head_dim);
    let diff = max_abs_diff(&cpu, &gpu);
    assert!(diff <= 1e-3, "max CPU-vs-cudarc diff {diff}");
    for row in 0..n_rows {
        let start = row * head_dim;
        let end = start + head_dim;
        let cos = cosine(&input[start..end], &gpu[start..end]);
        assert!(cos >= 0.99, "row {row} cosine {cos}");
    }

    if let Some(dir) = parity_dir() {
        std::fs::create_dir_all(&dir).expect("create parity dir");
        std::fs::write(dir.join("cudarc_iso3_dequant.f32le"), f32_bytes(&gpu))
            .expect("write cudarc parity fixture");
    }
}
