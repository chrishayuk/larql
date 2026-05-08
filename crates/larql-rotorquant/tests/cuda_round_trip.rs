#![cfg(feature = "cuda")]

use std::ffi::c_void;

use cudarc::driver::{CudaContext, DevicePtr, DevicePtrMut};
use half::f16;
use larql_rotorquant::{
    copy_f16_to_quantized_device, dequantize_to_f32_device, quantized_device_len_bytes, CudaStream,
    KvFormat,
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
