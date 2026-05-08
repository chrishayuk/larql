#![cfg(feature = "cuda-oxide")]

use larql_rotorquant::{
    cuda_oxide, dequantize_k, dequantize_v_with_inverse_rotation, quantize_k, quantize_v, KvFormat,
    QuantizedKv,
};
use std::path::PathBuf;

#[derive(Clone, Copy, Debug)]
enum KvKind {
    K,
    V,
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

fn read_f32le(path: PathBuf) -> Option<Vec<f32>> {
    let bytes = std::fs::read(path).ok()?;
    let mut values = Vec::with_capacity(bytes.len() / 4);
    for chunk in bytes.chunks_exact(4) {
        values.push(f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]));
    }
    Some(values)
}

fn parity_dir() -> Option<PathBuf> {
    std::env::var_os("LARQL_ROTORQUANT_PARITY_DIR").map(PathBuf::from)
}

fn format_seed(format: KvFormat) -> u64 {
    match format {
        KvFormat::Planar3 => 0xA11C_E003,
        KvFormat::Planar4 => 0xA11C_E004,
        KvFormat::Iso3 => 0x1500_0003,
        KvFormat::Iso4 => 0x1500_0004,
    }
}

fn quantize_kind(
    kind: KvKind,
    format: KvFormat,
    input: &[f32],
    n_rows: usize,
    head_dim: usize,
) -> QuantizedKv {
    match kind {
        KvKind::K => quantize_k(format, input, n_rows, head_dim).expect("quantize_k"),
        KvKind::V => quantize_v(format, input, n_rows, head_dim).expect("quantize_v"),
    }
}

fn dequantize_kind(kind: KvKind, qkv: &QuantizedKv) -> Vec<f32> {
    match kind {
        KvKind::K => dequantize_k(qkv).expect("dequantize_k"),
        KvKind::V => {
            dequantize_v_with_inverse_rotation(qkv).expect("dequantize_v_with_inverse_rotation")
        }
    }
}

fn assert_cuda_oxide_dequantizes_like_cpu(format: KvFormat, kind: KvKind) {
    if std::env::var("LARQL_CUDA_AVAILABLE").as_deref() != Ok("1") {
        eprintln!("skipping cuda-oxide round-trip: set LARQL_CUDA_AVAILABLE=1");
        return;
    }

    let n_rows = 64;
    let head_dim = 320;
    let kind_seed = match kind {
        KvKind::K => 0x0C0D_E001,
        KvKind::V => 0x0C0D_E002,
    };
    let input = synth(n_rows * head_dim, format_seed(format) ^ kind_seed);
    let qkv = quantize_kind(kind, format, &input, n_rows, head_dim);
    let cpu = dequantize_kind(kind, &qkv);

    let ctx = cuda_oxide::CudaContext::new(0).expect("cuda context");
    let gpu = cuda_oxide::dequantize(&ctx, &qkv).expect("cuda-oxide dequantize");

    assert_eq!(gpu.len(), cpu.len());
    let max_diff = max_abs_diff(&gpu, &cpu);
    assert!(
        max_diff <= 1e-3,
        "{format:?} {kind:?} max cpu-vs-cuda-oxide diff {max_diff}"
    );
}

#[test]
fn iso3_cuda_oxide_dequantize_matches_cpu() {
    if std::env::var("LARQL_CUDA_AVAILABLE").as_deref() != Ok("1") {
        eprintln!("skipping cuda-oxide round-trip: set LARQL_CUDA_AVAILABLE=1");
        return;
    }

    let n_rows = 64;
    let head_dim = 320;
    let input = synth(n_rows * head_dim, 0xC0DA);
    let qkv = quantize_k(KvFormat::Iso3, &input, n_rows, head_dim).expect("quantize_k");
    let cpu = dequantize_k(&qkv).expect("dequantize_k");

    let ctx = cuda_oxide::CudaContext::new(0).expect("cuda context");
    let gpu = cuda_oxide::dequantize_iso3(&ctx, &qkv).expect("cuda-oxide dequantize");

    assert_eq!(gpu.len(), cpu.len());
    let max_diff = gpu
        .iter()
        .zip(&cpu)
        .map(|(a, b)| (a - b).abs())
        .fold(0.0_f32, f32::max);
    assert!(max_diff <= 1e-3, "max diff {max_diff}");

    for row in 0..n_rows {
        let start = row * head_dim;
        let end = start + head_dim;
        let cos = cosine(&input[start..end], &gpu[start..end]);
        assert!(cos >= 0.99, "row {row} cosine {cos}");
    }

    if let Some(dir) = parity_dir() {
        if let Some(cudarc) = read_f32le(dir.join("cudarc_iso3_dequant.f32le")) {
            assert_eq!(cudarc.len(), gpu.len());
            let diff = max_abs_diff(&cudarc, &gpu);
            assert!(diff <= 1e-3, "max cudarc-vs-cuda-oxide diff {diff}");
        } else {
            eprintln!("skipping cudarc-vs-cuda-oxide fixture comparison: no cudarc fixture found");
        }
    }
}

#[test]
fn cuda_oxide_dequantize_matches_cpu_for_every_format_and_kind() {
    for format in [
        KvFormat::Planar3,
        KvFormat::Planar4,
        KvFormat::Iso3,
        KvFormat::Iso4,
    ] {
        for kind in [KvKind::K, KvKind::V] {
            assert_cuda_oxide_dequantizes_like_cpu(format, kind);
        }
    }
}
