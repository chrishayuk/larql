#![cfg(feature = "cuda-oxide")]

use larql_rotorquant::{cuda_oxide, dequantize_k, quantize_k, KvFormat};
use std::path::PathBuf;

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
