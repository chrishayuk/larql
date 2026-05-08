#![cfg(feature = "cuda-oxide")]

use larql_rotorquant::{cuda_oxide, dequantize_k, quantize_k, KvFormat};

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
}
