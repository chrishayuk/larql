//! CUDA fused-attention parity tests (`cuda-fused-attention`).
//!
//! Same env-gated pattern: `LARQL_CUDA_AVAILABLE=1` to run for real;
//! tests no-op cleanly otherwise. Reference is a naive scalar
//! implementation — no BLAS — to keep the oracle obvious.

#![cfg(feature = "cuda")]

use larql_compute::cuda::attn::{decode_attention, AttentionOpts};
use larql_compute::cuda::CudaBackend;

const TOL_ABS: f32 = 1e-3;
const TOL_COS: f32 = 0.9999;

fn gpu_or_skip() -> Option<()> {
    if std::env::var("LARQL_CUDA_AVAILABLE").ok().as_deref() != Some("1") {
        return None;
    }
    // Touch the backend to validate the driver / cublas / nvrtc init
    // path before each test. Init is cheap on warm host.
    CudaBackend::new().ok()?;
    Some(())
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

/// Naive scalar single-head attention reference. Computes
/// `softmax((Q @ K^T) * scale) @ V` row by row.
fn reference_attention(
    q: &[f32],
    k: &[f32],
    v: &[f32],
    n_q: usize,
    n_kv: usize,
    head_dim: usize,
    causal: bool,
    softcap: f32,
) -> Vec<f32> {
    let scale = 1.0_f32 / (head_dim as f32).sqrt();
    let mut out = vec![0.0_f32; n_q * head_dim];
    for i in 0..n_q {
        // logits[j] = q[i] · k[j] * scale (with optional causal/softcap)
        let mut logits = vec![0.0_f32; n_kv];
        let mut max = f32::NEG_INFINITY;
        for j in 0..n_kv {
            let mut s = 0.0_f32;
            for d in 0..head_dim {
                s += q[i * head_dim + d] * k[j * head_dim + d];
            }
            s *= scale;
            if softcap > 0.0 {
                s = softcap * (s / softcap).tanh();
            }
            if causal && j > i {
                s = f32::NEG_INFINITY;
            }
            logits[j] = s;
            if s > max {
                max = s;
            }
        }
        let mut sum = 0.0_f32;
        for l in &mut logits {
            *l = (*l - max).exp();
            sum += *l;
        }
        let inv = 1.0 / sum;
        for l in &mut logits {
            *l *= inv;
        }
        for d in 0..head_dim {
            let mut s = 0.0_f32;
            for j in 0..n_kv {
                s += logits[j] * v[j * head_dim + d];
            }
            out[i * head_dim + d] = s;
        }
    }
    out
}

#[test]
fn softmax_small_parity() {
    let Some(()) = gpu_or_skip() else { return };
    // 4 rows × 16 cols, no mask, scale = 1.0
    let n_q = 4;
    let n_kv = 16;
    let head_dim = 1;
    // Use decode_attention as a black-box softmax tester:
    //   set head_dim=1 and v=identity-ish so the result rows are the
    //   softmax rows multiplied by the v-vector.
    let q = synth(n_q * head_dim, 0xA1);
    let k = synth(n_kv * head_dim, 0xA2);
    let v = vec![1.0_f32; n_kv * head_dim];
    let opts = AttentionOpts {
        causal: false,
        softcap: 0.0,
    };
    let backend = CudaBackend::new().unwrap();
    // Driver lives inside the backend — we re-create it via the
    // public new() and access via a trait-bypass helper. The crate
    // doesn't expose drv, so we exercise the public `decode_attention`
    // directly.
    let cuda = decode_attention(
        &backend,
        &q,
        &k,
        &v,
        n_q,
        n_kv,
        head_dim,
        opts,
    )
    .expect("decode_attention");
    let cpu = reference_attention(&q, &k, &v, n_q, n_kv, head_dim, false, 0.0);
    let diff = max_abs_diff(&cpu, &cuda);
    let cos = cosine(&cpu, &cuda);
    assert!(diff <= TOL_ABS, "max abs diff {diff}");
    assert!(cos >= TOL_COS, "cosine {cos}");
}

#[test]
fn softmax_long_row_parity() {
    let Some(()) = gpu_or_skip() else { return };
    let n_q = 4;
    let n_kv = 4096;
    let head_dim = 1;
    let q = synth(n_q * head_dim, 0xB1);
    let k = synth(n_kv * head_dim, 0xB2);
    let v = synth(n_kv * head_dim, 0xB3);
    let opts = AttentionOpts {
        causal: false,
        softcap: 0.0,
    };
    let backend = CudaBackend::new().unwrap();
    let cuda = decode_attention(
        &backend,
        &q,
        &k,
        &v,
        n_q,
        n_kv,
        head_dim,
        opts,
    )
    .expect("decode_attention");
    let cpu = reference_attention(&q, &k, &v, n_q, n_kv, head_dim, false, 0.0);
    let diff = max_abs_diff(&cpu, &cuda);
    assert!(diff <= TOL_ABS, "max abs diff {diff}");
}

#[test]
fn softmax_causal_mask() {
    let Some(()) = gpu_or_skip() else { return };
    let n_q = 4;
    let n_kv = 4;
    let head_dim = 1;
    let q = synth(n_q * head_dim, 0xC1);
    let k = synth(n_kv * head_dim, 0xC2);
    let v = synth(n_kv * head_dim, 0xC3);
    let opts = AttentionOpts {
        causal: true,
        softcap: 0.0,
    };
    let backend = CudaBackend::new().unwrap();
    let cuda = decode_attention(
        &backend,
        &q,
        &k,
        &v,
        n_q,
        n_kv,
        head_dim,
        opts,
    )
    .expect("decode_attention");
    let cpu = reference_attention(&q, &k, &v, n_q, n_kv, head_dim, true, 0.0);
    let diff = max_abs_diff(&cpu, &cuda);
    assert!(
        diff <= TOL_ABS,
        "causal-mask max abs diff {diff} (cpu={cpu:?} cuda={cuda:?})"
    );
}

#[test]
fn softmax_softcap_50() {
    let Some(()) = gpu_or_skip() else { return };
    let n_q = 8;
    let n_kv = 16;
    let head_dim = 1;
    // Inflate Q so the dot-products land outside [-1, 1] and softcap
    // actually clamps.
    let q: Vec<f32> = synth(n_q * head_dim, 0xD1).into_iter().map(|v| v * 50.0).collect();
    let k: Vec<f32> = synth(n_kv * head_dim, 0xD2).into_iter().map(|v| v * 50.0).collect();
    let v = synth(n_kv * head_dim, 0xD3);
    let opts = AttentionOpts {
        causal: false,
        softcap: 50.0,
    };
    let backend = CudaBackend::new().unwrap();
    let cuda = decode_attention(
        &backend,
        &q,
        &k,
        &v,
        n_q,
        n_kv,
        head_dim,
        opts,
    )
    .expect("decode_attention");
    let cpu = reference_attention(&q, &k, &v, n_q, n_kv, head_dim, false, 50.0);
    let diff = max_abs_diff(&cpu, &cuda);
    assert!(diff <= TOL_ABS, "softcap-50 max abs diff {diff}");
}

#[test]
fn decode_attention_small_parity() {
    let Some(()) = gpu_or_skip() else { return };
    let n_q = 8;
    let n_kv = 8;
    let head_dim = 64;
    let q = synth(n_q * head_dim, 0xE1);
    let k = synth(n_kv * head_dim, 0xE2);
    let v = synth(n_kv * head_dim, 0xE3);
    let opts = AttentionOpts {
        causal: false,
        softcap: 0.0,
    };
    let backend = CudaBackend::new().unwrap();
    let cuda = decode_attention(
        &backend,
        &q,
        &k,
        &v,
        n_q,
        n_kv,
        head_dim,
        opts,
    )
    .expect("decode_attention");
    let cpu = reference_attention(&q, &k, &v, n_q, n_kv, head_dim, false, 0.0);
    let diff = max_abs_diff(&cpu, &cuda);
    let cos = cosine(&cpu, &cuda);
    assert!(diff <= TOL_ABS, "max abs diff {diff}");
    assert!(cos >= TOL_COS, "cosine {cos}");
}

#[test]
fn decode_attention_gemma4b_head_parity() {
    let Some(()) = gpu_or_skip() else { return };
    let n_q = 1;     // single decode token
    let n_kv = 2048; // mid-context
    let head_dim = 320; // Gemma 4B head_dim
    let q = synth(n_q * head_dim, 0xF1);
    let k = synth(n_kv * head_dim, 0xF2);
    let v = synth(n_kv * head_dim, 0xF3);
    let opts = AttentionOpts {
        causal: false,
        softcap: 0.0,
    };
    let backend = CudaBackend::new().unwrap();
    let cuda = decode_attention(
        &backend,
        &q,
        &k,
        &v,
        n_q,
        n_kv,
        head_dim,
        opts,
    )
    .expect("decode_attention");
    let cpu = reference_attention(&q, &k, &v, n_q, n_kv, head_dim, false, 0.0);
    let diff = max_abs_diff(&cpu, &cuda);
    let cos = cosine(&cpu, &cuda);
    assert!(diff <= TOL_ABS, "max abs diff {diff}");
    assert!(cos >= TOL_COS, "cosine {cos}");
}
