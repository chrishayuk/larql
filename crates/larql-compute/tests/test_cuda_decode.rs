//! CUDA DecodeBackend smoke tests (`cuda-decode-backend`).
//!
//! These are intentionally small and env-gated. The first CUDA decode backend
//! is correctness-first and host-visible, so these tests prove the trait
//! contract is live before performance work starts.

#![cfg(feature = "cuda")]

use larql_compute::backend::{Capability, ComputeBackend, DecodeBackend};
use larql_compute::cuda::CudaBackend;
use larql_compute::{Activation, FfnType, FullPipelineLayer, NormType, QuantFormat, QuantWeight};

fn gpu_or_skip() -> Option<CudaBackend> {
    if std::env::var("LARQL_CUDA_AVAILABLE").ok().as_deref() != Some("1") {
        eprintln!("skipping CUDA decode test: set LARQL_CUDA_AVAILABLE=1 to run");
        return None;
    }
    CudaBackend::new().ok()
}

fn synth(n: usize, seed: u64) -> Vec<f32> {
    let mut s = seed.wrapping_mul(0x9E37_79B9_7F4A_7C15);
    (0..n)
        .map(|_| {
            s ^= s << 13;
            s ^= s >> 7;
            s ^= s << 17;
            ((s & 0xFF_FFFF) as f32 / 8_388_608.0) - 0.5
        })
        .collect()
}

fn f32_bytes(vals: &[f32]) -> Vec<u8> {
    vals.iter().flat_map(|v| v.to_le_bytes()).collect()
}

fn qw(bytes: &[u8]) -> QuantWeight<'_> {
    QuantWeight {
        data: bytes,
        scales: None,
        format: QuantFormat::F32,
    }
}

fn f32_vals(bytes: &[u8]) -> Vec<f32> {
    bytes
        .chunks_exact(4)
        .map(|chunk| f32::from_le_bytes(chunk.try_into().unwrap()))
        .collect()
}

fn rms_norm(x: &[f32], weight: &[f32], eps: f32) -> Vec<f32> {
    let mean_sq = x.iter().map(|v| v * v).sum::<f32>() / x.len() as f32;
    let inv = 1.0 / (mean_sq + eps).sqrt();
    x.iter().zip(weight).map(|(v, w)| v * inv * w).collect()
}

fn matvec_rows(w: &[f32], x: &[f32], rows: usize, cols: usize) -> Vec<f32> {
    (0..rows)
        .map(|r| {
            let row = &w[r * cols..(r + 1) * cols];
            row.iter().zip(x).map(|(a, b)| a * b).sum()
        })
        .collect()
}

fn scalar_first_token_reference(
    x: &[f32],
    input_norm: &[f32],
    post_attn_norm: &[f32],
    pre_ffn_norm: &[f32],
    post_ffn_norm: &[f32],
    wv: &[u8],
    wo: &[u8],
    gate: &[u8],
    up: &[u8],
    down: &[u8],
    hidden: usize,
    inter: usize,
    q_dim: usize,
    kv_dim: usize,
    eps: f32,
) -> Vec<f32> {
    let x_norm = rms_norm(x, input_norm, eps);

    // For the first decode position there is only one key in the cache, so
    // softmax(QK^T) is exactly 1.0 and the attention output is V.
    let v = matvec_rows(&f32_vals(wv), &x_norm, kv_dim, hidden);
    let attn_delta = matvec_rows(&f32_vals(wo), &v, hidden, q_dim);
    let attn_normed = rms_norm(&attn_delta, post_attn_norm, eps);
    let mut h_post_attn = x.to_vec();
    for (h, a) in h_post_attn.iter_mut().zip(attn_normed) {
        *h += a;
    }

    let h_ffn = rms_norm(&h_post_attn, pre_ffn_norm, eps);
    let gate = matvec_rows(&f32_vals(gate), &h_ffn, inter, hidden);
    let up = matvec_rows(&f32_vals(up), &h_ffn, inter, hidden);
    let act: Vec<f32> = gate
        .iter()
        .zip(up)
        .map(|(g, u)| (g / (1.0 + (-g).exp())) * u)
        .collect();
    let ffn_delta = matvec_rows(&f32_vals(down), &act, hidden, inter);
    let ffn_normed = rms_norm(&ffn_delta, post_ffn_norm, eps);
    let mut h_out = h_post_attn;
    for (h, f) in h_out.iter_mut().zip(ffn_normed) {
        *h += f;
    }
    h_out
}

fn assert_close(got: &[f32], expected: &[f32], tol: f32) {
    assert_eq!(got.len(), expected.len());
    for (idx, (g, e)) in got.iter().zip(expected).enumerate() {
        let diff = (g - e).abs();
        assert!(
            diff <= tol,
            "mismatch at {idx}: got {g}, expected {e}, diff {diff}, tol {tol}"
        );
    }
}

#[test]
fn decode_token_one_layer_matches_scalar_reference() {
    let Some(backend) = gpu_or_skip() else { return };
    let hidden = 16;
    let inter = 32;
    let head_dim = 16;
    let num_q_heads = 1;
    let num_kv_heads = 1;
    let q_dim = num_q_heads * head_dim;
    let kv_dim = num_kv_heads * head_dim;

    let input_norm = vec![1.0; hidden];
    let post_attn_norm = vec![1.0; hidden];
    let pre_ffn_norm = vec![1.0; hidden];
    let post_ffn_norm = vec![1.0; hidden];
    let wq = f32_bytes(&synth(q_dim * hidden, 0x10));
    let wk = f32_bytes(&synth(kv_dim * hidden, 0x11));
    let wv = f32_bytes(&synth(kv_dim * hidden, 0x12));
    let wo = f32_bytes(&synth(hidden * q_dim, 0x13));
    let gate = f32_bytes(&synth(inter * hidden, 0x14));
    let up = f32_bytes(&synth(inter * hidden, 0x15));
    let down = f32_bytes(&synth(hidden * inter, 0x16));

    let layer = FullPipelineLayer {
        wq: qw(&wq),
        wk: qw(&wk),
        wv: qw(&wv),
        wo: qw(&wo),
        gate: qw(&gate),
        up: qw(&up),
        down: qw(&down),
        input_norm: &input_norm,
        post_attn_norm: &post_attn_norm,
        pre_ffn_norm: Some(&pre_ffn_norm),
        post_ffn_norm: Some(&post_ffn_norm),
        norm_offset: 0.0,
        eps: 1e-6,
        has_post_norms: true,
        norm_type: NormType::RmsNorm,
        ffn_type: FfnType::Gated,
        activation: Activation::Silu,
        attn_scale: 1.0 / (head_dim as f32).sqrt(),
        head_dim,
        num_q_heads,
        num_kv_heads,
        rope_base: 10_000.0,
        rotary_dim: head_dim,
        ..FullPipelineLayer::default()
    };
    let x = synth(hidden, 0x20);
    let out = backend
        .decode_token(
            &[layer],
            &x,
            hidden,
            inter,
            q_dim,
            kv_dim,
            num_q_heads,
            num_kv_heads,
            head_dim,
            10_000.0,
        )
        .expect("cuda decode_token should return Some");
    let expected = scalar_first_token_reference(
        &x,
        &input_norm,
        &post_attn_norm,
        &pre_ffn_norm,
        &post_ffn_norm,
        &wv,
        &wo,
        &gate,
        &up,
        &down,
        hidden,
        inter,
        q_dim,
        kv_dim,
        1e-6,
    );
    assert_close(&out, &expected, 1e-3);
    assert_eq!(backend.kv_cache_len(), 1);
}

#[test]
fn prefill_populates_kv_cache_len() {
    let Some(backend) = gpu_or_skip() else { return };
    assert!(backend.supports(Capability::DecodeToken));
    assert!(backend.supports(Capability::PrefillQ4));

    let hidden = 8;
    let inter = 16;
    let head_dim = 8;
    let num_q_heads = 1;
    let num_kv_heads = 1;
    let q_dim = head_dim;
    let kv_dim = head_dim;
    let input_norm = vec![1.0; hidden];
    let post_attn_norm = vec![1.0; hidden];
    let wq = f32_bytes(&synth(q_dim * hidden, 0x30));
    let wk = f32_bytes(&synth(kv_dim * hidden, 0x31));
    let wv = f32_bytes(&synth(kv_dim * hidden, 0x32));
    let wo = f32_bytes(&synth(hidden * q_dim, 0x33));
    let gate = f32_bytes(&synth(inter * hidden, 0x34));
    let up = f32_bytes(&synth(inter * hidden, 0x35));
    let down = f32_bytes(&synth(hidden * inter, 0x36));
    let layer = FullPipelineLayer {
        wq: qw(&wq),
        wk: qw(&wk),
        wv: qw(&wv),
        wo: qw(&wo),
        gate: qw(&gate),
        up: qw(&up),
        down: qw(&down),
        input_norm: &input_norm,
        post_attn_norm: &post_attn_norm,
        norm_type: NormType::RmsNorm,
        ffn_type: FfnType::Gated,
        activation: Activation::Silu,
        attn_scale: 1.0 / (head_dim as f32).sqrt(),
        head_dim,
        num_q_heads,
        num_kv_heads,
        rotary_dim: head_dim,
        ..FullPipelineLayer::default()
    };
    let seq_len = 3;
    let x = synth(seq_len * hidden, 0x40);
    let out = backend
        .prefill_q4(
            &[layer],
            &x,
            hidden,
            inter,
            q_dim,
            kv_dim,
            seq_len,
            num_q_heads,
            num_kv_heads,
            head_dim,
            10_000.0,
            false,
            0.0,
        )
        .expect("cuda prefill_q4 should return Some");
    assert_eq!(out.len(), seq_len * hidden);
    assert_eq!(backend.kv_cache_len(), seq_len);
}
