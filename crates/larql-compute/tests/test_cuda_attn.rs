//! CUDA fused-attention parity tests (`cuda-fused-attention`).
//!
//! Same env-gated pattern: `LARQL_CUDA_AVAILABLE=1` to run for real;
//! tests no-op cleanly otherwise. Reference is a naive scalar
//! implementation — no BLAS — to keep the oracle obvious.

#![cfg(feature = "cuda")]

use larql_compute::cuda::attn::{
    decode_attention, fused_decode_attention, qkv_rms_proj, AttentionOpts,
    FusedDecodeAttentionOpts, QkvProjDims,
};
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

fn rms_norm(x: &[f32], weight: &[f32], eps: f32, offset: f32) -> Vec<f32> {
    let mean_sq = x.iter().map(|v| v * v).sum::<f32>() / x.len() as f32;
    let inv = 1.0 / (mean_sq + eps).sqrt();
    x.iter()
        .zip(weight)
        .map(|(v, w)| v * inv * (w + offset))
        .collect()
}

fn matvec_rows(w: &[f32], x: &[f32], rows: usize, cols: usize) -> Vec<f32> {
    assert_eq!(w.len(), rows * cols);
    assert_eq!(x.len(), cols);
    let mut out = vec![0.0; rows];
    for r in 0..rows {
        let mut acc = 0.0;
        for c in 0..cols {
            acc += w[r * cols + c] * x[c];
        }
        out[r] = acc;
    }
    out
}

fn apply_qk_norm_rope(
    x: &[f32],
    weight: &[f32],
    eps: f32,
    offset: f32,
    pos: usize,
    rope_base: f32,
    rotary_dim: usize,
) -> Vec<f32> {
    let head_dim = x.len();
    let mean_sq = x.iter().map(|v| v * v).sum::<f32>() / head_dim as f32;
    let inv = 1.0 / (mean_sq + eps).sqrt();
    let mut out: Vec<f32> = x
        .iter()
        .zip(weight)
        .map(|(v, w)| v * inv * (w + offset))
        .collect();
    let rdim = if rotary_dim == 0 {
        head_dim
    } else {
        rotary_dim.min(head_dim)
    };
    let hdim = rdim / 2;
    for d in 0..hdim {
        let freq = 1.0 / rope_base.powf((2 * d) as f32 / rdim as f32);
        let angle = pos as f32 * freq;
        let (sin_a, cos_a) = angle.sin_cos();
        let re = out[d];
        let im = out[d + hdim];
        out[d] = re * cos_a - im * sin_a;
        out[d + hdim] = re * sin_a + im * cos_a;
    }
    out
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

#[allow(clippy::too_many_arguments)]
fn reference_fused_decode_attention(
    q: &[f32],
    k_new: &[f32],
    v_new: &[f32],
    k_cache: &[f32],
    v_cache: &[f32],
    q_norm: &[f32],
    k_norm: &[f32],
    opts: FusedDecodeAttentionOpts,
) -> (Vec<f32>, Vec<f32>, Vec<f32>) {
    let mut k_cache = k_cache.to_vec();
    let mut v_cache = v_cache.to_vec();
    let mut out = vec![0.0; opts.num_q_heads * opts.head_dim];
    let group = (opts.num_q_heads / opts.num_kv_heads).max(1);

    for kvh in 0..opts.num_kv_heads {
        let k_head = &k_new[kvh * opts.head_dim..(kvh + 1) * opts.head_dim];
        let k_rot = apply_qk_norm_rope(
            k_head,
            k_norm,
            opts.eps,
            opts.qk_norm_offset,
            opts.pos,
            opts.rope_base,
            opts.rotary_dim,
        );
        for d in 0..opts.head_dim {
            let idx = (opts.pos * opts.num_kv_heads + kvh) * opts.head_dim + d;
            k_cache[idx] = k_rot[d];
            v_cache[idx] = v_new[kvh * opts.head_dim + d];
        }
    }

    for qh in 0..opts.num_q_heads {
        let kvh = (qh / group).min(opts.num_kv_heads - 1);
        let q_head = &q[qh * opts.head_dim..(qh + 1) * opts.head_dim];
        let q_rot = apply_qk_norm_rope(
            q_head,
            q_norm,
            opts.eps,
            opts.qk_norm_offset,
            opts.pos,
            opts.rope_base,
            opts.rotary_dim,
        );

        let mut scores = vec![0.0; opts.pos + 1];
        let mut max = f32::NEG_INFINITY;
        for (j, score) in scores.iter_mut().enumerate() {
            let k_base = (j * opts.num_kv_heads + kvh) * opts.head_dim;
            let mut dot = 0.0;
            for d in 0..opts.head_dim {
                dot += q_rot[d] * k_cache[k_base + d];
            }
            let mut s = dot * opts.attn_scale;
            if opts.softcap > 0.0 {
                s = opts.softcap * (s / opts.softcap).tanh();
            }
            *score = s;
            max = max.max(s);
        }
        let mut sum = 0.0;
        for s in &mut scores {
            *s = (*s - max).exp();
            sum += *s;
        }
        for d in 0..opts.head_dim {
            let mut acc = 0.0;
            for (j, score) in scores.iter().enumerate() {
                let v_base = (j * opts.num_kv_heads + kvh) * opts.head_dim;
                acc += (*score / sum) * v_cache[v_base + d];
            }
            out[qh * opts.head_dim + d] = acc;
        }
    }

    (out, k_cache, v_cache)
}

#[test]
fn fused_qkv_rms_projection_matches_cpu_reference() {
    let Some(()) = gpu_or_skip() else { return };
    let hidden = 32;
    let q_dim = 16;
    let kv_dim = 8;
    let x = synth(hidden, 0x111);
    let norm_w: Vec<f32> = synth(hidden, 0x112)
        .into_iter()
        .map(|v| 1.0 + 0.05 * v)
        .collect();
    let wq = synth(q_dim * hidden, 0x113);
    let wk = synth(kv_dim * hidden, 0x114);
    let wv = synth(kv_dim * hidden, 0x115);
    let eps = 1e-6;
    let offset = 0.0;

    let backend = CudaBackend::new().unwrap();
    let cuda = qkv_rms_proj(
        &backend,
        &x,
        &norm_w,
        &wq,
        &wk,
        &wv,
        QkvProjDims {
            hidden,
            q_dim,
            kv_dim,
        },
        eps,
        offset,
    )
    .expect("qkv_rms_proj");

    let x_norm = rms_norm(&x, &norm_w, eps, offset);
    let q_cpu = matvec_rows(&wq, &x_norm, q_dim, hidden);
    let k_cpu = matvec_rows(&wk, &x_norm, kv_dim, hidden);
    let v_cpu = matvec_rows(&wv, &x_norm, kv_dim, hidden);
    assert!(max_abs_diff(&q_cpu, &cuda.q) <= 1e-3);
    assert!(max_abs_diff(&k_cpu, &cuda.k) <= 1e-3);
    assert!(max_abs_diff(&v_cpu, &cuda.v) <= 1e-3);
}

#[test]
fn fused_decode_attention_matches_cpu_reference() {
    let Some(()) = gpu_or_skip() else { return };
    let opts = FusedDecodeAttentionOpts {
        num_q_heads: 4,
        num_kv_heads: 2,
        head_dim: 32,
        pos: 5,
        max_seq: 8,
        rotary_dim: 16,
        rope_base: 10_000.0,
        eps: 1e-6,
        qk_norm_offset: 0.0,
        attn_scale: 1.0 / (32.0_f32).sqrt(),
        softcap: 50.0,
    };
    let q = synth(opts.num_q_heads * opts.head_dim, 0x211);
    let k_new = synth(opts.num_kv_heads * opts.head_dim, 0x212);
    let v_new = synth(opts.num_kv_heads * opts.head_dim, 0x213);
    let mut k_cache = synth(opts.max_seq * opts.num_kv_heads * opts.head_dim, 0x214);
    let mut v_cache = synth(opts.max_seq * opts.num_kv_heads * opts.head_dim, 0x215);
    for idx in opts.pos * opts.num_kv_heads * opts.head_dim..k_cache.len() {
        k_cache[idx] = 0.0;
        v_cache[idx] = 0.0;
    }
    let q_norm: Vec<f32> = synth(opts.head_dim, 0x216)
        .into_iter()
        .map(|v| 1.0 + 0.03 * v)
        .collect();
    let k_norm: Vec<f32> = synth(opts.head_dim, 0x217)
        .into_iter()
        .map(|v| 1.0 + 0.03 * v)
        .collect();

    let backend = CudaBackend::new().unwrap();
    let cuda = fused_decode_attention(
        &backend,
        &q,
        &k_new,
        &v_new,
        &k_cache,
        &v_cache,
        Some(&q_norm),
        Some(&k_norm),
        opts,
    )
    .expect("fused_decode_attention");
    let (out_cpu, k_cpu, v_cpu) = reference_fused_decode_attention(
        &q, &k_new, &v_new, &k_cache, &v_cache, &q_norm, &k_norm, opts,
    );

    let diff = max_abs_diff(&out_cpu, &cuda.out);
    let cos = cosine(&out_cpu, &cuda.out);
    assert!(diff <= 2e-3, "fused decode max abs diff {diff}");
    assert!(cos >= 0.9999, "fused decode cosine {cos}");
    assert!(max_abs_diff(&k_cpu, &cuda.k_cache) <= 1e-3);
    assert!(max_abs_diff(&v_cpu, &cuda.v_cache) <= 1e-6);
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
    let cuda = decode_attention(&backend, &q, &k, &v, n_q, n_kv, head_dim, opts)
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
    let cuda = decode_attention(&backend, &q, &k, &v, n_q, n_kv, head_dim, opts)
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
    let cuda = decode_attention(&backend, &q, &k, &v, n_q, n_kv, head_dim, opts)
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
    let q: Vec<f32> = synth(n_q * head_dim, 0xD1)
        .into_iter()
        .map(|v| v * 50.0)
        .collect();
    let k: Vec<f32> = synth(n_kv * head_dim, 0xD2)
        .into_iter()
        .map(|v| v * 50.0)
        .collect();
    let v = synth(n_kv * head_dim, 0xD3);
    let opts = AttentionOpts {
        causal: false,
        softcap: 50.0,
    };
    let backend = CudaBackend::new().unwrap();
    let cuda = decode_attention(&backend, &q, &k, &v, n_q, n_kv, head_dim, opts)
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
    let cuda = decode_attention(&backend, &q, &k, &v, n_q, n_kv, head_dim, opts)
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
    let n_q = 1; // single decode token
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
    let cuda = decode_attention(&backend, &q, &k, &v, n_q, n_kv, head_dim, opts)
        .expect("decode_attention");
    let cpu = reference_attention(&q, &k, &v, n_q, n_kv, head_dim, false, 0.0);
    let diff = max_abs_diff(&cpu, &cuda);
    let cos = cosine(&cpu, &cuda);
    assert!(diff <= TOL_ABS, "max abs diff {diff}");
    assert!(cos >= TOL_COS, "cosine {cos}");
}
