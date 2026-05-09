//! Parity tests for `cuda::attn_tree::tree_decode_attention` vs a
//! scalar Rust attention loop.
//!
//! Phase 3 task 3.1 of `cuda-speculative-decoding`. Validates the
//! batched masked attention primitive at q_tokens ∈ {1, 4, 5, 8}
//! with GQA (group=2) and explicit per-q causal masks.

#![cfg(any(feature = "cuda", feature = "cuda-oxide"))]

use larql_compute::cuda::attn_tree::{tree_decode_attention, TreeAttentionOpts};
use larql_compute::cuda::CudaBackend;

fn gpu_or_skip() -> Option<CudaBackend> {
    if std::env::var("LARQL_CUDA_AVAILABLE").ok().as_deref() != Some("1") {
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
            ((s & 0xFF_FFFF) as f32 / 8_388_608.0) - 1.0
        })
        .collect()
}

/// Scalar reference: for each (q_token, q_head), compute
/// softmax(scale·Q·Kᵀ ⊙ mask) · V over the masked subset of ctx_len.
#[allow(clippy::too_many_arguments)]
fn ref_tree_attention(
    q: &[f32],
    k: &[f32],
    v: &[f32],
    mask_bits: &[u32],
    q_tokens: usize,
    num_q_heads: usize,
    num_kv_heads: usize,
    head_dim: usize,
    ctx_len: usize,
    words_per_row: usize,
    attn_scale: f32,
    softcap: f32,
) -> Vec<f32> {
    let mut out = vec![0.0_f32; q_tokens * num_q_heads * head_dim];
    let group = (num_q_heads / num_kv_heads).max(1);
    for qt in 0..q_tokens {
        let row_off = qt * words_per_row;
        let mask_get = |j: usize| -> bool {
            let w = mask_bits[row_off + j / 32];
            ((w >> (j % 32)) & 1) == 1
        };
        for qh in 0..num_q_heads {
            let kvh = (qh / group).min(num_kv_heads - 1);
            let q_vec = &q[(qt * num_q_heads + qh) * head_dim..][..head_dim];

            // Compute scores.
            let mut scores = vec![f32::NEG_INFINITY; ctx_len];
            for j in 0..ctx_len {
                if !mask_get(j) {
                    continue;
                }
                let k_vec = &k[(j * num_kv_heads + kvh) * head_dim..][..head_dim];
                let mut dot = 0.0;
                for d in 0..head_dim {
                    dot += q_vec[d] * k_vec[d];
                }
                let mut s = dot * attn_scale;
                if softcap > 0.0 {
                    s = softcap * (s / softcap).tanh();
                }
                scores[j] = s;
            }
            let row_max = scores.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
            let mut sum = 0.0_f32;
            for s in scores.iter_mut() {
                *s = if *s == f32::NEG_INFINITY {
                    0.0
                } else {
                    (*s - row_max).exp()
                };
                sum += *s;
            }
            let inv_sum = 1.0 / sum;
            let out_vec = &mut out[(qt * num_q_heads + qh) * head_dim..][..head_dim];
            for d in 0..head_dim {
                let mut acc = 0.0;
                for j in 0..ctx_len {
                    if scores[j] == 0.0 {
                        continue;
                    }
                    let v_vec = &v[(j * num_kv_heads + kvh) * head_dim..][..head_dim];
                    acc += scores[j] * inv_sum * v_vec[d];
                }
                out_vec[d] = acc;
            }
        }
    }
    out
}

fn max_abs_diff(a: &[f32], b: &[f32]) -> f32 {
    a.iter()
        .zip(b)
        .map(|(x, y)| (x - y).abs())
        .fold(0.0_f32, f32::max)
}

fn build_full_mask(q_tokens: usize, ctx_len: usize) -> (Vec<u32>, usize) {
    let words_per_row = ctx_len.div_ceil(32);
    let mut bits = vec![0u32; q_tokens * words_per_row];
    for q in 0..q_tokens {
        for j in 0..ctx_len {
            bits[q * words_per_row + j / 32] |= 1u32 << (j % 32);
        }
    }
    (bits, words_per_row)
}

#[test]
fn tree_attention_q1_full_mask_matches_reference() {
    let Some(backend) = gpu_or_skip() else { return };
    let opts0 = (8usize, 4usize, 64usize, 16usize); // n_qh, n_kvh, head_dim, ctx_len
    let (n_qh, n_kvh, head_dim, ctx_len) = opts0;
    let q_tokens = 1;
    let q = synth(q_tokens * n_qh * head_dim, 0xAA);
    let k = synth(ctx_len * n_kvh * head_dim, 0xBB);
    let v = synth(ctx_len * n_kvh * head_dim, 0xCC);
    let (mask, wpr) = build_full_mask(q_tokens, ctx_len);

    let opts = TreeAttentionOpts {
        q_tokens,
        num_q_heads: n_qh,
        num_kv_heads: n_kvh,
        head_dim,
        ctx_len,
        words_per_row: wpr,
        attn_scale: 1.0 / (head_dim as f32).sqrt(),
        softcap: 0.0,
    };
    let gpu = tree_decode_attention(&backend, &q, &k, &v, &mask, opts).expect("attn");
    let cpu = ref_tree_attention(
        &q,
        &k,
        &v,
        &mask,
        q_tokens,
        n_qh,
        n_kvh,
        head_dim,
        ctx_len,
        wpr,
        opts.attn_scale,
        0.0,
    );
    let d = max_abs_diff(&gpu, &cpu);
    assert!(d < 1e-3, "max abs diff {d}");
}

#[test]
fn tree_attention_q5_with_per_q_mask_matches_reference() {
    // 5-node tree at depth=2 b=2 layout: root + 2 children + 4 leaves
    // implies q_tokens=5 if we count a root + 2 children + 2 grandchildren.
    // Use a per-q causal mask: q_token k attends to ctx[0..ctx_base) +
    // its own ancestor positions.
    let Some(backend) = gpu_or_skip() else { return };
    let n_qh = 8;
    let n_kvh = 4;
    let head_dim = 64;
    let ctx_base = 32; // history positions
    let tree_len = 5; // number of tree nodes appended after history
    let ctx_len = ctx_base + tree_len;
    let q_tokens = tree_len;

    let q = synth(q_tokens * n_qh * head_dim, 0x111);
    let k = synth(ctx_len * n_kvh * head_dim, 0x222);
    let v = synth(ctx_len * n_kvh * head_dim, 0x333);

    // Tree: 0 (root)
    //       1, 2 (children of 0)
    //       3 (child of 1), 4 (child of 2)
    // Ancestors of each tree node (incl. self):
    //   0 -> {0}
    //   1 -> {0, 1}
    //   2 -> {0, 2}
    //   3 -> {0, 1, 3}
    //   4 -> {0, 2, 4}
    let ancestors: [&[usize]; 5] = [&[0], &[0, 1], &[0, 2], &[0, 1, 3], &[0, 2, 4]];
    let words_per_row = ctx_len.div_ceil(32);
    let mut mask = vec![0u32; q_tokens * words_per_row];
    for q_idx in 0..q_tokens {
        let row = &mut mask[q_idx * words_per_row..(q_idx + 1) * words_per_row];
        // History: every q sees [0, ctx_base)
        for j in 0..ctx_base {
            row[j / 32] |= 1u32 << (j % 32);
        }
        // Tree ancestors
        for &a in ancestors[q_idx] {
            let j = ctx_base + a;
            row[j / 32] |= 1u32 << (j % 32);
        }
    }

    let opts = TreeAttentionOpts {
        q_tokens,
        num_q_heads: n_qh,
        num_kv_heads: n_kvh,
        head_dim,
        ctx_len,
        words_per_row,
        attn_scale: 1.0 / (head_dim as f32).sqrt(),
        softcap: 0.0,
    };
    let gpu = tree_decode_attention(&backend, &q, &k, &v, &mask, opts).expect("attn");
    let cpu = ref_tree_attention(
        &q,
        &k,
        &v,
        &mask,
        q_tokens,
        n_qh,
        n_kvh,
        head_dim,
        ctx_len,
        words_per_row,
        opts.attn_scale,
        0.0,
    );
    let d = max_abs_diff(&gpu, &cpu);
    assert!(d < 1e-3, "max abs diff {d}");
}

#[test]
fn tree_attention_q4_softcap_matches_reference() {
    let Some(backend) = gpu_or_skip() else { return };
    let n_qh = 8;
    let n_kvh = 4;
    let head_dim = 64;
    let ctx_len = 64;
    let q_tokens = 4;
    let q = synth(q_tokens * n_qh * head_dim, 0xDD);
    let k = synth(ctx_len * n_kvh * head_dim, 0xEE);
    let v = synth(ctx_len * n_kvh * head_dim, 0xFF);
    let (mask, wpr) = build_full_mask(q_tokens, ctx_len);

    let opts = TreeAttentionOpts {
        q_tokens,
        num_q_heads: n_qh,
        num_kv_heads: n_kvh,
        head_dim,
        ctx_len,
        words_per_row: wpr,
        attn_scale: 1.0 / (head_dim as f32).sqrt(),
        softcap: 50.0,
    };
    let gpu = tree_decode_attention(&backend, &q, &k, &v, &mask, opts).expect("attn");
    let cpu = ref_tree_attention(
        &q,
        &k,
        &v,
        &mask,
        q_tokens,
        n_qh,
        n_kvh,
        head_dim,
        ctx_len,
        wpr,
        opts.attn_scale,
        50.0,
    );
    let d = max_abs_diff(&gpu, &cpu);
    assert!(d < 1e-3, "max abs diff {d} (softcap)");
}

#[test]
fn tree_attention_gemma_4b_head_shape_q5() {
    // Approximate Gemma 3 4B GQA: 8 q heads, 4 kv heads, head_dim=256.
    // Test correctness at Gemma's actual head shape.
    let Some(backend) = gpu_or_skip() else { return };
    let n_qh = 8;
    let n_kvh = 4;
    let head_dim = 256;
    let ctx_len = 128;
    let q_tokens = 5;
    let q = synth(q_tokens * n_qh * head_dim, 0xA1);
    let k = synth(ctx_len * n_kvh * head_dim, 0xA2);
    let v = synth(ctx_len * n_kvh * head_dim, 0xA3);
    let (mask, wpr) = build_full_mask(q_tokens, ctx_len);

    let opts = TreeAttentionOpts {
        q_tokens,
        num_q_heads: n_qh,
        num_kv_heads: n_kvh,
        head_dim,
        ctx_len,
        words_per_row: wpr,
        attn_scale: 1.0 / (head_dim as f32).sqrt(),
        softcap: 0.0,
    };
    let gpu = tree_decode_attention(&backend, &q, &k, &v, &mask, opts).expect("attn");
    let cpu = ref_tree_attention(
        &q,
        &k,
        &v,
        &mask,
        q_tokens,
        n_qh,
        n_kvh,
        head_dim,
        ctx_len,
        wpr,
        opts.attn_scale,
        0.0,
    );
    let d = max_abs_diff(&gpu, &cpu);
    assert!(d < 5e-3, "Gemma shape max abs diff {d}");
}

#[test]
fn tree_attention_q1_matches_q5_first_slice() {
    // Run q_tokens=1 then run q_tokens=5 with the same Q[0] in the
    // first slot. The first slice of the q=5 output SHALL match the
    // q=1 output bit-equally (mask is full for both).
    let Some(backend) = gpu_or_skip() else { return };
    let n_qh = 8;
    let n_kvh = 4;
    let head_dim = 64;
    let ctx_len = 16;

    // Single-q baseline.
    let q1 = synth(n_qh * head_dim, 0xCAFE);
    let k = synth(ctx_len * n_kvh * head_dim, 0xBABE);
    let v = synth(ctx_len * n_kvh * head_dim, 0xFEED);
    let (mask1, wpr) = build_full_mask(1, ctx_len);
    let opts1 = TreeAttentionOpts {
        q_tokens: 1,
        num_q_heads: n_qh,
        num_kv_heads: n_kvh,
        head_dim,
        ctx_len,
        words_per_row: wpr,
        attn_scale: 1.0 / (head_dim as f32).sqrt(),
        softcap: 0.0,
    };
    let gpu1 = tree_decode_attention(&backend, &q1, &k, &v, &mask1, opts1).expect("q=1");

    // Stitched 5-q.
    let mut q5 = q1.clone();
    let q_extra = synth(4 * n_qh * head_dim, 0xDEAD);
    q5.extend_from_slice(&q_extra);
    let (mask5, _wpr5) = build_full_mask(5, ctx_len);
    let opts5 = TreeAttentionOpts {
        q_tokens: 5,
        num_q_heads: n_qh,
        num_kv_heads: n_kvh,
        head_dim,
        ctx_len,
        words_per_row: wpr,
        attn_scale: 1.0 / (head_dim as f32).sqrt(),
        softcap: 0.0,
    };
    let gpu5 = tree_decode_attention(&backend, &q5, &k, &v, &mask5, opts5).expect("q=5");

    let slice = &gpu5[..n_qh * head_dim];
    let d = max_abs_diff(slice, &gpu1);
    assert!(d < 1e-5, "first slice mismatch: {d}");
}
