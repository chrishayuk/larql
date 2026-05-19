#![cfg(target_os = "macos")]

//! Per-kernel tests for `kv_attention` — KV-cached single-token decode
//! attention. Companion to the prefill-side `fused_attention` tests.
//!
//! ## Why a focused file
//!
//! `kv_attention` is exercised only by the decode path
//! (`metal/decode/mod.rs::encode_kv_attend`), so any bug here surfaces
//! end-to-end only as a divergence between Metal-decode and a fresh
//! prefill at the same sequence length. The
//! `test_decode_consistency` integration suite catches that, but
//! doesn't tell us which kernel introduced the drift. These tests
//! pin the kernel itself against a hand-computed Rust reference so a
//! shader-level regression points to itself.
//!
//! ## What they assert
//!
//! For each (T, num_q, num_kv, head_dim) combination:
//!   - Compute attention via `kv_attention` shader (the actual decode
//!     pipeline used in production).
//!   - Compute the same softmax(QK·scale)·V on CPU.
//!   - Assert per-head cos > 0.999999 and max abs diff < 1e-3.
//!
//! Geometries chosen to cover production:
//!   - `(T=1,   num_q=8, num_kv=2,  head_dim=128)`  — Llama-2 7B-style
//!   - `(T=18,  num_q=8, num_kv=4,  head_dim=256)`  — Gemma 3 4B
//!   - `(T=18,  num_q=32, num_kv=16, head_dim=256)` — Gemma 4 31B sliding
//!   - `(T=18,  num_q=32, num_kv=4,  head_dim=512)` — Gemma 4 31B global ←
//!   - `(T=512, num_q=8, num_kv=2,  head_dim=128)` — short scores path
//!   - `(T=2048,num_q=32,num_kv=4,  head_dim=512)` — long scores path

extern crate blas_src;

#[path = "common/mod.rs"]
mod common;
use common::{cos_sim, get_metal, max_diff};

/// CPU reference: causal-masked GQA softmax-weighted attention. Single
/// query position (`Q.len() == num_q * head_dim`), `T` cached K/V
/// positions. Output is `[num_q, head_dim]` flat.
#[allow(clippy::too_many_arguments)]
fn cpu_kv_attention(
    q: &[f32],
    k_cache: &[f32],
    v_cache: &[f32],
    t: usize,
    num_q: usize,
    num_kv: usize,
    head_dim: usize,
    scale: f32,
) -> Vec<f32> {
    let mut out = vec![0.0f32; num_q * head_dim];
    let reps = num_q / num_kv;
    for h in 0..num_q {
        let kv_h = h / reps;
        let q_off = h * head_dim;
        // Q · K^T over all cached positions.
        let mut scores = vec![0.0f32; t];
        for (ki, score) in scores.iter_mut().enumerate() {
            let k_off = ki * num_kv * head_dim + kv_h * head_dim;
            let mut dot = 0.0f64;
            for d in 0..head_dim {
                dot += (q[q_off + d] as f64) * (k_cache[k_off + d] as f64);
            }
            *score = (dot as f32) * scale;
        }
        // Stable softmax.
        let max_s = scores.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        let mut exps: Vec<f32> = scores.iter().map(|s| (s - max_s).exp()).collect();
        let sum_exp: f32 = exps.iter().sum();
        for e in exps.iter_mut() {
            *e /= sum_exp;
        }
        // V-weighted sum.
        for d in 0..head_dim {
            let mut acc = 0.0f64;
            for (ki, &exp) in exps.iter().enumerate() {
                let v_off = ki * num_kv * head_dim + kv_h * head_dim;
                acc += (exp as f64) * (v_cache[v_off + d] as f64);
            }
            out[q_off + d] = acc as f32;
        }
    }
    out
}

#[allow(clippy::too_many_arguments)]
fn run_kv_attention(
    metal: &larql_compute_metal::MetalBackend,
    q: &[f32],
    k_cache: &[f32],
    v_cache: &[f32],
    t: usize,
    num_q: usize,
    num_kv: usize,
    head_dim: usize,
    scale: f32,
    window_size: u32,
) -> Vec<f32> {
    let q_buf = metal.bufs().transient_from_f32(q);
    let k_buf = metal.bufs().transient_from_f32(k_cache);
    let v_buf = metal.bufs().transient_from_f32(v_cache);
    let out_buf = metal.bufs().output((num_q * head_dim * 4) as u64);

    let t_val = t as u32;
    let hd = head_dim as u32;
    let nq_val = num_q as u32;
    let nkv = num_kv as u32;

    let cmd = metal.queue().new_command_buffer();
    let enc = cmd.new_compute_command_encoder();
    let span = larql_compute_metal::ops::kv_cache::attention_span(t_val, window_size);
    let pipeline = if span > larql_compute_metal::ops::kv_cache::SHORT_ATTENTION_SPAN {
        &metal.attention.kv_attend_long_pipeline
    } else {
        &metal.attention.kv_attend_pipeline
    };
    enc.set_compute_pipeline_state(pipeline);
    enc.set_buffer(0, Some(&q_buf), 0);
    enc.set_buffer(1, Some(&k_buf), 0);
    enc.set_buffer(2, Some(&v_buf), 0);
    enc.set_buffer(3, Some(&out_buf), 0);
    enc.set_bytes(4, 4, &t_val as *const u32 as *const std::ffi::c_void);
    enc.set_bytes(5, 4, &hd as *const u32 as *const std::ffi::c_void);
    enc.set_bytes(6, 4, &nq_val as *const u32 as *const std::ffi::c_void);
    enc.set_bytes(7, 4, &nkv as *const u32 as *const std::ffi::c_void);
    enc.set_bytes(8, 4, &scale as *const f32 as *const std::ffi::c_void);
    enc.set_bytes(9, 4, &window_size as *const u32 as *const std::ffi::c_void);
    enc.dispatch_thread_groups(
        metal::MTLSize::new(num_q as u64, 1, 1),
        metal::MTLSize::new(256.min(head_dim as u64), 1, 1),
    );
    enc.end_encoding();
    cmd.commit();
    cmd.wait_until_completed();

    larql_compute_metal::buffers::read_buffer_f32(&out_buf, num_q * head_dim)
}

#[allow(clippy::too_many_arguments)]
fn assert_kv_attention_matches_cpu(
    label: &str,
    t: usize,
    num_q: usize,
    num_kv: usize,
    head_dim: usize,
) {
    let metal = get_metal();
    let scale = 1.0f32; // Gemma 4 uses QK-norm so default scale is 1.0
    let window = 0u32; // 0 = no sliding window

    let q_total = num_q * head_dim;
    let kv_total_per_pos = num_kv * head_dim;

    // Deterministic synthetic data — non-trivial enough that any kernel
    // shape bug produces a detectable diff but not so wild that fp32
    // accumulation becomes the bottleneck.
    let q: Vec<f32> = (0..q_total)
        .map(|i| ((i as f32 * 0.017).sin() + 0.3 * ((i >> 5) as f32).cos()) * 0.4)
        .collect();
    let k_total = t * kv_total_per_pos;
    let k: Vec<f32> = (0..k_total)
        .map(|i| ((i as f32 * 0.013).cos() - 0.3 * ((i >> 4) as f32).sin()) * 0.4)
        .collect();
    let v: Vec<f32> = (0..k_total)
        .map(|i| ((i as f32 * 0.019).sin() + 0.2 * ((i >> 6) as f32).sin()) * 0.25)
        .collect();

    let cpu_out = cpu_kv_attention(&q, &k, &v, t, num_q, num_kv, head_dim, scale);
    let metal_out = run_kv_attention(
        &metal, &q, &k, &v, t, num_q, num_kv, head_dim, scale, window,
    );

    let diff = max_diff(&cpu_out, &metal_out);
    let cos = cos_sim(&cpu_out, &metal_out);
    assert!(
        diff < 1e-3 && cos > 0.999999,
        "kv_attention {label} (T={t} num_q={num_q} num_kv={num_kv} head_dim={head_dim}): \
         max_abs_diff={diff:.3e} cos={cos:.6} (thresholds: max<1e-3, cos>0.999999)\n\
         cpu[..8]={:?}\nmtl[..8]={:?}",
        &cpu_out[..8.min(cpu_out.len())],
        &metal_out[..8.min(metal_out.len())],
    );
}

#[test]
fn kv_attention_t1_llama2() {
    assert_kv_attention_matches_cpu("llama2 T=1", 1, 8, 2, 128);
}

#[test]
fn kv_attention_t18_gemma3() {
    assert_kv_attention_matches_cpu("gemma3 T=18", 18, 8, 4, 256);
}

#[test]
fn kv_attention_t18_gemma4_sliding() {
    // Gemma 4 31B sliding-layer geometry. head_dim=256 fits inside the
    // shader's max-256-thread TG cleanly.
    assert_kv_attention_matches_cpu("gemma4 sliding T=18", 18, 32, 16, 256);
}

#[test]
fn kv_attention_t18_gemma4_global_head_dim_512() {
    // **The decode-bug suspect.** Gemma 4 31B global layers use
    // head_dim=512; the kv_attention shader's TG is min(256, head_dim)
    // = 256 threads, so the per-head V-weighted-sum loop has to stride
    // (each thread handles 2 d values). Same shape that broke
    // `fused_attention` (caught by `fused_attention_head_dim_512`).
    // If the prefill version had a tg_q-init bug, the decode version
    // is the next place to look.
    assert_kv_attention_matches_cpu("gemma4 global T=18", 18, 32, 4, 512);
}

#[test]
fn kv_attention_t512_long_context() {
    // Stresses the score-accumulation buffer and softmax stability
    // across a much wider attention window. The shader's small-TG
    // scores buffer is sized 1024 — anything beyond that uses the
    // larger-buffer variant; this test sits inside the cheap path.
    assert_kv_attention_matches_cpu("long T=512", 512, 8, 2, 128);
}

#[test]
fn kv_attention_t2048_gemma4_global_long_context() {
    // Gemma 4 31B global layers are full-attention with head_dim=512.
    // Once T passes 1024 they must use kv_attention_long; the short shader's
    // 1024-entry scores buffer would otherwise write out of bounds.
    assert_kv_attention_matches_cpu("gemma4 global T=2048", 2048, 32, 4, 512);
}
