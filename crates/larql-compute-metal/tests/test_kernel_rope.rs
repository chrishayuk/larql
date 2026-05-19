#![cfg(target_os = "macos")]

//! Per-kernel tests for the three RoPE shader variants
//! (`metal/shaders/rope.rs`):
//!
//! 1. `rope_apply` — multi-position, used by Metal prefill.
//! 2. `rope_at_pos` — single vector at a fixed absolute position.
//! 3. `rope_at_pos_batched` — all heads at one position, used by
//!    Metal KV-cached decode.
//!
//! ## Why this file
//!
//! The decode-vs-prefill divergence on Gemma 4 31B
//! (`test_decode_consistency::decode_consistency_gemma4_31b_dense`)
//! has narrowed to "decode-only kernels misbehave at head_dim=512 with
//! partial-rotary 25%". RoPE is one of two remaining suspects (the
//! other is `kv_cache_append`). Decode and prefill use *different*
//! RoPE shaders, so the per-layer parity test on prefill doesn't tell
//! us anything about the decode form.
//!
//! Production geometries we cover here:
//!   - Llama-2 / Mistral (head_dim=128, full rotation)
//!   - Gemma 3 (head_dim=256, full rotation)
//!   - Gemma 4 sliding (head_dim=256, full rotation, rope_base=10000)
//!   - **Gemma 4 global (head_dim=512, 25% partial rotation, rope_base=500000)**
//!     ← the suspect.
//!
//! ## Reference
//!
//! All three shaders implement Llama-style split-half rotation:
//! pair `(x[i], x[i + rotary_dim/2])` rotated by angle `pos * freq(i)`
//! where `freq(i) = 1 / base^(2*i / rotary_dim)`. Dims past
//! `rotary_dim` pass through unchanged. Reference Rust implementation
//! mirrors that exactly.

extern crate blas_src;

#[path = "common/mod.rs"]
mod common;
use common::{cos_sim, get_metal, max_diff};

/// CPU reference: apply Llama-style split-half RoPE in place to a
/// single head vector at absolute position `pos`. `rotary_dim` of 0
/// means "rotate the entire head_dim".
fn cpu_rope_at_pos(head_dim: usize, rotary_dim: usize, base: f32, pos: usize, x: &mut [f32]) {
    debug_assert_eq!(x.len(), head_dim);
    let rdim = if rotary_dim == 0 {
        head_dim
    } else {
        rotary_dim.min(head_dim)
    };
    let hdim = rdim / 2;
    for d in 0..hdim {
        let freq = 1.0 / base.powf(2.0 * d as f32 / rdim as f32);
        let angle = pos as f32 * freq;
        let cos_a = angle.cos();
        let sin_a = angle.sin();
        let re = x[d];
        let im = x[d + hdim];
        x[d] = re * cos_a - im * sin_a;
        x[d + hdim] = re * sin_a + im * cos_a;
    }
}

/// CPU reference for the batched form used by decode: rotate every
/// head of a `[num_heads, head_dim]` flat buffer at the same position.
fn cpu_rope_at_pos_batched(
    x: &mut [f32],
    num_heads: usize,
    head_dim: usize,
    rotary_dim: usize,
    base: f32,
    pos: usize,
) {
    for h in 0..num_heads {
        let off = h * head_dim;
        let head = &mut x[off..off + head_dim];
        cpu_rope_at_pos(head_dim, rotary_dim, base, pos, head);
    }
}

// ── rope_at_pos_batched (decode path) ───────────────────────────────────────

#[allow(clippy::too_many_arguments)]
fn run_rope_at_pos_batched(
    metal: &larql_compute_metal::MetalBackend,
    x: &[f32],
    num_heads: usize,
    head_dim: usize,
    rotary_dim: usize,
    base: f32,
    pos: usize,
) -> Vec<f32> {
    let buf = metal.bufs().transient_from_f32(x);
    let hd_val = head_dim as u32;
    let rd_val = rotary_dim as u32;
    let nh_val = num_heads as u32;
    let pos_val = pos as u32;

    let cmd = metal.queue().new_command_buffer();
    let enc = cmd.new_compute_command_encoder();
    enc.set_compute_pipeline_state(&metal.attention.rope_at_pos_batched_pipeline);
    enc.set_buffer(0, Some(&buf), 0);
    enc.set_bytes(1, 4, &hd_val as *const u32 as *const std::ffi::c_void);
    enc.set_bytes(2, 4, &base as *const f32 as *const std::ffi::c_void);
    enc.set_bytes(3, 4, &pos_val as *const u32 as *const std::ffi::c_void);
    enc.set_bytes(4, 4, &rd_val as *const u32 as *const std::ffi::c_void);
    enc.set_bytes(5, 4, &nh_val as *const u32 as *const std::ffi::c_void);

    // Match the production decode dispatch (one thread per pair × per head).
    let rdim_eff = if rotary_dim == 0 {
        head_dim
    } else {
        rotary_dim
    };
    let pairs = (rdim_eff / 2) as u64;
    enc.dispatch_threads(
        metal::MTLSize::new(pairs, num_heads as u64, 1),
        metal::MTLSize::new(pairs.min(256), 1, 1),
    );
    enc.end_encoding();
    cmd.commit();
    cmd.wait_until_completed();

    larql_compute_metal::buffers::read_buffer_f32(&buf, num_heads * head_dim)
}

#[allow(clippy::too_many_arguments)]
fn assert_rope_at_pos_batched_matches_cpu(
    label: &str,
    num_heads: usize,
    head_dim: usize,
    rotary_dim: usize,
    base: f32,
    pos: usize,
) {
    let metal = get_metal();
    let n = num_heads * head_dim;
    let x: Vec<f32> = (0..n)
        .map(|i| ((i as f32 * 0.011).sin() + 0.4 * ((i >> 4) as f32).cos()) * 0.5)
        .collect();
    let mut expected = x.clone();
    cpu_rope_at_pos_batched(&mut expected, num_heads, head_dim, rotary_dim, base, pos);
    let result = run_rope_at_pos_batched(&metal, &x, num_heads, head_dim, rotary_dim, base, pos);
    let diff = max_diff(&expected, &result);
    let cos = cos_sim(&expected, &result);
    assert!(
        diff < 1e-4 && cos > 0.999999,
        "rope_at_pos_batched {label} (num_heads={num_heads} head_dim={head_dim} \
         rotary_dim={rotary_dim} base={base} pos={pos}): \
         max_abs={diff:.3e} cos={cos:.6}",
    );
}

#[test]
fn rope_at_pos_batched_llama2_full() {
    // 32 heads × 128 dim, full rotation, standard rope_base.
    for &pos in &[0, 1, 5, 17] {
        assert_rope_at_pos_batched_matches_cpu("llama2 full", 32, 128, 0, 10_000.0, pos);
    }
}

#[test]
fn rope_at_pos_batched_gemma3_full_256() {
    // Gemma 3 4B: 8 KV heads × 256 dim, full rotation.
    for &pos in &[0, 7, 23] {
        assert_rope_at_pos_batched_matches_cpu("gemma3 full 256", 8, 256, 0, 10_000.0, pos);
    }
}

#[test]
fn rope_at_pos_batched_gemma4_sliding() {
    // Gemma 4 31B sliding layer KV geometry: 16 heads × 256 dim,
    // full rotation, rope_base=10000.
    for &pos in &[0, 17, 100] {
        assert_rope_at_pos_batched_matches_cpu("gemma4 sliding", 16, 256, 0, 10_000.0, pos);
    }
}

#[test]
fn rope_at_pos_batched_gemma4_global_partial() {
    // **The decode-bug suspect.** Gemma 4 31B global: 4 KV heads × 512
    // dim, *25% partial* rotation (rotary_dim=128), rope_base=500000.
    // Same shape that broke `fused_attention` (caught by
    // `fused_attention_head_dim_512` previously). If the tg_q gating
    // bug has a sibling here, this test catches it.
    for &pos in &[0, 17, 100] {
        assert_rope_at_pos_batched_matches_cpu(
            "gemma4 global partial",
            4,
            512,
            128,
            500_000.0,
            pos,
        );
    }
}

#[test]
fn rope_at_pos_batched_q_heads_global() {
    // Q heads at the global geometry — same head_dim=512 / partial=128
    // but more heads (32 — Gemma 4 31B keeps num_q constant across
    // sliding/global). Ensures the per-head dispatch scales correctly.
    for &pos in &[0, 17] {
        assert_rope_at_pos_batched_matches_cpu(
            "gemma4 global Q heads",
            32,
            512,
            128,
            500_000.0,
            pos,
        );
    }
}

// `rope_apply` (prefill multi-position) is exercised end-to-end by
// `test_cpu_metal_parity` — full prefill matches CPU bit-exactly across
// all four test vindexes including Gemma 4 31B at head_dim=512 partial,
// so it's already pinned. Decoupling it into a kernel test would
// require exposing a pipeline accessor we don't have and isn't worth
// the surface change. The decode-only `rope_at_pos_batched` is what
// we don't have indirect coverage for, hence the targeted tests above.

// ── rope_at_pos_batched_qk: fused Q+K heads in one dispatch ─────────────────

/// Compare `rope_at_pos_batched_qk` (fused) against two separate
/// `rope_at_pos_batched` calls (Q heads, then K heads).
fn assert_rope_batched_qk_matches_separate(
    num_q_heads: usize,
    num_kv_heads: usize,
    head_dim: usize,
    rotary_dim: usize,
    rope_base: f32,
    pos: usize,
    label: &str,
) {
    let metal = get_metal();

    // Same input data for Q and K
    let q_in: Vec<f32> = (0..num_q_heads * head_dim)
        .map(|i| ((i as f32 * 0.011).sin() + 0.2) * 0.5)
        .collect();
    let k_in: Vec<f32> = (0..num_kv_heads * head_dim)
        .map(|i| ((i as f32 * 0.013).cos() + 0.1) * 0.5)
        .collect();

    // Reference: CPU RoPE on Q and K separately
    let mut ref_q = q_in.clone();
    let mut ref_k = k_in.clone();
    for h in 0..num_q_heads {
        cpu_rope_at_pos(
            head_dim,
            rotary_dim,
            rope_base,
            pos,
            &mut ref_q[h * head_dim..(h + 1) * head_dim],
        );
    }
    for h in 0..num_kv_heads {
        cpu_rope_at_pos(
            head_dim,
            rotary_dim,
            rope_base,
            pos,
            &mut ref_k[h * head_dim..(h + 1) * head_dim],
        );
    }

    // Fused: rope_at_pos_batched_qk
    let q_buf = metal.bufs().transient_from_f32(&q_in);
    let k_buf = metal.bufs().transient_from_f32(&k_in);

    let hd = head_dim as u32;
    let rdim = rotary_dim as u32;
    let pos_u = pos as u32;
    let nq = num_q_heads as u32;
    let rope_pairs = (if rotary_dim == 0 {
        head_dim
    } else {
        rotary_dim
    }) / 2;
    let total_heads = (num_q_heads + num_kv_heads) as u64;

    let cmd = metal.queue().new_command_buffer();
    let enc = cmd.new_compute_command_encoder();
    enc.set_compute_pipeline_state(&metal.attention.rope_at_pos_batched_qk_pipeline);
    enc.set_buffer(0, Some(&q_buf), 0);
    enc.set_buffer(1, Some(&k_buf), 0);
    enc.set_bytes(2, 4, &hd as *const u32 as *const std::ffi::c_void);
    enc.set_bytes(3, 4, &rope_base as *const f32 as *const std::ffi::c_void);
    enc.set_bytes(4, 4, &pos_u as *const u32 as *const std::ffi::c_void);
    enc.set_bytes(5, 4, &rdim as *const u32 as *const std::ffi::c_void);
    enc.set_bytes(6, 4, &nq as *const u32 as *const std::ffi::c_void);
    enc.dispatch_threads(
        metal::MTLSize::new(rope_pairs as u64, total_heads, 1),
        metal::MTLSize::new((rope_pairs as u64).min(256), 1, 1),
    );
    enc.end_encoding();
    cmd.commit();
    cmd.wait_until_completed();

    let got_q = larql_compute_metal::buffers::read_buffer_f32(&q_buf, num_q_heads * head_dim);
    let got_k = larql_compute_metal::buffers::read_buffer_f32(&k_buf, num_kv_heads * head_dim);

    let dq = max_diff(&ref_q, &got_q);
    assert!(dq < 1e-5, "{label} Q: max_diff {dq:.3e}");
    let dk = max_diff(&ref_k, &got_k);
    assert!(dk < 1e-5, "{label} K: max_diff {dk:.3e}");
}

#[test]
fn rope_at_pos_batched_qk_smoke() {
    assert_rope_batched_qk_matches_separate(4, 2, 16, 16, 10000.0, 5, "smoke");
}

#[test]
fn rope_at_pos_batched_qk_gemma3_4b() {
    // 32 Q + 16 KV heads, head_dim=256, full rotation, pos=42
    assert_rope_batched_qk_matches_separate(32, 16, 256, 256, 10000.0, 42, "gemma3-4b");
}

#[test]
fn rope_at_pos_batched_qk_partial_rotary() {
    // Gemma 4 global: head_dim=512, rotary_dim=128 (25%)
    assert_rope_batched_qk_matches_separate(4, 2, 512, 128, 500000.0, 7, "gemma4-global-partial");
}
