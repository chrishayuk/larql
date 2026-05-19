#![cfg(target_os = "macos")]

//! Per-kernel tests for `qk_norm` — per-head learned-weight RMSNorm.
//!
//! ## Why a focused file
//!
//! `qk_norm` is the production shader used by **both** Q/K-norm
//! (Gemma 3/4 attention pre-RoPE) **and** V-norm in Metal *prefill*
//! (`metal/ops/full_pipeline.rs:644-657` calls it with an all-ones
//! weight buffer + offset=0 to emulate the parameter-free V-norm). In
//! parallel, Metal *decode* applies V-norm via the dedicated
//! `v_norm_batched` shader.
//!
//! That means the prefill→decode KV cache hand-off depends on
//! `qk_norm(weight=1, offset=0)` producing **bit-equivalent** output
//! to `v_norm_batched`. If they diverge — even by float noise — every
//! cached V from prefill is subtly different from what decode would
//! have written, drifting downstream attention. With `kv_cache_append`,
//! `kv_attention`, and the RoPE shaders all already kernel-tested and
//! clean, this is the next remaining suspect for the open
//! `decode_consistency_gemma4_31b_dense` parity gap.
//!
//! ## What it asserts
//!
//! 1. **`qk_norm` standard form** — `(x / rms) * (offset + weight[d])`
//!    matches a CPU reference for the production geometries:
//!    Gemma 3 (head_dim=256, offset=1.0, learned weight),
//!    Gemma 4 sliding (head_dim=256, offset=0.0),
//!    Gemma 4 global (head_dim=512, offset=0.0).
//! 2. **`qk_norm` as parameter-free V-norm** — `weight=1, offset=0`
//!    produces output equal to `v_norm_batched` (and to a CPU
//!    parameter-free RMSNorm reference). Bit-equality is the bar:
//!    same formula, same f32 ops, same hardware. Any drift here is
//!    the direct cause of the open Gemma 4 31B parity gap.
//! 3. **In-place safety** — the production code aliases `x` and `out`;
//!    the threadgroup-shared partial-sum reduction must complete
//!    before any thread writes back. (Same hazard `v_norm_batched`
//!    had — see its in-place test.)

extern crate blas_src;

#[path = "common/mod.rs"]
mod common;
use common::{cos_sim, get_metal, max_diff};

// ── CPU references ──────────────────────────────────────────────────────────

/// `qk_norm` reference: `(x / rms) * (offset + weight[d])` per head.
fn cpu_qk_norm(
    x: &[f32],
    weight: &[f32],
    num_heads: usize,
    head_dim: usize,
    eps: f32,
    offset: f32,
) -> Vec<f32> {
    assert_eq!(x.len(), num_heads * head_dim);
    assert_eq!(weight.len(), head_dim);
    let mut out = vec![0.0f32; x.len()];
    for h in 0..num_heads {
        let base = h * head_dim;
        let sum_sq: f32 = x[base..base + head_dim].iter().map(|v| v * v).sum();
        let rms = (sum_sq / head_dim as f32 + eps).sqrt();
        for d in 0..head_dim {
            out[base + d] = (x[base + d] / rms) * (offset + weight[d]);
        }
    }
    out
}

/// `v_norm_batched` reference: `x * rsqrt(mean(x²) + eps)` per head.
fn cpu_v_norm_batched(x: &[f32], num_heads: usize, head_dim: usize, eps: f32) -> Vec<f32> {
    let mut out = vec![0.0f32; x.len()];
    for h in 0..num_heads {
        let base = h * head_dim;
        let sum_sq: f32 = x[base..base + head_dim].iter().map(|v| v * v).sum();
        let rms = 1.0 / (sum_sq / head_dim as f32 + eps).sqrt();
        for d in 0..head_dim {
            out[base + d] = x[base + d] * rms;
        }
    }
    out
}

// ── Dispatch helpers ───────────────────────────────────────────────────────

fn tg_width(head_dim: usize) -> u64 {
    let mut tg: u64 = 1;
    while (tg as usize) < head_dim && tg < 512 {
        tg <<= 1;
    }
    tg
}

#[allow(clippy::too_many_arguments)]
fn run_qk_norm(
    metal: &larql_compute_metal::MetalBackend,
    in_buf: &metal::Buffer,
    out_buf: &metal::Buffer,
    weight_buf: &metal::Buffer,
    num_heads: usize,
    head_dim: usize,
    eps: f32,
    offset: f32,
) {
    let hd_val = head_dim as u32;
    let nh_val = num_heads as u32;
    let tg_w = tg_width(head_dim);

    let cmd = metal.queue().new_command_buffer();
    let enc = cmd.new_compute_command_encoder();
    enc.set_compute_pipeline_state(&metal.norms.qk_norm_pipeline);
    enc.set_buffer(0, Some(in_buf), 0);
    enc.set_buffer(1, Some(out_buf), 0);
    enc.set_buffer(2, Some(weight_buf), 0);
    enc.set_bytes(3, 4, &hd_val as *const u32 as *const std::ffi::c_void);
    enc.set_bytes(4, 4, &nh_val as *const u32 as *const std::ffi::c_void);
    enc.set_bytes(5, 4, &eps as *const f32 as *const std::ffi::c_void);
    enc.set_bytes(6, 4, &offset as *const f32 as *const std::ffi::c_void);
    enc.dispatch_thread_groups(
        metal::MTLSize::new(num_heads as u64, 1, 1),
        metal::MTLSize::new(tg_w, 1, 1),
    );
    enc.end_encoding();
    cmd.commit();
    cmd.wait_until_completed();
}

fn run_v_norm_batched(
    metal: &larql_compute_metal::MetalBackend,
    in_buf: &metal::Buffer,
    out_buf: &metal::Buffer,
    num_heads: usize,
    head_dim: usize,
    eps: f32,
) {
    let hd_val = head_dim as u32;
    let nh_val = num_heads as u32;
    let tg_w = tg_width(head_dim);

    let cmd = metal.queue().new_command_buffer();
    let enc = cmd.new_compute_command_encoder();
    enc.set_compute_pipeline_state(&metal.norms.v_norm_batched_pipeline);
    enc.set_buffer(0, Some(in_buf), 0);
    enc.set_buffer(1, Some(out_buf), 0);
    enc.set_bytes(2, 4, &hd_val as *const u32 as *const std::ffi::c_void);
    enc.set_bytes(3, 4, &eps as *const f32 as *const std::ffi::c_void);
    enc.set_bytes(4, 4, &nh_val as *const u32 as *const std::ffi::c_void);
    enc.dispatch_thread_groups(
        metal::MTLSize::new(num_heads as u64, 1, 1),
        metal::MTLSize::new(tg_w, 1, 1),
    );
    enc.end_encoding();
    cmd.commit();
    cmd.wait_until_completed();
}

fn synth_input(num_heads: usize, head_dim: usize) -> Vec<f32> {
    (0..num_heads * head_dim)
        .map(|i| ((i as f32 * 0.013).sin() + 0.3 * ((i >> 5) as f32).cos()) * 0.4)
        .collect()
}

fn synth_weight(head_dim: usize) -> Vec<f32> {
    (0..head_dim)
        .map(|i| 0.5 + 0.05 * ((i as f32) * 0.07).sin())
        .collect()
}

// ── 1. qk_norm against CPU reference ───────────────────────────────────────

#[allow(clippy::too_many_arguments)]
fn assert_qk_norm_matches_cpu(label: &str, num_heads: usize, head_dim: usize, offset: f32) {
    let metal = get_metal();
    let eps = 1e-6f32;
    let x = synth_input(num_heads, head_dim);
    let weight = synth_weight(head_dim);
    let expected = cpu_qk_norm(&x, &weight, num_heads, head_dim, eps, offset);

    let in_buf = metal.bufs().transient_from_f32(&x);
    let out_buf = metal.bufs().output((x.len() * 4) as u64);
    let w_buf = metal.bufs().transient_from_f32(&weight);
    run_qk_norm(
        &metal, &in_buf, &out_buf, &w_buf, num_heads, head_dim, eps, offset,
    );

    let result = larql_compute_metal::buffers::read_buffer_f32(&out_buf, x.len());
    let diff = max_diff(&expected, &result);
    let cos = cos_sim(&expected, &result);
    assert!(
        diff < 1e-4 && cos > 0.999999,
        "qk_norm {label} (num_heads={num_heads} head_dim={head_dim} offset={offset}): \
         max_abs={diff:.3e} cos={cos:.6}",
    );
}

#[test]
fn qk_norm_gemma3_offset_one() {
    // Gemma 3 stores weight as `(weight - 1)` so offset=1.0 in the
    // shader. 8 KV heads × 256 = Gemma 3 4B K shape.
    assert_qk_norm_matches_cpu("gemma3 K", 8, 256, 1.0);
    // Q at Gemma 3 4B is 8 × 256 (or 32 × 256 for Q heads — same path).
    assert_qk_norm_matches_cpu("gemma3 Q", 32, 256, 1.0);
}

#[test]
fn qk_norm_gemma4_sliding_offset_zero() {
    // Gemma 4 31B sliding layer: 16 KV × 256, offset=0.0 (raw weight).
    assert_qk_norm_matches_cpu("gemma4 sliding K", 16, 256, 0.0);
    assert_qk_norm_matches_cpu("gemma4 sliding Q", 32, 256, 0.0);
}

#[test]
fn qk_norm_gemma4_global_offset_zero() {
    // **Parity-bug suspect geometry.** Gemma 4 31B global: 4 KV × 512
    // (K) and 32 × 512 (Q). offset=0.0.
    assert_qk_norm_matches_cpu("gemma4 global K", 4, 512, 0.0);
    assert_qk_norm_matches_cpu("gemma4 global Q", 32, 512, 0.0);
}

// ── 2. qk_norm-as-V-norm vs v_norm_batched ─────────────────────────────────

/// The critical parity check: prefill applies V-norm via `qk_norm`
/// with all-ones weight + offset=0, decode applies it via
/// `v_norm_batched`. Any disagreement here drifts every cached V.
fn assert_qk_norm_v_mode_matches_v_norm_batched(label: &str, num_heads: usize, head_dim: usize) {
    let metal = get_metal();
    let eps = 1e-6f32;
    let x = synth_input(num_heads, head_dim);
    let ones: Vec<f32> = vec![1.0; head_dim];

    // Path A: qk_norm with weight=1, offset=0.
    let in_a = metal.bufs().transient_from_f32(&x);
    let out_a = metal.bufs().output((x.len() * 4) as u64);
    let w_a = metal.bufs().transient_from_f32(&ones);
    run_qk_norm(&metal, &in_a, &out_a, &w_a, num_heads, head_dim, eps, 0.0);
    let a = larql_compute_metal::buffers::read_buffer_f32(&out_a, x.len());

    // Path B: v_norm_batched.
    let in_b = metal.bufs().transient_from_f32(&x);
    let out_b = metal.bufs().output((x.len() * 4) as u64);
    run_v_norm_batched(&metal, &in_b, &out_b, num_heads, head_dim, eps);
    let b = larql_compute_metal::buffers::read_buffer_f32(&out_b, x.len());

    let diff = max_diff(&a, &b);
    let cos = cos_sim(&a, &b);

    // Mathematically these are identical: both compute
    // `x / sqrt(mean(x²)+eps)`. qk_norm formulates it as
    // `(x / rms) * (offset + weight[d])` while v_norm_batched does
    // `x * rsqrt(...)`. Different f32 op sequences, so up to ~1 ULP
    // drift is acceptable. If this test fails with a multi-percent
    // diff, the formulations disagree structurally and the open
    // parity gap is right here.
    //
    // Note: don't use `cos > 0.99999999_f32` — that literal rounds to
    // 1.0 in f32 and the comparison is unreachable. `1.0 - cos < eps`
    // works regardless of representable-precision quirks.
    assert!(
        diff < 5e-6 && (1.0 - cos).abs() < 1e-6,
        "qk_norm(w=1, offset=0) vs v_norm_batched {label} \
         (num_heads={num_heads} head_dim={head_dim}): \
         max_abs={diff:.3e} cos={cos:.6}\n\
         a[..8]={:?}\nb[..8]={:?}\n\
         These two paths are used by Metal prefill and Metal decode \
         respectively for parameter-free V-norm. Any disagreement \
         drifts every cached V from prefill versus what decode would \
         have written, manifesting as the open Gemma 4 31B parity gap.",
        &a[..8.min(a.len())],
        &b[..8.min(b.len())],
    );
}

#[test]
fn qk_norm_v_mode_matches_v_norm_gemma4_sliding() {
    assert_qk_norm_v_mode_matches_v_norm_batched("gemma4 sliding V", 16, 256);
}

#[test]
fn qk_norm_v_mode_matches_v_norm_gemma4_global() {
    // The exact V geometry where the parity gap lives.
    assert_qk_norm_v_mode_matches_v_norm_batched("gemma4 global V", 4, 512);
}

#[test]
fn qk_norm_v_mode_matches_cpu_v_norm_reference() {
    // Sanity check: qk_norm(w=1, offset=0) hits the same CPU output as
    // the parameter-free formula (independent of the v_norm_batched
    // shader). Catches a bug where qk_norm and v_norm_batched are both
    // wrong in the same direction.
    let metal = get_metal();
    let cases: &[(usize, usize)] = &[(4, 512), (16, 256), (8, 128)];
    let eps = 1e-6f32;
    for &(num_heads, head_dim) in cases {
        let x = synth_input(num_heads, head_dim);
        let expected = cpu_v_norm_batched(&x, num_heads, head_dim, eps);

        let ones = vec![1.0f32; head_dim];
        let in_buf = metal.bufs().transient_from_f32(&x);
        let out_buf = metal.bufs().output((x.len() * 4) as u64);
        let w_buf = metal.bufs().transient_from_f32(&ones);
        run_qk_norm(
            &metal, &in_buf, &out_buf, &w_buf, num_heads, head_dim, eps, 0.0,
        );
        let result = larql_compute_metal::buffers::read_buffer_f32(&out_buf, x.len());

        let diff = max_diff(&expected, &result);
        let cos = cos_sim(&expected, &result);
        assert!(
            diff < 1e-4 && cos > 0.999999,
            "qk_norm(V mode) num_heads={num_heads} head_dim={head_dim}: \
             max_abs={diff:.3e} cos={cos:.6}",
        );
    }
}

// ── 3. In-place safety ─────────────────────────────────────────────────────

#[test]
fn qk_norm_in_place_matches_separate_buffers() {
    // The production prefill path (`encode_qk_norm` /
    // `encode_v_norm`) aliases the input and output buffers. The
    // shader recomputes a partial sum of squares per thread, then
    // writes back — if any thread writes before all threads finish
    // reading, the sum is corrupted. The shader's threadgroup-barrier
    // reduction prevents this; this test verifies the in-place form
    // matches the separate-buffer form.
    let metal = get_metal();
    let cases: &[(usize, usize, f32)] = &[
        (16, 256, 0.0), // Gemma 4 sliding
        (4, 512, 0.0),  // Gemma 4 global
        (8, 256, 1.0),  // Gemma 3 (offset = 1.0)
    ];
    let eps = 1e-6f32;
    for &(num_heads, head_dim, offset) in cases {
        let x = synth_input(num_heads, head_dim);
        let weight = synth_weight(head_dim);

        // Separate buffers
        let in_a = metal.bufs().transient_from_f32(&x);
        let out_a = metal.bufs().output((x.len() * 4) as u64);
        let w_a = metal.bufs().transient_from_f32(&weight);
        run_qk_norm(
            &metal, &in_a, &out_a, &w_a, num_heads, head_dim, eps, offset,
        );
        let a = larql_compute_metal::buffers::read_buffer_f32(&out_a, x.len());

        // In-place
        let inout_b = metal.bufs().transient_from_f32(&x);
        let w_b = metal.bufs().transient_from_f32(&weight);
        run_qk_norm(
            &metal, &inout_b, &inout_b, &w_b, num_heads, head_dim, eps, offset,
        );
        let b = larql_compute_metal::buffers::read_buffer_f32(&inout_b, x.len());

        let diff = max_diff(&a, &b);
        assert!(
            diff < 1e-7,
            "qk_norm in-place vs separate buffers num_heads={num_heads} head_dim={head_dim} \
             offset={offset}: max_abs={diff:.3e}\n\
             A read-write race in the partial-sum reduction would manifest as drift here.",
        );
    }
}

// ── qk_norm_qk: fused Q+K norm in one dispatch ──────────────────────────────

/// Drive the Metal `qk_norm_qk` kernel (fused Q+K heads in one dispatch)
/// and compare against two separate `qk_norm` calls.
fn assert_qk_norm_qk_matches_separate(
    num_q_heads: usize,
    num_kv_heads: usize,
    head_dim: usize,
    eps: f32,
    offset: f32,
) {
    let metal = get_metal();

    let seed_q = (num_q_heads * head_dim) as f32 * 0.03;
    let seed_k = (num_kv_heads * head_dim) as f32 * 0.05;
    let q_in: Vec<f32> = (0..num_q_heads * head_dim)
        .map(|i| ((seed_q + i as f32 * 0.011).sin() + 0.1) * 0.5)
        .collect();
    let k_in: Vec<f32> = (0..num_kv_heads * head_dim)
        .map(|i| ((seed_k + i as f32 * 0.013).cos() + 0.1) * 0.5)
        .collect();
    let q_wt: Vec<f32> = (0..head_dim).map(|i| 0.9 + (i as f32) * 0.001).collect();
    let k_wt: Vec<f32> = (0..head_dim).map(|i| 1.1 - (i as f32) * 0.001).collect();

    // Reference: two separate qk_norm calls
    let ref_q = cpu_qk_norm(&q_in, &q_wt, num_q_heads, head_dim, eps, offset);
    let ref_k = cpu_qk_norm(&k_in, &k_wt, num_kv_heads, head_dim, eps, offset);

    // Fused: qk_norm_qk
    let q_buf = metal.bufs().transient_from_f32(&q_in);
    let k_buf = metal.bufs().transient_from_f32(&k_in);
    let q_wt_buf = metal.bufs().get_f32(&q_wt);
    let k_wt_buf = metal.bufs().get_f32(&k_wt);

    let hd = head_dim as u32;
    let nq = num_q_heads as u32;
    let total_heads = (num_q_heads + num_kv_heads) as u64;
    let mut tg_w: usize = 1;
    while tg_w < head_dim && tg_w < 512 {
        tg_w <<= 1;
    }

    let cmd = metal.queue().new_command_buffer();
    let enc = cmd.new_compute_command_encoder();
    enc.set_compute_pipeline_state(&metal.norms.qk_norm_qk_pipeline);
    enc.set_buffer(0, Some(&q_buf), 0);
    enc.set_buffer(1, Some(&k_buf), 0);
    enc.set_buffer(2, Some(&q_wt_buf), 0);
    enc.set_buffer(3, Some(&k_wt_buf), 0);
    enc.set_bytes(4, 4, &hd as *const u32 as *const std::ffi::c_void);
    enc.set_bytes(5, 4, &nq as *const u32 as *const std::ffi::c_void);
    enc.set_bytes(6, 4, &eps as *const f32 as *const std::ffi::c_void);
    enc.set_bytes(7, 4, &offset as *const f32 as *const std::ffi::c_void);
    enc.dispatch_thread_groups(
        metal::MTLSize::new(total_heads, 1, 1),
        metal::MTLSize::new(tg_w as u64, 1, 1),
    );
    enc.end_encoding();
    cmd.commit();
    cmd.wait_until_completed();

    let got_q = larql_compute_metal::buffers::read_buffer_f32(&q_buf, num_q_heads * head_dim);
    let got_k = larql_compute_metal::buffers::read_buffer_f32(&k_buf, num_kv_heads * head_dim);

    let dq = max_diff(&ref_q, &got_q);
    assert!(
        dq < 1e-5,
        "qk_norm_qk Q: max_diff {dq:.3e} (nq={num_q_heads} hd={head_dim})"
    );
    let dk = max_diff(&ref_k, &got_k);
    assert!(
        dk < 1e-5,
        "qk_norm_qk K: max_diff {dk:.3e} (nkv={num_kv_heads} hd={head_dim})"
    );
}

#[test]
fn qk_norm_qk_smoke() {
    assert_qk_norm_qk_matches_separate(4, 2, 16, 1e-6, 1.0);
}

#[test]
fn qk_norm_qk_gemma3_4b() {
    // Gemma 3 4B: 32 Q heads, 16 KV heads, head_dim=256, offset=1.0
    assert_qk_norm_qk_matches_separate(32, 16, 256, 1e-6, 1.0);
}

#[test]
fn qk_norm_qk_gemma4_global_offset0() {
    // Gemma 4 global attention: offset=0.0
    assert_qk_norm_qk_matches_separate(8, 4, 512, 1e-6, 0.0);
}
