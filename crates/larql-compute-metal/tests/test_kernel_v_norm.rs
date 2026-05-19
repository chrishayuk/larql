#![cfg(target_os = "macos")]

//! Per-kernel tests for `v_norm_batched` — the parameter-free RMSNorm
//! used by Gemma 4's V-projection inside KV-cached decode.
//!
//! Why a focused file: `v_norm_batched` had two independent latent
//! bugs that only surfaced under specific shapes / call patterns:
//!
//! 1. **Heads > 1 silently dropped.** The original shader used
//!    `[[thread_position_in_grid]]: uint2` with a 2D dispatch, and on
//!    M3 only the first TG along Y actually wrote results — heads
//!    1..N stayed at the buffer's initial state (zero). Caught here
//!    by the `_all_ones_4x256` test: post-shader, indices 256+ were
//!    still 0.0.
//! 2. **In-place RMW race.** Production decode runs the shader with
//!    `x` and `out` aliased. Each thread re-reading the full head for
//!    `sum_sq` while other threads are mid-write produces drifted
//!    output. Caught by the `_in_place_matches_reference` test.
//!
//! Both fixed by switching to one TG per head + threadgroup-shared
//! `tg_partial[]` reduction with an explicit barrier between the read
//! and write phases (mirrors `qk_norm`'s structure). See
//! `metal/shaders/v_norm.rs`.

extern crate blas_src;

#[path = "common/mod.rs"]
mod common;
use common::{get_metal, max_diff};

/// Reference: per-head parameter-free RMSNorm.
fn cpu_v_norm_batched_reference(
    x: &[f32],
    num_heads: usize,
    head_dim: usize,
    eps: f32,
) -> Vec<f32> {
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

/// Drive `v_norm_batched` exactly the way `metal/decode/mod.rs` does:
/// one threadgroup per head along X; tg width is the next power of two
/// ≤ 512 for the in-shader tree reduction.
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
    let mut tg_w: u64 = 1;
    while tg_w < head_dim as u64 && tg_w < 512 {
        tg_w <<= 1;
    }

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

#[test]
fn all_ones_4x256_writes_every_head() {
    // Minimal smoke test: 4 heads × 256 dims, all-ones input. Each
    // head's RMS = 1.0, so output should also be ~1.0 everywhere.
    // The pre-fix shader silently left heads 1-3 at 0.0 (only head 0
    // got dispatched on M3 with the 2D `dispatch_threads` form).
    let metal = get_metal();
    let num_heads = 4usize;
    let head_dim = 256usize;
    let n = num_heads * head_dim;
    let x = vec![1.0f32; n];
    let eps = 1e-6f32;

    let x_buf = metal.bufs().transient_from_f32(&x);
    let out_buf = metal.bufs().output((n * 4) as u64);
    run_v_norm_batched(&metal, &x_buf, &out_buf, num_heads, head_dim, eps);

    let result = larql_compute_metal::buffers::read_buffer_f32(&out_buf, n);
    let expected = vec![1.0f32; n];
    let diff = max_diff(&expected, &result);

    // Locate first non-1.0 entry — useful when the bug regresses to
    // "head 0 fine, head 1+ zeros".
    let mut first_bad: Option<(usize, f32)> = None;
    for (i, &v) in result.iter().enumerate() {
        if (v - 1.0).abs() > 1e-3 {
            first_bad = Some((i, v));
            break;
        }
    }
    assert!(
        diff < 1e-4,
        "v_norm_batched(4×256, all-ones) max diff {diff:.3e}; \
         first non-1.0 at index {first_bad:?}; \
         heads 1-3 unwritten suggests the historical 2D-dispatch + \
         `tid.y = 0`-on-M3 bug has regressed.",
    );
}

#[test]
fn separate_buffers_match_reference_across_shapes() {
    // No aliasing — pure correctness check across the geometries we
    // actually run in production. (16, 256) is Gemma 4 31B sliding
    // L0; (4, 512) is Gemma 4 31B global L5 — the head_dim=512 case
    // historically tripped 256-thread-TG kernels (`fused_attention`
    // shipped a similar bug; see `fused_attention_head_dim_512`).
    let metal = get_metal();
    let cases: &[(usize, usize)] = &[(1, 64), (4, 256), (16, 256), (4, 512), (8, 128)];
    let eps = 1e-6f32;
    for &(num_heads, head_dim) in cases {
        let n = num_heads * head_dim;
        let x: Vec<f32> = (0..n)
            .map(|i| ((i as f32 * 0.013).sin() + 0.3 * ((i >> 5) as f32).cos()) * 0.4)
            .collect();
        let expected = cpu_v_norm_batched_reference(&x, num_heads, head_dim, eps);

        let x_buf = metal.bufs().transient_from_f32(&x);
        let out_buf = metal.bufs().output((n * 4) as u64);
        run_v_norm_batched(&metal, &x_buf, &out_buf, num_heads, head_dim, eps);

        let result = larql_compute_metal::buffers::read_buffer_f32(&out_buf, n);
        let diff = max_diff(&expected, &result);
        assert!(
            diff < 1e-4,
            "v_norm_batched (separate) num_heads={num_heads} head_dim={head_dim} \
             max diff {diff} exceeds 1e-4",
        );
    }
}

#[test]
fn in_place_matches_separate_buffer_reference() {
    // Production decode passes the same buffer for both `x` and `out`.
    // The shader recomputes `sum_sq` per thread by re-reading `x`; if
    // any thread starts writing before another finishes the read loop,
    // sum_sq is corrupted. Fixed by the threadgroup-barrier reduction.
    let metal = get_metal();
    let cases: &[(usize, usize)] = &[
        (16, 256), // Gemma 4 31B sliding L0
        (4, 512),  // Gemma 4 31B global L5+
    ];
    let eps = 1e-6f32;
    for &(num_heads, head_dim) in cases {
        let n = num_heads * head_dim;
        let x: Vec<f32> = (0..n)
            .map(|i| ((i as f32 * 0.013).sin() + 0.3 * ((i >> 5) as f32).cos()) * 0.4)
            .collect();
        let expected = cpu_v_norm_batched_reference(&x, num_heads, head_dim, eps);

        let inout_buf = metal.bufs().transient_from_f32(&x);
        run_v_norm_batched(&metal, &inout_buf, &inout_buf, num_heads, head_dim, eps);

        let result = larql_compute_metal::buffers::read_buffer_f32(&inout_buf, n);
        let diff = max_diff(&expected, &result);
        assert!(
            diff < 1e-4,
            "v_norm_batched (IN-PLACE) num_heads={num_heads} head_dim={head_dim} \
             max diff {diff} exceeds 1e-4 — race between threads in the \
             reduction phase and threads writing the output back to the \
             same buffer.",
        );
    }
}
