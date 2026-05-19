#![cfg(target_os = "macos")]

//! Per-kernel tests for `rope_at_pos` — the *single-head, single-vector*
//! RoPE shader used by Metal prefill via `metal/stages/rope.rs`. Looped
//! per-position per-head into one encoder.
//!
//! ## Why a focused file
//!
//! `test_kernel_rope` pins `rope_at_pos_batched` (the decode-time form
//! that rotates every head at one position in a single dispatch) and
//! `test_metal_shaders::rope_apply*` cover `rope_apply` (the
//! multi-position, in-place shader). Neither covers `rope_at_pos`,
//! which sits *between* those two — used only by Metal prefill when
//! the KV cache is populated, since the cache-write path needs RoPE'd
//! K and Q out of the projection step instead of folded into the
//! attention shader.
//!
//! That makes it the next suspect for the open
//! `decode_consistency_gemma4_31b_dense` parity gap: prefill RoPE'd K
//! lands in the cache; decode RoPE'd K lands at position N; if the two
//! shaders disagree at the Gemma 4 31B global geometry (head_dim=512,
//! rotary_dim=128), every cached K from prefill is subtly different
//! from what decode would have written, drifting all downstream
//! attention.
//!
//! ## What it asserts
//!
//! For each production geometry:
//!   - Run `rope_at_pos` against a CPU split-half reference.
//!   - Assert per-vector cos > 0.999999 and max_abs < 1e-4.
//!
//! Geometries:
//!   - Llama-2 7B / Mistral 7B (head_dim=128, full rotation, base=10000)
//!   - Gemma 3 4B (head_dim=256, full rotation, base=10000)
//!   - Gemma 4 31B sliding (head_dim=256, full rotation, base=10000)
//!   - **Gemma 4 31B global (head_dim=512, partial 25%, base=500000)**
//!     — the still-open parity-gap geometry.
//!
//! ## Reference
//!
//! Llama-style split-half rotation: pair `(x[i], x[i + rdim/2])`
//! rotated by angle `pos * freq(i)` where `freq(i) = 1/base^(2i/rdim)`.
//! Dims past `rotary_dim` pass through unchanged.

extern crate blas_src;

#[path = "common/mod.rs"]
mod common;
use common::{cos_sim, get_metal, max_diff};

/// CPU reference: split-half RoPE on a single head, in place.
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

/// Dispatch `rope_at_pos` once at the given offset. The shader rotates
/// `rotary_dim/2` pairs (one thread per pair) within a single head.
#[allow(clippy::too_many_arguments)]
fn run_rope_at_pos(
    metal: &larql_compute_metal::MetalBackend,
    x: &[f32],
    head_dim: usize,
    rotary_dim: usize,
    base: f32,
    pos: usize,
) -> Vec<f32> {
    assert_eq!(x.len(), head_dim);
    let buf = metal.bufs().transient_from_f32(x);

    let hd = head_dim as u32;
    let rd_val = rotary_dim as u32;
    let pos_val = pos as u32;
    let rdim_eff = if rotary_dim == 0 {
        head_dim
    } else {
        rotary_dim
    };
    let pairs = (rdim_eff / 2) as u64;

    let cmd = metal.queue().new_command_buffer();
    let enc = cmd.new_compute_command_encoder();
    enc.set_compute_pipeline_state(&metal.attention.rope_at_pos_pipeline);
    enc.set_buffer(0, Some(&buf), 0);
    enc.set_bytes(1, 4, &hd as *const u32 as *const std::ffi::c_void);
    enc.set_bytes(2, 4, &base as *const f32 as *const std::ffi::c_void);
    enc.set_bytes(3, 4, &pos_val as *const u32 as *const std::ffi::c_void);
    enc.set_bytes(4, 4, &rd_val as *const u32 as *const std::ffi::c_void);
    enc.dispatch_threads(
        metal::MTLSize::new(pairs, 1, 1),
        metal::MTLSize::new(pairs.min(256), 1, 1),
    );
    enc.end_encoding();
    cmd.commit();
    cmd.wait_until_completed();

    larql_compute_metal::buffers::read_buffer_f32(&buf, head_dim)
}

#[allow(clippy::too_many_arguments)]
fn assert_rope_at_pos_matches_cpu(
    label: &str,
    head_dim: usize,
    rotary_dim: usize,
    base: f32,
    pos: usize,
) {
    let metal = get_metal();
    let x: Vec<f32> = (0..head_dim)
        .map(|i| ((i as f32 * 0.011).sin() + 0.4 * ((i >> 4) as f32).cos()) * 0.5)
        .collect();

    let mut expected = x.clone();
    cpu_rope_at_pos(head_dim, rotary_dim, base, pos, &mut expected);

    let result = run_rope_at_pos(&metal, &x, head_dim, rotary_dim, base, pos);

    let diff = max_diff(&expected, &result);
    let cos = cos_sim(&expected, &result);
    assert!(
        diff < 1e-4 && cos > 0.999999,
        "rope_at_pos {label} (head_dim={head_dim} rotary_dim={rotary_dim} \
         base={base} pos={pos}): max_abs={diff:.3e} cos={cos:.6}",
    );

    // Also assert pass-through dims (those past rotary_dim) are
    // untouched. A bug that loops past `rdim` would manifest end-to-end
    // as silent drift on partial-rotary geometries (Gemma 4 global).
    let rdim_eff = if rotary_dim == 0 {
        head_dim
    } else {
        rotary_dim.min(head_dim)
    };
    if rdim_eff < head_dim {
        for d in rdim_eff..head_dim {
            let delta = (result[d] - x[d]).abs();
            assert!(
                delta < 1e-7,
                "rope_at_pos {label}: pass-through dim {d} changed (was {}, now {} delta {delta:.3e}). \
                 Indicates the kernel rotated past `rotary_dim`, which would silently shift the \
                 unrotated tail of every head on partial-rotary geometries.",
                x[d], result[d],
            );
        }
    }
}

#[test]
fn rope_at_pos_llama2_full() {
    // 128-dim head, full rotation, standard base. Same geometry as
    // Llama-2 7B / Mistral 7B / TinyLlama / etc. Position set matches
    // the sibling `test_kernel_rope` to keep the two suites moving in
    // lockstep — high-pos divergence is `Metal::pow` vs Rust `powf`
    // float precision noise, not a kernel bug.
    for &pos in &[0usize, 1, 5, 17] {
        assert_rope_at_pos_matches_cpu("llama2 full", 128, 0, 10_000.0, pos);
    }
}

#[test]
fn rope_at_pos_gemma3_full_256() {
    // Gemma 3 4B: 256-dim head, full rotation.
    for &pos in &[0usize, 7, 23] {
        assert_rope_at_pos_matches_cpu("gemma3 full 256", 256, 0, 10_000.0, pos);
    }
}

#[test]
fn rope_at_pos_gemma4_sliding() {
    // Gemma 4 31B sliding layer: 256-dim head, full rotation, base=10000.
    for &pos in &[0usize, 17, 100] {
        assert_rope_at_pos_matches_cpu("gemma4 sliding", 256, 0, 10_000.0, pos);
    }
}

#[test]
fn rope_at_pos_gemma4_global_partial() {
    // **The decode-bug suspect geometry.**
    //
    // Gemma 4 31B global layers: 512-dim head, 25 % partial rotation
    // (rotary_dim=128), rope_base=500000. This is the exact shape
    // where end-to-end parity fails on the open
    // `decode_consistency_gemma4_31b_dense` test. If `rope_at_pos`
    // (prefill stage) and `rope_at_pos_batched` (decode stage)
    // disagree here, every cached K from prefill is subtly off versus
    // what decode would have written, and the parity test fails.
    for &pos in &[0usize, 17, 100] {
        assert_rope_at_pos_matches_cpu("gemma4 global partial", 512, 128, 500_000.0, pos);
    }
}

#[test]
fn rope_at_pos_partial_pass_through_preserved() {
    // Stress the pass-through tail: half-rotation on a 128-dim head.
    // Dims [64..128) must come back bit-equal to the input. A previous
    // version of `rope_apply` once rotated the whole head when
    // `rotary_dim=0` was passed via a typo-path; an analogous bug here
    // would silently fail end-to-end without this check.
    for &pos in &[0usize, 5, 23] {
        assert_rope_at_pos_matches_cpu("half-rotation pass-through", 128, 64, 10_000.0, pos);
    }
}

#[test]
fn rope_at_pos_matches_rope_at_pos_batched_one_head() {
    // The two shaders should produce *identical* output for the same
    // single-head input at the same position. Discrepancies here are
    // the most likely sole-cause of the open Gemma 4 31B parity gap:
    // prefill writes K via rope_at_pos, decode writes K via
    // rope_at_pos_batched; if they disagree at head_dim=512 / partial
    // 128 / base=500000, the cache contents from prefill don't match
    // the freshly-RoPE'd K decode would have written.
    let metal = get_metal();
    let head_dim = 512usize;
    let rotary_dim = 128usize;
    let base = 500_000.0f32;
    let pos = 17usize;

    let x: Vec<f32> = (0..head_dim)
        .map(|i| ((i as f32 * 0.011).sin() + 0.4 * ((i >> 4) as f32).cos()) * 0.5)
        .collect();

    // rope_at_pos (prefill stage)
    let single = run_rope_at_pos(&metal, &x, head_dim, rotary_dim, base, pos);

    // rope_at_pos_batched (decode stage) — drive with one head.
    let buf = metal.bufs().transient_from_f32(&x);
    let hd = head_dim as u32;
    let rd_val = rotary_dim as u32;
    let nh = 1u32;
    let pos_val = pos as u32;
    let pairs = (rotary_dim / 2) as u64;
    let cmd = metal.queue().new_command_buffer();
    let enc = cmd.new_compute_command_encoder();
    enc.set_compute_pipeline_state(&metal.attention.rope_at_pos_batched_pipeline);
    enc.set_buffer(0, Some(&buf), 0);
    enc.set_bytes(1, 4, &hd as *const u32 as *const std::ffi::c_void);
    enc.set_bytes(2, 4, &base as *const f32 as *const std::ffi::c_void);
    enc.set_bytes(3, 4, &pos_val as *const u32 as *const std::ffi::c_void);
    enc.set_bytes(4, 4, &rd_val as *const u32 as *const std::ffi::c_void);
    enc.set_bytes(5, 4, &nh as *const u32 as *const std::ffi::c_void);
    enc.dispatch_threads(
        metal::MTLSize::new(pairs, 1, 1),
        metal::MTLSize::new(pairs.min(256), 1, 1),
    );
    enc.end_encoding();
    cmd.commit();
    cmd.wait_until_completed();
    let batched = larql_compute_metal::buffers::read_buffer_f32(&buf, head_dim);

    let diff = max_diff(&single, &batched);
    let cos = cos_sim(&single, &batched);
    // Bit-equality is the right bar here: same formula, same f32
    // intermediate ops on the same hardware.
    assert!(
        diff == 0.0 && cos == 1.0,
        "rope_at_pos vs rope_at_pos_batched (gemma4 global, single head) diverge: \
         max_abs={diff:.3e} cos={cos:.6}\n\
         single[..8]={:?}\nbatched[..8]={:?}\n\
         These shaders must produce identical output — they implement \
         the same formula on the same input. Any difference is the \
         direct cause of `decode_consistency_gemma4_31b_dense`.",
        &single[..8],
        &batched[..8],
    );
}
