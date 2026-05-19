#![cfg(target_os = "macos")]

//! Per-kernel tests for `q4k_ffn_gate_up` — the fused gate+up matvec
//! that runs once per layer in production Q4_K decode.
//!
//! ## Why a focused file
//!
//! Production Q4_K decode (`metal/decode/mod.rs`) dispatches this
//! shader exactly once per layer, with the layer's quantized
//! gate and up weights and the post-norm hidden as input. It produces
//! both `gate_out` and `up_out` in one dispatch by loading the input
//! into shared memory and striding rows of the two matrices into
//! parallel threadgroups.
//!
//! Coverage today: `multi_position_q4k_matches_individual` exercises
//! the regular `q4k_matvec` shader at multiple positions, but neither
//! that test nor any other pins `q4k_ffn_gate_up` directly. A
//! regression in the fused form (mismatched threadgroup count, the
//! `is_up` partition off by one, shared-memory overflow at large
//! `hidden`) would only show up end-to-end as nonsense FFN output.
//!
//! ## What it asserts
//!
//! For each (inter, hidden) production geometry:
//!   - Synth distinct gate/up f32 matrices, Q4_K-quantize each.
//!   - Run `q4k_ffn_gate_up` against a synthetic f32 input.
//!   - Compare each output against an independent CPU `q4k_matvec` of
//!     the same Q4_K bytes — i.e. the fused kernel must produce the
//!     same output its sibling single-matrix kernel does.
//!
//! Geometries:
//!   - Gemma 3 4B (hidden=2560, inter=10240) — production Q4_K decode
//!   - Gemma 4 31B sliding (hidden=5376, inter=21504) — large
//!   - Tight smoke (hidden=256, inter=64) — the smallest valid shape

extern crate blas_src;

#[path = "common/mod.rs"]
mod common;
use common::{cos_sim, get_metal, max_diff};

use larql_compute::prelude::*;

fn synth_matrix(rows: usize, cols: usize, seed: f32) -> Vec<f32> {
    (0..rows * cols)
        .map(|i| ((seed + i as f32 * 0.001).cos() + 0.3 * ((i >> 8) as f32).sin()) * 0.5)
        .collect()
}

fn synth_input(hidden: usize, seed: f32) -> Vec<f32> {
    (0..hidden)
        .map(|i| ((seed + i as f32 * 0.013).sin() + 0.2 * ((i >> 5) as f32).cos()) * 0.4)
        .collect()
}

/// Drive `q4k_ffn_gate_up` against a CPU `q4k_matvec` reference for
/// each output matrix.
fn assert_q4k_ffn_gate_up_matches_per_matrix(label: &str, inter: usize, hidden: usize) {
    assert_eq!(hidden % 256, 0, "Q4_K requires hidden divisible by 256");
    let metal = get_metal();
    let cpu = larql_compute::cpu::CpuBackend;

    // Distinct gate / up matrices so a "wrote up to gate's slot" bug
    // shows up as the wrong matrix in the wrong half of the output.
    let gate = synth_matrix(inter, hidden, 0.21);
    let up = synth_matrix(inter, hidden, 0.83);
    let x = synth_input(hidden, 0.41);

    let gate_q4k = larql_compute::cpu::ops::q4_common::quantize_q4_k(&gate);
    let up_q4k = larql_compute::cpu::ops::q4_common::quantize_q4_k(&up);

    // CPU references — independent matvecs, one per matrix.
    let gate_cpu = cpu.q4k_matvec(&gate_q4k, &x, inter, hidden).unwrap();
    let up_cpu = cpu.q4k_matvec(&up_q4k, &x, inter, hidden).unwrap();

    // Metal: one fused dispatch.
    use larql_compute_metal::shaders::q4k_ffn_gate_up as gu;
    let gate_w_buf = metal.bufs().get_bytes(&gate_q4k);
    let up_w_buf = metal.bufs().get_bytes(&up_q4k);
    let x_buf = metal.bufs().transient_from_f32(&x);
    let gate_out_buf = metal.bufs().output((inter * 4) as u64);
    let up_out_buf = metal.bufs().output((inter * 4) as u64);

    let n_val = inter as u32;
    let k_val = hidden as u32;
    let n_tgs_per_mat = (inter as u64).div_ceil(gu::ROWS_PER_TG);

    let cmd = metal.queue().new_command_buffer();
    let enc = cmd.new_compute_command_encoder();
    enc.set_compute_pipeline_state(&metal.ffn.q4k_ffn_gate_up_pipeline.state);
    enc.set_buffer(0, Some(&gate_w_buf), 0);
    enc.set_buffer(1, Some(&up_w_buf), 0);
    enc.set_buffer(2, Some(&x_buf), 0);
    enc.set_buffer(3, Some(&gate_out_buf), 0);
    enc.set_buffer(4, Some(&up_out_buf), 0);
    enc.set_bytes(5, 4, &n_val as *const u32 as *const std::ffi::c_void);
    enc.set_bytes(6, 4, &k_val as *const u32 as *const std::ffi::c_void);
    enc.dispatch_thread_groups(
        metal::MTLSize::new(n_tgs_per_mat * 2, 1, 1),
        metal::MTLSize::new(gu::THREADS_PER_TG, 1, 1),
    );
    enc.end_encoding();
    cmd.commit();
    cmd.wait_until_completed();

    let gate_metal = larql_compute_metal::buffers::read_buffer_f32(&gate_out_buf, inter);
    let up_metal = larql_compute_metal::buffers::read_buffer_f32(&up_out_buf, inter);

    // Metal Q4_K matvec and CPU Q4_K matvec are not bit-equal due to
    // f16 dequantization rounding, so use cos + max_diff with the
    // same threshold as `q4k_matvec_matches_cpu` (0.5 on similar
    // scale inputs) — but since this is the FUSED kernel against the
    // SINGLE kernel through Metal, we should also see the fused vs
    // separate-Metal-dispatch be much tighter. Cover both bars.
    let gate_diff = max_diff(&gate_cpu, &gate_metal);
    let gate_cos = cos_sim(&gate_cpu, &gate_metal);
    assert!(
        gate_diff < 0.5 && gate_cos > 0.999,
        "q4k_ffn_gate_up {label} GATE row: max_abs={gate_diff:.3e} cos={gate_cos:.6}",
    );

    let up_diff = max_diff(&up_cpu, &up_metal);
    let up_cos = cos_sim(&up_cpu, &up_metal);
    assert!(
        up_diff < 0.5 && up_cos > 0.999,
        "q4k_ffn_gate_up {label} UP row: max_abs={up_diff:.3e} cos={up_cos:.6}",
    );

    // Matrices are distinct, so gate output must NOT match up output.
    // Catches "wrote both halves to gate" / "ignored is_up flag" bugs.
    let gate_up_diff = max_diff(&gate_metal, &up_metal);
    assert!(
        gate_up_diff > 0.01,
        "q4k_ffn_gate_up {label}: gate_metal and up_metal nearly equal \
         (max_abs_between={gate_up_diff:.3e}). Indicates the kernel's \
         `is_up` flag isn't routing to distinct weight matrices.",
    );
}

#[test]
fn q4k_ffn_gate_up_smoke_256x64() {
    assert_q4k_ffn_gate_up_matches_per_matrix("smoke 256→64", 64, 256);
}

#[test]
fn q4k_ffn_gate_up_gemma3_4b() {
    // Gemma 3 4B: hidden=2560, inter=10240 — the production decode
    // shape this kernel runs at on every layer, every token.
    assert_q4k_ffn_gate_up_matches_per_matrix("gemma3-4b", 10240, 2560);
}

#[test]
fn q4k_ffn_gate_up_gemma4_26b_a4b_moe_shape() {
    // Gemma 4 26B-A4B MoE expert shape: hidden=2816, inter=704.
    // Pins the primitive suspected by the Metal MoE dispatch bug before
    // exercising the larger multi-expert dispatch chain.
    assert_q4k_ffn_gate_up_matches_per_matrix("gemma4-26b-a4b moe", 704, 2816);
}

#[test]
fn q4k_ffn_gate_up_max_k_boundary_4096() {
    // Right at the shader's Q4K_GU_MAX_K=4096 shared-memory cap. Should
    // pass — the threadgroup tile fits exactly. Anything past this is
    // out-of-bounds shared-memory access (Metal UB).
    assert_q4k_ffn_gate_up_matches_per_matrix("at MAX_K (4096)", 32, 4096);
}

/// Regression for the previously-broken shared-memory-cap bug. The
/// shader used to hard-code `Q4K_GU_MAX_K = 4096` and silently
/// produce garbage at any K > 4096; the fix dropped the threadgroup
/// `Xsh[]` tile and reads X directly from device memory (mirroring
/// `q4k_qkv_proj` which has always used that pattern). One
/// super-block past the old cap exercises the previously-broken
/// path.
#[test]
fn q4k_ffn_gate_up_just_past_max_k_4352() {
    assert_q4k_ffn_gate_up_matches_per_matrix("past MAX_K (4352)", 32, 4352);
}

/// Production Gemma 4 31B geometry (hidden=5376, inter=21504). With
/// the old `Xsh[]` tile this collapsed to `cos ≈ -0.08`; with the
/// direct-read fix it matches CPU at the standard Q4_K matvec
/// threshold. Pins the shader against any future regression of the
/// shared-memory-cap bug.
#[test]
fn q4k_ffn_gate_up_gemma4_31b_dense() {
    assert_q4k_ffn_gate_up_matches_per_matrix("gemma4-31b dense", 21504, 5376);
}

#[test]
fn q4k_ffn_gate_up_zero_input() {
    // Zero input → zero output (both gate and up). Sanity check that
    // the shared-memory load + per-row matvec produce no NaNs on
    // degenerate input. A bug like accumulating into uninitialised
    // shared memory would surface as nonzero out here.
    let metal = get_metal();
    let inter = 64usize;
    let hidden = 256usize;

    let gate = synth_matrix(inter, hidden, 0.11);
    let up = synth_matrix(inter, hidden, 0.71);
    let x = vec![0.0f32; hidden];
    let gate_q4k = larql_compute::cpu::ops::q4_common::quantize_q4_k(&gate);
    let up_q4k = larql_compute::cpu::ops::q4_common::quantize_q4_k(&up);

    use larql_compute_metal::shaders::q4k_ffn_gate_up as gu;
    let gate_w_buf = metal.bufs().get_bytes(&gate_q4k);
    let up_w_buf = metal.bufs().get_bytes(&up_q4k);
    let x_buf = metal.bufs().transient_from_f32(&x);
    let gate_out_buf = metal.bufs().output((inter * 4) as u64);
    let up_out_buf = metal.bufs().output((inter * 4) as u64);

    let n_val = inter as u32;
    let k_val = hidden as u32;
    let n_tgs_per_mat = (inter as u64).div_ceil(gu::ROWS_PER_TG);

    let cmd = metal.queue().new_command_buffer();
    let enc = cmd.new_compute_command_encoder();
    enc.set_compute_pipeline_state(&metal.ffn.q4k_ffn_gate_up_pipeline.state);
    enc.set_buffer(0, Some(&gate_w_buf), 0);
    enc.set_buffer(1, Some(&up_w_buf), 0);
    enc.set_buffer(2, Some(&x_buf), 0);
    enc.set_buffer(3, Some(&gate_out_buf), 0);
    enc.set_buffer(4, Some(&up_out_buf), 0);
    enc.set_bytes(5, 4, &n_val as *const u32 as *const std::ffi::c_void);
    enc.set_bytes(6, 4, &k_val as *const u32 as *const std::ffi::c_void);
    enc.dispatch_thread_groups(
        metal::MTLSize::new(n_tgs_per_mat * 2, 1, 1),
        metal::MTLSize::new(gu::THREADS_PER_TG, 1, 1),
    );
    enc.end_encoding();
    cmd.commit();
    cmd.wait_until_completed();

    let gate_metal = larql_compute_metal::buffers::read_buffer_f32(&gate_out_buf, inter);
    let up_metal = larql_compute_metal::buffers::read_buffer_f32(&up_out_buf, inter);

    let gate_max = gate_metal.iter().fold(0.0f32, |a, &v| a.max(v.abs()));
    let up_max = up_metal.iter().fold(0.0f32, |a, &v| a.max(v.abs()));
    assert!(
        gate_max < 1e-3 && up_max < 1e-3,
        "q4k_ffn_gate_up zero-input: gate_max={gate_max:.3e} up_max={up_max:.3e} (should be ~0)",
    );
    assert!(
        !gate_metal.iter().any(|v| v.is_nan()),
        "q4k_ffn_gate_up zero-input: gate output contains NaN"
    );
    assert!(
        !up_metal.iter().any(|v| v.is_nan()),
        "q4k_ffn_gate_up zero-input: up output contains NaN"
    );
}
