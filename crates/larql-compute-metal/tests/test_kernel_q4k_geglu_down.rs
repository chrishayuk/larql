//! Per-kernel tests for the fused GEGLU+down kernels:
//! - `q4k_geglu_silu_down`     (Llama / Mistral / Qwen activation)
//! - `q4k_geglu_gelu_tanh_down` (Gemma / GPT-2 / Phi activation)
//!
//! Both fuse `silu(gate) * up → matmul(W_down)` (or gelu_tanh) into a
//! single dispatch — no intermediate `inter`-sized activation buffer.
//! These were shipped, KernelHandle-wrapped, and contract-tested but
//! **never dispatched** in production until the wiring lands. This
//! file pins the fused kernel byte-equal to the separated path so a
//! future regression is caught at the kernel boundary.
//!
//! Reference (separated path):
//!   1. `geglu_silu` (or `geglu_gelu_tanh`) — element-wise:
//!      `act[i] = silu(gate[i]) * up[i]`
//!   2. `q4k_matvec` — `out[r] = Σᵢ W_down[r,i] * act[i]`
//!
//! Fused:
//!   `out[r] = Σᵢ W_down[r,i] * activation(gate[i]) * up[i]`

#![cfg(target_os = "macos")]

extern crate blas_src;

#[path = "common/mod.rs"]
mod common;
use common::{cos_sim, get_metal, max_diff};

use larql_compute::prelude::*;

fn synth_vec(n: usize, seed: f32) -> Vec<f32> {
    (0..n)
        .map(|i| ((seed + i as f32 * 0.013).sin() + 0.2 * ((i >> 5) as f32).cos()) * 0.4)
        .collect()
}

fn synth_matrix_q4k_friendly(rows: usize, cols: usize, seed: f32) -> Vec<f32> {
    // Q4_K super-blocks are 256 elements. Caller already arranges
    // hidden % 256 == 0; we just generate something whose dynamic
    // range stays within a few blocks' f16 scale precision.
    (0..rows * cols)
        .map(|i| ((seed + i as f32 * 0.001).cos() + 0.3 * ((i >> 8) as f32).sin()) * 0.5)
        .collect()
}

/// Compute the separated reference: `activation(gate) * up → W·x` on
/// CPU. The CPU Q4_K matvec lives on `CpuBackend`; the activation is
/// a few lines of arithmetic.
fn cpu_geglu_then_matvec(
    cpu: &dyn ComputeBackend,
    w_down_q4k: &[u8],
    gate: &[f32],
    up: &[f32],
    silu: bool,
    n: usize,
    inter: usize,
) -> Vec<f32> {
    let mut act = vec![0.0f32; inter];
    for i in 0..inter {
        let g = gate[i];
        let activated = if silu {
            g / (1.0 + (-g).exp())
        } else {
            // GELU-tanh: 0.5·x·(1 + tanh(√(2/π)·(x + 0.044715·x³)))
            let c = 0.797_884_6_f32;
            0.5 * g * (1.0 + (c * (g + 0.044715 * g * g * g)).tanh())
        };
        act[i] = activated * up[i];
    }
    cpu.q4k_matvec(w_down_q4k, &act, n, inter).unwrap()
}

/// Drive the fused kernel and return the f32 output vector.
fn metal_fused_geglu_down(
    metal: &larql_compute_metal::MetalBackend,
    w_down_q4k: &[u8],
    gate: &[f32],
    up: &[f32],
    silu: bool,
    n: usize,
    inter: usize,
) -> Vec<f32> {
    use larql_compute_metal::shaders::q4k_geglu_down as gd;
    let kernel = if silu {
        &metal.ffn.q4k_geglu_silu_down_pipeline
    } else {
        &metal.ffn.q4k_geglu_gelu_tanh_down_pipeline
    };

    let w_buf = metal.bufs().get_bytes(w_down_q4k);
    let gate_buf = metal.bufs().transient_from_f32(gate);
    let up_buf = metal.bufs().transient_from_f32(up);
    let out_buf = metal.bufs().output((n * 4) as u64);

    let n_val = n as u32;
    let k_val = inter as u32;
    let num_tgs = (n as u64).div_ceil(gd::ROWS_PER_TG);

    let cmd = metal.queue().new_command_buffer();
    let enc = cmd.new_compute_command_encoder();
    enc.set_compute_pipeline_state(&kernel.state);
    enc.set_buffer(0, Some(&w_buf), 0);
    enc.set_buffer(1, Some(&gate_buf), 0);
    enc.set_buffer(2, Some(&up_buf), 0);
    enc.set_buffer(3, Some(&out_buf), 0);
    enc.set_bytes(4, 4, &n_val as *const u32 as *const std::ffi::c_void);
    enc.set_bytes(5, 4, &k_val as *const u32 as *const std::ffi::c_void);
    enc.dispatch_thread_groups(
        metal::MTLSize::new(num_tgs, 1, 1),
        metal::MTLSize::new(gd::THREADS_PER_TG, 1, 1),
    );
    enc.end_encoding();
    cmd.commit();
    cmd.wait_until_completed();
    larql_compute_metal::buffers::read_buffer_f32(&out_buf, n)
}

/// Run the fused-vs-separated parity test for one geometry + activation.
fn assert_fused_geglu_down_matches_separated(label: &str, n: usize, inter: usize, silu: bool) {
    assert_eq!(inter % 256, 0, "Q4_K requires inter divisible by 256");
    let metal = get_metal();
    let cpu = larql_compute::cpu::CpuBackend;

    let down_f32 = synth_matrix_q4k_friendly(n, inter, 0.21);
    let gate = synth_vec(inter, 0.41);
    let up = synth_vec(inter, 0.83);
    let down_q4k = larql_compute::cpu::ops::q4_common::quantize_q4_k(&down_f32);

    let cpu_ref = cpu_geglu_then_matvec(&cpu, &down_q4k, &gate, &up, silu, n, inter);
    let fused = metal_fused_geglu_down(&metal, &down_q4k, &gate, &up, silu, n, inter);

    // Q4_K + activation accumulation is lossy — same threshold the
    // existing `q4k_matvec_matches_cpu` uses (cos > 0.999, max_abs
    // < 0.5 on similar-scale inputs).
    let cos = cos_sim(&cpu_ref, &fused);
    let diff = max_diff(&cpu_ref, &fused);
    assert!(
        cos > 0.999 && diff < 0.5,
        "{label} ({}): max_abs={diff:.3e} cos={cos:.6}",
        if silu { "silu" } else { "gelu_tanh" },
    );

    // Sanity: outputs are non-zero. Catches a "wrote nothing" bug
    // (the q4_matvec_v4 75 %-row drop class).
    let nonzero = fused.iter().filter(|&&v| v.abs() > 1e-6).count();
    assert!(
        nonzero > n / 10,
        "{label}: only {nonzero}/{n} fused rows non-zero — possible row-drop regression"
    );
}

#[test]
fn q4k_geglu_silu_down_smoke() {
    assert_fused_geglu_down_matches_separated("smoke 256→32", 32, 256, true);
}

#[test]
fn q4k_geglu_gelu_tanh_down_smoke() {
    assert_fused_geglu_down_matches_separated("smoke 256→32", 32, 256, false);
}

/// Production geometry (Gemma 3 4B FFN down): hidden=2560,
/// inter=10240. The path the wiring will hit on every layer of every
/// decode token.
#[test]
fn q4k_geglu_silu_down_gemma3_4b_ffn() {
    assert_fused_geglu_down_matches_separated("gemma3-4b ffn (silu)", 2560, 10240, true);
}

#[test]
fn q4k_geglu_gelu_tanh_down_gemma3_4b_ffn() {
    assert_fused_geglu_down_matches_separated("gemma3-4b ffn (gelu_tanh)", 2560, 10240, false);
}

/// Larger geometry (Gemma 4 31B sliding FFN): hidden=5376,
/// inter=21504. Catches "shader sized for K=4096" type bugs at scale.
#[test]
fn q4k_geglu_silu_down_gemma4_31b_ffn() {
    assert_fused_geglu_down_matches_separated("gemma4-31b ffn (silu)", 5376, 21504, true);
}

#[test]
fn q4k_geglu_gelu_tanh_down_gemma4_31b_ffn() {
    assert_fused_geglu_down_matches_separated("gemma4-31b ffn (gelu_tanh)", 5376, 21504, false);
}
