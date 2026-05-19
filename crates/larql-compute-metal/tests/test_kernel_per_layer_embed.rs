//! Correctness tests for the `ple_gate_apply` Metal shader (D-METAL-PLE).
//!
//! `ple_gate_apply` fuses `gate_out[i] = gelu_tanh(gate_in[i]) * per_layer_input[i]`
//! into one dispatch — the activation+multiply step in the middle of the
//! Per-Layer Embeddings block (Gemma 4 E2B). This test pins the kernel
//! against a CPU reference using the same `gelu_tanh` constants the
//! standalone `gelu_tanh` shader and CPU `apply_per_layer_embedding` use.
//!
//! Why a kernel-level test (not just an end-to-end parity run): the
//! gate-apply fusion is the only PLE-specific kernel.  The two matvecs
//! reuse `f32_gemv` (already covered by `test_kernel_lm_head_gemv.rs`)
//! and the closing norm+residual reuses `post_ffn_norm_residual_add`
//! (covered by the existing fused-norm tests).  This file is the only
//! one that catches per-element math drift in the new shader.

#![cfg(target_os = "macos")]

extern crate blas_src;

#[path = "common/mod.rs"]
mod common;
use common::max_diff;

/// CPU reference: `gate_out[i] = 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x³))) * pli[i]`.
/// Mirrors `crates/larql-inference/src/forward/ple.rs::apply_per_layer_embedding`'s
/// inner activation loop, multiplied through with the per-layer input.
fn ple_gate_apply_cpu(gate_in: &[f32], per_layer_input: &[f32]) -> Vec<f32> {
    let sqrt_2_over_pi = (2.0f32 / std::f32::consts::PI).sqrt();
    gate_in
        .iter()
        .zip(per_layer_input.iter())
        .map(|(&x, &pli)| {
            let inner = sqrt_2_over_pi * (x + 0.044715 * x * x * x);
            // Match the shader's clamp before tanh: keeps tanh away from
            // overflow on Apple Silicon.  Mathematically equivalent at f32
            // precision (tanh saturates well below |y|=15).
            let inner = inner.clamp(-15.0, 15.0);
            0.5 * x * (1.0 + inner.tanh()) * pli
        })
        .collect()
}

fn build_pipeline() -> (
    metal::Device,
    metal::ComputePipelineState,
    larql_compute_metal::buffers::BufferCache,
    metal::CommandQueue,
) {
    let device = metal::Device::system_default().unwrap();
    let src = larql_compute_metal::shaders::all_shaders();
    let lib = device
        .new_library_with_source(&src, &metal::CompileOptions::new())
        .unwrap();
    let pipeline = device
        .new_compute_pipeline_state_with_function(
            &lib.get_function("ple_gate_apply", None).unwrap(),
        )
        .unwrap();
    let bufs = larql_compute_metal::buffers::BufferCache::new(&device);
    let queue = device.new_command_queue();
    (device, pipeline, bufs, queue)
}

fn dispatch_ple_gate_apply(
    pipeline: &metal::ComputePipelineState,
    bufs: &larql_compute_metal::buffers::BufferCache,
    queue: &metal::CommandQueue,
    gate_in: &[f32],
    per_layer_input: &[f32],
) -> Vec<f32> {
    assert_eq!(gate_in.len(), per_layer_input.len());
    let n = gate_in.len();
    let buf_gate = bufs.transient_from_f32(gate_in);
    let buf_pli = bufs.transient_from_f32(per_layer_input);
    let buf_out = bufs.output((n * 4) as u64);
    let n_val = n as u32;

    let cmd = queue.new_command_buffer();
    let enc = cmd.new_compute_command_encoder();
    enc.set_compute_pipeline_state(pipeline);
    enc.set_buffer(0, Some(&buf_gate), 0);
    enc.set_buffer(1, Some(&buf_pli), 0);
    enc.set_buffer(2, Some(&buf_out), 0);
    enc.set_bytes(3, 4, &n_val as *const u32 as *const std::ffi::c_void);
    enc.dispatch_threads(
        metal::MTLSize::new(n as u64, 1, 1),
        metal::MTLSize::new((n as u64).min(256), 1, 1),
    );
    enc.end_encoding();
    cmd.commit();
    cmd.wait_until_completed();

    let ptr = buf_out.contents() as *const f32;
    unsafe { std::slice::from_raw_parts(ptr, n).to_vec() }
}

#[test]
fn ple_gate_apply_matches_cpu_smooth_inputs() {
    let (_dev, pipeline, bufs, queue) = build_pipeline();

    // Smooth inputs spanning a typical residual-stream range.
    let n = 256;
    let gate_in: Vec<f32> = (0..n).map(|i| (i as f32 * 0.05) - 6.0).collect();
    let pli: Vec<f32> = (0..n).map(|i| 0.3 + (i as f32 * 0.01).sin()).collect();

    let cpu = ple_gate_apply_cpu(&gate_in, &pli);
    let metal = dispatch_ple_gate_apply(&pipeline, &bufs, &queue, &gate_in, &pli);

    let diff = max_diff(&cpu, &metal);
    assert!(diff < 1e-5, "ple_gate_apply max diff {diff} (CPU vs Metal)");
}

#[test]
fn ple_gate_apply_zero_input_zero_output() {
    let (_dev, pipeline, bufs, queue) = build_pipeline();
    let n = 64;
    let gate_in = vec![0.0f32; n];
    let pli: Vec<f32> = (0..n).map(|i| (i as f32 + 1.0) * 0.1).collect();

    let cpu = ple_gate_apply_cpu(&gate_in, &pli);
    let metal = dispatch_ple_gate_apply(&pipeline, &bufs, &queue, &gate_in, &pli);

    // gelu_tanh(0) = 0, so output must be all zero regardless of pli.
    for v in &metal {
        assert_eq!(*v, 0.0);
    }
    assert_eq!(cpu, metal);
}

#[test]
fn ple_gate_apply_zero_per_layer_input_zero_output() {
    let (_dev, pipeline, bufs, queue) = build_pipeline();
    let n = 64;
    let gate_in: Vec<f32> = (0..n).map(|i| i as f32 * 0.1 - 3.0).collect();
    let pli = vec![0.0f32; n];

    let metal = dispatch_ple_gate_apply(&pipeline, &bufs, &queue, &gate_in, &pli);
    for v in &metal {
        assert_eq!(*v, 0.0);
    }
}

#[test]
fn ple_gate_apply_large_negative_inputs_dont_explode() {
    // GELU saturates near 0 for very negative inputs. The clamp inside
    // the shader keeps `tanh(inner)` away from `exp(2y)` overflow on
    // Apple Silicon. Pin that the output stays finite.
    let (_dev, pipeline, bufs, queue) = build_pipeline();
    let n = 64;
    let gate_in: Vec<f32> = (0..n).map(|i| -50.0 - (i as f32) * 0.5).collect();
    let pli = vec![1.0f32; n];

    let metal = dispatch_ple_gate_apply(&pipeline, &bufs, &queue, &gate_in, &pli);
    for (i, v) in metal.iter().enumerate() {
        assert!(v.is_finite(), "output[{i}] = {v} (input {})", gate_in[i]);
    }
}

#[test]
fn ple_gate_apply_e2b_shape() {
    // Gemma 4 E2B: ple_dim = 256.  Run at the production shape.
    let (_dev, pipeline, bufs, queue) = build_pipeline();
    let n = 256usize;
    // Plausible ranges — gate magnitudes in attention/FFN tend to be
    // small (post-norm), per_layer_input is the precomputed table which
    // has been RMS-normed and scaled by 1/sqrt(2).
    let gate_in: Vec<f32> = (0..n).map(|i| ((i as f32) * 0.013).sin() * 2.5).collect();
    let pli: Vec<f32> = (0..n).map(|i| ((i as f32) * 0.021).cos() * 0.7).collect();

    let cpu = ple_gate_apply_cpu(&gate_in, &pli);
    let metal = dispatch_ple_gate_apply(&pipeline, &bufs, &queue, &gate_in, &pli);

    let diff = max_diff(&cpu, &metal);
    assert!(diff < 1e-5, "E2B-shape ple_gate_apply max diff {diff}");
}
