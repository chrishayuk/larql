//! Per-layer residual scalar — Gemma 4's learned stabiliser.
//!
//! Multiplies the layer's final residual (`h_bufs[l + 1]`) by a per-layer
//! scalar typically in the range 0.02–0.8. Without this the residual
//! magnitude explodes across layers because Gemma 4's post-attention norm
//! weights can reach ~100. Mirrors `apply_layer_scalar` on the CPU path
//! and Step 8 of `decode_token`.
//!
//! Scoped to positions 0..seq_len for multi-position prefill; decode
//! calls with seq_len = 1.
//!
//! Caller owns the encoder lifecycle.

use metal::{Buffer, ComputeCommandEncoderRef, ComputePipelineState, MTLSize};
use std::ffi::c_void;

/// If `scalar` is non-zero, scale the f32 residual at each position by `scalar`.
///
/// * `h_buf` is the residual buffer holding `seq_len × hidden` f32s starting
///   at byte 0, one `hidden`-sized slice per position.
/// * `pipeline` must be the pipeline for the `scale_vector` shader.
pub fn encode(
    enc: &ComputeCommandEncoderRef,
    pipeline: &ComputePipelineState,
    h_buf: &Buffer,
    seq_len: usize,
    hidden: usize,
    scalar: f32,
) {
    if scalar == 0.0 {
        return;
    }
    let hidden_val = hidden as u32;
    for pos in 0..seq_len {
        let h_off = (pos * hidden * 4) as u64;
        enc.set_compute_pipeline_state(pipeline);
        enc.set_buffer(0, Some(h_buf), h_off);
        enc.set_buffer(1, Some(h_buf), h_off);
        enc.set_bytes(2, 4, &hidden_val as *const u32 as *const c_void);
        enc.set_bytes(3, 4, &scalar as *const f32 as *const c_void);
        // `scale_vector` uses `thread_position_in_grid` — one thread per
        // element, not a single 256-thread threadgroup.
        enc.dispatch_threads(
            MTLSize::new(hidden as u64, 1, 1),
            MTLSize::new(
                crate::kernels::DISPATCH_TG_MAX_THREADS.min(hidden as u64),
                1,
                1,
            ),
        );
    }
}

#[cfg(test)]
#[cfg(target_os = "macos")]
mod tests {
    use super::*;
    use crate::buffers::BufferCache;
    use crate::shaders;

    /// Pin Gemma 4's per-layer residual stabiliser: out = h * scalar applied
    /// in-place to each position's hidden vector. Uses a non-trivial
    /// scalar (0.5) and 3 positions to verify the per-position offset
    /// math too.
    #[test]
    fn encode_scales_each_position_in_place() {
        let device = match metal::Device::system_default() {
            Some(d) => d,
            None => return,
        };
        let src = shaders::all_shaders();
        let lib = device
            .new_library_with_source(&src, &metal::CompileOptions::new())
            .unwrap();
        let pipeline = device
            .new_compute_pipeline_state_with_function(
                &lib.get_function("scale_vector", None).unwrap(),
            )
            .unwrap();
        let bufs = BufferCache::new(&device);
        let queue = device.new_command_queue();

        let hidden = 16usize;
        let seq_len = 3usize;
        let scalar = 0.5f32;
        let h_init: Vec<f32> = (0..seq_len * hidden).map(|i| i as f32).collect();
        let h_buf = bufs.transient_from_f32(&h_init);

        let cmd = queue.new_command_buffer();
        let enc = cmd.new_compute_command_encoder();
        encode(enc, &pipeline, &h_buf, seq_len, hidden, scalar);
        enc.end_encoding();
        cmd.commit();
        cmd.wait_until_completed();

        let out_ptr = h_buf.contents() as *const f32;
        let metal_out: Vec<f32> =
            unsafe { std::slice::from_raw_parts(out_ptr, seq_len * hidden).to_vec() };

        for (i, v) in metal_out.iter().enumerate() {
            let expected = h_init[i] * scalar;
            assert!(
                (v - expected).abs() < 1e-6,
                "pos-element {i}: expected {expected} got {v}"
            );
        }
    }

    /// scalar == 0.0 is the "no-op" sentinel used for layers without a
    /// per-layer scalar (most archs). The function must early-return
    /// without touching the buffer.
    #[test]
    fn encode_zero_scalar_is_noop() {
        let device = match metal::Device::system_default() {
            Some(d) => d,
            None => return,
        };
        let src = shaders::all_shaders();
        let lib = device
            .new_library_with_source(&src, &metal::CompileOptions::new())
            .unwrap();
        let pipeline = device
            .new_compute_pipeline_state_with_function(
                &lib.get_function("scale_vector", None).unwrap(),
            )
            .unwrap();
        let bufs = BufferCache::new(&device);
        let queue = device.new_command_queue();

        let hidden = 8usize;
        let h_init: Vec<f32> = (0..hidden).map(|i| (i as f32) * 0.7 + 1.3).collect();
        let h_buf = bufs.transient_from_f32(&h_init);

        let cmd = queue.new_command_buffer();
        let enc = cmd.new_compute_command_encoder();
        encode(enc, &pipeline, &h_buf, 1, hidden, 0.0);
        enc.end_encoding();
        cmd.commit();
        cmd.wait_until_completed();

        let out_ptr = h_buf.contents() as *const f32;
        let out: Vec<f32> = unsafe { std::slice::from_raw_parts(out_ptr, hidden).to_vec() };
        for (i, v) in out.iter().enumerate() {
            assert!(
                (v - h_init[i]).abs() < 1e-6,
                "scalar=0 should leave element {i} unchanged: was {} now {v}",
                h_init[i]
            );
        }
    }
}
