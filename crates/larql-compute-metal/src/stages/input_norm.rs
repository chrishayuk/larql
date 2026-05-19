//! Input layer norm — the first stage of every transformer layer.
//!
//! Two code paths depending on what the QKV projection wants next:
//!
//! - **f32 output** (`encode_f32`): plain `rms_norm` writing f32 to the
//!   norm-out buffer. Used by Q4_K / Q4_KF / Q6_K attention which consume
//!   f32 input.
//! - **Fused norm + Q8 quantise** (`encode_q8`): single-dispatch
//!   `rms_norm_q8` writing Q8 int8s + per-32 f16-scaled blocks. Used by
//!   Q8_0 / Q4_0 attention which consume Q8 input.
//!
//! Both variants are per-position (single hidden vector per call); the
//! caller loops over positions. The caller owns the encoder lifecycle —
//! these helpers only issue dispatches.

use metal::{Buffer, ComputeCommandEncoderRef, ComputePipelineState, MTLSize};
use std::ffi::c_void;

/// f32-output input RMS norm.
///
/// Writes `out[hidden]` as `(x / rms(x)) * (weight + offset)` using the
/// cooperative single-threadgroup `rms_norm` shader.
#[allow(clippy::too_many_arguments)]
pub fn encode_f32(
    enc: &ComputeCommandEncoderRef,
    pipeline: &ComputePipelineState,
    h_buf: &Buffer,
    h_off: u64,
    norm_weight: &Buffer,
    out_buf: &Buffer,
    out_off: u64,
    hidden: usize,
    eps: f32,
    norm_offset: f32,
) {
    let hidden_val = hidden as u32;
    enc.set_compute_pipeline_state(pipeline);
    enc.set_buffer(0, Some(h_buf), h_off);
    enc.set_buffer(1, Some(norm_weight), 0);
    enc.set_buffer(2, Some(out_buf), out_off);
    enc.set_bytes(3, 4, &hidden_val as *const u32 as *const c_void);
    enc.set_bytes(4, 4, &eps as *const f32 as *const c_void);
    enc.set_bytes(5, 4, &norm_offset as *const f32 as *const c_void);
    enc.dispatch_thread_groups(
        MTLSize::new(1, 1, 1),
        MTLSize::new(
            crate::kernels::DISPATCH_TG_MAX_THREADS.min(hidden as u64),
            1,
            1,
        ),
    );
}

/// Fused RMS norm + Q8 quantise — writes Q8 int8 values and f32 scales.
#[allow(clippy::too_many_arguments)]
pub fn encode_q8(
    enc: &ComputeCommandEncoderRef,
    pipeline: &ComputePipelineState,
    h_buf: &Buffer,
    h_off: u64,
    norm_weight: &Buffer,
    q8_out: &Buffer,
    q8_out_off: u64,
    q8s_out: &Buffer,
    q8s_out_off: u64,
    hidden: usize,
    eps: f32,
    norm_offset: f32,
) {
    let hidden_val = hidden as u32;
    enc.set_compute_pipeline_state(pipeline);
    enc.set_buffer(0, Some(h_buf), h_off);
    enc.set_buffer(1, Some(norm_weight), 0);
    enc.set_buffer(2, Some(q8_out), q8_out_off);
    enc.set_buffer(3, Some(q8s_out), q8s_out_off);
    enc.set_bytes(4, 4, &hidden_val as *const u32 as *const c_void);
    enc.set_bytes(5, 4, &eps as *const f32 as *const c_void);
    enc.set_bytes(6, 4, &norm_offset as *const f32 as *const c_void);
    enc.dispatch_thread_groups(
        MTLSize::new(1, 1, 1),
        MTLSize::new(
            crate::kernels::DISPATCH_TG_MAX_THREADS.min(hidden as u64),
            1,
            1,
        ),
    );
}
