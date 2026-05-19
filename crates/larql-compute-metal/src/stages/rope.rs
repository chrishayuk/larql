//! Rotary position embedding (RoPE) — pre-attention when KV cache is used.
//!
//! Applies RoPE to Q and K in-place per head per position. Supports
//! partial rotation (Gemma 4 global layers use `rotary_dim = head_dim / 4`).
//!
//! The shader dispatched is `rope_at_pos` which rotates a single head's
//! `rotary_dim / 2` pairs. We loop per position, per head, dispatching
//! a thread per pair. One encoder batches all dispatches for efficiency.

use metal::{Buffer, ComputeCommandEncoderRef, ComputePipelineState, MTLSize};
use std::ffi::c_void;

/// Apply RoPE to Q and K per head per position.
///
/// `rotary_dim == 0` is treated by the shader as "rotate full head_dim".
/// Partial rotation (Gemma 4 global layers) uses `rotary_dim < head_dim`.
/// Caller owns the encoder lifecycle.
#[allow(clippy::too_many_arguments)]
pub fn encode(
    enc: &ComputeCommandEncoderRef,
    pipeline: &ComputePipelineState,
    q_buf: &Buffer,
    k_buf: &Buffer,
    seq_len: usize,
    num_q_heads: usize,
    num_kv_heads: usize,
    head_dim: usize,
    rotary_dim: usize,
    rope_base: f32,
) {
    let hd = head_dim as u32;
    let rdim_val = rotary_dim as u32;
    let rdim_effective = if rotary_dim == 0 {
        head_dim
    } else {
        rotary_dim
    };
    let hdim = (rdim_effective / 2) as u64;

    for pos in 0..seq_len {
        let pos_val = pos as u32;
        for qh in 0..num_q_heads {
            let offset = (pos * num_q_heads * head_dim + qh * head_dim) as u64 * 4;
            enc.set_compute_pipeline_state(pipeline);
            enc.set_buffer(0, Some(q_buf), offset);
            enc.set_bytes(1, 4, &hd as *const u32 as *const c_void);
            enc.set_bytes(2, 4, &rope_base as *const f32 as *const c_void);
            enc.set_bytes(3, 4, &pos_val as *const u32 as *const c_void);
            enc.set_bytes(4, 4, &rdim_val as *const u32 as *const c_void);
            enc.dispatch_threads(MTLSize::new(hdim, 1, 1), MTLSize::new(hdim.min(256), 1, 1));
        }
        for kvh in 0..num_kv_heads {
            let offset = (pos * num_kv_heads * head_dim + kvh * head_dim) as u64 * 4;
            enc.set_compute_pipeline_state(pipeline);
            enc.set_buffer(0, Some(k_buf), offset);
            enc.set_bytes(1, 4, &hd as *const u32 as *const c_void);
            enc.set_bytes(2, 4, &rope_base as *const f32 as *const c_void);
            enc.set_bytes(3, 4, &pos_val as *const u32 as *const c_void);
            enc.set_bytes(4, 4, &rdim_val as *const u32 as *const c_void);
            enc.dispatch_threads(MTLSize::new(hdim, 1, 1), MTLSize::new(hdim.min(256), 1, 1));
        }
    }
}
