//! Fused causal attention — one dispatch for the whole layer's QKV → attn_out.
//!
//! Dispatches `fused_attention` which handles RoPE (optional), QK-norm
//! (optional), causal GQA softmax, and softcap in a single Metal kernel.
//! Grid is `(num_q_heads, seq_len, 1)` threadgroups of 256 threads.
//!
//! When the caller has already applied QK-norm separately (via
//! `stages::qk_norm::encode_qk_norm`), pass `use_qk_norm = false`.
//! When the caller has already applied RoPE via `stages::rope::encode`,
//! pass `skip_rope = true`.

use metal::{Buffer, ComputeCommandEncoderRef, ComputePipelineState, MTLSize};
use std::ffi::c_void;

/// Flags for the fused attention dispatch. Keeps the parameter list
/// readable; every boolean has an obvious default.
#[derive(Clone, Copy)]
pub struct Flags {
    pub use_qk_norm: bool,
    pub skip_rope: bool,
    pub softcap: f32,
    pub rotary_dim: u32,
}

/// Dispatch `fused_attention` into the given encoder. Caller owns the
/// encoder lifecycle.
#[allow(clippy::too_many_arguments)]
pub fn encode(
    enc: &ComputeCommandEncoderRef,
    pipeline: &ComputePipelineState,
    q_buf: &Buffer,
    k_buf: &Buffer,
    v_buf: &Buffer,
    attn_out: &Buffer,
    seq_len: usize,
    num_q_heads: usize,
    num_kv_heads: usize,
    head_dim: usize,
    scale: f32,
    rope_base: f32,
    flags: Flags,
) {
    let seq_val = seq_len as u32;
    let hd_val = head_dim as u32;
    let nq_val = num_q_heads as u32;
    let nkv_val = num_kv_heads as u32;
    let qknorm_val: u32 = if flags.use_qk_norm { 1 } else { 0 };
    let skip_rope_val: u32 = if flags.skip_rope { 1 } else { 0 };

    enc.set_compute_pipeline_state(pipeline);
    enc.set_buffer(0, Some(q_buf), 0);
    enc.set_buffer(1, Some(k_buf), 0);
    enc.set_buffer(2, Some(v_buf), 0);
    enc.set_buffer(3, Some(attn_out), 0);
    enc.set_bytes(4, 4, &seq_val as *const u32 as *const c_void);
    enc.set_bytes(5, 4, &hd_val as *const u32 as *const c_void);
    enc.set_bytes(6, 4, &nq_val as *const u32 as *const c_void);
    enc.set_bytes(7, 4, &nkv_val as *const u32 as *const c_void);
    enc.set_bytes(8, 4, &scale as *const f32 as *const c_void);
    enc.set_bytes(9, 4, &rope_base as *const f32 as *const c_void);
    enc.set_bytes(10, 4, &qknorm_val as *const u32 as *const c_void);
    enc.set_bytes(11, 4, &flags.softcap as *const f32 as *const c_void);
    enc.set_bytes(12, 4, &skip_rope_val as *const u32 as *const c_void);
    enc.set_bytes(13, 4, &flags.rotary_dim as *const u32 as *const c_void);
    enc.dispatch_thread_groups(
        MTLSize::new(num_q_heads as u64, seq_len as u64, 1),
        MTLSize::new(256, 1, 1),
    );
}
