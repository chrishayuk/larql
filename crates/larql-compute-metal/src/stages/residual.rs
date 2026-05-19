//! Post-attention and post-FFN residual + norm fusions.
//!
//! Two block-level helpers that sit between the matmul-heavy stages:
//!
//! - [`encode_post_attn`] fuses the post-attention residual add, the
//!   pre-FFN RMS norm, and (for Q4_0 / Q8_0 FFN) the Q8 quantisation of
//!   the norm output. Produces both the f32 `h_post_attn` residual and
//!   the f32 `ffn_norm_out` per position.
//!
//! - [`encode_post_ffn`] fuses the post-FFN residual add with the
//!   optional post-FFN RMS norm (Gemma post-norm architectures).
//!
//! Pre-norm vs post-norm branching lives inside these helpers; callers
//! pass `has_post_norms` and the appropriate weight buffers.

use larql_models::quant::ggml::LEGACY_BLOCK_ELEMS;
use metal::{Buffer, ComputeCommandEncoderRef, ComputePipelineState, MTLSize};
use std::ffi::c_void;

/// Post-attention residual + pre-FFN norm (+ optional Q8 quant).
///
/// For every position in `0..seq_len`:
///   1. Build `h_post_attn = h + O` (pre-norm) or
///      `h_post_attn = h + norm(O, post_attn_norm)` (post-norm).
///   2. `ffn_norm_out = rms_norm(h_post_attn, pre_ffn_weight)`.
///   3. If `ffn_needs_q8`, Q8-quantise `ffn_norm_out` into
///      `ffn_q8_buf` + `ffn_q8s_buf`.
///
/// `pre_ffn_weight_buf` is the weight for step 2. For Gemma post-norm
/// models it's typically `pre_ffn_norm` (falling back to
/// `post_attn_norm_buf`); for pre-norm models pass `post_attn_norm_buf`
/// directly.
#[allow(clippy::too_many_arguments)]
pub fn encode_post_attn(
    enc: &ComputeCommandEncoderRef,
    rms_norm_pipeline: &ComputePipelineState,
    residual_add_pipeline: &ComputePipelineState,
    q8_quant_pipeline: &ComputePipelineState,
    scratch_alloc: &mut dyn FnMut(u64) -> Buffer,
    h_buf: &Buffer,
    o_out: &Buffer,
    h_post_attn: &Buffer,
    ffn_norm_out: &Buffer,
    post_attn_norm_buf: &Buffer,
    pre_ffn_weight_buf: &Buffer,
    ffn_q8_buf: &Buffer,
    ffn_q8s_buf: &Buffer,
    seq_len: usize,
    hidden: usize,
    eps: f32,
    norm_offset: f32,
    has_post_norms: bool,
    ffn_needs_q8: bool,
    h_stride_bytes: u64,
    q8_stride_bytes: u64,
    q8s_stride_bytes: u64,
    // Granite-style residual_multiplier (0.22 on 3B/8B, 0.175 on 30B);
    // 1.0 for every non-Granite model and bit-identical no-op there.
    // Bound at buffer(4) of the `residual_add` shader. See
    // `FullPipelineLayer::residual_multiplier`.
    b_scale: f32,
) {
    let hidden_val = hidden as u32;
    let tg_threads = crate::kernels::DISPATCH_TG_MAX_THREADS.min(hidden as u64);

    for pos in 0..seq_len {
        let h_off = pos as u64 * h_stride_bytes;
        let q8_off = pos as u64 * q8_stride_bytes;
        let q8s_off = pos as u64 * q8s_stride_bytes;

        if has_post_norms {
            // Post-norm: norm(O) first, then residual add.
            let normed = scratch_alloc((hidden * 4) as u64);
            enc.set_compute_pipeline_state(rms_norm_pipeline);
            enc.set_buffer(0, Some(o_out), h_off);
            enc.set_buffer(1, Some(post_attn_norm_buf), 0);
            enc.set_buffer(2, Some(&normed), 0);
            enc.set_bytes(3, 4, &hidden_val as *const u32 as *const c_void);
            enc.set_bytes(4, 4, &eps as *const f32 as *const c_void);
            enc.set_bytes(5, 4, &norm_offset as *const f32 as *const c_void);
            enc.dispatch_thread_groups(MTLSize::new(1, 1, 1), MTLSize::new(tg_threads, 1, 1));

            enc.set_compute_pipeline_state(residual_add_pipeline);
            enc.set_buffer(0, Some(h_buf), h_off);
            enc.set_buffer(1, Some(&normed), 0);
            enc.set_buffer(2, Some(h_post_attn), h_off);
            enc.set_bytes(3, 4, &hidden_val as *const u32 as *const c_void);
            enc.set_bytes(4, 4, &b_scale as *const f32 as *const c_void);
            enc.dispatch_threads(
                MTLSize::new(hidden as u64, 1, 1),
                MTLSize::new(tg_threads, 1, 1),
            );
        } else {
            // Pre-norm: residual add first (h + O), then norm below.
            enc.set_compute_pipeline_state(residual_add_pipeline);
            enc.set_buffer(0, Some(h_buf), h_off);
            enc.set_buffer(1, Some(o_out), h_off);
            enc.set_buffer(2, Some(h_post_attn), h_off);
            enc.set_bytes(3, 4, &hidden_val as *const u32 as *const c_void);
            enc.set_bytes(4, 4, &b_scale as *const f32 as *const c_void);
            enc.dispatch_threads(
                MTLSize::new(hidden as u64, 1, 1),
                MTLSize::new(tg_threads, 1, 1),
            );
        }

        // Pre-FFN rms_norm on h_post_attn → ffn_norm_out (f32).
        enc.set_compute_pipeline_state(rms_norm_pipeline);
        enc.set_buffer(0, Some(h_post_attn), h_off);
        enc.set_buffer(1, Some(pre_ffn_weight_buf), 0);
        enc.set_buffer(2, Some(ffn_norm_out), h_off);
        enc.set_bytes(3, 4, &hidden_val as *const u32 as *const c_void);
        enc.set_bytes(4, 4, &eps as *const f32 as *const c_void);
        enc.set_bytes(5, 4, &norm_offset as *const f32 as *const c_void);
        enc.dispatch_thread_groups(MTLSize::new(1, 1, 1), MTLSize::new(tg_threads, 1, 1));

        // Q8-quantise ffn_norm_out when the FFN needs Q8 input (Q4_0 / Q8_0).
        if ffn_needs_q8 {
            let blocks = (hidden as u64).div_ceil(LEGACY_BLOCK_ELEMS as u64);
            enc.set_compute_pipeline_state(q8_quant_pipeline);
            enc.set_buffer(0, Some(ffn_norm_out), h_off);
            enc.set_buffer(1, Some(ffn_q8_buf), q8_off);
            enc.set_buffer(2, Some(ffn_q8s_buf), q8s_off);
            enc.set_bytes(3, 4, &hidden_val as *const u32 as *const c_void);
            enc.dispatch_threads(
                MTLSize::new(blocks, 1, 1),
                MTLSize::new(crate::kernels::DISPATCH_TG_MAX_THREADS.min(blocks), 1, 1),
            );
        }
    }
}

/// Post-FFN residual + optional post-FFN RMS norm.
///
/// For every position:
///   - **Post-norm with post_ffn_norm weight**:
///     `h_next = h_post_attn + norm(down_out, post_ffn_norm)`.
///   - **Pre-norm or post-norm without post_ffn_norm**:
///     `h_next = h_post_attn + down_out`.
#[allow(clippy::too_many_arguments)]
pub fn encode_post_ffn(
    enc: &ComputeCommandEncoderRef,
    rms_norm_pipeline: &ComputePipelineState,
    residual_add_pipeline: &ComputePipelineState,
    scratch_alloc: &mut dyn FnMut(u64) -> Buffer,
    down_out: &Buffer,
    h_post_attn: &Buffer,
    h_next: &Buffer,
    post_ffn_norm_buf: Option<&Buffer>,
    seq_len: usize,
    hidden: usize,
    eps: f32,
    norm_offset: f32,
    has_post_norms: bool,
    h_stride_bytes: u64,
    // Granite-style residual_multiplier; 1.0 for non-Granite. See
    // `encode_post_attn` doc for context.
    b_scale: f32,
) {
    let hidden_val = hidden as u32;
    let tg_threads = crate::kernels::DISPATCH_TG_MAX_THREADS.min(hidden as u64);

    for pos in 0..seq_len {
        let h_off = pos as u64 * h_stride_bytes;

        if has_post_norms {
            if let Some(post_ffn_buf) = post_ffn_norm_buf {
                let normed = scratch_alloc((hidden * 4) as u64);
                enc.set_compute_pipeline_state(rms_norm_pipeline);
                enc.set_buffer(0, Some(down_out), h_off);
                enc.set_buffer(1, Some(post_ffn_buf), 0);
                enc.set_buffer(2, Some(&normed), 0);
                enc.set_bytes(3, 4, &hidden_val as *const u32 as *const c_void);
                enc.set_bytes(4, 4, &eps as *const f32 as *const c_void);
                enc.set_bytes(5, 4, &norm_offset as *const f32 as *const c_void);
                enc.dispatch_thread_groups(MTLSize::new(1, 1, 1), MTLSize::new(tg_threads, 1, 1));

                enc.set_compute_pipeline_state(residual_add_pipeline);
                enc.set_buffer(0, Some(h_post_attn), h_off);
                enc.set_buffer(1, Some(&normed), 0);
                enc.set_buffer(2, Some(h_next), h_off);
                enc.set_bytes(3, 4, &hidden_val as *const u32 as *const c_void);
                enc.set_bytes(4, 4, &b_scale as *const f32 as *const c_void);
                enc.dispatch_threads(
                    MTLSize::new(hidden as u64, 1, 1),
                    MTLSize::new(tg_threads, 1, 1),
                );
                continue;
            }
        }

        // Pre-norm or post-norm-without-post_ffn_norm: plain residual.
        enc.set_compute_pipeline_state(residual_add_pipeline);
        enc.set_buffer(0, Some(h_post_attn), h_off);
        enc.set_buffer(1, Some(down_out), h_off);
        enc.set_buffer(2, Some(h_next), h_off);
        enc.set_bytes(3, 4, &hidden_val as *const u32 as *const c_void);
        enc.set_bytes(4, 4, &b_scale as *const f32 as *const c_void);
        enc.dispatch_threads(
            MTLSize::new(hidden as u64, 1, 1),
            MTLSize::new(tg_threads, 1, 1),
        );
    }
}
