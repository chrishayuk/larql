//! Output projection (`attn_out → h_post_attn_input`) — per position.
//!
//! Thin wrapper over [`super::quant_matvec::encode`] that routes the
//! attention output through the right shader based on the O-weight format:
//!
//! - **Q4_K / Q4_KF / Q6_K**: f32 input directly; single matvec dispatch.
//! - **Q4_0 / Q8_0**: quantise `attn_out` to Q8 first (callers supply a
//!   staging buffer), then Q8 matvec.
//!
//! Single-vector per position. Multi-position prefill loops.

use larql_models::quant::ggml::LEGACY_BLOCK_ELEMS;
use metal::{Buffer, ComputeCommandEncoderRef, ComputePipelineState, MTLSize};
use std::ffi::c_void;

use super::quant_matvec;

/// Per-position O projection. Caller owns the encoder lifecycle.
///
/// For Q4_K / Q4_KF / Q6_K this is one dispatch. For Q4_0 / Q8_0 we first
/// quantise `attn_in` to the caller's Q8 staging buffer.
#[allow(clippy::too_many_arguments)]
pub fn encode(
    enc: &ComputeCommandEncoderRef,
    pipes: &quant_matvec::Pipelines<'_>,
    q8_quant_pipeline: &ComputePipelineState,
    format: larql_compute::QuantFormat,
    wo_buf: &Buffer,
    attn_in: &Buffer,
    attn_in_off: u64,
    q8_stage: &Buffer,
    q8_stage_off: u64,
    q8s_stage: &Buffer,
    q8s_stage_off: u64,
    o_out: &Buffer,
    o_out_off: u64,
    q_dim: usize,
    hidden: usize,
) {
    let is_f32_input = matches!(
        format,
        larql_compute::QuantFormat::Q4_K
            | larql_compute::QuantFormat::Q4_KF
            | larql_compute::QuantFormat::Q6_K
    );

    if !is_f32_input {
        // Q4_0 / Q8_0: quantise attn_in[q_dim] → Q8 int8 + per-32 f16 scale.
        let dim_val = q_dim as u32;
        let blocks = (q_dim as u64).div_ceil(LEGACY_BLOCK_ELEMS as u64);
        enc.set_compute_pipeline_state(q8_quant_pipeline);
        enc.set_buffer(0, Some(attn_in), attn_in_off);
        enc.set_buffer(1, Some(q8_stage), q8_stage_off);
        enc.set_buffer(2, Some(q8s_stage), q8s_stage_off);
        enc.set_bytes(3, 4, &dim_val as *const u32 as *const c_void);
        enc.dispatch_threads(
            MTLSize::new(blocks, 1, 1),
            MTLSize::new(crate::kernels::DISPATCH_TG_MAX_THREADS.min(blocks), 1, 1),
        );
    }

    quant_matvec::encode(
        enc,
        format,
        wo_buf,
        attn_in,
        attn_in_off,
        q8_stage,
        q8_stage_off,
        q8s_stage,
        q8s_stage_off,
        o_out,
        o_out_off,
        pipes,
        hidden,
        q_dim,
    );
}
