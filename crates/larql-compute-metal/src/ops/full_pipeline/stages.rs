//! Per-stage encoders extracted from the `dispatch_full_pipeline`
//! per-layer body.
//!
//! Each stage takes a context bundle so the function signatures stay
//! readable instead of carrying 20+ parameters. Behaviour mirrors the
//! inline code byte-for-byte — pure organisation, no logic change.

use metal::{CommandBufferRef, ComputePipelineState};

use super::buffers::LayerBuffers;
use crate::stages::{input_norm, qkv_proj, quant_matvec};
use larql_compute::FullPipelineLayer;

/// Per-layer geometry + offsets needed by the input-norm + QKV stage.
pub(super) struct LayerCtx {
    pub eps: f32,
    pub norm_offset: f32,
    pub layer_q_dim: usize,
    pub layer_kv_dim: usize,
    pub q8_row_max: usize,
    pub q8s_row_bytes: usize,
}

/// Pipeline references the input-norm + QKV stage may dispatch.
/// All matvec-side fields are bare `ComputePipelineState`s mirroring
/// the existing `dispatch_full_pipeline` signature; only `q4_matvec`
/// flows through the format-aware quant_matvec stage helper which
/// expects a [`crate::kernels::KernelHandle`].
#[allow(dead_code)]
pub(super) struct InputNormQkvPipes<'a> {
    pub rms_norm: &'a ComputePipelineState,
    pub rms_norm_q8: &'a ComputePipelineState,
    pub q8_qkv_proj: &'a ComputePipelineState,
    pub q4kf_qkv_proj: Option<&'a ComputePipelineState>,
    pub q4k_qkv_proj: Option<&'a ComputePipelineState>,
    pub qm_pipes: quant_matvec::Pipelines<'a>,
}

/// Stage 1+3 — input norm followed by Q/K/V projection. Format-aware
/// per layer (Q4_K family takes f32 input through a fused or
/// per-projection shader; Q4_0 fuses the norm with Q8 quant then
/// dispatches per-projection Q4_0 matvec; Q8_0 uses the fused-Q8-QKV
/// shader).
#[allow(clippy::too_many_arguments)]
pub(super) fn encode_input_norm_and_qkv(
    cmd: &CommandBufferRef,
    layer: &FullPipelineLayer<'_>,
    layer_idx: usize,
    seq_len: usize,
    hidden: usize,
    ctx: &LayerCtx,
    pipes: &InputNormQkvPipes<'_>,
    lb: &LayerBuffers,
) {
    let l = layer_idx;
    let attn_format = layer.wq.format;
    let uses_f32_input = matches!(
        attn_format,
        larql_compute::QuantFormat::Q4_K
            | larql_compute::QuantFormat::Q6_K
            | larql_compute::QuantFormat::Q4_KF
    );

    let h_off = |p: usize| (p * hidden * 4) as u64;
    let q_off = |p: usize| (p * ctx.layer_q_dim * 4) as u64;
    let kv_off = |p: usize| (p * ctx.layer_kv_dim * 4) as u64;
    let q8_off = |p: usize| (p * ctx.q8_row_max) as u64;
    let q8s_off = |p: usize| (p * ctx.q8s_row_bytes) as u64;

    let all_same_format = layer.wq.format == layer.wk.format && layer.wk.format == layer.wv.format;
    // Pick the fused kernel whose host-side TG geometry matches the
    // shader being dispatched. The two shaders use different rows/TG and
    // threads/TG counts; getting them out of sync silently leaves rows
    // unwritten because the kernel's `if (global_row >= total_rows)`
    // guard hides the under-coverage. Encoded as a (pipeline, kernel)
    // pair so the dispatcher can't use one without the other.
    let fused_qkv_pipe: Option<(&ComputePipelineState, qkv_proj::FusedQkvKernel)> =
        if all_same_format {
            match layer.wq.format {
                larql_compute::QuantFormat::Q4_KF => pipes
                    .q4kf_qkv_proj
                    .map(|p| (p, qkv_proj::FusedQkvKernel::Q4kf))
                    .or_else(|| {
                        pipes
                            .q4k_qkv_proj
                            .map(|p| (p, qkv_proj::FusedQkvKernel::Q4k))
                    }),
                larql_compute::QuantFormat::Q4_K => pipes
                    .q4k_qkv_proj
                    .map(|p| (p, qkv_proj::FusedQkvKernel::Q4k)),
                _ => None,
            }
        } else {
            None
        };

    // Encoder coalescing: hoist `cmd.new_compute_command_encoder()` and
    // `enc.end_encoding()` out of the per-position loop so we pay one
    // encoder-create + end_encoding per layer per stage instead of
    // `seq_len` of them. The per-position dispatches inside don't touch
    // encoder lifecycle (only set_pipeline_state / set_buffer / dispatch),
    // so they run back-to-back on the GPU. Saves ~5 µs × seq_len per layer
    // on prefill — see ROADMAP P0 "Prefill: per-position matvec → matmul"
    // entry, 2026-04-27.
    if uses_f32_input {
        // Q4_K / Q6_K / Q4_KF: f32 norm output, then either fused or
        // per-projection QKV matvec.
        let enc = cmd.new_compute_command_encoder();
        for pos in 0..seq_len {
            input_norm::encode_f32(
                enc,
                pipes.rms_norm,
                &lb.h[l],
                h_off(pos),
                &lb.input_norm[l],
                &lb.norm_out[l],
                h_off(pos),
                hidden,
                ctx.eps,
                ctx.norm_offset,
            );
            if let Some((fused_pipeline, fused_kernel)) = fused_qkv_pipe {
                qkv_proj::encode_fused_f32(
                    enc,
                    fused_pipeline,
                    fused_kernel,
                    &lb.wq[l],
                    &lb.wk[l],
                    &lb.wv[l],
                    &lb.norm_out[l],
                    h_off(pos),
                    &lb.q_out[l],
                    q_off(pos),
                    &lb.k_out[l],
                    kv_off(pos),
                    &lb.v_out[l],
                    kv_off(pos),
                    ctx.layer_q_dim,
                    ctx.layer_kv_dim,
                    hidden,
                );
            } else {
                let pos_qoff = q_off(pos);
                let pos_kvoff = kv_off(pos);
                qkv_proj::encode_per_proj(
                    enc,
                    &pipes.qm_pipes,
                    &lb.norm_out[l],
                    h_off(pos),
                    // Q8 input unused for f32-input formats — placeholder.
                    &lb.norm_out[l],
                    0,
                    &lb.norm_out[l],
                    0,
                    [
                        qkv_proj::Proj {
                            format: layer.wq.format,
                            w_buf: &lb.wq[l],
                            out_buf: &lb.q_out[l],
                            out_off: pos_qoff,
                            rows: ctx.layer_q_dim,
                        },
                        qkv_proj::Proj {
                            format: layer.wk.format,
                            w_buf: &lb.wk[l],
                            out_buf: &lb.k_out[l],
                            out_off: pos_kvoff,
                            rows: ctx.layer_kv_dim,
                        },
                        qkv_proj::Proj {
                            format: layer.wv.format,
                            w_buf: &lb.wv[l],
                            out_buf: &lb.v_out[l],
                            out_off: pos_kvoff,
                            rows: ctx.layer_kv_dim,
                        },
                    ],
                    hidden,
                );
            }
        }
        enc.end_encoding();
    } else {
        // Legacy Q8-input formats: first fuse rms_norm+Q8-quantise, then
        // route by weight layout. Q4_0 weights stay packed Q4_0 and must go
        // through the Q4_0 matvec helper; Q8_0 weights use the fused Q8 QKV
        // shader with separate per-row weight scales.
        let enc = cmd.new_compute_command_encoder();
        for pos in 0..seq_len {
            input_norm::encode_q8(
                enc,
                pipes.rms_norm_q8,
                &lb.h[l],
                h_off(pos),
                &lb.input_norm[l],
                &lb.q8[l],
                q8_off(pos),
                &lb.q8s[l],
                q8s_off(pos),
                hidden,
                ctx.eps,
                ctx.norm_offset,
            );
            if layer.wq.format == larql_compute::QuantFormat::Q8_0
                && layer.wk.format == larql_compute::QuantFormat::Q8_0
                && layer.wv.format == larql_compute::QuantFormat::Q8_0
            {
                qkv_proj::encode_fused_q8(
                    enc,
                    pipes.q8_qkv_proj,
                    &lb.wq[l],
                    &lb.wq_scale[l],
                    &lb.wk[l],
                    &lb.wk_scale[l],
                    &lb.wv[l],
                    &lb.wv_scale[l],
                    &lb.q8[l],
                    q8_off(pos),
                    &lb.q8s[l],
                    q8s_off(pos),
                    &lb.q_out[l],
                    q_off(pos),
                    &lb.k_out[l],
                    kv_off(pos),
                    &lb.v_out[l],
                    kv_off(pos),
                    ctx.layer_q_dim,
                    ctx.layer_kv_dim,
                    hidden,
                );
            } else {
                let pos_qoff = q_off(pos);
                let pos_kvoff = kv_off(pos);
                qkv_proj::encode_per_proj(
                    enc,
                    &pipes.qm_pipes,
                    &lb.h[l],
                    h_off(pos),
                    &lb.q8[l],
                    q8_off(pos),
                    &lb.q8s[l],
                    q8s_off(pos),
                    [
                        qkv_proj::Proj {
                            format: layer.wq.format,
                            w_buf: &lb.wq[l],
                            out_buf: &lb.q_out[l],
                            out_off: pos_qoff,
                            rows: ctx.layer_q_dim,
                        },
                        qkv_proj::Proj {
                            format: layer.wk.format,
                            w_buf: &lb.wk[l],
                            out_buf: &lb.k_out[l],
                            out_off: pos_kvoff,
                            rows: ctx.layer_kv_dim,
                        },
                        qkv_proj::Proj {
                            format: layer.wv.format,
                            w_buf: &lb.wv[l],
                            out_buf: &lb.v_out[l],
                            out_off: pos_kvoff,
                            rows: ctx.layer_kv_dim,
                        },
                    ],
                    hidden,
                );
            }
        }
        enc.end_encoding();
    }
}
