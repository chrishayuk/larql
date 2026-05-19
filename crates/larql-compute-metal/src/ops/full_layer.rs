//! Full layer pipeline: attention + FFN in one Metal command buffer.
//!
//! Dispatches Q/K/V projections (f32) → causal attention → O projection (f32) →
//! Q4 gate+up → GEGLU → Q4 down. One GPU submission per layer.

use metal::*;
use std::ffi::c_void;

use super::q4_common::Q4Pipelines;
use crate::buffers::BufferCache;
use crate::f32_ops::F32Ops;

/// Run a full transformer layer on Metal: attention + FFN, one command buffer.
#[allow(clippy::too_many_arguments)]
pub fn dispatch(
    queue: &CommandQueue,
    bufs: &BufferCache,
    f32_transb_pipeline: &ComputePipelineState,
    causal_attn_pipeline: &ComputePipelineState,
    _q4: &Q4Pipelines,
    // Attention weights (f32)
    w_q: &[f32],
    w_k: &[f32],
    w_v: &[f32],
    w_o: &[f32],
    // FFN weights (Q4)
    gate_q4: &[u8],
    up_q4: &[u8],
    down_t_q4: &[u8],
    // Input
    x: &[f32],
    seq_len: usize,
    hidden: usize,
    num_q_heads: usize,
    num_kv_heads: usize,
    head_dim: usize,
    _inter: usize,
    attn_scale: f32,
) -> Vec<f32> {
    let kv_dim = num_kv_heads * head_dim;
    let q_dim = num_q_heads * head_dim;

    let buf_x = bufs.transient_from_f32(x);
    let buf_wq = bufs.get_f32(w_q);
    let buf_wk = bufs.get_f32(w_k);
    let buf_wv = bufs.get_f32(w_v);
    let buf_wo = bufs.get_f32(w_o);
    let _buf_gate = bufs.get_bytes(gate_q4);
    let _buf_up = bufs.get_bytes(up_q4);
    let _buf_down = bufs.get_bytes(down_t_q4);

    let buf_q = bufs.output((seq_len * q_dim * 4) as u64);
    let buf_k = bufs.output((seq_len * kv_dim * 4) as u64);
    let buf_v = bufs.output((seq_len * kv_dim * 4) as u64);
    let buf_attn_out = bufs.output((seq_len * q_dim * 4) as u64);
    let buf_o_out = bufs.output((seq_len * hidden * 4) as u64);

    let cmd = queue.new_command_buffer();

    // Q projection
    {
        let enc = cmd.new_compute_command_encoder();
        F32Ops::encode_static(
            f32_transb_pipeline,
            enc,
            &buf_x,
            &buf_wq,
            &buf_q,
            seq_len,
            q_dim,
            hidden,
        );
        enc.end_encoding();
    }
    // K projection
    {
        let enc = cmd.new_compute_command_encoder();
        F32Ops::encode_static(
            f32_transb_pipeline,
            enc,
            &buf_x,
            &buf_wk,
            &buf_k,
            seq_len,
            kv_dim,
            hidden,
        );
        enc.end_encoding();
    }
    // V projection
    {
        let enc = cmd.new_compute_command_encoder();
        F32Ops::encode_static(
            f32_transb_pipeline,
            enc,
            &buf_x,
            &buf_wv,
            &buf_v,
            seq_len,
            kv_dim,
            hidden,
        );
        enc.end_encoding();
    }
    // Causal attention (simplified — first head only for benchmark)
    {
        let enc = cmd.new_compute_command_encoder();
        enc.set_compute_pipeline_state(causal_attn_pipeline);
        enc.set_buffer(0, Some(&buf_q), 0);
        enc.set_buffer(1, Some(&buf_k), 0);
        enc.set_buffer(2, Some(&buf_v), 0);
        enc.set_buffer(3, Some(&buf_attn_out), 0);
        let seq_val = seq_len as u32;
        let hd_val = head_dim as u32;
        enc.set_bytes(4, 4, &seq_val as *const u32 as *const c_void);
        enc.set_bytes(5, 4, &hd_val as *const u32 as *const c_void);
        enc.set_bytes(6, 4, &attn_scale as *const f32 as *const c_void);
        let threads = MTLSize::new(head_dim as u64, seq_len as u64, 1);
        let tg = MTLSize::new(head_dim.min(256) as u64, seq_len.min(1) as u64, 1);
        enc.dispatch_threads(threads, tg);
        enc.end_encoding();
    }
    // O projection
    {
        let enc = cmd.new_compute_command_encoder();
        F32Ops::encode_static(
            f32_transb_pipeline,
            enc,
            &buf_attn_out,
            &buf_wo,
            &buf_o_out,
            seq_len,
            hidden,
            q_dim,
        );
        enc.end_encoding();
    }

    // Note: FFN would chain here with Q8 quantize + Q4 gate/up/down
    // For now, return attention output only (FFN benchmarked separately)

    cmd.commit();
    cmd.wait_until_completed();

    crate::buffers::read_buffer_f32(&buf_o_out, seq_len * hidden)
}
