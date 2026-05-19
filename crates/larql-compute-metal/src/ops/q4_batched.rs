//! Batched Q4 operations — pair dispatch and multi-layer pipeline.
//!
//! Amortises GPU dispatch overhead by encoding multiple operations
//! into single command buffers.
//!
//! - `pair_batch`: gate+up for all seq positions in one submission
//! - `multi_layer_ffn`: 21 layers × (gate+up+GEGLU+down+Q8) in one submission

use metal::*;
use std::ffi::c_void;

use super::q4_common::{quantize_to_q8, Q4Pipelines};
use crate::buffers::BufferCache;
use larql_models::quant::ggml::LEGACY_BLOCK_ELEMS;

/// Batched gate+up for ALL seq positions in ONE GPU submission.
/// Encodes 2×seq_len Q4 matvec dispatches in a single command buffer.
#[allow(clippy::too_many_arguments)]
pub fn pair_batch(
    queue: &CommandQueue,
    bufs: &BufferCache,
    pipelines: &Q4Pipelines,
    gate_q4: &[u8],
    up_q4: &[u8],
    x_matrix: &[f32], // [seq_len * hidden] flattened
    seq_len: usize,
    num_rows: usize,
    hidden: usize,
) -> (Vec<Vec<f32>>, Vec<Vec<f32>>) {
    let n_val = num_rows as u32;
    let k_val = hidden as u32;
    // Geometry travels with the kernel — read both sides from the
    // same `KernelHandle` to guarantee num_tgs and threads_per_tg
    // agree with what the kernel was compiled for.
    let kernel = &pipelines.matvec;
    let num_tgs = (num_rows as u64).div_ceil(kernel.rows_per_tg);
    let grid = MTLSize::new(num_tgs, 1, 1);
    let tg_size = MTLSize::new(kernel.threads_per_tg, 1, 1);
    let out_bytes = (num_rows * 4) as u64;

    let buf_gate = bufs.get_bytes(gate_q4);
    let buf_up = bufs.get_bytes(up_q4);

    let cmd = queue.new_command_buffer();
    let mut gate_bufs = Vec::with_capacity(seq_len);
    let mut up_bufs = Vec::with_capacity(seq_len);

    for s in 0..seq_len {
        let x_slice = &x_matrix[s * hidden..(s + 1) * hidden];
        let (q8_x, q8_scales) = quantize_to_q8(x_slice);

        let buf_q8 = bufs.transient_from_i8(&q8_x);
        let buf_scales = bufs.transient_from_f32(&q8_scales);
        let buf_g_out = bufs.output(out_bytes);
        let buf_u_out = bufs.output(out_bytes);

        // Gate
        let enc = cmd.new_compute_command_encoder();
        enc.set_compute_pipeline_state(&kernel.state);
        enc.set_buffer(0, Some(&buf_gate), 0);
        enc.set_buffer(1, Some(&buf_q8), 0);
        enc.set_buffer(2, Some(&buf_scales), 0);
        enc.set_buffer(3, Some(&buf_g_out), 0);
        enc.set_bytes(4, 4, &n_val as *const u32 as *const c_void);
        enc.set_bytes(5, 4, &k_val as *const u32 as *const c_void);
        enc.dispatch_thread_groups(grid, tg_size);
        enc.end_encoding();

        // Up
        let enc = cmd.new_compute_command_encoder();
        enc.set_compute_pipeline_state(&kernel.state);
        enc.set_buffer(0, Some(&buf_up), 0);
        enc.set_buffer(1, Some(&buf_q8), 0);
        enc.set_buffer(2, Some(&buf_scales), 0);
        enc.set_buffer(3, Some(&buf_u_out), 0);
        enc.set_bytes(4, 4, &n_val as *const u32 as *const c_void);
        enc.set_bytes(5, 4, &k_val as *const u32 as *const c_void);
        enc.dispatch_thread_groups(grid, tg_size);
        enc.end_encoding();

        gate_bufs.push(buf_g_out);
        up_bufs.push(buf_u_out);
    }

    cmd.commit();
    cmd.wait_until_completed();

    let mut gate_results = Vec::with_capacity(seq_len);
    let mut up_results = Vec::with_capacity(seq_len);
    for s in 0..seq_len {
        gate_results.push(crate::buffers::read_buffer_f32(&gate_bufs[s], num_rows));
        up_results.push(crate::buffers::read_buffer_f32(&up_bufs[s], num_rows));
    }
    (gate_results, up_results)
}

/// Multi-layer Q4 FFN in ONE command buffer.
/// gate → up → GEGLU → down → Q8 quantize → next layer.
/// All on GPU, no CPU return between layers.
#[allow(clippy::too_many_arguments)]
pub fn multi_layer_ffn(
    queue: &CommandQueue,
    bufs: &BufferCache,
    pipelines: &Q4Pipelines,
    geglu_pipeline: &ComputePipelineState,
    q8_quant_pipeline: &ComputePipelineState,
    layers_q4: &[(&[u8], &[u8], &[u8])], // [(gate, up, down_t)]
    x: &[f32],
    inter: usize,
    hidden: usize,
) -> Vec<f32> {
    let num_layers = layers_q4.len();
    let n_val = inter as u32;
    let k_val = hidden as u32;
    let inter_val = inter as u32;
    let hidden_val = hidden as u32;
    let kernel = &pipelines.matvec;
    let num_tgs = (inter as u64).div_ceil(kernel.rows_per_tg);
    let tg_size = MTLSize::new(kernel.threads_per_tg, 1, 1);
    let n_blocks = (hidden / LEGACY_BLOCK_ELEMS) as u32;

    let (q8_init, q8s_init) = quantize_to_q8(x);

    // Pre-cache weight buffers
    let gate_bufs: Vec<_> = layers_q4
        .iter()
        .map(|(g, _, _)| bufs.get_bytes(g))
        .collect();
    let up_bufs: Vec<_> = layers_q4
        .iter()
        .map(|(_, u, _)| bufs.get_bytes(u))
        .collect();
    let down_bufs: Vec<_> = layers_q4
        .iter()
        .map(|(_, _, d)| bufs.get_bytes(d))
        .collect();

    // Pre-allocate ALL intermediate buffers
    let mut q8_bufs = Vec::with_capacity(num_layers + 1);
    let mut q8s_bufs = Vec::with_capacity(num_layers + 1);
    q8_bufs.push(bufs.transient_from_i8(&q8_init));
    q8s_bufs.push(bufs.transient_from_f32(&q8s_init));

    let mut gate_outs = Vec::with_capacity(num_layers);
    let mut up_outs = Vec::with_capacity(num_layers);
    let mut act_bufs_vec = Vec::with_capacity(num_layers);
    let mut down_outs = Vec::with_capacity(num_layers);

    for _ in 0..num_layers {
        gate_outs.push(bufs.output((inter * 4) as u64));
        up_outs.push(bufs.output((inter * 4) as u64));
        act_bufs_vec.push(bufs.output((inter * 4) as u64));
        down_outs.push(bufs.output((hidden * 4) as u64));
        q8_bufs.push(bufs.output(hidden as u64));
        q8s_bufs.push(bufs.output((hidden / LEGACY_BLOCK_ELEMS * 4) as u64));
    }

    let cmd = queue.new_command_buffer();

    for l in 0..num_layers {
        // Gate
        let enc = cmd.new_compute_command_encoder();
        enc.set_compute_pipeline_state(&kernel.state);
        enc.set_buffer(0, Some(&gate_bufs[l]), 0);
        enc.set_buffer(1, Some(&q8_bufs[l]), 0);
        enc.set_buffer(2, Some(&q8s_bufs[l]), 0);
        enc.set_buffer(3, Some(&gate_outs[l]), 0);
        enc.set_bytes(4, 4, &n_val as *const u32 as *const c_void);
        enc.set_bytes(5, 4, &k_val as *const u32 as *const c_void);
        enc.dispatch_thread_groups(MTLSize::new(num_tgs, 1, 1), tg_size);
        enc.end_encoding();

        // Up
        let enc = cmd.new_compute_command_encoder();
        enc.set_compute_pipeline_state(&kernel.state);
        enc.set_buffer(0, Some(&up_bufs[l]), 0);
        enc.set_buffer(1, Some(&q8_bufs[l]), 0);
        enc.set_buffer(2, Some(&q8s_bufs[l]), 0);
        enc.set_buffer(3, Some(&up_outs[l]), 0);
        enc.set_bytes(4, 4, &n_val as *const u32 as *const c_void);
        enc.set_bytes(5, 4, &k_val as *const u32 as *const c_void);
        enc.dispatch_thread_groups(MTLSize::new(num_tgs, 1, 1), tg_size);
        enc.end_encoding();

        // GEGLU
        let enc = cmd.new_compute_command_encoder();
        enc.set_compute_pipeline_state(geglu_pipeline);
        enc.set_buffer(0, Some(&gate_outs[l]), 0);
        enc.set_buffer(1, Some(&up_outs[l]), 0);
        enc.set_buffer(2, Some(&act_bufs_vec[l]), 0);
        enc.set_bytes(3, 4, &inter_val as *const u32 as *const c_void);
        enc.dispatch_threads(MTLSize::new(inter as u64, 1, 1), MTLSize::new(256, 1, 1));
        enc.end_encoding();

        // Down (f32_matvec on transposed weights)
        let enc = cmd.new_compute_command_encoder();
        enc.set_compute_pipeline_state(&pipelines.f32_matvec);
        enc.set_buffer(0, Some(&down_bufs[l]), 0);
        enc.set_buffer(1, Some(&act_bufs_vec[l]), 0);
        enc.set_buffer(2, Some(&down_outs[l]), 0);
        enc.set_bytes(3, 4, &hidden_val as *const u32 as *const c_void);
        enc.set_bytes(4, 4, &inter_val as *const u32 as *const c_void);
        enc.dispatch_threads(MTLSize::new(hidden as u64, 1, 1), MTLSize::new(256, 1, 1));
        enc.end_encoding();

        // Q8 quantize for next layer (skip last)
        if l + 1 < num_layers {
            let enc = cmd.new_compute_command_encoder();
            enc.set_compute_pipeline_state(q8_quant_pipeline);
            enc.set_buffer(0, Some(&down_outs[l]), 0);
            enc.set_buffer(1, Some(&q8_bufs[l + 1]), 0);
            enc.set_buffer(2, Some(&q8s_bufs[l + 1]), 0);
            enc.set_bytes(3, 4, &hidden_val as *const u32 as *const c_void);
            enc.dispatch_threads(
                MTLSize::new(n_blocks as u64, 1, 1),
                MTLSize::new(
                    crate::kernels::DISPATCH_TG_MAX_THREADS.min(n_blocks as u64),
                    1,
                    1,
                ),
            );
            enc.end_encoding();
        }
    }

    cmd.commit();
    cmd.wait_until_completed();

    let last = num_layers - 1;
    crate::buffers::read_buffer_f32(&down_outs[last], hidden)
}
