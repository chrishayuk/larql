//! The dense (non-expert) Q4_K FFN dispatch.
//!
//! The dense (non-expert) Q4_K FFN dispatch.
//!
//! Split out of `moe_dispatch.rs`; [`super`] composes the pieces.

#[allow(unused_imports)]
use super::*;
use crate::buffers::read_buffer_f32;
use crate::MetalBackend;
use metal::*;
use std::ffi::c_void;

impl MetalBackend {
    /// Run one dense (non-MoE) FFN layer on GPU using pre-loaded Q4K weight buffers.
    ///
    /// `h_norm` is the f32 FFN-input norm output, length = `hidden`.
    /// Gate and up projections run via `q4k_ffn_gate_up_8sg_pipeline`;
    /// activation via `geglu_gelu_tanh_pipeline`; down via `q4k_matvec_pipeline`.
    ///
    /// All three weight buffers must be pre-created from the mmap byte slices via
    /// `BufferCache::get_bytes` (zero-copy for page-aligned mmap data).
    ///
    /// Returns `Vec<f32>` of length `hidden` — the FFN delta (no residual add).
    #[allow(clippy::too_many_arguments)]
    pub fn run_dense_ffn_q4k(
        &self,
        h_norm: &[f32],
        gate_buf: &Buffer,
        up_buf: &Buffer,
        down_buf: &Buffer,
        hidden: usize,
        inter: usize,
        inter_padded: usize,
    ) -> Vec<f32> {
        if hidden == 0 || inter == 0 {
            return vec![0.0f32; hidden];
        }

        // Stage h_norm into a transient f32 buffer.
        let x_buf = self.bufs.transient_from_f32(h_norm);

        // Allocate scratch buffers.
        let gate_out = self.bufs.output((inter * 4) as u64);
        let up_out = self.bufs.output((inter * 4) as u64);
        let act_buf = self.bufs.output((inter_padded * 4) as u64);
        let out_buf = self.bufs.output((hidden * 4) as u64);

        let cmd = self.queue.new_command_buffer();
        let enc = cmd.new_compute_command_encoder();

        // 1. q4k_ffn_gate_up_8sg — gate and up projections.
        // Geometry pulled from the bound `KernelHandle` so a future
        // pipeline swap can't drift from the dispatched TG shape.
        let gate_up_kh = &self.ffn.q4k_ffn_gate_up_8sg_pipeline;
        let n_rows = inter as u32;
        let k_cols = hidden as u32;
        let n_tgs = (inter as u64).div_ceil(gate_up_kh.rows_per_tg);
        enc.set_compute_pipeline_state(&gate_up_kh.state);
        enc.set_buffer(0, Some(gate_buf), 0);
        enc.set_buffer(1, Some(up_buf), 0);
        enc.set_buffer(2, Some(&x_buf), 0);
        enc.set_buffer(3, Some(&gate_out), 0);
        enc.set_buffer(4, Some(&up_out), 0);
        enc.set_bytes(5, 4, &n_rows as *const u32 as *const c_void);
        enc.set_bytes(6, 4, &k_cols as *const u32 as *const c_void);
        enc.dispatch_thread_groups(
            MTLSize::new(n_tgs * 2, 1, 1),
            MTLSize::new(gate_up_kh.threads_per_tg, 1, 1),
        );

        // 2. geglu_gelu_tanh activation.
        let inter_u32 = inter as u32;
        enc.set_compute_pipeline_state(&self.ffn.geglu_gelu_tanh_pipeline);
        enc.set_buffer(0, Some(&gate_out), 0);
        enc.set_buffer(1, Some(&up_out), 0);
        enc.set_buffer(2, Some(&act_buf), 0);
        enc.set_bytes(3, 4, &inter_u32 as *const u32 as *const c_void);
        enc.dispatch_threads(
            MTLSize::new(inter as u64, 1, 1),
            MTLSize::new(
                crate::kernels::DISPATCH_TG_MAX_THREADS.min(inter as u64),
                1,
                1,
            ),
        );

        // 3. q4k_matvec down projection.
        // Pull dispatch geometry from the bound pipeline (not hardcoded) to avoid
        // the 4sg-vs-8sg dispatch geometry mismatch bug documented in ROADMAP.
        let n_out = hidden as u32;
        let k_in = inter_padded as u32;
        let down_rows_per_tg = self.quant.q4k_matvec_pipeline.rows_per_tg;
        let down_threads_per_tg = self.quant.q4k_matvec_pipeline.threads_per_tg;
        let down_tgs = (hidden as u64).div_ceil(down_rows_per_tg);
        enc.set_compute_pipeline_state(&self.quant.q4k_matvec_pipeline.state);
        enc.set_buffer(0, Some(down_buf), 0);
        enc.set_buffer(1, Some(&act_buf), 0);
        enc.set_buffer(2, Some(&out_buf), 0);
        enc.set_bytes(3, 4, &n_out as *const u32 as *const c_void);
        enc.set_bytes(4, 4, &k_in as *const u32 as *const c_void);
        enc.dispatch_thread_groups(
            MTLSize::new(down_tgs, 1, 1),
            MTLSize::new(down_threads_per_tg, 1, 1),
        );

        enc.end_encoding();
        cmd.commit();
        crate::cb_status::wait_or_abort(
            cmd,
            "crates/larql-compute-metal/src/moe_dispatch/dense.rs:109",
        );

        let result = read_buffer_f32(&out_buf, hidden);

        // Recycle scratch buffers back to the pool.
        self.bufs.recycle(gate_out);
        self.bufs.recycle(up_out);
        self.bufs.recycle(act_buf);
        self.bufs.recycle(out_buf);

        result
    }
}
