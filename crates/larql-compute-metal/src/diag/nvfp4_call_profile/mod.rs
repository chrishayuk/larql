//! Where does a single NVFP4 gemv call's ~300 us intercept go?
//!
//! Measured so far: interpreter glue is 7-8 ms/token, Metal's empty
//! commit+wait floor is ~19 us, and real calls fit
//! `300 us fixed + bytes / ~332 GB/s`. Across 209 calls that intercept is
//! ~63 ms/token — the single largest term in Glimmer decode.
//!
//! What has *not* been shown is that the intercept is host-side. An empty
//! command buffer does not exercise resource binding, residency of
//! multi-GB weight buffers, or GPU dispatch setup, so the remainder could
//! sit in any of those. This splits one call into phases that attribute
//! it, including the **GPU's own timestamps** — the measurement that
//! decides between "fix the host path" and "fuse or re-geometry the
//! kernel".
//!
//! `GPUStartTime`/`GPUEndTime` are not exposed by metal-rs 0.29, so they
//! are read through `objc`; both are `CFTimeInterval` (seconds) and are
//! valid only after the buffer completes.

use std::time::Instant;

use crate::MetalBackend;

/// One call, broken down. All values microseconds.
#[derive(Debug, Default, Clone, Copy)]
pub struct CallProfile {
    /// Acquiring the pooled input buffer and copying `x` into it.
    pub input_stage: f64,
    /// Acquiring weight buffers (cache hits) and the output buffer.
    pub buffer_acquire: f64,
    /// Command buffer + encoder creation, bindings, dispatch, end.
    pub encode: f64,
    /// `commit()` itself.
    pub commit: f64,
    /// `wait_until_completed()` — the CPU blocked.
    pub wait: f64,
    /// GPU-reported execution span (`GPUEndTime - GPUStartTime`).
    pub gpu_span: f64,
    /// Commit returning to the GPU starting work — queue latency.
    pub commit_to_gpu_start: f64,
    /// Reading the output back into a `Vec` and recycling buffers.
    pub readback: f64,
    /// Whole call, wall.
    pub total: f64,
}

/// Read `GPUStartTime` / `GPUEndTime` off a completed command buffer.
fn gpu_times(cmd: &metal::CommandBufferRef) -> (f64, f64) {
    use metal::foreign_types::ForeignTypeRef;
    use objc::{msg_send, sel, sel_impl};
    let raw: *mut objc::runtime::Object = cmd.as_ptr() as *mut _;
    // SAFETY: both selectors exist on MTLCommandBuffer and return
    // CFTimeInterval (double); the buffer has completed, so they are
    // populated.
    unsafe {
        let start: f64 = msg_send![raw, GPUStartTime];
        let end: f64 = msg_send![raw, GPUEndTime];
        (start, end)
    }
}

impl MetalBackend {
    /// Run one NVFP4 gemv with per-phase timing. Mirrors
    /// `nvfp4_gemv_multi`'s single-matrix path exactly — same kernel,
    /// same buffers, same order — so the phases add up to a real call
    /// rather than to a lookalike.
    pub fn profile_nvfp4_gemv(
        &self,
        packed: &[u8],
        scales: &[u8],
        tensor_scale: f32,
        x: &[f32],
        n: usize,
        k: usize,
    ) -> Option<CallProfile> {
        let mut p = CallProfile::default();
        let call_started = Instant::now();

        let t = Instant::now();
        let x_buf = self.bufs.output((x.len() * 4) as u64);
        let x_ptr = x_buf.contents() as *mut f32;
        if x_ptr.is_null() {
            return None;
        }
        // SAFETY: pooled buffer is at least x.len()*4 bytes, not yet bound.
        unsafe { std::ptr::copy_nonoverlapping(x.as_ptr(), x_ptr, x.len()) };
        p.input_stage = t.elapsed().as_secs_f64() * 1e6;

        let t = Instant::now();
        let p_buf = self.bufs.get_bytes(packed);
        let s_buf = self.bufs.get_bytes(scales);
        let out_buf = self.bufs.output((n * 4) as u64);
        p.buffer_acquire = t.elapsed().as_secs_f64() * 1e6;

        let t = Instant::now();
        let kernel = &self.quant.nvfp4_matvec_pipeline;
        let cmd = self.queue.new_command_buffer();
        let enc = cmd.new_compute_command_encoder();
        enc.set_compute_pipeline_state(&kernel.state);
        let (m_u32, k_u32) = (n as u32, k as u32);
        enc.set_buffer(0, Some(&p_buf), 0);
        enc.set_buffer(1, Some(&s_buf), 0);
        enc.set_buffer(2, Some(&x_buf), 0);
        enc.set_buffer(3, Some(&out_buf), 0);
        enc.set_bytes(4, 4, &m_u32 as *const u32 as *const std::ffi::c_void);
        enc.set_bytes(5, 4, &k_u32 as *const u32 as *const std::ffi::c_void);
        enc.set_bytes(6, 4, &tensor_scale as *const f32 as *const std::ffi::c_void);
        enc.dispatch_thread_groups(
            metal::MTLSize::new((n as u64).div_ceil(kernel.rows_per_tg), 1, 1),
            metal::MTLSize::new(kernel.threads_per_tg, 1, 1),
        );
        enc.end_encoding();
        p.encode = t.elapsed().as_secs_f64() * 1e6;

        let t = Instant::now();
        let commit_wall = Instant::now();
        cmd.commit();
        p.commit = t.elapsed().as_secs_f64() * 1e6;

        let t = Instant::now();
        crate::cb_status::wait_or_abort(
            cmd,
            "crates/larql-compute-metal/src/diag/nvfp4_call_profile.rs:121",
        );
        p.wait = t.elapsed().as_secs_f64() * 1e6;
        let commit_to_done = commit_wall.elapsed().as_secs_f64() * 1e6;

        let (gpu_start, gpu_end) = gpu_times(cmd);
        p.gpu_span = (gpu_end - gpu_start) * 1e6;
        // The GPU clock and Instant share no epoch, so queue latency is
        // derived: whatever the commit-to-completion wall was, minus the
        // span the GPU says it was busy.
        p.commit_to_gpu_start = commit_to_done - p.gpu_span;

        let t = Instant::now();
        let out = crate::buffers::try_read_buffer_f32(&out_buf, n)?;
        self.bufs.recycle(out_buf);
        self.bufs.recycle(x_buf);
        p.readback = t.elapsed().as_secs_f64() * 1e6;

        p.total = call_started.elapsed().as_secs_f64() * 1e6;
        debug_assert_eq!(out.len(), n);
        Some(p)
    }
}

impl MetalBackend {
    /// Commit `depth` identical dispatches back to back and wait **once**
    /// at the end, returning microseconds per dispatch.
    ///
    /// The commit-to-GPU-start latency measured above is ~230 us and flat
    /// in bytes. That is either an unavoidable per-dispatch charge, or the
    /// cost of letting the queue drain between every commit — a decode
    /// that waits after each of 209 dispatches never has a second buffer
    /// queued, so the GPU idles and re-wakes 209 times per token. If the
    /// latency pipelines, per-dispatch cost collapses as `depth` rises and
    /// the fix is to stop waiting per call; if it does not, the charge is
    /// real and only fusion removes it.
    // Mirrors `nvfp4_gemv`'s argument list on purpose: a diagnostic that
    // took a different shape from the call it models would be measuring
    // something else.
    #[allow(clippy::too_many_arguments)]
    pub fn nvfp4_pipelined_cost(
        &self,
        packed: &[u8],
        scales: &[u8],
        tensor_scale: f32,
        x: &[f32],
        n: usize,
        k: usize,
        depth: usize,
    ) -> Option<f64> {
        let x_buf = self.bufs.output((x.len() * 4) as u64);
        let x_ptr = x_buf.contents() as *mut f32;
        if x_ptr.is_null() {
            return None;
        }
        // SAFETY: pooled buffer is large enough and not yet bound.
        unsafe { std::ptr::copy_nonoverlapping(x.as_ptr(), x_ptr, x.len()) };
        let p_buf = self.bufs.get_bytes(packed);
        let s_buf = self.bufs.get_bytes(scales);
        // Distinct outputs so the dispatches stay genuinely independent;
        // sharing one would let the driver serialise on a write hazard.
        let outs: Vec<metal::Buffer> = (0..depth)
            .map(|_| self.bufs.output((n * 4) as u64))
            .collect();
        let kernel = &self.quant.nvfp4_matvec_pipeline;
        let (m_u32, k_u32) = (n as u32, k as u32);

        let started = Instant::now();
        let mut last: Option<metal::CommandBuffer> = None;
        for out_buf in &outs {
            let cmd = self.queue.new_command_buffer().to_owned();
            let enc = cmd.new_compute_command_encoder();
            enc.set_compute_pipeline_state(&kernel.state);
            enc.set_buffer(0, Some(&p_buf), 0);
            enc.set_buffer(1, Some(&s_buf), 0);
            enc.set_buffer(2, Some(&x_buf), 0);
            enc.set_buffer(3, Some(out_buf), 0);
            enc.set_bytes(4, 4, &m_u32 as *const u32 as *const std::ffi::c_void);
            enc.set_bytes(5, 4, &k_u32 as *const u32 as *const std::ffi::c_void);
            enc.set_bytes(6, 4, &tensor_scale as *const f32 as *const std::ffi::c_void);
            enc.dispatch_thread_groups(
                metal::MTLSize::new((n as u64).div_ceil(kernel.rows_per_tg), 1, 1),
                metal::MTLSize::new(kernel.threads_per_tg, 1, 1),
            );
            enc.end_encoding();
            cmd.commit();
            last = Some(cmd);
        }
        if let Some(cmd) = last {
            crate::cb_status::wait_or_abort(
                &cmd,
                "crates/larql-compute-metal/src/diag/nvfp4_call_profile.rs:209",
            );
        }
        let per = started.elapsed().as_secs_f64() * 1e6 / depth as f64;

        for b in outs {
            self.bufs.recycle(b);
        }
        self.bufs.recycle(x_buf);
        Some(per)
    }
}

#[cfg(test)]
mod tests;
