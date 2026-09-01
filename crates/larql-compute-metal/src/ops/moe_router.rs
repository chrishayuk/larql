//! Dispatch for the MoE GPU router projection (`shaders::moe_router`).
//!
//! Rung A of the GPU-dataflow routing ladder. Two surfaces:
//!
//! - [`MetalBackend::encode_moe_router_logits`] encodes into a caller-owned
//!   encoder and leaves the logits in a GPU buffer — the shape the later
//!   rungs need, where top-k and descriptor lookup chain after it inside
//!   one pre-encodable command buffer.
//! - [`MetalBackend::moe_router_logits`] is the round-trip wrapper (own
//!   command buffer + readback) used by the CPU-parity rig and any host
//!   caller that still consumes the route.

use crate::shaders::moe_router_select::{MAX_EXPERTS, MAX_TOP_K, TG_THREADS};
use crate::MetalBackend;
use larql_compute::{MoeExpertScalePolicy, MoeLayerWeights, MoeTopKWeightPolicy};

impl MetalBackend {
    /// Encode `logits[E] = W[E,H]·x + bias` into `enc`; returns the
    /// GPU-resident logits buffer. `bias_buf = None` means the
    /// architecture has none (`has_bias = 0`; the kernel never reads the
    /// placeholder binding).
    pub(crate) fn encode_moe_router_logits(
        &self,
        enc: &metal::ComputeCommandEncoderRef,
        w_buf: &metal::Buffer,
        x_buf: &metal::Buffer,
        bias_buf: Option<&metal::Buffer>,
        num_experts: usize,
        hidden: usize,
    ) -> metal::Buffer {
        let out_buf = self.bufs.output((num_experts * 4) as u64);
        let kernel = &self.moe_router_pipeline;
        let e_u32 = num_experts as u32;
        let h_u32 = hidden as u32;
        let has_bias: u32 = bias_buf.is_some() as u32;
        let num_tgs = (num_experts as u64).div_ceil(kernel.rows_per_tg);

        enc.set_compute_pipeline_state(&kernel.state);
        enc.set_buffer(0, Some(w_buf), 0);
        enc.set_buffer(1, Some(x_buf), 0);
        // Metal wants a valid binding even on the has_bias = 0 arm, where
        // the kernel is guarded against ever dereferencing it; x_buf is a
        // live buffer of convenience, not data.
        enc.set_buffer(2, Some(bias_buf.unwrap_or(x_buf)), 0);
        enc.set_buffer(3, Some(&out_buf), 0);
        enc.set_bytes(4, 4, &e_u32 as *const u32 as *const std::ffi::c_void);
        enc.set_bytes(5, 4, &h_u32 as *const u32 as *const std::ffi::c_void);
        enc.set_bytes(6, 4, &has_bias as *const u32 as *const std::ffi::c_void);
        enc.dispatch_thread_groups(
            metal::MTLSize::new(num_tgs, 1, 1),
            metal::MTLSize::new(kernel.threads_per_tg, 1, 1),
        );
        out_buf
    }

    /// Router projection with readback: `logits[e] = dot(x, w[e]) + bias[e]`.
    ///
    /// GPU twin of the CPU oracle
    /// `larql_compute::cpu::ops::moe::moe_router_logits`, and parity-gated
    /// against it (`tests/test_kernel_moe_router.rs`). `bias` empty = the
    /// architecture has none. Returns `None` on shape mismatch, mirroring
    /// the gemv family's contract.
    pub fn moe_router_logits(
        &self,
        router_proj: &[f32],
        router_bias: &[f32],
        x: &[f32],
        num_experts: usize,
        hidden: usize,
    ) -> Option<Vec<f32>> {
        if num_experts == 0
            || router_proj.len() != num_experts * hidden
            || x.len() != hidden
            || !(router_bias.is_empty() || router_bias.len() == num_experts)
        {
            return None;
        }
        let w_buf = self.bufs.get_f32(router_proj);
        let x_buf = self.bufs.transient_from_f32(x);
        let bias_buf = if router_bias.is_empty() {
            None
        } else {
            Some(self.bufs.transient_from_f32(router_bias))
        };

        let cmd = self.queue.new_command_buffer();
        let enc = cmd.new_compute_command_encoder();
        let out_buf = self.encode_moe_router_logits(
            enc,
            &w_buf,
            &x_buf,
            bias_buf.as_ref(),
            num_experts,
            hidden,
        );
        enc.end_encoding();
        cmd.commit();
        crate::cb_status::wait_or_abort(cmd, "crates/larql-compute-metal/src/ops/moe_router.rs:98");

        crate::buffers::try_read_buffer_f32(&out_buf, num_experts)
    }

    /// Encode the fused route selection (softmax → deterministic top-k →
    /// weight policy) over a GPU-resident logits buffer; returns
    /// `(selected_ids, selected_weights)` buffers, still GPU-resident.
    ///
    /// `pe_scale_buf = None` disables the per-expert scale arm;
    /// `renormalize` maps `MoeTopKWeightPolicy`. Callers must have
    /// enforced `num_experts ≤ MAX_EXPERTS` and `1 ≤ top_k ≤ MAX_TOP_K`.
    pub(crate) fn encode_moe_router_select(
        &self,
        enc: &metal::ComputeCommandEncoderRef,
        logits_buf: &metal::Buffer,
        pe_scale_buf: Option<&metal::Buffer>,
        num_experts: usize,
        top_k: usize,
        renormalize: bool,
    ) -> (metal::Buffer, metal::Buffer) {
        let ids_buf = self.bufs.output((top_k * 4) as u64);
        let w_buf = self.bufs.output((top_k * 4) as u64);
        let e_u32 = num_experts as u32;
        let k_u32 = top_k as u32;
        let renorm: u32 = renormalize as u32;
        let has_scale: u32 = pe_scale_buf.is_some() as u32;

        enc.set_compute_pipeline_state(&self.moe_router_select_pipeline);
        enc.set_buffer(0, Some(logits_buf), 0);
        // Placeholder binding on the has_scale = 0 arm, never read.
        enc.set_buffer(1, Some(pe_scale_buf.unwrap_or(logits_buf)), 0);
        enc.set_buffer(2, Some(&ids_buf), 0);
        enc.set_buffer(3, Some(&w_buf), 0);
        enc.set_bytes(4, 4, &e_u32 as *const u32 as *const std::ffi::c_void);
        enc.set_bytes(5, 4, &k_u32 as *const u32 as *const std::ffi::c_void);
        enc.set_bytes(6, 4, &renorm as *const u32 as *const std::ffi::c_void);
        enc.set_bytes(7, 4, &has_scale as *const u32 as *const std::ffi::c_void);
        enc.dispatch_thread_groups(
            metal::MTLSize::new(1, 1, 1),
            metal::MTLSize::new(TG_THREADS, 1, 1),
        );
        (ids_buf, w_buf)
    }

    /// Full GPU route: projection (rung A) chained into fused selection
    /// (rung B) in one command buffer, with readback.
    ///
    /// GPU twin of `larql_compute::cpu::ops::moe::moe_route_from_router_input`
    /// — same inputs, same `(indices, weights)` outputs, same policy
    /// semantics driven by `moe.routing_policy` — and parity-gated against
    /// it (`tests/test_kernel_moe_router_select.rs`). Does not fire the
    /// `moe_route_observe` trace hook; production integration decides
    /// where observation lives once routing stops being a host decision.
    ///
    /// Returns `None` on any shape the kernels don't cover
    /// (`num_experts > 256`, `top_k > 32`, mismatched tables) — the
    /// caller falls back to the CPU route, never a wrong dispatch.
    pub fn moe_route_gpu(
        &self,
        router_in: &[f32],
        moe: &MoeLayerWeights<'_>,
    ) -> Option<(Vec<usize>, Vec<f32>)> {
        let num_experts = moe.num_experts;
        let top_k = moe.top_k;
        let hidden = router_in.len();
        if num_experts == 0
            || num_experts > MAX_EXPERTS
            || top_k == 0
            || top_k > MAX_TOP_K
            || top_k > num_experts
            || moe.router_proj.len() != num_experts * hidden
            || !(moe.router_bias.is_empty() || moe.router_bias.len() == num_experts)
        {
            return None;
        }
        let renormalize =
            moe.routing_policy.selected_weight == MoeTopKWeightPolicy::RenormalizedSoftmax;
        // The CPU oracle engages the scale table only under the PerExpert
        // policy AND a non-empty table; it also tolerates a short table by
        // skipping out-of-range experts. A short table is a malformed
        // router description, not a case to reproduce — refuse it.
        let pe_scale = (moe.routing_policy.expert_scale == MoeExpertScalePolicy::PerExpert
            && !moe.router_per_expert_scale.is_empty())
        .then_some(moe.router_per_expert_scale);
        if let Some(s) = pe_scale {
            if s.len() != num_experts {
                return None;
            }
        }

        let w_buf = self.bufs.get_f32(moe.router_proj);
        let x_buf = self.bufs.transient_from_f32(router_in);
        let bias_buf = if moe.router_bias.is_empty() {
            None
        } else {
            Some(self.bufs.transient_from_f32(moe.router_bias))
        };
        let scale_buf = pe_scale.map(|s| self.bufs.transient_from_f32(s));

        let cmd = self.queue.new_command_buffer();
        let enc = cmd.new_compute_command_encoder();
        let logits_buf = self.encode_moe_router_logits(
            enc,
            &w_buf,
            &x_buf,
            bias_buf.as_ref(),
            num_experts,
            hidden,
        );
        let (ids_buf, weights_buf) = self.encode_moe_router_select(
            enc,
            &logits_buf,
            scale_buf.as_ref(),
            num_experts,
            top_k,
            renormalize,
        );
        enc.end_encoding();
        cmd.commit();
        crate::cb_status::wait_or_abort(
            cmd,
            "crates/larql-compute-metal/src/ops/moe_router.rs:218",
        );

        let weights = crate::buffers::try_read_buffer_f32(&weights_buf, top_k)?;
        let ids: Vec<usize> = unsafe {
            let ptr = ids_buf.contents() as *const u32;
            if ptr.is_null() {
                return None;
            }
            std::slice::from_raw_parts(ptr, top_k)
                .iter()
                .map(|&i| i as usize)
                .collect()
        };
        Some((ids, weights))
    }
}
