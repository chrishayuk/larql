//! Expert-block execution: prestaged, preselected, and the shared dispatch.
//!
//! Split out of `moe_dispatch.rs`; [`super`] composes the pieces.

#[allow(unused_imports)]
use super::*;
use crate::buffers::read_buffer_f32;
use crate::MetalBackend;
use metal::*;
use std::ffi::c_void;

impl MetalBackend {
    /// Pre-staged variant of `run_experts_preselected_metal`: takes per-expert
    /// `(gate_up_buf, down_buf)` Metal buffers (typically created once via
    /// `shared_buffer_no_copy` at server startup) instead of byte slices that
    /// would have to be memcpy'd into scratch on every call.
    ///
    /// Same wire output as `run_experts_preselected_metal` — only the staging
    /// path differs.  Because each expert's weights live in its own buffer we
    /// dispatch `q4k_ffn_gate_up` once per expert rather than once-for-all-K;
    /// the per-dispatch cost (~10–50µs on M3) is dwarfed by the eliminated
    /// memcpy (~1ms/layer at K=8).
    #[allow(clippy::too_many_arguments)]
    pub fn run_experts_prestaged_metal(
        &self,
        h_norm: &[f32],
        expert_bufs: &[(Buffer, Buffer)],
        expert_weights: &[f32],
        scratch: &MoeScratch,
    ) -> Vec<f32> {
        let hidden = h_norm.len();
        let inter = scratch.inter;
        let inter_padded = scratch.inter_padded;
        debug_assert_eq!(hidden, scratch.hidden);
        debug_assert_eq!(expert_bufs.len(), expert_weights.len());

        if expert_bufs.is_empty() || hidden == 0 || inter == 0 {
            return vec![0.0f32; hidden];
        }

        let timing_enabled =
            larql_compute::options::env_flag(larql_compute::options::ENV_METAL_MOE_TIMING);
        let t_start = std::time::Instant::now();

        let valid_count = expert_bufs.len().min(scratch.top_k);

        // Stage h_norm only (small — `hidden * 4` bytes).
        unsafe {
            let x_ptr = scratch.x_buf.contents() as *mut f32;
            std::ptr::copy_nonoverlapping(h_norm.as_ptr(), x_ptr, hidden);
        }
        let t_stage = t_start.elapsed();

        let cmd = self.queue.new_command_buffer();
        let enc = cmd.new_compute_command_encoder();

        // Per-expert gate+up dispatch.  Each expert's `gate_up_buf` holds
        // `[gate || up]`; the kernel takes them as separate buffers — pass
        // the same buffer twice with the up offset for the second binding.
        let row_bytes = scratch.row_bytes;
        let gate_half_bytes = (inter * row_bytes) as u64;
        let n_rows = inter as u32;
        let k_cols = hidden as u32;
        // Geometry travels with `q4k_ffn_gate_up_pipeline` — read it
        // off the `KernelHandle` rather than re-importing shader-module
        // constants, so a future bump of the field to a different
        // simdgroup variant doesn't silently drop rows. Same dispatch-
        // geometry-mismatch class as the q4_matvec_v4 ROADMAP entry.
        let gate_up_kh = &self.ffn.q4k_ffn_gate_up_pipeline;
        let tgs_per_mat = (inter as u64).div_ceil(gate_up_kh.rows_per_tg);

        for (e, (gate_up_buf, _)) in expert_bufs.iter().enumerate().take(valid_count) {
            enc.set_compute_pipeline_state(&gate_up_kh.state);
            // Wg = gate (offset 0), Wu = up (offset gate_half_bytes) within the
            // same per-expert mmap-backed buffer.
            enc.set_buffer(0, Some(gate_up_buf), 0);
            enc.set_buffer(1, Some(gate_up_buf), gate_half_bytes);
            enc.set_buffer(2, Some(&scratch.x_buf), 0);
            // Per-expert output offsets so K dispatches don't clobber each
            // other; same offsets the GELU/down dispatches read below.
            enc.set_buffer(3, Some(&scratch.g_out), (e * inter * 4) as u64);
            enc.set_buffer(4, Some(&scratch.u_out), (e * inter * 4) as u64);
            enc.set_bytes(5, 4, &n_rows as *const u32 as *const c_void);
            enc.set_bytes(6, 4, &k_cols as *const u32 as *const c_void);
            enc.dispatch_thread_groups(
                MTLSize::new(tgs_per_mat * 2, 1, 1),
                MTLSize::new(gate_up_kh.threads_per_tg, 1, 1),
            );
        }

        // GELU-tanh activation per expert (strided to inter_padded).
        let inter_u32 = inter as u32;
        for e in 0..valid_count {
            let g_offset = (e * inter * 4) as u64;
            let u_offset = (e * inter * 4) as u64;
            let a_offset = (e * inter_padded * 4) as u64;
            enc.set_compute_pipeline_state(&self.ffn.geglu_gelu_tanh_pipeline);
            enc.set_buffer(0, Some(&scratch.g_out), g_offset);
            enc.set_buffer(1, Some(&scratch.u_out), u_offset);
            enc.set_buffer(2, Some(&scratch.act_buf), a_offset);
            enc.set_bytes(3, 4, &inter_u32 as *const u32 as *const c_void);
            enc.dispatch_threads(
                MTLSize::new(inter as u64, 1, 1),
                MTLSize::new(
                    crate::kernels::DISPATCH_TG_MAX_THREADS.min(inter as u64),
                    1,
                    1,
                ),
            );
        }

        // Per-expert down projection — use each expert's pre-staged down buffer.
        let n_out = hidden as u32;
        let k_in = inter_padded as u32;
        // Pull dispatch geometry from the bound pipeline so this works for
        // both the 4sg and 8sg variants of `q4k_matvec` — hardcoding the
        // 4sg constants while dispatching the 8sg pipeline (the production
        // default since 2026-04-28) leaves simdgroups 4..7 unscheduled and
        // only writes rows 0..3 of each TG's 8-row range. See the matching
        // fix in `trait_impl/quant_matvec.rs::q4k_matvec`.
        let down_rows_per_tg = self.quant.q4k_matvec_pipeline.rows_per_tg;
        let down_threads_per_tg = self.quant.q4k_matvec_pipeline.threads_per_tg;
        let down_tgs = (hidden as u64).div_ceil(down_rows_per_tg);
        for (e, (_, down_buf)) in expert_bufs.iter().enumerate().take(valid_count) {
            let act_offset = (e * inter_padded * 4) as u64;
            let out_offset = (e * hidden * 4) as u64;
            enc.set_compute_pipeline_state(&self.quant.q4k_matvec_pipeline.state);
            enc.set_buffer(0, Some(down_buf), 0);
            enc.set_buffer(1, Some(&scratch.act_buf), act_offset);
            enc.set_buffer(2, Some(&scratch.expert_outs), out_offset);
            enc.set_bytes(3, 4, &n_out as *const u32 as *const c_void);
            enc.set_bytes(4, 4, &k_in as *const u32 as *const c_void);
            enc.dispatch_thread_groups(
                MTLSize::new(down_tgs, 1, 1),
                MTLSize::new(down_threads_per_tg, 1, 1),
            );
        }
        enc.end_encoding();
        cmd.commit();
        crate::cb_status::wait_or_abort(
            cmd,
            "crates/larql-compute-metal/src/moe_dispatch/experts.rs:140",
        );
        let t_gpu = t_start.elapsed();

        let all_expert_outputs = read_buffer_f32(&scratch.expert_outs, valid_count * hidden);
        let mut moe_out = vec![0.0f32; hidden];
        for e in 0..valid_count {
            let w = expert_weights[e];
            let out_slice = &all_expert_outputs[e * hidden..(e + 1) * hidden];
            for (acc, &v) in moe_out.iter_mut().zip(out_slice) {
                *acc += v * w;
            }
        }
        let t_total = t_start.elapsed();
        if timing_enabled {
            eprintln!(
                "[run_experts_metal/prestaged] K={valid_count} stage={:.2}ms gpu={:.2}ms \
                 readback+sum={:.2}ms total={:.2}ms",
                t_stage.as_secs_f32() * 1000.0,
                (t_gpu - t_stage).as_secs_f32() * 1000.0,
                (t_total - t_gpu).as_secs_f32() * 1000.0,
                t_total.as_secs_f32() * 1000.0,
            );
        }
        moe_out
    }

    /// Run a pre-selected set of MoE experts on the GPU and return their
    /// weighted sum.  Public surface used by `larql-server`'s shard endpoint —
    /// the client picks experts via its router, the server only computes them.
    ///
    /// `h_norm` is the *already* `pre_experts_norm`-applied residual.
    /// `expert_ids` and `expert_weights` are paired (both length K).
    /// `get_expert_bytes(eid)` returns `(gate_up_bytes, down_bytes)` mmap
    /// slices for one expert; if the shard does not own the expert it should
    /// return `None` (that expert is skipped).
    ///
    /// Returns the weighted sum **without** post-experts norm — the client
    /// applies post-norm once after summing across shards, since
    /// `rms_norm(a) + rms_norm(b) ≠ rms_norm(a + b)`.
    #[allow(clippy::too_many_arguments)]
    pub fn run_experts_preselected_metal<'w, F>(
        &self,
        h_norm: &[f32],
        expert_ids: &[usize],
        expert_weights: &[f32],
        scratch: &MoeScratch,
        get_expert_bytes: F,
    ) -> Vec<f32>
    where
        F: Fn(usize) -> Option<(&'w [u8], &'w [u8])>,
    {
        let hidden = h_norm.len();
        let inter = scratch.inter;
        let inter_padded = scratch.inter_padded;
        debug_assert_eq!(hidden, scratch.hidden, "h_norm hidden vs scratch.hidden");
        debug_assert!(
            expert_ids.len() == expert_weights.len(),
            "expert_ids and expert_weights must be same length"
        );

        if expert_ids.is_empty() || hidden == 0 || inter == 0 {
            return vec![0.0f32; hidden];
        }

        let timing_enabled =
            larql_compute::options::env_flag(larql_compute::options::ENV_METAL_MOE_TIMING);
        let t_start = std::time::Instant::now();

        // ── Stage expert weight bytes into pre-allocated Metal buffers ─────
        let row_bytes = scratch.row_bytes;
        let gate_half_bytes = inter * row_bytes;
        let up_half_bytes = inter * row_bytes;
        let down_expert_bytes = hidden * scratch.down_row_bytes;

        let gate_ptr = scratch.gate_buf.contents() as *mut u8;
        let up_ptr = scratch.up_buf.contents() as *mut u8;

        let mut valid_weights: Vec<f32> = Vec::with_capacity(expert_ids.len());
        let mut valid_count = 0usize;

        for (k, &ei) in expert_ids.iter().enumerate() {
            let Some((gu_bytes, dn_bytes)) = get_expert_bytes(ei) else {
                continue;
            };
            if gu_bytes.len() < 2 * gate_half_bytes {
                continue;
            }
            if valid_count >= scratch.top_k {
                // Caller passed more experts than scratch was sized for.
                // Truncate to fit; should not happen in practice (client's
                // top_k matches the architecture's top_k that scratch was
                // allocated for).
                break;
            }

            // Q4_K layout: gate || up, each `inter * row_bytes` bytes.
            // SAFETY: gate_ptr / up_ptr are StorageModeShared Metal buffer
            // contents; offsets are bounded by `top_k * gate_half_bytes`.
            unsafe {
                std::ptr::copy_nonoverlapping(
                    gu_bytes.as_ptr(),
                    gate_ptr.add(valid_count * gate_half_bytes),
                    gate_half_bytes,
                );
                std::ptr::copy_nonoverlapping(
                    gu_bytes.as_ptr().add(gate_half_bytes),
                    up_ptr.add(valid_count * up_half_bytes),
                    up_half_bytes,
                );
            }

            let dn_dst = scratch.down_bufs[valid_count].contents() as *mut u8;
            let copy_len = dn_bytes.len().min(down_expert_bytes);
            unsafe {
                std::ptr::copy_nonoverlapping(dn_bytes.as_ptr(), dn_dst, copy_len);
                if copy_len < down_expert_bytes {
                    std::ptr::write_bytes(dn_dst.add(copy_len), 0, down_expert_bytes - copy_len);
                }
            }

            valid_weights.push(expert_weights[k]);
            valid_count += 1;
        }

        if valid_count == 0 {
            return vec![0.0f32; hidden];
        }

        // ── Stage h_norm into pre-allocated x_buf ─────────────────────────
        unsafe {
            let x_ptr = scratch.x_buf.contents() as *mut f32;
            std::ptr::copy_nonoverlapping(h_norm.as_ptr(), x_ptr, hidden);
        }
        let t_stage = t_start.elapsed();

        let cmd = self.queue.new_command_buffer();
        let enc = cmd.new_compute_command_encoder();

        // q4k_ffn_gate_up over all valid_count experts at once.
        // Geometry travels with the `KernelHandle` (see `decode_hybrid.rs`
        // ship-log entry on this bug class).
        let gate_up_kh = &self.ffn.q4k_ffn_gate_up_pipeline;
        let n_rows = (valid_count * inter) as u32;
        let k_cols = hidden as u32;
        let tgs = (valid_count as u64 * inter as u64).div_ceil(gate_up_kh.rows_per_tg);

        enc.set_compute_pipeline_state(&gate_up_kh.state);
        enc.set_buffer(0, Some(&scratch.gate_buf), 0);
        enc.set_buffer(1, Some(&scratch.up_buf), 0);
        enc.set_buffer(2, Some(&scratch.x_buf), 0);
        enc.set_buffer(3, Some(&scratch.g_out), 0);
        enc.set_buffer(4, Some(&scratch.u_out), 0);
        enc.set_bytes(5, 4, &n_rows as *const u32 as *const c_void);
        enc.set_bytes(6, 4, &k_cols as *const u32 as *const c_void);
        enc.dispatch_thread_groups(
            MTLSize::new(tgs * 2, 1, 1),
            MTLSize::new(gate_up_kh.threads_per_tg, 1, 1),
        );

        // GELU-tanh activation per expert (strided to inter_padded).
        let inter_u32 = inter as u32;
        for e in 0..valid_count {
            let g_offset = (e * inter * 4) as u64;
            let u_offset = (e * inter * 4) as u64;
            let a_offset = (e * inter_padded * 4) as u64;
            enc.set_compute_pipeline_state(&self.ffn.geglu_gelu_tanh_pipeline);
            enc.set_buffer(0, Some(&scratch.g_out), g_offset);
            enc.set_buffer(1, Some(&scratch.u_out), u_offset);
            enc.set_buffer(2, Some(&scratch.act_buf), a_offset);
            enc.set_bytes(3, 4, &inter_u32 as *const u32 as *const c_void);
            enc.dispatch_threads(
                MTLSize::new(inter as u64, 1, 1),
                MTLSize::new(
                    crate::kernels::DISPATCH_TG_MAX_THREADS.min(inter as u64),
                    1,
                    1,
                ),
            );
        }

        // Down projection per expert.
        let n_out = hidden as u32;
        let k_in = inter_padded as u32;
        // Pull dispatch geometry from the bound pipeline so this works for
        // both the 4sg and 8sg variants of `q4k_matvec` — hardcoding the
        // 4sg constants while dispatching the 8sg pipeline (the production
        // default since 2026-04-28) leaves simdgroups 4..7 unscheduled and
        // only writes rows 0..3 of each TG's 8-row range. See the matching
        // fix in `trait_impl/quant_matvec.rs::q4k_matvec`.
        let down_rows_per_tg = self.quant.q4k_matvec_pipeline.rows_per_tg;
        let down_threads_per_tg = self.quant.q4k_matvec_pipeline.threads_per_tg;
        let down_tgs = (hidden as u64).div_ceil(down_rows_per_tg);

        for e in 0..valid_count {
            let act_offset = (e * inter_padded * 4) as u64;
            let out_offset = (e * hidden * 4) as u64;
            enc.set_compute_pipeline_state(&self.quant.q4k_matvec_pipeline.state);
            enc.set_buffer(0, Some(&scratch.down_bufs[e]), 0);
            enc.set_buffer(1, Some(&scratch.act_buf), act_offset);
            enc.set_buffer(2, Some(&scratch.expert_outs), out_offset);
            enc.set_bytes(3, 4, &n_out as *const u32 as *const c_void);
            enc.set_bytes(4, 4, &k_in as *const u32 as *const c_void);
            enc.dispatch_thread_groups(
                MTLSize::new(down_tgs, 1, 1),
                MTLSize::new(down_threads_per_tg, 1, 1),
            );
        }
        enc.end_encoding();
        cmd.commit();
        crate::cb_status::wait_or_abort(
            cmd,
            "crates/larql-compute-metal/src/moe_dispatch/experts.rs:349",
        );
        let t_gpu = t_start.elapsed();

        // CPU weighted sum (no post-experts norm — client does that across shards).
        let all_expert_outputs = read_buffer_f32(&scratch.expert_outs, valid_count * hidden);
        let mut moe_out = vec![0.0f32; hidden];
        for e in 0..valid_count {
            let w = valid_weights[e];
            let out_slice = &all_expert_outputs[e * hidden..(e + 1) * hidden];
            for (acc, &v) in moe_out.iter_mut().zip(out_slice) {
                *acc += v * w;
            }
        }
        let t_total = t_start.elapsed();
        if timing_enabled {
            eprintln!(
                "[run_experts_metal] K={valid_count} stage={:.2}ms gpu={:.2}ms readback+sum={:.2}ms total={:.2}ms",
                t_stage.as_secs_f32() * 1000.0,
                (t_gpu - t_stage).as_secs_f32() * 1000.0,
                (t_total - t_gpu).as_secs_f32() * 1000.0,
                t_total.as_secs_f32() * 1000.0,
            );
        }
        moe_out
    }
}
