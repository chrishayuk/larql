//! The dense (non-expert) Q4_K FFN dispatch.
//!
//! The GPU expert dispatch: route, stage, run, combine, against
//! pre-allocated scratch.
//!
//! Split out of `moe_dispatch.rs`; [`super`] composes the pieces.

#[allow(unused_imports)]
use super::*;
use crate::buffers::read_buffer_f32;
use crate::MetalBackend;
use larql_compute::cpu::ops::moe::{
    moe_expert_input, moe_post_expert_output, moe_route_from_router_input, moe_router_input,
};
use larql_compute::MoeLayerWeights;
use metal::*;
use std::ffi::c_void;

impl MetalBackend {
    /// The supplier is a trait object rather than a generic parameter on
    /// purpose. It is invoked `top_k` times per MoE layer — ~96 indirect
    /// calls per token against milliseconds of GPU work — while a generic
    /// monomorphises this whole ~330-line body once per distinct closure
    /// type at every call site, which bloats the binary and makes the
    /// file's measured coverage a function of how many closure types
    /// happen to exist rather than of what is tested.
    pub(crate) fn gpu_moe_dispatch_with_scratch<'w>(
        &self,
        h_post_attn: &[f32],
        moe: &MoeLayerWeights<'_>,
        eps: f32,
        scratch: &MoeScratch,
        get_expert_bytes: &dyn Fn(usize) -> Option<(&'w [u8], &'w [u8])>,
    ) -> Vec<f32> {
        let hidden = h_post_attn.len();
        let inter = moe.intermediate_size;
        let inter_padded = scratch.inter_padded;
        let top_k = moe.top_k;
        // ClampedGlu (with its biases) has a dedicated activation kernel
        // below. What has NO kernel is a Gated layer with expert biases —
        // no current model ships that combination, and running it bias-less
        // would be a plausible-looking wrong forward. Refuse it loudly.
        assert!(
            matches!(moe.gate_rule, larql_compute::MoeGateRule::ClampedGlu { .. })
                || (moe.experts_gate_up_bias.is_empty() && moe.experts_down_bias.is_empty()),
            "Metal MoE dispatch has no biased-Gated expert kernel; this \
             layer's biased experts must run on the CPU path"
        );
        // Checked HERE, at the top of the body, because everything below
        // it — this dispatch and the zero-copy fast path it delegates to
        // — encodes. Both carry a backstop arm on the combine match; a
        // refusal that reached one of those would abort the process
        // rather than report itself.
        assert!(
            crate::kernels::ffn::expert_activation_supported(&moe.gate_rule),
            "{}",
            crate::kernels::ffn::CLAMPED_GATED_REFUSAL
        );
        debug_assert_eq!(
            moe.gate_up_cols(hidden),
            scratch.weight_cols,
            "MoE scratch was sized for a different gate/up row width"
        );
        debug_assert_eq!(moe.expert_data_format, scratch.format);
        debug_assert_eq!(top_k, scratch.top_k, "MoE top_k drift across layers");
        debug_assert_eq!(
            inter, scratch.inter,
            "MoE intermediate_size drift across layers"
        );
        debug_assert_eq!(
            hidden, scratch.hidden,
            "MoE hidden_size drift across layers"
        );

        // ── 1. CPU expert input + router ─────────────────────────────────
        // The policy comes from `MoeLayerWeights`, so Gemma's
        // pre_experts_norm routing convention is no longer implicit here.
        let expert_input = moe_expert_input(h_post_attn, moe, 0.0, eps);
        let router_in = moe_router_input(h_post_attn, &expert_input, moe, 0.0, eps);
        let (expert_indices, expert_weights) = moe_route_from_router_input(&router_in, moe);

        // ── 2. Stage expert weight bytes into pre-allocated Metal buffers ─
        let row_bytes = scratch.row_bytes;
        let gate_half_bytes = inter * row_bytes;
        let up_half_bytes = inter * row_bytes;
        let down_expert_bytes = hidden * scratch.down_row_bytes;

        // ── 1.5 Zero-copy fast path ──────────────────────────────────────
        // When every selected expert's byte slices resolve inside a
        // registered mmap region (`register_weight_region` at engine
        // setup), bind them as region offsets and skip the staging
        // memcpys entirely — they are ~2 GB/token of CPU bandwidth on
        // GPT-OSS-class models. Any miss (unregistered region, short
        // slice) falls through to the staged path below unchanged.
        if let Some(resolved) =
            self.resolve_selected_experts(scratch, moe, &expert_indices, &expert_weights, |ei| {
                get_expert_bytes(ei)
            })
        {
            return self.dispatch_experts_zero_copy(&expert_input, moe, eps, scratch, &resolved);
        }

        let gate_ptr = scratch.gate_buf.contents() as *mut u8;
        let up_ptr = scratch.up_buf.contents() as *mut u8;

        let mut valid_weights: Vec<f32> = Vec::with_capacity(top_k);
        let mut valid_ids: Vec<usize> = Vec::with_capacity(top_k);
        let mut valid_count = 0usize;
        let stage_biases = !moe.experts_gate_up_bias.is_empty();

        for (k, &ei) in expert_indices.iter().enumerate() {
            // Scratch is sized for `scratch.top_k` experts; without this
            // guard a caller passing more indices than the scratch was
            // built for writes past `gate_buf`/`up_buf` (and would only
            // panic later on the `down_bufs` index). Bound against the
            // ALLOCATION, not `moe.top_k` — the two are only
            // debug_assert-ed equal, so in release they can drift. The
            // sibling staging loop in `run_experts_preselected_metal`
            // has carried this bound all along. Capability audit F10.
            if valid_count >= scratch.top_k {
                break;
            }
            let Some((gu_bytes, dn_bytes)) = get_expert_bytes(ei) else {
                continue;
            };
            if gu_bytes.len() < 2 * gate_half_bytes {
                continue;
            }

            // Q4_K layout: gate || up, each `inter * row_bytes` bytes.
            // SAFETY: gate_ptr / up_ptr are StorageModeShared Metal buffer
            // contents; offsets are bounded by `top_k * gate_half_bytes`
            // allocated up front (see `MoeScratch::new`), and the
            // `valid_count >= top_k` guard above enforces that bound.
            // Writes complete before the encoder dispatches the matvec
            // that reads them.
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

            // De-interleave this expert's fused gate/up bias row into the
            // staging buffers at the same slot the weights landed in, so
            // the activation kernel reads slot-aligned bias vectors. The
            // interleave convention is owned by `ExpertMlp::{gate,up}_bias`
            // (FusedHalf), not re-derived here.
            if stage_biases {
                let mlp = moe.expert_mlp(ei);
                unsafe {
                    let gb =
                        (scratch.gate_bias_buf.contents() as *mut f32).add(valid_count * inter);
                    let ub = (scratch.up_bias_buf.contents() as *mut f32).add(valid_count * inter);
                    for j in 0..inter {
                        *gb.add(j) = mlp.gate_bias(j);
                        *ub.add(j) = mlp.up_bias(j);
                    }
                }
            }

            valid_weights.push(expert_weights[k]);
            valid_ids.push(ei);
            valid_count += 1;
        }

        if valid_count == 0 {
            return vec![0.0f32; hidden];
        }

        // ── 3. Stage router-normed input into pre-allocated x_buf ─────────
        // The buffer is `weight_cols` wide with a permanently-zero tail, so
        // the writer's row padding contributes nothing to any dot product.
        unsafe {
            let x_ptr = scratch.x_buf.contents() as *mut f32;
            std::ptr::copy_nonoverlapping(expert_input.as_ptr(), x_ptr, hidden);
        }

        let cmd = self.queue.new_command_buffer();
        let enc = cmd.new_compute_command_encoder();

        // ── 4. Gate + up matvecs over all valid_count experts at once ────
        // Both staging buffers are one uniform-stride matrix of
        // `valid_count × inter` rows at `weight_cols` columns, so a single
        // dispatch (per projection) covers every selected expert.
        let n_rows = (valid_count * inter) as u32;
        let k_cols = scratch.weight_cols as u32;
        match scratch.format {
            larql_compute::QuantFormat::Q6_K => {
                // Two plain q6k_matvec dispatches. A fused Q6_K gate+up twin
                // of `q4k_ffn_gate_up` does not exist yet; at these shapes
                // the second dispatch costs ~µs against ~44 MB of weight
                // reads, so fusion is a tuning item, not a correctness one.
                let kh = &self.quant.q6k_matvec_pipeline;
                let tgs = (valid_count as u64 * inter as u64).div_ceil(kh.rows_per_tg);
                for (w_buf, out_buf) in [
                    (&scratch.gate_buf, &scratch.g_out),
                    (&scratch.up_buf, &scratch.u_out),
                ] {
                    enc.set_compute_pipeline_state(&kh.state);
                    enc.set_buffer(0, Some(w_buf), 0);
                    enc.set_buffer(1, Some(&scratch.x_buf), 0);
                    enc.set_buffer(2, Some(out_buf), 0);
                    enc.set_bytes(3, 4, &n_rows as *const u32 as *const c_void);
                    enc.set_bytes(4, 4, &k_cols as *const u32 as *const c_void);
                    enc.dispatch_thread_groups(
                        MTLSize::new(tgs, 1, 1),
                        MTLSize::new(kh.threads_per_tg, 1, 1),
                    );
                }
            }
            _ => {
                // Q4_K: the fused gate+up kernel. Geometry travels with the
                // `KernelHandle`; no shader-module ROWS_PER_TG re-imports.
                let gate_up_kh = &self.ffn.q4k_ffn_gate_up_pipeline;
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
            }
        }

        // ── 5. Gate-rule activation per expert (strided to inter_padded) ──
        // Gate/up output is packed at stride `inter`; activation must land at
        // stride `inter_padded` because down reads `K = inter_padded`. One
        // small dispatch per expert with the right offsets gets us strided
        // output without a new shader. valid_count × ~5µs ≪ allocation cost.
        //
        // The kernel comes from the layer's typed `MoeGateRule` — the same
        // authority the CPU paths combine under — so a Silu model can no
        // longer silently run the GELU kernel, and ClampedGlu (GPT-OSS)
        // gets its own kernel with the bias slots bound.
        let inter_u32 = inter as u32;
        for e in 0..valid_count {
            let g_offset = (e * inter * 4) as u64;
            let u_offset = (e * inter * 4) as u64;
            let a_offset = (e * inter_padded * 4) as u64;
            match moe.gate_rule {
                // Refused at admission by
                // `kernels::ffn::expert_activation_supported`, which
                // owns the reason. The backstop stays for any path that
                // does not come through that gate — and must never be
                // the thing that reports it, because a panic here
                // leaves the encoder unended.
                larql_compute::MoeGateRule::ClampedGated { .. } => {
                    unreachable!("{}", crate::kernels::ffn::CLAMPED_GATED_REFUSAL)
                }
                larql_compute::MoeGateRule::ClampedGlu { limit, alpha } => {
                    let has_bias: u32 = u32::from(stage_biases);
                    let b_offset = (e * inter * 4) as u64;
                    enc.set_compute_pipeline_state(&self.ffn.clamped_glu_bias_pipeline);
                    enc.set_buffer(0, Some(&scratch.g_out), g_offset);
                    enc.set_buffer(1, Some(&scratch.u_out), u_offset);
                    enc.set_buffer(2, Some(&scratch.act_buf), a_offset);
                    enc.set_bytes(3, 4, &inter_u32 as *const u32 as *const c_void);
                    enc.set_buffer(4, Some(&scratch.gate_bias_buf), b_offset);
                    enc.set_buffer(5, Some(&scratch.up_bias_buf), b_offset);
                    enc.set_bytes(6, 4, &has_bias as *const u32 as *const c_void);
                    enc.set_bytes(7, 4, &limit as *const f32 as *const c_void);
                    enc.set_bytes(8, 4, &alpha as *const f32 as *const c_void);
                }
                larql_compute::MoeGateRule::SituGlu { beta, linear_beta } => {
                    crate::kernels::ffn::bind_situ_glu(
                        enc,
                        &self.ffn.situ_glu_pipeline,
                        (&scratch.g_out, g_offset),
                        (&scratch.u_out, u_offset),
                        (&scratch.act_buf, a_offset),
                        inter_u32,
                        beta,
                        linear_beta,
                        stage_biases,
                    );
                }
                larql_compute::MoeGateRule::Gated(activation) => {
                    let pipeline = if activation.gate_up_is_gelu_tanh() {
                        &self.ffn.geglu_gelu_tanh_pipeline
                    } else {
                        &self.ffn.geglu_pipeline
                    };
                    enc.set_compute_pipeline_state(pipeline);
                    enc.set_buffer(0, Some(&scratch.g_out), g_offset);
                    enc.set_buffer(1, Some(&scratch.u_out), u_offset);
                    enc.set_buffer(2, Some(&scratch.act_buf), a_offset);
                    enc.set_bytes(3, 4, &inter_u32 as *const u32 as *const c_void);
                }
            }
            enc.dispatch_threads(
                MTLSize::new(inter as u64, 1, 1),
                MTLSize::new(
                    crate::kernels::DISPATCH_TG_MAX_THREADS.min(inter as u64),
                    1,
                    1,
                ),
            );
        }

        // ── 6. Down projection per expert ────────────────────────────────
        let n_out = hidden as u32;
        let k_in = inter_padded as u32;
        // Pull dispatch geometry from the bound pipeline so this works for
        // both the 4sg and 8sg variants of the matvec — hardcoding the
        // 4sg constants while dispatching the 8sg pipeline (the production
        // default since 2026-04-28) leaves simdgroups 4..7 unscheduled and
        // only writes rows 0..3 of each TG's 8-row range. See the matching
        // fix in `trait_impl/quant_matvec.rs::q4k_matvec`.
        let (down_kh, down_binding) = self.quant.expert_matvec_for(scratch.format);
        // The staged path memcpys expert payloads into scratch buffers; it has
        // no scratch for a second stream and no binding to reach one. A
        // split-scale bank therefore serves through the zero-copy route only,
        // which refuses at its own resolution step rather than arriving here.
        assert_eq!(
            down_binding,
            crate::kernels::quant::ExpertScaleBinding::InlineScales,
            "moe_dispatch: {:?} is a split-scale format and cannot be staged; \
             it serves through the zero-copy route, which requires the layer's \
             expert regions to be registered",
            scratch.format,
        );
        let down_tgs = (hidden as u64).div_ceil(down_kh.rows_per_tg);

        for e in 0..valid_count {
            let act_offset = (e * inter_padded * 4) as u64;
            let out_offset = (e * hidden * 4) as u64;
            enc.set_compute_pipeline_state(&down_kh.state);
            enc.set_buffer(0, Some(&scratch.down_bufs[e]), 0);
            enc.set_buffer(1, Some(&scratch.act_buf), act_offset);
            enc.set_buffer(2, Some(&scratch.expert_outs), out_offset);
            enc.set_bytes(3, 4, &n_out as *const u32 as *const c_void);
            enc.set_bytes(4, 4, &k_in as *const u32 as *const c_void);
            enc.dispatch_thread_groups(
                MTLSize::new(down_tgs, 1, 1),
                MTLSize::new(down_kh.threads_per_tg, 1, 1),
            );
        }
        enc.end_encoding();
        cmd.commit();
        crate::cb_status::wait_or_abort(
            cmd,
            "crates/larql-compute-metal/src/moe_dispatch/dispatch.rs:330",
        );

        // ── 7. CPU weighted sum + post-experts norm ──────────────────────
        // The down bias joins each expert's output BEFORE the routing
        // weight, matching the reference (`ExpertWeightFfn::run_expert`
        // adds it inside the expert, then the caller weights). Applied here
        // on readback rather than in a shader: it is `hidden` adds per
        // expert against ~22 MB of weight reads.
        let all_expert_outputs = read_buffer_f32(&scratch.expert_outs, valid_count * hidden);
        let mut moe_out = vec![0.0f32; hidden];
        for e in 0..valid_count {
            let w = valid_weights[e];
            let mlp = moe.expert_mlp(valid_ids[e]);
            let out_slice = &all_expert_outputs[e * hidden..(e + 1) * hidden];
            if mlp.down_bias.is_empty() {
                for (acc, &v) in moe_out.iter_mut().zip(out_slice) {
                    *acc += v * w;
                }
            } else {
                for ((acc, &v), &b) in moe_out.iter_mut().zip(out_slice).zip(mlp.down_bias) {
                    *acc += (v + b) * w;
                }
            }
        }

        moe_post_expert_output(&moe_out, moe, 0.0, eps)
    }
}

#[cfg(test)]
mod tests;
