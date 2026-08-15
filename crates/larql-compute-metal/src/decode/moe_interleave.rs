//! MoE interleave tail for decode.
//!
//! Hybrid MoE layers need a command-buffer split: attention produces
//! `h_post_attn`, the expert path runs on CPU or remotely, and the dense FFN
//! may be encoded on a second GPU command buffer so it overlaps the remote
//! expert round trip. This module owns that tail so `decode/mod.rs` can keep
//! the per-layer happy path readable.

use metal::{Buffer, CommandBuffer, ComputeCommandEncoder};

use super::{diag, encode_ffn, encode_post_ffn, gpu_timing, moe_combine};
use crate::MetalBackend;
use larql_compute::FullPipelineLayer;

pub(super) struct MoeInterleaveCtx<'a> {
    pub layer_idx: usize,
    pub num_layers: usize,
    pub hidden: usize,
    pub inter: usize,
    pub inter_padded: usize,
    pub defer_ffn_for_split: bool,
    pub stage_timing_split: bool,
    pub layer_in_snapshot: Option<&'a [f32]>,
    pub dump_l0_dir: Option<&'a str>,
}

pub(super) struct MoeInterleaveBufs<'a> {
    pub gate_w: &'a Buffer,
    pub up_w: &'a Buffer,
    pub down_w: &'a Buffer,
    pub h_post_attn: &'a Buffer,
    pub ffn_norm_out: &'a Buffer,
    pub ffn_q8: &'a Buffer,
    pub ffn_q8s: &'a Buffer,
    pub gate_out_scratch: &'a Buffer,
    pub up_out: &'a Buffer,
    pub act_buf: &'a Buffer,
    pub down_out: &'a Buffer,
    pub normed_scratch: &'a Buffer,
    pub new_h: &'a Buffer,
}

pub(super) struct MoeCommandState<'a> {
    pub cmd: &'a mut CommandBuffer,
    pub enc: &'a mut ComputeCommandEncoder,
    pub encoder_ended: &'a mut bool,
    pub gpu_time: &'a mut gpu_timing::TokenGpuTime,
    pub residual_dump: &'a mut diag::ResidualDump,
}

/// Names the first unmet merged-CB precondition, per layer.
const ENV_MOE_INLINE_DIAG: &str = "LARQL_MOE_INLINE_DIAG";

/// R16 dose-response control: insert N extra empty commit+wait pairs per MoE
/// layer. Each adds one submit->start->complete round trip and NO GPU work,
/// so the slope of ms/token against N is the per-barrier cost measured in
/// situ. Calibrates the claim that the 5.1 ms residual is rendezvous latency
/// rather than something else that merely correlates with layer count.
const ENV_EXTRA_BARRIERS: &str = "LARQL_EXTRA_BARRIERS";

/// Spin on the command buffer's status instead of blocking in
/// `wait_until_completed`.
///
/// The bubble between command buffers is ~240 us that is neither host compute
/// (~49 us measured) nor generic submission (~15 us for an EMPTY command
/// buffer). An empty buffer is already complete when the host asks, so the
/// host never actually blocks — which makes descheduling/wake latency the
/// candidate that both observations fit. Spinning keeps the thread on-core;
/// if the bubble collapses, the cost was the sleep/wake round trip.
const ENV_SPIN_WAIT: &str = "LARQL_SPIN_WAIT";

/// Context for the merged-command-buffer MoE path: routing on CPU, expert
/// dispatches + weighted combine encoded into the SAME command buffer the
/// next layer's attention will ride — one wait per layer instead of two.
/// Built by `decode_token_q4k_moe` when the backend's expert scratch is
/// live; layers that miss the preconditions fall back to the callback arm
/// unchanged.
pub struct InlineMoeCtx<'a> {
    pub(crate) scratch: &'a crate::moe_dispatch::MoeScratch,
    pub(crate) eps: f32,
}

impl<'a> InlineMoeCtx<'a> {
    pub(crate) fn new(scratch: &'a crate::moe_dispatch::MoeScratch, eps: f32) -> Self {
        Self { scratch, eps }
    }
}

impl MetalBackend {
    /// Every merged-CB precondition, in one place: the S2 GPU-route arm
    /// decides whether to SKIP the attention wait with exactly the same
    /// checks the CPU fast path uses to decide whether to run — a drift
    /// between the two would skip a wait some fallback arm still needs.
    fn inline_moe_preconditions<'m>(
        layer: &'m FullPipelineLayer<'_>,
        ctx: &MoeInterleaveCtx<'_>,
        scratch: &crate::moe_dispatch::MoeScratch,
    ) -> Result<&'m larql_compute::MoeLayerWeights<'m>, String> {
        let Some(moe) = layer.moe.as_ref() else {
            return Err("layer has no MoE weights".into());
        };
        if layer.ffn_is_remote {
            return Err("ffn_is_remote".into());
        }
        if ctx.defer_ffn_for_split {
            return Err("defer_ffn_for_split".into());
        }
        if ctx.stage_timing_split {
            return Err("stage_timing_split (LARQL_PROFILE_SPLIT)".into());
        }
        if layer.has_dense_ffn() {
            return Err("layer has a dense FFN branch".into());
        }
        if !matches!(
            moe.routing_policy.post_expert_norm,
            larql_compute::MoePostExpertNormPolicy::None
        ) {
            return Err("post_expert_norm is not None (not the identity-combine class)".into());
        }
        if layer.moe_combined_output_norm {
            return Err("moe_combined_output_norm is set".into());
        }
        if !(layer.layer_scalar == 0.0 || layer.layer_scalar == 1.0) {
            return Err("layer_scalar is neither 0 nor 1".into());
        }
        if ctx.layer_in_snapshot.is_some() {
            return Err("layer_in_snapshot capture is active".into());
        }
        if ctx.dump_l0_dir.is_some() {
            return Err("dump_l0_dir capture is active".into());
        }
        let biased_gated_servable =
            matches!(moe.gate_rule, larql_compute::MoeGateRule::ClampedGlu { .. })
                || (moe.experts_gate_up_bias.is_empty() && moe.experts_down_bias.is_empty());
        if !biased_gated_servable {
            return Err("a Gated layer with expert biases has no kernel".into());
        }
        if moe.top_k != scratch.top_k {
            return Err(format!("top_k {} != scratch {}", moe.top_k, scratch.top_k));
        }
        if moe.intermediate_size != scratch.inter {
            return Err(format!(
                "intermediate_size {} != scratch {}",
                moe.intermediate_size, scratch.inter
            ));
        }
        if ctx.hidden != scratch.hidden {
            return Err(format!(
                "hidden {} != scratch {}",
                ctx.hidden, scratch.hidden
            ));
        }
        if moe.expert_data_format != scratch.format {
            return Err(format!(
                "expert_data_format {:?} != scratch format {:?}",
                moe.expert_data_format, scratch.format
            ));
        }
        if moe.gate_up_cols(ctx.hidden) != scratch.weight_cols {
            return Err(format!(
                "gate_up_cols {} != scratch weight_cols {}",
                moe.gate_up_cols(ctx.hidden),
                scratch.weight_cols
            ));
        }
        Ok(moe)
    }

    #[allow(clippy::too_many_arguments, clippy::type_complexity)]
    pub(super) fn handle_moe_interleave(
        &self,
        layer: &FullPipelineLayer<'_>,
        ctx: MoeInterleaveCtx<'_>,
        bufs: MoeInterleaveBufs<'_>,
        state: MoeCommandState<'_>,
        moe_fn: &mut Option<&mut dyn FnMut(usize, &[f32]) -> Vec<f32>>,
        moe_collect_fn: &mut Option<&mut dyn FnMut(usize) -> Vec<f32>>,
        inline_moe: Option<&InlineMoeCtx<'_>>,
    ) {
        // Proceed when this is a hybrid-MoE layer (layer.moe is Some) OR when
        // the entire FFN is remote (ffn_is_remote), which also routes through
        // the moe_fn callback path instead of running a local GPU FFN.
        if layer.moe.is_none() && !layer.ffn_is_remote {
            return;
        }
        // Borrow the MoE weights if present (used only in the local-expert
        // fallback branch — never reached when moe_fn is Some or ffn_is_remote).
        let moe_ref = layer.moe.as_ref();

        // ── S2 (GPU-dataflow scheduling): the commit+wait below existed so
        // the CPU could read h_post_attn and route. When this layer will be
        // GPU-routed, that completion has no consumer — keep the command
        // buffer OPEN and encode the MoE straight after attention. Every
        // precondition is decided from the SAME authority the CPU fast path
        // uses (inline_moe_preconditions), plus the GPU-route support checks;
        // any miss falls through to the legacy wait with nothing changed.
        // (No moe_fn/moe_collect_fn gate: the merged-CB fast path has
        // always outranked the callback arm when its preconditions hold —
        // try_inline runs first regardless. S2 keeps that precedence.)
        if crate::moe_gpu_route::gpu_route_enabled() {
            if let Some(ictx) = inline_moe {
                if let Ok(moe) = Self::inline_moe_preconditions(layer, &ctx, ictx.scratch) {
                    if self.gpu_route_supported(moe, ictx.scratch) {
                        if let Some(table) = self.descriptor_table_for_layer(
                            ctx.layer_idx,
                            moe,
                            ictx.scratch.inter,
                            ictx.scratch.hidden,
                        ) {
                            let _t_encode = std::time::Instant::now();
                            self.encode_moe_layer_gpu_route(
                                state.enc,
                                moe,
                                ictx.scratch,
                                &table,
                                bufs.h_post_attn,
                                bufs.new_h,
                                ictx.eps,
                                ctx.layer_idx,
                            );
                            crate::route_witness::bump(&crate::route_witness::GPU_ROUTE_LAYERS);
                            gpu_timing::add_host_segment(
                                |h| &mut h.encode_ms,
                                _t_encode.elapsed().as_secs_f64() * 1000.0,
                            );
                            return;
                        }
                    }
                }
            }
        }

        let _t_commit = std::time::Instant::now();
        state.enc.end_encoding();
        state.cmd.commit();
        gpu_timing::add_host_segment(
            |h| &mut h.commit_ms,
            _t_commit.elapsed().as_secs_f64() * 1000.0,
        );
        crate::route_witness::bump(&crate::route_witness::WAIT_MOE_ROUTE_LEGACY);
        let _t_wait = std::time::Instant::now();
        if larql_compute::options::env_opt_in(ENV_SPIN_WAIT) {
            while state.cmd.status() != metal::MTLCommandBufferStatus::Completed
                && state.cmd.status() != metal::MTLCommandBufferStatus::Error
            {
                std::hint::spin_loop();
            }
        } else {
            state.cmd.wait_until_completed();
        }
        gpu_timing::add_host_segment(|h| &mut h.wait_ms, _t_wait.elapsed().as_secs_f64() * 1000.0);
        // R16 control — see ENV_EXTRA_BARRIERS.
        if let Some(extra) = larql_compute::options::env_usize(ENV_EXTRA_BARRIERS) {
            for _ in 0..extra {
                let cb = self.queue.new_command_buffer();
                cb.commit();
                cb.wait_until_completed();
            }
        }
        // In split mode the cb we just waited contains ONLY attention
        // (steps 1-5). In non-split mode it normally contains attention +
        // dense FFN; but when stage_timing_split was active, attention was
        // already committed at its own boundary so this cb contains only FFN
        // + post-residual.
        let cb_stage = if ctx.defer_ffn_for_split {
            gpu_timing::DecodeStage::Attention
        } else if ctx.stage_timing_split {
            gpu_timing::DecodeStage::DenseFfn
        } else {
            gpu_timing::DecodeStage::Other
        };
        state.gpu_time.record_stage(state.cmd, cb_stage);
        *state.encoder_ended = true;

        // MoE and dense FFN run on the SAME input (`h_post_attn`, the
        // post-attention residual). Dense FFN output is already in `new_h`.
        let attn_ptr = bufs.h_post_attn.contents() as *const f32;
        let attn_slice = unsafe { std::slice::from_raw_parts(attn_ptr, ctx.hidden) };

        // ── Merged-CB fast path ──────────────────────────────────────────
        // Route on CPU, then encode this layer's experts AND the weighted
        // combine into a FRESH command buffer that stays OPEN — the next
        // layer's attention rides it, so the per-layer wait count halves.
        // Every precondition miss falls through to the callback arm below,
        // byte-for-byte unchanged.
        if let Some(ictx) = inline_moe {
            if self.try_inline_zero_copy_moe(
                layer,
                &ctx,
                &bufs,
                ictx,
                attn_slice,
                &mut *state.cmd,
                &mut *state.enc,
                &mut *state.encoder_ended,
            ) {
                return;
            }
        }
        let moe_out = if ctx.defer_ffn_for_split {
            // Split path: fire MoE NOW, then encode dense FFN + post-FFN
            // residual on a fresh cb so GPU runs while the remote trip is in
            // flight. Pure-MoE layers have no dense branch to overlap —
            // fire and collect with no GPU work in between.
            let fire = moe_fn.as_deref_mut().expect("split_mode implies moe_fn");
            fire(ctx.layer_idx, attn_slice);

            *state.cmd = self.queue.new_command_buffer().to_owned();
            if layer.has_dense_ffn() {
                let ffn_enc = state.cmd.new_compute_command_encoder();

                self.encode_ffn_step(
                    ffn_enc,
                    layer,
                    encode_ffn::FfnBufs {
                        gate_w: bufs.gate_w,
                        up_w: bufs.up_w,
                        down_w: bufs.down_w,
                        ffn_norm_out: bufs.ffn_norm_out,
                        ffn_q8: bufs.ffn_q8,
                        ffn_q8s: bufs.ffn_q8s,
                        gate_out_scratch: bufs.gate_out_scratch,
                        up_out: bufs.up_out,
                        act_buf: bufs.act_buf,
                        down_out: bufs.down_out,
                    },
                    encode_ffn::FfnDims {
                        hidden: ctx.hidden,
                        inter: ctx.inter,
                        inter_padded: ctx.inter_padded,
                    },
                );

                // Always unfused here: this preserves the previous split-MoE path.
                // D-RMS-FUSE Phase 1 not applied: split-MoE path commits per-layer
                // boundaries that don't match the cross-layer fusion pattern.
                self.encode_post_ffn_residual(
                    ffn_enc,
                    layer,
                    encode_post_ffn::PostFfnBufs {
                        down_out: bufs.down_out,
                        h_post_attn: bufs.h_post_attn,
                        new_h: bufs.new_h,
                        normed_scratch: bufs.normed_scratch,
                    },
                    ctx.hidden,
                    false,
                    None,
                );
                ffn_enc.end_encoding();
            }
            state.cmd.commit();

            let collect = moe_collect_fn
                .as_deref_mut()
                .expect("split_mode implies moe_collect_fn");
            let result = collect(ctx.layer_idx);
            state.cmd.wait_until_completed();
            state
                .gpu_time
                .record_stage(state.cmd, gpu_timing::DecodeStage::DenseFfn);
            result
        } else if let Some(ref mut f) = moe_fn {
            f(ctx.layer_idx, attn_slice)
        } else {
            // Local expert fallback — only reachable when moe_fn is None and
            // ffn_is_remote is false (otherwise we'd have taken a branch above).
            let moe = moe_ref.expect("cpu_moe_forward requires moe weights");
            larql_compute::cpu::ops::moe::cpu_moe_forward(
                attn_slice,
                moe,
                layer.norm_offset,
                layer.eps,
            )
        };

        // Accumulate the FFN contribution into the output buffer.
        //
        // Dense hybrid MoE path: new_h = (h_post_attn + dense_ffn) + moe_out.
        //   The GPU has already written `h_post_attn + dense_ffn` into new_h,
        //   so we add moe_out in-place.
        //
        // Remote-FFN path (ffn_is_remote) and pure-MoE layers (no dense
        // branch extracted): new_h = h_post_attn + moe_out. The GPU did
        // NOT run a local FFN, so new_h is uninitialised for this layer;
        // set it directly rather than accumulating into garbage.
        let h_ptr = bufs.new_h.contents() as *mut f32;
        if layer.ffn_is_remote || !layer.has_dense_ffn() {
            // attn_ptr was already computed above (h_post_attn contents).
            unsafe {
                for (i, v) in moe_out.iter().enumerate() {
                    *h_ptr.add(i) = *attn_ptr.add(i) + v;
                }
            }
        } else {
            // Hybrid MoE: new_h already holds (h_post_attn + dense_ffn),
            // add the expert contribution.
            unsafe {
                for (i, v) in moe_out.iter().enumerate() {
                    *h_ptr.add(i) += v;
                }
            }
        }

        if ctx.layer_idx == 0 {
            if let Some(dir) = ctx.dump_l0_dir {
                diag::dump_l0_moe_intermediates(
                    dir,
                    bufs.h_post_attn,
                    bufs.ffn_norm_out,
                    bufs.gate_out_scratch,
                    bufs.up_out,
                    bufs.act_buf,
                    bufs.down_out,
                    bufs.new_h,
                    &moe_out,
                    ctx.hidden,
                    ctx.inter,
                );
            }
        }

        moe_combine::apply_outer_combine(layer, bufs.new_h, bufs.h_post_attn, ctx.hidden);

        if let Some(layer_in) = ctx.layer_in_snapshot {
            let ha = super::super::buffers::read_buffer_f32(bufs.h_post_attn, ctx.hidden);
            let lo = super::super::buffers::read_buffer_f32(bufs.new_h, ctx.hidden);
            state
                .residual_dump
                .record_layer(ctx.layer_idx, layer_in, &ha, &lo);
        }

        if ctx.layer_idx + 1 < ctx.num_layers {
            *state.cmd = self.queue.new_command_buffer().to_owned();
            *state.enc = state.cmd.new_compute_command_encoder().to_owned();
            *state.encoder_ended = false;
        }
    }

    /// The merged-CB arm's gate + execution. Returns `true` when the
    /// layer's experts and combine were encoded into a fresh, still-open
    /// command buffer (assigned into `cmd`/`enc`); `false` leaves all
    /// state untouched for the callback arm.
    ///
    /// Preconditions (any miss → `false`):
    /// - pure-MoE layer (no dense branch — the combine kernel writes
    ///   `new_h = h_post_attn + Σ`, which would drop a dense contribution),
    /// - identity combine class: `MoePostExpertNormPolicy::None`, no
    ///   combined-output norm, layer_scalar ∈ {0, 1} (both no-ops in
    ///   `apply_outer_combine`, which this path skips),
    /// - no diagnostic captures for this layer (they read `new_h` on the
    ///   CPU before the deferred commit would produce it),
    /// - every selected expert's bytes resolve into ONE registered region
    ///   per projection with u32-expressible offsets.
    #[allow(clippy::too_many_arguments)]
    fn try_inline_zero_copy_moe(
        &self,
        layer: &FullPipelineLayer<'_>,
        ctx: &MoeInterleaveCtx<'_>,
        bufs: &MoeInterleaveBufs<'_>,
        ictx: &InlineMoeCtx<'_>,
        h_post_attn: &[f32],
        cmd: &mut CommandBuffer,
        enc: &mut ComputeCommandEncoder,
        encoder_ended: &mut bool,
    ) -> bool {
        use larql_compute::cpu::ops::moe::{
            moe_expert_input, moe_route_from_router_input, moe_router_input,
        };
        // `LARQL_MOE_INLINE_DIAG=1` names the first unmet precondition per
        // layer. Without it a miss is invisible: decode stays correct and
        // just runs the callback arm, which is the 22.7 ms/token rung
        // instead of 16.7 — a 1.36x regression that no test can see, because
        // both arms produce identical tokens. `docs/k3-funnel.md` §4.10
        // records the same instrument-shaped failure one level up.
        let diag = larql_compute::options::env_opt_in(ENV_MOE_INLINE_DIAG);
        macro_rules! refuse {
            ($why:expr) => {{
                if diag {
                    eprintln!("[moe-inline] layer {} refused: {}", ctx.layer_idx, $why);
                }
                return false;
            }};
        }

        let moe = match Self::inline_moe_preconditions(layer, ctx, ictx.scratch) {
            Ok(m) => m,
            Err(why) => refuse!(why),
        };
        let scratch = ictx.scratch;

        // ── GPU-dataflow route (serve-integration rung S1, gated by
        // LARQL_GPU_ROUTE=1): router → select → descriptor-driven experts
        // consume the GPU-RESIDENT h_post_attn buffer; the host slice's
        // routing role ends here. Deliberately the SAME command-buffer
        // lifecycle as the CPU arm below — S1 proves semantics in the
        // real decode; removing the per-layer waits is S2's change, so
        // a failure here can never be confused with a scheduling defect.
        // Every precondition is checked before CB state is touched: an
        // unsupported policy/format/bank falls through to the CPU arm
        // with nothing to roll back.
        if crate::moe_gpu_route::gpu_route_enabled()
            && !self.gpu_route_supported(moe, scratch)
            && larql_compute::options::env_opt_in(ENV_MOE_INLINE_DIAG)
        {
            eprintln!(
                "[gpu-route] layer {}: unsupported — format={:?} layout={:?} \
                 weight_cols={} hidden={} transform={:?} E={} k={}",
                ctx.layer_idx,
                scratch.format,
                moe.fused_row_layout,
                scratch.weight_cols,
                scratch.hidden,
                crate::moe_gpu_route::router_input_transform(moe),
                moe.num_experts,
                moe.top_k,
            );
        }
        if crate::moe_gpu_route::gpu_route_enabled() && self.gpu_route_supported(moe, scratch) {
            // Dims come from the MoE scratch, NOT MoeInterleaveCtx: on a
            // hybrid layer ctx.inter is the DENSE FFN width; the expert
            // bank's own intermediate lives on the scratch (cost one
            // refused table + this comment to learn).
            let table =
                self.descriptor_table_for_layer(ctx.layer_idx, moe, scratch.inter, scratch.hidden);
            if table.is_none() && larql_compute::options::env_opt_in(ENV_MOE_INLINE_DIAG) {
                eprintln!(
                    "[gpu-route] layer {}: descriptor table refused — bank \
                     unregistered/ragged/cross-buffer, or bias tables \
                     contradict dims (E={} inter={} hidden={} gu_bias_len={} \
                     dn_bias_len={} gu0_len={:?})",
                    ctx.layer_idx,
                    moe.num_experts,
                    ctx.inter,
                    ctx.hidden,
                    moe.experts_gate_up_bias.len(),
                    moe.experts_down_bias.len(),
                    moe.experts_gate_up.first().map(|s| s.len()),
                );
            }
            if let Some(table) = table {
                let _t_encode = std::time::Instant::now();
                *cmd = self.queue.new_command_buffer().to_owned();
                *enc = cmd.new_compute_command_encoder().to_owned();
                self.encode_moe_layer_gpu_route(
                    enc,
                    moe,
                    scratch,
                    &table,
                    bufs.h_post_attn,
                    bufs.new_h,
                    ictx.eps,
                    ctx.layer_idx,
                );
                *encoder_ended = false;
                crate::route_witness::bump(&crate::route_witness::GPU_ROUTE_LAYERS);
                gpu_timing::add_host_segment(
                    |h| &mut h.encode_ms,
                    _t_encode.elapsed().as_secs_f64() * 1000.0,
                );
                return true;
            }
        }

        // CPU routing — identical helpers to the staged dispatcher.
        let _t_route = std::time::Instant::now();
        let expert_input = moe_expert_input(h_post_attn, moe, 0.0, ictx.eps);
        let router_in = moe_router_input(h_post_attn, &expert_input, moe, 0.0, ictx.eps);
        let (expert_indices, expert_weights) = moe_route_from_router_input(&router_in, moe);
        gpu_timing::add_host_segment(
            |h| &mut h.route_ms,
            _t_route.elapsed().as_secs_f64() * 1000.0,
        );

        let _t_resolve = std::time::Instant::now();

        let Some(resolved) =
            self.resolve_selected_experts(scratch, moe, &expert_indices, &expert_weights, |ei| {
                Some((
                    moe.experts_gate_up.get(ei).copied()?,
                    moe.experts_down.get(ei).copied()?,
                ))
            })
        else {
            refuse!("selected experts did not resolve to registered zero-copy regions")
        };

        gpu_timing::add_host_segment(
            |h| &mut h.resolve_ms,
            _t_resolve.elapsed().as_secs_f64() * 1000.0,
        );

        let _t_encode = std::time::Instant::now();
        // Fresh command buffer; encode experts + combine; leave it OPEN —
        // the caller's loop encodes the next layer's attention into it and
        // the next interleave (or the token epilogue) commits it.
        *cmd = self.queue.new_command_buffer().to_owned();
        *enc = cmd.new_compute_command_encoder().to_owned();
        self.encode_experts_and_combine_zero_copy(
            enc,
            &expert_input,
            moe,
            scratch,
            &resolved,
            bufs.h_post_attn,
            bufs.new_h,
            ctx.layer_idx,
        );
        *encoder_ended = false;
        gpu_timing::add_host_segment(
            |h| &mut h.encode_ms,
            _t_encode.elapsed().as_secs_f64() * 1000.0,
        );
        true
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::moe_dispatch::MoeScratch;
    use crate::MetalBackend;
    use larql_compute::pipeline::FullPipelineLayer;
    use larql_compute::{
        Activation, MoeGateRule, MoeLayerWeights, MoeRoutingPolicy, MoeWeightLayout, QuantFormat,
    };

    fn backend() -> MetalBackend {
        MetalBackend::new().expect("Metal device available on test host")
    }

    fn synth(n: usize, seed: f32) -> Vec<f32> {
        (0..n)
            .map(|i| (seed + i as f32 * 0.013).sin() * 0.2)
            .collect()
    }

    fn pad_rows_to_256(data: &[f32], rows: usize, cols: usize) -> Vec<f32> {
        let padded_cols = cols.div_ceil(256) * 256;
        if padded_cols == cols {
            return data.to_vec();
        }
        let mut out = vec![0.0f32; rows * padded_cols];
        for r in 0..rows {
            out[r * padded_cols..r * padded_cols + cols]
                .copy_from_slice(&data[r * cols..(r + 1) * cols]);
        }
        out
    }

    /// Same layout `tests/test_kernel_moe_expert_dispatch.rs` uses for
    /// Q4_K experts: fused `[gate | up]` halves, block-padded down rows.
    fn make_q4k_experts(hidden: usize, inter: usize, n: usize) -> (Vec<Vec<u8>>, Vec<Vec<u8>>) {
        let mut gate_up = Vec::with_capacity(n);
        let mut down = Vec::with_capacity(n);
        for e in 0..n {
            let gate = synth(inter * hidden, 0.11 + e as f32 * 0.13);
            let up = synth(inter * hidden, 0.41 + e as f32 * 0.17);
            let mut gu = Vec::with_capacity(2 * inter * hidden);
            gu.extend_from_slice(&gate);
            gu.extend_from_slice(&up);
            gate_up.push(larql_compute::cpu::ops::q4_common::quantize_q4_k(&gu));

            let raw_down = synth(hidden * inter, 0.73 + e as f32 * 0.07);
            let down_padded = pad_rows_to_256(&raw_down, hidden, inter);
            down.push(larql_compute::cpu::ops::q4_common::quantize_q4_k(
                &down_padded,
            ));
        }
        (gate_up, down)
    }

    /// Every `try_inline_zero_copy_moe` precondition satisfied: pure-MoE
    /// layer (no dense FFN branch, via `FullPipelineLayer::default()`'s
    /// empty `up`/`down` weights), identity-combine routing policy
    /// (`top_k_softmax`'s `post_expert_norm: None`, `layer_scalar: 0.0`,
    /// no combined-output norm), no diagnostic captures, and every
    /// expert's bytes pre-registered as a zero-copy region. This is the
    /// merged-CB fast path `handle_moe_interleave` takes when the
    /// backend's expert scratch is live — never reached by the
    /// staged-path tests in `moe_dispatch.rs`/the integration suite,
    /// which all use the hybrid (dense+MoE) or default routing-policy
    /// shape instead.
    #[test]
    fn try_inline_zero_copy_moe_encodes_experts_and_combine_on_registered_region() {
        let m = backend();
        let hidden = 256usize;
        let inter = 256usize;
        let top_k = 2usize;
        let num_experts = 4usize;

        let (expert_gu, expert_down) = make_q4k_experts(hidden, inter, num_experts);

        // Lay every expert out contiguously in one page-aligned anonymous
        // mmap, exactly the production `register_weight_region` contract.
        let total: usize = expert_gu
            .iter()
            .zip(expert_down.iter())
            .map(|(g, d)| g.len() + d.len())
            .sum();
        let mut region = memmap2::MmapMut::map_anon(total).expect("anon mmap");
        let mut offsets = Vec::with_capacity(num_experts);
        let mut cursor = 0usize;
        for (g, d) in expert_gu.iter().zip(expert_down.iter()) {
            region[cursor..cursor + g.len()].copy_from_slice(g);
            let g_off = cursor;
            cursor += g.len();
            region[cursor..cursor + d.len()].copy_from_slice(d);
            offsets.push((g_off, g.len(), cursor, d.len()));
            cursor += d.len();
        }
        let region = region.make_read_only().expect("read-only mmap");
        assert!(
            m.bufs.register_region(&region[..]),
            "page-aligned anon mmap must register"
        );

        // `moe.experts_gate_up`/`experts_down` MUST be slices into the
        // registered `region`, not the original `expert_gu`/`expert_down`
        // vectors those bytes were copied from — those still live at a
        // different, unregistered address. Passing the originals here was
        // the actual bug this test spent several CI round-trips finding:
        // every precondition matched, but `resolve_selected_experts`
        // still failed because `moe`'s own byte slices didn't point into
        // the region `register_region` was called on, so `resolve_region`
        // correctly reported no match for either selected expert.
        let experts_gate_up: Vec<&[u8]> = offsets
            .iter()
            .map(|&(g_off, g_len, _, _)| &region[g_off..g_off + g_len])
            .collect();
        let experts_down: Vec<&[u8]> = offsets
            .iter()
            .map(|&(_, _, d_off, d_len)| &region[d_off..d_off + d_len])
            .collect();

        let router_w: Vec<f32> = (0..num_experts * hidden)
            .map(|i| (i as f32 * 0.0003).sin() * 0.05)
            .collect();
        let pre_norm_w: Vec<f32> = (0..hidden).map(|i| 1.0 + (i as f32 * 0.0005)).collect();
        let router_scale: Vec<f32> = vec![1.0f32; hidden];
        let router_per_expert_scale: Vec<f32> = vec![1.0f32; num_experts];
        let moe = MoeLayerWeights {
            expert_scales: larql_compute::MoeExpertScales::Inline,
            fused_row_layout: larql_compute::MoeFusedRowLayout::ContiguousHalves,
            experts_gate_up,
            experts_down,
            // `top_k_softmax`, NOT the crate default (`gemma4_hybrid`):
            // the default's `post_expert_norm: RmsNorm` fails this
            // function's identity-combine precondition outright.
            routing_policy: MoeRoutingPolicy::top_k_softmax(),
            weight_layout: MoeWeightLayout::default(),
            expert_data_format: QuantFormat::Q4_K,
            router_proj: &router_w,
            router_scale: &router_scale,
            router_per_expert_scale: &router_per_expert_scale,
            router_norm: &[],
            router_norm_parameter_free: true,
            router_input_scalar: 1.0,
            pre_experts_norm: &pre_norm_w,
            post_ffn1_norm: &pre_norm_w,
            post_experts_norm: &pre_norm_w,
            num_experts,
            top_k,
            intermediate_size: inter,
            router_bias: &[],
            experts_gate_up_bias: &[],
            experts_down_bias: &[],
            gate_rule: MoeGateRule::Gated(Activation::GeluTanh),
        };

        let scratch = MoeScratch::new_public(&m, top_k, hidden, inter);
        // `FullPipelineLayer::default()` has empty `up`/`down` weights
        // (`has_dense_ffn() == false`), `layer_scalar: 0.0`,
        // `moe_combined_output_norm: false`, `ffn_is_remote: false` —
        // every non-MoE precondition this function checks.
        let layer = FullPipelineLayer {
            moe: Some(moe),
            ..Default::default()
        };
        let ctx = MoeInterleaveCtx {
            layer_idx: 0,
            num_layers: 1,
            hidden,
            inter,
            inter_padded: inter,
            defer_ffn_for_split: false,
            stage_timing_split: false,
            layer_in_snapshot: None,
            dump_l0_dir: None,
        };
        let ictx = InlineMoeCtx::new(&scratch, 1e-6);

        let h_post_attn_data = synth(hidden, 0.9);
        let h_post_attn_buf = m.bufs.transient_from_f32(&h_post_attn_data);
        let new_h_buf = m.bufs.transient_from_f32(&vec![0.0f32; hidden]);
        // Unused by this precondition/path combination — one shared dummy
        // buffer is enough for every field `try_inline_zero_copy_moe`
        // never reads.
        let dummy = m.bufs.transient_from_f32(&[0.0f32; 4]);
        let bufs = MoeInterleaveBufs {
            gate_w: &dummy,
            up_w: &dummy,
            down_w: &dummy,
            h_post_attn: &h_post_attn_buf,
            ffn_norm_out: &dummy,
            ffn_q8: &dummy,
            ffn_q8s: &dummy,
            gate_out_scratch: &dummy,
            up_out: &dummy,
            act_buf: &dummy,
            down_out: &dummy,
            normed_scratch: &dummy,
            new_h: &new_h_buf,
        };

        let mut cmd = m.queue.new_command_buffer().to_owned();
        let mut enc = cmd.new_compute_command_encoder().to_owned();
        let mut encoder_ended = true;
        // `try_inline_zero_copy_moe` REPLACES `*enc`/`*cmd` in place on the
        // fast-path hit — it assumes the caller already ended/committed
        // the incoming encoder (exactly what `handle_moe_interleave` does
        // right before calling it). Skipping this crashes the whole test
        // binary: Metal fatally asserts on dropping a command encoder
        // that was never `end_encoding()`'d.
        enc.end_encoding();
        cmd.commit();
        cmd.wait_until_completed();
        let took_zero_copy_path = m.try_inline_zero_copy_moe(
            &layer,
            &ctx,
            &bufs,
            &ictx,
            &h_post_attn_data,
            &mut cmd,
            &mut enc,
            &mut encoder_ended,
        );
        assert!(
            took_zero_copy_path,
            "every precondition was satisfied; the merged-CB fast path must fire"
        );
        assert!(!encoder_ended);

        enc.end_encoding();
        cmd.commit();
        cmd.wait_until_completed();

        let out = unsafe { std::slice::from_raw_parts(new_h_buf.contents() as *const f32, hidden) };
        assert!(
            out.iter().all(|v| v.is_finite()),
            "non-finite combine output"
        );
        assert!(
            out.iter().any(|&v| v.abs() > 1e-6),
            "combine wrote an all-zero buffer — vacuous dispatch"
        );
    }

    /// The test above puts every expert in ONE registered region, so
    /// `encode_experts_zero_copy`'s `single_base` check is always true and
    /// only the grouped-kernel arms run. Registering each expert in its
    /// OWN region instead forces `single_base` to false for both the
    /// gate/up and down dispatches regardless of which two experts the
    /// router selects, driving the per-expert (non-grouped) fused Q4_K
    /// kernel and per-expert down-matvec fallback arms — the other half
    /// of that function's dispatch-shape branching. Also sets non-empty
    /// `experts_gate_up_bias`/`experts_down_bias` to drive the bias-staging
    /// block here and the `has_bias` combine arm in
    /// `encode_experts_and_combine_zero_copy`, neither of which the
    /// bias-free test above reaches. Non-empty biases force
    /// `gate_rule: ClampedGlu` too — `biased_gated_servable` requires
    /// either ClampedGlu or both bias arrays empty, since a `Gated`
    /// layer with expert biases has no kernel — which additionally
    /// covers the ClampedGlu activation arm the first test never takes.
    #[test]
    fn try_inline_zero_copy_moe_uses_non_grouped_dispatch_across_separate_regions() {
        let m = backend();
        let hidden = 256usize;
        let inter = 256usize;
        let top_k = 2usize;
        let num_experts = 4usize;

        let (expert_gu, expert_down) = make_q4k_experts(hidden, inter, num_experts);

        // One page-aligned anonymous mmap PER expert — `resolve_region`
        // returns the same Metal buffer for any two sub-slices of the same
        // registered region, so this is what actually forces
        // `single_base` to observe distinct base buffers.
        let mut regions = Vec::with_capacity(num_experts);
        for (g, d) in expert_gu.iter().zip(expert_down.iter()) {
            let mut region = memmap2::MmapMut::map_anon(g.len() + d.len()).expect("anon mmap");
            region[..g.len()].copy_from_slice(g);
            region[g.len()..].copy_from_slice(d);
            let region = region.make_read_only().expect("read-only mmap");
            assert!(
                m.bufs.register_region(&region[..]),
                "page-aligned anon mmap must register"
            );
            regions.push(region);
        }
        let experts_gate_up: Vec<&[u8]> = regions
            .iter()
            .zip(expert_gu.iter())
            .map(|(region, g)| &region[..g.len()])
            .collect();
        let experts_down: Vec<&[u8]> = regions
            .iter()
            .zip(expert_gu.iter())
            .map(|(region, g)| &region[g.len()..])
            .collect();

        let router_w: Vec<f32> = (0..num_experts * hidden)
            .map(|i| (i as f32 * 0.0003).sin() * 0.05)
            .collect();
        let pre_norm_w: Vec<f32> = (0..hidden).map(|i| 1.0 + (i as f32 * 0.0005)).collect();
        let router_scale: Vec<f32> = vec![1.0f32; hidden];
        let router_per_expert_scale: Vec<f32> = vec![1.0f32; num_experts];
        // Non-empty so `expert_mlp(..).gate_up_bias`/`down_bias` are
        // non-empty too — `ExpertMlp::expert_mlp` slices these per-expert
        // at strides `2 * inter` and `hidden` respectively.
        let experts_gate_up_bias = vec![0.1f32; num_experts * 2 * inter];
        let experts_down_bias = vec![0.05f32; num_experts * hidden];
        let moe = MoeLayerWeights {
            expert_scales: larql_compute::MoeExpertScales::Inline,
            fused_row_layout: larql_compute::MoeFusedRowLayout::ContiguousHalves,
            experts_gate_up,
            experts_down,
            routing_policy: MoeRoutingPolicy::top_k_softmax(),
            weight_layout: MoeWeightLayout::default(),
            expert_data_format: QuantFormat::Q4_K,
            router_proj: &router_w,
            router_scale: &router_scale,
            router_per_expert_scale: &router_per_expert_scale,
            router_norm: &[],
            router_norm_parameter_free: true,
            router_input_scalar: 1.0,
            pre_experts_norm: &pre_norm_w,
            post_ffn1_norm: &pre_norm_w,
            post_experts_norm: &pre_norm_w,
            num_experts,
            top_k,
            intermediate_size: inter,
            router_bias: &[],
            experts_gate_up_bias: &experts_gate_up_bias,
            experts_down_bias: &experts_down_bias,
            // `biased_gated_servable` requires EITHER ClampedGlu OR both
            // bias arrays empty — "a Gated layer with expert biases has
            // no kernel" (see try_inline_zero_copy_moe's own comment).
            // Non-empty biases with `Gated` here made the function bail
            // at that check on the first attempt; this is also the
            // combination that drives the ClampedGlu activation arm
            // (limit/alpha values match tests/test_moe_clamped_glu_q6k.rs).
            gate_rule: MoeGateRule::ClampedGlu {
                limit: 7.0,
                alpha: 1.702,
            },
        };

        let scratch = MoeScratch::new_public(&m, top_k, hidden, inter);
        let layer = FullPipelineLayer {
            moe: Some(moe),
            ..Default::default()
        };
        let ctx = MoeInterleaveCtx {
            layer_idx: 0,
            num_layers: 1,
            hidden,
            inter,
            inter_padded: inter,
            defer_ffn_for_split: false,
            stage_timing_split: false,
            layer_in_snapshot: None,
            dump_l0_dir: None,
        };
        let ictx = InlineMoeCtx::new(&scratch, 1e-6);

        let h_post_attn_data = synth(hidden, 0.4);
        let h_post_attn_buf = m.bufs.transient_from_f32(&h_post_attn_data);
        let new_h_buf = m.bufs.transient_from_f32(&vec![0.0f32; hidden]);
        let dummy = m.bufs.transient_from_f32(&[0.0f32; 4]);
        let bufs = MoeInterleaveBufs {
            gate_w: &dummy,
            up_w: &dummy,
            down_w: &dummy,
            h_post_attn: &h_post_attn_buf,
            ffn_norm_out: &dummy,
            ffn_q8: &dummy,
            ffn_q8s: &dummy,
            gate_out_scratch: &dummy,
            up_out: &dummy,
            act_buf: &dummy,
            down_out: &dummy,
            normed_scratch: &dummy,
            new_h: &new_h_buf,
        };

        let mut cmd = m.queue.new_command_buffer().to_owned();
        let mut enc = cmd.new_compute_command_encoder().to_owned();
        let mut encoder_ended = true;
        enc.end_encoding();
        cmd.commit();
        cmd.wait_until_completed();
        let took_zero_copy_path = m.try_inline_zero_copy_moe(
            &layer,
            &ctx,
            &bufs,
            &ictx,
            &h_post_attn_data,
            &mut cmd,
            &mut enc,
            &mut encoder_ended,
        );
        assert!(
            took_zero_copy_path,
            "every precondition was satisfied; the merged-CB fast path must fire"
        );
        assert!(!encoder_ended);

        enc.end_encoding();
        cmd.commit();
        cmd.wait_until_completed();

        let out = unsafe { std::slice::from_raw_parts(new_h_buf.contents() as *const f32, hidden) };
        assert!(
            out.iter().all(|v| v.is_finite()),
            "non-finite combine output"
        );
        assert!(
            out.iter().any(|&v| v.abs() > 1e-6),
            "combine wrote an all-zero buffer — vacuous dispatch"
        );
    }

    /// Both tests above use `expert_data_format: QuantFormat::Q4_K` — the
    /// two Q6_K arms in `encode_experts_zero_copy` (grouped and
    /// non-grouped matvec) are entirely untested by either. Single shared
    /// region (`single_base` true) drives the Q6_K grouped kernel arm,
    /// same shape `tests/test_kernel_moe_expert_dispatch.rs`'s
    /// `zero_copy_grouped_q6k_dispatch_matches_staged_path` already
    /// proves numerically — this test only needs the fast path to fire
    /// and produce a non-vacuous result, not bit-exact parity.
    #[test]
    fn try_inline_zero_copy_moe_uses_q6k_grouped_dispatch() {
        use larql_compute::cpu::ops::q4_common::quantize_q6_k;

        let m = backend();
        let hidden = 256usize;
        let inter = 256usize;
        let top_k = 2usize;
        let num_experts = 4usize;

        let mut expert_gu: Vec<Vec<u8>> = Vec::with_capacity(num_experts);
        let mut expert_down: Vec<Vec<u8>> = Vec::with_capacity(num_experts);
        for e in 0..num_experts {
            let gate = synth(inter * hidden, 0.21 + e as f32 * 0.13);
            let up = synth(inter * hidden, 0.51 + e as f32 * 0.17);
            let mut gu = Vec::with_capacity(2 * inter * hidden);
            gu.extend_from_slice(&gate);
            gu.extend_from_slice(&up);
            expert_gu.push(quantize_q6_k(&gu));
            let raw_down = synth(hidden * inter, 0.83 + e as f32 * 0.07);
            let down_padded = pad_rows_to_256(&raw_down, hidden, inter);
            expert_down.push(quantize_q6_k(&down_padded));
        }

        let total: usize = expert_gu
            .iter()
            .zip(expert_down.iter())
            .map(|(g, d)| g.len() + d.len())
            .sum();
        let mut region = memmap2::MmapMut::map_anon(total).expect("anon mmap");
        let mut offsets = Vec::with_capacity(num_experts);
        let mut cursor = 0usize;
        for (g, d) in expert_gu.iter().zip(expert_down.iter()) {
            region[cursor..cursor + g.len()].copy_from_slice(g);
            let g_off = cursor;
            cursor += g.len();
            region[cursor..cursor + d.len()].copy_from_slice(d);
            offsets.push((g_off, g.len(), cursor, d.len()));
            cursor += d.len();
        }
        let region = region.make_read_only().expect("read-only mmap");
        assert!(
            m.bufs.register_region(&region[..]),
            "page-aligned anon mmap must register"
        );
        let experts_gate_up: Vec<&[u8]> = offsets
            .iter()
            .map(|&(g_off, g_len, _, _)| &region[g_off..g_off + g_len])
            .collect();
        let experts_down: Vec<&[u8]> = offsets
            .iter()
            .map(|&(_, _, d_off, d_len)| &region[d_off..d_off + d_len])
            .collect();

        let router_w: Vec<f32> = (0..num_experts * hidden)
            .map(|i| (i as f32 * 0.0004).cos() * 0.05)
            .collect();
        let pre_norm_w: Vec<f32> = (0..hidden).map(|i| 1.0 + (i as f32 * 0.0005)).collect();
        let router_scale: Vec<f32> = vec![1.0f32; hidden];
        let router_per_expert_scale: Vec<f32> = vec![1.0f32; num_experts];
        let moe = MoeLayerWeights {
            expert_scales: larql_compute::MoeExpertScales::Inline,
            fused_row_layout: larql_compute::MoeFusedRowLayout::ContiguousHalves,
            experts_gate_up,
            experts_down,
            routing_policy: MoeRoutingPolicy::top_k_softmax(),
            weight_layout: MoeWeightLayout::default(),
            expert_data_format: QuantFormat::Q6_K,
            router_proj: &router_w,
            router_scale: &router_scale,
            router_per_expert_scale: &router_per_expert_scale,
            router_norm: &[],
            router_norm_parameter_free: true,
            router_input_scalar: 1.0,
            pre_experts_norm: &pre_norm_w,
            post_ffn1_norm: &pre_norm_w,
            post_experts_norm: &pre_norm_w,
            num_experts,
            top_k,
            intermediate_size: inter,
            router_bias: &[],
            experts_gate_up_bias: &[],
            experts_down_bias: &[],
            gate_rule: MoeGateRule::Gated(Activation::GeluTanh),
        };

        let scratch = MoeScratch::new_public_with_format(
            &m,
            top_k,
            hidden,
            inter,
            QuantFormat::Q6_K,
            moe.gate_up_cols(hidden),
        );
        let layer = FullPipelineLayer {
            moe: Some(moe),
            ..Default::default()
        };
        let ctx = MoeInterleaveCtx {
            layer_idx: 0,
            num_layers: 1,
            hidden,
            inter,
            inter_padded: inter,
            defer_ffn_for_split: false,
            stage_timing_split: false,
            layer_in_snapshot: None,
            dump_l0_dir: None,
        };
        let ictx = InlineMoeCtx::new(&scratch, 1e-6);

        let h_post_attn_data = synth(hidden, 0.6);
        let h_post_attn_buf = m.bufs.transient_from_f32(&h_post_attn_data);
        let new_h_buf = m.bufs.transient_from_f32(&vec![0.0f32; hidden]);
        let dummy = m.bufs.transient_from_f32(&[0.0f32; 4]);
        let bufs = MoeInterleaveBufs {
            gate_w: &dummy,
            up_w: &dummy,
            down_w: &dummy,
            h_post_attn: &h_post_attn_buf,
            ffn_norm_out: &dummy,
            ffn_q8: &dummy,
            ffn_q8s: &dummy,
            gate_out_scratch: &dummy,
            up_out: &dummy,
            act_buf: &dummy,
            down_out: &dummy,
            normed_scratch: &dummy,
            new_h: &new_h_buf,
        };

        let mut cmd = m.queue.new_command_buffer().to_owned();
        let mut enc = cmd.new_compute_command_encoder().to_owned();
        let mut encoder_ended = true;
        enc.end_encoding();
        cmd.commit();
        cmd.wait_until_completed();
        let took_zero_copy_path = m.try_inline_zero_copy_moe(
            &layer,
            &ctx,
            &bufs,
            &ictx,
            &h_post_attn_data,
            &mut cmd,
            &mut enc,
            &mut encoder_ended,
        );
        assert!(
            took_zero_copy_path,
            "every precondition was satisfied; the merged-CB fast path must fire"
        );
        assert!(!encoder_ended);

        enc.end_encoding();
        cmd.commit();
        cmd.wait_until_completed();

        let out = unsafe { std::slice::from_raw_parts(new_h_buf.contents() as *const f32, hidden) };
        assert!(
            out.iter().all(|v| v.is_finite()),
            "non-finite combine output"
        );
        assert!(
            out.iter().any(|&v| v.abs() > 1e-6),
            "combine wrote an all-zero buffer — vacuous dispatch"
        );
    }

    /// `layer.moe.is_none()` is the first precondition check — must
    /// bail out before touching the command buffer/encoder at all.
    #[test]
    fn try_inline_zero_copy_moe_returns_false_without_moe_layer() {
        let m = backend();
        // `MoeScratch::new` debug-asserts `weight_cols.is_multiple_of(block)`
        // (Q4_K block = 256 elements) unconditionally, before this test's
        // early-return path is ever reached — must be a block multiple even
        // though the actual dispatch never runs.
        let hidden = 256usize;
        let layer = FullPipelineLayer {
            moe: None,
            ..Default::default()
        };
        let ctx = MoeInterleaveCtx {
            layer_idx: 0,
            num_layers: 1,
            hidden,
            inter: hidden,
            inter_padded: hidden,
            defer_ffn_for_split: false,
            stage_timing_split: false,
            layer_in_snapshot: None,
            dump_l0_dir: None,
        };
        let scratch = MoeScratch::new_public(&m, 1, hidden, hidden);
        let ictx = InlineMoeCtx::new(&scratch, 1e-6);
        let h_post_attn_data = vec![0.0f32; hidden];
        let dummy = m.bufs.transient_from_f32(&[0.0f32; 4]);
        let bufs = MoeInterleaveBufs {
            gate_w: &dummy,
            up_w: &dummy,
            down_w: &dummy,
            h_post_attn: &dummy,
            ffn_norm_out: &dummy,
            ffn_q8: &dummy,
            ffn_q8s: &dummy,
            gate_out_scratch: &dummy,
            up_out: &dummy,
            act_buf: &dummy,
            down_out: &dummy,
            normed_scratch: &dummy,
            new_h: &dummy,
        };
        let mut cmd = m.queue.new_command_buffer().to_owned();
        let mut enc = cmd.new_compute_command_encoder().to_owned();
        let mut encoder_ended = true;
        // `try_inline_zero_copy_moe` REPLACES `*enc`/`*cmd` in place on the
        // fast-path hit — it assumes the caller already ended/committed
        // the incoming encoder (exactly what `handle_moe_interleave` does
        // right before calling it). Skipping this crashes the whole test
        // binary: Metal fatally asserts on dropping a command encoder
        // that was never `end_encoding()`'d.
        enc.end_encoding();
        cmd.commit();
        cmd.wait_until_completed();
        let took_zero_copy_path = m.try_inline_zero_copy_moe(
            &layer,
            &ctx,
            &bufs,
            &ictx,
            &h_post_attn_data,
            &mut cmd,
            &mut enc,
            &mut encoder_ended,
        );
        assert!(!took_zero_copy_path);
        // Early-return arms never touch `*encoder_ended` — it must come
        // back exactly as the caller left it (already ended, per the
        // real `handle_moe_interleave` calling convention above), not
        // flipped to `false` as the success path would.
        assert!(
            encoder_ended,
            "must leave caller state untouched on bail-out"
        );
    }
}
