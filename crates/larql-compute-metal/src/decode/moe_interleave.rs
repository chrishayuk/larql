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
    pub ffn_uses_kquant: bool,
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

impl MetalBackend {
    #[allow(clippy::too_many_arguments, clippy::type_complexity)]
    pub(super) fn handle_moe_interleave(
        &self,
        layer: &FullPipelineLayer<'_>,
        ctx: MoeInterleaveCtx<'_>,
        bufs: MoeInterleaveBufs<'_>,
        state: MoeCommandState<'_>,
        moe_fn: &mut Option<&mut dyn FnMut(usize, &[f32]) -> Vec<f32>>,
        moe_collect_fn: &mut Option<&mut dyn FnMut(usize) -> Vec<f32>>,
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

        state.enc.end_encoding();
        state.cmd.commit();
        state.cmd.wait_until_completed();
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
        let moe_out = if ctx.defer_ffn_for_split {
            // Split path: fire MoE NOW, then encode dense FFN + post-FFN
            // residual on a fresh cb so GPU runs while the remote trip is in
            // flight.
            let fire = moe_fn.as_deref_mut().expect("split_mode implies moe_fn");
            fire(ctx.layer_idx, attn_slice);

            *state.cmd = self.queue.new_command_buffer().to_owned();
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
                ctx.ffn_uses_kquant,
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
        // Remote-FFN path (ffn_is_remote): new_h = h_post_attn + remote_ffn_out.
        //   The GPU did NOT run the local FFN, so new_h is uninitialised for
        //   this layer. We set new_h[i] = h_post_attn[i] + moe_out[i] directly.
        let h_ptr = bufs.new_h.contents() as *mut f32;
        if layer.ffn_is_remote {
            // Remote-FFN: new_h = h_post_attn + remote_ffn_out.
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
}
