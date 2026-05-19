//! Step 7: post-FFN residual + optional post-FFN norm.
//!
//! Three shapes covered, all behaviourally identical to the previously-inlined
//! versions (one in the dense branch, one inside the MoE-deferred FFN path):
//!
//! 1. `has_post_norms == false` — straight residual add `h_post_attn + down_out → new_h`.
//! 2. `has_post_norms && layer.post_ffn_norm.is_none()` — same straight residual
//!    add (post_ffn norm slot wasn't populated for this layer).
//! 3. `has_post_norms && layer.post_ffn_norm.is_some()` — RMS-norm `down_out` against
//!    `post_ffn_norm`, then residual-add against `h_post_attn` into `new_h`.
//!    When `use_fused == true`, dispatches the single fused
//!    `post_ffn_norm_residual_add` kernel (default-on for the dense path); when
//!    `use_fused == false`, falls back to the unfused `rms_norm` +
//!    `residual_add` two-dispatch chain (used by the MoE-deferred FFN path,
//!    matching prior behaviour exactly).
//!
//! `LARQL_FUSED_POST_FFN_NORM=0` is honoured only via the `use_fused` arg the
//! caller passes — the env-var resolution stays in the decode loop so this
//! helper has zero env-var I/O on the hot path.

use crate::ops::full_pipeline::{encode_residual_add, encode_rms_norm};
use crate::MetalBackend;
use larql_compute::FullPipelineLayer;
use metal::{Buffer, ComputeCommandEncoderRef, MTLSize};

pub(super) struct PostFfnBufs<'a> {
    pub down_out: &'a Buffer,
    pub h_post_attn: &'a Buffer,
    pub new_h: &'a Buffer,
    /// Scratch for the unfused chain. Unused when `use_fused == true`.
    pub normed_scratch: &'a Buffer,
}

/// D-RMS-FUSE Phase 1 hint: when present + `LARQL_FUSED_PRELAYER_NORM=1`,
/// the non-post-norms branch dispatches `residual_norm_store` instead of
/// plain `residual_add`, fusing the next layer's input rms_norm into the
/// same kernel call. The next layer's `encode_q4k_input_norm` then skips
/// its own dispatch (the data is already in the shared `norm_f32_buf`).
pub(super) struct PreLayerNormFusion<'a> {
    /// Next layer's `input_norm` weight slice.
    pub next_input_norm: &'a [f32],
    /// Shared `norm_f32_buf` (= next layer's `bufs.norm_out`) — written by
    /// the fused `residual_norm_store` dispatch.
    pub next_norm_out: &'a Buffer,
}

impl MetalBackend {
    pub(super) fn encode_post_ffn_residual(
        &self,
        enc: &ComputeCommandEncoderRef,
        layer: &FullPipelineLayer,
        bufs: PostFfnBufs<'_>,
        hidden: usize,
        use_fused: bool,
        prelayer_fusion: Option<&PreLayerNormFusion<'_>>,
    ) {
        // M2: read norm-related layer fields through the structured view.
        // `post_ffn_norm` is the only weight slice that doesn't have a
        // pre-extracted buffer in `bufs` — keep that as a direct field
        // access on the layer.
        let norms_view = layer.norms();

        // D-RMS-FUSE Phase 1: on the non-post-norms path (Llama / Mistral /
        // Qwen / etc.), if the caller passed in next-layer info AND the
        // env var is on, dispatch `residual_norm_store` to fuse the
        // residual-add with the next layer's input rms_norm in one kernel.
        // Saves 1 dispatch per layer × num_layers (~7 µs each).
        //
        // Skip the fusion when Granite's `residual_multiplier != 1.0`:
        // the `residual_norm_store` kernel hard-codes `out = a + b` and
        // has no `b_scale` binding, so firing it would silently drop
        // the residual-stream scaling. The unfused chain below uses
        // `encode_residual_add(..., layer.residual_multiplier)` which
        // does apply the scale.
        if let Some(fusion) = prelayer_fusion.filter(|_| {
            !norms_view.has_post_norms
                && self.decode_flags.fused_prelayer_norm
                && (layer.residual_multiplier - 1.0).abs() < f32::EPSILON
        }) {
            let next_input_norm_buf = self.bufs.get_f32(fusion.next_input_norm);
            let hidden_val = hidden as u32;
            let eps = norms_view.eps;
            let norm_offset = norms_view.norm_offset;
            // Granite multiplier is guarded above (this branch only fires
            // when residual_multiplier == 1.0). Still bind buffer(8) with
            // 1.0 since the shader now requires the slot.
            let b_scale: f32 = layer.residual_multiplier;
            enc.set_compute_pipeline_state(&self.norms.residual_norm_store_pipeline);
            enc.set_buffer(0, Some(bufs.h_post_attn), 0); // a (residual base)
            enc.set_buffer(1, Some(bufs.down_out), 0); // b (FFN output)
            enc.set_buffer(2, Some(&next_input_norm_buf), 0); // weight = next layer's input_norm
            enc.set_buffer(3, Some(fusion.next_norm_out), 0); // norm_out (next layer's normed input)
            enc.set_buffer(4, Some(bufs.new_h), 0); // sum_out (raw new_h for residual)
            enc.set_bytes(5, 4, &hidden_val as *const u32 as *const std::ffi::c_void);
            enc.set_bytes(6, 4, &eps as *const f32 as *const std::ffi::c_void);
            enc.set_bytes(7, 4, &norm_offset as *const f32 as *const std::ffi::c_void);
            enc.set_bytes(8, 4, &b_scale as *const f32 as *const std::ffi::c_void);
            enc.dispatch_thread_groups(
                MTLSize::new(1, 1, 1),
                MTLSize::new(
                    crate::kernels::DISPATCH_TG_MAX_THREADS.min(hidden as u64),
                    1,
                    1,
                ),
            );
            return;
        }

        // Granite never has post_norms, so the post_ffn_norm branch
        // below is dead for it — but guard the fused `post_ffn_norm_residual_add`
        // kernel anyway so a future hybrid (post-norm + residual_multiplier)
        // model doesn't silently regress. The kernel hard-codes
        // `out = down + h_post_attn` with no `b_scale` binding.
        let allow_fused_post_norm =
            use_fused && (layer.residual_multiplier - 1.0).abs() < f32::EPSILON;
        if norms_view.has_post_norms {
            if let Some(post_ffn) = layer.post_ffn_norm {
                let post_ffn_buf = self.bufs.get_f32(post_ffn);
                if allow_fused_post_norm {
                    let hidden_val = hidden as u32;
                    let eps = norms_view.eps;
                    let norm_offset = norms_view.norm_offset;
                    enc.set_compute_pipeline_state(&self.norms.post_ffn_norm_residual_add_pipeline);
                    enc.set_buffer(0, Some(bufs.down_out), 0);
                    enc.set_buffer(1, Some(bufs.h_post_attn), 0);
                    enc.set_buffer(2, Some(&post_ffn_buf), 0);
                    enc.set_buffer(3, Some(bufs.new_h), 0);
                    enc.set_bytes(4, 4, &hidden_val as *const u32 as *const std::ffi::c_void);
                    enc.set_bytes(5, 4, &eps as *const f32 as *const std::ffi::c_void);
                    enc.set_bytes(6, 4, &norm_offset as *const f32 as *const std::ffi::c_void);
                    enc.dispatch_thread_groups(
                        MTLSize::new(1, 1, 1),
                        MTLSize::new(
                            crate::kernels::DISPATCH_TG_MAX_THREADS.min(hidden as u64),
                            1,
                            1,
                        ),
                    );
                } else {
                    encode_rms_norm(
                        enc,
                        &self.norms.rms_norm_pipeline,
                        bufs.down_out,
                        &post_ffn_buf,
                        bufs.normed_scratch,
                        hidden,
                        norms_view.eps,
                        norms_view.norm_offset,
                    );
                    encode_residual_add(
                        enc,
                        &self.norms.residual_add_pipeline,
                        bufs.h_post_attn,
                        bufs.normed_scratch,
                        bufs.new_h,
                        hidden,
                        layer.residual_multiplier,
                    );
                }
            } else {
                encode_residual_add(
                    enc,
                    &self.norms.residual_add_pipeline,
                    bufs.h_post_attn,
                    bufs.down_out,
                    bufs.new_h,
                    hidden,
                    layer.residual_multiplier,
                );
            }
        } else {
            encode_residual_add(
                enc,
                &self.norms.residual_add_pipeline,
                bufs.h_post_attn,
                bufs.down_out,
                bufs.new_h,
                hidden,
                layer.residual_multiplier,
            );
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use larql_compute::pipeline::{
        Activation, FfnType, FullPipelineLayer, NormType, QuantFormat, QuantWeight,
    };

    fn backend() -> MetalBackend {
        MetalBackend::new().expect("Metal device available on test host")
    }

    /// Build a minimal `FullPipelineLayer` whose `norms()` view drives
    /// the `encode_post_ffn_residual` branches.  Pure CPU-side struct —
    /// no quantised weights are read by the residual encoder.
    fn layer_with(post_ffn_norm: Option<&[f32]>, has_post_norms: bool) -> FullPipelineLayer<'_> {
        let empty_q4 = QuantWeight {
            data: &[],
            scales: None,
            format: QuantFormat::Q4_K,
        };
        FullPipelineLayer {
            wq: empty_q4,
            wk: empty_q4,
            wv: empty_q4,
            wo: empty_q4,
            gate: empty_q4,
            up: empty_q4,
            down: empty_q4,
            input_norm: &[],
            post_attn_norm: &[],
            pre_ffn_norm: None,
            post_ffn_norm,
            input_norm_bias: None,
            post_attn_norm_bias: None,
            norm_offset: 1.0,
            qk_norm_offset: 0.0,
            eps: 1e-6,
            has_post_norms,
            norm_type: NormType::RmsNorm,
            ffn_type: FfnType::Gated,
            activation: Activation::Silu,
            attn_scale: 0.125,
            head_dim: 64,
            num_q_heads: 4,
            num_kv_heads: 4,
            rope_base: 10000.0,
            rotary_dim: 0,
            sliding_window: 0,
            has_v_norm: false,
            layer_scalar: 0.0,
            q_norm_weight: None,
            k_norm_weight: None,
            ffn_up_bias: None,
            ffn_down_bias: None,
            moe: None,
            ffn_is_remote: false,
            moe_combined_output_norm: false,
            moe_outer_post_norm: None,
            kv_shared_source: None,
            residual_multiplier: 1.0,
            ple_input_gate: None,
            ple_projection: None,
            ple_post_norm: None,
        }
    }

    fn drive(m: &MetalBackend, layer: &FullPipelineLayer<'_>, hidden: usize, use_fused: bool) {
        let down_out = m.bufs.transient_from_f32(&vec![0.5f32; hidden]);
        let h_post_attn = m.bufs.transient_from_f32(&vec![0.25f32; hidden]);
        let new_h = m.bufs.output((hidden * 4) as u64);
        let normed_scratch = m.bufs.output((hidden * 4) as u64);

        let cmd = m.queue.new_command_buffer();
        let enc = cmd.new_compute_command_encoder();
        m.encode_post_ffn_residual(
            enc,
            layer,
            PostFfnBufs {
                down_out: &down_out,
                h_post_attn: &h_post_attn,
                new_h: &new_h,
                normed_scratch: &normed_scratch,
            },
            hidden,
            use_fused,
            None,
        );
        enc.end_encoding();
        cmd.commit();
        cmd.wait_until_completed();

        let out = crate::buffers::read_buffer_f32(&new_h, hidden);
        assert_eq!(out.len(), hidden);
        assert!(out.iter().all(|v| v.is_finite()));
    }

    /// `has_post_norms == true` + `post_ffn_norm: Some` + `use_fused: true`
    /// drives the fused `post_ffn_norm_residual_add` dispatch (lines 98-117).
    #[test]
    fn post_norms_with_weight_and_fused_dispatches_fused_kernel() {
        let m = backend();
        let hidden = 64usize;
        let post_ffn = vec![1.0f32; hidden];
        let layer = layer_with(Some(&post_ffn), true);
        drive(&m, &layer, hidden, true);
    }

    /// Same shape, `use_fused: false` drives the unfused rms_norm +
    /// residual_add chain (lines 118-137).
    #[test]
    fn post_norms_with_weight_and_unfused_dispatches_rms_then_add() {
        let m = backend();
        let hidden = 64usize;
        let post_ffn = vec![1.0f32; hidden];
        let layer = layer_with(Some(&post_ffn), true);
        drive(&m, &layer, hidden, false);
    }

    /// `has_post_norms == true` + `post_ffn_norm: None` falls through
    /// to the plain residual-add (lines 138-147).
    #[test]
    fn post_norms_without_weight_falls_back_to_residual_add() {
        let m = backend();
        let hidden = 64usize;
        let layer = layer_with(None, true);
        drive(&m, &layer, hidden, true);
    }

    /// `has_post_norms == false` — outer `else` branch, plain residual
    /// add (lines 148-157).  Already exercised by the prefill_kquant tests
    /// but pinned here too for symmetry with the post_norms tests.
    #[test]
    fn no_post_norms_dispatches_plain_residual_add() {
        let m = backend();
        let hidden = 64usize;
        let layer = layer_with(None, false);
        drive(&m, &layer, hidden, true);
    }
}
