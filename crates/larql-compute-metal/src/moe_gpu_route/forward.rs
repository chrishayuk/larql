//! Test-facing forward wrappers (control vs descriptor arms) and the
//! rung-F token scheduling probe.

use crate::moe_descriptor::MoeExpertDescriptorTable;
use crate::moe_dispatch::MoeScratch;
use crate::MetalBackend;
use larql_compute::MoeLayerWeights;
use metal::Buffer;

/// Layer index reported to the execution seam by the SINGLE-layer
/// wrappers below. They encode one synthetic layer, so there is only one
/// index to report; a policy under test addresses it as layer 0, and any
/// other layer index is the negative control (the mask must not fire).
/// The multi-layer chain in [`MetalBackend::moe_token_forward_descriptor`]
/// reports its real chain position instead.
const SYNTHETIC_LAYER: usize = 0;

impl MetalBackend {
    /// Test-facing CONTROL arm: today's production CPU-routed layer,
    /// end to end — CPU route → `resolve_selected_experts` → legacy
    /// encode (offset/weight `set_bytes`, CPU bias staging) → readback.
    /// Moves the `route_witness` counters; that movement is the
    /// witness's own positive control.
    pub fn moe_layer_forward_control(
        &self,
        router_in: &[f32],
        moe: &MoeLayerWeights<'_>,
        h_post_attn: &[f32],
    ) -> Option<Vec<f32>> {
        let hidden = h_post_attn.len();
        let (ids, weights) =
            larql_compute::cpu::ops::moe::moe_route_from_router_input(router_in, moe);
        let scratch = MoeScratch::new_public_with_format(
            self,
            moe.top_k,
            hidden,
            moe.intermediate_size,
            moe.expert_data_format,
            hidden,
        );
        let resolved = self.resolve_selected_experts(&scratch, moe, &ids, &weights, |e| {
            Some((moe.experts_gate_up[e], moe.experts_down[e]))
        })?;
        let h_buf = self.bufs.transient_from_f32(h_post_attn);
        let new_h = self.bufs.output((hidden * 4) as u64);

        let cmd = self.queue.new_command_buffer();
        let enc = cmd.new_compute_command_encoder();
        self.encode_experts_and_combine_zero_copy(
            enc,
            router_in,
            moe,
            &scratch,
            &resolved,
            &h_buf,
            &new_h,
            SYNTHETIC_LAYER,
        );
        enc.end_encoding();
        cmd.commit();
        cmd.wait_until_completed();
        crate::buffers::try_read_buffer_f32(&new_h, hidden)
    }

    /// Test-facing CANDIDATE arm: GPU router → GPU select → descriptor
    /// arm, one command buffer, no host-visible route anywhere in the
    /// signature (type-level closure). Must NOT move the
    /// `route_witness` counters.
    ///
    /// `poison_staging_scratch` scribbles garbage into every scratch
    /// buffer the LEGACY path staged via CPU (`gate_bias_buf`,
    /// `up_bias_buf`, `down_bias_staged`) before encoding — the poison
    /// proof: the GPU kernels must fully overwrite them, so output must
    /// be identical whether or not the poison ran.
    pub fn moe_layer_forward_descriptor(
        &self,
        router_in: &[f32],
        moe: &MoeLayerWeights<'_>,
        table: &MoeExpertDescriptorTable,
        h_post_attn: &[f32],
        poison_staging_scratch: bool,
    ) -> Option<Vec<f32>> {
        use larql_compute::{MoeExpertScalePolicy, MoeTopKWeightPolicy};
        let hidden = h_post_attn.len();
        let num_experts = moe.num_experts;
        if router_in.len() != hidden
            || moe.router_proj.len() != num_experts * hidden
            || num_experts > crate::shaders::moe_router_select::MAX_EXPERTS
            || moe.top_k == 0
            || moe.top_k > crate::shaders::moe_router_select::MAX_TOP_K
        {
            return None;
        }
        let renormalize =
            moe.routing_policy.selected_weight == MoeTopKWeightPolicy::RenormalizedSoftmax;
        let pe_scale = (moe.routing_policy.expert_scale == MoeExpertScalePolicy::PerExpert
            && !moe.router_per_expert_scale.is_empty())
        .then_some(moe.router_per_expert_scale);
        if let Some(s) = pe_scale {
            if s.len() != num_experts {
                return None;
            }
        }

        let scratch = MoeScratch::new_public_with_format(
            self,
            moe.top_k,
            hidden,
            moe.intermediate_size,
            moe.expert_data_format,
            hidden,
        );
        if poison_staging_scratch {
            // Every value the LEGACY path would have host-staged becomes
            // unmistakable garbage; any read that escapes GPU staging
            // explodes the parity gate instead of coincidentally passing.
            for buf in [
                &scratch.gate_bias_buf,
                &scratch.up_bias_buf,
                &scratch.down_bias_staged,
            ] {
                let len = buf.length() as usize / 4;
                unsafe {
                    let p = buf.contents() as *mut f32;
                    for i in 0..len {
                        *p.add(i) = 1.0e30;
                    }
                }
            }
        }

        let w_buf = self.bufs.get_f32(moe.router_proj);
        let x_router = self.bufs.transient_from_f32(router_in);
        let bias_buf =
            (!moe.router_bias.is_empty()).then(|| self.bufs.transient_from_f32(moe.router_bias));
        let scale_buf = pe_scale.map(|s| self.bufs.transient_from_f32(s));
        let h_buf = self.bufs.transient_from_f32(h_post_attn);
        let new_h = self.bufs.output((hidden * 4) as u64);

        let cmd = self.queue.new_command_buffer();
        let enc = cmd.new_compute_command_encoder();
        let logits = self.encode_moe_router_logits(
            enc,
            &w_buf,
            &x_router,
            bias_buf.as_ref(),
            num_experts,
            hidden,
        );
        let (ids_buf, weights_buf) = self.encode_moe_router_select(
            enc,
            &logits,
            scale_buf.as_ref(),
            num_experts,
            moe.top_k,
            renormalize,
        );
        self.encode_experts_and_combine_descriptor(
            enc,
            router_in,
            moe,
            &scratch,
            table,
            &ids_buf,
            &weights_buf,
            &h_buf,
            &new_h,
            SYNTHETIC_LAYER,
        );
        enc.end_encoding();
        cmd.commit();
        cmd.wait_until_completed();
        crate::buffers::try_read_buffer_f32(&new_h, hidden)
    }
}

/// One synthetic token's scheduling measurements from
/// [`MetalBackend::moe_token_forward_descriptor`]. `out` carries the
/// final layer's output so arms can be compared numerically — a
/// submission policy must not change the numbers.
pub struct MoeTokenScheduleStats {
    pub cmd_bufs: usize,
    pub wall_ms: f64,
    /// Σ (GPUEndTime − GPUStartTime) over the token's command buffers.
    pub gpu_busy_ms: f64,
    /// Σ positive gaps between consecutive command buffers' GPU windows
    /// — the queue-starvation bubble. Zero by construction at one CB.
    pub bubble_ms: f64,
    pub out: Vec<f32>,
}

impl MetalBackend {
    /// Rung F's instrument: run `layers` chained descriptor-driven MoE
    /// layers (layer i+1's router input IS layer i's output buffer — no
    /// readback, no host staging between layers) under one of two
    /// submission policies:
    ///
    /// - `pre_encode = false` (JIT): one command buffer per layer,
    ///   commit + wait each — production decode's cadence today.
    /// - `pre_encode = true`: every layer encoded into ONE command
    ///   buffer, committed once — the shape E's semantic closure makes
    ///   legal.
    ///
    /// Identical kernels, buffers and encode order in both arms; only
    /// WHEN work is submitted differs.
    pub fn moe_token_forward_descriptor(
        &self,
        router_in: &[f32],
        moe: &MoeLayerWeights<'_>,
        table: &MoeExpertDescriptorTable,
        layers: usize,
        pre_encode: bool,
    ) -> Option<MoeTokenScheduleStats> {
        use larql_compute::{MoeExpertScalePolicy, MoeTopKWeightPolicy};
        use objc::{msg_send, sel, sel_impl};

        let hidden = router_in.len();
        let num_experts = moe.num_experts;
        if layers == 0
            || moe.router_proj.len() != num_experts * hidden
            || num_experts > crate::shaders::moe_router_select::MAX_EXPERTS
            || moe.top_k == 0
            || moe.top_k > crate::shaders::moe_router_select::MAX_TOP_K
        {
            return None;
        }
        let renormalize =
            moe.routing_policy.selected_weight == MoeTopKWeightPolicy::RenormalizedSoftmax;
        let pe_scale = (moe.routing_policy.expert_scale == MoeExpertScalePolicy::PerExpert
            && !moe.router_per_expert_scale.is_empty())
        .then_some(moe.router_per_expert_scale);

        let scratch = MoeScratch::new_public_with_format(
            self,
            moe.top_k,
            hidden,
            moe.intermediate_size,
            moe.expert_data_format,
            hidden,
        );
        let w_buf = self.bufs.get_f32(moe.router_proj);
        let bias_buf =
            (!moe.router_bias.is_empty()).then(|| self.bufs.transient_from_f32(moe.router_bias));
        let scale_buf = pe_scale.map(|s| self.bufs.transient_from_f32(s));
        let h0 = self.bufs.transient_from_f32(router_in);
        let new_hs: Vec<Buffer> = (0..layers)
            .map(|_| self.bufs.output((hidden * 4) as u64))
            .collect();

        let encode_layer =
            |enc: &metal::ComputeCommandEncoderRef, prev_h: &Buffer, out: &Buffer, layer: usize| {
                let logits = self.encode_moe_router_logits(
                    enc,
                    &w_buf,
                    prev_h,
                    bias_buf.as_ref(),
                    num_experts,
                    hidden,
                );
                let (ids_buf, weights_buf) = self.encode_moe_router_select(
                    enc,
                    &logits,
                    scale_buf.as_ref(),
                    num_experts,
                    moe.top_k,
                    renormalize,
                );
                self.encode_experts_and_combine_descriptor_x_buf(
                    enc,
                    prev_h,
                    moe,
                    &scratch,
                    table,
                    &ids_buf,
                    &weights_buf,
                    prev_h,
                    out,
                    layer,
                );
            };

        let t0 = std::time::Instant::now();
        let mut windows: Vec<(f64, f64)> = Vec::with_capacity(layers);
        let cmd_bufs;
        if pre_encode {
            let cmd = self.queue.new_command_buffer();
            let enc = cmd.new_compute_command_encoder();
            let mut prev = h0.clone();
            for (layer, out) in new_hs.iter().enumerate() {
                encode_layer(enc, &prev, out, layer);
                prev = out.clone();
            }
            enc.end_encoding();
            cmd.commit();
            cmd.wait_until_completed();
            windows.push(unsafe {
                let start: f64 = msg_send![cmd, GPUStartTime];
                let end: f64 = msg_send![cmd, GPUEndTime];
                (start, end)
            });
            cmd_bufs = 1;
        } else {
            let mut prev = h0.clone();
            for (layer, out) in new_hs.iter().enumerate() {
                let cmd = self.queue.new_command_buffer();
                let enc = cmd.new_compute_command_encoder();
                encode_layer(enc, &prev, out, layer);
                enc.end_encoding();
                cmd.commit();
                cmd.wait_until_completed();
                windows.push(unsafe {
                    let start: f64 = msg_send![cmd, GPUStartTime];
                    let end: f64 = msg_send![cmd, GPUEndTime];
                    (start, end)
                });
                prev = out.clone();
            }
            cmd_bufs = layers;
        }
        let wall_ms = t0.elapsed().as_secs_f64() * 1e3;

        let gpu_busy_ms: f64 = windows.iter().map(|(s, e)| (e - s) * 1e3).sum();
        let bubble_ms: f64 = windows
            .windows(2)
            .map(|w| ((w[1].0 - w[0].1) * 1e3).max(0.0))
            .sum();
        let out = crate::buffers::try_read_buffer_f32(&new_hs[layers - 1], hidden)?;
        Some(MoeTokenScheduleStats {
            cmd_bufs,
            wall_ms,
            gpu_busy_ms,
            bubble_ms,
            out,
        })
    }
}
