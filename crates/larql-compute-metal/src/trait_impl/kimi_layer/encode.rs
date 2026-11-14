//! Encoding a Kimi decoder layer (and chains of them) into one command buffer.

use super::super::grouped_experts::GroupedError;
use crate::shaders::kimi_layer as layer_shader;
use crate::MetalBackend;
use metal::{Buffer, ComputeCommandEncoderRef, MTLSize};

#[allow(unused_imports)]
use super::*;

impl MetalBackend {
    /// One decoder layer, one command buffer.
    pub fn kimi_decoder_layer(
        &self,
        w: KimiLayerWeights<'_>,
        x: &[f32],
    ) -> Result<(Vec<f32>, f64), GroupedError> {
        let p = self.kimi_decoder_layer_traced(w, x)?;
        Ok((p.output, p.gpu_ms))
    }

    /// **Several consecutive decoder layers in ONE command buffer.**
    ///
    /// Rung 5e. Layer `i+1`'s input is layer `i`'s output BUFFER — the
    /// hidden state never leaves the device, so layer `i+1`'s router
    /// scores a vector the host has not seen and its expert selection is
    /// dynamically downstream of layer `i`. That is the difference
    /// between chaining layers and merely batching them: a path that
    /// re-used the original input, or read a stale buffer, routes
    /// somewhere else and the gate catches it.
    ///
    /// Each layer carries its own recurrent state; `states` and
    /// `layers` are paired by position.
    pub fn kimi_decoder_layers(
        &self,
        layers: &[KimiLayerCall<'_>],
        x: &[f32],
        trace: Option<&mut ExecutionTrace>,
    ) -> Result<(Vec<f32>, f64), GroupedError> {
        let hidden = layers
            .first()
            .ok_or(GroupedError::NoExpertsSelected)?
            .weights
            .attention
            .hidden();
        let (scratch, kda, _, gpu_ms) = self.encode_layer_chain(layers, None, x)?;
        collect_routes(layers, &scratch, trace);
        // ONE readback: the last layer's output. Everything else stays
        // on device. Reading the traced planes here instead cost 64 ms a
        // token in the real trajectory — 12 buffers per layer over 19
        // layers, against 20 ms of actual GPU work.
        let out = crate::buffers::read_buffer_f32(&scratch[scratch.len() - 1].out, hidden);
        self.recycle_chain(scratch, kda);
        Ok((out, gpu_ms))
    }

    pub(super) fn recycle_chain(
        &self,
        scratch: Vec<LayerScratch>,
        kda_scratch: Vec<AttentionScratch>,
    ) {
        for s in scratch {
            self.recycle_layer(s);
        }
        for s in kda_scratch {
            match s {
                AttentionScratch::Kda(s) => self.recycle_scratch(s),
                AttentionScratch::Mla(s) => self.recycle_mla(s),
            }
        }
    }

    /// Encode the whole chain, commit, wait, and check every layer's
    /// refusal counter. Returns the scratch so the caller decides what
    /// to read back.
    #[allow(clippy::type_complexity)]
    #[allow(clippy::type_complexity)]
    pub(super) fn encode_layer_chain(
        &self,
        layers: &[KimiLayerCall<'_>],
        head: Option<&head::KimiHead<'_>>,
        x: &[f32],
    ) -> Result<
        (
            Vec<LayerScratch>,
            Vec<AttentionScratch>,
            Option<head::HeadScratch>,
            f64,
        ),
        GroupedError,
    > {
        if layers.is_empty() {
            return Err(GroupedError::NoExpertsSelected);
        }
        let hidden = layers[0].weights.attention.hidden();
        // Validate EVERY layer before encoding anything: an encoder
        // dropped without `end_encoding` aborts the process, so a
        // refusal found halfway through would not be recoverable. The
        // attention half validates through its own operator's checks.
        let mut visible = Vec::with_capacity(layers.len());
        for call in layers {
            let experts = call.weights.ffn.experts();
            visible.push(match call.weights.attention {
                AttentionSpec::Kda {
                    weights,
                    shape,
                    state,
                } => {
                    Self::validate_kda(weights, shape, state, hidden)?;
                    0
                }
                AttentionSpec::Mla { shape, state, .. } => {
                    Self::validate_mla(shape, state, hidden)?
                }
            });
            validate_layer(&call.weights, experts, call.weights.ffn.slots(), hidden)?;
        }
        if let Some(h) = head {
            Self::validate_head(h, hidden)?;
        }
        if x.len() != hidden {
            return Err(GroupedError::OffsetOutOfRange {
                slot: 0,
                offset: 0,
                need: hidden,
                have: x.len(),
            });
        }

        let mut kda_scratch = Vec::with_capacity(layers.len());
        let mut scratch = Vec::with_capacity(layers.len());
        for (call, &vis) in layers.iter().zip(&visible) {
            kda_scratch.push(match call.weights.attention {
                AttentionSpec::Kda { shape, .. } => AttentionScratch::Kda(self.kda_scratch(shape)),
                AttentionSpec::Mla { shape, .. } => {
                    AttentionScratch::Mla(self.mla_scratch(shape, vis))
                }
            });
            scratch.push(self.layer_scratch(
                hidden,
                call.weights.ffn.inter(),
                call.weights.ffn.experts(),
                call.weights.ffn.slots(),
                call.weights.ffn.top_k(),
            ));
        }
        let buf_x = self.bufs().transient_from_f32(x);

        let encode_clock = std::time::Instant::now();
        let cmd = self.queue().new_command_buffer();
        let enc = cmd.new_compute_command_encoder();
        // Buffers bound into the encoder must outlive the wait; the
        // cache clones held here guarantee it.
        let mut held = Vec::with_capacity(layers.len());
        for (i, call) in layers.iter().enumerate() {
            // The chain: layer 0 reads the upload, every later layer
            // reads its predecessor's OUTPUT buffer. Nothing in between
            // touches the host.
            let input = if i == 0 { &buf_x } else { &scratch[i - 1].out };
            held.push(self.encode_kimi_layer(
                enc,
                call.weights,
                input,
                &kda_scratch[i],
                &scratch[i],
                visible[i],
            ));
        }
        // The head reads the LAST layer's output buffer, inside this
        // same encoder — so the hidden state never crosses to the host
        // and the token stays at one epoch.
        let head_scratch = head.map(|h| {
            let s = self.kimi_head_scratch(hidden, h.vocab);
            held.push(self.encode_kimi_head(enc, h, &scratch[layers.len() - 1].out, &s, hidden));
            s
        });
        enc.end_encoding();
        let encode_ms = encode_clock.elapsed().as_secs_f64() * 1000.0;
        let wait_clock = std::time::Instant::now();
        cmd.commit();
        // A buffer that did not complete leaves every scratch holding
        // whatever it held before, and nothing here may read it or
        // advance a cache on it: refuse before the MLA positions move,
        // and hand the scratch back to the pool untouched.
        const WAIT_SITE: &str =
            "crates/larql-compute-metal/src/trait_impl/kimi_layer/mod.rs:layers";
        if let Err(detail) = crate::cb_status::wait_checked(cmd, WAIT_SITE) {
            drop(held);
            self.recycle_chain(scratch, kda_scratch);
            return Err(GroupedError::CommandBufferFailed {
                site: WAIT_SITE,
                detail,
            });
        }
        let wait_ms = wait_clock.elapsed().as_secs_f64() * 1000.0;
        let gpu_ms = crate::decode::gpu_timing::gpu_elapsed_ms(cmd);
        // Encode-vs-wait, because they have different fixes: encode is
        // host work per bound resource, wait is submission latency plus
        // GPU execution. Rung 5f's fixture bound 40 MiB a layer and this
        // binds ~900 MB, so the two do not scale together.
        ENCODE_MS.with(|c| c.set(c.get() + encode_ms));
        WAIT_MS.with(|c| c.set(c.get() + wait_ms));
        drop(held);

        // Every layer's refusal counter, read AFTER the wait. A route
        // that named a non-resident expert produced numbers from slot
        // 0's weights, and returning them would be the silent wrong
        // answer this seam exists to prevent.
        //
        // Checked before any cache advances: a refused call returns no
        // output, so it must leave no position behind either. The MLA
        // caches used to advance first, and a caller that handled the
        // refusal and retried ran one position ahead of its own stream.
        let mut refusal: Option<(usize, u32)> = None;
        for (i, s) in scratch.iter().enumerate() {
            let n = read_u32(&s.refusals, 1)[0];
            if n != 0 && refusal.is_none() {
                refusal = Some((i, n));
            }
        }
        if let Some((layer, count)) = refusal {
            self.recycle_chain(scratch, kda_scratch);
            return Err(GroupedError::LayerRouteNotResident {
                layer,
                refusals: count,
            });
        }

        // Each MLA layer's cache advanced by one position — but only
        // once the dispatch that wrote the latent has actually run, so
        // this happens after the wait and not at encode time, and only
        // for a call that is about to return its output.
        for (call, &vis) in layers.iter().zip(&visible) {
            if let AttentionSpec::Mla { state, .. } = call.weights.attention {
                state.advance_to(vis);
            }
        }
        Ok((scratch, kda_scratch, head_scratch, gpu_ms))
    }

    /// Encode one layer into an existing encoder, reading `input` and
    /// writing `s.out`. Returns the device buffers it bound, which the
    /// caller must hold until the wait.
    #[allow(clippy::too_many_arguments)]
    pub(super) fn encode_kimi_layer(
        &self,
        enc: &ComputeCommandEncoderRef,
        w: KimiLayerWeights<'_>,
        input: &Buffer,
        attention: &AttentionScratch,
        s: &LayerScratch,
        visible: usize,
    ) -> Vec<Buffer> {
        let hidden = w.attention.hidden();
        let f32b = |v: &[f32]| self.bufs().get_f32(v);
        let (norm_in, norm_post) = (f32b(w.input_norm), f32b(w.post_attention_norm));

        self.encode_rms_norm(enc, input, &norm_in, &s.input_normed, hidden, w.norm_eps);
        match (w.attention, attention) {
            (
                AttentionSpec::Kda {
                    weights,
                    shape,
                    state,
                },
                AttentionScratch::Kda(k),
            ) => self.encode_kda_attention(enc, weights, shape, state, &s.input_normed, k),
            (
                AttentionSpec::Mla {
                    weights,
                    shape,
                    state,
                },
                AttentionScratch::Mla(m),
            ) => self.encode_mla_attention_into(
                enc,
                weights,
                shape,
                state,
                &s.input_normed,
                m,
                visible,
            ),
            _ => unreachable!("the scratch is built from the same spec"),
        }
        // `after_attention = input + attention`. The attention plane
        // itself is `kda_scratch.out`, so nothing needs copying.
        self.encode_residual_add(
            enc,
            input,
            attention.out(),
            &s.after_attention,
            hidden,
            RESIDUAL_UNIT_SCALE,
        );
        self.encode_rms_norm(
            enc,
            &s.after_attention,
            &norm_post,
            &s.post_normed,
            hidden,
            w.norm_eps,
        );

        let ffn_held = match &w.ffn {
            ffn::FfnSpec::Moe(m) => self.encode_moe_ffn(enc, m, s, hidden),
            ffn::FfnSpec::Dense(d) => self.encode_dense_ffn(enc, d, s, hidden),
        };

        let mut held = vec![norm_in, norm_post];
        held.extend(ffn_held);
        held
    }

    pub(super) fn layer_scratch(
        &self,
        hidden: usize,
        inter: usize,
        experts: usize,
        slots: usize,
        top_k: usize,
    ) -> LayerScratch {
        // A dense layer has no router, so `experts` and `top_k` are
        // zero and the router planes below are never bound. Metal
        // refuses a zero-length buffer, so allocate one element rather
        // than special-case the struct: four bytes, never read.
        let f = |n: usize| self.bufs().output((n.max(1) * 4) as u64);
        let refusals = f(1);
        // Zeroed from the host before the encoder opens: the pool
        // recycles, and a recycled counter carries the previous route's
        // refusals.
        let ptr = refusals.contents() as *mut u32;
        if !ptr.is_null() {
            // SAFETY: a pooled 4-byte buffer not yet bound to any
            // encoder, so the GPU is not reading it.
            unsafe { std::ptr::write(ptr, 0) };
        }
        LayerScratch {
            input_normed: f(hidden),
            after_attention: f(hidden),
            post_normed: f(hidden),
            logits: f(experts),
            scores: f(experts),
            sel_scores: f(experts),
            chosen: f(top_k),
            // Routed slots only: the shared branch owns no entry in the
            // address tables — its region is bound directly.
            gate_offsets: f(top_k),
            up_offsets: f(top_k),
            down_offsets: f(top_k),
            // Always `top_k + 1`: the router writes the shared branch's
            // constant 1.0 unconditionally, and a layer without a shared
            // branch simply never reads it.
            weights: f(top_k + 1),
            refusals,
            gate_out: f(slots * inter),
            up_out: f(slots * inter),
            h: f(slots * inter),
            expert_out: f(slots * hidden),
            out: f(hidden),
        }
    }

    pub(super) fn recycle_layer(&self, s: LayerScratch) {
        for b in [
            s.input_normed,
            s.after_attention,
            s.post_normed,
            s.logits,
            s.scores,
            s.sel_scores,
            s.chosen,
            s.gate_offsets,
            s.up_offsets,
            s.down_offsets,
            s.weights,
            s.refusals,
            s.gate_out,
            s.up_out,
            s.h,
            s.expert_out,
            s.out,
        ] {
            self.bufs().recycle(b);
        }
    }

    pub(super) fn encode_rms_norm(
        &self,
        enc: &ComputeCommandEncoderRef,
        x: &Buffer,
        weight: &Buffer,
        out: &Buffer,
        len: usize,
        eps: f32,
    ) {
        let n = len as u32;
        enc.set_compute_pipeline_state(&self.norms.rms_norm_pipeline);
        enc.set_buffer(0, Some(x), 0);
        enc.set_buffer(1, Some(weight), 0);
        enc.set_buffer(2, Some(out), 0);
        enc.set_bytes(3, 4, &n as *const u32 as *const std::ffi::c_void);
        enc.set_bytes(4, 4, &eps as *const f32 as *const std::ffi::c_void);
        enc.set_bytes(
            5,
            4,
            &NORM_WEIGHT_OFFSET as *const f32 as *const std::ffi::c_void,
        );
        // One threadgroup, per the kernel's own contract.
        enc.dispatch_thread_groups(
            MTLSize::new(1, 1, 1),
            MTLSize::new(NORM_THREADS_PER_TG, 1, 1),
        );
    }

    pub(super) fn encode_router_select(
        &self,
        enc: &ComputeCommandEncoderRef,
        moe: &KimiMoeWeights<'_>,
        router_bias: &Buffer,
        s: &LayerScratch,
        experts: usize,
    ) {
        let (e, k) = (experts as u32, moe.top_k as u32);
        let renorm: u32 = u32::from(moe.renormalize);
        enc.set_compute_pipeline_state(&self.kimi.router_select);
        enc.set_buffer(0, Some(&s.logits), 0);
        enc.set_buffer(1, Some(router_bias), 0);
        enc.set_buffer(2, Some(&s.scores), 0);
        enc.set_buffer(3, Some(&s.sel_scores), 0);
        enc.set_buffer(4, Some(&s.chosen), 0);
        enc.set_buffer(5, Some(&s.weights), 0);
        enc.set_bytes(6, 4, &e as *const u32 as *const std::ffi::c_void);
        enc.set_bytes(7, 4, &k as *const u32 as *const std::ffi::c_void);
        enc.set_bytes(8, 4, &renorm as *const u32 as *const std::ffi::c_void);
        enc.set_bytes(
            9,
            4,
            &moe.branch_scale as *const f32 as *const std::ffi::c_void,
        );
        // One threadgroup: the selection is serial by design.
        enc.dispatch_thread_groups(
            MTLSize::new(1, 1, 1),
            MTLSize::new(layer_shader::SELECT_THREADS_PER_TG, 1, 1),
        );
    }

    /// `out = residual + sum_slot weight[slot] * branch[slot]`.
    ///
    /// The combine weights come from a DEVICE buffer, so the routed path
    /// passes the router's own output and the dense path passes a
    /// constant one — the same kernel, and the dense layer's residual is
    /// therefore the residual the routed layers already prove.
    pub(crate) fn encode_moe_combine(
        &self,
        enc: &ComputeCommandEncoderRef,
        s: &LayerScratch,
        weights: &Buffer,
        hidden: usize,
        slots: usize,
    ) {
        let (h, k) = (hidden as u32, slots as u32);
        enc.set_compute_pipeline_state(&self.kimi.moe_combine);
        enc.set_buffer(0, Some(&s.expert_out), 0);
        enc.set_buffer(1, Some(&s.after_attention), 0);
        enc.set_buffer(2, Some(weights), 0);
        enc.set_buffer(3, Some(&s.out), 0);
        enc.set_bytes(4, 4, &h as *const u32 as *const std::ffi::c_void);
        enc.set_bytes(5, 4, &k as *const u32 as *const std::ffi::c_void);
        crate::lowering::dispatch_linear(enc, &self.kimi.moe_combine, hidden);
    }
}
