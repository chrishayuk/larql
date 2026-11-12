//! `DecodeBackend` impl for `MetalBackend`.
//!
//! These methods drive the GPU full-pipeline / KV-cached decode /
//! prefill paths. Most of them delegate to dispatchers under
//! `metal::ops::full_pipeline` or to inherent helpers on
//! `MetalBackend` (e.g. `decode_token`, `decode_token_with_moe_fn`).
//!
//! The trait surface intentionally takes no scalar attention geometry —
//! all geometry is read per-layer from `FullPipelineLayer` inside the
//! dispatchers. The inner free-fns under `metal::decode` and
//! `metal::ops::full_pipeline` retain their existing scalar parameters
//! for synthetic-architecture tests; here we synthesise those values
//! from `layers[0]` since the dispatchers ignore them on production
//! paths anyway (per-layer reads are authoritative — see
//! `metal/decode/setup.rs:DecodeScratch::new` and
//! `metal/ops/full_pipeline/buffers.rs:LayerBuffers::allocate`).

use crate::{ops, MetalBackend};
use larql_compute::backend::DecodeBackend;

mod prefill;

/// `(q_dim, kv_dim, num_q_heads, num_kv_heads, head_dim, rope_base)` for
/// layer 0 — passed to the inner dispatchers as legacy scalars. Only
/// `q_dim` is read on a non-empty-layers path (as the empty-layers
/// fallback for scratch sizing); the rest are underscored downstream.
fn legacy_l0_geometry(
    layers: &[larql_compute::FullPipelineLayer<'_>],
) -> (usize, usize, usize, usize, usize, f32) {
    match layers.first() {
        Some(l) => (
            l.num_q_heads * l.head_dim,
            l.num_kv_heads * l.head_dim,
            l.num_q_heads,
            l.num_kv_heads,
            l.head_dim,
            l.rope_base,
        ),
        None => (0, 0, 0, 0, 0, 0.0),
    }
}

impl DecodeBackend for MetalBackend {
    #[allow(clippy::too_many_arguments)]
    fn full_pipeline_q4(
        &self,
        layers: &[larql_compute::FullPipelineLayer<'_>],
        x: &[f32],
        hidden: usize,
        inter: usize,
        seq_len: usize,
        use_qk_norm: bool,
        softcap: f32,
    ) -> Option<Vec<f32>> {
        let (q_dim, kv_dim, num_q_heads, num_kv_heads, head_dim, rope_base) =
            legacy_l0_geometry(layers);
        let geglu = if layers
            .first()
            .is_some_and(|l| l.activation == larql_compute::Activation::GeluTanh)
        {
            &self.ffn.geglu_gelu_tanh_pipeline
        } else {
            &self.ffn.geglu_pipeline
        };
        Some(ops::full_pipeline::dispatch_full_pipeline(
            &self.queue,
            &self.bufs,
            &self.q4,
            geglu,
            &self.ffn.geglu_gelu_tanh_pipeline,
            &self.ffn.silu_pipeline,
            &self.ffn.gelu_tanh_pipeline,
            &self.quant.q8_quant_pipeline,
            Some(&self.attention.fused_attn_pipeline),
            &self.quant.q8_matvec_pipeline.state,
            &self.attention.q8_qkv_proj_pipeline.state,
            &self.quant.q4k_matvec_pipeline,
            Some(&self.quant.q4k_matmul_pipeline),
            &self.quant.q6k_matvec_pipeline,
            &self.norms.rms_norm_pipeline,
            &self.norms.residual_add_pipeline,
            &self.norms.rms_norm_q8_pipeline,
            &self.norms.residual_norm_q8_pipeline,
            Some(&self.attention.q4k_qkv_proj_pipeline.state),
            Some(&self.attention.q4kf_qkv_proj_pipeline.state),
            Some(&self.attention.q4k_q6k_qkv_proj_pipeline),
            Some(&self.attention.q4kf_proj_pipeline.state),
            Some(&self.attention.bias_add_pipeline),
            None,
            Some(&self.norms.qk_norm_pipeline),
            Some(&self.norms.scale_vector_pipeline),
            Some(&self.ffn.q4k_geglu_silu_down_pipeline),
            Some(&self.ffn.q4k_geglu_gelu_tanh_down_pipeline),
            Some(&self.ffn.q6k_geglu_silu_down_pipeline),
            Some(&self.ffn.q6k_geglu_gelu_tanh_down_pipeline),
            None,
            layers,
            x,
            hidden,
            inter,
            q_dim,
            kv_dim,
            seq_len,
            num_q_heads,
            num_kv_heads,
            head_dim,
            rope_base,
            use_qk_norm,
            softcap,
            None, // moe_fn: no MoE callback for full_pipeline_q4
            None, // intervention: no head replacement
        ))
    }

    fn full_pipeline_q4_with_head_replacement(
        &self,
        layers: &[larql_compute::FullPipelineLayer<'_>],
        x: &[f32],
        hidden: usize,
        inter: usize,
        seq_len: usize,
        use_qk_norm: bool,
        softcap: f32,
        target_layer: usize,
        target_head: usize,
        replacement_delta: &[f32],
    ) -> Option<Vec<f32>> {
        use ops::full_pipeline::{dispatch_full_pipeline, PipelineIntervention};
        let (q_dim, kv_dim, num_q_heads, num_kv_heads, head_dim, rope_base) =
            legacy_l0_geometry(layers);
        let geglu = if layers
            .first()
            .is_some_and(|l| l.activation == larql_compute::Activation::GeluTanh)
        {
            &self.ffn.geglu_gelu_tanh_pipeline
        } else {
            &self.ffn.geglu_pipeline
        };
        // Intervention geometry must match the target layer (head_dim and
        // num_q_heads can differ across sliding/global layers on Gemma 4).
        let (target_head_dim, target_num_q_heads) = layers
            .get(target_layer)
            .map(|l| (l.head_dim, l.num_q_heads))
            .unwrap_or((head_dim, num_q_heads));
        let intervention = PipelineIntervention {
            target_layer,
            target_head,
            head_dim: target_head_dim,
            num_q_heads: target_num_q_heads,
            replacement_delta,
            pre_wo_capture: std::cell::RefCell::new(Vec::new()),
            stop_after_capture: false,
        };
        Some(dispatch_full_pipeline(
            &self.queue,
            &self.bufs,
            &self.q4,
            geglu,
            &self.ffn.geglu_gelu_tanh_pipeline,
            &self.ffn.silu_pipeline,
            &self.ffn.gelu_tanh_pipeline,
            &self.quant.q8_quant_pipeline,
            Some(&self.attention.fused_attn_pipeline),
            &self.quant.q8_matvec_pipeline.state,
            &self.attention.q8_qkv_proj_pipeline.state,
            &self.quant.q4k_matvec_pipeline,
            Some(&self.quant.q4k_matmul_pipeline),
            &self.quant.q6k_matvec_pipeline,
            &self.norms.rms_norm_pipeline,
            &self.norms.residual_add_pipeline,
            &self.norms.rms_norm_q8_pipeline,
            &self.norms.residual_norm_q8_pipeline,
            Some(&self.attention.q4k_qkv_proj_pipeline.state),
            Some(&self.attention.q4kf_qkv_proj_pipeline.state),
            Some(&self.attention.q4k_q6k_qkv_proj_pipeline),
            Some(&self.attention.q4kf_proj_pipeline.state),
            Some(&self.attention.bias_add_pipeline),
            Some(&self.attention.rope_at_pos_pipeline), // per-position RoPE — required for seq_len > 1
            Some(&self.norms.qk_norm_pipeline),
            Some(&self.norms.scale_vector_pipeline),
            Some(&self.ffn.q4k_geglu_silu_down_pipeline),
            Some(&self.ffn.q4k_geglu_gelu_tanh_down_pipeline),
            Some(&self.ffn.q6k_geglu_silu_down_pipeline),
            Some(&self.ffn.q6k_geglu_gelu_tanh_down_pipeline),
            None, // no KV cache — stateless prefill, each prompt independent
            layers,
            x,
            hidden,
            inter,
            q_dim,
            kv_dim,
            seq_len,
            num_q_heads,
            num_kv_heads,
            head_dim,
            rope_base,
            use_qk_norm,
            softcap,
            None, // no MoE
            Some(&intervention),
        ))
    }

    fn multi_layer_q4_ffn(
        &self,
        layers_q4: &[(&[u8], &[u8], &[u8])],
        x: &[f32],
        inter: usize,
        hidden: usize,
    ) -> Option<Vec<f32>> {
        Some(MetalBackend::multi_layer_q4_ffn(
            self, layers_q4, x, inter, hidden,
        ))
    }

    #[allow(clippy::too_many_arguments)]
    fn prefill_kquant(
        &self,
        layers: &[larql_compute::FullPipelineLayer<'_>],
        x: &[f32],
        hidden: usize,
        inter: usize,
        seq_len: usize,
        use_qk_norm: bool,
        softcap: f32,
    ) -> Option<Vec<f32>> {
        self.run_prefill_kquant(layers, x, hidden, inter, seq_len, use_qk_norm, softcap)
    }

    fn full_pipeline_kquant_capture_pre_wo(
        &self,
        layers: &[larql_compute::FullPipelineLayer<'_>],
        x: &[f32],
        hidden: usize,
        inter: usize,
        seq_len: usize,
        use_qk_norm: bool,
        softcap: f32,
        target_layer: usize,
        target_head: usize,
    ) -> Option<Vec<f32>> {
        self.run_full_pipeline_kquant_capture_pre_wo(
            layers,
            x,
            hidden,
            inter,
            seq_len,
            use_qk_norm,
            softcap,
            target_layer,
            target_head,
        )
    }

    fn prefill_kquant_with_head_replacement(
        &self,
        layers: &[larql_compute::FullPipelineLayer<'_>],
        x: &[f32],
        hidden: usize,
        inter: usize,
        seq_len: usize,
        use_qk_norm: bool,
        softcap: f32,
        target_layer: usize,
        target_head: usize,
        replacement_delta: &[f32],
    ) -> Option<Vec<f32>> {
        self.run_prefill_kquant_with_head_replacement(
            layers,
            x,
            hidden,
            inter,
            seq_len,
            use_qk_norm,
            softcap,
            target_layer,
            target_head,
            replacement_delta,
        )
    }

    fn has_kv_cache(&self) -> bool {
        true
    }

    fn populate_kv_layer(
        &self,
        layer: usize,
        k_data: &[f32],
        v_data: &[f32],
        seq_len: usize,
        num_kv_heads: usize,
        head_dim: usize,
    ) {
        let mut cache_guard = self.kv_cache.lock().unwrap();
        if cache_guard.is_none() {
            *cache_guard = Some(self.create_kv_cache(
                layer + 1,
                crate::decode::DEFAULT_KV_CACHE_MAX_SEQ,
                num_kv_heads,
                head_dim,
            ));
        }
        let kv = cache_guard.as_mut().unwrap();
        while kv.layers.len() <= layer {
            kv.layers.push(ops::kv_cache::LayerKVCache::new(
                &self.bufs,
                crate::decode::DEFAULT_KV_CACHE_MAX_SEQ,
                num_kv_heads,
                head_dim,
            ));
        }

        let total = seq_len * num_kv_heads * head_dim;
        let lc = &kv.layers[layer];
        // The copy below writes `total` floats into buffers allocated for
        // this layer's own shape. It was bounded by the source lengths only,
        // so a seed longer than the cache, or with another head shape, wrote
        // past the end of the Metal buffer. The seed is refused instead; the
        // lock is released first so the refusal does not poison the cache.
        let refusal = if (num_kv_heads, head_dim) != (lc.num_kv_heads, lc.head_dim) {
            Some(format!(
                "its shape is {num_kv_heads} KV heads x {head_dim}, and the layer's cache \
                 is {} x {}",
                lc.num_kv_heads, lc.head_dim
            ))
        } else if seq_len > lc.max_seq {
            Some(format!(
                "it holds {seq_len} rows, and the layer's cache holds at most {}",
                lc.max_seq
            ))
        } else if k_data.len() != total || v_data.len() != total {
            Some(format!(
                "it carries {} key and {} value floats, and {seq_len} rows of that shape \
                 are {total}",
                k_data.len(),
                v_data.len()
            ))
        } else {
            None
        };
        if let Some(reason) = refusal {
            drop(cache_guard);
            panic!("populate_kv_layer: the seed for layer {layer} was refused: {reason}");
        }
        let lc = &mut cache_guard
            .as_mut()
            .expect("the cache was installed above")
            .layers[layer];
        let k_ptr = lc.k_cache.contents() as *mut f32;
        let v_ptr = lc.v_cache.contents() as *mut f32;
        // SAFETY: k_ptr/v_ptr point to Metal buffers of
        // `max_seq * num_kv_heads * head_dim` floats, and the checks above
        // established that `total` is at most that and is exactly the
        // length of both sources.
        unsafe {
            std::ptr::copy_nonoverlapping(k_data.as_ptr(), k_ptr, total);
            std::ptr::copy_nonoverlapping(v_data.as_ptr(), v_ptr, total);
        }
        lc.seed(seq_len);
    }

    fn reset_kv_cache(&self) {
        let mut cache_guard = self.kv_cache.lock().unwrap();
        if let Some(ref mut kv) = *cache_guard {
            for layer in &mut kv.layers {
                // `clear()`, not `current_len = 0`. The cache splits
                // occupancy from stream position on purpose — a slid
                // window drops rows while the stream keeps advancing — and
                // `abs_position` is what RoPE is computed at. Zeroing only
                // the occupancy left `abs_position` climbing across every
                // reset, so the first token of the NEXT sequence was
                // rotated at the previous sequence's position and the
                // whole decode drifted. Found via issue #227: once the
                // O-projection stopped emitting zeros, two decodes over an
                // identical seeded history stopped matching.
                layer.clear();
            }
        }
    }

    fn kv_cache_len(&self) -> usize {
        self.kv_cache
            .lock()
            .unwrap()
            .as_ref()
            .map(|kv| kv.current_len())
            .unwrap_or(0)
    }

    fn truncate_kv_cache(&self, len: usize) {
        if let Some(ref mut kv) = *self.kv_cache.lock().unwrap() {
            for layer in &mut kv.layers {
                layer.roll_back_to(len);
            }
        }
    }

    fn preallocate_kv_cache_per_layer(&self, shapes: &[(usize, usize)], max_seq: usize) {
        // Replace any existing cache — callers invoke this once per
        // model load, before the first decode dispatch. If we kept an
        // old cache sized with the wrong per-layer dims the first
        // decode would read off the end of a global-layer buffer.
        let mut cache_guard = self.kv_cache.lock().unwrap();
        *cache_guard = Some(self.create_kv_cache_per_layer(shapes, max_seq));
    }

    fn preallocate_kv_cache_per_layer_with_capacity(
        &self,
        shapes: &[(usize, usize)],
        capacities: &[usize],
    ) {
        // Same replace-outright contract as the uniform variant above;
        // only the per-layer row count differs.
        let default_capacity = capacities.iter().copied().max().unwrap_or(0);
        let mut cache_guard = self.kv_cache.lock().unwrap();
        *cache_guard = Some(
            crate::ops::kv_cache::KVCache::new_per_layer_with_capacities(
                &self.bufs,
                shapes,
                capacities,
                default_capacity,
            ),
        );
    }

    fn decode_token(
        &self,
        layers: &[larql_compute::FullPipelineLayer<'_>],
        x: &[f32],
        hidden: usize,
        inter: usize,
    ) -> Option<Vec<f32>> {
        let (q_dim, kv_dim, num_q_heads, num_kv_heads, head_dim, rope_base) =
            legacy_l0_geometry(layers);
        let mut cache_guard = self.kv_cache.lock().unwrap();
        let kv = self.ensure_kv_cache_for_layers(
            &mut cache_guard,
            layers,
            crate::decode::DEFAULT_KV_CACHE_MAX_SEQ,
        );
        Some(MetalBackend::decode_token(
            self,
            kv,
            layers,
            x,
            hidden,
            inter,
            q_dim,
            kv_dim,
            num_q_heads,
            num_kv_heads,
            head_dim,
            rope_base,
        ))
    }

    fn decode_token_with_state_dump(
        &self,
        layers: &[larql_compute::FullPipelineLayer<'_>],
        x: &[f32],
        hidden: usize,
        inter: usize,
        state: Option<&mut larql_compute::DecodeStateDump>,
    ) -> Option<Vec<f32>> {
        self.decode_token_with_state_dump_masked(
            layers,
            x,
            hidden,
            inter,
            state,
            larql_compute::StateDumpMask::Full,
        )
    }

    fn decode_token_with_state_dump_masked(
        &self,
        layers: &[larql_compute::FullPipelineLayer<'_>],
        x: &[f32],
        hidden: usize,
        inter: usize,
        state: Option<&mut larql_compute::DecodeStateDump>,
        mask: larql_compute::StateDumpMask,
    ) -> Option<Vec<f32>> {
        let Some(state) = state else {
            // No state requested → fall back to the fast fused path.
            return <Self as DecodeBackend>::decode_token(self, layers, x, hidden, inter);
        };
        let (q_dim, kv_dim, num_q_heads, num_kv_heads, head_dim, rope_base) =
            legacy_l0_geometry(layers);
        let mut cache_guard = self.kv_cache.lock().unwrap();
        let kv = self.ensure_kv_cache_for_layers(
            &mut cache_guard,
            layers,
            crate::decode::DEFAULT_KV_CACHE_MAX_SEQ,
        );
        Some(MetalBackend::decode_token_with_state_dump_masked_fn(
            self,
            kv,
            layers,
            x,
            hidden,
            inter,
            q_dim,
            kv_dim,
            num_q_heads,
            num_kv_heads,
            head_dim,
            rope_base,
            state,
            mask,
        ))
    }

    fn decode_token_with_moe(
        &self,
        layers: &[larql_compute::FullPipelineLayer<'_>],
        x: &[f32],
        hidden: usize,
        inter: usize,
        moe_fn: &mut dyn FnMut(usize, &[f32]) -> Vec<f32>,
    ) -> Option<Vec<f32>> {
        let (q_dim, kv_dim, num_q_heads, num_kv_heads, head_dim, rope_base) =
            legacy_l0_geometry(layers);
        let mut cache_guard = self.kv_cache.lock().unwrap();
        let kv = self.ensure_kv_cache_for_layers(
            &mut cache_guard,
            layers,
            crate::decode::DEFAULT_KV_CACHE_MAX_SEQ,
        );
        Some(MetalBackend::decode_token_with_moe_fn(
            self,
            kv,
            layers,
            x,
            hidden,
            inter,
            q_dim,
            kv_dim,
            num_q_heads,
            num_kv_heads,
            head_dim,
            rope_base,
            Some(moe_fn),
        ))
    }

    fn decode_token_q4k_moe<'w>(
        &self,
        layers: &[larql_compute::FullPipelineLayer<'_>],
        x: &[f32],
        hidden: usize,
        inter: usize,
        norm_eps: f32,
        get_expert: &dyn Fn(usize, usize) -> Option<(&'w [u8], &'w [u8])>,
    ) -> Option<Vec<f32>> {
        let (q_dim, kv_dim, num_q_heads, num_kv_heads, head_dim, rope_base) =
            legacy_l0_geometry(layers);
        MetalBackend::decode_token_q4k_moe(
            self,
            layers,
            x,
            hidden,
            inter,
            q_dim,
            kv_dim,
            num_q_heads,
            num_kv_heads,
            head_dim,
            rope_base,
            norm_eps,
            get_expert,
            None, // head — this entry point returns the hidden state
        )
    }

    fn decode_token_q4k_moe_head<'w>(
        &self,
        layers: &[larql_compute::FullPipelineLayer<'_>],
        x: &[f32],
        hidden: usize,
        inter: usize,
        norm_eps: f32,
        get_expert: &dyn Fn(usize, usize) -> Option<(&'w [u8], &'w [u8])>,
        head: &larql_compute::DecodeHeadPlan<'_>,
    ) -> Option<Vec<(u32, f32)>> {
        let (q_dim, kv_dim, num_q_heads, num_kv_heads, head_dim, rope_base) =
            legacy_l0_geometry(layers);
        // The slot the encoded head writes into. It stays `None` when any
        // precondition refused, and the caller then runs the unfused head
        // — so a refusal costs a normal token, never a wrong one.
        let mut hits = None;
        let hidden_out = MetalBackend::decode_token_q4k_moe(
            self,
            layers,
            x,
            hidden,
            inter,
            q_dim,
            kv_dim,
            num_q_heads,
            num_kv_heads,
            head_dim,
            rope_base,
            norm_eps,
            get_expert,
            Some(crate::decode::HeadRequest {
                plan: head,
                out: &mut hits,
            }),
        );
        // `None` from the step means a command buffer inside it failed
        // (`cb_status`): the step *ran* — KV was appended — but its bytes
        // are poison. That must not read as a refusal, or the caller would
        // fall through to the unfused path and run the same token again on
        // top of the appended KV. Report it as an empty top-K instead:
        // "the step ran and has no usable candidates", which the caller
        // treats as terminal.
        if hidden_out.is_none() {
            return Some(Vec::new());
        }
        hits
    }

    fn decode_token_with_moe_split(
        &self,
        layers: &[larql_compute::FullPipelineLayer<'_>],
        x: &[f32],
        hidden: usize,
        inter: usize,
        moe_fire_fn: &mut dyn FnMut(usize, &[f32]),
        moe_collect_fn: &mut dyn FnMut(usize) -> Vec<f32>,
    ) -> Option<Vec<f32>> {
        let (q_dim, kv_dim, num_q_heads, num_kv_heads, head_dim, rope_base) =
            legacy_l0_geometry(layers);
        let mut cache_guard = self.kv_cache.lock().unwrap();
        let kv = self.ensure_kv_cache_for_layers(
            &mut cache_guard,
            layers,
            crate::decode::DEFAULT_KV_CACHE_MAX_SEQ,
        );
        // Wrap fire so its return value is ignored — the decode-loop closure
        // already discards moe_fn's output when split mode is active.
        let mut fire_wrapper = |layer: usize, h: &[f32]| -> Vec<f32> {
            moe_fire_fn(layer, h);
            Vec::new()
        };
        Some(MetalBackend::decode_token_with_moe_split_fn(
            self,
            kv,
            layers,
            x,
            hidden,
            inter,
            q_dim,
            kv_dim,
            num_q_heads,
            num_kv_heads,
            head_dim,
            rope_base,
            Some(&mut fire_wrapper),
            Some(moe_collect_fn),
            None, // no state capture on split fire/collect MoE path
            larql_compute::StateDumpMask::Full,
            None,
            None, // head — split fire/collect returns the hidden state
        ))
    }

    fn decode_token_split_profile(
        &self,
        layers: &[larql_compute::FullPipelineLayer<'_>],
        x: &[f32],
        hidden: usize,
        inter: usize,
    ) -> (Option<Vec<f32>>, f64, f64, f64) {
        // Per-stage GPU timing comes from `decode_token_with_moe_split_fn`
        // when `LARQL_PROFILE_SPLIT=1` is set: paired commit/wait boundaries
        // around the attention vs FFN blocks land per-stage GPU windows in
        // a thread-local. We read them back here. Without the env flag,
        // we fall back to whole-token wall time in `attn_ms` so callers
        // still see something useful — but they should set the flag to
        // get the actual split.
        use crate::decode::profile;
        let t0 = std::time::Instant::now();
        let result = <Self as DecodeBackend>::decode_token(self, layers, x, hidden, inter);
        let timings = profile::take_last_split_timings().unwrap_or_else(|| {
            // Fall back: report whole-step wall in attn_ms so the caller sees
            // a non-zero number when LARQL_PROFILE_SPLIT isn't set.
            let wall_ms = t0.elapsed().as_secs_f64() * 1000.0;
            profile::ProfileTimings {
                attn_ms: wall_ms,
                gate_up_ms: 0.0,
                down_ms: 0.0,
                // No split was recorded, so nothing is known about how much
                // of this wall was GPU. Report zero rather than guessing —
                // a fabricated `gpu_ms` here would be the same mistake the
                // "GPU fwd" counter made.
                gpu_ms: 0.0,
                wall_ms,
                cmd_buffers: 0,
            }
        });
        (result, timings.attn_ms, timings.gate_up_ms, timings.down_ms)
    }
}
