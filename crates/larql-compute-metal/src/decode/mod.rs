use super::*;

mod diag;
mod encode_attn;
mod encode_ffn;
mod encode_ple;
mod encode_post_ffn;
mod encode_qkv;
pub mod gpu_timing;
mod moe_combine;
mod moe_interleave;
pub mod profile;
mod setup;

pub use profile::ProfileTimings;

pub(crate) const DEFAULT_KV_CACHE_MAX_SEQ: usize = 4096;

impl MetalBackend {
    /// Create a KV cache for decode mode with uniform per-layer dims.
    ///
    /// Production decode/prefill should use [`Self::create_kv_cache_per_layer`]
    /// (or, on the trait surface, `preallocate_kv_cache_per_layer`) so models
    /// like Gemma 4 31B with sliding/global geometry alternation are sized
    /// correctly. This uniform helper is retained for synthetic-architecture
    /// tests and the lazy bootstrap inside `populate_kv_layer`, where the
    /// caller passes a single layer's `(num_kv_heads, head_dim)` and any
    /// subsequent layers are pushed via `kv.layers.push(...)` with their own
    /// per-layer dims.
    pub fn create_kv_cache(
        &self,
        num_layers: usize,
        max_seq: usize,
        num_kv_heads: usize,
        head_dim: usize,
    ) -> ops::kv_cache::KVCache {
        ops::kv_cache::KVCache::new(&self.bufs, num_layers, max_seq, num_kv_heads, head_dim)
    }

    /// Create a KV cache with per-layer shapes for models with asymmetric
    /// attention geometry (Gemma 4 31B sliding=16×256 / global=4×512).
    /// `shapes[i] = (num_kv_heads_i, head_dim_i)` for layer i.
    pub fn create_kv_cache_per_layer(
        &self,
        shapes: &[(usize, usize)],
        max_seq: usize,
    ) -> ops::kv_cache::KVCache {
        ops::kv_cache::KVCache::new_per_layer(&self.bufs, shapes, max_seq)
    }

    pub(crate) fn kv_shapes_for_layers(
        layers: &[larql_compute::FullPipelineLayer<'_>],
    ) -> Vec<(usize, usize)> {
        layers
            .iter()
            .map(|layer| (layer.num_kv_heads, layer.head_dim))
            .collect()
    }

    pub(crate) fn ensure_kv_cache_for_layers<'a>(
        &self,
        cache: &'a mut Option<ops::kv_cache::KVCache>,
        layers: &[larql_compute::FullPipelineLayer<'_>],
        max_seq: usize,
    ) -> &'a mut ops::kv_cache::KVCache {
        let shapes = Self::kv_shapes_for_layers(layers);
        self.ensure_kv_cache_for_shapes(cache, &shapes, max_seq)
    }

    pub(crate) fn ensure_kv_cache_for_shapes<'a>(
        &self,
        cache: &'a mut Option<ops::kv_cache::KVCache>,
        shapes: &[(usize, usize)],
        max_seq: usize,
    ) -> &'a mut ops::kv_cache::KVCache {
        let needs_rebuild = cache
            .as_ref()
            .is_none_or(|kv| kv.has_shape_mismatch(shapes));

        if needs_rebuild {
            *cache = Some(self.create_kv_cache_per_layer(shapes, max_seq));
        }

        let kv = cache.as_mut().expect("KV cache initialized above");
        kv.grow_to_shapes(&self.bufs, shapes, max_seq);
        kv
    }

    /// Decode one token through all layers with KV cache.
    ///
    /// **Single command buffer**, one encoder per layer, no explicit barriers
    /// (Apple Silicon serialises compute dispatches within an encoder).
    ///
    /// Per-layer pipeline (~10 dispatches):
    ///   1. Input norm
    ///   2. Fused QKV projection (Q4_K or Q4_KF)
    ///   3. Batched RoPE (all Q heads), batched RoPE (all K heads)
    ///   4. Batched V-norm (optional, Gemma 4)
    ///   5. KV cache append + KV attend
    ///   6. O projection
    ///   7. Residual + norm (f32 for Q4_K/Q4_KF, +Q8 for Q4_0)
    ///   8. FFN: fused gate+up (Q4_K) or separate gate/up (Q4_KF) + GEGLU + down
    ///   9. Post-FFN residual + optional layer scalar
    ///
    /// Format-aware FFN routing:
    ///   - Q4_KF: llama.cpp-exact kernel (q4kf_proj) with register-cached input
    ///   - Q4_K:  fused gate+up kernel + q4k_matvec (uint4, 8 rows/TG, nr0=2)
    ///   - Q4_0:  legacy Q8-input path
    ///
    /// Decode one token with an optional MoE override function.
    ///
    /// When `moe_fn` is `Some`, it is called instead of `cpu_moe_forward` for
    /// every MoE layer.  Signature: `moe_fn(layer_idx, h_post_attn) -> Vec<f32>`.
    /// The returned vec must have length == `hidden`.  Pass `None` for the
    /// normal local-expert path.
    ///
    /// When `moe_collect_fn` is also `Some` the per-layer pipeline switches to
    /// the split-encoder layout: attention is committed and waited, `moe_fn`
    /// is invoked as a non-blocking *fire* (return value discarded), dense
    /// FFN + post-FFN residual are encoded on a fresh command buffer and
    /// committed without waiting, then `moe_collect_fn(layer)` is called to
    /// retrieve the expert output — letting the remote round trip overlap
    /// with the dense-FFN GPU work.
    #[allow(clippy::too_many_arguments, clippy::type_complexity)]
    pub fn decode_token_with_moe_fn(
        &self,
        kv_cache: &mut ops::kv_cache::KVCache,
        layers: &[larql_compute::FullPipelineLayer],
        x: &[f32],
        hidden: usize,
        inter: usize,
        q_dim: usize,
        kv_dim: usize,
        _num_q_heads: usize,
        _num_kv_heads: usize,
        _head_dim: usize,
        _rope_base: f32,
        moe_fn: Option<&mut dyn FnMut(usize, &[f32]) -> Vec<f32>>,
    ) -> Vec<f32> {
        // Backwards-compat wrapper: forward to the split-aware impl with no
        // collect callback.
        self.decode_token_with_moe_split_fn(
            kv_cache,
            layers,
            x,
            hidden,
            inter,
            q_dim,
            kv_dim,
            _num_q_heads,
            _num_kv_heads,
            _head_dim,
            _rope_base,
            moe_fn,
            None,
        )
    }

    /// Split fire / collect variant of `decode_token_with_moe_fn`.  See the
    /// trait method `DecodeBackend::decode_token_with_moe_split` for the
    /// motivating use case (within-layer GPU/MoE overlap).
    #[allow(clippy::too_many_arguments, clippy::type_complexity)]
    pub fn decode_token_with_moe_split_fn(
        &self,
        kv_cache: &mut ops::kv_cache::KVCache,
        layers: &[larql_compute::FullPipelineLayer],
        x: &[f32],
        hidden: usize,
        inter: usize,
        q_dim: usize,
        kv_dim: usize,
        _num_q_heads: usize,
        _num_kv_heads: usize,
        _head_dim: usize,
        _rope_base: f32,
        mut moe_fn: Option<&mut dyn FnMut(usize, &[f32]) -> Vec<f32>>,
        mut moe_collect_fn: Option<&mut dyn FnMut(usize) -> Vec<f32>>,
    ) -> Vec<f32> {
        let _gpu_time_token_start = std::time::Instant::now();
        let mut gpu_time = gpu_timing::TokenGpuTime::default();

        // Residual dump (env-gated) for HF-reference diffs. Active only when
        // `LARQL_DUMP_RESIDUALS=<path>` is set.
        let mut residual_dump = diag::ResidualDump::from_env();

        // Input RMS debug (first 3 calls, env-gated).
        static CALL_COUNT: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);
        let call_n = CALL_COUNT.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        diag::log_decode_entry(call_n, x, hidden, inter, layers);

        // Per-layer weight-buffer caches + per-stage scratch + ping-pong
        // h-buffers. See `setup.rs` for the full inventory; previously
        // ~135 lines inline at the top of this method.
        let scratch =
            setup::DecodeScratch::new(&self.bufs, layers, x, hidden, inter, q_dim, kv_dim);
        let setup::DecodeScratch {
            wq_bufs,
            wk_bufs,
            wv_bufs,
            wo_bufs,
            wq_scale_bufs,
            wk_scale_bufs,
            wv_scale_bufs,
            wo_scale_bufs,
            gate_bufs,
            up_bufs,
            down_bufs,
            input_norm_bufs,
            post_attn_norm_bufs,
            h_init,
            h_a,
            h_b,
            q_out,
            k_out,
            v_out,
            norm_f32_buf,
            attn_out_buf,
            o_out_buf,
            h_post_attn,
            ffn_norm_out,
            ffn_q8,
            ffn_q8s,
            up_out,
            act_buf,
            down_out,
            gate_out_scratch,
            normed_scratch,
            o_q8_scratch,
            o_q8s_scratch,
            scaled_scratch,
            inter_padded,
            num_layers,
            has_moe,
            scratch_clones,
        } = scratch;
        // Return scratch buffers to the pool when this decode step exits.
        let _scratch_guard = {
            let mut g = super::buffers::ScratchGuard::new(&self.bufs);
            for buf in scratch_clones {
                g.track(&buf);
            }
            g
        };
        let mut h_buf = &h_init;
        // Per-Layer Embeddings precomputed table (Gemma 4 E2B): snapshot
        // once per token so the per-layer loop can read it without
        // re-locking the mutex on every iteration. `None` for non-PLE archs.
        let ple_inputs = self.ple_inputs_snapshot();
        // Split mode: when a fire+collect callback pair is present, defer
        // FFN encoding for MoE layers until *after* the remote MoE call has
        // been fired, so dense FFN runs on the GPU in parallel with the
        // network round trip.  Falls back to single-encoder per layer when
        // `moe_collect_fn` is `None` (existing local-MoE / unary HTTP path).
        let split_mode = moe_fn.is_some() && moe_collect_fn.is_some();
        let mut cmd = self.queue.new_command_buffer().to_owned();
        let mut enc = cmd.new_compute_command_encoder().to_owned();
        let mut encoder_ended = false;

        // Diagnostic: run only up to (and including) the specified layer,
        // then dump intermediates and exit. Pinpoints which sub-stage in
        // which layer first produces NaN on real-vindex decode.
        let diag_stop_layer: Option<usize> =
            larql_compute::options::env_usize(larql_compute::options::ENV_DECODE_DIAG_LAYER);

        for l in 0..num_layers {
            let layer = &layers[l];

            // Snapshot the layer input for HF-reference diff. Must be taken
            // before any compute since `h_buf` = layer-N input at this point
            // (it's the previous layer's `new_h`, or the embedding for L0).
            // GPU buffers are committed + waited at the end of each MoE
            // iteration so the read returns consistent data.
            let layer_in_snapshot: Option<Vec<f32>> = if residual_dump.is_enabled() {
                Some(super::buffers::read_buffer_f32(h_buf, hidden))
            } else {
                None
            };
            let dump_l0_dir = if l == 0 {
                larql_compute::options::env_value(larql_compute::options::ENV_DUMP_L0)
            } else {
                None
            };

            let norm_offset = layer.norm_offset;
            let eps = layer.eps;
            let layer_head_dim = layer.head_dim;
            let layer_num_q_heads = layer.num_q_heads;
            let layer_num_kv_heads = layer.num_kv_heads;
            let uses_kquant = layer.wq.format.is_kquant_family();
            let layer_q_dim = layer_num_q_heads * layer_head_dim;
            let layer_kv_dim = layer_num_kv_heads * layer_head_dim;

            // D-RMS-FUSE Phase 1: skip the input rms_norm dispatch when
            // `LARQL_FUSED_PRELAYER_NORM=1` AND we're not the first layer
            // AND the previous layer's `encode_post_ffn_residual` wrote
            // the pre-normalized data into `norm_f32_buf` via
            // `residual_norm_store` (only on the non-post-norms path).
            let prelayer_norm_active =
                l > 0 && !layers[l - 1].has_post_norms && self.decode_flags.fused_prelayer_norm;

            // ── Step 1: Input norm + Q/K/V projection ──
            // Format-aware: Q4_K family routes through fused QKV
            // shaders (uniform / mixed Q4K+Q6K-V / per-projection
            // fallback); Q4_0 routes through fused norm+Q8 then
            // Q8 QKV. Implementation lives in `encode_qkv.rs`.
            self.encode_input_norm_and_qkv(
                &enc,
                layer,
                encode_qkv::QkvBufs {
                    h_in: h_buf,
                    input_norm: &input_norm_bufs[l],
                    input_norm_bias: layer.input_norm_bias,
                    wq: &wq_bufs[l],
                    wk: &wk_bufs[l],
                    wv: &wv_bufs[l],
                    wq_scales: &wq_scale_bufs[l],
                    wk_scales: &wk_scale_bufs[l],
                    wv_scales: &wv_scale_bufs[l],
                    norm_out: &norm_f32_buf,
                    q_out: &q_out,
                    k_out: &k_out,
                    v_out: &v_out,
                    ffn_q8: &ffn_q8,
                    ffn_q8s: &ffn_q8s,
                },
                encode_qkv::QkvDims {
                    hidden,
                    layer_q_dim,
                    layer_kv_dim,
                    eps,
                    norm_offset,
                },
                uses_kquant,
                prelayer_norm_active,
            );

            // ── Steps 1.5–5: attention block ──
            //
            // QK-norm + RoPE (with optional `attn_fused` and `qk_norm_rope_fused`
            // variants), V-norm (Gemma 4), KV append + attend, O projection,
            // post-attn residual + ffn-input norm. See `encode_attn.rs` for the
            // full path map; previously ~470 lines inline here.
            self.encode_attention_block(
                &enc,
                layer,
                kv_cache,
                l,
                encode_attn::AttnBufs {
                    h_buf,
                    q_out: &q_out,
                    k_out: &k_out,
                    v_out: &v_out,
                    attn_out_buf: &attn_out_buf,
                    o_out_buf: &o_out_buf,
                    ffn_norm_out: &ffn_norm_out,
                    h_post_attn: &h_post_attn,
                    o_q8_scratch: &o_q8_scratch,
                    o_q8s_scratch: &o_q8s_scratch,
                    ffn_q8: &ffn_q8,
                    ffn_q8s: &ffn_q8s,
                    normed_scratch: &normed_scratch,
                    wo: &wo_bufs[l],
                    wo_scales: &wo_scale_bufs[l],
                    post_attn_norm: &post_attn_norm_bufs[l],
                },
                encode_attn::AttnDims {
                    hidden,
                    layer_q_dim,
                    uses_kquant,
                    ffn_uses_kquant: layer.gate.format.is_kquant_family(),
                },
            );
            let new_h = if l % 2 == 0 { &h_a } else { &h_b };
            let ffn_uses_kquant = layer.gate.format.is_kquant_family();

            // ── Steps 6-7: FFN + post-FFN residual ──
            //
            // Skip when in split mode AND this layer has MoE — they will be
            // re-encoded on a fresh command buffer inside the MoE block so
            // they can run in parallel with the remote MoE round trip.  For
            // non-MoE layers (or non-split mode) we encode them inline as
            // before.
            //
            // Also skip when ffn_is_remote: the entire FFN for this layer
            // will be provided by the remote server via moe_fn, so there
            // is no local FFN work to encode on the GPU.
            let defer_ffn_for_split = split_mode && layer.moe.is_some();

            // Stage-timing boundary: when LARQL_PROFILE_SPLIT=1 (or the legacy
            // alias LARQL_DECODE_STAGE_TIMING=1), close the encoder here so
            // attention CB time can be recorded separately from FFN CB time.
            // Adds ~1 commit/wait per layer (~30-50µs each on M3 Max) —
            // measurement-only mode, off by default. Skipped on MoE-deferred
            // layers because their interleave block handles its own commits.
            let stage_timing_split = !defer_ffn_for_split && profile::split_profile_requested();
            if stage_timing_split {
                enc.end_encoding();
                cmd.commit();
                cmd.wait_until_completed();
                gpu_time.record_stage(&cmd, gpu_timing::DecodeStage::Attention);
                cmd = self.queue.new_command_buffer().to_owned();
                enc = cmd.new_compute_command_encoder().to_owned();
                encoder_ended = false;
            }

            if !defer_ffn_for_split && !layer.ffn_is_remote {
                let ffn_bufs = encode_ffn::FfnBufs {
                    gate_w: &gate_bufs[l],
                    up_w: &up_bufs[l],
                    down_w: &down_bufs[l],
                    ffn_norm_out: &ffn_norm_out,
                    ffn_q8: &ffn_q8,
                    ffn_q8s: &ffn_q8s,
                    gate_out_scratch: &gate_out_scratch,
                    up_out: &up_out,
                    act_buf: &act_buf,
                    down_out: &down_out,
                };
                let ffn_dims = encode_ffn::FfnDims {
                    hidden,
                    inter,
                    inter_padded,
                };
                let use_fused_post_ffn = self.decode_flags.fused_post_ffn_norm;
                let post_ffn_bufs = encode_post_ffn::PostFfnBufs {
                    down_out: &down_out,
                    h_post_attn: &h_post_attn,
                    new_h,
                    normed_scratch: &normed_scratch,
                };

                // D-RMS-FUSE Phase 1: when env var on AND non-Gemma path
                // (no post_norms) AND there's a next layer, hand the next
                // layer's input_norm weight + the shared norm_f32_buf to
                // `encode_post_ffn_residual` so it can fuse the residual
                // add with the next layer's input rms_norm in one
                // `residual_norm_store` dispatch. Saves 1 dispatch/layer.
                let prelayer_fusion =
                    if !layer.has_post_norms && self.decode_flags.fused_prelayer_norm {
                        layers.get(l + 1).map(|next| {
                            super::decode::encode_post_ffn::PreLayerNormFusion {
                                next_input_norm: next.input_norm,
                                next_norm_out: &norm_f32_buf,
                            }
                        })
                    } else {
                        None
                    };

                if stage_timing_split && !has_moe {
                    // Fine split: gate+up in one CB, act+down+residual in another.
                    // Step 6a: gate+up
                    self.encode_ffn_gate_up_phase(
                        &enc,
                        layer,
                        &ffn_bufs,
                        ffn_dims,
                        ffn_uses_kquant,
                    );
                    enc.end_encoding();
                    cmd.commit();
                    cmd.wait_until_completed();
                    gpu_time.record_stage(&cmd, gpu_timing::DecodeStage::GateUp);
                    cmd = self.queue.new_command_buffer().to_owned();
                    enc = cmd.new_compute_command_encoder().to_owned();
                    // Step 6b + 7: activation+down + post-FFN residual
                    self.encode_ffn_down_phase(&enc, layer, &ffn_bufs, ffn_dims, ffn_uses_kquant);
                    self.encode_post_ffn_residual(
                        &enc,
                        layer,
                        post_ffn_bufs,
                        hidden,
                        use_fused_post_ffn,
                        prelayer_fusion.as_ref(),
                    );
                    enc.end_encoding();
                    cmd.commit();
                    cmd.wait_until_completed();
                    gpu_time.record_stage(&cmd, gpu_timing::DecodeStage::Down);
                    cmd = self.queue.new_command_buffer().to_owned();
                    enc = cmd.new_compute_command_encoder().to_owned();
                    encoder_ended = false;
                } else {
                    // Production path: whole FFN in one encoder block.
                    self.encode_ffn_step(&enc, layer, ffn_bufs, ffn_dims, ffn_uses_kquant);
                    self.encode_post_ffn_residual(
                        &enc,
                        layer,
                        post_ffn_bufs,
                        hidden,
                        use_fused_post_ffn,
                        prelayer_fusion.as_ref(),
                    );
                }

                // ── Step 8: Per-Layer Embeddings (Gemma 4 E2B) ──
                // Mirrors `crates/larql-inference/src/forward/ple.rs::apply_per_layer_embedding`.
                // Activates only when (a) the layer has the three PLE
                // weights wired and (b) the inference layer uploaded a
                // precomputed per-layer-input table via
                // `MetalBackend::prepare_ple_inputs` for this generation.
                if let Some(pli) = ple_inputs.as_ref() {
                    if layer.ple_spec().is_some() {
                        // Reuse two scratches that are dead after the
                        // post-FFN residual completes:
                        //   - `gate_out_scratch` (`inter` f32) holds the
                        //     `[ple_dim]` gate (ple_dim ≪ inter);
                        //   - `down_out` (`hidden` f32) holds the projection
                        //     output (`[hidden]`).
                        // Both buffers' previous data is consumed by
                        // `encode_post_ffn_residual` above.
                        self.encode_per_layer_embed(
                            &enc,
                            layer,
                            encode_ple::PleBufs {
                                h: new_h,
                                per_layer_input: &pli.buffer,
                                per_layer_input_offset: pli.row_offset_bytes(0, l),
                                gate_scratch: &gate_out_scratch,
                                contrib_scratch: &down_out,
                            },
                            hidden,
                            pli.ple_dim,
                        );
                    }
                }
            }

            h_buf = new_h;
            let _ = &scaled_scratch; // keep binding alive; no longer needed

            // Per-layer NaN diagnostic (LARQL_DEBUG_NAN_LAYERS=1).
            // Forces a commit+wait per layer — expensive, debug-only.
            if larql_compute::options::env_flag(larql_compute::options::ENV_DEBUG_NAN_LAYERS) {
                if !encoder_ended {
                    enc.end_encoding();
                }
                cmd.commit();
                cmd.wait_until_completed();
                let h = super::buffers::read_buffer_f32(h_buf, hidden);
                let nans = h.iter().filter(|v| v.is_nan()).count();
                eprintln!(
                    "[nan-debug] layer {l}: {nans}/{hidden} NaN (head_dim={} kv_heads={})",
                    layers[l].head_dim, layers[l].num_kv_heads
                );
                cmd = self.queue.new_command_buffer().to_owned();
                enc = cmd.new_compute_command_encoder().to_owned();
                encoder_ended = false;
            }

            // CPU MoE interleave for hybrid MoE models (e.g. Gemma 4 26B A4B).
            // After the GPU dense-FFN pass, flush the encoder, run the expert block
            // on CPU (direct shared-memory access), then restart for the next layer.
            // layer_scalar is applied AFTER MoE so it scales the combined output
            // (dense + MoE). Applying it before would leave the MoE contribution unscaled.
            if has_moe {
                self.handle_moe_interleave(
                    layer,
                    moe_interleave::MoeInterleaveCtx {
                        layer_idx: l,
                        num_layers,
                        hidden,
                        inter,
                        inter_padded,
                        ffn_uses_kquant,
                        defer_ffn_for_split,
                        stage_timing_split,
                        layer_in_snapshot: layer_in_snapshot.as_deref(),
                        dump_l0_dir: dump_l0_dir.as_deref(),
                    },
                    moe_interleave::MoeInterleaveBufs {
                        gate_w: &gate_bufs[l],
                        up_w: &up_bufs[l],
                        down_w: &down_bufs[l],
                        h_post_attn: &h_post_attn,
                        ffn_norm_out: &ffn_norm_out,
                        ffn_q8: &ffn_q8,
                        ffn_q8s: &ffn_q8s,
                        gate_out_scratch: &gate_out_scratch,
                        up_out: &up_out,
                        act_buf: &act_buf,
                        down_out: &down_out,
                        normed_scratch: &normed_scratch,
                        new_h,
                    },
                    moe_interleave::MoeCommandState {
                        cmd: &mut cmd,
                        enc: &mut enc,
                        encoder_ended: &mut encoder_ended,
                        gpu_time: &mut gpu_time,
                        residual_dump: &mut residual_dump,
                    },
                    &mut moe_fn,
                    &mut moe_collect_fn,
                );
            } else {
                // ── Step 8: Optional layer scalar (non-MoE layers) ──
                // GPU in-place scale on new_h before it becomes the next layer's input.
                if layer.layer_scalar != 0.0 {
                    crate::stages::layer_scalar::encode(
                        &enc,
                        &self.norms.scale_vector_pipeline,
                        new_h,
                        1,
                        hidden,
                        layer.layer_scalar,
                    );
                }
            }

            // Optional per-layer end-of-layer dump for decode-path
            // diagnostics. Flushes the encoder so `new_h` is readable,
            // writes `decode_layer_{LL}.f32`, then restarts the encoder
            // for the next layer. Paired with Metal prefill's
            // `metal_layer_{LL}_h_out.f32` hook so the two paths can be
            // diffed at the same layer boundaries. Gated on an env var to
            // keep normal decode free of flush overhead.
            //
            // When `LARQL_STAGE_DUMP_LAYER` names the current layer, also
            // dump every per-sub-stage scratch buffer
            // (`decode_layer_{LL}_{stage}.f32`). Names match the Metal
            // prefill side (`metal_layer_NN_{stage}.f32`) so the two
            // dump dirs can be diffed file-by-file. The end-of-layer
            // commit above is what makes these reads consistent — the
            // scratch buffers persist across layers, so without the
            // per-layer flush we'd be reading the *last* layer's value.
            if let Some(dir) =
                larql_compute::options::env_value(larql_compute::options::ENV_DECODE_DUMP_LAYERS)
            {
                if !encoder_ended {
                    enc.end_encoding();
                    cmd.commit();
                    cmd.wait_until_completed();
                    encoder_ended = true;
                }
                let hidden_bytes = super::buffers::read_buffer_f32(new_h, hidden);
                let as_bytes: Vec<u8> = hidden_bytes.iter().flat_map(|v| v.to_le_bytes()).collect();
                let path = format!("{dir}/decode_layer_{l:02}.f32");
                if let Err(e) = std::fs::write(&path, &as_bytes) {
                    eprintln!("[decode-dump] failed to write {path}: {e}");
                }

                // Per-stage dump for the layer named by
                // `LARQL_STAGE_DUMP_LAYER` (default 0). Helper lives in
                // `diag.rs`; the bundle of references is the same one
                // the early-exit diag mode uses.
                let stage_layer =
                    larql_compute::options::env_usize(larql_compute::options::ENV_STAGE_DUMP_LAYER)
                        .unwrap_or(0);
                if l == stage_layer {
                    let bufs = diag::LayerDiagBufs {
                        norm_f32_buf: &norm_f32_buf,
                        q_out: &q_out,
                        k_out: &k_out,
                        v_out: &v_out,
                        attn_out_buf: &attn_out_buf,
                        o_out_buf: &o_out_buf,
                        h_post_attn: &h_post_attn,
                        ffn_norm_out: &ffn_norm_out,
                        gate_out_scratch: &gate_out_scratch,
                        up_out: &up_out,
                        act_buf: &act_buf,
                        down_out: &down_out,
                        new_h,
                        hidden,
                        inter,
                        layer_q_dim,
                        layer_kv_dim: layer_num_kv_heads * layer_head_dim,
                    };
                    diag::dump_decode_stage_files(&dir, l, &bufs);
                }

                if l + 1 < num_layers {
                    cmd = self.queue.new_command_buffer().to_owned();
                    enc = cmd.new_compute_command_encoder().to_owned();
                    encoder_ended = false;
                }
            }

            // Diagnostic early-exit after layer `l`. Commits what we have,
            // reads the per-sub-stage buffers, and reports NaN counts.
            if diag_stop_layer == Some(l) {
                if !encoder_ended {
                    enc.end_encoding();
                    cmd.commit();
                    cmd.wait_until_completed();
                }
                let bufs = diag::LayerDiagBufs {
                    norm_f32_buf: &norm_f32_buf,
                    q_out: &q_out,
                    k_out: &k_out,
                    v_out: &v_out,
                    attn_out_buf: &attn_out_buf,
                    o_out_buf: &o_out_buf,
                    h_post_attn: &h_post_attn,
                    ffn_norm_out: &ffn_norm_out,
                    gate_out_scratch: &gate_out_scratch,
                    up_out: &up_out,
                    act_buf: &act_buf,
                    down_out: &down_out,
                    new_h,
                    hidden,
                    inter,
                    layer_q_dim,
                    layer_kv_dim: layer_num_kv_heads * layer_head_dim,
                };
                diag::dump_layer_buffers(l, &bufs);
                return super::buffers::read_buffer_f32(new_h, hidden);
            }
        }

        if !encoder_ended {
            enc.end_encoding();
            cmd.commit();
            cmd.wait_until_completed();
            gpu_time.record(&cmd);
        }

        // Diagnostic: dump per-call h after each layer, gated on
        // LARQL_PERCALL_LAYER_DUMP_DIR. Filename includes the call index
        // and layer index. Useful for pinpointing which call+layer
        // first diverges from CPU.
        if let Ok(dir) = std::env::var("LARQL_PERCALL_LAYER_DUMP_DIR") {
            for (idx, kv_layer) in kv_cache.layers.iter().enumerate() {
                if kv_layer.current_len == 0 {
                    continue;
                }
                let total = kv_layer.current_len * kv_layer.num_kv_heads * kv_layer.head_dim;
                let k_floats = super::buffers::read_buffer_f32(&kv_layer.k_cache, total);
                let k_bytes: Vec<u8> = k_floats.iter().flat_map(|v| v.to_le_bytes()).collect();
                let _ = std::fs::write(
                    format!("{dir}/metal_call{call_n:03}_L{idx:02}_K.f32"),
                    &k_bytes,
                );
            }
            // Dump h_buf (layer-output residual fed into next decode).
            let h_floats = super::buffers::read_buffer_f32(h_buf, hidden);
            let h_bytes: Vec<u8> = h_floats.iter().flat_map(|v| v.to_le_bytes()).collect();
            let _ = std::fs::write(format!("{dir}/metal_call{call_n:03}_h_final.f32"), &h_bytes);
            // Dump x (the input embed for this call) for comparison vs CPU
            // batched-prefill embed at the same position.
            let x_bytes: Vec<u8> = x.iter().flat_map(|v| v.to_le_bytes()).collect();
            let _ = std::fs::write(format!("{dir}/metal_call{call_n:03}_x_input.f32"), &x_bytes);
        }
        // Diagnostic: dump source-layer K/V caches when LARQL_KV_CACHE_DUMP_DIR
        // is set. Fires at the end of every decode_token call (file gets
        // overwritten — last call wins). Useful for byte-level comparison
        // against CPU's k_rope/v_final output for the same prompt.
        if let Ok(dir) = std::env::var("LARQL_KV_CACHE_DUMP_DIR") {
            for (idx, kv_layer) in kv_cache.layers.iter().enumerate() {
                if kv_layer.current_len == 0 {
                    continue;
                }
                let total = kv_layer.current_len * kv_layer.num_kv_heads * kv_layer.head_dim;
                let k_floats = super::buffers::read_buffer_f32(&kv_layer.k_cache, total);
                let v_floats = super::buffers::read_buffer_f32(&kv_layer.v_cache, total);
                let k_bytes: Vec<u8> = k_floats.iter().flat_map(|v| v.to_le_bytes()).collect();
                let v_bytes: Vec<u8> = v_floats.iter().flat_map(|v| v.to_le_bytes()).collect();
                let _ = std::fs::write(format!("{dir}/metal_L{idx:02}_K_cache.f32"), &k_bytes);
                let _ = std::fs::write(format!("{dir}/metal_L{idx:02}_V_cache.f32"), &v_bytes);
            }
        }

        let result = super::buffers::read_buffer_f32(h_buf, hidden);

        // Print GPU vs CPU split when LARQL_GPU_TIMING=1. Wall covers the
        // entire decode_token_with_moe_fn call including buffer reads;
        // gpu is the sum of MTLCommandBuffer.gpuStartTime/gpuEndTime
        // windows. Delta is CPU encoding + readback overhead.
        let wall_ms = _gpu_time_token_start.elapsed().as_secs_f64() * 1000.0;
        gpu_time.print_if_enabled(wall_ms);

        // When LARQL_PROFILE_SPLIT=1, store the per-stage breakdown for
        // `decode_token_split_profile` to read back. attn vs full-FFN
        // granularity (gate_up_ms carries the whole FFN block; down_ms
        // reserved for the next-finer split — see profile.rs doc-comment).
        if profile::split_profile_requested() {
            profile::store_last_split_timings(profile::ProfileTimings {
                attn_ms: gpu_time.attn_ms,
                gate_up_ms: gpu_time.gate_up_ms,
                down_ms: gpu_time.down_ms,
            });
        }

        result
    }

    /// Local-expert path — delegates to `decode_token_with_moe_fn` with no hook.
    #[allow(clippy::too_many_arguments)]
    pub fn decode_token(
        &self,
        kv_cache: &mut ops::kv_cache::KVCache,
        layers: &[larql_compute::FullPipelineLayer],
        x: &[f32],
        hidden: usize,
        inter: usize,
        q_dim: usize,
        kv_dim: usize,
        num_q_heads: usize,
        num_kv_heads: usize,
        head_dim: usize,
        rope_base: f32,
    ) -> Vec<f32> {
        self.decode_token_with_moe_fn(
            kv_cache,
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
            None,
        )
    }
}
