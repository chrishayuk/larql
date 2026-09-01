//! The decode token loop: one token through every layer, in one command
//! buffer wherever the layer's shape allows it.
//!
//! This is the implementation every entry point in `entry.rs` reaches, and
//! the place the per-token command-buffer count is decided. The stages a
//! layer encodes live in the `encode_*` modules beside it; what stays here
//! is the sequencing between them, the MoE fire/collect split, and the
//! single commit + wait at the bottom that TOKEN-B1 rung 2's fused head
//! rides (see `head.rs`).

use super::*;

impl MetalBackend {
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
        mut state_dump: Option<&mut larql_compute::DecodeStateDump>,
        state_dump_mask: larql_compute::StateDumpMask,
        inline_moe: Option<&moe_interleave::InlineMoeCtx<'_>>,
        head: Option<HeadRequest<'_, '_>>,
    ) -> Vec<f32> {
        // Refuse unroutable FFN formats BEFORE any command buffer or
        // encoder exists: a panic that unwinds past a live Metal
        // encoder trips the ObjC "released without endEncoding"
        // assertion and turns a clean refusal into a process-killing
        // SIGTRAP (the failure mode that hid #229 behind an earlier
        // test's abort). `encode_ffn_step` re-checks as defence in
        // depth for callers that bypass this entry point.
        for layer in layers {
            // A fully-remote FFN never runs locally; its dense weight
            // slots may be placeholders and are not validated.
            if !layer.ffn_is_remote {
                encode_ffn::validate_ffn_formats(layer);
            }
        }
        // W10 Phase B/C: capture flags. `dump_kv` controls the K/V
        // staging + readback (skipped under HOnly + None — Metal's own
        // kv cache still receives the K/V as a side effect for
        // attention). `dump_h` controls the h_in staging + readback
        // (skipped under None only — engines using `None` have no
        // CPU-side use for the residual stream, e.g.
        // MarkovResidualEngine with no window).
        let dump_kv = matches!(state_dump_mask, larql_compute::StateDumpMask::Full);
        let dump_h = !matches!(state_dump_mask, larql_compute::StateDumpMask::None);
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

        // W1-GPU step 7 (blit-encoder fusion). When `state_dump` is active,
        // pre-allocate per-layer staging buffers so the layer loop can
        // **blit** k_out / v_out / h_buf into them instead of forcing a
        // per-layer commit+wait+CPU-read. Reads run once after the final
        // commit. Saves ~1.7 ms / token (50 µs × num_layers) on M3 Max.
        let staging_bufs: Option<(Vec<metal::Buffer>, Vec<metal::Buffer>, Vec<metal::Buffer>)> =
            if state_dump.is_some() && (dump_kv || dump_h) {
                let mut sk = Vec::with_capacity(if dump_kv { num_layers } else { 0 });
                let mut sv = Vec::with_capacity(if dump_kv { num_layers } else { 0 });
                let mut sh = Vec::with_capacity(if dump_h { num_layers } else { 0 });
                let hidden_bytes = (hidden * 4) as u64;
                for layer in layers.iter() {
                    if dump_kv {
                        let kv_dim_bytes = (layer.num_kv_heads * layer.head_dim * 4) as u64;
                        sk.push(self.bufs.output(kv_dim_bytes));
                        sv.push(self.bufs.output(kv_dim_bytes));
                    }
                    if dump_h {
                        sh.push(self.bufs.output(hidden_bytes));
                    }
                }
                Some((sk, sv, sh))
            } else {
                None
            };
        // Track for recycling after final commit. Separate from the main
        // `_scratch_guard` since these buffers are allocated post-setup.
        let _staging_guard = {
            let mut g = super::buffers::ScratchGuard::new(&self.bufs);
            if let Some((sk, sv, sh)) = staging_bufs.as_ref() {
                for b in sk.iter().chain(sv.iter()).chain(sh.iter()) {
                    g.track(b);
                }
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
            // The only place that knows which layer is executing. The served
            // MoE route boundary reads this; without it a routing trace is
            // refused rather than attributed to a guessed layer.
            let _route_scope = larql_compute::moe_route_observe::LayerScope::new(l);

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

            // W1-GPU step 7 (blit fusion): capture h_in for state dump.
            // - L=0: `x` is on the CPU, push it directly into state_dump.
            // - L>=1: blit `h_buf` (previous layer's output) into the
            //   per-layer h-staging buffer. The blit is encoded into the
            //   same command buffer as the layer compute, so Metal's
            //   command-buffer ordering guarantees it sees the settled
            //   value once committed. Drained into state_dump after the
            //   single final commit at the bottom of the function.
            if dump_h {
                if let Some(s) = state_dump.as_deref_mut() {
                    if l == 0 {
                        s.h_in_per_layer.push(x.to_vec());
                    }
                }
                if let Some((_, _, ref sh)) = staging_bufs {
                    if l > 0 && !sh.is_empty() {
                        if !encoder_ended {
                            enc.end_encoding();
                        }
                        let blit = cmd.new_blit_command_encoder();
                        blit.copy_from_buffer(h_buf, 0, &sh[l], 0, (hidden * 4) as u64);
                        blit.end_encoding();
                        enc = cmd.new_compute_command_encoder().to_owned();
                        encoder_ended = false;
                    }
                }
            }
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
            //
            // When the fully-fused attention kernel will fire, it applies
            // the Q/K/V projection biases itself — the QKV stage must
            // skip its bias dispatches (shared `attn_fused_will_fire`
            // authority; disagreement = biases applied twice).
            let qkv_bias_deferred = self.attn_fused_will_fire(layer, kv_cache, l);
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
                    wq_scales: wq_scale_bufs[l].as_ref(),
                    wk_scales: wk_scale_bufs[l].as_ref(),
                    wv_scales: wv_scale_bufs[l].as_ref(),
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
                prelayer_norm_active,
                qkv_bias_deferred,
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
                    wo_scales: wo_scale_bufs[l].as_ref(),
                    post_attn_norm: &post_attn_norm_bufs[l],
                },
                encode_attn::AttnDims {
                    hidden,
                    layer_q_dim,
                    ffn_uses_kquant: layer.gate.format().is_kquant_family(),
                },
            );
            let new_h = if l % 2 == 0 { &h_a } else { &h_b };

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

            // Pure-MoE layers extract no dense FFN weights — encoding the
            // dense branch would run the kernels over empty slices and
            // poison `new_h` with garbage that the expert add can't
            // recover. Their FFN is the expert block alone; the MoE
            // interleave below writes `new_h = h_post_attn + moe_out`
            // directly (same combine as the remote-FFN arm).
            let layer_runs_dense_ffn = layer.has_dense_ffn() || layer.moe.is_none();

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
                crate::cb_status::wait_or_abort(
                    &cmd,
                    "crates/larql-compute-metal/src/decode/token.rs:359",
                );
                gpu_time.record_stage(&cmd, gpu_timing::DecodeStage::Attention);
                cmd = self.queue.new_command_buffer().to_owned();
                enc = cmd.new_compute_command_encoder().to_owned();
                encoder_ended = false;
            }

            if !defer_ffn_for_split && !layer.ffn_is_remote && layer_runs_dense_ffn {
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
                    self.encode_ffn_gate_up_phase(&enc, layer, &ffn_bufs, ffn_dims);
                    enc.end_encoding();
                    cmd.commit();
                    crate::cb_status::wait_or_abort(
                        &cmd,
                        "crates/larql-compute-metal/src/decode/token.rs:416",
                    );
                    gpu_time.record_stage(&cmd, gpu_timing::DecodeStage::GateUp);
                    cmd = self.queue.new_command_buffer().to_owned();
                    enc = cmd.new_compute_command_encoder().to_owned();
                    // Step 6b + 7: activation+down + post-FFN residual
                    self.encode_ffn_down_phase(&enc, layer, &ffn_bufs, ffn_dims);
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
                    crate::cb_status::wait_or_abort(
                        &cmd,
                        "crates/larql-compute-metal/src/decode/token.rs:432",
                    );
                    gpu_time.record_stage(&cmd, gpu_timing::DecodeStage::Down);
                    cmd = self.queue.new_command_buffer().to_owned();
                    enc = cmd.new_compute_command_encoder().to_owned();
                    encoder_ended = false;
                } else {
                    // Production path: whole FFN in one encoder block.
                    self.encode_ffn_step(&enc, layer, ffn_bufs, ffn_dims);
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

            // W1-GPU step 7 (blit fusion): capture k_new / v_new for state
            // dump. Instead of committing + waiting to safely read the
            // scratch k_out / v_out buffers before the next layer
            // overwrites them, we blit them into per-layer staging
            // buffers inside the same command buffer. The compute writes
            // to k_out / v_out happen-before the blit reads (Metal
            // command-buffer encode order), so the blit captures the
            // correct values. Drained into state_dump after the single
            // final commit at the bottom of the function.
            if dump_kv {
                if let Some((ref sk, ref sv, _)) = staging_bufs {
                    if !encoder_ended {
                        enc.end_encoding();
                    }
                    let blit = cmd.new_blit_command_encoder();
                    let layer_kv_dim_local = layer.num_kv_heads * layer.head_dim;
                    let bytes = (layer_kv_dim_local * 4) as u64;
                    blit.copy_from_buffer(&k_out, 0, &sk[l], 0, bytes);
                    blit.copy_from_buffer(&v_out, 0, &sv[l], 0, bytes);
                    blit.end_encoding();
                    enc = cmd.new_compute_command_encoder().to_owned();
                    encoder_ended = false;
                }
            }

            // Per-layer NaN diagnostic (LARQL_DEBUG_NAN_LAYERS=1).
            // Forces a commit+wait per layer — expensive, debug-only.
            if larql_compute::options::env_flag(larql_compute::options::ENV_DEBUG_NAN_LAYERS) {
                if !encoder_ended {
                    enc.end_encoding();
                }
                cmd.commit();
                crate::cb_status::wait_or_abort(
                    &cmd,
                    "crates/larql-compute-metal/src/decode/token.rs:518",
                );
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
            //
            // Branch on THIS LAYER being a MoE/remote layer, not on the
            // model-level `has_moe`: with the model-level test, a dense
            // layer of a hybrid-MoE model entered `handle_moe_interleave`
            // (which returns immediately for dense layers) and the
            // `else` arm's layer_scalar was never applied — the same
            // mis-scaling class as the 14x incident recorded in
            // `moe_combine.rs`. Capability audit F9. MoE layers get
            // their scalar inside `moe_combine::apply_outer_combine`.
            let layer_is_moe = layer.moe.is_some() || layer.ffn_is_remote;
            if layer_is_moe {
                self.handle_moe_interleave(
                    layer,
                    moe_interleave::MoeInterleaveCtx {
                        layer_idx: l,
                        num_layers,
                        hidden,
                        inter,
                        inter_padded,
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
                    inline_moe,
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

            // Issue #228: record the residual for DENSE layers too.
            //
            // `record_layer` used to be reachable only from
            // `handle_moe_interleave`, so `LARQL_DUMP_RESIDUALS` on a dense
            // model created the file, wrote its header, printed
            // "[residual-dump] writing to <path>" and then recorded
            // nothing — a well-formed 16-byte file that reads as "no
            // divergence found" rather than "nothing was measured". A
            // diagnostic that reports success while measuring nothing is
            // worse than one that is absent, because it gets trusted.
            //
            // Guarded on `!layer_is_moe`: the MoE arm already records at
            // its own boundary, and recording here as well would double
            // every MoE layer.
            //
            // The MoE arm commits at the end of each iteration, which is
            // what makes its read of `new_h` consistent. The dense arm
            // leaves the encoder open, so it must flush first AND restart
            // the encoder for the next layer — the same shape
            // `ENV_DECODE_DUMP_LAYERS` uses below. Omitting the restart
            // encodes the next layer into a finished encoder, which
            // segfaults rather than failing cleanly.
            if !layer_is_moe {
                if let Some(layer_in) = layer_in_snapshot.as_deref() {
                    if !encoder_ended {
                        enc.end_encoding();
                        cmd.commit();
                        crate::cb_status::wait_or_abort(
                            &cmd,
                            "crates/larql-compute-metal/src/decode/token.rs:627",
                        );
                        encoder_ended = true;
                    }
                    let ha = super::buffers::read_buffer_f32(&h_post_attn, hidden);
                    let lo = super::buffers::read_buffer_f32(new_h, hidden);
                    residual_dump.record_layer(l, layer_in, &ha, &lo);
                    if l + 1 < num_layers {
                        cmd = self.queue.new_command_buffer().to_owned();
                        enc = cmd.new_compute_command_encoder().to_owned();
                        encoder_ended = false;
                    }
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
                    crate::cb_status::wait_or_abort(
                        &cmd,
                        "crates/larql-compute-metal/src/decode/token.rs:663",
                    );
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
                    crate::cb_status::wait_or_abort(
                        &cmd,
                        "crates/larql-compute-metal/src/decode/token.rs:716",
                    );
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

        // TOKEN-B1 rung 2: the LM head rides this command buffer rather
        // than a second one. Encoded while `enc` is still open, so the
        // token pays one commit + wait and the hidden state never crosses
        // the host boundary. A refused plan leaves the encoder untouched
        // and `head_bufs` `None`, and the caller runs the unfused head off
        // the hidden state returned below — the path this is pinned to.
        //
        // Skipped when a diagnostic dump already ended the encoder: that
        // buffer is committed, so there is nothing left to ride.
        let mut head_bufs = None;
        if let Some(ref req) = head {
            if !encoder_ended {
                head_bufs = self.encode_decode_head(&enc, h_buf, hidden, req.plan);
            }
        }

        if !encoder_ended {
            enc.end_encoding();
            cmd.commit();
            crate::cb_status::wait_or_abort(
                &cmd,
                "crates/larql-compute-metal/src/decode/token.rs:761",
            );
            // A failed or ignored buffer returns from the wait just like a
            // finished one; only the status tells them apart. The MoE entry
            // seam turns any failure inside this step into `None`.
            gpu_time.record(&cmd);
        }

        // Reduce after the wait — the partials are only settled now — and
        // return the head's scratch to the pool in the same step.
        if let (Some(req), Some(head_out)) = (head, head_bufs) {
            *req.out = Some(head_out.reduce_and_recycle(&self.bufs));
        }

        // W1-GPU step 7 (blit fusion): drain per-layer staging buffers
        // into state_dump now that the single final commit has settled
        // all blits. `h_in_per_layer[0]` was already pushed inline (CPU
        // copy of `x`); indices 1..num_layers come from the h-staging
        // buffers populated by the blits at the top of each layer
        // body.
        if let Some(s) = state_dump.as_deref_mut() {
            if let Some((sk, sv, sh)) = staging_bufs.as_ref() {
                if dump_h {
                    for (l, _) in layers.iter().enumerate().skip(1) {
                        s.h_in_per_layer
                            .push(super::buffers::read_buffer_f32(&sh[l], hidden));
                    }
                }
                if dump_kv {
                    for (l, layer) in layers.iter().enumerate() {
                        let kv_dim_local = layer.num_kv_heads * layer.head_dim;
                        s.k_new_per_layer
                            .push(super::buffers::read_buffer_f32(&sk[l], kv_dim_local));
                        s.v_new_per_layer
                            .push(super::buffers::read_buffer_f32(&sv[l], kv_dim_local));
                    }
                }
            }
        }

        // Env-gated byte dumps for CPU/Metal bisection. Both are no-ops
        // unless their directory var is set; the bodies live in `diag` so
        // this function stays the token's control flow.
        diag::dump_percall_layers(kv_cache, h_buf, x, hidden, call_n);
        diag::dump_kv_caches(kv_cache);

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
                // The GPU/wall pair travels with the stage split so a caller
                // can report how much of the token was on the GPU at all.
                // The numbers were already measured here; they only ever
                // reached stderr via `print_if_enabled`, so every structured
                // consumer — the bench table, `--json` — attributed the whole
                // wall to "GPU fwd".
                gpu_ms: gpu_time.total_gpu_ms,
                wall_ms,
                cmd_buffers: gpu_time.n_cmd_buffers as u32,
            });
        }

        result
    }
}
