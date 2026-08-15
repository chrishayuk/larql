// Clippy's `needless_option_as_deref` lint flags `state_dump.as_deref_mut()`
// inside the per-layer loops here. The lint is wrong in this context: the
// loop reuses `state_dump` across iterations, and pattern-matching the
// `Option<&mut DecodeStateDump>` directly would move it on the first
// iteration. `as_deref_mut()` re-borrows each iteration without moving.
#![allow(clippy::needless_option_as_deref)]

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

pub(crate) use moe_interleave::InlineMoeCtx;
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

    /// Per-layer capacities implied by each layer's own attention window.
    pub(crate) fn kv_capacities_for_layers(
        layers: &[larql_compute::FullPipelineLayer<'_>],
        default_capacity: usize,
    ) -> Vec<usize> {
        layers
            .iter()
            .map(|layer| {
                larql_compute::pipeline_layer::kv_capacity_for_window(
                    layer.sliding_window,
                    default_capacity,
                )
            })
            .collect()
    }

    /// Ensure a cache sized by each layer's *own* capacity.
    ///
    /// This must not route through [`Self::ensure_kv_cache_for_shapes`]:
    /// that grows every layer to a single `max_seq`, which would
    /// immediately re-inflate a sliding layer that was deliberately
    /// allocated at `SLACK * W` and undo the saving on the first decode
    /// step.
    pub(crate) fn ensure_kv_cache_for_layers<'a>(
        &self,
        cache: &'a mut Option<ops::kv_cache::KVCache>,
        layers: &[larql_compute::FullPipelineLayer<'_>],
        max_seq: usize,
    ) -> &'a mut ops::kv_cache::KVCache {
        let shapes = Self::kv_shapes_for_layers(layers);
        let capacities = Self::kv_capacities_for_layers(layers, max_seq);

        let needs_rebuild = cache
            .as_ref()
            .is_none_or(|kv| kv.has_shape_mismatch(&shapes));
        if needs_rebuild {
            *cache = Some(ops::kv_cache::KVCache::new_per_layer_with_capacities(
                &self.bufs,
                &shapes,
                &capacities,
                max_seq,
            ));
        }
        let kv = cache.as_mut().expect("KV cache initialized above");
        kv.grow_to_capacities(&self.bufs, &shapes, &capacities, max_seq);
        kv
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
        // collect callback and no state dump.
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
            None,
            larql_compute::StateDumpMask::Full,
            None,
        )
    }

    /// Decode one token AND capture per-layer state (W1-GPU step 2).
    ///
    /// Same compute path as `decode_token_with_moe_fn` (no MoE; the
    /// per-layer engines that consume state — markov_residual, codec,
    /// turbo_quant — target dense-FFN architectures). On exit, `state`
    /// is populated with per-layer `h_in` (pre-attention residual),
    /// `k_new` and `v_new` (newly-projected K/V row). Implementation
    /// forces a commit + CPU readback at the end of each layer so the
    /// scratch K/V buffers can be sampled before the next layer
    /// overwrites them; the per-layer-commit overhead (~50µs each ×
    /// num_layers) is the dominant cost vs the fully-fused
    /// single-commit path.
    #[allow(clippy::too_many_arguments)]
    pub fn decode_token_with_state_dump_fn(
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
        state: &mut larql_compute::DecodeStateDump,
    ) -> Vec<f32> {
        self.decode_token_with_state_dump_masked_fn(
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
            state,
            larql_compute::StateDumpMask::Full,
        )
    }

    /// Mask-aware variant of [`Self::decode_token_with_state_dump_fn`].
    /// W10 Phase B: engines that treat K/V as derivative can pass
    /// [`larql_compute::StateDumpMask::HOnly`] to skip the K/V staging +
    /// readback.
    #[allow(clippy::too_many_arguments)]
    pub fn decode_token_with_state_dump_masked_fn(
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
        state: &mut larql_compute::DecodeStateDump,
        mask: larql_compute::StateDumpMask,
    ) -> Vec<f32> {
        self.decode_token_with_moe_split_fn(
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
            None,
            Some(state),
            mask,
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
        mut state_dump: Option<&mut larql_compute::DecodeStateDump>,
        state_dump_mask: larql_compute::StateDumpMask,
        inline_moe: Option<&moe_interleave::InlineMoeCtx<'_>>,
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
        // BW10: byte counters run unconditionally at the bind sites; this
        // scope just brackets one token's window so the delta is this
        // token's traffic.
        let movement_scope = larql_compute::movement_ledger::TokenScope::open();
        // The execution seam addresses tokens by (phase, step). Advancing
        // the index HERE, at the same boundary that opens the window
        // above, is what makes "the policy skipped at step 7" and "token
        // 7's ledger entry" the same token by construction rather than by
        // two loops agreeing. It buckets by phase, so a prefill position
        // never consumes a decode index — the defect the phase scope was
        // added to fix.
        larql_compute::exec_policy::step::advance();
        // Host segments must be per-token for the ledger to divide them
        // correctly; `print_if_enabled` clears on read but bails out early
        // when its env flags are off, so without this they accumulate.
        gpu_timing::reset_host_segments();

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
                cmd.wait_until_completed();
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
                    cmd.wait_until_completed();
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
                    cmd.wait_until_completed();
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

        // BW10: close this token's movement window and pair it with the
        // time terms already measured above. Peeks the host segments —
        // `print_if_enabled` below takes them, and whichever runs first
        // must not blind the other.
        {
            let seg = gpu_timing::peek_host_segments();
            // Every host-segment site lives on the MoE interleave path, and
            // S2 deletes the per-layer wait outright — so `wait_ms == 0`
            // there is a REAL zero, while a token that never touched the
            // path has no sampler at all. Distinguish them by whether any
            // segment moved, not by the wait alone.
            let host_sampled = seg.wait_ms > 0.0
                || seg.commit_ms > 0.0
                || seg.encode_ms > 0.0
                || seg.route_ms > 0.0;
            let record = movement_scope.close(larql_compute::movement_ledger::TimeAttribution {
                wall_ms,
                gpu_busy_ms: gpu_time.total_gpu_ms,
                gpu_bubble_ms: gpu_time.gpu_idle_between_ms,
                // No storage sampler on this path yet; a resident run's
                // real value IS zero, but only a sampler may say so.
                io_wait_ms: 0.0,
                io_wait_reported: false,
                host_wait_ms: seg.wait_ms,
                host_wait_reported: host_sampled,
            });
            larql_compute::movement_ledger::session::record_token(record);
        }

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

#[cfg(test)]
mod tests;
