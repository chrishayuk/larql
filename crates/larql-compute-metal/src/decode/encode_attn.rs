//! Per-layer attention block — Steps 1.5 through 5 of the decode loop.
//!
//! Inputs (already populated by `encode_input_norm_and_qkv`):
//! - `q_out`, `k_out`, `v_out`: raw Q/K/V projections (pre-norm, pre-RoPE).
//! - `h_buf`: layer-input residual.
//!
//! Outputs:
//! - `ffn_norm_out`: RMS-normed `h_buf + o_out` (FFN gate/up input).
//! - `h_post_attn`: raw `h_buf + o_out` (post-FFN residual base).
//! - `kv_cache.layers[l].current_len += 1` (the new token's K/V row is appended).
//!
//! Path selection (env-gated, defaults preserve the proven-win May-2026 fusion wave):
//! - `LARQL_FUSED_ATTN=1` (opt-in) — single `attn_fused` kernel covers QK-norm +
//!   RoPE + KV append + attend. Currently regresses on Gemma 3 4B (parallelism
//!   collapse 12 TGs → 8); kept registered for the multi-TG-per-head retry.
//! - `LARQL_FUSED_QK_NORM_ROPE=0` — opt out of the fused QK-norm + RoPE path.
//! - `LARQL_FUSED_KV_APPEND_ATTEND=0` — opt out of the fused KV append + attend.
//! - `LARQL_FUSED_POST_ATTN_NORM=0` — opt out of the triple-fused
//!   `post_attn_norm + residual + ffn_norm + store`.
//!
//! No behaviour change vs. the prior inline code; pure code motion to make the
//! per-stage profiler boundary tractable (next step) and shrink the decode
//! loop body.

use metal::{Buffer, ComputeCommandEncoderRef, MTLSize};

use super::ops;
use crate::ops::kv_cache::{MAX_HEAD_DIM_DOUBLE_SG, MAX_HEAD_DIM_SINGLE_SG};
use crate::MetalBackend;
use larql_compute::FullPipelineLayer;
use larql_models::quant::ggml::LEGACY_BLOCK_ELEMS;

pub(super) struct AttnBufs<'a> {
    /// Layer-input residual (read).
    pub h_buf: &'a Buffer,
    pub q_out: &'a Buffer,
    pub k_out: &'a Buffer,
    pub v_out: &'a Buffer,
    pub attn_out_buf: &'a Buffer,
    pub o_out_buf: &'a Buffer,
    /// FFN gate/up input (written).
    pub ffn_norm_out: &'a Buffer,
    /// Post-FFN residual base (written).
    pub h_post_attn: &'a Buffer,
    /// Scratch for Q8 quantize on the legacy O-proj path.
    pub o_q8_scratch: &'a Buffer,
    pub o_q8s_scratch: &'a Buffer,
    /// Scratch for the Q8-input residual+norm path.
    pub ffn_q8: &'a Buffer,
    pub ffn_q8s: &'a Buffer,
    /// Scratch for the unfused post-attn norm chain.
    pub normed_scratch: &'a Buffer,
    pub wo: &'a Buffer,
    pub wo_scales: &'a Buffer,
    pub post_attn_norm: &'a Buffer,
}

pub(super) struct AttnDims {
    pub hidden: usize,
    pub layer_q_dim: usize,
    pub uses_kquant: bool,
    /// True iff the FFN side will run Q4_K family (selects the fused
    /// `residual_norm_store` path that mirrors the FFN's input dtype).
    pub ffn_uses_kquant: bool,
}

impl MetalBackend {
    /// Encode the per-layer attention block (Steps 1.5–5). See the module
    /// doc-comment for the full input/output contract.
    #[allow(clippy::too_many_arguments)]
    pub(super) fn encode_attention_block(
        &self,
        enc: &ComputeCommandEncoderRef,
        layer: &FullPipelineLayer,
        kv_cache: &mut ops::kv_cache::KVCache,
        layer_idx: usize,
        bufs: AttnBufs<'_>,
        dims: AttnDims,
    ) {
        let AttnDims {
            hidden,
            layer_q_dim,
            uses_kquant,
            ffn_uses_kquant,
        } = dims;
        let hidden_val = hidden as u32;
        // M2: extract per-layer params via the structured views.
        // `attn_spec` carries the attention shape (head_dim, head counts,
        // RoPE, sliding_window, qk_norm flags); `norms` carries the eps
        // and offsets. The compiler optimises these getter methods to
        // plain field reads, so this is a clarity win — the function
        // signature can later narrow to `(AttentionSpec, LayerNorms,
        // ...)` once vindex/inference callers stop passing a flat layer.
        let attn_spec = layer.attention_spec();
        let norms_view = layer.norms();
        let norm_offset = norms_view.norm_offset;
        let eps = norms_view.eps;
        let scale = attn_spec.attn_scale;
        let layer_head_dim = attn_spec.head_dim;
        let layer_num_q_heads = attn_spec.num_q_heads;
        let layer_num_kv_heads = attn_spec.num_kv_heads;
        let layer_rope_base = attn_spec.rope_base;
        let layer_rotary_dim = if attn_spec.rotary_dim > 0 {
            attn_spec.rotary_dim
        } else {
            layer_head_dim
        };
        let window_size = attn_spec.sliding_window as u32;

        // Env flags governing kernel-level fusion. Cached at backend
        // startup (see `metal::flags::DecodeFlags`) so the decode hot
        // path doesn't pay 4× `getenv` per layer.  Defaults preserve
        // the proven-win May-2026 fusion wave; opts-out are diagnostic
        // only.
        let use_fused_attn = self.decode_flags.fused_attn;
        let use_fused_qkn_rope = self.decode_flags.fused_qk_norm_rope;
        // KV-cache sharing (Gemma 4 E2B): later layers reuse K/V from a
        // "source" layer that already ran this token's attention. Shared
        // layers compute their own Q (against their own residual + W_q)
        // but skip K/V append + read from the source's cache. The fused
        // attention paths write the layer's own cache, so they're disabled
        // for shared layers — the unfused encode_kv_attend branch below
        // handles the source-cache binding.
        //
        // Shared layers' position counter is pinned to the source's
        // (`source.current_len` is the count of stored positions
        // including the new token, since source has already finished its
        // block for this token by the time we reach the shared layer).
        let kv_shared_source = layer.kv_shared_source;
        let attend_cache_idx = kv_shared_source.unwrap_or(layer_idx);
        let pos = if let Some(src) = kv_shared_source {
            // Source has already incremented its current_len for this token.
            // Position to RoPE is the same as the source's last-written index.
            (kv_cache.layers[src].current_len.saturating_sub(1)) as u32
        } else {
            kv_cache.layers[layer_idx].current_len as u32
        };
        let t_val = if kv_shared_source.is_some() {
            // Source's current_len already counts this token; t_val is the
            // total positions to attend over (= source.current_len).
            kv_cache.layers[attend_cache_idx].current_len as u32
        } else {
            pos + 1
        };
        let attn_span = ops::kv_cache::attention_span(t_val, window_size);

        // kv_append_attend_fused uses a fixed tg_scores[SHORT_ATTENTION_SPAN]
        // threadgroup array. Spans beyond that overflow it — global-attention
        // layers (window_size=0) grow unboundedly and must fall back to
        // encode_kv_attend, which auto-selects kv_attention_long past the threshold.
        //
        // Additionally, the kernel is designed for head_dim <= MAX_HEAD_DIM_SINGLE_SG
        // (it dispatches exactly head_dim threads per group and assumes head_dim fits
        // in a single simdgroup). Layers with larger head_dim must use the unfused
        // encode_kv_append + encode_kv_attend path which handles arbitrary head_dim.
        let use_fused_kv_aa = attn_span <= ops::kv_cache::SHORT_ATTENTION_SPAN
            && layer_head_dim <= MAX_HEAD_DIM_SINGLE_SG
            && self.decode_flags.fused_kv_append_attend
            // Shared layers don't append to their own cache; the fused
            // append+attend kernel writes its argument cache, so it can't
            // be reused on the source's cache without corrupting the
            // source. Fall through to the unfused attend-only branch.
            && kv_shared_source.is_none();
        let use_fused_post_attn = self.decode_flags.fused_post_attn_norm;

        // Path 1: full attention fusion. Skips both qk_norm_rope dispatch AND
        // kv_append_attend_fused dispatch — handles them in `attn_fused`.
        // M2: `has_v_norm` and `q_norm_enabled` / `k_norm_enabled` come
        // from the structured AttentionSpec; the actual `q_norm_weight`
        // / `k_norm_weight` slices stay as direct layer reads because
        // the spec only carries presence flags, not the weight bytes.
        let did_fused_attn = use_fused_attn
            && layer_head_dim <= MAX_HEAD_DIM_SINGLE_SG
            && attn_span <= ops::kv_cache::SHORT_ATTENTION_SPAN
            && attn_spec.q_norm_enabled
            && attn_spec.k_norm_enabled
            && !attn_spec.has_v_norm
            // attn_fused writes the layer's own K/V cache; shared layers
            // must not, so we fall through to the manual qk_norm + RoPE
            // branch. The K-side work is wasted on shared layers (the
            // result is discarded), but correctness > 35-layer dispatch
            // saving on a non-default opt-in path.
            && kv_shared_source.is_none();

        // ── Step 1.5 + 2: QK-norm + RoPE ──
        if did_fused_attn {
            let cache = &kv_cache.layers[layer_idx];
            let q_w = layer.q_norm_weight.unwrap();
            let k_w = layer.k_norm_weight.unwrap();
            let q_w_buf = self.bufs.get_f32(q_w);
            let k_w_buf = self.bufs.get_f32(k_w);
            let t_val = (cache.current_len + 1) as u32;
            let hd_val = layer_head_dim as u32;
            let nq_val = layer_num_q_heads as u32;
            let nkv_val = cache.num_kv_heads as u32;
            let qk_off = norms_view.qk_norm_offset;
            let rdim = layer_rotary_dim as u32;
            let mut tg_w: u64 = 1;
            while tg_w < layer_head_dim as u64 && tg_w < MAX_HEAD_DIM_SINGLE_SG as u64 {
                tg_w <<= 1;
            }
            enc.set_compute_pipeline_state(&self.attention.attn_fused_pipeline);
            enc.set_buffer(0, Some(bufs.q_out), 0);
            enc.set_buffer(1, Some(bufs.k_out), 0);
            enc.set_buffer(2, Some(bufs.v_out), 0);
            enc.set_buffer(3, Some(&cache.k_cache), 0);
            enc.set_buffer(4, Some(&cache.v_cache), 0);
            enc.set_buffer(5, Some(bufs.attn_out_buf), 0);
            enc.set_buffer(6, Some(&q_w_buf), 0);
            enc.set_buffer(7, Some(&k_w_buf), 0);
            enc.set_bytes(8, 4, &t_val as *const u32 as *const std::ffi::c_void);
            enc.set_bytes(9, 4, &hd_val as *const u32 as *const std::ffi::c_void);
            enc.set_bytes(10, 4, &nq_val as *const u32 as *const std::ffi::c_void);
            enc.set_bytes(11, 4, &nkv_val as *const u32 as *const std::ffi::c_void);
            enc.set_bytes(12, 4, &scale as *const f32 as *const std::ffi::c_void);
            enc.set_bytes(13, 4, &window_size as *const u32 as *const std::ffi::c_void);
            enc.set_bytes(14, 4, &eps as *const f32 as *const std::ffi::c_void);
            enc.set_bytes(15, 4, &qk_off as *const f32 as *const std::ffi::c_void);
            enc.set_bytes(
                16,
                4,
                &layer_rope_base as *const f32 as *const std::ffi::c_void,
            );
            enc.set_bytes(17, 4, &rdim as *const u32 as *const std::ffi::c_void);
            enc.dispatch_thread_groups(
                MTLSize::new(layer_num_q_heads as u64, 1, 1),
                MTLSize::new(tg_w, 1, 1),
            );
            kv_cache.layers[layer_idx].current_len += 1;
        } else if use_fused_qkn_rope
            && layer.q_norm_weight.is_some()
            && layer.k_norm_weight.is_some()
        {
            let q_w = layer.q_norm_weight.unwrap();
            let k_w = layer.k_norm_weight.unwrap();
            let hd_val = layer_head_dim as u32;
            let nq_val = layer_num_q_heads as u32;
            let qk_off = norms_view.qk_norm_offset;
            let rdim = layer_rotary_dim as u32;
            let mut tg_w: usize = 1;
            while tg_w < layer_head_dim && tg_w < MAX_HEAD_DIM_DOUBLE_SG {
                tg_w <<= 1;
            }
            let q_w_buf = self.bufs.get_f32(q_w);
            let k_w_buf = self.bufs.get_f32(k_w);
            let total_heads = (layer_num_q_heads + layer_num_kv_heads) as u64;
            enc.set_compute_pipeline_state(&self.norms.qk_norm_rope_fused_pipeline);
            enc.set_buffer(0, Some(bufs.q_out), 0);
            enc.set_buffer(1, Some(bufs.k_out), 0);
            enc.set_buffer(2, Some(&q_w_buf), 0);
            enc.set_buffer(3, Some(&k_w_buf), 0);
            enc.set_bytes(4, 4, &hd_val as *const u32 as *const std::ffi::c_void);
            enc.set_bytes(5, 4, &nq_val as *const u32 as *const std::ffi::c_void);
            enc.set_bytes(6, 4, &eps as *const f32 as *const std::ffi::c_void);
            enc.set_bytes(7, 4, &qk_off as *const f32 as *const std::ffi::c_void);
            enc.set_bytes(
                8,
                4,
                &layer_rope_base as *const f32 as *const std::ffi::c_void,
            );
            enc.set_bytes(9, 4, &pos as *const u32 as *const std::ffi::c_void);
            enc.set_bytes(10, 4, &rdim as *const u32 as *const std::ffi::c_void);
            enc.dispatch_thread_groups(
                MTLSize::new(total_heads, 1, 1),
                MTLSize::new(tg_w as u64, 1, 1),
            );
        } else {
            if let (Some(q_w), Some(k_w)) = (layer.q_norm_weight, layer.k_norm_weight) {
                let hd_val = layer_head_dim as u32;
                let nq_val = layer_num_q_heads as u32;
                let qk_off = norms_view.qk_norm_offset;
                let mut tg_w: usize = 1;
                while tg_w < layer_head_dim && tg_w < MAX_HEAD_DIM_DOUBLE_SG {
                    tg_w <<= 1;
                }
                let q_w_buf = self.bufs.get_f32(q_w);
                let k_w_buf = self.bufs.get_f32(k_w);
                let total_heads = (layer_num_q_heads + layer_num_kv_heads) as u64;
                enc.set_compute_pipeline_state(&self.norms.qk_norm_qk_pipeline);
                enc.set_buffer(0, Some(bufs.q_out), 0);
                enc.set_buffer(1, Some(bufs.k_out), 0);
                enc.set_buffer(2, Some(&q_w_buf), 0);
                enc.set_buffer(3, Some(&k_w_buf), 0);
                enc.set_bytes(4, 4, &hd_val as *const u32 as *const std::ffi::c_void);
                enc.set_bytes(5, 4, &nq_val as *const u32 as *const std::ffi::c_void);
                enc.set_bytes(6, 4, &eps as *const f32 as *const std::ffi::c_void);
                enc.set_bytes(7, 4, &qk_off as *const f32 as *const std::ffi::c_void);
                enc.dispatch_thread_groups(
                    MTLSize::new(total_heads, 1, 1),
                    MTLSize::new(tg_w as u64, 1, 1),
                );
            }

            // ── Step 2: RoPE on Q and K heads (batched — one dispatch each) ──
            let hd = layer_head_dim as u32;
            let rdim = layer_rotary_dim as u32;
            let rope_pairs = (layer_rotary_dim / 2) as u64;
            let num_q = layer_num_q_heads as u32;
            let total_qk_heads = (layer_num_q_heads + layer_num_kv_heads) as u64;
            enc.set_compute_pipeline_state(&self.attention.rope_at_pos_batched_qk_pipeline);
            enc.set_buffer(0, Some(bufs.q_out), 0);
            enc.set_buffer(1, Some(bufs.k_out), 0);
            enc.set_bytes(2, 4, &hd as *const u32 as *const std::ffi::c_void);
            enc.set_bytes(
                3,
                4,
                &layer_rope_base as *const f32 as *const std::ffi::c_void,
            );
            enc.set_bytes(4, 4, &pos as *const u32 as *const std::ffi::c_void);
            enc.set_bytes(5, 4, &rdim as *const u32 as *const std::ffi::c_void);
            enc.set_bytes(6, 4, &num_q as *const u32 as *const std::ffi::c_void);
            enc.dispatch_threads(
                MTLSize::new(rope_pairs, total_qk_heads, 1),
                MTLSize::new(rope_pairs.min(256), 1, 1),
            );
        }

        // ── Step 3: V-norm batched (optional) ──
        if layer.has_v_norm {
            let hd_val = layer_head_dim as u32;
            let num_kv = layer_num_kv_heads as u32;
            let mut tg_w: u64 = 1;
            while tg_w < layer_head_dim as u64 && tg_w < MAX_HEAD_DIM_DOUBLE_SG as u64 {
                tg_w <<= 1;
            }
            enc.set_compute_pipeline_state(&self.norms.v_norm_batched_pipeline);
            enc.set_buffer(0, Some(bufs.v_out), 0);
            enc.set_buffer(1, Some(bufs.v_out), 0);
            enc.set_bytes(2, 4, &hd_val as *const u32 as *const std::ffi::c_void);
            enc.set_bytes(3, 4, &eps as *const f32 as *const std::ffi::c_void);
            enc.set_bytes(4, 4, &num_kv as *const u32 as *const std::ffi::c_void);
            enc.dispatch_thread_groups(
                MTLSize::new(layer_num_kv_heads as u64, 1, 1),
                MTLSize::new(tg_w, 1, 1),
            );
        }

        // ── Step 4: KV-append + KV-attend ──
        // Skipped entirely when `did_fused_attn` is true (the unified
        // `attn_fused` kernel above already wrote both cache rows + the
        // attention output and bumped current_len). Shared layers
        // (`kv_shared_source = Some(_)`) skip the append entirely and
        // attend against the source's cache.
        if did_fused_attn {
            // Already done — attn_fused wrote attn_out_buf + bumped current_len.
        } else if use_fused_kv_aa {
            let cache = &kv_cache.layers[layer_idx];
            let t_val = (cache.current_len + 1) as u32;
            let hd = cache.head_dim as u32;
            let num_q_val = layer_num_q_heads as u32;
            let num_kv = cache.num_kv_heads as u32;
            enc.set_compute_pipeline_state(&self.attention.kv_append_attend_fused_pipeline);
            enc.set_buffer(0, Some(bufs.q_out), 0);
            enc.set_buffer(1, Some(&cache.k_cache), 0);
            enc.set_buffer(2, Some(&cache.v_cache), 0);
            enc.set_buffer(3, Some(bufs.attn_out_buf), 0);
            enc.set_bytes(4, 4, &t_val as *const u32 as *const std::ffi::c_void);
            enc.set_bytes(5, 4, &hd as *const u32 as *const std::ffi::c_void);
            enc.set_bytes(6, 4, &num_q_val as *const u32 as *const std::ffi::c_void);
            enc.set_bytes(7, 4, &num_kv as *const u32 as *const std::ffi::c_void);
            enc.set_bytes(8, 4, &scale as *const f32 as *const std::ffi::c_void);
            enc.set_bytes(9, 4, &window_size as *const u32 as *const std::ffi::c_void);
            enc.set_buffer(10, Some(bufs.k_out), 0);
            enc.set_buffer(11, Some(bufs.v_out), 0);
            enc.dispatch_thread_groups(
                MTLSize::new(layer_num_q_heads as u64, 1, 1),
                MTLSize::new(
                    crate::kernels::DISPATCH_TG_MAX_THREADS.min(layer_head_dim as u64),
                    1,
                    1,
                ),
            );
        } else if let Some(src) = kv_shared_source {
            // Shared layer: attend against source's cache. Skip append
            // (the source already appended its K/V row this token).
            // `t_val = source.current_len` because source has already
            // incremented it to count the new token.
            let cache = &kv_cache.layers[src];
            let t_val_u = cache.current_len as u32;
            let hd_u = cache.head_dim as u32;
            let num_q_u = layer_num_q_heads as u32;
            let num_kv_u = cache.num_kv_heads as u32;
            // Pick the same pipeline encode_kv_attend would (long-form for
            // spans past SHORT_ATTENTION_SPAN; standard otherwise).
            let span_u = ops::kv_cache::attention_span(t_val_u, window_size);
            let pipeline = if span_u > ops::kv_cache::SHORT_ATTENTION_SPAN {
                &self.attention.kv_attend_long_pipeline
            } else {
                &self.attention.kv_attend_pipeline
            };
            enc.set_compute_pipeline_state(pipeline);
            enc.set_buffer(0, Some(bufs.q_out), 0);
            enc.set_buffer(1, Some(&cache.k_cache), 0);
            enc.set_buffer(2, Some(&cache.v_cache), 0);
            enc.set_buffer(3, Some(bufs.attn_out_buf), 0);
            enc.set_bytes(4, 4, &t_val_u as *const u32 as *const std::ffi::c_void);
            enc.set_bytes(5, 4, &hd_u as *const u32 as *const std::ffi::c_void);
            enc.set_bytes(6, 4, &num_q_u as *const u32 as *const std::ffi::c_void);
            enc.set_bytes(7, 4, &num_kv_u as *const u32 as *const std::ffi::c_void);
            enc.set_bytes(8, 4, &scale as *const f32 as *const std::ffi::c_void);
            enc.set_bytes(9, 4, &window_size as *const u32 as *const std::ffi::c_void);
            enc.dispatch_thread_groups(
                MTLSize::new(layer_num_q_heads as u64, 1, 1),
                MTLSize::new(
                    crate::kernels::DISPATCH_TG_MAX_THREADS.min(cache.head_dim as u64),
                    1,
                    1,
                ),
            );
        } else {
            ops::kv_cache::encode_kv_append(
                enc,
                &kv_cache.layers[layer_idx],
                &self.attention.kv_append_pipeline,
                bufs.k_out,
                bufs.v_out,
            );
            ops::kv_cache::encode_kv_attend(
                enc,
                &kv_cache.layers[layer_idx],
                &self.attention.kv_attend_pipeline,
                Some(&self.attention.kv_attend_long_pipeline),
                bufs.q_out,
                bufs.attn_out_buf,
                layer_num_q_heads,
                scale,
                window_size,
            );
        }
        // Only own-cache layers advance current_len; shared layers leave
        // their (unused) cache pointer at 0 forever.
        if !did_fused_attn && kv_shared_source.is_none() {
            kv_cache.layers[layer_idx].current_len += 1;
        }

        // ── Step 5a: O projection ──
        if uses_kquant {
            use crate::stages::quant_matvec::Pipelines;
            let pipes = Pipelines {
                q4kf_proj: Some(&self.attention.q4kf_proj_pipeline.state),
                q4k_matvec_fallback: &self.quant.q4k_matvec_pipeline,
                q6k_matvec: &self.quant.q6k_matvec_pipeline,
                q4_matvec: &self.q4.matvec,
                q4k_matmul: None,
            };
            crate::stages::o_proj::encode(
                enc,
                &pipes,
                &self.quant.q8_quant_pipeline,
                layer.wo.format,
                bufs.wo,
                bufs.attn_out_buf,
                0,
                bufs.o_q8_scratch,
                0,
                bufs.o_q8s_scratch,
                0,
                bufs.o_out_buf,
                0,
                layer_q_dim,
                hidden,
            );
        } else {
            // Q8 legacy path: decode-specific `q8_matvec` shader (not in
            // stages::quant_matvec which uses `q4_matvec` for Q4_0/Q8_0 with
            // a different buffer layout). Inline.
            let dim_val = layer_q_dim as u32;
            let blocks = (layer_q_dim / LEGACY_BLOCK_ELEMS) as u32;
            enc.set_compute_pipeline_state(&self.quant.q8_quant_pipeline);
            enc.set_buffer(0, Some(bufs.attn_out_buf), 0);
            enc.set_buffer(1, Some(bufs.o_q8_scratch), 0);
            enc.set_buffer(2, Some(bufs.o_q8s_scratch), 0);
            enc.set_bytes(3, 4, &dim_val as *const u32 as *const std::ffi::c_void);
            enc.dispatch_threads(
                MTLSize::new(blocks as u64, 1, 1),
                MTLSize::new(
                    crate::kernels::DISPATCH_TG_MAX_THREADS.min(blocks as u64),
                    1,
                    1,
                ),
            );

            let o_rows = hidden as u32;
            let o_k = layer_q_dim as u32;
            enc.set_compute_pipeline_state(&self.quant.q8_matvec_pipeline.state);
            enc.set_buffer(0, Some(bufs.wo), 0);
            enc.set_buffer(1, Some(bufs.o_q8_scratch), 0);
            enc.set_buffer(2, Some(bufs.wo_scales), 0);
            enc.set_buffer(3, Some(bufs.o_q8s_scratch), 0);
            enc.set_buffer(4, Some(bufs.o_out_buf), 0);
            enc.set_bytes(5, 4, &o_rows as *const u32 as *const std::ffi::c_void);
            enc.set_bytes(6, 4, &o_k as *const u32 as *const std::ffi::c_void);
            enc.dispatch_thread_groups(
                MTLSize::new((hidden as u64).div_ceil(8), 1, 1),
                MTLSize::new(256, 1, 1),
            );
        }

        // ── Step 5b: Residual + post-attn norm + ffn-input norm ──
        let has_post_norms = layer.has_post_norms;
        if has_post_norms {
            let pre_ffn_buf = if let Some(pfn) = layer.pre_ffn_norm {
                self.bufs.get_f32(pfn)
            } else {
                bufs.post_attn_norm.clone()
            };
            if use_fused_post_attn && ffn_uses_kquant {
                // Triple-fused: post_attn_norm + residual_norm + h_post_attn
                // store in ONE dispatch.
                enc.set_compute_pipeline_state(&self.norms.post_attn_residual_norm_store_pipeline);
                enc.set_buffer(0, Some(bufs.h_buf), 0);
                enc.set_buffer(1, Some(bufs.o_out_buf), 0);
                enc.set_buffer(2, Some(bufs.post_attn_norm), 0);
                enc.set_buffer(3, Some(&pre_ffn_buf), 0);
                enc.set_buffer(4, Some(bufs.ffn_norm_out), 0);
                enc.set_buffer(5, Some(bufs.h_post_attn), 0);
                enc.set_bytes(6, 4, &hidden_val as *const u32 as *const std::ffi::c_void);
                enc.set_bytes(7, 4, &eps as *const f32 as *const std::ffi::c_void);
                enc.set_bytes(8, 4, &norm_offset as *const f32 as *const std::ffi::c_void);
                enc.dispatch_thread_groups(
                    MTLSize::new(1, 1, 1),
                    MTLSize::new(
                        crate::kernels::DISPATCH_TG_MAX_THREADS.min(hidden as u64),
                        1,
                        1,
                    ),
                );
            } else {
                use crate::ops::full_pipeline::encode_rms_norm;
                encode_rms_norm(
                    enc,
                    &self.norms.rms_norm_pipeline,
                    bufs.o_out_buf,
                    bufs.post_attn_norm,
                    bufs.normed_scratch,
                    hidden,
                    eps,
                    norm_offset,
                );
                // Granite residual_multiplier; 1.0 for every other model.
                // `residual_norm_store` reads it at buffer(8),
                // `residual_norm_q8` at buffer(9).
                let b_scale: f32 = layer.residual_multiplier;
                if ffn_uses_kquant {
                    enc.set_compute_pipeline_state(&self.norms.residual_norm_store_pipeline);
                    enc.set_buffer(0, Some(bufs.h_buf), 0);
                    enc.set_buffer(1, Some(bufs.normed_scratch), 0);
                    enc.set_buffer(2, Some(&pre_ffn_buf), 0);
                    enc.set_buffer(3, Some(bufs.ffn_norm_out), 0);
                    enc.set_buffer(4, Some(bufs.h_post_attn), 0);
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
                } else {
                    enc.set_compute_pipeline_state(&self.norms.residual_norm_q8_pipeline);
                    enc.set_buffer(0, Some(bufs.h_buf), 0);
                    enc.set_buffer(1, Some(bufs.normed_scratch), 0);
                    enc.set_buffer(2, Some(&pre_ffn_buf), 0);
                    enc.set_buffer(3, Some(bufs.ffn_q8), 0);
                    enc.set_buffer(4, Some(bufs.ffn_q8s), 0);
                    enc.set_buffer(5, Some(bufs.h_post_attn), 0);
                    enc.set_bytes(6, 4, &hidden_val as *const u32 as *const std::ffi::c_void);
                    enc.set_bytes(7, 4, &eps as *const f32 as *const std::ffi::c_void);
                    enc.set_bytes(8, 4, &norm_offset as *const f32 as *const std::ffi::c_void);
                    enc.set_bytes(9, 4, &b_scale as *const f32 as *const std::ffi::c_void);
                    enc.dispatch_thread_groups(
                        MTLSize::new(1, 1, 1),
                        MTLSize::new(
                            crate::kernels::DISPATCH_TG_MAX_THREADS.min(hidden as u64),
                            1,
                            1,
                        ),
                    );
                }
            }
        } else if ffn_uses_kquant {
            let b_scale: f32 = layer.residual_multiplier;
            enc.set_compute_pipeline_state(&self.norms.residual_norm_store_pipeline);
            enc.set_buffer(0, Some(bufs.h_buf), 0);
            enc.set_buffer(1, Some(bufs.o_out_buf), 0);
            enc.set_buffer(2, Some(bufs.post_attn_norm), 0);
            enc.set_buffer(3, Some(bufs.ffn_norm_out), 0);
            enc.set_buffer(4, Some(bufs.h_post_attn), 0);
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
        } else {
            let b_scale: f32 = layer.residual_multiplier;
            enc.set_compute_pipeline_state(&self.norms.residual_norm_q8_pipeline);
            enc.set_buffer(0, Some(bufs.h_buf), 0);
            enc.set_buffer(1, Some(bufs.o_out_buf), 0);
            enc.set_buffer(2, Some(bufs.post_attn_norm), 0);
            enc.set_buffer(3, Some(bufs.ffn_q8), 0);
            enc.set_buffer(4, Some(bufs.ffn_q8s), 0);
            enc.set_buffer(5, Some(bufs.h_post_attn), 0);
            enc.set_bytes(6, 4, &hidden_val as *const u32 as *const std::ffi::c_void);
            enc.set_bytes(7, 4, &eps as *const f32 as *const std::ffi::c_void);
            enc.set_bytes(8, 4, &norm_offset as *const f32 as *const std::ffi::c_void);
            enc.set_bytes(9, 4, &b_scale as *const f32 as *const std::ffi::c_void);
            enc.dispatch_thread_groups(
                MTLSize::new(1, 1, 1),
                MTLSize::new(
                    crate::kernels::DISPATCH_TG_MAX_THREADS.min(hidden as u64),
                    1,
                    1,
                ),
            );
        }
    }
}
