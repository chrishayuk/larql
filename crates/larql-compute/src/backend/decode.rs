//! `DecodeBackend` — full-pipeline KV-cached decode + prefill.
//!
//! These methods cover the autoregressive inference loop: prefill
//! (multi-position with KV-cache population), decode (single token
//! against the cache), MoE-aware decode, and per-stage timing.
//!
//! All methods default to `None` / no-op; only the GPU backend
//! implements them today (CPU runs decode through the higher-level
//! `larql-inference` path, not through `ComputeBackend`).
//!
//! All attention geometry (head_dim, num_q_heads, num_kv_heads,
//! rope_base, sliding_window, etc.) is read per-layer from
//! `FullPipelineLayer`. The trait surface intentionally does **not**
//! take scalar geometry parameters — passing them would invite a
//! single-layer fallback to silently corrupt heterogeneous models
//! like Gemma 4 31B (50 sliding-attention layers + 10 global-attention
//! layers, with different head_dim and num_kv_heads on each class).

/// KV-cached generation primitives.
///
/// "Backend supports decode" means the backend can run a full forward
/// pass internally — attention + FFN + KV cache update — without
/// returning intermediate residuals to the caller.
pub trait DecodeBackend {
    /// Full pipeline: ALL Q4 (attention + FFN) for all layers in ONE
    /// command buffer. Each layer: Q4 Q/K/V proj → fused attention →
    /// Q4 O proj → Q4 FFN. No CPU-GPU round-trips between layers.
    #[allow(clippy::too_many_arguments)]
    fn full_pipeline_q4(
        &self,
        _layers: &[crate::FullPipelineLayer<'_>],
        _x: &[f32],
        _hidden: usize,
        _inter: usize,
        _seq_len: usize,
        _use_qk_norm: bool,
        _softcap: f32,
    ) -> Option<Vec<f32>> {
        None
    }

    /// Like `full_pipeline_q4` but replaces one attention head's residual
    /// contribution at `target_layer` with `replacement_delta`.
    ///
    /// This is the Metal-accelerated path for Mode D head injection used by
    /// the AHORD CEGIS loop. Default delegates to `full_pipeline_q4` (no
    /// intervention — callers must fall back to the CPU path if this returns
    /// `None`).
    #[allow(clippy::too_many_arguments)]
    fn full_pipeline_q4_with_head_replacement(
        &self,
        layers: &[crate::FullPipelineLayer<'_>],
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
        // Default: fall back to full pipeline without intervention.
        // Metal backend overrides this with the intervention-aware path.
        let _ = (target_layer, target_head, replacement_delta);
        self.full_pipeline_q4(layers, x, hidden, inter, seq_len, use_qk_norm, softcap)
    }

    /// Multi-layer Q4 FFN in one submission: gate → up → GEGLU → down.
    fn multi_layer_q4_ffn(
        &self,
        _layers_q4: &[(&[u8], &[u8], &[u8])],
        _x: &[f32],
        _inter: usize,
        _hidden: usize,
    ) -> Option<Vec<f32>> {
        None
    }

    /// Whether this backend supports KV-cache decode operations.
    fn has_kv_cache(&self) -> bool {
        false
    }

    /// Populate KV cache with prefill K/V data for one layer.
    ///
    /// `(num_kv_heads, head_dim)` here are this specific layer's
    /// geometry — not vestigial scalars. The caller must pass per-layer
    /// values (e.g. via `arch.num_kv_heads_for_layer(layer)`); a
    /// uniform-from-layer-0 fallback would corrupt heterogeneous models.
    fn populate_kv_layer(
        &self,
        _layer: usize,
        _k_data: &[f32],
        _v_data: &[f32],
        _seq_len: usize,
        _num_kv_heads: usize,
        _head_dim: usize,
    ) {
    }

    /// Reset KV cache (for new prompt).
    fn reset_kv_cache(&self) {}

    /// Return the number of token positions currently committed to the KV cache.
    fn kv_cache_len(&self) -> usize {
        0
    }

    /// Roll back the KV cache to a previously saved length.  Safe to call with
    /// any `len ≤ current_len`; the physical K/V data below `len` is preserved
    /// (positions 0..len are not zeroed), so a subsequent decode pass starting
    /// from position `len` will produce correct attention over the prior tokens.
    ///
    /// Used by iterative predispatch: all but the final Metal pass call
    /// `truncate_kv_cache(saved_len)` so that only the last pass permanently
    /// advances the sequence length.
    fn truncate_kv_cache(&self, _len: usize) {}

    /// Pre-allocate the KV cache with per-layer shapes. Required for
    /// asymmetric attention geometry (Gemma 4 alternates sliding/global).
    fn preallocate_kv_cache_per_layer(&self, _shapes: &[(usize, usize)], _max_seq: usize) {}

    /// Decode one token through all layers with KV cache.
    fn decode_token(
        &self,
        _layers: &[crate::FullPipelineLayer<'_>],
        _x: &[f32],
        _hidden: usize,
        _inter: usize,
    ) -> Option<Vec<f32>> {
        None
    }

    /// Like `decode_token` but calls `moe_fn(layer, h_post_attn)` for
    /// MoE layers (enables remote expert dispatch). Default delegates
    /// to `decode_token` and ignores the hook.
    fn decode_token_with_moe(
        &self,
        layers: &[crate::FullPipelineLayer<'_>],
        x: &[f32],
        hidden: usize,
        inter: usize,
        _moe_fn: &mut dyn FnMut(usize, &[f32]) -> Vec<f32>,
    ) -> Option<Vec<f32>> {
        self.decode_token(layers, x, hidden, inter)
    }

    /// Decode one token while dispatching Q4_K per-layer expert tensors on
    /// the backend. The expert callback returns borrowed `(gate_up, down)`
    /// byte slices for the requested `(layer, expert)` pair.
    fn decode_token_q4k_moe<'w>(
        &self,
        _layers: &[crate::FullPipelineLayer<'_>],
        _x: &[f32],
        _hidden: usize,
        _inter: usize,
        _norm_eps: f32,
        _get_expert: &dyn Fn(usize, usize) -> Option<(&'w [u8], &'w [u8])>,
    ) -> Option<Vec<f32>> {
        None
    }

    /// Split fire / collect variant of `decode_token_with_moe`.  At each MoE
    /// layer the implementation calls `moe_fire_fn(layer, h_post_attn)` once
    /// `h_post_attn` is computed, encodes dense FFN + post-FFN residual on a
    /// fresh command buffer, commits without waiting, then calls
    /// `moe_collect_fn(layer)` to retrieve the expert weighted-sum vector
    /// while the GPU runs the dense FFN in parallel.
    ///
    /// Default impl combines the two callbacks into a single synchronous
    /// closure and forwards to `decode_token_with_moe` — backends that don't
    /// support encoder splitting see no behaviour change.
    fn decode_token_with_moe_split(
        &self,
        layers: &[crate::FullPipelineLayer<'_>],
        x: &[f32],
        hidden: usize,
        inter: usize,
        moe_fire_fn: &mut dyn FnMut(usize, &[f32]),
        moe_collect_fn: &mut dyn FnMut(usize) -> Vec<f32>,
    ) -> Option<Vec<f32>> {
        // Default: synthesise a single synchronous moe_fn from the pair.
        let mut combined = |layer: usize, h: &[f32]| -> Vec<f32> {
            moe_fire_fn(layer, h);
            moe_collect_fn(layer)
        };
        self.decode_token_with_moe(layers, x, hidden, inter, &mut combined)
    }

    /// Like `decode_token` but splits each layer into attn / gate+up /
    /// down command buffers and times each. Returns `(result, attn_ms,
    /// gate_up_ms, down_ms)`. Default delegates to `decode_token` with
    /// zero timings.
    fn decode_token_split_profile(
        &self,
        layers: &[crate::FullPipelineLayer<'_>],
        x: &[f32],
        hidden: usize,
        inter: usize,
    ) -> (Option<Vec<f32>>, f64, f64, f64) {
        (self.decode_token(layers, x, hidden, inter), 0.0, 0.0, 0.0)
    }

    /// Multi-position prefill with KV-cache population. Stores
    /// post-RoPE K/V in the cache; returns the final hidden state
    /// `[seq_len * hidden]` for all positions.
    #[allow(clippy::too_many_arguments)]
    fn prefill_kquant(
        &self,
        _layers: &[crate::FullPipelineLayer<'_>],
        _x: &[f32],
        _hidden: usize,
        _inter: usize,
        _seq_len: usize,
        _use_qk_norm: bool,
        _softcap: f32,
    ) -> Option<Vec<f32>> {
        None
    }

    /// Capture the target head's pre-W_O output at `target_layer` via GPU,
    /// then stop. Returns `[seq_len × head_dim]` f32 — the raw attention output
    /// for `target_head` before W_O projection.
    ///
    /// For AHORD oracle code computation: runs only layers 0..=target_layer on GPU
    /// (not all 34 layers), giving ~34× speedup for target_layer=0 over CPU.
    #[allow(clippy::too_many_arguments)]
    fn full_pipeline_kquant_capture_pre_wo(
        &self,
        _layers: &[crate::FullPipelineLayer<'_>],
        _x: &[f32],
        _hidden: usize,
        _inter: usize,
        _seq_len: usize,
        _use_qk_norm: bool,
        _softcap: f32,
        _target_layer: usize,
        _target_head: usize,
    ) -> Option<Vec<f32>> {
        None
    }

    /// Like `prefill_kquant` but replaces one attention head's residual contribution
    /// at `target_layer` with `replacement_delta` — the AHORD Mode D injection path.
    ///
    /// Uses the same KV cache + per-position RoPE setup as `prefill_kquant`, so positional
    /// encodings are correct for all seq_len positions. Default returns `None`; the
    /// Metal backend overrides with the intervention-aware dispatch.
    #[allow(clippy::too_many_arguments)]
    fn prefill_kquant_with_head_replacement(
        &self,
        layers: &[crate::FullPipelineLayer<'_>],
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
        let _ = (target_layer, target_head, replacement_delta);
        self.prefill_kquant(layers, x, hidden, inter, seq_len, use_qk_norm, softcap)
    }
}
