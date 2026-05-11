//! `DecodeBackend` — full-pipeline KV-cached decode + prefill.
//!
//! These methods cover the autoregressive inference loop: prefill
//! (multi-position with KV-cache population), decode (single token
//! against the cache), MoE-aware decode, and per-stage timing.
//!
//! All methods default to `None` / no-op; only the GPU backend
//! implements them today (CPU runs decode through the higher-level
//! `larql-inference` path, not through `ComputeBackend`).

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
        _q_dim: usize,
        _kv_dim: usize,
        _seq_len: usize,
        _num_q_heads: usize,
        _num_kv_heads: usize,
        _head_dim: usize,
        _rope_base: f32,
        _use_qk_norm: bool,
        _softcap: f32,
    ) -> Option<Vec<f32>> {
        None
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
    #[allow(clippy::too_many_arguments)]
    fn decode_token(
        &self,
        _layers: &[crate::FullPipelineLayer<'_>],
        _x: &[f32],
        _hidden: usize,
        _inter: usize,
        _q_dim: usize,
        _kv_dim: usize,
        _num_q_heads: usize,
        _num_kv_heads: usize,
        _head_dim: usize,
        _rope_base: f32,
    ) -> Option<Vec<f32>> {
        None
    }

    /// Like `decode_token` but calls `moe_fn(layer, h_post_attn)` for
    /// MoE layers (enables remote expert dispatch). Default delegates
    /// to `decode_token` and ignores the hook.
    #[allow(clippy::too_many_arguments)]
    fn decode_token_with_moe(
        &self,
        layers: &[crate::FullPipelineLayer<'_>],
        x: &[f32],
        hidden: usize,
        inter: usize,
        q_dim: usize,
        kv_dim: usize,
        num_q_heads: usize,
        num_kv_heads: usize,
        head_dim: usize,
        rope_base: f32,
        _moe_fn: &mut dyn FnMut(usize, &[f32]) -> Vec<f32>,
    ) -> Option<Vec<f32>> {
        self.decode_token(
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
        )
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
    #[allow(clippy::too_many_arguments)]
    fn decode_token_with_moe_split(
        &self,
        layers: &[crate::FullPipelineLayer<'_>],
        x: &[f32],
        hidden: usize,
        inter: usize,
        q_dim: usize,
        kv_dim: usize,
        num_q_heads: usize,
        num_kv_heads: usize,
        head_dim: usize,
        rope_base: f32,
        moe_fire_fn: &mut dyn FnMut(usize, &[f32]),
        moe_collect_fn: &mut dyn FnMut(usize) -> Vec<f32>,
    ) -> Option<Vec<f32>> {
        // Default: synthesise a single synchronous moe_fn from the pair.
        let mut combined = |layer: usize, h: &[f32]| -> Vec<f32> {
            moe_fire_fn(layer, h);
            moe_collect_fn(layer)
        };
        self.decode_token_with_moe(
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
            &mut combined,
        )
    }

    /// Like `decode_token` but splits each layer into attn / gate+up /
    /// down command buffers and times each. Returns `(result, attn_ms,
    /// gate_up_ms, down_ms)`. Default delegates to `decode_token` with
    /// zero timings.
    #[allow(clippy::too_many_arguments)]
    fn decode_token_split_profile(
        &self,
        layers: &[crate::FullPipelineLayer<'_>],
        x: &[f32],
        hidden: usize,
        inter: usize,
        q_dim: usize,
        kv_dim: usize,
        num_q_heads: usize,
        num_kv_heads: usize,
        head_dim: usize,
        rope_base: f32,
    ) -> (Option<Vec<f32>>, f64, f64, f64) {
        (
            self.decode_token(
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
            ),
            0.0,
            0.0,
            0.0,
        )
    }

    /// Speculative-decode forward: process `x_per_token.len()` tokens
    /// through all layers with KV cache, returning the per-position
    /// hidden state for each.
    ///
    /// **Cache semantics**: this method advances the KV cache during
    /// processing then RESTORES it to its pre-call length. The
    /// caller (typically `larql_inference::speculative`) explores N
    /// candidate tokens without committing them; on acceptance, the
    /// caller separately calls `decode_token` for each accepted
    /// token to commit it permanently to the cache.
    ///
    /// Returns `Some(hiddens)` on success — `hiddens.len() ==
    /// x_per_token.len()`, with `hiddens[k]` the `[hidden]`-shaped
    /// output of the target's forward pass after consuming
    /// `x_per_token[..=k]`. Returns `None` if any per-token
    /// `decode_token` call fails (cache state restored regardless).
    ///
    /// **Default impl**: sequential `decode_token` calls + cache
    /// rollback via `truncate_kv_cache`. Backends with batched
    /// kernels (`cuda::q4k_batched` + `cuda::attn_tree`) MAY
    /// override for the perf win — phase 4c task C.2 implements
    /// this for `CudaBackend`.
    #[allow(clippy::too_many_arguments)]
    fn decode_tokens_speculative(
        &self,
        layers: &[crate::FullPipelineLayer<'_>],
        x_per_token: &[Vec<f32>],
        hidden: usize,
        inter: usize,
        q_dim: usize,
        kv_dim: usize,
        num_q_heads: usize,
        num_kv_heads: usize,
        head_dim: usize,
        rope_base: f32,
    ) -> Option<Vec<Vec<f32>>> {
        if !self.has_kv_cache() {
            return None;
        }
        let pre_len = self.kv_cache_len();
        let mut hiddens: Vec<Vec<f32>> = Vec::with_capacity(x_per_token.len());
        for x in x_per_token {
            match self.decode_token(
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
            ) {
                Some(h) => hiddens.push(h),
                None => {
                    // Restore cache before bailing.
                    self.truncate_kv_cache(pre_len);
                    return None;
                }
            }
        }
        self.truncate_kv_cache(pre_len);
        Some(hiddens)
    }

    /// `cuda-spec-branching-tree` T3.2: tree-shaped variant of
    /// [`Self::decode_tokens_speculative_keep_cache`]. Processes a
    /// branching draft tree in one batched forward, masking attention
    /// per-node so siblings don't leak.
    ///
    /// `ancestors[n]` SHALL be the u64 bitset for node `n` (per
    /// `DraftTree::ancestor_bitsets()`). `ancestors.len() ==
    /// x_per_node.len() <= 64`.
    ///
    /// On `None` (backend lacks tree-mask kernel, or tree size
    /// exceeds the kernel's cap), the caller SHALL fall back to the
    /// per-node `decode_token` path. Cache is restored to `pre_len`
    /// on any error.
    ///
    /// Default impl returns `None` so backends without the tree-mask
    /// kernel route to the conservative per-node fallback.
    #[allow(clippy::too_many_arguments)]
    fn decode_tokens_speculative_tree_keep_cache(
        &self,
        _layers: &[crate::FullPipelineLayer<'_>],
        _x_per_node: &[Vec<f32>],
        _ancestors: &[u64],
        _hidden: usize,
        _inter: usize,
        _q_dim: usize,
        _kv_dim: usize,
        _num_q_heads: usize,
        _num_kv_heads: usize,
        _head_dim: usize,
        _rope_base: f32,
    ) -> Option<Vec<Vec<f32>>> {
        None
    }

    /// Variant of [`Self::decode_tokens_speculative`] that does NOT
    /// truncate the KV cache on success — the cache is left advanced
    /// by `x_per_token.len()` positions. The CALLER is responsible for
    /// truncating to the desired position (typically `pre_len + R`
    /// where R is the number of accepted tokens after `verify_tree`)
    /// and re-decoding the bonus token at position `pre_len + R`.
    ///
    /// **Why this exists** (phase 4c skip-redundant-commit): the
    /// previous flow ran the helper, truncated cache back to pre_len,
    /// then re-decoded ALL R+1 emitted tokens to commit them. The
    /// first R of those re-decodes were redundant — the helper had
    /// just decoded the same R tokens (drafts[0..R-1]). With this
    /// keep-cache variant, the helper's chain decode commits drafts
    /// to cache; the caller then truncates to drop drafts[R..N-1]
    /// and re-decodes only the bonus (which is a resampled token,
    /// not equal to drafts[R]).
    ///
    /// On error, cache IS restored to pre_len (matches the truncate
    /// variant's error semantics).
    ///
    /// Returns `Some(hiddens)` with `hiddens[k]` = post-token-k hidden,
    /// same shape as [`Self::decode_tokens_speculative`].
    #[allow(clippy::too_many_arguments)]
    fn decode_tokens_speculative_keep_cache(
        &self,
        layers: &[crate::FullPipelineLayer<'_>],
        x_per_token: &[Vec<f32>],
        hidden: usize,
        inter: usize,
        q_dim: usize,
        kv_dim: usize,
        num_q_heads: usize,
        num_kv_heads: usize,
        head_dim: usize,
        rope_base: f32,
    ) -> Option<Vec<Vec<f32>>> {
        if !self.has_kv_cache() {
            return None;
        }
        let pre_len = self.kv_cache_len();
        let mut hiddens: Vec<Vec<f32>> = Vec::with_capacity(x_per_token.len());
        for x in x_per_token {
            match self.decode_token(
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
            ) {
                Some(h) => hiddens.push(h),
                None => {
                    // Restore cache before bailing — same as the
                    // truncate variant's error semantics so callers
                    // don't have to handle a partial-advance state.
                    self.truncate_kv_cache(pre_len);
                    return None;
                }
            }
        }
        // NO truncate on success — cache is left at pre_len + n.
        Some(hiddens)
    }

    /// Multi-position prefill with KV-cache population. Stores
    /// post-RoPE K/V in the cache; returns the final hidden state
    /// `[seq_len * hidden]` for all positions.
    #[allow(clippy::too_many_arguments)]
    fn prefill_q4(
        &self,
        _layers: &[crate::FullPipelineLayer<'_>],
        _x: &[f32],
        _hidden: usize,
        _inter: usize,
        _q_dim: usize,
        _kv_dim: usize,
        _seq_len: usize,
        _num_q_heads: usize,
        _num_kv_heads: usize,
        _head_dim: usize,
        _rope_base: f32,
        _use_qk_norm: bool,
        _softcap: f32,
    ) -> Option<Vec<f32>> {
        None
    }
}
