//! Decode-step attention — GQA for a single new token against a
//! growing KV cache.
//!
//! Prefill does full O(seq²) attention and returns K/V per layer. Decode
//! runs one token at a time with O(cached_len) attention: Q for the new
//! token attends against [K_cache | K_new] and [V_cache | V_new], with
//! no causal mask needed (the new query is at the end and can see every
//! cached position + itself).
//!
//! See `predict::generate_cached` for the prefill→decode driver.

use ndarray::Array2;

use super::rope::apply_rope_partial_at;
use super::SharedKV;

/// Per-layer K/V cache. Can grow unbounded or be clamped to a fixed
/// sliding window (Markov-residual-bounded strategy — keep the last W
/// positions' K/V, evict older). When bounded, attention becomes
/// "look at the last W tokens" — identical to StreamingLLM / sliding
/// window approaches.
///
/// Memory: O(num_layers × window × kv_dim × 4 bytes) when bounded,
/// O(num_layers × seq_len × kv_dim × 4 bytes) when unbounded.
#[derive(Clone, Debug, Default)]
pub struct KvCache {
    /// One entry per layer. `None` for layers that reuse another
    /// layer's K/V (Gemma 4 cross-layer sharing).
    pub layers: Vec<Option<SharedKV>>,
    /// When `Some(W)`, each layer's K/V is clipped to the last W
    /// positions after every append — the "bounded" part of the
    /// Markov Residual Bounded strategy. `None` = unbounded growth.
    pub max_window: Option<usize>,
    /// Absolute token position of the NEXT token to be appended.
    /// Used for RoPE: a new token's K needs RoPE at its true absolute
    /// position, not its row index in the clipped cache. Starts at 0
    /// and increments per append (not per eviction).
    pub next_position: usize,

    // ── RotorQuant compressed side-table ──────────────────────────
    //
    // Optional parallel storage for layers that have been moved into
    // RotorQuant compression via `quantize_layer`. The FP32 `layers`
    // entry is taken `None` once a layer is compressed; reading it
    // back via `dequantize_layer` reconstructs FP32 on demand.
    // This is an additive integration — the FP32 path stays the
    // canonical fast path; compressed storage is opt-in per layer
    // (the upstream "deferred-K" pattern).
    /// Compression format active for this cache. `None` = nothing
    /// compressed. Mixing formats across layers is unsupported (the
    /// parent design says session-level format selection).
    pub kv_format: Option<larql_rotorquant::KvFormat>,
    /// Compressed (K, V) per layer. `None` for layers that haven't
    /// been compressed; `Some` after `quantize_layer` succeeds.
    pub quantized_kv: Vec<Option<(larql_rotorquant::QuantizedKv, larql_rotorquant::QuantizedKv)>>,
    /// Total number of auto-promotes performed by `get_layer` since
    /// construction. Each successful `dequantize_layer` triggered
    /// by a compressed-only read increments this counter; reads
    /// that already had FP32 in the slot do NOT increment.
    /// `rotorquant-promote-on-read` change.
    pub promote_on_read_count: u64,
}

impl KvCache {
    fn new(
        num_layers: usize,
        max_window: Option<usize>,
        kv_format: Option<larql_rotorquant::KvFormat>,
    ) -> Self {
        Self {
            layers: vec![None; num_layers],
            max_window,
            next_position: 0,
            kv_format,
            quantized_kv: vec![None; num_layers],
            promote_on_read_count: 0,
        }
    }

    /// Unbounded cache — grows with every decode step.
    pub fn with_layers(num_layers: usize) -> Self {
        Self::new(num_layers, None, None)
    }

    /// Unbounded cache with a fixed KV compression format.
    pub fn with_layers_format(
        num_layers: usize,
        kv_format: Option<larql_rotorquant::KvFormat>,
    ) -> Self {
        Self::new(num_layers, None, kv_format)
    }

    /// Bounded (Markov-residual-bounded) — keeps only the last
    /// `window` positions per layer. Memory stays O(window).
    pub fn with_window(num_layers: usize, window: usize) -> Self {
        Self::with_window_format(num_layers, window, None)
    }

    /// Bounded cache with a fixed KV compression format.
    pub fn with_window_format(
        num_layers: usize,
        window: usize,
        kv_format: Option<larql_rotorquant::KvFormat>,
    ) -> Self {
        Self::new(
            num_layers,
            if window == 0 { None } else { Some(window) },
            kv_format,
        )
    }

    // ── RotorQuant compressed side-table ──────────────────────────

    /// Set the active KV compression format for the cache. Must be
    /// called before `quantize_layer`. Mixing formats across layers
    /// in the same cache instance is unsupported.
    pub fn set_kv_format(&mut self, format: larql_rotorquant::KvFormat) {
        assert!(
            self.try_set_kv_format(format),
            "KvCache format is fixed after construction; allocate a fresh cache to change it"
        );
    }

    /// Try to set the active compression format. Returns `false` if
    /// the cache already has a different format.
    pub fn try_set_kv_format(&mut self, format: larql_rotorquant::KvFormat) -> bool {
        match self.kv_format {
            Some(existing) if existing != format => false,
            Some(_) => true,
            None => {
                self.kv_format = Some(format);
                true
            }
        }
    }

    /// Compress an FP32 layer into the active KV format. The FP32
    /// slot is taken (set to `None`) so the cache memory does not
    /// double. No-op (returns `false`) if the layer is empty, out of
    /// range, or `kv_format` is unset.
    pub fn quantize_layer(&mut self, layer: usize) -> bool {
        let Some(format) = self.kv_format else {
            return false;
        };
        let Some(slot) = self.layers.get_mut(layer) else {
            return false;
        };
        let Some((k, v)) = slot.take() else {
            return false;
        };
        let n_rows = k.shape()[0];
        let head_dim = k.shape()[1];
        let k_flat = k.into_raw_vec_and_offset().0;
        let v_flat = v.into_raw_vec_and_offset().0;
        let qk = match larql_rotorquant::quantize_k(format, &k_flat, n_rows, head_dim) {
            Ok(q) => q,
            Err(_) => {
                // Restore on failure so the layer isn't silently lost.
                let k = ndarray::Array2::from_shape_vec((n_rows, head_dim), k_flat)
                    .expect("shape known");
                let v = ndarray::Array2::from_shape_vec((n_rows, head_dim), v_flat)
                    .expect("shape known");
                self.layers[layer] = Some((k, v));
                return false;
            }
        };
        let qv = match larql_rotorquant::quantize_v(format, &v_flat, n_rows, head_dim) {
            Ok(q) => q,
            Err(_) => {
                // K already consumed; can't recover the full FP32 pair
                // cleanly. Leave the layer empty and return failure.
                self.layers[layer] = None;
                return false;
            }
        };
        if let Some(slot) = self.quantized_kv.get_mut(layer) {
            *slot = Some((qk, qv));
        }
        true
    }

    /// Decompress a compressed layer back to FP32 SharedKV form.
    /// Returns the decompressed pair without disturbing the
    /// compressed side-table. `None` if the layer isn't compressed
    /// or out of range.
    pub fn dequantize_layer(&self, layer: usize) -> Option<SharedKV> {
        let (qk, qv) = self.quantized_kv.get(layer)?.as_ref()?;
        let k_flat = larql_rotorquant::dequantize_k(qk).ok()?;
        let v_flat = larql_rotorquant::dequantize_v_with_inverse_rotation(qv).ok()?;
        let k = ndarray::Array2::from_shape_vec((qk.n_rows, qk.head_dim), k_flat).ok()?;
        let v = ndarray::Array2::from_shape_vec((qv.n_rows, qv.head_dim), v_flat).ok()?;
        Some((k, v))
    }

    /// Move a compressed layer back into the FP32 `layers` slot
    /// (and clear the compressed side-table for that layer). Inverse
    /// of `quantize_layer`.
    pub fn promote_layer_to_fp32(&mut self, layer: usize) -> bool {
        let Some(kv) = self.dequantize_layer(layer) else {
            return false;
        };
        if let Some(slot) = self.layers.get_mut(layer) {
            *slot = Some(kv);
        }
        if let Some(qslot) = self.quantized_kv.get_mut(layer) {
            *qslot = None;
        }
        true
    }

    /// Whether the layer is currently compressed (backed by
    /// `quantized_kv` rather than `layers`).
    pub fn is_layer_compressed(&self, layer: usize) -> bool {
        self.quantized_kv
            .get(layer)
            .map(|opt| opt.is_some())
            .unwrap_or(false)
    }

    /// Number of cached positions for a given layer. Returns 0 if the
    /// layer has no cache yet.
    pub fn cached_len(&self, layer: usize) -> usize {
        self.layers
            .get(layer)
            .and_then(|opt| opt.as_ref())
            .map(|(k, _)| k.shape()[0])
            .unwrap_or(0)
    }

    /// Apply the window bound to a layer's cache: if the cache has more
    /// than `max_window` rows, drop the oldest rows (keeping the tail).
    /// No-op when unbounded or under the limit.
    pub fn clip_layer(&mut self, layer: usize) {
        let window = match self.max_window {
            Some(w) => w,
            None => return,
        };
        let Some(Some((k, v))) = self.layers.get_mut(layer) else {
            return;
        };
        let rows = k.shape()[0];
        if rows <= window {
            return;
        }
        let start = rows - window;
        let k_slice = k.slice(ndarray::s![start..rows, ..]).to_owned();
        let v_slice = v.slice(ndarray::s![start..rows, ..]).to_owned();
        *k = k_slice;
        *v = v_slice;
    }

    // ── KV surgery ──────────────────────────────────────────────────────────
    //
    // Lazarus's `prefill_inject` and `kv_inject_test` need to lift K/V from
    // one cache into another. The fields are pub so callers could reach in,
    // but these methods give a stable, documented API and handle the
    // `Vec<Option<_>>` indexing in one place.

    /// Read K/V for a layer (post-RoPE K, post-V-norm V). `None` if the
    /// layer index is out of range or that layer's cache is empty (e.g.
    /// before prefill, or when the layer reuses another layer's K/V).
    ///
    /// As of `rotorquant-promote-on-read`: if the FP32 slot is empty
    /// AND `quantized_kv[layer]` holds a compressed copy, this method
    /// transparently dequantises and populates the FP32 slot before
    /// returning. The compressed slot stays populated so the next
    /// `quantize_layer` call (e.g., from the auto-compress engine
    /// decorator) doesn't have to re-encode unchanged data.
    /// `promote_on_read_count` increments on each such auto-promote
    /// (not on cache hits).
    ///
    /// Use [`Self::get_layer_lazy`] for the explicit-no-promote variant
    /// (snapshot serialisation, diagnostics).
    pub fn get_layer(&mut self, layer: usize) -> Option<&SharedKV> {
        // Fast path: FP32 slot already populated.
        if self.layers.get(layer).map(|s| s.is_some()).unwrap_or(false) {
            return self.layers.get(layer).and_then(|opt| opt.as_ref());
        }
        // Slow path: try to promote from compressed storage.
        let promoted = self.dequantize_layer(layer);
        if let Some(kv) = promoted {
            if let Some(slot) = self.layers.get_mut(layer) {
                *slot = Some(kv);
                self.promote_on_read_count = self.promote_on_read_count.saturating_add(1);
            }
        }
        self.layers.get(layer).and_then(|opt| opt.as_ref())
    }

    /// Explicit no-side-effect variant of [`Self::get_layer`].
    ///
    /// Returns `None` for layers that exist only in compressed
    /// storage; **does not** promote them. Snapshot serialisation
    /// and read-only diagnostics use this variant. Production
    /// attention reads should use [`Self::get_layer`] which
    /// auto-promotes.
    pub fn get_layer_lazy(&self, layer: usize) -> Option<&SharedKV> {
        self.layers.get(layer).and_then(|opt| opt.as_ref())
    }

    /// Overwrite K/V for a layer with the supplied tensors. `K` and `V`
    /// must have the same row count. Caller is responsible for the rows
    /// being post-RoPE / post-V-norm — surgery happens at the same stage
    /// the forward pass writes.
    pub fn set_layer(&mut self, layer: usize, kv: SharedKV) {
        if layer >= self.layers.len() {
            return;
        }
        debug_assert_eq!(
            kv.0.shape()[0],
            kv.1.shape()[0],
            "K and V must have the same row count"
        );
        self.layers[layer] = Some(kv);
    }

    /// Clear a layer's cache. Subsequent decode at that layer will start
    /// fresh — i.e. attend only to new K/V. Symmetric: clears both the
    /// FP32 `layers[layer]` slot AND the compressed
    /// `quantized_kv[layer]` slot so a stale compressed copy doesn't
    /// outlive a clear.
    pub fn clear_layer(&mut self, layer: usize) {
        if let Some(slot) = self.layers.get_mut(layer) {
            *slot = None;
        }
        if let Some(qslot) = self.quantized_kv.get_mut(layer) {
            *qslot = None;
        }
    }

    /// Lift `other`'s entire K/V for `layer` into `self`. No-op if either
    /// side's layer is empty or out of range. Implements lazarus
    /// `kv_inject_test` (full-layer transplant).
    ///
    /// Uses [`Self::get_layer_lazy`] on the donor (immutable
    /// `&KvCache`) so a compressed-only donor layer is treated as
    /// "missing." Callers that want to lift a compressed layer
    /// should call `other.promote_layer_to_fp32(layer)` first.
    pub fn clone_layer_from(&mut self, other: &KvCache, layer: usize) {
        let Some((k, v)) = other.get_layer_lazy(layer) else {
            return;
        };
        self.set_layer(layer, (k.clone(), v.clone()));
    }

    /// Lift positions `[start..end]` of `other`'s `layer` K/V into `self`.
    /// Replaces `self`'s entire layer cache with the slice (it does not
    /// merge — concatenation/merge is the caller's job because each
    /// engine has its own append semantics).
    ///
    /// `start` is clamped to the donor's cache length; `end` is clamped
    /// to one past the last cached position. No-op if the resulting
    /// slice is empty or the donor's layer is missing.
    ///
    /// Implements lazarus `prefill_inject` (partial position transplant).
    pub fn clone_layer_position_range(
        &mut self,
        other: &KvCache,
        layer: usize,
        start: usize,
        end: usize,
    ) {
        let Some((k, v)) = other.get_layer_lazy(layer) else {
            return;
        };
        let cached = k.shape()[0];
        let s = start.min(cached);
        let e = end.min(cached);
        if s >= e {
            return;
        }
        let k_slice = k.slice(ndarray::s![s..e, ..]).to_owned();
        let v_slice = v.slice(ndarray::s![s..e, ..]).to_owned();
        self.set_layer(layer, (k_slice, v_slice));
    }
}

/// GQA attention for a single decode step.
///
/// `q_new`: `[1, num_q * head_dim]` — Q for the new token only.
/// `k_full`: `[total_len, num_kv * head_dim]` — K_cache concatenated
/// with the new token's K_rope. Same for `v_full`.
///
/// Returns `[1, num_q * head_dim]` attention output for the new token.
/// No causal mask — the new token naturally sees everything, and the
/// cache only grew by 1 at the end.
#[allow(clippy::too_many_arguments)]
pub fn gqa_attention_decode_step(
    q_new: &Array2<f32>,
    k_full: &Array2<f32>,
    v_full: &Array2<f32>,
    num_q: usize,
    head_dim: usize,
    reps: usize,
    scale: f64,
    softcap: Option<f32>,
) -> Array2<f32> {
    let total_len = k_full.shape()[0];
    let mut out = Array2::<f32>::zeros((1, num_q * head_dim));
    let scale_f32 = scale as f32;

    let mut scores = vec![0.0f32; total_len];
    for h in 0..num_q {
        let kv_h = h / reps;
        let q_off = h * head_dim;
        let kv_off = kv_h * head_dim;

        let q_row = q_new.slice(ndarray::s![0, q_off..q_off + head_dim]);
        let k_block = k_full.slice(ndarray::s![.., kv_off..kv_off + head_dim]);
        let raw: ndarray::Array1<f32> = k_block.dot(&q_row);
        for i in 0..total_len {
            let mut s = raw[i] * scale_f32;
            if let Some(cap) = softcap {
                s = (s / cap).tanh() * cap;
            }
            scores[i] = s;
        }
        // Softmax
        let max_val = scores.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        let mut sum = 0.0f64;
        for s in scores.iter_mut() {
            let e = ((*s - max_val) as f64).exp();
            *s = e as f32;
            sum += e;
        }
        let inv_sum = (1.0 / sum) as f32;
        for s in scores.iter_mut() {
            *s *= inv_sum;
        }
        // Weighted sum of V
        let v_block = v_full.slice(ndarray::s![.., kv_off..kv_off + head_dim]);
        let scores_view = ndarray::ArrayView1::from(&scores[..]);
        let weighted_v = v_block.t().dot(&scores_view);
        for d in 0..head_dim {
            out[[0, q_off + d]] = weighted_v[d];
        }
    }
    out
}

/// Run the attention block for one decode step using an incremental KV
/// cache. `h_new` is the `[1, hidden]` residual for the new token.
/// `kv_entry` is the layer's existing `(K_cache, V_cache)` or `None` on
/// first step. `abs_position` is the new token's absolute RoPE
/// position — the caller must pass its true position in the original
/// sequence, NOT the clipped cache length (those differ under a
/// sliding window). Returns the updated `(h_post_attn, new_kv)`.
///
/// CPU-only variant. For GPU projections use
/// [`run_attention_block_decode_step_backend`].
#[allow(clippy::too_many_arguments)]
#[allow(clippy::type_complexity)]
pub fn run_attention_block_decode_step(
    weights: &crate::model::ModelWeights,
    h_new: &Array2<f32>,
    layer: usize,
    kv_entry: Option<&SharedKV>,
    abs_position: usize,
) -> Option<(Array2<f32>, SharedKV)> {
    run_attention_block_decode_step_backend(weights, h_new, layer, kv_entry, abs_position, None)
}

/// Decode-step attention with optional GPU-accelerated projections
/// (Q/K/V/O matmuls route through `ComputeBackend::matmul_transb` when
/// `backend` is `Some`). GQA softmax + weighted-V stays on CPU —
/// that's O(cached_len × head_dim × num_q) per step and rarely the
/// bottleneck vs the hidden×hidden projection gemms.
#[allow(clippy::too_many_arguments)]
#[allow(clippy::type_complexity)]
pub fn run_attention_block_decode_step_backend(
    weights: &crate::model::ModelWeights,
    h_new: &Array2<f32>,
    layer: usize,
    kv_entry: Option<&SharedKV>,
    abs_position: usize,
    backend: Option<&dyn larql_compute::ComputeBackend>,
) -> Option<(Array2<f32>, SharedKV)> {
    use crate::forward::add_bias;
    use crate::residual::{rms_norm_heads, rms_norm_heads_no_weight};
    use larql_compute::dot_proj_gpu;

    let arch = &*weights.arch;
    let head_dim = arch.head_dim_for_layer(layer);
    let num_q = arch.num_q_heads_for_layer(layer);
    let num_kv = arch.num_kv_heads_for_layer(layer);
    let reps = num_q / num_kv;
    let scale = if arch.attention_multiplier() != 1.0 {
        arch.attention_multiplier() as f64
    } else {
        arch.attention_scale_for_layer(layer)
    };
    let norm_offset = arch.norm_weight_offset();
    let position = abs_position;

    let h_norm = crate::forward::apply_norm(
        weights,
        h_new,
        &arch.input_layernorm_key(layer),
        norm_offset,
    );

    let w_q = weights.tensors.get(&arch.attn_q_key(layer))?;
    let w_o = weights.tensors.get(&arch.attn_o_key(layer))?;
    let mut q_full = dot_proj_gpu(&h_norm, w_q, backend);
    if let Some(bias) = arch
        .attn_q_bias_key(layer)
        .and_then(|k| weights.vectors.get(&k))
    {
        add_bias(&mut q_full, bias);
    }

    let qk_offset = weights.arch.qk_norm_weight_offset();
    let qk_norm_off = if qk_offset != 0.0 {
        qk_offset
    } else {
        norm_offset
    };
    let q_normed = match arch
        .attn_q_norm_key(layer)
        .and_then(|k| weights.vectors.get(&k))
    {
        Some(norm_w) => rms_norm_heads(&q_full, norm_w, num_q, head_dim, qk_norm_off),
        None => q_full,
    };
    let layer_rope_base = arch.rope_base_for_layer(layer);
    let rotary_frac = arch.rotary_fraction_for_layer(layer);
    let q_rope = apply_rope_partial_at(
        &q_normed,
        num_q,
        head_dim,
        layer_rope_base,
        rotary_frac,
        position,
    );

    // New token's K, V — RoPE'd at `position`, then appended to cache.
    let w_k = weights.tensors.get(&arch.attn_k_key(layer))?;
    let v_from_k = !weights.tensors.contains_key(&arch.attn_v_key(layer));
    let w_v = if v_from_k {
        w_k
    } else {
        weights.tensors.get(&arch.attn_v_key(layer))?
    };

    let mut k_full_new = dot_proj_gpu(&h_norm, w_k, backend);
    let mut v_full_new = dot_proj_gpu(&h_norm, w_v, backend);
    if let Some(bias) = arch
        .attn_k_bias_key(layer)
        .and_then(|k| weights.vectors.get(&k))
    {
        add_bias(&mut k_full_new, bias);
    }
    if let Some(bias) = arch
        .attn_v_bias_key(layer)
        .and_then(|k| weights.vectors.get(&k))
    {
        add_bias(&mut v_full_new, bias);
    }
    if arch.has_v_norm() {
        v_full_new = rms_norm_heads_no_weight(&v_full_new, num_kv, head_dim);
    }
    let k_normed = match arch
        .attn_k_norm_key(layer)
        .and_then(|k| weights.vectors.get(&k))
    {
        Some(norm_w) => rms_norm_heads(&k_full_new, norm_w, num_kv, head_dim, qk_norm_off),
        None => k_full_new,
    };
    let k_new_rope = apply_rope_partial_at(
        &k_normed,
        num_kv,
        head_dim,
        layer_rope_base,
        rotary_frac,
        position,
    );

    // Concatenate cache + new along seq axis.
    let (k_concat, v_concat) = match kv_entry {
        Some((k_cached, v_cached)) => {
            let kv_dim = num_kv * head_dim;
            let total = k_cached.shape()[0] + 1;
            let mut k_out = Array2::<f32>::zeros((total, kv_dim));
            let mut v_out = Array2::<f32>::zeros((total, kv_dim));
            k_out
                .slice_mut(ndarray::s![..k_cached.shape()[0], ..])
                .assign(k_cached);
            v_out
                .slice_mut(ndarray::s![..v_cached.shape()[0], ..])
                .assign(v_cached);
            k_out
                .slice_mut(ndarray::s![k_cached.shape()[0].., ..])
                .assign(&k_new_rope);
            v_out
                .slice_mut(ndarray::s![v_cached.shape()[0].., ..])
                .assign(&v_full_new);
            (k_out, v_out)
        }
        None => (k_new_rope, v_full_new),
    };

    let softcap = arch.attn_logit_softcapping();
    let attn_out = gqa_attention_decode_step(
        &q_rope, &k_concat, &v_concat, num_q, head_dim, reps, scale, softcap,
    );

    let mut attn_projected = dot_proj_gpu(&attn_out, w_o, backend);
    if let Some(bias) = arch
        .attn_o_bias_key(layer)
        .and_then(|k| weights.vectors.get(&k))
    {
        add_bias(&mut attn_projected, bias);
    }

    let res_mult = arch.residual_multiplier();
    let h_post_attn = if arch.has_post_norms() {
        let normed = crate::forward::apply_norm(
            weights,
            &attn_projected,
            &arch.post_attention_layernorm_key(layer),
            norm_offset,
        );
        if res_mult != 1.0 {
            h_new + &(&normed * res_mult)
        } else {
            h_new + &normed
        }
    } else if res_mult != 1.0 {
        h_new + &(&attn_projected * res_mult)
    } else {
        h_new + &attn_projected
    };

    Some((h_post_attn, (k_concat, v_concat)))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::engines::test_utils::make_test_weights;
    use ndarray::Array2;

    // ── KvCache ───────────────────────────────────────────────────────────────

    #[test]
    fn kv_cache_starts_empty() {
        let mut cache = KvCache::with_layers(4);
        assert_eq!(cache.cached_len(0), 0);
        assert_eq!(cache.next_position, 0);
    }

    #[test]
    fn kv_cache_with_window_clips() {
        let kv_dim = 4usize;
        let mut cache = KvCache::with_window(1, 2);
        // Feed 3 entries into layer 0
        for step in 0..3usize {
            let k = Array2::from_elem((1, kv_dim), step as f32);
            let v = Array2::from_elem((1, kv_dim), step as f32);
            let prior = cache.layers[0].take();
            let new_kv = if let Some((pk, pv)) = prior {
                let mut nk = Array2::zeros((pk.shape()[0] + 1, kv_dim));
                nk.slice_mut(ndarray::s![..pk.shape()[0], ..]).assign(&pk);
                nk.slice_mut(ndarray::s![pk.shape()[0].., ..]).assign(&k);
                let mut nv = Array2::zeros((pv.shape()[0] + 1, kv_dim));
                nv.slice_mut(ndarray::s![..pv.shape()[0], ..]).assign(&pv);
                nv.slice_mut(ndarray::s![pv.shape()[0].., ..]).assign(&v);
                (nk, nv)
            } else {
                (k, v)
            };
            cache.layers[0] = Some(new_kv);
            cache.clip_layer(0);
        }
        assert!(cache.cached_len(0) <= 2, "window=2 should cap at 2 entries");
    }

    #[test]
    fn kv_cache_format_is_fixed_at_construction() {
        let mut cache = KvCache::with_layers_format(2, Some(larql_rotorquant::KvFormat::Iso3));
        assert_eq!(cache.kv_format, Some(larql_rotorquant::KvFormat::Iso3));
        assert!(cache.try_set_kv_format(larql_rotorquant::KvFormat::Iso3));
        assert!(!cache.try_set_kv_format(larql_rotorquant::KvFormat::Planar3));

        let windowed = KvCache::with_window_format(1, 8, Some(larql_rotorquant::KvFormat::Planar3));
        assert_eq!(windowed.max_window, Some(8));
        assert_eq!(
            windowed.kv_format,
            Some(larql_rotorquant::KvFormat::Planar3)
        );
    }

    // ── KV surgery (get / set / clear / clone) ────────────────────────────────

    fn fill_kv(layer_rows: usize, kv_dim: usize, fill: f32) -> SharedKV {
        let k = Array2::from_elem((layer_rows, kv_dim), fill);
        let v = Array2::from_elem((layer_rows, kv_dim), fill);
        (k, v)
    }

    #[test]
    fn get_layer_returns_none_when_empty() {
        let mut cache = KvCache::with_layers(2);
        assert!(cache.get_layer(0).is_none());
        assert!(cache.get_layer(99).is_none(), "out-of-range is None");
    }

    #[test]
    fn set_layer_then_get_layer_round_trips() {
        let mut cache = KvCache::with_layers(2);
        cache.set_layer(1, fill_kv(3, 4, 7.0));
        let (k, v) = cache.get_layer(1).expect("layer 1 set");
        assert_eq!(k.shape(), &[3, 4]);
        assert_eq!(v.shape(), &[3, 4]);
        assert_eq!(k[[0, 0]], 7.0);
        assert!(cache.get_layer(0).is_none());
    }

    #[test]
    fn set_layer_out_of_range_is_noop() {
        let mut cache = KvCache::with_layers(2);
        cache.set_layer(99, fill_kv(1, 4, 1.0));
        // No panic, no growth.
        assert_eq!(cache.layers.len(), 2);
    }

    #[test]
    fn clear_layer_removes_kv() {
        let mut cache = KvCache::with_layers(2);
        cache.set_layer(0, fill_kv(2, 4, 1.0));
        assert!(cache.get_layer(0).is_some());
        cache.clear_layer(0);
        assert!(cache.get_layer(0).is_none());
    }

    #[test]
    fn clone_layer_from_copies_donor_kv() {
        let mut donor = KvCache::with_layers(2);
        donor.set_layer(1, fill_kv(4, 6, 2.5));

        let mut recipient = KvCache::with_layers(2);
        recipient.clone_layer_from(&donor, 1);

        let (k, v) = recipient.get_layer(1).unwrap();
        assert_eq!(k.shape(), &[4, 6]);
        assert_eq!(v[[0, 0]], 2.5);
    }

    #[test]
    fn clone_layer_from_missing_donor_layer_is_noop() {
        let donor = KvCache::with_layers(2);
        let mut recipient = KvCache::with_layers(2);
        recipient.set_layer(0, fill_kv(1, 4, 9.0));
        recipient.clone_layer_from(&donor, 0);
        // Recipient's layer 0 should be unchanged because donor had nothing.
        assert_eq!(recipient.get_layer(0).unwrap().0[[0, 0]], 9.0);
    }

    #[test]
    fn clone_layer_position_range_slices_donor() {
        let mut donor = KvCache::with_layers(1);
        // Build a donor with row i = i*1.0 so we can verify the slice.
        let kv_dim = 3usize;
        let k = Array2::from_shape_fn((5, kv_dim), |(r, _)| r as f32);
        let v = Array2::from_shape_fn((5, kv_dim), |(r, _)| r as f32);
        donor.set_layer(0, (k, v));

        let mut recipient = KvCache::with_layers(1);
        recipient.clone_layer_position_range(&donor, 0, 1, 4);
        let (rk, _) = recipient.get_layer(0).unwrap();
        assert_eq!(rk.shape(), &[3, kv_dim]);
        assert_eq!(rk[[0, 0]], 1.0, "first sliced row is donor row 1");
        assert_eq!(rk[[2, 0]], 3.0, "last sliced row is donor row 3");
    }

    #[test]
    fn clone_layer_position_range_clamps_to_donor_length() {
        let mut donor = KvCache::with_layers(1);
        donor.set_layer(0, fill_kv(2, 3, 1.0));
        let mut recipient = KvCache::with_layers(1);
        // Ask for [0..99) — should clamp to [0..2).
        recipient.clone_layer_position_range(&donor, 0, 0, 99);
        let (rk, _) = recipient.get_layer(0).unwrap();
        assert_eq!(rk.shape(), &[2, 3]);
    }

    #[test]
    fn clone_layer_position_range_empty_slice_is_noop() {
        let mut donor = KvCache::with_layers(1);
        donor.set_layer(0, fill_kv(2, 3, 1.0));
        let mut recipient = KvCache::with_layers(1);
        recipient.clone_layer_position_range(&donor, 0, 5, 5);
        assert!(recipient.get_layer(0).is_none(), "empty range -> no write");
    }

    // ── decode step ───────────────────────────────────────────────────────────

    #[test]
    fn decode_step_output_shape() {
        let weights = make_test_weights();
        let h = Array2::from_elem((1, weights.hidden_size), 0.1f32);
        let (h_out, (k, v)) =
            run_attention_block_decode_step(&weights, &h, 0, None, 0).expect("decode_step failed");
        assert_eq!(h_out.shape(), &[1, weights.hidden_size]);
        assert_eq!(k.shape()[0], 1, "K should have 1 new row");
        assert_eq!(v.shape()[0], 1, "V should have 1 new row");
    }

    #[test]
    fn decode_step_output_finite() {
        let weights = make_test_weights();
        let h = Array2::from_elem((1, weights.hidden_size), 0.5f32);
        let (h_out, _) =
            run_attention_block_decode_step(&weights, &h, 0, None, 0).expect("decode_step failed");
        assert!(h_out.iter().all(|v| v.is_finite()));
    }

    #[test]
    fn decode_step_kv_grows_with_prior() {
        let weights = make_test_weights();
        let h = Array2::from_elem((1, weights.hidden_size), 0.1f32);
        // Step 0: no prior
        let (_, kv1) = run_attention_block_decode_step(&weights, &h, 0, None, 0).unwrap();
        assert_eq!(kv1.0.shape()[0], 1);
        // Step 1: prior has 1 entry → output K/V should have 2
        let (_, kv2) = run_attention_block_decode_step(&weights, &h, 0, Some(&kv1), 1).unwrap();
        assert_eq!(kv2.0.shape()[0], 2, "K should grow by 1 per step");
    }

    #[test]
    fn decode_step_all_layers_succeed() {
        let weights = make_test_weights();
        let h = Array2::from_elem((1, weights.hidden_size), 0.3f32);
        for layer in 0..weights.num_layers {
            let result = run_attention_block_decode_step(&weights, &h, layer, None, 0);
            assert!(result.is_some(), "layer {layer} decode step failed");
        }
    }

    // ── RotorQuant compressed side-table ──────────────────────────

    #[test]
    fn quantize_layer_no_op_when_format_unset() {
        let mut cache = KvCache::with_layers(1);
        cache.set_layer(0, fill_kv(4, 32, 0.5));
        // No format yet → quantize_layer returns false and the layer stays FP32.
        assert!(!cache.quantize_layer(0));
        assert!(cache.get_layer(0).is_some());
        assert!(!cache.is_layer_compressed(0));
    }

    #[test]
    fn quantize_then_dequantize_roundtrip_preserves_direction() {
        let mut cache = KvCache::with_layers(2);
        cache.set_kv_format(larql_rotorquant::KvFormat::Iso3);
        // Use head_dim=32 (multiple of 4 for Iso3) and 8 rows so the
        // synthetic data has a well-defined L2 norm.
        let original = fill_kv(8, 32, 0.7);
        cache.set_layer(1, original.clone());

        // Compress.
        assert!(cache.quantize_layer(1));
        assert!(cache.is_layer_compressed(1));
        // FP32 slot was taken so memory doesn't double. Use the
        // lazy variant so we observe the underlying state without
        // triggering auto-promote (rotorquant-promote-on-read).
        assert!(cache.get_layer_lazy(1).is_none());

        // Decompress.
        let recovered = cache.dequantize_layer(1).expect("dequantize");
        assert_eq!(recovered.0.shape(), original.0.shape());
        assert_eq!(recovered.1.shape(), original.1.shape());

        // Cosine similarity ≥ 0.95 (matches larql-rotorquant test threshold).
        let cos = |a: &Array2<f32>, b: &Array2<f32>| -> f32 {
            let av = a.iter().copied().collect::<Vec<_>>();
            let bv = b.iter().copied().collect::<Vec<_>>();
            let dot: f32 = av.iter().zip(&bv).map(|(x, y)| x * y).sum();
            let na: f32 = av.iter().map(|v| v * v).sum::<f32>().sqrt();
            let nb: f32 = bv.iter().map(|v| v * v).sum::<f32>().sqrt();
            dot / (na * nb + 1e-12)
        };
        let cos_k = cos(&original.0, &recovered.0);
        let cos_v = cos(&original.1, &recovered.1);
        assert!(cos_k >= 0.95, "K cosine after roundtrip: {cos_k}");
        assert!(cos_v >= 0.95, "V cosine after roundtrip: {cos_v}");
    }

    #[test]
    fn promote_layer_to_fp32_restores_layers_slot() {
        let mut cache = KvCache::with_layers(1);
        cache.set_kv_format(larql_rotorquant::KvFormat::Iso3);
        cache.set_layer(0, fill_kv(8, 32, 0.5));

        cache.quantize_layer(0);
        assert!(cache.is_layer_compressed(0));
        // Lazy variant: observe FP32-slot truth without auto-promote.
        assert!(cache.get_layer_lazy(0).is_none());

        // Promote restores the FP32 slot and clears compressed.
        assert!(cache.promote_layer_to_fp32(0));
        assert!(!cache.is_layer_compressed(0));
        assert!(cache.get_layer(0).is_some());
    }

    // ── rotorquant-promote-on-read ────────────────────────────────

    #[test]
    fn get_layer_returns_fp32_after_compress() {
        let mut cache = KvCache::with_layers(1);
        cache.set_kv_format(larql_rotorquant::KvFormat::Iso3);
        cache.set_layer(0, fill_kv(8, 32, 0.5));
        cache.quantize_layer(0);
        // FP32 slot was taken; compressed slot is populated.
        // get_layer auto-promotes and returns Some.
        assert!(cache.get_layer(0).is_some());
        assert_eq!(cache.promote_on_read_count, 1);
    }

    #[test]
    fn get_layer_caches_promoted_copy() {
        let mut cache = KvCache::with_layers(1);
        cache.set_kv_format(larql_rotorquant::KvFormat::Iso3);
        cache.set_layer(0, fill_kv(8, 32, 0.7));
        cache.quantize_layer(0);
        let _ = cache.get_layer(0); // first read promotes
        let _ = cache.get_layer(0); // second read hits cached FP32
                                    // Counter increments by exactly 1 (only the first read actually promoted).
        assert_eq!(cache.promote_on_read_count, 1);
    }

    #[test]
    fn get_layer_lazy_never_promotes() {
        let mut cache = KvCache::with_layers(1);
        cache.set_kv_format(larql_rotorquant::KvFormat::Iso3);
        cache.set_layer(0, fill_kv(8, 32, 0.3));
        cache.quantize_layer(0);
        // Lazy variant returns None for compressed-only layers and
        // does NOT mutate cache state.
        assert!(cache.get_layer_lazy(0).is_none());
        assert_eq!(cache.promote_on_read_count, 0);
        assert!(cache.is_layer_compressed(0));
    }

    #[test]
    fn clear_layer_erases_both_storages() {
        let mut cache = KvCache::with_layers(1);
        cache.set_kv_format(larql_rotorquant::KvFormat::Iso3);
        cache.set_layer(0, fill_kv(8, 32, 0.4));
        cache.quantize_layer(0);
        assert!(cache.is_layer_compressed(0));
        cache.clear_layer(0);
        assert!(!cache.is_layer_compressed(0));
        assert!(cache.get_layer(0).is_none());
        assert!(cache.get_layer_lazy(0).is_none());
    }
}
