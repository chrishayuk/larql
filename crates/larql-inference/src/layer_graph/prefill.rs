//! KV cache prefill — populate Metal KV cache from CPU attention.

use ndarray::Array2;

use crate::model::ModelWeights;
use larql_compute::prelude::*;

/// Prefill with KV cache population: run CPU attention, capture K/V, populate Metal KV cache.
/// Returns the final hidden state after all layers.
/// After this, `backend.decode_token()` can generate new tokens using the populated cache.
pub fn prefill_with_kv(
    weights: &ModelWeights,
    token_ids: &[u32],
    index: &larql_vindex::VectorIndex,
    backend: &dyn ComputeBackend,
    layer_range: std::ops::Range<usize>,
) -> Array2<f32> {
    let mut h = crate::forward::embed_tokens_pub(weights, token_ids);
    let seq_len = token_ids.len();

    for layer in layer_range {
        let (h_post_attn, k_rope, v) =
            crate::attention::gpu::run_attention_with_kv_backend(weights, &h, layer, Some(backend))
                .unwrap();

        if backend.has_kv_cache() {
            let layer_hd = weights.arch.head_dim_for_layer(layer);
            let layer_nkv = weights.arch.num_kv_heads_for_layer(layer);
            let k_flat = k_rope.as_slice().unwrap_or(&[]);
            let v_flat = v.as_slice().unwrap_or(&[]);
            backend.populate_kv_layer(layer, k_flat, v_flat, seq_len, layer_nkv, layer_hd);
        }

        let walk_ffn = crate::vindex::WalkFfn::new_unlimited(weights, index);
        let (h_out, _) = crate::forward::run_ffn(weights, &h_post_attn, layer, &walk_ffn, false);
        h = h_out;
    }
    h
}

/// Lightweight CPU KV cache population after GPU prefill.
pub(super) fn prefill_kv_cache_cpu(
    weights: &ModelWeights,
    token_ids: &[u32],
    index: &larql_vindex::VectorIndex,
    backend: &dyn ComputeBackend,
    layer_range: &std::ops::Range<usize>,
) {
    if !backend.has_kv_cache() {
        return;
    }
    let _ = prefill_with_kv(weights, token_ids, index, backend, layer_range.clone());
}

#[cfg(all(test, not(target_arch = "wasm32")))]
mod tests {
    use super::*;
    use crate::forward::hidden_to_raw_logits;
    use crate::test_utils::{make_test_vindex, make_test_weights};
    use larql_compute::CpuBackend;
    use larql_models::ModelWeights;
    use std::sync::OnceLock;

    fn weights() -> &'static ModelWeights {
        static W: OnceLock<ModelWeights> = OnceLock::new();
        W.get_or_init(make_test_weights)
    }

    // ── prefill_with_kv ───────────────────────────────────────────────────────

    #[test]
    fn prefill_output_shape_single_token() {
        let w = weights();
        let idx = make_test_vindex(w);
        let h = prefill_with_kv(w, &[0u32], &idx, &CpuBackend, 0..w.num_layers);
        assert_eq!(h.shape(), &[1, w.hidden_size]);
        assert!(h.iter().all(|v| v.is_finite()));
    }

    #[test]
    fn prefill_output_shape_multi_token() {
        let w = weights();
        let idx = make_test_vindex(w);
        let h = prefill_with_kv(w, &[0u32, 1, 2, 3], &idx, &CpuBackend, 0..w.num_layers);
        assert_eq!(h.shape(), &[4, w.hidden_size]);
        assert!(h.iter().all(|v| v.is_finite()));
    }

    #[test]
    fn prefill_partial_layer_range() {
        let w = weights();
        let idx = make_test_vindex(w);
        // Only run layer 0 — returns after one layer, still valid hidden state
        let h = prefill_with_kv(w, &[0u32], &idx, &CpuBackend, 0..1);
        assert_eq!(h.shape(), &[1, w.hidden_size]);
        assert!(h.iter().all(|v| v.is_finite()));
    }

    #[test]
    fn prefill_empty_range_returns_embed() {
        let w = weights();
        let idx = make_test_vindex(w);
        // Empty layer range → returns embeddings unchanged
        let h = prefill_with_kv(w, &[0u32], &idx, &CpuBackend, 0..0);
        assert_eq!(h.shape(), &[1, w.hidden_size]);
    }

    #[test]
    fn prefill_produces_usable_logits() {
        let w = weights();
        let idx = make_test_vindex(w);
        let h = prefill_with_kv(w, &[0u32, 1], &idx, &CpuBackend, 0..w.num_layers);
        let logits = hidden_to_raw_logits(
            w,
            &h.row(1)
                .into_owned()
                .into_shape_with_order((1, w.hidden_size))
                .unwrap(),
        );
        assert!(logits.iter().all(|v| v.is_finite()));
        assert_eq!(logits.len(), w.vocab_size);
    }

    // ── prefill_kv_cache_cpu ──────────────────────────────────────────────────

    #[test]
    fn prefill_kv_cache_cpu_noop_without_kv_cache() {
        // CpuBackend has no KV cache → function returns immediately, no panic
        let w = weights();
        let idx = make_test_vindex(w);
        prefill_kv_cache_cpu(w, &[0u32, 1], &idx, &CpuBackend, &(0..w.num_layers));
        // No assertion needed — the important thing is it doesn't panic
    }
}
