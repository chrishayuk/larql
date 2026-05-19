//! Q4K attention-weight dequantisation helper.
//!
//! Bridges Q4K vindex data (`VectorIndex::attn_kquant_layer_data`) into
//! `ModelWeights::tensors` as f32 tensors, so `KvDispatch` backends
//! that don't (yet) have native Q4K kernels can fall back to the f32
//! attention path.
//!
//! Lives here (not in `larql-kv`) so the `KvDispatch` trait impls
//! (`CpuBackend`, `MetalBackend` in `crate::kv_dispatch::*`) and the
//! engines that consume them can both reach it without a `larql-kv →
//! larql-inference → larql-kv` cycle.
//!
//! ## Phasing
//!
//! Phase 1 (current): callers invoke this upfront before the
//! `KvDispatch::attention_prefill` loop on a Q4K-loaded `ModelWeights`.
//! Memory cost: all-layer Q/K/V/O f32 tensors stay resident.
//!
//! Phase 3 (future): CpuBackend gains native Q4K matvec via
//! `larql_compute::QuantMatVec::q4k_matvec` per-call; this bulk-dequant
//! helper becomes a debug fallback only.
//!
//! See `docs/specs/kv-dispatch-quantization.md`.

use crate::model::ModelWeights;
use larql_vindex::VectorIndex;
use ndarray::Array2;

/// Dequantise attention Q4K weights (Q, K, V, O) for all layers into
/// `weights.tensors`. Idempotent — skips layers whose `attn_q_key` is
/// already present in `weights.tensors`.
///
/// No-op for layers where `index.attn_kquant_layer_data(layer)` returns
/// `None` (i.e., a layer with non-Q4K attention or no Q4K data at all).
pub fn ensure_attn_tensors_dequantised(weights: &mut ModelWeights, index: &VectorIndex) {
    let num_layers = weights.num_layers;
    for layer in 0..num_layers {
        let arch = &*weights.arch;
        let q_key = arch.attn_q_key(layer);
        if weights.tensors.contains_key(&q_key) {
            continue;
        }
        let Some(attn) = index.attn_kquant_layer_data(layer) else {
            continue;
        };
        let num_q = arch.num_q_heads_for_layer(layer);
        let num_kv = arch.num_kv_heads_for_layer(layer);
        let hd = arch.head_dim_for_layer(layer);
        let hidden = weights.hidden_size;
        let q_dim = num_q * hd;
        let kv_dim = num_kv * hd;
        let k_key = arch.attn_k_key(layer);
        let v_key = arch.attn_v_key(layer);
        let o_key = arch.attn_o_key(layer);
        let w_q = dequantize_matrix(attn[0].0, attn[0].1, q_dim, hidden);
        let w_k = dequantize_matrix(attn[1].0, attn[1].1, kv_dim, hidden);
        let w_v = dequantize_matrix(attn[2].0, attn[2].1, kv_dim, hidden);
        let w_o = dequantize_matrix(attn[3].0, attn[3].1, hidden, q_dim);
        weights.tensors.insert(q_key, w_q.into_shared());
        weights.tensors.insert(k_key, w_k.into_shared());
        weights.tensors.insert(v_key, w_v.into_shared());
        weights.tensors.insert(o_key, w_o.into_shared());
    }
}

fn dequantize_matrix(bytes: &[u8], format: &str, rows: usize, cols: usize) -> Array2<f32> {
    let n = rows * cols;
    let padded = n.div_ceil(256) * 256;
    let info = larql_vindex::quant::registry::lookup(format)
        .unwrap_or_else(|| panic!("unsupported quant format: {format}"));
    let floats =
        (info.dequantize)(bytes, padded).unwrap_or_else(|e| panic!("{format} dequant failed: {e}"));
    let truncated = if floats.len() > n {
        floats[..n].to_vec()
    } else {
        floats
    };
    Array2::from_shape_vec((rows, cols), truncated).expect("shape mismatch")
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_utils::{make_test_q4k_vindex, make_test_q4k_weights};

    /// `ensure_attn_tensors_dequantised` populates every layer's
    /// Q/K/V/O tensors when the vindex carries Q4K attention bytes.
    #[test]
    fn ensure_attn_tensors_populates_qkvo_per_layer() {
        let mut weights = make_test_q4k_weights();
        let index = make_test_q4k_vindex(&weights);
        // Capture per-layer keys upfront so we can drop the &arch
        // borrow before mutating weights.tensors.
        let num_layers = weights.num_layers;
        let keys: Vec<(String, String, String, String)> = (0..num_layers)
            .map(|l| {
                (
                    weights.arch.attn_q_key(l),
                    weights.arch.attn_k_key(l),
                    weights.arch.attn_v_key(l),
                    weights.arch.attn_o_key(l),
                )
            })
            .collect();
        // Strip the f32 attention tensors the synthetic fixture left
        // behind so we exercise the *insert* path, not the
        // already-present short-circuit.
        for (q, k, v, o) in &keys {
            weights.tensors.remove(q);
            weights.tensors.remove(k);
            weights.tensors.remove(v);
            weights.tensors.remove(o);
        }
        ensure_attn_tensors_dequantised(&mut weights, &index);
        for (l, (q, k, v, o)) in keys.iter().enumerate() {
            assert!(weights.tensors.contains_key(q), "Q missing layer {l}");
            assert!(weights.tensors.contains_key(k), "K missing layer {l}");
            assert!(weights.tensors.contains_key(v), "V missing layer {l}");
            assert!(weights.tensors.contains_key(o), "O missing layer {l}");
        }
    }

    /// Idempotent — calling twice doesn't re-dequantise (the
    /// `contains_key` short-circuit fires on the second pass; same
    /// data pointer means the tensor wasn't replaced).
    #[test]
    fn ensure_attn_tensors_is_idempotent() {
        let mut weights = make_test_q4k_weights();
        let index = make_test_q4k_vindex(&weights);
        let q_key = weights.arch.attn_q_key(0);
        ensure_attn_tensors_dequantised(&mut weights, &index);
        let q_ptr_before = weights
            .tensors
            .get(&q_key)
            .expect("Q present after first dequant")
            .as_ptr();
        ensure_attn_tensors_dequantised(&mut weights, &index);
        let q_ptr_after = weights.tensors.get(&q_key).unwrap().as_ptr();
        assert_eq!(
            q_ptr_before, q_ptr_after,
            "idempotent call must not replace the tensor"
        );
    }

    /// No-op when the vindex has no Q4K attention data (the
    /// `attn_kquant_layer_data → None` continue branch).
    #[test]
    fn ensure_attn_tensors_skips_layers_without_q4k_data() {
        let mut weights = make_test_q4k_weights();
        let empty_index = larql_vindex::VectorIndex::new(
            vec![None; weights.num_layers],
            vec![None; weights.num_layers],
            weights.num_layers,
            weights.hidden_size,
        );
        let q_key = weights.arch.attn_q_key(0);
        weights.tensors.remove(&q_key);
        ensure_attn_tensors_dequantised(&mut weights, &empty_index);
        assert!(
            !weights.tensors.contains_key(&q_key),
            "no Q4K data → no insert"
        );
    }
}
