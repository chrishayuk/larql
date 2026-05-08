//! Hybrid pipeline — GPU attention + vindex walk FFN.
//!
//! Replaces the dense GPU FFN (13.6ms for 34 layers) with vindex walk FFN
//! (~1ms/layer via gate KNN + sparse mmap accumulation).
//!
//! Pipeline per layer:
//!   1. GPU: norm → QKV → RoPE → KV cache → attend → O proj → residual
//!   2. CPU: pre-FFN norm → walk FFN (gate KNN → sparse down) → residual add
//!
//! Requires `--features metal` for GPU attention.

use super::CachedLayerGraph;
#[allow(unused_imports)]
use super::LayerGraph;
use crate::model::ModelWeights;
use larql_compute::prelude::*;

/// Hybrid decode: GPU attention + vindex walk FFN per layer.
///
/// Falls back to `predict_honest` if Metal is unavailable or walk data is missing.
#[allow(clippy::too_many_arguments)]
pub fn predict_hybrid(
    weights: &ModelWeights,
    tokenizer: &tokenizers::Tokenizer,
    token_ids: &[u32],
    top_k: usize,
    index: &larql_vindex::VectorIndex,
    backend: &dyn ComputeBackend,
    cached_layers: &CachedLayerGraph,
    layer_range: std::ops::Range<usize>,
) -> crate::forward::PredictResult {
    // Try the Metal hybrid path
    #[cfg(all(feature = "metal", target_os = "macos"))]
    {
        if let Some(result) = predict_hybrid_metal(
            weights,
            tokenizer,
            token_ids,
            top_k,
            index,
            backend,
            cached_layers,
            &layer_range,
        ) {
            return result;
        }
    }

    // Fallback: predict_honest (GPU decode_token with dense FFN)
    super::predict::predict_honest(
        weights,
        tokenizer,
        token_ids,
        top_k,
        index,
        backend,
        cached_layers,
        layer_range,
    )
}

/// Metal-specific hybrid implementation.
#[cfg(all(feature = "metal", target_os = "macos"))]
#[allow(clippy::too_many_arguments)]
fn predict_hybrid_metal(
    weights: &ModelWeights,
    tokenizer: &tokenizers::Tokenizer,
    token_ids: &[u32],
    top_k: usize,
    index: &larql_vindex::VectorIndex,
    backend: &dyn ComputeBackend,
    cached_layers: &CachedLayerGraph,
    layer_range: &std::ops::Range<usize>,
) -> Option<crate::forward::PredictResult> {
    // Check: Metal backend?
    let metal = backend
        .as_any()
        .downcast_ref::<larql_compute::metal::MetalBackend>()?;

    // Check: walk data available?
    let gate_index: &dyn larql_vindex::GateIndex = index;
    if !gate_index.has_down_features() {
        return None;
    }

    // Check: attention weights available?
    let has_attn = index.attn_q4k_layer_data(layer_range.start).is_some()
        || index.attn_q8_layer_data(layer_range.start).is_some();
    if !has_attn {
        return None;
    }

    let norm_offset = weights.arch.norm_weight_offset();
    let hidden = weights.hidden_size;
    // Build attention-only layer descriptors (FFN weights are dummies)
    let dummy = larql_compute::QuantWeight {
        data: &[],
        scales: None,
        format: larql_compute::QuantFormat::Q4_0,
    };
    let attn_layers: Vec<larql_compute::FullPipelineLayer> = layer_range
        .clone()
        .map(|layer| {
            let (wq, wk, wv, wo) = super::pipeline_layer::resolve_attn_weights(index, layer)
                .expect("No attention weights");
            super::pipeline_layer::build_arch_params(
                weights, layer, wq, wk, wv, wo, dummy, dummy, dummy,
            )
        })
        .collect();
    let kv_shapes: Vec<(usize, usize)> = (0..weights.num_layers)
        .map(|layer| {
            (
                weights.arch.num_kv_heads_for_layer(layer),
                weights.arch.head_dim_for_layer(layer),
            )
        })
        .collect();

    // ── Phase 0: Cached layers (template-fixed) ──
    let mut h = crate::forward::embed_tokens_pub(weights, token_ids);
    for layer in 0..layer_range.start {
        if let Some(output) = cached_layers.forward_layer(weights, &h, layer) {
            h = output.residual;
        }
    }

    // Populate KV cache for cached layers
    backend.reset_kv_cache();
    super::prefill::prefill_kv_cache_cpu(
        weights,
        token_ids,
        index,
        backend,
        &(0..layer_range.start),
    );

    // ── Phase 1: Hybrid GPU attention + CPU walk FFN ──
    let walk_ffn = crate::vindex::WalkFfn::new_unlimited_with_backend(weights, index, backend);

    for (rel_idx, abs_layer) in layer_range.clone().enumerate() {
        let x_vec: Vec<f32> = h.row(h.shape()[0] - 1).to_vec();

        // GPU: attention only
        let h_post_attn_vec = {
            let layer = &attn_layers[rel_idx];
            let layer_q_dim = layer.num_q_heads * layer.head_dim;
            let layer_kv_dim = layer.num_kv_heads * layer.head_dim;
            let mut cache_guard = metal.kv_cache_mut_for_shapes(&kv_shapes);
            let kv_cache = cache_guard.as_mut().unwrap();
            metal.decode_attention_layer(
                kv_cache,
                layer,
                abs_layer,
                &x_vec,
                hidden,
                layer_q_dim,
                layer_kv_dim,
            )
        };

        // CPU: walk FFN
        let h_post_attn = ndarray::Array2::from_shape_vec((1, hidden), h_post_attn_vec)
            .unwrap_or_else(|_| {
                h.slice(ndarray::s![h.shape()[0] - 1..h.shape()[0], ..])
                    .to_owned()
            });

        let (h_post_ffn, _) =
            crate::forward::run_ffn(weights, &h_post_attn, abs_layer, &walk_ffn, false);
        h = h_post_ffn;
    }

    // ── Phase 2: Logits ──
    Some(super::logits::finalize_logits(
        weights,
        tokenizer,
        &h,
        top_k,
        index,
        backend,
        norm_offset,
    ))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::engines::test_utils::{make_test_tokenizer, make_test_vindex, make_test_weights};
    use crate::layer_graph::CachedLayerGraph;
    use larql_compute::CpuBackend;

    #[test]
    fn predict_hybrid_runs_with_empty_cache() {
        let weights = make_test_weights();
        let tokenizer = make_test_tokenizer(weights.vocab_size);
        let index = make_test_vindex(&weights);
        let cached = CachedLayerGraph::from_residuals(vec![]);
        let num_layers = weights.num_layers;
        let result = predict_hybrid(
            &weights,
            &tokenizer,
            &[0u32, 1],
            3,
            &index,
            &CpuBackend,
            &cached,
            0..num_layers,
        );
        assert!(result.token_ids.len() <= 3);
    }

    #[test]
    fn predict_hybrid_with_partial_cache() {
        use crate::ffn::WeightFfn;
        let weights = make_test_weights();
        let tokenizer = make_test_tokenizer(weights.vocab_size);
        let index = make_test_vindex(&weights);
        let ffn = WeightFfn { weights: &weights };
        let cached = CachedLayerGraph::build(&weights, &[0u32], &[0], &ffn);
        let num_layers = weights.num_layers;
        let result = predict_hybrid(
            &weights,
            &tokenizer,
            &[0u32, 1],
            2,
            &index,
            &CpuBackend,
            &cached,
            0..num_layers,
        );
        assert!(result.token_ids.len() <= 2);
    }
}
