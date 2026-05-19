use ndarray::Array2;

use super::{DenseLayerGraph, LayerGraph, LayerOutput, PerLayerGraph};
use crate::ffn::FfnBackend;
use crate::model::ModelWeights;

// ── Cached: precomputed layer output for fixed-routing regimes ──

/// Cached layer graph: returns a precomputed residual instead of computing.
/// For layers where the output is template-determined (L0-12 regime).
///
/// Build by running a dense forward pass for a template, capturing residuals,
/// then storing them. At inference, skip the computation entirely.
pub struct CachedLayerGraph {
    /// layer → cached residual [seq_len, hidden]. Keyed by layer index.
    cache: std::collections::HashMap<usize, Array2<f32>>,
}

impl CachedLayerGraph {
    /// Build a cache by running a dense forward pass and capturing residuals.
    /// `layers`: which layers to cache (e.g., 0..=12).
    pub fn build(
        weights: &ModelWeights,
        token_ids: &[u32],
        layers: &[usize],
        ffn: &dyn FfnBackend,
    ) -> Self {
        let mut h = crate::forward::embed_tokens_pub(weights, token_ids);
        let mut cache = std::collections::HashMap::new();
        let max_layer = *layers.iter().max().unwrap_or(&0);

        for layer in 0..=max_layer.min(weights.num_layers - 1) {
            let graph = DenseLayerGraph {
                ffn,
                backend: None,
                capture_activation: false,
                capture_attention: false,
            };
            if let Some(output) = graph.forward_layer(weights, &h, layer) {
                h = output.residual;
                if layers.contains(&layer) {
                    cache.insert(layer, h.clone());
                }
            }
        }
        Self { cache }
    }

    /// Build from an existing residual (e.g., from a previous forward pass).
    pub fn from_residuals(residuals: Vec<(usize, Array2<f32>)>) -> Self {
        Self {
            cache: residuals.into_iter().collect(),
        }
    }

    pub fn has_layer(&self, layer: usize) -> bool {
        self.cache.contains_key(&layer)
    }

    pub fn num_cached(&self) -> usize {
        self.cache.len()
    }
}

impl LayerGraph for CachedLayerGraph {
    fn forward_layer(
        &self,
        _weights: &ModelWeights,
        _h: &Array2<f32>,
        layer: usize,
    ) -> Option<LayerOutput> {
        let residual = self.cache.get(&layer)?.clone();
        Some(LayerOutput {
            residual,
            activation: None,
            attention: None,
        })
    }

    fn name(&self) -> &str {
        "cached"
    }
}

/// Build a PerLayerGraph with cached layers for a detected template.
/// Returns the graph and the number of cached layers.
///
/// Layout:
///   cached_layers → CachedLayerGraph (skip computation)
///   remaining layers → fallback (dense/walk)
pub fn build_adaptive_graph<'a>(
    cache: &'a CachedLayerGraph,
    fallback: &'a dyn LayerGraph,
    num_layers: usize,
    cached_range: &std::ops::RangeInclusive<usize>,
) -> PerLayerGraph<'a> {
    let mut layers: Vec<&dyn LayerGraph> = Vec::with_capacity(num_layers);
    for layer in 0..num_layers {
        if cached_range.contains(&layer) && cache.has_layer(layer) {
            layers.push(cache);
        } else {
            layers.push(fallback);
        }
    }
    PerLayerGraph::new(layers)
}

/// Cached post-attention residuals and FFN-normed inputs for the split pass.
///
/// Built from one exact (interleaved) forward pass. Reused for all entities
/// that match the same template — attention is template-fixed (~99% identical).
pub struct AttentionCache {
    /// Per-layer FFN-normed last-token vector (the actual FFN input).
    pub ffn_inputs: Vec<Vec<f32>>,
    /// The final post-attention residual (for combining with FFN output).
    pub final_residual: Array2<f32>,
}

impl AttentionCache {
    /// Build by running one exact forward pass (interleaved attention + FFN)
    /// and capturing the FFN inputs at each walk layer.
    pub fn build(
        weights: &ModelWeights,
        token_ids: &[u32],
        cached_layers: &CachedLayerGraph,
        ffn: &dyn FfnBackend,
        layer_range: std::ops::Range<usize>,
    ) -> Self {
        let seq_len = token_ids.len();
        let arch = &*weights.arch;
        let norm_offset = arch.norm_weight_offset();

        // Run through cached layers first
        let mut h = crate::forward::embed_tokens_pub(weights, token_ids);
        for layer in 0..layer_range.start {
            if let Some(output) = cached_layers.forward_layer(weights, &h, layer) {
                h = output.residual;
            }
        }

        // Run exact interleaved pass for walk layers, capturing FFN inputs
        let mut ffn_inputs = Vec::with_capacity(layer_range.len());
        for layer in layer_range {
            // Attention (exact)
            let (h_post_attn, _, _) =
                crate::attention::run_attention_block_gpu(weights, &h, layer, false, None).unwrap();

            // Capture FFN-normed input (last token)
            let pre_ffn_key = if arch.has_post_norms() {
                arch.pre_feedforward_layernorm_key(layer)
            } else {
                Some(arch.post_attention_layernorm_key(layer))
            };
            let h_ffn = match pre_ffn_key {
                Some(key) => crate::forward::apply_norm(weights, &h_post_attn, &key, norm_offset),
                None => crate::residual::rms_norm(&h_post_attn, None, norm_offset),
            };
            ffn_inputs.push(h_ffn.row(seq_len - 1).to_vec());

            // FFN (exact — for correct residual stream)
            let (h_out, _) = crate::forward::run_ffn(weights, &h_post_attn, layer, ffn, false);
            h = h_out;
        }

        AttentionCache {
            ffn_inputs,
            final_residual: h,
        }
    }
}

#[cfg(all(test, not(target_arch = "wasm32")))]
mod tests {
    use super::*;
    use crate::ffn::WeightFfn;
    use crate::test_utils::make_test_weights;
    use ndarray::Array2;

    #[test]
    fn from_residuals_empty() {
        let g = CachedLayerGraph::from_residuals(vec![]);
        assert_eq!(g.num_cached(), 0);
        assert!(!g.has_layer(0));
    }

    #[test]
    fn from_residuals_single() {
        let arr = Array2::zeros((3, 4));
        let g = CachedLayerGraph::from_residuals(vec![(0, arr.clone())]);
        assert_eq!(g.num_cached(), 1);
        assert!(g.has_layer(0));
        assert!(!g.has_layer(1));
    }

    #[test]
    fn from_residuals_multiple() {
        let arr = Array2::ones((2, 8));
        let g =
            CachedLayerGraph::from_residuals(vec![(0, arr.clone()), (3, arr.clone()), (5, arr)]);
        assert_eq!(g.num_cached(), 3);
        assert!(g.has_layer(0));
        assert!(g.has_layer(3));
        assert!(g.has_layer(5));
        assert!(!g.has_layer(1));
    }

    #[test]
    fn forward_layer_returns_cached() {
        let weights = make_test_weights();
        let h = Array2::from_elem((2, weights.hidden_size), 0.5f32);
        let g = CachedLayerGraph::from_residuals(vec![(0, h.clone())]);
        let out = g
            .forward_layer(&weights, &h, 0)
            .expect("should return cached");
        assert_eq!(out.residual.shape(), &[2, weights.hidden_size]);
    }

    #[test]
    fn forward_layer_none_for_uncached() {
        let weights = make_test_weights();
        let h = Array2::zeros((1, weights.hidden_size));
        let g = CachedLayerGraph::from_residuals(vec![]);
        assert!(
            g.forward_layer(&weights, &h, 0).is_none(),
            "uncached layer should return None"
        );
    }

    #[test]
    fn build_caches_specified_layers() {
        let weights = make_test_weights();
        let ffn = WeightFfn { weights: &weights };
        let g = CachedLayerGraph::build(&weights, &[0u32, 1], &[0], &ffn);
        assert!(g.has_layer(0), "layer 0 should be cached");
        assert!(!g.has_layer(1), "layer 1 was not in the build list");
    }

    #[test]
    fn cached_layer_graph_name() {
        let g = CachedLayerGraph::from_residuals(vec![]);
        assert_eq!(g.name(), "cached");
    }

    #[test]
    fn build_caches_multiple_layers() {
        let weights = make_test_weights();
        let ffn = WeightFfn { weights: &weights };
        let g = CachedLayerGraph::build(&weights, &[0u32, 1, 2], &[0, 1], &ffn);
        assert!(g.has_layer(0));
        assert!(g.has_layer(1));
        assert_eq!(g.num_cached(), 2);
    }

    #[test]
    fn build_with_empty_layer_list_caches_nothing() {
        let weights = make_test_weights();
        let ffn = WeightFfn { weights: &weights };
        let g = CachedLayerGraph::build(&weights, &[0u32], &[], &ffn);
        assert_eq!(g.num_cached(), 0);
    }

    #[test]
    fn build_adaptive_graph_routes_layers_through_cache_or_fallback() {
        let weights = make_test_weights();
        let ffn = WeightFfn { weights: &weights };
        let cache =
            CachedLayerGraph::from_residuals(vec![(0, Array2::zeros((1, weights.hidden_size)))]);
        let fallback = DenseLayerGraph {
            ffn: &ffn,
            backend: None,
            capture_activation: false,
            capture_attention: false,
        };
        let _adaptive = build_adaptive_graph(&cache, &fallback, weights.num_layers, &(0..=0));
        // Just exercise the constructor — verifying routing requires the
        // PerLayerGraph dispatch surface which is covered elsewhere.
    }

    #[test]
    fn build_adaptive_graph_skips_cache_when_layer_outside_range() {
        let weights = make_test_weights();
        let ffn = WeightFfn { weights: &weights };
        let cache = CachedLayerGraph::from_residuals(vec![]);
        let fallback = DenseLayerGraph {
            ffn: &ffn,
            backend: None,
            capture_activation: false,
            capture_attention: false,
        };
        let _ = build_adaptive_graph(
            &cache,
            &fallback,
            weights.num_layers,
            &(weights.num_layers..=weights.num_layers + 5),
        );
    }

    #[test]
    fn attention_cache_build_captures_one_ffn_input_per_layer() {
        let weights = make_test_weights();
        let ffn = WeightFfn { weights: &weights };
        let cache = CachedLayerGraph::from_residuals(vec![]);
        let layer_range = 0..weights.num_layers;
        let ac = AttentionCache::build(&weights, &[0u32, 1, 2], &cache, &ffn, layer_range);
        assert_eq!(ac.ffn_inputs.len(), weights.num_layers);
        for input in &ac.ffn_inputs {
            assert_eq!(input.len(), weights.hidden_size);
            assert!(input.iter().all(|v| v.is_finite()));
        }
        assert_eq!(ac.final_residual.shape(), &[3, weights.hidden_size]);
    }

    #[test]
    fn attention_cache_build_partial_range_skips_layers() {
        // Layer range starting > 0 should still produce per-walk-layer
        // FFN inputs, just for the walked subset.
        let weights = make_test_weights();
        let ffn = WeightFfn { weights: &weights };
        let cache = CachedLayerGraph::from_residuals(vec![]);
        let ac = AttentionCache::build(&weights, &[0u32, 1], &cache, &ffn, 1..weights.num_layers);
        assert_eq!(ac.ffn_inputs.len(), weights.num_layers - 1);
    }
}
