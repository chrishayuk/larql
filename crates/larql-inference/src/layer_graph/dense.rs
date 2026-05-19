use ndarray::Array2;

use super::{LayerGraph, LayerOutput};
use crate::ffn::FfnBackend;
use crate::model::ModelWeights;
use larql_compute::prelude::*;

/// Dense baseline: standard matmul attention + pluggable FFN backend.
/// This is today's working path — nothing changes, just wrapped in the trait.
pub struct DenseLayerGraph<'a> {
    pub ffn: &'a dyn FfnBackend,
    pub backend: Option<&'a dyn ComputeBackend>,
    pub capture_activation: bool,
    pub capture_attention: bool,
}

impl<'a> LayerGraph for DenseLayerGraph<'a> {
    fn forward_layer(
        &self,
        weights: &ModelWeights,
        h: &Array2<f32>,
        layer: usize,
    ) -> Option<LayerOutput> {
        // Attention: dense matmul (Q·K·V), optionally GPU-accelerated
        let (h_post_attn, _attn_proj, attn_weights) = crate::attention::run_attention_block_gpu(
            weights,
            h,
            layer,
            self.capture_attention,
            self.backend,
        )?;

        // FFN: delegated to backend (dense, walk, sparse, etc.)
        let (h_out, activation) = crate::forward::run_ffn(
            weights,
            &h_post_attn,
            layer,
            self.ffn,
            self.capture_activation,
        );

        Some(LayerOutput {
            residual: h_out,
            activation,
            attention: attn_weights,
        })
    }

    fn name(&self) -> &str {
        "dense"
    }
}

/// Per-layer graph selection: different layers can use different backends.
pub struct PerLayerGraph<'a> {
    layers: Vec<&'a dyn LayerGraph>,
}

impl<'a> PerLayerGraph<'a> {
    pub fn new(layers: Vec<&'a dyn LayerGraph>) -> Self {
        Self { layers }
    }

    pub fn get(&self, layer: usize) -> &'a dyn LayerGraph {
        if layer < self.layers.len() {
            self.layers[layer]
        } else {
            *self.layers.last().unwrap()
        }
    }
}

impl<'a> LayerGraph for PerLayerGraph<'a> {
    fn forward_layer(
        &self,
        weights: &ModelWeights,
        h: &Array2<f32>,
        layer: usize,
    ) -> Option<LayerOutput> {
        self.get(layer).forward_layer(weights, h, layer)
    }

    fn name(&self) -> &str {
        "per-layer"
    }
}

#[cfg(all(test, not(target_arch = "wasm32")))]
mod tests {
    use super::*;
    use crate::ffn::WeightFfn;
    use crate::test_utils::make_test_weights;
    use larql_models::ModelWeights;
    use ndarray::Array2;
    use std::sync::OnceLock;

    fn weights() -> &'static ModelWeights {
        static W: OnceLock<ModelWeights> = OnceLock::new();
        W.get_or_init(make_test_weights)
    }

    fn input(seq: usize, hidden: usize) -> Array2<f32> {
        let data: Vec<f32> = (0..seq * hidden).map(|i| (i as f32 + 1.0) * 0.01).collect();
        Array2::from_shape_vec((seq, hidden), data).unwrap()
    }

    // ── DenseLayerGraph ───────────────────────────────────────────────────────

    #[test]
    fn dense_name() {
        let w = weights();
        let ffn = WeightFfn { weights: w };
        let g = DenseLayerGraph {
            ffn: &ffn,
            backend: None,
            capture_activation: false,
            capture_attention: false,
        };
        assert_eq!(g.name(), "dense");
    }

    #[test]
    fn dense_forward_shape_single_token() {
        let w = weights();
        let ffn = WeightFfn { weights: w };
        let g = DenseLayerGraph {
            ffn: &ffn,
            backend: None,
            capture_activation: false,
            capture_attention: false,
        };
        let h = input(1, w.hidden_size);
        let out = g.forward_layer(w, &h, 0).expect("layer 0 should succeed");
        assert_eq!(out.residual.shape(), &[1, w.hidden_size]);
        assert!(out.residual.iter().all(|v| v.is_finite()));
    }

    #[test]
    fn dense_forward_all_layers() {
        let w = weights();
        let ffn = WeightFfn { weights: w };
        let g = DenseLayerGraph {
            ffn: &ffn,
            backend: None,
            capture_activation: false,
            capture_attention: false,
        };
        let h = input(2, w.hidden_size);
        for layer in 0..w.num_layers {
            let out = g.forward_layer(w, &h, layer).expect("layer {layer}");
            assert_eq!(out.residual.shape(), &[2, w.hidden_size], "layer {layer}");
        }
    }

    #[test]
    fn dense_no_capture_has_no_activation() {
        let w = weights();
        let ffn = WeightFfn { weights: w };
        let g = DenseLayerGraph {
            ffn: &ffn,
            backend: None,
            capture_activation: false,
            capture_attention: false,
        };
        let out = g.forward_layer(w, &input(1, w.hidden_size), 0).unwrap();
        assert!(out.activation.is_none());
        assert!(out.attention.is_none());
    }

    #[test]
    fn dense_capture_activation_populates_field() {
        let w = weights();
        let ffn = WeightFfn { weights: w };
        let g = DenseLayerGraph {
            ffn: &ffn,
            backend: None,
            capture_activation: true,
            capture_attention: false,
        };
        let out = g.forward_layer(w, &input(1, w.hidden_size), 0).unwrap();
        assert!(
            out.activation.is_some(),
            "capture_activation=true should populate activation"
        );
    }

    // ── PerLayerGraph ─────────────────────────────────────────────────────────

    #[test]
    fn per_layer_get_in_range() {
        let w = weights();
        let ffn = WeightFfn { weights: w };
        let g0 = DenseLayerGraph {
            ffn: &ffn,
            backend: None,
            capture_activation: false,
            capture_attention: false,
        };
        let plg = PerLayerGraph::new(vec![&g0 as &dyn LayerGraph]);
        // layer 0 is in range
        let h = input(1, w.hidden_size);
        let out = plg.forward_layer(w, &h, 0);
        assert!(out.is_some());
    }

    #[test]
    fn per_layer_get_out_of_range_does_not_panic() {
        let w = weights();
        let ffn = WeightFfn { weights: w };
        let g0 = DenseLayerGraph {
            ffn: &ffn,
            backend: None,
            capture_activation: false,
            capture_attention: false,
        };
        let plg = PerLayerGraph::new(vec![&g0 as &dyn LayerGraph]);
        // layer 99 is out of range for the PerLayerGraph — uses last graph.
        // The underlying DenseLayerGraph returns None because weights don't have layer 99.
        // The important thing is it does not panic.
        let h = input(1, w.hidden_size);
        let _ = plg.forward_layer(w, &h, 99); // must not panic
    }

    #[test]
    fn per_layer_name() {
        let w = weights();
        let ffn = WeightFfn { weights: w };
        let g = DenseLayerGraph {
            ffn: &ffn,
            backend: None,
            capture_activation: false,
            capture_attention: false,
        };
        let plg = PerLayerGraph::new(vec![&g as &dyn LayerGraph]);
        assert_eq!(plg.name(), "per-layer");
    }
}
