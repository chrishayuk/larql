//! LayerGraph — pluggable per-layer routing for attention and FFN.
//!
//! The transformer layer loop receives a residual, routes through attention
//! and FFN, and produces the next residual. The mechanism behind each step
//! can vary:
//!
//! - Dense matmul (today's baseline)
//! - Walk/vindex (sparse FFN from mmap)
//! - Template cache (precomputed routing for known templates)
//! - Residual-adaptive graph (cluster-based routing)
//!
//! The `LayerGraph` trait abstracts this: given a residual, produce the
//! layer output. The implementation decides how attention and FFN are computed.

mod cached;
mod dense;
pub mod generate;
#[cfg(not(target_arch = "wasm32"))]
pub mod grid;
pub mod hybrid;
pub mod logits;
pub mod pipeline_layer;
pub mod predict;
pub mod prefill;
mod template;
mod walk;

pub use generate::{
    generate, generate_constrained, generate_constrained_streaming,
    generate_constrained_streaming_sampled, generate_streaming, generate_with_sampling,
    lm_head_topk, try_generate, try_generate_constrained, try_generate_constrained_streaming,
    try_generate_constrained_streaming_sampled, try_generate_streaming, try_generate_with_sampling,
    ChatMLRenderer, ChatSession, Detokenizer, EosConfig, GemmaRenderer, GenerateError,
    GenerateResult, Llama3Renderer, Sampler, SamplingConfig, StageTimings, TurnRenderer,
};

use ndarray::Array2;

use crate::attention::AttentionWeights;
use crate::model::ModelWeights;

// Re-export everything publicly
pub use cached::*;
pub use dense::*;
pub use predict::*;
pub use template::*;
pub use walk::*;

/// Output of a single layer's computation.
pub struct LayerOutput {
    /// Post-layer residual (input to next layer).
    pub residual: Array2<f32>,
    /// Optional: FFN activation capture (for tracing/analysis).
    pub activation: Option<Array2<f32>>,
    /// Optional: attention weight capture (for tracing/analysis).
    pub attention: Option<AttentionWeights>,
}

/// Per-layer routing trait. Takes a residual, produces the next residual.
///
/// Implementations control both attention and FFN computation.
/// The residual is always the input. The mechanism changes.
pub trait LayerGraph {
    /// Run one transformer layer: attention + FFN + residuals.
    fn forward_layer(
        &self,
        weights: &ModelWeights,
        h: &Array2<f32>,
        layer: usize,
    ) -> Option<LayerOutput>;

    /// Human-readable name for logging.
    fn name(&self) -> &str;
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

    // Verify that all three core LayerGraph implementations fulfil the trait
    // contract — they accept the same input shape and return a consistent output.

    #[test]
    fn dense_and_walk_produce_same_output_shape() {
        let w = weights();
        let ffn = WeightFfn { weights: w };
        let dense = DenseLayerGraph {
            ffn: &ffn,
            backend: None,
            capture_activation: false,
            capture_attention: false,
        };
        let walk = WalkLayerGraph {
            ffn: &ffn,
            backend: None,
        };
        let h = input(1, w.hidden_size);
        let out_d = dense.forward_layer(w, &h, 0).unwrap();
        let out_wk = walk.forward_layer(w, &h, 0).unwrap();
        assert_eq!(out_d.residual.shape(), out_wk.residual.shape());
    }

    #[test]
    fn layer_output_residual_is_finite_for_all_impls() {
        let w = weights();
        let ffn = WeightFfn { weights: w };
        let impls: Vec<(&str, Box<dyn LayerGraph>)> = vec![
            (
                "dense",
                Box::new(DenseLayerGraph {
                    ffn: &ffn,
                    backend: None,
                    capture_activation: false,
                    capture_attention: false,
                }),
            ),
            (
                "walk",
                Box::new(WalkLayerGraph {
                    ffn: &ffn,
                    backend: None,
                }),
            ),
        ];
        let h = input(1, w.hidden_size);
        for (name, g) in &impls {
            let out = g
                .forward_layer(w, &h, 0)
                .unwrap_or_else(|| panic!("{name} layer 0 returned None"));
            assert!(
                out.residual.iter().all(|v| v.is_finite()),
                "{name}: residual has non-finite values"
            );
        }
    }

    #[test]
    fn layer_graph_names_are_distinct() {
        let w = weights();
        let ffn = WeightFfn { weights: w };
        let dense = DenseLayerGraph {
            ffn: &ffn,
            backend: None,
            capture_activation: false,
            capture_attention: false,
        };
        let walk = WalkLayerGraph {
            ffn: &ffn,
            backend: None,
        };
        assert_ne!(dense.name(), walk.name());
    }
}
