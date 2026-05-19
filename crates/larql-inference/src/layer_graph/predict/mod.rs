//! Prediction entry points — the functions external code calls.
//!
//! All GPU pipeline layer construction goes through `pipeline_layer::build_pipeline_layers()`.
//! Logits computation goes through `logits::finalize_logits()`.
//! KV cache prefill goes through `prefill::prefill_with_kv()`.
//! Token generation goes through `generate::generate()`.
//!
//! ## Module layout
//!
//! - this file (`mod.rs`) — small variants + the dispatcher [`predict_pipeline`]
//!   + the trace helper [`trace_with_graph`].
//! - [`split`] — `predict_split_pass` (3-pass approximate-attention pipeline)
//!   and `predict_split_cached` (logits-only fast path on cached residuals).
//! - [`honest`] — `predict_honest`, the production GPU+CPU hybrid that
//!   `larql bench` and the streaming-demo runner use.

mod honest;
mod split;

pub use honest::predict_honest;
pub use split::{predict_split_cached, predict_split_pass};

use super::LayerGraph;
use crate::model::ModelWeights;

// Re-export moved functions for backward compatibility.
pub use super::generate::{generate, GenerateResult};
pub use super::logits::finalize_logits;
pub use super::prefill::prefill_with_kv;

/// Run a full forward pass using vindex logits (KNN against lm_head mmap).
/// Replaces the 231ms dense logits matmul with a ~1ms KNN lookup.
pub fn predict_with_graph_vindex_logits(
    weights: &ModelWeights,
    tokenizer: &tokenizers::Tokenizer,
    token_ids: &[u32],
    top_k: usize,
    graph: &dyn LayerGraph,
    index: &larql_vindex::VectorIndex,
) -> crate::forward::PredictResult {
    let seq_len = token_ids.len();
    let mut h = crate::forward::embed_tokens_pub(weights, token_ids);

    for layer in 0..weights.num_layers {
        match graph.forward_layer(weights, &h, layer) {
            Some(output) => h = output.residual,
            None => break,
        }
    }

    // Final norm
    let norm_offset = weights.arch.norm_weight_offset();
    let h_final =
        crate::forward::apply_norm(weights, &h, weights.arch.final_norm_key(), norm_offset);

    // Vindex logits: KNN against lm_head mmap
    let last_row = h_final.row(seq_len - 1).to_owned();

    let logits_scale = weights.arch.logits_scaling();
    let final_softcap = weights.arch.final_logit_softcapping();
    let inv_scale = 1.0 / logits_scale;

    // Get raw scores from KNN (dot products against lm_head)
    let hits = index.lm_head_knn(&last_row, top_k);

    // Apply scaling, softcap, softmax over top-K
    let scaled: Vec<(u32, f32)> = hits
        .iter()
        .map(|&(tid, score)| {
            let mut logit = score * inv_scale;
            if let Some(cap) = final_softcap {
                logit = (logit / cap).tanh() * cap;
            }
            (tid, logit)
        })
        .collect();

    let max_logit = scaled
        .iter()
        .map(|(_, l)| *l)
        .fold(f32::NEG_INFINITY, f32::max);
    let exp_sum: f64 = scaled
        .iter()
        .map(|(_, l)| ((*l - max_logit) as f64).exp())
        .sum();

    let predictions = scaled
        .iter()
        .filter_map(|&(tid, logit)| {
            let prob = ((logit - max_logit) as f64).exp() / exp_sum;
            tokenizer
                .decode(&[tid], true)
                .ok()
                .map(|s| (s.trim().to_string(), prob))
        })
        .collect();

    crate::forward::PredictResult {
        predictions,
        token_ids: Vec::new(),
    }
}

/// Run a full forward pass using a LayerGraph for per-layer routing.
/// This is the generic layer loop — embedding → layers → logits.
pub fn predict_with_graph(
    weights: &ModelWeights,
    tokenizer: &tokenizers::Tokenizer,
    token_ids: &[u32],
    top_k: usize,
    graph: &dyn LayerGraph,
) -> crate::forward::PredictResult {
    let mut h = crate::forward::embed_tokens_pub(weights, token_ids);

    for layer in 0..weights.num_layers {
        match graph.forward_layer(weights, &h, layer) {
            Some(output) => h = output.residual,
            None => break,
        }
    }

    crate::forward::logits_to_predictions_pub(weights, &h, tokenizer, top_k, 1.0)
}

/// Optimized predict: uses vindex logits when lm_head is loaded, falls back to full matmul.
///
/// This is the production entry point. It:
/// 1. Runs embedding → layer loop via LayerGraph
/// 2. Uses vindex lm_head KNN if available (eliminates 226ms logits matmul)
/// 3. Falls back to full vocab matmul if no lm_head loaded
pub fn predict_pipeline(
    weights: &ModelWeights,
    tokenizer: &tokenizers::Tokenizer,
    token_ids: &[u32],
    top_k: usize,
    graph: &dyn LayerGraph,
    index: Option<&larql_vindex::VectorIndex>,
) -> crate::forward::PredictResult {
    // Use vindex logits if lm_head is loaded
    if let Some(idx) = index {
        if idx.has_lm_head() {
            return predict_with_graph_vindex_logits(
                weights, tokenizer, token_ids, top_k, graph, idx,
            );
        }
    }
    // Fallback: full vocab matmul
    predict_with_graph(weights, tokenizer, token_ids, top_k, graph)
}

/// Run a full forward pass with tracing (residuals + activations + attention).
pub fn trace_with_graph(
    weights: &ModelWeights,
    token_ids: &[u32],
    capture_layers: &[usize],
    graph: &dyn LayerGraph,
) -> crate::forward::TraceResult {
    let seq_len = token_ids.len();
    let max_layer = *capture_layers.iter().max().unwrap_or(&0);

    let mut h = crate::forward::embed_tokens_pub(weights, token_ids);
    let mut results = Vec::new();
    let mut activations = Vec::new();
    let mut attention_captures = Vec::new();

    for layer in 0..=max_layer.min(weights.num_layers - 1) {
        match graph.forward_layer(weights, &h, layer) {
            Some(output) => {
                h = output.residual;

                if capture_layers.contains(&layer) {
                    let last_row = h.row(seq_len - 1);
                    results.push((layer, last_row.to_vec()));

                    if let Some(act) = output.activation {
                        let act_row = act.row(seq_len - 1);
                        let mut indexed: Vec<(usize, f32)> =
                            act_row.iter().copied().enumerate().collect();
                        indexed.sort_unstable_by(|a, b| b.1.abs().partial_cmp(&a.1.abs()).unwrap());
                        indexed.truncate(200);
                        activations.push((layer, indexed));
                    }

                    if let Some(attn) = output.attention {
                        attention_captures.push(crate::forward::LayerAttentionCapture {
                            layer,
                            weights: attn,
                        });
                    }
                }
            }
            None => break,
        }
    }

    crate::forward::TraceResult {
        residuals: results,
        activations,
        attention: attention_captures,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_utils::TestFixtures;
    use std::sync::OnceLock;

    fn fx() -> &'static TestFixtures {
        static F: OnceLock<TestFixtures> = OnceLock::new();
        F.get_or_init(TestFixtures::build)
    }
    use crate::ffn::WeightFfn;
    use crate::layer_graph::CachedLayerGraph;
    use larql_compute::CpuBackend;

    // ── predict_with_ffn ──────────────────────────────────────────────────────

    #[test]
    fn predict_with_ffn_returns_predictions() {
        let f = fx();
        let (weights, tokenizer) = (&f.weights, &f.tokenizer);
        let ffn = WeightFfn { weights };
        let result = crate::forward::predict_with_ffn(weights, tokenizer, &[0u32, 1], 3, &ffn);
        assert!(result.token_ids.len() <= 3);
        assert_eq!(result.predictions.len(), result.token_ids.len());
        assert!(result
            .token_ids
            .iter()
            .all(|&id| (id as usize) < weights.vocab_size));
    }

    #[test]
    fn predict_with_ffn_single_token() {
        let f = fx();
        let (weights, tokenizer) = (&f.weights, &f.tokenizer);
        let ffn = WeightFfn { weights };
        let result = crate::forward::predict_with_ffn(weights, tokenizer, &[5u32], 1, &ffn);
        assert!(result.token_ids.len() <= 1);
    }

    // ── predict_honest (CPU path via VectorIndex::new with no Q4K) ────────────

    #[test]
    fn predict_honest_runs_without_panic() {
        let f = fx();
        let (weights, tokenizer, index) = (&f.weights, &f.tokenizer, &f.index);
        let cached = CachedLayerGraph::from_residuals(vec![]);
        let num_layers = weights.num_layers;
        // predict_honest falls through to CPU path (no Q4K data in synthetic vindex)
        let result = predict_honest(
            weights,
            tokenizer,
            &[0u32, 1, 2],
            5,
            index,
            &CpuBackend,
            &cached,
            0..num_layers,
        );
        // lm_head_knn is empty → predictions may be empty, but no panic
        assert!(result.token_ids.len() <= 5);
    }

    #[test]
    fn predict_honest_single_token_decode_path() {
        let f = fx();
        let (weights, tokenizer, index) = (&f.weights, &f.tokenizer, &f.index);
        let cached = CachedLayerGraph::from_residuals(vec![]);
        let num_layers = weights.num_layers;
        let result = predict_honest(
            weights,
            tokenizer,
            &[3u32],
            3,
            index,
            &CpuBackend,
            &cached,
            0..num_layers,
        );
        assert!(result.token_ids.len() <= 3);
    }

    /// `predict_honest` against the synthetic Q4K vindex — drives the
    /// GPU-path branch (`backend.supports_quant(Q4_K) == true`) because
    /// `CpuBackend` advertises Q4_K matvec support. `decode_token` /
    /// `full_pipeline_q4` aren't implemented on CPU so they return None,
    /// pushing control into the `used_gpu = false` CPU fallback at the
    /// end of the function. End-to-end body coverage in either case.
    #[test]
    fn predict_honest_seq1_decode_against_synthetic_q4k_fixture() {
        use crate::test_utils::{make_test_q4k_vindex, make_test_q4k_weights, make_test_tokenizer};
        let weights = make_test_q4k_weights();
        let tokenizer = make_test_tokenizer(weights.vocab_size);
        let index = make_test_q4k_vindex(&weights);
        let cached = CachedLayerGraph::from_residuals(vec![]);
        let num_layers = weights.num_layers;
        let result = predict_honest(
            &weights,
            &tokenizer,
            &[0u32],
            3,
            &index,
            &CpuBackend,
            &cached,
            0..num_layers,
        );
        assert!(result.token_ids.len() <= 3);
    }

    /// `predict_honest` in prefill mode (seq_len > 1) against the
    /// synthetic Q4K vindex — exercises the `seq_len > 1` branch of the
    /// GPU pipeline (lines 122-152: pre-norm prefill path) and falls
    /// through to CPU fallback when `prefill_kquant` returns None.
    #[test]
    fn predict_honest_prefill_against_synthetic_q4k_fixture() {
        use crate::test_utils::{make_test_q4k_vindex, make_test_q4k_weights, make_test_tokenizer};
        let weights = make_test_q4k_weights();
        let tokenizer = make_test_tokenizer(weights.vocab_size);
        let index = make_test_q4k_vindex(&weights);
        let cached = CachedLayerGraph::from_residuals(vec![]);
        let num_layers = weights.num_layers;
        let result = predict_honest(
            &weights,
            &tokenizer,
            &[0u32, 1, 2, 3],
            5,
            &index,
            &CpuBackend,
            &cached,
            0..num_layers,
        );
        assert!(result.token_ids.len() <= 5);
    }

    #[test]
    fn predict_honest_with_cached_layers() {
        let f = fx();
        let (weights, tokenizer, index) = (&f.weights, &f.tokenizer, &f.index);
        let ffn = WeightFfn { weights };
        // Pre-cache layer 0
        let cached = CachedLayerGraph::build(weights, &[0u32], &[0], &ffn);
        let num_layers = weights.num_layers;
        let result = predict_honest(
            weights,
            tokenizer,
            &[0u32],
            3,
            index,
            &CpuBackend,
            &cached,
            0..num_layers,
        );
        assert!(result.token_ids.len() <= 3);
    }

    // ── DenseLayerGraph ───────────────────────────────────────────────────────

    #[test]
    fn dense_layer_graph_forward_runs() {
        use crate::layer_graph::{DenseLayerGraph, LayerGraph};
        let weights = &fx().weights;
        let ffn = WeightFfn { weights };
        let h = ndarray::Array2::from_elem((2, weights.hidden_size), 0.1f32);
        let g = DenseLayerGraph {
            ffn: &ffn,
            backend: None,
            capture_activation: false,
            capture_attention: false,
        };
        let out = g.forward_layer(weights, &h, 0);
        assert!(out.is_some(), "DenseLayerGraph should forward layer 0");
        assert_eq!(out.unwrap().residual.shape(), &[2, weights.hidden_size]);
    }

    #[test]
    fn dense_layer_graph_all_layers() {
        use crate::layer_graph::{DenseLayerGraph, LayerGraph};
        let weights = &fx().weights;
        let ffn = WeightFfn { weights };
        let h = ndarray::Array2::from_elem((1, weights.hidden_size), 0.5f32);
        let g = DenseLayerGraph {
            ffn: &ffn,
            backend: None,
            capture_activation: false,
            capture_attention: false,
        };
        for layer in 0..weights.num_layers {
            let out = g.forward_layer(weights, &h, layer);
            assert!(out.is_some(), "layer {layer} should succeed");
        }
    }

    // ── WalkLayerGraph ────────────────────────────────────────────────────────

    #[test]
    fn walk_layer_graph_forward_runs() {
        use crate::layer_graph::{LayerGraph, WalkLayerGraph};
        let weights = &fx().weights;
        let ffn = WeightFfn { weights };
        let g = WalkLayerGraph {
            ffn: &ffn,
            backend: None,
        };
        let h = ndarray::Array2::from_elem((2, weights.hidden_size), 0.1f32);
        let out = g.forward_layer(weights, &h, 0);
        assert!(out.is_some());
        assert_eq!(out.unwrap().residual.shape(), &[2, weights.hidden_size]);
    }

    // ── predict_pipeline ─────────────────────────────────────────────────────

    #[test]
    fn predict_pipeline_runs() {
        use crate::layer_graph::LayerGraph;
        let f = fx();
        let (weights, tokenizer, index) = (&f.weights, &f.tokenizer, &f.index);
        let ffn = WeightFfn { weights };
        let g = crate::layer_graph::WalkLayerGraph {
            ffn: &ffn,
            backend: None,
        };
        let graph: &dyn LayerGraph = &g;
        // predict_pipeline takes Option<&VectorIndex>
        let result = predict_pipeline(weights, tokenizer, &[0u32, 1], 3, graph, Some(index));
        assert!(result.token_ids.len() <= 3);
    }

    #[test]
    fn predict_pipeline_falls_back_when_no_index() {
        // index = None → must skip the vindex-logits path and use full
        // matmul via predict_with_graph.
        use crate::layer_graph::LayerGraph;
        let f = fx();
        let (weights, tokenizer) = (&f.weights, &f.tokenizer);
        let ffn = WeightFfn { weights };
        let g = crate::layer_graph::WalkLayerGraph {
            ffn: &ffn,
            backend: None,
        };
        let graph: &dyn LayerGraph = &g;
        let result = predict_pipeline(weights, tokenizer, &[0u32], 3, graph, None);
        assert!(result.predictions.len() <= 3);
    }

    #[test]
    fn predict_pipeline_falls_back_when_index_lacks_lm_head() {
        // Synthetic vindex doesn't load lm_head — `has_lm_head` returns
        // false → still routes to predict_with_graph (full matmul).
        use crate::layer_graph::LayerGraph;
        let f = fx();
        let ffn = WeightFfn {
            weights: &f.weights,
        };
        let g = crate::layer_graph::WalkLayerGraph {
            ffn: &ffn,
            backend: None,
        };
        let graph: &dyn LayerGraph = &g;
        // Synthetic index has no lm_head data populated.
        let result = predict_pipeline(&f.weights, &f.tokenizer, &[0u32], 3, graph, Some(&f.index));
        assert!(result.predictions.len() <= 3);
    }

    #[test]
    fn predict_with_graph_returns_predictions() {
        use crate::layer_graph::{LayerGraph, WalkLayerGraph};
        let f = fx();
        let ffn = WeightFfn {
            weights: &f.weights,
        };
        let g = WalkLayerGraph {
            ffn: &ffn,
            backend: None,
        };
        let graph: &dyn LayerGraph = &g;
        let result = predict_with_graph(&f.weights, &f.tokenizer, &[0u32, 1, 2], 4, graph);
        assert!(result.predictions.len() <= 4);
    }

    #[test]
    fn trace_with_graph_returns_per_layer_residuals() {
        use crate::layer_graph::{LayerGraph, WalkLayerGraph};
        let f = fx();
        let ffn = WeightFfn {
            weights: &f.weights,
        };
        let g = WalkLayerGraph {
            ffn: &ffn,
            backend: None,
        };
        let graph: &dyn LayerGraph = &g;
        let trace = trace_with_graph(&f.weights, &[0u32, 1, 2], &[0, 1], graph);
        assert_eq!(trace.residuals.len(), 2);
        assert_eq!(trace.residuals[0].0, 0);
        assert_eq!(trace.residuals[1].0, 1);
        for (_, r) in &trace.residuals {
            assert_eq!(r.len(), f.weights.hidden_size);
            assert!(r.iter().all(|v| v.is_finite()));
        }
    }

    #[test]
    fn trace_with_graph_empty_capture_runs_zero_layers() {
        use crate::layer_graph::{LayerGraph, WalkLayerGraph};
        let f = fx();
        let ffn = WeightFfn {
            weights: &f.weights,
        };
        let g = WalkLayerGraph {
            ffn: &ffn,
            backend: None,
        };
        let graph: &dyn LayerGraph = &g;
        // Empty capture → max_layer defaults to 0 (per `unwrap_or(&0)`)
        // → still walks layer 0 but never records.
        let trace = trace_with_graph(&f.weights, &[0u32], &[], graph);
        assert!(trace.residuals.is_empty());
        assert!(trace.activations.is_empty());
    }

    /// Direct call to `predict_with_graph_vindex_logits` so the function
    /// body is exercised even if `predict_pipeline`'s `has_lm_head` gate
    /// shifts. Same Q4K fixture as above.
    #[test]
    fn predict_with_graph_vindex_logits_directly() {
        use crate::layer_graph::WalkLayerGraph;
        use crate::test_utils::{
            make_test_q4k_weights, make_test_tokenizer, write_synthetic_q4k_model_dir,
        };
        use crate::vindex::open_inference_vindex;

        let tmp = tempfile::tempdir().unwrap();
        write_synthetic_q4k_model_dir(tmp.path()).expect("fixture");
        let index = open_inference_vindex(tmp.path()).expect("loader");

        let weights = make_test_q4k_weights();
        let tokenizer = make_test_tokenizer(weights.vocab_size);
        let ffn = WeightFfn { weights: &weights };
        let g = WalkLayerGraph {
            ffn: &ffn,
            backend: None,
        };
        let result =
            predict_with_graph_vindex_logits(&weights, &tokenizer, &[0u32, 1, 2], 5, &g, &index);
        assert!(result.predictions.len() <= 5);
    }

    /// Trace capture with activation+attention populated — drives lines
    /// 175-189 (the `if let Some(act)` / `if let Some(attn)` arms of
    /// `trace_with_graph`).
    #[test]
    fn trace_with_graph_captures_activation_and_attention() {
        use crate::layer_graph::{DenseLayerGraph, LayerGraph};
        let f = fx();
        let ffn = WeightFfn {
            weights: &f.weights,
        };
        let g = DenseLayerGraph {
            ffn: &ffn,
            backend: None,
            capture_activation: true,
            capture_attention: true,
        };
        let graph: &dyn LayerGraph = &g;
        let trace = trace_with_graph(&f.weights, &[0u32, 1], &[0], graph);
        assert_eq!(trace.residuals.len(), 1);
        // DenseLayerGraph with capture_activation=true populates the
        // Some(act) arm, capture_attention=true populates Some(attn).
        assert!(
            !trace.activations.is_empty(),
            "capture_activation should populate trace.activations"
        );
        assert!(
            !trace.attention.is_empty(),
            "capture_attention should populate trace.attention"
        );
    }
}
