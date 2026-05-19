//! Trace capture — decomposed forward pass recording attn and FFN deltas.

use std::collections::HashMap;

use ndarray::Array2;

use crate::attention::SharedKV;
use crate::ffn::{FfnBackend, WeightFfn};
use crate::forward::hooks::LayerHook;
use crate::forward::ple::precompute_per_layer_inputs;
use crate::forward::{embed_tokens_pub, run_layer_with_capture_hooked};
use crate::model::ModelWeights;

use super::types::*;

/// Which positions to capture.
pub enum TracePositions {
    Last,
    All,
    Positions(Vec<usize>),
}

#[derive(Default)]
struct TraceLayerHook {
    post_attention: Option<Array2<f32>>,
}

impl LayerHook for TraceLayerHook {
    fn on_post_attention(&mut self, _layer: usize, h: &mut Array2<f32>) {
        self.post_attention = Some(h.clone());
    }
}

/// Capture a complete residual stream trace.
pub fn trace_residuals(
    weights: &ModelWeights,
    token_ids: &[u32],
    positions: TracePositions,
    capture_attention: bool,
    ffn: &dyn FfnBackend,
) -> ResidualTrace {
    let seq_len = token_ids.len();
    let hidden = weights.hidden_size;
    let num_layers = weights.num_layers;

    let pos_list: Vec<usize> = match positions {
        TracePositions::Last => vec![seq_len - 1],
        TracePositions::All => (0..seq_len).collect(),
        TracePositions::Positions(ref ps) => ps.clone(),
    };

    let mut h = embed_tokens_pub(weights, token_ids);
    let ple_inputs = precompute_per_layer_inputs(weights, &h, token_ids);
    let mut kv_cache: HashMap<usize, SharedKV> = HashMap::new();
    let mut nodes = Vec::new();
    let mut attention_captures = Vec::new();
    let zero = vec![0.0f32; hidden];

    // Embedding layer (-1)
    for &p in &pos_list {
        nodes.push(TraceNode {
            layer: -1,
            position: p,
            residual: h.row(p).to_vec(),
            attn_delta: zero.clone(),
            ffn_delta: zero.clone(),
        });
    }

    // Transformer layers
    for layer in 0..num_layers {
        let pre = h.clone();

        let shared_kv = weights
            .arch
            .kv_shared_source_layer(layer)
            .and_then(|src| kv_cache.get(&src));
        let mut hook = TraceLayerHook::default();
        let Some((h_out, _, attn_weights, kv_out)) = run_layer_with_capture_hooked(
            weights,
            &h,
            layer,
            ffn,
            false,
            capture_attention,
            ple_inputs.get(layer),
            shared_kv,
            &mut hook,
        ) else {
            continue;
        };
        let h_post_attn = hook.post_attention.unwrap_or_else(|| pre.clone());

        for &p in &pos_list {
            let attn_delta: Vec<f32> = h_post_attn
                .row(p)
                .iter()
                .zip(pre.row(p).iter())
                .map(|(&a, &b)| a - b)
                .collect();
            let ffn_delta: Vec<f32> = h_out
                .row(p)
                .iter()
                .zip(h_post_attn.row(p).iter())
                .map(|(&a, &b)| a - b)
                .collect();

            nodes.push(TraceNode {
                layer: layer as i32,
                position: p,
                residual: h_out.row(p).to_vec(),
                attn_delta,
                ffn_delta,
            });
        }

        if let Some(w) = attn_weights {
            attention_captures.push((layer, w));
        }
        if let Some(kv) = kv_out {
            kv_cache.insert(layer, kv);
        }
        h = h_out;
    }

    let tokens: Vec<String> = token_ids.iter().map(|&id| format!("t{}", id)).collect();

    ResidualTrace {
        prompt: String::new(),
        tokens,
        token_ids: token_ids.to_vec(),
        n_layers: num_layers,
        hidden_size: hidden,
        nodes,
        attention: attention_captures,
    }
}

/// Convenience: trace with default WeightFfn.
pub fn trace(
    weights: &ModelWeights,
    token_ids: &[u32],
    positions: TracePositions,
) -> ResidualTrace {
    let ffn = WeightFfn { weights };
    trace_residuals(weights, token_ids, positions, false, &ffn)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ffn::FfnBackend;
    use crate::forward::trace_forward_with_ffn;
    // `forward_raw_logits` / `hidden_to_raw_logits` are only used by
    // `trace_final_residual_matches_raw_forward_logits`, which is gated
    // off on Windows. Keep the imports under the same gate so `clippy
    // -D warnings` doesn't trip on unused imports.
    #[cfg(not(windows))]
    use crate::forward::{forward_raw_logits, hidden_to_raw_logits};
    use crate::test_utils::make_test_weights;
    use larql_models::ModelWeights;
    use ndarray::Array2;
    use std::sync::OnceLock;

    fn weights() -> &'static ModelWeights {
        static W: OnceLock<ModelWeights> = OnceLock::new();
        W.get_or_init(make_test_weights)
    }

    struct ZeroFfn;

    impl FfnBackend for ZeroFfn {
        fn forward(&self, _layer: usize, x: &Array2<f32>) -> Array2<f32> {
            Array2::zeros((x.nrows(), x.ncols()))
        }

        fn forward_with_activation(
            &self,
            _layer: usize,
            x: &Array2<f32>,
        ) -> (Array2<f32>, Array2<f32>) {
            (
                Array2::zeros((x.nrows(), x.ncols())),
                Array2::zeros((x.nrows(), x.ncols())),
            )
        }

        fn name(&self) -> &str {
            "zero"
        }
    }

    // ── trace (WeightFfn path) ────────────────────────────────────────────────

    #[test]
    fn trace_all_positions_populates_nodes() {
        let w = weights();
        let t = trace(w, &[0u32, 1, 2], TracePositions::All);
        // Each position has (n_layers + 1) nodes (embedding + transformer layers)
        let expected = 3 * (w.num_layers + 1);
        assert_eq!(t.nodes.len(), expected, "expected {expected} nodes");
        assert_eq!(t.n_layers, w.num_layers);
        assert_eq!(t.hidden_size, w.hidden_size);
    }

    #[test]
    fn trace_last_position_only() {
        let w = weights();
        let t = trace(w, &[0u32, 1, 2, 3], TracePositions::Last);
        // Only last position: (n_layers + 1) nodes
        assert_eq!(t.nodes.len(), w.num_layers + 1);
        assert!(t.nodes.iter().all(|n| n.position == 3));
    }

    #[test]
    fn trace_specific_positions() {
        let w = weights();
        let t = trace(w, &[0u32, 1, 2, 3], TracePositions::Positions(vec![0, 2]));
        // 2 positions × (n_layers + 1) nodes
        assert_eq!(t.nodes.len(), 2 * (w.num_layers + 1));
        let positions: std::collections::HashSet<usize> =
            t.nodes.iter().map(|n| n.position).collect();
        assert_eq!(positions.len(), 2);
        assert!(positions.contains(&0) && positions.contains(&2));
    }

    #[test]
    fn trace_nodes_are_finite() {
        let w = weights();
        let t = trace(w, &[0u32, 1], TracePositions::All);
        for node in &t.nodes {
            assert!(
                node.residual.iter().all(|v| v.is_finite()),
                "layer {} pos {} residual has non-finite",
                node.layer,
                node.position
            );
        }
    }

    #[test]
    fn trace_deltas_correct_residual_len() {
        let w = weights();
        let t = trace(w, &[0u32], TracePositions::All);
        for node in &t.nodes {
            assert_eq!(node.residual.len(), w.hidden_size);
            assert_eq!(node.attn_delta.len(), w.hidden_size);
            assert_eq!(node.ffn_delta.len(), w.hidden_size);
        }
    }

    #[test]
    fn trace_embedding_layer_minus_one_present() {
        let w = weights();
        let t = trace(w, &[0u32, 1], TracePositions::All);
        // Each position should have layer -1 (embedding)
        assert!(t.nodes.iter().any(|n| n.layer == -1));
    }

    #[test]
    fn trace_edges_reconstruct_residuals() {
        let w = weights();
        let t = trace(w, &[0u32, 1, 2], TracePositions::Last);
        let pos = 2;

        for layer in 0..w.num_layers as i32 {
            let prev = if layer == 0 {
                t.node(-1, pos).expect("embedding node")
            } else {
                t.node(layer - 1, pos).expect("previous layer node")
            };
            let node = t.node(layer, pos).expect("current layer node");
            for i in 0..w.hidden_size {
                let reconstructed = prev.residual[i] + node.attn_delta[i] + node.ffn_delta[i];
                assert!(
                    (reconstructed - node.residual[i]).abs() < 1e-4,
                    "layer {layer} dim {i}: reconstructed {reconstructed} != residual {}",
                    node.residual[i]
                );
            }
        }
    }

    // `trace()` and `forward_raw_logits()` compute the same forward pass
    // but through slightly different BLAS dispatch paths. Linux + macOS
    // produce bit-stable enough output that the residual matches within
    // 1e-4; Windows OpenBLAS uses parallel reduction whose summation
    // order isn't reproducible across these two paths, so the residual
    // diverges by ~1e-1 well past the tolerance. Same pattern, same gate
    // as `generate_cached_hooked_with_noop_matches_baseline`.
    #[cfg(not(windows))]
    #[test]
    fn trace_final_residual_matches_raw_forward_logits() {
        let w = weights();
        let tokens = &[0u32, 1, 2, 3];
        let t = trace(w, tokens, TracePositions::Last);
        let node = t
            .node(w.num_layers as i32 - 1, tokens.len() - 1)
            .expect("final trace node");
        let raw = forward_raw_logits(w, tokens, None);

        let traced_h =
            Array2::from_shape_vec((1, w.hidden_size), node.residual.clone()).expect("trace row");
        let raw_last = tokens.len() - 1;
        for i in 0..w.hidden_size {
            let expected = raw.h_pre_norm[[raw_last, i]];
            let got = traced_h[[0, i]];
            assert!(
                (got - expected).abs() < 1e-4,
                "final residual dim {i}: trace {got} != raw forward {expected}"
            );
        }

        let traced_logits = hidden_to_raw_logits(w, &traced_h);
        for (i, &got) in traced_logits.iter().enumerate() {
            let expected = raw.logits[i];
            assert!(
                (got - expected).abs() < 1e-3,
                "logit {i}: trace projection {got} != raw forward {expected}"
            );
        }
    }

    #[test]
    fn trace_custom_ffn_matches_hooked_forward_final_residual() {
        let w = weights();
        let tokens = &[0u32, 1, 2, 3];
        let ffn = ZeroFfn;
        let t = trace_residuals(w, tokens, TracePositions::Last, false, &ffn);
        let traced = t
            .node(w.num_layers as i32 - 1, tokens.len() - 1)
            .expect("final trace node");
        let forward = trace_forward_with_ffn(w, tokens, &[w.num_layers - 1], false, 0, &ffn);
        let (_, expected) = forward.residuals.first().expect("captured final residual");

        // BLAS on Windows runs successive matmuls with different
        // reduction orders (parallel accumulation), so the two
        // code paths drift by a few times 1e-3 on equivalent inputs.
        // Linux/macOS BLAS is deterministic and stays well under 1e-4.
        // 1e-2 still catches real algorithmic regressions while letting
        // the documented Windows drift through.
        const FINAL_RESIDUAL_TOL: f32 = if cfg!(windows) { 1e-2 } else { 1e-4 };

        for i in 0..w.hidden_size {
            let got = traced.residual[i];
            let expected = expected[i];
            assert!(
                (got - expected).abs() < FINAL_RESIDUAL_TOL,
                "custom backend final residual dim {i}: trace {got} != hooked forward {expected}"
            );
        }
    }

    #[test]
    fn trace_with_capture_attention_records_per_layer_weights() {
        // Exercises the `if let Some(w) = attn_weights` branch — captures the
        // attention weight matrix at every layer so callers can analyse head
        // patterns alongside the residual deltas.
        let w = weights();
        let tokens = &[0u32, 1, 2];
        let ffn = WeightFfn { weights: w };
        let t = trace_residuals(w, tokens, TracePositions::Last, true, &ffn);
        assert_eq!(
            t.attention.len(),
            w.num_layers,
            "expected one attention capture per layer when capture_attention=true"
        );
        for (layer, attn) in &t.attention {
            for (head_idx, head) in attn.heads.iter().enumerate() {
                assert!(
                    head.iter().all(|v| v.is_finite()),
                    "non-finite attention weight at layer {layer} head {head_idx}"
                );
            }
        }
    }

    #[test]
    fn trace_residuals_positions_clone_preserves_order() {
        // Covers the `TracePositions::Positions(ref ps) => ps.clone()` arm
        // with non-sorted, non-contiguous indices — guards against any
        // future implementation that silently sorts/dedups.
        let w = weights();
        let t = trace(w, &[0u32, 1, 2, 3], TracePositions::Positions(vec![3, 0]));
        let order: Vec<usize> = t
            .nodes
            .iter()
            .filter(|n| n.layer == -1)
            .map(|n| n.position)
            .collect();
        assert_eq!(order, vec![3, 0]);
    }
}
