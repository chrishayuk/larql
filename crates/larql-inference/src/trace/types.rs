//! Core trace types.

use crate::attention::AttentionWeights;
use crate::model::ModelWeights;
use serde::{Deserialize, Serialize};

/// A single waypoint in the residual stream.
#[derive(Clone)]
pub struct TraceNode {
    /// Layer index (-1 = embedding).
    pub layer: i32,
    /// Token position in the sequence.
    pub position: usize,
    /// Residual vector after this layer. Shape: (hidden_size,).
    pub residual: Vec<f32>,
    /// What attention added at this layer. Zero for embedding layer.
    pub attn_delta: Vec<f32>,
    /// What the post-attention path added at this layer. Zero for embedding
    /// layer. On plain decoder blocks this is the FFN residual write; on
    /// architectures with PLE/post norms/layer scales it includes those
    /// model-specific terms so that:
    /// `residual[layer] = residual[layer-1] + attn_delta + ffn_delta`.
    pub ffn_delta: Vec<f32>,
}

/// Per-layer summary across all captured positions.
#[derive(Clone, Serialize, Deserialize)]
pub struct LayerSummary {
    pub layer: i32,
    pub residual_norm: f32,
    pub attn_delta_norm: f32,
    pub ffn_delta_norm: f32,
    pub top1_token: String,
    pub top1_prob: f32,
}

/// One waypoint in an answer trajectory.
#[derive(Clone, Serialize, Deserialize)]
pub struct AnswerWaypoint {
    pub layer: i32,
    pub rank: u32,
    pub prob: f32,
    pub attn_logit: f32,
    pub ffn_logit: f32,
    pub residual_norm: f32,
}

/// Complete in-memory inference trace.
pub struct ResidualTrace {
    pub prompt: String,
    pub tokens: Vec<String>,
    pub token_ids: Vec<u32>,
    pub n_layers: usize,
    pub hidden_size: usize,
    pub nodes: Vec<TraceNode>,
    pub attention: Vec<(usize, AttentionWeights)>,
}

impl ResidualTrace {
    pub fn node(&self, layer: i32, position: usize) -> Option<&TraceNode> {
        self.nodes
            .iter()
            .find(|n| n.layer == layer && n.position == position)
    }

    pub fn last_node(&self, layer: i32) -> Option<&TraceNode> {
        let last_pos = self.tokens.len().saturating_sub(1);
        self.node(layer, last_pos)
    }

    pub fn layer_nodes(&self, layer: i32) -> Vec<&TraceNode> {
        self.nodes.iter().filter(|n| n.layer == layer).collect()
    }

    pub fn position_trajectory(&self, position: usize) -> Vec<&TraceNode> {
        let mut traj: Vec<&TraceNode> = self
            .nodes
            .iter()
            .filter(|n| n.position == position)
            .collect();
        traj.sort_by_key(|n| n.layer);
        traj
    }

    pub fn vocab_project(&self, weights: &ModelWeights, vec: &[f32]) -> Vec<f32> {
        super::vocab::project_to_logits(weights, vec)
    }

    pub fn top_k(
        &self,
        weights: &ModelWeights,
        tokenizer: &tokenizers::Tokenizer,
        layer: i32,
        position: usize,
        k: usize,
    ) -> Vec<(String, f32)> {
        let node = match self.node(layer, position) {
            Some(n) => n,
            None => return vec![],
        };
        let logits = super::vocab::project_to_logits(weights, &node.residual);
        super::vocab::top_k_from_logits(&logits, tokenizer, k)
    }

    pub fn answer_trajectory(
        &self,
        weights: &ModelWeights,
        answer_token_id: u32,
    ) -> Vec<AnswerWaypoint> {
        let last_pos = self.tokens.len().saturating_sub(1);
        let mut traj = Vec::new();

        for layer in -1..self.n_layers as i32 {
            let node = match self.node(layer, last_pos) {
                Some(n) => n,
                None => continue,
            };
            let logits = super::vocab::project_to_logits(weights, &node.residual);
            let probs = super::vocab::softmax(&logits);
            let prob = probs[answer_token_id as usize];
            let rank = probs.iter().filter(|&&p| p > prob).count() as u32 + 1;

            let attn_logit = if node.attn_delta.iter().any(|&x| x != 0.0) {
                super::vocab::project_to_logits(weights, &node.attn_delta)[answer_token_id as usize]
            } else {
                0.0
            };
            let ffn_logit = if node.ffn_delta.iter().any(|&x| x != 0.0) {
                super::vocab::project_to_logits(weights, &node.ffn_delta)[answer_token_id as usize]
            } else {
                0.0
            };

            traj.push(AnswerWaypoint {
                layer,
                rank,
                prob,
                attn_logit,
                ffn_logit,
                residual_norm: super::vocab::vec_norm(&node.residual),
            });
        }
        traj
    }

    pub fn layer_summaries<'a>(
        &'a self,
        weights: &'a ModelWeights,
        tokenizer: &'a tokenizers::Tokenizer,
    ) -> Vec<LayerSummary> {
        let last_pos = self.tokens.len().saturating_sub(1);
        let mut summaries = Vec::new();
        for layer in -1..self.n_layers as i32 {
            let node = match self.node(layer, last_pos) {
                Some(n) => n,
                None => continue,
            };
            let logits = super::vocab::project_to_logits(weights, &node.residual);
            let top = super::vocab::top_k_from_logits(&logits, tokenizer, 1);
            let (tok, prob) = top
                .first()
                .map(|(t, p)| (t.clone(), *p))
                .unwrap_or_default();
            summaries.push(LayerSummary {
                layer,
                residual_norm: super::vocab::vec_norm(&node.residual),
                attn_delta_norm: super::vocab::vec_norm(&node.attn_delta),
                ffn_delta_norm: super::vocab::vec_norm(&node.ffn_delta),
                top1_token: tok,
                top1_prob: prob,
            });
        }
        summaries
    }
}

#[cfg(all(test, not(target_arch = "wasm32")))]
mod tests {
    use super::*;

    fn node(layer: i32, position: usize) -> TraceNode {
        TraceNode {
            layer,
            position,
            residual: vec![layer as f32, position as f32],
            attn_delta: vec![0.0, 0.0],
            ffn_delta: vec![0.0, 0.0],
        }
    }

    fn make_trace(n_layers: usize, n_tokens: usize) -> ResidualTrace {
        let mut nodes = Vec::new();
        for pos in 0..n_tokens {
            // embedding layer (-1) + transformer layers 0..n_layers
            nodes.push(node(-1, pos));
            for l in 0..n_layers as i32 {
                nodes.push(node(l, pos));
            }
        }
        ResidualTrace {
            prompt: "test".into(),
            tokens: (0..n_tokens).map(|i| format!("t{i}")).collect(),
            token_ids: (0..n_tokens as u32).collect(),
            n_layers,
            hidden_size: 2,
            nodes,
            attention: Vec::new(),
        }
    }

    // ── node ──────────────────────────────────────────────────────────────────

    #[test]
    fn node_found_at_correct_layer_and_position() {
        let t = make_trace(3, 4);
        let n = t.node(1, 2).expect("layer 1, pos 2 should exist");
        assert_eq!(n.layer, 1);
        assert_eq!(n.position, 2);
    }

    #[test]
    fn node_returns_none_for_missing_layer() {
        let t = make_trace(3, 2);
        assert!(t.node(99, 0).is_none());
    }

    #[test]
    fn node_returns_none_for_missing_position() {
        let t = make_trace(3, 2);
        assert!(t.node(0, 99).is_none());
    }

    #[test]
    fn embedding_layer_minus_one_accessible() {
        let t = make_trace(2, 3);
        assert!(t.node(-1, 0).is_some());
        assert_eq!(t.node(-1, 0).unwrap().layer, -1);
    }

    // ── last_node ─────────────────────────────────────────────────────────────

    #[test]
    fn last_node_returns_node_at_last_token() {
        let t = make_trace(2, 4); // 4 tokens, last pos = 3
        let n = t.last_node(0).expect("layer 0 last node");
        assert_eq!(n.position, 3);
    }

    #[test]
    fn last_node_returns_none_for_missing_layer() {
        let t = make_trace(2, 2);
        assert!(t.last_node(99).is_none());
    }

    // ── layer_nodes ───────────────────────────────────────────────────────────

    #[test]
    fn layer_nodes_returns_all_positions_for_layer() {
        let t = make_trace(3, 5); // 5 tokens
        let nodes = t.layer_nodes(2);
        assert_eq!(nodes.len(), 5, "one node per token at layer 2");
        assert!(nodes.iter().all(|n| n.layer == 2));
    }

    #[test]
    fn layer_nodes_returns_empty_for_missing_layer() {
        let t = make_trace(2, 3);
        assert!(t.layer_nodes(99).is_empty());
    }

    // ── position_trajectory ───────────────────────────────────────────────────

    #[test]
    fn position_trajectory_sorted_ascending_by_layer() {
        let t = make_trace(4, 3);
        let traj = t.position_trajectory(1); // position 1
                                             // Should have embedding (-1) + 4 transformer layers = 5 nodes
        assert_eq!(traj.len(), 5);
        for w in traj.windows(2) {
            assert!(w[0].layer <= w[1].layer, "trajectory not sorted");
        }
        assert_eq!(traj[0].layer, -1);
    }

    #[test]
    fn position_trajectory_empty_for_missing_position() {
        let t = make_trace(2, 2);
        assert!(t.position_trajectory(99).is_empty());
    }

    // ── vocab-projection methods (need real ModelWeights + Tokenizer) ──

    fn real_trace() -> (
        ResidualTrace,
        crate::model::ModelWeights,
        tokenizers::Tokenizer,
    ) {
        let weights = crate::test_utils::make_test_weights();
        let tokenizer = crate::test_utils::make_test_tokenizer(weights.vocab_size);
        let n_layers = weights.num_layers;
        let hidden = weights.hidden_size;
        let n_tokens = 3;
        let mut nodes = Vec::new();
        for pos in 0..n_tokens {
            for l in -1..n_layers as i32 {
                nodes.push(TraceNode {
                    layer: l,
                    position: pos,
                    residual: vec![0.05f32; hidden],
                    attn_delta: vec![0.01f32; hidden],
                    ffn_delta: vec![0.02f32; hidden],
                });
            }
        }
        let trace = ResidualTrace {
            prompt: "hello".into(),
            tokens: vec!["[0]".into(), "[1]".into(), "[2]".into()],
            token_ids: vec![0u32, 1, 2],
            n_layers,
            hidden_size: hidden,
            nodes,
            attention: Vec::new(),
        };
        (trace, weights, tokenizer)
    }

    #[test]
    fn vocab_project_returns_full_vocab_logits() {
        let (trace, weights, _) = real_trace();
        let v = vec![0.05f32; weights.hidden_size];
        let logits = trace.vocab_project(&weights, &v);
        assert_eq!(logits.len(), weights.vocab_size);
        assert!(logits.iter().all(|x| x.is_finite()));
    }

    #[test]
    fn top_k_returns_at_most_k_predictions() {
        let (trace, weights, tokenizer) = real_trace();
        let top = trace.top_k(&weights, &tokenizer, 0, 1, 4);
        assert!(top.len() <= 4);
        for (_, p) in &top {
            assert!(p.is_finite());
        }
    }

    #[test]
    fn top_k_returns_empty_for_missing_node() {
        let (trace, weights, tokenizer) = real_trace();
        let top = trace.top_k(&weights, &tokenizer, 99, 99, 5);
        assert!(top.is_empty());
    }

    #[test]
    fn answer_trajectory_one_waypoint_per_layer_including_embedding() {
        let (trace, weights, _) = real_trace();
        let traj = trace.answer_trajectory(&weights, 0);
        // -1 embedding + n_layers transformer layers.
        assert_eq!(traj.len(), trace.n_layers + 1);
        assert_eq!(traj[0].layer, -1);
        for w in &traj {
            assert!(w.prob.is_finite());
            assert!(w.rank >= 1);
            assert!(w.residual_norm.is_finite());
        }
    }

    #[test]
    fn layer_summaries_one_per_layer_including_embedding() {
        let (trace, weights, tokenizer) = real_trace();
        let sums = trace.layer_summaries(&weights, &tokenizer);
        assert_eq!(sums.len(), trace.n_layers + 1);
        assert_eq!(sums[0].layer, -1);
        for s in &sums {
            assert!(s.residual_norm.is_finite());
            assert!(s.attn_delta_norm.is_finite());
            assert!(s.ffn_delta_norm.is_finite());
            // tinymodel is greedy-deterministic; just confirm prob is finite.
            assert!(s.top1_prob.is_finite());
        }
    }
}
