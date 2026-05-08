//! MoE router index — load and query router weights for expert selection.
//!
//! For MoE models, the router determines which experts handle each input.
//! The router weights are small (128 × hidden_size × num_layers) and stored
//! at full precision (bf16/f16), making them a reliable knowledge signal.
//!
//! Used by MoE DESCRIBE to show which experts activate for an entity.

use std::path::Path;

use ndarray::{Array1, Array2};

/// MoE router weights for all layers.
pub struct RouterIndex {
    /// Per-layer router weight matrices: `[num_experts, hidden_size]`
    pub weights: Vec<Array2<f32>>,
    /// Per-layer router biases: `[num_experts]`
    pub biases: Vec<Array1<f32>>,
    /// Number of experts per layer.
    pub num_experts: usize,
    /// Top-K experts selected per token.
    pub top_k: usize,
}

/// Result of routing an entity through the MoE router.
#[derive(Debug, Clone)]
pub struct RouteResult {
    /// Selected expert IDs, ordered by score.
    pub experts: Vec<usize>,
    /// Softmax probabilities for each selected expert.
    pub probs: Vec<f32>,
    /// Raw router scores for each selected expert.
    pub scores: Vec<f32>,
}

impl RouterIndex {
    /// Load router weights from a vindex directory.
    /// Returns None if router_weights.bin doesn't exist (dense model).
    pub fn load(dir: &Path, config: &crate::config::VindexConfig) -> Option<Self> {
        let path = dir.join("router_weights.bin");
        if !path.exists() {
            return None;
        }

        let moe_config = config.model_config.as_ref()?.moe.as_ref()?;
        let num_experts = moe_config.num_experts;
        let top_k = moe_config.top_k;
        let hidden_size = config.hidden_size;
        let num_layers = config.num_layers;

        let file = std::fs::File::open(&path).ok()?;
        let mmap = unsafe { memmap2::Mmap::map(&file).ok()? };
        let floats = crate::config::dtype::decode_floats(&mmap, config.dtype);

        let weight_size = num_experts * hidden_size;
        let bias_size = num_experts;
        let per_layer = weight_size + bias_size;

        let mut weights = Vec::with_capacity(num_layers);
        let mut biases = Vec::with_capacity(num_layers);

        for layer in 0..num_layers {
            let base = layer * per_layer;
            if base + per_layer > floats.len() {
                break;
            }

            let w_data = &floats[base..base + weight_size];
            let w = Array2::from_shape_vec((num_experts, hidden_size), w_data.to_vec()).ok()?;
            weights.push(w);

            let b_data = &floats[base + weight_size..base + per_layer];
            let b = Array1::from_vec(b_data.to_vec());
            biases.push(b);
        }

        Some(RouterIndex {
            weights,
            biases,
            num_experts,
            top_k,
        })
    }

    /// Route an entity embedding through the router at a specific layer.
    pub fn route(&self, layer: usize, embedding: &Array1<f32>) -> Option<RouteResult> {
        if layer >= self.weights.len() {
            return None;
        }

        let hidden = embedding.len();
        let x = embedding.view().into_shape_with_order((1, hidden)).unwrap();
        let cpu = larql_compute::CpuBackend;
        use larql_compute::MatMul;
        // weights[layer] is (num_experts, hidden_size) — HF nn.Linear convention
        // (out_features × in_features). To compute scores = x @ W.T (yielding
        // [1, num_experts]) we use matmul_transb. Plain matmul would require
        // hidden_size == num_experts, which only happens by accident.
        let proj = cpu.matmul_transb(x, self.weights[layer].view()); // [1, num_experts]
        let scores_1d = ndarray::Array1::from_vec(proj.into_raw_vec_and_offset().0);
        let scores_raw = scores_1d + &self.biases[layer];

        // Top-K selection
        let mut indexed: Vec<(usize, f32)> = scores_raw.iter().copied().enumerate().collect();
        indexed.sort_unstable_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
        indexed.truncate(self.top_k);

        let experts: Vec<usize> = indexed.iter().map(|(i, _)| *i).collect();
        let scores: Vec<f32> = indexed.iter().map(|(_, s)| *s).collect();

        // Softmax of top-K
        let max_score = scores[0];
        let exp_scores: Vec<f32> = scores.iter().map(|s| (s - max_score).exp()).collect();
        let sum: f32 = exp_scores.iter().sum();
        let probs: Vec<f32> = exp_scores.iter().map(|e| e / sum).collect();

        Some(RouteResult {
            experts,
            probs,
            scores,
        })
    }

    /// Route an entity across all layers and find the most common experts.
    pub fn route_all_layers(
        &self,
        embedding: &Array1<f32>,
        layer_range: std::ops::RangeInclusive<usize>,
    ) -> Vec<(usize, usize, f32)> {
        // Count how often each expert is selected across layers, with avg probability
        let mut expert_counts: std::collections::HashMap<usize, (usize, f32)> =
            std::collections::HashMap::new();

        for layer in layer_range {
            if let Some(result) = self.route(layer, embedding) {
                for (i, &eid) in result.experts.iter().enumerate() {
                    let entry = expert_counts.entry(eid).or_insert((0, 0.0));
                    entry.0 += 1;
                    entry.1 += result.probs[i];
                }
            }
        }

        let mut sorted: Vec<(usize, usize, f32)> = expert_counts
            .into_iter()
            .map(|(eid, (count, total_prob))| (eid, count, total_prob / count as f32))
            .collect();
        sorted.sort_unstable_by(|a, b| b.1.cmp(&a.1).then(b.2.partial_cmp(&a.2).unwrap()));
        sorted
    }
}
