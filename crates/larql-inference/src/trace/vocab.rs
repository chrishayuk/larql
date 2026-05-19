//! Vocabulary projection helpers — project residual vectors through lm_head.

use crate::model::ModelWeights;
use larql_models::NormType;
use ndarray::Array2;

/// Project a vector through final_norm → lm_head → logits.
pub fn project_to_logits(weights: &ModelWeights, vec: &[f32]) -> Vec<f32> {
    let hidden = weights.hidden_size;
    let norm_offset = weights.arch.norm_weight_offset();

    let v = Array2::from_shape_vec((1, hidden), vec.to_vec()).unwrap();
    let normed = apply_norm(weights, &v, weights.arch.final_norm_key(), norm_offset);
    let normed_row = normed.row(0);

    let logits_scale = weights.arch.logits_scaling();
    let softcap = weights.arch.final_logit_softcapping();
    let mut logits = Vec::with_capacity(weights.vocab_size);
    for tok_id in 0..weights.vocab_size {
        let lm_row = weights.lm_head.row(tok_id);
        let dot: f64 = normed_row
            .iter()
            .zip(lm_row.iter())
            .map(|(&a, &b)| a as f64 * b as f64)
            .sum();
        let mut logit = (dot / logits_scale as f64) as f32;
        if let Some(cap) = softcap {
            logit = (logit / cap).tanh() * cap;
        }
        logits.push(logit);
    }
    logits
}

pub use crate::forward::softmax;

pub fn top_k_from_logits(
    logits: &[f32],
    tokenizer: &tokenizers::Tokenizer,
    k: usize,
) -> Vec<(String, f32)> {
    let probs = softmax(logits);
    let mut indexed: Vec<(usize, f32)> = probs.iter().copied().enumerate().collect();
    let k = k.min(indexed.len());
    // `select_nth_unstable_by(k, ...)` requires `k < len`. When the
    // caller asks for the entire vocabulary (k == len) — or k == 0 for
    // an empty vocab — skip the partition; the subsequent sort handles
    // ordering on its own.
    if k > 0 && k < indexed.len() {
        indexed.select_nth_unstable_by(k, |a, b| b.1.partial_cmp(&a.1).unwrap());
        indexed.truncate(k);
    } else if k == 0 {
        return Vec::new();
    }
    indexed.sort_unstable_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
    indexed
        .into_iter()
        .filter_map(|(idx, prob)| {
            tokenizer
                .decode(&[idx as u32], true)
                .ok()
                .map(|s| (s.trim().to_string(), prob))
        })
        .collect()
}

pub fn vec_norm(v: &[f32]) -> f32 {
    v.iter().map(|x| x * x).sum::<f32>().sqrt()
}

fn apply_norm(
    weights: &ModelWeights,
    x: &Array2<f32>,
    weight_key: &str,
    norm_offset: f32,
) -> Array2<f32> {
    match weights.arch.norm_type() {
        NormType::LayerNorm => {
            let bias_key = weight_key.replace(".weight", ".bias");
            crate::residual::layer_norm(
                x,
                weights.vectors.get(weight_key),
                weights.vectors.get(&bias_key),
            )
        }
        _ => crate::residual::rms_norm(x, weights.vectors.get(weight_key), norm_offset),
    }
}

#[cfg(all(test, not(target_arch = "wasm32")))]
mod tests {
    use super::*;
    use crate::test_utils::make_test_weights;

    #[test]
    fn vec_norm_known_value() {
        assert!((vec_norm(&[3.0f32, 4.0]) - 5.0).abs() < 1e-5);
    }

    #[test]
    fn vec_norm_zero_vector() {
        assert_eq!(vec_norm(&[0.0f32, 0.0]), 0.0);
    }

    #[test]
    fn project_to_logits_returns_vocab_size_values() {
        let w = make_test_weights();
        let logits = project_to_logits(&w, &vec![0.1f32; w.hidden_size]);
        assert_eq!(logits.len(), w.vocab_size);
        assert!(logits.iter().all(|v| v.is_finite()));
    }

    #[test]
    fn project_to_logits_nonzero_input_gives_nonzero_output() {
        let w = make_test_weights();
        let logits = project_to_logits(&w, &vec![1.0f32; w.hidden_size]);
        assert!(logits.iter().any(|v| v.abs() > 1e-8));
    }

    #[test]
    fn project_to_logits_layernorm_arch_routes_through_layernorm_branch() {
        // Starcoder2 has NormType::LayerNorm — exercises the LayerNorm
        // arm in `apply_norm` (vs the default RMSNorm).
        let w = crate::test_utils::make_starcoder2_test_weights();
        let logits = project_to_logits(&w, &vec![0.1f32; w.hidden_size]);
        assert_eq!(logits.len(), w.vocab_size);
        assert!(logits.iter().all(|v| v.is_finite()));
    }

    #[test]
    fn top_k_from_logits_returns_top_predictions() {
        let w = make_test_weights();
        let tok = crate::test_utils::make_test_tokenizer(w.vocab_size);
        let logits: Vec<f32> = (0..w.vocab_size).map(|i| i as f32 * 0.1).collect();
        let top = top_k_from_logits(&logits, &tok, 3);
        assert!(top.len() <= 3);
        // Sorted by prob descending.
        for w in top.windows(2) {
            assert!(w[0].1 >= w[1].1, "top-K not sorted by prob");
        }
        // Last element of logits is highest, so its decoded form should be first.
        if let Some((_, top_prob)) = top.first() {
            assert!(top_prob.is_finite());
        }
    }

    #[test]
    fn top_k_from_logits_k_zero_returns_empty() {
        let w = make_test_weights();
        let tok = crate::test_utils::make_test_tokenizer(w.vocab_size);
        let logits = vec![1.0f32; w.vocab_size];
        let top = top_k_from_logits(&logits, &tok, 0);
        assert!(top.is_empty());
    }

    #[test]
    fn top_k_from_logits_k_larger_than_vocab_clamps() {
        let w = make_test_weights();
        let tok = crate::test_utils::make_test_tokenizer(w.vocab_size);
        let logits = vec![1.0f32; 5];
        let top = top_k_from_logits(&logits, &tok, 1_000_000);
        assert_eq!(top.len(), 5);
    }
}
