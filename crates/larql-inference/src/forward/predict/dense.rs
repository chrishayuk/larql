//! Dense (full-weight) forward passes and logit projection utilities.

use super::super::embed::embed_tokens;
use super::super::layer::run_layer_with_ffn;
use super::super::ple::precompute_per_layer_inputs;
use super::super::{apply_norm, dot_proj};
use super::types::{PredictResult, PredictResultWithResiduals};
use crate::attention::SharedKV;
use crate::ffn::WeightFfn;
use crate::model::ModelWeights;
use ndarray::Array2;

/// Descending order on the probability field of `(index, prob)` pairs,
/// with NaN probabilities treated as the smallest value so they never
/// displace a real top-k hit. Used by every top-k selector in this file
/// — a forward pass that produces the occasional NaN (bad quant, runaway
/// softmax) still surfaces the real maximum instead of whatever NaN
/// happened to land in the pivot.
pub(super) fn cmp_desc_nan_last(a: &(usize, f32), b: &(usize, f32)) -> std::cmp::Ordering {
    use std::cmp::Ordering;
    match (a.1.is_nan(), b.1.is_nan()) {
        (true, true) => Ordering::Equal,
        (true, false) => Ordering::Greater, // NaN sorts after real in descending order
        (false, true) => Ordering::Less,
        _ => b.1.partial_cmp(&a.1).unwrap_or(Ordering::Equal),
    }
}

/// Project the final hidden state to logits and return top-k predictions.
pub fn logits_to_predictions_pub(
    weights: &ModelWeights,
    h: &Array2<f32>,
    tokenizer: &tokenizers::Tokenizer,
    top_k: usize,
    temperature: f32,
) -> PredictResult {
    logits_to_predictions(weights, h, tokenizer, top_k, temperature)
}

pub(crate) fn logits_to_predictions(
    weights: &ModelWeights,
    h: &Array2<f32>,
    tokenizer: &tokenizers::Tokenizer,
    top_k: usize,
    temperature: f32,
) -> PredictResult {
    let seq_len = h.shape()[0];
    let norm_offset = weights.arch.norm_weight_offset();

    let h_final = apply_norm(weights, h, weights.arch.final_norm_key(), norm_offset);

    let logits_scale = weights.arch.logits_scaling();
    let final_softcap = weights.arch.final_logit_softcapping();

    let last_2d = h_final.slice(ndarray::s![seq_len - 1..seq_len, ..]);
    let logits_raw = dot_proj(&last_2d, &weights.lm_head);
    let inv_scale = 1.0 / logits_scale;
    let logits: Vec<f32> = logits_raw
        .row(0)
        .iter()
        .map(|&v| {
            let mut logit = v * inv_scale;
            if let Some(cap) = final_softcap {
                logit = (logit / cap).tanh() * cap;
            }
            logit / temperature.max(1e-6)
        })
        .collect();

    let max_logit = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let exp_sum: f64 = logits.iter().map(|l| ((l - max_logit) as f64).exp()).sum();
    let probs: Vec<f32> = logits
        .iter()
        .map(|l| (((l - max_logit) as f64).exp() / exp_sum) as f32)
        .collect();

    let mut indexed: Vec<(usize, f32)> = probs.iter().copied().enumerate().collect();
    let k = top_k.min(indexed.len());
    // `select_nth_unstable_by(k, …)` requires `k < len`. When the
    // caller asks for the full vocabulary (k == indexed.len()) we
    // skip the partial sort and let the full sort below order
    // everything.
    if k > 0 && k < indexed.len() {
        indexed.select_nth_unstable_by(k, cmp_desc_nan_last);
        indexed.truncate(k);
    }
    indexed.sort_unstable_by(cmp_desc_nan_last);

    let mut predictions = Vec::with_capacity(indexed.len());
    let mut token_ids = Vec::with_capacity(indexed.len());
    for (idx, prob) in indexed {
        let id = idx as u32;
        if let Ok(s) = tokenizer.decode(&[id], true) {
            // Preserve leading whitespace — necessary for autoregressive
            // detokenization where stripping would collapse "Paris" and
            // " Paris" to the same token on re-encode.
            predictions.push((s, prob as f64));
            token_ids.push(id);
        }
    }

    PredictResult {
        predictions,
        token_ids,
    }
}

/// Run a full forward pass and return the top-k next token predictions.
pub fn predict(
    weights: &ModelWeights,
    tokenizer: &tokenizers::Tokenizer,
    token_ids: &[u32],
    top_k: usize,
) -> PredictResult {
    predict_with_temperature(weights, tokenizer, token_ids, top_k, 1.0)
}

pub fn predict_with_temperature(
    weights: &ModelWeights,
    tokenizer: &tokenizers::Tokenizer,
    token_ids: &[u32],
    top_k: usize,
    temperature: f32,
) -> PredictResult {
    let ffn = WeightFfn { weights };
    let num_layers = weights.num_layers;
    let mut h = embed_tokens(weights, token_ids);
    let ple_inputs = precompute_per_layer_inputs(weights, &h, token_ids);
    let mut kv_cache: std::collections::HashMap<usize, SharedKV> = std::collections::HashMap::new();
    for layer in 0..num_layers {
        let shared_kv = weights
            .arch
            .kv_shared_source_layer(layer)
            .and_then(|src| kv_cache.get(&src));
        match run_layer_with_ffn(
            weights,
            &h,
            layer,
            &ffn,
            false,
            ple_inputs.get(layer),
            shared_kv,
        ) {
            Some((h_new, _, kv_out)) => {
                h = h_new;
                if let Some(kv) = kv_out {
                    kv_cache.insert(layer, kv);
                }
            }
            None => continue,
        }
    }
    logits_to_predictions(weights, &h, tokenizer, top_k, temperature)
}

/// Project a single residual vector through final norm + lm_head to get top-1 prediction.
pub fn logit_lens_top1(
    weights: &ModelWeights,
    tokenizer: &tokenizers::Tokenizer,
    residual: &[f32],
) -> Option<(String, f64)> {
    let hidden = weights.hidden_size;
    if residual.len() != hidden {
        return None;
    }

    let h = Array2::from_shape_vec((1, hidden), residual.to_vec()).ok()?;
    let result = logits_to_predictions(weights, &h, tokenizer, 1, 1.0);
    result.predictions.into_iter().next()
}

/// Resume a forward pass from a pre-computed hidden state.
pub fn predict_from_hidden(
    weights: &ModelWeights,
    tokenizer: &tokenizers::Tokenizer,
    h_init: &Array2<f32>,
    start_layer: usize,
    top_k: usize,
) -> PredictResult {
    let ffn = WeightFfn { weights };
    predict_from_hidden_with_ffn(weights, tokenizer, h_init, start_layer, top_k, &ffn, &[])
}

/// Resume a forward pass from a pre-computed hidden state with a custom FFN backend.
pub fn predict_from_hidden_with_ffn(
    weights: &ModelWeights,
    tokenizer: &tokenizers::Tokenizer,
    h_init: &Array2<f32>,
    start_layer: usize,
    top_k: usize,
    ffn: &dyn crate::ffn::FfnBackend,
    token_ids: &[u32],
) -> PredictResult {
    let num_layers = weights.num_layers;
    let mut h = h_init.clone();
    let ple_inputs: Vec<Array2<f32>> = if token_ids.is_empty() {
        Vec::new()
    } else {
        let embeds = embed_tokens(weights, token_ids);
        precompute_per_layer_inputs(weights, &embeds, token_ids)
    };

    for layer in start_layer..num_layers {
        h = match run_layer_with_ffn(weights, &h, layer, ffn, false, ple_inputs.get(layer), None) {
            Some((h_new, _, _)) => h_new,
            None => continue,
        };
    }

    logits_to_predictions(weights, &h, tokenizer, top_k, 1.0)
}

/// Forward pass with residual capture — predictions + per-layer residuals.
pub fn predict_with_ffn_trace(
    weights: &ModelWeights,
    tokenizer: &tokenizers::Tokenizer,
    token_ids: &[u32],
    top_k: usize,
    ffn: &dyn crate::ffn::FfnBackend,
) -> PredictResultWithResiduals {
    let num_layers = weights.num_layers;
    let mut h = embed_tokens(weights, token_ids);
    let ple_inputs = precompute_per_layer_inputs(weights, &h, token_ids);
    let mut residuals = Vec::with_capacity(num_layers);

    for layer in 0..num_layers {
        let last_pos = h.shape()[0] - 1;
        residuals.push(h.row(last_pos).to_vec());

        h = match run_layer_with_ffn(weights, &h, layer, ffn, false, ple_inputs.get(layer), None) {
            Some((h_new, _, _)) => h_new,
            None => continue,
        };
    }

    let result = logits_to_predictions(weights, &h, tokenizer, top_k, 1.0);
    PredictResultWithResiduals {
        predictions: result.predictions,
        residuals,
    }
}

#[cfg(all(test, not(target_arch = "wasm32")))]
mod tests {
    use super::*;
    use crate::test_utils::TestFixtures;
    use ndarray::Array2;

    #[test]
    fn cmp_desc_nan_last_orders_descending_with_nan_last() {
        let mut v = [(0, 1.0f32), (1, 3.0), (2, f32::NAN), (3, 2.0)];
        v.sort_by(cmp_desc_nan_last);
        // 3.0 first (highest), 2.0, 1.0, NaN last.
        let order: Vec<usize> = v.iter().map(|(i, _)| *i).collect();
        assert_eq!(order, vec![1, 3, 0, 2]);
    }

    #[test]
    fn predict_returns_top_k_predictions() {
        let fx = TestFixtures::build();
        let r = predict(&fx.weights, &fx.tokenizer, &[0u32, 1], 5);
        assert!(r.predictions.len() <= 5);
    }

    #[test]
    fn predict_with_temperature_high_temp_smooths_distribution() {
        // Higher temperature → flatter distribution. Top probability at
        // T=10 should be lower than top at T=1 (synthetic weights make
        // the actual predictions chaotic but the relationship holds).
        let fx = TestFixtures::build();
        let cold = predict_with_temperature(&fx.weights, &fx.tokenizer, &[0u32, 1], 10, 1.0);
        let hot = predict_with_temperature(&fx.weights, &fx.tokenizer, &[0u32, 1], 10, 10.0);
        assert_eq!(cold.predictions.len(), hot.predictions.len());
        if !cold.predictions.is_empty() && !hot.predictions.is_empty() {
            assert!(
                cold.predictions[0].1 >= hot.predictions[0].1 - 1e-6,
                "high T should not produce a sharper top-1 than low T"
            );
        }
    }

    #[test]
    fn logit_lens_top1_returns_some_for_correct_shape() {
        let fx = TestFixtures::build();
        let residual = vec![0.5f32; fx.weights.hidden_size];
        let result = logit_lens_top1(&fx.weights, &fx.tokenizer, &residual);
        assert!(result.is_some());
    }

    #[test]
    fn logit_lens_top1_returns_none_for_wrong_shape() {
        let fx = TestFixtures::build();
        let residual = vec![0.5f32; fx.weights.hidden_size + 1];
        assert!(logit_lens_top1(&fx.weights, &fx.tokenizer, &residual).is_none());
    }

    #[test]
    fn predict_from_hidden_resumes_at_start_layer() {
        let fx = TestFixtures::build();
        let h = Array2::<f32>::zeros((2, fx.weights.hidden_size));
        let r = predict_from_hidden(&fx.weights, &fx.tokenizer, &h, 0, 3);
        assert!(r.predictions.len() <= 3);
    }

    #[test]
    fn predict_from_hidden_with_ffn_handles_empty_token_ids() {
        // Empty token_ids → ple_inputs stays empty; predict still runs.
        let fx = TestFixtures::build();
        let h = Array2::<f32>::ones((1, fx.weights.hidden_size));
        let ffn = crate::ffn::WeightFfn {
            weights: &fx.weights,
        };
        let r = predict_from_hidden_with_ffn(&fx.weights, &fx.tokenizer, &h, 0, 5, &ffn, &[]);
        assert!(r.predictions.len() <= 5);
    }

    #[test]
    fn predict_with_ffn_trace_returns_per_layer_residuals() {
        let fx = TestFixtures::build();
        let ffn = crate::ffn::WeightFfn {
            weights: &fx.weights,
        };
        let r = predict_with_ffn_trace(&fx.weights, &fx.tokenizer, &[0u32, 1], 3, &ffn);
        assert_eq!(r.residuals.len(), fx.weights.num_layers);
        for residual in &r.residuals {
            assert_eq!(residual.len(), fx.weights.hidden_size);
            assert!(residual.iter().all(|v| v.is_finite()));
        }
        assert!(r.predictions.len() <= 3);
    }

    #[test]
    fn logits_to_predictions_pub_matches_internal() {
        let fx = TestFixtures::build();
        let h = Array2::<f32>::from_elem((2, fx.weights.hidden_size), 0.1f32);
        let r_pub = logits_to_predictions_pub(&fx.weights, &h, &fx.tokenizer, 5, 1.0);
        let r_priv = logits_to_predictions(&fx.weights, &h, &fx.tokenizer, 5, 1.0);
        assert_eq!(r_pub.predictions.len(), r_priv.predictions.len());
        for ((a_t, a_p), (b_t, b_p)) in r_pub.predictions.iter().zip(r_priv.predictions.iter()) {
            assert_eq!(a_t, b_t);
            assert!((a_p - b_p).abs() < 1e-6);
        }
    }
}
