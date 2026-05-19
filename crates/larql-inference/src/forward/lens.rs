//! Logit lens — project an arbitrary-layer residual through the model's
//! final norm + lm_head to read off vocabulary distributions mid-stack.
//!
//! Built on the existing [`super::predict::hidden_to_raw_logits`]
//! projection. No new forward passes; everything here operates on a
//! captured residual (e.g. one returned by a [`super::hooks::RecordHook`]).
//!
//! Three operations cover the lazarus tool surface:
//!
//! - [`logit_lens_topk`] — top-k tokens at a single residual.
//! - [`track_token`] — probability of one specific token at a residual.
//! - [`track_race`] — top-k per layer for a list of residuals (one pass
//!   each, batched in a single call).
//!
//! All three are tokenizer-free — they return raw token IDs and probs.
//! Decode IDs to strings on the caller side if needed.

use super::predict::raw::hidden_to_raw_logits;
use super::softmax;
use crate::model::ModelWeights;
use ndarray::Array2;

/// Top-k `(token_id, probability)` pairs at the given residual, projected
/// through the model's final norm + lm_head. Probabilities sum to 1.0
/// across the full vocab (top-k truncation happens after softmax, not
/// before, so the listed probs are real likelihoods).
///
/// Returns an empty vec on dimension mismatch. NaN-safe top-k: NaN probs
/// sort last and never displace a real hit.
pub fn logit_lens_topk(weights: &ModelWeights, residual: &[f32], k: usize) -> Vec<(u32, f32)> {
    let probs = match residual_to_probs(weights, residual) {
        Some(p) => p,
        None => return Vec::new(),
    };
    topk_from_probs(&probs, k)
}

/// Probability of `target_id` at the given residual. Returns 0.0 on
/// dimension mismatch or out-of-range token id.
pub fn track_token(weights: &ModelWeights, residual: &[f32], target_id: u32) -> f32 {
    let probs = match residual_to_probs(weights, residual) {
        Some(p) => p,
        None => return 0.0,
    };
    let idx = target_id as usize;
    if idx >= probs.len() {
        0.0
    } else {
        probs[idx]
    }
}

/// Top-k per layer for a list of `(layer, residual)` pairs. Equivalent to
/// calling [`logit_lens_topk`] in a loop, but returned in one allocation
/// for caller convenience. Layer ordering preserved.
pub fn track_race(
    weights: &ModelWeights,
    residuals: &[(usize, Vec<f32>)],
    k: usize,
) -> Vec<(usize, Vec<(u32, f32)>)> {
    residuals
        .iter()
        .map(|(layer, r)| (*layer, logit_lens_topk(weights, r, k)))
        .collect()
}

// ── internals ───────────────────────────────────────────────────────────────

fn residual_to_probs(weights: &ModelWeights, residual: &[f32]) -> Option<Vec<f32>> {
    let hidden = weights.hidden_size;
    if residual.len() != hidden {
        return None;
    }
    let h = Array2::from_shape_vec((1, hidden), residual.to_vec()).ok()?;
    let logits = hidden_to_raw_logits(weights, &h);
    Some(softmax(&logits))
}

fn topk_from_probs(probs: &[f32], k: usize) -> Vec<(u32, f32)> {
    let mut indexed: Vec<(usize, f32)> = probs.iter().copied().enumerate().collect();
    let n = indexed.len();
    let k = k.min(n);
    if k == 0 {
        return Vec::new();
    }
    let pivot = k.min(n - 1);
    indexed.select_nth_unstable_by(pivot, cmp_desc_nan_last);
    indexed.truncate(k);
    indexed.sort_unstable_by(cmp_desc_nan_last);
    indexed
        .into_iter()
        .map(|(idx, p)| (idx as u32, p))
        .collect()
}

fn cmp_desc_nan_last(a: &(usize, f32), b: &(usize, f32)) -> std::cmp::Ordering {
    use std::cmp::Ordering;
    match (a.1.is_nan(), b.1.is_nan()) {
        (true, true) => Ordering::Equal,
        (true, false) => Ordering::Greater,
        (false, true) => Ordering::Less,
        _ => b.1.partial_cmp(&a.1).unwrap_or(Ordering::Equal),
    }
}

#[cfg(all(test, not(target_arch = "wasm32")))]
mod tests {
    use super::*;
    use crate::model::ModelWeights;
    use crate::test_utils::make_test_weights;
    use std::sync::OnceLock;

    fn shared_weights() -> &'static ModelWeights {
        static W: OnceLock<ModelWeights> = OnceLock::new();
        W.get_or_init(make_test_weights)
    }

    fn synth_residual(weights: &ModelWeights) -> Vec<f32> {
        // A finite, non-degenerate residual.
        (0..weights.hidden_size)
            .map(|i| (i as f32 + 1.0) * 0.01)
            .collect()
    }

    #[test]
    fn topk_returns_correct_count() {
        let weights = shared_weights();
        let r = synth_residual(weights);
        let result = logit_lens_topk(weights, &r, 5);
        assert_eq!(result.len(), 5);
    }

    #[test]
    fn topk_descending_by_prob() {
        let weights = shared_weights();
        let r = synth_residual(weights);
        let result = logit_lens_topk(weights, &r, 10);
        for w in result.windows(2) {
            assert!(
                w[0].1 >= w[1].1,
                "top-k must be descending: {:?} then {:?}",
                w[0],
                w[1]
            );
        }
    }

    #[test]
    fn topk_probs_in_unit_interval() {
        let weights = shared_weights();
        let r = synth_residual(weights);
        for (_id, p) in logit_lens_topk(weights, &r, 5) {
            assert!((0.0..=1.0).contains(&p), "prob {p} out of range");
            assert!(p.is_finite());
        }
    }

    #[test]
    fn topk_dim_mismatch_returns_empty() {
        let weights = shared_weights();
        let bad = vec![0.0; weights.hidden_size + 1];
        assert!(logit_lens_topk(weights, &bad, 5).is_empty());
    }

    #[test]
    fn topk_zero_k_returns_empty() {
        let weights = shared_weights();
        let r = synth_residual(weights);
        assert!(logit_lens_topk(weights, &r, 0).is_empty());
    }

    #[test]
    fn track_token_matches_topk_when_token_is_top() {
        let weights = shared_weights();
        let r = synth_residual(weights);
        let top = logit_lens_topk(weights, &r, 1);
        assert_eq!(top.len(), 1);
        let (top_id, top_prob) = top[0];
        let tracked = track_token(weights, &r, top_id);
        assert!(
            (tracked - top_prob).abs() < 1e-6,
            "tracked={tracked} top={top_prob}"
        );
    }

    #[test]
    fn track_token_out_of_range_returns_zero() {
        let weights = shared_weights();
        let r = synth_residual(weights);
        assert_eq!(track_token(weights, &r, u32::MAX), 0.0);
    }

    #[test]
    fn track_token_dim_mismatch_returns_zero() {
        let weights = shared_weights();
        let bad = vec![0.0; 1];
        assert_eq!(track_token(weights, &bad, 0), 0.0);
    }

    #[test]
    fn track_race_preserves_layer_order() {
        let weights = shared_weights();
        let r = synth_residual(weights);
        let inputs = vec![(2usize, r.clone()), (0usize, r.clone()), (5usize, r)];
        let race = track_race(weights, &inputs, 3);
        let layers: Vec<usize> = race.iter().map(|(l, _)| *l).collect();
        assert_eq!(layers, vec![2, 0, 5]);
        for (_, top) in &race {
            assert_eq!(top.len(), 3);
        }
    }

    #[test]
    fn track_race_total_prob_per_layer_sums_close_to_full_vocab() {
        // Sanity: top-k of a long-tail distribution should account for
        // *some* mass; nothing pathological.
        let weights = shared_weights();
        let r = synth_residual(weights);
        let race = track_race(weights, &[(0, r)], weights.vocab_size);
        let total: f32 = race[0].1.iter().map(|(_, p)| p).sum();
        assert!(
            (total - 1.0).abs() < 1e-3,
            "full-vocab top-k probs should sum to ~1, got {total}"
        );
    }
}
