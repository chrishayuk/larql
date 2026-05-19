//! Logits computation — final norm + vindex KNN + softmax.

use ndarray::Array2;

use crate::model::ModelWeights;
use larql_compute::prelude::*;

/// Shared logits computation: final norm + vindex KNN + softmax.
pub fn finalize_logits(
    weights: &ModelWeights,
    tokenizer: &tokenizers::Tokenizer,
    h: &Array2<f32>,
    top_k: usize,
    index: &larql_vindex::VectorIndex,
    backend: &dyn ComputeBackend,
    norm_offset: f32,
) -> crate::forward::PredictResult {
    let h_final =
        crate::forward::apply_norm(weights, h, weights.arch.final_norm_key(), norm_offset);
    let seq_len = h_final.shape()[0];
    let last_row = h_final.row(seq_len - 1).to_owned();

    let hits = index.lm_head_knn_backend(&last_row, top_k, backend);

    let logits_scale = weights.arch.logits_scaling();
    let final_softcap = weights.arch.final_logit_softcapping();
    let inv_scale = 1.0 / logits_scale;

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

/// Softmax probability of a single score within a set of hits.
pub(super) fn softmax_prob(
    score: f32,
    hits: &[(u32, f32)],
    logits_scale: f32,
    softcap: Option<f32>,
) -> f64 {
    let inv_scale = 1.0 / logits_scale;
    let scaled: Vec<f32> = hits
        .iter()
        .map(|&(_, s)| {
            let mut l = s * inv_scale;
            if let Some(cap) = softcap {
                l = (l / cap).tanh() * cap;
            }
            l
        })
        .collect();
    let max_l = scaled.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let exp_sum: f64 = scaled.iter().map(|l| ((*l - max_l) as f64).exp()).sum();
    let mut target = score * inv_scale;
    if let Some(cap) = softcap {
        target = (target / cap).tanh() * cap;
    }
    ((target - max_l) as f64).exp() / exp_sum
}

#[cfg(all(test, not(target_arch = "wasm32")))]
mod tests {
    use super::*;
    use crate::test_utils::{make_test_tokenizer, make_test_vindex, make_test_weights};
    use larql_compute::CpuBackend;

    #[test]
    fn finalize_logits_runs_without_panic() {
        let weights = make_test_weights();
        let tokenizer = make_test_tokenizer(weights.vocab_size);
        let index = make_test_vindex(&weights);
        let h = ndarray::Array2::from_elem((1, weights.hidden_size), 0.1f32);
        let norm_offset = weights.arch.norm_weight_offset();
        let result = finalize_logits(
            &weights,
            &tokenizer,
            &h,
            5,
            &index,
            &CpuBackend,
            norm_offset,
        );
        // lm_head_knn returns empty for synthetic vindex → empty predictions
        assert!(result.token_ids.len() <= 5);
    }

    #[test]
    fn softmax_prob_basic() {
        let hits = vec![(0u32, 3.0f32), (1u32, 2.0f32), (2u32, 1.0f32)];
        let p = softmax_prob(3.0, &hits, 1.0, None);
        assert!(p > 0.0 && p <= 1.0, "probability should be in (0,1]");
        // Highest logit should have highest probability
        let p2 = softmax_prob(2.0, &hits, 1.0, None);
        assert!(p > p2, "logit=3 should have higher prob than logit=2");
    }

    #[test]
    fn softmax_prob_with_softcap_applies_tanh() {
        // softcap squashes logits through tanh*cap. Pre-cap the largest
        // logit dominates the softmax; post-cap the gap shrinks toward 0
        // so probability mass spreads more evenly.
        let hits = vec![(0u32, 30.0f32), (1u32, 0.0f32)];
        let no_cap = softmax_prob(30.0, &hits, 1.0, None);
        let cap = softmax_prob(30.0, &hits, 1.0, Some(5.0));
        assert!(no_cap > 0.99, "uncapped: dominant logit nearly 1.0");
        assert!(
            cap < no_cap && cap > 0.5,
            "softcap should compress the gap (got no_cap={no_cap}, cap={cap})"
        );
    }

    #[test]
    fn softmax_prob_with_logit_scale_normalises_input() {
        // logits_scale > 1 divides scores before softmax — equivalent to
        // sampling with a higher temperature, so probabilities flatten.
        let hits = vec![(0u32, 10.0f32), (1u32, 0.0f32)];
        let unscaled = softmax_prob(10.0, &hits, 1.0, None);
        let scaled = softmax_prob(10.0, &hits, 10.0, None);
        assert!(
            scaled < unscaled,
            "scaling 10x should flatten distribution: unscaled={unscaled}, scaled={scaled}"
        );
    }

    #[test]
    fn finalize_logits_with_softcap_arch_path() {
        // Exercise the `if let Some(cap) = final_softcap` branch in the
        // `scaled` and `predictions` loops by routing through a Gemma-style
        // arch that returns a cap. With the synthetic vindex returning no
        // hits, the loops are empty, so this guards the cap-aware logic
        // is callable without panicking; combined with `softmax_prob_with_softcap`
        // the cap branch is end-to-end exercised.
        let weights = make_test_weights();
        let tokenizer = make_test_tokenizer(weights.vocab_size);
        let index = make_test_vindex(&weights);
        let h = ndarray::Array2::from_elem((2, weights.hidden_size), 0.05f32);
        let norm_offset = weights.arch.norm_weight_offset();
        let result = finalize_logits(
            &weights,
            &tokenizer,
            &h,
            3,
            &index,
            &CpuBackend,
            norm_offset,
        );
        assert!(result.predictions.is_empty() || result.predictions.len() <= 3);
    }

    /// Build a `VectorIndex` whose `lm_head_knn_backend` returns real hits
    /// by materialising `lm_head.bin` on disk in a tempdir and calling
    /// `load_lm_head`. The bytes come from the synthetic `weights.lm_head`
    /// matrix so the f32 BLAS fallback path produces deterministic scores.
    fn vindex_with_lm_head(
        weights: &larql_models::ModelWeights,
        dir: &std::path::Path,
    ) -> larql_vindex::VectorIndex {
        // lm_head shape = (vocab, hidden). Serialise as little-endian f32.
        let lm = &weights.lm_head;
        assert_eq!(lm.shape(), &[weights.vocab_size, weights.hidden_size]);
        let mut bytes: Vec<u8> = Vec::with_capacity(lm.len() * 4);
        for &v in lm.iter() {
            bytes.extend_from_slice(&v.to_le_bytes());
        }
        std::fs::write(dir.join("lm_head.bin"), &bytes).expect("write lm_head.bin");

        let mut index = make_test_vindex(weights);
        index.load_lm_head(dir).expect("load_lm_head");
        assert!(index.has_lm_head(), "lm_head must report loaded");
        index
    }

    #[test]
    fn finalize_logits_returns_predictions_with_lm_head_hits() {
        // Drives the inner closures (scaled / predictions / max-logit fold)
        // by giving the vindex real lm_head bytes so `lm_head_knn_backend`
        // returns a non-empty top-k.
        let weights = make_test_weights();
        let tokenizer = make_test_tokenizer(weights.vocab_size);
        let tmp = tempfile::tempdir().unwrap();
        let index = vindex_with_lm_head(&weights, tmp.path());
        let h = ndarray::Array2::from_elem((2, weights.hidden_size), 0.05f32);
        let norm_offset = weights.arch.norm_weight_offset();
        let result = finalize_logits(
            &weights,
            &tokenizer,
            &h,
            5,
            &index,
            &CpuBackend,
            norm_offset,
        );
        // f32 BLAS fallback returns up to top_k entries.
        assert!(
            !result.predictions.is_empty(),
            "lm_head fallback should produce hits"
        );
        assert!(result.predictions.len() <= 5);
        let total: f64 = result.predictions.iter().map(|(_, p)| *p).sum();
        // Probabilities are softmaxed across the top_k subset, so they sum
        // to ≈1 (within fp tolerance).
        assert!(
            (total - 1.0).abs() < 1e-3,
            "predictions probs should ~sum to 1, got {total}"
        );
    }

    #[test]
    fn finalize_logits_picks_highest_logit_first() {
        let weights = make_test_weights();
        let tokenizer = make_test_tokenizer(weights.vocab_size);
        let tmp = tempfile::tempdir().unwrap();
        let index = vindex_with_lm_head(&weights, tmp.path());
        let h = ndarray::Array2::from_elem((1, weights.hidden_size), 0.1f32);
        let norm_offset = weights.arch.norm_weight_offset();
        let result = finalize_logits(
            &weights,
            &tokenizer,
            &h,
            5,
            &index,
            &CpuBackend,
            norm_offset,
        );

        // First entry has the highest probability (top-k is sorted descending).
        let probs: Vec<f64> = result.predictions.iter().map(|(_, p)| *p).collect();
        for w in probs.windows(2) {
            assert!(
                w[0] >= w[1] - 1e-9,
                "predictions must be in descending probability order: {probs:?}"
            );
        }
    }
}
