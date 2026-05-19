//! Dense (full-weight) forward passes and logit projection utilities.

use super::super::apply_norm;
use super::super::embed::embed_tokens;
use super::super::layer::run_layer_with_ffn;
use super::super::ple::precompute_per_layer_inputs;
use super::types::{PredictResult, PredictResultWithResiduals};
use crate::attention::SharedKV;
use crate::ffn::WeightFfn;
use crate::model::ModelWeights;
use ndarray::Array2;
use rayon::prelude::*;

/// Row-parallel matvec: `out[v] = sum_h x[0, h] * lm_head[v, h]`.
/// `lm_head` is `[vocab, hidden]` row-major; `x` is `[1, hidden]`.
/// Each row's dot product runs independently; rayon fans out across
/// performance cores and the inner dot dispatches to a NEON kernel on
/// aarch64. Bypasses ndarray's BLAS fall-back, which collapses to
/// scalar on `lm_head.t()` (transposed view = non-standard layout).
fn parallel_lm_head_logits(
    x: &ndarray::ArrayView2<'_, f32>,
    lm_head: &larql_models::WeightArray,
) -> Vec<f32> {
    let hidden = lm_head.shape()[1];
    let vocab = lm_head.shape()[0];
    let x_row: &[f32] = x.row(0).to_slice().expect("h_final last row contiguous");
    let lm_slice: &[f32] = lm_head
        .as_slice()
        .expect("lm_head expected contiguous row-major");
    let mut out = vec![0.0f32; vocab];
    out.par_iter_mut().enumerate().for_each(|(v, slot)| {
        let row = &lm_slice[v * hidden..(v + 1) * hidden];
        *slot = f32_dot(row, x_row);
    });
    out
}

/// f32 vector dot product. Dispatches to NEON on aarch64 (Apple
/// Silicon always has it), scalar elsewhere. Handles arbitrary
/// length; processes 16 elements per NEON iteration with 4-wide FMA
/// accumulators, scalar tail.
#[inline]
fn f32_dot(a: &[f32], b: &[f32]) -> f32 {
    debug_assert_eq!(a.len(), b.len());
    #[cfg(target_arch = "aarch64")]
    {
        unsafe { f32_dot_neon(a, b) }
    }
    #[cfg(not(target_arch = "aarch64"))]
    {
        let mut acc = 0.0f32;
        for k in 0..a.len() {
            acc += a[k] * b[k];
        }
        acc
    }
}

#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
unsafe fn f32_dot_neon(a: &[f32], b: &[f32]) -> f32 {
    use core::arch::aarch64::*;
    let n = a.len();
    let mut i = 0usize;
    // Four independent accumulators to hide FMA latency (4-cycle on
    // M3); we process 16 lanes per iteration.
    let mut acc0 = vdupq_n_f32(0.0);
    let mut acc1 = vdupq_n_f32(0.0);
    let mut acc2 = vdupq_n_f32(0.0);
    let mut acc3 = vdupq_n_f32(0.0);
    while i + 16 <= n {
        let ap = a.as_ptr().add(i);
        let bp = b.as_ptr().add(i);
        let a0 = vld1q_f32(ap);
        let a1 = vld1q_f32(ap.add(4));
        let a2 = vld1q_f32(ap.add(8));
        let a3 = vld1q_f32(ap.add(12));
        let b0 = vld1q_f32(bp);
        let b1 = vld1q_f32(bp.add(4));
        let b2 = vld1q_f32(bp.add(8));
        let b3 = vld1q_f32(bp.add(12));
        acc0 = vfmaq_f32(acc0, a0, b0);
        acc1 = vfmaq_f32(acc1, a1, b1);
        acc2 = vfmaq_f32(acc2, a2, b2);
        acc3 = vfmaq_f32(acc3, a3, b3);
        i += 16;
    }
    let sum01 = vaddq_f32(acc0, acc1);
    let sum23 = vaddq_f32(acc2, acc3);
    let sum = vaddq_f32(sum01, sum23);
    let mut acc = vaddvq_f32(sum);
    // Scalar tail. hidden = 2560 is /16 cleanly on Gemma 3 4B so this
    // loop is unreached, but keep it for correctness on other shapes.
    while i < n {
        acc += a[i] * b[i];
        i += 1;
    }
    acc
}

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

/// Q4_K-aware variant: when the vindex carries a synthesized Q4_K view
/// of the LM head, route the matmul through it. Drops lm_head
/// bandwidth from ~2.7 GB (f32) to ~0.7 GB (Q4_K) per step on a 262K-
/// vocab head like Gemma 3 4B. Falls back to the f32 path when the
/// vindex doesn't have a Q4 lm_head.
///
/// Gemma 3 / Llama tied-embedding models always get a Q4_K view via
/// `synthesize_lm_head_kquant` at vindex load. Untied models need a
/// separate `lm_head_q4.bin` (extract with the quantised writer).
#[allow(clippy::too_many_arguments)]
pub fn logits_to_predictions_q4_lm_head(
    weights: &ModelWeights,
    h: &Array2<f32>,
    q4_lm_head: &[u8],
    vocab: usize,
    backend: &dyn larql_compute::ComputeBackend,
    tokenizer: &tokenizers::Tokenizer,
    top_k: usize,
    temperature: f32,
) -> PredictResult {
    let seq_len = h.shape()[0];
    let norm_offset = weights.arch.norm_weight_offset();
    let h_final = apply_norm(weights, h, weights.arch.final_norm_key(), norm_offset);
    let hidden = h_final.shape()[1];

    let last_row: &[f32] = h_final
        .row(seq_len - 1)
        .to_slice()
        .or_else(|| h_final.as_slice())
        .expect("final hidden last row contiguous");

    let logits_scale = weights.arch.logits_scaling();
    let final_softcap = weights.arch.final_logit_softcapping();
    let inv_scale = 1.0 / logits_scale;
    let inv_temp = 1.0 / temperature.max(1e-6);

    // Q4_K × Q8_K via NEON sdot for the lm_head matvec — same approach
    // as the per-layer projections. Quantising the hidden vector once
    // amortises across the full 262K-vocab matmul. Wrapped with
    // `par_chunks_mut` because `q4k_q8k_matvec_into` itself is
    // single-threaded; vocab=262K is more than enough to scale
    // linearly across M3 Max's 11 perf cores.
    let raw = {
        use larql_compute::cpu::ops::q4k_q8k_dot::{
            q4k_q8k_matvec_into, quantize_x_to_q8k_into, Q8KActivation,
        };
        use rayon::prelude::*;
        let mut h_q8k = Q8KActivation::with_capacity(hidden);
        quantize_x_to_q8k_into(&mut h_q8k, last_row);
        let bytes_per_row = (hidden / 256) * 144; // Q4_K row size
        let mut out = vec![0.0f32; vocab];
        const CHUNK_ROWS: usize = 64;
        out.par_chunks_mut(CHUNK_ROWS)
            .enumerate()
            .for_each(|(chunk_idx, chunk)| {
                let row_start = chunk_idx * CHUNK_ROWS;
                let chunk_len = chunk.len().min(vocab.saturating_sub(row_start));
                if chunk_len == 0 {
                    return;
                }
                let w_chunk =
                    &q4_lm_head[row_start * bytes_per_row..(row_start + chunk_len) * bytes_per_row];
                q4k_q8k_matvec_into(&mut chunk[..chunk_len], &h_q8k, w_chunk, chunk_len, hidden);
            });
        out
    };
    let _ = backend;
    let _ = parallel_lm_head_logits;

    let logits: Vec<f32> = raw
        .into_iter()
        .map(|v| {
            let mut logit = v * inv_scale;
            if let Some(cap) = final_softcap {
                logit = (logit / cap).tanh() * cap;
            }
            logit * inv_temp
        })
        .collect();

    finalize_topk_predictions(logits, tokenizer, top_k)
}

/// Shared softmax + top-k decode used by both the f32 and Q4 lm_head
/// paths. Pulled out so the two flavours diverge only in how they
/// compute the raw logits.
fn finalize_topk_predictions(
    logits: Vec<f32>,
    tokenizer: &tokenizers::Tokenizer,
    top_k: usize,
) -> PredictResult {
    let max_logit = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let exp_sum: f64 = logits.iter().map(|l| ((l - max_logit) as f64).exp()).sum();
    let probs: Vec<f32> = logits
        .iter()
        .map(|l| (((l - max_logit) as f64).exp() / exp_sum) as f32)
        .collect();

    let mut indexed: Vec<(usize, f32)> = probs.iter().copied().enumerate().collect();
    let k = top_k.min(indexed.len());
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
            predictions.push((s, prob as f64));
            token_ids.push(id);
        }
    }

    PredictResult {
        predictions,
        token_ids,
    }
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
    // ndarray's `last_2d.dot(&lm_head.t())` falls off the BLAS fast path
    // because `lm_head.t()` is a transposed view (non-standard layout) —
    // sgemv was running at ~10 GB/s on M3 Max. Hand-roll the row-parallel
    // dot product over `lm_head` (row-major, shape [vocab, hidden]) so
    // we read each row contiguously and let rayon spread vocab across
    // cores. Measured ~10× faster on Gemma 3 4B's 262K × 2560 head.
    let logits_raw = parallel_lm_head_logits(&last_2d, &weights.lm_head);
    let inv_scale = 1.0 / logits_scale;
    let logits: Vec<f32> = logits_raw
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

#[cfg(test)]
mod dot_tests {
    use super::*;

    fn scalar_dot(a: &[f32], b: &[f32]) -> f32 {
        let mut s = 0.0f32;
        for k in 0..a.len() {
            s += a[k] * b[k];
        }
        s
    }

    #[test]
    fn f32_dot_matches_scalar_on_aligned_length() {
        // 2560 = Gemma 3 4B hidden — clean multiple of 16.
        let a: Vec<f32> = (0..2560).map(|i| (i as f32 * 0.013).sin()).collect();
        let b: Vec<f32> = (0..2560).map(|i| (i as f32 * 0.021).cos()).collect();
        let s = scalar_dot(&a, &b);
        let g = f32_dot(&a, &b);
        // Pairwise-summed NEON ordering vs left-to-right scalar — allow
        // small relative drift.
        let rel = ((s - g).abs() / s.abs().max(1e-6)) as f64;
        assert!(rel < 1e-4, "scalar={s} neon={g}");
    }

    #[test]
    fn f32_dot_handles_unaligned_tail() {
        // 23 is not a multiple of 16 — exercises the scalar tail.
        let a: Vec<f32> = (0..23).map(|i| (i + 1) as f32).collect();
        let b: Vec<f32> = (0..23).map(|i| (i as f32 * 0.5) + 1.0).collect();
        let s = scalar_dot(&a, &b);
        let g = f32_dot(&a, &b);
        assert!((s - g).abs() < 1e-4, "scalar={s} neon={g}");
    }

    #[test]
    fn f32_dot_empty_returns_zero() {
        assert_eq!(f32_dot(&[], &[]), 0.0);
    }
}

#[cfg(test)]
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

#[cfg(test)]
mod q4_lm_head_tests {
    use super::*;
    use crate::test_utils::Q4KTestFixtures;

    /// Round-trip the Q4_K lm_head matmul path: build a synthetic
    /// hidden vector, run it through `logits_to_predictions_q4_lm_head`
    /// against the synth Q4_K lm_head bytes, and confirm the
    /// predictions are well-formed.
    #[test]
    fn logits_to_predictions_q4_lm_head_produces_valid_top_k() {
        let fx = Q4KTestFixtures::build();
        let h = Array2::from_shape_fn((1, fx.weights.hidden_size), |(_, j)| {
            ((j as f32) * 0.013).sin() * 0.5
        });
        let q4_bytes = fx
            .index
            .storage
            .lm_head_kquant_view()
            .expect("synth Q4 lm_head present")
            .as_ref()
            .to_vec();
        let backend = larql_compute::CpuBackend;

        let result = logits_to_predictions_q4_lm_head(
            &fx.weights,
            &h,
            &q4_bytes,
            fx.weights.vocab_size,
            &backend,
            &fx.tokenizer,
            5,
            1.0,
        );

        // Top-5 with vocab=256 → returns 5 entries.
        assert_eq!(result.token_ids.len(), 5);
        assert_eq!(result.predictions.len(), 5);
        // Probabilities should be in [0, 1] and descending.
        for (i, (_, p)) in result.predictions.iter().enumerate() {
            assert!(
                *p >= 0.0 && *p <= 1.0,
                "prob[{i}] = {p} should be in [0, 1]"
            );
            if i > 0 {
                let prev = result.predictions[i - 1].1;
                assert!(prev >= *p, "prob should be descending: prev={prev} cur={p}");
            }
        }
        // Top token is a valid vocab ID.
        assert!((result.token_ids[0] as usize) < fx.weights.vocab_size);
    }

    /// Multi-position hidden state: the function takes the LAST row.
    #[test]
    fn logits_to_predictions_q4_lm_head_uses_last_row() {
        let fx = Q4KTestFixtures::build();
        // Two-row hidden: row 0 deliberately set to a value that
        // would yield different argmax than row 1 — confirms we read
        // the last row, not the first.
        let mut h = Array2::<f32>::zeros((2, fx.weights.hidden_size));
        for j in 0..fx.weights.hidden_size {
            h[[0, j]] = -1.0; // row 0 (ignored)
            h[[1, j]] = ((j as f32) * 0.013).sin() * 0.5; // row 1 (used)
        }
        // Single-row reference for comparison.
        let h1 = Array2::from_shape_fn((1, fx.weights.hidden_size), |(_, j)| {
            ((j as f32) * 0.013).sin() * 0.5
        });
        let q4_bytes = fx
            .index
            .storage
            .lm_head_kquant_view()
            .expect("Q4 lm_head present")
            .as_ref()
            .to_vec();
        let backend = larql_compute::CpuBackend;

        let r2 = logits_to_predictions_q4_lm_head(
            &fx.weights,
            &h,
            &q4_bytes,
            fx.weights.vocab_size,
            &backend,
            &fx.tokenizer,
            1,
            1.0,
        );
        let r1 = logits_to_predictions_q4_lm_head(
            &fx.weights,
            &h1,
            &q4_bytes,
            fx.weights.vocab_size,
            &backend,
            &fx.tokenizer,
            1,
            1.0,
        );
        assert_eq!(
            r2.token_ids[0], r1.token_ids[0],
            "two-row input's last row should produce the same top-1 as the single-row input"
        );
    }

    /// Temperature scales the softmax — high T flattens the
    /// distribution, low T sharpens it. Confirm the top-1 probability
    /// is non-increasing as T grows.
    #[test]
    fn logits_to_predictions_q4_lm_head_high_temp_flattens() {
        let fx = Q4KTestFixtures::build();
        let h = Array2::from_shape_fn((1, fx.weights.hidden_size), |(_, j)| {
            ((j as f32) * 0.011).cos() * 0.4
        });
        let q4_bytes = fx
            .index
            .storage
            .lm_head_kquant_view()
            .expect("Q4 lm_head present")
            .as_ref()
            .to_vec();
        let backend = larql_compute::CpuBackend;

        let cold = logits_to_predictions_q4_lm_head(
            &fx.weights,
            &h,
            &q4_bytes,
            fx.weights.vocab_size,
            &backend,
            &fx.tokenizer,
            1,
            1.0,
        );
        let hot = logits_to_predictions_q4_lm_head(
            &fx.weights,
            &h,
            &q4_bytes,
            fx.weights.vocab_size,
            &backend,
            &fx.tokenizer,
            1,
            10.0,
        );
        assert!(
            cold.predictions[0].1 >= hot.predictions[0].1 - 1e-6,
            "T=10 should not produce a sharper top-1 than T=1: cold={} hot={}",
            cold.predictions[0].1,
            hot.predictions[0].1
        );
    }
}
