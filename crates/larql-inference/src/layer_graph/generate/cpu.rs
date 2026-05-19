//! CPU Q4K generate path — used when the active backend does not support the
//! fused Q4 prefill + KV-cached decode pipeline (today: CpuBackend).

use super::{
    eos::EosConfig,
    types::{GenerateError, GenerateResult, StageTimings},
};
use crate::forward::PredictResult;
use crate::model::ModelWeights;
use larql_compute::prelude::*;

// ── Backend capability probe + CPU Q4K delegation ────────────────────────────
//
// `generate` / `generate_constrained` assume the backend implements the fused
// Q4 prefill + KV-cached decode pipeline (currently: Metal). Backends that
// lack it (CpuBackend) delegate to the per-layer CPU Q4K dequant path
// (`predict_kquant_hidden`), which mutates `weights.tensors` per layer — that's
// the single reason these functions take `&mut ModelWeights`.

/// True when the backend can handle the fused Q4 prefill + decode pipeline
/// directly. Metal: yes. Pure CPU: no — that path produces correct forward
/// results via the vindex Q4K dequant loop in `crate::vindex::kquant_forward`.
pub(super) fn backend_supports_fused_q4_pipeline(backend: &dyn ComputeBackend) -> bool {
    backend.supports(Capability::PrefillQ4) && backend.supports(Capability::DecodeToken)
}

/// CPU Q4K generate path. For dense single-stream architectures (no
/// hybrid MoE, no cross-layer KV sharing) this uses the KV-cached
/// driver in [`crate::vindex::predict_kquant_prefill`] +
/// [`crate::vindex::predict_kquant_decode_step`]: full prompt once at
/// prefill, then 1-row attention + 1-row FFN per generated token.
/// Falls back to the original O(N²) per-step `predict_kquant_hidden`
/// loop for hybrid MoE (Gemma 4 26B A4B) and Gemma 4 E2B
/// (cross-layer KV sharing).
pub(super) fn generate_via_cpu_q4k(
    weights: &mut ModelWeights,
    tokenizer: &tokenizers::Tokenizer,
    token_ids: &[u32],
    max_tokens: usize,
    index: &larql_vindex::VectorIndex,
    eos: &EosConfig,
) -> GenerateResult {
    if max_tokens == 0 {
        return GenerateResult::empty_success();
    }

    if crate::vindex::supports_cached_decode(weights) {
        let use_direct = crate::vindex::supports_direct_matvec_decode(weights, index);
        generate_via_cpu_q4k_cached(
            weights, tokenizer, token_ids, max_tokens, index, eos, use_direct,
        )
    } else {
        generate_via_cpu_q4k_uncached(weights, tokenizer, token_ids, max_tokens, index, eos)
    }
}

/// KV-cached path. Decode work per step is O(1) in N (single-row
/// attention vs growing K/V) instead of O(N²). Dense architectures
/// without cross-layer KV sharing only.
///
/// `direct_matvec`: when true, decode steps skip the per-layer Q4_K →
/// f32 dequant staging and call `backend.quant_matvec` against the
/// vindex's raw Q4_K/Q6_K bytes directly. Massive win — dequant was
/// ~93% of CPU forward time on Gemma 3 4B Q4_K.
#[allow(clippy::too_many_arguments)]
fn generate_via_cpu_q4k_cached(
    weights: &mut ModelWeights,
    tokenizer: &tokenizers::Tokenizer,
    token_ids: &[u32],
    max_tokens: usize,
    index: &larql_vindex::VectorIndex,
    eos: &EosConfig,
    direct_matvec: bool,
) -> GenerateResult {
    // ── Prefill ────────────────────────────────────────────────────
    let prefill_start = std::time::Instant::now();
    let (h_prompt, mut cache, _prefill_timings) =
        crate::vindex::predict_kquant_prefill(weights, token_ids, index);
    // Don't fold prefill dequant into per-step averages — bench numbers
    // already account for the prompt pass via `prefill_ms`. Mixing them
    // here would mis-attribute the one-shot prefill cost to decode.

    let backend: Box<dyn larql_compute::ComputeBackend> = Box::new(larql_compute::CpuBackend);
    // Q4_K view of the lm_head, synthesised from the f16 embeddings at
    // vindex load for tied-embedding models (Gemma, Llama). When
    // available we route the lm_head matmul through `q4k_matvec`
    // instead of the f32 row-parallel sgemv — drops lm_head bandwidth
    // from ~2.7 GB to ~0.7 GB per step.
    let lm_head_q4: Option<&[u8]> = index.storage.lm_head_kquant_view().map(|b| b.as_ref());
    let lm_head_vocab = index.vocab_size;

    // lm_head + argmax on the last prompt position to seed decode.
    let h_last = last_row_as_2d(&h_prompt);
    let lm_head_start = std::time::Instant::now();
    let first = lm_head_predict(
        weights,
        &h_last,
        tokenizer,
        backend.as_ref(),
        lm_head_q4,
        lm_head_vocab,
    );
    let mut t_lm_head = lm_head_start.elapsed().as_secs_f64() * 1000.0;
    let prefill_ms = prefill_start.elapsed().as_secs_f64() * 1000.0;

    let mut tokens: Vec<(String, f64)> = Vec::with_capacity(max_tokens);
    let mut decode_ms = Vec::with_capacity(max_tokens);
    let mut t_cpu_fwd = 0.0f64;
    let mut t_dequant = 0.0f64;

    let mut next_id = match (first.token_ids.first(), first.predictions.first()) {
        (Some(&id), Some(first_pred)) => {
            tokens.push((first_pred.0.clone(), 1.0));
            if eos.is_eos_with_tokenizer(id, &first_pred.0, tokenizer) {
                return GenerateResult {
                    tokens,
                    prefill_ms,
                    decode_ms,
                    stage_timings: StageTimings {
                        cpu_fwd_ms_total: t_cpu_fwd,
                        lm_head_ms_total: t_lm_head,
                        dequant_ms_total: t_dequant,
                        ..Default::default()
                    },
                    error: None,
                };
            }
            id
        }
        _ => {
            return GenerateResult {
                tokens,
                prefill_ms,
                decode_ms,
                stage_timings: StageTimings::default(),
                error: Some(GenerateError::empty_output(
                    "CPU Q4K generation produced no first token",
                )),
            };
        }
    };

    // ── Decode loop ────────────────────────────────────────────────
    let prompt_len = token_ids.len();
    for step in 1..max_tokens {
        let abs_position = prompt_len + (step - 1);
        let t0 = std::time::Instant::now();
        let h_new = if direct_matvec {
            match crate::vindex::predict_kquant_decode_step_direct(
                weights,
                next_id,
                index,
                backend.as_ref(),
                &mut cache,
                abs_position,
            ) {
                Some(h) => h,
                None => break,
            }
        } else {
            match crate::vindex::predict_kquant_decode_step(
                weights,
                next_id,
                index,
                &mut cache,
                abs_position,
            ) {
                Some((h, step_timings)) => {
                    t_dequant += step_timings.dequant_ms;
                    h
                }
                None => break,
            }
        };
        let hidden_ms = t0.elapsed().as_secs_f64() * 1000.0;
        t_cpu_fwd += hidden_ms;

        let lm_head_start = std::time::Instant::now();
        let result = lm_head_predict(
            weights,
            &h_new,
            tokenizer,
            backend.as_ref(),
            lm_head_q4,
            lm_head_vocab,
        );
        let lm_head_ms = lm_head_start.elapsed().as_secs_f64() * 1000.0;
        t_lm_head += lm_head_ms;
        decode_ms.push(hidden_ms + lm_head_ms);

        let id = match result.token_ids.first() {
            Some(&id) => id,
            None => break,
        };
        let tok = result
            .predictions
            .first()
            .map(|p| p.0.clone())
            .unwrap_or_default();
        let stop = eos.is_eos_with_tokenizer(id, &tok, tokenizer);
        tokens.push((tok, 1.0));
        if stop {
            break;
        }
        next_id = id;
    }

    GenerateResult {
        tokens,
        prefill_ms,
        decode_ms,
        stage_timings: StageTimings {
            embed_ms_total: 0.0,
            gpu_ms_total: 0.0,
            cpu_fwd_ms_total: t_cpu_fwd,
            gate_up_ms_total: 0.0,
            down_ms_total: 0.0,
            norm_ms_total: 0.0,
            lm_head_ms_total: t_lm_head,
            detok_ms_total: 0.0,
            dequant_ms_total: t_dequant,
        },
        error: None,
    }
}

/// Legacy O(N²) loop for architectures the cached path can't handle
/// (hybrid MoE, KV sharing). Re-runs `predict_kquant_hidden` over the
/// growing token sequence at every decode step.
fn generate_via_cpu_q4k_uncached(
    weights: &mut ModelWeights,
    tokenizer: &tokenizers::Tokenizer,
    token_ids: &[u32],
    max_tokens: usize,
    index: &larql_vindex::VectorIndex,
    eos: &EosConfig,
) -> GenerateResult {
    let prefill_start = std::time::Instant::now();
    let (first, _, _) = predict_q4k_timed(weights, tokenizer, token_ids, 5, index);
    let prefill_ms = prefill_start.elapsed().as_secs_f64() * 1000.0;

    let mut tokens: Vec<(String, f64)> = Vec::with_capacity(max_tokens);
    let mut decode_ms = Vec::with_capacity(max_tokens);
    let mut t_cpu_fwd = 0.0f64;
    let mut t_lm_head = 0.0f64;

    let mut ids = token_ids.to_vec();
    if let (Some(&id), Some(first_pred)) = (first.token_ids.first(), first.predictions.first()) {
        tokens.push((first_pred.0.clone(), 1.0));
        let stop = eos.is_eos_with_tokenizer(id, &first_pred.0, tokenizer);
        ids.push(id);
        if stop {
            return GenerateResult {
                tokens,
                prefill_ms,
                decode_ms,
                stage_timings: StageTimings::default(),
                error: None,
            };
        }
    } else {
        return GenerateResult {
            tokens,
            prefill_ms,
            decode_ms,
            stage_timings: StageTimings::default(),
            error: Some(GenerateError::empty_output(
                "CPU Q4K generation produced no first token",
            )),
        };
    }

    for _step in 1..max_tokens {
        let t0 = std::time::Instant::now();
        let (result, hidden_ms, lm_head_ms) = predict_q4k_timed(weights, tokenizer, &ids, 5, index);
        let step_ms = t0.elapsed().as_secs_f64() * 1000.0;
        decode_ms.push(step_ms);
        t_cpu_fwd += hidden_ms;
        t_lm_head += lm_head_ms;

        match result.token_ids.first() {
            Some(&id) => {
                let tok = result
                    .predictions
                    .first()
                    .map(|p| p.0.clone())
                    .unwrap_or_default();
                let stop = eos.is_eos_with_tokenizer(id, &tok, tokenizer);
                tokens.push((tok, 1.0));
                ids.push(id);
                if stop {
                    break;
                }
            }
            None => break,
        }
    }

    GenerateResult {
        tokens,
        prefill_ms,
        decode_ms,
        stage_timings: StageTimings {
            embed_ms_total: 0.0,
            gpu_ms_total: 0.0,
            cpu_fwd_ms_total: t_cpu_fwd,
            gate_up_ms_total: 0.0,
            down_ms_total: 0.0,
            norm_ms_total: 0.0,
            lm_head_ms_total: t_lm_head,
            detok_ms_total: 0.0,
            dequant_ms_total: 0.0,
        },
        error: None,
    }
}

fn last_row_as_2d(h: &ndarray::Array2<f32>) -> ndarray::Array2<f32> {
    let seq_len = h.shape()[0];
    let hidden = h.shape()[1];
    let mut out = ndarray::Array2::<f32>::zeros((1, hidden));
    out.row_mut(0).assign(&h.row(seq_len - 1));
    out
}

/// Decode-loop lm_head wrapper. When the vindex carries a Q4_K view of
/// the LM head (always true for tied-embedding models — Gemma 3/4,
/// Llama with `tie_word_embeddings=true`) the matmul runs through
/// `q4k_matvec` against the synthesised Q4_K bytes, saving ~2 GB of
/// f32 bandwidth per step on Gemma 3 4B. Falls back to the f32
/// row-parallel path when the Q4 view is missing.
fn lm_head_predict(
    weights: &ModelWeights,
    h: &ndarray::Array2<f32>,
    tokenizer: &tokenizers::Tokenizer,
    backend: &dyn larql_compute::ComputeBackend,
    q4_lm_head: Option<&[u8]>,
    vocab: usize,
) -> crate::forward::PredictResult {
    if let Some(q4_bytes) = q4_lm_head {
        if vocab > 0 {
            return crate::forward::predict::logits_to_predictions_q4_lm_head(
                weights, h, q4_bytes, vocab, backend, tokenizer, 5, 1.0,
            );
        }
    }
    crate::forward::predict::logits_to_predictions_pub(weights, h, tokenizer, 5, 1.0)
}

fn predict_q4k_timed(
    weights: &mut ModelWeights,
    tokenizer: &tokenizers::Tokenizer,
    token_ids: &[u32],
    top_k: usize,
    index: &larql_vindex::VectorIndex,
) -> (PredictResult, f64, f64) {
    let hidden_start = std::time::Instant::now();
    let h = crate::vindex::predict_kquant_hidden(weights, token_ids, index, None);
    let hidden_ms = hidden_start.elapsed().as_secs_f64() * 1000.0;

    let lm_head_start = std::time::Instant::now();
    let result =
        crate::forward::predict::logits_to_predictions_pub(weights, &h, tokenizer, top_k, 1.0);
    let lm_head_ms = lm_head_start.elapsed().as_secs_f64() * 1000.0;

    (result, hidden_ms, lm_head_ms)
}

/// Sampling-aware bridge to the CPU Q4_K constrained decoder. Threads
/// the caller's `SamplingConfig` (temperature/top_p/seed/penalties)
/// through to token selection over the masked logits.
#[allow(clippy::too_many_arguments)]
pub(super) fn generate_constrained_via_cpu_q4k_streaming_sampled<M, F>(
    weights: &mut ModelWeights,
    tokenizer: &tokenizers::Tokenizer,
    token_ids: &[u32],
    max_tokens: usize,
    index: &larql_vindex::VectorIndex,
    mask_fn: M,
    on_token: F,
    sampling: super::sampling::SamplingConfig,
    eos: &EosConfig,
) -> GenerateResult
where
    M: FnMut(&[u32], &mut Vec<f32>),
    F: FnMut(u32, &str, f64),
{
    if max_tokens == 0 {
        return GenerateResult::empty_success();
    }

    let prefill_start = std::time::Instant::now();
    let out = crate::vindex::generate_kquant_cpu_constrained_streaming_sampled_with_eos(
        weights, tokenizer, token_ids, max_tokens, index, mask_fn, on_token, sampling, eos,
    );
    let total_ms = prefill_start.elapsed().as_secs_f64() * 1000.0;
    // Heuristic split: attribute the first token to prefill, the rest to
    // decode. Matches the semantics of the GPU path closely enough for
    // bench-report purposes without tracking per-step timing inside the
    // constrained CPU loop.
    let n = out.len();
    let (prefill_ms, decode_ms_each) = if n == 0 {
        (total_ms, 0.0)
    } else {
        let avg = total_ms / n as f64;
        (avg, avg)
    };
    let tokens: Vec<(String, f64)> = out.into_iter().map(|(t, _)| (t, 1.0)).collect();
    let decode_ms = (1..tokens.len()).map(|_| decode_ms_each).collect();
    GenerateResult {
        tokens,
        prefill_ms,
        decode_ms,
        stage_timings: StageTimings::default(),
        error: None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_utils::Q4KTestFixtures;

    /// `generate_via_cpu_q4k` routes through `_cached` when
    /// `supports_cached_decode` holds. On the Q4_K synthetic fixture,
    /// the Gemma 3-style arch satisfies the cached-decode contract, so
    /// the cached path should run and produce a valid result.
    #[test]
    fn generate_via_cpu_q4k_cached_path_produces_tokens() {
        let mut fx = Q4KTestFixtures::build();
        // Build a small prompt directly — `make_test_tokenizer` decodes
        // token N to `[N]` so any vocab IDs work.
        let token_ids = vec![1u32, 2, 3];
        let eos = EosConfig::builtin();

        let result = generate_via_cpu_q4k(
            &mut fx.weights,
            &fx.tokenizer,
            &token_ids,
            4,
            &fx.index,
            &eos,
        );
        assert!(
            result.error.is_none(),
            "cached path must succeed; got error: {:?}",
            result.error
        );
        assert!(
            !result.tokens.is_empty(),
            "expect at least the seed token from the prefill argmax"
        );
        assert!(
            result.prefill_ms > 0.0,
            "prefill should be timed (synthetic, but nonzero)"
        );
    }

    /// max_tokens=0 short-circuits with an empty-success result.
    #[test]
    fn generate_via_cpu_q4k_zero_tokens_returns_empty_success() {
        let mut fx = Q4KTestFixtures::build();
        let token_ids = vec![1u32, 2];
        let eos = EosConfig::builtin();

        let result = generate_via_cpu_q4k(
            &mut fx.weights,
            &fx.tokenizer,
            &token_ids,
            0,
            &fx.index,
            &eos,
        );
        assert!(result.tokens.is_empty());
        assert!(result.error.is_none());
        assert!(result.decode_ms.is_empty());
    }

    /// `lm_head_predict` dispatches to the Q4_K matvec path when the
    /// vindex carries a Q4_K view of the LM head — the synthetic
    /// fixture installs one via `set_lm_head_kquant_mmap`.
    #[test]
    fn lm_head_predict_uses_q4_lm_head_when_available() {
        let fx = Q4KTestFixtures::build();
        // Build a synthetic hidden vector roughly the right scale
        // (post-RMS-norm + final-norm magnitudes ≈ 1).
        let h = ndarray::Array2::from_shape_fn((1, fx.weights.hidden_size), |(_, j)| {
            ((j as f32) * 0.013).sin() * 0.5
        });
        let backend = larql_compute::CpuBackend;
        let q4_lm_head: Option<&[u8]> = fx.index.storage.lm_head_kquant_view().map(|b| b.as_ref());
        assert!(
            q4_lm_head.is_some(),
            "synthetic vindex must carry a Q4_K lm_head view"
        );
        let vocab = fx.weights.vocab_size;
        let result = lm_head_predict(&fx.weights, &h, &fx.tokenizer, &backend, q4_lm_head, vocab);
        assert!(
            !result.token_ids.is_empty(),
            "must return at least one token id"
        );
        for &id in &result.token_ids {
            assert!((id as usize) < vocab);
        }
    }

    /// When the vindex doesn't carry a Q4_K lm_head, `lm_head_predict`
    /// falls back to the f32 sgemv path via
    /// `logits_to_predictions_pub`.
    #[test]
    fn lm_head_predict_falls_back_to_f32_when_q4_unavailable() {
        let fx = Q4KTestFixtures::build();
        let h = ndarray::Array2::from_shape_fn((1, fx.weights.hidden_size), |(_, j)| {
            ((j as f32) * 0.013).sin() * 0.5
        });
        let backend = larql_compute::CpuBackend;
        let vocab = fx.weights.vocab_size;
        // Pass None to force the f32 fallback path.
        let result = lm_head_predict(&fx.weights, &h, &fx.tokenizer, &backend, None, vocab);
        assert!(!result.token_ids.is_empty());
    }

    /// `last_row_as_2d` extracts the final position's hidden vector
    /// as a `[1, hidden]` array.
    #[test]
    fn last_row_as_2d_extracts_final_position() {
        let h = ndarray::Array2::from_shape_fn((3, 4), |(r, c)| (r * 10 + c) as f32);
        let last = last_row_as_2d(&h);
        assert_eq!(last.shape(), &[1, 4]);
        assert_eq!(last[[0, 0]], 20.0); // row 2, col 0 = 2*10 + 0
        assert_eq!(last[[0, 3]], 23.0); // row 2, col 3 = 2*10 + 3
    }
}

#[cfg(test)]
mod more_tests {
    use super::*;
    use crate::test_utils::Q4KTestFixtures;
    use ndarray::Array2;

    /// `predict_q4k_timed` returns the prediction + decomposed timing.
    /// Sanity-check shapes and that both timings come back finite + ≥ 0.
    #[test]
    fn predict_q4k_timed_returns_finite_timings() {
        let mut fx = Q4KTestFixtures::build();
        let (result, hidden_ms, lm_head_ms) =
            predict_q4k_timed(&mut fx.weights, &fx.tokenizer, &[1u32, 2], 3, &fx.index);
        assert!(!result.token_ids.is_empty());
        assert!(hidden_ms.is_finite() && hidden_ms >= 0.0);
        assert!(lm_head_ms.is_finite() && lm_head_ms >= 0.0);
    }

    /// `generate_constrained_via_cpu_q4k_streaming_sampled` short-circuits
    /// when `max_tokens == 0`.
    #[test]
    fn generate_constrained_zero_tokens_returns_empty_success() {
        let mut fx = Q4KTestFixtures::build();
        let token_ids = vec![1u32];
        let eos = EosConfig::builtin();
        let mask = |_ids: &[u32], _logits: &mut Vec<f32>| {};
        let on_tok = |_id: u32, _s: &str, _p: f64| {};
        let sampling = super::super::sampling::SamplingConfig::greedy();
        let result = generate_constrained_via_cpu_q4k_streaming_sampled(
            &mut fx.weights,
            &fx.tokenizer,
            &token_ids,
            0,
            &fx.index,
            mask,
            on_tok,
            sampling,
            &eos,
        );
        assert!(result.tokens.is_empty());
        assert!(result.error.is_none());
    }

    /// `last_row_as_2d` on a single-row hidden returns the same row.
    #[test]
    fn last_row_as_2d_single_row_is_identity_shaped() {
        let h = Array2::from_shape_vec((1, 3), vec![1.0f32, 2.0, 3.0]).unwrap();
        let last = last_row_as_2d(&h);
        assert_eq!(last.shape(), &[1, 3]);
        assert_eq!(last[[0, 0]], 1.0);
        assert_eq!(last[[0, 1]], 2.0);
        assert_eq!(last[[0, 2]], 3.0);
    }

    /// `lm_head_predict` with vocab=0 falls back to the f32 path
    /// (the Q4 branch requires vocab > 0).
    #[test]
    fn lm_head_predict_zero_vocab_falls_back() {
        let fx = Q4KTestFixtures::build();
        let h = Array2::from_shape_fn((1, fx.weights.hidden_size), |(_, j)| {
            ((j as f32) * 0.013).sin() * 0.5
        });
        let backend = larql_compute::CpuBackend;
        let q4_lm_head: Option<&[u8]> = fx.index.storage.lm_head_kquant_view().map(|b| b.as_ref());
        // vocab=0 — Q4 path is gated by `vocab > 0`; falls back to f32.
        let _result = lm_head_predict(&fx.weights, &h, &fx.tokenizer, &backend, q4_lm_head, 0);
        // The f32 fallback may produce predictions or empty depending on
        // tokenizer — what we're verifying is that it doesn't panic.
    }
}

#[cfg(test)]
mod uncached_path_tests {
    use super::*;
    use crate::test_utils::Q4KTestFixtures;

    /// Direct call to the legacy O(N²) `generate_via_cpu_q4k_uncached`.
    /// In production this path only fires for hybrid-MoE or KV-shared
    /// architectures (where the cached path can't run); for testing,
    /// call it directly on the dense synthetic fixture to exercise
    /// the loop without needing a hybrid-MoE arch fixture.
    #[test]
    fn generate_via_cpu_q4k_uncached_produces_tokens() {
        let mut fx = Q4KTestFixtures::build();
        let token_ids = vec![1u32, 2];
        let eos = EosConfig::builtin();
        let result = generate_via_cpu_q4k_uncached(
            &mut fx.weights,
            &fx.tokenizer,
            &token_ids,
            3,
            &fx.index,
            &eos,
        );
        // Either succeeds and produces tokens, or returns a typed
        // error — both are valid (no panic, no NaN propagation).
        if let Some(err) = &result.error {
            panic!("uncached path errored unexpectedly: {err:?}");
        }
        assert!(
            result.prefill_ms > 0.0,
            "prefill should be measured even on the legacy path"
        );
    }

    /// `generate_via_cpu_q4k_uncached` with `max_tokens=1` exercises
    /// the prefill-only-then-return branch (loop body never runs).
    #[test]
    fn generate_via_cpu_q4k_uncached_max_tokens_one_returns_seed_only() {
        let mut fx = Q4KTestFixtures::build();
        let token_ids = vec![1u32, 2];
        let eos = EosConfig::builtin();
        let result = generate_via_cpu_q4k_uncached(
            &mut fx.weights,
            &fx.tokenizer,
            &token_ids,
            1,
            &fx.index,
            &eos,
        );
        assert!(result.error.is_none(), "expected success");
        // With max_tokens=1, we emit the seed and skip the decode loop.
        assert!(result.decode_ms.is_empty());
    }

    /// EOS-on-first-token early-return path of `generate_via_cpu_q4k_cached`
    /// (lines 114-127). Build an EosConfig where every vocab id is EOS so
    /// whatever the prefill argmax returns triggers the stop.
    #[test]
    fn generate_via_cpu_q4k_cached_eos_on_first_token_returns_early() {
        let mut fx = Q4KTestFixtures::build();
        let token_ids = vec![1u32, 2, 3];
        let mut eos = EosConfig::empty();
        for id in 0..fx.weights.vocab_size as u32 {
            eos = eos.with_eos_id(id);
        }
        let result = generate_via_cpu_q4k(
            &mut fx.weights,
            &fx.tokenizer,
            &token_ids,
            10,
            &fx.index,
            &eos,
        );
        assert!(result.error.is_none(), "EOS-stop must be a success");
        // First token is emitted before the EOS check; no decode steps run.
        assert_eq!(
            result.tokens.len(),
            1,
            "exactly one token (the seed) is emitted before EOS stops decode"
        );
        assert!(result.decode_ms.is_empty());
    }

    /// EOS-on-first-token early-return path of
    /// `generate_via_cpu_q4k_uncached` (lines 252-260). Same shape as
    /// the cached test but routes through the uncached driver by
    /// disabling cached-decode support (hybrid MoE / cross-layer KV).
    /// We can't toggle the architecture here, so we call the uncached
    /// path directly via its private name.
    #[test]
    fn generate_via_cpu_q4k_uncached_eos_on_first_token_returns_early() {
        let mut fx = Q4KTestFixtures::build();
        let token_ids = vec![1u32, 2];
        let mut eos = EosConfig::empty();
        for id in 0..fx.weights.vocab_size as u32 {
            eos = eos.with_eos_id(id);
        }
        let result = generate_via_cpu_q4k_uncached(
            &mut fx.weights,
            &fx.tokenizer,
            &token_ids,
            10,
            &fx.index,
            &eos,
        );
        assert!(result.error.is_none());
        assert_eq!(result.tokens.len(), 1);
    }
}

#[cfg(test)]
mod constrained_streaming_tests {
    use super::*;
    use crate::test_utils::Q4KTestFixtures;

    /// Cover the body of `generate_constrained_via_cpu_q4k_streaming_sampled`
    /// (only the `max_tokens=0` short-circuit was previously hit).
    /// The constrained loop calls into `vindex::generate_kquant_cpu_constrained_streaming_sampled_with_eos`
    /// which produces token strings — we exercise the wrapper's
    /// prefill/decode-ms accounting and the empty-output handling.
    #[test]
    fn generate_constrained_via_cpu_q4k_streaming_sampled_nonzero_runs() {
        let mut fx = Q4KTestFixtures::build();
        let token_ids = vec![1u32, 2];
        let eos = EosConfig::builtin();
        // Identity mask — pass logits through unchanged.
        let mask = |_ids: &[u32], _logits: &mut Vec<f32>| {};
        let mut emitted: Vec<u32> = Vec::new();
        let on_tok = |id: u32, _s: &str, _p: f64| emitted.push(id);
        let sampling = super::super::sampling::SamplingConfig::greedy();
        let result = generate_constrained_via_cpu_q4k_streaming_sampled(
            &mut fx.weights,
            &fx.tokenizer,
            &token_ids,
            2,
            &fx.index,
            mask,
            on_tok,
            sampling,
            &eos,
        );
        // Constrained streaming returns no typed error and either
        // emits tokens or stops early; we just confirm no panic and
        // sane timing fields. The Q4K constrained decoder may produce
        // zero tokens against the synthetic vindex if the masked
        // logits don't admit any candidate, which is fine — the
        // wrapper still computes prefill_ms.
        assert!(result.error.is_none());
        assert!(result.prefill_ms >= 0.0);
    }
}
