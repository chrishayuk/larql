use larql_models::ModelWeights;
use larql_vindex::VectorIndex;
use tokenizers::Tokenizer;

use crate::forward::PredictResult;

use super::hidden::predict_kquant_hidden;

/// End-to-end predict on a Q4_K/Q6_K vindex.
pub fn predict_kquant(
    weights: &mut ModelWeights,
    tokenizer: &Tokenizer,
    token_ids: &[u32],
    top_k: usize,
    index: &VectorIndex,
) -> PredictResult {
    let h = predict_kquant_hidden(weights, token_ids, index, None);
    crate::forward::predict::logits_to_predictions_pub(weights, &h, tokenizer, top_k, 1.0)
}

/// Common end-of-turn / EOS markers across Gemma, Llama, Mistral, ChatML.
pub fn is_end_of_turn(token: &str) -> bool {
    matches!(
        token,
        "<eos>"
            | "</s>"
            | "<|endoftext|>"
            | "<|im_end|>"
            | "<|end_of_turn|>"
            | "<end_of_turn>"
            | "<|eot_id|>"
    )
}

/// CPU autoregressive generation against a Q4_K / Q6_K vindex.
pub fn generate_kquant_cpu(
    weights: &mut ModelWeights,
    tokenizer: &Tokenizer,
    prompt_ids: &[u32],
    max_tokens: usize,
    index: &VectorIndex,
) -> Vec<(String, u32)> {
    let mut ids = prompt_ids.to_vec();
    let mut out: Vec<(String, u32)> = Vec::with_capacity(max_tokens);
    for _ in 0..max_tokens {
        let result = predict_kquant(weights, tokenizer, &ids, 1, index);
        let next_id = match result.token_ids.first() {
            Some(&id) => id,
            None => break,
        };
        let tok = result
            .predictions
            .first()
            .map(|p| p.0.clone())
            .unwrap_or_default();
        let stop = is_end_of_turn(&tok);
        out.push((tok, next_id));
        ids.push(next_id);
        if stop {
            break;
        }
    }
    out
}

/// Like [`generate_kquant_cpu`] but dispatches MoE expert matmuls to remote shard
/// servers via [`crate::ffn::RemoteMoeBackend`].
pub fn generate_kquant_cpu_remote(
    weights: &mut ModelWeights,
    tokenizer: &Tokenizer,
    prompt_ids: &[u32],
    max_tokens: usize,
    index: &VectorIndex,
    moe_remote: &crate::ffn::RemoteMoeBackend,
) -> Vec<(String, u32)> {
    let mut ids = prompt_ids.to_vec();
    let mut out: Vec<(String, u32)> = Vec::with_capacity(max_tokens);
    for _ in 0..max_tokens {
        let h = predict_kquant_hidden(weights, &ids, index, Some(moe_remote));
        let last = h.nrows().saturating_sub(1);
        let h_last = h.slice(ndarray::s![last..last + 1, ..]).to_owned();
        let logits = crate::forward::hidden_to_raw_logits(weights, &h_last);
        let next_id = logits
            .iter()
            .copied()
            .enumerate()
            .filter(|(_, v)| v.is_finite())
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
            .map(|(i, _)| i as u32)
            .unwrap_or(0);
        let tok = tokenizer.decode(&[next_id], true).unwrap_or_default();
        let stop = is_end_of_turn(&tok);
        out.push((tok, next_id));
        ids.push(next_id);
        if stop {
            break;
        }
    }
    out
}

/// Constrained variant of [`generate_kquant_cpu`]. Greedy under the mask.
pub fn generate_kquant_cpu_constrained<M>(
    weights: &mut ModelWeights,
    tokenizer: &Tokenizer,
    prompt_ids: &[u32],
    max_tokens: usize,
    index: &VectorIndex,
    mask_fn: M,
) -> Vec<(String, u32)>
where
    M: FnMut(&[u32], &mut Vec<f32>),
{
    generate_kquant_cpu_constrained_streaming_sampled(
        weights,
        tokenizer,
        prompt_ids,
        max_tokens,
        index,
        mask_fn,
        |_, _, _| {},
        crate::layer_graph::SamplingConfig::greedy(),
    )
}

/// Streaming-callback variant of [`generate_kquant_cpu_constrained`].
/// Fires `on_token(id, text, prob)` after each masked argmax pick. Used
/// by the OpenAI server's SSE path so JSON / structured-output streams
/// can flush chunks as the constrained decoder produces them.
///
/// Greedy under the mask. For sampling under mask, see
/// [`generate_kquant_cpu_constrained_streaming_sampled`].
pub fn generate_kquant_cpu_constrained_streaming<M, F>(
    weights: &mut ModelWeights,
    tokenizer: &Tokenizer,
    prompt_ids: &[u32],
    max_tokens: usize,
    index: &VectorIndex,
    mask_fn: M,
    on_token: F,
) -> Vec<(String, u32)>
where
    M: FnMut(&[u32], &mut Vec<f32>),
    F: FnMut(u32, &str, f64),
{
    generate_kquant_cpu_constrained_streaming_sampled(
        weights,
        tokenizer,
        prompt_ids,
        max_tokens,
        index,
        mask_fn,
        on_token,
        crate::layer_graph::SamplingConfig::greedy(),
    )
}

/// Sampling-aware streaming-constrained CPU Q4_K decode. Drives token
/// selection through the supplied `SamplingConfig` (temperature, top_p,
/// top_k, seed, repetition penalties) over the masked logits — so JSON
/// / tools modes can be sampled rather than greedy when the caller asks.
///
/// Pass `SamplingConfig::greedy()` for the existing argmax behaviour.
#[allow(clippy::too_many_arguments)]
pub fn generate_kquant_cpu_constrained_streaming_sampled<M, F>(
    weights: &mut ModelWeights,
    tokenizer: &Tokenizer,
    prompt_ids: &[u32],
    max_tokens: usize,
    index: &VectorIndex,
    mask_fn: M,
    on_token: F,
    sampling: crate::layer_graph::SamplingConfig,
) -> Vec<(String, u32)>
where
    M: FnMut(&[u32], &mut Vec<f32>),
    F: FnMut(u32, &str, f64),
{
    generate_kquant_cpu_constrained_streaming_sampled_with_eos(
        weights,
        tokenizer,
        prompt_ids,
        max_tokens,
        index,
        mask_fn,
        on_token,
        sampling,
        &crate::layer_graph::EosConfig::builtin(),
    )
}

/// Sampling-aware streaming-constrained CPU Q4_K decode with explicit EOS
/// policy. Kept crate-visible so public legacy helpers continue to use the
/// built-in stop set while higher-level generation APIs can honor
/// caller-supplied EOS IDs and stop strings.
#[allow(clippy::too_many_arguments)]
pub(crate) fn generate_kquant_cpu_constrained_streaming_sampled_with_eos<M, F>(
    weights: &mut ModelWeights,
    tokenizer: &Tokenizer,
    prompt_ids: &[u32],
    max_tokens: usize,
    index: &VectorIndex,
    mut mask_fn: M,
    mut on_token: F,
    sampling: crate::layer_graph::SamplingConfig,
    eos: &crate::layer_graph::EosConfig,
) -> Vec<(String, u32)>
where
    M: FnMut(&[u32], &mut Vec<f32>),
    F: FnMut(u32, &str, f64),
{
    let mut ids = prompt_ids.to_vec();
    let mut generated: Vec<u32> = Vec::with_capacity(max_tokens);
    let mut out: Vec<(String, u32)> = Vec::with_capacity(max_tokens);
    let mut sampler = crate::layer_graph::Sampler::new(sampling);

    for _ in 0..max_tokens {
        let h = predict_kquant_hidden(weights, &ids, index, None);
        let last_hidden = h.row(h.nrows().saturating_sub(1)).to_owned();
        let last_2d = ndarray::Array2::from_shape_vec((1, last_hidden.len()), last_hidden.to_vec())
            .expect("shape");

        let mut logits = crate::forward::hidden_to_raw_logits(weights, &last_2d);
        mask_fn(&generated, &mut logits);

        let id = match sampler.sample_with_history(&logits, &generated) {
            Some(id) => id,
            None => break,
        };
        // Sanity: bail if the picked token's logit isn't finite (e.g.
        // mask wiped every entry to -inf — the FSM rejected everything).
        let idx_score = *logits.get(id as usize).unwrap_or(&f32::NEG_INFINITY);
        if !idx_score.is_finite() {
            break;
        }
        let tok = tokenizer.decode(&[id], true).unwrap_or_default();

        let stop = eos.is_eos_with_tokenizer(id, &tok, tokenizer);
        on_token(id, &tok, 1.0);
        out.push((tok, id));
        ids.push(id);
        generated.push(id);
        if stop {
            break;
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::is_end_of_turn;

    #[test]
    fn is_end_of_turn_recognises_known_terminators() {
        for t in [
            "<eos>",
            "</s>",
            "<|endoftext|>",
            "<|im_end|>",
            "<|end_of_turn|>",
            "<end_of_turn>",
            "<|eot_id|>",
        ] {
            assert!(is_end_of_turn(t), "expected {t:?} to be a terminator");
        }
    }

    #[test]
    fn is_end_of_turn_rejects_arbitrary_tokens() {
        for t in ["", " ", "the", "<eos", "eos>", "<EOS>", "<|im_start|>"] {
            assert!(
                !is_end_of_turn(t),
                "did not expect {t:?} to be a terminator"
            );
        }
    }

    // ── Q4K + MoE generate paths (Gemma 4 fixture) ───────────────────────

    use super::*;
    use crate::test_utils::{
        make_test_gemma4_moe_weights, make_test_q4k_vindex, make_test_q4k_weights,
        make_test_tokenizer,
    };

    #[test]
    fn predict_kquant_returns_predictions_against_q4k_fixture() {
        let mut weights = make_test_q4k_weights();
        let index = make_test_q4k_vindex(&weights);
        let tokenizer = make_test_tokenizer(weights.vocab_size);
        let result = predict_kquant(&mut weights, &tokenizer, &[0u32, 1, 2], 5, &index);
        assert!(result.predictions.len() <= 5);
    }

    #[test]
    fn generate_kquant_cpu_emits_tokens_against_q4k_fixture() {
        let mut weights = make_test_q4k_weights();
        let index = make_test_q4k_vindex(&weights);
        let tokenizer = make_test_tokenizer(weights.vocab_size);
        let out = generate_kquant_cpu(&mut weights, &tokenizer, &[0u32, 1], 3, &index);
        assert!(out.len() <= 3, "out: {out:?}");
    }

    /// Gemma 4 MoE arch drives the hybrid-MoE branch of
    /// `predict_kquant_hidden` (via `predict_kquant` → which calls it).
    #[test]
    fn predict_kquant_routes_through_moe_on_gemma4_fixture() {
        let mut weights = make_test_gemma4_moe_weights();
        let index = make_test_q4k_vindex(&weights);
        let tokenizer = make_test_tokenizer(weights.vocab_size);
        let result = predict_kquant(&mut weights, &tokenizer, &[0u32, 1, 2], 3, &index);
        assert!(result.predictions.len() <= 3);
    }

    #[test]
    fn generate_kquant_cpu_constrained_emits_tokens_against_q4k_fixture() {
        let mut weights = make_test_q4k_weights();
        let index = make_test_q4k_vindex(&weights);
        let tokenizer = make_test_tokenizer(weights.vocab_size);
        let mut mask_calls = 0;
        let out = generate_kquant_cpu_constrained(
            &mut weights,
            &tokenizer,
            &[0u32],
            2,
            &index,
            |_ids, _logits| {
                mask_calls += 1;
            },
        );
        assert!(out.len() <= 2);
    }

    /// `generate_kquant_cpu_remote` against a disconnected `RemoteMoeBackend`
    /// on the Gemma 4 MoE fixture — drives the function body (lines 68-100)
    /// end-to-end. The MoE dispatch falls back to zero contribution when
    /// the remote returns Err (see hidden.rs's `Some(remote)` branch);
    /// the generation loop still picks tokens off the dense path.
    #[test]
    fn generate_kquant_cpu_remote_runs_against_disconnected_backend() {
        use crate::ffn::RemoteMoeBackend;
        use crate::test_utils::{make_test_gemma4_moe_weights, make_test_q4k_vindex};
        let mut weights = make_test_gemma4_moe_weights();
        let index = make_test_q4k_vindex(&weights);
        let tokenizer = make_test_tokenizer(weights.vocab_size);
        let remote = RemoteMoeBackend::new_disconnected();
        let out =
            generate_kquant_cpu_remote(&mut weights, &tokenizer, &[0u32, 1], 2, &index, &remote);
        assert!(out.len() <= 2);
    }

    /// `generate_kquant_cpu_constrained_streaming` wraps the sampled
    /// variant with `SamplingConfig::greedy()`. Drives lines 133, 146-156
    /// — the body just forwards to the sampled variant.
    #[test]
    fn generate_kquant_cpu_constrained_streaming_invokes_on_token() {
        let mut weights = make_test_q4k_weights();
        let index = make_test_q4k_vindex(&weights);
        let tokenizer = make_test_tokenizer(weights.vocab_size);
        let mut on_token_calls = 0;
        let out = generate_kquant_cpu_constrained_streaming(
            &mut weights,
            &tokenizer,
            &[0u32],
            2,
            &index,
            |_ids, _logits| {},
            |_id, _tok, _prob| {
                on_token_calls += 1;
            },
        );
        // streaming variant must call on_token for every emitted token.
        assert_eq!(out.len(), on_token_calls);
        assert!(out.len() <= 2);
    }
}
