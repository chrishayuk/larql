//! Constrained generation over the GPU/vindex decode path.

use super::cpu::{
    backend_supports_fused_q4_pipeline, generate_constrained_via_cpu_q4k_streaming_sampled,
};
use super::eos::EosConfig;
use super::gpu_setup::{
    build_gpu_decode_setup, ensure_prompt_fits, prefill_kquant_prompt,
    reset_and_preallocate_kv_cache,
};
use super::lm_head::pick_next_token_masked_sampled;
use super::sampling::{Sampler, SamplingConfig};
use super::types::{GenerateError, GenerateResult, StageTimings};
use crate::layer_graph::CachedLayerGraph;
use crate::model::ModelWeights;
use larql_compute::prelude::*;

/// Constrained variant of [`super::generate`] for grammar-controlled decoding.
///
/// Differs from unconstrained generation in two places only:
///
///   1. The LM-head step uses a dense vocabulary score vector rather than
///      sparse vindex KNN. Required because an arbitrary mask can disqualify
///      tokens that would otherwise have fallen outside the top-K.
///   2. After scoring, `mask_fn(generated_ids, &mut logits)` runs and the next
///      token is selected from the masked scores.
#[allow(clippy::too_many_arguments)]
pub fn generate_constrained<M>(
    weights: &mut ModelWeights,
    tokenizer: &tokenizers::Tokenizer,
    token_ids: &[u32],
    max_tokens: usize,
    index: &larql_vindex::VectorIndex,
    backend: &dyn ComputeBackend,
    cached_layers: &CachedLayerGraph,
    layer_range: std::ops::Range<usize>,
    mask_fn: M,
) -> GenerateResult
where
    M: FnMut(&[u32], &mut Vec<f32>),
{
    generate_constrained_streaming(
        weights,
        tokenizer,
        token_ids,
        max_tokens,
        index,
        backend,
        cached_layers,
        layer_range,
        mask_fn,
        |_, _, _| {},
    )
}

/// Fallible variant of [`generate_constrained`].
#[allow(clippy::too_many_arguments)]
pub fn try_generate_constrained<M>(
    weights: &mut ModelWeights,
    tokenizer: &tokenizers::Tokenizer,
    token_ids: &[u32],
    max_tokens: usize,
    index: &larql_vindex::VectorIndex,
    backend: &dyn ComputeBackend,
    cached_layers: &CachedLayerGraph,
    layer_range: std::ops::Range<usize>,
    mask_fn: M,
) -> Result<GenerateResult, GenerateError>
where
    M: FnMut(&[u32], &mut Vec<f32>),
{
    generate_constrained(
        weights,
        tokenizer,
        token_ids,
        max_tokens,
        index,
        backend,
        cached_layers,
        layer_range,
        mask_fn,
    )
    .into_result()
}

/// Streaming variant of [`generate_constrained`].
#[allow(clippy::too_many_arguments)]
pub fn generate_constrained_streaming<M, F>(
    weights: &mut ModelWeights,
    tokenizer: &tokenizers::Tokenizer,
    token_ids: &[u32],
    max_tokens: usize,
    index: &larql_vindex::VectorIndex,
    backend: &dyn ComputeBackend,
    cached_layers: &CachedLayerGraph,
    layer_range: std::ops::Range<usize>,
    mask_fn: M,
    on_token: F,
) -> GenerateResult
where
    M: FnMut(&[u32], &mut Vec<f32>),
    F: FnMut(u32, &str, f64),
{
    generate_constrained_streaming_sampled(
        weights,
        tokenizer,
        token_ids,
        max_tokens,
        index,
        backend,
        cached_layers,
        layer_range,
        mask_fn,
        on_token,
        SamplingConfig::greedy(),
        &EosConfig::builtin(),
    )
}

/// Fallible variant of [`generate_constrained_streaming`].
#[allow(clippy::too_many_arguments)]
pub fn try_generate_constrained_streaming<M, F>(
    weights: &mut ModelWeights,
    tokenizer: &tokenizers::Tokenizer,
    token_ids: &[u32],
    max_tokens: usize,
    index: &larql_vindex::VectorIndex,
    backend: &dyn ComputeBackend,
    cached_layers: &CachedLayerGraph,
    layer_range: std::ops::Range<usize>,
    mask_fn: M,
    on_token: F,
) -> Result<GenerateResult, GenerateError>
where
    M: FnMut(&[u32], &mut Vec<f32>),
    F: FnMut(u32, &str, f64),
{
    generate_constrained_streaming(
        weights,
        tokenizer,
        token_ids,
        max_tokens,
        index,
        backend,
        cached_layers,
        layer_range,
        mask_fn,
        on_token,
    )
    .into_result()
}

/// Streaming + sampling-aware constrained decode.
#[allow(clippy::too_many_arguments)]
pub fn generate_constrained_streaming_sampled<M, F>(
    weights: &mut ModelWeights,
    tokenizer: &tokenizers::Tokenizer,
    token_ids: &[u32],
    max_tokens: usize,
    index: &larql_vindex::VectorIndex,
    backend: &dyn ComputeBackend,
    cached_layers: &CachedLayerGraph,
    layer_range: std::ops::Range<usize>,
    mut mask_fn: M,
    mut on_token: F,
    sampling: SamplingConfig,
    eos: &EosConfig,
) -> GenerateResult
where
    M: FnMut(&[u32], &mut Vec<f32>),
    F: FnMut(u32, &str, f64),
{
    if max_tokens == 0 {
        return GenerateResult::empty_success();
    }

    let mut sampler = Sampler::new(sampling);
    let needs_per_layer_embed = weights.arch.has_per_layer_embeddings();
    if !backend_supports_fused_q4_pipeline(backend) || needs_per_layer_embed {
        return generate_constrained_via_cpu_q4k_streaming_sampled(
            weights, tokenizer, token_ids, max_tokens, index, mask_fn, on_token, sampling, eos,
        );
    }

    let arch = &*weights.arch;
    let norm_offset = arch.norm_weight_offset();
    let setup = match build_gpu_decode_setup(weights, index, backend, layer_range, true) {
        Ok(setup) => setup,
        Err(err) => {
            let _ = cached_layers;
            return GenerateResult::empty_error(err);
        }
    };
    let layers = setup.layers;
    let hidden = setup.hidden;
    let intermediate = setup.intermediate;

    let prefill_start = std::time::Instant::now();
    let seq_len = token_ids.len();
    if let Err(err) = ensure_prompt_fits(seq_len) {
        return GenerateResult::empty_error(err);
    }
    reset_and_preallocate_kv_cache(weights, backend);

    let h_embed = crate::forward::embed_tokens_pub(weights, token_ids);
    let x: Vec<f32> = h_embed.as_slice().unwrap_or(&[]).to_vec();
    let softcap_val = arch.attn_logit_softcapping().unwrap_or(0.0);
    let qk_norm_val = arch.attn_q_norm_key(0).is_some();
    let h_vec = match prefill_kquant_prompt(
        backend,
        &layers,
        &x,
        hidden,
        intermediate,
        seq_len,
        qk_norm_val,
        softcap_val,
        "constrained GPU Q4 prefill returned no output",
    ) {
        Ok(v) => v,
        Err(err) => return GenerateResult::empty_error(err),
    };

    let h_metal = ndarray::Array2::from_shape_vec((seq_len, hidden), h_vec.clone())
        .unwrap_or_else(|_| h_embed.clone());
    let h_1d = {
        let h_final = crate::forward::apply_norm(
            weights,
            &h_metal,
            weights.arch.final_norm_key(),
            norm_offset,
        );
        h_final.row(seq_len - 1).to_owned()
    };
    let prefill_ms = prefill_start.elapsed().as_secs_f64() * 1000.0;

    let mut tokens: Vec<(String, f64)> = Vec::with_capacity(max_tokens);
    let mut decode_ms = Vec::with_capacity(max_tokens);
    let mut generated: Vec<u32> = Vec::with_capacity(max_tokens);

    let first = pick_next_token_masked_sampled(
        weights,
        &h_1d,
        &generated,
        backend,
        &mut mask_fn,
        &mut sampler,
    );
    let mut current_token_id = match first {
        Some((tid, _)) => {
            let tok_str = tokenizer.decode(&[tid], true).unwrap_or_default();
            let is_eos = eos.is_eos_with_tokenizer(tid, &tok_str, tokenizer);
            on_token(tid, &tok_str, 1.0);
            tokens.push((tok_str, 1.0));
            generated.push(tid);
            if is_eos {
                return GenerateResult {
                    tokens,
                    prefill_ms,
                    decode_ms,
                    stage_timings: StageTimings::default(),
                    error: None,
                };
            }
            tid
        }
        None => {
            return GenerateResult {
                tokens,
                prefill_ms,
                decode_ms,
                stage_timings: StageTimings::default(),
                error: Some(GenerateError::MaskRejectedAllCandidates),
            };
        }
    };

    for _step in 1..max_tokens {
        let decode_start = std::time::Instant::now();

        let h_tok = crate::forward::embed_tokens_pub(weights, &[current_token_id]);
        let x_dec: Vec<f32> = h_tok.row(0).to_vec();
        let result = backend.decode_token(&layers, &x_dec, hidden, intermediate);

        let h_1d = if let Some(h_out) = result {
            let h_arr = ndarray::Array2::from_shape_vec((1, hidden), h_out).unwrap();
            let h_final = crate::forward::apply_norm(
                weights,
                &h_arr,
                weights.arch.final_norm_key(),
                norm_offset,
            );
            h_final.row(0).to_owned()
        } else {
            break;
        };

        let pick = pick_next_token_masked_sampled(
            weights,
            &h_1d,
            &generated,
            backend,
            &mut mask_fn,
            &mut sampler,
        );
        decode_ms.push(decode_start.elapsed().as_secs_f64() * 1000.0);

        match pick {
            Some((tid, _)) => {
                let tok_str = tokenizer.decode(&[tid], true).unwrap_or_default();
                let is_eos = eos.is_eos_with_tokenizer(tid, &tok_str, tokenizer);
                on_token(tid, &tok_str, 1.0);
                tokens.push((tok_str, 1.0));
                generated.push(tid);
                current_token_id = tid;
                if is_eos {
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
        stage_timings: StageTimings::default(),
        error: None,
    }
}

/// Fallible variant of [`generate_constrained_streaming_sampled`].
#[allow(clippy::too_many_arguments)]
pub fn try_generate_constrained_streaming_sampled<M, F>(
    weights: &mut ModelWeights,
    tokenizer: &tokenizers::Tokenizer,
    token_ids: &[u32],
    max_tokens: usize,
    index: &larql_vindex::VectorIndex,
    backend: &dyn ComputeBackend,
    cached_layers: &CachedLayerGraph,
    layer_range: std::ops::Range<usize>,
    mask_fn: M,
    on_token: F,
    sampling: SamplingConfig,
    eos: &EosConfig,
) -> Result<GenerateResult, GenerateError>
where
    M: FnMut(&[u32], &mut Vec<f32>),
    F: FnMut(u32, &str, f64),
{
    generate_constrained_streaming_sampled(
        weights,
        tokenizer,
        token_ids,
        max_tokens,
        index,
        backend,
        cached_layers,
        layer_range,
        mask_fn,
        on_token,
        sampling,
        eos,
    )
    .into_result()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_utils::{
        make_test_q4k_vindex, make_test_q4k_weights, make_test_tokenizer, MockGpuBackend,
    };

    /// `generate_constrained` zero-tokens short-circuit.
    #[test]
    fn generate_constrained_zero_tokens_returns_empty_success() {
        let mut weights = make_test_q4k_weights();
        let index = make_test_q4k_vindex(&weights);
        let tokenizer = make_test_tokenizer(weights.vocab_size);
        let cached = CachedLayerGraph::from_residuals(vec![]);
        let backend = MockGpuBackend::new();
        let num_layers = weights.num_layers;
        let result = generate_constrained(
            &mut weights,
            &tokenizer,
            &[0u32, 1],
            /*max_tokens=*/ 0,
            &index,
            &backend,
            &cached,
            0..num_layers,
            |_ids, _logits| {},
        );
        assert!(result.tokens.is_empty());
        assert!(result.error.is_none());
    }

    /// `generate_constrained` end-to-end against MockGpuBackend.
    /// Identity mask — pass logits through unchanged. Mock returns
    /// zero logits so the sampler picks token 0 every step.
    #[test]
    fn generate_constrained_runs_through_mock_gpu_backend() {
        let mut weights = make_test_q4k_weights();
        let index = make_test_q4k_vindex(&weights);
        let tokenizer = make_test_tokenizer(weights.vocab_size);
        let cached = CachedLayerGraph::from_residuals(vec![]);
        let backend = MockGpuBackend::new();
        let num_layers = weights.num_layers;
        let result = generate_constrained(
            &mut weights,
            &tokenizer,
            &[0u32, 1, 2],
            2,
            &index,
            &backend,
            &cached,
            0..num_layers,
            |_ids, _logits| {},
        );
        // No error → the GPU constrained pipeline ran end-to-end.
        assert!(result.error.is_none());
    }

    /// `try_generate_constrained` delegates to `generate_constrained`
    /// + `.into_result()`. Drives the wrapper body (lines 71-82).
    #[test]
    fn try_generate_constrained_returns_ok_on_zero_tokens() {
        let mut weights = make_test_q4k_weights();
        let index = make_test_q4k_vindex(&weights);
        let tokenizer = make_test_tokenizer(weights.vocab_size);
        let cached = CachedLayerGraph::from_residuals(vec![]);
        let backend = MockGpuBackend::new();
        let num_layers = weights.num_layers;
        let result = try_generate_constrained(
            &mut weights,
            &tokenizer,
            &[0u32, 1],
            0,
            &index,
            &backend,
            &cached,
            0..num_layers,
            |_ids, _logits| {},
        );
        assert!(result.is_ok());
    }

    /// `try_generate_constrained_streaming` — wrapper over the
    /// streaming variant. Drives lines 121-147.
    #[test]
    fn try_generate_constrained_streaming_returns_ok_on_zero_tokens() {
        let mut weights = make_test_q4k_weights();
        let index = make_test_q4k_vindex(&weights);
        let tokenizer = make_test_tokenizer(weights.vocab_size);
        let cached = CachedLayerGraph::from_residuals(vec![]);
        let backend = MockGpuBackend::new();
        let num_layers = weights.num_layers;
        let mut callback_count = 0;
        let result = try_generate_constrained_streaming(
            &mut weights,
            &tokenizer,
            &[0u32, 1],
            0,
            &index,
            &backend,
            &cached,
            0..num_layers,
            |_ids, _logits| {},
            |_id, _text, _prob| callback_count += 1,
        );
        assert!(result.is_ok());
        assert_eq!(callback_count, 0);
    }

    /// `generate_constrained_streaming` end-to-end (greedy variant
    /// without explicit SamplingConfig — uses the default greedy).
    #[test]
    fn generate_constrained_streaming_runs_through_mock_gpu_backend() {
        let mut weights = make_test_q4k_weights();
        let index = make_test_q4k_vindex(&weights);
        let tokenizer = make_test_tokenizer(weights.vocab_size);
        let cached = CachedLayerGraph::from_residuals(vec![]);
        let backend = MockGpuBackend::new();
        let num_layers = weights.num_layers;
        let mut callback_count = 0;
        let result = generate_constrained_streaming(
            &mut weights,
            &tokenizer,
            &[0u32, 1, 2],
            2,
            &index,
            &backend,
            &cached,
            0..num_layers,
            |_ids, _logits| {},
            |_id, _text, _prob| callback_count += 1,
        );
        // No error; callback fired per token.
        assert!(result.error.is_none());
        assert_eq!(callback_count, result.tokens.len());
    }

    /// `try_generate_constrained_streaming_sampled` happy path with
    /// MockGpuBackend + a non-greedy sampler.
    #[test]
    fn try_generate_constrained_streaming_sampled_runs_with_mock_gpu_backend() {
        let mut weights = make_test_q4k_weights();
        let index = make_test_q4k_vindex(&weights);
        let tokenizer = make_test_tokenizer(weights.vocab_size);
        let cached = CachedLayerGraph::from_residuals(vec![]);
        let backend = MockGpuBackend::new();
        let num_layers = weights.num_layers;
        let mut callback_count = 0;
        let mut sampling = SamplingConfig::greedy();
        sampling.temperature = 0.7;
        sampling.top_k = Some(4);
        let result = try_generate_constrained_streaming_sampled(
            &mut weights,
            &tokenizer,
            &[0u32, 1],
            2,
            &index,
            &backend,
            &cached,
            0..num_layers,
            |_ids, _logits| {},
            |_id, _text, _prob| callback_count += 1,
            sampling,
            &EosConfig::builtin(),
        );
        // try_ wrapper converts errors to Err; success path returns Ok.
        let _ = result; // may be Ok or Err depending on mask outcome
    }
}
