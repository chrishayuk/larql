//! Autoregressive generation with a CPU [`KvCache`].
//!
//! Two-phase decoder:
//!
//! 1. **Prefill.** Run a full forward pass over the prompt via
//!    `predict_with_ffn` (which already handles all Gemma 3 / Gemma 4
//!    specifics — QK norm, V norm, cross-layer KV sharing, PLE, layer
//!    scalar). During the pass, capture post-RoPE K and post-V-norm V
//!    per layer into a [`KvCache`].
//! 2. **Decode.** For each new token: embed it as a single row, run
//!    the decode-step attention (Q of new token attends against
//!    cached K/V + the new token's own K/V), FFN, next layer. At end
//!    of layer stack, logits → argmax → next token. Streams tokens
//!    to a caller-supplied callback.
//!
//! This is **not** a full re-implementation of the prefill path — the
//! prefill reuses `predict_with_ffn` verbatim. Only the decode step
//! has new code, gated to single-token inputs where per-step cost is
//! O(cached_len) instead of O(cached_len²).
//!
//! Works with any [`FfnBackend`] — local `WalkFfn`, `RemoteWalkBackend`
//! (FFN over HTTP), etc.
//!
//! Lifted from `larql-inference::forward::kv_generate` in 2026-05-16.
//! These loops drive every engine's `prefill` / `decode_step` impl via
//! [`generate_with_engine`]; [`generate_cached_backend`] is retained as
//! the parity oracle for the unification migration (see
//! `larql-inference/docs/specs/kv-engine-unification.md` §8.7).

use larql_inference::attention::{
    run_attention_block_decode_step_backend, run_attention_with_kv_backend,
};
use larql_inference::ffn::FfnBackend;
use larql_inference::forward::hooks::{LayerHook, NoopHook};
use larql_inference::forward::{
    embed_tokens_pub, hidden_to_raw_logits, logits_to_predictions_pub, run_ffn,
};
use larql_inference::ModelWeights;
use ndarray::Array2;

use crate::cache::KvCache;

/// Stream autoregressive generation with a KV cache.
///
/// `on_token` receives `(token_id, decoded_string)` for each generated
/// token as it arrives (including the first, which comes out of the
/// prefill step).
///
/// Returns the concatenated generated IDs. Stops on EOS or when
/// `max_new_tokens` have been produced.
pub fn generate_cached<F>(
    weights: &ModelWeights,
    tokenizer: &larql_inference::tokenizers::Tokenizer,
    ffn: &dyn FfnBackend,
    prompt_ids: &[u32],
    max_new_tokens: usize,
    mut on_token: F,
) -> Vec<u32>
where
    F: FnMut(u32, &str),
{
    generate_cached_bounded(
        weights,
        tokenizer,
        ffn,
        prompt_ids,
        max_new_tokens,
        None,
        None,
        &mut on_token,
    )
}

/// Variant of [`generate_cached`] that runs Q/K/V/O projections on a
/// GPU `ComputeBackend` when provided. GQA softmax stays on CPU.
#[allow(clippy::too_many_arguments)]
pub fn generate_cached_backend<F>(
    weights: &ModelWeights,
    tokenizer: &larql_inference::tokenizers::Tokenizer,
    ffn: &dyn FfnBackend,
    prompt_ids: &[u32],
    max_new_tokens: usize,
    backend: Option<&dyn larql_compute::ComputeBackend>,
    window: Option<usize>,
    mut on_token: F,
) -> Vec<u32>
where
    F: FnMut(u32, &str),
{
    generate_cached_bounded(
        weights,
        tokenizer,
        ffn,
        prompt_ids,
        max_new_tokens,
        window,
        backend,
        &mut on_token,
    )
}

/// Sliding-window (Markov-residual-bounded) variant of
/// [`generate_cached`]. Keeps only the last `window` positions of K/V
/// per layer — older tokens drop off the back of the cache and are no
/// longer attendable. Memory stays O(num_layers × window × kv_dim)
/// regardless of total generation length. Pass `window = None` for
/// unbounded growth.
pub fn generate_cached_with_window<F>(
    weights: &ModelWeights,
    tokenizer: &larql_inference::tokenizers::Tokenizer,
    ffn: &dyn FfnBackend,
    prompt_ids: &[u32],
    max_new_tokens: usize,
    window: Option<usize>,
    mut on_token: F,
) -> Vec<u32>
where
    F: FnMut(u32, &str),
{
    generate_cached_bounded(
        weights,
        tokenizer,
        ffn,
        prompt_ids,
        max_new_tokens,
        window,
        None,
        &mut on_token,
    )
}

#[allow(clippy::too_many_arguments)]
fn generate_cached_bounded(
    weights: &ModelWeights,
    tokenizer: &larql_inference::tokenizers::Tokenizer,
    ffn: &dyn FfnBackend,
    prompt_ids: &[u32],
    max_new_tokens: usize,
    window: Option<usize>,
    backend: Option<&dyn larql_compute::ComputeBackend>,
    on_token: &mut dyn FnMut(u32, &str),
) -> Vec<u32> {
    generate_cached_hooked_inner(
        weights,
        tokenizer,
        ffn,
        prompt_ids,
        max_new_tokens,
        window,
        backend,
        &mut NoopHook,
        on_token,
    )
}

/// Hook-aware autoregressive generation on the CPU KV-cache path.
///
/// Same prefill + decode loop as [`generate_cached`], but fires
/// [`LayerHook`] callbacks at every layer of every step (prefill **and**
/// every decode step):
///
/// - `on_pre_layer` — residual entering the layer.
/// - `on_post_attention(&mut h)` — post-attention residual; mutating it
///   here changes what the layer's FFN sees.
/// - `on_post_layer(&mut h)` — full-layer output; mutating it here
///   changes what the **next** layer sees.
///
/// The Metal-fast `layer_graph::generate::gpu::generate*` path is
/// hook-free by design (the kernel pipeline is fused; threading hooks
/// through it would force per-layer kernel splits even when no hook is
/// registered, so we keep the fast path fast). When you need hooks
/// during multi-token generation use this CPU path instead — typically
/// 5–20× slower than the Metal path on the same model, but every
/// primitive in [`larql_inference::forward::hooks`] works end-to-end.
///
/// The `on_attention_weights` and `on_ffn_activation` callbacks do
/// **not** fire on this path — the production decode kernels don't
/// capture those intermediates. Use
/// [`larql_inference::forward::trace_forward_full_hooked`] for a single
/// forward pass when you need them.
#[allow(clippy::too_many_arguments)]
pub fn generate_cached_hooked<F>(
    weights: &ModelWeights,
    tokenizer: &larql_inference::tokenizers::Tokenizer,
    ffn: &dyn FfnBackend,
    prompt_ids: &[u32],
    max_new_tokens: usize,
    window: Option<usize>,
    backend: Option<&dyn larql_compute::ComputeBackend>,
    hook: &mut dyn LayerHook,
    mut on_token: F,
) -> Vec<u32>
where
    F: FnMut(u32, &str),
{
    generate_cached_hooked_inner(
        weights,
        tokenizer,
        ffn,
        prompt_ids,
        max_new_tokens,
        window,
        backend,
        hook,
        &mut on_token,
    )
}

/// Drive autoregressive generation through any [`crate::KvEngine`].
///
/// This is the engine-trait-based equivalent of [`generate_cached_backend`]:
/// same prefill → sample → decode loop → sample → ... shape, but the
/// per-stage forward passes are delegated to `engine.prefill` /
/// `engine.decode_step`. Sampling, tokenizer decoding, and EOS detection
/// remain centralized here so every engine produces a stream with
/// identical sampling semantics.
///
/// Parity contract: with `engine = StandardEngine::new(window)`, the
/// returned `Vec<u32>` is bit-identical to
/// `generate_cached_backend(weights, tokenizer, ffn, prompt, max,
/// backend, window, ...)`. This is the parity gate for the unification
/// migration (see `larql-inference/docs/specs/kv-engine-unification.md` §8.4).
pub fn generate_with_engine<F>(
    engine: &mut dyn crate::KvEngine,
    weights: &ModelWeights,
    tokenizer: &larql_inference::tokenizers::Tokenizer,
    ffn: &dyn FfnBackend,
    prompt_ids: &[u32],
    max_new_tokens: usize,
    mut on_token: F,
) -> Vec<u32>
where
    F: FnMut(u32, &str),
{
    if max_new_tokens == 0 || prompt_ids.is_empty() {
        return Vec::new();
    }

    // ── Phase 1: prefill ──
    let last_hidden = match engine.prefill(weights, ffn, prompt_ids) {
        Some(h) => h,
        None => return Vec::new(),
    };

    // Sample first new token from the prefill-end hidden state.
    let first = match argmax_next_token(weights, tokenizer, &last_hidden) {
        Some(t) => t,
        None => return Vec::new(),
    };
    on_token(first.0, &first.1);

    let mut generated = Vec::with_capacity(max_new_tokens);
    generated.push(first.0);
    if is_stop_token_str(&first.1) {
        return generated;
    }
    if max_new_tokens == 1 {
        return generated;
    }

    // ── Phase 2: decode loop ──
    let mut current_id = first.0;
    for _step in 1..max_new_tokens {
        let h_step = match engine.decode_step(weights, ffn, current_id) {
            Some(h) => h,
            None => break,
        };
        let (id, tok_str) = match argmax_next_token(weights, tokenizer, &h_step) {
            Some(t) => t,
            None => break,
        };
        on_token(id, &tok_str);
        generated.push(id);
        if is_stop_token_str(&tok_str) {
            break;
        }
        current_id = id;
    }

    generated
}

/// Prefill phase as a reusable building block: runs a full forward over
/// `prompt_ids`, populates a fresh [`KvCache`] (bounded if `window` is
/// `Some`), and returns `(last_hidden_1xD, populated_cache)`.
///
/// Returns `None` if the prompt is empty or if any layer's attention
/// fails. This is the production K/V cache prefill loop, extracted so
/// `KvEngine::prefill` impls can call it directly.
///
/// The caller applies `final_norm + lm_head` to the returned hidden
/// state to get logits.
#[allow(clippy::too_many_arguments)]
pub fn kv_prefill_run(
    weights: &ModelWeights,
    ffn: &dyn FfnBackend,
    prompt_ids: &[u32],
    window: Option<usize>,
    backend: Option<&dyn larql_compute::ComputeBackend>,
    hook: &mut dyn LayerHook,
) -> Option<(Array2<f32>, KvCache)> {
    if prompt_ids.is_empty() {
        return None;
    }
    let num_layers = weights.num_layers;
    let mut cache = match window {
        Some(w) => KvCache::with_window(num_layers, w),
        None => KvCache::with_layers(num_layers),
    };

    let mut h = embed_tokens_pub(weights, prompt_ids);
    for layer in 0..num_layers {
        hook.on_pre_layer(layer, &h);

        let (mut h_post_attn, k_rope, v) =
            run_attention_with_kv_backend(weights, &h, layer, backend)?;
        cache.layers[layer] = Some((k_rope, v));
        cache.clip_layer(layer);

        hook.on_post_attention(layer, &mut h_post_attn);

        let (mut h_out, _) = run_ffn(weights, &h_post_attn, layer, ffn, false);

        hook.on_post_layer(layer, &mut h_out);
        h = h_out;
    }
    cache.next_position = prompt_ids.len();

    Some((last_row_as_2d(&h), cache))
}

/// Decode-step phase as a reusable building block: takes one new
/// `token_id`, runs the autoregressive attention against an existing
/// populated [`KvCache`], mutates the cache to append the new K/V (and
/// clip to window), and returns the new token's hidden state (shape
/// `[1, hidden_dim]`).
///
/// Returns `None` if any layer's attention fails. This is the
/// production decode step extracted so `KvEngine::decode_step` impls
/// can call it directly.
#[allow(clippy::too_many_arguments)]
pub fn kv_decode_step_run(
    weights: &ModelWeights,
    ffn: &dyn FfnBackend,
    cache: &mut KvCache,
    token_id: u32,
    backend: Option<&dyn larql_compute::ComputeBackend>,
    hook: &mut dyn LayerHook,
) -> Option<Array2<f32>> {
    let num_layers = weights.num_layers;
    let h_new = embed_tokens_pub(weights, &[token_id]);
    let abs_position = cache.next_position;
    let mut h_step = h_new;
    for layer in 0..num_layers {
        hook.on_pre_layer(layer, &h_step);

        let kv_entry = cache.layers[layer].as_ref();
        let (mut h_post_attn, new_kv) = run_attention_block_decode_step_backend(
            weights,
            &h_step,
            layer,
            kv_entry,
            abs_position,
            backend,
        )?;
        cache.layers[layer] = Some(new_kv);
        cache.clip_layer(layer);

        hook.on_post_attention(layer, &mut h_post_attn);

        let (mut h_out, _) = run_ffn(weights, &h_post_attn, layer, ffn, false);

        hook.on_post_layer(layer, &mut h_out);
        h_step = h_out;
    }
    cache.next_position += 1;
    Some(h_step)
}

#[allow(clippy::too_many_arguments)]
fn generate_cached_hooked_inner(
    weights: &ModelWeights,
    tokenizer: &larql_inference::tokenizers::Tokenizer,
    ffn: &dyn FfnBackend,
    prompt_ids: &[u32],
    max_new_tokens: usize,
    window: Option<usize>,
    backend: Option<&dyn larql_compute::ComputeBackend>,
    hook: &mut dyn LayerHook,
    on_token: &mut dyn FnMut(u32, &str),
) -> Vec<u32> {
    if max_new_tokens == 0 || prompt_ids.is_empty() {
        return Vec::new();
    }

    // ── Phase 1: prefill ──
    let (last_hidden, mut cache) =
        match kv_prefill_run(weights, ffn, prompt_ids, window, backend, hook) {
            Some(t) => t,
            None => return Vec::new(),
        };

    let first = match argmax_next_token(weights, tokenizer, &last_hidden) {
        Some(t) => t,
        None => return Vec::new(),
    };
    on_token(first.0, &first.1);

    let mut generated = Vec::with_capacity(max_new_tokens);
    generated.push(first.0);
    if is_stop_token_str(&first.1) {
        return generated;
    }
    if max_new_tokens == 1 {
        return generated;
    }

    let mut current_id = first.0;
    for _step in 1..max_new_tokens {
        let h_step = match kv_decode_step_run(weights, ffn, &mut cache, current_id, backend, hook) {
            Some(h) => h,
            None => break,
        };
        let (id, tok_str) = match argmax_next_token(weights, tokenizer, &h_step) {
            Some(t) => t,
            None => break,
        };
        on_token(id, &tok_str);
        generated.push(id);
        if is_stop_token_str(&tok_str) {
            break;
        }
        current_id = id;
    }

    generated
}

fn last_row_as_2d(h: &Array2<f32>) -> Array2<f32> {
    let seq_len = h.shape()[0];
    let hidden = h.shape()[1];
    let mut out = Array2::<f32>::zeros((1, hidden));
    out.row_mut(0).assign(&h.row(seq_len - 1));
    out
}

fn argmax_next_token(
    weights: &ModelWeights,
    tokenizer: &larql_inference::tokenizers::Tokenizer,
    h_single: &Array2<f32>,
) -> Option<(u32, String)> {
    let result = logits_to_predictions_pub(weights, h_single, tokenizer, 1, 1.0);
    let id = *result.token_ids.first()?;
    let (decoded, _) = result.predictions.first()?.clone();
    Some((id, decoded))
}

fn is_stop_token_str(s: &str) -> bool {
    matches!(
        s,
        "<eos>"
            | "</s>"
            | "<|endoftext|>"
            | "<|im_end|>"
            | "<|end_of_turn|>"
            | "<end_of_turn>"
            | "<|end_of_text|>"
            | "<|eom_id|>"
            | "<|eot_id|>"
    )
}

/// Autoregressive generation where a caller-supplied closure can mask the raw
/// logits before each argmax step.
///
/// `mask_fn(generated_ids, logits)` is called after computing logits for each
/// new token. It may modify `logits` in place (e.g. set unwanted token positions
/// to `f32::NEG_INFINITY`) before the argmax is applied. Returning without
/// modification gives the same result as unconstrained generation.
///
/// Useful for grammar-constrained generation: the caller tracks the partial
/// output and restricts the vocabulary to tokens valid at each position.
pub fn generate_cached_constrained<F, M>(
    weights: &ModelWeights,
    tokenizer: &larql_inference::tokenizers::Tokenizer,
    ffn: &dyn FfnBackend,
    prompt_ids: &[u32],
    max_new_tokens: usize,
    mut mask_fn: M,
    mut on_token: F,
) -> Vec<u32>
where
    F: FnMut(u32, &str),
    M: FnMut(&[u32], &mut Vec<f32>),
{
    if max_new_tokens == 0 || prompt_ids.is_empty() {
        return Vec::new();
    }

    let num_layers = weights.num_layers;
    let mut cache = KvCache::with_layers(num_layers);

    let mut h = embed_tokens_pub(weights, prompt_ids);
    for layer in 0..num_layers {
        let (h_post_attn, k_rope, v) = match run_attention_with_kv_backend(weights, &h, layer, None)
        {
            Some(t) => t,
            None => return Vec::new(),
        };
        cache.layers[layer] = Some((k_rope, v));
        let (h_out, _) = run_ffn(weights, &h_post_attn, layer, ffn, false);
        h = h_out;
    }
    cache.next_position = prompt_ids.len();

    let last_hidden = last_row_as_2d(&h);
    let mut logits = hidden_to_raw_logits(weights, &last_hidden);
    let mut generated: Vec<u32> = Vec::with_capacity(max_new_tokens);
    mask_fn(&generated, &mut logits);
    let (first_id, first_str) = match masked_argmax(&logits, tokenizer) {
        Some(t) => t,
        None => return Vec::new(),
    };
    on_token(first_id, &first_str);
    generated.push(first_id);
    if is_stop_token_str(&first_str) || max_new_tokens == 1 {
        return generated;
    }

    let mut current_id = first_id;
    for _step in 1..max_new_tokens {
        let h_new = embed_tokens_pub(weights, &[current_id]);
        let abs_position = cache.next_position;
        let mut h_step = h_new;
        for layer in 0..num_layers {
            let kv_entry = cache.layers[layer].as_ref();
            let (h_post_attn, new_kv) = match run_attention_block_decode_step_backend(
                weights,
                &h_step,
                layer,
                kv_entry,
                abs_position,
                None,
            ) {
                Some(t) => t,
                None => return generated,
            };
            cache.layers[layer] = Some(new_kv);
            let (h_out, _) = run_ffn(weights, &h_post_attn, layer, ffn, false);
            h_step = h_out;
        }
        cache.next_position += 1;

        let mut logits = hidden_to_raw_logits(weights, &h_step);
        mask_fn(&generated, &mut logits);
        let (id, tok_str) = match masked_argmax(&logits, tokenizer) {
            Some(t) => t,
            None => break,
        };
        on_token(id, &tok_str);
        generated.push(id);
        if is_stop_token_str(&tok_str) {
            break;
        }
        current_id = id;
    }

    generated
}

fn masked_argmax(
    logits: &[f32],
    tokenizer: &larql_inference::tokenizers::Tokenizer,
) -> Option<(u32, String)> {
    let (idx, _) = logits
        .iter()
        .enumerate()
        .filter(|(_, &v)| !v.is_nan())
        .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))?;
    let id = idx as u32;
    let decoded = tokenizer.decode(&[id], true).ok()?;
    Some((id, decoded))
}

#[cfg(test)]
mod tests {
    use super::*;
    use larql_inference::ffn::WeightFfn;
    use larql_inference::test_utils::{make_test_tokenizer, make_test_weights};

    #[test]
    fn generate_cached_returns_token_ids() {
        let weights = make_test_weights();
        let tokenizer = make_test_tokenizer(weights.vocab_size);
        let ffn = WeightFfn { weights: &weights };
        let mut decoded_tokens: Vec<String> = Vec::new();
        let ids = generate_cached(&weights, &tokenizer, &ffn, &[0u32, 1], 3, |_id, text| {
            decoded_tokens.push(text.to_string())
        });
        assert!(ids.len() <= 3, "should generate at most 3 tokens");
        assert_eq!(
            ids.len(),
            decoded_tokens.len(),
            "callback called once per token"
        );
    }

    #[test]
    fn generate_cached_with_window_limits_cache() {
        let weights = make_test_weights();
        let tokenizer = make_test_tokenizer(weights.vocab_size);
        let ffn = WeightFfn { weights: &weights };
        let ids =
            generate_cached_with_window(&weights, &tokenizer, &ffn, &[0u32], 4, Some(2), |_, _| {});
        assert!(ids.len() <= 4);
    }

    // ── generate_with_engine coverage ─────────────────────────────────────
    //
    // Synthetic engine that returns deterministic hidden states to drive
    // the helper through each branch: empty inputs, max_new_tokens=0,
    // max_new_tokens=1, normal multi-step generation, prefill failure,
    // decode failure.

    struct StubEngine {
        cache: Option<KvCache>,
        fail_prefill: bool,
        fail_decode_after: Option<usize>,
        decode_count: usize,
    }

    impl crate::KvEngine for StubEngine {
        fn name(&self) -> &str {
            "stub"
        }
        fn info(&self) -> crate::EngineInfo {
            crate::EngineInfo {
                name: "stub".into(),
                description: "test fixture".into(),
                backend: "cpu".into(),
                config: String::new(),
            }
        }
        fn prefill(
            &mut self,
            weights: &ModelWeights,
            ffn: &dyn FfnBackend,
            token_ids: &[u32],
        ) -> Option<Array2<f32>> {
            if self.fail_prefill {
                return None;
            }
            let (hidden, cache) =
                kv_prefill_run(weights, ffn, token_ids, None, None, &mut NoopHook)?;
            self.cache = Some(cache);
            Some(hidden)
        }
        fn decode_step(
            &mut self,
            weights: &ModelWeights,
            ffn: &dyn FfnBackend,
            token_id: u32,
        ) -> Option<Array2<f32>> {
            self.decode_count += 1;
            if let Some(limit) = self.fail_decode_after {
                if self.decode_count > limit {
                    return None;
                }
            }
            let cache = self.cache.as_mut()?;
            kv_decode_step_run(weights, ffn, cache, token_id, None, &mut NoopHook)
        }
        fn memory_bytes(&self) -> usize {
            0
        }
    }

    fn fresh_stub() -> StubEngine {
        StubEngine {
            cache: None,
            fail_prefill: false,
            fail_decode_after: None,
            decode_count: 0,
        }
    }

    #[test]
    fn generate_with_engine_empty_prompt_returns_empty() {
        let weights = make_test_weights();
        let tokenizer = make_test_tokenizer(weights.vocab_size);
        let ffn = WeightFfn { weights: &weights };
        let mut eng = fresh_stub();
        let out = generate_with_engine(&mut eng, &weights, &tokenizer, &ffn, &[], 5, |_, _| {});
        assert!(out.is_empty());
    }

    #[test]
    fn generate_with_engine_zero_max_returns_empty() {
        let weights = make_test_weights();
        let tokenizer = make_test_tokenizer(weights.vocab_size);
        let ffn = WeightFfn { weights: &weights };
        let mut eng = fresh_stub();
        let out = generate_with_engine(
            &mut eng,
            &weights,
            &tokenizer,
            &ffn,
            &[0u32, 1],
            0,
            |_, _| {},
        );
        assert!(out.is_empty());
    }

    #[test]
    fn generate_with_engine_max_one_returns_single_token() {
        let weights = make_test_weights();
        let tokenizer = make_test_tokenizer(weights.vocab_size);
        let ffn = WeightFfn { weights: &weights };
        let mut eng = fresh_stub();
        let out = generate_with_engine(
            &mut eng,
            &weights,
            &tokenizer,
            &ffn,
            &[0u32, 1],
            1,
            |_, _| {},
        );
        assert_eq!(out.len(), 1);
    }

    #[test]
    fn generate_with_engine_multi_step_fires_callback_per_token() {
        let weights = make_test_weights();
        let tokenizer = make_test_tokenizer(weights.vocab_size);
        let ffn = WeightFfn { weights: &weights };
        let mut eng = fresh_stub();
        let mut callbacks = 0usize;
        let out = generate_with_engine(
            &mut eng,
            &weights,
            &tokenizer,
            &ffn,
            &[0u32, 1],
            4,
            |_, _| callbacks += 1,
        );
        assert_eq!(out.len(), callbacks);
        assert!(out.len() <= 4);
    }

    #[test]
    fn generate_with_engine_prefill_failure_returns_empty() {
        let weights = make_test_weights();
        let tokenizer = make_test_tokenizer(weights.vocab_size);
        let ffn = WeightFfn { weights: &weights };
        let mut eng = fresh_stub();
        eng.fail_prefill = true;
        let out = generate_with_engine(&mut eng, &weights, &tokenizer, &ffn, &[0u32], 3, |_, _| {});
        assert!(out.is_empty());
    }

    #[test]
    fn generate_with_engine_decode_failure_breaks_loop_early() {
        let weights = make_test_weights();
        let tokenizer = make_test_tokenizer(weights.vocab_size);
        let ffn = WeightFfn { weights: &weights };
        let mut eng = fresh_stub();
        eng.fail_decode_after = Some(1);
        let out = generate_with_engine(
            &mut eng,
            &weights,
            &tokenizer,
            &ffn,
            &[0u32, 1],
            5,
            |_, _| {},
        );
        assert!(
            out.len() <= 2,
            "should break after decode failure, got {} tokens",
            out.len()
        );
    }

    #[test]
    fn generate_cached_backend_cpu() {
        let weights = make_test_weights();
        let tokenizer = make_test_tokenizer(weights.vocab_size);
        let ffn = WeightFfn { weights: &weights };
        let ids = generate_cached_backend(
            &weights,
            &tokenizer,
            &ffn,
            &[2u32, 3],
            2,
            None,
            None,
            |_, _| {},
        );
        assert!(ids.len() <= 2);
    }

    #[test]
    fn generate_cached_constrained_restricts_tokens() {
        let weights = make_test_weights();
        let tokenizer = make_test_tokenizer(weights.vocab_size);
        let ffn = WeightFfn { weights: &weights };
        let allowed: std::collections::HashSet<u32> = (0u32..8).collect();
        let ids = generate_cached_constrained(
            &weights,
            &tokenizer,
            &ffn,
            &[0u32],
            3,
            |_generated, logits| {
                for (id, logit) in logits.iter_mut().enumerate() {
                    if !allowed.contains(&(id as u32)) {
                        *logit = f32::NEG_INFINITY;
                    }
                }
            },
            |_, _| {},
        );
        for &id in &ids {
            assert!(
                allowed.contains(&id),
                "generated token {id} outside allowed set"
            );
        }
    }

    #[test]
    fn generate_cached_empty_prompt() {
        let weights = make_test_weights();
        let tokenizer = make_test_tokenizer(weights.vocab_size);
        let ffn = WeightFfn { weights: &weights };
        let ids = generate_cached(&weights, &tokenizer, &ffn, &[], 2, |_, _| {});
        assert!(ids.len() <= 2);
    }

    // ── generate_cached_hooked ────────────────────────────────────────────────

    // The unhooked and hooked decode paths are mathematically equivalent
    // under NoopHook, but BLAS reduction order can drift call-to-call on
    // Windows OpenBLAS — observed argmax flipping after the first decode
    // step. Linux/macOS BLAS implementations are bit-stable enough for
    // this assertion to hold, so we keep the coverage there.
    #[cfg(not(windows))]
    #[test]
    fn generate_cached_hooked_with_noop_matches_baseline() {
        let weights = make_test_weights();
        let tokenizer = make_test_tokenizer(weights.vocab_size);
        let ffn = WeightFfn { weights: &weights };

        let baseline = generate_cached(&weights, &tokenizer, &ffn, &[0u32, 1, 2], 4, |_, _| {});

        let hooked = generate_cached_hooked(
            &weights,
            &tokenizer,
            &ffn,
            &[0u32, 1, 2],
            4,
            None,
            None,
            &mut NoopHook,
            |_, _| {},
        );

        assert_eq!(baseline, hooked, "noop hook must not change generated ids");
    }

    #[test]
    fn generate_cached_hooked_record_fires_during_prefill_and_decode() {
        struct CountHook {
            calls: std::collections::HashMap<usize, usize>,
        }
        impl LayerHook for CountHook {
            fn on_post_layer(&mut self, layer: usize, _h: &mut Array2<f32>) {
                *self.calls.entry(layer).or_insert(0) += 1;
            }
        }

        let weights = make_test_weights();
        let tokenizer = make_test_tokenizer(weights.vocab_size);
        let ffn = WeightFfn { weights: &weights };
        let max_new = 3usize;
        let mut hook = CountHook {
            calls: std::collections::HashMap::new(),
        };

        let _ = generate_cached_hooked(
            &weights,
            &tokenizer,
            &ffn,
            &[0u32, 1],
            max_new,
            None,
            None,
            &mut hook,
            |_, _| {},
        );

        for layer in 0..weights.num_layers {
            let count = *hook.calls.get(&layer).unwrap_or(&0);
            assert!(
                count >= 1,
                "hook should fire at least once per layer (got {count} for layer {layer})"
            );
            assert!(
                count <= max_new,
                "hook fires at most max_new times per layer (got {count} for layer {layer})"
            );
        }
    }

    #[test]
    fn generate_cached_hooked_steer_changes_output() {
        use larql_inference::forward::SteerHook;
        use ndarray::Array1;

        let weights = make_test_weights();
        let tokenizer = make_test_tokenizer(weights.vocab_size);
        let ffn = WeightFfn { weights: &weights };
        let prompt = vec![1u32, 2, 3];

        let baseline = generate_cached(&weights, &tokenizer, &ffn, &prompt, 4, |_, _| {});

        let v = Array1::from_vec(
            (0..weights.hidden_size)
                .map(|i| (i as f32 + 1.0) * 0.1)
                .collect(),
        );
        let mut steer = SteerHook::new().add(0, v, 5.0);

        let steered = generate_cached_hooked(
            &weights,
            &tokenizer,
            &ffn,
            &prompt,
            4,
            None,
            None,
            &mut steer,
            |_, _| {},
        );

        if !baseline.is_empty() && !steered.is_empty() {
            assert_ne!(
                baseline, steered,
                "steering with α=5 must change generated tokens"
            );
        }
    }
}
