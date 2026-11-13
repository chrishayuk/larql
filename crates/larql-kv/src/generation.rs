//! Autoregressive generation with a CPU [`KvCache`].
//!
//! Two-phase decoder:
//!
//! 1. **Prefill.** Run a full forward pass over the prompt: per layer,
//!    attention (capturing post-RoPE K and post-V-norm V into the
//!    [`KvCache`]) → FFN → per-layer embedding (PLE, Gemma-4) →
//!    layer-scalar (Gemma-4). PLE and layer-scalar are no-ops on
//!    archs that don't define those keys (Gemma-3, TinyModel, etc.).
//! 2. **Decode.** For each new token: embed it as a single row,
//!    precompute the single-token PLE input, run decode-step attention
//!    (Q of new token attends against cached K/V + the new token's
//!    own K/V), FFN, PLE, layer-scalar, next layer. At end of layer
//!    stack, logits → argmax → next token. Streams tokens to a
//!    caller-supplied callback.
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
use larql_inference::forward::layer::apply_layer_scalar;
use larql_inference::forward::ple::{apply_per_layer_embedding, precompute_per_layer_inputs};
use larql_inference::forward::{
    embed_tokens_pub, hidden_to_raw_logits, logits_to_predictions_pub, run_ffn,
};
use larql_inference::ModelWeights;
use ndarray::Array2;

use crate::cache::KvCache;

mod kv_run;

pub use kv_run::{kv_decode_step_run, kv_prefill_run};

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
    engine: &mut crate::AnyEngine,
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
        Ok(h) => h,
        Err(_) => return Vec::new(),
    };

    // Sample first new token from the prefill-end hidden state.
    let first = match argmax_next_token(weights, tokenizer, &last_hidden) {
        Some(t) => t,
        None => return Vec::new(),
    };
    on_token(first.0, &first.1);

    let mut generated = Vec::with_capacity(max_new_tokens);
    generated.push(first.0);
    if is_stop(first.0, &first.1, tokenizer) {
        return generated;
    }
    if max_new_tokens == 1 {
        return generated;
    }

    // ── Phase 2: decode loop ──
    let mut current_id = first.0;
    for _step in 1..max_new_tokens {
        let h_step = match engine.decode_step(weights, ffn, current_id) {
            Ok(h) => h,
            Err(_) => break,
        };
        let (id, tok_str) = match argmax_next_token(weights, tokenizer, &h_step) {
            Some(t) => t,
            None => break,
        };
        on_token(id, &tok_str);
        generated.push(id);
        if is_stop(id, &tok_str, tokenizer) {
            break;
        }
        current_id = id;
    }

    generated
}

/// Like [`generate_with_engine`] but drives the engine's **resident-weights
/// quant** path (`prefill_resident` / `decode_step_resident`), threading the
/// `index` so a backend with a Q4K-direct attention kernel
/// (`LARQL_Q4K_DIRECT_ATTN`) reads packed bytes instead of `weights.tensors`.
///
/// Takes `&ModelWeights` (immutable) — the caller must have already made the
/// client weights f32-resident (so no lazy dequant / `&mut` is needed), which
/// lets `ffn` borrow the same `&weights` concurrently (the moe-shards path's
/// `RemoteMoeFfn`). With `LARQL_Q4K_DIRECT_ATTN` unset, the backend ignores the
/// index and runs the f32 path — output is identical to [`generate_with_engine`].
///
/// An engine failure ends the stream silently here. A caller whose FFN can
/// fail mid-generation (a remote expert shard) must use
/// [`generate_with_engine_resident_checked`], which returns the failure.
#[allow(clippy::too_many_arguments)]
pub fn generate_with_engine_resident<F>(
    engine: &mut crate::AnyEngine,
    weights: &ModelWeights,
    tokenizer: &larql_inference::tokenizers::Tokenizer,
    ffn: &dyn FfnBackend,
    index: &larql_inference::larql_vindex::VectorIndex,
    prompt_ids: &[u32],
    max_new_tokens: usize,
    on_token: F,
) -> Vec<u32>
where
    F: FnMut(u32, &str),
{
    generate_with_engine_resident_checked(
        engine,
        weights,
        tokenizer,
        ffn,
        index,
        prompt_ids,
        max_new_tokens,
        on_token,
    )
    .tokens
}

/// What a checked generation produced.
///
/// `tokens` is every token emitted before the stream ended; `error` is why it
/// ended early, when an engine step failed. A stream with `error: Some` is not
/// a complete answer — the tokens are what came out before the failure, and a
/// caller must not present them as the model's continuation.
#[derive(Debug)]
pub struct EngineGeneration {
    pub tokens: Vec<u32>,
    pub error: Option<larql_inference::kv_engine::EngineError>,
}

/// [`generate_with_engine_resident`], returning the engine failure that ended
/// the stream instead of swallowing it.
///
/// A prefill failure yields no tokens; a decode failure yields the tokens
/// emitted before it. Either way `error` carries the engine's error — for a
/// strict remote-MoE FFN, the failed shard, layer, experts and cause.
#[allow(clippy::too_many_arguments)]
pub fn generate_with_engine_resident_checked<F>(
    engine: &mut crate::AnyEngine,
    weights: &ModelWeights,
    tokenizer: &larql_inference::tokenizers::Tokenizer,
    ffn: &dyn FfnBackend,
    index: &larql_inference::larql_vindex::VectorIndex,
    prompt_ids: &[u32],
    max_new_tokens: usize,
    mut on_token: F,
) -> EngineGeneration
where
    F: FnMut(u32, &str),
{
    let mut generated = Vec::new();
    if max_new_tokens == 0 || prompt_ids.is_empty() {
        return EngineGeneration {
            tokens: generated,
            error: None,
        };
    }
    let finish = |tokens: Vec<u32>, error| EngineGeneration { tokens, error };

    let last_hidden = match engine.prefill_resident(weights, ffn, index, prompt_ids) {
        Ok(h) => h,
        Err(e) => return finish(generated, Some(e)),
    };

    let Some(first) = argmax_next_token_resident(weights, tokenizer, index, &last_hidden) else {
        return finish(generated, None);
    };
    on_token(first.0, &first.1);

    generated.reserve(max_new_tokens);
    generated.push(first.0);
    if is_stop(first.0, &first.1, tokenizer) || max_new_tokens == 1 {
        return finish(generated, None);
    }

    let mut current_id = first.0;
    for _step in 1..max_new_tokens {
        let h_step = match engine.decode_step_resident(weights, ffn, index, current_id) {
            Ok(h) => h,
            Err(e) => return finish(generated, Some(e)),
        };
        let Some((id, tok_str)) = argmax_next_token_resident(weights, tokenizer, index, &h_step)
        else {
            break;
        };
        on_token(id, &tok_str);
        generated.push(id);
        if is_stop(id, &tok_str, tokenizer) {
            break;
        }
        current_id = id;
    }

    finish(generated, None)
}

/// Multi-modal-capable peer of [`generate_with_engine`]. Same shape;
/// the only difference is the prefill input: pre-built initial hidden
/// state (e.g. from `larql_compute::forward::embed_plan` on an
/// `EmbeddingPlan` mixing `Tokens` and `Precomputed` chunks) instead of
/// a token-id slice.
///
/// **Contract: caller MUST verify `engine.supports_multimodal()` returns
/// true BEFORE calling this function** (see ADR-0023). The default
/// `prefill_from_hidden` impl panics on engines that don't support
/// MM; the capability check is the contract and the panic is
/// defense-in-depth. For text-only inputs on any engine, use
/// `generate_with_engine` instead.
///
/// `max_new_tokens` accounting is independent of `initial_hidden.nrows()`
/// — the budget counts only newly *decoded* tokens, never prefill rows
/// (which may include vision/audio embeddings that aren't tokens at all).
/// Pinned by the `max_tokens_independent_of_hidden_rows` test below.
///
/// Bit-identity contract: feeding the single-Tokens-chunk plan
/// `EmbeddingPlan::from_tokens(prompt_ids)` through `embed_plan` then
/// this function produces the same token stream as
/// `generate_with_engine(engine, ..., prompt_ids, max_new_tokens, on_token)`
/// — same sampling, same EOS, same callback shape. Pinned by the
/// `wrapper_text_only_plan_matches_generate_with_engine` test below.
pub fn generate_with_engine_from_hidden<F>(
    engine: &mut crate::AnyEngine,
    weights: &ModelWeights,
    tokenizer: &larql_inference::tokenizers::Tokenizer,
    ffn: &dyn FfnBackend,
    initial_hidden: &Array2<f32>,
    max_new_tokens: usize,
    mut on_token: F,
) -> Vec<u32>
where
    F: FnMut(u32, &str),
{
    if max_new_tokens == 0 || initial_hidden.nrows() == 0 {
        return Vec::new();
    }

    // ── Phase 1: prefill from pre-built hidden state ──
    // Panics if engine doesn't support MM; capability check is upstream
    // per ADR-0023. An Err return (e.g. BackendFailure on dispatch) is
    // mapped to an empty stream, matching `generate_with_engine`'s
    // post-refactor behaviour.
    let last_hidden = match engine.prefill_from_hidden(weights, ffn, initial_hidden) {
        Ok(h) => h,
        Err(_) => return Vec::new(),
    };

    // ── Sample first new token (identical to generate_with_engine) ──
    let first = match argmax_next_token(weights, tokenizer, &last_hidden) {
        Some(t) => t,
        None => return Vec::new(),
    };
    on_token(first.0, &first.1);

    let mut generated = Vec::with_capacity(max_new_tokens);
    generated.push(first.0);
    if is_stop(first.0, &first.1, tokenizer) {
        return generated;
    }
    if max_new_tokens == 1 {
        return generated;
    }

    // ── Phase 2: decode loop (verbatim from generate_with_engine; ADR-0023
    // scoped-out: decode-loop embedding stays text-token-based) ──
    let mut current_id = first.0;
    for _step in 1..max_new_tokens {
        let h_step = match engine.decode_step(weights, ffn, current_id) {
            Ok(h) => h,
            Err(_) => break,
        };
        let (id, tok_str) = match argmax_next_token(weights, tokenizer, &h_step) {
            Some(t) => t,
            None => break,
        };
        on_token(id, &tok_str);
        generated.push(id);
        if is_stop(id, &tok_str, tokenizer) {
            break;
        }
        current_id = id;
    }

    generated
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
    let (last_hidden, mut cache) = match kv_prefill_run(
        larql_inference::WeightsView::dense(weights),
        ffn,
        prompt_ids,
        window,
        backend,
        hook,
    ) {
        Ok(t) => t,
        Err(_) => return Vec::new(),
    };

    let first = match argmax_next_token(weights, tokenizer, &last_hidden) {
        Some(t) => t,
        None => return Vec::new(),
    };
    on_token(first.0, &first.1);

    let mut generated = Vec::with_capacity(max_new_tokens);
    generated.push(first.0);
    if is_stop(first.0, &first.1, tokenizer) {
        return generated;
    }
    if max_new_tokens == 1 {
        return generated;
    }

    let mut current_id = first.0;
    for _step in 1..max_new_tokens {
        let h_step = match kv_decode_step_run(weights, ffn, &mut cache, current_id, backend, hook) {
            Ok(h) => h,
            Err(_) => break,
        };
        let (id, tok_str) = match argmax_next_token(weights, tokenizer, &h_step) {
            Some(t) => t,
            None => break,
        };
        on_token(id, &tok_str);
        generated.push(id);
        if is_stop(id, &tok_str, tokenizer) {
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
    // lm_head (vocab projection) dominates this call — time it for the
    // decode-stage split (`LARQL_DECODE_STAGES=1`).
    let _t_lmhead = std::time::Instant::now();
    let result = logits_to_predictions_pub(weights, h_single, tokenizer, 1, 1.0);
    larql_inference::decode_stages::record_lmhead(_t_lmhead.elapsed().as_nanos());
    let id = *result.token_ids.first()?;
    let (decoded, _) = result.predictions.first()?.clone();
    Some((id, decoded))
}

/// `LARQL_Q4K_LM_HEAD=1` routes the resident-path lm_head matvec through
/// the vindex's Q4_K lm_head view (synthesised from f16 embeddings at load
/// for tied-embedding models) instead of the f32 row-parallel sgemv. On a
/// 262K-vocab head this drops lm_head bandwidth ~4× (e.g. 2.95 GB → 0.42 GB
/// per step on Gemma 4 26B-A4B). **Default on** (`LARQL_Q4K_LM_HEAD=0` opts
/// out); falls back to the f32 path when no Q4_K head view exists.
fn q4k_lm_head_enabled() -> bool {
    larql_compute::options::q4k_lm_head_enabled()
}

/// Resident-path argmax: like [`argmax_next_token`] but with the vindex at
/// hand, so the lm_head matvec can run Q4_K-direct under
/// `LARQL_Q4K_LM_HEAD=1`. Falls back to the f32 path when the flag is off
/// or the vindex has no Q4_K lm_head view (untied model without
/// `lm_head_q4.bin`).
fn argmax_next_token_resident(
    weights: &ModelWeights,
    tokenizer: &larql_inference::tokenizers::Tokenizer,
    index: &larql_inference::larql_vindex::VectorIndex,
    h_single: &Array2<f32>,
) -> Option<(u32, String)> {
    if q4k_lm_head_enabled() && index.vocab_size > 0 {
        if let Some(q4_bytes) = index.storage.lm_head_kquant_view() {
            let _t_lmhead = std::time::Instant::now();
            // Argmax-only fast path: skips the full-vocab softmax + top-k
            // temporaries (scaling/softcap/temperature are monotone, so the
            // selected token is identical to the softmax route).
            let result = larql_inference::forward::predict::q4_lm_head_argmax(
                weights,
                h_single,
                q4_bytes.as_ref(),
                index.vocab_size,
                tokenizer,
            );
            larql_inference::decode_stages::record_lmhead(_t_lmhead.elapsed().as_nanos());
            if let Some((id, decoded)) = result {
                return Some((id, decoded));
            }
        }
    }
    argmax_next_token(weights, tokenizer, h_single)
}

/// Whether `id` ends generation, by the shared EOS config's built-in stops.
/// Matched on the raw token when the clean decode dropped it as a special
/// token — the case a string-only check misses (`<end_of_turn>`,
/// `<|eot_id|>` decode to `""` with `skip_special_tokens`).
fn is_stop(id: u32, decoded: &str, tokenizer: &larql_inference::tokenizers::Tokenizer) -> bool {
    static EOS: std::sync::OnceLock<larql_inference::EosConfig> = std::sync::OnceLock::new();
    EOS.get_or_init(larql_inference::EosConfig::builtin)
        .is_eos_with_tokenizer(id, decoded, tokenizer)
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
    // Per-Layer Embedding inputs for Gemma-4 archs — same per-layer
    // sequence as `kv_prefill_run` (empty Vec / no-ops elsewhere).
    let ple_inputs = precompute_per_layer_inputs(weights, &h, prompt_ids);
    for layer in 0..num_layers {
        let (h_post_attn, k_rope, v) = match run_attention_with_kv_backend(
            larql_inference::WeightsView::dense(weights),
            &h,
            layer,
            None,
            None,
        ) {
            Some(t) => t,
            None => return Vec::new(),
        };
        cache.layers[layer] = Some((k_rope, v));
        let (h_post_ffn, _) = run_ffn(weights, &h_post_attn, layer, ffn, false);
        let mut h_out =
            apply_per_layer_embedding(weights, &h_post_ffn, layer, ple_inputs.get(layer));
        apply_layer_scalar(weights, &mut h_out, layer);
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
    if is_stop(first_id, &first_str, tokenizer) || max_new_tokens == 1 {
        return generated;
    }

    let mut current_id = first_id;
    for _step in 1..max_new_tokens {
        let h_new = embed_tokens_pub(weights, &[current_id]);
        let abs_position = cache.next_position;
        // PLE inputs are per-token — recompute for this single-token
        // decode step, same sequence as `kv_decode_step_run`.
        let ple_inputs = precompute_per_layer_inputs(weights, &h_new, &[current_id]);
        let mut h_step = h_new;
        for layer in 0..num_layers {
            let kv_entry = cache.layers[layer].as_ref();
            let (h_post_attn, new_kv) = match run_attention_block_decode_step_backend(
                larql_inference::WeightsView::dense(weights),
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
            let (h_post_ffn, _) = run_ffn(weights, &h_post_attn, layer, ffn, false);
            let mut h_out =
                apply_per_layer_embedding(weights, &h_post_ffn, layer, ple_inputs.get(layer));
            apply_layer_scalar(weights, &mut h_out, layer);
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
        if is_stop(id, &tok_str, tokenizer) {
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
mod tests;
