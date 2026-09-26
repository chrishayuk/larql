//! `POST /v1/completions` — OpenAI-compatible legacy text completions (N0.2).
//!
//! Implements the [OpenAI Completions API](https://platform.openai.com/docs/api-reference/completions/create)
//! shape so existing `openai` SDKs and eval harnesses work unmodified:
//!
//! ```python
//! from openai import OpenAI
//! client = OpenAI(base_url="http://larql:8080/v1", api_key="sk-...")
//! resp = client.completions.create(
//!     model="gemma-3-4b",
//!     prompt="The capital of France is",
//!     max_tokens=10,
//! )
//! ```
//!
//! ## Generation path
//!
//! Buffered + SSE both run the **KV-cached** generation loop in
//! `larql_inference::layer_graph::generate{,_with_sampling,_streaming}`.
//! The buffered path uses `generate_with_sampling`; the SSE path uses
//! `generate_streaming` and pumps the per-token callback into an mpsc
//! channel. Generation acquires an exclusive write guard on
//! `LoadedModel.weights` for the duration; concurrent reads block,
//! other endpoints are unaffected in steady state.
//!
//! ## Streaming (slice 3)
//!
//! `stream: true` returns an SSE response — `text/event-stream` with
//! one `data: {chunk}\n\n` event per generated token, terminated by
//! `data: [DONE]\n\n`. Each chunk's shape mirrors the OpenAI
//! Completions stream: `{id, object: "text_completion", created,
//! model, choices: [{text, index, finish_reason, logprobs: null}]}`.
//! The final chunk before `[DONE]` carries `finish_reason: "stop" |
//! "length"`.
//!
//! Generation runs on the blocking pool; the stream channel is
//! capacity-bounded so the producer back-pressures naturally on slow
//! clients. Client disconnect cleans up early on the next
//! `blocking_send` failure.
//!
//! ## Logprobs
//!
//! `logprobs: int` returns `null` in the response. Top-k log-probabilities
//! over the lm_head distribution land in F18.

use std::convert::Infallible;
use std::sync::Arc;

use axum::extract::State;
use axum::response::sse::{Event, KeepAlive, Sse};
use axum::response::{IntoResponse, Response};
use axum::Json;
use futures::stream::Stream;
use serde::{Deserialize, Serialize};
use tokio_stream::wrappers::ReceiverStream;
use tokio_stream::StreamExt as _;

use crate::error::ServerError;
use crate::routes::openai::OpenAIError;
use crate::state::{AppState, LoadedModel};

use super::util::{
    contains_any, error_chunk, new_id_suffix, trim_at_stop, unix_now, StopSpec,
    FINISH_REASON_LENGTH, FINISH_REASON_STOP, SSE_CHANNEL_DEPTH, SSE_DONE,
};

pub(super) const TEXT_COMPLETION_OBJECT: &str = "text_completion";
pub(super) const DEFAULT_MAX_TOKENS: usize = 16;

#[derive(Deserialize)]
#[serde(untagged)]
pub enum CompletionPrompt {
    Single(String),
    Batch(Vec<String>),
}

#[derive(Deserialize)]
pub struct CompletionsRequest {
    pub model: Option<String>,
    pub prompt: CompletionPrompt,
    #[serde(default)]
    pub max_tokens: Option<usize>,
    #[serde(default)]
    pub temperature: Option<f32>,
    /// Nucleus (top-p) filter applied after temperature scaling. Only
    /// honoured when `temperature > 0`; for greedy decoding it's a no-op.
    #[serde(default)]
    pub top_p: Option<f32>,
    /// Streaming via SSE — emits one `text_completion` chunk per token,
    /// terminated by `data: [DONE]\n\n`.
    #[serde(default)]
    pub stream: Option<bool>,
    /// Number of completions per prompt — only `n=1` supported; values
    /// >1 return 501.
    #[serde(default)]
    pub n: Option<usize>,
    /// Stop strings — accepted; first match halts generation.
    #[serde(default)]
    pub stop: Option<StopSpec>,
    /// Echo the prompt in the completion text (OpenAI legacy behaviour).
    #[serde(default)]
    pub echo: Option<bool>,
    /// Top-k log-probs — returns `null` in the response (F18 follow-up).
    #[serde(default)]
    pub logprobs: Option<usize>,
    /// Best-of — accepted, ignored (treats as 1).
    #[serde(default)]
    pub best_of: Option<usize>,
    /// Seed for reproducible sampling. Same seed + same temperature +
    /// same prompt produces the same tokens. No-op for greedy mode.
    #[serde(default)]
    pub seed: Option<u64>,
    /// End-user id — logged via tracing if set, otherwise no-op.
    #[serde(default)]
    pub user: Option<String>,
    /// OpenAI repetition penalty: subtract `freq * count(token)` from
    /// each candidate's logit before softmax. Range `[-2.0, 2.0]`;
    /// values outside that band are clamped server-side.
    #[serde(default)]
    pub frequency_penalty: Option<f32>,
    /// OpenAI presence penalty: subtract `presence * 1` from any token
    /// that's already appeared. Range `[-2.0, 2.0]`.
    #[serde(default)]
    pub presence_penalty: Option<f32>,
}

#[derive(Serialize)]
pub struct CompletionChoice {
    pub text: String,
    pub index: usize,
    pub finish_reason: &'static str,
    /// Populated when the request set `logprobs: int`. `None`
    /// (serialised as `null`) otherwise.
    pub logprobs: Option<CompletionLogprobs>,
}

/// Legacy `/v1/completions` logprobs shape — parallel arrays of
/// per-token info. Different from chat completions' nested-content
/// envelope, but the inner data is the same.
///
/// `top_logprobs` is one map per token of `{candidate → logprob}`;
/// empty maps until the inference layer exposes top-K alternatives
/// (follow-up). The picked-token entry alone preserves wire shape so
/// existing eval harnesses parse cleanly.
#[derive(Serialize)]
pub struct CompletionLogprobs {
    pub tokens: Vec<String>,
    pub token_logprobs: Vec<f64>,
    pub top_logprobs: Vec<std::collections::BTreeMap<String, f64>>,
    pub text_offset: Vec<usize>,
}

#[derive(Serialize)]
pub struct CompletionsUsage {
    pub prompt_tokens: usize,
    pub completion_tokens: usize,
    pub total_tokens: usize,
}

#[derive(Serialize)]
pub struct CompletionsResponse {
    pub id: String,
    pub object: &'static str,
    pub created: u64,
    pub model: String,
    pub choices: Vec<CompletionChoice>,
    pub usage: CompletionsUsage,
}

#[utoipa::path(
    post,
    path = "/v1/completions",
    tag = "openai",
    request_body = crate::openapi::schemas::OpenAiCompletionsRequest,
    responses(
        (status = 200, description = "Non-streaming JSON response.",
         body = crate::openapi::schemas::OpenAiCompletionsResponse),
        (status = 200, description = "SSE stream when `stream: true`. Each event is `data: <CompletionsChunk JSON>\\n\\n`, terminated by `data: [DONE]`.",
         content_type = "text/event-stream", body = String),
        (status = 400, body = crate::routes::openai::error::OpenAIErrorBody),
        (status = 500, body = crate::routes::openai::error::OpenAIErrorBody),
    ),
)]
pub async fn handle_completions(
    State(state): State<Arc<AppState>>,
    Json(req): Json<CompletionsRequest>,
) -> Result<Response, OpenAIError> {
    state.bump_requests();

    if req.n.unwrap_or(1) > 1 {
        return Err(OpenAIError::invalid_request(
            "n>1 not yet supported; only n=1 (single completion per prompt)",
        ));
    }

    // The one V2/V3 decision point: the registry resolves the request's
    // model to a runtime binding; below this match nothing re-detects
    // the container generation.
    let served = state.served_or_err(req.model.as_deref())?;
    if let crate::state::ServedModel::V2(model) = &served {
        if model.infer_disabled {
            return Err(OpenAIError::service_unavailable(
                "inference disabled (--no-infer / --embed-only / --ffn-only)",
            ));
        }
    }

    let prompts: Vec<String> = match req.prompt {
        CompletionPrompt::Single(s) => vec![s],
        CompletionPrompt::Batch(v) => v,
    };
    if prompts.is_empty() {
        return Err(OpenAIError::invalid_request("prompt is empty"));
    }

    let max_tokens = req.max_tokens.unwrap_or(DEFAULT_MAX_TOKENS);
    let sampling_params = super::util::SamplingParams {
        temperature: req.temperature,
        top_p: req.top_p,
        seed: req.seed,
        frequency_penalty: req.frequency_penalty,
        presence_penalty: req.presence_penalty,
    };
    let stop_strings: Vec<String> = req
        .stop
        .as_ref()
        .map(|s| s.as_slice().to_vec())
        .unwrap_or_default();
    let echo = req.echo.unwrap_or(false);
    let stream = req.stream.unwrap_or(false);

    if stream {
        // Streaming mode: SSE response. `echo` and batched prompts are
        // not supported in stream mode (OpenAI's stream contract is
        // one prompt → one stream of chunks). Validated here, before
        // runtime dispatch, so both runtimes share one contract.
        if echo {
            return Err(OpenAIError::invalid_request(
                "echo=true is not supported with stream=true",
            ));
        }
        if prompts.len() > 1 {
            return Err(OpenAIError::invalid_request(
                "batched prompts (prompt: [...]) are not supported with stream=true; \
                 send one prompt per request",
            ));
        }
    }

    // Model id for the response (matches the request when given,
    // otherwise the loaded model's id).
    let model_id = req.model.clone().unwrap_or_else(|| match &served {
        crate::state::ServedModel::V2(m) => m.id.clone(),
        crate::state::ServedModel::V3(m) => m.id.clone(),
    });

    let model_arc = match served {
        crate::state::ServedModel::V3(v3) => {
            return super::v3_completions::respond(
                v3.clone(),
                prompts,
                max_tokens,
                sampling_params,
                stop_strings,
                echo,
                stream,
                model_id,
                state.infer_timeout,
                Arc::clone(&state.runtime),
            )
            .await;
        }
        crate::state::ServedModel::V2(model) => model.clone(),
    };

    if stream {
        let prompt = prompts.into_iter().next().unwrap();
        return Ok(stream_completions(
            model_arc,
            prompt,
            max_tokens,
            sampling_params,
            stop_strings,
            model_id,
            Arc::clone(&state.runtime),
        )
        .into_response());
    }

    // Non-streaming: the existing buffered path. Racing the blocking
    // generation against `state.infer_timeout` mirrors
    // `run_infer_with_timeout` (routes/infer.rs, BUG-infer-deadlock §5.6):
    // without it, a stuck/slow generation call holds `LoadedModel.weights`'
    // write guard for as long as the spawned thread runs, and every other
    // OpenAI-route request queues on that guard indefinitely. On timeout we
    // drop the JoinHandle and respond 504; the spawned thread finishes (or
    // doesn't) in the background, same tradeoff /v1/infer already accepts.
    let logprobs_requested = req.logprobs;
    let started = std::time::Instant::now();
    let _gen_guard = Arc::clone(&state.runtime).enter_generation();
    let handle = tokio::task::spawn_blocking(move || -> Result<_, ServerError> {
        let mut tally = crate::runtime_stats::GenerationTally::new();
        let out = run_completions_loop(
            &model_arc,
            &prompts,
            max_tokens,
            sampling_params,
            &stop_strings,
            echo,
            logprobs_requested,
            &mut tally,
        )?;
        Ok((out, tally))
    });
    let timeout = state.infer_timeout;
    let ((choices, prompt_tokens, completion_tokens), tally) = if timeout.is_zero() {
        handle
            .await
            .map_err(|e| ServerError::Internal(e.to_string()))??
    } else {
        match tokio::time::timeout(timeout, handle).await {
            Ok(join_result) => join_result.map_err(|e| ServerError::Internal(e.to_string()))??,
            Err(_elapsed) => {
                tracing::warn!(
                    target: "larql_server::openai::completions",
                    "completion timed out after {:.1}s; dropping in-flight task and \
                     responding 504 (background thread will finish on its own)",
                    started.elapsed().as_secs_f64(),
                );
                return Err(OpenAIError::from(ServerError::Timeout(format!(
                    "completion exceeded server-side timeout of {}s",
                    timeout.as_secs(),
                ))));
            }
        }
    };
    state
        .runtime
        .record(tally.into_sample(crate::state::elapsed_ms(started)));

    Ok(Json(CompletionsResponse {
        id: format!("cmpl-{}", new_id_suffix()),
        object: TEXT_COMPLETION_OBJECT,
        created: unix_now(),
        model: model_id,
        choices,
        usage: CompletionsUsage {
            prompt_tokens,
            completion_tokens,
            total_tokens: prompt_tokens + completion_tokens,
        },
    })
    .into_response())
}

/// Build an SSE response that streams one chunk per generated token.
/// Final chunk carries `finish_reason`; the stream terminates with
/// `data: [DONE]\n\n`.
#[allow(clippy::too_many_arguments)]
fn stream_completions(
    model: Arc<LoadedModel>,
    prompt: String,
    max_tokens: usize,
    sampling_params: super::util::SamplingParams,
    stop_strings: Vec<String>,
    model_id: String,
    runtime: Arc<crate::runtime_stats::RuntimeRecorder>,
) -> Sse<impl Stream<Item = Result<Event, Infallible>>> {
    let (tx, rx) = tokio::sync::mpsc::channel::<String>(SSE_CHANNEL_DEPTH);
    let cmpl_id = format!("cmpl-{}", new_id_suffix());
    let call_started = std::time::Instant::now();

    tokio::task::spawn_blocking(move || {
        let _gen_guard = runtime.clone().enter_generation();

        // BitNet (--keep-quant) vindexes take the native-ternary
        // streaming path: skips the dense weights write-lock (which
        // serialises all generation) and runs
        // generate_streaming_bitnet against the pre-loaded
        // BitnetModel at ~1.4 GB resident instead of ~5 GB.
        if model.is_bitnet() {
            let bitnet_guard = match model.get_or_load_bitnet() {
                Ok(g) => g,
                Err(e) => {
                    let _ = tx.blocking_send(error_chunk(&e));
                    return;
                }
            };
            let bitnet: &larql_inference::ternary::BitnetModel = &bitnet_guard;
            let encoding = match model.tokenizer.encode(prompt.as_str(), true) {
                Ok(e) => e,
                Err(e) => {
                    let _ = tx.blocking_send(error_chunk(&format!("tokenize: {e}")));
                    return;
                }
            };
            let prompt_ids: Vec<u32> = encoding.get_ids().to_vec();
            if prompt_ids.is_empty() {
                let _ = tx.blocking_send(error_chunk("prompt tokenises to empty"));
                return;
            }

            let (sampling, eos) = super::util::build_sampling_eos(sampling_params, &stop_strings);

            let cmpl_id_cb = cmpl_id.clone();
            let model_id_cb = model_id.clone();
            let tx_cb = tx.clone();
            let stop_strings_cb = stop_strings.clone();
            let mut completion_text = String::new();
            let mut early_stop = false;
            let mut emitted = 0usize;
            let _ = larql_inference::ternary::generate_streaming_bitnet(
                bitnet,
                &model.tokenizer,
                &prompt_ids,
                max_tokens,
                sampling,
                &eos,
                |_id, text, _ms| {
                    if early_stop {
                        return;
                    }
                    let chunk =
                        build_text_completion_chunk(&cmpl_id_cb, &model_id_cb, Some(text), None);
                    if tx_cb.blocking_send(chunk).is_err() {
                        early_stop = true;
                        return;
                    }
                    completion_text.push_str(text);
                    emitted += 1;
                    if !stop_strings_cb.is_empty()
                        && contains_any(&completion_text, &stop_strings_cb)
                    {
                        early_stop = true;
                    }
                },
            );

            // Record the generation the same way the dense path does,
            // or `/v1/stats` would report BitNet traffic as zero
            // throughput. `add_v3` takes plain counts, which is all the
            // ternary path produces (no GenerateResult); the split
            // between prefill and decode is not separately measured
            // here, so the whole span is attributed to decode.
            let mut tally = crate::runtime_stats::GenerationTally::new();
            let elapsed = crate::state::elapsed_ms(call_started);
            tally.add_v3(prompt_ids.len(), emitted, 0.0, elapsed);
            runtime.record(tally.into_sample(elapsed));

            let finish_reason: &'static str = if early_stop || emitted < max_tokens {
                FINISH_REASON_STOP
            } else {
                FINISH_REASON_LENGTH
            };
            let final_chunk =
                build_text_completion_chunk(&cmpl_id, &model_id, None, Some(finish_reason));
            let _ = tx.blocking_send(final_chunk);
            return;
        }

        let mut weights_guard = match model.lock_weights_for_gen() {
            Ok(w) => w,
            Err(e) => {
                let _ = tx.blocking_send(error_chunk(&e));
                return;
            }
        };
        let weights: &mut larql_inference::ModelWeights = &mut weights_guard;
        let encoding = match model.tokenizer.encode(prompt.as_str(), true) {
            Ok(e) => e,
            Err(e) => {
                let _ = tx.blocking_send(error_chunk(&format!("tokenize: {e}")));
                return;
            }
        };
        let prompt_ids: Vec<u32> = encoding.get_ids().to_vec();
        if prompt_ids.is_empty() {
            let _ = tx.blocking_send(error_chunk("prompt tokenises to empty"));
            return;
        }

        let (sampling, eos) = super::util::build_sampling_eos(sampling_params, &stop_strings);

        let patched = model.patched.blocking_read();
        let index = patched.base();
        let backend = larql_compute::default_backend();
        let cached_layers = larql_inference::CachedLayerGraph::from_residuals(Vec::new());
        let num_layers = weights.num_layers;

        let cmpl_id_cb = cmpl_id.clone();
        let model_id_cb = model_id.clone();
        let tx_cb = tx.clone();
        let stop_strings_cb = stop_strings.clone();
        let mut completion_text = String::new();
        let mut early_stop = false;
        let result = larql_inference::layer_graph::generate_streaming(
            weights,
            &model.tokenizer,
            &prompt_ids,
            max_tokens,
            index,
            &*backend,
            &cached_layers,
            0..num_layers,
            sampling,
            &eos,
            |_id, text, _prob| {
                if early_stop {
                    return;
                }
                let chunk =
                    build_text_completion_chunk(&cmpl_id_cb, &model_id_cb, Some(text), None);
                if tx_cb.blocking_send(chunk).is_err() {
                    early_stop = true;
                    return;
                }
                completion_text.push_str(text);
                if !stop_strings_cb.is_empty() && contains_any(&completion_text, &stop_strings_cb) {
                    early_stop = true;
                }
            },
            None,
        );

        let finish_reason: &'static str = if early_stop || result.tokens.len() < max_tokens {
            FINISH_REASON_STOP
        } else {
            FINISH_REASON_LENGTH
        };
        let mut tally = crate::runtime_stats::GenerationTally::new();
        tally.add_v2(&result, prompt_ids.len());
        runtime.record(tally.into_sample(crate::state::elapsed_ms(call_started)));
        let final_chunk =
            build_text_completion_chunk(&cmpl_id, &model_id, None, Some(finish_reason));
        let _ = tx.blocking_send(final_chunk);
    });

    let stream = ReceiverStream::new(rx)
        .map(|data| Event::default().data(data))
        .chain(tokio_stream::once(Event::default().data(SSE_DONE)))
        .map(Ok::<_, Infallible>);

    Sse::new(stream).keep_alive(KeepAlive::default())
}

pub(super) fn build_text_completion_chunk(
    id: &str,
    model: &str,
    text: Option<&str>,
    finish_reason: Option<&'static str>,
) -> String {
    let chunk = serde_json::json!({
        "id": id,
        "object": TEXT_COMPLETION_OBJECT,
        "created": unix_now(),
        "model": model,
        "choices": [{
            "text": text.unwrap_or(""),
            "index": 0,
            "logprobs": serde_json::Value::Null,
            "finish_reason": match finish_reason {
                Some(r) => serde_json::Value::String(r.to_string()),
                None => serde_json::Value::Null,
            },
        }]
    });
    chunk.to_string()
}

/// Generate completions for every prompt. Returns
/// `(choices, prompt_tokens_sum, completion_tokens_sum)`.
/// Post-process a generated token stream into `(completion_text,
/// completion_tokens, finish_reason)`.
///
/// Pure — depends only on the tokens and stop strings, not on the model or
/// platform — so the EOS-break and stop-trim branches are unit-testable
/// deterministically. (Driving them through real generation is
/// non-deterministic across platforms: the synthetic CI model emits
/// different tokens on Ubuntu vs macOS, which is exactly what made
/// `completions.rs` coverage flap. Keeping this logic pure pins it.)
pub(super) fn finalize_completion(
    tokens: &[(String, f64)],
    stop_strings: &[String],
) -> (String, Vec<(String, f64)>, &'static str) {
    let mut completion_text = String::new();
    let mut completion_tokens: Vec<(String, f64)> = Vec::new();
    let mut finish_reason = FINISH_REASON_LENGTH;
    for (text, prob) in tokens {
        completion_text.push_str(text);
        completion_tokens.push((text.clone(), *prob));
        if larql_inference::vindex::is_end_of_turn(text) {
            finish_reason = FINISH_REASON_STOP;
            break;
        }
    }
    if !stop_strings.is_empty() && contains_any(&completion_text, stop_strings) {
        completion_text = trim_at_stop(&completion_text, stop_strings);
        finish_reason = FINISH_REASON_STOP;
        // Drop tokens past the byte boundary so logprobs and text stay
        // length-aligned.
        let target = completion_text.len();
        let mut acc = 0usize;
        completion_tokens.retain(|(t, _)| {
            if acc >= target {
                return false;
            }
            acc += t.len();
            true
        });
    }
    (completion_text, completion_tokens, finish_reason)
}

#[allow(clippy::too_many_arguments)]
fn run_completions_loop(
    model: &LoadedModel,
    prompts: &[String],
    max_tokens: usize,
    sampling_params: super::util::SamplingParams,
    stop_strings: &[String],
    echo: bool,
    logprobs_requested: Option<usize>,
    tally: &mut crate::runtime_stats::GenerationTally,
) -> Result<(Vec<CompletionChoice>, usize, usize), ServerError> {
    // Take an exclusive write guard on the weights. Each prompt in
    // the batch is generated in turn under the same guard so the
    // dequant cache only warms once.
    let mut weights_guard = model
        .lock_weights_for_gen()
        .map_err(ServerError::InferenceUnavailable)?;
    let weights: &mut larql_inference::ModelWeights = &mut weights_guard;

    let patched = model.patched.blocking_read();
    let index = patched.base();
    let backend = larql_compute::default_backend();
    let cached_layers = larql_inference::CachedLayerGraph::from_residuals(Vec::new());
    let num_layers = weights.num_layers;

    let mut choices = Vec::with_capacity(prompts.len());
    let mut total_prompt_tokens = 0usize;
    let mut total_completion_tokens = 0usize;

    for (idx, prompt) in prompts.iter().enumerate() {
        let encoding = model
            .tokenizer
            .encode(prompt.as_str(), true)
            .map_err(|e| ServerError::Internal(format!("tokenize: {e}")))?;
        let prompt_ids: Vec<u32> = encoding.get_ids().to_vec();
        if prompt_ids.is_empty() {
            return Err(ServerError::BadRequest(format!(
                "prompt[{idx}] tokenises to empty"
            )));
        }
        total_prompt_tokens += prompt_ids.len();

        // Build a fresh (sampling, eos) per prompt so the seed advances
        // deterministically — `SamplingConfig::with_seed` keeps the same
        // RNG seed across each prompt, which is what callers expect when
        // a seed is provided.
        let (sampling, eos) = super::util::build_sampling_eos(sampling_params, stop_strings);

        let result = larql_inference::layer_graph::generate_with_sampling(
            weights,
            &model.tokenizer,
            &prompt_ids,
            max_tokens,
            index,
            &*backend,
            &cached_layers,
            0..num_layers,
            sampling,
            &eos,
        );
        tally.add_v2(&result, prompt_ids.len());

        let (completion_text, completion_tokens, finish_reason) =
            finalize_completion(&result.tokens, stop_strings);

        total_completion_tokens += completion_tokens.len();

        let logprobs = logprobs_requested.map(|_| build_completion_logprobs(&completion_tokens));

        let text_out = if echo {
            format!("{prompt}{completion_text}")
        } else {
            completion_text
        };

        choices.push(CompletionChoice {
            text: text_out,
            index: idx,
            finish_reason,
            logprobs,
        });
    }

    Ok((choices, total_prompt_tokens, total_completion_tokens))
}

/// Map per-token `(text, prob)` pairs to OpenAI's legacy completions
/// `logprobs` envelope. `prob` from the inference layer is currently a
/// `1.0` placeholder (per-token softmax not yet exposed), so logprob
/// resolves to `0.0` for every token. `top_logprobs` is an empty map
/// per token until top-K alternatives are surfaced (follow-up).
fn build_completion_logprobs(tokens: &[(String, f64)]) -> CompletionLogprobs {
    use std::collections::BTreeMap;

    let mut text_offset = Vec::with_capacity(tokens.len());
    let mut acc = 0usize;
    for (text, _) in tokens {
        text_offset.push(acc);
        acc += text.len();
    }
    CompletionLogprobs {
        tokens: tokens.iter().map(|(t, _)| t.clone()).collect(),
        token_logprobs: tokens
            .iter()
            .map(|(_, p)| p.max(f64::MIN_POSITIVE).ln())
            .collect(),
        top_logprobs: tokens
            .iter()
            .map(|(t, p)| {
                let mut m: BTreeMap<String, f64> = BTreeMap::new();
                m.insert(t.clone(), p.max(f64::MIN_POSITIVE).ln());
                m
            })
            .collect(),
        text_offset,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn deserialize_single_string_prompt() {
        let json = serde_json::json!({"prompt": "hello"});
        let req: CompletionsRequest = serde_json::from_value(json).unwrap();
        match req.prompt {
            CompletionPrompt::Single(s) => assert_eq!(s, "hello"),
            _ => panic!(),
        }
    }

    #[test]
    fn deserialize_string_array_prompt() {
        let json = serde_json::json!({"prompt": ["a", "b"]});
        let req: CompletionsRequest = serde_json::from_value(json).unwrap();
        match req.prompt {
            CompletionPrompt::Batch(v) => assert_eq!(v, vec!["a", "b"]),
            _ => panic!(),
        }
    }

    fn toks(parts: &[&str]) -> Vec<(String, f64)> {
        parts.iter().map(|s| (s.to_string(), 1.0)).collect()
    }

    #[test]
    fn build_text_completion_chunk_shapes_token_and_final_events() {
        // Per-token chunk: text present, finish_reason null.
        let mid = build_text_completion_chunk("cmpl-1", "synthetic", Some(" Paris"), None);
        let v: serde_json::Value = serde_json::from_str(&mid).unwrap();
        assert_eq!(v["object"], TEXT_COMPLETION_OBJECT);
        assert_eq!(v["model"], "synthetic");
        assert_eq!(v["choices"][0]["text"], " Paris");
        assert!(v["choices"][0]["finish_reason"].is_null());
        assert!(v["choices"][0]["logprobs"].is_null());

        // Final chunk: no text (defaults to ""), finish_reason set.
        let last =
            build_text_completion_chunk("cmpl-1", "synthetic", None, Some(FINISH_REASON_STOP));
        let v: serde_json::Value = serde_json::from_str(&last).unwrap();
        assert_eq!(v["choices"][0]["text"], "");
        assert_eq!(v["choices"][0]["finish_reason"], "stop");
    }

    #[test]
    fn finalize_completion_plain_run_is_length_and_keeps_all_tokens() {
        let (text, kept, reason) = finalize_completion(&toks(&["Par", "is"]), &[]);
        assert_eq!(text, "Paris");
        assert_eq!(kept.len(), 2);
        assert_eq!(reason, "length");
    }

    #[test]
    fn finalize_completion_stops_and_truncates_on_end_of_turn_token() {
        // The `<eos>` marker ends the turn: it's included, anything after is
        // dropped, finish_reason flips to "stop".
        let (text, kept, reason) = finalize_completion(&toks(&["hi", "<eos>", "ignored"]), &[]);
        assert_eq!(text, "hi<eos>");
        assert_eq!(kept.len(), 2);
        assert_eq!(reason, "stop");
    }

    #[test]
    fn finalize_completion_trims_at_stop_string_and_realigns_tokens() {
        // "STOP" is a stop string; text is trimmed at it and the token list
        // is truncated to stay byte-aligned with the trimmed text.
        let (text, kept, reason) =
            finalize_completion(&toks(&["foo", "STOP", "bar"]), &["STOP".to_string()]);
        assert_eq!(text, "foo");
        assert_eq!(reason, "stop");
        // Only the "foo" token survives the byte-boundary retention.
        assert_eq!(kept.len(), 1);
        assert_eq!(kept[0].0, "foo");
    }

    #[test]
    fn finalize_completion_no_stop_match_is_unchanged() {
        let (text, kept, reason) = finalize_completion(&toks(&["a", "b"]), &["zzz".to_string()]);
        assert_eq!(text, "ab");
        assert_eq!(kept.len(), 2);
        assert_eq!(reason, "length");
    }

    #[test]
    fn build_completion_logprobs_aligns_offsets_and_arrays() {
        let toks = vec![("Paris".to_string(), 1.0), (" is".to_string(), 1.0)];
        let lp = build_completion_logprobs(&toks);
        assert_eq!(lp.tokens, vec!["Paris".to_string(), " is".to_string()]);
        assert_eq!(lp.token_logprobs.len(), 2);
        assert_eq!(lp.text_offset, vec![0, 5]);
        assert_eq!(lp.top_logprobs.len(), 2);
        // prob=1.0 → logprob=0.0.
        assert!((lp.token_logprobs[0] - 0.0).abs() < 1e-6);
        // top_logprobs[i] currently contains just the picked token.
        assert!(lp.top_logprobs[0].contains_key("Paris"));
    }
}
