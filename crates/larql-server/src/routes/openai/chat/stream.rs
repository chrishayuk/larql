//! SSE streaming for `POST /v1/chat/completions` — the per-token
//! chunk pump plus the `chat.completion.chunk` builders.

use axum::response::sse::{Event, KeepAlive, Sse};
use futures::stream::Stream;
use std::convert::Infallible;
use std::sync::Arc;
use tokio_stream::wrappers::ReceiverStream;
use tokio_stream::StreamExt as _;

use crate::routes::openai::prompt::{pick_template, render, ASSISTANT_ROLE};
use crate::routes::openai::schema::Schema;
use crate::routes::openai::token_tap::{EmitFailure, TokenTap};
use crate::routes::openai::util::{
    self, error_chunk, new_id_suffix, unix_now, FINISH_REASON_LENGTH, FINISH_REASON_STOP,
    FINISH_REASON_TOOL_CALLS, SSE_CHANNEL_DEPTH, SSE_DONE,
};
use crate::state::LoadedModel;

use super::tools::{build_constrained_mask, build_tool_call_message};
use super::types::{ChatMessage, ToolCall};
use super::CHAT_COMPLETION_CHUNK_OBJECT;

/// SSE stream for `/v1/chat/completions`. First chunk emits
/// `delta: {role: "assistant"}`; subsequent chunks emit
/// `delta: {content: "<token text>"}`; the final chunk has empty
/// `delta` and `finish_reason`. Stream terminates with `data: [DONE]`.
#[allow(clippy::too_many_arguments)]
pub(super) fn stream_chat_completion(
    model: Arc<LoadedModel>,
    messages: Vec<ChatMessage>,
    max_tokens: usize,
    sampling_params: util::SamplingParams,
    stop_strings: Vec<String>,
    constrained_schema: Option<Schema>,
    tools_active: bool,
    model_id: String,
    runtime: Arc<crate::runtime_stats::RuntimeRecorder>,
) -> Sse<impl Stream<Item = Result<Event, Infallible>>> {
    let (tx, rx) = tokio::sync::mpsc::channel::<String>(SSE_CHANNEL_DEPTH);
    let chat_id = format!("chatcmpl-{}", new_id_suffix());
    let call_started = std::time::Instant::now();

    tokio::task::spawn_blocking(move || {
        let _gen_guard = runtime.clone().enter_generation();

        // BitNet (--keep-quant) vindexes take the native-ternary
        // streaming path: skips the dense weights write-lock and runs
        // generate_streaming_bitnet against the pre-loaded
        // BitnetModel at ~1.4 GB resident instead of ~5 GB.
        //
        // Tools and constrained generation are refused rather than
        // silently ignored: both need masked logits over the dense
        // path, and answering a tool request with prose would look
        // like a model that chose not to call the tool.
        if model.is_bitnet() {
            if tools_active || constrained_schema.is_some() {
                let _ = tx.blocking_send(error_chunk(
                    "tools / constrained generation not supported on BitNet \
                     (--keep-quant) models yet",
                ));
                return;
            }
            let bitnet_guard = match model.get_or_load_bitnet() {
                Ok(g) => g,
                Err(e) => {
                    let _ = tx.blocking_send(error_chunk(&e));
                    return;
                }
            };
            let bitnet: &larql_inference::ternary::BitnetModel = &bitnet_guard;
            // `pick_template` needs `&ModelWeights`, which the ternary
            // path deliberately never loads. Resolve the template from
            // the container's declared family instead — the same string
            // `ModelWeights::arch.family()` would have yielded.
            let template = larql_inference::prompt::ChatTemplate::for_family(&model.config.family);
            let prompt = render(template, &messages);
            let encoding = match model.tokenizer.encode(prompt.as_str(), true) {
                Ok(e) => e,
                Err(e) => {
                    let _ = tx.blocking_send(error_chunk(&format!("tokenize: {e}")));
                    return;
                }
            };
            let prompt_ids: Vec<u32> = encoding.get_ids().to_vec();
            if prompt_ids.is_empty() {
                let _ = tx.blocking_send(error_chunk("rendered prompt tokenises to empty"));
                return;
            }

            // Initial role=assistant chunk — OpenAI contract.
            let first = build_chat_chunk(&chat_id, &model_id, Some(ASSISTANT_ROLE), None, None);
            if tx.blocking_send(first).is_err() {
                return;
            }

            let (sampling, eos) = util::build_sampling_eos(sampling_params, &stop_strings);
            // Same `TokenTap` the dense path uses, so stop-string
            // handling and halt semantics are one implementation
            // rather than two that have to agree.
            let tap = std::rc::Rc::new(std::cell::RefCell::new(TokenTap::new(
                &stop_strings,
                EmitFailure::Halt,
            )));
            let chat_id_cb = chat_id.clone();
            let model_id_cb = model_id.clone();
            let tx_cb = tx.clone();
            let tap_cb = std::rc::Rc::clone(&tap);
            let result = larql_inference::ternary::generate_streaming_bitnet(
                bitnet,
                &model.tokenizer,
                &prompt_ids,
                max_tokens,
                sampling,
                &eos,
                move |_id: u32, text: &str, _ms: f64| {
                    tap_cb.borrow_mut().feed(text, |t| {
                        let chunk =
                            build_chat_chunk(&chat_id_cb, &model_id_cb, None, Some(t), None);
                        tx_cb.blocking_send(chunk).is_ok()
                    });
                },
            );

            let emitted = result;
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

            let finish_reason: &'static str = if tap.borrow().halted() || emitted < max_tokens {
                FINISH_REASON_STOP
            } else {
                FINISH_REASON_LENGTH
            };
            let final_chunk =
                build_chat_chunk(&chat_id, &model_id, None, None, Some(finish_reason));
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
        let template = pick_template(weights);
        let prompt = render(template, &messages);
        let encoding = match model.tokenizer.encode(prompt.as_str(), true) {
            Ok(e) => e,
            Err(e) => {
                let _ = tx.blocking_send(error_chunk(&format!("tokenize: {e}")));
                return;
            }
        };
        let prompt_ids: Vec<u32> = encoding.get_ids().to_vec();
        if prompt_ids.is_empty() {
            let _ = tx.blocking_send(error_chunk("rendered prompt tokenises to empty"));
            return;
        }

        // First chunk: role="assistant" delta. OpenAI's chat completion
        // stream contract starts with this, even before any content.
        let first = build_chat_chunk(&chat_id, &model_id, Some(ASSISTANT_ROLE), None, None);
        if tx.blocking_send(first).is_err() {
            return;
        }

        let patched = model.patched.blocking_read();
        let index = patched.base();
        let backend = larql_compute::default_backend();
        let cached_layers = larql_inference::CachedLayerGraph::from_residuals(Vec::new());
        let num_layers = weights.num_layers;

        // Per-token callback used by the unconstrained / json-mode
        // streaming paths: one SSE content-delta chunk per token, with
        // buffering and stop-string halting delegated to `TokenTap`.
        // For `tools_active` runs the tap is buffering-only — the
        // OpenAI tool_calls delta shape only makes sense once the full
        // tool name + arguments JSON is parsed after generation. The
        // tap is shared with the post-loop finish-reason check via
        // Rc<RefCell> — ergonomic single-threaded mutable state, since
        // the whole spawn_blocking body runs on one thread.
        let tap = std::rc::Rc::new(std::cell::RefCell::new(if tools_active {
            TokenTap::buffering_only(&stop_strings)
        } else {
            TokenTap::new(&stop_strings, EmitFailure::Halt)
        }));
        let chat_id_cb = chat_id.clone();
        let model_id_cb = model_id.clone();
        let tx_cb = tx.clone();
        let tap_cb = std::rc::Rc::clone(&tap);
        let on_token = move |_id: u32, text: &str, _prob: f64| {
            tap_cb.borrow_mut().feed(text, |t| {
                let chunk = build_chat_chunk(&chat_id_cb, &model_id_cb, None, Some(t), None);
                tx_cb.blocking_send(chunk).is_ok()
            });
        };

        let result = if let Some(schema) = constrained_schema {
            // Sampling under mask: temperature/top_p/seed/penalties drive
            // selection over the masked logits, falling back to greedy
            // when the request didn't set them.
            let (sampling, eos) = util::build_sampling_eos(sampling_params, &stop_strings);
            let mask = build_constrained_mask(&model.tokenizer, schema);
            larql_inference::layer_graph::generate_constrained_streaming_sampled(
                weights,
                &model.tokenizer,
                &prompt_ids,
                max_tokens,
                index,
                &*backend,
                &cached_layers,
                0..num_layers,
                mask,
                on_token,
                sampling,
                &eos,
            )
        } else {
            let (sampling, eos) = util::build_sampling_eos(sampling_params, &stop_strings);
            larql_inference::layer_graph::generate_streaming(
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
                on_token,
                None,
            )
        };

        let mut tally = crate::runtime_stats::GenerationTally::new();
        tally.add_v2(&result, prompt_ids.len());
        runtime.record(tally.into_sample(crate::state::elapsed_ms(call_started)));

        // Final-chunk finish reason: layer_graph::generate halts on
        // EOS internally; tokens.len() < max_tokens implies stop.
        let finish_reason: &'static str = if tools_active {
            FINISH_REASON_TOOL_CALLS
        } else if tap.borrow().halted() || result.tokens.len() < max_tokens {
            FINISH_REASON_STOP
        } else {
            FINISH_REASON_LENGTH
        };

        // Tool-call delta: parse the buffered constrained output once
        // generation finishes and emit a single chunk carrying the
        // full `tool_calls[0]` payload. Per-token argument streaming
        // is a tightening that lives in a follow-up — most OpenAI
        // clients accumulate `tool_calls[i].function.arguments`
        // incrementally and trigger only on `finish_reason: "tool_calls"`,
        // so a single fat chunk is wire-compatible.
        if tools_active {
            let buffered = tap.borrow().text().to_string();
            match build_tool_call_message(&buffered) {
                Ok(msg) => {
                    if let Some(calls) = msg.tool_calls.as_ref() {
                        let chunk = build_chat_tool_calls_chunk(&chat_id, &model_id, calls);
                        let _ = tx.blocking_send(chunk);
                    }
                }
                Err(e) => {
                    let _ = tx.blocking_send(error_chunk(&format!(
                        "tool_call output failed to parse: {e}"
                    )));
                }
            }
        }

        let final_chunk = build_chat_chunk(&chat_id, &model_id, None, None, Some(finish_reason));
        let _ = tx.blocking_send(final_chunk);
    });

    let stream = ReceiverStream::new(rx)
        .map(|data| Event::default().data(data))
        .chain(tokio_stream::once(Event::default().data(SSE_DONE)))
        .map(Ok::<_, Infallible>);

    Sse::new(stream).keep_alive(KeepAlive::default())
}

pub(super) fn build_chat_chunk(
    id: &str,
    model: &str,
    role: Option<&str>,
    content: Option<&str>,
    finish_reason: Option<&'static str>,
) -> String {
    let mut delta = serde_json::Map::new();
    if let Some(r) = role {
        delta.insert("role".into(), serde_json::Value::String(r.to_string()));
    }
    if let Some(c) = content {
        delta.insert("content".into(), serde_json::Value::String(c.to_string()));
    }
    let chunk = serde_json::json!({
        "id": id,
        "object": CHAT_COMPLETION_CHUNK_OBJECT,
        "created": unix_now(),
        "model": model,
        "choices": [{
            "index": 0,
            "delta": serde_json::Value::Object(delta),
            "finish_reason": match finish_reason {
                Some(r) => serde_json::Value::String(r.to_string()),
                None => serde_json::Value::Null,
            },
            "logprobs": serde_json::Value::Null,
        }]
    });
    chunk.to_string()
}

/// Build a streaming chunk that carries the full `tool_calls` payload
/// in the delta. Each call gets an `index` field per OpenAI's chunk
/// shape (so clients can demux multiple parallel tool calls); we emit
/// the entire `name` + `arguments` in one chunk rather than splitting
/// arguments per-token (a follow-up tightening).
pub(super) fn build_chat_tool_calls_chunk(id: &str, model: &str, calls: &[ToolCall]) -> String {
    let tool_calls_json: Vec<serde_json::Value> = calls
        .iter()
        .enumerate()
        .map(|(i, c)| {
            serde_json::json!({
                "index": i,
                "id": c.id,
                "type": c.kind,
                "function": {
                    "name": c.function.name,
                    "arguments": c.function.arguments,
                },
            })
        })
        .collect();
    serde_json::json!({
        "id": id,
        "object": CHAT_COMPLETION_CHUNK_OBJECT,
        "created": unix_now(),
        "model": model,
        "choices": [{
            "index": 0,
            "delta": {"tool_calls": tool_calls_json},
            "finish_reason": serde_json::Value::Null,
            "logprobs": serde_json::Value::Null,
        }]
    })
    .to_string()
}
