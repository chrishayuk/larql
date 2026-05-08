//! `POST /v1/chat/completions` — OpenAI-compatible chat completions (N0.1, slice 2).
//!
//! Implements the [OpenAI Chat Completions API](https://platform.openai.com/docs/api-reference/chat/create)
//! shape so existing `openai` SDKs work unmodified:
//!
//! ```python
//! from openai import OpenAI
//! client = OpenAI(base_url="http://larql:8080/v1", api_key="sk-...")
//! resp = client.chat.completions.create(
//!     model="gemma-3-4b",
//!     messages=[
//!         {"role": "system", "content": "You are a helpful assistant."},
//!         {"role": "user",   "content": "What is the capital of France?"},
//!     ],
//!     max_tokens=20,
//! )
//! ```
//!
//! ## Chat template handling
//!
//! `messages` is rendered to a single prompt via the model's chat
//! template (Gemma / Llama / ChatML / Mistral / plain), detected from
//! the model's `family` and `id`. The rendered prompt then runs through
//! the same generation loop as `/v1/completions`.
//!
//! Template detection precedence:
//! 1. `arch.family()` (authoritative when available)
//! 2. Substring match on `model.id` ("gemma", "llama", "qwen", …)
//! 3. Plain (fallback for unknown families and base models)
//!
//! ## Generation path
//!
//! Buffered + SSE streaming both call
//! `larql_inference::layer_graph::generate{,_streaming}` which is KV-
//! cached on f16 vindexes (and falls back to a per-step Q4_K decode
//! when the backend is CPU + Q4K). Generation acquires an exclusive
//! write guard on `LoadedModel.weights` for the duration; concurrent
//! reads block but other endpoints are unaffected in steady state.
//!
//! ## Slice 2-3 limitations
//!
//! - `tools` / `tool_choice` returns 400 (slice 4 = N0.6 constrained decoding)
//! - `response_format: json_object | json_schema` returns 400 (slice 4)
//! - `n>1` returns 400
//! - `logprobs` request field accepted, response field always `null` (F18)

use axum::extract::State;
use axum::response::sse::{Event, KeepAlive, Sse};
use axum::response::{IntoResponse, Response};
use axum::Json;
use futures::stream::Stream;
use serde::{Deserialize, Serialize};
use std::convert::Infallible;
use std::sync::Arc;
use tokio_stream::wrappers::ReceiverStream;
use tokio_stream::StreamExt as _;

use crate::error::ServerError;
use crate::state::{AppState, LoadedModel};

use super::schema::{ObjectSchema, Schema};
use super::util::{contains_any, error_chunk, new_id_suffix, trim_at_stop, unix_now, StopSpec};

const CHAT_COMPLETION_OBJECT: &str = "chat.completion";
const CHAT_COMPLETION_CHUNK_OBJECT: &str = "chat.completion.chunk";
const ASSISTANT_ROLE: &str = "assistant";
const SYSTEM_ROLE: &str = "system";
const USER_ROLE: &str = "user";
const TOOL_ROLE: &str = "tool";
const DEFAULT_MAX_TOKENS: usize = 256;

#[derive(Deserialize)]
pub struct ChatMessage {
    pub role: String,
    /// Free-text content. Optional because assistant messages that
    /// emitted tool_calls send `content: null` per OpenAI's wire shape.
    #[serde(default)]
    pub content: Option<String>,
    /// Echoed back on `role: "assistant"` messages in multi-turn
    /// conversations so the model can see its own prior tool dispatch.
    #[serde(default)]
    pub tool_calls: Option<serde_json::Value>,
    /// Set on `role: "tool"` messages — the call id this result
    /// corresponds to.
    #[serde(default)]
    pub tool_call_id: Option<String>,
    /// Optional `function.name` echoed on tool messages by some clients.
    /// Treated as informational; we already get the name from the
    /// matching `tool_calls[i].function.name` when available.
    #[serde(default)]
    pub name: Option<String>,
}

#[derive(Deserialize)]
pub struct ChatCompletionsRequest {
    pub model: Option<String>,
    pub messages: Vec<ChatMessage>,
    #[serde(default)]
    pub max_tokens: Option<usize>,
    #[serde(default)]
    pub temperature: Option<f32>,
    /// Nucleus (top-p) filter applied after temperature scaling. Only
    /// honoured when `temperature > 0`; for greedy decoding it's a no-op.
    #[serde(default)]
    pub top_p: Option<f32>,
    /// Streaming via SSE — emits one `chat.completion.chunk` per token,
    /// terminated by `data: [DONE]\n\n`.
    #[serde(default)]
    pub stream: Option<bool>,
    /// Number of completions per prompt — only n=1 supported.
    #[serde(default)]
    pub n: Option<usize>,
    /// Stop strings — first match halts generation.
    #[serde(default)]
    pub stop: Option<StopSpec>,
    /// Top-k log-probs — request accepted, response field always null.
    #[serde(default)]
    pub logprobs: Option<bool>,
    /// Newer log-probs field used by recent SDKs — same handling as `logprobs`.
    #[serde(default)]
    pub top_logprobs: Option<usize>,
    /// Tool definitions — slice 4 (N0.6 constrained decoding); 400 if non-empty.
    #[serde(default)]
    pub tools: Option<serde_json::Value>,
    /// Tool choice — same as `tools` (slice 4).
    #[serde(default)]
    pub tool_choice: Option<serde_json::Value>,
    /// Response format (`{type: "json_object" | "json_schema", ...}`) —
    /// slice 4. Returns 400 for any non-text response_format.
    #[serde(default)]
    pub response_format: Option<serde_json::Value>,
    /// Seed for reproducible sampling. Same seed + same temperature +
    /// same prompt produces the same tokens. No-op for greedy mode
    /// (greedy is already deterministic on argmax).
    #[serde(default)]
    pub seed: Option<u64>,
    /// End-user id — logged via tracing if set.
    #[serde(default)]
    pub user: Option<String>,
    /// Frequency / presence penalties — accepted for shape compat;
    /// the sampler does not yet apply repetition penalties (F19).
    #[serde(default)]
    pub frequency_penalty: Option<f32>,
    #[serde(default)]
    pub presence_penalty: Option<f32>,
}

#[derive(Serialize)]
pub struct ChatChoiceMessage {
    pub role: &'static str,
    /// Always present, but `null` when the assistant emitted tool_calls
    /// rather than free text. Serialised as `content: null` in that case
    /// (OpenAI's contract).
    pub content: Option<String>,
    /// One or more tool calls produced by constrained decoding when
    /// `tools` was on the request. Omitted entirely for plain text
    /// completions so non-tools responses stay shape-clean.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tool_calls: Option<Vec<ToolCall>>,
}

/// OpenAI's tool-call shape on the response side: `id`, `type`,
/// `function: {name, arguments}`. `arguments` is JSON-stringified.
#[derive(Serialize)]
pub struct ToolCall {
    pub id: String,
    #[serde(rename = "type")]
    pub kind: &'static str,
    pub function: ToolCallFunction,
}

#[derive(Serialize)]
pub struct ToolCallFunction {
    pub name: String,
    /// JSON-encoded string, not a nested object — preserves the wire
    /// shape SDKs expect.
    pub arguments: String,
}

#[derive(Serialize)]
pub struct ChatChoice {
    pub index: usize,
    pub message: ChatChoiceMessage,
    pub finish_reason: &'static str,
    /// Populated when the request set `logprobs: true`. `None`
    /// (serialised as `null`) otherwise — the OpenAI default.
    pub logprobs: Option<ChatLogprobs>,
}

/// `choices[i].logprobs` payload for chat completions. Mirrors
/// OpenAI's `{content: [{token, logprob, bytes, top_logprobs}]}`.
#[derive(Serialize)]
pub struct ChatLogprobs {
    pub content: Vec<TokenLogprob>,
}

/// One per-token entry in a logprobs payload (chat or completions —
/// the chat shape is identical for the inner item).
///
/// `top_logprobs` is an empty array until the inference layer exposes
/// per-step top-K alternatives (follow-up). Until then we still emit
/// the picked-token entry so client parsers don't break on the field.
#[derive(Serialize)]
pub struct TokenLogprob {
    pub token: String,
    pub logprob: f64,
    pub bytes: Vec<u8>,
    pub top_logprobs: Vec<TokenLogprob>,
}

#[derive(Serialize)]
pub struct ChatUsage {
    pub prompt_tokens: usize,
    pub completion_tokens: usize,
    pub total_tokens: usize,
}

#[derive(Serialize)]
pub struct ChatCompletionsResponse {
    pub id: String,
    pub object: &'static str,
    pub created: u64,
    pub model: String,
    pub choices: Vec<ChatChoice>,
    pub usage: ChatUsage,
}

#[utoipa::path(
    post,
    path = "/v1/chat/completions",
    tag = "openai",
    request_body = crate::openapi::schemas::OpenAiChatRequest,
    responses(
        (status = 200, description = "Non-streaming JSON response.",
         body = crate::openapi::schemas::OpenAiChatResponse),
        (status = 200, description = "SSE stream when `stream: true`. Each event is `data: <ChatCompletionChunk JSON>\\n\\n`, terminated by `data: [DONE]`.",
         content_type = "text/event-stream", body = String),
        (status = 400, body = crate::error::ErrorBody),
        (status = 500, body = crate::error::ErrorBody),
    ),
)]
pub async fn handle_chat_completions(
    State(state): State<Arc<AppState>>,
    Json(req): Json<ChatCompletionsRequest>,
) -> Result<Response, ServerError> {
    state.bump_requests();

    if req.n.unwrap_or(1) > 1 {
        return Err(ServerError::BadRequest(
            "n>1 not yet supported; only n=1 (single completion per prompt)".into(),
        ));
    }
    // Tools take precedence over response_format. If tools are
    // present and not disabled by `tool_choice="none"`, the model is
    // constrained to emit JSON matching one of the supplied function
    // schemas; the response is then reshaped into `tool_calls`.
    let (constrained_schema, tools_active) = match resolve_tools(&req)? {
        Some(schema) => (Some(schema), true),
        None => (
            schema_for_response_format(req.response_format.as_ref())?,
            false,
        ),
    };

    let model = state.model_or_err(req.model.as_deref())?;
    if model.infer_disabled {
        return Err(ServerError::InferenceUnavailable(
            "inference disabled (--no-infer / --embed-only / --ffn-only)".into(),
        ));
    }
    if req.messages.is_empty() {
        return Err(ServerError::BadRequest("messages is empty".into()));
    }
    for (i, m) in req.messages.iter().enumerate() {
        if !matches!(
            m.role.as_str(),
            USER_ROLE | ASSISTANT_ROLE | SYSTEM_ROLE | TOOL_ROLE
        ) {
            return Err(ServerError::BadRequest(format!(
                "messages[{i}].role must be 'user' | 'assistant' | 'system' | 'tool' (got {:?})",
                m.role
            )));
        }
        // Per-role shape validation — only enforce constraints OpenAI
        // clients can violate; missing-content + tool_calls is normal
        // for assistant turns, missing tool_call_id is an error on
        // tool turns.
        match m.role.as_str() {
            TOOL_ROLE => {
                if m.tool_call_id.is_none() {
                    return Err(ServerError::BadRequest(format!(
                        "messages[{i}] role=tool requires tool_call_id"
                    )));
                }
                if m.content.is_none() {
                    return Err(ServerError::BadRequest(format!(
                        "messages[{i}] role=tool requires content"
                    )));
                }
            }
            ASSISTANT_ROLE => {
                let has_tool_calls = m
                    .tool_calls
                    .as_ref()
                    .is_some_and(|v| !v.is_null() && !is_empty_json_array(v));
                if !has_tool_calls && m.content.is_none() {
                    return Err(ServerError::BadRequest(format!(
                        "messages[{i}] role=assistant requires content (or tool_calls)"
                    )));
                }
            }
            USER_ROLE | SYSTEM_ROLE => {
                if m.content.is_none() {
                    return Err(ServerError::BadRequest(format!(
                        "messages[{i}] role={} requires content",
                        m.role
                    )));
                }
            }
            _ => {}
        }
    }

    let max_tokens = req.max_tokens.unwrap_or(DEFAULT_MAX_TOKENS);
    let stop_strings: Vec<String> = req
        .stop
        .as_ref()
        .map(|s| s.as_slice().to_vec())
        .unwrap_or_default();
    let sampling_params = super::util::SamplingParams {
        temperature: req.temperature,
        top_p: req.top_p,
        seed: req.seed,
        frequency_penalty: req.frequency_penalty,
        presence_penalty: req.presence_penalty,
    };
    let model_id = req.model.clone().unwrap_or_else(|| model.id.clone());
    let model_arc = model.clone();
    let messages = req.messages;

    if req.stream.unwrap_or(false) {
        return Ok(stream_chat_completion(
            model_arc,
            messages,
            max_tokens,
            sampling_params,
            stop_strings,
            constrained_schema,
            tools_active,
            model_id,
        )
        .into_response());
    }

    let logprobs_requested = req.logprobs.unwrap_or(false);
    let output = tokio::task::spawn_blocking(move || -> Result<_, ServerError> {
        run_chat_completion(
            &model_arc,
            &messages,
            max_tokens,
            sampling_params,
            &stop_strings,
            constrained_schema,
        )
    })
    .await
    .map_err(|e| ServerError::Internal(e.to_string()))??;

    let logprobs = if logprobs_requested && !tools_active {
        Some(build_chat_logprobs(&output.tokens))
    } else {
        None
    };

    let (message, finish_reason) = if tools_active {
        match build_tool_call_message(&output.text) {
            Ok(m) => (m, "tool_calls"),
            Err(e) => {
                return Err(ServerError::Internal(format!(
                    "tool_call output failed to parse: {e}; raw: {:?}",
                    output.text
                )));
            }
        }
    } else {
        (
            ChatChoiceMessage {
                role: ASSISTANT_ROLE,
                content: Some(output.text),
                tool_calls: None,
            },
            output.finish_reason,
        )
    };

    Ok(Json(ChatCompletionsResponse {
        id: format!("chatcmpl-{}", new_id_suffix()),
        object: CHAT_COMPLETION_OBJECT,
        created: unix_now(),
        model: model_id,
        choices: vec![ChatChoice {
            index: 0,
            message,
            finish_reason,
            logprobs,
        }],
        usage: ChatUsage {
            prompt_tokens: output.prompt_tokens,
            completion_tokens: output.completion_tokens,
            total_tokens: output.prompt_tokens + output.completion_tokens,
        },
    })
    .into_response())
}

/// Map per-token `(text, prob)` pairs to OpenAI's `ChatLogprobs`
/// envelope. `prob` is currently `1.0` placeholder from the inference
/// layer until per-token softmax is exposed; logprob then becomes
/// `0.0` for every token. `top_logprobs` is empty until top-K
/// alternatives are surfaced in a follow-up.
fn build_chat_logprobs(tokens: &[(String, f64)]) -> ChatLogprobs {
    ChatLogprobs {
        content: tokens
            .iter()
            .map(|(text, prob)| TokenLogprob {
                token: text.clone(),
                logprob: prob.max(f64::MIN_POSITIVE).ln(),
                bytes: text.as_bytes().to_vec(),
                top_logprobs: Vec::new(),
            })
            .collect(),
    }
}

/// SSE stream for `/v1/chat/completions`. First chunk emits
/// `delta: {role: "assistant"}`; subsequent chunks emit
/// `delta: {content: "<token text>"}`; the final chunk has empty
/// `delta` and `finish_reason`. Stream terminates with `data: [DONE]`.
#[allow(clippy::too_many_arguments)]
fn stream_chat_completion(
    model: Arc<LoadedModel>,
    messages: Vec<ChatMessage>,
    max_tokens: usize,
    sampling_params: super::util::SamplingParams,
    stop_strings: Vec<String>,
    constrained_schema: Option<Schema>,
    tools_active: bool,
    model_id: String,
) -> Sse<impl Stream<Item = Result<Event, Infallible>>> {
    let (tx, rx) = tokio::sync::mpsc::channel::<String>(64);
    let chat_id = format!("chatcmpl-{}", new_id_suffix());

    tokio::task::spawn_blocking(move || {
        let mut weights_guard = match model.lock_weights_for_gen() {
            Ok(w) => w,
            Err(e) => {
                let _ = tx.blocking_send(error_chunk(&e));
                return;
            }
        };
        let weights: &mut larql_inference::ModelWeights = &mut weights_guard;
        let template = pick_template(&model);
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
        // streaming paths. Pushes one SSE content-delta chunk per token
        // and tracks completion text so client-supplied stop strings
        // can halt early. For `tools_active` runs the callback runs in
        // *buffer* mode — it accumulates text without emitting chunks,
        // because the OpenAI tool_calls delta shape only makes sense
        // once the full tool name + arguments JSON is parsed.
        // `early_stop` is shared with the post-loop finish-reason check
        // via Rc<Cell<bool>> — ergonomic single-threaded mutable state,
        // since the whole spawn_blocking body runs on one thread.
        let chat_id_cb = chat_id.clone();
        let model_id_cb = model_id.clone();
        let tx_cb = tx.clone();
        let stop_strings_cb = stop_strings.clone();
        let early_stop = std::rc::Rc::new(std::cell::Cell::new(false));
        let early_stop_cb = early_stop.clone();
        let buffered_text = std::rc::Rc::new(std::cell::RefCell::new(String::new()));
        let buffered_text_cb = buffered_text.clone();
        let on_token = move |_id: u32, text: &str, _prob: f64| {
            if early_stop_cb.get() {
                return;
            }
            // Always buffer; tools_active reads from `buffered_text`
            // after generation, content streaming reads token-by-token.
            buffered_text_cb.borrow_mut().push_str(text);
            if !tools_active {
                let chunk = build_chat_chunk(&chat_id_cb, &model_id_cb, None, Some(text), None);
                if tx_cb.blocking_send(chunk).is_err() {
                    early_stop_cb.set(true);
                    return;
                }
            }
            if !stop_strings_cb.is_empty()
                && contains_any(&buffered_text_cb.borrow(), &stop_strings_cb)
            {
                early_stop_cb.set(true);
            }
        };

        let result = if let Some(schema) = constrained_schema {
            // Sampling under mask: temperature/top_p/seed/penalties drive
            // selection over the masked logits, falling back to greedy
            // when the request didn't set them.
            let (sampling, eos) = super::util::build_sampling_eos(sampling_params, &stop_strings);
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
            let (sampling, eos) = super::util::build_sampling_eos(sampling_params, &stop_strings);
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
            )
        };

        // Final-chunk finish reason: layer_graph::generate halts on
        // EOS internally; tokens.len() < max_tokens implies stop.
        let finish_reason: &'static str = if tools_active {
            "tool_calls"
        } else if early_stop.get() || result.tokens.len() < max_tokens {
            "stop"
        } else {
            "length"
        };

        // Tool-call delta: parse the buffered constrained output once
        // generation finishes and emit a single chunk carrying the
        // full `tool_calls[0]` payload. Per-token argument streaming
        // is a tightening that lives in a follow-up — most OpenAI
        // clients accumulate `tool_calls[i].function.arguments`
        // incrementally and trigger only on `finish_reason: "tool_calls"`,
        // so a single fat chunk is wire-compatible.
        if tools_active {
            let buffered = buffered_text.borrow().clone();
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
        .chain(tokio_stream::once(Event::default().data("[DONE]")))
        .map(Ok::<_, Infallible>);

    Sse::new(stream).keep_alive(KeepAlive::default())
}

fn build_chat_chunk(
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
fn build_chat_tool_calls_chunk(id: &str, model: &str, calls: &[ToolCall]) -> String {
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

/// Render `messages` to a single prompt, then run the generation loop.
/// Returns `(text, finish_reason, prompt_tokens, completion_tokens)`.
///
/// Branches on `constrained_schema`:
/// - `None` → sampling path (`generate_with_sampling`).
/// - `Some(schema)` → grammar-mask path (`generate_constrained`).
///   Sampling fields (temperature/top_p/seed) are accepted but ignored
///   in this slice — constrained decoding is greedy by design so JSON /
///   structured output is deterministic.
#[allow(clippy::too_many_arguments)]
fn run_chat_completion(
    model: &LoadedModel,
    messages: &[ChatMessage],
    max_tokens: usize,
    sampling_params: super::util::SamplingParams,
    stop_strings: &[String],
    constrained_schema: Option<Schema>,
) -> Result<ChatGenerationOutput, ServerError> {
    // Take an exclusive write guard on the weights for the duration
    // of generation. `larql_inference::layer_graph::generate` mutates
    // `weights.tensors` (the per-layer Q4_K dequant cache), so other
    // read paths block while one chat completion runs.
    let mut weights_guard = model
        .lock_weights_for_gen()
        .map_err(ServerError::InferenceUnavailable)?;
    let weights: &mut larql_inference::ModelWeights = &mut weights_guard;

    let template = pick_template(model);
    let prompt = render(template, messages);

    let encoding = model
        .tokenizer
        .encode(prompt.as_str(), true)
        .map_err(|e| ServerError::Internal(format!("tokenize: {e}")))?;
    let prompt_ids: Vec<u32> = encoding.get_ids().to_vec();
    if prompt_ids.is_empty() {
        return Err(ServerError::BadRequest(
            "rendered prompt tokenises to empty".into(),
        ));
    }
    let prompt_token_count = prompt_ids.len();

    let patched = model.patched.blocking_read();
    let index = patched.base();
    let backend = larql_compute::default_backend();
    let cached_layers = larql_inference::CachedLayerGraph::from_residuals(Vec::new());
    let num_layers = weights.num_layers;

    let result = if let Some(schema) = constrained_schema {
        // Sampling under mask via the new `_sampled` variant — drives
        // selection through the user's SamplingConfig over the masked
        // logits. Greedy when no sampling fields are set.
        let (sampling, eos) = super::util::build_sampling_eos(sampling_params, stop_strings);
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
            |_, _, _| {}, // buffered path: no per-token callback
            sampling,
            &eos,
        )
    } else {
        let (sampling, eos) = super::util::build_sampling_eos(sampling_params, stop_strings);
        larql_inference::layer_graph::generate_with_sampling(
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
        )
    };

    let mut completion_text = String::new();
    let mut completion_tokens: Vec<(String, f64)> = Vec::new();
    let mut finish_reason: &'static str = "length";
    for (text, prob) in &result.tokens {
        completion_text.push_str(text);
        completion_tokens.push((text.clone(), *prob));
        if larql_inference::vindex::is_end_of_turn(text) {
            finish_reason = "stop";
            break;
        }
    }
    if !stop_strings.is_empty() && contains_any(&completion_text, stop_strings) {
        completion_text = trim_at_stop(&completion_text, stop_strings);
        finish_reason = "stop";
        // Also trim the per-token list to the same length so logprobs
        // align with the truncated text. We can't perfectly reverse the
        // textual trim, but discarding tokens past the byte boundary is
        // a good approximation.
        completion_tokens = trim_tokens_to_text(&completion_tokens, &completion_text);
    }

    let completion_token_count = completion_tokens.len();
    Ok(ChatGenerationOutput {
        text: completion_text,
        tokens: completion_tokens,
        finish_reason,
        prompt_tokens: prompt_token_count,
        completion_tokens: completion_token_count,
    })
}

/// Output of [`run_chat_completion`]. Carries per-token info so the
/// handler can emit logprobs without re-running generation.
struct ChatGenerationOutput {
    text: String,
    tokens: Vec<(String, f64)>,
    finish_reason: &'static str,
    prompt_tokens: usize,
    completion_tokens: usize,
}

/// Truncate `tokens` so concatenated surface forms cover at most the
/// byte length of `truncated_text`. Used after `trim_at_stop` chops
/// the joined string to keep `tokens.len()` matching `text.len()`.
fn trim_tokens_to_text(tokens: &[(String, f64)], truncated_text: &str) -> Vec<(String, f64)> {
    let target_len = truncated_text.len();
    let mut acc = 0usize;
    let mut out = Vec::with_capacity(tokens.len());
    for (t, p) in tokens {
        if acc >= target_len {
            break;
        }
        acc += t.len();
        out.push((t.clone(), *p));
    }
    out
}

// ── Template selection ───────────────────────────────────────────────────────
//
// The multi-turn rendering itself lives in
// `larql_inference::prompt::ChatTemplate::render_messages`. This handler
// only needs to pick the right template variant for the loaded model.

fn pick_template(model: &LoadedModel) -> larql_inference::prompt::ChatTemplate {
    use larql_inference::prompt::ChatTemplate;
    // Prefer the architecture's family signal when weights are loaded;
    // fall back to id heuristics when weights haven't been touched yet.
    if let Some(cell) = model.weights.get() {
        if let Ok(weights) = cell.read() {
            return ChatTemplate::for_family(weights.arch.family());
        }
    }
    ChatTemplate::for_model_id(&model.id)
}

/// Adapter: convert our wire `ChatMessage` list to the `(role, content)`
/// shape `ChatTemplate::render_messages` accepts. The chat templates
/// natively handle `system` / `user` / `assistant` only, so tool turns
/// are flattened into text content that fits within those slots:
///
/// - Assistant message with `tool_calls` (and `content: null`) →
///   assistant turn whose content is a serialised summary of the tool
///   calls (`Tool call: <name>(<arguments>)`). Any prior `content`
///   takes precedence when both are set.
/// - Tool message → user turn with `[Tool result for <id>: <content>]`,
///   so the model sees the result inline before generating the next
///   assistant turn.
fn render(template: larql_inference::prompt::ChatTemplate, messages: &[ChatMessage]) -> String {
    let pairs: Vec<(String, String)> = messages
        .iter()
        .map(|m| match m.role.as_str() {
            TOOL_ROLE => (
                USER_ROLE.to_string(),
                format_tool_result(m.tool_call_id.as_deref(), m.content.as_deref()),
            ),
            ASSISTANT_ROLE => {
                if let Some(c) = m.content.as_deref() {
                    (ASSISTANT_ROLE.to_string(), c.to_string())
                } else if let Some(tc) = m.tool_calls.as_ref() {
                    (ASSISTANT_ROLE.to_string(), format_tool_calls(tc))
                } else {
                    (ASSISTANT_ROLE.to_string(), String::new())
                }
            }
            other => (other.to_string(), m.content.clone().unwrap_or_default()),
        })
        .collect();
    template.render_messages(pairs.iter().map(|(r, c)| (r.as_str(), c.as_str())))
}

/// Render a tool-result message as a user-side text turn so the model
/// sees the tool output before the next assistant generation.
fn format_tool_result(tool_call_id: Option<&str>, content: Option<&str>) -> String {
    let id = tool_call_id.unwrap_or("?");
    let body = content.unwrap_or("");
    format!("[Tool result for {id}]: {body}")
}

/// Render an assistant `tool_calls` echo as text. Multiple parallel
/// tool calls are listed; arguments stay JSON-encoded.
fn format_tool_calls(tool_calls: &serde_json::Value) -> String {
    let arr = match tool_calls.as_array() {
        Some(a) => a,
        None => return String::new(),
    };
    let mut out = String::new();
    for (i, tc) in arr.iter().enumerate() {
        if i > 0 {
            out.push('\n');
        }
        let name = tc
            .get("function")
            .and_then(|f| f.get("name"))
            .and_then(|n| n.as_str())
            .unwrap_or("?");
        let args = tc
            .get("function")
            .and_then(|f| f.get("arguments"))
            .and_then(|a| a.as_str())
            .unwrap_or("");
        out.push_str(&format!("[Tool call: {name}({args})]"));
    }
    out
}

// ── chat-only request validation helper ─────────────────────────────────────

fn is_empty_json_array(v: &serde_json::Value) -> bool {
    v.as_array().map(|a| a.is_empty()).unwrap_or(false)
}

/// Resolve `tools` + `tool_choice` into a synthesised `Schema`.
///
/// Returns `Ok(None)` when no tools are bound (or `tool_choice="none"`)
/// so the caller falls through to `response_format` /unconstrained.
/// Returns `Ok(Some(schema))` with the discriminated-union shape over
/// each function (one branch per tool); the chat handler then post-
/// parses the JSON output into `tool_calls`.
fn resolve_tools(req: &ChatCompletionsRequest) -> Result<Option<Schema>, ServerError> {
    use super::schema::{resolve_tool_choice, synth_tools_schema};

    let tools_present = req
        .tools
        .as_ref()
        .is_some_and(|v| !v.is_null() && !is_empty_json_array(v));

    let tool_names: Vec<String> = req
        .tools
        .as_ref()
        .and_then(|v| v.as_array())
        .map(|arr| {
            arr.iter()
                .filter_map(|t| {
                    t.get("function")
                        .and_then(|f| f.get("name"))
                        .and_then(|n| n.as_str())
                        .map(|s| s.to_string())
                })
                .collect()
        })
        .unwrap_or_default();

    let mode = resolve_tool_choice(tools_present, req.tool_choice.as_ref(), &tool_names)
        .map_err(ServerError::BadRequest)?;

    if !tools_present || matches!(mode, super::schema::ToolMode::None) {
        return Ok(None);
    }

    let tools = req
        .tools
        .as_ref()
        .expect("tools_present checked above")
        .clone();
    let result = synth_tools_schema(&tools, &mode).map_err(ServerError::BadRequest)?;
    Ok(result.map(|(schema, _names)| schema))
}

/// Parse a constrained-decoder output back into a `ChatChoiceMessage`
/// with `tool_calls` populated. Constrained decoding guarantees a
/// well-formed JSON object, but we still tolerate incidental leading
/// or trailing whitespace.
fn build_tool_call_message(text: &str) -> Result<ChatChoiceMessage, String> {
    let trimmed = text.trim();
    let (start, end) = trimmed
        .find('{')
        .and_then(|s| trimmed.rfind('}').map(|e| (s, e + 1)))
        .ok_or_else(|| "no `{...}` JSON object in tool output".to_string())?;
    let json_slice = &trimmed[start..end];
    let parsed: serde_json::Value =
        serde_json::from_str(json_slice).map_err(|e| format!("invalid JSON: {e}"))?;
    let name = parsed
        .get("name")
        .and_then(|n| n.as_str())
        .ok_or_else(|| "tool output missing `name`".to_string())?
        .to_string();
    let arguments_value = parsed
        .get("arguments")
        .ok_or_else(|| "tool output missing `arguments`".to_string())?;
    // OpenAI sends arguments as a JSON-stringified object — reserialise
    // to canonical compact form so SDKs `json.loads` cleanly.
    let arguments = serde_json::to_string(arguments_value)
        .map_err(|e| format!("failed to serialise arguments: {e}"))?;
    Ok(ChatChoiceMessage {
        role: ASSISTANT_ROLE,
        content: None,
        tool_calls: Some(vec![ToolCall {
            id: format!("call_{}", new_id_suffix()),
            kind: "function",
            function: ToolCallFunction { name, arguments },
        }]),
    })
}

/// Map an OpenAI `response_format` field to the `Schema` the FSM
/// should enforce. `None` (or `{type: "text"}`) means "no constrained
/// decoding" — fall through to the sampling path.
///
/// `json_object` compiles to `Schema::Object(any)`. `json_schema`
/// reaches into `json_schema.schema` and runs the JSON Schema parser
/// with `strict: true` when the `strict` field is set (matching
/// OpenAI's structured-outputs contract).
fn schema_for_response_format(
    rf: Option<&serde_json::Value>,
) -> Result<Option<Schema>, ServerError> {
    let Some(rf) = rf else {
        return Ok(None);
    };
    let kind = rf.get("type").and_then(|t| t.as_str()).unwrap_or("text");
    match kind {
        "text" => Ok(None),
        "json_object" => Ok(Some(Schema::object(ObjectSchema::any()))),
        "json_schema" => {
            let js = rf.get("json_schema").ok_or_else(|| {
                ServerError::BadRequest(
                    "response_format.type=json_schema requires a json_schema field".into(),
                )
            })?;
            let schema_value = js.get("schema").ok_or_else(|| {
                ServerError::BadRequest("response_format.json_schema.schema is required".into())
            })?;
            // OpenAI's `strict: true` flips the additionalProperties default
            // to false. Default is `false` here so non-strict callers can
            // still send extra keys.
            let strict = js.get("strict").and_then(|v| v.as_bool()).unwrap_or(false);
            let opts = super::schema::ParseOptions { strict };
            let parsed = super::schema::parse_schema_with(schema_value, opts)
                .map_err(|e| ServerError::BadRequest(format!("invalid json_schema: {e}")))?;
            Ok(Some(parsed))
        }
        other => Err(ServerError::BadRequest(format!(
            "response_format.type {other:?} is not supported (expected \
             \"text\" | \"json_object\" | \"json_schema\")"
        ))),
    }
}

/// Resolve common end-of-turn token ids for the loaded model. The
/// constrained-mask uses these to gate EOS — the model can't truncate
/// while the FSM is mid-structure, but once the FSM is complete the
/// EOS tokens become legal again.
///
/// Looks up a small set of well-known special markers
/// (`<end_of_turn>`, `<|im_end|>`, `<eos>`, `</s>`, etc.) via
/// `tokenizer.token_to_id` and ignores any that aren't present in the
/// vocab.
fn resolve_eos_token_ids(
    tokenizer: &larql_inference::tokenizers::Tokenizer,
) -> std::collections::HashSet<u32> {
    let mut ids = std::collections::HashSet::new();
    for tok in [
        "<end_of_turn>",
        "<|end_of_turn|>",
        "<|im_end|>",
        "<|eot_id|>",
        "<|eom_id|>",
        "<|endoftext|>",
        "<|end_of_text|>",
        "<eos>",
        "</s>",
    ] {
        if let Some(id) = tokenizer.token_to_id(tok) {
            ids.insert(id);
        }
    }
    ids
}

/// Build the masked-vocab callback the constrained generator expects.
/// Wraps the tokenizer in `Arc` (the schema mask caches surface forms
/// per id), seeds a fresh FSM from `schema`, and includes the model's
/// EOS marker ids so structured output can terminate cleanly once the
/// FSM hits `is_complete()`.
fn build_constrained_mask(
    tokenizer: &larql_inference::tokenizers::Tokenizer,
    schema: Schema,
) -> impl FnMut(&[u32], &mut Vec<f32>) {
    let eos_ids = resolve_eos_token_ids(tokenizer);
    let tk: std::sync::Arc<larql_inference::tokenizers::Tokenizer> =
        std::sync::Arc::new(tokenizer.clone());
    super::schema::build_mask(tk, super::schema::Fsm::new(schema), String::new(), eos_ids)
}

#[cfg(test)]
mod tests {
    use super::*;

    // Multi-turn template rendering is tested in
    // `larql_inference::prompt::render_messages_tests` (Gemma, ChatML,
    // Llama, Mistral, Plain). This handler only marshals JSON to the
    // inference helper, so our tests focus on the request-validation
    // surface and shape decisions specific to the OpenAI wire.

    #[test]
    fn deserialize_chat_request_min() {
        let json = serde_json::json!({
            "messages": [{"role": "user", "content": "hi"}]
        });
        let req: ChatCompletionsRequest = serde_json::from_value(json).unwrap();
        assert_eq!(req.messages.len(), 1);
        assert_eq!(req.messages[0].role, "user");
    }

    #[test]
    fn deserialize_chat_request_full() {
        let json = serde_json::json!({
            "model": "gemma-3-4b",
            "messages": [
                {"role": "system", "content": "You are concise."},
                {"role": "user", "content": "What is 2+2?"}
            ],
            "max_tokens": 50,
            "temperature": 0.0,
            "top_p": 0.9,
            "n": 1,
            "stream": false,
            "stop": ["\n\n"],
            "seed": 42
        });
        let req: ChatCompletionsRequest = serde_json::from_value(json).unwrap();
        assert_eq!(req.messages.len(), 2);
        assert_eq!(req.max_tokens, Some(50));
        assert_eq!(req.temperature, Some(0.0));
    }

    #[test]
    fn format_tool_result_includes_call_id_and_body() {
        let s = format_tool_result(Some("call_abc"), Some("23 C"));
        assert!(s.contains("call_abc"));
        assert!(s.contains("23 C"));
    }

    #[test]
    fn format_tool_calls_summarises_function_calls() {
        let tc = serde_json::json!([
            {"id": "call_1", "type": "function",
             "function": {"name": "calc", "arguments": "{\"a\":1}"}}
        ]);
        let out = format_tool_calls(&tc);
        assert!(out.contains("calc"), "missing name in {out}");
        assert!(out.contains("{\"a\":1}"), "missing args in {out}");
    }

    #[test]
    fn build_chat_tool_calls_chunk_shapes_delta_correctly() {
        let calls = vec![ToolCall {
            id: "call_xyz".into(),
            kind: "function",
            function: ToolCallFunction {
                name: "calc".into(),
                arguments: "{\"a\":1,\"b\":2}".into(),
            },
        }];
        let chunk = build_chat_tool_calls_chunk("chatcmpl-x", "gemma", &calls);
        let v: serde_json::Value = serde_json::from_str(&chunk).unwrap();
        assert_eq!(v["object"], "chat.completion.chunk");
        assert_eq!(v["choices"][0]["delta"]["tool_calls"][0]["index"], 0);
        assert_eq!(v["choices"][0]["delta"]["tool_calls"][0]["id"], "call_xyz");
        assert_eq!(
            v["choices"][0]["delta"]["tool_calls"][0]["type"],
            "function"
        );
        assert_eq!(
            v["choices"][0]["delta"]["tool_calls"][0]["function"]["name"],
            "calc"
        );
        // arguments is JSON-stringified.
        assert_eq!(
            v["choices"][0]["delta"]["tool_calls"][0]["function"]["arguments"],
            "{\"a\":1,\"b\":2}"
        );
        assert!(v["choices"][0]["finish_reason"].is_null());
    }

    #[test]
    fn build_chat_logprobs_emits_one_entry_per_token() {
        let toks = vec![("Paris".to_string(), 1.0), (".".to_string(), 1.0)];
        let lp = build_chat_logprobs(&toks);
        assert_eq!(lp.content.len(), 2);
        assert_eq!(lp.content[0].token, "Paris");
        assert_eq!(lp.content[0].bytes, b"Paris".to_vec());
        assert!(lp.content[0].top_logprobs.is_empty());
        // prob=1.0 → logprob=0.0 (placeholder until inference exposes
        // real per-token softmax probs).
        assert!((lp.content[0].logprob - 0.0).abs() < 1e-6);
    }

    #[test]
    fn deserialize_chat_message_with_tool_call_replay() {
        // Multi-turn shape OpenAI clients send back: assistant tool-call
        // + tool result + (next) assistant turn the model would emit.
        let json = serde_json::json!({
            "messages": [
                {"role": "user", "content": "Weather?"},
                {"role": "assistant", "content": null, "tool_calls": [
                    {"id": "call_1", "type": "function",
                     "function": {"name": "get_weather", "arguments": "{\"city\":\"London\"}"}}
                ]},
                {"role": "tool", "tool_call_id": "call_1", "content": "23C"}
            ]
        });
        let req: ChatCompletionsRequest = serde_json::from_value(json).unwrap();
        assert_eq!(req.messages.len(), 3);
        assert!(req.messages[1].content.is_none());
        assert!(req.messages[1].tool_calls.is_some());
        assert_eq!(req.messages[2].role, "tool");
        assert_eq!(req.messages[2].tool_call_id.as_deref(), Some("call_1"));
        assert_eq!(req.messages[2].content.as_deref(), Some("23C"));
    }
}
