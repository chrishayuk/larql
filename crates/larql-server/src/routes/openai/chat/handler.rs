//! Buffered `POST /v1/chat/completions` handling — request validation,
//! the blocking generation loop, and logprobs/finish-reason shaping.

use axum::extract::State;
use axum::response::{IntoResponse, Response};
use axum::Json;
use std::sync::Arc;

use crate::error::ServerError;
use crate::routes::openai::prompt::{
    pick_template, render, ASSISTANT_ROLE, SYSTEM_ROLE, TOOL_ROLE, USER_ROLE,
};
use crate::routes::openai::schema::Schema;
use crate::routes::openai::util::{
    self, contains_any, new_id_suffix, trim_at_stop, unix_now, FINISH_REASON_LENGTH,
    FINISH_REASON_STOP, FINISH_REASON_TOOL_CALLS,
};
use crate::routes::openai::OpenAIError;
use crate::state::{AppState, LoadedModel};

use super::stream::stream_chat_completion;
use super::tools::{
    build_constrained_mask, build_tool_call_message, is_empty_json_array, resolve_tools,
    schema_for_response_format,
};
use super::types::{
    ChatChoice, ChatChoiceMessage, ChatCompletionsRequest, ChatCompletionsResponse, ChatLogprobs,
    ChatMessage, ChatUsage, TokenLogprob,
};
use super::{CHAT_COMPLETION_OBJECT, DEFAULT_MAX_TOKENS};

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
        (status = 400, description = "Bad request", body = crate::routes::openai::error::OpenAIErrorBody),
        (status = 500, description = "Internal server error", body = crate::routes::openai::error::OpenAIErrorBody),
    ),
)]
pub async fn handle_chat_completions(
    State(state): State<Arc<AppState>>,
    Json(req): Json<ChatCompletionsRequest>,
) -> Result<Response, OpenAIError> {
    state.bump_requests();

    if req.n.unwrap_or(1) > 1 {
        return Err(OpenAIError::invalid_request(
            "n>1 not yet supported; only n=1 (single completion per prompt)",
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

    // Resolve across BOTH registries — the V2/V3 decision is made here
    // and nowhere below (same contract as `/v1/completions` and
    // `/v1/responses`). Message validation is model-independent, so the
    // V3 dispatch happens after it, once the knobs are computed.
    let served = state.served_or_err(req.model.as_deref())?;
    let model = match &served {
        crate::state::ServedModel::V2(m) => {
            if m.infer_disabled {
                return Err(OpenAIError::service_unavailable(
                    "inference disabled (--no-infer / --embed-only / --ffn-only)",
                ));
            }
            Some((*m).clone())
        }
        crate::state::ServedModel::V3(_) => None,
    };
    let v3_model = match &served {
        crate::state::ServedModel::V3(m) => Some((*m).clone()),
        crate::state::ServedModel::V2(_) => None,
    };
    if req.messages.is_empty() {
        return Err(OpenAIError::invalid_request("messages is empty"));
    }
    for (i, m) in req.messages.iter().enumerate() {
        if !matches!(
            m.role.as_str(),
            USER_ROLE | ASSISTANT_ROLE | SYSTEM_ROLE | TOOL_ROLE
        ) {
            return Err(OpenAIError::invalid_request(format!(
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
                    return Err(OpenAIError::invalid_request(format!(
                        "messages[{i}] role=tool requires tool_call_id"
                    )));
                }
                if m.content.is_none() {
                    return Err(OpenAIError::invalid_request(format!(
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
                    return Err(OpenAIError::invalid_request(format!(
                        "messages[{i}] role=assistant requires content (or tool_calls)"
                    )));
                }
            }
            USER_ROLE | SYSTEM_ROLE if m.content.is_none() => {
                return Err(OpenAIError::invalid_request(format!(
                    "messages[{i}] role={} requires content",
                    m.role
                )));
            }
            _ => {}
        }
    }

    let max_tokens = crate::routes::limits::generation_tokens(
        "max_completion_tokens",
        req.max_completion_tokens.or(req.max_tokens),
        DEFAULT_MAX_TOKENS,
    )
    .map_err(OpenAIError::invalid_request)?;
    let stop_strings: Vec<String> = req
        .stop
        .as_ref()
        .map(|s| s.as_slice().to_vec())
        .unwrap_or_default();
    let sampling_params = util::SamplingParams {
        temperature: req.temperature,
        top_p: req.top_p,
        seed: req.seed,
        frequency_penalty: req.frequency_penalty,
        presence_penalty: req.presence_penalty,
    };
    // V3 dispatch: everything wire-shaped below is shared; only the
    // token source differs (see `chat/v3.rs`).
    if let Some(v3) = v3_model {
        let model_id = req.model.clone().unwrap_or_else(|| v3.id.clone());
        return super::v3::respond(
            v3,
            req.messages,
            max_tokens,
            sampling_params,
            stop_strings,
            constrained_schema,
            tools_active,
            req.stream.unwrap_or(false),
            req.logprobs.unwrap_or(false),
            model_id,
            state.infer_timeout,
            Arc::clone(&state.runtime),
        )
        .await;
    }

    let model = model.expect("V2 binding checked above");
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
            Arc::clone(&state.runtime),
        )
        .into_response());
    }

    // Race the blocking generation against `state.infer_timeout`, mirroring
    // `run_infer_with_timeout` (routes/infer.rs, BUG-infer-deadlock §5.6).
    // Without this, a stuck/slow chat completion holds `LoadedModel.weights`'
    // write guard for as long as the spawned thread runs, and every other
    // OpenAI-route request queues on that guard indefinitely. On timeout we
    // drop the JoinHandle and respond 504; the spawned thread finishes (or
    // doesn't) in the background, same tradeoff /v1/infer already accepts.
    let logprobs_requested = req.logprobs.unwrap_or(false);
    let started = std::time::Instant::now();
    let _gen_guard = Arc::clone(&state.runtime).enter_generation();
    let handle = tokio::task::spawn_blocking(move || -> Result<_, ServerError> {
        let mut tally = crate::runtime_stats::GenerationTally::new();
        let out = run_chat_completion(
            &model_arc,
            &messages,
            max_tokens,
            sampling_params,
            &stop_strings,
            constrained_schema,
            &mut tally,
        )?;
        Ok((out, tally))
    });
    let timeout = state.infer_timeout;
    let (output, tally) = if timeout.is_zero() {
        handle
            .await
            .map_err(|e| ServerError::Internal(e.to_string()))??
    } else {
        match tokio::time::timeout(timeout, handle).await {
            Ok(join_result) => join_result.map_err(|e| ServerError::Internal(e.to_string()))??,
            Err(_elapsed) => {
                tracing::warn!(
                    target: "larql_server::openai::chat",
                    "chat completion timed out after {:.1}s; dropping in-flight task and \
                     responding 504 (background thread will finish on its own)",
                    started.elapsed().as_secs_f64(),
                );
                return Err(OpenAIError::from(ServerError::Timeout(format!(
                    "chat completion exceeded server-side timeout of {}s",
                    timeout.as_secs(),
                ))));
            }
        }
    };
    state
        .runtime
        .record(tally.into_sample(crate::state::elapsed_ms(started)));

    let logprobs = if logprobs_requested && !tools_active {
        Some(build_chat_logprobs(&output.tokens))
    } else {
        None
    };

    let (message, finish_reason) = if tools_active {
        match build_tool_call_message(&output.text) {
            Ok(m) => (m, FINISH_REASON_TOOL_CALLS),
            Err(e) => {
                // 400 not 500: the failure is recoverable (client can
                // retry, simplify tool schema, or fall back).
                return Err(OpenAIError::invalid_request(format!(
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
pub(super) fn build_chat_logprobs(tokens: &[(String, f64)]) -> ChatLogprobs {
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
pub(in crate::routes::openai) fn run_chat_completion(
    model: &LoadedModel,
    messages: &[ChatMessage],
    max_tokens: usize,
    sampling_params: util::SamplingParams,
    stop_strings: &[String],
    constrained_schema: Option<Schema>,
    tally: &mut crate::runtime_stats::GenerationTally,
) -> Result<ChatGenerationOutput, ServerError> {
    // Take an exclusive write guard on the weights for the duration
    // of generation. `larql_inference::layer_graph::generate` mutates
    // `weights.tensors` (the per-layer Q4_K dequant cache), so other
    // read paths block while one chat completion runs.
    let mut weights_guard = model
        .lock_weights_for_gen()
        .map_err(ServerError::InferenceUnavailable)?;
    let weights: &mut larql_inference::ModelWeights = &mut weights_guard;

    let template = pick_template(weights);
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
        let (sampling, eos) = util::build_sampling_eos(sampling_params, stop_strings);
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
        let (sampling, eos) = util::build_sampling_eos(sampling_params, stop_strings);
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
    tally.add_v2(&result, prompt_token_count);

    let mut completion_text = String::new();
    let mut completion_tokens: Vec<(String, f64)> = Vec::new();
    let mut finish_reason: &'static str = FINISH_REASON_LENGTH;
    for (text, prob) in &result.tokens {
        completion_text.push_str(text);
        completion_tokens.push((text.clone(), *prob));
        if larql_inference::vindex::is_end_of_turn(text) {
            finish_reason = FINISH_REASON_STOP;
            break;
        }
    }
    // The generation loop halts on EOS internally without necessarily
    // surfacing an end-of-turn marker token; fewer tokens than the
    // budget means it stopped, not that it ran out — the same rule the
    // streaming path applies. (Found live: 3/8 tokens reported as
    // "length".)
    if finish_reason == FINISH_REASON_LENGTH && result.tokens.len() < max_tokens {
        finish_reason = FINISH_REASON_STOP;
    }
    if !stop_strings.is_empty() && contains_any(&completion_text, stop_strings) {
        completion_text = trim_at_stop(&completion_text, stop_strings);
        finish_reason = FINISH_REASON_STOP;
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
pub(in crate::routes::openai) struct ChatGenerationOutput {
    pub(in crate::routes::openai) text: String,
    pub(in crate::routes::openai) tokens: Vec<(String, f64)>,
    pub(in crate::routes::openai) finish_reason: &'static str,
    pub(in crate::routes::openai) prompt_tokens: usize,
    pub(in crate::routes::openai) completion_tokens: usize,
}

/// Truncate `tokens` so concatenated surface forms cover at most the
/// byte length of `truncated_text`. Used after `trim_at_stop` chops
/// the joined string to keep `tokens.len()` matching `text.len()`.
pub(super) fn trim_tokens_to_text(
    tokens: &[(String, f64)],
    truncated_text: &str,
) -> Vec<(String, f64)> {
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
