//! `POST /v1/responses` — request validation, planning, and the
//! buffered (non-streaming) serving path.
//!
//! Streaming shares everything up to generation ([`RequestPlan`]) and
//! then diverges into `super::stream` for the typed-event SSE shape.

use std::sync::Arc;

use axum::extract::State;
use axum::http::HeaderMap;
use axum::response::{IntoResponse, Response};
use axum::Json;

use crate::error::ServerError;
use crate::response_store::{StoredMessage, StoredResponse};
use crate::routes::openai::OpenAIError;
use crate::session::{extract_session_id, SessionLease};
use crate::state::{AppState, ServedModel};

use super::super::chat::{build_tool_call_message, schema_for_response_format, ChatMessage};
use super::super::prompt::{format_tool_calls, format_tool_result, SYSTEM_ROLE, TOOL_ROLE};
use super::super::schema::{resolve_tool_choice, synth_tools_schema, Schema, ToolMode};
use super::super::util::{join_generation, new_id_suffix, unix_now, SamplingParams};
use super::engine::{GenerationOutcome, ResponsesEngine};
use super::input::input_to_messages;
use super::tools::{chat_shaped_tool_choice, chat_shaped_tools, response_format_from_text};
use super::types::{
    IncompleteDetails, OutputContent, OutputItem, ResponseObject, ResponseUsage, ResponsesRequest,
    CALL_ID_PREFIX, DEFAULT_MAX_OUTPUT_TOKENS, FUNCTION_CALL_ID_PREFIX,
    INCOMPLETE_MAX_OUTPUT_TOKENS, MESSAGE_ID_PREFIX, RESPONSE_ID_PREFIX, RESPONSE_OBJECT,
    STATUS_COMPLETED, STATUS_INCOMPLETE,
};

/// Everything both serving paths need, resolved once up front.
pub(super) struct RequestPlan {
    pub id: String,
    pub created_at: u64,
    pub engine: ResponsesEngine,
    pub model_id: String,
    pub messages: Vec<ChatMessage>,
    pub max_tokens: usize,
    pub sampling: SamplingParams,
    pub stop_strings: Vec<String>,
    pub schema: Option<Schema>,
    pub tools_active: bool,
    pub store: bool,
    /// The session this request is bound to, when it carried an
    /// `X-Session-Id`. Owns whatever KV continuation the generation
    /// retains, and collects this session's resumption counters.
    pub session: Option<Arc<SessionLease>>,
    // Echoed on the envelope.
    pub previous_response_id: Option<String>,
    pub instructions: Option<String>,
    pub metadata: Option<serde_json::Value>,
    pub temperature: Option<f32>,
    pub top_p: Option<f32>,
    pub max_output_tokens: Option<usize>,
}

#[utoipa::path(
    post,
    path = "/v1/responses",
    tag = "openai",
    request_body = crate::openapi::schemas::OpenAiResponsesRequest,
    responses(
        (status = 200, description = "Non-streaming response envelope.",
         body = crate::openapi::schemas::OpenAiResponsesResponse),
        (status = 200, description = "SSE stream when `stream: true`: typed events \
          (`response.created` … `response.output_text.delta` … `response.completed`), \
          terminated by `data: [DONE]`.",
         content_type = "text/event-stream", body = String),
        (status = 400, body = crate::routes::openai::error::OpenAIErrorBody),
        (status = 404, body = crate::routes::openai::error::OpenAIErrorBody),
        (status = 500, body = crate::routes::openai::error::OpenAIErrorBody),
    ),
)]
pub async fn handle_responses(
    State(state): State<Arc<AppState>>,
    headers: HeaderMap,
    Json(req): Json<ResponsesRequest>,
) -> Result<Response, OpenAIError> {
    state.bump_requests();
    let stream = req.stream.unwrap_or(false);
    let mut plan = plan_request(&state, req)?;
    plan.session = bind_session(&state, &headers, &plan).await;

    if stream {
        return Ok(super::stream::sse_response(state, plan).into_response());
    }

    let engine = plan.engine.clone();
    let messages = plan.messages.clone();
    let (max_tokens, sampling, stops, schema) = (
        plan.max_tokens,
        plan.sampling,
        plan.stop_strings.clone(),
        plan.schema.clone(),
    );
    let resume = take_kv_resume(&state, &plan);
    let started = std::time::Instant::now();
    let _gen_guard = Arc::clone(&state.runtime).enter_generation();
    let handle = tokio::task::spawn_blocking(move || -> Result<GenerationOutcome, ServerError> {
        super::engine::generate(
            &engine,
            &messages,
            max_tokens,
            sampling,
            &stops,
            schema,
            resume,
            &mut |_text| true,
        )
    });
    let mut outcome = join_generation(handle, state.infer_timeout).await?;
    state
        .runtime
        .record(outcome.tally.into_sample(crate::state::elapsed_ms(started)));
    retain_kv_handoff(&state, &plan, &mut outcome);

    let (output, status, incomplete) =
        build_output(plan.tools_active, &outcome).map_err(OpenAIError::invalid_request)?;
    let usage = usage_of(&outcome);
    let envelope = build_envelope(&plan, status, incomplete, output, Some(usage));
    persist(&state, &plan, &envelope);
    Ok(Json(envelope).into_response())
}

/// Bind this request to a session when the client named one.
///
/// Binding is the client's opt-in: without `X-Session-Id` the request
/// owns nothing and the continuation it retains is governed by capacity
/// and TTL alone. With it, the session is created if absent (cheaply —
/// no patch overlay is materialised), its idle clock is refreshed, and
/// it becomes the owner of whatever KV state this generation leaves
/// behind. The session is bound to the *runtime binding's* model id, the
/// same identity the KV cache is keyed by.
async fn bind_session(
    state: &AppState,
    headers: &HeaderMap,
    plan: &RequestPlan,
) -> Option<Arc<SessionLease>> {
    let session_id = extract_session_id(headers)?;
    Some(
        state
            .sessions
            .bind(&session_id, plan.engine.model_id())
            .await,
    )
}

/// N1 — take the previous turn's resident KV continuation state, when
/// this request chains from a stored response served by a V3 runtime.
/// Take-once: a concurrent second chain from the same id falls back to
/// a full prefill (identical output, just not accelerated).
pub(super) fn take_kv_resume(
    state: &AppState,
    plan: &RequestPlan,
) -> Option<crate::vindex3::V3KvHandoff> {
    if !plan.engine.is_v3() || plan.tools_active || !state.v3_kv.enabled() {
        return None;
    }
    let prev = plan.previous_response_id.as_deref()?;
    // Keyed by the BINDING's id, not the request's display model id —
    // a KV state is only meaningful under the weights that produced it.
    state.v3_kv.take(prev, plan.engine.model_id())
}

/// N1 — retain this generation's KV continuation state under the new
/// response id, so the next chain link can resume from it. Only stored
/// responses are chainable, so `store: false` retains nothing. Also
/// records whether resumption actually ENGAGED this generation — the
/// `hits`-vs-`resumptions` gap in `/v1/stats` is the live measure of
/// exact-prefix survival.
pub(super) fn retain_kv_handoff(
    state: &AppState,
    plan: &RequestPlan,
    outcome: &mut GenerationOutcome,
) {
    if outcome.continuation_refused.is_some() {
        state.v3_kv.record_refusal();
    }
    if outcome.reused_prompt_tokens > 0 {
        state.v3_kv.record_resumption(outcome.reused_prompt_tokens);
        if let Some(session) = &plan.session {
            session.record_resumption(outcome.reused_prompt_tokens);
        }
    }
    if let Some(session) = &plan.session {
        // Generation can outlast the request that started it; refresh
        // the idle clock at the end too so a long turn cannot expire
        // the session it belongs to.
        session.touch();
    }
    if let Some(handoff) = outcome.kv_handoff.take() {
        if plan.store {
            state.v3_kv.insert(
                &plan.id,
                plan.engine.model_id(),
                plan.session.clone(),
                handoff,
            );
        }
    }
}

/// Validate the request and resolve every input into a [`RequestPlan`].
pub(super) fn plan_request(
    state: &AppState,
    req: ResponsesRequest,
) -> Result<RequestPlan, OpenAIError> {
    if req.background.unwrap_or(false) {
        return Err(OpenAIError::invalid_request(
            "background: true is not supported (larql serves responses synchronously)",
        ));
    }
    if let Some(user) = req.user.as_deref() {
        tracing::debug!(target: "larql_server::openai::responses", user, "request user tag");
    }

    let mut messages: Vec<ChatMessage> = Vec::new();
    if let Some(instructions) = req.instructions.as_deref() {
        messages.push(ChatMessage {
            role: SYSTEM_ROLE.to_string(),
            content: Some(instructions.to_string()),
            tool_calls: None,
            tool_call_id: None,
            name: None,
        });
    }

    // Conversation chaining: replay the stored conversation ahead of
    // this request's input. The stored model id becomes the default
    // model so a follow-up hits the same runtime.
    let mut model_hint = req.model.clone();
    if let Some(prev_id) = req.previous_response_id.as_deref() {
        let prev = state.responses.get(prev_id).ok_or_else(|| {
            OpenAIError::not_found(format!("previous response '{prev_id}' not found"))
        })?;
        if model_hint.is_none() {
            model_hint = Some(prev.model_id.clone());
        }
        messages.extend(prev.conversation.iter().map(|m| ChatMessage {
            role: m.role.clone(),
            content: Some(m.content.clone()),
            tool_calls: None,
            tool_call_id: None,
            name: None,
        }));
    }

    messages.extend(input_to_messages(&req.input).map_err(OpenAIError::invalid_request)?);
    if messages.is_empty() {
        return Err(OpenAIError::invalid_request("input is empty"));
    }

    let engine = match state.served_or_err(model_hint.as_deref())? {
        ServedModel::V2(m) => {
            if m.infer_disabled {
                return Err(OpenAIError::service_unavailable(
                    "inference disabled (--no-infer / --embed-only / --ffn-only)",
                ));
            }
            ResponsesEngine::V2(m.clone())
        }
        ServedModel::V3(m) => ResponsesEngine::V3(m.clone()),
    };
    let model_id = model_hint.unwrap_or_else(|| engine.model_id().to_string());

    let (schema, tools_active) = resolve_constraints(&req)?;

    let stop_strings: Vec<String> = req
        .stop
        .as_ref()
        .map(|s| s.as_slice().to_vec())
        .unwrap_or_default();

    Ok(RequestPlan {
        id: format!("{RESPONSE_ID_PREFIX}{}", new_id_suffix()),
        created_at: unix_now(),
        engine,
        model_id,
        messages,
        max_tokens: req.max_output_tokens.unwrap_or(DEFAULT_MAX_OUTPUT_TOKENS),
        sampling: SamplingParams {
            temperature: req.temperature,
            top_p: req.top_p,
            ..SamplingParams::default()
        },
        stop_strings,
        schema,
        tools_active,
        store: req.store.unwrap_or(true),
        // Bound by the handler once the plan's model binding is known.
        session: None,
        previous_response_id: req.previous_response_id,
        instructions: req.instructions,
        metadata: req.metadata,
        temperature: req.temperature,
        top_p: req.top_p,
        max_output_tokens: req.max_output_tokens,
    })
}

/// Tools take precedence over `text.format`, mirroring the chat
/// endpoint's contract.
fn resolve_constraints(req: &ResponsesRequest) -> Result<(Option<Schema>, bool), OpenAIError> {
    let tools_present = req
        .tools
        .as_ref()
        .is_some_and(|v| v.as_array().is_some_and(|a| !a.is_empty()));
    if tools_present {
        let tools = req.tools.as_ref().expect("checked above");
        let chat_tools = chat_shaped_tools(tools).map_err(OpenAIError::invalid_request)?;
        let choice = req.tool_choice.as_ref().map(chat_shaped_tool_choice);
        let names: Vec<String> = chat_tools
            .as_array()
            .into_iter()
            .flatten()
            .filter_map(|t| {
                t.get("function")
                    .and_then(|f| f.get("name"))
                    .and_then(|n| n.as_str())
                    .map(str::to_string)
            })
            .collect();
        let mode = resolve_tool_choice(true, choice.as_ref(), &names)
            .map_err(OpenAIError::invalid_request)?;
        if !matches!(mode, ToolMode::None) {
            let synth =
                synth_tools_schema(&chat_tools, &mode).map_err(OpenAIError::invalid_request)?;
            if let Some((schema, _names)) = synth {
                return Ok((Some(schema), true));
            }
        }
    }
    let response_format =
        response_format_from_text(req.text.as_ref()).map_err(OpenAIError::invalid_request)?;
    let schema = schema_for_response_format(response_format.as_ref())?;
    Ok((schema, false))
}

/// Map a finished generation to `output[]` + envelope status.
pub(super) fn build_output(
    tools_active: bool,
    outcome: &GenerationOutcome,
) -> Result<(Vec<OutputItem>, &'static str, Option<IncompleteDetails>), String> {
    if tools_active {
        let msg = build_tool_call_message(&outcome.text)
            .map_err(|e| format!("tool_call output failed to parse: {e}"))?;
        let call = msg
            .tool_calls
            .as_ref()
            .and_then(|calls| calls.first())
            .ok_or_else(|| "tool_call output produced no call".to_string())?;
        let item = OutputItem::FunctionCall {
            id: format!("{FUNCTION_CALL_ID_PREFIX}{}", new_id_suffix()),
            status: STATUS_COMPLETED.to_string(),
            call_id: format!("{CALL_ID_PREFIX}{}", new_id_suffix()),
            name: call.function.name.clone(),
            arguments: call.function.arguments.clone(),
        };
        return Ok((vec![item], STATUS_COMPLETED, None));
    }
    let item = message_item(&outcome.text, STATUS_COMPLETED);
    if outcome.stopped {
        Ok((vec![item], STATUS_COMPLETED, None))
    } else {
        Ok((
            vec![item],
            STATUS_INCOMPLETE,
            Some(IncompleteDetails {
                reason: INCOMPLETE_MAX_OUTPUT_TOKENS,
            }),
        ))
    }
}

/// Build a completed message output item around `text`.
pub(super) fn message_item(text: &str, status: &str) -> OutputItem {
    OutputItem::Message {
        id: format!("{MESSAGE_ID_PREFIX}{}", new_id_suffix()),
        status: status.to_string(),
        role: super::super::prompt::ASSISTANT_ROLE,
        content: vec![OutputContent::OutputText {
            text: text.to_string(),
            annotations: Vec::new(),
        }],
    }
}

pub(super) fn usage_of(outcome: &GenerationOutcome) -> ResponseUsage {
    ResponseUsage {
        input_tokens: outcome.prompt_tokens,
        input_tokens_details: super::types::InputTokensDetails {
            cached_tokens: outcome.reused_prompt_tokens,
            continuation_refused: outcome.continuation_refused.as_ref().map(|refusal| {
                super::types::ContinuationRefusal {
                    kind: refusal.kind(),
                    reason: refusal.to_string(),
                    recovery: super::types::FRESH_PREFILL_RECOVERY,
                }
            }),
        },
        output_tokens: outcome.completion_tokens,
        total_tokens: outcome.prompt_tokens + outcome.completion_tokens,
    }
}

pub(super) fn build_envelope(
    plan: &RequestPlan,
    status: &str,
    incomplete_details: Option<IncompleteDetails>,
    output: Vec<OutputItem>,
    usage: Option<ResponseUsage>,
) -> ResponseObject {
    ResponseObject {
        id: plan.id.clone(),
        object: RESPONSE_OBJECT,
        created_at: plan.created_at,
        status: status.to_string(),
        error: None,
        incomplete_details,
        model: plan.model_id.clone(),
        output,
        previous_response_id: plan.previous_response_id.clone(),
        instructions: plan.instructions.clone(),
        max_output_tokens: plan.max_output_tokens,
        temperature: plan.temperature,
        top_p: plan.top_p,
        metadata: plan.metadata.clone(),
        usage,
    }
}

/// Store the finished response for `previous_response_id` chaining and
/// `GET /v1/responses/{id}` retrieval, honouring `store: false`.
pub(super) fn persist(state: &AppState, plan: &RequestPlan, envelope: &ResponseObject) {
    if !plan.store {
        return;
    }
    let mut conversation: Vec<StoredMessage> = plan.messages.iter().map(flatten_message).collect();
    conversation.push(StoredMessage {
        role: super::super::prompt::ASSISTANT_ROLE.to_string(),
        content: assistant_turn_text(envelope),
    });
    state.responses.insert(StoredResponse {
        id: plan.id.clone(),
        model_id: plan.model_id.clone(),
        conversation,
        envelope: serde_json::to_value(envelope).unwrap_or(serde_json::Value::Null),
    });
}

/// Text form of the assistant turn for replay: plain text output, or
/// the rendered tool-call summary when the output was a function call.
pub(super) fn assistant_turn_text(envelope: &ResponseObject) -> String {
    let text = envelope.output_text();
    if !text.is_empty() {
        return text;
    }
    for item in &envelope.output {
        if let OutputItem::FunctionCall {
            call_id,
            name,
            arguments,
            ..
        } = item
        {
            let calls = serde_json::json!([{
                "id": call_id,
                "type": "function",
                "function": {"name": name, "arguments": arguments},
            }]);
            return format_tool_calls(&calls);
        }
    }
    String::new()
}

/// Flatten one wire message into the stored `(role, content)` form —
/// tool traffic is rendered to text exactly the way the prompt
/// renderer would, so replay produces the same prompt.
pub(super) fn flatten_message(m: &ChatMessage) -> StoredMessage {
    match m.role.as_str() {
        TOOL_ROLE => StoredMessage {
            role: super::super::prompt::USER_ROLE.to_string(),
            content: format_tool_result(m.tool_call_id.as_deref(), m.content.as_deref()),
        },
        _ => {
            let content = match (&m.content, &m.tool_calls) {
                (Some(c), _) => c.clone(),
                (None, Some(tc)) => format_tool_calls(tc),
                (None, None) => String::new(),
            };
            StoredMessage {
                role: m.role.clone(),
                content,
            }
        }
    }
}
