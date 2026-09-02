//! `/v1/chat/completions` served by a VINDEX3 runtime.
//!
//! Reached only through [`crate::state::AppState::served`] resolving
//! the request's model to a [`V3Model`] — the same single decision
//! point `/v1/completions` and `/v1/responses` use. Everything
//! wire-shaped is shared with the V2 path (the response structs, the
//! chunk builder, finish reasons), so the two runtimes cannot drift in
//! what a client sees; only the token source differs:
//!
//! ```text
//! V2: ModelWeights + VectorIndex → generate_streaming
//! V3: Vindex3Runtime → CanonicalKvState → prefill_into
//!       → session_with_kv → continue_session
//! ```
//!
//! Tools / structured output (N0.6) run through the SAME schema →
//! FSM → logits-mask pipeline as V2 (`build_constrained_mask`), fed
//! into the V3 driver's masked variant — one grammar implementation,
//! two runtimes.

use std::convert::Infallible;
use std::sync::Arc;

use axum::response::sse::{Event, KeepAlive, Sse};
use axum::response::{IntoResponse, Response};
use axum::Json;
use futures::stream::Stream;
use tokio_stream::wrappers::ReceiverStream;
use tokio_stream::StreamExt as _;

use crate::error::ServerError;
use crate::routes::openai::prompt::{render, ASSISTANT_ROLE};
use crate::routes::openai::schema::Schema;
use crate::routes::openai::token_tap::{EmitFailure, TokenTap};
use crate::routes::openai::util::{
    build_sampling_eos_from, contains_any, error_chunk, join_generation, new_id_suffix,
    trim_at_stop, unix_now, SamplingParams, FINISH_REASON_LENGTH, FINISH_REASON_STOP,
    SSE_CHANNEL_DEPTH, SSE_DONE,
};
use crate::routes::openai::OpenAIError;
use crate::vindex3::{generate_v3, generate_v3_constrained, V3Generation, V3Model};

use super::handler::build_chat_logprobs;
use super::stream::{build_chat_chunk, build_chat_tool_calls_chunk};
use super::tools::{build_constrained_mask, build_tool_call_message};
use super::types::{
    ChatChoice, ChatChoiceMessage, ChatCompletionsResponse, ChatMessage, ChatUsage,
};
use super::CHAT_COMPLETION_OBJECT;

/// Serve one already-validated chat request on a V3 runtime. The
/// handler has validated messages/knobs; this function renders,
/// generates (masked when a constraint schema is present), and shapes.
#[allow(clippy::too_many_arguments)]
pub(super) async fn respond(
    model: Arc<V3Model>,
    messages: Vec<ChatMessage>,
    max_tokens: usize,
    sampling_params: SamplingParams,
    stop_strings: Vec<String>,
    constrained_schema: Option<Schema>,
    tools_active: bool,
    stream: bool,
    logprobs_requested: bool,
    model_id: String,
    timeout: std::time::Duration,
    runtime: Arc<crate::runtime_stats::RuntimeRecorder>,
) -> Result<Response, OpenAIError> {
    if stream {
        return Ok(stream_v3_chat(
            model,
            messages,
            max_tokens,
            sampling_params,
            stop_strings,
            constrained_schema,
            tools_active,
            model_id,
            runtime,
        )
        .into_response());
    }

    let started = std::time::Instant::now();
    let _gen_guard = Arc::clone(&runtime).enter_generation();
    let handle = tokio::task::spawn_blocking(move || -> Result<_, ServerError> {
        let mut tally = crate::runtime_stats::GenerationTally::new();
        let out = run_v3_chat(
            &model,
            &messages,
            max_tokens,
            sampling_params,
            &stop_strings,
            constrained_schema,
            &mut tally,
        )?;
        Ok((out, tally))
    });
    let ((text, tokens, finish_reason, prompt_tokens), tally) =
        join_generation(handle, timeout).await?;
    runtime.record(tally.into_sample(crate::state::elapsed_ms(started)));

    // Tool shaping mirrors the V2 handler: the constrained output
    // parses into `tool_calls`, and an unparseable output is a 400
    // (recoverable), not a 500.
    let (message, finish_reason) = if tools_active {
        match build_tool_call_message(&text) {
            Ok(m) => (m, super::super::util::FINISH_REASON_TOOL_CALLS),
            Err(e) => {
                return Err(OpenAIError::invalid_request(format!(
                    "tool_call output failed to parse: {e}; raw: {text:?}"
                )));
            }
        }
    } else {
        (
            ChatChoiceMessage {
                role: ASSISTANT_ROLE,
                content: Some(text),
                tool_calls: None,
            },
            finish_reason,
        )
    };

    let logprobs = (logprobs_requested && !tools_active).then(|| build_chat_logprobs(&tokens));
    let completion_tokens = tokens.len();
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
            prompt_tokens,
            completion_tokens,
            total_tokens: prompt_tokens + completion_tokens,
        },
    })
    .into_response())
}

/// Render + tokenise the conversation for the V3 runtime. No
/// `ModelWeights` exists on this path, so template choice falls to the
/// container's declared family, then the id heuristic
/// (`V3Model::chat_template`).
fn encode_chat_prompt(model: &V3Model, messages: &[ChatMessage]) -> Result<Vec<u32>, ServerError> {
    let template = model.chat_template();
    let prompt = render(template, messages);
    let encoding = model
        .tokenizer
        .encode(prompt.as_str(), true)
        .map_err(|e| ServerError::Internal(format!("tokenize: {e}")))?;
    let ids: Vec<u32> = encoding.get_ids().to_vec();
    if ids.is_empty() {
        return Err(ServerError::BadRequest(
            "rendered prompt tokenises to empty".into(),
        ));
    }
    Ok(ids)
}

type BufferedChat = (String, Vec<(String, f64)>, &'static str, usize);

/// Buffered generation: returns `(text, scored_tokens, finish_reason,
/// prompt_tokens)` — the same tail shape the V2 loop produces, with
/// the per-token probability fixed at 1.0 (the V3 session does not
/// surface per-token softmax yet, matching `v3_completions`).
fn run_v3_chat(
    model: &V3Model,
    messages: &[ChatMessage],
    max_tokens: usize,
    sampling_params: SamplingParams,
    stop_strings: &[String],
    constrained_schema: Option<Schema>,
    tally: &mut crate::runtime_stats::GenerationTally,
) -> Result<BufferedChat, ServerError> {
    let prompt_ids = encode_chat_prompt(model, messages)?;
    let (sampling, eos) = build_sampling_eos_from(model.eos.clone(), sampling_params, stop_strings);
    let generation = generate_maybe_masked(
        model,
        &prompt_ids,
        max_tokens,
        sampling,
        &eos,
        constrained_schema,
        |_, _| {},
    )?;
    tally.add_v3(
        generation.prompt_tokens,
        generation.texts.len(),
        generation.prefill_ms,
        generation.decode_ms_total,
    );

    let mut text: String = generation.texts.concat();
    let mut tokens: Vec<(String, f64)> =
        generation.texts.iter().map(|t| (t.clone(), 1.0)).collect();
    let mut finish_reason = if generation.stopped_early || generation.texts.len() < max_tokens {
        FINISH_REASON_STOP
    } else {
        FINISH_REASON_LENGTH
    };
    if !stop_strings.is_empty() && contains_any(&text, stop_strings) {
        text = trim_at_stop(&text, stop_strings);
        finish_reason = FINISH_REASON_STOP;
        tokens.truncate(count_tokens_covering(&tokens, text.len()));
    }
    Ok((text, tokens, finish_reason, generation.prompt_tokens))
}

/// Number of leading tokens whose concatenated surface forms fit in
/// `byte_len` — the V3 twin of the V2 path's `trim_tokens_to_text`.
pub(super) fn count_tokens_covering(tokens: &[(String, f64)], byte_len: usize) -> usize {
    let mut acc = 0usize;
    let mut kept = 0usize;
    for (t, _) in tokens {
        if acc >= byte_len {
            break;
        }
        acc += t.len();
        kept += 1;
    }
    kept
}

/// Run one V3 generation, masked when a constraint schema is present.
/// The mask pipeline (`build_constrained_mask`) is the V2 one, so the
/// grammar semantics cannot differ between runtimes.
fn generate_maybe_masked(
    model: &V3Model,
    prompt_ids: &[u32],
    max_tokens: usize,
    sampling: larql_inference::SamplingConfig,
    eos: &larql_inference::EosConfig,
    constrained_schema: Option<Schema>,
    on_token: impl FnMut(u32, &str),
) -> Result<V3Generation, ServerError> {
    match constrained_schema {
        Some(schema) => {
            let mut mask = build_constrained_mask(&model.tokenizer, schema);
            generate_v3_constrained(
                model, prompt_ids, max_tokens, sampling, eos, &mut mask, on_token,
            )
        }
        None => generate_v3(model, prompt_ids, max_tokens, sampling, eos, on_token),
    }
}

/// SSE streaming over the V3 stack — chunk shape, role-first contract,
/// stop handling, tools buffering, and termination identical to the V2
/// `stream_chat_completion`.
#[allow(clippy::too_many_arguments)]
fn stream_v3_chat(
    model: Arc<V3Model>,
    messages: Vec<ChatMessage>,
    max_tokens: usize,
    sampling_params: SamplingParams,
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
        let prompt_ids = match encode_chat_prompt(&model, &messages) {
            Ok(ids) => ids,
            Err(e) => {
                let _ = tx.blocking_send(error_chunk(&e.to_string()));
                return;
            }
        };

        // First chunk: role="assistant" delta, per the stream contract.
        let first = build_chat_chunk(&chat_id, &model_id, Some(ASSISTANT_ROLE), None, None);
        if tx.blocking_send(first).is_err() {
            return;
        }

        let (sampling, eos) =
            build_sampling_eos_from(model.eos.clone(), sampling_params, &stop_strings);
        // Tools buffer (the tool_calls delta shape only makes sense
        // once the full JSON has parsed); content streams per token.
        let mut tap = if tools_active {
            TokenTap::buffering_only(&stop_strings)
        } else {
            TokenTap::new(&stop_strings, EmitFailure::Halt)
        };
        let result = generate_maybe_masked(
            &model,
            &prompt_ids,
            max_tokens,
            sampling,
            &eos,
            constrained_schema,
            |_id, text| {
                tap.feed(text, |t| {
                    let chunk = build_chat_chunk(&chat_id, &model_id, None, Some(t), None);
                    tx.blocking_send(chunk).is_ok()
                });
            },
        );
        let finish_reason: &'static str = match &result {
            Ok(_) if tools_active => super::super::util::FINISH_REASON_TOOL_CALLS,
            Ok(generation)
                if tap.halted()
                    || generation.stopped_early
                    || generation.texts.len() < max_tokens =>
            {
                FINISH_REASON_STOP
            }
            Ok(_) => FINISH_REASON_LENGTH,
            Err(e) => {
                let _ = tx.blocking_send(error_chunk(&e.to_string()));
                return;
            }
        };
        if let Ok(generation) = &result {
            let mut tally = crate::runtime_stats::GenerationTally::new();
            tally.add_v3(
                generation.prompt_tokens,
                generation.texts.len(),
                generation.prefill_ms,
                generation.decode_ms_total,
            );
            runtime.record(tally.into_sample(crate::state::elapsed_ms(call_started)));
        }
        if tools_active {
            // One chunk carrying the whole parsed tool_calls payload —
            // the V2 stream's contract.
            match build_tool_call_message(tap.text()) {
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
