//! Generation dispatch for `/v1/responses` — one callback-driven entry
//! point over both runtimes.
//!
//! ```text
//! V2: ModelWeights + VectorIndex → generate_streaming
//!                                  (or the constrained-mask variant)
//! V3: Vindex3Runtime → generate_v3
//! ```
//!
//! The V2/V3 decision is made once, at model resolution in the handler
//! ([`crate::state::AppState::served`]); this module only executes the
//! chosen binding. Buffered and streaming callers share the same
//! function — buffered callers pass a no-op token callback.

use std::sync::Arc;

use crate::error::ServerError;
use crate::state::LoadedModel;
use crate::vindex3::{generate_v3_request, V3KvHandoff, V3Model};

use super::super::chat::ChatMessage;
use super::super::prompt::{pick_template, render};
use super::super::schema::Schema;
use super::super::token_tap::{EmitFailure, TokenTap};
use super::super::util::{
    build_sampling_eos, build_sampling_eos_from, contains_any, trim_at_stop, SamplingParams,
};

/// The resolved runtime binding for one request, cloned out of
/// `AppState` so generation can move onto a blocking thread.
#[derive(Clone)]
pub(super) enum ResponsesEngine {
    V2(Arc<LoadedModel>),
    V3(Arc<V3Model>),
}

impl ResponsesEngine {
    pub(super) fn model_id(&self) -> &str {
        match self {
            ResponsesEngine::V2(m) => &m.id,
            ResponsesEngine::V3(m) => &m.id,
        }
    }

    pub(super) fn is_v3(&self) -> bool {
        matches!(self, ResponsesEngine::V3(_))
    }
}

/// What generation produced, independent of runtime.
pub(super) struct GenerationOutcome {
    /// Full completion text, trimmed at the first client stop string.
    pub text: String,
    /// True when generation halted before `max_tokens` (EOS or a stop
    /// string) — maps to envelope status `completed` vs `incomplete`.
    pub stopped: bool,
    pub prompt_tokens: usize,
    pub completion_tokens: usize,
    /// V3 only (N1): the continuation state through this generation,
    /// for the caller to retain when the response is stored. `None` on
    /// the V2 runtime, which has no detachable KV state.
    pub kv_handoff: Option<V3KvHandoff>,
    /// Prompt tokens served from a resumed KV instead of re-prefill.
    pub reused_prompt_tokens: usize,
    /// This generation's measured performance, for the caller to feed
    /// into `RuntimeRecorder::record` — see [`crate::runtime_stats`].
    pub tally: crate::runtime_stats::GenerationTally,
}

/// Run one generation, invoking `on_token` per decoded token. The
/// callback returns `false` to stop *emission* (client disconnect);
/// stop strings additionally halt V2 generation via the EOS config.
#[allow(clippy::too_many_arguments)]
pub(super) fn generate(
    engine: &ResponsesEngine,
    messages: &[ChatMessage],
    max_tokens: usize,
    params: SamplingParams,
    stop_strings: &[String],
    schema: Option<Schema>,
    resume: Option<V3KvHandoff>,
    on_token: &mut dyn FnMut(&str) -> bool,
) -> Result<GenerationOutcome, ServerError> {
    match engine {
        ResponsesEngine::V2(model) => generate_v2(
            model,
            messages,
            max_tokens,
            params,
            stop_strings,
            schema,
            on_token,
        ),
        ResponsesEngine::V3(model) => generate_on_v3(
            model,
            messages,
            max_tokens,
            params,
            stop_strings,
            schema,
            resume,
            on_token,
        ),
    }
}

#[allow(clippy::too_many_arguments)]
fn generate_v2(
    model: &LoadedModel,
    messages: &[ChatMessage],
    max_tokens: usize,
    params: SamplingParams,
    stop_strings: &[String],
    schema: Option<Schema>,
    on_token: &mut dyn FnMut(&str) -> bool,
) -> Result<GenerationOutcome, ServerError> {
    // Exclusive weights guard for the duration — the generate family
    // takes `&mut ModelWeights` (per-layer Q4_K dequant cache).
    let mut weights_guard = model
        .lock_weights_for_gen()
        .map_err(ServerError::InferenceUnavailable)?;
    let weights: &mut larql_inference::ModelWeights = &mut weights_guard;

    let template = pick_template(weights);
    let prompt = render(template, messages);
    let prompt_ids = encode(&model.tokenizer, &prompt)?;
    let prompt_tokens = prompt_ids.len();

    let patched = model.patched.blocking_read();
    let index = patched.base();
    let backend = larql_compute::default_backend();
    let cached_layers = larql_inference::CachedLayerGraph::from_residuals(Vec::new());
    let num_layers = weights.num_layers;

    // A disconnect stops emission but not buffering — the envelope is
    // still built and stored after the client goes away.
    let mut tap = TokenTap::new(stop_strings, EmitFailure::StopEmitting);
    let token_cb = |_id: u32, text: &str, _prob: f64| {
        tap.feed(text, |t| on_token(t));
    };

    let (sampling, eos) = build_sampling_eos(params, stop_strings);
    let result = if let Some(schema) = schema {
        let mask = super::super::chat::build_constrained_mask(&model.tokenizer, schema);
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
            token_cb,
            sampling,
            &eos,
        )
    } else {
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
            token_cb,
            None,
        )
    };

    let mut tally = crate::runtime_stats::GenerationTally::new();
    tally.add_v2(&result, prompt_tokens);

    let completion_tokens = result.tokens.len();
    let stopped = tap.halted() || completion_tokens < max_tokens;
    // Assemble the final text from the result's token list, not the
    // callback buffer: the CPU-Q4K arm of `generate_streaming` returns
    // its tokens without ever invoking the callback, so the buffer can
    // be empty while tokens exist (found live-serving Gemma-3-4B Q4K).
    let mut text: String = result.tokens.iter().map(|(t, _)| t.as_str()).collect();
    if !stop_strings.is_empty() && contains_any(&text, stop_strings) {
        text = trim_at_stop(&text, stop_strings);
    }
    Ok(GenerationOutcome {
        text,
        stopped,
        prompt_tokens,
        completion_tokens,
        kv_handoff: None,
        reused_prompt_tokens: 0,
        tally,
    })
}

#[allow(clippy::too_many_arguments)]
fn generate_on_v3(
    model: &V3Model,
    messages: &[ChatMessage],
    max_tokens: usize,
    params: SamplingParams,
    stop_strings: &[String],
    schema: Option<Schema>,
    resume: Option<V3KvHandoff>,
    on_token: &mut dyn FnMut(&str) -> bool,
) -> Result<GenerationOutcome, ServerError> {
    // No `ModelWeights` on the V3 path — the container binds as an
    // executable program — so template choice falls to the id
    // container's declared family, then the id heuristic.
    let template = model.chat_template();
    let prompt = render(template, messages);
    let prompt_ids = encode(&model.tokenizer, &prompt)?;

    let (sampling, eos) = build_sampling_eos_from(model.eos.clone(), params, stop_strings);
    let mut tap = TokenTap::new(stop_strings, EmitFailure::StopEmitting);
    // The mask pipeline is V2's `build_constrained_mask` verbatim —
    // one grammar implementation, two runtimes.
    let mut mask = schema.map(|s| super::super::chat::build_constrained_mask(&model.tokenizer, s));
    let mask_ref: Option<larql_inference::vindex3::LogitsMask<'_>> = match mask.as_mut() {
        Some(m) => Some(m),
        None => None,
    };
    let (generation, handoff) = generate_v3_request(
        model,
        &prompt_ids,
        resume,
        max_tokens,
        sampling,
        &eos,
        mask_ref,
        |_id, text| {
            tap.feed(text, |t| on_token(t));
        },
    )?;

    let mut tally = crate::runtime_stats::GenerationTally::new();
    tally.add_v3(
        generation.prompt_tokens,
        generation.texts.len(),
        generation.prefill_ms,
        generation.decode_ms_total,
    );

    let stopped = tap.halted() || generation.stopped_early || generation.texts.len() < max_tokens;
    let mut text = tap.into_text();
    if !stop_strings.is_empty() && contains_any(&text, stop_strings) {
        text = trim_at_stop(&text, stop_strings);
    }
    Ok(GenerationOutcome {
        text,
        stopped,
        prompt_tokens: generation.prompt_tokens,
        completion_tokens: generation.texts.len(),
        kv_handoff: Some(handoff),
        reused_prompt_tokens: generation.reused_prompt_tokens,
        tally,
    })
}

fn encode(
    tokenizer: &larql_vindex::tokenizers::Tokenizer,
    prompt: &str,
) -> Result<Vec<u32>, ServerError> {
    let encoding = tokenizer
        .encode(prompt, true)
        .map_err(|e| ServerError::Internal(format!("tokenize: {e}")))?;
    let ids: Vec<u32> = encoding.get_ids().to_vec();
    if ids.is_empty() {
        return Err(ServerError::BadRequest(
            "rendered prompt tokenises to empty".into(),
        ));
    }
    Ok(ids)
}
