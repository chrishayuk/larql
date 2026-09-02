//! VI3-SERVE-1: a VINDEX3 container served over the normal API.
//!
//! One vertical slice, deliberately boring:
//!
//! ```text
//! VINDEX3 container → Vindex3Runtime → CanonicalKvState
//!     → prefill_into() → session_with_kv() → continue_session()
//!     → existing SSE/JSON shaping
//! ```
//!
//! What deliberately does **not** happen here: no `load_vindex3() ->
//! ModelWeights`, no `VectorIndex`, no `ModelArchitecture` — a V3
//! container binds as an executable program and is served through the
//! runtime stack INF-0..3 and KV-1 gated bit-for-bit. The V2/V3
//! distinction is decided once, at model binding
//! ([`crate::bootstrap::load_artifact`] /
//! [`crate::state::AppState::served`]); generation code below the
//! binding never asks which format it is running.
//!
//! Operand residency: the container's operands are lowered into the
//! backend's execution form **once, at bind time**
//! ([`Vindex3Runtime::prepare`]), and every request reads that one
//! image. Before that, batch prefill and the decode session each
//! materialised the whole model for themselves — 3.8 s + 3.3 s against
//! 0.13 s of actual decode on a 3 B container, i.e. ~94% of a warm
//! request spent loading a model the server already had.

use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU32, Ordering};
use std::sync::Arc;

use larql_inference::layer_graph::generate::detok::Detokenizer;
use larql_inference::vindex3::{
    continue_session, continue_session_masked, LogitsMask, PreparedVindex3, Vindex3Runtime,
};
use larql_inference::{EosConfig, SamplingConfig};
use larql_kv::CanonicalKvState;
use larql_vindex::format::vindex3::opplan::exec::production::ProductionBackend;
use larql_vindex::tokenizers;

use crate::error::ServerError;
use crate::state::model_id_from_name;

/// Component id a container's text stack is served under.
const SERVED_COMPONENT: &str = "target";

/// One bound VINDEX3 container: the opened runtime plus the serving
/// glue (tokenizer, id). Holds no `ModelWeights` and no `VectorIndex`
/// — structurally, the old inference path is unreachable from here.
pub struct V3Model {
    /// Model ID (derived from the container directory name).
    pub id: String,
    /// Container directory on disk.
    pub path: PathBuf,
    /// The program with its operands already lowered into the
    /// backend's execution form — model lifetime, shared by every
    /// request. Requests contribute only continuation state.
    pub runtime: PreparedVindex3<ProductionBackend>,
    /// Tokenizer for the text-facing API (`tokenizer.json` in the
    /// container directory).
    pub tokenizer: tokenizers::Tokenizer,
    /// The model family the container itself declares (`index.json`).
    ///
    /// The *container* is the only authority here — a V3 binding has no
    /// architecture registry entry, and the model id is just the
    /// directory basename, so id-substring matching answers "plain" for
    /// any container whose folder is not named after its family.
    pub family: String,
    /// The end-of-turn tokens the container declares
    /// (`generation_config.json::eos_token_id`), resolved once at load
    /// through the same authority the CLI's V3 arm uses. The V3 driver
    /// judges EOS on ids alone — it has no tokenizer to re-decode a
    /// special token's surface form the way the V2 loop does — so a
    /// route that starts from the empty built-in set runs every
    /// generation to `max_tokens`. Every V3 route builds its request
    /// EOS on top of this.
    pub eos: EosConfig,
    /// Count of in-flight generations on this container — the V3
    /// counterpart of `LoadedModel.requests_in_flight`, which V3 had
    /// none of before this (no walk-ffn/grid participation to have
    /// borrowed one from). Private: the only way to change it is
    /// [`V3GenerationGuard`], entered once at the
    /// [`generate_v3_request`] choke point every V3 route funnels
    /// through — never a raw field a caller could increment without
    /// the matching decrement. Read it via
    /// [`V3Model::requests_in_flight`].
    requests_in_flight: Arc<AtomicU32>,
}

impl V3Model {
    /// Current count of in-flight generations on this model. One
    /// relaxed atomic load — cheap enough to call from a status
    /// endpoint on every request, no lock. Exact count of what's
    /// genuinely running: the guard that maintains this lives inside
    /// [`generate_v3_request`], so it covers a streaming request's
    /// real decode-loop lifetime, not just however long the async
    /// route handler took to return its (immediate, for a stream) HTTP
    /// response.
    pub fn requests_in_flight(&self) -> u32 {
        self.requests_in_flight.load(Ordering::Relaxed)
    }

    /// The chat template to render conversations with.
    ///
    /// Family first (the container's own declaration), model id second
    /// (the historical heuristic, kept so a renamed container still
    /// resolves), `Plain` last.
    ///
    /// This is not cosmetic. `Plain` ends an assistant turn with a bare
    /// newline, so the last emitted token can merge with the text of
    /// the following turn when the conversation is re-rendered — which
    /// breaks N1's exact-ids-prefix rule at the seam and costs every
    /// chained request its KV resumption. A template that terminates
    /// the turn with an atomic special token does not have that
    /// failure mode.
    pub fn chat_template(&self) -> larql_inference::prompt::ChatTemplate {
        resolve_chat_template(&self.family, &self.id)
    }
}

/// [`V3Model::chat_template`]'s rule, as a free function so it is
/// testable without opening a container.
pub fn resolve_chat_template(family: &str, id: &str) -> larql_inference::prompt::ChatTemplate {
    use larql_inference::prompt::ChatTemplate;
    match ChatTemplate::for_family(family) {
        ChatTemplate::Plain => ChatTemplate::for_model_id(id),
        resolved => resolved,
    }
}

/// Bind a VINDEX3 container for serving: open the component's plan
/// and operand store (refusing closure defects), and load the
/// container's tokenizer — the text API cannot serve ids-only.
pub fn load_v3_model(path: &Path) -> Result<V3Model, Box<dyn std::error::Error + Send + Sync>> {
    let runtime = Vindex3Runtime::open(path, SERVED_COMPONENT, ProductionBackend::new())
        .map_err(|e| format!("open VINDEX3 container: {e}"))?
        .prepare()
        .map_err(|e| format!("prepare VINDEX3 operands: {e}"))?;
    let tokenizer = larql_vindex::load_vindex_tokenizer(path)
        .map_err(|e| format!("VINDEX3 container has no servable tokenizer.json: {e}"))?;
    // The container names itself (`index.model`); the directory name is
    // only the last-resort fallback for a container encoded nameless.
    // The family comes from the same inspection — never a second open
    // whose failure would silently default to "no family".
    let name = match runtime.model_name() {
        "" => path
            .file_name()
            .map(|n| n.to_string_lossy().into_owned())
            .unwrap_or_else(|| "vindex3".to_string()),
        named => named.to_string(),
    };
    let family = runtime.family().to_string();
    let model = V3Model {
        id: model_id_from_name(&name),
        path: path.to_path_buf(),
        runtime,
        tokenizer,
        family,
        eos: EosConfig::from_vindex_dir(path),
        requests_in_flight: Arc::new(AtomicU32::new(0)),
    };
    if matches!(
        model.chat_template(),
        larql_inference::prompt::ChatTemplate::Plain
    ) {
        tracing::warn!(
            model = %model.id,
            family = %model.family,
            "no chat template matches this container: serving with the Plain fallback. \
             Conversations get no chat scaffolding, and N1 KV resumption will usually \
             miss — Plain ends an assistant turn with a bare newline, so its last token \
             re-tokenises differently once the next turn follows."
        );
    }
    Ok(model)
}

/// What one V3 generation produced, shaped for the OpenAI routes:
/// per-token surface text in emission order plus the ids behind them.
pub struct V3Generation {
    pub ids: Vec<u32>,
    pub texts: Vec<String>,
    pub prompt_tokens: usize,
    /// True when generation ended before the token budget — the EOS
    /// signal the routes fold into `finish_reason`.
    pub stopped_early: bool,
    /// How many of the run's prompt tokens were served from a resumed
    /// KV state instead of being re-prefilled (0 on a fresh run).
    pub reused_prompt_tokens: usize,
    /// Wall-clock time of the `prefill_into` call below, in ms. The V3
    /// driver ([`larql_inference::vindex3::generate`]) carries no
    /// timing of its own, so this is measured here, around the two
    /// calls the server already makes — a real number, not one
    /// invented by an observability endpoint after the fact. Feeds
    /// `GET /v1/runtime` via [`crate::runtime_stats::GenerationTally::add_v3`].
    pub prefill_ms: f64,
    /// Wall-clock time of the `continue_session[_masked]` call below,
    /// in ms — the decode-loop counterpart of `prefill_ms`.
    pub decode_ms_total: f64,
}

/// A generation's continuation state, detached from any session so it
/// can outlive the request (N1): the KV plus exactly the token ids the
/// KV has absorbed. `absorbed_ids` can be one short of prompt+emitted —
/// the driver never steps the final emitted token on a budget stop —
/// which is why the ids travel with the state instead of being
/// re-derived by callers.
pub struct V3KvHandoff {
    pub kv: CanonicalKvState,
    pub absorbed_ids: Vec<u32>,
}

/// The SERVE-1 stack for one request: fresh caller-owned continuation
/// state, batch prefill, resume, drive the sampler — streaming each
/// token's `(id, text)` through `on_token` as it is emitted.
pub fn generate_v3(
    model: &V3Model,
    prompt_ids: &[u32],
    max_tokens: usize,
    sampling: SamplingConfig,
    eos: &EosConfig,
    on_token: impl FnMut(u32, &str),
) -> Result<V3Generation, ServerError> {
    generate_v3_resumable(model, prompt_ids, None, max_tokens, sampling, eos, on_token)
        .map(|(generation, _)| generation)
}

/// [`generate_v3`] under a logits mask (N0.6 — tools / structured
/// output on the V3 runtime). `mask_fn` is the V2 constrained driver's
/// contract verbatim — generated-so-far ids plus mutable logits, FSM
/// state in the closure — so one schema-to-mask pipeline serves both
/// runtimes. Constrained runs never resume from a KV handoff (the
/// callers gate that), so this is the fresh-prefill path only.
pub fn generate_v3_constrained(
    model: &V3Model,
    prompt_ids: &[u32],
    max_tokens: usize,
    sampling: SamplingConfig,
    eos: &EosConfig,
    mask_fn: LogitsMask<'_>,
    on_token: impl FnMut(u32, &str),
) -> Result<V3Generation, ServerError> {
    generate_v3_request(
        model,
        prompt_ids,
        None,
        max_tokens,
        sampling,
        eos,
        Some(mask_fn),
        on_token,
    )
    .map(|(generation, _)| generation)
}

/// [`generate_v3`] with KV continuation (N1). When `resume` carries a
/// prior turn's [`V3KvHandoff`] whose `absorbed_ids` are a strict
/// prefix of `prompt_ids`, only the unseen suffix is prefilled — the
/// resumed positions cost nothing. Any mismatch (different rendering,
/// tokenizer seam effects, an exhausted prompt) falls back to a full
/// fresh prefill, so reuse is purely an optimisation: the produced
/// tokens are identical either way, which the V3 serve tests pin.
///
/// The returned handoff holds the state through this generation for
/// the next chain link.
pub fn generate_v3_resumable(
    model: &V3Model,
    prompt_ids: &[u32],
    resume: Option<V3KvHandoff>,
    max_tokens: usize,
    sampling: SamplingConfig,
    eos: &EosConfig,
    on_token: impl FnMut(u32, &str),
) -> Result<(V3Generation, V3KvHandoff), ServerError> {
    generate_v3_request(
        model, prompt_ids, resume, max_tokens, sampling, eos, None, on_token,
    )
}

/// RAII marker for one in-flight V3 generation. Entered once, as the
/// first statement of [`generate_v3_request`] — the single choke
/// point every V3 route (completions, chat, responses; buffered and
/// streaming) funnels through — and dropped on every exit from that
/// function: the normal `Ok` return, any of its `?` early-returns on
/// prefill/session/decode failure, and (were the function to ever
/// panic) unwind too. Nothing at any call site has to remember to
/// decrement — there's exactly one place the counter changes.
struct V3GenerationGuard {
    counter: Arc<AtomicU32>,
}

impl V3GenerationGuard {
    fn enter(counter: Arc<AtomicU32>) -> Self {
        counter.fetch_add(1, Ordering::AcqRel);
        Self { counter }
    }
}

impl Drop for V3GenerationGuard {
    fn drop(&mut self) {
        self.counter.fetch_sub(1, Ordering::AcqRel);
    }
}

/// The one V3 generation body behind the free, resumable, and
/// constrained entry points — and the direct entry for callers that
/// need BOTH knobs (a chained Responses request under a
/// `response_format` constraint resumes AND masks; the two are
/// orthogonal: resume shapes prefill, the mask shapes sampling).
#[allow(clippy::too_many_arguments)]
pub fn generate_v3_request(
    model: &V3Model,
    prompt_ids: &[u32],
    resume: Option<V3KvHandoff>,
    max_tokens: usize,
    sampling: SamplingConfig,
    eos: &EosConfig,
    mask_fn: Option<LogitsMask<'_>>,
    mut on_token: impl FnMut(u32, &str),
) -> Result<(V3Generation, V3KvHandoff), ServerError> {
    // Covers this call's entire body, success or failure — see
    // `V3GenerationGuard`'s doc comment for why entering here (rather
    // than in each route handler) is load-bearing for streaming
    // callers.
    let _gen_guard = V3GenerationGuard::enter(Arc::clone(&model.requests_in_flight));

    // A handoff is resumable only when the new prompt extends exactly
    // what the KV already absorbed.
    let resumed = resume.filter(|h| {
        !h.absorbed_ids.is_empty()
            && h.absorbed_ids.len() < prompt_ids.len()
            && prompt_ids.starts_with(&h.absorbed_ids)
    });
    let (mut kv, reused_prompt_tokens) = match resumed {
        Some(h) => (h.kv, h.absorbed_ids.len()),
        None => (CanonicalKvState::new(), 0),
    };

    let prefill_start = std::time::Instant::now();
    let prefill_logits = model
        .runtime
        .prefill_into(&prompt_ids[reused_prompt_tokens..], &mut kv)
        .map_err(|e| ServerError::Internal(format!("v3 prefill: {e}")))?;
    let prefill_ms = crate::state::elapsed_ms(prefill_start);
    let mut session = model
        .runtime
        .session_with_kv(&mut kv)
        .map_err(|e| ServerError::Internal(format!("v3 session: {e}")))?;

    let mut detok = Detokenizer::new(&model.tokenizer);
    detok.seed(prompt_ids);
    let mut texts = Vec::new();
    let mut emit = |id: u32| {
        let text = detok.push(id);
        on_token(id, &text);
        texts.push(text);
    };
    let decode_start = std::time::Instant::now();
    let result = match mask_fn {
        Some(mask_fn) => continue_session_masked(
            &mut session,
            prefill_logits,
            max_tokens,
            sampling,
            eos,
            mask_fn,
            &mut emit,
        ),
        None => continue_session(
            &mut session,
            prefill_logits,
            max_tokens,
            sampling,
            eos,
            &mut emit,
        ),
    }
    .map_err(|e| ServerError::Internal(format!("v3 decode: {e}")))?;
    let decode_ms_total = crate::state::elapsed_ms(decode_start);
    drop(session);

    // The KV's logical position says exactly how many of
    // prompt + emitted it absorbed (the driver never steps the final
    // emitted token on a budget stop).
    let absorbed_len = larql_vindex::format::vindex3::opplan::exec::kv::KvState::position(&kv);
    let mut absorbed_ids = Vec::with_capacity(absorbed_len);
    absorbed_ids.extend_from_slice(prompt_ids);
    absorbed_ids.extend_from_slice(&result.tokens);
    absorbed_ids.truncate(absorbed_len);

    let stopped_early = result.tokens.len() < max_tokens;
    Ok((
        V3Generation {
            ids: result.tokens,
            texts,
            prompt_tokens: prompt_ids.len(),
            stopped_early,
            reused_prompt_tokens,
            prefill_ms,
            decode_ms_total,
        },
        V3KvHandoff { kv, absorbed_ids },
    ))
}
