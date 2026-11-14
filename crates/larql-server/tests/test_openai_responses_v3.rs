//! `/v1/responses` served by a VINDEX3 runtime.
//!
//! Mirrors the VI3-SERVE-1 gate shape (`test_vindex3_serve.rs`): arm A
//! is the direct runtime stack (`Vindex3Runtime` → `CanonicalKvState`
//! → `prefill_into` → `session_with_kv` → `continue_session`) fed the
//! *rendered* prompt the endpoint would build; arm B is the HTTP
//! request through `/v1/responses`. The gate demands the envelope's
//! output text equal arm A's emission text exactly (greedy decoding on
//! both arms), plus the V3-specific contracts: streaming deltas (the
//! V3 token callback fires per token, unlike the CPU-Q4K V2 path),
//! `previous_response_id` chaining, and a clean 400 for tools.

mod common;

use std::path::Path;
use std::sync::Arc;

use larql_inference::layer_graph::generate::detok::Detokenizer;
use larql_inference::test_utils::synthetic_tokenizer_json;
use larql_inference::vindex3::{continue_session, Vindex3Runtime};
use larql_inference::{EosConfig, SamplingConfig};
use larql_kv::CanonicalKvState;
use larql_server::bootstrap::{load_artifact, LoadVindexOptions, LoadedArtifact};
use larql_server::state::AppState;
use larql_vindex::format::vindex3::fixtures::{
    encode_fixture_container, miniature_glimmer, G_VOCAB,
};
use larql_vindex::format::vindex3::opplan::exec::production::ProductionBackend;

use axum::body::Body;
use axum::http::{header, Request, StatusCode};
use tower::ServiceExt;

const NEW_TOKENS: usize = 16;
/// The user turn as sent over the wire. The endpoint renders it through
/// the model's chat template before tokenising.
const USER_INPUT: &str = "[3]";
const COMPONENT: &str = "target";

fn v3_container() -> tempfile::TempDir {
    let checkpoint = tempfile::tempdir().unwrap();
    let container = tempfile::tempdir().unwrap();
    encode_fixture_container(
        miniature_glimmer,
        checkpoint.path(),
        container.path(),
        "serve-fixture",
    );
    std::fs::write(
        container.path().join("tokenizer.json"),
        synthetic_tokenizer_json(G_VOCAB),
    )
    .unwrap();
    common::chat_template::write_transcript_template(container.path());
    container
}

fn v3_state(container: &Path) -> Arc<AppState> {
    v3_state_with_kv_entries(container, larql_server::response_kv::DEFAULT_MAX_ENTRIES)
}

/// [`v3_state`] with an explicit KV continuation cache size — 0
/// disables resumption so parity tests can compare cached vs
/// stateless chains.
fn v3_state_with_kv_entries(container: &Path, kv_entries: usize) -> Arc<AppState> {
    let artifact =
        load_artifact(&container.to_string_lossy(), LoadVindexOptions::default()).unwrap();
    let v3 = match artifact {
        LoadedArtifact::V3(m) => Arc::new(*m),
        LoadedArtifact::V2(_) => panic!("a VINDEX3 container must bind as V3"),
    };
    Arc::new(AppState {
        model_set: std::sync::RwLock::new(larql_server::state::ModelSet {
            models: Vec::new(),
            v3_models: vec![v3],
            aliases: Default::default(),
        }),
        router_topology: larql_server::state::RouterTopology::SingleModel,
        lifecycle: std::sync::Mutex::new(larql_server::state::LifecycleState::Idle),
        started_at: std::time::Instant::now(),
        requests_served: std::sync::atomic::AtomicU64::new(0),
        api_key: None,
        sessions: larql_server::session::SessionManager::new(3600),
        describe_cache: larql_server::cache::DescribeCache::new(0),
        infer_timeout: std::time::Duration::from_secs(60),
        patch_sources: Default::default(),
        responses: larql_server::response_store::ResponseStore::new(),
        v3_kv: larql_server::response_kv::ResponseKvCache::new(
            kv_entries,
            larql_server::response_kv::DEFAULT_TTL_SECS,
        ),
        runtime: Arc::new(larql_server::runtime_stats::RuntimeRecorder::new()),
    })
}

/// Arm A: direct runtime, fed the prompt ids the shared derivation
/// (`render_conversation` through the container's own template, then
/// `encode`) gives the user turn. Returns the concatenated emission text.
fn direct_arm_text(container: &Path, max_tokens: usize) -> String {
    let runtime = Vindex3Runtime::open(container, COMPONENT, ProductionBackend::new()).unwrap();
    let tokenizer = larql_vindex::load_vindex_tokenizer(container).unwrap();
    let prompt_ids: Vec<u32> = larql_inference::chat::render_conversation(
        container,
        runtime.family(),
        &[larql_inference::chat::ChatTurn::user(USER_INPUT)],
    )
    .unwrap()
    .encode(
        &tokenizer,
        larql_inference::vindex3::declared_prompt_bos(container),
    )
    .unwrap();
    assert!(!prompt_ids.is_empty(), "rendered prompt must tokenise");

    let mut kv = CanonicalKvState::new();
    let prefill = runtime.prefill_into(&prompt_ids, &mut kv).unwrap();
    let mut session = runtime.session_with_kv(&mut kv).unwrap();
    let mut detok = Detokenizer::new(&tokenizer);
    detok.seed(&prompt_ids);
    let mut text = String::new();
    continue_session(
        &mut session,
        prefill,
        max_tokens,
        SamplingConfig::greedy(),
        &EosConfig::builtin(),
        |id| {
            text.push_str(&detok.push(id));
        },
    )
    .unwrap();
    text
}

async fn post_responses(app: &axum::Router, body: serde_json::Value) -> axum::http::Response<Body> {
    app.clone()
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/v1/responses")
                .header(header::CONTENT_TYPE, "application/json")
                .body(Body::from(serde_json::to_vec(&body).unwrap()))
                .unwrap(),
        )
        .await
        .unwrap()
}

async fn drain(resp: axum::http::Response<Body>) -> (StatusCode, serde_json::Value) {
    let status = resp.status();
    let bytes = axum::body::to_bytes(resp.into_body(), usize::MAX)
        .await
        .unwrap();
    let json = serde_json::from_slice(&bytes)
        .unwrap_or_else(|_| serde_json::json!({"raw": String::from_utf8_lossy(&bytes)}));
    (status, json)
}

fn output_text(envelope: &serde_json::Value) -> String {
    envelope["output"]
        .as_array()
        .into_iter()
        .flatten()
        .filter(|item| item["type"] == "message")
        .flat_map(|item| item["content"].as_array().into_iter().flatten())
        .filter(|part| part["type"] == "output_text")
        .filter_map(|part| part["text"].as_str())
        .collect()
}

#[tokio::test]
async fn v3_responses_match_the_direct_runtime_text_for_text() {
    let container = v3_container();
    let expected = direct_arm_text(container.path(), NEW_TOKENS);
    assert!(!expected.is_empty(), "fixture must emit text");

    let state = v3_state(container.path());
    assert!(
        state.models_snapshot().models.is_empty(),
        "no V2 model may exist"
    );
    let app = larql_server::routes::single_model_router(state);
    let resp = post_responses(
        &app,
        serde_json::json!({"input": USER_INPUT, "max_output_tokens": NEW_TOKENS}),
    )
    .await;
    let (status, v) = drain(resp).await;
    assert_eq!(status, StatusCode::OK, "{v}");
    assert_eq!(output_text(&v), expected, "{v}");
    assert!(v["usage"]["input_tokens"].as_u64().unwrap() > 0, "{v}");
    assert_eq!(
        v["usage"]["output_tokens"].as_u64().unwrap() as usize,
        NEW_TOKENS,
        "{v}"
    );
}

#[tokio::test]
async fn v3_responses_streaming_carries_per_token_deltas() {
    let container = v3_container();
    let expected = direct_arm_text(container.path(), NEW_TOKENS);
    let app = larql_server::routes::single_model_router(v3_state(container.path()));
    let resp = post_responses(
        &app,
        serde_json::json!({
            "input": USER_INPUT,
            "max_output_tokens": NEW_TOKENS,
            "stream": true,
        }),
    )
    .await;
    assert_eq!(resp.status(), StatusCode::OK);
    let bytes = axum::body::to_bytes(resp.into_body(), usize::MAX)
        .await
        .unwrap();
    let body = String::from_utf8_lossy(&bytes);

    // The V3 runtime invokes the token callback per token, so unlike
    // the CPU-Q4K V2 fixture the delta events must be present — and
    // their concatenation must reproduce the direct arm's text.
    let deltas: String = body
        .lines()
        .filter_map(|line| line.strip_prefix("data: "))
        .filter(|data| *data != "[DONE]")
        .filter_map(|data| serde_json::from_str::<serde_json::Value>(data).ok())
        .filter(|v| v["type"] == "response.output_text.delta")
        .filter_map(|v| v["delta"].as_str().map(str::to_string))
        .collect();
    assert_eq!(deltas, expected, "streamed deltas must match direct arm");
    assert!(body.contains("event: response.completed"), "{body}");
    assert!(body.contains("[DONE]"), "{body}");
}

#[tokio::test]
async fn v3_responses_chain_via_previous_response_id() {
    let container = v3_container();
    let app = larql_server::routes::single_model_router(v3_state(container.path()));

    let resp = post_responses(
        &app,
        serde_json::json!({"input": USER_INPUT, "max_output_tokens": 4}),
    )
    .await;
    let (status, first) = drain(resp).await;
    assert_eq!(status, StatusCode::OK, "{first}");
    let id = first["id"].as_str().unwrap().to_string();

    let resp = post_responses(
        &app,
        serde_json::json!({
            "input": "[5]",
            "previous_response_id": id,
            "max_output_tokens": 4,
        }),
    )
    .await;
    let (status, second) = drain(resp).await;
    assert_eq!(status, StatusCode::OK, "{second}");
    assert_eq!(second["previous_response_id"], id.as_str(), "{second}");
    assert!(!output_text(&second).is_empty(), "{second}");
}

#[tokio::test]
async fn v3_responses_tools_fail_closed_on_this_vocabulary() {
    // N0.6: tools now RUN on V3 through the shared grammar mask. This
    // fixture's tokenizer (no pre-tokenizer, no JSON lexemes) cannot
    // spell any grammar-valid token, so the mask rejects every first
    // candidate and the request fails closed — never ungrammatical
    // output. The affirmative tools-on-V3 gates live in
    // `test_openai_v3_tools.rs` on a JSON-capable vocabulary.
    let container = v3_container();
    let app = larql_server::routes::single_model_router(v3_state(container.path()));
    let resp = post_responses(
        &app,
        serde_json::json!({
            "input": USER_INPUT,
            "tools": [{"type": "function", "name": "f", "parameters": {"type": "object"}}],
        }),
    )
    .await;
    let (status, v) = drain(resp).await;
    assert!(status.is_server_error(), "{v}");
}

#[tokio::test]
async fn v3_model_retrieve_reports_generation_3() {
    let container = v3_container();
    let state = v3_state(container.path());
    let model_id = state.models_snapshot().v3_models[0].id.clone();
    let app = larql_server::routes::single_model_router(state);
    let resp = app
        .clone()
        .oneshot(
            Request::builder()
                .method("GET")
                .uri(format!("/v1/models/{model_id}"))
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();
    let (status, v) = drain(resp).await;
    assert_eq!(status, StatusCode::OK, "{v}");
    assert_eq!(v["id"], model_id.as_str(), "{v}");
    assert_eq!(v["generation"], 3, "{v}");
}

// ── /v1/chat/completions on the same V3 runtime ─────────────────────────────

async fn post_chat(app: &axum::Router, body: serde_json::Value) -> axum::http::Response<Body> {
    app.clone()
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/v1/chat/completions")
                .header(header::CONTENT_TYPE, "application/json")
                .body(Body::from(serde_json::to_vec(&body).unwrap()))
                .unwrap(),
        )
        .await
        .unwrap()
}

#[tokio::test]
async fn v3_chat_matches_the_direct_runtime_text_for_text() {
    let container = v3_container();
    let expected = direct_arm_text(container.path(), NEW_TOKENS);
    let app = larql_server::routes::single_model_router(v3_state(container.path()));
    let resp = post_chat(
        &app,
        serde_json::json!({
            "messages": [{"role": "user", "content": USER_INPUT}],
            "max_tokens": NEW_TOKENS,
        }),
    )
    .await;
    let (status, v) = drain(resp).await;
    assert_eq!(status, StatusCode::OK, "{v}");
    assert_eq!(
        v["choices"][0]["message"]["content"],
        expected.as_str(),
        "{v}"
    );
    assert_eq!(v["choices"][0]["message"]["role"], "assistant", "{v}");
    assert!(v["usage"]["prompt_tokens"].as_u64().unwrap() > 0, "{v}");
    assert_eq!(
        v["usage"]["completion_tokens"].as_u64().unwrap() as usize,
        NEW_TOKENS,
        "{v}"
    );
}

#[tokio::test]
async fn v3_chat_streaming_content_chunks_match_direct_arm() {
    let container = v3_container();
    let expected = direct_arm_text(container.path(), NEW_TOKENS);
    let app = larql_server::routes::single_model_router(v3_state(container.path()));
    let resp = post_chat(
        &app,
        serde_json::json!({
            "messages": [{"role": "user", "content": USER_INPUT}],
            "max_tokens": NEW_TOKENS,
            "stream": true,
        }),
    )
    .await;
    assert_eq!(resp.status(), StatusCode::OK);
    let bytes = axum::body::to_bytes(resp.into_body(), usize::MAX)
        .await
        .unwrap();
    let body = String::from_utf8_lossy(&bytes);
    assert!(body.contains("[DONE]"), "{body}");

    let chunks: Vec<serde_json::Value> = body
        .lines()
        .filter_map(|line| line.strip_prefix("data: "))
        .filter(|data| *data != "[DONE]")
        .map(|data| serde_json::from_str(data).expect("chunk is JSON"))
        .collect();
    assert_eq!(
        chunks[0]["choices"][0]["delta"]["role"], "assistant",
        "{body}"
    );
    let content: String = chunks
        .iter()
        .filter_map(|c| c["choices"][0]["delta"]["content"].as_str())
        .collect();
    assert_eq!(content, expected, "{body}");
    let last = chunks.last().unwrap();
    let finish = last["choices"][0]["finish_reason"].as_str().unwrap();
    assert!(finish == "stop" || finish == "length", "{body}");
}

#[tokio::test]
async fn v3_chat_tools_fail_closed_on_this_vocabulary() {
    // See `v3_responses_tools_fail_closed_on_this_vocabulary` — same
    // contract on the chat surface.
    let container = v3_container();
    let app = larql_server::routes::single_model_router(v3_state(container.path()));
    let resp = post_chat(
        &app,
        serde_json::json!({
            "messages": [{"role": "user", "content": USER_INPUT}],
            "tools": [{"type": "function",
                       "function": {"name": "f", "parameters": {"type": "object"}}}],
        }),
    )
    .await;
    let (status, v) = drain(resp).await;
    assert!(status.is_server_error(), "{v}");
}

#[tokio::test]
async fn v3_chat_stop_string_trims_deterministically() {
    // V3 emits real per-token text (unlike the CPU-Q4K V2 path), so a
    // stop string built from the direct arm's own output is GUARANTEED
    // to fire: use the first character of the expected emission. The
    // endpoint must trim the text at the stop and report finish=stop.
    let container = v3_container();
    let expected = direct_arm_text(container.path(), NEW_TOKENS);
    let stop = expected
        .chars()
        .next()
        .expect("nonempty emission")
        .to_string();
    let expected_trimmed = &expected[..expected.find(&stop).unwrap()];

    let app = larql_server::routes::single_model_router(v3_state(container.path()));
    let resp = post_chat(
        &app,
        serde_json::json!({
            "messages": [{"role": "user", "content": USER_INPUT}],
            "max_tokens": NEW_TOKENS,
            "stop": [stop],
            "logprobs": true,
        }),
    )
    .await;
    let (status, v) = drain(resp).await;
    assert_eq!(status, StatusCode::OK, "{v}");
    assert_eq!(v["choices"][0]["finish_reason"], "stop", "{v}");
    assert_eq!(
        v["choices"][0]["message"]["content"], expected_trimmed,
        "{v}"
    );
    assert!(
        v["choices"][0]["logprobs"].is_object(),
        "logprobs requested but missing: {v}"
    );
}

#[tokio::test]
async fn v3_chat_streaming_stop_string_halts_the_stream() {
    let container = v3_container();
    let expected = direct_arm_text(container.path(), NEW_TOKENS);
    let stop = expected
        .chars()
        .next()
        .expect("nonempty emission")
        .to_string();

    let app = larql_server::routes::single_model_router(v3_state(container.path()));
    let resp = post_chat(
        &app,
        serde_json::json!({
            "messages": [{"role": "user", "content": USER_INPUT}],
            "max_tokens": NEW_TOKENS,
            "stream": true,
            "stop": [stop],
        }),
    )
    .await;
    assert_eq!(resp.status(), StatusCode::OK);
    let bytes = axum::body::to_bytes(resp.into_body(), usize::MAX)
        .await
        .unwrap();
    let body = String::from_utf8_lossy(&bytes);
    assert!(body.contains("[DONE]"), "{body}");
    let chunks: Vec<serde_json::Value> = body
        .lines()
        .filter_map(|line| line.strip_prefix("data: "))
        .filter(|data| *data != "[DONE]")
        .map(|data| serde_json::from_str(data).expect("chunk is JSON"))
        .collect();
    let last = chunks.last().unwrap();
    assert_eq!(last["choices"][0]["finish_reason"], "stop", "{body}");
    // The stop halts emission: fewer content chunks than a full run
    // (role chunk + at most a couple of content chunks + final chunk).
    let content_chunks = chunks
        .iter()
        .filter(|c| c["choices"][0]["delta"]["content"].is_string())
        .count();
    assert!(content_chunks < NEW_TOKENS, "stop did not halt: {body}");
}

#[tokio::test]
async fn v3_responses_stop_string_trims_deterministically() {
    // Same construction as the chat stop test: the stop string is the
    // first character of the direct arm's own emission, so it must
    // fire — covering the engine's V3 halt + trim path.
    let container = v3_container();
    let expected = direct_arm_text(container.path(), NEW_TOKENS);
    let stop = expected
        .chars()
        .next()
        .expect("nonempty emission")
        .to_string();
    let expected_trimmed = expected[..expected.find(&stop).unwrap()].to_string();

    let app = larql_server::routes::single_model_router(v3_state(container.path()));
    let resp = post_responses(
        &app,
        serde_json::json!({
            "input": USER_INPUT,
            "max_output_tokens": NEW_TOKENS,
            "stop": [stop],
        }),
    )
    .await;
    let (status, v) = drain(resp).await;
    assert_eq!(status, StatusCode::OK, "{v}");
    assert_eq!(v["status"], "completed", "stop implies completed: {v}");
    assert_eq!(output_text(&v), expected_trimmed, "{v}");
}

// ── N1: KV-resident continuation ─────────────────────────────────────────────

/// Drive a two-turn chain against `state` and return
/// `(second_output_text, second_cached_tokens)`.
async fn run_chain(state: Arc<AppState>) -> (String, u64) {
    let app = larql_server::routes::single_model_router(state);
    let resp = post_responses(
        &app,
        serde_json::json!({"input": USER_INPUT, "max_output_tokens": NEW_TOKENS}),
    )
    .await;
    let (status, first) = drain(resp).await;
    assert_eq!(status, StatusCode::OK, "{first}");
    let id = first["id"].as_str().unwrap().to_string();

    let resp = post_responses(
        &app,
        serde_json::json!({
            "input": "[5]",
            "previous_response_id": id,
            "max_output_tokens": NEW_TOKENS,
        }),
    )
    .await;
    let (status, second) = drain(resp).await;
    assert_eq!(status, StatusCode::OK, "{second}");
    let cached = second["usage"]["input_tokens_details"]["cached_tokens"]
        .as_u64()
        .expect("usage carries input_tokens_details.cached_tokens");
    (output_text(&second), cached)
}

#[tokio::test]
async fn v3_chained_turn_with_kv_cache_matches_stateless_replay() {
    // The N1 fallback-parity gate: the SAME chain on the same
    // container, once with the KV continuation cache and once fully
    // stateless, must produce byte-identical second-turn text (greedy
    // decoding). On THIS fixture the resume prefix can never match —
    // the synthetic tokenizer is WordLevel with no pre-tokenizer, so a
    // whole rendered prompt encodes to one [UNK] id and a chained
    // prompt cannot extend the absorbed ids — which makes this exactly
    // the fallback path a real model hits on re-tokenization drift:
    // cached_tokens stays 0 on both arms and the text must not differ.
    // The resumption mechanism itself is pinned at the ids level in
    // `v3_resumed_decode_is_bit_identical_to_fresh_prefill`.
    let container = v3_container();
    let (with_kv_text, with_kv_cached) =
        run_chain(v3_state_with_kv_entries(container.path(), 4)).await;
    let (stateless_text, stateless_cached) =
        run_chain(v3_state_with_kv_entries(container.path(), 0)).await;

    assert_eq!(stateless_cached, 0, "disabled cache must never resume");
    assert_eq!(
        with_kv_cached, 0,
        "this fixture's tokenizer cannot produce a resumable prefix; \
         a hit here means the fixture changed — rewrite this gate"
    );
    assert_eq!(
        with_kv_text, stateless_text,
        "the KV-cache arm's fallback changed the produced text"
    );
}

#[test]
fn v3_resumed_decode_is_bit_identical_to_fresh_prefill() {
    // The N1 resumption gate, at the seam where the contract lives
    // (token ids — no template/tokenizer in the loop): generate turn 1,
    // extend its absorbed ids into a turn-2 prompt, and decode turn 2
    // twice — resumed from the handoff vs a fresh full prefill. The
    // two runs must emit bit-identical token streams, and the resumed
    // run must report every absorbed position as reused.
    let container = v3_container();
    let artifact = load_artifact(
        &container.path().to_string_lossy(),
        LoadVindexOptions::default(),
    )
    .unwrap();
    let model = match artifact {
        LoadedArtifact::V3(m) => *m,
        LoadedArtifact::V2(_) => panic!("a VINDEX3 container must bind as V3"),
    };
    let sampling = SamplingConfig::default();
    let eos = EosConfig::default();

    let turn1_ids: Vec<u32> = vec![1, 2, 3];
    let (gen1, handoff) = larql_server::vindex3::generate_v3_resumable(
        &model,
        &turn1_ids,
        None,
        4,
        sampling,
        &eos,
        |_, _| {},
    )
    .expect("turn 1 generates");
    assert_eq!(gen1.reused_prompt_tokens, 0);
    assert!(
        handoff.absorbed_ids.starts_with(&turn1_ids),
        "absorbed ids must begin with the prompt"
    );

    let mut turn2_ids = handoff.absorbed_ids.clone();
    turn2_ids.extend_from_slice(&[4, 5]);

    let (resumed, _) = larql_server::vindex3::generate_v3_resumable(
        &model,
        &turn2_ids,
        Some(handoff),
        4,
        sampling,
        &eos,
        |_, _| {},
    )
    .expect("resumed turn 2 generates");
    let (fresh, _) = larql_server::vindex3::generate_v3_resumable(
        &model,
        &turn2_ids,
        None,
        4,
        sampling,
        &eos,
        |_, _| {},
    )
    .expect("fresh turn 2 generates");

    assert!(
        resumed.reused_prompt_tokens > 0,
        "the resumed run must actually skip absorbed positions"
    );
    assert_eq!(fresh.reused_prompt_tokens, 0);
    assert_eq!(
        resumed.ids, fresh.ids,
        "resumed decode diverged from fresh prefill"
    );
    assert_eq!(resumed.texts, fresh.texts);
}

#[tokio::test]
async fn v3_store_false_retains_no_kv_state() {
    let container = v3_container();
    let state = v3_state_with_kv_entries(container.path(), 4);
    let app = larql_server::routes::single_model_router(Arc::clone(&state));
    let resp = post_responses(
        &app,
        serde_json::json!({
            "input": USER_INPUT,
            "max_output_tokens": 4,
            "store": false,
        }),
    )
    .await;
    let (status, v) = drain(resp).await;
    assert_eq!(status, StatusCode::OK, "{v}");
    assert!(
        state.v3_kv.is_empty(),
        "store:false must not retain continuation state"
    );
}

#[tokio::test]
async fn v3_stored_response_retains_kv_state_for_the_next_link() {
    let container = v3_container();
    let state = v3_state_with_kv_entries(container.path(), 4);
    let app = larql_server::routes::single_model_router(Arc::clone(&state));
    let resp = post_responses(
        &app,
        serde_json::json!({"input": USER_INPUT, "max_output_tokens": 4}),
    )
    .await;
    let (status, v) = drain(resp).await;
    assert_eq!(status, StatusCode::OK, "{v}");
    assert_eq!(state.v3_kv.len(), 1, "stored response must retain its KV");

    // Chaining takes the entry (take-once) and retains a new one under
    // the new response id.
    let id = v["id"].as_str().unwrap().to_string();
    let resp = post_responses(
        &app,
        serde_json::json!({
            "input": "[5]",
            "previous_response_id": id,
            "max_output_tokens": 4,
        }),
    )
    .await;
    let (status, second) = drain(resp).await;
    assert_eq!(status, StatusCode::OK, "{second}");
    assert_eq!(state.v3_kv.len(), 1, "old entry consumed, new one retained");
    let model_id = state.models_snapshot().v3_models[0].id.clone();
    assert!(
        state
            .v3_kv
            .take(second["id"].as_str().unwrap(), &model_id)
            .is_some(),
        "the retained entry must key on the NEW response id"
    );
}
