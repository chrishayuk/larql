//! VI3-SERVE-1 gates: a VINDEX3 container over the normal server API.
//!
//! The authoritative arm (A) is the direct runtime stack —
//! `Vindex3Runtime` → `CanonicalKvState` → `prefill_into` →
//! `session_with_kv` → `continue_session` — assembled by hand in this
//! file. Arm B is an HTTP request through the server's model registry
//! into `/v1/completions`. The gate demands the streamed tokens match
//! arm A token-for-token: same first token, same ordering, same
//! count, same finish behaviour.
//!
//! The negative control pins the architectural regression this rung
//! exists to prevent: the served container **cannot** be opened by
//! the V2 path at all (`load_vindex_config` refuses the generation,
//! `load_single_vindex` errors), and the serving state holds zero V2
//! models while requests succeed — so the server provably did not
//! reconstitute an old-style model behind the scenes.

mod common;

use std::path::Path;
use std::sync::Arc;

use larql_inference::layer_graph::generate::detok::Detokenizer;
use larql_inference::test_utils::synthetic_tokenizer_json;
use larql_inference::vindex3::{continue_session, Vindex3Runtime};
use larql_inference::{EosConfig, SamplingConfig};
use larql_kv::CanonicalKvState;
use larql_server::bootstrap::{
    load_artifact, load_single_vindex, LoadVindexOptions, LoadedArtifact,
};
use larql_server::state::AppState;
use larql_server::vindex3::generate_v3;
use larql_vindex::format::load::load_vindex_config;
use larql_vindex::format::vindex3::fixtures::{
    encode_fixture_container, miniature_glimmer, G_VOCAB,
};
use larql_vindex::format::vindex3::opplan::exec::production::ProductionBackend;

use axum::body::Body;
use axum::http::{Request, StatusCode};
use tower::ServiceExt;

const NEW_TOKENS: usize = 16;
const PROMPT: &str = "[3]";
const COMPONENT: &str = "target";

/// Encode the miniature container and give it a servable tokenizer
/// (`[N]` ↔ id N, no pre-tokenizer).
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
    container
}

/// Arm A: the direct runtime stack, by hand. Returns per-token
/// `(id, text)` pairs in emission order.
fn direct_arm(container: &Path, max_tokens: usize) -> Vec<(u32, String)> {
    let runtime = Vindex3Runtime::open(container, COMPONENT, ProductionBackend::new()).unwrap();
    let tokenizer = larql_vindex::load_vindex_tokenizer(container).unwrap();
    let prompt_ids: Vec<u32> = tokenizer.encode(PROMPT, true).unwrap().get_ids().to_vec();
    assert!(!prompt_ids.is_empty());

    let mut kv = CanonicalKvState::new();
    let prefill = runtime.prefill_into(&prompt_ids, &mut kv).unwrap();
    let mut session = runtime.session_with_kv(&mut kv).unwrap();
    let mut detok = Detokenizer::new(&tokenizer);
    detok.seed(&prompt_ids);
    let mut pairs = Vec::new();
    continue_session(
        &mut session,
        prefill,
        max_tokens,
        SamplingConfig::greedy(),
        &EosConfig::builtin(),
        |id| {
            let text = detok.push(id);
            pairs.push((id, text));
        },
    )
    .unwrap();
    pairs
}

/// A serving state holding ONLY the V3 model — bound through the same
/// `load_artifact` the real bootstrap uses.
fn v3_state(container: &Path) -> Arc<AppState> {
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
        }),
        router_topology: larql_server::state::RouterTopology::SingleModel,
        lifecycle: std::sync::Mutex::new(larql_server::state::LifecycleState::Idle),
        started_at: std::time::Instant::now(),
        requests_served: std::sync::atomic::AtomicU64::new(0),
        api_key: None,
        sessions: larql_server::session::SessionManager::new(3600),
        describe_cache: larql_server::cache::DescribeCache::new(0),
        infer_timeout: std::time::Duration::from_secs(60),
        responses: larql_server::response_store::ResponseStore::new(),
        v3_kv: larql_server::response_kv::ResponseKvCache::new(
            larql_server::response_kv::DEFAULT_MAX_ENTRIES,
            larql_server::response_kv::DEFAULT_TTL_SECS,
        ),
        runtime: Arc::new(larql_server::runtime_stats::RuntimeRecorder::new()),
    })
}

/// Parse an SSE body into its JSON data chunks (excluding `[DONE]`).
fn sse_chunks(body: &str) -> Vec<serde_json::Value> {
    body.lines()
        .filter_map(|line| line.strip_prefix("data: "))
        .filter(|data| *data != "[DONE]")
        .map(|data| serde_json::from_str(data).expect("SSE chunk is JSON"))
        .collect()
}

#[tokio::test]
async fn v3_stream_over_the_api_matches_the_direct_runtime_token_for_token() {
    let container = v3_container();
    let expected = direct_arm(container.path(), NEW_TOKENS);
    assert_eq!(expected.len(), NEW_TOKENS, "fixture must fill the budget");

    let state = v3_state(container.path());
    assert!(
        state.models_snapshot().models.is_empty(),
        "no V2 model may exist while V3 serves"
    );
    let app = larql_server::routes::single_model_router(state);
    let resp = common::post_json(
        app,
        "/v1/completions",
        serde_json::json!({
            "prompt": PROMPT,
            "max_tokens": NEW_TOKENS,
            "stream": true,
        }),
    )
    .await;
    assert_eq!(resp.status(), axum::http::StatusCode::OK);
    let content_type = resp
        .headers()
        .get("content-type")
        .and_then(|v| v.to_str().ok())
        .unwrap_or_default()
        .to_string();
    assert!(content_type.contains("event-stream"), "{content_type}");

    let bytes = axum::body::to_bytes(resp.into_body(), usize::MAX)
        .await
        .unwrap();
    let body = String::from_utf8(bytes.to_vec()).unwrap();
    assert!(body.contains("[DONE]"));

    let chunks = sse_chunks(&body);
    // One chunk per token plus the final finish_reason chunk.
    assert_eq!(chunks.len(), NEW_TOKENS + 1, "chunk count");
    for (chunk, (_, text)) in chunks[..NEW_TOKENS].iter().zip(&expected) {
        assert_eq!(chunk["choices"][0]["text"], text.as_str());
        assert_eq!(
            chunk["choices"][0]["finish_reason"],
            serde_json::Value::Null
        );
        assert_eq!(chunk["object"], "text_completion");
    }
    // Same first token, same ordering (asserted above), same EOS
    // behaviour: the greedy run never hits a stop, so "length".
    assert_eq!(chunks[NEW_TOKENS]["choices"][0]["finish_reason"], "length");
}

#[tokio::test]
async fn v3_buffered_response_matches_the_direct_runtime() {
    let container = v3_container();
    let expected = direct_arm(container.path(), NEW_TOKENS);
    let expected_text: String = expected.iter().map(|(_, t)| t.as_str()).collect();

    let app = larql_server::routes::single_model_router(v3_state(container.path()));
    let resp = common::post_json(
        app,
        "/v1/completions",
        serde_json::json!({"prompt": PROMPT, "max_tokens": NEW_TOKENS}),
    )
    .await;
    assert_eq!(resp.status(), axum::http::StatusCode::OK);
    let json = common::body_json(resp.into_body()).await;
    assert_eq!(json["choices"][0]["text"], expected_text.as_str());
    assert_eq!(json["choices"][0]["finish_reason"], "length");
    assert_eq!(json["usage"]["prompt_tokens"], 1);
    assert_eq!(json["usage"]["completion_tokens"], NEW_TOKENS);
    assert_eq!(json["object"], "text_completion");
}

#[tokio::test]
async fn v3_stream_honours_client_stop_strings() {
    let container = v3_container();
    let expected = direct_arm(container.path(), NEW_TOKENS);
    // Stop on the third token's surface text: chunks 1..=3 stream,
    // then the final chunk closes with "stop".
    let stop = expected[2].1.clone();
    assert!(!stop.trim().is_empty(), "stop token must have surface text");

    let app = larql_server::routes::single_model_router(v3_state(container.path()));
    let resp = common::post_json(
        app,
        "/v1/completions",
        serde_json::json!({
            "prompt": PROMPT,
            "max_tokens": NEW_TOKENS,
            "stream": true,
            "stop": stop,
        }),
    )
    .await;
    assert_eq!(resp.status(), axum::http::StatusCode::OK);
    let bytes = axum::body::to_bytes(resp.into_body(), usize::MAX)
        .await
        .unwrap();
    let body = String::from_utf8(bytes.to_vec()).unwrap();
    let chunks = sse_chunks(&body);
    assert_eq!(chunks.len(), 3 + 1, "stream must stop after the match");
    assert_eq!(chunks[3]["choices"][0]["finish_reason"], "stop");
}

/// The negative control: the container this server is happily serving
/// CANNOT be opened by the V2 path at all — so V3 serving is provably
/// not "reconstitute an old model, run old inference" in disguise.
#[test]
fn the_served_container_cannot_take_the_v2_path() {
    let container = v3_container();

    let config = load_vindex_config(container.path());
    assert!(
        config.is_err(),
        "V2 config loader must refuse a V3 container"
    );

    let v2_load = load_single_vindex(
        &container.path().to_string_lossy(),
        LoadVindexOptions::default(),
    );
    assert!(
        v2_load.is_err(),
        "V2 model loader must refuse a V3 container"
    );

    let artifact = load_artifact(
        &container.path().to_string_lossy(),
        LoadVindexOptions::default(),
    )
    .unwrap();
    assert!(
        matches!(artifact, LoadedArtifact::V3(_)),
        "binding must resolve to the V3 runtime"
    );
}

#[tokio::test]
async fn v3_model_appears_in_the_models_listing() {
    let container = v3_container();
    let app = larql_server::routes::single_model_router(v3_state(container.path()));
    let resp = common::get(app, "/v1/models").await;
    assert_eq!(resp.status(), axum::http::StatusCode::OK);
    let json = common::body_json(resp.into_body()).await;
    assert_eq!(json["object"], "list");
    let data = json["data"].as_array().unwrap();
    assert_eq!(data.len(), 1);
    assert_eq!(data[0]["object"], "model");
    assert_eq!(data[0]["generation"], 3);
    assert_eq!(data[0]["loaded"], true);
    assert!(data[0]["id"].as_str().is_some_and(|s| !s.is_empty()));
}

#[tokio::test]
async fn v3_buffered_supports_echo_and_batched_prompts() {
    let container = v3_container();
    let app = larql_server::routes::single_model_router(v3_state(container.path()));
    let resp = common::post_json(
        app,
        "/v1/completions",
        serde_json::json!({
            "prompt": ["[3]", "[5]"],
            "max_tokens": 2,
            "echo": true,
        }),
    )
    .await;
    assert_eq!(resp.status(), axum::http::StatusCode::OK);
    let json = common::body_json(resp.into_body()).await;
    let choices = json["choices"].as_array().unwrap();
    assert_eq!(choices.len(), 2);
    assert!(choices[0]["text"].as_str().unwrap().starts_with("[3]"));
    assert!(choices[1]["text"].as_str().unwrap().starts_with("[5]"));
    assert_eq!(json["usage"]["prompt_tokens"], 2);
    assert_eq!(json["usage"]["completion_tokens"], 4);
}

#[tokio::test]
async fn v3_buffered_trims_at_client_stop_strings() {
    let container = v3_container();
    let expected = direct_arm(container.path(), NEW_TOKENS);
    let stop = expected[2].1.clone();

    let app = larql_server::routes::single_model_router(v3_state(container.path()));
    let resp = common::post_json(
        app,
        "/v1/completions",
        serde_json::json!({
            "prompt": PROMPT,
            "max_tokens": NEW_TOKENS,
            "stop": stop,
        }),
    )
    .await;
    assert_eq!(resp.status(), axum::http::StatusCode::OK);
    let json = common::body_json(resp.into_body()).await;
    assert_eq!(json["choices"][0]["finish_reason"], "stop");
    let text = json["choices"][0]["text"].as_str().unwrap();
    let full: String = expected.iter().map(|(_, t)| t.as_str()).collect();
    assert!(text.len() < full.len(), "stop must trim the completion");
}

#[tokio::test]
async fn v3_stream_reports_an_untokenizable_prompt_as_an_error_chunk() {
    let container = v3_container();
    let app = larql_server::routes::single_model_router(v3_state(container.path()));
    // An empty prompt string passes the handler's list-level check but
    // tokenises to zero ids — the in-stream failure path.
    let resp = common::post_json(
        app,
        "/v1/completions",
        serde_json::json!({
            "prompt": "",
            "max_tokens": 4,
            "stream": true,
        }),
    )
    .await;
    // Headers are already SSE by the time tokenisation runs, so the
    // failure arrives as an in-stream error chunk, mirroring V2.
    assert_eq!(resp.status(), axum::http::StatusCode::OK);
    let bytes = axum::body::to_bytes(resp.into_body(), usize::MAX)
        .await
        .unwrap();
    let body = String::from_utf8(bytes.to_vec()).unwrap();
    assert!(body.contains("error"), "{body}");
    assert!(body.contains("[DONE]"));
}

/// Binding refuses a directory that is not a V3 container, naming the
/// open step — never a panic, never a half-bound model.
#[test]
fn load_v3_model_refuses_a_non_container_directory() {
    let empty = tempfile::tempdir().unwrap();
    let err = larql_server::vindex3::load_v3_model(empty.path())
        .err()
        .expect("an empty directory must not bind");
    assert!(err.to_string().contains("open VINDEX3 container"), "{err}");
}

/// A valid container without `tokenizer.json` cannot serve the
/// text-facing API; the refusal names the missing capability.
#[test]
fn load_v3_model_refuses_a_tokenizerless_container() {
    let checkpoint = tempfile::tempdir().unwrap();
    let container = tempfile::tempdir().unwrap();
    encode_fixture_container(
        miniature_glimmer,
        checkpoint.path(),
        container.path(),
        "serve-fixture",
    );
    let err = larql_server::vindex3::load_v3_model(container.path())
        .err()
        .expect("a tokenizerless container must not bind for serving");
    assert!(err.to_string().contains("tokenizer.json"), "{err}");
}

/// A container encoded nameless falls back to the directory name —
/// the last-resort identity, never an empty id.
#[test]
fn a_nameless_container_takes_its_id_from_the_directory() {
    let checkpoint = tempfile::tempdir().unwrap();
    let container = tempfile::tempdir().unwrap();
    encode_fixture_container(miniature_glimmer, checkpoint.path(), container.path(), "");
    std::fs::write(
        container.path().join("tokenizer.json"),
        synthetic_tokenizer_json(G_VOCAB),
    )
    .unwrap();
    let model = larql_server::vindex3::load_v3_model(container.path()).unwrap();
    let dir_name = container
        .path()
        .file_name()
        .unwrap()
        .to_string_lossy()
        .into_owned();
    assert_eq!(model.id, dir_name);
}

/// A prefill failure surfaces as a server error naming the stage, not
/// a panic — driven through the same `generate_v3` the routes use.
#[test]
fn generate_v3_reports_a_prefill_failure_as_a_server_error() {
    let container = v3_container();
    let model = larql_server::vindex3::load_v3_model(container.path()).unwrap();
    let result = larql_server::vindex3::generate_v3(
        &model,
        &[],
        4,
        SamplingConfig::greedy(),
        &EosConfig::builtin(),
        |_, _| {},
    );
    match result {
        Err(e) => assert!(e.to_string().contains("prefill"), "{e}"),
        // If the runtime ever learns to prefill zero tokens this arm
        // keeps the gate honest instead of silently passing.
        Ok(generation) => panic!("empty prefill unexpectedly succeeded: {:?}", generation.ids),
    }
}

/// …and the other arm of the same binding decision: an ordinary V2
/// vindex must resolve to the V2 runtime through the identical
/// `load_artifact` call, so the dispatch is proven in both directions.
#[test]
fn a_v2_vindex_binds_as_v2_through_the_same_artifact_loader() {
    let fixture = common::synthetic_vindex::build();
    let artifact =
        load_artifact(&fixture.dir.to_string_lossy(), LoadVindexOptions::default()).unwrap();
    assert!(
        matches!(artifact, LoadedArtifact::V2(_)),
        "a V2 vindex must bind as V2"
    );
}

/// `/v1/stats` must answer on a V3-only server.
///
/// It used to 404: the handler resolved V2 only, so the `server` block
/// — the sole surface carrying the N1 continuation counters — was
/// unreachable on exactly the deployments N1 runs on. Found by serving
/// a real container, not by a fixture.
#[tokio::test]
async fn stats_answers_on_a_v3_only_server_and_carries_the_server_block() {
    let container = v3_container();
    let app = larql_server::routes::single_model_router(v3_state(container.path()));
    let resp = app
        .oneshot(
            Request::builder()
                .method("GET")
                .uri("/v1/stats")
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(
        resp.status(),
        StatusCode::OK,
        "V3-only /v1/stats must not 404"
    );
    let bytes = axum::body::to_bytes(resp.into_body(), usize::MAX)
        .await
        .unwrap();
    let json: serde_json::Value = serde_json::from_slice(&bytes).unwrap();

    // The program's own shape, read from the opened plan.
    assert_eq!(json["generation"], 3);
    assert_eq!(json["component"], "target");
    assert!(json["layers"].as_u64().expect("layers") > 0);
    assert!(json["has_output_head"].as_bool().expect("head"));
    // The V2 vocabulary must not be faked onto a V3 binding.
    assert!(json.get("features").is_none());
    assert!(json.get("extract_level").is_none());

    // The reason this endpoint matters on V3 at all.
    let kv = &json["server"]["v3_kv"];
    assert!(kv["enabled"].as_bool().expect("enabled"));
    for counter in ["hits", "misses", "resumptions", "reused_tokens_total"] {
        assert!(kv[counter].is_number(), "missing v3_kv.{counter}");
    }
    assert!(json["server"]["sessions"]["active"].is_number());
}

/// Service modes without a V3 implementation must refuse rather than
/// silently accept flags such as `--no-infer`. CPU layer slicing is
/// tested separately through its actual HTTP contract.
#[test]
fn a_v3_container_refuses_options_it_cannot_honour() {
    let container = v3_container();
    let path = container.path().to_string_lossy().to_string();

    for (label, opts) in [
        (
            "--no-infer",
            LoadVindexOptions {
                no_infer: true,
                ..LoadVindexOptions::default()
            },
        ),
        (
            "--embed-only",
            LoadVindexOptions {
                embed_only: true,
                ..LoadVindexOptions::default()
            },
        ),
    ] {
        let msg = match load_artifact(&path, opts) {
            Err(e) => e.to_string(),
            Ok(_) => panic!("{label} must be refused, not silently ignored"),
        };
        assert!(
            msg.contains("do not support") && msg.contains(label),
            "{label}: refusal must name the option — got {msg}"
        );
    }

    // And the same container binds cleanly with no such options.
    let ok = load_artifact(&path, LoadVindexOptions::default())
        .map_err(|e| e.to_string())
        .expect("a V3 container with no unsupported options binds");
    assert!(matches!(ok, LoadedArtifact::V3(_)));
}

/// `V3Model::requests_in_flight` must reflect generation genuinely
/// running, not just "a request handler is somewhere on the stack" —
/// the guard lives inside `generate_v3_request` for exactly this
/// reason (see `docs/runtime-lifecycle-design.md` and the doc comment
/// on `V3GenerationGuard`). A before/after check alone can't catch an
/// incorrectly scoped guard (e.g. one that decrements immediately
/// after entering, or one entered too late to cover the decode loop):
/// this test needs to observe `1` *while* a real generation is
/// mid-flight on another thread, then `0` once it's done.
#[test]
fn v3_generation_in_flight_counter_reflects_genuine_concurrency() {
    let container = v3_container();
    let artifact = load_artifact(
        &container.path().to_string_lossy(),
        LoadVindexOptions::default(),
    )
    .unwrap();
    let model = match artifact {
        LoadedArtifact::V3(m) => Arc::new(*m),
        LoadedArtifact::V2(_) => panic!("a VINDEX3 container must bind as V3"),
    };
    assert_eq!(model.requests_in_flight(), 0, "idle before any generation");

    let prompt_ids: Vec<u32> = model
        .tokenizer
        .encode(PROMPT, true)
        .unwrap()
        .get_ids()
        .to_vec();
    assert!(!prompt_ids.is_empty());

    let model_bg = Arc::clone(&model);
    let handle = std::thread::spawn(move || {
        generate_v3(
            &model_bg,
            &prompt_ids,
            NEW_TOKENS,
            SamplingConfig::greedy(),
            &EosConfig::builtin(),
            |_id, _text| {
                // Widen the in-flight window enough for the polling
                // loop below to catch it deterministically — the same
                // technique `ensure_weights_cell_single_flights_
                // concurrent_loaders` (state/loaded_model.rs) uses to
                // make a race window observable in a unit test.
                std::thread::sleep(std::time::Duration::from_millis(15));
            },
        )
        .expect("generation must succeed against the fixture container");
    });

    // Poll for genuine in-flight work rather than sleeping a fixed
    // guess-and-hope duration — fail loudly if it's never observed,
    // rather than passing on a lucky timing coincidence.
    let deadline = std::time::Instant::now() + std::time::Duration::from_secs(5);
    let mut observed_in_flight = false;
    while std::time::Instant::now() < deadline {
        if model.requests_in_flight() == 1 {
            observed_in_flight = true;
            break;
        }
        std::thread::sleep(std::time::Duration::from_millis(2));
    }
    assert!(
        observed_in_flight,
        "never observed requests_in_flight() == 1 while a generation with a 15ms/token \
         callback delay ({} tokens) was running on another thread",
        NEW_TOKENS
    );

    handle.join().expect("generation thread must not panic");
    assert_eq!(
        model.requests_in_flight(),
        0,
        "the guard must decrement back to 0 once generation returns"
    );
}

/// Ids named by `[N]` tokens in a synthetic-tokenizer surface string, in
/// order — the chat route returns text, and on this fixture the text IS
/// the id sequence.
fn ids_in_surface(text: &str) -> Vec<u32> {
    text.split('[')
        .filter_map(|piece| piece.split(']').next())
        .filter_map(|n| n.parse().ok())
        .collect()
}

/// Encode the fixture and declare `eos_id` as its end-of-turn token in
/// `generation_config.json`, the file the CLI's V3 arm already reads.
fn v3_container_declaring_eos(eos_id: u32) -> tempfile::TempDir {
    let container = v3_container();
    std::fs::write(
        container
            .path()
            .join(larql_vindex::format::filenames::GENERATION_CONFIG_JSON),
        serde_json::json!({ "eos_token_id": eos_id }).to_string(),
    )
    .unwrap();
    container
}

/// **A container's declared end-of-turn token stops served generation.**
///
/// The V3 driver judges EOS on ids alone, so the server must hand it the
/// ids the container declares — an empty built-in set means every V3
/// completion runs to `max_tokens`. The direct arm says what the fixture
/// emits for PROMPT; its second id is declared as EOS; the same request
/// then finishes with `stop` after exactly the tokens before that id,
/// buffered and streamed, and a chat turn stops the same way. The
/// identical container without the declaration runs to `length`, which
/// is the control that the stop came from the declaration.
#[tokio::test]
async fn v3_generation_stops_on_the_containers_declared_eos_token() {
    // Control: no declaration, the budget is filled.
    let plain = v3_container();
    let emitted = direct_arm(plain.path(), NEW_TOKENS);
    let eos_id = emitted[1].0;
    let expected_len = emitted
        .iter()
        .position(|(id, _)| *id == eos_id)
        .expect("the declared id is one the fixture emits");
    let expected_text: String = emitted[..expected_len]
        .iter()
        .map(|(_, t)| t.as_str())
        .collect();
    let plain_app = larql_server::routes::single_model_router(v3_state(plain.path()));
    let resp = common::post_json(
        plain_app.clone(),
        "/v1/completions",
        serde_json::json!({"prompt": PROMPT, "max_tokens": NEW_TOKENS}),
    )
    .await;
    let json = common::body_json(resp.into_body()).await;
    assert_eq!(
        json["choices"][0]["finish_reason"], "length",
        "control: {json}"
    );
    assert_eq!(
        json["usage"]["completion_tokens"], NEW_TOKENS,
        "control: {json}"
    );
    let plain_chat = common::post_json(
        plain_app,
        "/v1/chat/completions",
        serde_json::json!({
            "messages": [{"role": "user", "content": "[1]"}],
            "max_tokens": NEW_TOKENS,
        }),
    )
    .await;
    let plain_chat = common::body_json(plain_chat.into_body()).await;
    assert_eq!(
        plain_chat["choices"][0]["finish_reason"], "length",
        "control: {plain_chat}"
    );
    let chat_ids = ids_in_surface(
        plain_chat["choices"][0]["message"]["content"]
            .as_str()
            .unwrap(),
    );
    assert!(
        chat_ids.len() >= 2,
        "control: the chat turn must emit ids: {plain_chat}"
    );
    let chat_eos_id = chat_ids[1];
    let chat_expected_len = chat_ids.iter().position(|&id| id == chat_eos_id).unwrap();

    // Declared: the same fixture with `generation_config.json`.
    let container = v3_container_declaring_eos(eos_id);
    let app = larql_server::routes::single_model_router(v3_state(container.path()));

    let resp = common::post_json(
        app.clone(),
        "/v1/completions",
        serde_json::json!({"prompt": PROMPT, "max_tokens": NEW_TOKENS}),
    )
    .await;
    assert_eq!(resp.status(), axum::http::StatusCode::OK);
    let json = common::body_json(resp.into_body()).await;
    assert_eq!(
        json["choices"][0]["finish_reason"], "stop",
        "buffered: {json}"
    );
    assert_eq!(
        json["usage"]["completion_tokens"], expected_len,
        "buffered: {json}"
    );
    assert_eq!(
        json["choices"][0]["text"],
        expected_text.as_str(),
        "buffered: {json}"
    );

    let resp = common::post_json(
        app.clone(),
        "/v1/completions",
        serde_json::json!({"prompt": PROMPT, "max_tokens": NEW_TOKENS, "stream": true}),
    )
    .await;
    assert_eq!(resp.status(), axum::http::StatusCode::OK);
    let bytes = axum::body::to_bytes(resp.into_body(), usize::MAX)
        .await
        .unwrap();
    let chunks = sse_chunks(core::str::from_utf8(&bytes).unwrap());
    assert_eq!(
        chunks.len(),
        expected_len + 1,
        "streamed: one chunk per kept token plus the stop"
    );
    assert_eq!(
        chunks[expected_len]["choices"][0]["finish_reason"], "stop",
        "streamed: {chunks:?}"
    );

    // Chat: its own prompt, its own sequence, the same stop.
    let chat_container = v3_container_declaring_eos(chat_eos_id);
    let chat_app = larql_server::routes::single_model_router(v3_state(chat_container.path()));
    let resp = common::post_json(
        chat_app,
        "/v1/chat/completions",
        serde_json::json!({
            "messages": [{"role": "user", "content": "[1]"}],
            "max_tokens": NEW_TOKENS,
        }),
    )
    .await;
    assert_eq!(resp.status(), axum::http::StatusCode::OK);
    let json = common::body_json(resp.into_body()).await;
    assert_eq!(json["choices"][0]["finish_reason"], "stop", "chat: {json}");
    assert_eq!(
        json["usage"]["completion_tokens"], chat_expected_len,
        "chat: {json}"
    );
    assert_eq!(
        ids_in_surface(json["choices"][0]["message"]["content"].as_str().unwrap()),
        chat_ids[..chat_expected_len],
        "chat: {json}"
    );
}

/// An `AppState` with nothing bound: the control for "absent".
fn empty_state() -> Arc<AppState> {
    let state = v3_state(v3_container().path());
    state
        .model_set
        .write()
        .unwrap_or_else(|p| p.into_inner())
        .v3_models
        .clear();
    state
}

/// **A loaded-but-unsupported model never masquerades as absent.**
///
/// The V2-only surfaces resolve a VINDEX2 model; on a server that has
/// only a VINDEX3 container bound they used to answer 404 "no model
/// loaded" while `/v1/models` listed the container. The rule is three-
/// way: no model is 404, a V2 model takes the route, and a V3 model on
/// a V2-only capability is 501 naming VINDEX3 — a truthful refusal, not
/// V3 support, which those routes do not have.
#[tokio::test]
async fn v2_only_surfaces_refuse_a_v3_container_as_unsupported_not_absent() {
    let container = v3_container();
    let state = v3_state(container.path());
    let app = larql_server::routes::single_model_router(state.clone());

    let gets = [
        "/v1/describe?entity=%5B1%5D",
        "/v1/relations",
        "/v1/patches",
        "/v1/walk?prompt=%5B1%5D&top=1",
    ];
    for path in gets {
        let resp = app
            .clone()
            .oneshot(
                Request::builder()
                    .method("GET")
                    .uri(path)
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        let status = resp.status();
        let json = common::body_json(resp.into_body()).await;
        assert_eq!(status, StatusCode::NOT_IMPLEMENTED, "{path}: {json}");
        let error = json["error"].as_str().unwrap_or_default().to_string();
        assert!(
            error.contains("VINDEX3"),
            "{path}: refusal must name VINDEX3: {json}"
        );
        assert!(
            !error.contains("not found"),
            "{path}: a bound model is not absent: {json}"
        );
    }
    let resp = common::post_json(
        app.clone(),
        "/v1/infer",
        serde_json::json!({"prompt": "[1]"}),
    )
    .await;
    let status = resp.status();
    let json = common::body_json(resp.into_body()).await;
    assert_eq!(status, StatusCode::NOT_IMPLEMENTED, "/v1/infer: {json}");
    assert!(
        json["error"]
            .as_str()
            .unwrap_or_default()
            .contains("VINDEX3"),
        "{json}"
    );

    // An OpenAI-shaped V2-only route answers in the OpenAI envelope.
    let resp = common::post_json(
        app.clone(),
        "/v1/embeddings",
        serde_json::json!({"input": "[1]"}),
    )
    .await;
    let status = resp.status();
    let json = common::body_json(resp.into_body()).await;
    assert_eq!(
        status,
        StatusCode::NOT_IMPLEMENTED,
        "/v1/embeddings: {json}"
    );
    assert_eq!(json["error"]["type"], "not_implemented_error", "{json}");
    assert!(
        json["error"]["message"]
            .as_str()
            .unwrap_or_default()
            .contains("VINDEX3"),
        "{json}"
    );

    // gRPC: the same rule, in gRPC's vocabulary.
    use larql_server::grpc::proto::vindex_service_server::VindexService;
    let svc = larql_server::grpc::VindexGrpcService {
        state: state.clone(),
    };
    let err = svc
        .get_stats(tonic::Request::new(
            larql_server::grpc::proto::StatsRequest {},
        ))
        .await
        .expect_err("a V3-only server refuses the V2 gRPC surface");
    assert_eq!(err.code(), tonic::Code::Unimplemented, "{err}");
    assert!(err.message().contains("VINDEX3"), "{err}");

    // Control: nothing bound is still "absent".
    let empty = larql_server::routes::single_model_router(empty_state());
    let resp = empty
        .oneshot(
            Request::builder()
                .method("GET")
                .uri("/v1/describe?entity=%5B1%5D")
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();
    let status = resp.status();
    let json = common::body_json(resp.into_body()).await;
    assert_eq!(status, StatusCode::NOT_FOUND, "control: {json}");
    assert_eq!(json["error"], "no model loaded", "control: {json}");
}

#[tokio::test]
async fn lifecycle_reports_selected_backend_and_refuses_an_implicit_switch() {
    let container = v3_container();
    let state = common::state(vec![]);
    let app = larql_server::routes::single_model_router(state);
    let path = container.path().to_string_lossy();
    let loaded = common::post_json(
        app.clone(),
        "/v1/runtime/model",
        serde_json::json!({"path": path, "backend": "cpu"}),
    )
    .await;
    assert_eq!(loaded.status(), StatusCode::OK);
    let body = common::body_json(loaded.into_body()).await;
    assert_eq!(body["backend"]["selected"], "cpu");
    let conflict = common::post_json(
        app.clone(),
        "/v1/runtime/model",
        serde_json::json!({"path": path, "backend": "metal"}),
    )
    .await;
    assert_eq!(conflict.status(), StatusCode::CONFLICT);
    let bad = common::post_json(
        app,
        "/v1/runtime/model",
        serde_json::json!({"path": path, "backend": "typo"}),
    )
    .await;
    assert_eq!(bad.status(), StatusCode::UNPROCESSABLE_ENTITY);
}

#[cfg(not(all(feature = "vindex3-metal", target_os = "macos")))]
#[test]
fn unavailable_metal_is_refused_instead_of_serving_on_cpu() {
    let container = v3_container();
    let result = load_artifact(
        container.path().to_str().unwrap(),
        LoadVindexOptions {
            v3_backend: larql_server::vindex3::V3Backend::Metal,
            ..Default::default()
        },
    );
    let err = result
        .err()
        .expect("Metal must not fall back to CPU")
        .to_string();
    assert!(err.contains("vindex3-metal"), "{err}");
}

/// Explicit opt-in: requires a real Metal device; it never passes by skipping
/// device creation or falling back to CPU. Run serially with vindex3-metal.
#[cfg(all(feature = "vindex3-metal", target_os = "macos"))]
#[test]
#[ignore = "requires a real Metal device"]
fn selected_metal_backend_executes_the_v3_fixture() {
    use larql_server::vindex3::{load_v3_model_with_backend, V3Backend};
    let container = v3_container();
    let model = load_v3_model_with_backend(container.path(), V3Backend::Metal).unwrap();
    assert_eq!(model.backend, V3Backend::Metal);
    let ids = model
        .tokenizer
        .encode(PROMPT, true)
        .unwrap()
        .get_ids()
        .to_vec();
    let result = generate_v3(
        &model,
        &ids,
        NEW_TOKENS,
        SamplingConfig::greedy(),
        &EosConfig::builtin(),
        |_, _| {},
    )
    .unwrap();
    let expected: Vec<_> = direct_arm(container.path(), NEW_TOKENS)
        .into_iter()
        .map(|(id, _)| id)
        .collect();
    assert_eq!(
        result.ids, expected,
        "fixture's greedy tokens must agree with CPU"
    );
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn v3_layer_workers_over_http_match_local_execution_and_refuse_bad_requests() {
    use larql_inference::vindex3::{
        distributed::{artifact_identity, DistributedSession},
        LogitsSession,
    };
    use larql_router::vindex3::HttpLayerShards;
    use larql_router_protocol::vindex3::{Binding, PATH};
    use larql_vindex::format::vindex3::opplan::exec::prepared::ExecutionSlice;
    let container = v3_container();
    let full = larql_server::routes::single_model_router(v3_state(container.path()));
    let response = full
        .oneshot(Request::builder().uri(PATH).body(Body::empty()).unwrap())
        .await
        .unwrap();
    assert_eq!(response.status(), StatusCode::NOT_IMPLEMENTED);
    for (range, backend) in [
        ((2, 3), larql_server::vindex3::V3Backend::Cpu),
        ((1, 0), larql_server::vindex3::V3Backend::Cpu),
        ((0, usize::MAX), larql_server::vindex3::V3Backend::Cpu),
        ((0, 0), larql_server::vindex3::V3Backend::Metal),
    ] {
        assert!(load_artifact(
            container.path().to_str().unwrap(),
            LoadVindexOptions {
                layer_range: Some(range),
                v3_backend: backend,
                ..Default::default()
            }
        )
        .is_err());
    }
    let mut urls = Vec::new();
    let mut servers = Vec::new();
    for layer in 0..2 {
        let state = v3_state(container.path());
        let LoadedArtifact::V3(model) = load_artifact(
            container.path().to_str().unwrap(),
            LoadVindexOptions {
                layer_range: Some(
                    larql_server::bootstrap::parse_layer_range(&format!("{layer}-{layer}"))
                        .unwrap(),
                ),
                ..Default::default()
            },
        )
        .unwrap() else {
            panic!("V3")
        };
        assert_eq!(model.runtime.operands().layer_count(), 1);
        assert!(!model.runtime.operands().has_output());
        state.model_set.write().unwrap().v3_models = vec![Arc::new(*model)];
        let app = larql_server::routes::single_model_router(state);
        // A worker never serves a partial stack as a complete language model.
        let denied = common::post_json(
            app.clone(),
            "/v1/completions",
            serde_json::json!({"prompt":PROMPT,"max_tokens":1}),
        )
        .await;
        assert!(!denied.status().is_success());
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        urls.push(format!("http://{}", listener.local_addr().unwrap()));
        servers.push(tokio::spawn(async move {
            axum::serve(listener, app).await.unwrap()
        }));
    }
    let client = reqwest::Client::new();
    let binding: Binding = client
        .get(format!("{}{PATH}", urls[0]))
        .send()
        .await
        .unwrap()
        .json()
        .await
        .unwrap();
    let mut wrong = binding.clone();
    wrong.end = 2;
    for body in [
        serde_json::json!({"binding":wrong,"rows":[vec![0.0;binding.hidden]]}),
        serde_json::json!({"binding":binding,"rows":[[0.0]]}),
    ] {
        assert_eq!(
            client
                .post(format!("{}{PATH}", urls[0]))
                .json(&body)
                .send()
                .await
                .unwrap()
                .status(),
            StatusCode::BAD_REQUEST
        );
    }
    let path = container.path().to_path_buf();
    tokio::task::spawn_blocking(move || {
        let runtime = Vindex3Runtime::open(&path, COMPONENT, ProductionBackend::new()).unwrap();
        let identity = artifact_identity(&path, runtime.plan()).unwrap();
        let endpoints = runtime.prepare_slice(ExecutionSlice::Endpoints).unwrap();
        let transport = HttpLayerShards::connect(&urls, None).unwrap();
        let mut remote = DistributedSession::new(
            endpoints.plan(),
            endpoints.operands(),
            endpoints.backend(),
            &identity,
            transport,
        )
        .unwrap();
        let local = Vindex3Runtime::open(&path, COMPONENT, ProductionBackend::new())
            .unwrap()
            .prepare()
            .unwrap();
        let mut kv = CanonicalKvState::new();
        let mut reference = local.session_with_kv(&mut kv).unwrap();
        // Sliding window is three: this fixture actually crosses its boundary.
        for id in [3, 17, 28, 0, 11, 3, 17, 28, 0, 11] {
            let a = reference.step(id).unwrap();
            let b = remote.step(id).unwrap();
            let delta = a
                .iter()
                .zip(&b)
                .map(|(x, y)| (x - y).abs())
                .fold(0.0f32, f32::max);
            assert!(delta < 1e-5, "HTTP logit delta {delta}");
        }
    })
    .await
    .unwrap();
    for server in servers {
        server.abort();
    }
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn v3_dense_ffn_workers_over_http_preserve_local_continuation() {
    use larql_inference::vindex3::{
        dense_ffn::{prepare_coordinator, DenseFfnSession},
        LogitsSession,
    };
    use larql_router::vindex3_ffn::HttpFfnShards;
    use larql_router_protocol::vindex3_ffn::{Binding, PATH};
    use larql_vindex::format::vindex3::opplan::exec::prepared::ExecutionSlice;
    let container = v3_container();
    let mut urls = Vec::new();
    let mut servers = Vec::new();
    assert!(load_artifact(
        container.path().to_str().unwrap(),
        LoadVindexOptions {
            ffn_only: true,
            ..Default::default()
        }
    )
    .is_err());
    for layer in 0..2 {
        let LoadedArtifact::V3(model) = load_artifact(
            container.path().to_str().unwrap(),
            LoadVindexOptions {
                ffn_only: true,
                layer_range: Some(
                    larql_server::bootstrap::parse_layer_range(&format!("{layer}-{layer}"))
                        .unwrap(),
                ),
                ..Default::default()
            },
        )
        .unwrap() else {
            panic!("V3")
        };
        let ops = model.runtime.operands();
        assert_eq!(
            ops.slice(),
            &ExecutionSlice::DenseFfns {
                start: layer,
                end: layer + 1
            }
        );
        assert!(!ops.has_output());
        let census = ops.residency_census();
        assert_eq!(census.attention.total(), 0);
        assert_eq!(census.embedding.total(), 0);
        assert_eq!(census.glue.total(), 0);
        assert!(census.ffn.total() > 0);
        let state = v3_state(container.path());
        state.model_set.write().unwrap().v3_models = vec![Arc::new(*model)];
        let app = larql_server::routes::single_model_router(state);
        let denied = common::post_json(
            app.clone(),
            "/v1/completions",
            serde_json::json!({"prompt":PROMPT,"max_tokens":1}),
        )
        .await;
        assert!(!denied.status().is_success());
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        urls.push(format!("http://{}", listener.local_addr().unwrap()));
        servers.push(tokio::spawn(async move {
            axum::serve(listener, app).await.unwrap()
        }));
    }
    let client = reqwest::Client::new();
    let binding: Binding = client
        .get(format!("{}{PATH}", urls[0]))
        .send()
        .await
        .unwrap()
        .json()
        .await
        .unwrap();
    let good =
        serde_json::json!({"binding":binding,"layer":0,"row":vec![0.3;binding.program.hidden]});
    let a: serde_json::Value = client
        .post(format!("{}{PATH}", urls[0]))
        .json(&good)
        .send()
        .await
        .unwrap()
        .json()
        .await
        .unwrap();
    let b: serde_json::Value = client
        .post(format!("{}{PATH}", urls[0]))
        .json(&good)
        .send()
        .await
        .unwrap()
        .json()
        .await
        .unwrap();
    assert_eq!(a, b);
    let mut wrong = binding.clone();
    wrong.program.artifact = "0".repeat(64);
    for body in [
        serde_json::json!({"binding":binding,"layer":1,"row":vec![0.3;binding.program.hidden]}),
        serde_json::json!({"binding":binding,"layer":0,"row":[0.3]}),
        serde_json::json!({"binding":wrong,"layer":0,"row":vec![0.3;binding.program.hidden]}),
    ] {
        assert_eq!(
            client
                .post(format!("{}{PATH}", urls[0]))
                .json(&body)
                .send()
                .await
                .unwrap()
                .status(),
            StatusCode::BAD_REQUEST
        );
    }
    use larql_router_protocol::vindex3_ffn::binary::{self, Direction};
    let opened: binary::Opened = client
        .post(format!("{}{}", urls[0], binary::OPEN_PATH))
        .json(&binding)
        .send()
        .await
        .unwrap()
        .error_for_status()
        .unwrap()
        .json()
        .await
        .unwrap();
    assert_eq!(opened.binding, binding);
    let mut mismatched = binding.clone();
    mismatched.program.artifact = "0".repeat(64);
    assert_eq!(
        client
            .post(format!("{}{}", urls[0], binary::OPEN_PATH))
            .json(&mismatched)
            .send()
            .await
            .unwrap()
            .status(),
        StatusCode::BAD_REQUEST
    );
    let packet = binary::encode(
        Direction::Request,
        opened.handle,
        17,
        0,
        &vec![0.3; binding.program.hidden],
    )
    .unwrap();
    for _ in 0..2 {
        let response = client
            .post(format!("{}{}", urls[0], binary::PATH))
            .header(reqwest::header::CONTENT_TYPE, binary::CONTENT_TYPE)
            .body(packet.clone())
            .send()
            .await
            .unwrap()
            .error_for_status()
            .unwrap()
            .bytes()
            .await
            .unwrap();
        let decoded =
            binary::decode(&response, Direction::Response, binding.program.hidden).unwrap();
        assert_eq!(decoded.sequence, 17);
        let expected: larql_router_protocol::vindex3_ffn::Response =
            serde_json::from_value(a.clone()).unwrap();
        assert_eq!(
            decoded.row.iter().map(|x| x.to_bits()).collect::<Vec<_>>(),
            expected.row.iter().map(|x| x.to_bits()).collect::<Vec<_>>()
        );
    }
    // A fresh preparation of the very same artifact invalidates old handles.
    let LoadedArtifact::V3(reopened) = load_artifact(
        container.path().to_str().unwrap(),
        LoadVindexOptions {
            ffn_only: true,
            layer_range: Some((0, 1)),
            ..Default::default()
        },
    )
    .unwrap() else {
        panic!("V3")
    };
    assert_ne!(reopened.ffn_wire.as_ref().unwrap().handle, opened.handle);
    for mode in 0..6 {
        let mut bad = packet.clone();
        match mode {
            0 => bad[4..20].copy_from_slice(&reopened.ffn_wire.as_ref().unwrap().handle),
            1 => bad[28..32].copy_from_slice(&1u32.to_le_bytes()),
            2 => {
                bad.pop();
            }
            3 => bad.push(0),
            4 => bad[32..36].copy_from_slice(&0u32.to_le_bytes()),
            _ => bad[binary::HEADER_BYTES..binary::HEADER_BYTES + 4]
                .copy_from_slice(&f32::NAN.to_bits().to_le_bytes()),
        }
        assert_eq!(
            client
                .post(format!("{}{}", urls[0], binary::PATH))
                .header(reqwest::header::CONTENT_TYPE, binary::CONTENT_TYPE)
                .body(bad)
                .send()
                .await
                .unwrap()
                .status(),
            StatusCode::BAD_REQUEST
        );
    }
    let path = container.path().to_path_buf();
    tokio::task::spawn_blocking(move || {
        use tungstenite::{client::IntoClientRequest, Message};
        let opened: binary::Opened = reqwest::blocking::Client::new().post(format!("{}{}", urls[0], binary::OPEN_PATH)).json(&binding).send().unwrap().json().unwrap();
        for fault in 0..7 {
            let address = urls[0].strip_prefix("http://").unwrap();
            let tcp = std::net::TcpStream::connect(address).unwrap();
            tcp.set_read_timeout(Some(std::time::Duration::from_secs(2))).unwrap();
            let mut request = format!("ws://{address}{}", binary::STREAM_PATH).into_client_request().unwrap();
            request.headers_mut().insert("Sec-WebSocket-Protocol", binary::STREAM_PROTOCOL.parse().unwrap());
            let (mut socket, _) = tungstenite::client(request, tcp).unwrap();
            socket.send(Message::Text(r#"{"profile":false}"#.into())).unwrap();
            let mut frame = binary::encode(binary::Direction::Request, opened.handle, 1, 0, &vec![0.3;binding.program.hidden]).unwrap();
            // A valid operation first proves the stream is admitted. Duplicate
            // sequence, unlike stateless HTTP replay, is then a protocol error.
            socket.send(Message::Binary(frame.clone().into())).unwrap();
            assert!(matches!(socket.read().unwrap(), Message::Binary(_)));
            frame[20..28].copy_from_slice(&2u64.to_le_bytes());
            match fault {
                0 => frame[4] ^= 1,
                1 => frame[28..32].copy_from_slice(&1u32.to_le_bytes()),
                2 => frame[20..28].copy_from_slice(&1u64.to_le_bytes()),
                3 => frame[36..40].copy_from_slice(&f32::NAN.to_bits().to_le_bytes()),
                4 => { frame.pop(); },
                5 => frame.push(0),
                6 => frame[32..36].copy_from_slice(&0u32.to_le_bytes()),
                _ => unreachable!(),
            }
            socket.send(Message::Binary(frame.into())).unwrap();
            assert!(!matches!(socket.read(), Ok(Message::Binary(_))), "stream accepted fault {fault}");
        }
        let runtime = Vindex3Runtime::open(&path, COMPONENT, ProductionBackend::new()).unwrap();
        let stream_transport = HttpFfnShards::connect_stream(&urls, None).unwrap();
        let stream_ops = prepare_coordinator(&path, runtime.plan(), runtime.operands(), runtime.backend(), stream_transport).unwrap();
        let mut stream_session = DenseFfnSession::new(runtime.plan(), &stream_ops, runtime.backend()).unwrap();
        let binary_transport = HttpFfnShards::connect_binary(&urls, None).unwrap();
        let binary_ops = prepare_coordinator(&path, runtime.plan(), runtime.operands(), runtime.backend(), binary_transport).unwrap();
        let mut binary_session = DenseFfnSession::new(runtime.plan(), &binary_ops, runtime.backend()).unwrap();
        let transport = HttpFfnShards::connect(&urls, None).unwrap();
        let ops = prepare_coordinator(
            &path,
            runtime.plan(),
            runtime.operands(),
            runtime.backend(),
            transport,
        )
        .unwrap();
        let mut remote = DenseFfnSession::new(runtime.plan(), &ops, runtime.backend()).unwrap();
        let mut local = runtime.session().unwrap();
        let mut smoke = Vec::new();
        for (position, id) in [3, 17, 28, 0, 11, 3, 17, 28, 0, 11].into_iter().enumerate() {
            use larql_inference::vindex3::dense_ffn::profile::Capture;
            let capture = Capture::start().unwrap();
            let expected = local.step(id).unwrap();
            let local_rows = capture.finish();
            assert_eq!(local_rows.len(), 1);
            assert!(local_rows[0].provider_calls.is_empty());
            let capture = Capture::start().unwrap();
            let actual = remote.step(id).unwrap();
            let rows = capture.finish();
            assert_eq!(rows.len(), 1);
            let row = &rows[0];
            assert!(row.complete);
            assert_eq!(row.position, position);
            assert_eq!(row.total_ns, row.attention_ns + row.ffn_ns + row.reentry_ns + row.other_ns);
            assert_eq!(row.provider_calls.len(), 2);
            for (layer, call) in row.provider_calls.iter().enumerate() {
                assert_eq!(call["layer"], layer);
                assert_eq!(call["complete"], true);
                assert!(call["request_bytes"].as_u64().unwrap() > 0);
                assert!(call["response_bytes"].as_u64().unwrap() > 0);
                let worker = &call["worker"];
                assert!(worker["ffn_ns"].as_u64().unwrap() <= worker["execute_ns"].as_u64().unwrap());
                assert!(worker["execute_ns"].as_u64().unwrap() <= worker["handler_ns"].as_u64().unwrap());
            }
            let capture = Capture::start().unwrap();
            let binary_logits = binary_session.step(id).unwrap();
            let binary_rows = capture.finish();
            assert_eq!(binary_logits.iter().map(|x| x.to_bits()).collect::<Vec<_>>(), expected.iter().map(|x| x.to_bits()).collect::<Vec<_>>());
            for call in &binary_rows[0].provider_calls {
                assert_eq!(call["request_bytes"], binary::HEADER_BYTES + 4 * ops.hidden());
                assert_eq!(call["response_bytes"], call["request_bytes"]);
            }
            let capture = Capture::start().unwrap();
            let stream_logits = stream_session.step(id).unwrap();
            let stream_rows = capture.finish();
            assert_eq!(stream_logits.iter().map(|x| x.to_bits()).collect::<Vec<_>>(), expected.iter().map(|x| x.to_bits()).collect::<Vec<_>>());
            for call in &stream_rows[0].provider_calls {
                assert_eq!(call["request_bytes"], binary::HEADER_BYTES + 4 * ops.hidden());
                assert_eq!(call["response_bytes"], call["request_bytes"]);
                assert!(call["telemetry_bytes"].as_u64().unwrap() > 0);
                assert!(call["websocket_overhead_bytes"].as_u64().unwrap() > 0);
                assert!(call["worker"]["ffn_ns"].as_u64().unwrap() <= call["worker"]["handler_ns"].as_u64().unwrap());
            }
            smoke.push(serde_json::json!({"token_id": id, "local": local_rows[0], "remote": rows[0]}));
            assert_eq!(
                expected.iter().map(|x| x.to_bits()).collect::<Vec<_>>(),
                actual.iter().map(|x| x.to_bits()).collect::<Vec<_>>()
            );
        }
        // Optional diagnostic artifact, explicitly a tiny debug fixture, not a benchmark.
        if let Some(path) = std::env::var_os("LARQL_V3_FFN_SMOKE_PROFILE") {
            use std::io::Write;
            let mut file = std::fs::OpenOptions::new().write(true).create_new(true).open(path).unwrap();
            writeln!(file, "{}", serde_json::json!({"schema": "larql.v3.ffn-smoke.v1", "benchmark": false, "layers": 2, "hidden": ops.hidden(), "profile": "debug synthetic HTTP loopback; no exclusivity or warmup claim"})).unwrap();
            for row in smoke { writeln!(file, "{row}").unwrap(); }
        }
    })
    .await
    .unwrap();
    for server in servers {
        server.abort();
    }
}
