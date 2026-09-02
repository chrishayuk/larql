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

/// A V3 container must refuse the slicing / service-mode options rather
/// than accept and ignore them. Accepting `--layers 0-9` silently
/// loaded the *whole* model and answered complete requests; `--no-infer`
/// did not disable inference.
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
            "--layers",
            LoadVindexOptions {
                layer_range: Some((0, 1)),
                ..LoadVindexOptions::default()
            },
        ),
        (
            "--ffn-only",
            LoadVindexOptions {
                ffn_only: true,
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
