//! N0.6 on V3 — tools / structured output through the V3 runtime.
//!
//! The schema → FSM → logits-mask pipeline is V2's
//! (`build_constrained_mask`), fed into the V3 driver's masked variant
//! (`continue_session_masked`): one grammar implementation, two
//! runtimes. These gates prove the wiring END TO END: the fixture's
//! tokenizer carries JSON *lexemes* as whole WordLevel tokens (`{`,
//! `}`, `:`, `,`, and the quoted strings the tool schema needs), so
//! the mask can steer the miniature model into emitting a genuinely
//! parseable tool call — no canned output, the grammar does the work.
//!
//! The tool's `parameters` are `{"type":"object","properties":{},
//! "additionalProperties":false}`, so `arguments` must be exactly the
//! empty object — the constrained emission is deterministic under
//! greedy sampling and the gate can assert the parsed call, not just
//! "some JSON".

mod common;

use std::path::{Path, PathBuf};
use std::sync::Arc;

use larql_server::bootstrap::{load_artifact, LoadVindexOptions, LoadedArtifact};
use larql_server::state::AppState;
use larql_vindex::format::vindex3::fixtures::{
    encode_fixture_container, miniature_glimmer, G_VOCAB,
};

use axum::body::Body;
use axum::http::{header, Request, StatusCode};
use tower::ServiceExt;

/// Token budget: the forced call is 9 lexemes; slack for key order.
const NEW_TOKENS: usize = 12;
/// The one tool the fixture can spell.
const TOOL_NAME: &str = "get";

/// WordLevel + WhitespaceSplit whose vocab carries the JSON lexemes a
/// `{"name":"get","arguments":{}}` call needs, as whole tokens. The
/// FSM replays token SURFACES (no join spaces), so grammar validity is
/// judged on the raw lexeme stream; the decoded TEXT is space-joined,
/// which serde tolerates between JSON tokens.
fn json_capable_tokenizer_json() -> String {
    let scaffold = ["User:", "Assistant:", "System:"];
    let lexemes = ["{", "}", ":", ",", "\"name\"", "\"arguments\"", "\"get\""];
    let word_ids = G_VOCAB - scaffold.len() - lexemes.len();
    let mut vocab = serde_json::Map::new();
    for i in 0..word_ids {
        vocab.insert(format!("[{i}]"), serde_json::json!(i));
    }
    for (k, lex) in lexemes.iter().enumerate() {
        vocab.insert((*lex).to_string(), serde_json::json!(word_ids + k));
    }
    for (k, word) in scaffold.iter().enumerate() {
        vocab.insert(
            (*word).to_string(),
            serde_json::json!(word_ids + lexemes.len() + k),
        );
    }
    serde_json::json!({
        "version": "1.0",
        "truncation": null,
        "padding": null,
        "added_tokens": [],
        "normalizer": null,
        "pre_tokenizer": {"type": "WhitespaceSplit"},
        "post_processor": null,
        "decoder": null,
        "model": {"type": "WordLevel", "vocab": vocab, "unk_token": "[0]"},
    })
    .to_string()
}

fn v3_container(root: &Path, name: &str) -> PathBuf {
    let checkpoint = root.join(format!("{name}-checkpoint"));
    let container = root.join(name);
    std::fs::create_dir_all(&checkpoint).unwrap();
    std::fs::create_dir_all(&container).unwrap();
    encode_fixture_container(miniature_glimmer, &checkpoint, &container, name);
    std::fs::write(
        container.join("tokenizer.json"),
        json_capable_tokenizer_json(),
    )
    .unwrap();
    container
}

fn v3_app(container: &Path) -> axum::Router {
    let artifact =
        load_artifact(&container.to_string_lossy(), LoadVindexOptions::default()).unwrap();
    let v3 = match artifact {
        LoadedArtifact::V3(m) => Arc::new(*m),
        LoadedArtifact::V2(_) => panic!("a VINDEX3 container must bind as V3"),
    };
    larql_server::routes::single_model_router(Arc::new(AppState {
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
            larql_server::response_kv::DEFAULT_MAX_ENTRIES,
            larql_server::response_kv::DEFAULT_TTL_SECS,
        ),
        runtime: Arc::new(larql_server::runtime_stats::RuntimeRecorder::new()),
    }))
}

fn tool_definition() -> serde_json::Value {
    serde_json::json!({
        "type": "function",
        "function": {
            "name": TOOL_NAME,
            "description": "the one call this vocabulary can spell",
            "parameters": {
                "type": "object",
                "properties": {},
                "additionalProperties": false
            }
        }
    })
}

async fn post(app: &axum::Router, uri: &str, body: serde_json::Value) -> (StatusCode, String) {
    let resp = app
        .clone()
        .oneshot(
            Request::builder()
                .method("POST")
                .uri(uri)
                .header(header::CONTENT_TYPE, "application/json")
                .body(Body::from(serde_json::to_vec(&body).unwrap()))
                .unwrap(),
        )
        .await
        .unwrap();
    let status = resp.status();
    let bytes = axum::body::to_bytes(resp.into_body(), usize::MAX)
        .await
        .unwrap();
    (status, String::from_utf8_lossy(&bytes).into_owned())
}

fn parse(body: &str) -> serde_json::Value {
    serde_json::from_str(body).unwrap_or_else(|_| serde_json::json!({"raw": body}))
}

// ── chat completions ─────────────────────────────────────────────────────────

#[tokio::test]
async fn v3_chat_tools_emit_a_parsed_tool_call() {
    let root = tempfile::tempdir().unwrap();
    let app = v3_app(&v3_container(root.path(), "tools-chat"));
    let (status, body) = post(
        &app,
        "/v1/chat/completions",
        serde_json::json!({
            "messages": [{"role": "user", "content": "[1]"}],
            "max_tokens": NEW_TOKENS,
            "tools": [tool_definition()],
        }),
    )
    .await;
    let v = parse(&body);
    assert_eq!(status, StatusCode::OK, "{v}");
    assert_eq!(v["choices"][0]["finish_reason"], "tool_calls", "{v}");
    let call = &v["choices"][0]["message"]["tool_calls"][0];
    assert_eq!(call["function"]["name"], TOOL_NAME, "{v}");
    let args: serde_json::Value =
        serde_json::from_str(call["function"]["arguments"].as_str().unwrap_or("null"))
            .expect("arguments must be JSON");
    assert_eq!(args, serde_json::json!({}), "{v}");
    assert!(
        v["choices"][0]["message"]["content"].is_null(),
        "a tool call carries no content: {v}"
    );
}

#[tokio::test]
async fn v3_chat_tools_stream_the_tool_call_chunk() {
    let root = tempfile::tempdir().unwrap();
    let app = v3_app(&v3_container(root.path(), "tools-chat-sse"));
    let (status, body) = post(
        &app,
        "/v1/chat/completions",
        serde_json::json!({
            "messages": [{"role": "user", "content": "[1]"}],
            "max_tokens": NEW_TOKENS,
            "stream": true,
            "tools": [tool_definition()],
        }),
    )
    .await;
    assert_eq!(status, StatusCode::OK, "{body}");
    assert!(body.contains(r#""tool_calls""#), "{body}");
    assert!(body.contains(TOOL_NAME), "{body}");
    assert!(body.contains(r#""finish_reason":"tool_calls""#), "{body}");
    assert!(body.contains("[DONE]"), "{body}");
}

#[tokio::test]
async fn v3_chat_json_object_yields_parseable_json() {
    let root = tempfile::tempdir().unwrap();
    let app = v3_app(&v3_container(root.path(), "tools-json"));
    let (status, body) = post(
        &app,
        "/v1/chat/completions",
        serde_json::json!({
            "messages": [{"role": "user", "content": "[1]"}],
            "max_tokens": NEW_TOKENS,
            "response_format": {"type": "json_object"},
        }),
    )
    .await;
    let v = parse(&body);
    assert_eq!(status, StatusCode::OK, "{v}");
    let content = v["choices"][0]["message"]["content"]
        .as_str()
        .unwrap_or_default();
    serde_json::from_str::<serde_json::Value>(content)
        .unwrap_or_else(|e| panic!("constrained output must parse as JSON ({e}): {content:?}"));
}

// ── responses api ────────────────────────────────────────────────────────────

#[tokio::test]
async fn v3_responses_tools_emit_a_function_call_item() {
    let root = tempfile::tempdir().unwrap();
    let app = v3_app(&v3_container(root.path(), "tools-resp"));
    let (status, body) = post(
        &app,
        "/v1/responses",
        serde_json::json!({
            "input": "[1]",
            "max_output_tokens": NEW_TOKENS,
            "tools": [{
                "type": "function",
                "name": TOOL_NAME,
                "parameters": {
                    "type": "object",
                    "properties": {},
                    "additionalProperties": false
                }
            }],
        }),
    )
    .await;
    let v = parse(&body);
    assert_eq!(status, StatusCode::OK, "{v}");
    let item = &v["output"][0];
    assert_eq!(item["type"], "function_call", "{v}");
    assert_eq!(item["name"], TOOL_NAME, "{v}");
    let args: serde_json::Value =
        serde_json::from_str(item["arguments"].as_str().unwrap_or("null"))
            .expect("arguments must be JSON");
    assert_eq!(args, serde_json::json!({}), "{v}");
}

#[tokio::test]
async fn v3_responses_tools_stream_the_function_call_item() {
    let root = tempfile::tempdir().unwrap();
    let app = v3_app(&v3_container(root.path(), "tools-resp-sse"));
    let (status, body) = post(
        &app,
        "/v1/responses",
        serde_json::json!({
            "input": "[1]",
            "max_output_tokens": NEW_TOKENS,
            "stream": true,
            "tools": [{
                "type": "function",
                "name": TOOL_NAME,
                "parameters": {
                    "type": "object",
                    "properties": {},
                    "additionalProperties": false
                }
            }],
        }),
    )
    .await;
    assert_eq!(status, StatusCode::OK, "{body}");
    assert!(body.contains("event: response.output_item.added"), "{body}");
    assert!(body.contains(r#""type":"function_call""#), "{body}");
    assert!(body.contains("event: response.completed"), "{body}");
    assert!(body.contains("[DONE]"), "{body}");
}

/// The fail-closed control: a vocabulary that cannot spell JSON makes
/// the mask reject every first candidate — the request errors instead
/// of emitting ungrammatical output (mirrors V2's
/// `MaskRejectedAllCandidates`).
#[tokio::test]
async fn v3_tools_fail_closed_on_a_json_incapable_vocabulary() {
    let root = tempfile::tempdir().unwrap();
    // Build a container with the words-only tokenizer: no JSON lexemes.
    let checkpoint = root.path().join("nojson-checkpoint");
    let container = root.path().join("nojson");
    std::fs::create_dir_all(&checkpoint).unwrap();
    std::fs::create_dir_all(&container).unwrap();
    encode_fixture_container(miniature_glimmer, &checkpoint, &container, "nojson");
    let mut vocab = serde_json::Map::new();
    for i in 0..G_VOCAB {
        vocab.insert(format!("[{i}]"), serde_json::json!(i));
    }
    std::fs::write(
        container.join("tokenizer.json"),
        serde_json::json!({
            "version": "1.0", "truncation": null, "padding": null,
            "added_tokens": [], "normalizer": null,
            "pre_tokenizer": {"type": "WhitespaceSplit"},
            "post_processor": null, "decoder": null,
            "model": {"type": "WordLevel", "vocab": vocab, "unk_token": "[0]"},
        })
        .to_string(),
    )
    .unwrap();

    let app = v3_app(&container);
    let (status, body) = post(
        &app,
        "/v1/chat/completions",
        serde_json::json!({
            "messages": [{"role": "user", "content": "[1]"}],
            "max_tokens": NEW_TOKENS,
            "tools": [tool_definition()],
        }),
    )
    .await;
    assert!(
        status.is_server_error(),
        "an unspellable grammar must fail closed, got {status}: {body}"
    );
}
