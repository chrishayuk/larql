//! Coverage for `GET /v1/runtime` (`routes/runtime.rs`) and the
//! recorder it snapshots (`runtime_stats.rs`).
//!
//! Three things worth pinning end-to-end, not just in the recorder's
//! own unit tests:
//!   - the zero-model boot reports `model: null` rather than 404 or a
//!     fabricated value;
//!   - a bound VINDEX2 model's `model`/`memory` block reflects the
//!     real config, not a guess;
//!   - a completed generation on `/v1/completions` actually shows up
//!     in the next `/v1/runtime` read — proving the recorder wiring in
//!     `routes/openai/completions.rs` is reached in production, not
//!     just exercised by `runtime_stats`'s own unit tests.

mod common;

use std::sync::Arc;

use axum::body::Body;
use axum::http::{header, Request, StatusCode};
use tower::ServiceExt;

use larql_inference::test_utils::synthetic_tokenizer_json;
use larql_server::bootstrap::{load_artifact, LoadVindexOptions, LoadedArtifact};
use larql_server::state::AppState;
use larql_vindex::format::vindex3::fixtures::{
    encode_fixture_container, miniature_glimmer, G_VOCAB,
};

/// A serving state holding only a VINDEX3 model — the same
/// `encode_fixture_container` + `load_artifact` binding path
/// `test_vindex3_serve.rs` uses, kept minimal here since `/v1/runtime`
/// never runs generation against it.
fn v3_only_state() -> (Arc<AppState>, tempfile::TempDir) {
    let checkpoint = tempfile::tempdir().unwrap();
    let container = tempfile::tempdir().unwrap();
    encode_fixture_container(
        miniature_glimmer,
        checkpoint.path(),
        container.path(),
        "runtime-fixture",
    );
    std::fs::write(
        container.path().join("tokenizer.json"),
        synthetic_tokenizer_json(G_VOCAB),
    )
    .unwrap();
    let artifact = load_artifact(
        &container.path().to_string_lossy(),
        LoadVindexOptions::default(),
    )
    .unwrap();
    let v3 = match artifact {
        LoadedArtifact::V3(m) => Arc::new(*m),
        LoadedArtifact::V2(_) => panic!("a VINDEX3 container must bind as V3"),
    };
    let state = Arc::new(AppState {
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
    });
    (state, container)
}

async fn get_runtime(app: axum::Router) -> serde_json::Value {
    let resp = app
        .oneshot(
            Request::builder()
                .method("GET")
                .uri("/v1/runtime")
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(resp.status(), StatusCode::OK);
    let bytes = axum::body::to_bytes(resp.into_body(), usize::MAX)
        .await
        .unwrap();
    serde_json::from_slice(&bytes).unwrap()
}

#[tokio::test]
async fn runtime_reports_no_model_and_no_activity_before_any_request() {
    let state = common::state(vec![]);
    let app = larql_server::routes::single_model_router(state);
    let v = get_runtime(app).await;

    assert_eq!(v["status"], "ready");
    assert!(v["model"].is_null(), "no model bound → null, not a 404");
    assert!(v["memory"]["model_bytes"].is_null());
    assert!(v["performance"]["decode_tokens_per_second"].is_null());
    assert!(v["performance"]["prefill_tokens_per_second"].is_null());
    assert!(v["performance"]["last_request_latency_ms"].is_null());
    assert_eq!(v["generation"]["active"], false);
    assert_eq!(v["generation"]["active_requests"], 0);
    // uptime_ms / version are process facts — just confirm they're the
    // right shape, not a specific value.
    assert!(v["uptime_ms"].as_u64().is_some());
    assert_eq!(v["version"], env!("CARGO_PKG_VERSION"));
}

#[tokio::test]
async fn runtime_reports_the_bound_v2_model() {
    let (model, fixture) = common::model_with_real_weights("synthetic");
    let state = common::state(vec![model]);
    let app = larql_server::routes::single_model_router(state);
    let v = get_runtime(app).await;
    drop(fixture);

    assert_eq!(v["model"]["id"], "synthetic");
    // `architecture` is `VindexConfig::family`, whatever the fixture's
    // `build_vindex` call derived it as — not pinned to a literal
    // here, just confirmed to be the real (non-empty) field rather
    // than a placeholder.
    assert!(!v["model"]["architecture"].as_str().unwrap().is_empty());
    assert_eq!(v["model"]["format"], "vindex2");
    assert_eq!(v["model"]["quantization"], "none");
    assert!(
        v["memory"]["model_bytes"].as_u64().unwrap() > 0,
        "estimate_resident_bytes should report something nonzero for a real config: {v}"
    );
    // resident_bytes is `getrusage`'s peak RSS for this test process —
    // real regardless of which model is bound. `getrusage` has no
    // Windows equivalent (`runtime_stats::resident_bytes` returns
    // `None` there by design), so the assertion is platform-gated the
    // same way `resident_bytes`'s own unit test is.
    #[cfg(unix)]
    assert!(v["memory"]["resident_bytes"].as_u64().unwrap() > 0);
    #[cfg(not(unix))]
    assert!(v["memory"]["resident_bytes"].is_null());
    assert_eq!(
        v["backend"]["metal_compiled"],
        cfg!(all(
            any(feature = "metal-experts", feature = "vindex3-metal"),
            target_os = "macos"
        ))
    );
}

#[tokio::test]
async fn runtime_records_performance_after_a_completion() {
    let (model, fixture) = common::model_with_q4k_weights("synthetic");
    let state = common::state(vec![model]);
    let app = larql_server::routes::single_model_router(state);

    let post_resp = app
        .clone()
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/v1/completions")
                .header(header::CONTENT_TYPE, "application/json")
                .body(Body::from(
                    serde_json::to_vec(&serde_json::json!({
                        "model": "synthetic",
                        "prompt": "the capital of France is",
                        "max_tokens": 4,
                    }))
                    .unwrap(),
                ))
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(post_resp.status(), StatusCode::OK);
    let completion_bytes = axum::body::to_bytes(post_resp.into_body(), usize::MAX)
        .await
        .unwrap();
    let completion: serde_json::Value = serde_json::from_slice(&completion_bytes).unwrap();
    let completion_tokens = completion["usage"]["completion_tokens"].as_u64().unwrap();

    let v = get_runtime(app).await;
    drop(fixture);

    // Every completed request leaves a latency sample, tokens or not.
    assert!(
        v["performance"]["last_request_latency_ms"]
            .as_f64()
            .unwrap()
            >= 0.0,
        "expected a recorded latency after a completed generation: {v}"
    );
    // The guard is scoped to the request's own generation work, so it
    // must be back to zero once the response has been built.
    assert_eq!(v["generation"]["active_requests"], 0);
    assert_eq!(v["generation"]["active"], false);
    if completion_tokens > 0 {
        assert!(
            v["performance"]["decode_tokens_per_second"]
                .as_f64()
                .unwrap()
                > 0.0,
            "generated {completion_tokens} tokens but decode_tokens_per_second is not \
             a positive number: {v}"
        );
    }
}

#[tokio::test]
async fn runtime_reports_the_bound_v3_model() {
    let (state, container) = v3_only_state();
    let app = larql_server::routes::single_model_router(state);
    let v = get_runtime(app).await;
    drop(container);

    assert_eq!(v["model"]["format"], "vindex3");
    assert!(!v["model"]["id"].as_str().unwrap().is_empty());
    assert!(!v["model"]["architecture"].as_str().unwrap().is_empty());
    // A V3 container carries no single top-level quant tag the way a
    // V2 index.json does — `null`, not a guess.
    assert!(v["model"]["quantization"].is_null());
    // No V2 estimator applies to a V3 container yet either.
    assert!(v["memory"]["model_bytes"].is_null());
}
