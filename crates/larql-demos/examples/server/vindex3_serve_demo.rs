//! Serving a VINDEX3 container over the normal OpenAI-style API
//! (VI3-SERVE-1), end to end and self-contained.
//!
//! Encodes the miniature judged-semantics checkpoint into a real
//! VINDEX3 container (with a tokenizer), binds it through the SAME
//! `load_artifact` path the production bootstrap uses, builds the
//! real single-model router, and drives `/v1/completions` as a
//! client — buffered JSON and SSE streaming — plus `/v1/models`.
//!
//! The generation behind every token: batch prefill into caller-owned
//! continuation state (`CanonicalKvState`), resumed decode through
//! the canonical plan executor. No `ModelWeights`, no `VectorIndex`,
//! no `ModelArchitecture` anywhere on the path.
//!
//! Run: cargo run -p larql-demos --example vindex3_serve_demo
//!
//! Pass a container path to serve a REAL container instead (it must
//! carry `tokenizer.json` — the text API cannot serve ids-only):
//!
//! ```sh
//! cargo run -p larql-demos --example vindex3_serve_demo -- path/to/model.vindex3
//! ```

use std::sync::atomic::AtomicU64;
use std::sync::Arc;
use std::time::Instant;

use axum::body::Body;
use axum::http::{header, Request};
use larql_inference::test_utils::synthetic_tokenizer_json;
use larql_server::bootstrap::{load_artifact, LoadVindexOptions, LoadedArtifact};
use larql_server::cache::DescribeCache;
use larql_server::session::SessionManager;
use larql_server::state::AppState;
use larql_vindex::format::vindex3::fixtures::{
    encode_fixture_container, miniature_glimmer, G_VOCAB,
};
use serde_json::Value;
use tower::ServiceExt;

async fn post(app: axum::Router, path: &str, body: &Value) -> (axum::http::StatusCode, String) {
    let resp = app
        .oneshot(
            Request::builder()
                .method("POST")
                .uri(path)
                .header(header::CONTENT_TYPE, "application/json")
                .body(Body::from(serde_json::to_vec(body).unwrap()))
                .unwrap(),
        )
        .await
        .expect("oneshot post");
    let status = resp.status();
    let bytes = axum::body::to_bytes(resp.into_body(), 64 * 1024 * 1024)
        .await
        .expect("read body");
    (status, String::from_utf8_lossy(&bytes).into_owned())
}

async fn get(app: axum::Router, path: &str) -> String {
    let resp = app
        .oneshot(Request::builder().uri(path).body(Body::empty()).unwrap())
        .await
        .expect("oneshot get");
    let bytes = axum::body::to_bytes(resp.into_body(), 64 * 1024 * 1024)
        .await
        .expect("read body");
    String::from_utf8_lossy(&bytes).into_owned()
}

#[tokio::main(flavor = "current_thread")]
async fn main() {
    println!("=== VINDEX3 over /v1/completions (VI3-SERVE-1) ===\n");

    // A container to serve: the one on the command line, or a
    // self-encoded miniature with a tokenizer.
    let _tmp;
    let container_path = match std::env::args().nth(1) {
        Some(path) => std::path::PathBuf::from(path),
        None => {
            let checkpoint = tempfile::tempdir().expect("tempdir");
            let container = tempfile::tempdir().expect("tempdir");
            encode_fixture_container(
                miniature_glimmer,
                checkpoint.path(),
                container.path(),
                "serve-demo-glimmer",
            );
            std::fs::write(
                container.path().join("tokenizer.json"),
                synthetic_tokenizer_json(G_VOCAB),
            )
            .expect("write tokenizer");
            let path = container.path().to_path_buf();
            _tmp = container;
            path
        }
    };

    // Bind through the production bootstrap path: the container's own
    // generation marker decides V3; a V2 vindex would bind V2 here.
    println!("binding {} ...", container_path.display());
    let artifact = load_artifact(
        &container_path.to_string_lossy(),
        LoadVindexOptions::default(),
    )
    .expect("bind container");
    let v3 = match artifact {
        LoadedArtifact::V3(model) => Arc::new(*model),
        LoadedArtifact::V2(_) => panic!("expected a VINDEX3 container"),
    };
    println!("bound: model `{}` (VINDEX3)\n", v3.id);

    let state = Arc::new(AppState {
        model_set: std::sync::RwLock::new(larql_server::state::ModelSet {
            models: Vec::new(),
            v3_models: vec![v3],
            aliases: Default::default(),
        }),
        router_topology: larql_server::state::RouterTopology::SingleModel,
        lifecycle: std::sync::Mutex::new(larql_server::state::LifecycleState::Idle),
        started_at: Instant::now(),
        requests_served: AtomicU64::new(0),
        api_key: None,
        sessions: SessionManager::new(3600),
        describe_cache: DescribeCache::new(60),
        infer_timeout: std::time::Duration::from_secs(60),
        patch_sources: Default::default(),
        responses: larql_server::response_store::ResponseStore::new(),
        v3_kv: larql_server::response_kv::ResponseKvCache::new(
            larql_server::response_kv::DEFAULT_MAX_ENTRIES,
            larql_server::response_kv::DEFAULT_TTL_SECS,
        ),
        runtime: Arc::new(larql_server::runtime_stats::RuntimeRecorder::new()),
    });

    // GET /v1/models — the registry lists the V3 model.
    let models = get(
        larql_server::routes::single_model_router(state.clone()),
        "/v1/models",
    )
    .await;
    println!("GET /v1/models\n{models}\n");

    // POST /v1/completions — buffered.
    let (status, body) = post(
        larql_server::routes::single_model_router(state.clone()),
        "/v1/completions",
        &serde_json::json!({"prompt": "[3]", "max_tokens": 12}),
    )
    .await;
    println!("POST /v1/completions (buffered) -> {status}\n{body}\n");

    // POST /v1/completions — SSE streaming, one chunk per token.
    let (status, body) = post(
        larql_server::routes::single_model_router(state.clone()),
        "/v1/completions",
        &serde_json::json!({"prompt": "[3]", "max_tokens": 8, "stream": true}),
    )
    .await;
    println!("POST /v1/completions (stream) -> {status}\n{body}");

    println!("=== Done ===");
}
