//! The PUBLIC_EXPLORER surface over HTTP.
//!
//! The invariant under test: `/v1/query` executes **real LQL** against
//! the served container through the same `Session::execute` seam every
//! transport uses — no simulation, no route-side filtering — and the
//! capability profile's judgement maps honestly onto HTTP: 200 for the
//! read surface, 403 (nothing failed) for a refused statement, 400 for
//! a parse error, 404 for routes that simply are not mounted.

use std::sync::Arc;

use axum::body::Body;
use axum::http::{Request, StatusCode};
use larql_inference::test_utils::synthetic_tokenizer_json;
use larql_server::bootstrap::{load_artifact, LoadVindexOptions, LoadedArtifact};
use larql_server::state::AppState;
use larql_vindex::format::vindex3::fixtures::{
    encode_fixture_container, miniature_glimmer, G_VOCAB,
};
use tower::ServiceExt;

fn v3_container() -> tempfile::TempDir {
    let checkpoint = tempfile::tempdir().unwrap();
    let container = tempfile::tempdir().unwrap();
    encode_fixture_container(
        miniature_glimmer,
        checkpoint.path(),
        container.path(),
        "explorer-fixture",
    );
    std::fs::write(
        container.path().join("tokenizer.json"),
        synthetic_tokenizer_json(G_VOCAB),
    )
    .unwrap();
    container
}

fn v3_state(container: &std::path::Path) -> Arc<AppState> {
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

/// The full public router over the fixture: state + bridged session.
fn public_app(container: &std::path::Path) -> axum::Router {
    let state = v3_state(container);
    let bridge = Arc::new(
        larql_server::lql_bridge::spawn(container, std::time::Duration::from_secs(60)).unwrap(),
    );
    larql_server::routes::public_explorer_router(state, bridge)
}

async fn query(app: &axum::Router, statement: &str) -> (StatusCode, serde_json::Value) {
    let req = Request::builder()
        .method("POST")
        .uri("/v1/query")
        .header("content-type", "application/json")
        .body(Body::from(
            serde_json::json!({ "statement": statement }).to_string(),
        ))
        .unwrap();
    let response = app.clone().oneshot(req).await.unwrap();
    let status = response.status();
    let bytes = axum::body::to_bytes(response.into_body(), usize::MAX)
        .await
        .unwrap();
    let body = serde_json::from_slice(&bytes).unwrap_or(serde_json::Value::Null);
    (status, body)
}

#[tokio::test]
async fn the_read_surface_executes_real_lql_over_http() {
    let container = v3_container();
    let app = public_app(container.path());

    for stmt in [
        "SHOW COMPONENTS;",
        "SHOW REPRESENTATIONS;",
        "SHOW PROVENANCE;",
        "SHOW AUTHORITY;",
        "SHOW LAYERS;",
        "STATS;",
        "INFER \"[3]\" GENERATE 2;",
    ] {
        let (status, body) = query(&app, stmt).await;
        assert_eq!(status, StatusCode::OK, "{stmt}: {body}");
        assert_eq!(body["profile"], "PUBLIC_EXPLORER", "{stmt}: {body}");
        assert!(
            !body["lines"].as_array().unwrap().is_empty(),
            "{stmt}: {body}"
        );
    }

    // The components really come from the served container's graph.
    let (_, body) = query(&app, "SHOW COMPONENTS;").await;
    let joined = body["lines"]
        .as_array()
        .unwrap()
        .iter()
        .map(|l| l.as_str().unwrap())
        .collect::<Vec<_>>()
        .join("\n");
    assert!(joined.contains("target"), "{joined}");
}

#[tokio::test]
async fn a_refused_statement_is_403_and_names_the_profile() {
    let container = v3_container();
    let app = public_app(container.path());

    for stmt in [
        "DELETE FROM EDGES WHERE layer = 0;",
        r#"INSERT INTO EDGES (entity, relation, target) VALUES ("a", "b", "c");"#,
        "USE \"/etc/anything.vindex\";",
        "COMPILE CURRENT INTO VINDEX \"out\";",
        "BEGIN PATCH \"p.vlp\";",
        "INFER \"[3]\" GENERATE 4096;",
    ] {
        let (status, body) = query(&app, stmt).await;
        assert_eq!(status, StatusCode::FORBIDDEN, "{stmt}: {body}");
        let msg = body["error"].as_str().unwrap();
        assert!(msg.contains("PUBLIC_EXPLORER"), "{stmt}: {msg}");
    }
}

#[tokio::test]
async fn a_parse_error_is_400_not_500() {
    let container = v3_container();
    let app = public_app(container.path());
    let (status, body) = query(&app, "FROBNICATE THE WEIGHTS;").await;
    assert_eq!(status, StatusCode::BAD_REQUEST, "{body}");
}

#[tokio::test]
async fn an_oversized_statement_is_413() {
    let container = v3_container();
    let app = public_app(container.path());
    let big = format!("FIND {};", "x".repeat(9 * 1024));
    let (status, _) = query(&app, &big).await;
    assert_eq!(status, StatusCode::PAYLOAD_TOO_LARGE);
}

/// Mutating routes are absent, not gated: the router never mounted
/// them, so there is nothing to mis-configure.
#[tokio::test]
async fn mutating_routes_are_not_mounted() {
    let container = v3_container();
    let app = public_app(container.path());

    for (method, uri) in [
        ("POST", "/v1/insert"),
        ("POST", "/v1/patches/apply"),
        ("POST", "/v1/infer"),
        ("POST", "/v1/runtime/model"),
        ("GET", "/v1/shard/x/0-1"),
        ("POST", "/v1/chat/completions"),
    ] {
        let req = Request::builder()
            .method(method)
            .uri(uri)
            .header("content-type", "application/json")
            .body(Body::from("{}"))
            .unwrap();
        let response = app.clone().oneshot(req).await.unwrap();
        assert_eq!(
            response.status(),
            StatusCode::NOT_FOUND,
            "{method} {uri} must not exist on the public router"
        );
    }
}

/// The typed container facts — the protocol's read nouns — answer
/// with structure a client can render, from the container alone.
#[tokio::test]
async fn container_facts_answer_typed() {
    let container = v3_container();
    let app = public_app(container.path());

    let get = |uri: &'static str| {
        let app = app.clone();
        async move {
            let req = Request::builder().uri(uri).body(Body::empty()).unwrap();
            let response = app.oneshot(req).await.unwrap();
            assert_eq!(response.status(), StatusCode::OK, "{uri}");
            let bytes = axum::body::to_bytes(response.into_body(), usize::MAX)
                .await
                .unwrap();
            serde_json::from_slice::<serde_json::Value>(&bytes).unwrap()
        }
    };

    let components = get("/v1/components").await;
    assert_eq!(components["components"][0]["id"], "target", "{components}");
    assert!(components["coherent"].as_bool().unwrap(), "{components}");

    let reps = get("/v1/representations").await;
    assert!(!reps["entries"].as_array().unwrap().is_empty(), "{reps}");

    let prov = get("/v1/provenance").await;
    let first = &prov["entries"][0];
    assert!(
        first["payload_sha256"].as_str().unwrap().len() >= 32,
        "{prov}"
    );

    let auth = get("/v1/authority").await;
    assert!(
        auth["authority"] == "canonical" || auth["authority"] == "derived",
        "{auth}"
    );
}

/// The read-only REST routes stay available beside /v1/query — the
/// protocol has two dialects (REST nouns and LQL statements) over one
/// state.
#[tokio::test]
async fn the_rest_read_routes_are_mounted() {
    let container = v3_container();
    let app = public_app(container.path());
    for uri in ["/v1/health", "/v1/models", "/v1/stats"] {
        let req = Request::builder().uri(uri).body(Body::empty()).unwrap();
        let response = app.clone().oneshot(req).await.unwrap();
        assert_eq!(response.status(), StatusCode::OK, "{uri}");
    }
}
