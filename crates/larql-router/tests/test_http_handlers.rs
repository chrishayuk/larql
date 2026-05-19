//! End-to-end tests for the router's HTTP surface (`/v1/walk-ffn`,
//! `/v1/stats`, `/v1/health`).
//!
//! Each test stands up a loopback "fake shard" that echoes the request
//! back as JSON, points the router at it via `--shards`, then drives the
//! router via real HTTP requests. This exercises `handle_walk_ffn`,
//! `handle_walk_ffn_inner`, `proxy_raw`, `handle_stats`, and `handle_health`
//! end-to-end.

use std::net::SocketAddr;
use std::sync::Arc;
use std::time::Duration;

use axum::body::Body;
use axum::extract::State;
use axum::http::{header, Request, StatusCode};
use axum::routing::{get, post};
use axum::Json;
use serde_json::{json, Value};
use tokio::sync::Mutex;
use tower::ServiceExt; // for `oneshot`

use larql_router::http::{build_router, AppState, BINARY_CT};
use larql_router::shards::parse_shards;

// ── Mock shard server ────────────────────────────────────────────────────────

#[derive(Clone, Default)]
struct ShardCalls {
    inner: Arc<Mutex<Vec<Value>>>,
}

async fn fake_walk_ffn(
    State(calls): State<ShardCalls>,
    body: axum::extract::Json<Value>,
) -> Json<Value> {
    calls.inner.lock().await.push(body.0.clone());

    // Echo back the layer(s) plus a fake latency so the router's merge
    // path has a concrete max latency to surface.
    let body = &body.0;
    let layer = body.get("layer").and_then(|v| v.as_u64()).unwrap_or(0);
    let layers = body
        .get("layers")
        .and_then(|v| v.as_array())
        .cloned()
        .unwrap_or_default();
    let results: Vec<Value> = if layers.is_empty() {
        vec![json!({"layer": layer, "value": "ok"})]
    } else {
        layers
            .iter()
            .map(|l| json!({"layer": l.as_u64().unwrap_or(0), "value": "ok"}))
            .collect()
    };
    Json(json!({
        "results": results,
        "latency_ms": 5.5,
    }))
}

async fn fake_walk_ffn_binary(
    State(_calls): State<ShardCalls>,
    body: axum::body::Bytes,
) -> axum::response::Response {
    // For binary requests we mirror the body back so the router's
    // proxy_raw path can be inspected.
    axum::response::Response::builder()
        .status(StatusCode::OK)
        .header(header::CONTENT_TYPE, BINARY_CT)
        .body(Body::from(body))
        .unwrap()
}

async fn fake_stats() -> Json<Value> {
    Json(json!({"hidden_size": 2560, "num_layers": 34}))
}

async fn spawn_fake_shard() -> (SocketAddr, ShardCalls) {
    let calls = ShardCalls::default();
    let app_calls = calls.clone();
    let app = axum::Router::new()
        .route(
            "/v1/walk-ffn",
            post(
                |st: State<ShardCalls>, req: axum::extract::Request| async move {
                    let is_binary = req
                        .headers()
                        .get(header::CONTENT_TYPE)
                        .and_then(|v| v.to_str().ok())
                        .map(|ct| ct.starts_with(BINARY_CT))
                        .unwrap_or(false);
                    if is_binary {
                        let body = axum::body::to_bytes(req.into_body(), 64 * 1024 * 1024)
                            .await
                            .unwrap();
                        fake_walk_ffn_binary(st, body).await
                    } else {
                        let body = axum::body::to_bytes(req.into_body(), 64 * 1024 * 1024)
                            .await
                            .unwrap();
                        let json: Value = serde_json::from_slice(&body).unwrap();
                        fake_walk_ffn(st, axum::extract::Json(json))
                            .await
                            .into_response()
                    }
                },
            ),
        )
        .route("/v1/stats", get(fake_stats))
        .with_state(app_calls);

    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap();
    tokio::spawn(async move {
        axum::serve(listener, app).await.unwrap();
    });
    tokio::time::sleep(Duration::from_millis(50)).await;
    (addr, calls)
}

use axum::response::IntoResponse;

fn make_router(static_shards: &str) -> axum::Router {
    let shards = parse_shards(static_shards).unwrap();
    let client = reqwest::Client::builder()
        .timeout(Duration::from_secs(5))
        .build()
        .unwrap();
    let state = Arc::new(AppState {
        static_shards: shards,
        grid: None,
        client,
        metrics: None,
        #[cfg(feature = "http3")]
        h3_client: None,
        hedge_after: None,
    });
    build_router(state)
}

// ── Tests ────────────────────────────────────────────────────────────────────

#[tokio::test]
async fn health_endpoint_returns_ok() {
    let app = make_router("0-3=http://127.0.0.1:1"); // shard URL unused for /v1/health
    let resp = app
        .oneshot(
            Request::builder()
                .uri("/v1/health")
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(resp.status(), StatusCode::OK);
    let body = axum::body::to_bytes(resp.into_body(), 1024).await.unwrap();
    let v: Value = serde_json::from_slice(&body).unwrap();
    assert_eq!(v["status"], "ok");
}

#[tokio::test]
async fn walk_ffn_rejects_invalid_json_body() {
    let app = make_router("0-3=http://127.0.0.1:1");
    let resp = app
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/v1/walk-ffn")
                .header(header::CONTENT_TYPE, "application/json")
                .body(Body::from("not json"))
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(resp.status(), StatusCode::BAD_REQUEST);
    let body = axum::body::to_bytes(resp.into_body(), 1024).await.unwrap();
    let v: Value = serde_json::from_slice(&body).unwrap();
    assert!(v["error"].as_str().unwrap().contains("invalid JSON"));
}

#[tokio::test]
async fn walk_ffn_rejects_missing_layer_field() {
    let app = make_router("0-3=http://127.0.0.1:1");
    let resp = app
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/v1/walk-ffn")
                .header(header::CONTENT_TYPE, "application/json")
                .body(Body::from(r#"{"foo":"bar"}"#))
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(resp.status(), StatusCode::BAD_REQUEST);
    let body = axum::body::to_bytes(resp.into_body(), 1024).await.unwrap();
    let v: Value = serde_json::from_slice(&body).unwrap();
    assert!(v["error"].as_str().unwrap().contains("must provide"));
}

#[tokio::test]
async fn walk_ffn_rejects_empty_layer_list() {
    let app = make_router("0-3=http://127.0.0.1:1");
    let resp = app
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/v1/walk-ffn")
                .header(header::CONTENT_TYPE, "application/json")
                .body(Body::from(r#"{"layers":[]}"#))
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(resp.status(), StatusCode::BAD_REQUEST);
}

#[tokio::test]
async fn walk_ffn_rejects_layer_outside_shard_map() {
    let app = make_router("0-3=http://127.0.0.1:1");
    let resp = app
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/v1/walk-ffn")
                .header(header::CONTENT_TYPE, "application/json")
                .body(Body::from(r#"{"layer":99}"#))
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(resp.status(), StatusCode::BAD_REQUEST);
    let body = axum::body::to_bytes(resp.into_body(), 1024).await.unwrap();
    let v: Value = serde_json::from_slice(&body).unwrap();
    assert!(v["error"].as_str().unwrap().contains("no owning shard"));
}

#[tokio::test]
async fn walk_ffn_proxies_single_shard_json_unchanged() {
    let (addr, calls) = spawn_fake_shard().await;
    let app = make_router(&format!("0-3=http://{addr}"));
    let resp = app
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/v1/walk-ffn")
                .header(header::CONTENT_TYPE, "application/json")
                .body(Body::from(r#"{"layer":2}"#))
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(resp.status(), StatusCode::OK);
    let body = axum::body::to_bytes(resp.into_body(), 4096).await.unwrap();
    let v: Value = serde_json::from_slice(&body).unwrap();
    assert!(v["results"].is_array());
    let stored = calls.inner.lock().await.clone();
    assert_eq!(stored.len(), 1);
    assert_eq!(stored[0]["layer"], 2);
}

#[tokio::test]
async fn walk_ffn_fans_out_to_multiple_shards_and_merges() {
    let (addr_a, _calls_a) = spawn_fake_shard().await;
    let (addr_b, _calls_b) = spawn_fake_shard().await;
    let app = make_router(&format!("0-3=http://{addr_a},4-7=http://{addr_b}"));
    let resp = app
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/v1/walk-ffn")
                .header(header::CONTENT_TYPE, "application/json")
                .body(Body::from(r#"{"layers":[1,5]}"#))
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(resp.status(), StatusCode::OK);
    let body = axum::body::to_bytes(resp.into_body(), 4096).await.unwrap();
    let v: Value = serde_json::from_slice(&body).unwrap();
    let results = v["results"].as_array().unwrap();
    assert_eq!(results.len(), 2);
    // Sorted by layer.
    assert_eq!(results[0]["layer"], 1);
    assert_eq!(results[1]["layer"], 5);
    // latency_ms is the max of both shards (both reported 5.5).
    assert!((v["latency_ms"].as_f64().unwrap() - 5.5).abs() < 1e-6);
}

#[tokio::test]
async fn walk_ffn_binary_single_shard_round_trips() {
    let (addr, _) = spawn_fake_shard().await;
    let app = make_router(&format!("0-3=http://{addr}"));
    // Binary body: 4-byte little-endian layer id only.
    let body = 2u32.to_le_bytes().to_vec();
    let resp = app
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/v1/walk-ffn")
                .header(header::CONTENT_TYPE, BINARY_CT)
                .body(Body::from(body.clone()))
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(resp.status(), StatusCode::OK);
    let resp_ct = resp
        .headers()
        .get(header::CONTENT_TYPE)
        .map(|v| v.to_str().unwrap().to_string());
    assert_eq!(resp_ct.as_deref(), Some(BINARY_CT));
    let echoed = axum::body::to_bytes(resp.into_body(), 1024).await.unwrap();
    assert_eq!(echoed.as_ref(), body.as_slice());
}

#[tokio::test]
async fn walk_ffn_binary_rejects_truncated_header() {
    let app = make_router("0-3=http://127.0.0.1:1");
    let resp = app
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/v1/walk-ffn")
                .header(header::CONTENT_TYPE, BINARY_CT)
                .body(Body::from(vec![0u8, 1u8])) // < 4 bytes
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(resp.status(), StatusCode::BAD_REQUEST);
}

#[tokio::test]
async fn walk_ffn_binary_rejects_multi_shard_fanout() {
    let app = make_router("0-3=http://127.0.0.1:1,4-7=http://127.0.0.1:2");
    // Binary batch header: layers 1 and 5 live on different shards.
    let mut body = Vec::new();
    body.extend_from_slice(&0xFFFF_FFFFu32.to_le_bytes()); // BATCH_MARKER
    body.extend_from_slice(&2u32.to_le_bytes()); // n=2
    body.extend_from_slice(&1u32.to_le_bytes());
    body.extend_from_slice(&5u32.to_le_bytes());
    let resp = app
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/v1/walk-ffn")
                .header(header::CONTENT_TYPE, BINARY_CT)
                .body(Body::from(body))
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(resp.status(), StatusCode::BAD_REQUEST);
    let body = axum::body::to_bytes(resp.into_body(), 1024).await.unwrap();
    let v: Value = serde_json::from_slice(&body).unwrap();
    assert!(v["error"].as_str().unwrap().contains("binary fan-out"));
}

#[tokio::test]
async fn stats_proxies_to_first_reachable_shard() {
    let (addr, _) = spawn_fake_shard().await;
    let app = make_router(&format!("0-3=http://{addr}"));
    let resp = app
        .oneshot(
            Request::builder()
                .uri("/v1/stats")
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(resp.status(), StatusCode::OK);
    let body = axum::body::to_bytes(resp.into_body(), 1024).await.unwrap();
    let v: Value = serde_json::from_slice(&body).unwrap();
    assert_eq!(v["hidden_size"], 2560);
}

#[tokio::test]
async fn stats_returns_503_when_no_shard_reachable() {
    let app = make_router("0-3=http://127.0.0.1:1"); // unreachable
    let resp = app
        .oneshot(
            Request::builder()
                .uri("/v1/stats")
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(resp.status(), StatusCode::SERVICE_UNAVAILABLE);
    let body = axum::body::to_bytes(resp.into_body(), 1024).await.unwrap();
    let v: Value = serde_json::from_slice(&body).unwrap();
    assert!(v["error"].as_str().unwrap().contains("no shard reachable"));
}

#[tokio::test]
async fn walk_ffn_routes_via_grid_when_grid_state_is_set() {
    use larql_router::grid::{GridState, ServerEntry};
    use parking_lot::RwLock;
    use std::collections::HashMap;

    let (addr, calls) = spawn_fake_shard().await;
    let grid = Arc::new(RwLock::new(GridState::default()));
    grid.write().register(ServerEntry {
        server_id: "grid-srv".into(),
        listen_url: format!("http://{addr}"),
        model_id: "m".into(),
        layer_start: 0,
        layer_end: 9,
        vindex_hash: "h".into(),
        cpu_pct: 0.0,
        ram_used: 0,
        requests_in_flight: 0,
        last_seen: std::time::Instant::now(),
        layer_latencies: HashMap::new(),
        req_per_sec: 0.0,
        rtt_ms: None,
        expert_start: 0,
        expert_end: 0,
    });

    let client = reqwest::Client::builder()
        .timeout(Duration::from_secs(5))
        .build()
        .unwrap();
    let state = Arc::new(AppState {
        static_shards: parse_shards("99-100=http://unused:1").unwrap(),
        grid: Some(grid),
        client,
        metrics: None,
        #[cfg(feature = "http3")]
        h3_client: None,
        hedge_after: None,
    });
    let app = build_router(state);

    // Request layer 3 — covered by the grid, not the static map.
    let resp = app
        .clone()
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/v1/walk-ffn")
                .header(header::CONTENT_TYPE, "application/json")
                .body(Body::from(r#"{"model_id":"m","layer":3}"#))
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(resp.status(), StatusCode::OK);
    let stored = calls.inner.lock().await.clone();
    assert_eq!(stored.len(), 1);
    assert_eq!(stored[0]["layer"], 3);

    // /v1/stats hits the grid first; should reach the same fake shard.
    let stats = app
        .oneshot(
            Request::builder()
                .uri("/v1/stats")
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(stats.status(), StatusCode::OK);
    let body = axum::body::to_bytes(stats.into_body(), 1024).await.unwrap();
    let v: Value = serde_json::from_slice(&body).unwrap();
    assert_eq!(v["hidden_size"], 2560);
}

#[tokio::test]
async fn walk_ffn_grid_layer_missing_falls_back_to_static_shards() {
    use larql_router::grid::GridState;
    use parking_lot::RwLock;

    let (addr, _calls) = spawn_fake_shard().await;
    let grid = Arc::new(RwLock::new(GridState::default()));
    // No servers registered — grid lookup returns None for every layer.
    // Static map covers it.

    let client = reqwest::Client::builder()
        .timeout(Duration::from_secs(5))
        .build()
        .unwrap();
    let state = Arc::new(AppState {
        static_shards: parse_shards(&format!("0-9=http://{addr}")).unwrap(),
        grid: Some(grid),
        client,
        metrics: None,
        #[cfg(feature = "http3")]
        h3_client: None,
        hedge_after: None,
    });
    let app = build_router(state);

    let resp = app
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/v1/walk-ffn")
                .header(header::CONTENT_TYPE, "application/json")
                .body(Body::from(r#"{"layer":5}"#))
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(resp.status(), StatusCode::OK);
}

#[tokio::test]
async fn walk_ffn_500s_on_shard_connection_failure() {
    let app = make_router("0-3=http://127.0.0.1:1"); // unreachable
    let resp = app
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/v1/walk-ffn")
                .header(header::CONTENT_TYPE, "application/json")
                .body(Body::from(r#"{"layer":0}"#))
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(resp.status(), StatusCode::BAD_GATEWAY);
}

// ── ADR-0017: /metrics endpoint ─────────────────────────────────────────────

/// `/metrics` returns Prometheus text format when a registry is wired
/// in. Every documented metric family appears in the output with a
/// pre-touched zero value (so dashboards don't see "missing metric"
/// for a freshly-started router).
#[tokio::test]
async fn metrics_endpoint_serves_prometheus_text_with_zero_values() {
    use larql_router::metrics::RouterMetrics;

    let shards = parse_shards("0-3=http://127.0.0.1:1").unwrap();
    let client = reqwest::Client::builder()
        .timeout(Duration::from_secs(1))
        .build()
        .unwrap();
    let metrics = RouterMetrics::new();
    let state = Arc::new(AppState {
        static_shards: shards,
        grid: None,
        client,
        metrics: Some(metrics.clone()),
        #[cfg(feature = "http3")]
        h3_client: None,
        hedge_after: None,
    });
    let app = build_router(state);

    let resp = app
        .oneshot(
            Request::builder()
                .uri("/metrics")
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(resp.status(), StatusCode::OK);
    let ct = resp
        .headers()
        .get(header::CONTENT_TYPE)
        .and_then(|v| v.to_str().ok())
        .unwrap_or("");
    assert!(
        ct.starts_with("text/plain"),
        "/metrics must serve text/plain, got {ct}"
    );
    let body = axum::body::to_bytes(resp.into_body(), usize::MAX)
        .await
        .unwrap();
    let text = String::from_utf8(body.to_vec()).unwrap();
    for required in [
        "larql_router_build_info",
        "larql_router_grid_servers",
        "larql_router_grid_models",
        "larql_router_grid_coverage_gaps",
        "larql_router_grid_elevated_ranges",
        "larql_router_target_replicas",
        "larql_router_grid_registers_total",
        "larql_router_grid_deregisters_total",
        "larql_router_rebalancer_actions_total",
        "larql_router_rtt_probes_total",
        "larql_router_walk_ffn_requests_total",
        "larql_router_walk_ffn_duration_seconds",
    ] {
        assert!(
            text.contains(required),
            "/metrics output missing {required}; got:\n{text}"
        );
    }
}

/// `/metrics` returns 503 when the AppState lacks a registry —
/// integration tests sometimes build a router without one, and the
/// handler shouldn't panic.
#[tokio::test]
async fn metrics_endpoint_returns_503_when_no_registry() {
    let app = make_router("0-3=http://127.0.0.1:1");
    let resp = app
        .oneshot(
            Request::builder()
                .uri("/metrics")
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(resp.status(), StatusCode::SERVICE_UNAVAILABLE);
}

// ── ADR-0018: MoE expert routing dispatch ──────────────────────────────────

/// MoE request with no grid configured 503s — there's no static-shard
/// fallback path for expert routing.
#[tokio::test]
async fn moe_request_without_grid_returns_503() {
    let app = make_router("0-3=http://127.0.0.1:1");
    let resp = app
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/v1/walk-ffn")
                .header(header::CONTENT_TYPE, "application/json")
                .body(Body::from(r#"{"layer":0,"experts":[0,3]}"#))
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(resp.status(), StatusCode::SERVICE_UNAVAILABLE);
    let body = axum::body::to_bytes(resp.into_body(), 1024).await.unwrap();
    let v: Value = serde_json::from_slice(&body).unwrap();
    assert!(v["error"]
        .as_str()
        .unwrap()
        .contains("MoE routing requires"));
}

/// MoE request against a grid with no shard owning the requested
/// `(layer, expert)` returns 503.
#[tokio::test]
async fn moe_request_with_no_owner_returns_503() {
    use larql_router::grid::{GridState, ServerEntry};
    let grid = Arc::new(parking_lot::RwLock::new(GridState::default()));
    {
        let mut g = grid.write();
        g.register(ServerEntry {
            server_id: "moe-a".into(),
            listen_url: "http://moe-a".into(),
            model_id: "m".into(),
            layer_start: 0,
            layer_end: 0,
            vindex_hash: "h".into(),
            cpu_pct: 0.0,
            ram_used: 0,
            requests_in_flight: 0,
            last_seen: std::time::Instant::now(),
            layer_latencies: std::collections::HashMap::new(),
            req_per_sec: 0.0,
            rtt_ms: None,
            expert_start: 0,
            expert_end: 3,
        });
    }
    let shards = parse_shards("99-100=http://unused:1").unwrap();
    let client = reqwest::Client::builder()
        .timeout(Duration::from_secs(1))
        .build()
        .unwrap();
    let state = Arc::new(AppState {
        static_shards: shards,
        grid: Some(grid),
        client,
        metrics: None,
        #[cfg(feature = "http3")]
        h3_client: None,
        hedge_after: None,
    });
    let app = build_router(state);

    let resp = app
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/v1/walk-ffn")
                .header(header::CONTENT_TYPE, "application/json")
                .body(Body::from(r#"{"layer":0,"experts":[99],"model_id":"m"}"#))
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(resp.status(), StatusCode::SERVICE_UNAVAILABLE);
    let body = axum::body::to_bytes(resp.into_body(), 1024).await.unwrap();
    let v: Value = serde_json::from_slice(&body).unwrap();
    assert!(v["error"]
        .as_str()
        .unwrap()
        .contains("(layer 0, expert 99)"));
}

/// MoE dispatch with two expert shards and an `experts` request —
/// fans out to both shards, merges the responses.
#[tokio::test]
async fn moe_request_fans_out_to_owning_shards_and_merges() {
    use larql_router::grid::{GridState, ServerEntry};

    let (addr_lo, _calls_lo) = spawn_fake_shard().await;
    let (addr_hi, _calls_hi) = spawn_fake_shard().await;

    let grid = Arc::new(parking_lot::RwLock::new(GridState::default()));
    {
        let mut g = grid.write();
        g.register(ServerEntry {
            server_id: "moe-lo".into(),
            listen_url: format!("http://{addr_lo}"),
            model_id: "m".into(),
            layer_start: 0,
            layer_end: 0,
            vindex_hash: "h".into(),
            cpu_pct: 0.0,
            ram_used: 0,
            requests_in_flight: 0,
            last_seen: std::time::Instant::now(),
            layer_latencies: std::collections::HashMap::new(),
            req_per_sec: 0.0,
            rtt_ms: None,
            expert_start: 0,
            expert_end: 3,
        });
        g.register(ServerEntry {
            server_id: "moe-hi".into(),
            listen_url: format!("http://{addr_hi}"),
            model_id: "m".into(),
            layer_start: 0,
            layer_end: 0,
            vindex_hash: "h".into(),
            cpu_pct: 0.0,
            ram_used: 0,
            requests_in_flight: 0,
            last_seen: std::time::Instant::now(),
            layer_latencies: std::collections::HashMap::new(),
            req_per_sec: 0.0,
            rtt_ms: None,
            expert_start: 4,
            expert_end: 7,
        });
    }
    let shards = parse_shards("99-100=http://unused:1").unwrap();
    let client = reqwest::Client::builder()
        .timeout(Duration::from_secs(5))
        .build()
        .unwrap();
    let state = Arc::new(AppState {
        static_shards: shards,
        grid: Some(grid),
        client,
        metrics: None,
        #[cfg(feature = "http3")]
        h3_client: None,
        hedge_after: None,
    });
    let app = build_router(state);

    let resp = app
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/v1/walk-ffn")
                .header(header::CONTENT_TYPE, "application/json")
                .body(Body::from(
                    r#"{"layer":0,"experts":[0,3,5,7],"model_id":"m"}"#,
                ))
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(
        resp.status(),
        StatusCode::OK,
        "MoE fan-out should merge two shard responses"
    );
    let body = axum::body::to_bytes(resp.into_body(), 64 * 1024)
        .await
        .unwrap();
    let v: Value = serde_json::from_slice(&body).unwrap();
    assert!(v["results"].is_array(), "merged envelope must have results");
}

/// A walk-ffn call that 502s should increment the `error_5xx` counter
/// on the registry. Proves the instrumentation hook in the handler
/// fires on the error path.
#[tokio::test]
async fn walk_ffn_5xx_increments_error_counter() {
    use larql_router::metrics::{encode_metrics_text, RouterMetrics};

    let shards = parse_shards("0-3=http://127.0.0.1:1").unwrap();
    let client = reqwest::Client::builder()
        .timeout(Duration::from_secs(1))
        .build()
        .unwrap();
    let metrics = RouterMetrics::new();
    let state = Arc::new(AppState {
        static_shards: shards,
        grid: None,
        client,
        metrics: Some(metrics.clone()),
        #[cfg(feature = "http3")]
        h3_client: None,
        hedge_after: None,
    });
    let app = build_router(state);

    let resp = app
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/v1/walk-ffn")
                .header(header::CONTENT_TYPE, "application/json")
                .body(Body::from(r#"{"layer":0}"#))
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(resp.status(), StatusCode::BAD_GATEWAY);

    let text = encode_metrics_text(&metrics).unwrap();
    assert!(
        text.contains("larql_router_walk_ffn_requests_total{status=\"error_5xx\"} 1"),
        "expected error_5xx=1, got:\n{text}"
    );
}

// ── ADR-0019: HTTP/3 end-to-end smoke ────────────────────────────────────────
//
// Phase 4c: prove the full router→server h3 wire works.
// 1. Spin up an h3 axum listener that records the MoE sub-request body.
// 2. Configure AppState with the matching H3Client (no fingerprint pin —
//    LAN/dev mode).
// 3. Issue a MoE `experts` request to the router and assert the h3 server
//    received the rewritten `layer_experts` payload.

#[cfg(feature = "http3")]
#[tokio::test]
async fn moe_fanout_dispatches_through_h3_client_when_configured() {
    use larql_router::grid::{GridState, ServerEntry};
    use larql_router_protocol::transport::h3::{serve_axum, server_endpoint, H3Client};
    use larql_router_protocol::transport::quic::self_signed_tls;
    use tokio::sync::Mutex;

    let _ = rustls::crypto::ring::default_provider().install_default();

    // ── Stand up a single h3 echo server that records every body it sees.
    let recorded: Arc<Mutex<Vec<Value>>> = Arc::new(Mutex::new(Vec::new()));
    let recorded_handler = recorded.clone();
    let h3_app = axum::Router::new().route(
        "/v1/walk-ffn",
        axum::routing::post(move |body: axum::extract::Json<Value>| {
            let recorded = recorded_handler.clone();
            async move {
                recorded.lock().await.push(body.0.clone());
                axum::Json(json!({
                    "results": [{"layer": 0, "expert": 0, "out": "ok"}],
                    "latency_ms": 1.0
                }))
            }
        }),
    );
    let tls = self_signed_tls("h3-shard").expect("self_signed_tls");
    let endpoint = server_endpoint("127.0.0.1:0".parse().unwrap(), &tls).expect("server_endpoint");
    let h3_addr = endpoint.local_addr().expect("local_addr");
    tokio::spawn(async move {
        let _ = serve_axum(endpoint, h3_app).await;
    });
    tokio::time::sleep(Duration::from_millis(50)).await;

    // ── Grid: one MoE shard owning experts 0-7 of layer 0, listen URL
    //    pointing at the h3 listener. The router's dispatch path will
    //    parse `host:port` out of this and call H3Client::post_json.
    let grid = Arc::new(parking_lot::RwLock::new(GridState::default()));
    {
        let mut g = grid.write();
        g.register(ServerEntry {
            server_id: "moe-h3".into(),
            listen_url: format!("http://127.0.0.1:{}", h3_addr.port()),
            model_id: "m".into(),
            layer_start: 0,
            layer_end: 0,
            vindex_hash: "h".into(),
            cpu_pct: 0.0,
            ram_used: 0,
            requests_in_flight: 0,
            last_seen: std::time::Instant::now(),
            layer_latencies: std::collections::HashMap::new(),
            req_per_sec: 0.0,
            rtt_ms: None,
            expert_start: 0,
            expert_end: 7,
        });
    }

    // ── Router AppState with the matching H3Client.
    let shards = parse_shards("99-100=http://unused:1").unwrap();
    let client = reqwest::Client::builder()
        .timeout(Duration::from_secs(2))
        .build()
        .unwrap();
    let h3_client =
        Arc::new(H3Client::new("127.0.0.1:0".parse().unwrap(), None).expect("h3 client"));
    let state = Arc::new(AppState {
        static_shards: shards,
        grid: Some(grid),
        client,
        metrics: None,
        h3_client: Some(h3_client),
        hedge_after: None,
    });
    let app = build_router(state);

    // ── Issue a MoE request — four picked experts all on the same shard.
    let resp = app
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/v1/walk-ffn")
                .header(header::CONTENT_TYPE, "application/json")
                .body(Body::from(
                    r#"{"layer":0,"experts":[0,3,5,7],"model_id":"m"}"#,
                ))
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(
        resp.status(),
        StatusCode::OK,
        "router→shard h3 dispatch must return OK"
    );

    let recv = recorded.lock().await;
    assert_eq!(recv.len(), 1, "h3 shard saw exactly one sub-request");
    let layer_experts = recv[0]["layer_experts"].as_array().expect("layer_experts");
    assert_eq!(layer_experts.len(), 1);
    // The router groups all four picked experts onto the single shard
    // that owns 0-7, so the sub-request payload lists all of them.
    let experts = layer_experts[0]["experts"].as_array().expect("experts");
    let ids: Vec<u64> = experts.iter().filter_map(|v| v.as_u64()).collect();
    assert_eq!(ids, vec![0, 3, 5, 7]);
}

/// ADR-0020 — when every replica that owns the requested layer is at
/// or above the configured saturation ceiling, the router must emit
/// `503 Service Unavailable` (not `400 Bad Request`), set the
/// `Retry-After` hint, and bump the `route_saturation_total` counter.
#[tokio::test]
async fn walk_ffn_returns_503_with_retry_after_when_replicas_saturated() {
    use larql_router::grid::{GridState, ServerEntry};
    use larql_router::metrics::{encode_metrics_text, RouterMetrics};
    use parking_lot::RwLock;
    use std::collections::HashMap;

    let grid = Arc::new(RwLock::new(GridState::default()));
    // One owner, requests_in_flight already at the ceiling.
    grid.write().register(ServerEntry {
        server_id: "saturated".into(),
        listen_url: "http://unreachable:9".into(),
        model_id: "m".into(),
        layer_start: 0,
        layer_end: 9,
        vindex_hash: "h".into(),
        cpu_pct: 0.0,
        ram_used: 0,
        requests_in_flight: 8,
        last_seen: std::time::Instant::now(),
        layer_latencies: HashMap::new(),
        req_per_sec: 0.0,
        rtt_ms: None,
        expert_start: 0,
        expert_end: 0,
    });
    grid.write().set_saturation_ceiling(Some(8));

    let metrics = RouterMetrics::new();
    let client = reqwest::Client::builder()
        .timeout(Duration::from_secs(5))
        .build()
        .unwrap();
    let state = Arc::new(AppState {
        static_shards: parse_shards("99-100=http://unused:1").unwrap(),
        grid: Some(grid),
        client,
        metrics: Some(metrics.clone()),
        #[cfg(feature = "http3")]
        h3_client: None,
        hedge_after: None,
    });
    let app = build_router(state);

    let resp = app
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/v1/walk-ffn")
                .header(header::CONTENT_TYPE, "application/json")
                .body(Body::from(r#"{"model_id":"m","layer":3}"#))
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(resp.status(), StatusCode::SERVICE_UNAVAILABLE);
    assert_eq!(
        resp.headers()
            .get(header::RETRY_AFTER)
            .map(|v| v.to_str().unwrap()),
        Some("0.5"),
        "Retry-After hint must be set on saturation 503s"
    );
    let body = axum::body::to_bytes(resp.into_body(), 1024).await.unwrap();
    let v: Value = serde_json::from_slice(&body).unwrap();
    assert!(
        v["error"].as_str().unwrap().contains("saturation ceiling"),
        "error body should explain saturation; got {}",
        v["error"]
    );

    let text = encode_metrics_text(&metrics).unwrap();
    assert!(
        text.lines()
            .any(|l| l.starts_with("larql_router_route_saturation_total ") && l.ends_with(" 1")),
        "route_saturation_total must increment exactly once; got:\n{text}"
    );
}

// ── ADR-0021 hedged dispatch ───────────────────────────────────────────────

/// Like `spawn_fake_shard`, but the `/v1/walk-ffn` handler waits
/// `delay` before responding. Used to construct a deliberately-slow
/// primary that the hedge should beat.
async fn spawn_slow_shard(delay: Duration) -> (SocketAddr, ShardCalls) {
    let calls = ShardCalls::default();
    let app_calls = calls.clone();
    let app = axum::Router::new()
        .route(
            "/v1/walk-ffn",
            post(
                move |st: State<ShardCalls>, body: axum::extract::Json<Value>| async move {
                    tokio::time::sleep(delay).await;
                    fake_walk_ffn(st, body).await
                },
            ),
        )
        .with_state(app_calls);
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap();
    tokio::spawn(async move {
        axum::serve(listener, app).await.unwrap();
    });
    tokio::time::sleep(Duration::from_millis(50)).await;
    (addr, calls)
}

/// Build a 2-replica grid where layer 0 has a slow primary + fast
/// secondary, layer 1 has a single fast owner. Multi-layer requests
/// force the fan-out path; the slow primary triggers the hedge.
async fn build_hedge_topology(
    slow_delay: Duration,
) -> (
    SocketAddr,
    ShardCalls,
    SocketAddr,
    ShardCalls,
    SocketAddr,
    ShardCalls,
    Arc<parking_lot::RwLock<larql_router::grid::GridState>>,
) {
    use larql_router::grid::{GridState, ServerEntry};
    use std::collections::HashMap;

    let (slow_addr, slow_calls) = spawn_slow_shard(slow_delay).await;
    let (fast_addr, fast_calls) = spawn_fake_shard().await;
    let (b_addr, b_calls) = spawn_fake_shard().await;

    let grid = Arc::new(parking_lot::RwLock::new(GridState::default()));
    // Layer 0 owned by slow_a (primary, lower in_flight = priority) +
    // fast_a (secondary). Heartbeats set in_flight so the comparator
    // picks slow_a first deterministically.
    grid.write().register(ServerEntry {
        server_id: "slow-a".into(),
        listen_url: format!("http://{slow_addr}"),
        model_id: "m".into(),
        layer_start: 0,
        layer_end: 0,
        vindex_hash: "h".into(),
        cpu_pct: 0.0,
        ram_used: 0,
        requests_in_flight: 0,
        last_seen: std::time::Instant::now(),
        layer_latencies: HashMap::new(),
        req_per_sec: 0.0,
        rtt_ms: None,
        expert_start: 0,
        expert_end: 0,
    });
    grid.write().register(ServerEntry {
        server_id: "fast-a".into(),
        listen_url: format!("http://{fast_addr}"),
        model_id: "m".into(),
        layer_start: 0,
        layer_end: 0,
        vindex_hash: "h".into(),
        cpu_pct: 0.0,
        ram_used: 0,
        requests_in_flight: 5,
        last_seen: std::time::Instant::now(),
        layer_latencies: HashMap::new(),
        req_per_sec: 0.0,
        rtt_ms: None,
        expert_start: 0,
        expert_end: 0,
    });
    // Layer 1 owned by a single fast shard.
    grid.write().register(ServerEntry {
        server_id: "b".into(),
        listen_url: format!("http://{b_addr}"),
        model_id: "m".into(),
        layer_start: 1,
        layer_end: 1,
        vindex_hash: "h".into(),
        cpu_pct: 0.0,
        ram_used: 0,
        requests_in_flight: 0,
        last_seen: std::time::Instant::now(),
        layer_latencies: HashMap::new(),
        req_per_sec: 0.0,
        rtt_ms: None,
        expert_start: 0,
        expert_end: 0,
    });
    (
        slow_addr, slow_calls, fast_addr, fast_calls, b_addr, b_calls, grid,
    )
}

#[tokio::test]
async fn walk_ffn_hedge_fires_when_primary_is_slow() {
    use larql_router::metrics::{encode_metrics_text, RouterMetrics};

    // Primary delays 500ms; hedge fires after 30ms; secondary responds
    // ~immediately. The hedge must fire (counter +1) AND win (counter +1).
    let (_slow_addr, _slow_calls, _fast_addr, fast_calls, _b_addr, _b_calls, grid) =
        build_hedge_topology(Duration::from_millis(500)).await;

    let metrics = RouterMetrics::new();
    let client = reqwest::Client::builder()
        .timeout(Duration::from_secs(5))
        .build()
        .unwrap();
    let state = Arc::new(AppState {
        static_shards: parse_shards("99-100=http://unused:1").unwrap(),
        grid: Some(grid),
        client,
        metrics: Some(metrics.clone()),
        #[cfg(feature = "http3")]
        h3_client: None,
        hedge_after: Some(Duration::from_millis(30)),
    });
    let app = build_router(state);

    let resp = app
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/v1/walk-ffn")
                .header(header::CONTENT_TYPE, "application/json")
                .body(Body::from(r#"{"model_id":"m","layers":[0,1]}"#))
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(resp.status(), StatusCode::OK);

    let text = encode_metrics_text(&metrics).unwrap();
    let line = |needle: &str| {
        text.lines()
            .find(|l| l.starts_with(needle))
            .map(String::from)
    };
    let fires = line("larql_router_route_hedge_fires_total ").unwrap();
    let wins = line("larql_router_route_hedge_wins_total ").unwrap();
    assert!(
        fires.ends_with(" 1"),
        "hedge_fires_total expected 1, got line: {fires}"
    );
    assert!(
        wins.ends_with(" 1"),
        "hedge_wins_total expected 1, got line: {wins}"
    );

    // The fast secondary received the layer-0 sub-request.
    let fast_seen = fast_calls.inner.lock().await.clone();
    assert!(
        !fast_seen.is_empty(),
        "secondary (fast_a) must have served the hedged sub-request"
    );
}

#[tokio::test]
async fn walk_ffn_hedge_does_not_fire_on_fast_primary() {
    use larql_router::metrics::{encode_metrics_text, RouterMetrics};

    // Primary delay 0 — no hedge should fire. hedge_after stays set,
    // so any spurious firing would surface as a non-zero counter.
    let (_slow_addr, _slow_calls, _fast_addr, fast_calls, _b_addr, _b_calls, grid) =
        build_hedge_topology(Duration::from_millis(0)).await;

    let metrics = RouterMetrics::new();
    let client = reqwest::Client::builder()
        .timeout(Duration::from_secs(5))
        .build()
        .unwrap();
    let state = Arc::new(AppState {
        static_shards: parse_shards("99-100=http://unused:1").unwrap(),
        grid: Some(grid),
        client,
        metrics: Some(metrics.clone()),
        #[cfg(feature = "http3")]
        h3_client: None,
        hedge_after: Some(Duration::from_millis(500)),
    });
    let app = build_router(state);

    let resp = app
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/v1/walk-ffn")
                .header(header::CONTENT_TYPE, "application/json")
                .body(Body::from(r#"{"model_id":"m","layers":[0,1]}"#))
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(resp.status(), StatusCode::OK);

    let text = encode_metrics_text(&metrics).unwrap();
    let line = |needle: &str| {
        text.lines()
            .find(|l| l.starts_with(needle))
            .map(String::from)
    };
    let fires = line("larql_router_route_hedge_fires_total ").unwrap();
    let wins = line("larql_router_route_hedge_wins_total ").unwrap();
    assert!(
        fires.ends_with(" 0"),
        "hedge_fires_total expected 0 when primary is fast, got: {fires}"
    );
    assert!(
        wins.ends_with(" 0"),
        "hedge_wins_total expected 0, got: {wins}"
    );
    // Secondary should NOT have received anything.
    let fast_seen = fast_calls.inner.lock().await.clone();
    assert!(
        fast_seen.is_empty(),
        "secondary should not be touched when primary wins, but saw {fast_seen:?}"
    );
}

#[tokio::test]
async fn walk_ffn_no_hedge_when_only_one_replica() {
    use larql_router::grid::{GridState, ServerEntry};
    use larql_router::metrics::{encode_metrics_text, RouterMetrics};
    use std::collections::HashMap;

    // Topology has no secondary for any layer; even with hedging
    // configured, the hedge path is skipped because route_with_rank
    // returns only one URL.
    let (a_addr, _a_calls) = spawn_fake_shard().await;
    let (b_addr, _b_calls) = spawn_fake_shard().await;
    let grid = Arc::new(parking_lot::RwLock::new(GridState::default()));
    for (id, addr, ls, le) in [("a", a_addr, 0u32, 0u32), ("b", b_addr, 1u32, 1u32)] {
        grid.write().register(ServerEntry {
            server_id: id.into(),
            listen_url: format!("http://{addr}"),
            model_id: "m".into(),
            layer_start: ls,
            layer_end: le,
            vindex_hash: "h".into(),
            cpu_pct: 0.0,
            ram_used: 0,
            requests_in_flight: 0,
            last_seen: std::time::Instant::now(),
            layer_latencies: HashMap::new(),
            req_per_sec: 0.0,
            rtt_ms: None,
            expert_start: 0,
            expert_end: 0,
        });
    }

    let metrics = RouterMetrics::new();
    let client = reqwest::Client::builder()
        .timeout(Duration::from_secs(5))
        .build()
        .unwrap();
    let state = Arc::new(AppState {
        static_shards: parse_shards("99-100=http://unused:1").unwrap(),
        grid: Some(grid),
        client,
        metrics: Some(metrics.clone()),
        #[cfg(feature = "http3")]
        h3_client: None,
        hedge_after: Some(Duration::from_millis(20)),
    });
    let app = build_router(state);

    let resp = app
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/v1/walk-ffn")
                .header(header::CONTENT_TYPE, "application/json")
                .body(Body::from(r#"{"model_id":"m","layers":[0,1]}"#))
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(resp.status(), StatusCode::OK);

    let text = encode_metrics_text(&metrics).unwrap();
    let fires_line = text
        .lines()
        .find(|l| l.starts_with("larql_router_route_hedge_fires_total "))
        .unwrap();
    assert!(
        fires_line.ends_with(" 0"),
        "no secondary exists — hedge must never fire; got: {fires_line}"
    );
}
