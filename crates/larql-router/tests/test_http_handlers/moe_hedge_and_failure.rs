//! ADR-0018 MoE dispatch under ADR-0021 hedging, and a failing expert
//! shard: the hedge counters move on the MoE path, and a sub-request
//! failure surfaces as 502 rather than a partial merge.

use super::*;
use larql_router::grid::{GridState, ServerEntry};
use larql_router::metrics::RouterMetrics;

/// Primary delay long enough that the hedge always fires first.
const SLOW_PRIMARY: Duration = Duration::from_millis(400);
/// Hedge threshold, well under `SLOW_PRIMARY`.
const HEDGE_AFTER: Duration = Duration::from_millis(30);
/// In-flight count that ranks a replica behind an idle one.
const BUSY: u32 = 5;
const EXPERT_REQUEST: &str = r#"{"layer":0,"experts":[1,2],"model_id":"m"}"#;

fn moe_entry(id: &str, url: String, requests_in_flight: u32) -> ServerEntry {
    ServerEntry {
        server_id: id.into(),
        listen_url: url,
        model_id: "m".into(),
        layer_start: 0,
        layer_end: 0,
        vindex_hash: "h".into(),
        shard_sha256: String::new(),
        cpu_pct: 0.0,
        ram_used: 0,
        requests_in_flight,
        last_seen: std::time::Instant::now(),
        layer_latencies: std::collections::HashMap::new(),
        req_per_sec: 0.0,
        rtt_ms: None,
        expert_start: 0,
        expert_end: 3,
        serves_openai: false,
    }
}

fn moe_router(
    grid: GridState,
    hedge_after: Option<Duration>,
    metrics: &Arc<RouterMetrics>,
) -> axum::Router {
    let client = reqwest::Client::builder()
        .timeout(Duration::from_secs(5))
        .build()
        .unwrap();
    build_router(Arc::new(AppState {
        static_shards: parse_shards("99-100=http://unused:1").unwrap(),
        grid: Some(Arc::new(parking_lot::RwLock::new(grid))),
        client,
        metrics: Some(metrics.clone()),
        #[cfg(feature = "http3")]
        h3_client: None,
        hedge_after,
        openai_responses: Default::default(),
    }))
}

async fn post_experts(app: axum::Router) -> axum::http::Response<Body> {
    app.oneshot(
        Request::builder()
            .method("POST")
            .uri("/v1/walk-ffn")
            .header(header::CONTENT_TYPE, "application/json")
            .body(Body::from(EXPERT_REQUEST))
            .unwrap(),
    )
    .await
    .unwrap()
}

/// Keep the primary pending and make the secondary fail while parsing an
/// actual TCP response. Neither a connection-refused delay nor a fixed
/// primary-response window determines the winner. Dropping the JoinSet
/// cleans up both listeners, including on an assertion failure.
async fn spawn_pending_primary_and_dead_secondary(
) -> (SocketAddr, SocketAddr, tokio::task::JoinSet<()>) {
    use tokio::io::{AsyncReadExt, AsyncWriteExt};

    let primary = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let secondary = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let primary_addr = primary.local_addr().unwrap();
    let secondary_addr = secondary.local_addr().unwrap();
    let (started_tx, started_rx) = tokio::sync::oneshot::channel();
    let started_tx = Arc::new(Mutex::new(Some(started_tx)));
    let app = axum::Router::new().route(
        "/v1/walk-ffn",
        post(move || {
            let started_tx = started_tx.clone();
            async move {
                if let Some(tx) = started_tx.lock().await.take() {
                    let _ = tx.send(());
                }
                std::future::pending::<Json<Value>>().await
            }
        }),
    );
    let mut servers = tokio::task::JoinSet::new();
    servers.spawn(async move { axum::serve(primary, app).await.unwrap() });
    servers.spawn(async move {
        let (mut stream, _) = secondary.accept().await.unwrap();
        let mut request = [0; 4096];
        assert!(stream.read(&mut request).await.unwrap() > 0);
        started_rx.await.expect("the primary received its request");
        // Force an HTTP transport/parser error, rather than an application
        // status whose JSON body might be accepted as a successful result.
        stream
            .write_all(b"invalid HTTP response\r\n\r\n")
            .await
            .unwrap();
        stream.shutdown().await.unwrap();
    });
    (primary_addr, secondary_addr, servers)
}

#[tokio::test]
async fn moe_hedge_to_the_secondary_replica_wins_and_is_counted() {
    let (slow, _slow_calls) = spawn_slow_shard(SLOW_PRIMARY).await;
    let (fast, fast_calls) = spawn_fake_shard().await;
    let mut grid = GridState::default();
    grid.register(moe_entry("slow", format!("http://{slow}"), 0));
    grid.register(moe_entry("fast", format!("http://{fast}"), BUSY));
    let metrics = RouterMetrics::new();

    let resp = post_experts(moe_router(grid, Some(HEDGE_AFTER), &metrics)).await;
    assert_eq!(resp.status(), StatusCode::OK);
    assert_eq!(metrics.route_hedge_fires_total.get(), 1);
    assert_eq!(metrics.route_hedge_wins_total.get(), 1);

    let calls = fast_calls.inner.lock().await;
    assert_eq!(calls.len(), 1, "the hedge reached the secondary");
    assert_eq!(
        calls[0]["layer_experts"],
        json!([{"layer": 0, "experts": [1, 2]}]),
        "the secondary gets the same per-shard sub-request"
    );
}

#[tokio::test]
async fn moe_hedge_without_a_second_replica_stays_on_the_primary() {
    let (only, only_calls) = spawn_fake_shard().await;
    let mut grid = GridState::default();
    grid.register(moe_entry("only", format!("http://{only}"), 0));
    let metrics = RouterMetrics::new();

    let resp = post_experts(moe_router(grid, Some(HEDGE_AFTER), &metrics)).await;
    assert_eq!(resp.status(), StatusCode::OK);
    assert_eq!(metrics.route_hedge_fires_total.get(), 0);
    assert_eq!(only_calls.inner.lock().await.len(), 1);
}

/// The hedge fires against a slow primary and the dead secondary answers
/// first — with an error. The failure is still a 502, and the hedge is
/// still counted as fired and won (the secondary finished first).
#[tokio::test]
async fn moe_hedge_that_lands_on_a_dead_secondary_is_a_502_and_counted() {
    let (slow, dead, _servers) = spawn_pending_primary_and_dead_secondary().await;
    let mut grid = GridState::default();
    grid.register(moe_entry("slow", format!("http://{slow}"), 0));
    grid.register(moe_entry("dead", format!("http://{dead}"), BUSY));
    let metrics = RouterMetrics::new();

    let resp = tokio::time::timeout(
        Duration::from_secs(10),
        post_experts(moe_router(grid, Some(HEDGE_AFTER), &metrics)),
    )
    .await
    .expect("the failing hedge must complete");
    assert_eq!(resp.status(), StatusCode::BAD_GATEWAY);
    assert_eq!(metrics.route_hedge_fires_total.get(), 1);
    assert_eq!(metrics.route_hedge_wins_total.get(), 1);
}

#[tokio::test]
async fn moe_sub_request_failure_is_a_502() {
    let mut grid = GridState::default();
    // Nothing listens on port 1: the sub-request fails to connect.
    grid.register(moe_entry("dead", "http://127.0.0.1:1".into(), 0));
    let metrics = RouterMetrics::new();

    let resp = post_experts(moe_router(grid, None, &metrics)).await;
    assert_eq!(resp.status(), StatusCode::BAD_GATEWAY);
    let body = axum::body::to_bytes(resp.into_body(), 4096).await.unwrap();
    let v: Value = serde_json::from_slice(&body).unwrap();
    assert!(
        v["error"]
            .as_str()
            .unwrap()
            .contains("MoE sub-request to shard failed"),
        "got: {v}"
    );
    assert_eq!(metrics.route_hedge_fires_total.get(), 0);
}
