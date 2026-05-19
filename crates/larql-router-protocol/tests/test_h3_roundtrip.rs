//! ADR-0019 — end-to-end round-trip test for the HTTP/3 transport.
//!
//! Spins up an axum::Router with one route, serves it over h3 via
//! `transport::h3::serve_axum`, then issues a POST through
//! `H3Client::post_json` and asserts the round-trip works
//! end-to-end.
//!
//! Feature-gated under `http3` — same as the transport module.

#![cfg(feature = "http3")]

use std::net::SocketAddr;
use std::sync::Arc;
use std::time::Duration;

use axum::extract::State;
use axum::routing::post;
use axum::Json;
use serde_json::{json, Value};
use tokio::sync::Mutex;

use larql_router_protocol::transport::h3::{serve_axum, server_endpoint, H3Client};
use larql_router_protocol::transport::quic::self_signed_tls;

/// Server-side state: records every request body the handler sees.
/// Shared between the spawned server task and the test assertions so
/// we can verify the request actually went through.
#[derive(Clone, Default)]
struct Calls(Arc<Mutex<Vec<Value>>>);

async fn echo_handler(State(calls): State<Calls>, Json(body): Json<Value>) -> Json<Value> {
    calls.0.lock().await.push(body.clone());
    // Echo the body back plus a server-side tag — exercises the
    // response-body path.
    Json(json!({
        "received": body,
        "ack": "h3",
    }))
}

/// Spawn an h3 server with `/echo` and return its socket address
/// plus the shared call recorder.
async fn spawn_h3_echo_server() -> (SocketAddr, Calls) {
    let _ = rustls::crypto::ring::default_provider().install_default();

    let calls = Calls::default();
    let app = axum::Router::new()
        .route("/echo", post(echo_handler))
        .with_state(calls.clone());

    let tls = self_signed_tls("h3-test").expect("self_signed_tls");
    let endpoint = server_endpoint("127.0.0.1:0".parse().unwrap(), &tls).expect("server_endpoint");
    let addr = endpoint.local_addr().expect("local_addr");

    tokio::spawn(async move {
        if let Err(e) = serve_axum(endpoint, app).await {
            eprintln!("h3 server: {e}");
        }
    });

    // Give the listener a moment to start accepting.
    tokio::time::sleep(Duration::from_millis(50)).await;

    (addr, calls)
}

/// End-to-end: POST a JSON body, check status, check echoed body,
/// check the server actually received the request.
#[tokio::test]
async fn h3_post_json_round_trips() {
    let (server_addr, calls) = spawn_h3_echo_server().await;

    let client = H3Client::new("127.0.0.1:0".parse().unwrap(), None).expect("client setup");

    let body = bytes::Bytes::from_static(br#"{"hello":"world"}"#);
    let resp = client
        .post_json(server_addr, "h3-test", "/echo", body)
        .await
        .expect("post_json");

    assert_eq!(resp.status, 200, "expected 200 from /echo");

    let parsed: Value = serde_json::from_slice(&resp.body).expect("response body must be JSON");
    assert_eq!(parsed["received"]["hello"], "world");
    assert_eq!(parsed["ack"], "h3");

    let recorded = calls.0.lock().await;
    assert_eq!(recorded.len(), 1, "server saw exactly one request");
    assert_eq!(recorded[0]["hello"], "world");
}

/// Two concurrent requests over the same client. Each call opens its
/// own connection (the MVP client doesn't pool), so this also
/// confirms multiple connect attempts against the same server
/// succeed.
#[tokio::test]
async fn h3_two_concurrent_requests_round_trip() {
    let (server_addr, calls) = spawn_h3_echo_server().await;

    let client = H3Client::new("127.0.0.1:0".parse().unwrap(), None).expect("client setup");

    let body_a = bytes::Bytes::from_static(br#"{"id":"a"}"#);
    let body_b = bytes::Bytes::from_static(br#"{"id":"b"}"#);

    let (ra, rb) = tokio::join!(
        client.post_json(server_addr, "h3-test", "/echo", body_a),
        client.post_json(server_addr, "h3-test", "/echo", body_b),
    );
    let ra = ra.expect("request a");
    let rb = rb.expect("request b");
    assert_eq!(ra.status, 200);
    assert_eq!(rb.status, 200);

    let recorded = calls.0.lock().await;
    assert_eq!(recorded.len(), 2, "server saw both requests");
    let ids: Vec<&str> = recorded
        .iter()
        .map(|v| v["id"].as_str().unwrap_or(""))
        .collect();
    assert!(ids.contains(&"a"));
    assert!(ids.contains(&"b"));
}

/// A client that opens a QUIC connection then drops it without
/// issuing any HTTP/3 request exercises the server's graceful-
/// close branch in `handle_connection` (the `accept().await` →
/// `Ok(None)` path). No assertion on observable side effects — the
/// test passes if the server task doesn't leak / panic after the
/// client disconnects.
#[tokio::test]
async fn h3_client_disconnect_without_request_is_graceful() {
    use larql_router_protocol::transport::h3::client_endpoint;
    let (server_addr, _) = spawn_h3_echo_server().await;

    let endpoint = client_endpoint("127.0.0.1:0".parse().unwrap(), None).unwrap();
    let connecting = endpoint
        .connect(server_addr, "h3-test")
        .expect("endpoint.connect");
    let conn = connecting.await.expect("connect");
    // Drop the connection immediately. The server should see the
    // QUIC connection terminate cleanly; its h3::server accept()
    // loop returns Ok(None) or a graceful Err — either way the
    // handle_connection helper exits cleanly without a panic.
    drop(conn);
    drop(endpoint);

    // Give the server's spawned handle_connection a chance to
    // observe the close. No assertion — we're just verifying nothing
    // panics or hangs.
    tokio::time::sleep(Duration::from_millis(100)).await;
}

/// A POST to a path that doesn't exist returns 404 — exercises the
/// axum-router routing through the h3 adapter, not just the happy
/// path.
#[tokio::test]
async fn h3_unknown_route_returns_404() {
    let (server_addr, _) = spawn_h3_echo_server().await;

    let client = H3Client::new("127.0.0.1:0".parse().unwrap(), None).expect("client setup");

    let resp = client
        .post_json(
            server_addr,
            "h3-test",
            "/does-not-exist",
            bytes::Bytes::from_static(b"{}"),
        )
        .await
        .expect("post_json");

    assert_eq!(resp.status, 404, "unknown path must surface a 404");
}
