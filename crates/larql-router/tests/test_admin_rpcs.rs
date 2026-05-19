//! Integration tests for the ADR-0004 Phase 5 admin RPCs:
//! `DrainServer` and `AssignRange`. Stands up a real `GridServiceImpl`
//! over loopback tonic, registers fixtures via the gRPC `Join` stream,
//! then drives the admin RPCs and verifies the side-effects on the live
//! grid state.

use std::sync::Arc;
use std::time::Duration;

use parking_lot::RwLock;
use tokio::sync::mpsc;
use tokio_stream::wrappers::ReceiverStream;
use tokio_stream::StreamExt;

use larql_router::grid::service::GridServiceImpl;
use larql_router::grid::GridState;
use larql_router_protocol::{
    grid_service_server::GridServiceServer, AnnounceMsg, AssignRangeRequest, AvailableMsg,
    DrainRequest, GridServiceClient, RouterMessage, RouterPayload, ServerMessage, ServerPayload,
};
use tonic::transport::Server;

async fn spawn_router() -> (std::net::SocketAddr, Arc<RwLock<GridState>>) {
    let metrics = larql_router::metrics::RouterMetrics::new();
    let state = Arc::new(RwLock::new(GridState::default()));
    let svc = GridServiceImpl::new(state.clone()).with_metrics(metrics);

    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap();
    let stream = tokio_stream::wrappers::TcpListenerStream::new(listener);
    tokio::spawn(async move {
        Server::builder()
            .add_service(GridServiceServer::new(svc))
            .serve_with_incoming(stream)
            .await
            .unwrap();
    });
    tokio::time::sleep(Duration::from_millis(100)).await;
    (addr, state)
}

async fn join_and_announce(
    addr: std::net::SocketAddr,
    listen_url: &str,
    layers: (u32, u32),
    hash: &str,
) -> (mpsc::Sender<ServerMessage>, tonic::Streaming<RouterMessage>) {
    let mut client = GridServiceClient::connect(format!("http://{addr}"))
        .await
        .unwrap();
    let (tx, rx) = mpsc::channel::<ServerMessage>(32);
    let inbound = client
        .join(ReceiverStream::new(rx))
        .await
        .unwrap()
        .into_inner();
    tx.send(ServerMessage {
        payload: Some(ServerPayload::Announce(AnnounceMsg {
            model_id: "m".into(),
            layer_start: layers.0,
            layer_end: layers.1,
            ram_bytes: 1024 * 1024 * 1024,
            listen_url: listen_url.into(),
            vindex_hash: hash.into(),
            expert_start: 0,
            expert_end: 0,
        })),
    })
    .await
    .unwrap();
    (tx, inbound)
}

async fn join_and_available(
    addr: std::net::SocketAddr,
) -> (mpsc::Sender<ServerMessage>, tonic::Streaming<RouterMessage>) {
    let mut client = GridServiceClient::connect(format!("http://{addr}"))
        .await
        .unwrap();
    let (tx, rx) = mpsc::channel::<ServerMessage>(32);
    let inbound = client
        .join(ReceiverStream::new(rx))
        .await
        .unwrap()
        .into_inner();
    tx.send(ServerMessage {
        payload: Some(ServerPayload::Available(AvailableMsg {
            ram_bytes: 8 * 1024 * 1024 * 1024,
            disk_bytes: 0,
            store_path: "/tmp".into(),
        })),
    })
    .await
    .unwrap();
    (tx, inbound)
}

// ── DrainServer ──────────────────────────────────────────────────────────────

#[tokio::test]
async fn drain_server_unknown_id_returns_not_ok() {
    let (addr, _state) = spawn_router().await;

    let mut client = GridServiceClient::connect(format!("http://{addr}"))
        .await
        .unwrap();
    let resp = client
        .drain_server(DrainRequest {
            server_id: "no-such-server".into(),
            reason: "test".into(),
        })
        .await
        .unwrap()
        .into_inner();
    assert!(!resp.ok);
    assert!(
        resp.message.contains("not currently serving"),
        "got: {}",
        resp.message
    );
}

#[tokio::test]
async fn drain_server_known_id_dispatches_unassign() {
    let (addr, state) = spawn_router().await;

    let (_donor_tx, mut donor_inbound) =
        join_and_announce(addr, "http://srv:8080", (0, 4), "hash-a").await;
    // Consume the Ack the router sends on registration.
    tokio::time::timeout(Duration::from_secs(1), donor_inbound.next())
        .await
        .unwrap();

    // Look up the server_id the router assigned.
    let server_id = tokio::time::timeout(Duration::from_secs(2), async {
        loop {
            tokio::time::sleep(Duration::from_millis(50)).await;
            let g = state.read();
            let found = g.servers().next().map(|(id, _)| id.clone());
            if let Some(id) = found {
                return id;
            }
        }
    })
    .await
    .expect("server must register");

    // Drain.
    let mut client = GridServiceClient::connect(format!("http://{addr}"))
        .await
        .unwrap();
    let resp = client
        .drain_server(DrainRequest {
            server_id,
            reason: "rebalance test".into(),
        })
        .await
        .unwrap()
        .into_inner();
    assert!(resp.ok, "drain should succeed: {}", resp.message);

    // The donor must observe an UnassignMsg on its inbound stream.
    let observed = tokio::time::timeout(Duration::from_secs(2), donor_inbound.next())
        .await
        .unwrap()
        .unwrap()
        .unwrap();
    match observed.payload {
        Some(RouterPayload::Unassign(u)) => assert_eq!(u.reason, "rebalance test"),
        other => panic!("expected Unassign, got {other:?}"),
    }
}

// ── AssignRange ──────────────────────────────────────────────────────────────

#[tokio::test]
async fn assign_range_with_no_origin_returns_not_ok() {
    let (addr, _state) = spawn_router().await;

    // Spare in the pool, but no live replica anywhere for the requested
    // range and no explicit origin given → admin RPC must refuse.
    let (_spare_tx, _spare_inbound) = join_and_available(addr).await;
    tokio::time::sleep(Duration::from_millis(150)).await;

    let mut client = GridServiceClient::connect(format!("http://{addr}"))
        .await
        .unwrap();
    let resp = client
        .assign_range(AssignRangeRequest {
            model_id: "m".into(),
            layer_start: 0,
            layer_end: 4,
            target_server_id: String::new(),
            explicit_origin_url: String::new(),
            explicit_origin_hash: String::new(),
        })
        .await
        .unwrap()
        .into_inner();
    assert!(!resp.ok);
    assert!(
        resp.message.contains("no live replica"),
        "got: {}",
        resp.message
    );
}

#[tokio::test]
async fn assign_range_with_live_replica_dispatches_to_any_spare() {
    let (addr, _state) = spawn_router().await;

    // A donor covers layers 0-4 — that's the origin for the spare's new
    // replica.
    let (_donor_tx, _donor_inbound) =
        join_and_announce(addr, "http://donor:8080", (0, 4), "hash-d").await;
    // Spare in the available pool.
    let (_spare_tx, mut spare_inbound) = join_and_available(addr).await;
    tokio::time::sleep(Duration::from_millis(150)).await;

    let mut client = GridServiceClient::connect(format!("http://{addr}"))
        .await
        .unwrap();
    let resp = client
        .assign_range(AssignRangeRequest {
            model_id: "m".into(),
            layer_start: 0,
            layer_end: 4,
            target_server_id: String::new(),
            explicit_origin_url: String::new(),
            explicit_origin_hash: String::new(),
        })
        .await
        .unwrap()
        .into_inner();
    assert!(resp.ok, "assign should succeed: {}", resp.message);

    // The spare must see an AssignMsg with the donor as origin.
    let observed = tokio::time::timeout(Duration::from_secs(2), spare_inbound.next())
        .await
        .unwrap()
        .unwrap()
        .unwrap();
    let Some(RouterPayload::Assign(a)) = observed.payload else {
        panic!(
            "expected Assign on the spare's inbound, got {:?}",
            observed.payload
        );
    };
    assert_eq!(a.model_id, "m");
    assert_eq!(a.layer_start, 0);
    assert_eq!(a.layer_end, 4);
    assert_eq!(a.origin_url, "http://donor:8080");
    assert_eq!(a.shard_hash, "hash-d");
}

// ── Library RPC-wrapper coverage ─────────────────────────────────────────────
//
// These exercise `larql_router::admin::admin_status` etc. so the wrappers
// hit the 90% per-file floor without main.rs needing to fire its CLI
// dispatch path in tests.

#[tokio::test]
async fn admin_status_returns_rendered_lines() {
    let (addr, _state) = spawn_router().await;
    let (_donor, _inbound) = join_and_announce(addr, "http://srv:8080", (0, 4), "h").await;
    tokio::time::sleep(Duration::from_millis(150)).await;

    let lines = larql_router::admin::admin_status(&format!("http://{addr}"))
        .await
        .expect("admin_status RPC");
    let joined = lines.join("\n");
    assert!(joined.contains("Model: m"));
    assert!(joined.contains("Servers:"));
}

#[tokio::test]
async fn admin_gaps_reports_no_gaps_when_grid_empty() {
    let (addr, _state) = spawn_router().await;
    let lines = larql_router::admin::admin_gaps(&format!("http://{addr}"), None)
        .await
        .expect("admin_gaps RPC");
    assert_eq!(lines, vec!["No gaps.".to_string()]);
}

#[tokio::test]
async fn admin_drain_returns_ack() {
    let (addr, state) = spawn_router().await;
    let (_donor, mut inbound) = join_and_announce(addr, "http://srv:8080", (0, 4), "h").await;
    let _ = tokio::time::timeout(Duration::from_secs(1), inbound.next()).await; // Ack

    let server_id = tokio::time::timeout(Duration::from_secs(2), async {
        loop {
            tokio::time::sleep(Duration::from_millis(50)).await;
            let g = state.read();
            let found = g.servers().next().map(|(id, _)| id.clone());
            if let Some(id) = found {
                return id;
            }
        }
    })
    .await
    .expect("server registers");

    let ack =
        larql_router::admin::admin_drain(&format!("http://{addr}"), &server_id, "test-reason")
            .await
            .expect("admin_drain RPC");
    assert!(ack.ok);
}

#[tokio::test]
async fn admin_drain_unknown_id_returns_failure_ack() {
    let (addr, _state) = spawn_router().await;
    let ack = larql_router::admin::admin_drain(&format!("http://{addr}"), "no-such", "test-reason")
        .await
        .expect("admin_drain RPC");
    assert!(!ack.ok);
    assert!(ack.message.contains("not currently serving"));
}

#[tokio::test]
async fn admin_assign_with_explicit_origin_succeeds() {
    let (addr, _state) = spawn_router().await;
    // No serving server, but a spare is in the available pool.
    let mut client = GridServiceClient::connect(format!("http://{addr}"))
        .await
        .unwrap();
    let (spare_tx, spare_rx) = mpsc::channel::<ServerMessage>(32);
    let _inbound = client
        .join(ReceiverStream::new(spare_rx))
        .await
        .unwrap()
        .into_inner();
    spare_tx
        .send(ServerMessage {
            payload: Some(ServerPayload::Available(AvailableMsg {
                ram_bytes: 4 * 1024 * 1024 * 1024,
                disk_bytes: 0,
                store_path: "/tmp".into(),
            })),
        })
        .await
        .unwrap();
    tokio::time::sleep(Duration::from_millis(150)).await;

    let ack = larql_router::admin::admin_assign(
        &format!("http://{addr}"),
        "m",
        "0-4",
        None,
        Some("https://shard-bucket/m/0-4.tar"),
        "deadbeef",
    )
    .await
    .expect("admin_assign RPC");
    assert!(ack.ok, "got ack: {ack:?}");
}

#[tokio::test]
async fn admin_assign_invalid_layers_errors_before_rpc() {
    let (addr, _state) = spawn_router().await;
    let err = larql_router::admin::admin_assign(
        &format!("http://{addr}"),
        "m",
        "not-a-range",
        None,
        None,
        "",
    )
    .await
    .expect_err("invalid --layers should fail before the RPC");
    let msg = format!("{err}");
    assert!(
        msg.contains("expected START-END") || msg.contains("--layers"),
        "msg: {msg}"
    );
}

#[tokio::test]
async fn assign_range_explicit_origin_bypasses_live_lookup() {
    let (addr, _state) = spawn_router().await;
    let (_spare_tx, mut spare_inbound) = join_and_available(addr).await;
    tokio::time::sleep(Duration::from_millis(150)).await;

    let mut client = GridServiceClient::connect(format!("http://{addr}"))
        .await
        .unwrap();
    let resp = client
        .assign_range(AssignRangeRequest {
            model_id: "m".into(),
            layer_start: 10,
            layer_end: 14,
            target_server_id: String::new(),
            // No donor for this range — but the operator supplies an
            // external origin (S3, etc.) so the admin RPC accepts.
            explicit_origin_url: "https://shard-bucket/m/10-14.tar".into(),
            explicit_origin_hash: "deadbeef".into(),
        })
        .await
        .unwrap()
        .into_inner();
    assert!(
        resp.ok,
        "explicit-origin assign should succeed: {}",
        resp.message
    );

    let observed = tokio::time::timeout(Duration::from_secs(2), spare_inbound.next())
        .await
        .unwrap()
        .unwrap()
        .unwrap();
    let Some(RouterPayload::Assign(a)) = observed.payload else {
        panic!("expected Assign, got {:?}", observed.payload);
    };
    assert_eq!(a.origin_url, "https://shard-bucket/m/10-14.tar");
    assert_eq!(a.shard_hash, "deadbeef");
}
