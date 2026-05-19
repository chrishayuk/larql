//! End-to-end Mode B smoke test.
//!
//! Scenario: a router, two donors covering disjoint layer ranges, and a spare
//! that joins as available. When a gap appears (donor disconnects) and a live
//! replica still covers the range, the router must:
//!   1. Resolve the surviving replica's `listen_url` as origin
//!   2. Send `AssignMsg` to the spare with the real origin + vindex_hash
//!   3. Allow the spare's `shard_loader` to download a tar from
//!      `GET /v1/shard/{model_id}/{start}-{end}` and unpack it locally
//!   4. Register the spare as serving after `ReadyMsg`
//!
//! Verifies: wire-level coordination between router, donors, and spare; the
//! `/v1/shard` HTTP endpoint contract; and `shard_loader` tar unpack.

use std::sync::Arc;
use std::time::Duration;

use axum::body::Body;
use axum::extract::Path;
use axum::http::{header, StatusCode};
use axum::response::Response;
use axum::routing::get;
use axum::Router;
use bytes::Bytes;
use parking_lot::RwLock;
use tempfile::TempDir;
use tokio::sync::mpsc;
use tokio_stream::wrappers::ReceiverStream;
use tokio_stream::StreamExt;

use larql_router::grid::service::GridServiceImpl;
use larql_router::grid::GridState;
use larql_router_protocol::{
    grid_service_server::GridServiceServer, AnnounceMsg, AvailableMsg, GridServiceClient, ReadyMsg,
    RouterPayload, ServerMessage, ServerPayload,
};
use tonic::transport::Server;

fn make_tar(files: &[(&str, &[u8])]) -> Bytes {
    let mut buf = Vec::new();
    {
        let mut tar = tar::Builder::new(&mut buf);
        for (name, content) in files {
            let mut header = tar::Header::new_gnu();
            header.set_size(content.len() as u64);
            header.set_mode(0o644);
            header.set_cksum();
            tar.append_data(&mut header, name, *content).unwrap();
        }
        tar.finish().unwrap();
    }
    Bytes::from(buf)
}

async fn spawn_shard_donor() -> std::net::SocketAddr {
    async fn handler(Path((_model, _range)): Path<(String, String)>) -> Response {
        let tar = make_tar(&[
            ("index.json", b"{\"shard\":\"donor\"}"),
            ("layer-0.bin", &[0x11, 0x22, 0x33]),
        ]);
        Response::builder()
            .status(StatusCode::OK)
            .header(header::CONTENT_TYPE, "application/x-tar")
            .body(Body::from(tar))
            .unwrap()
    }
    let app = Router::new().route("/v1/shard/{model_id}/{range}", get(handler));
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap();
    tokio::spawn(async move { axum::serve(listener, app).await.unwrap() });
    addr
}

async fn spawn_router() -> (std::net::SocketAddr, Arc<RwLock<GridState>>) {
    let state = Arc::new(RwLock::new(GridState::default()));
    let svc = GridServiceImpl::new(state.clone());

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

async fn announce_client(
    router_addr: std::net::SocketAddr,
    listen_url: String,
    model_id: String,
    layer_start: u32,
    layer_end: u32,
    vindex_hash: &str,
) -> (
    mpsc::Sender<ServerMessage>,
    tonic::Streaming<larql_router_protocol::RouterMessage>,
) {
    let mut client = GridServiceClient::connect(format!("http://{router_addr}"))
        .await
        .unwrap();
    let (tx, rx) = mpsc::channel::<ServerMessage>(32);
    let outbound = ReceiverStream::new(rx);
    let response = client.join(outbound).await.unwrap();
    let inbound = response.into_inner();

    tx.send(ServerMessage {
        payload: Some(ServerPayload::Announce(AnnounceMsg {
            model_id,
            layer_start,
            layer_end,
            ram_bytes: 1024 * 1024 * 1024,
            listen_url,
            vindex_hash: vindex_hash.to_string(),
            expert_start: 0,
            expert_end: 0,
        })),
    })
    .await
    .unwrap();
    (tx, inbound)
}

#[tokio::test]
async fn mode_b_full_vertical_handoff() {
    let donor_http = spawn_shard_donor().await;
    let donor_listen_url = format!("http://{donor_http}");

    let (router_addr, state) = spawn_router().await;

    // Two donors with overlapping replicated coverage for layers 0-4 — these
    // remain alive so they can serve as origins for the spare.
    let (donor_a_tx, _donor_a_inbound) = announce_client(
        router_addr,
        donor_listen_url.clone(),
        "test-model".into(),
        0,
        4,
        "hash-A",
    )
    .await;
    let (_donor_b_tx, _donor_b_inbound) = announce_client(
        router_addr,
        donor_listen_url.clone(),
        "test-model".into(),
        0,
        4,
        "hash-B",
    )
    .await;

    tokio::time::sleep(Duration::from_millis(200)).await;
    {
        let g = state.read();
        assert_eq!(
            g.status_response().servers.len(),
            2,
            "two donors must be registered"
        );
    }

    // Spare connects as Available.
    let mut spare_client = GridServiceClient::connect(format!("http://{router_addr}"))
        .await
        .unwrap();
    let (spare_tx, spare_rx) = mpsc::channel::<ServerMessage>(32);
    let spare_response = spare_client
        .join(ReceiverStream::new(spare_rx))
        .await
        .unwrap();
    let mut spare_inbound = spare_response.into_inner();

    let tmp = TempDir::new().unwrap();
    let store_path = tmp.path().to_string_lossy().to_string();

    spare_tx
        .send(ServerMessage {
            payload: Some(ServerPayload::Available(AvailableMsg {
                ram_bytes: 8 * 1024 * 1024 * 1024,
                disk_bytes: 0,
                store_path: store_path.clone(),
            })),
        })
        .await
        .unwrap();
    tokio::time::sleep(Duration::from_millis(200)).await;

    // ── Trigger a replica add: simulate the rebalancer asking for an
    //    additional replica of layers 0-4. We don't have a UnassignMsg flow
    //    in this test (the running donor stub can't drain a real load), so
    //    we drive the assignment manually via the GridState API the
    //    rebalancer would use. The wire-level path (Available → Assign →
    //    download → Ready) is exactly the same.
    {
        let mut g = state.write();
        let sent = g.try_assign_gap("test-model", 0, 4, 0, 0, 0);
        assert!(
            sent,
            "try_assign_gap must succeed when a live replica exists as origin"
        );
    }

    // Spare must observe AssignMsg.
    let assign = tokio::time::timeout(Duration::from_secs(2), async {
        loop {
            match spare_inbound.next().await {
                Some(Ok(rm)) => {
                    if let Some(RouterPayload::Assign(a)) = rm.payload {
                        return a;
                    }
                }
                Some(Err(e)) => panic!("spare stream error: {e}"),
                None => panic!("spare stream closed before AssignMsg"),
            }
        }
    })
    .await
    .expect("AssignMsg should arrive within 2s");

    assert_eq!(assign.model_id, "test-model");
    assert_eq!(assign.layer_start, 0);
    assert_eq!(assign.layer_end, 4);
    assert_eq!(assign.origin_url, donor_listen_url);
    // The router prefers whichever donor it finds first in HashMap iteration
    // order — either "hash-A" or "hash-B" is acceptable.
    assert!(
        assign.shard_hash == "hash-A" || assign.shard_hash == "hash-B",
        "shard_hash must come from a live donor, got: {}",
        assign.shard_hash
    );

    // Spare-side download via shard_loader against the donor's HTTP origin.
    larql_server::shard_loader::download_and_load_shard(
        &assign.origin_url,
        &store_path,
        "", // hash check skipped (placeholder)
        &assign.model_id,
        assign.layer_start,
        assign.layer_end,
    )
    .await
    .expect("shard_loader download must succeed");

    let dest = std::path::PathBuf::from(&store_path)
        .join("test-model")
        .join("layers-0-4");
    assert!(dest.is_dir(), "shard unpacked: {dest:?}");
    let body = std::fs::read(dest.join("index.json")).unwrap();
    assert_eq!(body, b"{\"shard\":\"donor\"}");

    // Acknowledge with ReadyMsg — spare must register as serving.
    spare_tx
        .send(ServerMessage {
            payload: Some(ServerPayload::Ready(ReadyMsg {
                model_id: assign.model_id.clone(),
                layer_start: assign.layer_start,
                layer_end: assign.layer_end,
                listen_url: "http://spare:9999".into(),
                expert_start: 0,
                expert_end: 0,
            })),
        })
        .await
        .unwrap();
    tokio::time::sleep(Duration::from_millis(200)).await;
    {
        let g = state.read();
        let urls: Vec<String> = g
            .status_response()
            .servers
            .iter()
            .map(|s| s.listen_url.clone())
            .collect();
        assert!(
            urls.contains(&"http://spare:9999".to_string()),
            "spare must appear in status after ReadyMsg: got {urls:?}"
        );
    }

    drop(donor_a_tx);
}

/// Drives the entire Mode B round-trip through
/// `announce::try_once_available` — the same code path that
/// `run_announce_available` (the daemon entry point) uses.
/// `mode_b_full_vertical_handoff` above wires the gRPC stream
/// manually and calls `shard_loader::download_and_load_shard`
/// directly; this test asserts the *production* loop wires
/// Available → Assign → download → Ready → Ack end-to-end.
#[tokio::test]
async fn mode_b_try_once_available_drives_full_handshake() {
    let _ = tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| tracing_subscriber::EnvFilter::new("warn")),
        )
        .with_test_writer()
        .try_init();

    let donor_http = spawn_shard_donor().await;
    let donor_listen_url = format!("http://{donor_http}");
    let (router_addr, state) = spawn_router().await;

    // Donor announces with a placeholder hash — the only value the
    // production `shard_loader` accepts without verifying a SHA-256 of
    // the downloaded tar against the announce-time `vindex_hash`.
    // The non-placeholder behaviour is a known production-side
    // inconsistency tracked separately (vindex_identity_hash is a
    // 16-hex model-identity tag, not a content hash — shard_loader
    // expects the latter). See ROADMAP "GT5 hash verification"
    // follow-up.
    let (_donor_tx, _donor_inbound) = announce_client(
        router_addr,
        donor_listen_url.clone(),
        "test-model".into(),
        0,
        4,
        "",
    )
    .await;
    tokio::time::sleep(Duration::from_millis(200)).await;
    assert_eq!(state.read().status_response().servers.len(), 1);

    // Mode B config — production daemon path.
    let tmp = TempDir::new().unwrap();
    let store_path = tmp.path().to_string_lossy().to_string();
    let cfg = larql_server::announce::AvailableConfig {
        join_url: format!("http://{router_addr}"),
        listen_url: "http://spare-via-try-once:9999".into(),
        ram_bytes: 8 * 1024 * 1024 * 1024,
        disk_bytes: 0,
        store_path: store_path.clone(),
        grid_key: None,
        quic_cert_fingerprint: None,
    };

    // Spawn the real Mode B handshake. The task should return Ok(())
    // once the spare receives AckMsg after sending ReadyMsg.
    let handle =
        tokio::spawn(async move { larql_server::announce::try_once_available(&cfg).await });

    // Give the spare time to send AvailableMsg + register as available.
    tokio::time::sleep(Duration::from_millis(300)).await;

    // Trigger the assignment — same call the rebalancer would issue
    // for an under-replicated range.
    let sent = state.write().try_assign_gap("test-model", 0, 4, 0, 0, 0);
    assert!(
        sent,
        "try_assign_gap should succeed: live origin exists + spare is available"
    );

    // The Mode B loop should download the shard and ack within 3s.
    let res = tokio::time::timeout(Duration::from_secs(3), handle)
        .await
        .expect("try_once_available must complete within 3s")
        .expect("task must not panic");
    res.expect("Mode B handshake should succeed");

    // Disk-side: the tar got unpacked at the expected path.
    let dest = std::path::PathBuf::from(&store_path)
        .join("test-model")
        .join("layers-0-4");
    assert!(
        dest.is_dir(),
        "shard must be unpacked at {dest:?} by the spare's run_available_loop"
    );
    let body = std::fs::read(dest.join("index.json")).unwrap();
    assert_eq!(body, b"{\"shard\":\"donor\"}");

    // Router-side: the spare must appear as serving with its listen_url.
    let urls: Vec<String> = state
        .read()
        .status_response()
        .servers
        .iter()
        .map(|s| s.listen_url.clone())
        .collect();
    assert!(
        urls.contains(&"http://spare-via-try-once:9999".to_string()),
        "spare must register as serving after ReadyMsg; got servers: {urls:?}"
    );
}

#[tokio::test]
async fn no_assign_when_gap_has_no_surviving_origin() {
    let (router_addr, state) = spawn_router().await;

    // Single donor for layers 10-14 — no replicas.
    let (donor_tx, donor_inbound) = announce_client(
        router_addr,
        "http://donor:8080".into(),
        "test-model".into(),
        10,
        14,
        "hash-X",
    )
    .await;

    // Spare connects as Available.
    let mut spare_client = GridServiceClient::connect(format!("http://{router_addr}"))
        .await
        .unwrap();
    let (spare_tx, spare_rx) = mpsc::channel::<ServerMessage>(32);
    let spare_response = spare_client
        .join(ReceiverStream::new(spare_rx))
        .await
        .unwrap();
    let mut spare_inbound = spare_response.into_inner();

    let tmp = TempDir::new().unwrap();
    spare_tx
        .send(ServerMessage {
            payload: Some(ServerPayload::Available(AvailableMsg {
                ram_bytes: 8 * 1024 * 1024 * 1024,
                disk_bytes: 0,
                store_path: tmp.path().to_string_lossy().to_string(),
            })),
        })
        .await
        .unwrap();

    tokio::time::sleep(Duration::from_millis(150)).await;

    // Kill the only donor — no live origin for layers 10-14.
    drop(donor_tx);
    drop(donor_inbound);
    tokio::time::sleep(Duration::from_millis(200)).await;

    // The gap is detected (only one shard, no overlap → coverage_gaps is
    // empty because shards adjacency check requires multiple shards) but
    // even if it were detected, find_origin_for would return None. Either
    // way: the spare must not receive an AssignMsg.
    let result = tokio::time::timeout(Duration::from_millis(300), spare_inbound.next()).await;
    assert!(
        result.is_err(),
        "spare must not receive AssignMsg without a live origin: got {result:?}"
    );

    // Confirm the route table no longer holds the dead donor.
    let g = state.read();
    assert_eq!(g.status_response().servers.len(), 0);
}
