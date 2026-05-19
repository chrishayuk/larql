//! Phase B2 drain-then-reassign integration test.
//!
//! Drives the real production announce loop (`announce::try_once`) against an
//! in-process router. When the router sends `UnassignMsg`, the server must:
//!   1. Drain in-flight (counter is zero → immediate)
//!   2. Send `DroppingMsg` with reason="reassigned"
//!   3. Stay on the same stream and send `AvailableMsg` (because
//!      `available_after_drain` is populated)
//!
//! The router must then move the server into the available pool and be able
//! to send it a fresh `AssignMsg` for a different layer range.

use std::sync::atomic::AtomicU32;
use std::sync::Arc;
use std::time::Duration;

use parking_lot::RwLock;
use tempfile::TempDir;
use tokio::sync::mpsc;
use tokio_stream::wrappers::ReceiverStream;

use larql_router::grid::service::GridServiceImpl;
use larql_router::grid::GridState;
use larql_router_protocol::{
    grid_service_server::GridServiceServer, AnnounceMsg, GridServiceClient, RouterMessage,
    RouterPayload, ServerMessage, ServerPayload, UnassignMsg,
};
use larql_server::announce::{self, AnnounceConfig, AvailableConfig};
use larql_server::metrics::LayerLatencyTracker;
use tonic::transport::Server;

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

/// Pre-register a "live" donor on the router so it can serve as origin for the
/// re-assignment after the drained server enters the available pool.
async fn spawn_live_origin(
    router_addr: std::net::SocketAddr,
    listen_url: &str,
    layer_start: u32,
    layer_end: u32,
) -> (mpsc::Sender<ServerMessage>, tonic::Streaming<RouterMessage>) {
    let mut client = GridServiceClient::connect(format!("http://{router_addr}"))
        .await
        .unwrap();
    let (tx, rx) = mpsc::channel::<ServerMessage>(32);
    let outbound = ReceiverStream::new(rx);
    let inbound = client.join(outbound).await.unwrap().into_inner();

    tx.send(ServerMessage {
        payload: Some(ServerPayload::Announce(AnnounceMsg {
            model_id: "test-model".into(),
            layer_start,
            layer_end,
            ram_bytes: 1024 * 1024 * 1024,
            listen_url: listen_url.into(),
            vindex_hash: "live-origin-hash".into(),
            expert_start: 0,
            expert_end: 0,
        })),
    })
    .await
    .unwrap();
    (tx, inbound)
}

#[tokio::test]
async fn drain_then_reassign_via_available_after_drain() {
    let (router_addr, state) = spawn_router().await;

    // A live donor that will provide the origin for the post-drain reassign.
    let (_origin_tx, _origin_inbound) =
        spawn_live_origin(router_addr, "http://origin:8080", 20, 24).await;

    // The drained server we will exercise. Its `available_after_drain` is set
    // to a Mode B config that points at the same store_path and lists 8 GB RAM.
    let tmp = TempDir::new().unwrap();
    let store_path = tmp.path().to_string_lossy().to_string();

    let cfg = AnnounceConfig {
        join_url: format!("http://{router_addr}"),
        model_id: "test-model".into(),
        layer_start: 0,
        layer_end: 4,
        listen_url: "http://drained:8080".into(),
        ram_bytes: 0,
        grid_key: None,
        vindex_hash: "drained-hash".into(),
        latency_tracker: Arc::new(LayerLatencyTracker::new()),
        requests_in_flight: Arc::new(AtomicU32::new(0)),
        requests_total: Arc::new(std::sync::atomic::AtomicU64::new(0)),
        available_after_drain: Some(AvailableConfig {
            join_url: format!("http://{router_addr}"),
            listen_url: "http://drained:8080".into(),
            ram_bytes: 8 * 1024 * 1024 * 1024,
            disk_bytes: 0,
            store_path,
            grid_key: None,
            quic_cert_fingerprint: None,
        }),
        quic_cert_fingerprint: None,
    };

    // Drive the real announce loop in the background.
    let announce_task = tokio::spawn(async move { announce::try_once(&cfg).await });

    // Wait for the drained server to register as serving.
    let drained_server_id = tokio::time::timeout(Duration::from_secs(2), async {
        loop {
            tokio::time::sleep(Duration::from_millis(50)).await;
            let g = state.read();
            for (sid, entry) in g.servers() {
                if entry.listen_url == "http://drained:8080" {
                    return sid.clone();
                }
            }
        }
    })
    .await
    .expect("drained server must register within 2s");

    // Confirm: 2 servers serving (origin + drained), 0 in available pool.
    {
        let g = state.read();
        assert_eq!(g.status_response().servers.len(), 2);
        assert!(!g.has_available_servers());
    }

    // Trigger UnassignMsg → router sends down the drained server's
    // serving_sender channel. The production announce loop must then drain,
    // send DroppingMsg, and re-enter run_available_loop on the same stream.
    // Clone the sender then drop the read guard before awaiting — the
    // parking_lot guard is !Send and can't be held across an `.await`.
    let sender = state
        .read()
        .serving_sender(&drained_server_id)
        .expect("router must hold the drained server's sender");
    sender
        .send(Ok(RouterMessage {
            payload: Some(RouterPayload::Unassign(UnassignMsg {
                model_id: "test-model".into(),
                layer_start: 0,
                layer_end: 4,
                reason: "rebalancing".into(),
            })),
        }))
        .await
        .unwrap();

    // The drained server now: drain → DroppingMsg → re-enter Mode B.
    // After that, GridState should:
    //   - deregister the serving entry
    //   - register an available entry
    //   - (because the live origin still covers 0-4, gap re-fill might or
    //     might not fire here — coverage_gaps uses consecutive shard gaps,
    //     and the survivor at 20-24 is in a different range, so no gap to
    //     fill for 0-4. Confirm available pool is populated either way.)
    tokio::time::timeout(Duration::from_secs(3), async {
        loop {
            tokio::time::sleep(Duration::from_millis(50)).await;
            let g = state.read();
            if g.has_available_servers() {
                return;
            }
        }
    })
    .await
    .expect("drained server must appear in available pool within 3s");

    {
        let g = state.read();
        // The original serving entry is gone.
        let serving_urls: Vec<String> = g
            .status_response()
            .servers
            .iter()
            .map(|s| s.listen_url.clone())
            .collect();
        assert!(
            !serving_urls.contains(&"http://drained:8080".to_string()),
            "drained server must have left the serving set: {serving_urls:?}"
        );
    }

    // Now drive an assignment to the newly-available server for the
    // live-origin range (20-24). The router must resolve the origin from the
    // surviving donor and dispatch AssignMsg.
    {
        let mut g = state.write();
        let sent = g.try_assign_gap("test-model", 20, 24, 0, 0, 0);
        assert!(sent, "router must dispatch AssignMsg to the available pool");
    }

    // The announce_task may still be running run_available_loop. We don't
    // wait for it to complete (it would only end when the stream closes or
    // an Ack is received after Ready). Abort it as cleanup.
    announce_task.abort();
}
