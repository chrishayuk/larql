//! Phase 4 replication integration test.
//!
//! Sets `target_replicas=2` on the router, registers a single donor, then
//! has a spare join as Available. The router must:
//!   1. Detect that the donor's range is under-replicated (count=1 < 2)
//!   2. Resolve the donor as origin and send `AssignMsg` to the spare
//!   3. Once the spare sends `ReadyMsg`, the range is at target — no more
//!      assignments should fire on a subsequent spare.

use std::sync::Arc;
use std::time::Duration;

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

async fn spawn_router(target_replicas: u32) -> (std::net::SocketAddr, Arc<RwLock<GridState>>) {
    let state = Arc::new(RwLock::new(GridState::default()));
    state.write().set_target_replicas(target_replicas);
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

async fn announce(
    addr: std::net::SocketAddr,
    listen_url: &str,
    layers: (u32, u32),
    hash: &str,
) -> (
    mpsc::Sender<ServerMessage>,
    tonic::Streaming<larql_router_protocol::RouterMessage>,
) {
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
            model_id: "test-model".into(),
            layer_start: layers.0,
            layer_end: layers.1,
            ram_bytes: 1024 * 1024 * 1024,
            listen_url: listen_url.to_string(),
            vindex_hash: hash.to_string(),
            expert_start: 0,
            expert_end: 0,
        })),
    })
    .await
    .unwrap();
    (tx, inbound)
}

async fn available(
    addr: std::net::SocketAddr,
    ram_gb: u64,
    store_path: &str,
) -> (
    mpsc::Sender<ServerMessage>,
    tonic::Streaming<larql_router_protocol::RouterMessage>,
) {
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
            ram_bytes: ram_gb * 1024 * 1024 * 1024,
            disk_bytes: 0,
            store_path: store_path.to_string(),
        })),
    })
    .await
    .unwrap();
    (tx, inbound)
}

#[tokio::test]
async fn spare_replicates_under_replicated_range() {
    let (addr, state) = spawn_router(2).await;

    // One donor covering layers 0-4 — under-replicated by 1.
    let (_donor_tx, _donor_inbound) =
        announce(addr, "http://donor:8080", (0, 4), "donor-hash").await;
    tokio::time::sleep(Duration::from_millis(150)).await;

    // Spare joins.
    let tmp = TempDir::new().unwrap();
    let (_spare_tx, mut spare_inbound) = available(addr, 8, &tmp.path().to_string_lossy()).await;

    // Spare must receive AssignMsg for layers 0-4 with donor as origin.
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
    .expect("under-replicated range must trigger an AssignMsg");

    assert_eq!(assign.model_id, "test-model");
    assert_eq!(assign.layer_start, 0);
    assert_eq!(assign.layer_end, 4);
    assert_eq!(assign.origin_url, "http://donor:8080");
    assert_eq!(assign.shard_hash, "donor-hash");

    // Confirm the available pool drained.
    {
        let g = state.read();
        assert!(
            !g.has_available_servers(),
            "spare must have been consumed by replication"
        );
    }
}

#[tokio::test]
async fn second_spare_not_assigned_after_target_met() {
    let (addr, state) = spawn_router(2).await;

    // Donor + a pre-registered second replica via a fake Ready (we mock it
    // by registering directly through the state API to avoid building two
    // full announce streams).
    let (_donor_tx, _donor_inbound) =
        announce(addr, "http://donor:8080", (0, 4), "donor-hash").await;

    // Simulate a second already-serving replica by registering a second
    // donor at the same range.
    let (_donor2_tx, _donor2_inbound) =
        announce(addr, "http://donor-2:8080", (0, 4), "donor2-hash").await;
    tokio::time::sleep(Duration::from_millis(200)).await;
    {
        let g = state.read();
        assert_eq!(g.status_response().servers.len(), 2);
        assert!(g.under_replicated_ranges().is_empty());
    }

    // Now a spare joins — replicas already at target, so no AssignMsg
    // should fire.
    let tmp = TempDir::new().unwrap();
    let (_spare_tx, mut spare_inbound) = available(addr, 8, &tmp.path().to_string_lossy()).await;

    let result = tokio::time::timeout(Duration::from_millis(300), spare_inbound.next()).await;
    assert!(
        result.is_err(),
        "no AssignMsg should arrive when at target_replicas: got {result:?}"
    );

    // Spare must remain in the available pool.
    let g = state.read();
    assert!(g.has_available_servers());
}

#[tokio::test]
async fn ready_replica_satisfies_target() {
    let (addr, state) = spawn_router(2).await;
    let (_donor_tx, _donor_inbound) =
        announce(addr, "http://donor:8080", (0, 4), "donor-hash").await;
    tokio::time::sleep(Duration::from_millis(150)).await;

    let tmp = TempDir::new().unwrap();
    let (spare_tx, mut spare_inbound) = available(addr, 8, &tmp.path().to_string_lossy()).await;

    // Spare gets AssignMsg.
    let assign = tokio::time::timeout(Duration::from_secs(2), async {
        loop {
            match spare_inbound.next().await {
                Some(Ok(rm)) => {
                    if let Some(RouterPayload::Assign(a)) = rm.payload {
                        return a;
                    }
                }
                _ => panic!("expected AssignMsg"),
            }
        }
    })
    .await
    .expect("AssignMsg must land");

    // Spare sends ReadyMsg — now the range has 2 replicas.
    spare_tx
        .send(ServerMessage {
            payload: Some(ServerPayload::Ready(ReadyMsg {
                model_id: assign.model_id,
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
        assert_eq!(g.status_response().servers.len(), 2);
        assert!(g.under_replicated_ranges().is_empty());
    }
}
