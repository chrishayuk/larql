//! Router-side integration tests for `GridServiceImpl::join`.
//!
//! These mirror coverage that the server-crate integration tests provide via
//! a full live grid; we run them here so `cargo llvm-cov -p larql-router`
//! attributes the gRPC handler lines back to the router crate.

use std::sync::Arc;
use std::time::Duration;

use parking_lot::RwLock;
use tokio::sync::mpsc;
use tokio_stream::wrappers::ReceiverStream;
use tokio_stream::StreamExt;

use larql_router::grid::service::GridServiceImpl;
use larql_router::grid::GridState;
use larql_router_protocol::{
    grid_service_server::GridServiceServer, AnnounceMsg, AvailableMsg, GridServiceClient,
    HeartbeatMsg, LayerLatency, ReadyMsg, RouterMessage, RouterPayload, ServerMessage,
    ServerPayload, StatusRequest, UnassignMsg,
};
use tonic::transport::Server;

async fn spawn_router(grid_key: Option<String>) -> (std::net::SocketAddr, Arc<RwLock<GridState>>) {
    // Install a metrics registry so the service's instrumentation
    // branches (`if let Some(m) = &metrics`) are exercised end-to-end.
    let metrics = larql_router::metrics::RouterMetrics::new();
    let state = Arc::new(RwLock::new(GridState::default()));
    let svc = GridServiceImpl::new_with_key(state.clone(), grid_key).with_metrics(metrics);
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

async fn join_with_auth(
    addr: std::net::SocketAddr,
    bearer: Option<&'static str>,
) -> Result<(mpsc::Sender<ServerMessage>, tonic::Streaming<RouterMessage>), tonic::Status> {
    let channel = tonic::transport::Channel::from_shared(format!("http://{addr}"))
        .unwrap()
        .connect()
        .await
        .unwrap();
    // tonic's interceptor closure returns `Result<Request<()>, tonic::Status>`,
    // and `Status` is ~176 bytes — that's the tonic API, not something we can
    // box without forking the trait. Allow the lint locally.
    #[allow(clippy::result_large_err)]
    let mut client =
        GridServiceClient::with_interceptor(channel, move |mut req: tonic::Request<()>| {
            if let Some(b) = bearer {
                req.metadata_mut()
                    .insert("authorization", format!("Bearer {b}").parse().unwrap());
            }
            Ok(req)
        });
    let (tx, rx) = mpsc::channel::<ServerMessage>(32);
    let response = client.join(ReceiverStream::new(rx)).await?;
    Ok((tx, response.into_inner()))
}

#[tokio::test]
async fn announce_then_heartbeat_then_dropping() {
    let (addr, state) = spawn_router(None).await;

    let (tx, mut inbound) = join_with_auth(addr, None).await.unwrap();

    // Announce.
    tx.send(ServerMessage {
        payload: Some(ServerPayload::Announce(AnnounceMsg {
            model_id: "m".into(),
            layer_start: 0,
            layer_end: 4,
            ram_bytes: 1,
            listen_url: "http://srv".into(),
            vindex_hash: "h".into(),
            expert_start: 0,
            expert_end: 0,
        })),
    })
    .await
    .unwrap();

    // Router should send AckMsg.
    let ack = tokio::time::timeout(Duration::from_secs(1), inbound.next())
        .await
        .unwrap()
        .unwrap()
        .unwrap();
    assert!(matches!(ack.payload, Some(RouterPayload::Ack(_))));

    // Heartbeat with layer stats — exercises update_heartbeat path.
    tx.send(ServerMessage {
        payload: Some(ServerPayload::Heartbeat(HeartbeatMsg {
            cpu_pct: 12.5,
            ram_used: 4096,
            requests_in_flight: 2,
            layer_stats: vec![LayerLatency {
                layer: 0,
                avg_ms: 1.5,
                p99_ms: 3.0,
            }],
            req_per_sec: 0.0,
        })),
    })
    .await
    .unwrap();
    tokio::time::sleep(Duration::from_millis(100)).await;

    {
        let g = state.read();
        let s = g.status_response();
        assert_eq!(s.servers.len(), 1);
        assert_eq!(s.servers[0].cpu_pct, 12.5);
        assert_eq!(s.servers[0].requests_in_flight, 2);
        assert_eq!(s.servers[0].layer_stats.len(), 1);
    }

    // Dropping deregisters.
    tx.send(ServerMessage {
        payload: Some(ServerPayload::Dropping(
            larql_router_protocol::DroppingMsg {
                model_id: "m".into(),
                layer_start: 0,
                layer_end: 4,
                reason: "shutdown".into(),
            },
        )),
    })
    .await
    .unwrap();
    tokio::time::sleep(Duration::from_millis(150)).await;
    {
        let g = state.read();
        assert_eq!(g.status_response().servers.len(), 0);
    }
}

#[tokio::test]
async fn missing_grid_key_is_rejected_with_unauthenticated() {
    let (addr, _state) = spawn_router(Some("topsecret".into())).await;
    let err = join_with_auth(addr, None)
        .await
        .expect_err("join must fail without bearer");
    assert_eq!(err.code(), tonic::Code::Unauthenticated);
}

#[tokio::test]
async fn wrong_grid_key_is_rejected() {
    let (addr, _state) = spawn_router(Some("topsecret".into())).await;
    let err = join_with_auth(addr, Some("wrong"))
        .await
        .expect_err("join must fail with wrong bearer");
    assert_eq!(err.code(), tonic::Code::Unauthenticated);
}

#[tokio::test]
async fn correct_grid_key_accepted() {
    let (addr, _state) = spawn_router(Some("topsecret".into())).await;
    let (tx, _inbound) = join_with_auth(addr, Some("topsecret"))
        .await
        .expect("join must succeed with correct bearer");
    drop(tx);
}

#[tokio::test]
async fn available_followed_by_ready_registers_as_serving() {
    let (addr, state) = spawn_router(None).await;
    let (tx, _inbound) = join_with_auth(addr, None).await.unwrap();

    tx.send(ServerMessage {
        payload: Some(ServerPayload::Available(AvailableMsg {
            ram_bytes: 1024 * 1024 * 1024,
            disk_bytes: 0,
            store_path: "/tmp".into(),
        })),
    })
    .await
    .unwrap();
    tokio::time::sleep(Duration::from_millis(100)).await;
    {
        let g = state.read();
        assert!(g.has_available_servers());
    }

    // Ready transitions the connection into the serving set.
    tx.send(ServerMessage {
        payload: Some(ServerPayload::Ready(ReadyMsg {
            model_id: "m".into(),
            layer_start: 0,
            layer_end: 4,
            listen_url: "http://spare:9999".into(),
            expert_start: 0,
            expert_end: 0,
        })),
    })
    .await
    .unwrap();
    tokio::time::sleep(Duration::from_millis(150)).await;
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
            "got {urls:?}"
        );
    }
}

#[tokio::test]
async fn refuse_is_logged_without_panicking() {
    let (addr, _state) = spawn_router(None).await;
    let (tx, _inbound) = join_with_auth(addr, None).await.unwrap();

    // Refuse without prior Available — server is just logging; the router
    // must accept it without panicking and keep the stream alive.
    tx.send(ServerMessage {
        payload: Some(ServerPayload::Refuse(larql_router_protocol::RefuseMsg {
            model_id: "m".into(),
            layer_start: 0,
            layer_end: 4,
            reason: "insufficient_disk".into(),
        })),
    })
    .await
    .unwrap();
    tokio::time::sleep(Duration::from_millis(100)).await;
}

#[tokio::test]
async fn status_rpc_returns_current_grid() {
    let (addr, _state) = spawn_router(None).await;
    let (tx, _inbound) = join_with_auth(addr, None).await.unwrap();
    tx.send(ServerMessage {
        payload: Some(ServerPayload::Announce(AnnounceMsg {
            model_id: "m".into(),
            layer_start: 0,
            layer_end: 4,
            ram_bytes: 1,
            listen_url: "http://srv".into(),
            vindex_hash: "h".into(),
            expert_start: 0,
            expert_end: 0,
        })),
    })
    .await
    .unwrap();
    tokio::time::sleep(Duration::from_millis(100)).await;

    let mut client = GridServiceClient::connect(format!("http://{addr}"))
        .await
        .unwrap();
    let resp = client.status(StatusRequest {}).await.unwrap().into_inner();
    assert_eq!(resp.servers.len(), 1);
    assert_eq!(resp.servers[0].listen_url, "http://srv");
}

#[tokio::test]
async fn unassign_via_serving_sender_reaches_client() {
    // Drives the rebalancer's UnassignMsg path through the server channel —
    // verifies the gRPC outbound stream actually delivers it.
    let (addr, state) = spawn_router(None).await;
    let (tx, mut inbound) = join_with_auth(addr, None).await.unwrap();

    tx.send(ServerMessage {
        payload: Some(ServerPayload::Announce(AnnounceMsg {
            model_id: "m".into(),
            layer_start: 0,
            layer_end: 4,
            ram_bytes: 1,
            listen_url: "http://srv".into(),
            vindex_hash: "h".into(),
            expert_start: 0,
            expert_end: 0,
        })),
    })
    .await
    .unwrap();

    // Consume Ack.
    let _ = tokio::time::timeout(Duration::from_secs(1), inbound.next())
        .await
        .unwrap()
        .unwrap();

    // Drive UnassignMsg via the serving_sender the router holds.
    let server_id = {
        let g = state.read();
        let id = g.servers().next().map(|(id, _)| id.clone()).unwrap();
        id
    };
    let sender = state.read().serving_sender(&server_id).unwrap();
    sender
        .send(Ok(RouterMessage {
            payload: Some(RouterPayload::Unassign(UnassignMsg {
                model_id: "m".into(),
                layer_start: 0,
                layer_end: 4,
                reason: "test".into(),
            })),
        }))
        .await
        .unwrap();

    // Client must observe the Unassign.
    let observed = tokio::time::timeout(Duration::from_secs(1), inbound.next())
        .await
        .unwrap()
        .unwrap()
        .unwrap();
    assert!(matches!(observed.payload, Some(RouterPayload::Unassign(_))));
}

/// Exercises the Mode B Available → replicate-from-available path
/// (no gap, but the existing range is under-replicated).
#[tokio::test]
async fn available_with_under_replication_triggers_replicate() {
    let (addr, state) = spawn_router(None).await;
    {
        let mut g = state.write();
        g.set_target_replicas(2);
    }

    // One server covers layers 0-4 — under-replicated by 1 against target=2.
    let (tx_serving, _inbound_serving) = join_with_auth(addr, None).await.unwrap();
    tx_serving
        .send(ServerMessage {
            payload: Some(ServerPayload::Announce(AnnounceMsg {
                model_id: "m".into(),
                layer_start: 0,
                layer_end: 4,
                ram_bytes: 1,
                listen_url: "http://serving".into(),
                vindex_hash: "h".into(),
                expert_start: 0,
                expert_end: 0,
            })),
        })
        .await
        .unwrap();
    tokio::time::sleep(Duration::from_millis(100)).await;

    // Spare joins as Available.
    let (tx_spare, mut inbound_spare) = join_with_auth(addr, None).await.unwrap();
    tx_spare
        .send(ServerMessage {
            payload: Some(ServerPayload::Available(AvailableMsg {
                ram_bytes: 1_000_000_000,
                disk_bytes: 0,
                store_path: "/tmp".into(),
            })),
        })
        .await
        .unwrap();

    let assign = tokio::time::timeout(Duration::from_secs(2), inbound_spare.next())
        .await
        .expect("router must respond within timeout")
        .expect("stream must yield a message")
        .expect("ok payload");
    match assign.payload {
        Some(RouterPayload::Assign(a)) => {
            assert_eq!(a.model_id, "m");
            assert_eq!(a.layer_start, 0);
            assert_eq!(a.layer_end, 4);
            assert_eq!(a.origin_url, "http://serving");
        }
        other => panic!("expected Assign from replicate path, got {other:?}"),
    }
}

/// Exercises the post-stream cleanup `filled > 0 || replicated > 0`
/// path: drop a serving stream that owns the only replica of an
/// under-replicated range, with a Mode B spare standing by. The
/// disconnect should trigger gap-fill + replicate, which should
/// dispatch an AssignMsg to the spare.
#[tokio::test]
async fn serving_disconnect_triggers_post_stream_replicate() {
    let (addr, state) = spawn_router(None).await;
    {
        let mut g = state.write();
        g.set_target_replicas(2);
    }

    // Two serving streams cover layers 0-4. Once one drops, the other
    // remains and the spare should pick up the under-replicated slot.
    let (tx_a, _inbound_a) = join_with_auth(addr, None).await.unwrap();
    tx_a.send(ServerMessage {
        payload: Some(ServerPayload::Announce(AnnounceMsg {
            model_id: "m".into(),
            layer_start: 0,
            layer_end: 4,
            ram_bytes: 1,
            listen_url: "http://srv-a".into(),
            vindex_hash: "h".into(),
            expert_start: 0,
            expert_end: 0,
        })),
    })
    .await
    .unwrap();
    let (tx_b, _inbound_b) = join_with_auth(addr, None).await.unwrap();
    tx_b.send(ServerMessage {
        payload: Some(ServerPayload::Announce(AnnounceMsg {
            model_id: "m".into(),
            layer_start: 0,
            layer_end: 4,
            ram_bytes: 1,
            listen_url: "http://srv-b".into(),
            vindex_hash: "h".into(),
            expert_start: 0,
            expert_end: 0,
        })),
    })
    .await
    .unwrap();

    // Mode B spare, just sitting in the available pool.
    let (tx_spare, mut inbound_spare) = join_with_auth(addr, None).await.unwrap();
    tx_spare
        .send(ServerMessage {
            payload: Some(ServerPayload::Available(AvailableMsg {
                ram_bytes: 1_000_000_000,
                disk_bytes: 0,
                store_path: "/tmp".into(),
            })),
        })
        .await
        .unwrap();
    tokio::time::sleep(Duration::from_millis(150)).await;

    // Drop server B's stream — disconnect path should fire.
    drop(tx_b);
    let assign = tokio::time::timeout(Duration::from_secs(2), inbound_spare.next())
        .await
        .expect("router must respond to disconnect within timeout")
        .expect("stream must yield a message")
        .expect("ok payload");
    assert!(matches!(assign.payload, Some(RouterPayload::Assign(_))));
}

/// Exercises the `payload: None` arm in the join handler's main
/// match. A `ServerMessage` with no payload must be silently skipped
/// — the router stays connected and processes a subsequent
/// well-formed payload normally.
#[tokio::test]
async fn payload_none_is_silently_skipped() {
    let (addr, state) = spawn_router(None).await;
    let (tx, mut inbound) = join_with_auth(addr, None).await.unwrap();

    // Send an empty payload first.
    tx.send(ServerMessage { payload: None }).await.unwrap();
    tokio::time::sleep(Duration::from_millis(50)).await;

    // Stream must still be alive — follow with a real Announce.
    tx.send(ServerMessage {
        payload: Some(ServerPayload::Announce(AnnounceMsg {
            model_id: "m".into(),
            layer_start: 0,
            layer_end: 4,
            ram_bytes: 1,
            listen_url: "http://srv".into(),
            vindex_hash: "h".into(),
            expert_start: 0,
            expert_end: 0,
        })),
    })
    .await
    .unwrap();

    let ack = tokio::time::timeout(Duration::from_secs(1), inbound.next())
        .await
        .expect("router must Ack after the empty-payload skip")
        .expect("stream must yield")
        .expect("ok payload");
    assert!(matches!(ack.payload, Some(RouterPayload::Ack(_))));
    assert_eq!(state.read().status_response().servers.len(), 1);
}

/// Exercises the `Dropping → filled/replicated > 0` success-log path
/// in the join handler: graceful Dropping of an under-replicated
/// shard with a Mode B spare waiting should trigger
/// `try_replicate_from_available` and route an AssignMsg to the
/// spare.
#[tokio::test]
async fn dropping_under_replicated_shard_triggers_replicate_log() {
    let (addr, state) = spawn_router(None).await;
    {
        let mut g = state.write();
        g.set_target_replicas(2);
    }

    // Two serving replicas of layers 0-4.
    let (tx_a, _inbound_a) = join_with_auth(addr, None).await.unwrap();
    tx_a.send(ServerMessage {
        payload: Some(ServerPayload::Announce(AnnounceMsg {
            model_id: "m".into(),
            layer_start: 0,
            layer_end: 4,
            ram_bytes: 1,
            listen_url: "http://srv-a".into(),
            vindex_hash: "h".into(),
            expert_start: 0,
            expert_end: 0,
        })),
    })
    .await
    .unwrap();
    let (tx_b, _inbound_b) = join_with_auth(addr, None).await.unwrap();
    tx_b.send(ServerMessage {
        payload: Some(ServerPayload::Announce(AnnounceMsg {
            model_id: "m".into(),
            layer_start: 0,
            layer_end: 4,
            ram_bytes: 1,
            listen_url: "http://srv-b".into(),
            vindex_hash: "h".into(),
            expert_start: 0,
            expert_end: 0,
        })),
    })
    .await
    .unwrap();

    // Mode B spare.
    let (tx_spare, mut inbound_spare) = join_with_auth(addr, None).await.unwrap();
    tx_spare
        .send(ServerMessage {
            payload: Some(ServerPayload::Available(AvailableMsg {
                ram_bytes: 1_000_000_000,
                disk_bytes: 0,
                store_path: "/tmp".into(),
            })),
        })
        .await
        .unwrap();
    tokio::time::sleep(Duration::from_millis(150)).await;

    // Graceful Dropping from server B — fires the same cleanup as
    // disconnect, but goes through the explicit Dropping arm so the
    // success log on line 180 of service.rs is exercised.
    tx_b.send(ServerMessage {
        payload: Some(ServerPayload::Dropping(
            larql_router_protocol::DroppingMsg {
                model_id: "m".into(),
                layer_start: 0,
                layer_end: 4,
                reason: "test".into(),
            },
        )),
    })
    .await
    .unwrap();

    let assign = tokio::time::timeout(Duration::from_secs(2), inbound_spare.next())
        .await
        .expect("router must respond to Dropping within timeout")
        .expect("stream must yield a message")
        .expect("ok payload");
    assert!(matches!(assign.payload, Some(RouterPayload::Assign(_))));
}
