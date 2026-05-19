//! End-to-end QUIC transport test for the grid gRPC stream.
//!
//! Wiring:
//!   1. Generate a self-signed cert + pinning fingerprint.
//!   2. Stand up a quinn server endpoint listening on a loopback port,
//!      hand its accept stream to a tonic Server that serves a stub
//!      `GridService` implementation.
//!   3. Build a quinn client endpoint with the fingerprint pinned.
//!   4. Connect → run a unary `Status` RPC and a bidi `Join` exchange
//!      over the QUIC stream pair.
//!
//! Verifies that one full `Announce → Ack` round trip clears over QUIC
//! with the same `GridServiceClient` generated stubs the TCP path uses.

#![cfg(feature = "quic")]

use std::net::SocketAddr;
use std::pin::Pin;
use std::sync::Arc;
use std::time::Duration;

use larql_router_protocol::transport::quic::{
    client_endpoint, connect_grpc_channel, self_signed_tls, server_endpoint, spawn_accept_loop,
    QuicConnectInfo,
};
use larql_router_protocol::{
    grid_service_server::{GridService, GridServiceServer},
    AckMsg, AdminAck, AnnounceMsg, AssignRangeRequest, DrainRequest, GridServiceClient,
    RouterMessage, RouterPayload, ServerMessage, ServerPayload, StatusRequest, StatusResponse,
};
use tokio::sync::{mpsc, RwLock};
use tokio_stream::wrappers::ReceiverStream;
use tokio_stream::StreamExt;
use tonic::transport::Server;
use tonic::{Request, Response, Status, Streaming};

// ── Stub GridService used for the test ───────────────────────────────────────

#[derive(Default)]
struct StubService {
    announces_received: Arc<RwLock<Vec<AnnounceMsg>>>,
}

#[tonic::async_trait]
impl GridService for StubService {
    type JoinStream =
        Pin<Box<dyn tokio_stream::Stream<Item = Result<RouterMessage, Status>> + Send>>;

    async fn join(
        &self,
        request: Request<Streaming<ServerMessage>>,
    ) -> Result<Response<Self::JoinStream>, Status> {
        // Confirm the QUIC connect_info was attached (sanity check that
        // the transport's `Connected` impl wired through).
        let info = request.extensions().get::<QuicConnectInfo>();
        assert!(info.is_some(), "QuicConnectInfo must reach the handler");

        let announces = self.announces_received.clone();
        let mut inbound = request.into_inner();
        let (tx, rx) = mpsc::channel::<Result<RouterMessage, Status>>(4);

        tokio::spawn(async move {
            while let Some(msg) = inbound.next().await {
                let Ok(server_msg) = msg else { break };
                if let Some(ServerPayload::Announce(a)) = server_msg.payload {
                    announces.write().await.push(a);
                    let _ = tx
                        .send(Ok(RouterMessage {
                            payload: Some(RouterPayload::Ack(AckMsg {
                                server_id: "stub-server-1".into(),
                            })),
                        }))
                        .await;
                }
            }
        });

        Ok(Response::new(Box::pin(ReceiverStream::new(rx))))
    }

    async fn status(
        &self,
        _req: Request<StatusRequest>,
    ) -> Result<Response<StatusResponse>, Status> {
        Ok(Response::new(StatusResponse::default()))
    }

    // Phase 5 admin RPCs — the QUIC roundtrip test doesn't exercise them,
    // so stub-acknowledge so this file keeps compiling against the trait.
    async fn drain_server(
        &self,
        _req: Request<DrainRequest>,
    ) -> Result<Response<AdminAck>, Status> {
        Ok(Response::new(AdminAck::default()))
    }

    async fn assign_range(
        &self,
        _req: Request<AssignRangeRequest>,
    ) -> Result<Response<AdminAck>, Status> {
        Ok(Response::new(AdminAck::default()))
    }
}

// ── Test ─────────────────────────────────────────────────────────────────────

async fn spawn_quic_server(
    bind: SocketAddr,
    tls_cert_pem: String,
    tls_key_pem: String,
    state: Arc<StubService>,
) -> std::net::SocketAddr {
    // server_endpoint takes its own owned PEM; we reconstruct the
    // SelfSignedTls fields it needs.
    let tls = larql_router_protocol::transport::quic::SelfSignedTls {
        cert_pem: tls_cert_pem,
        key_pem: tls_key_pem,
        fingerprint: String::new(), // unused server-side
        server_name: "router".into(),
    };
    let endpoint = server_endpoint(bind, &tls).expect("server_endpoint");
    let actual_addr = endpoint.local_addr().expect("local_addr");

    let rx = spawn_accept_loop(endpoint);
    let incoming = ReceiverStream::new(rx);
    let svc = GridServiceServer::new(StubService {
        announces_received: state.announces_received.clone(),
    });
    tokio::spawn(async move {
        if let Err(e) = Server::builder()
            .add_service(svc)
            .serve_with_incoming(incoming)
            .await
        {
            eprintln!("[test] tonic serve_with_incoming exited: {e}");
        }
    });

    // Give tonic a tick to start accepting.
    tokio::time::sleep(Duration::from_millis(100)).await;
    actual_addr
}

#[tokio::test]
async fn quic_round_trip_announce_to_ack() {
    rustls::crypto::ring::default_provider()
        .install_default()
        .ok();

    let tls = self_signed_tls("router").expect("rcgen");
    let stub = Arc::new(StubService::default());
    let server_state = stub.clone();

    let bind: SocketAddr = "127.0.0.1:0".parse().unwrap();
    let server_addr = spawn_quic_server(
        bind,
        tls.cert_pem.clone(),
        tls.key_pem.clone(),
        server_state,
    )
    .await;

    // Client endpoint binds an ephemeral port and pins the server cert by
    // fingerprint.
    let client_bind: SocketAddr = "127.0.0.1:0".parse().unwrap();
    let endpoint =
        client_endpoint(client_bind, Some(tls.fingerprint.clone())).expect("client_endpoint");

    let (_conn, channel) = connect_grpc_channel(&endpoint, server_addr, "router")
        .await
        .expect("connect_grpc_channel");

    let mut client = GridServiceClient::new(channel);

    // Run the Join bidi stream over QUIC.
    let (out_tx, out_rx) = mpsc::channel::<ServerMessage>(4);
    let outbound = ReceiverStream::new(out_rx);
    let response = client.join(outbound).await.expect("join over QUIC");
    let mut inbound = response.into_inner();

    out_tx
        .send(ServerMessage {
            payload: Some(ServerPayload::Announce(AnnounceMsg {
                model_id: "quic-test".into(),
                layer_start: 0,
                layer_end: 4,
                ram_bytes: 1024,
                listen_url: "http://quic-srv:8080".into(),
                vindex_hash: "feedface".into(),
                expert_start: 0,
                expert_end: 0,
            })),
        })
        .await
        .expect("send Announce");

    let ack = tokio::time::timeout(Duration::from_secs(3), inbound.next())
        .await
        .expect("timed out waiting for Ack")
        .expect("stream closed")
        .expect("server error");

    let payload = ack.payload.expect("Ack must have payload");
    match payload {
        RouterPayload::Ack(a) => assert_eq!(a.server_id, "stub-server-1"),
        other => panic!("expected Ack, got {other:?}"),
    }

    // The stub stored the Announce.
    let stored = stub.announces_received.read().await;
    assert_eq!(stored.len(), 1);
    assert_eq!(stored[0].model_id, "quic-test");

    // Tidy.
    drop(out_tx);
}

#[tokio::test]
async fn quic_status_unary_call_succeeds() {
    rustls::crypto::ring::default_provider()
        .install_default()
        .ok();

    let tls = self_signed_tls("router").expect("rcgen");
    let stub = Arc::new(StubService::default());
    let server_state = stub.clone();

    let bind: SocketAddr = "127.0.0.1:0".parse().unwrap();
    let server_addr = spawn_quic_server(
        bind,
        tls.cert_pem.clone(),
        tls.key_pem.clone(),
        server_state,
    )
    .await;

    let client_bind: SocketAddr = "127.0.0.1:0".parse().unwrap();
    let endpoint =
        client_endpoint(client_bind, Some(tls.fingerprint.clone())).expect("client_endpoint");
    let (_conn, channel) = connect_grpc_channel(&endpoint, server_addr, "router")
        .await
        .expect("connect_grpc_channel");

    let mut client = GridServiceClient::new(channel);
    let resp = client.status(StatusRequest::default()).await;

    // The stub returns Default::default(); just confirm we got *a*
    // successful response over QUIC.
    assert!(resp.is_ok(), "Status over QUIC: {:?}", resp.err());
}

/// LAN-only path: client connects with no fingerprint so the
/// AcceptAny verifier is wired into the TLS config and its
/// verify_server_cert / verify_tls13_signature run during the real
/// handshake. Pairs with the unit tests that exercise AcceptAny's
/// pure methods directly.
#[tokio::test]
async fn quic_round_trip_with_skip_verify_uses_accept_any() {
    rustls::crypto::ring::default_provider()
        .install_default()
        .ok();

    let tls = self_signed_tls("router").expect("rcgen");
    let stub = Arc::new(StubService::default());
    let server_state = stub.clone();

    let bind: SocketAddr = "127.0.0.1:0".parse().unwrap();
    let server_addr = spawn_quic_server(
        bind,
        tls.cert_pem.clone(),
        tls.key_pem.clone(),
        server_state,
    )
    .await;

    // `None` → AcceptAny verifier; the cert hashes are never compared.
    let client_bind: SocketAddr = "127.0.0.1:0".parse().unwrap();
    let endpoint = client_endpoint(client_bind, None).expect("client_endpoint");
    let (_conn, channel) = connect_grpc_channel(&endpoint, server_addr, "router")
        .await
        .expect("connect_grpc_channel with skip_verify");

    let mut client = GridServiceClient::new(channel);
    let resp = client.status(StatusRequest::default()).await;
    assert!(
        resp.is_ok(),
        "Status over QUIC with skip-verify: {:?}",
        resp.err()
    );
}
