//! A router started with a grid key refuses every grid RPC without it.
//!
//! The key used to gate `Join` only. `Status`, `DrainServer` and
//! `AssignRange` answered anyone who could reach the port: a caller with
//! no key could list every server, drain a serving shard, or point a
//! spare at an origin of their choosing.

use std::sync::Arc;
use std::time::Duration;

use parking_lot::RwLock;
use tokio::sync::mpsc;
use tokio_stream::wrappers::ReceiverStream;

use larql_router::grid::service::GridServiceImpl;
use larql_router::grid::GridState;
use larql_router_protocol::{
    grid_service_server::GridServiceServer, AnnounceMsg, AssignRangeRequest, DrainRequest,
    GridServiceClient, RouterMessage, ServerMessage, ServerPayload, StatusRequest,
};
use tonic::transport::Server;
use tonic::Code;

const GRID_KEY: &str = "the-grid-key";
const WRONG_KEY: &str = "not-the-grid-key";
const MODEL: &str = "m";
const LAYERS: (u32, u32) = (0, 4);
/// Long enough for the listener to accept, and for an announce to register.
const SETTLE: Duration = Duration::from_millis(150);

async fn spawn_router(grid_key: Option<&str>) -> (String, Arc<RwLock<GridState>>) {
    let state = Arc::new(RwLock::new(GridState::default()));
    let svc = GridServiceImpl::new_with_key(state.clone(), grid_key.map(str::to_string))
        .with_metrics(larql_router::metrics::RouterMetrics::new());
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
    tokio::time::sleep(SETTLE).await;
    (format!("http://{addr}"), state)
}

fn with_key<T>(message: T, key: &str) -> tonic::Request<T> {
    let mut request = tonic::Request::new(message);
    request
        .metadata_mut()
        .insert("authorization", format!("Bearer {key}").parse().unwrap());
    request
}

/// Join with the key and announce one serving shard. Returns its server id
/// and the stream halves, which keep the server registered while held.
async fn serving_server(
    url: &str,
    state: &Arc<RwLock<GridState>>,
) -> (
    String,
    mpsc::Sender<ServerMessage>,
    tonic::Streaming<RouterMessage>,
) {
    let mut client = GridServiceClient::connect(url.to_string()).await.unwrap();
    let (tx, rx) = mpsc::channel::<ServerMessage>(8);
    let inbound = client
        .join(with_key(ReceiverStream::new(rx), GRID_KEY))
        .await
        .expect("a join with the key is accepted")
        .into_inner();
    tx.send(ServerMessage {
        payload: Some(ServerPayload::Announce(AnnounceMsg {
            model_id: MODEL.into(),
            layer_start: LAYERS.0,
            layer_end: LAYERS.1,
            listen_url: "http://srv:8080".into(),
            ..Default::default()
        })),
    })
    .await
    .unwrap();
    tokio::time::sleep(SETTLE).await;
    let server_id = state
        .read()
        .servers()
        .map(|(id, _)| id.clone())
        .next()
        .expect("the announced server is registered");
    (server_id, tx, inbound)
}

fn drain(server_id: &str) -> DrainRequest {
    DrainRequest {
        server_id: server_id.to_string(),
        reason: "test".into(),
    }
}

fn assign() -> AssignRangeRequest {
    AssignRangeRequest {
        model_id: MODEL.into(),
        layer_start: LAYERS.0,
        layer_end: LAYERS.1,
        target_server_id: String::new(),
        explicit_origin_url: "http://attacker.example/shard.tar".into(),
        explicit_origin_sha256: String::new(),
    }
}

#[tokio::test]
async fn status_drain_and_assign_are_refused_without_the_key() {
    let (url, state) = spawn_router(Some(GRID_KEY)).await;
    let (server_id, _tx, _inbound) = serving_server(&url, &state).await;
    let mut client = GridServiceClient::connect(url).await.unwrap();

    let status = client.status(StatusRequest {}).await.unwrap_err();
    let drained = client.drain_server(drain(&server_id)).await.unwrap_err();
    let assigned = client.assign_range(assign()).await.unwrap_err();

    assert_eq!(status.code(), Code::Unauthenticated);
    assert_eq!(drained.code(), Code::Unauthenticated);
    assert_eq!(assigned.code(), Code::Unauthenticated);
    assert!(
        state.read().servers().any(|(id, _)| *id == server_id),
        "the server must still be serving"
    );
}

#[tokio::test]
async fn a_wrong_key_is_refused_on_every_rpc() {
    let (url, state) = spawn_router(Some(GRID_KEY)).await;
    let (server_id, _tx, _inbound) = serving_server(&url, &state).await;
    let mut client = GridServiceClient::connect(url).await.unwrap();

    let status = client
        .status(with_key(StatusRequest {}, WRONG_KEY))
        .await
        .unwrap_err();
    let drained = client
        .drain_server(with_key(drain(&server_id), WRONG_KEY))
        .await
        .unwrap_err();
    let assigned = client
        .assign_range(with_key(assign(), WRONG_KEY))
        .await
        .unwrap_err();

    for refused in [status, drained, assigned] {
        assert_eq!(refused.code(), Code::Unauthenticated);
    }
}

#[tokio::test]
async fn a_router_with_no_key_answers_without_one() {
    // Development mode is unchanged: no key set, no key needed.
    let (url, _state) = spawn_router(None).await;
    let mut client = GridServiceClient::connect(url).await.unwrap();

    client
        .status(StatusRequest {})
        .await
        .expect("status on an open router");
}

#[tokio::test]
async fn the_key_opens_every_rpc() {
    // The control for the refusals: the key does not refuse everything.
    let (url, state) = spawn_router(Some(GRID_KEY)).await;
    let (server_id, _tx, _inbound) = serving_server(&url, &state).await;
    let mut client = GridServiceClient::connect(url).await.unwrap();

    let status = client
        .status(with_key(StatusRequest {}, GRID_KEY))
        .await
        .expect("status with the key")
        .into_inner();
    let ack = client
        .drain_server(with_key(drain(&server_id), GRID_KEY))
        .await
        .expect("drain with the key")
        .into_inner();

    assert!(!status.models.is_empty());
    assert!(ack.ok, "{}", ack.message);
}
