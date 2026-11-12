//! The admin client sends the grid key, so the admin commands work against
//! a router that requires one. (The RPC-level refusals are in
//! `test_grid_key_rpcs.rs`.)
//!
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

use larql_router::admin::AdminClient;
use larql_router::grid::service::GridServiceImpl;
use larql_router::grid::GridState;
use larql_router_protocol::{
    grid_service_server::GridServiceServer, AnnounceMsg, GridServiceClient, RouterMessage,
    ServerMessage, ServerPayload,
};
use tonic::transport::Server;

const GRID_KEY: &str = "the-grid-key";
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

#[tokio::test]
async fn the_admin_client_with_the_key_reads_status_and_drains() {
    let (url, state) = spawn_router(Some(GRID_KEY)).await;
    let (server_id, _tx, _inbound) = serving_server(&url, &state).await;
    let admin = AdminClient::new(&url, Some(GRID_KEY));

    let status = admin
        .status()
        .await
        .expect("status with the key")
        .join("\n");
    let gaps = admin.gaps(None).await.expect("gaps with the key");
    let ack = admin
        .drain(&server_id, "test")
        .await
        .expect("drain with the key");

    assert!(status.contains(&format!("Model: {MODEL}")), "{status}");
    assert!(!gaps.is_empty());
    assert!(ack.ok, "{}", ack.message);
}

#[tokio::test]
async fn the_admin_client_without_the_key_is_refused() {
    let (url, _state) = spawn_router(Some(GRID_KEY)).await;
    let admin = AdminClient::new(&url, None);

    let status = admin.status().await.expect_err("status without the key");
    let assigned = admin
        .assign(MODEL, "0-4", None, None, "")
        .await
        .expect_err("assign without the key");

    assert!(status.to_string().contains("invalid grid key"), "{status}");
    assert!(
        assigned.to_string().contains("invalid grid key"),
        "{assigned}"
    );
}

#[tokio::test]
async fn a_router_with_no_key_answers_without_one() {
    // Development mode is unchanged: no key set, no key needed.
    let (url, _state) = spawn_router(None).await;

    let lines = AdminClient::new(&url, None)
        .status()
        .await
        .expect("status on an open router");

    assert!(!lines.is_empty());
}

#[tokio::test]
async fn a_grid_key_that_is_not_a_header_value_is_refused_before_sending() {
    let (url, _state) = spawn_router(None).await;

    let err = AdminClient::new(&url, Some("line\nbreak"))
        .status()
        .await
        .expect_err("a key with a line break cannot be sent");

    assert!(
        err.to_string().contains("not a valid header value"),
        "{err}"
    );
}
