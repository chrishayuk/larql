//! End-to-end round-trip test for the Exp 53 `ShardService`.
//!
//! Unit tests inside `shard_query.rs` exercise the handler trait method
//! directly. This file spins up the tonic server on an ephemeral TCP
//! port, drives one `Query` through the generated client, and asserts
//! the response decodes correctly — proving the proto + service
//! registration compile and serve correctly end-to-end.

use std::sync::Arc;

use larql_router_protocol::{ShardQuery, ShardServiceClient, ShardServiceServer};
use larql_server::shard_query::{encode_f32_le, ShardCache, ShardGrpcService};
use tokio::net::TcpListener;
use tokio::sync::RwLock;
use tokio_stream::wrappers::TcpListenerStream;
use tonic::transport::Server;

async fn spawn_server() -> (std::net::SocketAddr, Arc<RwLock<ShardCache>>) {
    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap();
    let cache = Arc::new(RwLock::new(ShardCache::new(0.97)));
    {
        let mut guard = cache.write().await;
        guard
            .seed_from_normed(
                26,
                vec![1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0],
                vec![10.0, 20.0, 30.0, 40.0, -1.0, -2.0, -3.0, -4.0],
                2,
                4,
            )
            .unwrap();
    }
    let svc = ShardGrpcService::from_cache(Arc::clone(&cache));
    let server_cache = Arc::clone(&cache);
    tokio::spawn(async move {
        Server::builder()
            .add_service(ShardServiceServer::new(svc))
            .serve_with_incoming(TcpListenerStream::new(listener))
            .await
            .unwrap();
        // Keep cache reference alive until the server task ends (it
        // never does in these tests — the runtime is dropped first).
        drop(server_cache);
    });
    // Give tonic a tick to start accepting before we dial.
    tokio::time::sleep(std::time::Duration::from_millis(50)).await;
    (addr, cache)
}

#[tokio::test]
async fn shard_query_round_trip_hit() {
    let (addr, _cache) = spawn_server().await;
    let mut client = ShardServiceClient::connect(format!("http://{addr}"))
        .await
        .expect("connect to shard server");

    let resp = client
        .query(ShardQuery {
            layer_id: 26,
            k: 1,
            query_vec: encode_f32_le(&[1.0, 0.0, 0.0, 0.0]),
            tau_override: 0.0,
        })
        .await
        .expect("query rpc")
        .into_inner();

    assert!(resp.hit, "exact match should hit");
    assert!((resp.best_sim - 1.0).abs() < 1e-6);
    // Decode the mlp_out payload — should match the seeded row 0 output.
    let mlp = larql_server::shard_query::decode_f32_le(&resp.mlp_out).unwrap();
    assert_eq!(mlp, vec![10.0, 20.0, 30.0, 40.0]);
}

#[tokio::test]
async fn shard_query_round_trip_miss_below_tau() {
    let (addr, _cache) = spawn_server().await;
    let mut client = ShardServiceClient::connect(format!("http://{addr}"))
        .await
        .expect("connect");

    // Orthogonal-to-everything query → best_sim ≈ 0 → miss.
    let resp = client
        .query(ShardQuery {
            layer_id: 26,
            k: 1,
            query_vec: encode_f32_le(&[0.0, 0.0, 1.0, 0.0]),
            tau_override: 0.0,
        })
        .await
        .expect("query rpc")
        .into_inner();

    assert!(!resp.hit);
    assert!(resp.mlp_out.is_empty());
    assert!(resp.best_sim < 0.97);
}

#[tokio::test]
async fn shard_query_round_trip_unknown_layer() {
    let (addr, _cache) = spawn_server().await;
    let mut client = ShardServiceClient::connect(format!("http://{addr}"))
        .await
        .expect("connect");
    let resp = client
        .query(ShardQuery {
            layer_id: 99,
            k: 1,
            query_vec: encode_f32_le(&[1.0, 0.0, 0.0, 0.0]),
            tau_override: 0.0,
        })
        .await
        .expect("query rpc")
        .into_inner();
    assert!(!resp.hit);
}

/// Regression test for the shared-`Arc<RwLock<PatchedVindex>>` refactor:
/// a vindex patch added through one `Arc` handle is visible to a shard
/// service holding another handle to the *same* Arc. Proves the
/// follow-up from Exp 53 (live patch propagation, no startup snapshot)
/// actually works end to end.
#[tokio::test]
async fn shard_view_sees_patches_added_through_a_shared_arc_handle() {
    use larql_models::TopKEntry;
    use larql_router_protocol::{ShardServiceClient, ShardServiceServer};
    use larql_server::shard_query::{decode_f32_le, encode_f32_le, ShardGrpcService, ShardSource};
    use larql_vindex::{FeatureMeta, PatchedVindex, VectorIndex};

    // Two handles to the same Arc: `patched_writer` will be mutated;
    // `patched_for_source` is what the gRPC service reads from.
    let base = VectorIndex::new(vec![None], vec![None], 1, 4);
    let patched_writer = Arc::new(RwLock::new(PatchedVindex::new(base)));
    let patched_for_source = Arc::clone(&patched_writer);

    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap();
    let svc = ShardGrpcService::new(ShardSource::vindex(patched_for_source, 0.5));
    tokio::spawn(async move {
        tonic::transport::Server::builder()
            .add_service(ShardServiceServer::new(svc))
            .serve_with_incoming(TcpListenerStream::new(listener))
            .await
            .unwrap();
    });
    tokio::time::sleep(std::time::Duration::from_millis(50)).await;
    let mut client = ShardServiceClient::connect(format!("http://{addr}"))
        .await
        .expect("connect");

    // Before the patch: every query misses (empty index).
    let before = client
        .query(ShardQuery {
            layer_id: 0,
            k: 1,
            query_vec: encode_f32_le(&[1.0, 0.0, 0.0, 0.0]),
            tau_override: 0.0,
        })
        .await
        .expect("query")
        .into_inner();
    assert!(!before.hit, "no patches yet → miss");

    // Add a gate patch *through the writer handle* — the shard
    // service should see it on the next call because both handles
    // share the same `Arc<RwLock<…>>`.
    {
        let mut guard = patched_writer.write().await;
        guard.insert_feature(
            0,
            0,
            vec![1.0, 0.0, 0.0, 0.0],
            FeatureMeta {
                top_token: "f0".into(),
                top_token_id: 0,
                c_score: 1.0,
                top_k: vec![TopKEntry {
                    token: "f0".into(),
                    token_id: 0,
                    logit: 1.0,
                }],
            },
        );
        // No down weights wired — that's fine for this regression
        // test. We're proving the patch propagates to gate_knn; the
        // down lookup will fall through, but `best_sim` reflects the
        // newly-patched gate match.
    }

    let after = client
        .query(ShardQuery {
            layer_id: 0,
            k: 1,
            query_vec: encode_f32_le(&[1.0, 0.0, 0.0, 0.0]),
            tau_override: 0.0,
        })
        .await
        .expect("query")
        .into_inner();
    assert!(
        !after.hit,
        "down not patched → still a miss, but best_sim must show the gate hit"
    );
    assert!(
        after.best_sim >= 0.99,
        "patched gate vector must surface via gate_knn; got best_sim={}",
        after.best_sim
    );
    // Decode-side sanity: mlp_out is empty on a miss.
    assert!(decode_f32_le(&after.mlp_out).unwrap().is_empty());
}
