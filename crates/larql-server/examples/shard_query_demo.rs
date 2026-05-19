//! Exp 53 ShardService end-to-end demo.
//!
//! Spins up a tonic `ShardServiceServer` on a loopback port backed by
//! an `Arc<RwLock<PatchedVindex>>`, then exercises three paths through
//! the generated `ShardServiceClient`:
//!
//!   1. Query an empty vindex → miss with `best_sim = 0.0`.
//!   2. Add a gate patch via the shared `Arc` handle → re-query →
//!      gate match surfaces immediately (proves live patch
//!      propagation, no startup snapshot).
//!   3. Query a different layer / orthogonal vector → miss.
//!
//! Runs in-process — no separate `larql-server` binary needed. The
//! point is to demonstrate the production query path end to end:
//! same proto, same `ShardSource`, same `Arc`-shared `PatchedVindex`.
//!
//! ```bash
//! cargo run --release -p larql-server --example shard_query_demo
//! ```

use std::sync::Arc;
use std::time::{Duration, Instant};

use larql_models::TopKEntry;
use larql_router_protocol::{ShardQuery, ShardServiceClient, ShardServiceServer};
use larql_server::shard_query::{decode_f32_le, encode_f32_le, ShardGrpcService, ShardSource};
use larql_vindex::{FeatureMeta, PatchedVindex, VectorIndex};
use tokio::net::TcpListener;
use tokio::sync::RwLock;
use tokio_stream::wrappers::TcpListenerStream;
use tonic::transport::Server;

const HIDDEN: usize = 4;
const LAYER: u32 = 0;
const TAU: f32 = 0.5;

fn banner(title: &str) {
    println!();
    println!("──── {title} ────");
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("Exp 53 ShardService end-to-end demo");
    println!("hidden={HIDDEN}, layer={LAYER}, tau={TAU}");

    // ── Step 1: empty vindex behind a shared Arc ────────────────────────
    let base = VectorIndex::new(vec![None], vec![None], 1, HIDDEN);
    let patched = Arc::new(RwLock::new(PatchedVindex::new(base)));

    // ── Step 2: tonic ShardServiceServer on an ephemeral TCP port ──────
    let listener = TcpListener::bind("127.0.0.1:0").await?;
    let server_addr = listener.local_addr()?;
    let server_handle = Arc::clone(&patched);
    let svc = ShardGrpcService::new(ShardSource::vindex(server_handle, TAU));
    tokio::spawn(async move {
        Server::builder()
            .add_service(ShardServiceServer::new(svc))
            .serve_with_incoming(TcpListenerStream::new(listener))
            .await
            .expect("tonic serve");
    });
    // Give tonic a tick to start accepting.
    tokio::time::sleep(Duration::from_millis(80)).await;

    let mut client = ShardServiceClient::connect(format!("http://{server_addr}")).await?;
    let query = encode_f32_le(&[1.0_f32, 0.0, 0.0, 0.0]);

    // ── Step 3: empty index → miss ──────────────────────────────────────
    banner("step 1: empty index, expect miss");
    let t0 = Instant::now();
    let resp = client
        .query(ShardQuery {
            layer_id: LAYER,
            k: 1,
            query_vec: query.clone(),
            tau_override: 0.0,
        })
        .await?
        .into_inner();
    let rtt = t0.elapsed();
    println!(
        "  hit={} best_sim={:.3} mlp_out_bytes={} rtt={:?}",
        resp.hit,
        resp.best_sim,
        resp.mlp_out.len(),
        rtt
    );
    assert!(!resp.hit, "empty index must miss");

    // ── Step 4: add a gate patch via the shared Arc handle ─────────────
    banner("step 2: insert_feature via shared Arc, expect propagation");
    {
        let mut guard = patched.write().await;
        guard.insert_feature(
            LAYER as usize,
            0,
            vec![1.0_f32, 0.0, 0.0, 0.0],
            FeatureMeta {
                top_token: "demo".into(),
                top_token_id: 0,
                c_score: 1.0,
                top_k: vec![TopKEntry {
                    token: "demo".into(),
                    token_id: 0,
                    logit: 1.0,
                }],
            },
        );
        println!("  inserted gate-only patch at layer {LAYER}, feature 0");
    }

    let t0 = Instant::now();
    let resp = client
        .query(ShardQuery {
            layer_id: LAYER,
            k: 1,
            query_vec: query.clone(),
            tau_override: 0.0,
        })
        .await?
        .into_inner();
    let rtt = t0.elapsed();
    println!(
        "  hit={} best_sim={:.3} mlp_out_bytes={} rtt={:?}",
        resp.hit,
        resp.best_sim,
        resp.mlp_out.len(),
        rtt
    );
    assert!(
        resp.best_sim >= 0.99,
        "patched gate must surface via gate_knn; got best_sim={}",
        resp.best_sim
    );
    // No down weights wired → still a miss, but the cosine match
    // proves the patch propagated to the shard view through the
    // shared Arc — the whole point of the refactor.
    println!(
        "  → gate match propagated through the shared Arc; down not wired so mlp_out is empty"
    );

    // ── Step 5: orthogonal query → miss ────────────────────────────────
    banner("step 3: orthogonal query, expect miss");
    let t0 = Instant::now();
    let resp = client
        .query(ShardQuery {
            layer_id: LAYER,
            k: 1,
            query_vec: encode_f32_le(&[0.0_f32, 0.0, 1.0, 0.0]),
            tau_override: 0.0,
        })
        .await?
        .into_inner();
    let rtt = t0.elapsed();
    println!(
        "  hit={} best_sim={:.3} mlp_out_bytes={} rtt={:?}",
        resp.hit,
        resp.best_sim,
        resp.mlp_out.len(),
        rtt
    );
    assert!(!resp.hit, "orthogonal query must miss");

    // ── Step 6: simulate a wired down row via a cache-backed source ────
    // The vindex path covers the production wiring but needs real FFN
    // storage to return a non-empty mlp_out. Swap to a Cache-backed
    // source for the round-trip-with-payload demonstration.
    banner("step 4: same wire, but ShardSource::Cache so mlp_out arrives");
    let cache_listener = TcpListener::bind("127.0.0.1:0").await?;
    let cache_addr = cache_listener.local_addr()?;
    let cache = Arc::new(RwLock::new(larql_server::shard_query::ShardCache::new(TAU)));
    {
        let mut guard = cache.write().await;
        guard.seed_from_normed(
            LAYER,
            vec![1.0_f32, 0.0, 0.0, 0.0],
            vec![10.0_f32, 20.0, 30.0, 40.0],
            1,
            HIDDEN,
        )?;
    }
    let cache_svc = ShardGrpcService::from_cache(Arc::clone(&cache));
    tokio::spawn(async move {
        Server::builder()
            .add_service(ShardServiceServer::new(cache_svc))
            .serve_with_incoming(TcpListenerStream::new(cache_listener))
            .await
            .expect("tonic serve");
    });
    tokio::time::sleep(Duration::from_millis(80)).await;
    let mut cache_client = ShardServiceClient::connect(format!("http://{cache_addr}")).await?;
    let t0 = Instant::now();
    let resp = cache_client
        .query(ShardQuery {
            layer_id: LAYER,
            k: 1,
            query_vec: query.clone(),
            tau_override: 0.0,
        })
        .await?
        .into_inner();
    let rtt = t0.elapsed();
    let mlp = decode_f32_le(&resp.mlp_out)?;
    println!(
        "  hit={} best_sim={:.3} mlp_out={:?} rtt={:?}",
        resp.hit, resp.best_sim, mlp, rtt
    );
    assert!(resp.hit && mlp == vec![10.0, 20.0, 30.0, 40.0]);

    println!();
    println!("done — all four scenarios passed");
    Ok(())
}
