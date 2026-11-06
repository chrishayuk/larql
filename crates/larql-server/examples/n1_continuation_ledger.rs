//! The N1 continuation A/B, frozen so it can be re-run unchanged.
//!
//! Answers one question — *what does the KV continuation cache save on
//! a real chained conversation* — by running the **same** four-turn
//! `/v1/responses` chain twice against the same container: once with
//! the cache enabled and once with it disabled. Both arms run
//! in-process through the real router, so there are no ports to manage
//! and no second server to keep honest.
//!
//! # Why this is an example and not a scratch script
//!
//! Its value is comparability across rungs, and that survives only if
//! the workload does not drift. The turns, the output budget and the
//! protocol below are the measurement — **do not tune them** to make a
//! future number look better. A new question deserves a new example.
//!
//! # The ledger so far
//!
//! Four-turn chain, cache on vs off, min of two chains, on battery:
//!
//! ```text
//! state                      cache off   cache on   speedup   reused
//! pre-2B   gemma-2-2b           41.75 s    41.14 s     1.5%    noise
//! post-2B  gpt-oss-20b          45.17 s    18.52 s    2.44x   254 tok
//! post-2B  granite-4.1-3b       15.69 s     7.74 s    2.03x   146 tok
//! post-2C  (to measure)
//! ```
//!
//! (The post-2B rows are this example's own output. An ad-hoc script
//! measured 2.40x and 2.07x on the same models minutes earlier; the
//! agreement within ~2% is what says the frozen workload reproduces
//! what it was written from. `cached_tokens` matched exactly.)
//!
//! ```text
//! ```
//!
//! The per-turn shape is stronger evidence than the total, and it is
//! model-independent — without the cache, turn time grows with history
//! because the whole conversation is re-prefilled; with it, turn time
//! is flat because only the new turn is:
//!
//! ```text
//! cache OFF   4.33 -> 8.81 -> 13.49 -> 18.38 s     linear
//! cache ON    4.31 -> 4.96 ->  4.57 ->  4.92 s     flat
//! ```
//!
//! Read the post-2C row against this one carefully: making prefill
//! faster should *reduce* the absolute seconds N1 saves while leaving
//! the shape intact. If the shape survives, N1 is structurally
//! valuable rather than a workaround for slow prefill.
//!
//! Run:
//!
//! ```sh
//! cargo run --release -p larql-server --example n1_continuation_ledger \
//!     -- path/to/model.vindex3
//! ```
//!
//! The container must carry its own `tokenizer.json` (V3-SERVE-4).

use std::sync::atomic::AtomicU64;
use std::sync::Arc;
use std::time::Instant;

use axum::body::Body;
use axum::http::{header, Request};
use larql_server::bootstrap::{load_artifact, LoadVindexOptions, LoadedArtifact};
use larql_server::cache::DescribeCache;
use larql_server::session::SessionManager;
use larql_server::state::AppState;
use serde_json::Value;
use tower::ServiceExt;

/// The workload. Four turns of a growing conversation — the shape N1
/// exists to serve. **Frozen**: changing these invalidates every row of
/// the ledger above.
const TURNS: [&str; 4] = [
    "Explain in one sentence why distributed consensus protocols need a quorum.",
    "And why does an even number of nodes not help?",
    "What happens during a network partition?",
    "Summarise the trade-off in one line.",
];

/// Short enough that decode does not swamp prefill, which is what the
/// cache actually affects.
const OUT_TOKENS: usize = 8;

/// Chains per arm; the ledger reports the fastest, since the floor is
/// the least noisy estimator of a wall-clock cost.
const CHAINS_PER_ARM: usize = 2;

/// Cache capacity for the enabled arm.
const CACHE_ENTRIES: usize = 4;

struct Turn {
    seconds: f64,
    input_tokens: u64,
    cached_tokens: u64,
}

fn state_for(container: &str, kv_entries: usize) -> Arc<AppState> {
    let artifact = load_artifact(container, LoadVindexOptions::default())
        .unwrap_or_else(|e| panic!("open {container}: {e}"));
    let v3 = match artifact {
        LoadedArtifact::V3(m) => Arc::new(*m),
        LoadedArtifact::V2(_) => panic!("this ledger measures the V3 continuation cache"),
    };
    Arc::new(AppState {
        model_set: std::sync::RwLock::new(larql_server::state::ModelSet {
            models: Vec::new(),
            v3_models: vec![v3],
            aliases: Default::default(),
        }),
        router_topology: larql_server::state::RouterTopology::SingleModel,
        lifecycle: std::sync::Mutex::new(larql_server::state::LifecycleState::Idle),
        started_at: Instant::now(),
        requests_served: AtomicU64::new(0),
        api_key: None,
        sessions: SessionManager::new(3600),
        describe_cache: DescribeCache::new(0),
        infer_timeout: std::time::Duration::from_secs(3600),
        patch_sources: Default::default(),
        responses: larql_server::response_store::ResponseStore::new(),
        v3_kv: larql_server::response_kv::ResponseKvCache::new(
            kv_entries,
            larql_server::response_kv::DEFAULT_TTL_SECS,
        ),
        runtime: Arc::new(larql_server::runtime_stats::RuntimeRecorder::new()),
    })
}

async fn respond(state: &Arc<AppState>, body: Value) -> (f64, Value) {
    let app = larql_server::routes::single_model_router(Arc::clone(state));
    let req = Request::builder()
        .method("POST")
        .uri("/v1/responses")
        .header(header::CONTENT_TYPE, "application/json")
        .body(Body::from(serde_json::to_vec(&body).unwrap()))
        .unwrap();
    let t0 = Instant::now();
    let resp = app.oneshot(req).await.unwrap();
    let status = resp.status();
    let bytes = axum::body::to_bytes(resp.into_body(), usize::MAX)
        .await
        .unwrap();
    let elapsed = t0.elapsed().as_secs_f64();
    assert!(status.is_success(), "turn failed: {status} {bytes:?}");
    (elapsed, serde_json::from_slice(&bytes).unwrap())
}

/// One whole conversation. Returns each turn's cost and reuse.
async fn chain(state: &Arc<AppState>) -> Vec<Turn> {
    let mut previous: Option<String> = None;
    let mut turns = Vec::with_capacity(TURNS.len());
    for text in TURNS {
        let mut body = serde_json::json!({
            "input": text,
            "max_output_tokens": OUT_TOKENS,
            "temperature": 0.0,
        });
        if let Some(id) = &previous {
            body["previous_response_id"] = Value::String(id.clone());
        }
        let (seconds, envelope) = respond(state, body).await;
        previous = Some(envelope["id"].as_str().unwrap().to_string());
        let usage = &envelope["usage"];
        turns.push(Turn {
            seconds,
            input_tokens: usage["input_tokens"].as_u64().unwrap(),
            cached_tokens: usage["input_tokens_details"]["cached_tokens"]
                .as_u64()
                .unwrap(),
        });
    }
    turns
}

async fn arm(label: &str, container: &str, kv_entries: usize) -> f64 {
    let state = state_for(container, kv_entries);
    // Warm the mmap pages so the first measured turn is not paying for
    // first touch; the prepared image is already built by binding.
    respond(
        &state,
        serde_json::json!({"input": "warm", "max_output_tokens": 2, "temperature": 0.0}),
    )
    .await;

    let mut best: Option<Vec<Turn>> = None;
    for _ in 0..CHAINS_PER_ARM {
        let turns = chain(&state).await;
        let total: f64 = turns.iter().map(|t| t.seconds).sum();
        if best
            .as_ref()
            .is_none_or(|b| total < b.iter().map(|t| t.seconds).sum::<f64>())
        {
            best = Some(turns);
        }
    }
    let turns = best.expect("at least one chain");

    println!("  {label}");
    for (i, t) in turns.iter().enumerate() {
        println!(
            "    turn {}: {:7.2}s  input={:4}  cached={:4}",
            i + 1,
            t.seconds,
            t.input_tokens,
            t.cached_tokens
        );
    }
    let total: f64 = turns.iter().map(|t| t.seconds).sum();
    let reused: u64 = turns.iter().map(|t| t.cached_tokens).sum();
    println!("    total  : {total:7.2}s   reused={reused} tokens");
    total
}

#[tokio::main]
async fn main() {
    let container = std::env::args()
        .nth(1)
        .expect("usage: n1_continuation_ledger <container.vindex3>");

    println!("N1 continuation ledger — {container}\n");
    let on = arm(
        &format!("cache ON ({CACHE_ENTRIES} entries)"),
        &container,
        CACHE_ENTRIES,
    )
    .await;
    let off = arm("cache OFF", &container, 0).await;

    println!("\n── ledger ───────────────────────────────────────────────");
    println!("  cache off  {off:7.2}s");
    println!("  cache on   {on:7.2}s");
    println!("  saving     {:7.2}s   ({:.2}x)", off - on, off / on);
    println!(
        "\n  Read the per-turn column, not just the total: without the cache\n  \
         turn time grows with history; with it, it stays near new-turn cost."
    );
}
