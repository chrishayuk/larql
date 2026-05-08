//! Latency benchmarks for the attention-service-routes endpoints.
//!
//! Drives the real axum router via tower::ServiceExt::oneshot against
//! the same synthetic `make_test_weights` model the
//! test_attention_validation harness uses. Numbers are not directly
//! comparable to a real-model deployment (vocab=32, hidden=16,
//! num_layers=2 → tiny working set), but they do gate the
//! request/response wire path's overhead and detect regressions in
//! the spawn_blocking + mpsc plumbing.
//!
//! Shapes: prefill at `seq_len ∈ {1, 8, 32, 128}`, decode at
//! `prefilled_len = 32`, snapshot/restore on the same.
//!
//! Run:
//!   cargo bench -p larql-server --bench attention_service

use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::atomic::AtomicU64;
use std::sync::{Arc, OnceLock};

use axum::body::Body;
use axum::http::Request;
use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use larql_inference::engines::test_utils::{
    make_test_weights, make_test_weights_with, SyntheticDims,
};
use larql_inference::ndarray::Array2;
use larql_inference::ModelWeights;
use larql_server::attention_session::AttentionSessionMap;
use larql_server::cache::DescribeCache;
use larql_server::ffn_l2_cache::FfnL2Cache;
use larql_server::routes::single_model_router;
use larql_server::session::SessionManager;
use larql_server::state::{AppState, LoadedModel};
use larql_server::tokenizer_cache::TokenizerCache;
use larql_vindex::{
    ndarray::Array2 as VArray2, ExtractLevel, LayerBands, PatchedVindex, QuantFormat, VectorIndex,
    VindexConfig, VindexLayerInfo,
};
use tower::ServiceExt;

const MODEL_ID: &str = "tiny-bench";

fn build_loaded_model(weights: ModelWeights) -> LoadedModel {
    let hidden = weights.hidden_size;
    let num_layers = weights.num_layers;
    let intermediate = weights.intermediate_size;
    let vocab = weights.vocab_size;

    let gate = VArray2::<f32>::zeros((1, hidden));
    let gate_vectors: Vec<Option<VArray2<f32>>> =
        (0..num_layers).map(|_| Some(gate.clone())).collect();
    let down_meta = vec![None; num_layers];
    let index = VectorIndex::new(gate_vectors, down_meta, num_layers, hidden);
    let patched = PatchedVindex::new(index);

    let tok_json =
        r#"{"version":"1.0","model":{"type":"BPE","vocab":{},"merges":[]},"added_tokens":[]}"#;
    let tokenizer = larql_vindex::tokenizers::Tokenizer::from_bytes(tok_json).unwrap();

    let config = VindexConfig {
        version: 2,
        model: MODEL_ID.into(),
        family: "tiny".into(),
        source: None,
        checksums: None,
        num_layers,
        hidden_size: hidden,
        intermediate_size: intermediate,
        vocab_size: vocab,
        embed_scale: 1.0,
        extract_level: ExtractLevel::Inference,
        dtype: larql_vindex::StorageDtype::default(),
        quant: QuantFormat::None,
        layer_bands: Some(LayerBands {
            syntax: (0, 0),
            knowledge: (0, 0),
            output: (0, num_layers.saturating_sub(1)),
        }),
        layers: (0..num_layers)
            .map(|l| VindexLayerInfo {
                layer: l,
                num_features: 1,
                offset: 0,
                length: (hidden * 4) as u64,
                num_experts: None,
                num_features_per_expert: None,
            })
            .collect(),
        down_top_k: 1,
        has_model_weights: true,
        model_config: None,
        ffn_layout: None,
        fp4: None,
    };

    let lock = OnceLock::new();
    lock.set(std::sync::RwLock::new(weights)).ok();

    LoadedModel {
        id: MODEL_ID.into(),
        path: PathBuf::from("/nonexistent"),
        config,
        patched: tokio::sync::RwLock::new(patched),
        embeddings: Array2::zeros((vocab, hidden)),
        embed_scale: 1.0,
        tokenizer,
        infer_disabled: false,
        ffn_only: false,
        embed_only: false,
        embed_store: None,
        release_mmap_after_request: false,
        weights: lock,
        probe_labels: HashMap::new(),
        ffn_l2_cache: FfnL2Cache::new(1),
        expert_filter: None,
        unit_filter: None,
        moe_remote: None,
        tokenizer_cache: Arc::new(TokenizerCache::new(0, 0)),
    }
}

fn build_state() -> Arc<AppState> {
    build_state_for(make_test_weights())
}

fn build_state_for(weights: ModelWeights) -> Arc<AppState> {
    Arc::new(AppState {
        models: vec![Arc::new(build_loaded_model(weights))],
        started_at: std::time::Instant::now(),
        requests_served: AtomicU64::new(0),
        api_key: None,
        sessions: SessionManager::new(60),
        describe_cache: DescribeCache::new(0),
        attention_sessions: Arc::new(AttentionSessionMap::new(60, 256)),
    })
}

/// Gemma-3-4B-shaped synthetic state, cached because a 4-layer
/// build is ~100 MB f32 and otherwise dominates per-iter cost.
/// `OnceLock` rather than `LazyLock` so the criterion harness
/// loads it lazily on the first `gemma_*` bench only.
fn gemma_4layer_state() -> Arc<AppState> {
    use std::sync::OnceLock;
    static CACHED: OnceLock<Arc<AppState>> = OnceLock::new();
    CACHED
        .get_or_init(|| build_state_for(make_test_weights_with(SyntheticDims::gemma_3_4b_4layer())))
        .clone()
}

fn deterministic_embeddings(seq_len: usize, hidden: usize) -> Vec<Vec<f32>> {
    (0..seq_len)
        .map(|i| {
            (0..hidden)
                .map(|j| ((i * 37 + j * 13) as f32 * 0.001).sin() * 0.1)
                .collect()
        })
        .collect()
}

async fn post_json(
    app: axum::Router,
    path: &str,
    body: serde_json::Value,
) -> axum::http::Response<Body> {
    app.oneshot(
        Request::builder()
            .method("POST")
            .uri(path)
            .header("content-type", "application/json")
            .body(Body::from(serde_json::to_vec(&body).unwrap()))
            .unwrap(),
    )
    .await
    .unwrap()
}

async fn body_value(body: Body) -> serde_json::Value {
    let bytes = axum::body::to_bytes(body, usize::MAX).await.unwrap();
    serde_json::from_slice(&bytes).unwrap_or(serde_json::Value::Null)
}

async fn create_session(state: Arc<AppState>) -> String {
    let app = single_model_router(state);
    let resp = post_json(
        app,
        "/v1/attention/session",
        serde_json::json!({"model_id": MODEL_ID, "kv_format": "fp32"}),
    )
    .await;
    let body = body_value(resp.into_body()).await;
    body["session_id"].as_str().unwrap().to_string()
}

async fn one_prefill(state: Arc<AppState>, sid: &str, embeddings: &[Vec<f32>]) {
    let app = single_model_router(state);
    let resp = post_json(
        app,
        "/v1/attention/prefill",
        serde_json::json!({
            "session_id": sid,
            "token_embeddings": embeddings,
        }),
    )
    .await;
    let _ = body_value(resp.into_body()).await;
}

async fn one_decode(state: Arc<AppState>, sid: &str, embedding: &[f32]) {
    let app = single_model_router(state);
    let resp = post_json(
        app,
        "/v1/attention/decode",
        serde_json::json!({
            "session_id": sid,
            "query_token_embedding": embedding,
        }),
    )
    .await;
    let _ = body_value(resp.into_body()).await;
}

async fn one_snapshot(state: Arc<AppState>, sid: &str) {
    let app = single_model_router(state);
    let resp = post_json(
        app,
        "/v1/kv-cache/snapshot",
        serde_json::json!({"session_id": sid}),
    )
    .await;
    let _ = body_value(resp.into_body()).await;
}

/// `iter_with_setup` would let us amortise state creation, but criterion's
/// async `to_async(&rt)` already drives futures via `rt.block_on` under
/// the hood — calling `rt.block_on` again inside the setup panics with
/// "Cannot start a runtime from within a runtime". The simplest fix is
/// to inline the setup inside the timed `async move { … }` block. The
/// state-build cost is small (no real model load; synthetic weights are
/// already in memory) and steady across iterations, so it lifts every
/// bench by the same constant — the relative shape of the curve is
/// preserved.
fn bench_prefill(c: &mut Criterion) {
    let rt = tokio::runtime::Runtime::new().unwrap();
    let mut group = c.benchmark_group("prefill_e2e_fresh_session");
    let hidden = 16; // make_test_weights default

    for seq_len in [1usize, 8, 32, 128] {
        let embeddings = Arc::new(deterministic_embeddings(seq_len, hidden));
        group.throughput(Throughput::Elements(seq_len as u64));
        group.bench_with_input(
            BenchmarkId::from_parameter(seq_len),
            &embeddings,
            |b, embeddings| {
                let embeddings = embeddings.clone();
                b.to_async(&rt).iter(|| {
                    let embeddings = embeddings.clone();
                    async move {
                        let state = build_state();
                        let sid = create_session(state.clone()).await;
                        one_prefill(state, &sid, &embeddings).await;
                    }
                });
            },
        );
    }
    group.finish();
}

fn bench_decode(c: &mut Criterion) {
    let rt = tokio::runtime::Runtime::new().unwrap();
    let hidden = 16;
    let prefill_len = 32;
    let prefill_embeddings = Arc::new(deterministic_embeddings(prefill_len, hidden));
    let query: Arc<Vec<f32>> = Arc::new(
        (0..hidden)
            .map(|j| ((j * 17) as f32 * 0.01).cos() * 0.1)
            .collect(),
    );

    c.bench_function("decode_e2e_after_prefill_32", |b| {
        let prefill = prefill_embeddings.clone();
        let q = query.clone();
        b.to_async(&rt).iter(|| {
            let prefill = prefill.clone();
            let q = q.clone();
            async move {
                let state = build_state();
                let sid = create_session(state.clone()).await;
                one_prefill(state.clone(), &sid, &prefill).await;
                one_decode(state, &sid, &q).await;
            }
        });
    });
}

fn bench_snapshot(c: &mut Criterion) {
    let rt = tokio::runtime::Runtime::new().unwrap();
    let hidden = 16;
    let prefill_len = 32;
    let prefill_embeddings = Arc::new(deterministic_embeddings(prefill_len, hidden));

    c.bench_function("snapshot_e2e_after_prefill_32", |b| {
        let prefill = prefill_embeddings.clone();
        b.to_async(&rt).iter(|| {
            let prefill = prefill.clone();
            async move {
                let state = build_state();
                let sid = create_session(state.clone()).await;
                one_prefill(state.clone(), &sid, &prefill).await;
                one_snapshot(state, &sid).await;
            }
        });
    });
}

/// Gemma-3-4B-shaped (hidden=2560, num_q=8, num_kv=4, head_dim=320,
/// intermediate=10240) but with only 4 layers so build/iter time
/// stays bench-friendly. Every iteration starts from a fresh
/// session against the cached AppState — this gates whether the
/// per-call overhead (session create + spawn_blocking + serde
/// round-trip) scales sanely with realistic dims.
fn bench_prefill_gemma_shaped(c: &mut Criterion) {
    let rt = tokio::runtime::Runtime::new().unwrap();
    let mut group = c.benchmark_group("prefill_gemma_4layer");
    let hidden = SyntheticDims::gemma_3_4b_4layer().hidden;
    // Sample-size and warm-up are tightened — Gemma-shaped is
    // ~10× the work of the tiny variant, and we don't need
    // criterion's full statistical sweep for a regression gate.
    group.sample_size(20);
    group.measurement_time(std::time::Duration::from_secs(8));

    for seq_len in [1usize, 8, 32] {
        let embeddings = Arc::new(deterministic_embeddings(seq_len, hidden));
        group.throughput(Throughput::Elements(seq_len as u64));
        group.bench_with_input(
            BenchmarkId::from_parameter(seq_len),
            &embeddings,
            |b, embeddings| {
                let embeddings = embeddings.clone();
                let state = gemma_4layer_state();
                b.to_async(&rt).iter(|| {
                    let embeddings = embeddings.clone();
                    let state = state.clone();
                    async move {
                        let sid = create_session(state.clone()).await;
                        one_prefill(state, &sid, &embeddings).await;
                    }
                });
            },
        );
    }
    group.finish();
}

criterion_group!(
    benches,
    bench_prefill,
    bench_decode,
    bench_snapshot,
    bench_prefill_gemma_shaped,
);
criterion_main!(benches);
