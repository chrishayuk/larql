//! Shared HTTP test infrastructure for larql-server integration tests.
//!
//! Uses axum's tower::ServiceExt::oneshot pattern — requests are dispatched
//! in-process to the full router with no network socket. Every test builds a
//! synthetic in-memory VectorIndex (1 layer, 3 features, hidden=4).
//!
//! For tests that need a real `LoadedModel.weights`-populating model on
//! disk (`full_output=true` paths into walk_ffn / explain / generation),
//! see [`synthetic_vindex`] + [`model_with_real_weights`].

#![allow(dead_code, unused_imports)]

pub mod synthetic_q4k_vindex;
pub mod synthetic_vindex;

use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::atomic::AtomicU64;
use std::sync::Arc;

use axum::body::Body;
use axum::http::{Request, StatusCode};
use larql_server::cache::DescribeCache;
use larql_server::ffn_l2_cache::FfnL2Cache;
use larql_server::session::SessionManager;
use larql_server::state::{AppState, LoadedModel};
use larql_vindex::{
    ndarray::Array2, ExtractLevel, FeatureMeta, LayerBands, PatchedVindex, QuantFormat,
    VectorIndex, VindexConfig, VindexLayerInfo,
};
use tower::ServiceExt;

// ══════════════════════════════════════════════════════════════
// Index / config helpers
// ══════════════════════════════════════════════════════════════

pub fn make_feature(token: &str, id: u32, score: f32) -> FeatureMeta {
    FeatureMeta {
        top_token: token.to_string(),
        top_token_id: id,
        c_score: score,
        top_k: vec![
            larql_models::TopKEntry {
                token: token.to_string(),
                token_id: id,
                logit: score,
            },
            larql_models::TopKEntry {
                token: "also".into(),
                token_id: id + 1,
                logit: score * 0.5,
            },
        ],
    }
}

pub fn test_index() -> VectorIndex {
    let hidden = 4;
    let mut gate = Array2::<f32>::zeros((3, hidden));
    gate[[0, 0]] = 1.0; // Paris  → dim 0
    gate[[1, 1]] = 1.0; // French → dim 1
    gate[[2, 2]] = 1.0; // Europe → dim 2

    let meta: Vec<Option<FeatureMeta>> = vec![
        Some(make_feature("Paris", 100, 0.95)),
        Some(make_feature("French", 101, 0.88)),
        Some(make_feature("Europe", 102, 0.75)),
    ];

    VectorIndex::new(vec![Some(gate)], vec![Some(meta)], 1, hidden)
}

pub fn test_config() -> VindexConfig {
    VindexConfig {
        version: 2,
        model: "test/model-4".to_string(),
        family: "test".to_string(),
        source: None,
        checksums: None,
        num_layers: 1,
        hidden_size: 4,
        intermediate_size: 12,
        vocab_size: 8,
        embed_scale: 1.0,
        extract_level: ExtractLevel::Browse,
        dtype: larql_vindex::StorageDtype::default(),
        quant: QuantFormat::None,
        layer_bands: Some(LayerBands {
            syntax: (0, 0),
            knowledge: (0, 0),
            output: (0, 0),
        }),
        layers: vec![VindexLayerInfo {
            layer: 0,
            num_features: 3,
            offset: 0,
            length: 48,
            num_experts: None,
            num_features_per_expert: None,
        }],
        down_top_k: 5,
        has_model_weights: false,
        model_config: None,
        fp4: None,
        ffn_layout: None,
    }
}

pub fn empty_tokenizer() -> larql_vindex::tokenizers::Tokenizer {
    let json =
        r#"{"version":"1.0","model":{"type":"BPE","vocab":{},"merges":[]},"added_tokens":[]}"#;
    larql_vindex::tokenizers::Tokenizer::from_bytes(json).unwrap()
}

/// WordLevel tokenizer: France→0, Germany→1, capital→2, language→3, UNK→7
/// Used by tests that need real tokenization without a full model file.
pub fn functional_tokenizer() -> larql_vindex::tokenizers::Tokenizer {
    let json = r#"{"version":"1.0","truncation":null,"padding":null,"added_tokens":[],"normalizer":null,"pre_tokenizer":null,"post_processor":null,"decoder":null,"model":{"type":"WordLevel","vocab":{"France":0,"Germany":1,"capital":2,"language":3,"UNK":7},"unk_token":"UNK"}}"#;
    larql_vindex::tokenizers::Tokenizer::from_bytes(json.as_bytes()).unwrap()
}

/// Model using the functional tokenizer.
/// Embeddings: row 0=[1,0,0,0] → matches gate feature 0 ("Paris")
///             row 1=[0,1,0,0] → matches gate feature 1 ("French")
pub fn model_functional(id: &str) -> Arc<LoadedModel> {
    Arc::new(LoadedModel {
        id: id.to_string(),
        path: std::path::PathBuf::from("/nonexistent"),
        config: test_config(),
        patched: std::sync::Arc::new(tokio::sync::RwLock::new(PatchedVindex::new(test_index()))),
        embeddings: {
            let mut e = Array2::<f32>::zeros((8, 4));
            e[[0, 0]] = 1.0;
            e[[1, 1]] = 1.0;
            e[[2, 2]] = 1.0;
            e[[3, 3]] = 1.0;
            e
        },
        embed_scale: 1.0,
        tokenizer: functional_tokenizer(),
        infer_disabled: true,
        ffn_only: false,
        embed_only: false,
        embed_store: None,
        release_mmap_after_request: false,
        weights: std::sync::OnceLock::new(),
        probe_labels: std::collections::HashMap::new(),
        ffn_l2_cache: larql_server::ffn_l2_cache::FfnL2Cache::new(1),
        layer_latency_tracker: std::sync::Arc::new(
            larql_server::metrics::LayerLatencyTracker::new(),
        ),
        requests_in_flight: std::sync::Arc::new(std::sync::atomic::AtomicU32::new(0)),
        requests_total: std::sync::Arc::new(std::sync::atomic::AtomicU64::new(0)),
        expert_filter: None,
        unit_filter: None,
        moe_remote: None,
        #[cfg(all(feature = "metal-experts", target_os = "macos"))]
        metal_backend: std::sync::OnceLock::new(),
        #[cfg(all(feature = "metal-experts", target_os = "macos"))]
        moe_scratches: std::sync::Mutex::new(std::collections::HashMap::new()),
        #[cfg(all(feature = "metal-experts", target_os = "macos"))]
        metal_ffn_layer_bufs: std::sync::OnceLock::new(),
    })
}

/// ModelBuilder with optional infer_disabled override (defaults true).
pub fn model_infer_enabled(id: &str) -> Arc<LoadedModel> {
    Arc::new(LoadedModel {
        id: id.to_string(),
        path: PathBuf::from("/nonexistent"),
        config: test_config(),
        patched: std::sync::Arc::new(tokio::sync::RwLock::new(PatchedVindex::new(test_index()))),
        embeddings: {
            let mut e = Array2::<f32>::zeros((8, 4));
            e[[0, 0]] = 1.0;
            e[[1, 1]] = 1.0;
            e[[2, 2]] = 1.0;
            e[[3, 3]] = 1.0;
            e
        },
        embed_scale: 1.0,
        tokenizer: empty_tokenizer(),
        infer_disabled: false,
        ffn_only: false,
        embed_only: false,
        embed_store: None,
        release_mmap_after_request: false,
        weights: std::sync::OnceLock::new(),
        probe_labels: std::collections::HashMap::new(),
        ffn_l2_cache: larql_server::ffn_l2_cache::FfnL2Cache::new(1),
        layer_latency_tracker: std::sync::Arc::new(
            larql_server::metrics::LayerLatencyTracker::new(),
        ),
        requests_in_flight: std::sync::Arc::new(std::sync::atomic::AtomicU32::new(0)),
        requests_total: std::sync::Arc::new(std::sync::atomic::AtomicU64::new(0)),
        expert_filter: None,
        unit_filter: None,
        moe_remote: None,
        #[cfg(all(feature = "metal-experts", target_os = "macos"))]
        metal_backend: std::sync::OnceLock::new(),
        #[cfg(all(feature = "metal-experts", target_os = "macos"))]
        moe_scratches: std::sync::Mutex::new(std::collections::HashMap::new()),
        #[cfg(all(feature = "metal-experts", target_os = "macos"))]
        metal_ffn_layer_bufs: std::sync::OnceLock::new(),
    })
}

// ══════════════════════════════════════════════════════════════
// ModelBuilder
// ══════════════════════════════════════════════════════════════

pub struct ModelBuilder {
    pub id: String,
    pub ffn_only: bool,
    pub embed_only: bool,
    pub infer_disabled: bool,
    pub probe_labels: HashMap<(usize, usize), String>,
    pub config: VindexConfig,
}

impl ModelBuilder {
    pub fn new(id: &str) -> Self {
        Self {
            id: id.to_string(),
            ffn_only: false,
            embed_only: false,
            infer_disabled: true,
            probe_labels: HashMap::new(),
            config: test_config(),
        }
    }
    pub fn ffn_only(mut self) -> Self {
        self.ffn_only = true;
        self
    }
    pub fn embed_only(mut self) -> Self {
        self.embed_only = true;
        self
    }
    pub fn infer_disabled(mut self, v: bool) -> Self {
        self.infer_disabled = v;
        self
    }
    pub fn with_labels(mut self, labels: HashMap<(usize, usize), String>) -> Self {
        self.probe_labels = labels;
        self
    }
    pub fn build(self) -> Arc<LoadedModel> {
        Arc::new(LoadedModel {
            id: self.id,
            path: PathBuf::from("/nonexistent"),
            config: self.config,
            patched: std::sync::Arc::new(tokio::sync::RwLock::new(
                PatchedVindex::new(test_index()),
            )),
            embeddings: {
                let mut e = Array2::<f32>::zeros((8, 4));
                e[[0, 0]] = 1.0;
                e[[1, 1]] = 1.0;
                e[[2, 2]] = 1.0;
                e[[3, 3]] = 1.0;
                e
            },
            embed_scale: 1.0,
            tokenizer: empty_tokenizer(),
            infer_disabled: self.infer_disabled,
            ffn_only: self.ffn_only,
            embed_only: self.embed_only,
            embed_store: None,
            release_mmap_after_request: false,
            weights: std::sync::OnceLock::new(),
            probe_labels: self.probe_labels,
            ffn_l2_cache: FfnL2Cache::new(1),
            layer_latency_tracker: std::sync::Arc::new(
                larql_server::metrics::LayerLatencyTracker::new(),
            ),
            requests_in_flight: std::sync::Arc::new(std::sync::atomic::AtomicU32::new(0)),
            requests_total: std::sync::Arc::new(std::sync::atomic::AtomicU64::new(0)),
            expert_filter: None,
            unit_filter: None,
            moe_remote: None,
            #[cfg(all(feature = "metal-experts", target_os = "macos"))]
            metal_backend: std::sync::OnceLock::new(),
            #[cfg(all(feature = "metal-experts", target_os = "macos"))]
            moe_scratches: std::sync::Mutex::new(std::collections::HashMap::new()),
            #[cfg(all(feature = "metal-experts", target_os = "macos"))]
            metal_ffn_layer_bufs: std::sync::OnceLock::new(),
        })
    }
}

pub fn model(id: &str) -> Arc<LoadedModel> {
    ModelBuilder::new(id).build()
}

/// Build a `LoadedModel` backed by a real synthetic vindex on disk.
/// `LoadedModel.path` points at the tempdir and
/// `LoadedModel.get_or_load_weights()` will mmap the synthetic
/// tensors when the route handler calls it — so `full_output=true`
/// paths (walk_ffn, explain, generation, lm_head) all execute their
/// real codepaths instead of bailing on the empty `OnceLock`.
///
/// The returned `SyntheticVindex` owns the tempdir; the test must
/// keep it alive for the duration of the test (drop after assertions).
pub fn model_with_real_weights(id: &str) -> (Arc<LoadedModel>, synthetic_vindex::SyntheticVindex) {
    model_with_real_weights_and_labels(id, HashMap::new())
}

/// Like [`model_with_real_weights`] but seeds `LoadedModel.probe_labels`
/// so the `relations_only` branches in `routes/explain.rs` actually fire.
pub fn model_with_real_weights_and_labels(
    id: &str,
    probe_labels: HashMap<(usize, usize), String>,
) -> (Arc<LoadedModel>, synthetic_vindex::SyntheticVindex) {
    use larql_vindex::{SilentLoadCallbacks, VectorIndex};

    let fixture = synthetic_vindex::build();
    let mut cb = SilentLoadCallbacks;
    let index = VectorIndex::load_vindex(&fixture.dir, &mut cb).expect("load synthetic vindex");
    let config = larql_vindex::load_vindex_config(&fixture.dir).expect("load vindex config");

    // Reload the same tokenizer the fixture wrote to disk so the
    // route handler's `model.tokenizer.encode(prompt)` produces real
    // token ids (and therefore exercises the per-token branches in
    // walk_ffn / explain / predict). Without this, an empty BPE
    // tokenizer would encode every prompt to 0 tokens and the meat
    // of the route handlers stays uncovered.
    let tok_bytes = std::fs::read(fixture.dir.join("tokenizer.json")).expect("read tokenizer.json");
    let fixture_tokenizer = larql_vindex::tokenizers::Tokenizer::from_bytes(&tok_bytes)
        .expect("parse fixture tokenizer");

    // Embed matrix — copy what `build_vindex` wrote so the embed
    // lookup hits a non-zero row per token. We reload it from the
    // fixture so the test sees the exact same data the loader would.
    let embed_path = fixture.dir.join("embeddings.bin");
    let embed_bytes = std::fs::read(&embed_path).expect("read embeddings.bin");
    let n_floats = embed_bytes.len() / std::mem::size_of::<f32>();
    let mut embed_floats = Vec::with_capacity(n_floats);
    for chunk in embed_bytes.chunks_exact(4) {
        embed_floats.push(f32::from_le_bytes(chunk.try_into().unwrap()));
    }
    let embeddings = Array2::from_shape_vec((fixture.vocab_size, fixture.hidden), embed_floats)
        .expect("embeddings shape");

    let model = Arc::new(LoadedModel {
        id: id.to_string(),
        path: fixture.dir.clone(),
        config,
        patched: std::sync::Arc::new(tokio::sync::RwLock::new(PatchedVindex::new(index))),
        embeddings,
        embed_scale: 1.0,
        tokenizer: fixture_tokenizer,
        infer_disabled: false,
        ffn_only: false,
        embed_only: false,
        embed_store: None,
        release_mmap_after_request: false,
        weights: std::sync::OnceLock::new(),
        probe_labels,
        ffn_l2_cache: FfnL2Cache::new(1),
        layer_latency_tracker: std::sync::Arc::new(
            larql_server::metrics::LayerLatencyTracker::new(),
        ),
        requests_in_flight: std::sync::Arc::new(std::sync::atomic::AtomicU32::new(0)),
        requests_total: std::sync::Arc::new(std::sync::atomic::AtomicU64::new(0)),
        expert_filter: None,
        unit_filter: None,
        moe_remote: None,
        #[cfg(all(feature = "metal-experts", target_os = "macos"))]
        metal_backend: std::sync::OnceLock::new(),
        #[cfg(all(feature = "metal-experts", target_os = "macos"))]
        moe_scratches: std::sync::Mutex::new(std::collections::HashMap::new()),
        #[cfg(all(feature = "metal-experts", target_os = "macos"))]
        metal_ffn_layer_bufs: std::sync::OnceLock::new(),
    });
    (model, fixture)
}

/// Build a `LoadedModel` backed by a real synthetic **Q4K-quantised**
/// vindex on disk. Same shape as [`model_with_real_weights`] but the
/// on-disk vindex carries `attn_weights_q4k.bin` +
/// `interleaved_kquant.bin` so `generate_with_sampling`'s
/// `insert_q4k_layer_tensors` actually finds the K-quant data it
/// expects (instead of panicking with "attn Q4K slices missing for
/// layer 0"). Use this for tests that exercise the OpenAI generation
/// endpoints, the streaming SSE path, or `routes/walk_ffn/q8k.rs`.
pub fn model_with_q4k_weights(
    id: &str,
) -> (Arc<LoadedModel>, synthetic_q4k_vindex::SyntheticQ4kVindex) {
    use larql_vindex::{SilentLoadCallbacks, VectorIndex};

    let fixture = synthetic_q4k_vindex::build();
    let mut cb = SilentLoadCallbacks;
    let mut index =
        VectorIndex::load_vindex(&fixture.dir, &mut cb).expect("load synthetic Q4K vindex");
    // Production `bootstrap.rs` calls these two loaders explicitly
    // after `load_vindex` when the vindex is Q4K-quantised — without
    // them, `insert_q4k_layer_tensors` (called from the generation
    // path) panics with "attn Q4K slices missing for layer N".
    index
        .load_attn_kquant(&fixture.dir)
        .expect("load attn_weights_q4k.bin into VectorIndex");
    index
        .load_interleaved_kquant(&fixture.dir)
        .expect("load interleaved_kquant.bin into VectorIndex");
    let config = larql_vindex::load_vindex_config(&fixture.dir).expect("load Q4K vindex config");

    let tok_bytes = std::fs::read(fixture.dir.join("tokenizer.json")).expect("read tokenizer.json");
    let fixture_tokenizer =
        larql_vindex::tokenizers::Tokenizer::from_bytes(&tok_bytes).expect("parse Q4K tokenizer");

    // Q4K vindex still writes `embeddings.bin` as plain f32 (only
    // attn + interleaved are Q4K-packed) so the read pattern is the
    // same as the f32 fixture.
    let embed_bytes =
        std::fs::read(fixture.dir.join("embeddings.bin")).expect("read embeddings.bin");
    let mut embed_floats = Vec::with_capacity(embed_bytes.len() / 4);
    for chunk in embed_bytes.chunks_exact(4) {
        embed_floats.push(f32::from_le_bytes(chunk.try_into().unwrap()));
    }
    let embeddings = Array2::from_shape_vec((fixture.vocab_size, fixture.hidden), embed_floats)
        .expect("Q4K embeddings shape");

    let model = Arc::new(LoadedModel {
        id: id.to_string(),
        path: fixture.dir.clone(),
        config,
        patched: std::sync::Arc::new(tokio::sync::RwLock::new(PatchedVindex::new(index))),
        embeddings,
        embed_scale: 1.0,
        tokenizer: fixture_tokenizer,
        infer_disabled: false,
        ffn_only: false,
        embed_only: false,
        embed_store: None,
        release_mmap_after_request: false,
        weights: std::sync::OnceLock::new(),
        probe_labels: HashMap::new(),
        ffn_l2_cache: FfnL2Cache::new(1),
        layer_latency_tracker: std::sync::Arc::new(
            larql_server::metrics::LayerLatencyTracker::new(),
        ),
        requests_in_flight: std::sync::Arc::new(std::sync::atomic::AtomicU32::new(0)),
        requests_total: std::sync::Arc::new(std::sync::atomic::AtomicU64::new(0)),
        expert_filter: None,
        unit_filter: None,
        moe_remote: None,
        #[cfg(all(feature = "metal-experts", target_os = "macos"))]
        metal_backend: std::sync::OnceLock::new(),
        #[cfg(all(feature = "metal-experts", target_os = "macos"))]
        moe_scratches: std::sync::Mutex::new(std::collections::HashMap::new()),
        #[cfg(all(feature = "metal-experts", target_os = "macos"))]
        metal_ffn_layer_bufs: std::sync::OnceLock::new(),
    });
    (model, fixture)
}

// ══════════════════════════════════════════════════════════════
// State builders
// ══════════════════════════════════════════════════════════════

pub fn state(models: Vec<Arc<LoadedModel>>) -> Arc<AppState> {
    Arc::new(AppState {
        models,
        started_at: std::time::Instant::now(),
        requests_served: AtomicU64::new(0),
        api_key: None,
        sessions: SessionManager::new(3600),
        describe_cache: DescribeCache::new(0),
    })
}

pub fn state_with_key(models: Vec<Arc<LoadedModel>>, key: &str) -> Arc<AppState> {
    Arc::new(AppState {
        models,
        started_at: std::time::Instant::now(),
        requests_served: AtomicU64::new(0),
        api_key: Some(key.to_string()),
        sessions: SessionManager::new(3600),
        describe_cache: DescribeCache::new(0),
    })
}

pub fn state_with_cache(models: Vec<Arc<LoadedModel>>, cache_size: u64) -> Arc<AppState> {
    Arc::new(AppState {
        models,
        started_at: std::time::Instant::now(),
        requests_served: AtomicU64::new(0),
        api_key: None,
        sessions: SessionManager::new(3600),
        describe_cache: DescribeCache::new(cache_size),
    })
}

// ══════════════════════════════════════════════════════════════
// HTTP helpers
// ══════════════════════════════════════════════════════════════

pub async fn body_json(body: Body) -> serde_json::Value {
    let bytes = axum::body::to_bytes(body, usize::MAX).await.unwrap();
    serde_json::from_slice(&bytes).unwrap_or(serde_json::Value::Null)
}

pub async fn get(app: axum::Router, path: &str) -> axum::http::Response<Body> {
    app.oneshot(
        Request::builder()
            .method("GET")
            .uri(path)
            .body(Body::empty())
            .unwrap(),
    )
    .await
    .unwrap()
}

pub async fn get_h(app: axum::Router, path: &str, h: (&str, &str)) -> axum::http::Response<Body> {
    app.oneshot(
        Request::builder()
            .method("GET")
            .uri(path)
            .header(h.0, h.1)
            .body(Body::empty())
            .unwrap(),
    )
    .await
    .unwrap()
}

pub async fn post_json(
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

pub async fn post_json_h(
    app: axum::Router,
    path: &str,
    body: serde_json::Value,
    h: (&str, &str),
) -> axum::http::Response<Body> {
    app.oneshot(
        Request::builder()
            .method("POST")
            .uri(path)
            .header("content-type", "application/json")
            .header(h.0, h.1)
            .body(Body::from(serde_json::to_vec(&body).unwrap()))
            .unwrap(),
    )
    .await
    .unwrap()
}

pub async fn delete(app: axum::Router, path: &str) -> axum::http::Response<Body> {
    app.oneshot(
        Request::builder()
            .method("DELETE")
            .uri(path)
            .body(Body::empty())
            .unwrap(),
    )
    .await
    .unwrap()
}

// ══════════════════════════════════════════════════════════════
// Patch helpers
// ══════════════════════════════════════════════════════════════

pub fn inline_delete_patch(name: &str) -> serde_json::Value {
    serde_json::json!({
        "patch": {
            "version": 1,
            "base_model": "test",
            "base_checksum": null,
            "created_at": "2026-04-26",
            "description": name,
            "author": null,
            "tags": [],
            "operations": [
                {"op": "delete", "layer": 0, "feature": 2}
            ]
        }
    })
}

// Re-export commonly-used router constructors
pub use larql_server::routes::{multi_model_router, single_model_router};
