//! Shared HTTP test infrastructure for larql-server integration tests.
//!
//! Uses axum's tower::ServiceExt::oneshot pattern — requests are dispatched
//! in-process to the full router with no network socket. Every test builds a
//! synthetic in-memory VectorIndex (1 layer, 3 features, hidden=4).

#![allow(dead_code, unused_imports)]

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
        patched: tokio::sync::RwLock::new(PatchedVindex::new(test_index())),
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
        expert_filter: None,
        unit_filter: None,
        moe_remote: None,
        #[cfg(feature = "metal-experts")]
        metal_backend: std::sync::OnceLock::new(),
        #[cfg(feature = "metal-experts")]
        moe_scratches: std::sync::Mutex::new(std::collections::HashMap::new()),
        #[cfg(feature = "metal-experts")]
        metal_ffn_layer_bufs: std::sync::OnceLock::new(),
    })
}

/// ModelBuilder with optional infer_disabled override (defaults true).
pub fn model_infer_enabled(id: &str) -> Arc<LoadedModel> {
    Arc::new(LoadedModel {
        id: id.to_string(),
        path: PathBuf::from("/nonexistent"),
        config: test_config(),
        patched: tokio::sync::RwLock::new(PatchedVindex::new(test_index())),
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
        expert_filter: None,
        unit_filter: None,
        moe_remote: None,
        #[cfg(feature = "metal-experts")]
        metal_backend: std::sync::OnceLock::new(),
        #[cfg(feature = "metal-experts")]
        moe_scratches: std::sync::Mutex::new(std::collections::HashMap::new()),
        #[cfg(feature = "metal-experts")]
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
            patched: tokio::sync::RwLock::new(PatchedVindex::new(test_index())),
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
            expert_filter: None,
            unit_filter: None,
            moe_remote: None,
            #[cfg(feature = "metal-experts")]
            metal_backend: std::sync::OnceLock::new(),
            #[cfg(feature = "metal-experts")]
            moe_scratches: std::sync::Mutex::new(std::collections::HashMap::new()),
            #[cfg(feature = "metal-experts")]
            metal_ffn_layer_bufs: std::sync::OnceLock::new(),
        })
    }
}

pub fn model(id: &str) -> Arc<LoadedModel> {
    ModelBuilder::new(id).build()
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
