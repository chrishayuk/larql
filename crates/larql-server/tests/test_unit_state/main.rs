//! Pure unit tests: AppState, model ID, multi-model lookup, infer mode parsing,
//! auth, rate limit, cache, ETag, session, announce hash, warmup_model,
//! probe labels, content token, server error mapping, infer disabled logic.

use axum::response::IntoResponse;
use larql_server::cache::DescribeCache;
use larql_server::error::ServerError;
use larql_server::ffn_l2_cache::FfnL2Cache;
use larql_server::session::SessionManager;
use larql_server::state::{load_probe_labels, model_id_from_name, AppState, LoadedModel};
use larql_vindex::ndarray::Array2;
use larql_vindex::{
    ExtractLevel, FeatureMeta, PatchedVindex, QuantFormat, VectorIndex, VindexConfig,
    VindexLayerInfo,
};
use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::atomic::AtomicU64;
use std::sync::Arc;

// Tiny fixture helpers (local copies — ~50 LOC)

fn make_top_k(token: &str, id: u32, logit: f32) -> larql_models::TopKEntry {
    larql_models::TopKEntry {
        token: token.to_string(),
        token_id: id,
        logit,
    }
}

fn make_meta(token: &str, id: u32, score: f32) -> FeatureMeta {
    FeatureMeta {
        top_token: token.to_string(),
        top_token_id: id,
        c_score: score,
        top_k: vec![
            make_top_k(token, id, score),
            make_top_k("also", id + 1, score * 0.5),
        ],
    }
}

fn make_tiny_model(id: &str) -> Arc<LoadedModel> {
    let hidden = 4;
    let gate = Array2::<f32>::zeros((2, hidden));
    let index = VectorIndex::new(vec![Some(gate)], vec![None], 1, hidden);
    let patched = PatchedVindex::new(index);
    let tok_json =
        r#"{"version":"1.0","model":{"type":"BPE","vocab":{},"merges":[]},"added_tokens":[]}"#;
    let tokenizer = larql_vindex::tokenizers::Tokenizer::from_bytes(tok_json).unwrap();
    Arc::new(LoadedModel {
        id: id.to_string(),
        path: PathBuf::from("/nonexistent"),
        config: VindexConfig {
            version: 2,
            model: "test/model".to_string(),
            family: "test".to_string(),
            source: None,
            checksums: None,
            num_layers: 1,
            hidden_size: hidden,
            intermediate_size: 8,
            vocab_size: 4,
            embed_scale: 1.0,
            extract_level: ExtractLevel::Browse,
            dtype: larql_vindex::StorageDtype::default(),
            quant: QuantFormat::None,
            layer_bands: None,
            layers: vec![VindexLayerInfo {
                layer: 0,
                num_features: 2,
                offset: 0,
                length: 32,
                num_experts: None,
                num_features_per_expert: None,
            }],
            down_top_k: 2,
            has_model_weights: false,
            model_config: None,
            fp4: None,
            ffn_layout: None,
            bitnet_layout: None,
        },
        patched: std::sync::Arc::new(tokio::sync::RwLock::new(patched)),
        embeddings: Array2::<f32>::zeros((4, hidden)),
        embed_scale: 1.0,
        tokenizer,
        infer_disabled: true,
        ffn_only: false,
        embed_only: false,
        embed_store: None,
        release_mmap_after_request: false,
        weights: std::sync::OnceLock::new(),
        weights_init: std::sync::Mutex::new(()),
        bitnet_model: std::sync::OnceLock::new(),
        bitnet_init: std::sync::Mutex::new(()),
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
    })
}

fn make_tiny_state(models: Vec<Arc<LoadedModel>>) -> Arc<AppState> {
    Arc::new(AppState {
        model_set: std::sync::RwLock::new(larql_server::state::ModelSet {
            models,
            v3_models: Vec::new(),
            aliases: Default::default(),
        }),
        router_topology: larql_server::state::RouterTopology::SingleModel,
        lifecycle: std::sync::Mutex::new(larql_server::state::LifecycleState::Idle),
        started_at: std::time::Instant::now(),
        requests_served: AtomicU64::new(0),
        api_key: None,
        sessions: SessionManager::new(3600),
        describe_cache: DescribeCache::new(0),
        infer_timeout: std::time::Duration::from_secs(60),
        patch_sources: Default::default(),
        responses: larql_server::response_store::ResponseStore::new(),
        v3_kv: larql_server::response_kv::ResponseKvCache::new(
            larql_server::response_kv::DEFAULT_MAX_ENTRIES,
            larql_server::response_kv::DEFAULT_TTL_SECS,
        ),
        runtime: Arc::new(larql_server::runtime_stats::RuntimeRecorder::new()),
    })
}

fn make_loaded_model_for_warmup() -> Arc<LoadedModel> {
    let hidden = 4;
    let gate = Array2::<f32>::zeros((3, hidden));
    let meta = vec![Some(make_meta("Paris", 100, 0.9))];
    let index = VectorIndex::new(vec![Some(gate)], vec![Some(meta)], 1, hidden);

    let config = VindexConfig {
        version: 2,
        model: "test/warmup-model".to_string(),
        family: "test".to_string(),
        source: None,
        checksums: None,
        num_layers: 1,
        hidden_size: hidden,
        intermediate_size: 12,
        vocab_size: 8,
        embed_scale: 1.0,
        extract_level: ExtractLevel::Browse,
        dtype: larql_vindex::StorageDtype::default(),
        quant: QuantFormat::None,
        layer_bands: Some(larql_vindex::LayerBands {
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
        bitnet_layout: None,
    };

    let tok_json =
        r#"{"version":"1.0","model":{"type":"BPE","vocab":{},"merges":[]},"added_tokens":[]}"#;
    let tokenizer = larql_vindex::tokenizers::Tokenizer::from_bytes(tok_json).unwrap();

    Arc::new(LoadedModel {
        id: "warmup-test".into(),
        path: PathBuf::from("/nonexistent"),
        config,
        patched: std::sync::Arc::new(tokio::sync::RwLock::new(PatchedVindex::new(index))),
        embeddings: Array2::<f32>::zeros((8, hidden)),
        embed_scale: 1.0,
        tokenizer,
        infer_disabled: true,
        ffn_only: false,
        embed_only: false,
        embed_store: None,
        release_mmap_after_request: false,
        weights: std::sync::OnceLock::new(),
        weights_init: std::sync::Mutex::new(()),
        bitnet_model: std::sync::OnceLock::new(),
        bitnet_init: std::sync::Mutex::new(()),
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
    })
}

// APPSTATE UNIT TESTS

// MODEL_ID_FROM_NAME EDGE CASES

fn model_id(name: &str) -> String {
    name.rsplit('/').next().unwrap_or(name).to_string()
}

// MULTI-MODEL LOOKUP

// INFER MODE PARSING

// AUTH LOGIC

// RATE LIMITER (inline logic)

fn rate_limit_parse(spec: &str) -> Option<(f64, f64)> {
    let parts: Vec<&str> = spec.split('/').collect();
    if parts.len() != 2 {
        return None;
    }
    let count: f64 = parts[0].trim().parse().ok()?;
    let per_sec = match parts[1].trim() {
        "sec" | "s" | "second" => count,
        "min" | "m" | "minute" => count / 60.0,
        "hour" | "h" => count / 3600.0,
        _ => return None,
    };
    Some((count, per_sec))
}

use larql_server::ratelimit::RateLimiter;

// DESCRIBE CACHE

// ETAG

use larql_server::etag::{compute_etag, matches_etag};

// SESSION — get_or_create, session_count

// ANNOUNCE — vindex_identity_hash

// WARMUP — warmup_model unit tests

// PROBE LABELS (load_probe_labels)

// RELATIONS CONTENT-TOKEN FILTER

fn is_content_token_test(tok: &str) -> bool {
    let tok = tok.trim();
    if tok.is_empty() || tok.len() > 30 {
        return false;
    }
    let readable = tok
        .chars()
        .filter(|c| {
            c.is_ascii_alphanumeric()
                || *c == ' '
                || *c == '-'
                || *c == '\''
                || *c == '.'
                || *c == ','
        })
        .count();
    let total = tok.chars().count();
    if readable * 2 < total || total == 0 {
        return false;
    }
    let chars: Vec<char> = tok.chars().collect();
    if chars.len() < 3 || chars.len() > 25 {
        return false;
    }
    let alpha = chars.iter().filter(|c| c.is_ascii_alphabetic()).count();
    if alpha < chars.len() * 2 / 3 {
        return false;
    }
    for w in chars.windows(2) {
        if w[0].is_ascii_lowercase() && w[1].is_ascii_uppercase() {
            return false;
        }
    }
    if !chars.iter().any(|c| c.is_ascii_alphabetic()) {
        return false;
    }
    let lower = tok.to_lowercase();
    !matches!(
        lower.as_str(),
        "the"
            | "and"
            | "for"
            | "but"
            | "not"
            | "you"
            | "all"
            | "can"
            | "her"
            | "was"
            | "one"
            | "our"
            | "out"
            | "are"
            | "has"
            | "his"
            | "how"
            | "its"
            | "may"
            | "new"
            | "now"
            | "old"
            | "see"
            | "way"
            | "who"
            | "did"
            | "get"
            | "let"
            | "say"
            | "she"
            | "too"
            | "use"
            | "from"
            | "have"
            | "been"
            | "will"
            | "with"
            | "this"
            | "that"
            | "they"
            | "were"
            | "some"
            | "them"
            | "than"
            | "when"
            | "what"
            | "your"
            | "each"
            | "make"
            | "like"
            | "just"
            | "over"
            | "such"
            | "take"
            | "also"
            | "into"
            | "only"
            | "very"
            | "more"
            | "does"
            | "most"
            | "about"
            | "which"
            | "their"
            | "would"
            | "there"
            | "could"
            | "other"
            | "after"
            | "being"
            | "where"
            | "these"
            | "those"
            | "first"
            | "should"
            | "because"
            | "through"
            | "before"
            | "par"
            | "aux"
            | "che"
            | "del"
    )
}

// SERVER ERROR → HTTP RESPONSE

// STATS — mode advertisement

// INFER DISABLED LOGIC

// ERROR HANDLING (model lookup)

// RATELIMIT MIDDLEWARE

use axum::body::Body;
use axum::extract::ConnectInfo;
use axum::http::{Request, StatusCode};
use axum::{middleware, routing::get, Router};
use larql_server::ratelimit::{rate_limit_middleware, RateLimitState};
use std::net::SocketAddr;
use tower::ServiceExt as TowerServiceExt;

async fn ok_handler() -> &'static str {
    "ok"
}

fn router_with_limiter(rl: Arc<RateLimiter>) -> Router {
    router_with_limiter_trust_forwarded_for(rl, false)
}

fn router_with_limiter_trust_forwarded_for(
    rl: Arc<RateLimiter>,
    trust_forwarded_for: bool,
) -> Router {
    let state = Arc::new(RateLimitState {
        limiter: rl,
        trust_forwarded_for,
    });
    Router::new()
        .route("/v1/stats", get(ok_handler))
        .route("/v1/health", get(ok_handler))
        .layer(middleware::from_fn_with_state(state, rate_limit_middleware))
}

mod appstate_unit_tests;
mod etag;
mod rate_limiter_inline_logic;
mod ratelimit_middleware;
mod stats_mode_advertisement;
mod warmup_warmup_model_unit_tests;
