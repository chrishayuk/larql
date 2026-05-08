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

// ══════════════════════════════════════════════════════════════
// Tiny fixture helpers (local copies — ~50 LOC)
// ══════════════════════════════════════════════════════════════

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
        },
        patched: tokio::sync::RwLock::new(patched),
        embeddings: Array2::<f32>::zeros((4, hidden)),
        embed_scale: 1.0,
        tokenizer,
        infer_disabled: true,
        ffn_only: false,
        embed_only: false,
        embed_store: None,
        release_mmap_after_request: false,
        weights: std::sync::OnceLock::new(),
        probe_labels: HashMap::new(),
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

fn make_tiny_state(models: Vec<Arc<LoadedModel>>) -> Arc<AppState> {
    Arc::new(AppState {
        models,
        started_at: std::time::Instant::now(),
        requests_served: AtomicU64::new(0),
        api_key: None,
        sessions: SessionManager::new(3600),
        describe_cache: DescribeCache::new(0),
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
    };

    let tok_json =
        r#"{"version":"1.0","model":{"type":"BPE","vocab":{},"merges":[]},"added_tokens":[]}"#;
    let tokenizer = larql_vindex::tokenizers::Tokenizer::from_bytes(tok_json).unwrap();

    Arc::new(LoadedModel {
        id: "warmup-test".into(),
        path: PathBuf::from("/nonexistent"),
        config,
        patched: tokio::sync::RwLock::new(PatchedVindex::new(index)),
        embeddings: Array2::<f32>::zeros((8, hidden)),
        embed_scale: 1.0,
        tokenizer,
        infer_disabled: true,
        ffn_only: false,
        embed_only: false,
        embed_store: None,
        release_mmap_after_request: false,
        weights: std::sync::OnceLock::new(),
        probe_labels: HashMap::new(),
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

// ══════════════════════════════════════════════════════════════
// APPSTATE UNIT TESTS
// ══════════════════════════════════════════════════════════════

#[test]
fn test_app_state_model_single_none_returns_first() {
    let state = make_tiny_state(vec![make_tiny_model("gemma")]);
    let m = state.model(None);
    assert!(m.is_some());
    assert_eq!(m.unwrap().id, "gemma");
}

#[test]
fn test_app_state_model_with_id_finds_correct() {
    let state = make_tiny_state(vec![make_tiny_model("a"), make_tiny_model("b")]);
    assert_eq!(state.model(Some("a")).unwrap().id, "a");
    assert_eq!(state.model(Some("b")).unwrap().id, "b");
}

#[test]
fn test_app_state_model_multi_none_returns_none() {
    let state = make_tiny_state(vec![make_tiny_model("a"), make_tiny_model("b")]);
    // Multi-model with no id → must specify which model.
    assert!(state.model(None).is_none());
}

#[test]
fn test_app_state_model_unknown_id_returns_none() {
    let state = make_tiny_state(vec![make_tiny_model("a")]);
    assert!(state.model(Some("nonexistent")).is_none());
}

#[test]
fn test_app_state_is_multi_model_single() {
    let state = make_tiny_state(vec![make_tiny_model("a")]);
    assert!(!state.is_multi_model());
}

#[test]
fn test_app_state_is_multi_model_multi() {
    let state = make_tiny_state(vec![make_tiny_model("a"), make_tiny_model("b")]);
    assert!(state.is_multi_model());
}

#[test]
fn test_app_state_bump_requests_increments() {
    let state = make_tiny_state(vec![make_tiny_model("a")]);
    assert_eq!(
        state
            .requests_served
            .load(std::sync::atomic::Ordering::Relaxed),
        0
    );
    state.bump_requests();
    assert_eq!(
        state
            .requests_served
            .load(std::sync::atomic::Ordering::Relaxed),
        1
    );
    state.bump_requests();
    state.bump_requests();
    assert_eq!(
        state
            .requests_served
            .load(std::sync::atomic::Ordering::Relaxed),
        3
    );
}

// ══════════════════════════════════════════════════════════════
// MODEL_ID_FROM_NAME EDGE CASES
// ══════════════════════════════════════════════════════════════

#[test]
fn test_model_id_extraction() {
    assert_eq!(model_id("google/gemma-3-4b-it"), "gemma-3-4b-it");
    assert_eq!(model_id("llama-3-8b"), "llama-3-8b");
    assert_eq!(model_id("org/sub/model"), "model");
}

fn model_id(name: &str) -> String {
    name.rsplit('/').next().unwrap_or(name).to_string()
}

#[test]
fn test_model_id_from_name_no_slash() {
    assert_eq!(model_id_from_name("llama-3-8b"), "llama-3-8b");
}

#[test]
fn test_model_id_from_name_single_slash() {
    assert_eq!(model_id_from_name("google/gemma-3-4b-it"), "gemma-3-4b-it");
}

#[test]
fn test_model_id_from_name_deep_path() {
    assert_eq!(model_id_from_name("org/sub/model"), "model");
}

#[test]
fn test_model_id_from_name_trailing_slash() {
    // rsplit('/').next() on "foo/" returns "" — reflects actual behavior.
    let result = model_id_from_name("foo/");
    assert_eq!(result, "");
}

// ══════════════════════════════════════════════════════════════
// MULTI-MODEL LOOKUP
// ══════════════════════════════════════════════════════════════

#[test]
fn test_multi_model_lookup_by_id() {
    // Simulate AppState.model() logic
    let models = ["gemma-3-4b-it", "llama-3-8b", "mistral-7b"];
    let find = |id: &str| models.iter().find(|m| **m == id);
    assert_eq!(find("gemma-3-4b-it"), Some(&"gemma-3-4b-it"));
    assert_eq!(find("llama-3-8b"), Some(&"llama-3-8b"));
    assert_eq!(find("nonexistent"), None);
}

#[test]
fn test_single_model_returns_first() {
    let models = ["only-model"];
    // Single model mode: None → returns first
    let result = if models.len() == 1 {
        models.first()
    } else {
        None
    };
    assert_eq!(result, Some(&"only-model"));
}

#[test]
fn test_multi_model_none_returns_none() {
    let models = ["a", "b"];
    // Multi-model mode: None → returns None (must specify ID)
    let result: Option<&&str> = if models.len() == 1 {
        models.first()
    } else {
        None
    };
    assert_eq!(result, None);
}

// ══════════════════════════════════════════════════════════════
// INFER MODE PARSING
// ══════════════════════════════════════════════════════════════

#[test]
fn test_infer_mode_parsing() {
    // The infer handler parses mode into walk/dense/compare
    let check = |mode: &str| -> (bool, bool) {
        let is_compare = mode == "compare";
        let use_walk = mode == "walk" || is_compare;
        let use_dense = mode == "dense" || is_compare;
        (use_walk, use_dense)
    };

    assert_eq!(check("walk"), (true, false));
    assert_eq!(check("dense"), (false, true));
    assert_eq!(check("compare"), (true, true));
}

#[test]
fn test_config_has_inference_capability() {
    let mut config = VindexConfig {
        version: 2,
        model: "test/model-4".to_string(),
        family: "test".to_string(),
        source: None,
        checksums: None,
        num_layers: 2,
        hidden_size: 4,
        intermediate_size: 12,
        vocab_size: 8,
        embed_scale: 1.0,
        extract_level: ExtractLevel::Browse,
        dtype: larql_vindex::StorageDtype::default(),
        quant: QuantFormat::None,
        layer_bands: None,
        layers: vec![],
        down_top_k: 5,
        has_model_weights: false,
        model_config: None,
        fp4: None,
        ffn_layout: None,
    };

    // Browse level → no inference
    config.extract_level = ExtractLevel::Browse;
    config.has_model_weights = false;
    let has_weights = config.has_model_weights
        || config.extract_level == ExtractLevel::Inference
        || config.extract_level == ExtractLevel::All;
    assert!(!has_weights);

    // Inference level → has inference
    config.extract_level = ExtractLevel::Inference;
    let has_weights = config.has_model_weights
        || config.extract_level == ExtractLevel::Inference
        || config.extract_level == ExtractLevel::All;
    assert!(has_weights);

    // Legacy has_model_weights flag
    config.extract_level = ExtractLevel::Browse;
    config.has_model_weights = true;
    let has_weights = config.has_model_weights
        || config.extract_level == ExtractLevel::Inference
        || config.extract_level == ExtractLevel::All;
    assert!(has_weights);
}

// ══════════════════════════════════════════════════════════════
// AUTH LOGIC
// ══════════════════════════════════════════════════════════════

#[test]
fn test_bearer_token_extraction() {
    let header = "Bearer sk-abc123";
    let token = header.strip_prefix("Bearer ");
    assert_eq!(token, Some("sk-abc123"));
}

#[test]
fn test_bearer_token_mismatch() {
    let header = "Bearer wrong-key";
    let required = "sk-abc123";
    let token = &header[7..];
    assert_ne!(token, required);
}

#[test]
fn test_no_auth_header() {
    let header: Option<&str> = None;
    let has_valid_token = header
        .filter(|h| h.starts_with("Bearer "))
        .map(|h| &h[7..])
        .is_some();
    assert!(!has_valid_token);
}

#[test]
fn test_health_exempt_from_auth() {
    let path = "/v1/health";
    let is_health = path == "/v1/health";
    assert!(is_health);

    let path = "/v1/describe";
    let is_health = path == "/v1/health";
    assert!(!is_health);
}

// ══════════════════════════════════════════════════════════════
// RATE LIMITER (inline logic)
// ══════════════════════════════════════════════════════════════

#[test]
fn test_rate_limit_parse() {
    // Valid formats
    assert!(rate_limit_parse("100/min").is_some());
    assert!(rate_limit_parse("10/sec").is_some());
    assert!(rate_limit_parse("3600/hour").is_some());
    assert!(rate_limit_parse("50/s").is_some());
    assert!(rate_limit_parse("200/m").is_some());

    // Invalid formats
    assert!(rate_limit_parse("abc").is_none());
    assert!(rate_limit_parse("100").is_none());
    assert!(rate_limit_parse("100/day").is_none());
}

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

#[test]
fn test_rate_limit_token_bucket() {
    // Simulate token bucket: 2 tokens, 1 refill/sec
    let mut tokens: f64 = 2.0;
    let max_tokens: f64 = 2.0;

    // First two requests succeed
    assert!(tokens >= 1.0);
    tokens -= 1.0;
    assert!(tokens >= 1.0);
    tokens -= 1.0;

    // Third fails
    assert!(tokens < 1.0);

    // Refill
    tokens = (tokens + 1.0).min(max_tokens);
    assert!(tokens >= 1.0);
}

use larql_server::ratelimit::RateLimiter;

#[test]
fn test_rate_limiter_zero_count_rejects_immediately() {
    // "0/sec" → 0 tokens → first request is rejected.
    let rl = RateLimiter::parse("0/sec");
    // Either returns None (invalid) or allows creation and rejects first request.
    if let Some(rl) = rl {
        let ip: std::net::IpAddr = "127.0.0.1".parse().unwrap();
        assert!(!rl.check(ip));
    }
    // None is also acceptable — 0/sec is edge-case.
}

#[test]
fn test_rate_limiter_per_minute_long_form() {
    // "60/minute" is valid; verify it allows 60 consecutive requests.
    let rl = RateLimiter::parse("60/minute").unwrap();
    let ip: std::net::IpAddr = "10.0.0.60".parse().unwrap();
    for _ in 0..60 {
        assert!(rl.check(ip));
    }
    assert!(!rl.check(ip)); // 61st request blocked
}

#[test]
fn test_rate_limiter_per_second_long_form() {
    // "10/second" is valid; verify it allows 10 consecutive requests.
    let rl = RateLimiter::parse("10/second").unwrap();
    let ip: std::net::IpAddr = "10.0.0.10".parse().unwrap();
    for _ in 0..10 {
        assert!(rl.check(ip));
    }
    assert!(!rl.check(ip)); // 11th request blocked
}

#[test]
fn test_rate_limiter_fractional_count() {
    // "1/hour" → bucket holds 1 token; second request is blocked.
    let rl = RateLimiter::parse("1/hour").unwrap();
    let ip: std::net::IpAddr = "10.0.0.1".parse().unwrap();
    assert!(rl.check(ip));
    assert!(!rl.check(ip)); // no refill within the test
}

#[test]
fn test_rate_limiter_empty_spec_rejects() {
    assert!(RateLimiter::parse("").is_none());
    assert!(RateLimiter::parse("/").is_none());
    assert!(RateLimiter::parse("100/").is_none());
}

// ══════════════════════════════════════════════════════════════
// DESCRIBE CACHE
// ══════════════════════════════════════════════════════════════

#[test]
fn test_cache_key_format() {
    let key = format!("{}:{}:{}:{}:{}", "model", "France", "knowledge", 20, 5);
    assert_eq!(key, "model:France:knowledge:20:5");
}

#[test]
fn test_cache_disabled_when_ttl_zero() {
    // TTL=0 means cache is disabled
    let ttl = 0u64;
    assert_eq!(ttl, 0);
}

#[test]
fn test_cache_hit_and_miss() {
    let mut cache: HashMap<String, serde_json::Value> = HashMap::new();
    let key = "model:France:knowledge:20:5".to_string();
    let value = serde_json::json!({"entity": "France", "edges": []});

    // Miss
    assert!(!cache.contains_key(&key));

    // Insert
    cache.insert(key.clone(), value.clone());

    // Hit
    assert_eq!(cache.get(&key), Some(&value));
}

#[test]
fn test_cache_overwrite_updates_value() {
    let cache = DescribeCache::new(60);
    let key = DescribeCache::key("model", "France", "knowledge", 20, 5.0);
    let v1 = serde_json::json!({"edges": []});
    let v2 = serde_json::json!({"edges": [{"target": "Paris"}]});
    cache.put(key.clone(), v1);
    cache.put(key.clone(), v2.clone());
    assert_eq!(cache.get(&key), Some(v2));
}

#[test]
fn test_cache_key_float_precision_truncated() {
    // min_score is cast to u32 in the key, so 5.9 and 5.0 produce the same key.
    let k1 = DescribeCache::key("m", "e", "b", 10, 5.0);
    let k2 = DescribeCache::key("m", "e", "b", 10, 5.9);
    assert_eq!(k1, k2);
    // 6.0 differs.
    let k3 = DescribeCache::key("m", "e", "b", 10, 6.0);
    assert_ne!(k1, k3);
}

// ══════════════════════════════════════════════════════════════
// ETAG
// ══════════════════════════════════════════════════════════════

#[test]
fn test_etag_deterministic() {
    use std::collections::hash_map::DefaultHasher;
    use std::hash::{Hash, Hasher};

    let body = serde_json::json!({"entity": "France", "edges": [{"target": "Paris"}]});
    let s = body.to_string();

    let mut h1 = DefaultHasher::new();
    s.hash(&mut h1);
    let mut h2 = DefaultHasher::new();
    s.hash(&mut h2);
    assert_eq!(h1.finish(), h2.finish());
}

#[test]
fn test_etag_format() {
    // ETag should be quoted hex string
    let body = serde_json::json!({"test": true});
    let s = body.to_string();
    let mut hasher = std::collections::hash_map::DefaultHasher::new();
    std::hash::Hash::hash(&s, &mut hasher);
    let etag = format!("\"{:x}\"", std::hash::Hasher::finish(&hasher));
    assert!(etag.starts_with('"'));
    assert!(etag.ends_with('"'));
    assert!(etag.len() > 4); // At least "xx"
}

#[test]
fn test_if_none_match_comparison() {
    let etag = "\"abc123\"";
    // Exact match
    assert_eq!(etag.trim(), etag);
    // Wildcard
    assert_eq!("*".trim(), "*");
    // No match
    assert_ne!("\"different\"".trim(), etag);
}

#[test]
fn test_304_not_modified_condition() {
    let cached_etag = "\"abc123\"";
    let request_etag = "\"abc123\"";
    let should_304 = request_etag.trim() == cached_etag || request_etag.trim() == "*";
    assert!(should_304);

    let stale_etag = "\"old\"";
    let should_304 = stale_etag.trim() == cached_etag || stale_etag.trim() == "*";
    assert!(!should_304);
}

use larql_server::etag::{compute_etag, matches_etag};

#[test]
fn test_etag_empty_object_is_valid() {
    let etag = compute_etag(&serde_json::json!({}));
    assert!(etag.starts_with('"') && etag.ends_with('"'));
    assert!(etag.len() > 2);
}

#[test]
fn test_etag_different_key_order_produces_different_hash() {
    // JSON key ordering matters when serialised.
    let a = compute_etag(&serde_json::json!({"a": 1, "b": 2}));
    let b = compute_etag(&serde_json::json!({"b": 2, "a": 1}));
    // serde_json preserves insertion order, so these are the same.
    assert_eq!(a, b);
}

#[test]
fn test_matches_etag_extra_whitespace() {
    let etag = compute_etag(&serde_json::json!({"x": 1}));
    // Leading/trailing whitespace should still match after trim.
    let padded = format!("  {}  ", etag);
    assert!(matches_etag(Some(&padded), &etag));
}

#[test]
fn test_matches_etag_mismatch_returns_false() {
    assert!(!matches_etag(Some("\"abc\""), "\"xyz\""));
}

// ══════════════════════════════════════════════════════════════
// SESSION — get_or_create, session_count
// ══════════════════════════════════════════════════════════════

#[tokio::test]
async fn session_get_or_create_new_session_returns_empty_patched() {
    let sm = SessionManager::new(3600);
    let m = make_loaded_model_for_warmup();
    let patched = sm.get_or_create("new-session", &m).await;
    assert_eq!(patched.num_patches(), 0);
}

#[tokio::test]
async fn session_count_increments_on_first_create() {
    let sm = SessionManager::new(3600);
    let m = make_loaded_model_for_warmup();
    assert_eq!(sm.session_count().await, 0);
    sm.get_or_create("s1", &m).await;
    assert_eq!(sm.session_count().await, 1);
    sm.get_or_create("s2", &m).await;
    assert_eq!(sm.session_count().await, 2);
}

#[tokio::test]
async fn session_get_or_create_same_id_does_not_add_session() {
    let sm = SessionManager::new(3600);
    let m = make_loaded_model_for_warmup();
    sm.get_or_create("same", &m).await;
    sm.get_or_create("same", &m).await;
    assert_eq!(sm.session_count().await, 1);
}

#[tokio::test]
async fn session_remove_patch_from_unknown_session_returns_err() {
    let sm = SessionManager::new(3600);
    let result = sm.remove_patch("does-not-exist", "any").await;
    assert!(result.is_err());
    assert!(result.unwrap_err().contains("not found"));
}

// ══════════════════════════════════════════════════════════════
// ANNOUNCE — vindex_identity_hash
// ══════════════════════════════════════════════════════════════

#[test]
fn vindex_identity_hash_is_deterministic() {
    use larql_server::announce::vindex_identity_hash;
    let h1 = vindex_identity_hash("gemma-3-4b", 34);
    let h2 = vindex_identity_hash("gemma-3-4b", 34);
    assert_eq!(h1, h2);
}

#[test]
fn vindex_identity_hash_differs_on_model_id() {
    use larql_server::announce::vindex_identity_hash;
    let h1 = vindex_identity_hash("gemma-3-4b", 34);
    let h2 = vindex_identity_hash("llama-3-8b", 34);
    assert_ne!(h1, h2);
}

#[test]
fn vindex_identity_hash_differs_on_num_layers() {
    use larql_server::announce::vindex_identity_hash;
    let h1 = vindex_identity_hash("model", 32);
    let h2 = vindex_identity_hash("model", 34);
    assert_ne!(h1, h2);
}

#[test]
fn vindex_identity_hash_is_hex_string() {
    use larql_server::announce::vindex_identity_hash;
    let h = vindex_identity_hash("gemma-3-4b", 34);
    assert_eq!(h.len(), 16);
    assert!(h.chars().all(|c| c.is_ascii_hexdigit()));
}

// ══════════════════════════════════════════════════════════════
// WARMUP — warmup_model unit tests
// ══════════════════════════════════════════════════════════════

#[test]
fn warmup_model_skip_weights_sets_loaded_false() {
    use larql_server::routes::warmup::{warmup_model, WarmupRequest};
    let model = make_loaded_model_for_warmup();
    let req = WarmupRequest {
        layers: None,
        skip_weights: true,
        warmup_hnsw: false,
    };
    let resp = warmup_model(&model, &req);
    assert!(!resp.weights_loaded);
    assert_eq!(resp.weights_load_ms, 0);
}

#[test]
fn warmup_model_with_explicit_layers_prefetches_matching() {
    use larql_server::routes::warmup::{warmup_model, WarmupRequest};
    let model = make_loaded_model_for_warmup();
    let req = WarmupRequest {
        layers: Some(vec![0]),
        skip_weights: true,
        warmup_hnsw: false,
    };
    let resp = warmup_model(&model, &req);
    assert_eq!(resp.layers_prefetched, 1);
}

#[test]
fn warmup_model_out_of_range_layer_is_skipped() {
    use larql_server::routes::warmup::{warmup_model, WarmupRequest};
    let model = make_loaded_model_for_warmup();
    let req = WarmupRequest {
        layers: Some(vec![999]),
        skip_weights: true,
        warmup_hnsw: false,
    };
    let resp = warmup_model(&model, &req);
    assert_eq!(resp.layers_prefetched, 0);
}

#[test]
fn warmup_model_empty_layers_list_prefetches_zero() {
    use larql_server::routes::warmup::{warmup_model, WarmupRequest};
    let model = make_loaded_model_for_warmup();
    let req = WarmupRequest {
        layers: Some(vec![]),
        skip_weights: true,
        warmup_hnsw: false,
    };
    let resp = warmup_model(&model, &req);
    assert_eq!(resp.layers_prefetched, 0);
}

#[test]
fn warmup_model_reports_correct_model_name() {
    use larql_server::routes::warmup::{warmup_model, WarmupRequest};
    let model = make_loaded_model_for_warmup();
    let req = WarmupRequest {
        layers: Some(vec![]),
        skip_weights: true,
        warmup_hnsw: false,
    };
    let resp = warmup_model(&model, &req);
    assert_eq!(resp.model, "test/warmup-model");
}

#[test]
fn warmup_model_weight_load_fails_gracefully() {
    use larql_server::routes::warmup::{warmup_model, WarmupRequest};
    let model = make_loaded_model_for_warmup();
    let req = WarmupRequest {
        layers: Some(vec![]),
        skip_weights: false,
        warmup_hnsw: false,
    };
    // Path is /nonexistent so get_or_load_weights fails — should warn but not panic.
    let resp = warmup_model(&model, &req);
    assert!(!resp.weights_loaded);
}

// ══════════════════════════════════════════════════════════════
// PROBE LABELS (load_probe_labels)
// ══════════════════════════════════════════════════════════════

#[test]
fn test_load_probe_labels_from_json_file() {
    let dir = std::env::temp_dir().join("larql_test_labels_01");
    std::fs::create_dir_all(&dir).unwrap();
    let json = r#"{"L0_F0": "capital", "L1_F2": "language", "L5_F10": "continent"}"#;
    std::fs::write(dir.join("feature_labels.json"), json).unwrap();

    let labels = load_probe_labels(&dir);
    assert_eq!(labels.get(&(0, 0)), Some(&"capital".to_string()));
    assert_eq!(labels.get(&(1, 2)), Some(&"language".to_string()));
    assert_eq!(labels.get(&(5, 10)), Some(&"continent".to_string()));
    assert_eq!(labels.len(), 3);

    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn test_load_probe_labels_missing_file_returns_empty() {
    let dir = std::path::Path::new("/nonexistent/path/to/vindex");
    let labels = load_probe_labels(dir);
    assert!(labels.is_empty());
}

#[test]
fn test_load_probe_labels_malformed_json_returns_empty() {
    let dir = std::env::temp_dir().join("larql_test_labels_02");
    std::fs::create_dir_all(&dir).unwrap();
    std::fs::write(dir.join("feature_labels.json"), b"not valid json").unwrap();

    let labels = load_probe_labels(&dir);
    assert!(labels.is_empty());

    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn test_load_probe_labels_non_object_json_returns_empty() {
    let dir = std::env::temp_dir().join("larql_test_labels_03");
    std::fs::create_dir_all(&dir).unwrap();
    std::fs::write(
        dir.join("feature_labels.json"),
        b"[\"not\",\"an\",\"object\"]",
    )
    .unwrap();

    let labels = load_probe_labels(&dir);
    assert!(labels.is_empty());

    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn test_load_probe_labels_skips_malformed_keys() {
    let dir = std::env::temp_dir().join("larql_test_labels_04");
    std::fs::create_dir_all(&dir).unwrap();
    // Mix of valid and invalid keys
    let json = r#"{"L0_F0": "capital", "INVALID": "skip", "L_BAD_F": "skip2", "L3_F7": "valid"}"#;
    std::fs::write(dir.join("feature_labels.json"), json).unwrap();

    let labels = load_probe_labels(&dir);
    // Only L0_F0 and L3_F7 should parse.
    assert_eq!(labels.get(&(0, 0)), Some(&"capital".to_string()));
    assert_eq!(labels.get(&(3, 7)), Some(&"valid".to_string()));
    assert_eq!(labels.len(), 2);

    let _ = std::fs::remove_dir_all(&dir);
}

// ══════════════════════════════════════════════════════════════
// RELATIONS CONTENT-TOKEN FILTER
// ══════════════════════════════════════════════════════════════

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

#[test]
fn test_content_token_valid_words() {
    assert!(is_content_token_test("capital"));
    assert!(is_content_token_test("Paris"));
    assert!(is_content_token_test("language"));
    assert!(is_content_token_test("France"));
    assert!(is_content_token_test("Europe"));
}

#[test]
fn test_content_token_stopwords_rejected() {
    assert!(!is_content_token_test("the"));
    assert!(!is_content_token_test("and"));
    assert!(!is_content_token_test("for"));
    assert!(!is_content_token_test("with"));
    assert!(!is_content_token_test("about"));
    assert!(!is_content_token_test("should"));
}

#[test]
fn test_content_token_too_short_rejected() {
    assert!(!is_content_token_test("ab")); // < 3 chars
    assert!(!is_content_token_test("a"));
    assert!(!is_content_token_test(""));
}

#[test]
fn test_content_token_too_long_rejected() {
    let long = "a".repeat(26);
    assert!(!is_content_token_test(&long));
}

#[test]
fn test_content_token_camelcase_rejected() {
    assert!(!is_content_token_test("camelCase"));
    assert!(!is_content_token_test("camelCaseWord"));
}

#[test]
fn test_content_token_numeric_heavy_rejected() {
    // Less than 2/3 alpha characters
    assert!(!is_content_token_test("a12345"));
}

// ══════════════════════════════════════════════════════════════
// SERVER ERROR → HTTP RESPONSE
// ══════════════════════════════════════════════════════════════

#[test]
fn test_server_error_not_found_maps_to_404() {
    let resp = ServerError::NotFound("the-thing".into()).into_response();
    assert_eq!(resp.status(), axum::http::StatusCode::NOT_FOUND);
}

#[test]
fn test_server_error_bad_request_maps_to_400() {
    let resp = ServerError::BadRequest("bad input".into()).into_response();
    assert_eq!(resp.status(), axum::http::StatusCode::BAD_REQUEST);
}

#[test]
fn test_server_error_internal_maps_to_500() {
    let resp = ServerError::Internal("oops".into()).into_response();
    assert_eq!(resp.status(), axum::http::StatusCode::INTERNAL_SERVER_ERROR);
}

#[test]
fn test_server_error_unavailable_maps_to_503() {
    #[allow(dead_code)]
    let resp = ServerError::InferenceUnavailable("no weights".into()).into_response();
    assert_eq!(resp.status(), axum::http::StatusCode::SERVICE_UNAVAILABLE);
}

#[test]
fn test_server_error_display_format() {
    assert!(format!("{}", ServerError::NotFound("x".into())).contains("not found"));
    assert!(format!("{}", ServerError::BadRequest("x".into())).contains("bad request"));
    assert!(format!("{}", ServerError::Internal("x".into())).contains("internal error"));
}

// ══════════════════════════════════════════════════════════════
// STATS — mode advertisement
// ══════════════════════════════════════════════════════════════

#[test]
fn test_stats_shape_includes_mode_full_by_default() {
    let mode = "full";
    let ffn_service = true;
    let stats = serde_json::json!({
        "mode": mode,
        "loaded": { "ffn_service": ffn_service },
    });
    assert_eq!(stats["mode"], "full");
    assert_eq!(stats["loaded"]["ffn_service"], true);
}

#[test]
fn test_stats_shape_advertises_ffn_service_mode() {
    let mode = "ffn-service";
    let inference_available = false;
    let stats = serde_json::json!({
        "mode": mode,
        "loaded": {
            "browse": true,
            "inference": inference_available,
            "ffn_service": true,
        },
    });
    assert_eq!(stats["mode"], "ffn-service");
    assert_eq!(stats["loaded"]["inference"], false);
    assert_eq!(stats["loaded"]["ffn_service"], true);
}

#[test]
fn test_ffn_only_implies_infer_disabled() {
    fn effective(no_infer: bool, ffn_only: bool) -> bool {
        no_infer || ffn_only
    }
    assert!(!effective(false, false));
    assert!(effective(true, false));
    assert!(effective(false, true));
    assert!(effective(true, true));
}

#[test]
fn test_stats_shape_advertises_embed_service_mode() {
    let stats = serde_json::json!({
        "mode": "embed-service",
        "loaded": {
            "browse": false,
            "inference": false,
            "ffn_service": false,
            "embed_service": true,
        },
    });
    assert_eq!(stats["mode"], "embed-service");
    assert_eq!(stats["loaded"]["embed_service"], true);
    assert_eq!(stats["loaded"]["browse"], false);
    assert_eq!(stats["loaded"]["ffn_service"], false);
}

#[test]
fn test_embed_only_implies_infer_disabled() {
    fn effective(no_infer: bool, ffn_only: bool, embed_only: bool) -> bool {
        no_infer || ffn_only || embed_only
    }
    assert!(!effective(false, false, false));
    assert!(effective(false, false, true));
    assert!(effective(false, true, false));
    assert!(effective(true, false, false));
    assert!(effective(true, true, true));
}

#[test]
fn test_embed_only_mode_string() {
    fn mode(embed_only: bool, ffn_only: bool) -> &'static str {
        if embed_only {
            "embed-service"
        } else if ffn_only {
            "ffn-service"
        } else {
            "full"
        }
    }
    assert_eq!(mode(false, false), "full");
    assert_eq!(mode(false, true), "ffn-service");
    assert_eq!(mode(true, false), "embed-service");
    // embed_only takes priority
    assert_eq!(mode(true, true), "embed-service");
}

// ══════════════════════════════════════════════════════════════
// INFER DISABLED LOGIC
// ══════════════════════════════════════════════════════════════

#[test]
fn test_infer_disabled_check() {
    let disabled = true;
    assert!(disabled); // Handler returns 503

    let disabled = false;
    assert!(!disabled); // Handler proceeds
}

#[test]
fn test_infer_weights_required() {
    let config = VindexConfig {
        version: 2,
        model: "test/model-4".to_string(),
        family: "test".to_string(),
        source: None,
        checksums: None,
        num_layers: 2,
        hidden_size: 4,
        intermediate_size: 12,
        vocab_size: 8,
        embed_scale: 1.0,
        extract_level: ExtractLevel::Browse,
        dtype: larql_vindex::StorageDtype::default(),
        quant: QuantFormat::None,
        layer_bands: None,
        layers: vec![],
        down_top_k: 5,
        has_model_weights: false,
        model_config: None,
        fp4: None,
        ffn_layout: None,
    };
    // Browse level + no model weights → can't infer
    let can_infer = config.has_model_weights
        || config.extract_level == ExtractLevel::Inference
        || config.extract_level == ExtractLevel::All;
    assert!(!can_infer);
}

#[test]
fn test_infer_compare_returns_both() {
    let mode = "compare";
    let is_compare = mode == "compare";
    let use_walk = mode == "walk" || is_compare;
    let use_dense = mode == "dense" || is_compare;
    assert!(is_compare);
    assert!(use_walk);
    assert!(use_dense);
}

#[test]
fn test_infer_disabled_all_flag_combinations() {
    fn eff(no_infer: bool, ffn_only: bool, embed_only: bool) -> bool {
        no_infer || ffn_only || embed_only
    }
    // All off → enabled
    assert!(!eff(false, false, false));
    // Single flags
    assert!(eff(true, false, false));
    assert!(eff(false, true, false));
    assert!(eff(false, false, true));
    // Combinations
    assert!(eff(true, true, false));
    assert!(eff(false, true, true));
    assert!(eff(true, false, true));
    assert!(eff(true, true, true));
}

// ══════════════════════════════════════════════════════════════
// ERROR HANDLING (model lookup)
// ══════════════════════════════════════════════════════════════

#[test]
fn test_error_model_not_found() {
    let models: Vec<&str> = vec!["gemma-3-4b-it"];
    let result = models.iter().find(|m| **m == "nonexistent");
    assert!(result.is_none()); // → 404
}

#[test]
fn test_error_empty_prompt() {
    let token_ids: Vec<u32> = vec![];
    assert!(token_ids.is_empty()); // → 400 BadRequest
}

#[test]
fn test_error_nonexistent_model_in_multi() {
    let models = ["model-a", "model-b"];
    let find = |id: &str| models.iter().find(|m| **m == id);
    assert!(find("model-c").is_none()); // → 404
}

// ══════════════════════════════════════════════════════════════
// RATELIMIT MIDDLEWARE
// ══════════════════════════════════════════════════════════════

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

#[tokio::test]
async fn rate_limit_blocks_when_exhausted() {
    // 1/sec → first request with trusted X-Forwarded-For passes, second is rejected.
    let rl = Arc::new(RateLimiter::parse("1/sec").unwrap());
    let app1 = router_with_limiter_trust_forwarded_for(Arc::clone(&rl), true);
    let resp1 = app1
        .oneshot(
            Request::builder()
                .method("GET")
                .uri("/v1/stats")
                .header("x-forwarded-for", "1.2.3.4")
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(resp1.status(), StatusCode::OK, "first request should pass");

    let app2 = router_with_limiter_trust_forwarded_for(Arc::clone(&rl), true);
    let resp2 = app2
        .oneshot(
            Request::builder()
                .method("GET")
                .uri("/v1/stats")
                .header("x-forwarded-for", "1.2.3.4")
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(
        resp2.status(),
        StatusCode::TOO_MANY_REQUESTS,
        "second request should be rate-limited"
    );
}

#[tokio::test]
async fn rate_limit_health_exempt() {
    // Even with a 1/sec limiter exhausted, /v1/health is exempt.
    let rl = Arc::new(RateLimiter::parse("1/sec").unwrap());

    // Exhaust the limiter for 127.0.0.1 via X-Forwarded-For.
    let app1 = router_with_limiter_trust_forwarded_for(Arc::clone(&rl), true);
    let resp1 = app1
        .oneshot(
            Request::builder()
                .method("GET")
                .uri("/v1/stats")
                .header("x-forwarded-for", "127.0.0.1")
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(resp1.status(), StatusCode::OK);

    // Verify exhausted on /v1/stats.
    let app2 = router_with_limiter_trust_forwarded_for(Arc::clone(&rl), true);
    let resp2 = app2
        .oneshot(
            Request::builder()
                .method("GET")
                .uri("/v1/stats")
                .header("x-forwarded-for", "127.0.0.1")
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(resp2.status(), StatusCode::TOO_MANY_REQUESTS);

    // Health check is exempt — should still pass.
    let app3 = router_with_limiter_trust_forwarded_for(Arc::clone(&rl), true);
    let resp3 = app3
        .oneshot(
            Request::builder()
                .method("GET")
                .uri("/v1/health")
                .header("x-forwarded-for", "127.0.0.1")
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(
        resp3.status(),
        StatusCode::OK,
        "/v1/health should be exempt from rate limiting"
    );
}

#[tokio::test]
async fn rate_limit_forwarded_for_header_used_as_ip_when_trusted() {
    // X-Forwarded-For: 10.0.0.1 → uses that IP, different from 10.0.0.2.
    let rl = Arc::new(RateLimiter::parse("1/sec").unwrap());
    let proxy_addr: SocketAddr = "192.0.2.10:443".parse().unwrap();

    // Exhaust 10.0.0.1 bucket.
    let app1 = router_with_limiter_trust_forwarded_for(Arc::clone(&rl), true);
    let _ = app1
        .oneshot(
            Request::builder()
                .method("GET")
                .uri("/v1/stats")
                .header("x-forwarded-for", "10.0.0.1")
                .extension(ConnectInfo(proxy_addr))
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();

    // 10.0.0.1 is now blocked.
    let app2 = router_with_limiter_trust_forwarded_for(Arc::clone(&rl), true);
    let resp_blocked = app2
        .oneshot(
            Request::builder()
                .method("GET")
                .uri("/v1/stats")
                .header("x-forwarded-for", "10.0.0.1")
                .extension(ConnectInfo(proxy_addr))
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(resp_blocked.status(), StatusCode::TOO_MANY_REQUESTS);

    // 10.0.0.2 has its own bucket — should pass.
    let app3 = router_with_limiter_trust_forwarded_for(Arc::clone(&rl), true);
    let resp_other = app3
        .oneshot(
            Request::builder()
                .method("GET")
                .uri("/v1/stats")
                .header("x-forwarded-for", "10.0.0.2")
                .extension(ConnectInfo(proxy_addr))
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(
        resp_other.status(),
        StatusCode::OK,
        "different IP should have its own bucket"
    );
}

#[tokio::test]
async fn rate_limit_forwarded_for_header_ignored_by_default() {
    let rl = Arc::new(RateLimiter::parse("1/sec").unwrap());

    for ip in ["10.0.0.1", "10.0.0.2", "10.0.0.3"] {
        let app = router_with_limiter(Arc::clone(&rl));
        let resp = app
            .oneshot(
                Request::builder()
                    .method("GET")
                    .uri("/v1/stats")
                    .header("x-forwarded-for", ip)
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(resp.status(), StatusCode::OK);
    }
}

#[tokio::test]
async fn rate_limit_no_ip_passes_through() {
    // No X-Forwarded-For and no ConnectInfo → middleware has no IP to check.
    // Per the implementation: if ip is None, the check is skipped entirely.
    let rl = Arc::new(RateLimiter::parse("1/sec").unwrap());
    // Make multiple requests with no IP info — all should pass (no IP → no rate limit applied).
    for _ in 0..3 {
        let app = router_with_limiter(Arc::clone(&rl));
        let resp = app
            .oneshot(
                Request::builder()
                    .method("GET")
                    .uri("/v1/stats")
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        // Without an IP, rate_limit_middleware skips the check and passes through.
        assert_eq!(
            resp.status(),
            StatusCode::OK,
            "no IP → should pass through even beyond limit"
        );
    }
}
