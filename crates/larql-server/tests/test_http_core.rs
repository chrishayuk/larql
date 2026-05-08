//! HTTP integration tests: health, models, stats, auth, error responses,
//! request counter, probe labels.

mod common;
use common::*;

use axum::http::StatusCode;
use axum::middleware;
use axum::response::IntoResponse;
use larql_server::auth::auth_middleware;
use larql_server::cache::DescribeCache;
use larql_server::error::ServerError;
use larql_server::session::SessionManager;
use larql_server::state::AppState;
use std::sync::atomic::AtomicU64;
use std::sync::Arc;

// ══════════════════════════════════════════════════════════════
// GET /v1/health
// ══════════════════════════════════════════════════════════════

#[tokio::test]
async fn http_health_returns_200() {
    let app = single_model_router(state(vec![model("test")]));
    let resp = get(app, "/v1/health").await;
    assert_eq!(resp.status(), StatusCode::OK);
}

#[tokio::test]
async fn http_health_body_has_required_fields() {
    let app = single_model_router(state(vec![model("test")]));
    let resp = get(app, "/v1/health").await;
    let body = body_json(resp.into_body()).await;
    assert_eq!(body["status"], "ok");
    assert!(body["uptime_seconds"].as_u64().is_some());
    assert!(body["requests_served"].as_u64().is_some());
}

#[tokio::test]
async fn http_health_bumps_request_counter() {
    let st = state(vec![model("test")]);
    let app = single_model_router(st.clone());
    get(app, "/v1/health").await;
    assert_eq!(
        st.requests_served
            .load(std::sync::atomic::Ordering::Relaxed),
        1
    );
}

// ══════════════════════════════════════════════════════════════
// GET /v1/models
// ══════════════════════════════════════════════════════════════

#[tokio::test]
async fn http_models_single_lists_one_model() {
    let app = single_model_router(state(vec![model("gemma")]));
    let resp = get(app, "/v1/models").await;
    assert_eq!(resp.status(), StatusCode::OK);
    let body = body_json(resp.into_body()).await;
    // OpenAI-compat shape: {object: "list", data: [...]}
    assert_eq!(body["object"], "list");
    let data = body["data"].as_array().unwrap();
    assert_eq!(data.len(), 1);
    assert_eq!(data[0]["id"], "gemma");
    assert_eq!(data[0]["object"], "model");
    assert_eq!(data[0]["owned_by"], "larql");
    assert!(data[0]["created"].is_u64());
    // larql-specific extras still present.
    assert!(data[0]["features"].as_u64().is_some());
    assert_eq!(data[0]["loaded"], true);
}

#[tokio::test]
async fn http_models_single_path_is_v1() {
    let app = single_model_router(state(vec![model("m")]));
    let resp = get(app, "/v1/models").await;
    let body = body_json(resp.into_body()).await;
    assert_eq!(body["data"][0]["path"], "/v1");
}

#[tokio::test]
async fn http_models_multi_path_includes_model_id() {
    let app = multi_model_router(state(vec![model("a"), model("b")]));
    let resp = get(app, "/v1/models").await;
    let body = body_json(resp.into_body()).await;
    let data = body["data"].as_array().unwrap();
    assert_eq!(data.len(), 2);
    // Multi-model paths are /v1/{id}
    let paths: Vec<&str> = data.iter().map(|m| m["path"].as_str().unwrap()).collect();
    assert!(paths.contains(&"/v1/a"));
    assert!(paths.contains(&"/v1/b"));
}

// ══════════════════════════════════════════════════════════════
// GET /v1/stats — single model
// ══════════════════════════════════════════════════════════════

#[tokio::test]
async fn http_stats_returns_model_info() {
    let app = single_model_router(state(vec![model("test")]));
    let resp = get(app, "/v1/stats").await;
    assert_eq!(resp.status(), StatusCode::OK);
    let body = body_json(resp.into_body()).await;
    assert_eq!(body["model"], "test/model-4");
    assert_eq!(body["family"], "test");
    assert_eq!(body["layers"], 1);
    assert_eq!(body["features"], 3);
    assert_eq!(body["hidden_size"], 4);
    assert_eq!(body["vocab_size"], 8);
    assert!(body["layer_bands"].is_object());
}

#[tokio::test]
async fn http_stats_mode_full_by_default() {
    let app = single_model_router(state(vec![model("test")]));
    let resp = get(app, "/v1/stats").await;
    let body = body_json(resp.into_body()).await;
    assert_eq!(body["mode"], "full");
    assert_eq!(body["loaded"]["ffn_service"], true);
}

#[tokio::test]
async fn http_stats_mode_ffn_service_when_ffn_only() {
    let m = ModelBuilder::new("test").ffn_only().build();
    let app = single_model_router(state(vec![m]));
    let resp = get(app, "/v1/stats").await;
    let body = body_json(resp.into_body()).await;
    assert_eq!(body["mode"], "ffn-service");
    assert_eq!(body["loaded"]["inference"], false);
}

#[tokio::test]
async fn http_stats_mode_embed_service_when_embed_only() {
    let m = ModelBuilder::new("test").embed_only().build();
    let app = single_model_router(state(vec![m]));
    let resp = get(app, "/v1/stats").await;
    let body = body_json(resp.into_body()).await;
    assert_eq!(body["mode"], "embed-service");
    assert_eq!(body["loaded"]["embed_service"], true);
    assert_eq!(body["loaded"]["browse"], false);
}

#[tokio::test]
async fn http_stats_layer_bands_shape() {
    let app = single_model_router(state(vec![model("test")]));
    let resp = get(app, "/v1/stats").await;
    let body = body_json(resp.into_body()).await;
    let bands = &body["layer_bands"];
    assert!(bands["syntax"].is_array());
    assert!(bands["knowledge"].is_array());
    assert!(bands["output"].is_array());
}

// ══════════════════════════════════════════════════════════════
// MULTI-MODEL stats
// ══════════════════════════════════════════════════════════════

#[tokio::test]
async fn http_multi_health_returns_200() {
    let app = multi_model_router(state(vec![model("a"), model("b")]));
    let resp = get(app, "/v1/health").await;
    assert_eq!(resp.status(), StatusCode::OK);
}

#[tokio::test]
async fn http_multi_models_lists_both() {
    let app = multi_model_router(state(vec![model("a"), model("b")]));
    let resp = get(app, "/v1/models").await;
    let body = body_json(resp.into_body()).await;
    assert_eq!(body["object"], "list");
    assert_eq!(body["data"].as_array().unwrap().len(), 2);
}

#[tokio::test]
async fn http_multi_stats_valid_model_returns_200() {
    let app = multi_model_router(state(vec![model("alpha"), model("beta")]));
    let resp = get(app, "/v1/alpha/stats").await;
    assert_eq!(resp.status(), StatusCode::OK);
    let body = body_json(resp.into_body()).await;
    assert_eq!(body["model"], "test/model-4");
}

#[tokio::test]
async fn http_multi_stats_unknown_model_returns_404() {
    let app = multi_model_router(state(vec![model("a")]));
    let resp = get(app, "/v1/unknown/stats").await;
    assert_eq!(resp.status(), StatusCode::NOT_FOUND);
}

// ══════════════════════════════════════════════════════════════
// AUTH MIDDLEWARE
// ══════════════════════════════════════════════════════════════

#[tokio::test]
async fn http_auth_no_api_key_configured_allows_all() {
    // No api_key in state → middleware passes everything.
    let app = single_model_router(state(vec![model("test")]));
    let resp = get(app, "/v1/stats").await;
    assert_eq!(resp.status(), StatusCode::OK);
}

#[tokio::test]
async fn http_auth_correct_bearer_returns_200() {
    let st = state_with_key(vec![model("test")], "secret123");
    let app =
        single_model_router(st.clone()).layer(middleware::from_fn_with_state(st, auth_middleware));
    let resp = get_h(app, "/v1/stats", ("authorization", "Bearer secret123")).await;
    assert_eq!(resp.status(), StatusCode::OK);
}

#[tokio::test]
async fn http_auth_wrong_bearer_returns_401() {
    let st = state_with_key(vec![model("test")], "secret123");
    let app =
        single_model_router(st.clone()).layer(middleware::from_fn_with_state(st, auth_middleware));
    let resp = get_h(app, "/v1/stats", ("authorization", "Bearer wrongkey")).await;
    assert_eq!(resp.status(), StatusCode::UNAUTHORIZED);
}

#[tokio::test]
async fn http_auth_missing_header_returns_401() {
    let st = state_with_key(vec![model("test")], "secret123");
    let app =
        single_model_router(st.clone()).layer(middleware::from_fn_with_state(st, auth_middleware));
    let resp = get(app, "/v1/stats").await; // no auth header
    assert_eq!(resp.status(), StatusCode::UNAUTHORIZED);
}

#[tokio::test]
async fn http_auth_health_exempt_without_key() {
    let st = state_with_key(vec![model("test")], "secret123");
    let app =
        single_model_router(st.clone()).layer(middleware::from_fn_with_state(st, auth_middleware));
    // /v1/health must be reachable even without auth.
    let resp = get(app, "/v1/health").await;
    assert_eq!(resp.status(), StatusCode::OK);
}

#[tokio::test]
async fn http_auth_non_bearer_format_rejected() {
    let st = state_with_key(vec![model("test")], "secret123");
    let app =
        single_model_router(st.clone()).layer(middleware::from_fn_with_state(st, auth_middleware));
    let resp = get_h(app, "/v1/stats", ("authorization", "Token secret123")).await;
    assert_eq!(resp.status(), StatusCode::UNAUTHORIZED);
}

// ══════════════════════════════════════════════════════════════
// SERVER ERROR → HTTP RESPONSE (async body read)
// ══════════════════════════════════════════════════════════════

#[tokio::test]
async fn http_server_error_not_found_body_has_error_key() {
    let resp = ServerError::NotFound("entity not found".into()).into_response();
    let status = resp.status();
    let body = body_json(resp.into_body()).await;
    assert_eq!(status, StatusCode::NOT_FOUND);
    assert!(body["error"].as_str().unwrap().contains("entity not found"));
}

#[tokio::test]
async fn http_server_error_bad_request_body_has_error_key() {
    let resp = ServerError::BadRequest("invalid param".into()).into_response();
    let status = resp.status();
    let body = body_json(resp.into_body()).await;
    assert_eq!(status, StatusCode::BAD_REQUEST);
    assert!(body["error"].as_str().unwrap().contains("invalid param"));
}

#[tokio::test]
async fn http_server_error_internal_body_has_error_key() {
    let resp = ServerError::Internal("disk failure".into()).into_response();
    let status = resp.status();
    let body = body_json(resp.into_body()).await;
    assert_eq!(status, StatusCode::INTERNAL_SERVER_ERROR);
    assert!(body["error"].as_str().unwrap().contains("disk failure"));
}

#[tokio::test]
async fn http_server_error_unavailable_body_has_error_key() {
    let resp = ServerError::InferenceUnavailable("no weights loaded".into()).into_response();
    let status = resp.status();
    let body = body_json(resp.into_body()).await;
    assert_eq!(status, StatusCode::SERVICE_UNAVAILABLE);
    assert!(body["error"]
        .as_str()
        .unwrap()
        .contains("no weights loaded"));
}

// ══════════════════════════════════════════════════════════════
// REQUEST COUNTER
// ══════════════════════════════════════════════════════════════

#[tokio::test]
async fn http_requests_served_increments_per_request() {
    let st = state(vec![model("test")]);
    let before = st
        .requests_served
        .load(std::sync::atomic::Ordering::Relaxed);

    let app = single_model_router(st.clone());
    get(app, "/v1/health").await;

    let after = st
        .requests_served
        .load(std::sync::atomic::Ordering::Relaxed);
    assert_eq!(after, before + 1);
}

#[tokio::test]
async fn http_select_increments_request_counter() {
    let st = state(vec![model("test")]);
    let app = single_model_router(st.clone());
    post_json(app, "/v1/select", serde_json::json!({})).await;
    assert_eq!(
        st.requests_served
            .load(std::sync::atomic::Ordering::Relaxed),
        1
    );
}

// ══════════════════════════════════════════════════════════════
// LOAD PROBE LABELS (async round-trip via file I/O)
// ══════════════════════════════════════════════════════════════

#[tokio::test]
async fn http_load_probe_labels_roundtrip() {
    use larql_server::state::load_probe_labels;
    let dir = std::env::temp_dir().join("larql_http_labels_01");
    tokio::fs::create_dir_all(&dir).await.unwrap();
    let json = r#"{"L0_F0":"capital","L1_F2":"language"}"#;
    tokio::fs::write(dir.join("feature_labels.json"), json)
        .await
        .unwrap();

    let labels = load_probe_labels(&dir);
    assert_eq!(labels.get(&(0, 0)), Some(&"capital".to_string()));
    assert_eq!(labels.get(&(1, 2)), Some(&"language".to_string()));

    let _ = tokio::fs::remove_dir_all(&dir).await;
}

// ══════════════════════════════════════════════════════════════
// WARMUP — no model → 404
// ══════════════════════════════════════════════════════════════

#[tokio::test]
async fn http_warmup_no_model_returns_404() {
    // single_model_router with empty model list → model(None) returns None → 404.
    let st = Arc::new(AppState {
        models: vec![],
        started_at: std::time::Instant::now(),
        requests_served: AtomicU64::new(0),
        api_key: None,
        sessions: SessionManager::new(3600),
        describe_cache: DescribeCache::new(0),
        attention_sessions: std::sync::Arc::new(
            larql_server::attention_session::AttentionSessionMap::new(600, 256),
        ),
        default_kv_format: None,
    });
    let app = single_model_router(st);
    let resp = post_json(app, "/v1/warmup", serde_json::json!({})).await;
    assert_eq!(resp.status(), StatusCode::NOT_FOUND);
}
