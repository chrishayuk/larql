#![cfg(not(target_arch = "wasm32"))]
//! HTTP integration tests: warmup, walk, infer, explain-infer, insert (all variants).

mod common;
use common::*;

use axum::http::StatusCode;

// ══════════════════════════════════════════════════════════════
// POST /v1/warmup
// ══════════════════════════════════════════════════════════════

#[tokio::test]
async fn http_warmup_skip_weights_returns_200() {
    let app = single_model_router(state(vec![model("test")]));
    let resp = post_json(app, "/v1/warmup", serde_json::json!({"skip_weights": true})).await;
    assert_eq!(resp.status(), StatusCode::OK);
    let body = body_json(resp.into_body()).await;
    assert_eq!(body["weights_loaded"], false);
    assert!(body["layers_prefetched"].as_u64().is_some());
    assert!(body["total_ms"].as_u64().is_some());
}

#[tokio::test]
async fn http_warmup_empty_body_returns_200() {
    let app = single_model_router(state(vec![model("test")]));
    let resp = post_json(app, "/v1/warmup", serde_json::json!({})).await;
    assert_eq!(resp.status(), StatusCode::OK);
    let body = body_json(resp.into_body()).await;
    assert!(body["model"].as_str().is_some());
    assert!(body["hnsw_built"].as_bool().is_some());
}

#[tokio::test]
async fn http_warmup_with_layer_list_returns_prefetch_count() {
    let app = single_model_router(state(vec![model("test")]));
    let resp = post_json(
        app,
        "/v1/warmup",
        serde_json::json!({"skip_weights": true, "layers": [0]}),
    )
    .await;
    assert_eq!(resp.status(), StatusCode::OK);
    let body = body_json(resp.into_body()).await;
    assert_eq!(body["layers_prefetched"], 1);
}

#[tokio::test]
async fn http_warmup_with_out_of_range_layers_returns_zero_prefetch() {
    let app = single_model_router(state(vec![model("test")]));
    let resp = post_json(
        app,
        "/v1/warmup",
        serde_json::json!({"skip_weights": true, "layers": [999]}),
    )
    .await;
    assert_eq!(resp.status(), StatusCode::OK);
    let body = body_json(resp.into_body()).await;
    assert_eq!(body["layers_prefetched"], 0);
}

// ══════════════════════════════════════════════════════════════
// GET /v1/walk
// ══════════════════════════════════════════════════════════════

#[tokio::test]
async fn http_walk_empty_prompt_returns_400() {
    // Empty BPE tokenizer produces no token ids → "empty prompt" BadRequest.
    let app = single_model_router(state(vec![model("test")]));
    let resp = get(app, "/v1/walk?prompt=hello").await;
    assert_eq!(resp.status(), StatusCode::BAD_REQUEST);
    let body = body_json(resp.into_body()).await;
    assert!(body["error"].as_str().unwrap().contains("empty prompt"));
}

#[tokio::test]
async fn http_walk_bumps_request_counter() {
    let st = state(vec![model("test")]);
    let app = single_model_router(st.clone());
    get(app, "/v1/walk?prompt=test").await;
    assert_eq!(
        st.requests_served
            .load(std::sync::atomic::Ordering::Relaxed),
        1
    );
}

#[tokio::test]
async fn http_walk_multi_model_not_found_returns_404() {
    let app = multi_model_router(state(vec![model("a")]));
    let resp = get(app, "/v1/nosuchmodel/walk?prompt=hello").await;
    assert_eq!(resp.status(), StatusCode::NOT_FOUND);
}

// ══════════════════════════════════════════════════════════════
// POST /v1/infer
// ══════════════════════════════════════════════════════════════

#[tokio::test]
async fn http_infer_disabled_returns_503() {
    // model() builder sets infer_disabled=true.
    let app = single_model_router(state(vec![model("test")]));
    let resp = post_json(app, "/v1/infer", serde_json::json!({"prompt": "hello"})).await;
    assert_eq!(resp.status(), StatusCode::SERVICE_UNAVAILABLE);
    let body = body_json(resp.into_body()).await;
    assert!(body["error"].as_str().is_some());
}

#[tokio::test]
async fn http_infer_missing_prompt_returns_422() {
    let app = single_model_router(state(vec![model("test")]));
    let resp = post_json(app, "/v1/infer", serde_json::json!({})).await;
    // axum JSON extractor returns 422 for missing required field.
    assert_eq!(resp.status(), StatusCode::UNPROCESSABLE_ENTITY);
}

#[tokio::test]
async fn http_infer_multi_model_not_found_returns_404() {
    let app = multi_model_router(state(vec![model("a")]));
    let resp = post_json(
        app,
        "/v1/nosuchmodel/infer",
        serde_json::json!({"prompt": "hello"}),
    )
    .await;
    assert_eq!(resp.status(), StatusCode::NOT_FOUND);
}

#[tokio::test]
async fn http_infer_bumps_request_counter() {
    let st = state(vec![model("test")]);
    let app = single_model_router(st.clone());
    post_json(app, "/v1/infer", serde_json::json!({"prompt": "hello"})).await;
    assert_eq!(
        st.requests_served
            .load(std::sync::atomic::Ordering::Relaxed),
        1
    );
}

// ══════════════════════════════════════════════════════════════
// POST /v1/explain-infer
// ══════════════════════════════════════════════════════════════

#[tokio::test]
async fn http_explain_no_weights_returns_503() {
    // explain-infer calls get_or_load_weights(); path=/nonexistent → fails → 503.
    let app = single_model_router(state(vec![model("test")]));
    let resp = post_json(
        app,
        "/v1/explain-infer",
        serde_json::json!({"prompt": "hello"}),
    )
    .await;
    assert_eq!(resp.status(), StatusCode::SERVICE_UNAVAILABLE);
}

#[tokio::test]
async fn http_explain_multi_model_not_found_returns_404() {
    let app = multi_model_router(state(vec![model("a")]));
    let resp = post_json(
        app,
        "/v1/nosuchmodel/explain-infer",
        serde_json::json!({"prompt": "hello"}),
    )
    .await;
    assert_eq!(resp.status(), StatusCode::NOT_FOUND);
}

#[tokio::test]
async fn http_explain_bumps_request_counter() {
    let st = state(vec![model("test")]);
    let app = single_model_router(st.clone());
    post_json(app, "/v1/explain-infer", serde_json::json!({"prompt": "x"})).await;
    assert_eq!(
        st.requests_served
            .load(std::sync::atomic::Ordering::Relaxed),
        1
    );
}

// ══════════════════════════════════════════════════════════════
// POST /v1/insert
// ══════════════════════════════════════════════════════════════

#[tokio::test]
async fn http_insert_returns_200_with_embedding_mode() {
    // has_model_weights=false → compute_residuals returns empty → embedding fallback.
    let app = single_model_router(state(vec![model("test")]));
    let resp = post_json(
        app,
        "/v1/insert",
        serde_json::json!({
            "entity": "France",
            "relation": "capital",
            "target": "Paris"
        }),
    )
    .await;
    assert_eq!(resp.status(), StatusCode::OK);
    let body = body_json(resp.into_body()).await;
    assert_eq!(body["entity"], "France");
    assert_eq!(body["relation"], "capital");
    assert_eq!(body["target"], "Paris");
    assert_eq!(body["mode"], "embedding");
    assert!(body["inserted"].as_u64().is_some());
    assert!(body["latency_ms"].is_number());
}

#[tokio::test]
async fn http_insert_with_session_header_returns_session_field() {
    let app = single_model_router(state(vec![model("test")]));
    let resp = post_json_h(
        app,
        "/v1/insert",
        serde_json::json!({
            "entity": "Germany",
            "relation": "capital",
            "target": "Berlin"
        }),
        ("x-session-id", "test-session"),
    )
    .await;
    assert_eq!(resp.status(), StatusCode::OK);
    let body = body_json(resp.into_body()).await;
    assert_eq!(body["session"], "test-session");
}

#[tokio::test]
async fn http_insert_multi_model_not_found_returns_404() {
    let app = multi_model_router(state(vec![model("a")]));
    let resp = post_json(
        app,
        "/v1/nosuchmodel/insert",
        serde_json::json!({
            "entity": "X",
            "relation": "y",
            "target": "Z"
        }),
    )
    .await;
    assert_eq!(resp.status(), StatusCode::NOT_FOUND);
}

#[tokio::test]
async fn http_insert_with_explicit_layer_returns_200() {
    let app = single_model_router(state(vec![model("test")]));
    let resp = post_json(
        app,
        "/v1/insert",
        serde_json::json!({
            "entity": "Japan",
            "relation": "capital",
            "target": "Tokyo",
            "layer": 0
        }),
    )
    .await;
    assert_eq!(resp.status(), StatusCode::OK);
    let body = body_json(resp.into_body()).await;
    assert_eq!(body["entity"], "Japan");
}

#[tokio::test]
async fn http_insert_bumps_request_counter() {
    let st = state(vec![model("test")]);
    let app = single_model_router(st.clone());
    post_json(
        app,
        "/v1/insert",
        serde_json::json!({
            "entity": "X", "relation": "y", "target": "Z"
        }),
    )
    .await;
    assert_eq!(
        st.requests_served
            .load(std::sync::atomic::Ordering::Relaxed),
        1
    );
}

// ══════════════════════════════════════════════════════════════
// POST /v1/infer — no weights (has_model_weights=false, Browse level)
// ══════════════════════════════════════════════════════════════

#[tokio::test]
async fn http_infer_no_weights_check_returns_503() {
    // infer_disabled=false but has_model_weights=false + ExtractLevel::Browse
    // → handler should return 503 "vindex does not contain model weights".
    // model_infer_enabled() uses infer_disabled=false + empty tokenizer.
    // The infer route checks has_model_weights before calling get_or_load_weights.
    // Since extract_level=Browse and has_model_weights=false, it returns 503.
    let app = single_model_router(state(vec![model_infer_enabled("test")]));
    let resp = post_json(app, "/v1/infer", serde_json::json!({"prompt": "hello"})).await;
    assert_eq!(resp.status(), StatusCode::SERVICE_UNAVAILABLE);
    let body = body_json(resp.into_body()).await;
    assert!(
        body["error"]
            .as_str()
            .unwrap_or("")
            .contains("model weights"),
        "expected 'model weights' in error, got: {:?}",
        body["error"]
    );
}
