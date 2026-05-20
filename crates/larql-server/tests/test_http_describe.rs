#![cfg(not(target_arch = "wasm32"))]
//! HTTP integration tests: describe endpoint (all band variants, verbose,
//! cache, ETag, multi-model).

mod common;
use common::*;

use axum::body::Body;
use axum::http::{Request, StatusCode};
use tower::ServiceExt;

// ══════════════════════════════════════════════════════════════
// GET /v1/describe
// ══════════════════════════════════════════════════════════════

#[tokio::test]
async fn http_describe_returns_200_with_entity_field() {
    let app = single_model_router(state(vec![model("test")]));
    let resp = get(app, "/v1/describe?entity=France").await;
    assert_eq!(resp.status(), StatusCode::OK);
    let body = body_json(resp.into_body()).await;
    assert_eq!(body["entity"], "France");
    assert!(body["edges"].is_array());
    assert!(body["latency_ms"].as_f64().is_some());
}

#[tokio::test]
async fn http_describe_empty_vocab_returns_empty_edges() {
    // Empty BPE tokenizer → empty token_ids → graceful empty response.
    let app = single_model_router(state(vec![model("test")]));
    let resp = get(app, "/v1/describe?entity=Germany").await;
    assert_eq!(resp.status(), StatusCode::OK);
    let body = body_json(resp.into_body()).await;
    assert_eq!(body["edges"].as_array().unwrap().len(), 0);
}

#[tokio::test]
async fn http_describe_missing_entity_returns_400() {
    let app = single_model_router(state(vec![model("test")]));
    let resp = get(app, "/v1/describe").await; // no entity param
                                               // axum rejects the missing required query param
    assert_eq!(resp.status(), StatusCode::BAD_REQUEST);
}

// ══════════════════════════════════════════════════════════════
// Band variants
// ══════════════════════════════════════════════════════════════

#[tokio::test]
async fn http_describe_band_syntax_returns_200() {
    let app = single_model_router(state(vec![model("test")]));
    let resp = get(app, "/v1/describe?entity=France&band=syntax").await;
    assert_eq!(resp.status(), StatusCode::OK);
    let body = body_json(resp.into_body()).await;
    assert_eq!(body["entity"], "France");
    assert!(body["edges"].is_array());
}

#[tokio::test]
async fn http_describe_band_output_returns_200() {
    let app = single_model_router(state(vec![model("test")]));
    let resp = get(app, "/v1/describe?entity=France&band=output").await;
    assert_eq!(resp.status(), StatusCode::OK);
}

#[tokio::test]
async fn http_describe_band_all_returns_200() {
    let app = single_model_router(state(vec![model("test")]));
    let resp = get(app, "/v1/describe?entity=France&band=all").await;
    assert_eq!(resp.status(), StatusCode::OK);
    let body = body_json(resp.into_body()).await;
    assert!(body["edges"].is_array());
}

#[tokio::test]
async fn http_describe_verbose_mode_returns_200() {
    let app = single_model_router(state(vec![model("test")]));
    let resp = get(app, "/v1/describe?entity=France&verbose=true").await;
    assert_eq!(resp.status(), StatusCode::OK);
}

#[tokio::test]
async fn http_describe_empty_entity_returns_empty_edges() {
    // Empty tokenizer → empty token ids → early return with edges=[].
    let app = single_model_router(state(vec![model("test")]));
    let resp = get(app, "/v1/describe?entity=hello").await;
    assert_eq!(resp.status(), StatusCode::OK);
    let body = body_json(resp.into_body()).await;
    // Empty BPE → no token ids → describe_entity returns edges=[].
    assert!(body["edges"].is_array());
}

// ══════════════════════════════════════════════════════════════
// ETag and cache
// ══════════════════════════════════════════════════════════════

#[tokio::test]
async fn http_describe_has_etag_header() {
    let app = single_model_router(state(vec![model("test")]));
    let resp = get(app, "/v1/describe?entity=France").await;
    assert_eq!(resp.status(), StatusCode::OK);
    assert!(resp.headers().contains_key("etag"));
}

#[tokio::test]
async fn http_describe_cache_hit_returns_cached_response() {
    let st = state_with_cache(vec![model("test")], 100);
    // First request populates cache.
    let app1 = single_model_router(st.clone());
    let r1 = get(app1, "/v1/describe?entity=France").await;
    assert_eq!(r1.status(), StatusCode::OK);
    let etag = r1.headers()["etag"].to_str().unwrap().to_string();

    // Second request — same key, cache enabled — returns cached with same etag.
    let app2 = single_model_router(st.clone());
    let r2 = get(app2, "/v1/describe?entity=France").await;
    assert_eq!(r2.status(), StatusCode::OK);
    assert_eq!(r2.headers()["etag"].to_str().unwrap(), etag);
}

#[tokio::test]
async fn http_describe_if_none_match_returns_304() {
    let st = state_with_cache(vec![model("test")], 100);
    // Get etag from first request.
    let app1 = single_model_router(st.clone());
    let r1 = get(app1, "/v1/describe?entity=France").await;
    let etag = r1.headers()["etag"].to_str().unwrap().to_string();

    // Second request with If-None-Match → 304.
    let app2 = single_model_router(st.clone());
    let resp = app2
        .oneshot(
            Request::builder()
                .method("GET")
                .uri("/v1/describe?entity=France")
                .header("if-none-match", &etag)
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(resp.status(), StatusCode::NOT_MODIFIED);
}

// ══════════════════════════════════════════════════════════════
// Multi-model describe
// ══════════════════════════════════════════════════════════════

#[tokio::test]
async fn http_describe_multi_model_returns_200() {
    let app = multi_model_router(state(vec![model("a"), model("b")]));
    let resp = get(app, "/v1/a/describe?entity=France").await;
    assert_eq!(resp.status(), StatusCode::OK);
}

#[tokio::test]
async fn http_describe_multi_model_not_found_returns_404() {
    let app = multi_model_router(state(vec![model("a")]));
    let resp = get(app, "/v1/nosuchmodel/describe?entity=France").await;
    assert_eq!(resp.status(), StatusCode::NOT_FOUND);
}
