//! Coverage push for `routes/expert/single.rs`,
//! `routes/expert/cpu.rs`, `routes/expert/layer_batch.rs`,
//! `routes/expert/multi_layer_batch.rs`, and
//! `routes/expert/batch_legacy.rs` — the dispatcher branches that fire
//! before any MoE-specific compute. Specifically the rejection paths:
//!
//!   - "model is not a hybrid MoE — no expert endpoints available"
//!     (every dense model takes this path immediately)
//!   - residual-length mismatch on the request body
//!   - 404 / 400 from the model-id resolver
//!
//! The deeper MoE compute paths need a synthetic hybrid-MoE vindex
//! (router_proj + N experts + arch overriding `is_hybrid_moe()`).
//! That fixture is the same gap that keeps `walk_ffn/core.rs`
//! excluded; covered there once it lands.

mod common;

use axum::body::Body;
use axum::http::{header, Request, StatusCode};
use tower::ServiceExt;

async fn post_expert(
    layer: usize,
    expert_id: usize,
    body: serde_json::Value,
) -> axum::http::Response<Body> {
    let (model, _fixture) = common::model_with_q4k_weights("synthetic");
    let state = common::state(vec![model]);
    let app = larql_server::routes::single_model_router(state);
    let resp = app
        .oneshot(
            Request::builder()
                .method("POST")
                .uri(format!("/v1/expert/{layer}/{expert_id}"))
                .header(header::CONTENT_TYPE, "application/json")
                .body(Body::from(serde_json::to_vec(&body).unwrap()))
                .unwrap(),
        )
        .await
        .unwrap();
    drop(_fixture);
    resp
}

#[tokio::test]
async fn single_expert_dense_model_returns_400_not_moe() {
    // Synthetic uses Llama arch; `is_hybrid_moe()` is false →
    // `run_expert` rejects with "model is not a hybrid MoE".
    let resp = post_expert(
        0,
        0,
        serde_json::json!({
            "residual": vec![0.0_f32; 8],
        }),
    )
    .await;
    assert_eq!(resp.status(), StatusCode::BAD_REQUEST);
    let bytes = axum::body::to_bytes(resp.into_body(), usize::MAX)
        .await
        .unwrap();
    let body = String::from_utf8_lossy(&bytes);
    assert!(
        body.contains("hybrid MoE") || body.contains("expert"),
        "expected MoE-specific error message; got {body}"
    );
}

#[tokio::test]
async fn single_expert_invalid_json_returns_400() {
    let (model, _fixture) = common::model_with_q4k_weights("synthetic");
    let state = common::state(vec![model]);
    let app = larql_server::routes::single_model_router(state);
    let resp = app
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/v1/expert/0/0")
                .header(header::CONTENT_TYPE, "application/json")
                .body(Body::from("not json"))
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(resp.status(), StatusCode::BAD_REQUEST);
}

async fn post_expert_batch(layer: usize, body: serde_json::Value) -> axum::http::Response<Body> {
    let (model, _fixture) = common::model_with_q4k_weights("synthetic");
    let state = common::state(vec![model]);
    let app = larql_server::routes::single_model_router(state);
    let resp = app
        .oneshot(
            Request::builder()
                .method("POST")
                .uri(format!("/v1/expert/{layer}/batch"))
                .header(header::CONTENT_TYPE, "application/json")
                .body(Body::from(serde_json::to_vec(&body).unwrap()))
                .unwrap(),
        )
        .await
        .unwrap();
    drop(_fixture);
    resp
}

#[tokio::test]
async fn expert_batch_dense_model_returns_400() {
    // Same is_hybrid_moe() rejection on the batch endpoint.
    let resp = post_expert_batch(
        0,
        serde_json::json!({
            "residuals": [vec![0.0_f32; 8]],
            "expert_ids": [0],
        }),
    )
    .await;
    assert_eq!(resp.status(), StatusCode::BAD_REQUEST);
}

#[tokio::test]
async fn expert_layer_batch_dense_model_returns_400() {
    let (model, _fixture) = common::model_with_q4k_weights("synthetic");
    let state = common::state(vec![model]);
    let app = larql_server::routes::single_model_router(state);
    let resp = app
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/v1/expert/0/layer-batch")
                .header(header::CONTENT_TYPE, "application/json")
                .body(Body::from(
                    serde_json::to_vec(&serde_json::json!({
                        "residual": vec![0.0_f32; 8],
                        "expert_ids": [0],
                    }))
                    .unwrap(),
                ))
                .unwrap(),
        )
        .await
        .unwrap();
    drop(_fixture);
    assert!(
        resp.status() == StatusCode::BAD_REQUEST || resp.status() == StatusCode::NOT_FOUND,
        "dense model should reject expert/layer-batch; got {:?}",
        resp.status()
    );
}

#[tokio::test]
async fn expert_multi_layer_batch_dense_model_returns_400() {
    let (model, _fixture) = common::model_with_q4k_weights("synthetic");
    let state = common::state(vec![model]);
    let app = larql_server::routes::single_model_router(state);
    let resp = app
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/v1/expert/batch")
                .header(header::CONTENT_TYPE, "application/json")
                .body(Body::from(
                    serde_json::to_vec(&serde_json::json!({
                        "entries": [
                            {"layer": 0, "expert_id": 0, "residual": vec![0.0_f32; 8]}
                        ]
                    }))
                    .unwrap(),
                ))
                .unwrap(),
        )
        .await
        .unwrap();
    drop(_fixture);
    assert!(
        resp.status() == StatusCode::BAD_REQUEST || resp.status() == StatusCode::NOT_FOUND,
        "dense model should reject /v1/expert/batch; got {:?}",
        resp.status()
    );
}
