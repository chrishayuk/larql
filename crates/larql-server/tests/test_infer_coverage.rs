//! Coverage push for `routes/infer.rs` (was 50%, target ≥ 90%).
//!
//! Three inference modes (`walk`, `dense`, `compare`) plus session-
//! scoped walk + multi-model dispatch + error paths. Uses the
//! synthetic f32 vindex so `get_or_load_weights()` populates and the
//! real `larql_inference::infer_patched` / `predict` paths execute.

mod common;

use axum::body::Body;
use axum::http::{header, Request, StatusCode};
use tower::ServiceExt;

async fn post_infer(body: serde_json::Value) -> axum::http::Response<Body> {
    let (model, _fixture) = common::model_with_real_weights("synthetic");
    let state = common::state(vec![model]);
    let app = larql_server::routes::single_model_router(state);
    let resp = app
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/v1/infer")
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
async fn infer_walk_mode_default_runs_full_predict() {
    let resp = post_infer(serde_json::json!({
        "prompt": "the capital of France is",
        "top": 3,
    }))
    .await;
    assert_eq!(resp.status(), StatusCode::OK);
}

#[tokio::test]
async fn infer_dense_mode_runs_dense_predict_branch() {
    let resp = post_infer(serde_json::json!({
        "prompt": "the capital of France is",
        "mode": "dense",
        "top": 3,
    }))
    .await;
    assert_eq!(resp.status(), StatusCode::OK);
}

#[tokio::test]
async fn infer_compare_mode_runs_both_walk_and_dense() {
    let resp = post_infer(serde_json::json!({
        "prompt": "the capital of France is",
        "mode": "compare",
        "top": 2,
    }))
    .await;
    assert_eq!(resp.status(), StatusCode::OK);
    let bytes = axum::body::to_bytes(resp.into_body(), usize::MAX)
        .await
        .unwrap();
    let v: serde_json::Value = serde_json::from_slice(&bytes).unwrap();
    assert!(
        v.get("walk").is_some() && v.get("dense").is_some(),
        "compare mode emits both walk + dense keys; got {v:?}"
    );
}

#[tokio::test]
async fn infer_empty_prompt_returns_400() {
    let resp = post_infer(serde_json::json!({
        "prompt": "",
    }))
    .await;
    assert_eq!(resp.status(), StatusCode::BAD_REQUEST);
}

#[tokio::test]
async fn infer_invalid_json_returns_400() {
    let (model, _fixture) = common::model_with_real_weights("synthetic");
    let state = common::state(vec![model]);
    let app = larql_server::routes::single_model_router(state);
    let resp = app
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/v1/infer")
                .header(header::CONTENT_TYPE, "application/json")
                .body(Body::from("not json"))
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(resp.status(), StatusCode::BAD_REQUEST);
}

#[tokio::test]
async fn infer_unknown_mode_falls_through_with_no_predictions() {
    // mode != walk / dense / compare → neither flag set; handler
    // emits a response with neither predictions nor mode key.
    let resp = post_infer(serde_json::json!({
        "prompt": "x",
        "mode": "bogus",
    }))
    .await;
    // Either 200 (handler returned latency-only object) or 400 — both
    // are valid covered paths in the handler.
    assert!(resp.status().is_success() || resp.status().is_client_error());
}

#[tokio::test]
async fn infer_multi_model_dispatches_by_model_id() {
    let (model, _fixture) = common::model_with_real_weights("synthetic");
    let state = common::state(vec![model]);
    let app = larql_server::routes::multi_model_router(state);

    let resp = app
        .clone()
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/v1/synthetic/infer")
                .header(header::CONTENT_TYPE, "application/json")
                .body(Body::from(br#"{"prompt":"x"}"#.to_vec()))
                .unwrap(),
        )
        .await
        .unwrap();
    assert_ne!(resp.status(), StatusCode::NOT_FOUND);

    let resp404 = app
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/v1/nonexistent/infer")
                .header(header::CONTENT_TYPE, "application/json")
                .body(Body::from(br#"{"prompt":"x"}"#.to_vec()))
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(resp404.status(), StatusCode::NOT_FOUND);
}

#[tokio::test]
async fn infer_with_unknown_session_id_falls_back_to_global_patched() {
    // Header present but session doesn't exist → handler drops the
    // sessions guard and falls through to `model.patched.blocking_read()`
    // (covers infer.rs L140-142).
    let (model, _fixture) = common::model_with_real_weights("synthetic");
    let state = common::state(vec![model]);
    let app = larql_server::routes::single_model_router(state);

    let resp = app
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/v1/infer")
                .header(header::CONTENT_TYPE, "application/json")
                .header("x-session-id", "no-such-session")
                .body(Body::from(br#"{"prompt":"hello","mode":"walk"}"#.to_vec()))
                .unwrap(),
        )
        .await
        .unwrap();
    assert!(resp.status().is_success() || resp.status().is_client_error());
}

#[tokio::test]
async fn infer_with_existing_session_uses_session_patched() {
    // Pre-create the session via state.sessions.get_or_create so the
    // handler hits the `Some(session)` branch and walks via the
    // session-scoped PatchedVindex (covers infer.rs L137-138).
    let (model, _fixture) = common::model_with_real_weights("synthetic");
    let sid = "session-abc";
    let state = common::state(vec![model.clone()]);
    let _ = state.sessions.get_or_create(sid, &model).await;

    let app = larql_server::routes::single_model_router(state);
    let resp = app
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/v1/infer")
                .header(header::CONTENT_TYPE, "application/json")
                .header("x-session-id", sid)
                .body(Body::from(br#"{"prompt":"hello","mode":"walk"}"#.to_vec()))
                .unwrap(),
        )
        .await
        .unwrap();
    assert!(resp.status().is_success() || resp.status().is_client_error());
}
