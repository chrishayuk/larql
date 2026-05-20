#![cfg(not(target_arch = "wasm32"))]
//! HTTP integration tests: patches apply/list/delete (global + session-scoped).

mod common;
use common::*;

use axum::http::StatusCode;

// ══════════════════════════════════════════════════════════════
// GET /v1/patches  •  DELETE /v1/patches/{name}
// ══════════════════════════════════════════════════════════════

#[tokio::test]
async fn http_patches_list_empty_returns_empty_array() {
    let app = single_model_router(state(vec![model("test")]));
    let resp = get(app, "/v1/patches").await;
    assert_eq!(resp.status(), StatusCode::OK);
    let body = body_json(resp.into_body()).await;
    let patches = body["patches"].as_array().unwrap();
    assert!(patches.is_empty());
}

#[tokio::test]
async fn http_patches_delete_nonexistent_returns_404() {
    let app = single_model_router(state(vec![model("test")]));
    let resp = delete(app, "/v1/patches/nonexistent-patch").await;
    assert_eq!(resp.status(), StatusCode::NOT_FOUND);
}

#[tokio::test]
async fn http_patches_session_list_returns_session_field() {
    let app = single_model_router(state(vec![model("test")]));
    let resp = get_h(app, "/v1/patches", ("x-session-id", "sess-abc")).await;
    assert_eq!(resp.status(), StatusCode::OK);
    let body = body_json(resp.into_body()).await;
    assert_eq!(body["session"], "sess-abc");
    assert!(body["patches"].as_array().unwrap().is_empty());
}

// ══════════════════════════════════════════════════════════════
// POST /v1/patches/apply  •  GET /v1/patches  •  DELETE /v1/patches/{name}
// ══════════════════════════════════════════════════════════════

#[tokio::test]
async fn http_patches_apply_no_url_no_patch_returns_400() {
    let app = single_model_router(state(vec![model("test")]));
    let resp = post_json(app, "/v1/patches/apply", serde_json::json!({})).await;
    assert_eq!(resp.status(), StatusCode::BAD_REQUEST);
    let body = body_json(resp.into_body()).await;
    assert!(body["error"].as_str().unwrap().contains("url"));
}

#[tokio::test]
async fn http_patches_apply_inline_returns_200() {
    let app = single_model_router(state(vec![model("test")]));
    let resp = post_json(app, "/v1/patches/apply", inline_delete_patch("my-patch")).await;
    assert_eq!(resp.status(), StatusCode::OK);
    let body = body_json(resp.into_body()).await;
    assert_eq!(body["applied"], "my-patch");
    assert!(body["active_patches"].as_u64().is_some());
}

#[tokio::test]
async fn http_patches_list_after_apply_shows_patch() {
    let st = state(vec![model("test")]);
    // Apply the patch.
    let app1 = single_model_router(st.clone());
    post_json(
        app1,
        "/v1/patches/apply",
        inline_delete_patch("visible-patch"),
    )
    .await;
    // List patches.
    let app2 = single_model_router(st.clone());
    let resp = get(app2, "/v1/patches").await;
    assert_eq!(resp.status(), StatusCode::OK);
    let body = body_json(resp.into_body()).await;
    let patches = body["patches"].as_array().unwrap();
    assert!(patches.iter().any(|p| p["name"] == "visible-patch"));
}

#[tokio::test]
async fn http_patches_delete_named_returns_200() {
    let st = state(vec![model("test")]);
    // Apply, then delete.
    let app1 = single_model_router(st.clone());
    post_json(app1, "/v1/patches/apply", inline_delete_patch("to-delete")).await;
    let app2 = single_model_router(st.clone());
    let resp = delete(app2, "/v1/patches/to-delete").await;
    assert_eq!(resp.status(), StatusCode::OK);
    let body = body_json(resp.into_body()).await;
    assert_eq!(body["removed"], "to-delete");
    assert!(body["active_patches"].as_u64().is_some());
}

#[tokio::test]
async fn http_patches_session_apply_returns_session_field() {
    // apply_patch uses blocking_read when creating a new session inside an async
    // write-lock guard, which panics. Pre-create the session via get_or_create
    // (uses read().await, safe) so the entry already exists when the HTTP handler
    // calls apply_patch, skipping the blocking_read path entirely.
    let st = state(vec![model("test")]);
    let m = st.models[0].clone();
    st.sessions.get_or_create("sid-abc", &m).await;

    let app = single_model_router(st);
    let resp = post_json_h(
        app,
        "/v1/patches/apply",
        inline_delete_patch("sess-patch"),
        ("x-session-id", "sid-abc"),
    )
    .await;
    assert_eq!(resp.status(), StatusCode::OK);
    let body = body_json(resp.into_body()).await;
    assert_eq!(body["session"], "sid-abc");
    assert!(body["active_patches"].as_u64().is_some());
}

#[tokio::test]
async fn http_patches_session_list_after_session_apply() {
    let st = state(vec![model("test")]);
    let m = st.models[0].clone();
    st.sessions.get_or_create("sid-list", &m).await;

    let app1 = single_model_router(st.clone());
    post_json_h(
        app1,
        "/v1/patches/apply",
        inline_delete_patch("session-visible"),
        ("x-session-id", "sid-list"),
    )
    .await;
    let app2 = single_model_router(st.clone());
    let resp = get_h(app2, "/v1/patches", ("x-session-id", "sid-list")).await;
    assert_eq!(resp.status(), StatusCode::OK);
    let body = body_json(resp.into_body()).await;
    assert_eq!(body["session"], "sid-list");
    let patches = body["patches"].as_array().unwrap();
    assert!(patches.iter().any(|p| p["name"] == "session-visible"));
}

#[tokio::test]
async fn http_patches_multi_model_apply_not_found_returns_404() {
    let app = multi_model_router(state(vec![model("a")]));
    let resp = post_json(
        app,
        "/v1/nosuchmodel/patches/apply",
        inline_delete_patch("p"),
    )
    .await;
    assert_eq!(resp.status(), StatusCode::NOT_FOUND);
}
