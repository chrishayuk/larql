//! `POST /v1/patches/apply` answers `applied` only for a patch that was
//! applied.
//!
//! The route used to call an infallible apply that discarded the checked
//! apply's result, so a patch the overlay dropped was answered 200
//! `"applied"` with an operation count. A refused patch is now a 400, and
//! the model, the session and the patch list are as they were.

mod common;
use common::*;

use axum::http::StatusCode;

const SESSION_HEADER: &str = "x-session-id";
/// Past the end of the test model's layers and features.
const OUT_OF_BOUNDS: usize = 1_000_000;

/// An inline patch whose one operation names a slot the model does not have.
fn out_of_bounds_patch(name: &str) -> serde_json::Value {
    serde_json::json!({
        "patch": {
            "version": 1,
            "base_model": "test",
            "base_checksum": null,
            "created_at": "2026-11-10",
            "description": name,
            "author": null,
            "tags": [],
            "operations": [
                {"op": "delete", "layer": 0, "feature": OUT_OF_BOUNDS}
            ]
        }
    })
}

/// A valid operation followed by one out of bounds: the first must not land.
fn half_valid_patch(name: &str) -> serde_json::Value {
    let mut patch = inline_delete_patch(name);
    patch["patch"]["operations"]
        .as_array_mut()
        .expect("operations")
        .push(serde_json::json!({"op": "delete", "layer": OUT_OF_BOUNDS, "feature": 0}));
    patch
}

async fn patch_names(st: &std::sync::Arc<larql_server::state::AppState>) -> Vec<String> {
    let resp = get(single_model_router(st.clone()), "/v1/patches").await;
    assert_eq!(resp.status(), StatusCode::OK);
    body_json(resp.into_body()).await["patches"]
        .as_array()
        .expect("patches")
        .iter()
        .map(|p| p["name"].as_str().unwrap_or_default().to_string())
        .collect()
}

#[tokio::test]
async fn a_refused_patch_is_a_400_and_is_not_listed() {
    let st = state(vec![model("test")]);

    let resp = post_json(
        single_model_router(st.clone()),
        "/v1/patches/apply",
        out_of_bounds_patch("refused"),
    )
    .await;

    assert_eq!(resp.status(), StatusCode::BAD_REQUEST);
    let body = body_json(resp.into_body()).await.to_string();
    assert!(body.contains("out of bounds"), "{body}");
    assert!(patch_names(&st).await.is_empty());
}

#[tokio::test]
async fn one_invalid_operation_stops_the_valid_ones_before_it() {
    let st = state(vec![model("test")]);

    let resp = post_json(
        single_model_router(st.clone()),
        "/v1/patches/apply",
        half_valid_patch("half"),
    )
    .await;

    assert_eq!(resp.status(), StatusCode::BAD_REQUEST);
    assert!(patch_names(&st).await.is_empty());
}

#[tokio::test]
async fn a_valid_patch_still_applies_after_a_refusal() {
    // The control for the two above: the route does not refuse everything.
    let st = state(vec![model("test")]);
    post_json(
        single_model_router(st.clone()),
        "/v1/patches/apply",
        out_of_bounds_patch("refused"),
    )
    .await;

    let resp = post_json(
        single_model_router(st.clone()),
        "/v1/patches/apply",
        inline_delete_patch("accepted"),
    )
    .await;

    assert_eq!(resp.status(), StatusCode::OK);
    assert_eq!(patch_names(&st).await, ["accepted"]);
}

#[tokio::test]
async fn a_refused_first_patch_leaves_no_session_behind() {
    // A session's first patch materialises its overlay. If that happened
    // before the patch was checked, a refusal would leave a session holding
    // an empty overlay, which reads as "patched".
    let st = state(vec![model("test")]);

    let resp = post_json_h(
        single_model_router(st.clone()),
        "/v1/patches/apply",
        out_of_bounds_patch("refused"),
        (SESSION_HEADER, "never-seen"),
    )
    .await;

    assert_eq!(resp.status(), StatusCode::BAD_REQUEST);
    assert!(st.sessions.get("never-seen").await.is_none());
}

#[tokio::test]
async fn a_refused_patch_leaves_a_sessions_existing_patches() {
    let st = state(vec![model("test")]);
    let apply = |patch| {
        post_json_h(
            single_model_router(st.clone()),
            "/v1/patches/apply",
            patch,
            (SESSION_HEADER, "stacked"),
        )
    };
    assert_eq!(
        apply(inline_delete_patch("first")).await.status(),
        StatusCode::OK
    );

    assert_eq!(
        apply(half_valid_patch("half")).await.status(),
        StatusCode::BAD_REQUEST
    );

    let listed = st.sessions.list_patches("stacked").await;
    assert_eq!(listed.len(), 1, "{listed:?}");
    assert_eq!(listed[0]["name"], "first");
}
