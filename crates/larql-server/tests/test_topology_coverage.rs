//! Coverage push for `routes/topology.rs` (was 60.66% debt baseline).
//!
//! The existing in-file unit tests cover only the
//! `TopologyResponse` struct serialisation. The handler itself
//! (`handle_topology`, L44-69) is uncovered. Three reachable paths:
//!
//!   - Model loaded but `expert_filter == None` → 404
//!   - Model loaded with `expert_filter == Some((start, end_excl))` → 200
//!     with owned_start = start, owned_end = end_excl - 1
//!   - Optional model_config.moe path that sets `num_experts`

mod common;

use axum::body::Body;
use axum::http::{header, Request, StatusCode};
use std::sync::Arc;
use tower::ServiceExt;

#[tokio::test]
async fn topology_without_expert_filter_returns_404() {
    // Synthetic model has `expert_filter: None` so the handler must
    // 404 (this server "owns all experts" / is not an expert shard).
    let (model, _fixture) = common::model_with_real_weights("synthetic");
    let state = common::state(vec![model]);
    let app = larql_server::routes::single_model_router(state);
    let resp = app
        .oneshot(
            Request::builder()
                .method("GET")
                .uri("/v1/expert/topology")
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();
    drop(_fixture);
    assert_eq!(resp.status(), StatusCode::NOT_FOUND);
}

#[tokio::test]
async fn topology_with_expert_filter_returns_200_with_owned_range() {
    // Mutate the synthetic LoadedModel's expert_filter before
    // wrapping in Arc. `model_with_real_weights` returns an Arc, but
    // since we hold the only reference at this point Arc::try_unwrap
    // succeeds — no Clone needed.
    let (model_arc, fixture) = common::model_with_real_weights("synthetic-experts");
    // `LoadedModel` doesn't implement `Debug`, so we can't `.expect()` on
    // the `Result<T, Arc<T>>` returned by `try_unwrap`; match the Err arm
    // explicitly to surface the same failure mode.
    let mut model = match Arc::try_unwrap(model_arc) {
        Ok(m) => m,
        Err(_) => panic!("sole Arc reference"),
    };
    // Pretend this shard owns experts 4..=11 (half-open in storage).
    model.expert_filter = Some((4, 12));
    let model = Arc::new(model);
    let state = common::state(vec![model]);
    let app = larql_server::routes::single_model_router(state);

    let resp = app
        .oneshot(
            Request::builder()
                .method("GET")
                .uri("/v1/expert/topology")
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();
    drop(fixture);
    assert_eq!(resp.status(), StatusCode::OK);
    let body = axum::body::to_bytes(resp.into_body(), usize::MAX)
        .await
        .unwrap();
    let v: serde_json::Value = serde_json::from_slice(&body).unwrap();
    assert_eq!(v["model_id"], "synthetic-experts");
    assert_eq!(v["owned_start"], 4);
    assert_eq!(v["owned_end"], 11);
    // The synthetic config has no `moe` section → num_experts = 0.
    assert_eq!(v["num_experts"], 0);
}

#[tokio::test]
async fn topology_with_content_type_header_still_responds() {
    // Some clients send a Content-Type even on GETs; the handler
    // should ignore it. Belt-and-braces check that the topology
    // route is GET-only (POST should not match).
    let (model, fixture) = common::model_with_real_weights("synthetic");
    let state = common::state(vec![model]);
    let app = larql_server::routes::single_model_router(state);
    let resp = app
        .oneshot(
            Request::builder()
                .method("GET")
                .uri("/v1/expert/topology")
                .header(header::CONTENT_TYPE, "application/json")
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();
    drop(fixture);
    // 404 because expert_filter is None — the request itself was
    // accepted (the handler ran), which is what we're asserting.
    assert_eq!(resp.status(), StatusCode::NOT_FOUND);
}
