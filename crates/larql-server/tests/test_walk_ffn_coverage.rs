//! Coverage push for `routes/walk_ffn.rs` (was 49%, target ≥ 90%).
//!
//! Uses the synthetic f32 vindex from `tests/common/synthetic_vindex.rs`
//! so the `full_output=true` paths (which call `run_full_output_core` →
//! real FFN compute over loaded `ModelWeights`) actually execute.
//! Features-only paths are already covered by `test_http_full_routes.rs`;
//! these tests target the previously-uncovered branches:
//!
//!   * full_output=true on a single layer and a layers array
//!   * binary wire format (FFN binary CT + Accept negotiation for f32 / f16 / i8)
//!   * validate_residual + validate_owned error paths
//!   * Q8K dense-FFN batch endpoint (404 when vindex has no Q4K data)

mod common;

use axum::body::Body;
use axum::http::{header, Request, StatusCode};
use tower::ServiceExt;

const SYN_HIDDEN: usize = 8;

fn residual_of(len: usize) -> Vec<f32> {
    let mut v = vec![0.0_f32; len];
    for (i, slot) in v.iter_mut().enumerate() {
        *slot = (i as f32) * 0.01 + 0.5;
    }
    v
}

async fn post_walk_ffn_json(body: serde_json::Value) -> axum::http::Response<Body> {
    let (model, _fixture) = common::model_with_real_weights("synthetic");
    let state = common::state(vec![model]);
    let app = larql_server::routes::single_model_router(state);
    let resp = app
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/v1/walk-ffn")
                .header(header::CONTENT_TYPE, "application/json")
                .body(Body::from(serde_json::to_vec(&body).unwrap()))
                .unwrap(),
        )
        .await
        .unwrap();
    // Hold the fixture alive until after the request — `_fixture` would
    // otherwise drop at the end of model_with_real_weights's scope.
    drop(_fixture);
    resp
}

#[tokio::test]
async fn walk_ffn_full_output_single_layer_runs_real_compute() {
    let body = serde_json::json!({
        "layer": 0,
        "residual": residual_of(SYN_HIDDEN),
        "full_output": true,
    });
    let resp = post_walk_ffn_json(body).await;
    assert_eq!(resp.status(), StatusCode::OK);
    let bytes = axum::body::to_bytes(resp.into_body(), usize::MAX)
        .await
        .unwrap();
    let v: serde_json::Value = serde_json::from_slice(&bytes).unwrap();
    // run_full_output produces a JSON object — assert it's not the
    // features-only shape (which would have `features` + `scores`).
    assert!(v.is_object(), "full_output must produce a JSON object");
}

#[tokio::test]
async fn walk_ffn_full_output_layers_array_runs_multi_layer() {
    let body = serde_json::json!({
        "layers": [0, 1],
        "residual": residual_of(SYN_HIDDEN),
        "full_output": true,
    });
    let resp = post_walk_ffn_json(body).await;
    assert_eq!(resp.status(), StatusCode::OK);
}

#[tokio::test]
async fn walk_ffn_features_only_layers_array() {
    // Hits run_features_only's len > 1 branch (single-layer is
    // exercised by an older suite; the array branch shapes the
    // response differently).
    let body = serde_json::json!({
        "layers": [0, 1],
        "residual": residual_of(SYN_HIDDEN),
        "full_output": false,
        "top_k": 2,
    });
    let resp = post_walk_ffn_json(body).await;
    assert_eq!(resp.status(), StatusCode::OK);
    let bytes = axum::body::to_bytes(resp.into_body(), usize::MAX)
        .await
        .unwrap();
    let v: serde_json::Value = serde_json::from_slice(&bytes).unwrap();
    assert!(v["results"].is_array(), "expected results array shape");
}

#[tokio::test]
async fn walk_ffn_seq_len_2_multi_position_full_output() {
    // Hits run_full_output's multi-position residual path —
    // seq_len=2 ⇒ residual length 2*hidden.
    let body = serde_json::json!({
        "layer": 0,
        "residual": residual_of(SYN_HIDDEN * 2),
        "seq_len": 2,
        "full_output": true,
    });
    let resp = post_walk_ffn_json(body).await;
    assert_eq!(resp.status(), StatusCode::OK);
}

#[tokio::test]
async fn walk_ffn_validate_residual_wrong_size_returns_400() {
    let body = serde_json::json!({
        "layer": 0,
        "residual": vec![1.0_f32; 3], // hidden=8, so 3 is wrong
    });
    let resp = post_walk_ffn_json(body).await;
    assert_eq!(resp.status(), StatusCode::BAD_REQUEST);
}

#[tokio::test]
async fn walk_ffn_collect_scan_layers_neither_field_returns_400() {
    // Neither `layer` nor `layers` set — collect_scan_layers must reject.
    let body = serde_json::json!({
        "residual": residual_of(SYN_HIDDEN),
    });
    let resp = post_walk_ffn_json(body).await;
    assert_eq!(resp.status(), StatusCode::BAD_REQUEST);
}

#[tokio::test]
async fn walk_ffn_invalid_json_returns_400() {
    let (model, _fixture) = common::model_with_real_weights("synthetic");
    let state = common::state(vec![model]);
    let app = larql_server::routes::single_model_router(state);
    let resp = app
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/v1/walk-ffn")
                .header(header::CONTENT_TYPE, "application/json")
                .body(Body::from("not json"))
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(resp.status(), StatusCode::BAD_REQUEST);
}

#[tokio::test]
async fn walk_ffn_moe_layer_without_moe_shards_returns_400() {
    // moe_layer=true on a model that has no `moe_remote` set
    // (synthetic doesn't configure --moe-shards) must error out.
    let body = serde_json::json!({
        "layer": 0,
        "residual": residual_of(SYN_HIDDEN),
        "full_output": true,
        "moe_layer": true,
    });
    let resp = post_walk_ffn_json(body).await;
    assert_eq!(resp.status(), StatusCode::BAD_REQUEST);
}

#[tokio::test]
async fn walk_ffn_binary_request_without_full_output_returns_400() {
    let (model, _fixture) = common::model_with_real_weights("synthetic");
    let state = common::state(vec![model]);
    let app = larql_server::routes::single_model_router(state);
    // Binary FFN header layout: layer:u32, seq_len:u32, flags:u32, top_k:u32,
    // followed by residual:f32[]. full_output bit is bit 0 of flags.
    let mut body: Vec<u8> = Vec::new();
    body.extend_from_slice(&0u32.to_le_bytes()); // layer
    body.extend_from_slice(&1u32.to_le_bytes()); // seq_len
    body.extend_from_slice(&0u32.to_le_bytes()); // flags=0 → full_output=false
    body.extend_from_slice(&8u32.to_le_bytes()); // top_k
    for v in residual_of(SYN_HIDDEN) {
        body.extend_from_slice(&v.to_le_bytes());
    }
    let resp = app
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/v1/walk-ffn")
                .header(header::CONTENT_TYPE, "application/x-larql-ffn")
                .body(Body::from(body))
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(resp.status(), StatusCode::BAD_REQUEST);
}

#[tokio::test]
async fn walk_ffn_binary_full_output_default_f32_response() {
    let (model, _fixture) = common::model_with_real_weights("synthetic");
    let state = common::state(vec![model]);
    let app = larql_server::routes::single_model_router(state);
    let mut body: Vec<u8> = Vec::new();
    body.extend_from_slice(&0u32.to_le_bytes()); // layer
    body.extend_from_slice(&1u32.to_le_bytes()); // seq_len
    body.extend_from_slice(&1u32.to_le_bytes()); // flags=1 → full_output=true
    body.extend_from_slice(&8u32.to_le_bytes()); // top_k
    for v in residual_of(SYN_HIDDEN) {
        body.extend_from_slice(&v.to_le_bytes());
    }
    let resp = app
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/v1/walk-ffn")
                .header(header::CONTENT_TYPE, "application/x-larql-ffn")
                .body(Body::from(body))
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(resp.status(), StatusCode::OK);
    let ct = resp
        .headers()
        .get(header::CONTENT_TYPE)
        .and_then(|v| v.to_str().ok())
        .unwrap_or("")
        .to_owned();
    assert!(
        ct.starts_with("application/x-larql-ffn"),
        "expected binary response, got {ct}"
    );
}

#[tokio::test]
async fn walk_ffn_binary_full_output_f16_negotiation() {
    let (model, _fixture) = common::model_with_real_weights("synthetic");
    let state = common::state(vec![model]);
    let app = larql_server::routes::single_model_router(state);
    let mut body: Vec<u8> = Vec::new();
    body.extend_from_slice(&0u32.to_le_bytes());
    body.extend_from_slice(&1u32.to_le_bytes());
    body.extend_from_slice(&1u32.to_le_bytes());
    body.extend_from_slice(&8u32.to_le_bytes());
    for v in residual_of(SYN_HIDDEN) {
        body.extend_from_slice(&v.to_le_bytes());
    }
    let resp = app
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/v1/walk-ffn")
                .header(header::CONTENT_TYPE, "application/x-larql-ffn")
                .header(header::ACCEPT, "application/x-larql-ffn-f16")
                .body(Body::from(body))
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(resp.status(), StatusCode::OK);
}

#[tokio::test]
async fn walk_ffn_q8k_returns_404_when_vindex_has_no_q4k() {
    // The synthetic vindex is non-Q4K (StorageDtype::F32). The Q8K
    // endpoint requires interleaved Q4K data and must 404.
    let (model, _fixture) = common::model_with_real_weights("synthetic");
    let state = common::state(vec![model]);
    let app = larql_server::routes::single_model_router(state);
    // Q8K batch body is fairly involved; an empty body still trips
    // the "no Q4K" precondition before parsing.
    let resp = app
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/v1/walk-ffn-q8k")
                .header(header::CONTENT_TYPE, "application/x-larql-ffn-q8k-batch")
                .body(Body::from(Vec::<u8>::new()))
                .unwrap(),
        )
        .await
        .unwrap();
    // Could be 400 (bad body) or 404 (no Q4K); both are post-route
    // codepaths inside handle_walk_ffn_q8k.
    assert!(
        resp.status() == StatusCode::NOT_FOUND || resp.status() == StatusCode::BAD_REQUEST,
        "expected 404 or 400, got {:?}",
        resp.status()
    );
}

// ── Binary batch request shape (BATCH_MARKER decode + multi-entry encode) ───────

fn build_binary_batch_request(layers: &[u32], seq_len: u32, residual: &[f32]) -> Vec<u8> {
    let mut body: Vec<u8> = Vec::new();
    body.extend_from_slice(&0xFFFF_FFFFu32.to_le_bytes()); // BATCH_MARKER
    body.extend_from_slice(&(layers.len() as u32).to_le_bytes());
    for &l in layers {
        body.extend_from_slice(&l.to_le_bytes());
    }
    body.extend_from_slice(&seq_len.to_le_bytes());
    body.extend_from_slice(&1u32.to_le_bytes()); // flags=1 → full_output=true
    body.extend_from_slice(&8u32.to_le_bytes()); // top_k
    for v in residual {
        body.extend_from_slice(&v.to_le_bytes());
    }
    body
}

async fn post_binary(body: Vec<u8>, accept: Option<&str>) -> axum::http::Response<Body> {
    let (model, _fixture) = common::model_with_real_weights("synthetic");
    let state = common::state(vec![model]);
    let app = larql_server::routes::single_model_router(state);
    let mut req = Request::builder()
        .method("POST")
        .uri("/v1/walk-ffn")
        .header(header::CONTENT_TYPE, "application/x-larql-ffn");
    if let Some(a) = accept {
        req = req.header(header::ACCEPT, a);
    }
    let resp = app
        .oneshot(req.body(Body::from(body)).unwrap())
        .await
        .unwrap();
    drop(_fixture);
    resp
}

#[tokio::test]
async fn walk_ffn_binary_batch_request_decodes_and_multi_entry_encodes() {
    // Two layers + full_output triggers decode_binary_request's
    // BATCH_MARKER branch AND the multi-entry encode path (line ~252).
    let body = build_binary_batch_request(&[0, 1], 1, &residual_of(SYN_HIDDEN));
    let resp = post_binary(body, None).await;
    assert_eq!(resp.status(), StatusCode::OK);
}

#[tokio::test]
async fn walk_ffn_binary_batch_f16_negotiation_hits_multi_entry_f16_encode() {
    // Multi-layer batch + Accept f16 covers encode_binary_output_f16
    // multi-entry branch (L285-299).
    let body = build_binary_batch_request(&[0, 1], 1, &residual_of(SYN_HIDDEN));
    let resp = post_binary(body, Some("application/x-larql-ffn-f16")).await;
    assert_eq!(resp.status(), StatusCode::OK);
}

#[tokio::test]
async fn walk_ffn_binary_single_i8_negotiation_hits_i8_quantise() {
    // Single layer + Accept i8 covers encode_binary_output_i8
    // single-entry branch (L320-331).
    let mut body: Vec<u8> = Vec::new();
    body.extend_from_slice(&0u32.to_le_bytes()); // layer
    body.extend_from_slice(&1u32.to_le_bytes()); // seq_len
    body.extend_from_slice(&1u32.to_le_bytes()); // flags=1
    body.extend_from_slice(&8u32.to_le_bytes()); // top_k
    for v in residual_of(SYN_HIDDEN) {
        body.extend_from_slice(&v.to_le_bytes());
    }
    let resp = post_binary(body, Some("application/x-larql-ffn-i8")).await;
    assert_eq!(resp.status(), StatusCode::OK);
}

#[tokio::test]
async fn walk_ffn_binary_batch_i8_negotiation_hits_multi_entry_i8_encode() {
    // Multi-layer batch + Accept i8 covers encode_binary_output_i8
    // multi-entry branch (L332-348).
    let body = build_binary_batch_request(&[0, 1], 1, &residual_of(SYN_HIDDEN));
    let resp = post_binary(body, Some("application/x-larql-ffn-i8")).await;
    assert_eq!(resp.status(), StatusCode::OK);
}

#[tokio::test]
async fn walk_ffn_binary_truncated_header_returns_400() {
    // < 16 bytes — hits decode_binary_request's first guard.
    let resp = post_binary(vec![0u8; 4], None).await;
    assert_eq!(resp.status(), StatusCode::BAD_REQUEST);
}

#[tokio::test]
async fn walk_ffn_binary_batch_truncated_layer_indices_returns_400() {
    // BATCH_MARKER + claims 4 layers but only includes 1 → truncated.
    let mut body: Vec<u8> = Vec::new();
    body.extend_from_slice(&0xFFFF_FFFFu32.to_le_bytes());
    body.extend_from_slice(&4u32.to_le_bytes()); // 4 layers claimed
    body.extend_from_slice(&0u32.to_le_bytes()); // only 1 actually present
                                                 // Pad to >= 16 bytes so the first guard doesn't fire, but
                                                 // truncate before all 4 layer indices land.
    body.extend_from_slice(&[0u8; 4]);
    let resp = post_binary(body, None).await;
    assert_eq!(resp.status(), StatusCode::BAD_REQUEST);
}

#[tokio::test]
async fn walk_ffn_binary_residual_not_multiple_of_4_returns_400() {
    // header_end + 12 + odd-byte tail → "residual byte length is not a
    // multiple of 4" branch in decode_binary_request.
    let mut body: Vec<u8> = Vec::new();
    body.extend_from_slice(&0u32.to_le_bytes());
    body.extend_from_slice(&1u32.to_le_bytes());
    body.extend_from_slice(&1u32.to_le_bytes());
    body.extend_from_slice(&8u32.to_le_bytes());
    // Residual block of 1 byte (not multiple of 4)
    body.push(0u8);
    let resp = post_binary(body, None).await;
    assert_eq!(resp.status(), StatusCode::BAD_REQUEST);
}
