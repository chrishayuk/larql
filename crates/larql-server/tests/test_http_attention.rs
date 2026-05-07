//! End-to-end HTTP tests for the `/v1/attention/*` and
//! `/v1/kv-cache/*` routes added by the `attention-service-routes`
//! change.
//!
//! Exercises the slice that ships before the prefill/decode runner
//! lands: session lifecycle, snapshot/restore round-trip, free, and
//! the role-aware error responses (404 for unknown id, 400 for
//! malformed snapshot, 503 for cap exhaustion).

mod common;
use common::*;

use axum::http::StatusCode;

// ── helpers ────────────────────────────────────────────────────────────────

async fn create_session(app: &axum::Router, model_id: &str) -> serde_json::Value {
    let resp = post_json(
        app.clone(),
        "/v1/attention/session",
        serde_json::json!({"model_id": model_id}),
    )
    .await;
    assert_eq!(resp.status(), StatusCode::OK, "create_session must succeed");
    body_json(resp.into_body()).await
}

async fn create_session_with_format(
    app: &axum::Router,
    model_id: &str,
    fmt: &str,
) -> serde_json::Value {
    let resp = post_json(
        app.clone(),
        "/v1/attention/session",
        serde_json::json!({"model_id": model_id, "kv_format": fmt}),
    )
    .await;
    assert_eq!(resp.status(), StatusCode::OK);
    body_json(resp.into_body()).await
}

// ── create / get / delete ──────────────────────────────────────────────────

#[tokio::test]
async fn create_session_returns_id_and_layer_range() {
    let app = single_model_router(state(vec![model("test")]));
    let body = create_session(&app, "test").await;
    assert!(body["session_id"].is_string());
    let sid = body["session_id"].as_str().unwrap();
    assert_eq!(sid.len(), 26, "session id is a 26-char ULID");
    assert_eq!(body["kv_format"], "fp32");
    assert!(body["layer_range"].is_array());
}

#[tokio::test]
async fn create_session_for_unknown_model_returns_404() {
    let app = single_model_router(state(vec![model("test")]));
    let resp = post_json(
        app,
        "/v1/attention/session",
        serde_json::json!({"model_id": "no-such-model"}),
    )
    .await;
    assert_eq!(resp.status(), StatusCode::NOT_FOUND);
    let body = body_json(resp.into_body()).await;
    assert_eq!(body["error"], "no_such_model");
}

#[tokio::test]
async fn create_session_with_unknown_kv_format_returns_400() {
    let app = single_model_router(state(vec![model("test")]));
    let resp = post_json(
        app,
        "/v1/attention/session",
        serde_json::json!({"model_id": "test", "kv_format": "nf4"}),
    )
    .await;
    assert_eq!(resp.status(), StatusCode::BAD_REQUEST);
    let body = body_json(resp.into_body()).await;
    assert_eq!(body["error"], "kv_format_unknown");
}

#[tokio::test]
async fn get_session_returns_state() {
    let st = state(vec![model("test")]);
    let app = single_model_router(st.clone());

    let body = create_session_with_format(&app, "test", "iso3").await;
    let sid = body["session_id"].as_str().unwrap().to_string();

    // Re-arm the router with a fresh tower::ServiceExt — oneshot consumes.
    let app2 = single_model_router(st);
    let resp = get(app2, &format!("/v1/attention/session/{sid}")).await;
    assert_eq!(resp.status(), StatusCode::OK);
    let got = body_json(resp.into_body()).await;
    assert_eq!(got["session_id"], sid);
    assert_eq!(got["model_id"], "test");
    assert_eq!(got["kv_format"], "iso3");
    assert_eq!(got["seq_len"], 0);
    assert_eq!(got["prefilled"], false);
}

#[tokio::test]
async fn get_unknown_session_returns_404() {
    let app = single_model_router(state(vec![model("test")]));
    let resp = get(app, "/v1/attention/session/01HM1MISSING0000000000000A").await;
    assert_eq!(resp.status(), StatusCode::NOT_FOUND);
    let body = body_json(resp.into_body()).await;
    assert_eq!(body["error"], "no_such_session");
}

#[tokio::test]
async fn delete_session_then_get_returns_404() {
    let st = state(vec![model("test")]);
    let app = single_model_router(st.clone());

    let body = create_session(&app, "test").await;
    let sid = body["session_id"].as_str().unwrap().to_string();

    let app2 = single_model_router(st.clone());
    let resp = delete(app2, &format!("/v1/attention/session/{sid}")).await;
    assert_eq!(resp.status(), StatusCode::NO_CONTENT);

    let app3 = single_model_router(st);
    let resp = get(app3, &format!("/v1/attention/session/{sid}")).await;
    assert_eq!(resp.status(), StatusCode::NOT_FOUND);
}

#[tokio::test]
async fn delete_unknown_session_returns_404() {
    let app = single_model_router(state(vec![model("test")]));
    let resp = delete(app, "/v1/attention/session/01HM1MISSING0000000000000A").await;
    assert_eq!(resp.status(), StatusCode::NOT_FOUND);
}

// ── snapshot / restore ─────────────────────────────────────────────────────

#[tokio::test]
async fn snapshot_returns_base64_with_correct_magic() {
    let st = state(vec![model("test")]);
    let app = single_model_router(st.clone());
    let body = create_session(&app, "test").await;
    let sid = body["session_id"].as_str().unwrap().to_string();

    let app2 = single_model_router(st);
    let resp = post_json(
        app2,
        "/v1/kv-cache/snapshot",
        serde_json::json!({"session_id": sid}),
    )
    .await;
    assert_eq!(resp.status(), StatusCode::OK);
    let body = body_json(resp.into_body()).await;
    let b64 = body["snapshot"].as_str().expect("snapshot is base64");
    use base64::Engine;
    let bytes = base64::engine::general_purpose::STANDARD
        .decode(b64)
        .expect("base64 decodes");
    // Magic bytes: 0x4C 0x41 0x51 0x41 → little-endian "LAQA".
    assert_eq!(&bytes[..4], &[0x41, 0x51, 0x41, 0x4C]);
    assert_eq!(&bytes[4..6], &[0x01, 0x00], "version is 1");
}

#[tokio::test]
async fn restore_round_trips_through_a_new_session() {
    let st = state(vec![model("test")]);
    let app = single_model_router(st.clone());
    let body = create_session(&app, "test").await;
    let sid_a = body["session_id"].as_str().unwrap().to_string();

    // Snapshot the (empty) session.
    let app2 = single_model_router(st.clone());
    let resp = post_json(
        app2,
        "/v1/kv-cache/snapshot",
        serde_json::json!({"session_id": sid_a}),
    )
    .await;
    let snap = body_json(resp.into_body()).await;
    let snap_b64 = snap["snapshot"].as_str().unwrap().to_string();

    // Create a brand-new session restored from that snapshot.
    let app3 = single_model_router(st.clone());
    let resp = post_json(
        app3,
        "/v1/attention/session",
        serde_json::json!({
            "model_id": "test",
            "restore_from_snapshot": snap_b64,
        }),
    )
    .await;
    assert_eq!(resp.status(), StatusCode::OK);
    let body = body_json(resp.into_body()).await;
    let sid_b = body["session_id"].as_str().unwrap();
    assert_ne!(sid_b, &sid_a, "restore allocates a fresh id");
}

#[tokio::test]
async fn restore_with_bad_magic_returns_400_with_typed_error() {
    let st = state(vec![model("test")]);
    let app = single_model_router(st.clone());
    let body = create_session(&app, "test").await;
    let sid = body["session_id"].as_str().unwrap().to_string();

    use base64::Engine;
    let bad = base64::engine::general_purpose::STANDARD.encode(vec![0u8; 64]);

    let app2 = single_model_router(st);
    let resp = post_json(
        app2,
        "/v1/kv-cache/restore",
        serde_json::json!({"session_id": sid, "snapshot": bad}),
    )
    .await;
    assert_eq!(resp.status(), StatusCode::BAD_REQUEST);
    let body = body_json(resp.into_body()).await;
    assert_eq!(body["error"], "snapshot_magic_mismatch");
}

#[tokio::test]
async fn restore_with_bad_base64_returns_400() {
    let st = state(vec![model("test")]);
    let app = single_model_router(st.clone());
    let body = create_session(&app, "test").await;
    let sid = body["session_id"].as_str().unwrap().to_string();

    let app2 = single_model_router(st);
    let resp = post_json(
        app2,
        "/v1/kv-cache/restore",
        serde_json::json!({"session_id": sid, "snapshot": "@@@bogus@@@"}),
    )
    .await;
    assert_eq!(resp.status(), StatusCode::BAD_REQUEST);
    let body = body_json(resp.into_body()).await;
    assert_eq!(body["error"], "snapshot_base64_decode_failed");
}

#[tokio::test]
async fn restore_unknown_session_returns_404() {
    let app = single_model_router(state(vec![model("test")]));
    let resp = post_json(
        app,
        "/v1/kv-cache/restore",
        serde_json::json!({
            "session_id": "01HM1MISSING0000000000000A",
            "snapshot": "",
        }),
    )
    .await;
    assert_eq!(resp.status(), StatusCode::NOT_FOUND);
}

// ── free ───────────────────────────────────────────────────────────────────

#[tokio::test]
async fn free_unknown_session_returns_404() {
    let app = single_model_router(state(vec![model("test")]));
    let resp = post_json(
        app,
        "/v1/kv-cache/free",
        serde_json::json!({"session_id": "01HM1MISSING0000000000000A"}),
    )
    .await;
    assert_eq!(resp.status(), StatusCode::NOT_FOUND);
}

#[tokio::test]
async fn free_layer_out_of_range_returns_400() {
    let st = state(vec![model("test")]);
    let app = single_model_router(st.clone());
    let body = create_session(&app, "test").await;
    let sid = body["session_id"].as_str().unwrap().to_string();

    let app2 = single_model_router(st);
    let resp = post_json(
        app2,
        "/v1/kv-cache/free",
        serde_json::json!({"session_id": sid, "layer": 999}),
    )
    .await;
    assert_eq!(resp.status(), StatusCode::BAD_REQUEST);
    let body = body_json(resp.into_body()).await;
    assert_eq!(body["error"], "layer_out_of_range");
}

#[tokio::test]
async fn free_all_layers_returns_layers_freed() {
    let st = state(vec![model("test")]);
    let app = single_model_router(st.clone());
    let body = create_session(&app, "test").await;
    let sid = body["session_id"].as_str().unwrap().to_string();

    let app2 = single_model_router(st);
    let resp = post_json(
        app2,
        "/v1/kv-cache/free",
        serde_json::json!({"session_id": sid}),
    )
    .await;
    assert_eq!(resp.status(), StatusCode::OK);
    let body = body_json(resp.into_body()).await;
    assert!(body["layers_freed"].as_u64().unwrap() >= 1);
}

// ── prefill / decode (501 stubs — runner integration pending) ─────────────

#[tokio::test]
async fn prefill_unknown_session_returns_404() {
    let app = single_model_router(state(vec![model("test")]));
    let resp = post_json(
        app,
        "/v1/attention/prefill",
        serde_json::json!({
            "session_id": "01HM1MISSING0000000000000A",
            "token_embeddings": [[0.0, 0.0, 0.0, 0.0]],
        }),
    )
    .await;
    assert_eq!(resp.status(), StatusCode::NOT_FOUND);
}

#[tokio::test]
async fn prefill_with_empty_embeddings_returns_400() {
    let st = state(vec![model("test")]);
    let app = single_model_router(st.clone());
    let body = create_session(&app, "test").await;
    let sid = body["session_id"].as_str().unwrap().to_string();

    let app2 = single_model_router(st);
    let resp = post_json(
        app2,
        "/v1/attention/prefill",
        serde_json::json!({
            "session_id": sid,
            "token_embeddings": [],
        }),
    )
    .await;
    assert_eq!(resp.status(), StatusCode::BAD_REQUEST);
    let body = body_json(resp.into_body()).await;
    assert_eq!(body["error"], "empty_token_embeddings");
}

#[tokio::test]
async fn prefill_with_ragged_rows_returns_400() {
    let st = state(vec![model("test")]);
    let app = single_model_router(st.clone());
    let body = create_session(&app, "test").await;
    let sid = body["session_id"].as_str().unwrap().to_string();

    let app2 = single_model_router(st);
    let resp = post_json(
        app2,
        "/v1/attention/prefill",
        serde_json::json!({
            "session_id": sid,
            "token_embeddings": [[0.0, 1.0, 2.0], [0.0]],
        }),
    )
    .await;
    assert_eq!(resp.status(), StatusCode::BAD_REQUEST);
    let body = body_json(resp.into_body()).await;
    assert_eq!(body["error"], "ragged_token_embeddings");
}

#[tokio::test]
async fn prefill_with_valid_shape_returns_501_with_typed_error() {
    let st = state(vec![model("test")]);
    let app = single_model_router(st.clone());
    let body = create_session(&app, "test").await;
    let sid = body["session_id"].as_str().unwrap().to_string();

    let app2 = single_model_router(st);
    let resp = post_json(
        app2,
        "/v1/attention/prefill",
        serde_json::json!({
            "session_id": sid,
            "token_embeddings": [[0.0, 0.1, 0.2, 0.3]],
        }),
    )
    .await;
    assert_eq!(resp.status(), StatusCode::NOT_IMPLEMENTED);
    let body = body_json(resp.into_body()).await;
    assert_eq!(body["error"], "prefill_not_implemented");
}

#[tokio::test]
async fn decode_unknown_session_returns_404() {
    let app = single_model_router(state(vec![model("test")]));
    let resp = post_json(
        app,
        "/v1/attention/decode",
        serde_json::json!({
            "session_id": "01HM1MISSING0000000000000A",
            "query_token_embedding": [0.0, 0.0, 0.0, 0.0],
        }),
    )
    .await;
    assert_eq!(resp.status(), StatusCode::NOT_FOUND);
}

#[tokio::test]
async fn decode_before_prefill_returns_400() {
    let st = state(vec![model("test")]);
    let app = single_model_router(st.clone());
    let body = create_session(&app, "test").await;
    let sid = body["session_id"].as_str().unwrap().to_string();

    let app2 = single_model_router(st);
    let resp = post_json(
        app2,
        "/v1/attention/decode",
        serde_json::json!({
            "session_id": sid,
            "query_token_embedding": [0.0, 0.0, 0.0, 0.0],
        }),
    )
    .await;
    assert_eq!(resp.status(), StatusCode::BAD_REQUEST);
    let body = body_json(resp.into_body()).await;
    assert_eq!(body["error"], "decode_before_prefill");
}

// ── format coverage ────────────────────────────────────────────────────────

#[tokio::test]
async fn create_session_supports_every_kv_format() {
    let st = state(vec![model("test")]);
    for fmt in ["fp32", "planar3", "planar4", "iso3", "iso4"] {
        let app = single_model_router(st.clone());
        let body = create_session_with_format(&app, "test", fmt).await;
        assert_eq!(body["kv_format"], fmt, "format {fmt} round-trips");
    }
}
