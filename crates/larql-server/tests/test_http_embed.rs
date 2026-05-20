#![cfg(not(target_arch = "wasm32"))]
//! HTTP integration tests: embed, logits, token encode/decode (single + multi).

mod common;
use common::*;

use axum::body::Body;
use axum::http::Request;
use axum::http::StatusCode;
use larql_server::http::BINARY_FFN_CONTENT_TYPE;
use tower::ServiceExt;

fn binary_embed_body(token_ids: &[u32]) -> Vec<u8> {
    let mut body = Vec::with_capacity(4 + token_ids.len() * 4);
    body.extend_from_slice(&(token_ids.len() as u32).to_le_bytes());
    for &token_id in token_ids {
        body.extend_from_slice(&token_id.to_le_bytes());
    }
    body
}

fn binary_logits_body(values: &[f32]) -> Vec<u8> {
    let mut body = Vec::with_capacity(values.len() * 4);
    for &value in values {
        body.extend_from_slice(&value.to_le_bytes());
    }
    body
}

async fn post_binary(app: axum::Router, path: &str, body: Vec<u8>) -> axum::http::Response<Body> {
    app.oneshot(
        Request::builder()
            .method("POST")
            .uri(path)
            .header("content-type", BINARY_FFN_CONTENT_TYPE)
            .body(Body::from(body))
            .unwrap(),
    )
    .await
    .unwrap()
}

// ══════════════════════════════════════════════════════════════
// POST /v1/embed
// ══════════════════════════════════════════════════════════════

#[tokio::test]
async fn http_embed_valid_token_ids_returns_200() {
    let app = single_model_router(state(vec![model("test")]));
    let resp = post_json(
        app,
        "/v1/embed",
        serde_json::json!({"token_ids": [0, 1, 2]}),
    )
    .await;
    assert_eq!(resp.status(), StatusCode::OK);
    let body = body_json(resp.into_body()).await;
    assert_eq!(body["seq_len"], 3);
    assert_eq!(body["hidden_size"], 4);
    assert!(body["residual"].is_array());
}

#[tokio::test]
async fn http_embed_empty_token_ids_returns_400() {
    let app = single_model_router(state(vec![model("test")]));
    let resp = post_json(app, "/v1/embed", serde_json::json!({"token_ids": []})).await;
    assert_eq!(resp.status(), StatusCode::BAD_REQUEST);
}

#[tokio::test]
async fn http_embed_out_of_range_token_returns_400() {
    // vocab_size=8, token_id=100 is out of range.
    let app = single_model_router(state(vec![model("test")]));
    let resp = post_json(app, "/v1/embed", serde_json::json!({"token_ids": [100]})).await;
    assert_eq!(resp.status(), StatusCode::BAD_REQUEST);
}

#[tokio::test]
async fn http_embed_single_token_returns_correct_shape() {
    let app = single_model_router(state(vec![model("test")]));
    let resp = post_json(app, "/v1/embed", serde_json::json!({"token_ids": [0]})).await;
    assert_eq!(resp.status(), StatusCode::OK);
    let body = body_json(resp.into_body()).await;
    // seq_len=1, hidden_size=4 → residual[0] has 4 values.
    let row = body["residual"][0].as_array().unwrap();
    assert_eq!(row.len(), 4);
}

#[tokio::test]
async fn http_embed_invalid_json_returns_400() {
    let app = single_model_router(state(vec![model("test")]));
    let resp = app
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/v1/embed")
                .header("content-type", "application/json")
                .body(Body::from("{not json"))
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(resp.status(), StatusCode::BAD_REQUEST);
}

#[tokio::test]
async fn http_embed_no_model_returns_404() {
    let app = single_model_router(state(vec![]));
    let resp = post_json(app, "/v1/embed", serde_json::json!({"token_ids": [0]})).await;
    assert_eq!(resp.status(), StatusCode::NOT_FOUND);
}

#[tokio::test]
async fn http_embed_binary_returns_binary_response() {
    let app = single_model_router(state(vec![model("test")]));
    let resp = post_binary(app, "/v1/embed", binary_embed_body(&[0, 1])).await;
    assert_eq!(resp.status(), StatusCode::OK);
    assert_eq!(
        resp.headers()
            .get("content-type")
            .and_then(|v| v.to_str().ok()),
        Some(BINARY_FFN_CONTENT_TYPE)
    );
    let bytes = axum::body::to_bytes(resp.into_body(), usize::MAX)
        .await
        .unwrap();
    assert_eq!(u32::from_le_bytes(bytes[0..4].try_into().unwrap()), 2);
    assert_eq!(u32::from_le_bytes(bytes[4..8].try_into().unwrap()), 4);
    assert_eq!(bytes.len(), 8 + 2 * 4 * 4);
}

#[tokio::test]
async fn http_embed_binary_truncated_returns_400() {
    let app = single_model_router(state(vec![model("test")]));
    let mut body = Vec::new();
    body.extend_from_slice(&2u32.to_le_bytes());
    body.extend_from_slice(&0u32.to_le_bytes());
    let resp = post_binary(app, "/v1/embed", body).await;
    assert_eq!(resp.status(), StatusCode::BAD_REQUEST);
}

// ══════════════════════════════════════════════════════════════
// GET /v1/embed/{token_id}  (single-token lookup)
// ══════════════════════════════════════════════════════════════

#[tokio::test]
async fn http_embed_single_get_returns_200() {
    let app = single_model_router(state(vec![model("test")]));
    let resp = get(app, "/v1/embed/0").await;
    assert_eq!(resp.status(), StatusCode::OK);
}

#[tokio::test]
async fn http_embed_single_get_json_accept_returns_json() {
    let app = single_model_router(state(vec![model("test")]));
    let resp = get_h(app, "/v1/embed/0", ("accept", "application/json")).await;
    assert_eq!(resp.status(), StatusCode::OK);
    assert_eq!(
        resp.headers()
            .get("cache-control")
            .and_then(|v| v.to_str().ok()),
        Some("public, max-age=31536000, immutable")
    );
    let body = body_json(resp.into_body()).await;
    assert_eq!(body["token_id"], 0);
    assert_eq!(body["hidden_size"], 4);
}

#[tokio::test]
async fn http_embed_single_get_out_of_range_returns_400() {
    let app = single_model_router(state(vec![model("test")]));
    let resp = get(app, "/v1/embed/100").await;
    assert_eq!(resp.status(), StatusCode::BAD_REQUEST);
}

#[tokio::test]
async fn http_multi_embed_single_get_unknown_model_returns_404() {
    let app = multi_model_router(state(vec![model("test")]));
    let resp = get(app, "/v1/missing/embed/0").await;
    assert_eq!(resp.status(), StatusCode::NOT_FOUND);
}

// ══════════════════════════════════════════════════════════════
// POST /v1/logits
// ══════════════════════════════════════════════════════════════

#[tokio::test]
async fn http_logits_invalid_json_returns_400() {
    let app = single_model_router(state(vec![model("test")]));
    let resp = app
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/v1/logits")
                .header("content-type", "application/json")
                .body(Body::from("{bad"))
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(resp.status(), StatusCode::BAD_REQUEST);
}

#[tokio::test]
async fn http_logits_binary_odd_length_returns_400() {
    let app = single_model_router(state(vec![model("test")]));
    let resp = post_binary(app, "/v1/logits", vec![0, 1, 2, 3, 4]).await;
    assert_eq!(resp.status(), StatusCode::BAD_REQUEST);
}

#[tokio::test]
async fn http_logits_hidden_mismatch_returns_400() {
    let app = single_model_router(state(vec![model("test")]));
    let resp = post_json(
        app,
        "/v1/logits",
        serde_json::json!({"residual": [1.0, 2.0], "top_k": 2}),
    )
    .await;
    assert_eq!(resp.status(), StatusCode::BAD_REQUEST);
}

#[tokio::test]
async fn http_logits_binary_hidden_mismatch_returns_400() {
    let app = single_model_router(state(vec![model("test")]));
    let resp = post_binary(app, "/v1/logits", binary_logits_body(&[1.0, 2.0])).await;
    assert_eq!(resp.status(), StatusCode::BAD_REQUEST);
}

#[tokio::test]
async fn http_logits_no_model_returns_404() {
    let app = single_model_router(state(vec![]));
    let resp = post_json(
        app,
        "/v1/logits",
        serde_json::json!({"residual": [0.0, 0.0, 0.0, 0.0]}),
    )
    .await;
    assert_eq!(resp.status(), StatusCode::NOT_FOUND);
}

// ══════════════════════════════════════════════════════════════
// GET /v1/token/decode
// ══════════════════════════════════════════════════════════════

#[tokio::test]
async fn http_token_decode_empty_ids_returns_200() {
    let app = single_model_router(state(vec![model("test")]));
    let resp = get(app, "/v1/token/decode?ids=").await;
    assert_eq!(resp.status(), StatusCode::OK);
    let body = body_json(resp.into_body()).await;
    assert!(body["token_ids"].as_array().unwrap().is_empty());
}

#[tokio::test]
async fn http_token_decode_invalid_id_returns_400() {
    let app = single_model_router(state(vec![model("test")]));
    let resp = get(app, "/v1/token/decode?ids=notanumber").await;
    assert_eq!(resp.status(), StatusCode::BAD_REQUEST);
}

#[tokio::test]
async fn http_token_decode_missing_ids_param_returns_400() {
    let app = single_model_router(state(vec![model("test")]));
    let resp = get(app, "/v1/token/decode").await;
    assert_eq!(resp.status(), StatusCode::BAD_REQUEST);
}

// ══════════════════════════════════════════════════════════════
// GET /v1/token/encode
// ══════════════════════════════════════════════════════════════

#[tokio::test]
async fn http_token_encode_returns_200() {
    let app = single_model_router(state(vec![model("test")]));
    let resp = get(app, "/v1/token/encode?text=hello").await;
    assert_eq!(resp.status(), StatusCode::OK);
    let body = body_json(resp.into_body()).await;
    assert_eq!(body["text"], "hello");
    assert!(body["token_ids"].is_array());
}

#[tokio::test]
async fn http_token_encode_missing_text_returns_400() {
    let app = single_model_router(state(vec![model("test")]));
    let resp = get(app, "/v1/token/encode").await;
    assert_eq!(resp.status(), StatusCode::BAD_REQUEST);
}

// ══════════════════════════════════════════════════════════════
// POST /v1/embeddings — OpenAI-compatible embeddings (N0.4)
// ══════════════════════════════════════════════════════════════

#[tokio::test]
async fn http_openai_embeddings_string_input_returns_200_with_pooled_vector() {
    // Uses the functional tokenizer so "France" tokenises cleanly.
    let app = single_model_router(state(vec![model_functional("gemma")]));
    let resp = post_json(
        app,
        "/v1/embeddings",
        serde_json::json!({"input": "France"}),
    )
    .await;
    assert_eq!(resp.status(), StatusCode::OK);
    let body = body_json(resp.into_body()).await;
    assert_eq!(body["object"], "list");
    assert_eq!(body["model"], "gemma");
    let data = body["data"].as_array().unwrap();
    assert_eq!(data.len(), 1);
    assert_eq!(data[0]["object"], "embedding");
    assert_eq!(data[0]["index"], 0);
    let embedding = data[0]["embedding"].as_array().unwrap();
    assert_eq!(embedding.len(), 4); // hidden_size=4 in synthetic model
    assert!(body["usage"]["prompt_tokens"].as_u64().unwrap() > 0);
    assert_eq!(
        body["usage"]["prompt_tokens"],
        body["usage"]["total_tokens"]
    );
}

#[tokio::test]
async fn http_openai_embeddings_string_array_returns_indexed_data() {
    let app = single_model_router(state(vec![model_functional("gemma")]));
    let resp = post_json(
        app,
        "/v1/embeddings",
        serde_json::json!({"input": ["France", "Germany", "capital"]}),
    )
    .await;
    assert_eq!(resp.status(), StatusCode::OK);
    let body = body_json(resp.into_body()).await;
    let data = body["data"].as_array().unwrap();
    assert_eq!(data.len(), 3);
    for (i, entry) in data.iter().enumerate() {
        assert_eq!(entry["index"], i);
        assert_eq!(entry["object"], "embedding");
        let v = entry["embedding"].as_array().unwrap();
        assert_eq!(v.len(), 4);
    }
}

#[tokio::test]
async fn http_openai_embeddings_pretokenised_single_works() {
    let app = single_model_router(state(vec![model("test")]));
    let resp = post_json(
        app,
        "/v1/embeddings",
        serde_json::json!({"input": [0u32, 1u32, 2u32]}),
    )
    .await;
    assert_eq!(resp.status(), StatusCode::OK);
    let body = body_json(resp.into_body()).await;
    let data = body["data"].as_array().unwrap();
    assert_eq!(data.len(), 1);
    assert_eq!(body["usage"]["prompt_tokens"], 3);
}

#[tokio::test]
async fn http_openai_embeddings_base64_format_returns_string() {
    // base64 is now supported — the embedding field is a base64 string
    // of the LE f32 bytes instead of a JSON array. Use pretokenised
    // input so the synthetic tokenizer doesn't gate the test path.
    let app = single_model_router(state(vec![model("test")]));
    let resp = post_json(
        app,
        "/v1/embeddings",
        serde_json::json!({
            "input": [0u32, 1u32, 2u32],
            "encoding_format": "base64",
        }),
    )
    .await;
    assert_eq!(resp.status(), StatusCode::OK);
    let body = body_json(resp.into_body()).await;
    let embedding = &body["data"][0]["embedding"];
    assert!(
        embedding.is_string(),
        "expected base64 string, got {embedding}"
    );
    let s = embedding.as_str().unwrap();
    // Decode + sanity-check length: 4 bytes per f32, must be ≥1 f32.
    use base64::Engine;
    let bytes = base64::engine::general_purpose::STANDARD
        .decode(s.as_bytes())
        .expect("valid base64");
    assert!(!bytes.is_empty());
    assert_eq!(
        bytes.len() % 4,
        0,
        "len must be 4·hidden, got {}",
        bytes.len()
    );
}

#[tokio::test]
async fn http_openai_embeddings_unknown_format_returns_400() {
    let app = single_model_router(state(vec![model("test")]));
    let resp = post_json(
        app,
        "/v1/embeddings",
        serde_json::json!({"input": [0u32, 1u32], "encoding_format": "binary"}),
    )
    .await;
    assert_eq!(resp.status(), StatusCode::BAD_REQUEST);
}

#[tokio::test]
async fn http_openai_embeddings_empty_input_returns_400() {
    let app = single_model_router(state(vec![model("test")]));
    let resp = post_json(app, "/v1/embeddings", serde_json::json!({"input": []})).await;
    assert_eq!(resp.status(), StatusCode::BAD_REQUEST);
}

// ══════════════════════════════════════════════════════════════
// POST /v1/completions — OpenAI-compatible completions (N0.2)
//
// These tests exercise request validation (the parts that don't
// require a real model + weights). End-to-end generation is exercised
// via the `larql run` CLI smoke test against a real vindex.
// ══════════════════════════════════════════════════════════════

#[tokio::test]
async fn http_openai_completions_stream_with_echo_returns_400() {
    // echo=true is not supported in stream mode (one-prompt-one-stream).
    let app = single_model_router(state(vec![model_infer_enabled("gemma")]));
    let resp = post_json(
        app,
        "/v1/completions",
        serde_json::json!({
            "prompt": "hi",
            "stream": true,
            "echo": true,
            "max_tokens": 1
        }),
    )
    .await;
    assert_eq!(resp.status(), StatusCode::BAD_REQUEST);
}

#[tokio::test]
async fn http_openai_completions_stream_with_batched_prompts_returns_400() {
    // Batched prompts not supported with stream=true.
    let app = single_model_router(state(vec![model_infer_enabled("gemma")]));
    let resp = post_json(
        app,
        "/v1/completions",
        serde_json::json!({
            "prompt": ["hi", "there"],
            "stream": true,
            "max_tokens": 1
        }),
    )
    .await;
    assert_eq!(resp.status(), StatusCode::BAD_REQUEST);
}

#[tokio::test]
async fn http_openai_completions_stream_returns_event_stream_content_type() {
    use axum::http::header;
    let app = single_model_router(state(vec![model_infer_enabled("gemma")]));
    let resp = post_json(
        app,
        "/v1/completions",
        serde_json::json!({
            "prompt": "hi",
            "stream": true,
            "max_tokens": 2
        }),
    )
    .await;
    assert_eq!(resp.status(), StatusCode::OK);
    let ct = resp
        .headers()
        .get(header::CONTENT_TYPE)
        .and_then(|v| v.to_str().ok())
        .unwrap_or("");
    assert!(
        ct.starts_with("text/event-stream"),
        "expected SSE content-type, got {ct:?}"
    );
}

#[tokio::test]
async fn http_openai_completions_n_gt_1_returns_400() {
    let app = single_model_router(state(vec![model_infer_enabled("gemma")]));
    let resp = post_json(
        app,
        "/v1/completions",
        serde_json::json!({"prompt": "hi", "n": 2, "max_tokens": 1}),
    )
    .await;
    assert_eq!(resp.status(), StatusCode::BAD_REQUEST);
}

#[tokio::test]
async fn http_openai_completions_infer_disabled_returns_503() {
    // model() builds with infer_disabled=true.
    let app = single_model_router(state(vec![model("gemma")]));
    let resp = post_json(
        app,
        "/v1/completions",
        serde_json::json!({"prompt": "hi", "max_tokens": 1}),
    )
    .await;
    // ServerError::InferenceUnavailable maps to 503.
    assert_eq!(resp.status(), StatusCode::SERVICE_UNAVAILABLE);
}

#[tokio::test]
async fn http_openai_completions_missing_prompt_returns_422() {
    let app = single_model_router(state(vec![model_infer_enabled("gemma")]));
    let resp = post_json(app, "/v1/completions", serde_json::json!({"max_tokens": 1})).await;
    // Missing required `prompt` field — serde returns 422 via axum's
    // Json extractor.
    assert!(
        resp.status() == StatusCode::UNPROCESSABLE_ENTITY
            || resp.status() == StatusCode::BAD_REQUEST,
        "got {}",
        resp.status()
    );
}

// ══════════════════════════════════════════════════════════════
// OpenAI endpoints — multi-model routing
//
// In multi-model mode the client passes `model` in the request body
// (OpenAI convention). The endpoints route to the right loaded vindex
// without needing a path-prefixed `/v1/{model_id}/...` URL.
// ══════════════════════════════════════════════════════════════

#[tokio::test]
async fn http_openai_models_multi_lists_all_with_openai_shape() {
    let app = multi_model_router(state(vec![
        model_functional("gemma-a"),
        model_functional("gemma-b"),
    ]));
    let resp = get(app, "/v1/models").await;
    assert_eq!(resp.status(), StatusCode::OK);
    let body = body_json(resp.into_body()).await;
    assert_eq!(body["object"], "list");
    let data = body["data"].as_array().unwrap();
    assert_eq!(data.len(), 2);
    let ids: Vec<&str> = data.iter().map(|m| m["id"].as_str().unwrap()).collect();
    assert!(ids.contains(&"gemma-a"));
    assert!(ids.contains(&"gemma-b"));
    for entry in data {
        assert_eq!(entry["object"], "model");
        assert_eq!(entry["owned_by"], "larql");
    }
}

#[tokio::test]
async fn http_openai_embeddings_multi_routes_via_model_field() {
    let app = multi_model_router(state(vec![
        model_functional("gemma-a"),
        model_functional("gemma-b"),
    ]));
    let resp = post_json(
        app,
        "/v1/embeddings",
        serde_json::json!({"model": "gemma-b", "input": "France"}),
    )
    .await;
    assert_eq!(resp.status(), StatusCode::OK);
    let body = body_json(resp.into_body()).await;
    assert_eq!(body["model"], "gemma-b");
    let data = body["data"].as_array().unwrap();
    assert_eq!(data.len(), 1);
    assert_eq!(data[0]["index"], 0);
}

#[tokio::test]
async fn http_openai_embeddings_multi_unknown_model_returns_404() {
    let app = multi_model_router(state(vec![model_functional("gemma-a")]));
    let resp = post_json(
        app,
        "/v1/embeddings",
        serde_json::json!({"model": "missing", "input": "France"}),
    )
    .await;
    assert_eq!(resp.status(), StatusCode::NOT_FOUND);
}

#[tokio::test]
async fn http_openai_embeddings_no_model_field_in_single_model_works() {
    // Single-model mode: omitting `model` is fine; we use the loaded one.
    let app = single_model_router(state(vec![model_functional("gemma")]));
    let resp = post_json(
        app,
        "/v1/embeddings",
        serde_json::json!({"input": "France"}),
    )
    .await;
    assert_eq!(resp.status(), StatusCode::OK);
    let body = body_json(resp.into_body()).await;
    assert_eq!(body["model"], "gemma");
}

#[tokio::test]
async fn http_openai_completions_multi_routes_via_model_field() {
    // Use ModelBuilder to flip infer_disabled=false.
    use larql_server::state::LoadedModel;
    use std::sync::Arc;
    let m = ModelBuilder::new("gemma-a").build();
    let n = ModelBuilder::new("gemma-b").build();
    let _: Arc<LoadedModel> = Arc::clone(&m);
    let app = multi_model_router(state(vec![m, n]));
    // infer_disabled=true on default ModelBuilder → expect 503.
    // We're testing routing, not generation — 503 from the right model
    // confirms routing worked.
    let resp = post_json(
        app,
        "/v1/completions",
        serde_json::json!({"model": "gemma-b", "prompt": "x", "max_tokens": 1}),
    )
    .await;
    assert_eq!(resp.status(), StatusCode::SERVICE_UNAVAILABLE);
}

#[tokio::test]
async fn http_openai_completions_multi_unknown_model_returns_404() {
    let app = multi_model_router(state(vec![model_functional("gemma-a")]));
    let resp = post_json(
        app,
        "/v1/completions",
        serde_json::json!({"model": "missing", "prompt": "x", "max_tokens": 1}),
    )
    .await;
    assert_eq!(resp.status(), StatusCode::NOT_FOUND);
}

// ══════════════════════════════════════════════════════════════
// OpenAI endpoints — auth flow
// ══════════════════════════════════════════════════════════════

#[tokio::test]
async fn http_openai_embeddings_with_auth_required_no_token_returns_401() {
    use axum::middleware;
    let app_state = state_with_key(vec![model_functional("gemma")], "sk-secret");
    let app = single_model_router(app_state.clone()).layer(middleware::from_fn_with_state(
        app_state,
        larql_server::auth::auth_middleware,
    ));
    let resp = post_json(
        app,
        "/v1/embeddings",
        serde_json::json!({"input": "France"}),
    )
    .await;
    assert_eq!(resp.status(), StatusCode::UNAUTHORIZED);
}

#[tokio::test]
async fn http_openai_embeddings_with_auth_correct_bearer_returns_200() {
    use axum::middleware;
    let app_state = state_with_key(vec![model_functional("gemma")], "sk-secret");
    let app = single_model_router(app_state.clone()).layer(middleware::from_fn_with_state(
        app_state,
        larql_server::auth::auth_middleware,
    ));
    let resp = post_json_h(
        app,
        "/v1/embeddings",
        serde_json::json!({"input": "France"}),
        ("authorization", "Bearer sk-secret"),
    )
    .await;
    assert_eq!(resp.status(), StatusCode::OK);
}

// ══════════════════════════════════════════════════════════════
// POST /v1/chat/completions — N0.1 slice 2
// ══════════════════════════════════════════════════════════════

#[tokio::test]
async fn http_openai_chat_stream_returns_event_stream_content_type() {
    // model() has infer_disabled=true, but the dispatch happens before
    // the inference step — actually no, infer_disabled is checked first
    // and returns 503 even for stream. Use model_infer_enabled (empty
    // tokenizer) — generation will tokenise the prompt to empty and
    // emit an error chunk before [DONE], but the response headers and
    // status should be SSE.
    use axum::http::header;
    let app = single_model_router(state(vec![model_infer_enabled("gemma")]));
    let resp = post_json(
        app,
        "/v1/chat/completions",
        serde_json::json!({
            "messages": [{"role": "user", "content": "hi"}],
            "stream": true,
            "max_tokens": 2
        }),
    )
    .await;
    assert_eq!(resp.status(), StatusCode::OK);
    let ct = resp
        .headers()
        .get(header::CONTENT_TYPE)
        .and_then(|v| v.to_str().ok())
        .unwrap_or("");
    assert!(
        ct.starts_with("text/event-stream"),
        "expected SSE content-type, got {ct:?}"
    );
}

#[tokio::test]
async fn http_openai_chat_n_gt_1_returns_400() {
    let app = single_model_router(state(vec![model_infer_enabled("gemma")]));
    let resp = post_json(
        app,
        "/v1/chat/completions",
        serde_json::json!({
            "messages": [{"role": "user", "content": "hi"}],
            "n": 3,
            "max_tokens": 1
        }),
    )
    .await;
    assert_eq!(resp.status(), StatusCode::BAD_REQUEST);
}

#[tokio::test]
async fn http_openai_chat_tools_are_accepted() {
    // Tools synthesise a constrained-decoding schema. Synthetic model
    // is infer_disabled so we 503 — confirms the schema synth +
    // ToolMode resolution succeeded.
    let app = single_model_router(state(vec![model("gemma")]));
    let resp = post_json(
        app,
        "/v1/chat/completions",
        serde_json::json!({
            "messages": [{"role": "user", "content": "hi"}],
            "tools": [{
                "type": "function",
                "function": {
                    "name": "get_weather",
                    "parameters": {
                        "type": "object",
                        "properties": {"location": {"type": "string"}},
                        "required": ["location"]
                    }
                }
            }],
            "max_tokens": 1
        }),
    )
    .await;
    assert_eq!(resp.status(), StatusCode::SERVICE_UNAVAILABLE);
}

#[tokio::test]
async fn http_openai_chat_tools_with_specific_choice_is_accepted() {
    let app = single_model_router(state(vec![model("gemma")]));
    let resp = post_json(
        app,
        "/v1/chat/completions",
        serde_json::json!({
            "messages": [{"role": "user", "content": "hi"}],
            "tools": [
                {"type": "function", "function": {"name": "calc", "parameters": {"type": "object"}}},
                {"type": "function", "function": {"name": "search", "parameters": {"type": "object"}}}
            ],
            "tool_choice": {"type": "function", "function": {"name": "calc"}},
            "max_tokens": 1
        }),
    )
    .await;
    assert_eq!(resp.status(), StatusCode::SERVICE_UNAVAILABLE);
}

#[tokio::test]
async fn http_openai_chat_tools_unknown_choice_returns_400() {
    let app = single_model_router(state(vec![model_infer_enabled("gemma")]));
    let resp = post_json(
        app,
        "/v1/chat/completions",
        serde_json::json!({
            "messages": [{"role": "user", "content": "hi"}],
            "tools": [{"type": "function", "function": {"name": "calc", "parameters": {}}}],
            "tool_choice": {"type": "function", "function": {"name": "missing"}},
            "max_tokens": 1
        }),
    )
    .await;
    assert_eq!(resp.status(), StatusCode::BAD_REQUEST);
}

#[tokio::test]
async fn http_openai_chat_tools_with_stream_returns_event_stream() {
    // Slice 4.11: tools + stream is now wired. Synthetic model has
    // infer_disabled=true, but the SSE response shape is determined
    // before the inference gate fires — confirm we get a 200 SSE
    // content-type, not 400.
    use axum::http::header;
    let app = single_model_router(state(vec![model_infer_enabled("gemma")]));
    let resp = post_json(
        app,
        "/v1/chat/completions",
        serde_json::json!({
            "messages": [{"role": "user", "content": "hi"}],
            "tools": [{"type": "function", "function": {"name": "calc", "parameters": {}}}],
            "stream": true,
            "max_tokens": 1
        }),
    )
    .await;
    assert_eq!(resp.status(), StatusCode::OK);
    let ct = resp
        .headers()
        .get(header::CONTENT_TYPE)
        .and_then(|v| v.to_str().ok())
        .unwrap_or("");
    assert!(
        ct.starts_with("text/event-stream"),
        "expected SSE content-type, got {ct:?}"
    );
}

#[tokio::test]
async fn http_openai_chat_tool_choice_none_skips_constraint() {
    // tool_choice="none" disables constrained decoding even when tools
    // are listed — falls through to the standard text completion path.
    let app = single_model_router(state(vec![model("gemma")]));
    let resp = post_json(
        app,
        "/v1/chat/completions",
        serde_json::json!({
            "messages": [{"role": "user", "content": "hi"}],
            "tools": [{"type": "function", "function": {"name": "calc", "parameters": {}}}],
            "tool_choice": "none",
            "max_tokens": 1
        }),
    )
    .await;
    assert_eq!(resp.status(), StatusCode::SERVICE_UNAVAILABLE);
}

#[tokio::test]
async fn http_openai_chat_response_format_json_schema_missing_schema_field_returns_400() {
    // {type: "json_schema"} requires `json_schema: {schema: ...}` —
    // the empty inner object has no `schema` key, so we 400 with a
    // pointer at the missing field.
    let app = single_model_router(state(vec![model_infer_enabled("gemma")]));
    let resp = post_json(
        app,
        "/v1/chat/completions",
        serde_json::json!({
            "messages": [{"role": "user", "content": "hi"}],
            "response_format": {"type": "json_schema", "json_schema": {}},
            "max_tokens": 1
        }),
    )
    .await;
    assert_eq!(resp.status(), StatusCode::BAD_REQUEST);
}

#[tokio::test]
async fn http_openai_chat_response_format_json_schema_is_accepted() {
    // Full {type: "json_schema", json_schema: {name, schema, strict}}
    // request — synthetic model 503s because infer_disabled, which
    // confirms the schema parsed cleanly through to the inference gate.
    let app = single_model_router(state(vec![model("gemma")]));
    let resp = post_json(
        app,
        "/v1/chat/completions",
        serde_json::json!({
            "messages": [{"role": "user", "content": "hi"}],
            "response_format": {
                "type": "json_schema",
                "json_schema": {
                    "name": "Person",
                    "strict": true,
                    "schema": {
                        "type": "object",
                        "properties": {
                            "name": {"type": "string"},
                            "age": {"type": "integer"}
                        },
                        "required": ["name", "age"]
                    }
                }
            },
            "max_tokens": 1
        }),
    )
    .await;
    assert_eq!(resp.status(), StatusCode::SERVICE_UNAVAILABLE);
}

#[tokio::test]
async fn http_openai_chat_response_format_json_schema_invalid_returns_400() {
    // Schema uses an unsupported feature ($ref) — parser bubbles up
    // a clear 400.
    let app = single_model_router(state(vec![model_infer_enabled("gemma")]));
    let resp = post_json(
        app,
        "/v1/chat/completions",
        serde_json::json!({
            "messages": [{"role": "user", "content": "hi"}],
            "response_format": {
                "type": "json_schema",
                "json_schema": {"schema": {"$ref": "#/foo"}}
            },
            "max_tokens": 1
        }),
    )
    .await;
    assert_eq!(resp.status(), StatusCode::BAD_REQUEST);
}

#[tokio::test]
async fn http_openai_chat_response_format_text_is_accepted() {
    // {type: "text"} is the OpenAI default — should pass through, fall
    // through to infer_disabled gate (synthetic model) → 503.
    let app = single_model_router(state(vec![model("gemma")]));
    let resp = post_json(
        app,
        "/v1/chat/completions",
        serde_json::json!({
            "messages": [{"role": "user", "content": "hi"}],
            "response_format": {"type": "text"},
            "max_tokens": 1
        }),
    )
    .await;
    assert_eq!(resp.status(), StatusCode::SERVICE_UNAVAILABLE);
}

#[tokio::test]
async fn http_openai_chat_response_format_json_object_is_accepted() {
    // {type: "json_object"} compiles to a Schema::Object(any) FSM and
    // routes through generate_constrained. The synthetic model has
    // infer_disabled=true so we still 503 — that's our signal that the
    // request shape parsed cleanly through the constrained-mode path.
    let app = single_model_router(state(vec![model("gemma")]));
    let resp = post_json(
        app,
        "/v1/chat/completions",
        serde_json::json!({
            "messages": [{"role": "user", "content": "hi"}],
            "response_format": {"type": "json_object"},
            "max_tokens": 1
        }),
    )
    .await;
    assert_eq!(resp.status(), StatusCode::SERVICE_UNAVAILABLE);
}

#[tokio::test]
async fn http_openai_chat_response_format_unknown_type_returns_400() {
    let app = single_model_router(state(vec![model_infer_enabled("gemma")]));
    let resp = post_json(
        app,
        "/v1/chat/completions",
        serde_json::json!({
            "messages": [{"role": "user", "content": "hi"}],
            "response_format": {"type": "yaml"},
            "max_tokens": 1
        }),
    )
    .await;
    assert_eq!(resp.status(), StatusCode::BAD_REQUEST);
}

#[tokio::test]
async fn http_openai_chat_invalid_role_returns_400() {
    let app = single_model_router(state(vec![model_infer_enabled("gemma")]));
    let resp = post_json(
        app,
        "/v1/chat/completions",
        serde_json::json!({
            "messages": [{"role": "function", "content": "x"}],
            "max_tokens": 1
        }),
    )
    .await;
    assert_eq!(resp.status(), StatusCode::BAD_REQUEST);
}

#[tokio::test]
async fn http_openai_chat_tool_message_without_tool_call_id_returns_400() {
    let app = single_model_router(state(vec![model_infer_enabled("gemma")]));
    let resp = post_json(
        app,
        "/v1/chat/completions",
        serde_json::json!({
            "messages": [
                {"role": "user", "content": "Weather?"},
                {"role": "tool", "content": "23C"} // missing tool_call_id
            ],
            "max_tokens": 1
        }),
    )
    .await;
    assert_eq!(resp.status(), StatusCode::BAD_REQUEST);
}

#[tokio::test]
async fn http_openai_chat_tool_replay_is_accepted() {
    // Full multi-turn tool flow: user → assistant tool_call → tool
    // result → expects another assistant turn. Synthetic model is
    // infer_disabled, so we 503 — confirming the wire shape parsed
    // through validation.
    let app = single_model_router(state(vec![model("gemma")]));
    let resp = post_json(
        app,
        "/v1/chat/completions",
        serde_json::json!({
            "messages": [
                {"role": "user", "content": "Weather in London?"},
                {"role": "assistant", "content": null, "tool_calls": [
                    {"id": "call_1", "type": "function",
                     "function": {"name": "get_weather", "arguments": "{\"city\":\"London\"}"}}
                ]},
                {"role": "tool", "tool_call_id": "call_1", "content": "23C, sunny"}
            ],
            "max_tokens": 16
        }),
    )
    .await;
    assert_eq!(resp.status(), StatusCode::SERVICE_UNAVAILABLE);
}

#[tokio::test]
async fn http_openai_chat_assistant_with_only_tool_calls_is_accepted() {
    // Some clients send assistant messages with content: null but
    // populated tool_calls — must not 400 on the missing content.
    let app = single_model_router(state(vec![model("gemma")]));
    let resp = post_json(
        app,
        "/v1/chat/completions",
        serde_json::json!({
            "messages": [
                {"role": "user", "content": "x"},
                {"role": "assistant", "content": null, "tool_calls": [
                    {"id": "call_1", "type": "function",
                     "function": {"name": "calc", "arguments": "{}"}}
                ]}
            ],
            "max_tokens": 1
        }),
    )
    .await;
    assert_eq!(resp.status(), StatusCode::SERVICE_UNAVAILABLE);
}

#[tokio::test]
async fn http_openai_chat_logprobs_request_field_is_accepted() {
    // logprobs: true should be accepted on chat completions; the
    // synthetic model 503s but the field passes validation.
    let app = single_model_router(state(vec![model("gemma")]));
    let resp = post_json(
        app,
        "/v1/chat/completions",
        serde_json::json!({
            "messages": [{"role": "user", "content": "hi"}],
            "logprobs": true,
            "top_logprobs": 5,
            "max_tokens": 1
        }),
    )
    .await;
    assert_eq!(resp.status(), StatusCode::SERVICE_UNAVAILABLE);
}

#[tokio::test]
async fn http_openai_completions_repetition_penalties_are_accepted() {
    // F19: frequency_penalty + presence_penalty land in SamplingConfig
    // and clamp to [-2.0, 2.0]. Synthetic model 503s but the field
    // parses cleanly through to the inference gate.
    let app = single_model_router(state(vec![model("gemma")]));
    let resp = post_json(
        app,
        "/v1/completions",
        serde_json::json!({
            "prompt": "hi",
            "temperature": 0.7,
            "frequency_penalty": 1.5,
            "presence_penalty": -0.3,
            "max_tokens": 4
        }),
    )
    .await;
    assert_eq!(resp.status(), StatusCode::SERVICE_UNAVAILABLE);
}

#[tokio::test]
async fn http_openai_chat_repetition_penalties_are_accepted() {
    let app = single_model_router(state(vec![model("gemma")]));
    let resp = post_json(
        app,
        "/v1/chat/completions",
        serde_json::json!({
            "messages": [{"role": "user", "content": "hi"}],
            "temperature": 0.5,
            "frequency_penalty": 1.0,
            "presence_penalty": 0.5,
            "max_tokens": 4
        }),
    )
    .await;
    assert_eq!(resp.status(), StatusCode::SERVICE_UNAVAILABLE);
}

#[tokio::test]
async fn http_openai_completions_logprobs_request_field_is_accepted() {
    let app = single_model_router(state(vec![model("gemma")]));
    let resp = post_json(
        app,
        "/v1/completions",
        serde_json::json!({
            "prompt": "hi",
            "logprobs": 3,
            "max_tokens": 1
        }),
    )
    .await;
    assert_eq!(resp.status(), StatusCode::SERVICE_UNAVAILABLE);
}

#[tokio::test]
async fn http_openai_chat_assistant_with_no_content_or_tools_returns_400() {
    let app = single_model_router(state(vec![model_infer_enabled("gemma")]));
    let resp = post_json(
        app,
        "/v1/chat/completions",
        serde_json::json!({
            "messages": [
                {"role": "user", "content": "hi"},
                {"role": "assistant"} // no content, no tool_calls
            ],
            "max_tokens": 1
        }),
    )
    .await;
    assert_eq!(resp.status(), StatusCode::BAD_REQUEST);
}

#[tokio::test]
async fn http_openai_chat_empty_messages_returns_400() {
    let app = single_model_router(state(vec![model_infer_enabled("gemma")]));
    let resp = post_json(
        app,
        "/v1/chat/completions",
        serde_json::json!({"messages": [], "max_tokens": 1}),
    )
    .await;
    assert_eq!(resp.status(), StatusCode::BAD_REQUEST);
}

#[tokio::test]
async fn http_openai_chat_infer_disabled_returns_503() {
    let app = single_model_router(state(vec![model("gemma")]));
    let resp = post_json(
        app,
        "/v1/chat/completions",
        serde_json::json!({
            "messages": [{"role": "user", "content": "hi"}],
            "max_tokens": 1
        }),
    )
    .await;
    assert_eq!(resp.status(), StatusCode::SERVICE_UNAVAILABLE);
}

#[tokio::test]
async fn http_openai_chat_multi_routes_via_model_field() {
    let app = multi_model_router(state(vec![model("gemma-a"), model("gemma-b")]));
    let resp = post_json(
        app,
        "/v1/chat/completions",
        serde_json::json!({
            "model": "gemma-b",
            "messages": [{"role": "user", "content": "hi"}],
            "max_tokens": 1
        }),
    )
    .await;
    // Routing succeeds; infer_disabled on the synthetic model → 503.
    assert_eq!(resp.status(), StatusCode::SERVICE_UNAVAILABLE);
}

#[tokio::test]
async fn http_openai_chat_multi_unknown_model_returns_404() {
    let app = multi_model_router(state(vec![model("gemma-a")]));
    let resp = post_json(
        app,
        "/v1/chat/completions",
        serde_json::json!({
            "model": "missing",
            "messages": [{"role": "user", "content": "hi"}],
            "max_tokens": 1
        }),
    )
    .await;
    assert_eq!(resp.status(), StatusCode::NOT_FOUND);
}

#[tokio::test]
async fn http_openai_chat_sampling_params_accepted() {
    // Wire-shape contract: temperature, top_p, seed, stop must be
    // accepted on the request and not rejected by validation. The
    // synthetic model has infer_disabled=true so the request reaches
    // the inference gate (503) — that's our signal that all sampling
    // fields parsed cleanly upstream.
    let app = single_model_router(state(vec![model("gemma")]));
    let resp = post_json(
        app,
        "/v1/chat/completions",
        serde_json::json!({
            "messages": [{"role": "user", "content": "hi"}],
            "temperature": 0.7,
            "top_p": 0.9,
            "seed": 42,
            "stop": ["\n\n", "STOP"],
            "max_tokens": 4
        }),
    )
    .await;
    assert_eq!(resp.status(), StatusCode::SERVICE_UNAVAILABLE);
}

#[tokio::test]
async fn http_openai_chat_stop_accepts_single_string() {
    // OpenAI's `stop` is `string | string[]`; the StopSpec untagged
    // enum should accept a bare string without validation errors.
    let app = single_model_router(state(vec![model("gemma")]));
    let resp = post_json(
        app,
        "/v1/chat/completions",
        serde_json::json!({
            "messages": [{"role": "user", "content": "hi"}],
            "stop": "\n",
            "max_tokens": 1
        }),
    )
    .await;
    assert_eq!(resp.status(), StatusCode::SERVICE_UNAVAILABLE);
}

#[tokio::test]
async fn http_openai_completions_sampling_params_accepted() {
    let app = single_model_router(state(vec![model("gemma")]));
    let resp = post_json(
        app,
        "/v1/completions",
        serde_json::json!({
            "prompt": "hi",
            "temperature": 0.7,
            "top_p": 0.9,
            "seed": 42,
            "stop": ["\n\n"],
            "max_tokens": 4
        }),
    )
    .await;
    assert_eq!(resp.status(), StatusCode::SERVICE_UNAVAILABLE);
}

// ══════════════════════════════════════════════════════════════
// REV4 — OpenAI error envelope shape.
//
// /v1/embeddings, /v1/completions, /v1/chat/completions must return
// the nested `{error: {message, type, param, code}}` shape so the
// OpenAI Python and JS SDKs parse errors without special-casing.
// LARQL paradigm endpoints keep the flat `{error: "msg"}` shape
// (covered by other tests in this file).
// ══════════════════════════════════════════════════════════════

fn assert_openai_error_envelope(v: &serde_json::Value, expected_type: &str) {
    let err = v
        .get("error")
        .and_then(|e| e.as_object())
        .expect("response body must be {\"error\": {...}} (nested)");
    assert!(
        err.get("message").and_then(|m| m.as_str()).is_some(),
        "error.message must be a non-null string; got {:?}",
        err.get("message")
    );
    assert_eq!(
        err.get("type").and_then(|t| t.as_str()),
        Some(expected_type),
        "error.type mismatch"
    );
    assert!(
        err.contains_key("param"),
        "error.param key must be present (even if null) — SDKs hard-key on it"
    );
    assert!(
        err.contains_key("code"),
        "error.code key must be present (even if null) — SDKs hard-key on it"
    );
}

#[tokio::test]
async fn http_openai_embeddings_400_uses_nested_envelope() {
    let app = single_model_router(state(vec![model("test")]));
    let resp = post_json(
        app,
        "/v1/embeddings",
        serde_json::json!({"input": [0u32, 1u32], "encoding_format": "binary"}),
    )
    .await;
    assert_eq!(resp.status(), StatusCode::BAD_REQUEST);
    let v = body_json(resp.into_body()).await;
    assert_openai_error_envelope(&v, "invalid_request_error");
    let msg = v["error"]["message"].as_str().unwrap();
    assert!(
        msg.contains("encoding_format='binary'"),
        "message should reference the bad input; got {msg:?}"
    );
}

#[tokio::test]
async fn http_openai_embeddings_empty_uses_nested_envelope() {
    let app = single_model_router(state(vec![model("test")]));
    let resp = post_json(app, "/v1/embeddings", serde_json::json!({"input": []})).await;
    assert_eq!(resp.status(), StatusCode::BAD_REQUEST);
    let v = body_json(resp.into_body()).await;
    assert_openai_error_envelope(&v, "invalid_request_error");
}

#[tokio::test]
async fn http_openai_completions_400_uses_nested_envelope() {
    let app = single_model_router(state(vec![model_infer_enabled("gemma")]));
    let resp = post_json(
        app,
        "/v1/completions",
        serde_json::json!({"prompt": "hi", "stream": true, "echo": true, "max_tokens": 1}),
    )
    .await;
    assert_eq!(resp.status(), StatusCode::BAD_REQUEST);
    let v = body_json(resp.into_body()).await;
    assert_openai_error_envelope(&v, "invalid_request_error");
}

#[tokio::test]
async fn http_openai_completions_503_uses_nested_envelope() {
    // model has infer_disabled = true → ServiceUnavailable.
    let app = single_model_router(state(vec![model("gemma")]));
    let resp = post_json(
        app,
        "/v1/completions",
        serde_json::json!({"prompt": "hi", "max_tokens": 1}),
    )
    .await;
    assert_eq!(resp.status(), StatusCode::SERVICE_UNAVAILABLE);
    let v = body_json(resp.into_body()).await;
    assert_openai_error_envelope(&v, "service_unavailable_error");
}

#[tokio::test]
async fn http_openai_chat_completions_400_uses_nested_envelope() {
    let app = single_model_router(state(vec![model_infer_enabled("gemma")]));
    let resp = post_json(
        app,
        "/v1/chat/completions",
        serde_json::json!({"messages": []}),
    )
    .await;
    assert_eq!(resp.status(), StatusCode::BAD_REQUEST);
    let v = body_json(resp.into_body()).await;
    assert_openai_error_envelope(&v, "invalid_request_error");
}

#[tokio::test]
async fn http_openai_chat_completions_503_uses_nested_envelope() {
    let app = single_model_router(state(vec![model("gemma")]));
    let resp = post_json(
        app,
        "/v1/chat/completions",
        serde_json::json!({
            "messages": [{"role": "user", "content": "hi"}],
            "max_tokens": 1
        }),
    )
    .await;
    assert_eq!(resp.status(), StatusCode::SERVICE_UNAVAILABLE);
    let v = body_json(resp.into_body()).await;
    assert_openai_error_envelope(&v, "service_unavailable_error");
}
