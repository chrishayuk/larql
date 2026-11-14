//! End-to-end coverage for `routes/openai/responses/*` — the OpenAI
//! Responses API surface (`POST /v1/responses`, `GET`/`DELETE
//! /v1/responses/{id}`) plus `GET /v1/models/{model}`.
//!
//! Uses the Q4K synthetic vindex (see `test_openai_chat_coverage.rs`)
//! so the generation loop actually runs. Response bodies are always
//! drained so the handler's success paths register in coverage.

mod common;

use axum::body::Body;
use axum::http::{header, Request, StatusCode};
use tower::ServiceExt;

fn q4k_app() -> (
    axum::Router,
    common::synthetic_q4k_vindex::SyntheticQ4kVindex,
) {
    let (model, fixture) = common::model_with_q4k_weights("synthetic");
    let state = common::state(vec![model]);
    (larql_server::routes::single_model_router(state), fixture)
}

async fn send(
    app: &axum::Router,
    method: &str,
    uri: &str,
    body: Option<serde_json::Value>,
) -> axum::http::Response<Body> {
    let builder = Request::builder()
        .method(method)
        .uri(uri)
        .header(header::CONTENT_TYPE, "application/json");
    let request = match body {
        Some(v) => builder.body(Body::from(serde_json::to_vec(&v).unwrap())),
        None => builder.body(Body::empty()),
    }
    .unwrap();
    app.clone().oneshot(request).await.unwrap()
}

/// Drain the body and parse JSON; returns `(status, json)` so callers
/// can assert on both without double-consuming the response.
async fn drain(resp: axum::http::Response<Body>) -> (StatusCode, serde_json::Value) {
    let status = resp.status();
    let bytes = axum::body::to_bytes(resp.into_body(), usize::MAX)
        .await
        .unwrap();
    let json = serde_json::from_slice(&bytes)
        .unwrap_or_else(|_| serde_json::json!({"raw": String::from_utf8_lossy(&bytes)}));
    (status, json)
}

async fn drain_text(resp: axum::http::Response<Body>) -> (StatusCode, String) {
    let status = resp.status();
    let bytes = axum::body::to_bytes(resp.into_body(), usize::MAX)
        .await
        .unwrap();
    (status, String::from_utf8_lossy(&bytes).into_owned())
}

// ── happy paths ──────────────────────────────────────────────────────────────

#[tokio::test]
async fn responses_non_streaming_returns_envelope() {
    let (app, _fx) = q4k_app();
    let resp = send(
        &app,
        "POST",
        "/v1/responses",
        Some(serde_json::json!({
            "model": "synthetic",
            "input": "the capital of France is",
            "max_output_tokens": 4,
        })),
    )
    .await;
    let (status, v) = drain(resp).await;
    assert_eq!(status, StatusCode::OK, "{v}");
    assert!(v["id"].as_str().unwrap().starts_with("resp_"), "{v}");
    assert_eq!(v["object"], "response", "{v}");
    assert!(
        v["status"] == "completed" || v["status"] == "incomplete",
        "{v}"
    );
    assert_eq!(v["output"][0]["type"], "message", "{v}");
    assert_eq!(v["output"][0]["role"], "assistant", "{v}");
    assert_eq!(v["output"][0]["content"][0]["type"], "output_text", "{v}");
    let usage = &v["usage"];
    assert!(usage["input_tokens"].as_u64().unwrap() > 0, "{v}");
    assert_eq!(
        usage["total_tokens"].as_u64().unwrap(),
        usage["input_tokens"].as_u64().unwrap() + usage["output_tokens"].as_u64().unwrap(),
        "{v}"
    );
}

#[tokio::test]
async fn responses_accepts_item_list_and_instructions() {
    let (app, _fx) = q4k_app();
    let resp = send(
        &app,
        "POST",
        "/v1/responses",
        Some(serde_json::json!({
            "model": "synthetic",
            "instructions": "Answer briefly.",
            "input": [
                {"role": "user", "content": [{"type": "input_text", "text": "the sky is"}]},
                {"type": "reasoning", "content": []},
            ],
            "max_output_tokens": 2,
        })),
    )
    .await;
    let (status, v) = drain(resp).await;
    assert_eq!(status, StatusCode::OK, "{v}");
    assert_eq!(v["instructions"], "Answer briefly.", "{v}");
}

#[tokio::test]
async fn responses_streaming_emits_typed_events_and_done() {
    let (app, _fx) = q4k_app();
    let resp = send(
        &app,
        "POST",
        "/v1/responses",
        Some(serde_json::json!({
            "model": "synthetic",
            "input": "x",
            "max_output_tokens": 3,
            "stream": true,
        })),
    )
    .await;
    assert_eq!(resp.status(), StatusCode::OK);
    let ct = resp
        .headers()
        .get(header::CONTENT_TYPE)
        .and_then(|v| v.to_str().ok())
        .unwrap_or("")
        .to_string();
    assert!(ct.contains("event-stream"), "expected SSE; got {ct}");
    let (_, body) = drain_text(resp).await;
    for needle in [
        "event: response.created",
        "event: response.in_progress",
        "event: response.output_item.added",
        "event: response.content_part.added",
        "event: response.output_text.done",
        "event: response.content_part.done",
        "event: response.output_item.done",
        "event: response.completed",
        "[DONE]",
        "sequence_number",
    ] {
        assert!(body.contains(needle), "missing {needle:?} in:\n{body}");
    }
    // A delta event is sent for each token that decodes to text. Tokens of
    // this synthetic fixture can decode to nothing, and those send no delta,
    // so the delta event is asserted only when the stream carried output
    // text.
    if body.contains("\"text\":\"\"") {
        assert!(
            !body.contains("event: response.output_text.delta"),
            "empty output must not carry deltas:\n{body}"
        );
    } else {
        assert!(
            body.contains("event: response.output_text.delta"),
            "non-empty output should stream deltas:\n{body}"
        );
    }
}

#[tokio::test]
async fn responses_store_chain_get_and_delete_round_trip() {
    let (app, _fx) = q4k_app();

    // 1. Create (store defaults to true).
    let resp = send(
        &app,
        "POST",
        "/v1/responses",
        Some(serde_json::json!({
            "model": "synthetic",
            "input": "count: one",
            "max_output_tokens": 2,
        })),
    )
    .await;
    let (status, v) = drain(resp).await;
    assert_eq!(status, StatusCode::OK, "{v}");
    let id = v["id"].as_str().unwrap().to_string();

    // 2. Retrieve it.
    let resp = send(&app, "GET", &format!("/v1/responses/{id}"), None).await;
    let (status, got) = drain(resp).await;
    assert_eq!(status, StatusCode::OK, "{got}");
    assert_eq!(got["id"], id.as_str(), "{got}");

    // 3. Chain a follow-up from it — model comes from the stored
    //    response, so the request may omit it.
    let resp = send(
        &app,
        "POST",
        "/v1/responses",
        Some(serde_json::json!({
            "input": "count: two",
            "previous_response_id": id,
            "max_output_tokens": 2,
        })),
    )
    .await;
    let (status, chained) = drain(resp).await;
    assert_eq!(status, StatusCode::OK, "{chained}");
    assert_eq!(chained["previous_response_id"], id.as_str(), "{chained}");

    // 4. Delete the original; a second delete and a get are 404.
    let resp = send(&app, "DELETE", &format!("/v1/responses/{id}"), None).await;
    let (status, receipt) = drain(resp).await;
    assert_eq!(status, StatusCode::OK, "{receipt}");
    assert_eq!(receipt["deleted"], true, "{receipt}");
    let resp = send(&app, "DELETE", &format!("/v1/responses/{id}"), None).await;
    assert_eq!(drain(resp).await.0, StatusCode::NOT_FOUND);
    let resp = send(&app, "GET", &format!("/v1/responses/{id}"), None).await;
    assert_eq!(drain(resp).await.0, StatusCode::NOT_FOUND);
}

#[tokio::test]
async fn responses_store_false_is_not_retrievable() {
    let (app, _fx) = q4k_app();
    let resp = send(
        &app,
        "POST",
        "/v1/responses",
        Some(serde_json::json!({
            "model": "synthetic",
            "input": "x",
            "max_output_tokens": 2,
            "store": false,
        })),
    )
    .await;
    let (status, v) = drain(resp).await;
    assert_eq!(status, StatusCode::OK, "{v}");
    let id = v["id"].as_str().unwrap().to_string();
    let resp = send(&app, "GET", &format!("/v1/responses/{id}"), None).await;
    assert_eq!(drain(resp).await.0, StatusCode::NOT_FOUND);
}

#[tokio::test]
async fn responses_tools_produce_function_call_item() {
    let (app, _fx) = q4k_app();
    let resp = send(
        &app,
        "POST",
        "/v1/responses",
        Some(serde_json::json!({
            "model": "synthetic",
            "input": "add one and two",
            "max_output_tokens": 64,
            "tools": [{
                "type": "function",
                "name": "calc",
                "description": "adds numbers",
                "parameters": {
                    "type": "object",
                    "properties": {"a": {"type": "integer"}, "b": {"type": "integer"}},
                    "required": ["a", "b"],
                },
            }],
            "tool_choice": "required",
        })),
    )
    .await;
    let (status, v) = drain(resp).await;
    // Constrained decoding on the synthetic model can legitimately
    // produce unparseable output (400); when it parses, the shape must
    // be a function_call item.
    if status == StatusCode::OK {
        assert_eq!(v["output"][0]["type"], "function_call", "{v}");
        assert_eq!(v["output"][0]["name"], "calc", "{v}");
        assert!(
            v["output"][0]["call_id"]
                .as_str()
                .unwrap()
                .starts_with("call_"),
            "{v}"
        );
    } else {
        assert_eq!(status, StatusCode::BAD_REQUEST, "{v}");
    }
}

#[tokio::test]
async fn responses_json_object_format_constrains_output() {
    let (app, _fx) = q4k_app();
    let resp = send(
        &app,
        "POST",
        "/v1/responses",
        Some(serde_json::json!({
            "model": "synthetic",
            "input": "emit an object",
            "max_output_tokens": 8,
            "text": {"format": {"type": "json_object"}},
        })),
    )
    .await;
    let (status, v) = drain(resp).await;
    assert_eq!(status, StatusCode::OK, "{v}");
}

// ── validation / error paths ─────────────────────────────────────────────────

#[tokio::test]
async fn responses_background_true_is_400() {
    let (app, _fx) = q4k_app();
    let resp = send(
        &app,
        "POST",
        "/v1/responses",
        Some(serde_json::json!({"model": "synthetic", "input": "x", "background": true})),
    )
    .await;
    assert_eq!(drain(resp).await.0, StatusCode::BAD_REQUEST);
}

#[tokio::test]
async fn responses_empty_item_list_is_400() {
    let (app, _fx) = q4k_app();
    let resp = send(
        &app,
        "POST",
        "/v1/responses",
        Some(serde_json::json!({"model": "synthetic", "input": []})),
    )
    .await;
    assert_eq!(drain(resp).await.0, StatusCode::BAD_REQUEST);
}

#[tokio::test]
async fn responses_unknown_item_type_is_400() {
    let (app, _fx) = q4k_app();
    let resp = send(
        &app,
        "POST",
        "/v1/responses",
        Some(serde_json::json!({
            "model": "synthetic",
            "input": [{"type": "input_image", "role": "user"}],
        })),
    )
    .await;
    let (status, v) = drain(resp).await;
    assert_eq!(status, StatusCode::BAD_REQUEST, "{v}");
}

#[tokio::test]
async fn responses_unknown_previous_response_is_404() {
    let (app, _fx) = q4k_app();
    let resp = send(
        &app,
        "POST",
        "/v1/responses",
        Some(serde_json::json!({
            "model": "synthetic",
            "input": "x",
            "previous_response_id": "resp_does_not_exist",
        })),
    )
    .await;
    assert_eq!(drain(resp).await.0, StatusCode::NOT_FOUND);
}

#[tokio::test]
async fn responses_unknown_model_is_404() {
    let (app, _fx) = q4k_app();
    let resp = send(
        &app,
        "POST",
        "/v1/responses",
        Some(serde_json::json!({"model": "nope", "input": "x"})),
    )
    .await;
    assert_eq!(drain(resp).await.0, StatusCode::NOT_FOUND);
}

#[tokio::test]
async fn responses_invalid_json_is_400() {
    let (app, _fx) = q4k_app();
    let resp = app
        .clone()
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/v1/responses")
                .header(header::CONTENT_TYPE, "application/json")
                .body(Body::from("not json"))
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(resp.status(), StatusCode::BAD_REQUEST);
}

#[tokio::test]
async fn responses_get_unknown_id_is_404() {
    let (app, _fx) = q4k_app();
    let resp = send(&app, "GET", "/v1/responses/resp_missing", None).await;
    assert_eq!(drain(resp).await.0, StatusCode::NOT_FOUND);
}

#[tokio::test]
async fn responses_infer_disabled_model_is_503() {
    let state = common::state(vec![common::model("browse-only")]);
    let app = larql_server::routes::single_model_router(state);
    let resp = send(
        &app,
        "POST",
        "/v1/responses",
        Some(serde_json::json!({"model": "browse-only", "input": "x"})),
    )
    .await;
    assert_eq!(drain(resp).await.0, StatusCode::SERVICE_UNAVAILABLE);
}

// ── GET /v1/models/{model} ───────────────────────────────────────────────────

#[tokio::test]
async fn model_retrieve_returns_entry_and_404() {
    let (app, _fx) = q4k_app();
    let resp = send(&app, "GET", "/v1/models/synthetic", None).await;
    let (status, v) = drain(resp).await;
    assert_eq!(status, StatusCode::OK, "{v}");
    assert_eq!(v["id"], "synthetic", "{v}");
    assert_eq!(v["object"], "model", "{v}");
    assert_eq!(v["owned_by"], "larql", "{v}");

    let resp = send(&app, "GET", "/v1/models/absent", None).await;
    assert_eq!(drain(resp).await.0, StatusCode::NOT_FOUND);
}

#[tokio::test]
async fn model_retrieve_works_on_multi_model_router() {
    let state = common::state(vec![common::model("alpha"), common::model("beta")]);
    let app = larql_server::routes::multi_model_router(state);
    let resp = send(&app, "GET", "/v1/models/beta", None).await;
    let (status, v) = drain(resp).await;
    assert_eq!(status, StatusCode::OK, "{v}");
    assert_eq!(v["id"], "beta", "{v}");
    // Multi-model path extras point at the per-model prefix.
    assert_eq!(v["path"], "/v1/beta", "{v}");
}

#[tokio::test]
async fn responses_user_tag_is_accepted() {
    let (app, _fx) = q4k_app();
    let resp = send(
        &app,
        "POST",
        "/v1/responses",
        Some(serde_json::json!({
            "model": "synthetic",
            "input": "x",
            "max_output_tokens": 2,
            "user": "tester-1",
        })),
    )
    .await;
    let (status, v) = drain(resp).await;
    assert_eq!(status, StatusCode::OK, "{v}");
}

#[tokio::test]
async fn responses_streaming_generation_failure_emits_failed_event() {
    // Deliberately destroy the fixture BEFORE the request: the
    // synthetic weights load lazily from the fixture dir during
    // generation (while the stream is consumed), so the load fails and
    // the stream must carry `response.failed` (with an error envelope)
    // and still terminate with [DONE]. Dropping after the request would
    // race the generation thread's load.
    let (app, fixture) = q4k_app();
    drop(fixture);
    let resp = send(
        &app,
        "POST",
        "/v1/responses",
        Some(serde_json::json!({
            "model": "synthetic",
            "input": "x",
            "max_output_tokens": 2,
            "stream": true,
        })),
    )
    .await;
    assert_eq!(resp.status(), StatusCode::OK);
    let (_, body) = drain_text(resp).await;
    assert!(body.contains("event: response.failed"), "{body}");
    assert!(body.contains("server_error"), "{body}");
    assert!(body.contains("[DONE]"), "{body}");
}

#[tokio::test]
async fn responses_streaming_with_tools_fails_or_emits_function_call() {
    // The synthetic vocab cannot spell tool-call JSON: the constrained
    // generation either errors up front (unrepresentable prompt) or
    // produces unparseable output, so the stream must end in either a
    // `response.failed` event or — if parsing somehow succeeds — a
    // function-call item pair. All three arms are valid terminations;
    // what matters is a typed terminal event plus [DONE].
    let (app, _fx) = q4k_app();
    let resp = send(
        &app,
        "POST",
        "/v1/responses",
        Some(serde_json::json!({
            "model": "synthetic",
            "input": "x",
            "max_output_tokens": 4,
            "stream": true,
            "tools": [{
                "type": "function",
                "name": "get_weather",
                "parameters": {
                    "type": "object",
                    "properties": {"city": {"type": "string"}},
                    "required": ["city"]
                }
            }],
        })),
    )
    .await;
    if resp.status() != StatusCode::OK {
        assert_eq!(resp.status(), StatusCode::BAD_REQUEST);
        return;
    }
    let (_, body) = drain_text(resp).await;
    assert!(body.contains("[DONE]"), "{body}");
    assert!(
        body.contains("event: response.failed") || body.contains("function_call"),
        "stream must end in failed or a function_call item:\n{body}"
    );
}
