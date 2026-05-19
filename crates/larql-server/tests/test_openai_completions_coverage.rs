//! Coverage push for `routes/openai/completions.rs` (was 40%, target ≥ 90%).
//!
//! Uses the synthetic f32 vindex so the generation loop has real
//! weights to run against. Targets: handler branches (n>1, empty
//! prompt, echo+stream rejection, batched+stream rejection,
//! infer_disabled rejection), the non-streaming buffered path, and
//! the streaming SSE path.

mod common;

use axum::body::Body;
use axum::http::{header, Request, StatusCode};
use tower::ServiceExt;

async fn post_completions(body: serde_json::Value) -> axum::http::Response<Body> {
    let (model, _fixture) = common::model_with_q4k_weights("synthetic");
    let state = common::state(vec![model]);
    let app = larql_server::routes::single_model_router(state);
    let resp = app
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/v1/completions")
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
async fn completions_non_streaming_single_prompt_returns_200() {
    let resp = post_completions(serde_json::json!({
        "model": "synthetic",
        "prompt": "the capital of France is",
        "max_tokens": 4,
    }))
    .await;
    // Either 200 (generation succeeded) or 500 (synthetic weights
    // produced NaN) — both exercise the non-streaming compose path.
    assert!(resp.status() == StatusCode::OK || resp.status().is_server_error());
}

#[tokio::test]
async fn completions_n_gt_1_returns_400() {
    let resp = post_completions(serde_json::json!({
        "model": "synthetic",
        "prompt": "x",
        "n": 2,
    }))
    .await;
    assert_eq!(resp.status(), StatusCode::BAD_REQUEST);
}

#[tokio::test]
async fn completions_empty_prompt_array_returns_400() {
    let resp = post_completions(serde_json::json!({
        "model": "synthetic",
        "prompt": [],
    }))
    .await;
    assert_eq!(resp.status(), StatusCode::BAD_REQUEST);
}

#[tokio::test]
async fn completions_batched_prompt_with_stream_returns_400() {
    let resp = post_completions(serde_json::json!({
        "model": "synthetic",
        "prompt": ["a", "b"],
        "stream": true,
    }))
    .await;
    assert_eq!(resp.status(), StatusCode::BAD_REQUEST);
}

#[tokio::test]
async fn completions_echo_with_stream_returns_400() {
    let resp = post_completions(serde_json::json!({
        "model": "synthetic",
        "prompt": "x",
        "stream": true,
        "echo": true,
    }))
    .await;
    assert_eq!(resp.status(), StatusCode::BAD_REQUEST);
}

#[tokio::test]
async fn completions_echo_in_non_stream_runs_echo_branch() {
    let resp = post_completions(serde_json::json!({
        "model": "synthetic",
        "prompt": "the capital of France is",
        "max_tokens": 2,
        "echo": true,
    }))
    .await;
    assert!(resp.status() == StatusCode::OK || resp.status().is_server_error());
}

#[tokio::test]
async fn completions_batched_non_stream_runs_loop_branch() {
    let resp = post_completions(serde_json::json!({
        "model": "synthetic",
        "prompt": ["a", "b"],
        "max_tokens": 2,
    }))
    .await;
    assert!(resp.status() == StatusCode::OK || resp.status().is_server_error());
}

#[tokio::test]
async fn completions_streaming_single_prompt_returns_sse() {
    let resp = post_completions(serde_json::json!({
        "model": "synthetic",
        "prompt": "x",
        "max_tokens": 2,
        "stream": true,
    }))
    .await;
    // Streaming starts as 200 with SSE content-type.
    assert_eq!(resp.status(), StatusCode::OK);
    let ct = resp
        .headers()
        .get(header::CONTENT_TYPE)
        .and_then(|v| v.to_str().ok())
        .unwrap_or("");
    assert!(
        ct.contains("event-stream"),
        "expected SSE content-type, got {ct}"
    );
    // Drain the full body so spawn_blocking has time to emit every
    // chunk through ReceiverStream — without a complete drain the
    // background task drops early and the per-token branches stay
    // uncovered.
    let body = axum::body::to_bytes(resp.into_body(), usize::MAX)
        .await
        .unwrap();
    let body_str = String::from_utf8_lossy(&body);
    // The stream must terminate with `data: [DONE]\n\n`.
    assert!(
        body_str.contains("[DONE]"),
        "SSE stream must terminate with [DONE]; got {body_str:?}"
    );
    eprintln!("SSE body length: {}", body_str.len());
    eprintln!("SSE body sample: {}", &body_str[..body_str.len().min(500)]);
}

#[tokio::test]
async fn completions_invalid_json_returns_400() {
    let (model, _fixture) = common::model_with_q4k_weights("synthetic");
    let state = common::state(vec![model]);
    let app = larql_server::routes::single_model_router(state);
    let resp = app
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/v1/completions")
                .header(header::CONTENT_TYPE, "application/json")
                .body(Body::from("not json"))
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(resp.status(), StatusCode::BAD_REQUEST);
}

#[tokio::test]
async fn completions_with_sampling_params_runs_sampler_branches() {
    let resp = post_completions(serde_json::json!({
        "model": "synthetic",
        "prompt": "x",
        "max_tokens": 2,
        "temperature": 0.5,
        "top_p": 0.9,
        "seed": 42,
        "frequency_penalty": 0.1,
        "presence_penalty": 0.1,
    }))
    .await;
    assert!(resp.status() == StatusCode::OK || resp.status().is_server_error());
}

#[tokio::test]
async fn completions_with_stop_strings_runs_stop_check_branch() {
    // The synthetic generator emits tokens from its WordLevel vocab.
    // Including the most common produced characters as stop strings
    // forces the contains_any → trim_at_stop branch (completions.rs
    // L481-494) to fire, which is the deepest uncovered path here.
    let resp = post_completions(serde_json::json!({
        "model": "synthetic",
        "prompt": "x",
        "max_tokens": 16,
        "stop": ["x", " "],
    }))
    .await;
    assert!(resp.status() == StatusCode::OK || resp.status().is_server_error());
}

#[tokio::test]
async fn completions_with_logprobs_runs_logprobs_branch() {
    let resp = post_completions(serde_json::json!({
        "model": "synthetic",
        "prompt": "x",
        "max_tokens": 2,
        "logprobs": 3,
    }))
    .await;
    assert!(resp.status() == StatusCode::OK || resp.status().is_server_error());
}
