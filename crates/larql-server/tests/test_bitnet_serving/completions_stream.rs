//! `POST /v1/completions` with `stream: true` on a BitNet container: the
//! native-ternary SSE arm, and its stop-string and failure paths.

use axum::http::StatusCode;

use super::{
    bitnet_app, broken_bitnet_app, finish_reason, post, sse_events, token_prompt, BitnetBuild,
};

const COMPLETIONS: &str = "/v1/completions";
const MAX_TOKENS: usize = 3;

fn completion_body(prompt: &str) -> serde_json::Value {
    serde_json::json!({"prompt": prompt, "max_tokens": MAX_TOKENS, "stream": true})
}

fn texts(events: &[serde_json::Value]) -> Vec<String> {
    events
        .iter()
        .filter_map(|e| e["choices"][0]["text"].as_str())
        .filter(|t| !t.is_empty())
        .map(str::to_string)
        .collect()
}

#[tokio::test]
async fn stream_emits_ternary_tokens_and_a_finish_reason() {
    let (app, _fixture) = bitnet_app(BitnetBuild::DenseOnly);
    let (status, body) = post(
        &app,
        COMPLETIONS,
        completion_body(&token_prompt(&[7, 8, 9])),
    )
    .await;
    assert_eq!(status, StatusCode::OK, "{body}");
    assert!(body.ends_with("data: [DONE]\n\n"), "{body}");
    let events = sse_events(&body);
    assert!(
        events.iter().all(|e| e.get("error").is_none()),
        "no error chunk expected:\n{body}"
    );
    let emitted = texts(&events);
    assert!(!emitted.is_empty() && emitted.len() <= MAX_TOKENS, "{body}");
    let reason = finish_reason(&events).expect("final chunk finish_reason");
    let expected = if emitted.len() < MAX_TOKENS {
        "stop"
    } else {
        "length"
    };
    assert_eq!(reason, expected, "{body}");
}

#[tokio::test]
async fn stream_stops_early_on_a_stop_string() {
    let (app, _fixture) = bitnet_app(BitnetBuild::DenseOnly);
    let mut body = completion_body(&token_prompt(&[7, 8, 9]));
    // Every fixture token spells `[N]`, so `[` matches the first one.
    body["stop"] = serde_json::json!(["["]);
    let (status, text) = post(&app, COMPLETIONS, body).await;
    assert_eq!(status, StatusCode::OK, "{text}");
    let events = sse_events(&text);
    assert_eq!(
        texts(&events).len(),
        1,
        "halts after the first token: {text}"
    );
    assert_eq!(finish_reason(&events).as_deref(), Some("stop"), "{text}");
}

#[tokio::test]
async fn failed_ternary_load_surfaces_as_an_error_chunk() {
    let (app, _fixture) = broken_bitnet_app(BitnetBuild::DenseOnly);
    let (status, text) = post(&app, COMPLETIONS, completion_body("[1]")).await;
    assert_eq!(status, StatusCode::OK, "{text}");
    assert!(text.contains("failed to load bitnet model"), "{text}");
}

#[tokio::test]
async fn a_prompt_that_tokenises_to_nothing_surfaces_as_an_error_chunk() {
    let (app, _fixture) = bitnet_app(BitnetBuild::DenseOnly);
    let (status, text) = post(&app, COMPLETIONS, completion_body("")).await;
    assert_eq!(status, StatusCode::OK, "{text}");
    assert!(text.contains("tokenises to empty"), "{text}");
}

#[tokio::test]
async fn non_streaming_completion_names_the_bitnet_refusal() {
    let (app, _fixture) = bitnet_app(BitnetBuild::DenseOnly);
    let mut body = completion_body("[1]");
    body["stream"] = serde_json::json!(false);
    let (status, text) = post(&app, COMPLETIONS, body).await;
    assert!(
        status.is_client_error() || status.is_server_error(),
        "{text}"
    );
    assert!(text.contains("--keep-quant"), "{text}");
}
