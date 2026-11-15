//! `POST /v1/chat/completions` with `stream: true` on a BitNet container:
//! the native-ternary SSE arm, and the up-front 400 for tools /
//! constrained generation (refused before the stream opens, not as an
//! error chunk inside a 200).

use axum::http::StatusCode;

use super::{
    bitnet_app, broken_bitnet_app, finish_reason, post, sse_events, token_prompt, BitnetBuild,
};

const CHAT: &str = "/v1/chat/completions";
const MAX_TOKENS: usize = 3;

fn chat_body(content: &str) -> serde_json::Value {
    serde_json::json!({
        "messages": [{"role": "user", "content": content}],
        "max_tokens": MAX_TOKENS,
        "stream": true,
    })
}

fn deltas(events: &[serde_json::Value]) -> Vec<String> {
    events
        .iter()
        .filter_map(|e| e["choices"][0]["delta"]["content"].as_str())
        .map(str::to_string)
        .collect()
}

#[tokio::test]
async fn stream_emits_role_ternary_content_and_finish() {
    let (app, _fixture) = bitnet_app(BitnetBuild::DenseOnly);
    let (status, body) = post(&app, CHAT, chat_body(&token_prompt(&[4, 5, 6]))).await;
    assert_eq!(status, StatusCode::OK, "{body}");
    assert!(body.ends_with("data: [DONE]\n\n"), "{body}");
    let events = sse_events(&body);
    assert!(
        events.iter().all(|e| e.get("error").is_none()),
        "no error chunk expected:\n{body}"
    );
    assert_eq!(
        events[0]["choices"][0]["delta"]["role"], "assistant",
        "{body}"
    );
    let content = deltas(&events);
    assert!(!content.is_empty() && content.len() <= MAX_TOKENS, "{body}");
    let reason = finish_reason(&events).expect("final chunk finish_reason");
    assert!(reason == "stop" || reason == "length", "{body}");
}

#[tokio::test]
async fn stream_halts_on_a_stop_string() {
    let (app, _fixture) = bitnet_app(BitnetBuild::DenseOnly);
    let mut body = chat_body(&token_prompt(&[4, 5, 6]));
    // Every fixture token spells `[N]`, so `[` matches the first one.
    body["stop"] = serde_json::json!(["["]);
    let (status, text) = post(&app, CHAT, body).await;
    assert_eq!(status, StatusCode::OK, "{text}");
    let events = sse_events(&text);
    assert_eq!(finish_reason(&events).as_deref(), Some("stop"), "{text}");
    assert!(
        deltas(&events).len() < MAX_TOKENS,
        "halts before max_tokens: {text}"
    );
}

#[tokio::test]
async fn tools_are_refused_before_streaming_with_a_400() {
    let (app, _fixture) = bitnet_app(BitnetBuild::DenseOnly);
    let mut body = chat_body("[1]");
    body["tools"] = serde_json::json!([{
        "type": "function",
        "function": {
            "name": "lookup",
            "parameters": {"type": "object", "properties": {"q": {"type": "string"}}},
        },
    }]);
    let (status, text) = post(&app, CHAT, body).await;
    assert_eq!(status, StatusCode::BAD_REQUEST, "{text}");
    assert!(!text.contains("data:"), "no stream opened: {text}");
    let v: serde_json::Value = serde_json::from_str(&text).expect("json error body");
    let msg = v["error"]["message"].as_str().unwrap_or_default();
    assert!(msg.contains("BitNet") && msg.contains("tools"), "{text}");
}

#[tokio::test]
async fn constrained_response_format_is_refused_before_streaming_with_a_400() {
    let (app, _fixture) = bitnet_app(BitnetBuild::DenseOnly);
    let mut body = chat_body("[1]");
    body["response_format"] = serde_json::json!({"type": "json_object"});
    let (status, text) = post(&app, CHAT, body).await;
    assert_eq!(status, StatusCode::BAD_REQUEST, "{text}");
    assert!(text.contains("constrained generation"), "{text}");
}

#[tokio::test]
async fn failed_ternary_load_surfaces_as_an_error_chunk() {
    let (app, _fixture) = broken_bitnet_app(BitnetBuild::DenseOnly);
    let (status, text) = post(&app, CHAT, chat_body("[1]")).await;
    assert_eq!(status, StatusCode::OK, "{text}");
    assert!(text.contains("failed to load bitnet model"), "{text}");
    assert!(text.contains("[DONE]"), "{text}");
}
