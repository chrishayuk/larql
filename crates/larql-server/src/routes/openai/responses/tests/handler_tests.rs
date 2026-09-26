//! Unit tests for handler helpers whose end-to-end trigger needs a
//! model that can emit tool-call JSON — out of reach for the synthetic
//! fixtures, so the shaping logic is pinned directly.

use super::super::engine::GenerationOutcome;
use super::super::handler::{assistant_turn_text, build_output, flatten_message};
use super::super::types::{OutputContent, OutputItem, ResponseObject, RESPONSE_OBJECT};
use crate::routes::openai::chat::ChatMessage;

fn outcome(text: &str, stopped: bool) -> GenerationOutcome {
    GenerationOutcome {
        text: text.to_string(),
        stopped,
        prompt_tokens: 3,
        completion_tokens: 2,
        kv_handoff: None,
        reused_prompt_tokens: 0,
        continuation_refused: None,
        tally: crate::runtime_stats::GenerationTally::new(),
    }
}

fn envelope_with(output: Vec<OutputItem>) -> ResponseObject {
    ResponseObject {
        id: "resp_test".to_string(),
        object: RESPONSE_OBJECT,
        created_at: 0,
        status: "completed".to_string(),
        error: None,
        incomplete_details: None,
        model: "m".to_string(),
        output,
        previous_response_id: None,
        instructions: None,
        max_output_tokens: None,
        temperature: None,
        top_p: None,
        metadata: None,
        usage: None,
    }
}

// ── build_output, tools active ──────────────────────────────────────

#[test]
fn build_output_tools_parses_function_call_item() {
    let out = outcome(
        r#"{"name":"get_weather","arguments":{"city":"Paris"}}"#,
        true,
    );
    let (items, status, incomplete) = build_output(true, &out).unwrap();
    assert_eq!(status, "completed");
    assert!(incomplete.is_none());
    match &items[0] {
        OutputItem::FunctionCall {
            id,
            call_id,
            name,
            arguments,
            status,
        } => {
            assert!(id.starts_with("fc_"), "{id}");
            assert!(call_id.starts_with("call_"), "{call_id}");
            assert_eq!(name, "get_weather");
            assert_eq!(arguments, r#"{"city":"Paris"}"#);
            assert_eq!(status, "completed");
        }
        other => panic!("expected FunctionCall, got {other:?}"),
    }
}

#[test]
fn build_output_tools_unparseable_text_errors() {
    let err = build_output(true, &outcome("not json", true)).unwrap_err();
    assert!(err.contains("tool_call output failed to parse"), "{err}");
}

// ── assistant_turn_text ─────────────────────────────────────────────

#[test]
fn assistant_turn_text_prefers_message_text() {
    let envelope = envelope_with(vec![OutputItem::Message {
        id: "msg_1".to_string(),
        status: "completed".to_string(),
        role: "assistant",
        content: vec![OutputContent::OutputText {
            text: "Paris".to_string(),
            annotations: Vec::new(),
        }],
    }]);
    assert_eq!(assistant_turn_text(&envelope), "Paris");
}

#[test]
fn assistant_turn_text_renders_function_call_when_no_text() {
    let envelope = envelope_with(vec![OutputItem::FunctionCall {
        id: "fc_1".to_string(),
        status: "completed".to_string(),
        call_id: "call_1".to_string(),
        name: "get_weather".to_string(),
        arguments: r#"{"city":"Paris"}"#.to_string(),
    }]);
    let text = assistant_turn_text(&envelope);
    assert!(text.contains("get_weather"), "{text}");
    assert!(text.contains(r#"{"city":"Paris"}"#), "{text}");
}

#[test]
fn assistant_turn_text_empty_output_is_empty() {
    assert_eq!(assistant_turn_text(&envelope_with(Vec::new())), "");
}

// ── flatten_message ─────────────────────────────────────────────────

#[test]
fn flatten_message_tool_role_becomes_user_tool_result() {
    let m = ChatMessage {
        role: "tool".to_string(),
        content: Some("22C".to_string()),
        tool_calls: None,
        tool_call_id: Some("call_1".to_string()),
        name: None,
    };
    let stored = flatten_message(&m);
    assert_eq!(stored.role, "user");
    assert!(stored.content.contains("22C"), "{}", stored.content);
    assert!(stored.content.contains("call_1"), "{}", stored.content);
}

#[test]
fn flatten_message_assistant_tool_calls_render_to_text() {
    let m = ChatMessage {
        role: "assistant".to_string(),
        content: None,
        tool_calls: Some(serde_json::json!([{
            "id": "call_9",
            "type": "function",
            "function": {"name": "f", "arguments": "{}"},
        }])),
        tool_call_id: None,
        name: None,
    };
    let stored = flatten_message(&m);
    assert_eq!(stored.role, "assistant");
    assert!(stored.content.contains('f'), "{}", stored.content);
}

#[test]
fn flatten_message_no_content_no_calls_is_empty() {
    let m = ChatMessage {
        role: "assistant".to_string(),
        content: None,
        tool_calls: None,
        tool_call_id: None,
        name: None,
    };
    assert_eq!(flatten_message(&m).content, "");
}
