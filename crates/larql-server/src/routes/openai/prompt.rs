//! Chat-template selection + message rendering shared by the OpenAI
//! conversation endpoints (`/v1/chat/completions`, `/v1/responses`).
//!
//! This module (a) picks the heuristic template variant for a VINDEX2
//! model, whose rendering lives in
//! `larql_inference::prompt::ChatTemplate::render_messages`, and (b)
//! flattens tool-call turns into the `system`/`user`/`assistant` slots
//! every template natively understands ([`flatten`]) — the one
//! flattening both runtimes use. A VINDEX3 container renders through
//! its own template instead ([`super::v3_prompt`]). Centralised here so
//! the chat and responses endpoints cannot drift in how a conversation
//! becomes a prompt.

use super::chat::ChatMessage;

/// Role strings shared across the OpenAI conversation endpoints.
pub(super) const ASSISTANT_ROLE: &str = "assistant";
pub(super) const SYSTEM_ROLE: &str = "system";
pub(super) const USER_ROLE: &str = "user";
pub(super) const TOOL_ROLE: &str = "tool";

/// Pick the chat template for a V2 model from the caller's weights
/// guard.
///
/// The caller must pass the guard it already holds — reading the
/// `model.weights` RwLock here instead once deadlocked the chat
/// handler, because `run_chat_completion` holds the write lock from
/// `lock_weights_for_gen()` and `std::sync::RwLock` blocks the same
/// thread on a read after a write (POSIX `pthread_rwlock_rdlock`
/// semantics on macOS). Every generation path holds the guard, so no
/// guard-less variant exists.
pub(super) fn pick_template(
    weights: &larql_inference::ModelWeights,
) -> larql_inference::prompt::ChatTemplate {
    larql_inference::prompt::ChatTemplate::for_family(weights.arch.family())
}

/// Render a wire `ChatMessage` list with a heuristic VINDEX2 template,
/// over the [`flatten`]ed turns.
pub(super) fn render(
    template: larql_inference::prompt::ChatTemplate,
    messages: &[ChatMessage],
) -> String {
    let pairs = flatten(messages);
    template.render_messages(pairs.iter().map(|(r, c)| (r.as_str(), c.as_str())))
}

/// Adapter: convert a wire `ChatMessage` list to `(role, content)`
/// pairs. Chat templates reliably handle `system` / `user` /
/// `assistant` only, so tool turns are flattened into text content that
/// fits within those slots:
///
/// - Assistant message with `tool_calls` (and `content: null`) →
///   assistant turn whose content is a serialised summary of the tool
///   calls (`Tool call: <name>(<arguments>)`). Any prior `content`
///   takes precedence when both are set.
/// - Tool message → user turn with `[Tool result for <id>: <content>]`,
///   so the model sees the result inline before generating the next
///   assistant turn.
pub(super) fn flatten(messages: &[ChatMessage]) -> Vec<(String, String)> {
    messages
        .iter()
        .map(|m| match m.role.as_str() {
            TOOL_ROLE => (
                USER_ROLE.to_string(),
                format_tool_result(m.tool_call_id.as_deref(), m.content.as_deref()),
            ),
            ASSISTANT_ROLE => {
                if let Some(c) = m.content.as_deref() {
                    (ASSISTANT_ROLE.to_string(), c.to_string())
                } else if let Some(tc) = m.tool_calls.as_ref() {
                    (ASSISTANT_ROLE.to_string(), format_tool_calls(tc))
                } else {
                    (ASSISTANT_ROLE.to_string(), String::new())
                }
            }
            other => (other.to_string(), m.content.clone().unwrap_or_default()),
        })
        .collect()
}

/// Render a tool-result message as a user-side text turn so the model
/// sees the tool output before the next assistant generation.
pub(super) fn format_tool_result(tool_call_id: Option<&str>, content: Option<&str>) -> String {
    let id = tool_call_id.unwrap_or("?");
    let body = content.unwrap_or("");
    format!("[Tool result for {id}]: {body}")
}

/// Render an assistant `tool_calls` echo as text. Multiple parallel
/// tool calls are listed; arguments stay JSON-encoded.
pub(super) fn format_tool_calls(tool_calls: &serde_json::Value) -> String {
    let arr = match tool_calls.as_array() {
        Some(a) => a,
        None => return String::new(),
    };
    let mut out = String::new();
    for (i, tc) in arr.iter().enumerate() {
        if i > 0 {
            out.push('\n');
        }
        let name = tc
            .get("function")
            .and_then(|f| f.get("name"))
            .and_then(|n| n.as_str())
            .unwrap_or("?");
        let args = tc
            .get("function")
            .and_then(|f| f.get("arguments"))
            .and_then(|a| a.as_str())
            .unwrap_or("");
        out.push_str(&format!("[Tool call: {name}({args})]"));
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn format_tool_result_includes_call_id_and_body() {
        let s = format_tool_result(Some("call_abc"), Some("23 C"));
        assert!(s.contains("call_abc"));
        assert!(s.contains("23 C"));
    }

    #[test]
    fn format_tool_result_placeholders_for_missing_fields() {
        let s = format_tool_result(None, None);
        assert!(s.contains('?'));
    }

    #[test]
    fn format_tool_calls_summarises_function_calls() {
        let tc = serde_json::json!([
            {"id": "call_1", "type": "function",
             "function": {"name": "calc", "arguments": "{\"a\":1}"}}
        ]);
        let out = format_tool_calls(&tc);
        assert!(out.contains("calc"), "missing name in {out}");
        assert!(out.contains("{\"a\":1}"), "missing args in {out}");
    }

    #[test]
    fn format_tool_calls_non_array_renders_empty() {
        assert_eq!(format_tool_calls(&serde_json::json!({"not": "array"})), "");
    }

    #[test]
    fn format_tool_calls_joins_parallel_calls_with_newlines() {
        let tc = serde_json::json!([
            {"function": {"name": "f1", "arguments": "{}"}},
            {"function": {"name": "f2", "arguments": "{}"}},
        ]);
        let out = format_tool_calls(&tc);
        assert_eq!(out.lines().count(), 2, "{out}");
        assert!(out.contains("f1") && out.contains("f2"), "{out}");
    }

    fn msg(role: &str, content: Option<&str>) -> ChatMessage {
        ChatMessage {
            role: role.to_string(),
            content: content.map(str::to_string),
            tool_calls: None,
            tool_call_id: None,
            name: None,
        }
    }

    #[test]
    fn render_flattens_tool_turn_into_user_slot() {
        let mut tool_msg = msg(TOOL_ROLE, Some("22C"));
        tool_msg.tool_call_id = Some("call_7".to_string());
        let out = render(
            larql_inference::prompt::ChatTemplate::Plain,
            &[msg(USER_ROLE, Some("weather?")), tool_msg],
        );
        assert!(out.contains("[Tool result for call_7]: 22C"), "{out}");
    }

    #[test]
    fn render_assistant_content_takes_precedence_over_tool_calls() {
        let mut m = msg(ASSISTANT_ROLE, Some("said this"));
        m.tool_calls = Some(serde_json::json!([
            {"function": {"name": "f", "arguments": "{}"}}
        ]));
        let out = render(larql_inference::prompt::ChatTemplate::Plain, &[m]);
        assert!(out.contains("said this"), "{out}");
        assert!(!out.contains("[Tool call:"), "{out}");
    }

    #[test]
    fn render_assistant_tool_calls_without_content_render_as_text() {
        let mut m = msg(ASSISTANT_ROLE, None);
        m.tool_calls = Some(serde_json::json!([
            {"function": {"name": "get_weather", "arguments": "{\"c\":1}"}}
        ]));
        let out = render(larql_inference::prompt::ChatTemplate::Plain, &[m]);
        assert!(out.contains("[Tool call: get_weather({\"c\":1})]"), "{out}");
    }

    #[test]
    fn render_assistant_with_neither_renders_empty_turn() {
        let out = render(
            larql_inference::prompt::ChatTemplate::Plain,
            &[msg(USER_ROLE, Some("hi")), msg(ASSISTANT_ROLE, None)],
        );
        assert!(out.contains("hi"), "{out}");
    }
}
