//! Wire types for `/v1/responses` (request + response object).
//!
//! Shapes follow the OpenAI Responses API. Field-by-field support is
//! documented on each struct; unsupported-but-harmless fields are
//! accepted and echoed, unsupported-and-meaningful fields are rejected
//! with a 400 in the handler so clients never silently lose semantics.

use serde::{Deserialize, Serialize};

use super::super::util::StopSpec;

/// `object` value on a response envelope.
pub const RESPONSE_OBJECT: &str = "response";
/// Default `max_output_tokens` when the request omits it — matches the
/// chat endpoint's default so the two conversation APIs behave alike.
pub const DEFAULT_MAX_OUTPUT_TOKENS: usize = 256;
/// Id prefixes, per OpenAI's convention.
pub const RESPONSE_ID_PREFIX: &str = "resp_";
pub const MESSAGE_ID_PREFIX: &str = "msg_";
pub const FUNCTION_CALL_ID_PREFIX: &str = "fc_";
pub const CALL_ID_PREFIX: &str = "call_";

/// Lifecycle status strings for the response envelope and output items.
pub const STATUS_COMPLETED: &str = "completed";
pub const STATUS_INCOMPLETE: &str = "incomplete";
pub const STATUS_FAILED: &str = "failed";
pub const STATUS_IN_PROGRESS: &str = "in_progress";

/// `incomplete_details.reason` when generation hit `max_output_tokens`.
pub const INCOMPLETE_MAX_OUTPUT_TOKENS: &str = "max_output_tokens";

/// `input` — either a bare string (single user turn) or a list of
/// input items (conversation turns, function calls, function outputs).
#[derive(Deserialize, Debug)]
#[serde(untagged)]
pub enum ResponseInput {
    Text(String),
    Items(Vec<InputItem>),
}

/// One item in the `input` list. The Responses API discriminates on
/// `type` (defaulting to `"message"` when `role` is present); we keep
/// the fields flat and let [`super::input::items_to_messages`] enforce
/// per-type shape so unknown item types produce a clear 400 instead of
/// a serde soup.
#[derive(Deserialize, Debug)]
pub struct InputItem {
    #[serde(rename = "type", default)]
    pub kind: Option<String>,
    #[serde(default)]
    pub role: Option<String>,
    #[serde(default)]
    pub content: Option<ItemContent>,
    /// `function_call` / `function_call_output`: correlation id.
    #[serde(default)]
    pub call_id: Option<String>,
    /// `function_call`: tool name.
    #[serde(default)]
    pub name: Option<String>,
    /// `function_call`: JSON-stringified arguments.
    #[serde(default)]
    pub arguments: Option<String>,
    /// `function_call_output`: the tool's result.
    #[serde(default)]
    pub output: Option<ItemContent>,
}

/// Message content — a bare string or a list of typed parts.
#[derive(Deserialize, Debug)]
#[serde(untagged)]
pub enum ItemContent {
    Text(String),
    Parts(Vec<ContentPart>),
}

/// One typed content part. Only text parts are supported
/// (`input_text` / `output_text` / `text`); image and audio parts are
/// rejected in the handler.
#[derive(Deserialize, Debug)]
pub struct ContentPart {
    #[serde(rename = "type")]
    pub kind: String,
    #[serde(default)]
    pub text: Option<String>,
}

/// `POST /v1/responses` request body.
#[derive(Deserialize)]
pub struct ResponsesRequest {
    pub model: Option<String>,
    pub input: ResponseInput,
    /// System / developer instructions, prepended as a system turn.
    #[serde(default)]
    pub instructions: Option<String>,
    #[serde(default)]
    pub max_output_tokens: Option<usize>,
    #[serde(default)]
    pub temperature: Option<f32>,
    /// Nucleus filter — only honoured when `temperature > 0`, same as
    /// the chat endpoint.
    #[serde(default)]
    pub top_p: Option<f32>,
    /// Streaming via SSE typed events (`response.created` …
    /// `response.completed`), terminated by `data: [DONE]`.
    #[serde(default)]
    pub stream: Option<bool>,
    /// Persist this response for `previous_response_id` chaining.
    /// Defaults to true (OpenAI's default). Storage is in-memory and
    /// bounded — see [`super::store::ResponseStore`].
    #[serde(default)]
    pub store: Option<bool>,
    /// Continue from a stored response: its conversation (inputs +
    /// output) is replayed ahead of this request's `input`.
    #[serde(default)]
    pub previous_response_id: Option<String>,
    /// Function tools, Responses shape: `[{type:"function", name,
    /// description?, parameters, strict?}]`. Runs through the same
    /// constrained decoder as chat tools.
    #[serde(default)]
    pub tools: Option<serde_json::Value>,
    /// `"auto" | "none" | "required" | {type:"function", name}`.
    #[serde(default)]
    pub tool_choice: Option<serde_json::Value>,
    /// `{format: {type: "text" | "json_object" | "json_schema", ...}}` —
    /// the Responses analog of chat's `response_format`.
    #[serde(default)]
    pub text: Option<serde_json::Value>,
    /// Opaque client metadata, echoed on the response envelope.
    #[serde(default)]
    pub metadata: Option<serde_json::Value>,
    /// Background mode is not supported — 400 when true.
    #[serde(default)]
    pub background: Option<bool>,
    /// Accepted for SDK compatibility; not applied (single-turn
    /// truncation policy is the engine's own).
    #[serde(default)]
    pub truncation: Option<String>,
    /// End-user id — accepted, logged via tracing only.
    #[serde(default)]
    pub user: Option<String>,
    /// Not part of the upstream Responses API, but harmless and useful
    /// against a local engine: extra stop strings.
    #[serde(default)]
    pub stop: Option<StopSpec>,
}

/// One `output[]` item on the response envelope.
#[derive(Serialize, Clone, Debug)]
#[serde(tag = "type")]
pub enum OutputItem {
    #[serde(rename = "message")]
    Message {
        id: String,
        status: String,
        role: &'static str,
        content: Vec<OutputContent>,
    },
    #[serde(rename = "function_call")]
    FunctionCall {
        id: String,
        status: String,
        call_id: String,
        name: String,
        arguments: String,
    },
}

/// One content part inside a message output item.
#[derive(Serialize, Clone, Debug)]
#[serde(tag = "type")]
pub enum OutputContent {
    #[serde(rename = "output_text")]
    OutputText {
        text: String,
        /// Always empty — larql produces no citations/annotations.
        annotations: Vec<serde_json::Value>,
    },
}

/// Token accounting, Responses field names (`input_tokens`, not
/// `prompt_tokens`).
#[derive(Serialize, Clone, Debug)]
pub struct ResponseUsage {
    pub input_tokens: usize,
    /// OpenAI's cached-token detail: how many `input_tokens` were
    /// served from a resumed KV state (N1) instead of re-prefilled.
    pub input_tokens_details: InputTokensDetails,
    pub output_tokens: usize,
    pub total_tokens: usize,
}

/// The `usage.input_tokens_details` object (OpenAI Responses shape).
#[derive(Serialize, Clone, Debug)]
pub struct InputTokensDetails {
    pub cached_tokens: usize,
    /// larql extension, present only when it happened (C4): the chained
    /// response's continuation state refused to resume, and this
    /// response was generated from a fresh prefill instead.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub continuation_refused: Option<ContinuationRefusal>,
}

/// What `usage.input_tokens_details.continuation_refused` records: the
/// refusal's kind and reason, and the recovery the server took.
#[derive(Serialize, Clone, Debug, PartialEq, Eq)]
pub struct ContinuationRefusal {
    /// `provider_absent`, `revision_changed` or `configuration_changed`.
    pub kind: &'static str,
    /// The refusal as the resume contract stated it, naming both sides.
    pub reason: String,
    /// What the server did instead. Always [`FRESH_PREFILL_RECOVERY`].
    pub recovery: &'static str,
}

/// The only recovery the server takes from a refused continuation: a
/// full prefill of the prompt under this model's own continuation.
pub const FRESH_PREFILL_RECOVERY: &str = "fresh_prefill";

/// `incomplete_details` on the envelope when status is `incomplete`.
#[derive(Serialize, Clone, Debug)]
pub struct IncompleteDetails {
    pub reason: &'static str,
}

/// The response envelope returned buffered, embedded in
/// `response.created` / `response.completed` stream events, and served
/// by `GET /v1/responses/{id}`.
#[derive(Serialize, Clone, Debug)]
pub struct ResponseObject {
    pub id: String,
    pub object: &'static str,
    pub created_at: u64,
    pub status: String,
    /// Always null on a served response; failures surface as HTTP
    /// errors or a `response.failed` stream event instead.
    pub error: Option<serde_json::Value>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub incomplete_details: Option<IncompleteDetails>,
    pub model: String,
    pub output: Vec<OutputItem>,
    pub previous_response_id: Option<String>,
    pub instructions: Option<String>,
    pub max_output_tokens: Option<usize>,
    pub temperature: Option<f32>,
    pub top_p: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub metadata: Option<serde_json::Value>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub usage: Option<ResponseUsage>,
}

impl ResponseObject {
    /// Concatenated text of every `output_text` part — the SDKs'
    /// `output_text` convenience, used internally for storage.
    pub fn output_text(&self) -> String {
        let mut out = String::new();
        for item in &self.output {
            if let OutputItem::Message { content, .. } = item {
                for OutputContent::OutputText { text, .. } in content {
                    out.push_str(text);
                }
            }
        }
        out
    }
}
