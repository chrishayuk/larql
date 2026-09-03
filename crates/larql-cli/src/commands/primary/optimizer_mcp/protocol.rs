//! **JSON-RPC 2.0 over stdio**, which is all MCP's stdio transport is.
//!
//! Hand-written rather than taken from a client library, because the
//! subset a read-only server needs is four methods and an error code,
//! and a dependency that brings an async runtime to read lines off
//! stdin would be the larger thing to justify.

use serde::{Deserialize, Serialize};

/// The only JSON-RPC version this server speaks.
pub const JSONRPC_VERSION: &str = "2.0";

/// The MCP revision this server implements.
pub const PROTOCOL_VERSION: &str = "2025-06-18";

pub const METHOD_INITIALIZE: &str = "initialize";
pub const METHOD_TOOLS_LIST: &str = "tools/list";
pub const METHOD_TOOLS_CALL: &str = "tools/call";
pub const METHOD_PING: &str = "ping";

// JSON-RPC 2.0's own error codes. Named, because `-32601` in a match
// arm is the kind of literal that gets mistyped once and then travels.
pub const ERROR_PARSE: i64 = -32700;
pub const ERROR_INVALID_REQUEST: i64 = -32600;
pub const ERROR_METHOD_NOT_FOUND: i64 = -32601;
pub const ERROR_INVALID_PARAMS: i64 = -32602;

/// One incoming call. A request with no `id` is a notification and is
/// answered with silence, which is the protocol's rule and not an
/// omission.
#[derive(Debug, Clone, Deserialize)]
pub struct Request {
    pub jsonrpc: String,
    #[serde(default)]
    pub id: Option<serde_json::Value>,
    pub method: String,
    #[serde(default)]
    pub params: serde_json::Value,
}

impl Request {
    pub fn is_notification(&self) -> bool {
        self.id.is_none()
    }
}

/// One outgoing answer: a result or an error, never both.
#[derive(Debug, Clone, Serialize)]
pub struct Response {
    pub jsonrpc: &'static str,
    pub id: serde_json::Value,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub result: Option<serde_json::Value>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub error: Option<ResponseError>,
}

#[derive(Debug, Clone, Serialize)]
pub struct ResponseError {
    pub code: i64,
    pub message: String,
}

impl Response {
    pub fn result(id: serde_json::Value, result: serde_json::Value) -> Self {
        Self {
            jsonrpc: JSONRPC_VERSION,
            id,
            result: Some(result),
            error: None,
        }
    }

    pub fn error(id: serde_json::Value, code: i64, message: impl Into<String>) -> Self {
        Self {
            jsonrpc: JSONRPC_VERSION,
            id,
            result: None,
            error: Some(ResponseError {
                code,
                message: message.into(),
            }),
        }
    }
}

/// A tool's answer. `is_error` marks a call the server understood and
/// could not satisfy — an id the record does not hold — as distinct
/// from a protocol failure, which is a [`Response::error`].
#[derive(Debug, Clone, Serialize)]
pub struct ToolResult {
    pub content: Vec<Content>,
    #[serde(rename = "isError")]
    pub is_error: bool,
}

#[derive(Debug, Clone, Serialize)]
#[serde(tag = "type")]
pub enum Content {
    #[serde(rename = "text")]
    Text { text: String },
}

impl ToolResult {
    /// A view, rendered. The text is the view's own serialisation and
    /// nothing is added to it.
    pub fn of(value: &impl Serialize, is_error: bool) -> Result<Self, serde_json::Error> {
        Ok(Self {
            content: vec![Content::Text {
                text: serde_json::to_string_pretty(value)?,
            }],
            is_error,
        })
    }
}
