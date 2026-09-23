//! Provider-agnostic request/response types.
//!
//! These mirror the parts of the OpenAI API pg_infer cares about — a
//! deliberately small subset.  Bedrock impls translate to/from these
//! at the boundary so pg_infer never sees Anthropic's Messages API
//! shape directly.

use serde::{Deserialize, Serialize};

#[derive(Debug, thiserror::Error)]
pub enum ProviderError {
    #[error("HTTP transport error: {0}")]
    Transport(String),

    #[error("provider returned HTTP {status}: {body}")]
    Server { status: u16, body: String },

    #[error("authentication failed: {0}")]
    Auth(String),

    #[error("missing required environment variable: {0}")]
    MissingEnv(&'static str),

    #[error("failed to parse provider response: {0}")]
    Parse(String),

    #[error("provider does not support {operation}")]
    Unsupported { operation: &'static str },

    #[error("invalid argument: {0}")]
    Invalid(String),
}

impl From<reqwest::Error> for ProviderError {
    fn from(e: reqwest::Error) -> Self {
        ProviderError::Transport(e.to_string())
    }
}

// ── Infer (next-token / completion) ─────────────────────────────────────────

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InferRequest {
    pub prompt: String,
    /// Number of top tokens to return.  pg_infer typically asks for 5.
    pub top: usize,
    /// Sampling temperature; `None` means provider default.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub temperature: Option<f32>,
    /// Maximum output tokens.  `None` means provider default.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub max_tokens: Option<u32>,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct InferResponse {
    pub predictions: Vec<Prediction>,
    #[serde(default)]
    pub latency_ms: f64,
    #[serde(default)]
    pub usage: Option<Usage>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Prediction {
    pub token: String,
    /// 0..=1 confidence — for completion-style providers we synthesise
    /// this from the relative position (rank → 1/(rank+1)) since few
    /// non-OpenAI APIs return logprobs.
    pub probability: f64,
}

// ── Embed ──────────────────────────────────────────────────────────────────

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EmbedRequest {
    pub input: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct EmbedResponse {
    pub vectors: Vec<Vec<f32>>,
    #[serde(default)]
    pub latency_ms: f64,
    #[serde(default)]
    pub usage: Option<Usage>,
}

// ── Chat ───────────────────────────────────────────────────────────────────

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "lowercase")]
pub enum MessageRole {
    System,
    User,
    Assistant,
    Tool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChatMessage {
    pub role: MessageRole,
    pub content: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChatRequest {
    pub messages: Vec<ChatMessage>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub temperature: Option<f32>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub max_tokens: Option<u32>,
    /// Stop strings; ignored by providers that don't support them.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub stop: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct ChatResponse {
    pub message: ChatMessage,
    pub finish: ChoiceFinish,
    #[serde(default)]
    pub latency_ms: f64,
    #[serde(default)]
    pub usage: Option<Usage>,
}

impl Default for ChatMessage {
    fn default() -> Self {
        Self {
            role: MessageRole::Assistant,
            content: String::new(),
        }
    }
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq, Default)]
#[serde(rename_all = "snake_case")]
pub enum ChoiceFinish {
    #[default]
    Stop,
    Length,
    ToolCall,
    ContentFilter,
    Unknown,
}

// ── Usage ──────────────────────────────────────────────────────────────────

#[derive(Debug, Clone, Copy, Serialize, Deserialize, Default)]
pub struct Usage {
    pub prompt_tokens: u32,
    pub completion_tokens: u32,
    pub total_tokens: u32,
}
