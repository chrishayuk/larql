//! Outbound LLM clients for larql-server's proxy mode.
//!
//! larql-server already speaks the OpenAI API as an *answerer* (see
//! `crates/larql-server/src/routes/openai/`).  This crate is the
//! mirror: a uniform Rust interface that *calls* OpenAI-compatible
//! services (OpenAI proper, Together AI, Fireworks, Exoscale.ch's AI
//! gateway, vLLM, llama.cpp server, ollama) and AWS Bedrock.
//!
//! The single trait [`CloudClient`] exposes the three operations
//! pg_infer needs to delegate to a cloud LLM:
//!
//! - [`infer`](CloudClient::infer): next-token-style completion.
//! - [`embed`](CloudClient::embed):  text → fixed-dim vector.
//! - [`chat`](CloudClient::chat):   role-tagged messages → reply.
//!
//! `larql-server` accepts a `--proxy <provider>` flag and constructs
//! one of the impls; pg_infer hits `/v1/infer` / `/v1/embeddings` /
//! `/v1/chat/completions` and never knows whether the back end was a
//! local vindex or a remote API.
//!
//! ## Provider matrix
//!
//! | Backend              | Auth                              | Model name shape         |
//! |---------------------|-----------------------------------|--------------------------|
//! | OpenAI              | `OPENAI_API_KEY` Bearer            | `gpt-4o-mini`            |
//! | Exoscale.ch SKS-LLM | `EXOSCALE_API_KEY` Bearer          | `meta-llama-3.1-8b`      |
//! | Together AI         | `TOGETHER_API_KEY` Bearer          | `meta-llama/Llama-3-…`   |
//! | Fireworks           | `FIREWORKS_API_KEY` Bearer         | `accounts/fireworks/…`   |
//! | vLLM / llama.cpp    | none (or proxy)                   | server-defined           |
//! | ollama              | none                              | `qwen2:1.5b`             |
//! | AWS Bedrock (token) | `AWS_BEARER_TOKEN_BEDROCK` Bearer  | `anthropic.claude-3-…`   |
//! | AWS Bedrock (SigV4) | `AWS_ACCESS_KEY_ID` + secret      | `anthropic.claude-3-…`   |
//!
//! All seven OpenAI-compatible variants share a single
//! [`OpenAiCompatible`] impl differing only in `base_url` and
//! optional auth header.  Bedrock has its own impl because the wire
//! format is the Anthropic Messages API (or Titan / Llama variants),
//! and the SigV4 path needs request signing.

use async_trait::async_trait;

pub mod bedrock;
pub mod openai;
pub mod types;

pub use bedrock::{BedrockAuth, BedrockClient};
pub use openai::OpenAiCompatible;
pub use types::{
    ChatMessage, ChatRequest, ChatResponse, ChoiceFinish, EmbedRequest, EmbedResponse,
    InferRequest, InferResponse, MessageRole, ProviderError, Usage,
};

/// Common interface every cloud LLM client implements.
///
/// Methods are async + take `&self` so a single client can fan out
/// across many concurrent requests.  Implementations are expected to
/// honor cancellation by dropping the in-flight future at any await
/// point — see `larql-server` for the cancellation token wiring.
#[async_trait]
pub trait CloudClient: Send + Sync {
    /// Single-shot text completion: prompt → top-K next tokens or a
    /// streaming-eager continuation.  Used by pg_infer's `infer()`.
    async fn infer(&self, req: InferRequest) -> Result<InferResponse, ProviderError>;

    /// Text → vector.  Used by pg_infer's `similar_to_many` when the
    /// configured backend is `--proxy openai-compat` against an
    /// embeddings-capable endpoint.
    async fn embed(&self, req: EmbedRequest) -> Result<EmbedResponse, ProviderError>;

    /// Role-tagged conversation.  Used by larql-server's
    /// `/v1/chat/completions` proxy passthrough.
    async fn chat(&self, req: ChatRequest) -> Result<ChatResponse, ProviderError>;

    /// Provider-stable id (e.g. `"openai"`, `"bedrock-sigv4"`).  Used
    /// in logs and `/v1/stats` responses so operators can see which
    /// proxy is in front of a model.
    fn provider_id(&self) -> &'static str;

    /// Model id passed to the backend.  Distinct from the SQL-side
    /// model name in pg_infer's `infer.models` registry; this is
    /// the *upstream* identifier (`gpt-4o-mini`, `anthropic.claude-3
    /// -haiku-20240307-v1:0`, etc.).
    fn model_id(&self) -> &str;
}
