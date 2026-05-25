//! `larql-cloud-proxy` — serve a CloudClient as an OpenAI-compatible
//! JSON API on a port.
//!
//! Single binary that fronts any provider (OpenAI, Exoscale, Bedrock,
//! Together, ollama, vLLM) behind the same wire format pg_infer's
//! `RemoteBackend` already speaks.  Runs colocated with
//! `larql-server` (different port) or behind nginx on the same host.
//!
//! Endpoints:
//! - `GET  /v1/health`            — liveness
//! - `GET  /v1/stats`             — `{model, mode: "cloud-proxy",
//!                                    provider, ...}` so pg_infer's
//!                                    registration probe succeeds
//! - `POST /v1/infer`             — `{prompt, top}` → predictions
//! - `POST /v1/embeddings`        — `{input: [...]}` → vectors
//! - `POST /v1/chat/completions`  — OpenAI-shaped chat passthrough
//! - `GET  /v1/walk`              — `501` "no vindex on cloud proxy"
//! - `GET  /v1/describe`          — same
//! - `GET  /v1/relations`         — same
//!
//! Auth env vars (per provider) are picked up from the shell or
//! systemd `Environment=` directives.
//!
//! Example:
//! ```sh
//! larql-cloud-proxy --provider bedrock \
//!     --model anthropic.claude-3-haiku-20240307-v1:0 \
//!     --region us-east-1 \
//!     --port 8080
//! ```

use std::net::SocketAddr;
use std::sync::Arc;
use std::time::Duration;

use axum::extract::{Query, State};
use axum::http::StatusCode;
use axum::response::IntoResponse;
use axum::routing::{get, post};
use axum::{Json, Router};
use clap::{Parser, ValueEnum};
use larql_cloud::{
    BedrockAuth, BedrockClient, ChatRequest, ChatResponse, CloudClient, EmbedRequest,
    InferRequest, InferResponse, MessageRole, OpenAiCompatible, ProviderError,
};
use serde::Deserialize;
use tracing::{error, info};

const PROXY_VERSION: &str = env!("CARGO_PKG_VERSION");

#[derive(Parser, Debug)]
#[command(
    name = "larql-cloud-proxy",
    version,
    about = "Serve a CloudClient as an OpenAI-compatible JSON API"
)]
struct Cli {
    /// Provider backend.
    #[arg(long, value_enum)]
    provider: Provider,

    /// Upstream model id (e.g. `gpt-4o-mini`,
    /// `anthropic.claude-3-haiku-20240307-v1:0`).
    #[arg(long)]
    model: String,

    /// Listen port.
    #[arg(long, default_value_t = 8080)]
    port: u16,

    /// Bind address.
    #[arg(long, default_value = "0.0.0.0")]
    host: String,

    /// Per-request timeout (seconds).
    #[arg(long, default_value_t = 60)]
    timeout_secs: u64,

    /// AWS region (Bedrock only).  Default: `AWS_REGION` env var.
    #[arg(long)]
    region: Option<String>,

    /// Override base URL for OpenAI-compatible providers (vLLM,
    /// llama.cpp, ollama, custom Exoscale endpoint).  Required for
    /// `--provider local`; ignored otherwise.
    #[arg(long)]
    base_url: Option<String>,

    /// Optional bearer token for `--provider local`.  Picked up from
    /// `LARQL_PROXY_API_KEY` env if set.
    #[arg(long, env = "LARQL_PROXY_API_KEY")]
    local_api_key: Option<String>,

    /// Log level filter.
    #[arg(long, default_value = "info")]
    log_level: String,
}

#[derive(Clone, Copy, Debug, ValueEnum)]
enum Provider {
    Openai,
    Exoscale,
    Together,
    Bedrock,
    Local,
}

#[derive(Clone)]
struct AppState {
    client: Arc<dyn CloudClient>,
    provider_id: &'static str,
    model_id: String,
}

// ── handlers ────────────────────────────────────────────────────────────────

async fn handle_health() -> Json<serde_json::Value> {
    Json(serde_json::json!({"status": "ok", "version": PROXY_VERSION}))
}

async fn handle_stats(State(state): State<AppState>) -> Json<serde_json::Value> {
    Json(serde_json::json!({
        "model": state.model_id,
        "mode": "cloud-proxy",
        "provider": state.provider_id,
        "version": PROXY_VERSION,
        // Layer/hidden are unknown for a cloud model.  pg_infer reads
        // these into `infer.models` but tolerates zeros for remote rows.
        "layers": 0,
        "hidden_size": 0,
        "vocab_size": 0,
        "extract_level": "cloud",
        "loaded": {
            "browse": false,
            "infer": true,
            "embeddings": true,
            "chat": true,
        },
    }))
}

#[derive(Deserialize)]
struct InferBody {
    prompt: String,
    #[serde(default = "default_top")]
    top: usize,
    #[serde(default)]
    temperature: Option<f32>,
    #[serde(default)]
    max_tokens: Option<u32>,
}
fn default_top() -> usize {
    5
}

async fn handle_infer(
    State(state): State<AppState>,
    Json(body): Json<InferBody>,
) -> Result<Json<InferResponse>, ProxyError> {
    let resp = state
        .client
        .infer(InferRequest {
            prompt: body.prompt,
            top: body.top,
            temperature: body.temperature,
            max_tokens: body.max_tokens,
        })
        .await
        .map_err(ProxyError::from)?;
    Ok(Json(resp))
}

async fn handle_embeddings(
    State(state): State<AppState>,
    Json(body): Json<EmbeddingsBody>,
) -> Result<Json<serde_json::Value>, ProxyError> {
    let resp = state
        .client
        .embed(EmbedRequest {
            input: body.input.into_strings(),
        })
        .await
        .map_err(ProxyError::from)?;
    // Emit OpenAI-compatible response shape so existing OpenAI SDKs
    // (and pg_infer's RemoteBackend) parse it without special cases.
    let data: Vec<serde_json::Value> = resp
        .vectors
        .iter()
        .enumerate()
        .map(|(i, v)| serde_json::json!({"object": "embedding", "embedding": v, "index": i}))
        .collect();
    Ok(Json(serde_json::json!({
        "object": "list",
        "model": state.model_id,
        "data": data,
        "usage": resp.usage,
    })))
}

#[derive(Deserialize)]
struct EmbeddingsBody {
    input: EmbeddingsInput,
    #[serde(default)]
    #[allow(dead_code)]
    model: Option<String>,
    #[serde(default)]
    #[allow(dead_code)]
    encoding_format: Option<String>,
}

#[derive(Deserialize)]
#[serde(untagged)]
enum EmbeddingsInput {
    Single(String),
    Many(Vec<String>),
}

impl EmbeddingsInput {
    fn into_strings(self) -> Vec<String> {
        match self {
            EmbeddingsInput::Single(s) => vec![s],
            EmbeddingsInput::Many(v) => v,
        }
    }
}

#[derive(Deserialize)]
struct ChatBody {
    #[serde(default)]
    messages: Vec<ChatBodyMessage>,
    #[serde(default)]
    temperature: Option<f32>,
    #[serde(default)]
    max_tokens: Option<u32>,
    #[serde(default)]
    stop: Option<StopValue>,
    #[serde(default)]
    #[allow(dead_code)]
    model: Option<String>,
    #[serde(default)]
    #[allow(dead_code)]
    stream: Option<bool>,
}

#[derive(Deserialize)]
struct ChatBodyMessage {
    role: String,
    content: String,
}

#[derive(Deserialize)]
#[serde(untagged)]
enum StopValue {
    One(String),
    Many(Vec<String>),
}

impl StopValue {
    fn into_vec(self) -> Vec<String> {
        match self {
            StopValue::One(s) => vec![s],
            StopValue::Many(v) => v,
        }
    }
}

async fn handle_chat(
    State(state): State<AppState>,
    Json(body): Json<ChatBody>,
) -> Result<Json<serde_json::Value>, ProxyError> {
    let messages: Vec<larql_cloud::ChatMessage> = body
        .messages
        .into_iter()
        .map(|m| larql_cloud::ChatMessage {
            role: parse_role(&m.role),
            content: m.content,
        })
        .collect();
    let resp: ChatResponse = state
        .client
        .chat(ChatRequest {
            messages,
            temperature: body.temperature,
            max_tokens: body.max_tokens,
            stop: body.stop.map(|s| s.into_vec()).unwrap_or_default(),
        })
        .await
        .map_err(ProxyError::from)?;
    // OpenAI-compatible chat response shape.
    Ok(Json(serde_json::json!({
        "id": format!("chatcmpl-{}", uuid_like()),
        "object": "chat.completion",
        "created": chrono::Utc::now().timestamp(),
        "model": state.model_id,
        "choices": [{
            "index": 0,
            "message": {
                "role": role_str(resp.message.role),
                "content": resp.message.content,
            },
            "finish_reason": finish_str(resp.finish),
        }],
        "usage": resp.usage,
    })))
}

async fn handle_unsupported() -> impl IntoResponse {
    (
        StatusCode::NOT_IMPLEMENTED,
        Json(serde_json::json!({
            "error": "vindex-only endpoint not available on cloud-proxy",
        })),
    )
}

#[derive(Deserialize)]
struct WalkParams {
    #[serde(default)]
    #[allow(dead_code)]
    prompt: Option<String>,
}

async fn handle_walk(
    Query(_q): Query<WalkParams>,
) -> impl IntoResponse {
    handle_unsupported().await
}

// ── error mapping ───────────────────────────────────────────────────────────

#[derive(Debug)]
struct ProxyError(ProviderError);

impl From<ProviderError> for ProxyError {
    fn from(e: ProviderError) -> Self {
        Self(e)
    }
}

impl IntoResponse for ProxyError {
    fn into_response(self) -> axum::response::Response {
        let (status, body) = match &self.0 {
            ProviderError::Auth(_) | ProviderError::MissingEnv(_) => (
                StatusCode::UNAUTHORIZED,
                serde_json::json!({"error": self.0.to_string()}),
            ),
            ProviderError::Server { status, body } => (
                StatusCode::from_u16(*status).unwrap_or(StatusCode::BAD_GATEWAY),
                serde_json::json!({"error": "upstream returned non-2xx", "body": body}),
            ),
            ProviderError::Unsupported { operation } => (
                StatusCode::NOT_IMPLEMENTED,
                serde_json::json!({"error": format!("provider does not support {operation}")}),
            ),
            ProviderError::Invalid(msg) => (
                StatusCode::BAD_REQUEST,
                serde_json::json!({"error": msg}),
            ),
            _ => (
                StatusCode::BAD_GATEWAY,
                serde_json::json!({"error": self.0.to_string()}),
            ),
        };
        (status, Json(body)).into_response()
    }
}

// ── helpers ─────────────────────────────────────────────────────────────────

fn parse_role(s: &str) -> MessageRole {
    match s.to_lowercase().as_str() {
        "system" => MessageRole::System,
        "user" => MessageRole::User,
        "tool" => MessageRole::Tool,
        _ => MessageRole::Assistant,
    }
}

fn role_str(r: MessageRole) -> &'static str {
    match r {
        MessageRole::System => "system",
        MessageRole::User => "user",
        MessageRole::Assistant => "assistant",
        MessageRole::Tool => "tool",
    }
}

fn finish_str(f: larql_cloud::ChoiceFinish) -> &'static str {
    use larql_cloud::ChoiceFinish::*;
    match f {
        Stop => "stop",
        Length => "length",
        ToolCall => "tool_calls",
        ContentFilter => "content_filter",
        Unknown => "stop",
    }
}

/// Tiny non-cryptographic id for `chatcmpl-…`.  We don't pull `uuid`
/// for one cosmetic field.
fn uuid_like() -> String {
    let nanos = chrono::Utc::now().timestamp_nanos_opt().unwrap_or_default();
    let mix = (nanos as u128).wrapping_mul(0x9e3779b97f4a7c15);
    format!("{:016x}", mix as u64)
}

// ── main ────────────────────────────────────────────────────────────────────

fn build_client(cli: &Cli) -> Result<Arc<dyn CloudClient>, Box<dyn std::error::Error>> {
    let timeout = Duration::from_secs(cli.timeout_secs);
    let client: Arc<dyn CloudClient> = match cli.provider {
        Provider::Openai => Arc::new(OpenAiCompatible::openai(&cli.model, timeout)?),
        Provider::Exoscale => Arc::new(OpenAiCompatible::exoscale(&cli.model, timeout)?),
        Provider::Together => Arc::new(OpenAiCompatible::together(&cli.model, timeout)?),
        Provider::Bedrock => {
            let region = cli
                .region
                .clone()
                .or_else(|| std::env::var("AWS_REGION").ok())
                .or_else(|| std::env::var("AWS_DEFAULT_REGION").ok())
                .ok_or("AWS region required: pass --region or set AWS_REGION")?;
            let auth = BedrockAuth::from_env()?;
            Arc::new(BedrockClient::new(region, &cli.model, auth, timeout)?)
        }
        Provider::Local => {
            let base = cli
                .base_url
                .clone()
                .ok_or("--provider local requires --base-url")?;
            Arc::new(OpenAiCompatible::local(
                base,
                &cli.model,
                cli.local_api_key.clone(),
                timeout,
            )?)
        }
    };
    Ok(client)
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let cli = Cli::parse();

    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| tracing_subscriber::EnvFilter::new(&cli.log_level)),
        )
        .init();

    let client = build_client(&cli).map_err(|e| {
        error!("failed to build cloud client: {e}");
        e
    })?;
    let provider_id = client.provider_id();
    info!(
        "larql-cloud-proxy v{PROXY_VERSION} provider={} model={} listen=http://{}:{}",
        provider_id, cli.model, cli.host, cli.port
    );

    let state = AppState {
        client,
        provider_id,
        model_id: cli.model.clone(),
    };

    let app = Router::new()
        .route("/v1/health", get(handle_health))
        .route("/v1/stats", get(handle_stats))
        .route("/v1/infer", post(handle_infer))
        .route("/v1/embeddings", post(handle_embeddings))
        .route("/v1/chat/completions", post(handle_chat))
        .route("/v1/walk", get(handle_walk))
        .route("/v1/describe", get(handle_unsupported))
        .route("/v1/relations", get(handle_unsupported))
        .with_state(state);

    let addr: SocketAddr = format!("{}:{}", cli.host, cli.port).parse()?;
    let listener = tokio::net::TcpListener::bind(&addr).await?;

    let shutdown = async {
        let _ = tokio::signal::ctrl_c().await;
        info!("shutdown signal received");
    };

    axum::serve(listener, app)
        .with_graceful_shutdown(shutdown)
        .await?;
    Ok(())
}

