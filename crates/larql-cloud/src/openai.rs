//! OpenAI-compatible client.
//!
//! Drives any backend that speaks the OpenAI Chat / Completions /
//! Embeddings JSON API: OpenAI proper, Together AI, Fireworks,
//! Exoscale.ch's AI gateway, vLLM, llama.cpp server, ollama.
//! Auth is a single optional `Authorization: Bearer …` header.

use std::time::{Duration, Instant};

use async_trait::async_trait;
use reqwest::Client;
use serde::{Deserialize, Serialize};

use crate::types::{
    ChatMessage, ChatRequest, ChatResponse, ChoiceFinish, EmbedRequest, EmbedResponse,
    InferRequest, InferResponse, MessageRole, Prediction, ProviderError, Usage,
};
use crate::CloudClient;

/// One client per provider/model pair.  Cheap to clone (`reqwest::Client`
/// is internally `Arc`'d).
#[derive(Clone)]
pub struct OpenAiCompatible {
    /// Base URL like `https://api.openai.com` or
    /// `https://ai.exoscale.com` or `http://localhost:11434`.  No
    /// trailing slash, no `/v1`.
    base_url: String,
    /// Upstream model id passed in every request body.
    model: String,
    /// Bearer token, if the provider requires auth.
    api_key: Option<String>,
    /// Stable identifier surfaced in `/v1/stats`.
    provider_id: &'static str,
    client: Client,
}

impl OpenAiCompatible {
    /// Build a client.  Common wiring; see provider-specific
    /// constructors below for typical defaults.
    pub fn new(
        base_url: impl Into<String>,
        model: impl Into<String>,
        api_key: Option<String>,
        provider_id: &'static str,
        timeout: Duration,
    ) -> Result<Self, ProviderError> {
        let base_url = base_url.into().trim_end_matches('/').to_string();
        url::Url::parse(&base_url)
            .map_err(|e| ProviderError::Invalid(format!("base_url '{base_url}': {e}")))?;
        let client = Client::builder()
            .timeout(timeout)
            .connect_timeout(Duration::from_secs(5))
            .pool_idle_timeout(Duration::from_secs(90))
            .pool_max_idle_per_host(16)
            .build()?;
        Ok(Self {
            base_url,
            model: model.into(),
            api_key,
            provider_id,
            client,
        })
    }

    /// OpenAI proper.  Reads `OPENAI_API_KEY` from env.
    pub fn openai(model: impl Into<String>, timeout: Duration) -> Result<Self, ProviderError> {
        let key = std::env::var("OPENAI_API_KEY")
            .map_err(|_| ProviderError::MissingEnv("OPENAI_API_KEY"))?;
        Self::new("https://api.openai.com", model, Some(key), "openai", timeout)
    }

    /// Exoscale.ch's AI gateway.  Reads `EXOSCALE_API_KEY` from env.
    /// Endpoint pattern matches their public docs (OpenAI-compatible).
    pub fn exoscale(model: impl Into<String>, timeout: Duration) -> Result<Self, ProviderError> {
        let key = std::env::var("EXOSCALE_API_KEY")
            .map_err(|_| ProviderError::MissingEnv("EXOSCALE_API_KEY"))?;
        let endpoint = std::env::var("EXOSCALE_AI_BASE")
            .unwrap_or_else(|_| "https://ai.exoscale.com".to_string());
        Self::new(endpoint, model, Some(key), "exoscale", timeout)
    }

    /// Together AI.  Reads `TOGETHER_API_KEY`.
    pub fn together(model: impl Into<String>, timeout: Duration) -> Result<Self, ProviderError> {
        let key = std::env::var("TOGETHER_API_KEY")
            .map_err(|_| ProviderError::MissingEnv("TOGETHER_API_KEY"))?;
        Self::new("https://api.together.xyz", model, Some(key), "together", timeout)
    }

    /// Local-or-LAN OpenAI-compatible server (vLLM, llama.cpp,
    /// ollama, etc.).  No auth by default; pass `api_key` if the
    /// reverse proxy in front needs it.
    pub fn local(
        base_url: impl Into<String>,
        model: impl Into<String>,
        api_key: Option<String>,
        timeout: Duration,
    ) -> Result<Self, ProviderError> {
        Self::new(base_url, model, api_key, "openai-compat", timeout)
    }

    fn auth(&self, b: reqwest::RequestBuilder) -> reqwest::RequestBuilder {
        match &self.api_key {
            Some(k) => b.bearer_auth(k),
            None => b,
        }
    }
}

#[async_trait]
impl CloudClient for OpenAiCompatible {
    fn provider_id(&self) -> &'static str {
        self.provider_id
    }
    fn model_id(&self) -> &str {
        &self.model
    }

    async fn chat(&self, req: ChatRequest) -> Result<ChatResponse, ProviderError> {
        let start = Instant::now();
        let url = format!("{}/v1/chat/completions", self.base_url);
        let body = serde_json::json!({
            "model": self.model,
            "messages": req
                .messages
                .iter()
                .map(|m| serde_json::json!({"role": role_str(m.role), "content": m.content}))
                .collect::<Vec<_>>(),
            "temperature": req.temperature,
            "max_tokens": req.max_tokens,
            "stop": req.stop,
            "stream": false,
        });
        let resp = self.auth(self.client.post(&url)).json(&body).send().await?;
        let status = resp.status();
        let bytes = resp.bytes().await?;
        if !status.is_success() {
            return Err(ProviderError::Server {
                status: status.as_u16(),
                body: String::from_utf8_lossy(&bytes).into_owned(),
            });
        }
        let parsed: OpenAiChatResponse = serde_json::from_slice(&bytes)
            .map_err(|e| ProviderError::Parse(e.to_string()))?;
        let choice = parsed
            .choices
            .into_iter()
            .next()
            .ok_or_else(|| ProviderError::Parse("no choices in response".into()))?;
        Ok(ChatResponse {
            message: ChatMessage {
                role: parse_role(&choice.message.role),
                content: choice.message.content,
            },
            finish: parse_finish(choice.finish_reason.as_deref().unwrap_or("stop")),
            latency_ms: start.elapsed().as_secs_f64() * 1000.0,
            usage: parsed.usage.map(Usage::from),
        })
    }

    async fn infer(&self, req: InferRequest) -> Result<InferResponse, ProviderError> {
        // Most OpenAI-compatible servers no longer expose the legacy
        // /v1/completions endpoint.  Use chat with a single user
        // message and synthesise top-K predictions from the resulting
        // text by emitting the first `top` whitespace-separated
        // tokens.  This is intentionally crude — pg_infer's `infer()`
        // is for next-token-style hints, not real generation; if you
        // want generation, use `chat()`.
        let start = Instant::now();
        let chat_req = ChatRequest {
            messages: vec![ChatMessage {
                role: MessageRole::User,
                content: req.prompt,
            }],
            temperature: req.temperature,
            max_tokens: req.max_tokens.or(Some(32)),
            stop: vec![],
        };
        let chat = self.chat(chat_req).await?;
        let tokens: Vec<&str> = chat
            .message
            .content
            .split_whitespace()
            .take(req.top.max(1))
            .collect();
        let predictions = tokens
            .into_iter()
            .enumerate()
            .map(|(i, tok)| Prediction {
                token: tok.to_string(),
                probability: 1.0 / (i as f64 + 1.0),
            })
            .collect();
        Ok(InferResponse {
            predictions,
            latency_ms: start.elapsed().as_secs_f64() * 1000.0,
            usage: chat.usage,
        })
    }

    async fn embed(&self, req: EmbedRequest) -> Result<EmbedResponse, ProviderError> {
        let start = Instant::now();
        let url = format!("{}/v1/embeddings", self.base_url);
        let body = serde_json::json!({
            "model": self.model,
            "input": req.input,
            "encoding_format": "float",
        });
        let resp = self.auth(self.client.post(&url)).json(&body).send().await?;
        let status = resp.status();
        let bytes = resp.bytes().await?;
        if !status.is_success() {
            return Err(ProviderError::Server {
                status: status.as_u16(),
                body: String::from_utf8_lossy(&bytes).into_owned(),
            });
        }
        let parsed: OpenAiEmbedResponse = serde_json::from_slice(&bytes)
            .map_err(|e| ProviderError::Parse(e.to_string()))?;
        let vectors = parsed.data.into_iter().map(|d| d.embedding).collect();
        Ok(EmbedResponse {
            vectors,
            latency_ms: start.elapsed().as_secs_f64() * 1000.0,
            usage: parsed.usage.map(Usage::from),
        })
    }
}

// ── helpers ─────────────────────────────────────────────────────────────────

fn role_str(r: MessageRole) -> &'static str {
    match r {
        MessageRole::System => "system",
        MessageRole::User => "user",
        MessageRole::Assistant => "assistant",
        MessageRole::Tool => "tool",
    }
}

fn parse_role(s: &str) -> MessageRole {
    match s {
        "system" => MessageRole::System,
        "user" => MessageRole::User,
        "tool" => MessageRole::Tool,
        _ => MessageRole::Assistant,
    }
}

fn parse_finish(s: &str) -> ChoiceFinish {
    match s {
        "stop" => ChoiceFinish::Stop,
        "length" => ChoiceFinish::Length,
        "tool_calls" | "function_call" => ChoiceFinish::ToolCall,
        "content_filter" => ChoiceFinish::ContentFilter,
        _ => ChoiceFinish::Unknown,
    }
}

// ── wire types (we only deserialize the fields we need) ────────────────────

#[derive(Deserialize)]
struct OpenAiChatResponse {
    choices: Vec<OpenAiChoice>,
    #[serde(default)]
    usage: Option<OpenAiUsage>,
}

#[derive(Deserialize)]
struct OpenAiChoice {
    message: OpenAiMessage,
    #[serde(default)]
    finish_reason: Option<String>,
}

#[derive(Deserialize)]
struct OpenAiMessage {
    #[serde(default = "default_role")]
    role: String,
    #[serde(default)]
    content: String,
}

fn default_role() -> String {
    "assistant".to_string()
}

#[derive(Deserialize)]
struct OpenAiEmbedResponse {
    data: Vec<OpenAiEmbedData>,
    #[serde(default)]
    usage: Option<OpenAiUsage>,
}

#[derive(Deserialize)]
struct OpenAiEmbedData {
    embedding: Vec<f32>,
}

#[derive(Deserialize, Serialize, Clone, Copy)]
struct OpenAiUsage {
    #[serde(default)]
    prompt_tokens: u32,
    #[serde(default)]
    completion_tokens: u32,
    #[serde(default)]
    total_tokens: u32,
}

impl From<OpenAiUsage> for Usage {
    fn from(u: OpenAiUsage) -> Self {
        Usage {
            prompt_tokens: u.prompt_tokens,
            completion_tokens: u.completion_tokens,
            total_tokens: u.total_tokens,
        }
    }
}

// ── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::{Arc, Mutex};
    use std::time::Duration;

    use bytes::Bytes;
    use http_body_util::Full;
    use hyper::body::Incoming;
    use hyper::{Request, Response};
    use hyper_util::rt::TokioIo;

    /// Stand up a deterministic mock OpenAI server on an ephemeral
    /// TCP port.  Records every request URI for assertion.
    async fn spawn_mock() -> (String, Arc<Mutex<Vec<String>>>) {
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();
        let log: Arc<Mutex<Vec<String>>> = Arc::new(Mutex::new(Vec::new()));
        let log_bg = log.clone();
        tokio::spawn(async move {
            let deadline = tokio::time::Instant::now() + Duration::from_secs(10);
            loop {
                tokio::select! {
                    _ = tokio::time::sleep_until(deadline) => return,
                    res = listener.accept() => {
                        let (stream, _) = match res { Ok(v) => v, Err(_) => continue };
                        let log_conn = log_bg.clone();
                        tokio::spawn(async move {
                            let io = TokioIo::new(stream);
                            let svc = hyper::service::service_fn(move |req: Request<Incoming>| {
                                let log = log_conn.clone();
                                async move {
                                    let path = req.uri().path().to_string();
                                    log.lock().expect("log").push(path.clone());
                                    let body = match path.as_str() {
                                        "/v1/chat/completions" => serde_json::json!({
                                            "id": "chat-1",
                                            "object": "chat.completion",
                                            "model": "test-model",
                                            "choices": [{
                                                "index": 0,
                                                "message": {"role": "assistant", "content": "hello world from mock"},
                                                "finish_reason": "stop"
                                            }],
                                            "usage": {"prompt_tokens": 5, "completion_tokens": 4, "total_tokens": 9}
                                        }),
                                        "/v1/embeddings" => serde_json::json!({
                                            "object": "list",
                                            "data": [
                                                {"object": "embedding", "embedding": [0.1, 0.2, 0.3], "index": 0},
                                                {"object": "embedding", "embedding": [0.4, 0.5, 0.6], "index": 1}
                                            ],
                                            "model": "test-model",
                                            "usage": {"prompt_tokens": 2, "completion_tokens": 0, "total_tokens": 2}
                                        }),
                                        _ => serde_json::json!({"error": "not found"}),
                                    };
                                    let bytes = serde_json::to_vec(&body).expect("vec");
                                    Ok::<Response<Full<Bytes>>, std::io::Error>(
                                        Response::builder()
                                            .status(if path.starts_with("/v1/") { 200 } else { 404 })
                                            .header("content-type", "application/json")
                                            .body(Full::new(Bytes::from(bytes)))
                                            .expect("resp"))
                                }
                            });
                            let _ = hyper::server::conn::http1::Builder::new()
                                .serve_connection(io, svc).await;
                        });
                    }
                }
            }
        });
        (format!("http://{addr}"), log)
    }

    #[tokio::test]
    async fn chat_round_trip() {
        let (url, log) = spawn_mock().await;
        let c = OpenAiCompatible::local(url, "test-model", None, Duration::from_secs(5))
            .expect("client");
        let resp = c
            .chat(ChatRequest {
                messages: vec![ChatMessage {
                    role: MessageRole::User,
                    content: "hi".into(),
                }],
                temperature: Some(0.7),
                max_tokens: Some(64),
                stop: vec![],
            })
            .await
            .expect("chat");
        assert_eq!(resp.message.content, "hello world from mock");
        assert_eq!(resp.finish, ChoiceFinish::Stop);
        assert_eq!(resp.usage.unwrap().total_tokens, 9);
        assert_eq!(log.lock().unwrap()[0], "/v1/chat/completions");
    }

    #[tokio::test]
    async fn infer_synthesises_from_chat() {
        let (url, _log) = spawn_mock().await;
        let c = OpenAiCompatible::local(url, "test-model", None, Duration::from_secs(5))
            .expect("client");
        let resp = c
            .infer(InferRequest {
                prompt: "the capital of France is".into(),
                top: 3,
                temperature: None,
                max_tokens: None,
            })
            .await
            .expect("infer");
        // Mock returns "hello world from mock"; first 3 whitespace-tokens.
        assert_eq!(resp.predictions.len(), 3);
        assert_eq!(resp.predictions[0].token, "hello");
        assert_eq!(resp.predictions[1].token, "world");
        assert_eq!(resp.predictions[2].token, "from");
        assert!(resp.predictions[0].probability > resp.predictions[1].probability);
    }

    #[tokio::test]
    async fn embed_round_trip() {
        let (url, _log) = spawn_mock().await;
        let c = OpenAiCompatible::local(url, "test-model", None, Duration::from_secs(5))
            .expect("client");
        let resp = c
            .embed(EmbedRequest {
                input: vec!["a".into(), "b".into()],
            })
            .await
            .expect("embed");
        assert_eq!(resp.vectors.len(), 2);
        assert_eq!(resp.vectors[0], vec![0.1, 0.2, 0.3]);
        assert_eq!(resp.vectors[1].len(), 3);
    }

    #[tokio::test]
    async fn server_error_propagates() {
        // Hit a non-routable port to force a transport error.
        let bad = OpenAiCompatible::local(
            "http://127.0.0.1:1",
            "x",
            None,
            Duration::from_millis(200),
        )
        .expect("client");
        let err = bad
            .chat(ChatRequest {
                messages: vec![ChatMessage {
                    role: MessageRole::User,
                    content: "hi".into(),
                }],
                temperature: None,
                max_tokens: None,
                stop: vec![],
            })
            .await;
        assert!(matches!(err, Err(ProviderError::Transport(_))), "got {err:?}");
    }
}
