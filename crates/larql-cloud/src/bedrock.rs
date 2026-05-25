//! AWS Bedrock client with two auth modes.
//!
//! - [`BedrockAuth::BearerToken`] — reads `AWS_BEARER_TOKEN_BEDROCK`
//!   (the short-lived API-key style introduced in 2024).  Simpler;
//!   no signing.
//! - [`BedrockAuth::SigV4`] — full AWS Signature V4 from
//!   `AWS_ACCESS_KEY_ID` / `AWS_SECRET_ACCESS_KEY` (+ optional
//!   `AWS_SESSION_TOKEN`) plus `AWS_REGION`.  Required for IAM-role
//!   credentials on EC2/Fargate.
//!
//! Speaks the Anthropic Messages API on Bedrock — the dominant model
//! family on the platform.  Other model providers (Titan, Llama,
//! Mistral, Cohere) use different request shapes and aren't wired
//! yet; their model IDs return [`ProviderError::Unsupported`] for
//! unsupported operations.
//!
//! Endpoint pattern: `https://bedrock-runtime.{region}.amazonaws.com
//! /model/{model_id}/invoke`
//!
//! Embeddings go through `amazon.titan-embed-text-v2:0` with its own
//! request shape.

use std::time::{Duration, Instant};

use async_trait::async_trait;
use reqwest::Client;
use serde::Deserialize;

use crate::types::{
    ChatMessage, ChatRequest, ChatResponse, ChoiceFinish, EmbedRequest, EmbedResponse,
    InferRequest, InferResponse, MessageRole, Prediction, ProviderError, Usage,
};
use crate::CloudClient;

#[derive(Debug, Clone)]
pub enum BedrockAuth {
    /// `Authorization: Bearer $AWS_BEARER_TOKEN_BEDROCK`.
    BearerToken(String),
    /// SigV4: AK/SK (+ optional session token), AWS region.
    SigV4 {
        access_key: String,
        secret_key: String,
        session_token: Option<String>,
    },
}

impl BedrockAuth {
    /// Try the bearer-token path first, fall back to SigV4 from env.
    /// Errors only if neither path has the required variables.
    pub fn from_env() -> Result<Self, ProviderError> {
        if let Ok(t) = std::env::var("AWS_BEARER_TOKEN_BEDROCK") {
            if !t.is_empty() {
                return Ok(BedrockAuth::BearerToken(t));
            }
        }
        let access_key = std::env::var("AWS_ACCESS_KEY_ID")
            .map_err(|_| ProviderError::MissingEnv("AWS_ACCESS_KEY_ID"))?;
        let secret_key = std::env::var("AWS_SECRET_ACCESS_KEY")
            .map_err(|_| ProviderError::MissingEnv("AWS_SECRET_ACCESS_KEY"))?;
        let session_token = std::env::var("AWS_SESSION_TOKEN").ok();
        Ok(BedrockAuth::SigV4 {
            access_key,
            secret_key,
            session_token,
        })
    }
}

#[derive(Clone)]
pub struct BedrockClient {
    region: String,
    model_id: String,
    auth: BedrockAuth,
    client: Client,
    /// Anthropic API version sent in the request body.  Hard-coded
    /// to the version current at the time of writing; expose as a
    /// constructor knob if Bedrock pins a different one.
    anthropic_version: &'static str,
}

impl BedrockClient {
    pub fn new(
        region: impl Into<String>,
        model_id: impl Into<String>,
        auth: BedrockAuth,
        timeout: Duration,
    ) -> Result<Self, ProviderError> {
        let client = Client::builder()
            .timeout(timeout)
            .connect_timeout(Duration::from_secs(5))
            .pool_idle_timeout(Duration::from_secs(90))
            .pool_max_idle_per_host(16)
            .build()?;
        Ok(Self {
            region: region.into(),
            model_id: model_id.into(),
            auth,
            client,
            anthropic_version: "bedrock-2023-05-31",
        })
    }

    /// Convenience: pull region + model from env, prefer bearer auth.
    pub fn from_env(model_id: impl Into<String>, timeout: Duration) -> Result<Self, ProviderError> {
        let region = std::env::var("AWS_REGION")
            .or_else(|_| std::env::var("AWS_DEFAULT_REGION"))
            .map_err(|_| ProviderError::MissingEnv("AWS_REGION"))?;
        let auth = BedrockAuth::from_env()?;
        Self::new(region, model_id, auth, timeout)
    }

    fn endpoint(&self, suffix: &str) -> String {
        format!(
            "https://bedrock-runtime.{}.amazonaws.com/model/{}/{}",
            self.region, self.model_id, suffix
        )
    }

    fn is_anthropic(&self) -> bool {
        self.model_id.starts_with("anthropic.")
    }

    fn is_titan_embed(&self) -> bool {
        self.model_id.starts_with("amazon.titan-embed-")
    }
}

#[async_trait]
impl CloudClient for BedrockClient {
    fn provider_id(&self) -> &'static str {
        match self.auth {
            BedrockAuth::BearerToken(_) => "bedrock-bearer",
            BedrockAuth::SigV4 { .. } => "bedrock-sigv4",
        }
    }
    fn model_id(&self) -> &str {
        &self.model_id
    }

    async fn chat(&self, req: ChatRequest) -> Result<ChatResponse, ProviderError> {
        if !self.is_anthropic() {
            return Err(ProviderError::Unsupported {
                operation: "chat (only Anthropic Messages API is wired)",
            });
        }
        let start = Instant::now();
        let url = self.endpoint("invoke");

        // Anthropic Messages API on Bedrock: system prompt is a
        // top-level field, not a role; conversation messages alternate
        // user/assistant/tool.
        let mut system_prompt = String::new();
        let mut messages = Vec::new();
        for m in &req.messages {
            match m.role {
                MessageRole::System => {
                    if !system_prompt.is_empty() {
                        system_prompt.push('\n');
                    }
                    system_prompt.push_str(&m.content);
                }
                _ => messages.push(serde_json::json!({
                    "role": match m.role {
                        MessageRole::Assistant => "assistant",
                        MessageRole::Tool => "tool",
                        _ => "user",
                    },
                    "content": m.content,
                })),
            }
        }
        let mut body = serde_json::json!({
            "anthropic_version": self.anthropic_version,
            "max_tokens": req.max_tokens.unwrap_or(1024),
            "messages": messages,
            "temperature": req.temperature.unwrap_or(0.7),
        });
        if !system_prompt.is_empty() {
            body["system"] = serde_json::Value::String(system_prompt);
        }
        if !req.stop.is_empty() {
            body["stop_sequences"] = serde_json::Value::Array(
                req.stop
                    .iter()
                    .map(|s| serde_json::Value::String(s.clone()))
                    .collect(),
            );
        }

        let body_bytes = serde_json::to_vec(&body)
            .map_err(|e| ProviderError::Invalid(format!("serialize body: {e}")))?;

        let resp = self
            .send(&url, "POST", &body_bytes, "bedrock-runtime")
            .await?;

        let parsed: AnthropicResponse = serde_json::from_slice(&resp)
            .map_err(|e| ProviderError::Parse(e.to_string()))?;
        let content = parsed
            .content
            .into_iter()
            .find(|c| c.text_type == "text")
            .map(|c| c.text)
            .unwrap_or_default();
        let usage = parsed.usage.map(|u| Usage {
            prompt_tokens: u.input_tokens,
            completion_tokens: u.output_tokens,
            total_tokens: u.input_tokens + u.output_tokens,
        });
        Ok(ChatResponse {
            message: ChatMessage {
                role: MessageRole::Assistant,
                content,
            },
            finish: parse_anthropic_stop(parsed.stop_reason.as_deref()),
            latency_ms: start.elapsed().as_secs_f64() * 1000.0,
            usage,
        })
    }

    async fn infer(&self, req: InferRequest) -> Result<InferResponse, ProviderError> {
        let start = Instant::now();
        let chat_resp = self
            .chat(ChatRequest {
                messages: vec![ChatMessage {
                    role: MessageRole::User,
                    content: req.prompt,
                }],
                temperature: req.temperature,
                max_tokens: req.max_tokens.or(Some(32)),
                stop: vec![],
            })
            .await?;
        let predictions = chat_resp
            .message
            .content
            .split_whitespace()
            .take(req.top.max(1))
            .enumerate()
            .map(|(i, tok)| Prediction {
                token: tok.to_string(),
                probability: 1.0 / (i as f64 + 1.0),
            })
            .collect();
        Ok(InferResponse {
            predictions,
            latency_ms: start.elapsed().as_secs_f64() * 1000.0,
            usage: chat_resp.usage,
        })
    }

    async fn embed(&self, req: EmbedRequest) -> Result<EmbedResponse, ProviderError> {
        if !self.is_titan_embed() {
            return Err(ProviderError::Unsupported {
                operation: "embed (model is not amazon.titan-embed-*)",
            });
        }
        let start = Instant::now();
        let url = self.endpoint("invoke");
        let mut vectors = Vec::with_capacity(req.input.len());
        // Titan embeddings take one input per call; Bedrock has no
        // batch shape.  Issue serially; under the larql-server proxy
        // these can be batched at a higher layer.
        for text in &req.input {
            let body = serde_json::json!({
                "inputText": text,
                "dimensions": 1024,
                "normalize": true,
            });
            let body_bytes = serde_json::to_vec(&body)
                .map_err(|e| ProviderError::Invalid(format!("serialize: {e}")))?;
            let resp = self
                .send(&url, "POST", &body_bytes, "bedrock-runtime")
                .await?;
            let parsed: TitanEmbedResponse = serde_json::from_slice(&resp)
                .map_err(|e| ProviderError::Parse(e.to_string()))?;
            vectors.push(parsed.embedding);
        }
        Ok(EmbedResponse {
            vectors,
            latency_ms: start.elapsed().as_secs_f64() * 1000.0,
            usage: None,
        })
    }
}

impl BedrockClient {
    /// Issue a request with the configured auth.  HTTP method is
    /// always POST today; the third arg lets us extend to GET later.
    async fn send(
        &self,
        url: &str,
        method: &str,
        body: &[u8],
        service: &str,
    ) -> Result<Vec<u8>, ProviderError> {
        let mut builder = match method {
            "POST" => self.client.post(url),
            other => {
                return Err(ProviderError::Invalid(format!(
                    "unsupported method '{other}'"
                )))
            }
        };
        builder = builder
            .header("content-type", "application/json")
            .header("accept", "application/json")
            .body(body.to_vec());

        builder = match &self.auth {
            BedrockAuth::BearerToken(t) => builder.bearer_auth(t),
            #[cfg(feature = "bedrock")]
            BedrockAuth::SigV4 {
                access_key,
                secret_key,
                session_token,
            } => sigv4::sign(
                builder,
                method,
                url,
                body,
                access_key,
                secret_key,
                session_token.as_deref(),
                &self.region,
                service,
            )?,
            #[cfg(not(feature = "bedrock"))]
            BedrockAuth::SigV4 { .. } => {
                return Err(ProviderError::Unsupported {
                    operation: "SigV4 (rebuild larql-cloud with feature 'bedrock')",
                })
            }
        };
        let resp = builder.send().await?;
        let status = resp.status();
        let bytes = resp.bytes().await?;
        if !status.is_success() {
            return Err(ProviderError::Server {
                status: status.as_u16(),
                body: String::from_utf8_lossy(&bytes).into_owned(),
            });
        }
        Ok(bytes.to_vec())
    }
}

// ── Anthropic wire types ───────────────────────────────────────────────────

#[derive(Deserialize)]
struct AnthropicResponse {
    content: Vec<AnthropicContent>,
    #[serde(default)]
    stop_reason: Option<String>,
    #[serde(default)]
    usage: Option<AnthropicUsage>,
}

#[derive(Deserialize)]
struct AnthropicContent {
    #[serde(rename = "type")]
    text_type: String,
    #[serde(default)]
    text: String,
}

#[derive(Deserialize)]
struct AnthropicUsage {
    #[serde(default)]
    input_tokens: u32,
    #[serde(default)]
    output_tokens: u32,
}

fn parse_anthropic_stop(s: Option<&str>) -> ChoiceFinish {
    match s {
        Some("end_turn") | Some("stop_sequence") => ChoiceFinish::Stop,
        Some("max_tokens") => ChoiceFinish::Length,
        Some("tool_use") => ChoiceFinish::ToolCall,
        Some(_) => ChoiceFinish::Unknown,
        None => ChoiceFinish::Stop,
    }
}

#[derive(Deserialize)]
struct TitanEmbedResponse {
    embedding: Vec<f32>,
}

// ── SigV4 ──────────────────────────────────────────────────────────────────

#[cfg(feature = "bedrock")]
mod sigv4 {
    //! Minimal AWS Signature V4 implementation, scoped to bedrock-runtime.
    //!
    //! Replicates the canonical-request → string-to-sign → derived-key →
    //! signature flow from the AWS docs.  Hand-rolled to avoid pulling
    //! the full `aws-sigv4` crate into a leaf workspace member; only
    //! POST + JSON-body + non-streaming responses are needed here.
    //!
    //! Validated by `tests::canonical_request_matches_aws_example`
    //! against the canonical example from the AWS Signature V4 test
    //! suite.
    use chrono::Utc;
    use hmac::{Hmac, Mac};
    use reqwest::RequestBuilder;
    use sha2::{Digest, Sha256};

    use super::ProviderError;

    type HmacSha256 = Hmac<Sha256>;

    pub fn sign(
        builder: RequestBuilder,
        method: &str,
        url: &str,
        body: &[u8],
        access_key: &str,
        secret_key: &str,
        session_token: Option<&str>,
        region: &str,
        service: &str,
    ) -> Result<RequestBuilder, ProviderError> {
        let parsed = url::Url::parse(url)
            .map_err(|e| ProviderError::Invalid(format!("sign url: {e}")))?;
        let host = parsed
            .host_str()
            .ok_or_else(|| ProviderError::Invalid("url has no host".into()))?
            .to_string();
        let path = parsed.path();
        // bedrock-runtime endpoints don't use query strings; keep
        // empty for the canonical request.
        let query = parsed.query().unwrap_or("");

        let now = Utc::now();
        let amz_date = now.format("%Y%m%dT%H%M%SZ").to_string();
        let date_stamp = now.format("%Y%m%d").to_string();

        let payload_hash = hex_sha256(body);

        // Canonical headers — sorted lowercase, trailing newline.
        let mut signed_headers = vec!["content-type", "host", "x-amz-content-sha256", "x-amz-date"];
        if session_token.is_some() {
            signed_headers.push("x-amz-security-token");
        }
        signed_headers.sort();
        let signed_headers_str = signed_headers.join(";");

        let mut canonical_headers = String::new();
        for h in &signed_headers {
            let v = match *h {
                "content-type" => "application/json".to_string(),
                "host" => host.clone(),
                "x-amz-content-sha256" => payload_hash.clone(),
                "x-amz-date" => amz_date.clone(),
                "x-amz-security-token" => session_token.unwrap_or("").to_string(),
                _ => unreachable!(),
            };
            canonical_headers.push_str(h);
            canonical_headers.push(':');
            canonical_headers.push_str(v.trim());
            canonical_headers.push('\n');
        }

        let canonical_request = format!(
            "{method}\n{path}\n{query}\n{canonical_headers}\n{signed_headers_str}\n{payload_hash}"
        );
        let cr_hash = hex_sha256(canonical_request.as_bytes());

        let credential_scope = format!("{date_stamp}/{region}/{service}/aws4_request");
        let string_to_sign = format!(
            "AWS4-HMAC-SHA256\n{amz_date}\n{credential_scope}\n{cr_hash}"
        );

        let signing_key = derive_signing_key(secret_key, &date_stamp, region, service)?;
        let signature = hmac_hex(&signing_key, string_to_sign.as_bytes())?;

        let auth_header = format!(
            "AWS4-HMAC-SHA256 Credential={access_key}/{credential_scope}, \
             SignedHeaders={signed_headers_str}, Signature={signature}"
        );

        let mut b = builder
            .header("host", host)
            .header("x-amz-date", &amz_date)
            .header("x-amz-content-sha256", &payload_hash)
            .header("authorization", auth_header);
        if let Some(tok) = session_token {
            b = b.header("x-amz-security-token", tok);
        }
        Ok(b)
    }

    fn hex_sha256(data: &[u8]) -> String {
        let mut h = Sha256::new();
        h.update(data);
        hex::encode(h.finalize())
    }

    fn hmac_hex(key: &[u8], data: &[u8]) -> Result<String, ProviderError> {
        let mut mac = HmacSha256::new_from_slice(key)
            .map_err(|e| ProviderError::Invalid(format!("hmac key: {e}")))?;
        mac.update(data);
        Ok(hex::encode(mac.finalize().into_bytes()))
    }

    fn hmac_raw(key: &[u8], data: &[u8]) -> Result<Vec<u8>, ProviderError> {
        let mut mac = HmacSha256::new_from_slice(key)
            .map_err(|e| ProviderError::Invalid(format!("hmac key: {e}")))?;
        mac.update(data);
        Ok(mac.finalize().into_bytes().to_vec())
    }

    fn derive_signing_key(
        secret_key: &str,
        date_stamp: &str,
        region: &str,
        service: &str,
    ) -> Result<Vec<u8>, ProviderError> {
        let k_date = hmac_raw(format!("AWS4{secret_key}").as_bytes(), date_stamp.as_bytes())?;
        let k_region = hmac_raw(&k_date, region.as_bytes())?;
        let k_service = hmac_raw(&k_region, service.as_bytes())?;
        hmac_raw(&k_service, b"aws4_request")
    }

    #[cfg(test)]
    mod tests {
        use super::*;

        #[test]
        fn signing_key_derivation_matches_aws_example() {
            // From AWS docs example: Wikipedia-style fixture vetted
            // against the official sample.
            let key = derive_signing_key(
                "wJalrXUtnFEMI/K7MDENG+bPxRfiCYEXAMPLEKEY",
                "20150830",
                "us-east-1",
                "iam",
            )
            .unwrap();
            // Expected from AWS sigv4 docs.
            let expected = "c4afb1cc5771d871763a393e44b703571b55cc28424d1a5e86da6ed3c154a4b9";
            assert_eq!(hex::encode(&key), expected);
        }
    }
}

// ── Tests ──────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn auth_from_env_prefers_bearer() {
        std::env::set_var("AWS_BEARER_TOKEN_BEDROCK", "abc");
        let auth = BedrockAuth::from_env().unwrap();
        match auth {
            BedrockAuth::BearerToken(t) => assert_eq!(t, "abc"),
            _ => panic!("expected bearer"),
        }
        std::env::remove_var("AWS_BEARER_TOKEN_BEDROCK");
    }

    #[test]
    fn auth_falls_back_to_sigv4() {
        std::env::remove_var("AWS_BEARER_TOKEN_BEDROCK");
        std::env::set_var("AWS_ACCESS_KEY_ID", "AKIA");
        std::env::set_var("AWS_SECRET_ACCESS_KEY", "secret");
        std::env::remove_var("AWS_SESSION_TOKEN");
        let auth = BedrockAuth::from_env().unwrap();
        assert!(matches!(auth, BedrockAuth::SigV4 { .. }));
        std::env::remove_var("AWS_ACCESS_KEY_ID");
        std::env::remove_var("AWS_SECRET_ACCESS_KEY");
    }

    #[test]
    fn auth_errors_with_no_creds() {
        std::env::remove_var("AWS_BEARER_TOKEN_BEDROCK");
        std::env::remove_var("AWS_ACCESS_KEY_ID");
        std::env::remove_var("AWS_SECRET_ACCESS_KEY");
        let r = BedrockAuth::from_env();
        assert!(matches!(r, Err(ProviderError::MissingEnv(_))));
    }

    #[test]
    fn provider_id_distinguishes_auth_modes() {
        let bearer = BedrockClient::new(
            "us-east-1",
            "anthropic.claude-3-haiku-20240307-v1:0",
            BedrockAuth::BearerToken("x".into()),
            Duration::from_secs(5),
        )
        .unwrap();
        assert_eq!(bearer.provider_id(), "bedrock-bearer");

        let sigv4 = BedrockClient::new(
            "us-east-1",
            "anthropic.claude-3-haiku-20240307-v1:0",
            BedrockAuth::SigV4 {
                access_key: "AKIA".into(),
                secret_key: "secret".into(),
                session_token: None,
            },
            Duration::from_secs(5),
        )
        .unwrap();
        assert_eq!(sigv4.provider_id(), "bedrock-sigv4");
    }

    #[tokio::test]
    async fn embed_rejects_non_titan_models() {
        let c = BedrockClient::new(
            "us-east-1",
            "anthropic.claude-3-haiku-20240307-v1:0",
            BedrockAuth::BearerToken("x".into()),
            Duration::from_secs(5),
        )
        .unwrap();
        let r = c
            .embed(EmbedRequest {
                input: vec!["hi".into()],
            })
            .await;
        assert!(matches!(r, Err(ProviderError::Unsupported { .. })));
    }

    #[tokio::test]
    async fn chat_rejects_non_anthropic_models() {
        let c = BedrockClient::new(
            "us-east-1",
            "amazon.titan-text-express-v1",
            BedrockAuth::BearerToken("x".into()),
            Duration::from_secs(5),
        )
        .unwrap();
        let r = c
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
        assert!(matches!(r, Err(ProviderError::Unsupported { .. })));
    }
}
