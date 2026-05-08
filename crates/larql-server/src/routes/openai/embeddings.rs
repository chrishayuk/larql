//! `POST /v1/embeddings` — OpenAI-compatible embeddings endpoint (N0.4).
//!
//! Implements the [OpenAI Embeddings API](https://platform.openai.com/docs/api-reference/embeddings/create)
//! shape so existing `openai` SDKs work unmodified:
//!
//! ```python
//! from openai import OpenAI
//! client = OpenAI(base_url="http://larql:8080/v1", api_key="sk-...")
//! resp = client.embeddings.create(model="gemma-3-4b", input="hello world")
//! ```
//!
//! ## Pooling semantics
//!
//! OpenAI's text-embedding models output one pooled vector per input.
//! This endpoint emulates that by **mean-pooling** the per-token static
//! embeddings (`embeddings.bin` row lookup) over the input sequence.
//! Static embeddings are not the same as a contrastively-trained sentence
//! encoder — clients should treat results as "lookup-pooled" rather than
//! "semantic" embeddings until a dedicated embedding head is added.
//!
//! For per-token embeddings (no pooling), use the native `/v1/embed`
//! endpoint instead.
//!
//! ## Input variants supported
//!
//! - `string` — one input
//! - `string[]` — batched inputs
//! - `int[]` — one pre-tokenised input
//! - `int[][]` — batched pre-tokenised inputs
//!
//! ## Encoding format
//!
//! - `encoding_format: "float"` (default) — JSON array of f32.
//! - `encoding_format: "base64"` — base64-encoded little-endian f32
//!   bytes (~33% smaller wire than the JSON array form). Many
//!   production OpenAI clients default to base64 for embeddings.

use std::sync::Arc;

use axum::extract::State;
use axum::Json;
use base64::Engine;
use serde::{Deserialize, Serialize};

use crate::error::ServerError;
use crate::state::{AppState, LoadedModel};

use crate::routes::embed::embed_tokens;

const EMBEDDING_OBJECT: &str = "embedding";
const LIST_OBJECT: &str = "list";

/// Choice between the OpenAI `"float"` (default) and `"base64"` wire
/// formats. `Float` produces `embedding: [f32, ...]`; `Base64` produces
/// `embedding: "<base64 of LE f32 bytes>"`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum EncodingFormat {
    Float,
    Base64,
}

/// Per-request `embedding` field — `Vec<f32>` for float mode, `String`
/// for base64. Untagged so serde picks a single shape per object based
/// on which variant was constructed.
#[derive(Serialize)]
#[serde(untagged)]
pub enum EmbeddingValue {
    Floats(Vec<f32>),
    Base64(String),
}

#[derive(Deserialize)]
#[serde(untagged)]
pub enum EmbeddingInput {
    Single(String),
    Batch(Vec<String>),
    SingleTokens(Vec<u32>),
    BatchTokens(Vec<Vec<u32>>),
}

#[derive(Deserialize)]
pub struct EmbeddingsRequest {
    /// Model id (matched against the loaded model's id; ignored in
    /// single-model mode).
    pub model: Option<String>,
    pub input: EmbeddingInput,
    /// `"float"` (default) or `"base64"`. Anything else returns 400.
    #[serde(default)]
    pub encoding_format: Option<String>,
    /// Optional caller-supplied dimensionality. Larql ignores this — the
    /// vector size is the model's `hidden_size`. Logged but not enforced.
    #[serde(default)]
    pub dimensions: Option<usize>,
    /// Optional end-user id (OpenAI tracks this for abuse monitoring;
    /// larql logs it via tracing if set, otherwise no-op).
    #[serde(default)]
    pub user: Option<String>,
}

#[derive(Serialize)]
pub struct EmbeddingObject {
    pub object: &'static str,
    pub embedding: EmbeddingValue,
    pub index: usize,
}

#[derive(Serialize)]
pub struct EmbeddingsUsage {
    pub prompt_tokens: usize,
    pub total_tokens: usize,
}

#[derive(Serialize)]
pub struct EmbeddingsResponse {
    pub object: &'static str,
    pub data: Vec<EmbeddingObject>,
    pub model: String,
    pub usage: EmbeddingsUsage,
}

#[utoipa::path(
    post,
    path = "/v1/embeddings",
    tag = "openai",
    request_body = crate::openapi::schemas::OpenAiEmbeddingsRequest,
    responses(
        (status = 200, description = "Mean-pooled embeddings (not contrastively trained — use at your own risk).",
         body = crate::openapi::schemas::OpenAiEmbeddingsResponse),
        (status = 400, body = crate::error::ErrorBody),
        (status = 500, body = crate::error::ErrorBody),
    ),
)]
pub async fn handle_embeddings(
    State(state): State<Arc<AppState>>,
    Json(req): Json<EmbeddingsRequest>,
) -> Result<Json<EmbeddingsResponse>, ServerError> {
    state.bump_requests();

    let encoding = match req.encoding_format.as_deref() {
        None | Some("float") => EncodingFormat::Float,
        Some("base64") => EncodingFormat::Base64,
        Some(fmt) => {
            return Err(ServerError::BadRequest(format!(
                "encoding_format='{fmt}' is not supported (expected 'float' or 'base64')"
            )));
        }
    };

    let model = state.model_or_err(req.model.as_deref())?;

    // Resolve input to one or more token-id sequences. Strings get
    // tokenised; pre-tokenised inputs pass through.
    let model_ref: &LoadedModel = model.as_ref();
    let token_seqs: Vec<Vec<u32>> = match req.input {
        EmbeddingInput::Single(s) => vec![tokenize_one(model_ref, &s)?],
        EmbeddingInput::Batch(strs) => strs
            .iter()
            .map(|s| tokenize_one(model_ref, s))
            .collect::<Result<_, _>>()?,
        EmbeddingInput::SingleTokens(ids) => vec![ids],
        EmbeddingInput::BatchTokens(idses) => idses,
    };

    if token_seqs.iter().all(|s| s.is_empty()) {
        return Err(ServerError::BadRequest("input is empty".into()));
    }

    let mut data = Vec::with_capacity(token_seqs.len());
    let mut total_tokens = 0usize;
    for (idx, ids) in token_seqs.iter().enumerate() {
        if ids.is_empty() {
            return Err(ServerError::BadRequest(format!(
                "input[{idx}] is empty — every input must have ≥1 token"
            )));
        }
        let h = embed_tokens(model_ref, ids)?;
        let pooled = mean_pool(&h);
        total_tokens += ids.len();
        let value = match encoding {
            EncodingFormat::Float => EmbeddingValue::Floats(pooled),
            EncodingFormat::Base64 => EmbeddingValue::Base64(encode_floats_base64(&pooled)),
        };
        data.push(EmbeddingObject {
            object: EMBEDDING_OBJECT,
            embedding: value,
            index: idx,
        });
    }

    Ok(Json(EmbeddingsResponse {
        object: LIST_OBJECT,
        data,
        model: model.id.clone(),
        usage: EmbeddingsUsage {
            prompt_tokens: total_tokens,
            total_tokens,
        },
    }))
}

fn tokenize_one(model: &LoadedModel, text: &str) -> Result<Vec<u32>, ServerError> {
    let enc = model
        .tokenizer
        .encode(text, false)
        .map_err(|e| ServerError::Internal(format!("tokenize: {e}")))?;
    Ok(enc.get_ids().to_vec())
}

/// Encode a float vector as base64 of its little-endian f32 bytes.
/// Wire shape OpenAI clients expect when `encoding_format="base64"`:
/// `len(vector) * 4` bytes → standard-alphabet base64 string.
fn encode_floats_base64(values: &[f32]) -> String {
    let mut bytes = Vec::with_capacity(values.len() * 4);
    for v in values {
        bytes.extend_from_slice(&v.to_le_bytes());
    }
    base64::engine::general_purpose::STANDARD.encode(&bytes)
}

/// Mean pool a `[seq_len × hidden]` matrix to a `[hidden]` vector.
/// Returns zeros for empty sequences (caller should reject upstream).
fn mean_pool(h: &larql_vindex::ndarray::Array2<f32>) -> Vec<f32> {
    let seq_len = h.shape()[0];
    let hidden = h.shape()[1];
    if seq_len == 0 {
        return vec![0.0; hidden];
    }
    let mut out = vec![0.0f32; hidden];
    for row in h.rows() {
        for (a, &v) in out.iter_mut().zip(row.iter()) {
            *a += v;
        }
    }
    let inv_n = 1.0 / seq_len as f32;
    for v in out.iter_mut() {
        *v *= inv_n;
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use larql_vindex::ndarray::array;

    #[test]
    fn mean_pool_single_row_returns_row() {
        let h = array![[1.0f32, 2.0, 3.0]];
        let pooled = mean_pool(&h);
        assert_eq!(pooled, vec![1.0, 2.0, 3.0]);
    }

    #[test]
    fn mean_pool_two_rows_averages_per_column() {
        let h = array![[1.0f32, 4.0], [3.0, 6.0]];
        let pooled = mean_pool(&h);
        assert_eq!(pooled, vec![2.0, 5.0]);
    }

    #[test]
    fn mean_pool_empty_sequence_returns_zero_vector() {
        let h: larql_vindex::ndarray::Array2<f32> = larql_vindex::ndarray::Array2::zeros((0, 4));
        let pooled = mean_pool(&h);
        assert_eq!(pooled, vec![0.0, 0.0, 0.0, 0.0]);
    }

    #[test]
    fn embedding_input_deserializes_single_string() {
        let json = serde_json::json!({"input": "hello"});
        let req: EmbeddingsRequest = serde_json::from_value(json).unwrap();
        match req.input {
            EmbeddingInput::Single(s) => assert_eq!(s, "hello"),
            _ => panic!("expected Single"),
        }
    }

    #[test]
    fn embedding_input_deserializes_string_batch() {
        let json = serde_json::json!({"input": ["a", "b"]});
        let req: EmbeddingsRequest = serde_json::from_value(json).unwrap();
        match req.input {
            EmbeddingInput::Batch(v) => assert_eq!(v, vec!["a", "b"]),
            _ => panic!("expected Batch"),
        }
    }

    #[test]
    fn embedding_input_deserializes_pretokenised_single() {
        let json = serde_json::json!({"input": [1, 2, 3]});
        let req: EmbeddingsRequest = serde_json::from_value(json).unwrap();
        match req.input {
            EmbeddingInput::SingleTokens(v) => assert_eq!(v, vec![1, 2, 3]),
            other => panic!(
                "expected SingleTokens, got {:?}",
                std::mem::discriminant(&other)
            ),
        }
    }

    #[test]
    fn embedding_input_deserializes_pretokenised_batch() {
        let json = serde_json::json!({"input": [[1, 2], [3, 4]]});
        let req: EmbeddingsRequest = serde_json::from_value(json).unwrap();
        match req.input {
            EmbeddingInput::BatchTokens(v) => assert_eq!(v, vec![vec![1, 2], vec![3, 4]]),
            _ => panic!("expected BatchTokens"),
        }
    }

    #[test]
    fn encode_floats_base64_round_trip() {
        let v = vec![1.0f32, -2.5, 0.5, 0.0];
        let encoded = encode_floats_base64(&v);
        let decoded = base64::engine::general_purpose::STANDARD
            .decode(encoded.as_bytes())
            .expect("base64 decode");
        // 4 bytes per f32, little-endian.
        assert_eq!(decoded.len(), v.len() * 4);
        let recovered: Vec<f32> = decoded
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect();
        for (a, b) in v.iter().zip(recovered.iter()) {
            assert!((a - b).abs() < 1e-6, "{a} != {b}");
        }
    }

    #[test]
    fn embedding_value_serialises_floats_as_array() {
        let v = EmbeddingValue::Floats(vec![1.0, 2.0, 3.0]);
        let json = serde_json::to_value(&v).unwrap();
        assert!(json.is_array());
        assert_eq!(json[0], 1.0);
    }

    #[test]
    fn embedding_value_serialises_base64_as_string() {
        let v = EmbeddingValue::Base64("AAA=".to_string());
        let json = serde_json::to_value(&v).unwrap();
        assert!(json.is_string());
        assert_eq!(json.as_str().unwrap(), "AAA=");
    }
}
