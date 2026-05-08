//! Embed server endpoints — POST /v1/embed, POST /v1/logits, GET /v1/token/*.
//!
//! These endpoints expose the static lookup half of the transformer:
//! embeddings (token_ids → residual_0) and lm_head (residual_final → logits).
//! Both are pure table lookups / one matmul against static matrices — no
//! per-layer computation required.
//!
//! Activated when the server is started with `--embed-only`.

use std::sync::Arc;

use axum::body::Body;
use axum::extract::{Path, Query, State};
use axum::http::header;
use axum::response::{IntoResponse, Response};
use axum::Json;
use serde::{Deserialize, Serialize};
use utoipa::ToSchema;

use larql_inference::forward::predict::logits_to_predictions_pub;
use larql_vindex::ndarray::Array2;

use crate::error::ServerError;
use crate::http::{
    BINARY_FFN_CONTENT_TYPE, JSON_CONTENT_TYPE, REQUEST_BODY_LIMIT_BYTES,
    REQUEST_BODY_LIMIT_LARGE_BYTES,
};
use crate::state::{AppState, LoadedModel};

// ── Request / response types ──────────────────────────────────────────────────

#[derive(Deserialize, ToSchema)]
pub struct EmbedRequest {
    pub token_ids: Vec<u32>,
}

#[derive(Serialize, ToSchema)]
pub struct EmbedResponse {
    /// Row-major: seq_len × hidden_size f32 values.
    pub residual: Vec<Vec<f32>>,
    pub seq_len: usize,
    pub hidden_size: usize,
    pub latency_ms: f32,
}

#[derive(Deserialize, ToSchema)]
pub struct LogitsRequest {
    /// Flat f32 residual of length hidden_size (one position, post-all-layers).
    pub residual: Vec<f32>,
    #[serde(default = "default_top_k")]
    pub top_k: usize,
    #[serde(default = "default_temperature")]
    pub temperature: f32,
}

fn default_top_k() -> usize {
    5
}
fn default_temperature() -> f32 {
    1.0
}

fn error_response(error: ServerError) -> Response {
    error.into_response()
}

fn parse_binary_embed_request(bytes: &[u8]) -> Result<Vec<u32>, ServerError> {
    if bytes.len() < 4 {
        return Err(ServerError::BadRequest(
            "binary embed: need ≥4 bytes".into(),
        ));
    }
    let num_tokens = u32::from_le_bytes(bytes[..4].try_into().unwrap()) as usize;
    let expected_len = 4 + num_tokens * 4;
    if bytes.len() < expected_len {
        return Err(ServerError::BadRequest(
            "binary embed: truncated token_ids".into(),
        ));
    }
    Ok((0..num_tokens)
        .map(|i| u32::from_le_bytes(bytes[4 + i * 4..4 + i * 4 + 4].try_into().unwrap()))
        .collect())
}

fn encode_binary_embed_response(h: &Array2<f32>) -> Vec<u8> {
    let seq_len = h.shape()[0];
    let hidden = h.shape()[1];
    let mut out = Vec::with_capacity(8 + seq_len * hidden * 4);
    out.extend_from_slice(&(seq_len as u32).to_le_bytes());
    out.extend_from_slice(&(hidden as u32).to_le_bytes());
    for val in h.iter() {
        out.extend_from_slice(&val.to_le_bytes());
    }
    out
}

fn parse_binary_logits_request(bytes: &[u8]) -> Result<Vec<f32>, ServerError> {
    if !bytes.len().is_multiple_of(4) {
        return Err(ServerError::BadRequest(
            "binary logits: byte length not multiple of 4".into(),
        ));
    }
    Ok(bytes
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes(c.try_into().unwrap()))
        .collect())
}

#[derive(Serialize, ToSchema)]
pub struct TokenProb {
    pub token_id: u32,
    pub token: String,
    pub prob: f32,
}

#[derive(Serialize, ToSchema)]
pub struct LogitsResponse {
    pub top_k: Vec<TokenProb>,
    pub latency_ms: f32,
}

#[derive(Deserialize)]
pub struct TokenEncodeQuery {
    pub text: String,
}

#[derive(Deserialize)]
pub struct TokenDecodeQuery {
    pub ids: String,
}

// ── Core helpers ──────────────────────────────────────────────────────────────

/// Look up embedding rows for the given token IDs and apply the embed scale.
/// Returns shape [seq_len, hidden_size].
///
/// Uses the f16-at-rest store (with L1 cache) when available; falls back to
/// the eagerly-decoded f32 `model.embeddings` matrix otherwise.
pub(crate) fn embed_tokens(
    model: &LoadedModel,
    token_ids: &[u32],
) -> Result<Array2<f32>, ServerError> {
    let hidden = model.config.hidden_size;
    let mut h = Array2::<f32>::zeros((token_ids.len(), hidden));

    if let Some(ref store) = model.embed_store {
        // f16 path — per-row decode with L1 cache.
        for (i, &tok_id) in token_ids.iter().enumerate() {
            let row = store.lookup(tok_id).map_err(ServerError::BadRequest)?;
            let mut dst = h.row_mut(i);
            for (j, &v) in row.iter().enumerate() {
                dst[j] = v;
            }
        }
    } else {
        // f32 path — direct row copy.
        let vocab = model.embeddings.shape()[0];
        let scale = model.embed_scale;
        for (i, &tok_id) in token_ids.iter().enumerate() {
            let tid = tok_id as usize;
            if tid >= vocab {
                return Err(ServerError::BadRequest(format!(
                    "token_id {tok_id} out of range (vocab={vocab})"
                )));
            }
            let src = model.embeddings.row(tid);
            let mut dst = h.row_mut(i);
            for j in 0..hidden {
                dst[j] = src[j] * scale;
            }
        }
    }
    Ok(h)
}

// ── Handlers ──────────────────────────────────────────────────────────────────

/// `POST /v1/embed`
///
/// JSON request: `{"token_ids": [...]}`.
/// Binary request (`Content-Type: application/x-larql-ffn`):
///   - 4 bytes: num_tokens (u32 LE)
///   - num_tokens × 4 bytes: token_ids (u32 LE)
///
/// JSON response: `{"residual": [[f32, ...], ...], "seq_len": N, ...}`.
/// Binary response: seq_len×hidden_size f32 LE, prefixed by two u32 headers.
#[utoipa::path(
    post,
    path = "/v1/embed",
    tag = "admin",
    request_body(
        content = EmbedRequest,
        description = "JSON `{token_ids: [u32]}` OR binary `application/x-larql-ffn`: \
                       `[num_tokens u32 LE][token_ids u32 LE...]`.",
    ),
    responses(
        (status = 200, description = "JSON response", body = EmbedResponse),
        (status = 200, content_type = "application/x-larql-ffn",
         body = Vec<u8>,
         description = "Binary response when the request used `Content-Type: application/x-larql-ffn`: \
                        `[seq_len u32][hidden u32][seq_len × hidden f32 LE]`."),
        (status = 400, body = crate::error::ErrorBody),
        (status = 404, body = crate::error::ErrorBody),
    ),
)]
pub async fn handle_embed(
    State(state): State<Arc<AppState>>,
    headers: axum::http::HeaderMap,
    body: Body,
) -> Response {
    handle_embed_inner(&state, None, headers, body).await
}

#[utoipa::path(
    post,
    path = "/v1/{model_id}/embed",
    tag = "admin",
    params(("model_id" = String, Path, description = "Id of a loaded vindex.")),
    request_body(content = EmbedRequest, description = "JSON or binary `application/x-larql-ffn`."),
    responses(
        (status = 200, body = EmbedResponse),
        (status = 200, content_type = "application/x-larql-ffn", body = Vec<u8>),
        (status = 400, body = crate::error::ErrorBody),
        (status = 404, body = crate::error::ErrorBody),
    ),
)]
pub async fn handle_embed_multi(
    State(state): State<Arc<AppState>>,
    Path(model_id): Path<String>,
    headers: axum::http::HeaderMap,
    body: Body,
) -> Response {
    handle_embed_inner(&state, Some(model_id.as_str()), headers, body).await
}

async fn handle_embed_inner(
    state: &AppState,
    model_id: Option<&str>,
    headers: axum::http::HeaderMap,
    body: Body,
) -> Response {
    state.bump_requests();
    let model = match state.model(model_id) {
        Some(m) => m,
        None => {
            return error_response(ServerError::NotFound("model not found".into()));
        }
    };

    let is_binary = crate::wire::has_content_type(&headers, BINARY_FFN_CONTENT_TYPE);

    let bytes = match axum::body::to_bytes(body, REQUEST_BODY_LIMIT_BYTES).await {
        Ok(b) => b,
        Err(e) => {
            return error_response(ServerError::BadRequest(format!("read body: {e}")));
        }
    };

    let start = std::time::Instant::now();

    let token_ids: Vec<u32> = if is_binary {
        match parse_binary_embed_request(&bytes) {
            Ok(token_ids) => token_ids,
            Err(e) => return error_response(e),
        }
    } else {
        let req: EmbedRequest = match serde_json::from_slice(&bytes) {
            Ok(r) => r,
            Err(e) => {
                return error_response(ServerError::BadRequest(format!(
                    "parse embed request: {e}"
                )));
            }
        };
        req.token_ids
    };

    if token_ids.is_empty() {
        return error_response(ServerError::BadRequest(
            "token_ids must be non-empty".into(),
        ));
    }

    let h = match embed_tokens(model, &token_ids) {
        Ok(h) => h,
        Err(e) => return e.into_response(),
    };

    let seq_len = h.shape()[0];
    let hidden = h.shape()[1];
    let latency_ms = start.elapsed().as_secs_f32() * 1000.0;

    // Return binary if the client asked for it.
    if is_binary {
        let out = encode_binary_embed_response(&h);
        return ([(header::CONTENT_TYPE, BINARY_FFN_CONTENT_TYPE)], out).into_response();
    }

    let residual: Vec<Vec<f32>> = h.rows().into_iter().map(|row| row.to_vec()).collect();

    Json(EmbedResponse {
        residual,
        seq_len,
        hidden_size: hidden,
        latency_ms,
    })
    .into_response()
}

// ─────────────────────────────────────────────────────────────────────────────

/// `POST /v1/logits`
///
/// Accepts JSON (`{"residual": [...], "top_k": 5, "temperature": 1.0}`) or
/// binary (`Content-Type: application/x-larql-ffn`, raw hidden_size f32 LE
/// bytes). Returns JSON top-k tokens.
#[utoipa::path(
    post,
    path = "/v1/logits",
    tag = "admin",
    request_body(
        content = LogitsRequest,
        description = "JSON `{residual: [f32; hidden_size], top_k, temperature}` OR binary \
                       `application/x-larql-ffn` with raw hidden_size f32 LE bytes.",
    ),
    responses(
        (status = 200, description = "Top-K tokens from lm_head", body = LogitsResponse),
        (status = 400, body = crate::error::ErrorBody),
        (status = 500, body = crate::error::ErrorBody),
    ),
)]
pub async fn handle_logits(
    State(state): State<Arc<AppState>>,
    headers: axum::http::HeaderMap,
    body: Body,
) -> Response {
    handle_logits_inner(&state, None, headers, body).await
}

#[utoipa::path(
    post,
    path = "/v1/{model_id}/logits",
    tag = "admin",
    params(("model_id" = String, Path, description = "Id of a loaded vindex.")),
    request_body(content = LogitsRequest),
    responses(
        (status = 200, body = LogitsResponse),
        (status = 400, body = crate::error::ErrorBody),
        (status = 404, body = crate::error::ErrorBody),
    ),
)]
pub async fn handle_logits_multi(
    State(state): State<Arc<AppState>>,
    Path(model_id): Path<String>,
    headers: axum::http::HeaderMap,
    body: Body,
) -> Response {
    handle_logits_inner(&state, Some(model_id.as_str()), headers, body).await
}

async fn handle_logits_inner(
    state: &AppState,
    model_id: Option<&str>,
    headers: axum::http::HeaderMap,
    body: Body,
) -> Response {
    state.bump_requests();
    let model = match state.model(model_id) {
        Some(m) => m,
        None => return error_response(ServerError::NotFound("model not found".into())),
    };

    let is_binary = crate::wire::has_content_type(&headers, BINARY_FFN_CONTENT_TYPE);

    let bytes = match axum::body::to_bytes(body, REQUEST_BODY_LIMIT_LARGE_BYTES).await {
        Ok(b) => b,
        Err(e) => return error_response(ServerError::BadRequest(format!("read body: {e}"))),
    };

    let (residual_flat, top_k, temperature): (Vec<f32>, usize, f32) = if is_binary {
        match parse_binary_logits_request(&bytes) {
            Ok(floats) => (floats, default_top_k(), default_temperature()),
            Err(e) => return error_response(e),
        }
    } else {
        let req: LogitsRequest = match serde_json::from_slice(&bytes) {
            Ok(r) => r,
            Err(e) => {
                return error_response(ServerError::BadRequest(format!(
                    "parse logits request: {e}"
                )));
            }
        };
        (req.residual, req.top_k, req.temperature)
    };

    let hidden = model.config.hidden_size;
    if residual_flat.len() != hidden {
        return error_response(ServerError::BadRequest(format!(
            "residual length {} != hidden_size {}",
            residual_flat.len(),
            hidden
        )));
    }

    let weights_guard = match model.get_or_load_weights() {
        Ok(w) => w,
        Err(e) => {
            return error_response(ServerError::Internal(format!("load weights: {e}")));
        }
    };
    let weights: &larql_inference::ModelWeights = &weights_guard;

    let start = std::time::Instant::now();

    // Wrap the flat residual as [1, hidden] for logits_to_predictions_pub.
    let h = Array2::from_shape_vec((1, hidden), residual_flat).unwrap();
    let result = logits_to_predictions_pub(weights, &h, &model.tokenizer, top_k, temperature);

    let latency_ms = start.elapsed().as_secs_f32() * 1000.0;

    let top_k_out: Vec<TokenProb> = result
        .predictions
        .iter()
        .zip(result.token_ids.iter())
        .map(|((token, prob), &token_id)| TokenProb {
            token_id,
            token: token.clone(),
            prob: *prob as f32,
        })
        .collect();

    Json(LogitsResponse {
        top_k: top_k_out,
        latency_ms,
    })
    .into_response()
}

// ─────────────────────────────────────────────────────────────────────────────

/// `GET /v1/token/encode?text=Paris`
#[utoipa::path(
    get,
    path = "/v1/token/encode",
    tag = "admin",
    params(
        ("text" = String, Query, description = "Text to tokenize."),
    ),
    responses(
        (status = 200, description = "Token IDs for the text", body = crate::openapi::schemas::TokenEncodeResponse),
        (status = 500, body = crate::error::ErrorBody),
    ),
)]
pub async fn handle_token_encode(
    State(state): State<Arc<AppState>>,
    Query(q): Query<TokenEncodeQuery>,
) -> Result<Json<serde_json::Value>, ServerError> {
    handle_token_encode_inner(&state, None, q)
}

#[utoipa::path(
    get,
    path = "/v1/{model_id}/token/encode",
    tag = "admin",
    params(
        ("model_id" = String, Path, description = "Id of a loaded vindex."),
        ("text" = String, Query, description = "Text to tokenize."),
    ),
    responses(
        (status = 200, body = crate::openapi::schemas::TokenEncodeResponse),
        (status = 404, body = crate::error::ErrorBody),
    ),
)]
pub async fn handle_token_encode_multi(
    State(state): State<Arc<AppState>>,
    Path(model_id): Path<String>,
    Query(q): Query<TokenEncodeQuery>,
) -> Result<Json<serde_json::Value>, ServerError> {
    handle_token_encode_inner(&state, Some(&model_id), q)
}

fn handle_token_encode_inner(
    state: &AppState,
    model_id: Option<&str>,
    q: TokenEncodeQuery,
) -> Result<Json<serde_json::Value>, ServerError> {
    state.bump_requests();
    let model = state.model_or_err(model_id)?;

    let enc = model
        .tokenizer
        .encode(q.text.as_str(), false)
        .map_err(|e| ServerError::Internal(format!("tokenize: {e}")))?;
    let ids: Vec<u32> = enc.get_ids().to_vec();

    Ok(Json(serde_json::json!({
        "token_ids": ids,
        "text": q.text,
    })))
}

// ─────────────────────────────────────────────────────────────────────────────

/// `GET /v1/token/decode?ids=9515,235,1234`
#[utoipa::path(
    get,
    path = "/v1/token/decode",
    tag = "admin",
    params(
        ("ids" = String, Query, description = "Comma-separated token IDs, e.g. `9515,235,1234`."),
    ),
    responses(
        (status = 200, description = "Decoded text", body = crate::openapi::schemas::TokenDecodeResponse),
        (status = 400, body = crate::error::ErrorBody),
        (status = 500, body = crate::error::ErrorBody),
    ),
)]
pub async fn handle_token_decode(
    State(state): State<Arc<AppState>>,
    Query(q): Query<TokenDecodeQuery>,
) -> Result<Json<serde_json::Value>, ServerError> {
    handle_token_decode_inner(&state, None, q)
}

#[utoipa::path(
    get,
    path = "/v1/{model_id}/token/decode",
    tag = "admin",
    params(
        ("model_id" = String, Path, description = "Id of a loaded vindex."),
        ("ids" = String, Query, description = "Comma-separated token IDs."),
    ),
    responses(
        (status = 200, body = crate::openapi::schemas::TokenDecodeResponse),
        (status = 400, body = crate::error::ErrorBody),
        (status = 404, body = crate::error::ErrorBody),
    ),
)]
pub async fn handle_token_decode_multi(
    State(state): State<Arc<AppState>>,
    Path(model_id): Path<String>,
    Query(q): Query<TokenDecodeQuery>,
) -> Result<Json<serde_json::Value>, ServerError> {
    handle_token_decode_inner(&state, Some(&model_id), q)
}

fn handle_token_decode_inner(
    state: &AppState,
    model_id: Option<&str>,
    q: TokenDecodeQuery,
) -> Result<Json<serde_json::Value>, ServerError> {
    state.bump_requests();
    let model = state.model_or_err(model_id)?;

    let ids: Vec<u32> = q
        .ids
        .split(',')
        .filter(|s| !s.trim().is_empty())
        .map(|s| {
            s.trim()
                .parse::<u32>()
                .map_err(|_| ServerError::BadRequest(format!("invalid token id: '{s}'")))
        })
        .collect::<Result<Vec<u32>, _>>()?;

    let text = model
        .tokenizer
        .decode(&ids, true)
        .map_err(|e| ServerError::Internal(format!("decode: {e}")))?;

    Ok(Json(serde_json::json!({
        "text": text,
        "token_ids": ids,
    })))
}

// ─────────────────────────────────────────────────────────────────────────────

/// `GET /v1/embed/{token_id}`
///
/// Returns the scaled f32 embedding vector for a single token ID.
/// The key (token_id) is a 32-bit integer; the value is a deterministic
/// function of the model weights — so the response is immutably cacheable:
///
///   Cache-Control: public, max-age=31536000, immutable
///
/// CDN-friendly: a reverse proxy or browser can cache the embedding for
/// any token permanently, eliminating repeated lookups for high-frequency
/// tokens (the, a, in, …) on the decode path.
///
/// Response (binary, 10 KB for hidden=2560):
///   [f32 × hidden_size] — LE bytes, pre-scaled
///
/// Response (JSON, if Accept: application/json):
///   {"token_id": N, "embedding": [f32, ...], "hidden_size": N}
#[utoipa::path(
    get,
    path = "/v1/embed/{token_id}",
    tag = "admin",
    params(
        ("token_id" = u32, Path, description = "Vocabulary token id."),
    ),
    responses(
        (status = 200, description = "Binary f32 LE embedding (default)", content_type = "application/x-larql-ffn", body = Vec<u8>),
        (status = 200, description = "JSON response when `Accept: application/json`", body = crate::openapi::schemas::EmbedSingleJsonResponse),
        (status = 400, body = crate::error::ErrorBody),
        (status = 404, body = crate::error::ErrorBody),
    ),
)]
pub async fn handle_embed_single(
    State(state): State<Arc<AppState>>,
    Path(token_id): Path<u32>,
    headers: axum::http::HeaderMap,
) -> Response {
    handle_embed_single_inner(&state, None, token_id, headers)
}

#[utoipa::path(
    get,
    path = "/v1/{model_id}/embed/{token_id}",
    tag = "admin",
    params(
        ("model_id" = String, Path, description = "Id of a loaded vindex."),
        ("token_id" = u32, Path),
    ),
    responses(
        (status = 200, content_type = "application/x-larql-ffn", body = Vec<u8>),
        (status = 200, body = crate::openapi::schemas::EmbedSingleJsonResponse),
        (status = 400, body = crate::error::ErrorBody),
        (status = 404, body = crate::error::ErrorBody),
    ),
)]
pub async fn handle_embed_single_multi(
    State(state): State<Arc<AppState>>,
    Path((model_id, token_id)): Path<(String, u32)>,
    headers: axum::http::HeaderMap,
) -> Response {
    handle_embed_single_inner(&state, Some(model_id.as_str()), token_id, headers)
}

fn handle_embed_single_inner(
    state: &AppState,
    model_id: Option<&str>,
    token_id: u32,
    headers: axum::http::HeaderMap,
) -> Response {
    state.bump_requests();
    let model = match state.model(model_id) {
        Some(m) => m,
        None => return error_response(ServerError::NotFound("model not found".into())),
    };

    let row: Vec<f32> = if let Some(ref store) = model.embed_store {
        match store.lookup(token_id) {
            Ok(r) => r,
            Err(e) => return error_response(ServerError::BadRequest(e)),
        }
    } else {
        let vocab = model.embeddings.shape()[0];
        let scale = model.embed_scale;
        let tid = token_id as usize;
        if tid >= vocab {
            return error_response(ServerError::BadRequest(format!(
                "token_id {token_id} out of range (vocab={vocab})"
            )));
        }
        model
            .embeddings
            .row(tid)
            .iter()
            .map(|&v| v * scale)
            .collect()
    };

    let cache_headers = [
        (header::CACHE_CONTROL, "public, max-age=31536000, immutable"),
        (header::VARY, "Accept"),
    ];

    let want_json = headers
        .get(header::ACCEPT)
        .and_then(|v| v.to_str().ok())
        .map(|s| s.contains(JSON_CONTENT_TYPE))
        .unwrap_or(false);

    if want_json {
        let body = serde_json::json!({
            "token_id": token_id,
            "embedding": row,
            "hidden_size": row.len(),
        });
        return (cache_headers, Json(body)).into_response();
    }

    // Default: binary f32 LE.
    let mut out = Vec::with_capacity(row.len() * 4);
    for v in &row {
        out.extend_from_slice(&v.to_le_bytes());
    }
    (
        [
            (header::CONTENT_TYPE, BINARY_FFN_CONTENT_TYPE),
            (header::CACHE_CONTROL, "public, max-age=31536000, immutable"),
            (header::VARY, "Accept"),
        ],
        out,
    )
        .into_response()
}

// ── Inline tests ──────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use larql_vindex::ndarray::Array2;

    // ── Binary wire format helpers ───────────────────────────────────────────

    fn make_binary_embed_request(token_ids: &[u32]) -> Vec<u8> {
        let mut out = Vec::with_capacity(4 + token_ids.len() * 4);
        out.extend_from_slice(&(token_ids.len() as u32).to_le_bytes());
        for &id in token_ids {
            out.extend_from_slice(&id.to_le_bytes());
        }
        out
    }

    fn make_binary_logits_request(floats: &[f32]) -> Vec<u8> {
        let mut out = Vec::with_capacity(floats.len() * 4);
        for &v in floats {
            out.extend_from_slice(&v.to_le_bytes());
        }
        out
    }

    // ── Embed binary encode/decode ───────────────────────────────────────────

    #[test]
    fn binary_embed_request_encodes_num_tokens() {
        let body = make_binary_embed_request(&[1, 2, 3]);
        let num = u32::from_le_bytes(body[..4].try_into().unwrap());
        assert_eq!(num, 3);
        assert_eq!(parse_binary_embed_request(&body).unwrap(), vec![1, 2, 3]);
    }

    #[test]
    fn binary_embed_request_encodes_token_ids() {
        let ids = [100u32, 200, 300];
        let body = make_binary_embed_request(&ids);
        for (i, &expected) in ids.iter().enumerate() {
            let got = u32::from_le_bytes(body[4 + i * 4..4 + i * 4 + 4].try_into().unwrap());
            assert_eq!(got, expected);
        }
    }

    #[test]
    fn binary_embed_request_total_length() {
        // 4 (num_tokens u32) + N × 4 (token_id u32)
        let body = make_binary_embed_request(&[1, 2, 3, 4, 5]);
        assert_eq!(body.len(), 4 + 5 * 4);
    }

    #[test]
    fn binary_embed_response_header_fields() {
        // Response format: [seq_len u32][hidden_size u32][seq_len × hidden_size f32]
        let seq_len = 2usize;
        let hidden = 4usize;
        let h = Array2::<f32>::from_elem((seq_len, hidden), 1.23);
        let out = encode_binary_embed_response(&h);
        assert_eq!(
            u32::from_le_bytes(out[..4].try_into().unwrap()) as usize,
            seq_len
        );
        assert_eq!(
            u32::from_le_bytes(out[4..8].try_into().unwrap()) as usize,
            hidden
        );
        assert_eq!(out.len(), 8 + seq_len * hidden * 4);
    }

    #[test]
    fn binary_embed_response_float_roundtrip() {
        let seq_len = 1usize;
        let hidden = 4usize;
        let values = [0.1f32, -0.5, 1.0, 2.5];
        let mut out = vec![0u8; 8];
        for &v in &values {
            out.extend_from_slice(&v.to_le_bytes());
        }
        let payload = &out[8..];
        for (i, chunk) in payload.chunks_exact(4).enumerate() {
            let got = f32::from_le_bytes(chunk.try_into().unwrap());
            assert!(
                (got - values[i]).abs() < 1e-6,
                "float[{i}]: {got} != {}",
                values[i]
            );
        }
        let _ = (seq_len, hidden);
    }

    // ── Logits binary encode/decode ──────────────────────────────────────────

    #[test]
    fn binary_logits_request_byte_length() {
        let residual: Vec<f32> = (0..8).map(|i| i as f32).collect();
        let body = make_binary_logits_request(&residual);
        assert_eq!(body.len(), 8 * 4);
    }

    #[test]
    fn binary_logits_request_float_roundtrip() {
        let residual = [1.5f32, -2.0, 0.0, 99.9];
        let body = make_binary_logits_request(&residual);
        assert_eq!(parse_binary_logits_request(&body).unwrap(), residual);
        for (i, chunk) in body.chunks_exact(4).enumerate() {
            let got = f32::from_le_bytes(chunk.try_into().unwrap());
            assert!((got - residual[i]).abs() < 1e-6);
        }
    }

    #[test]
    fn binary_logits_odd_length_is_invalid() {
        // A body of 5 bytes is not a multiple of 4.
        let body = [0u8; 5];
        assert_ne!(body.len() % 4, 0, "5 bytes must fail the alignment check");
        assert!(matches!(
            parse_binary_logits_request(&body),
            Err(ServerError::BadRequest(_))
        ));
    }

    #[test]
    fn binary_embed_rejects_short_header() {
        assert!(matches!(
            parse_binary_embed_request(&[0, 1, 2]),
            Err(ServerError::BadRequest(_))
        ));
    }

    #[test]
    fn binary_embed_rejects_truncated_token_ids() {
        let mut body = Vec::new();
        body.extend_from_slice(&2u32.to_le_bytes());
        body.extend_from_slice(&7u32.to_le_bytes());
        assert!(matches!(
            parse_binary_embed_request(&body),
            Err(ServerError::BadRequest(_))
        ));
    }

    // ── Token decode query parsing ───────────────────────────────────────────

    #[test]
    fn token_decode_query_parse_csv() {
        let q = "9515,235,1234";
        let ids: Vec<u32> = q
            .split(',')
            .filter(|s| !s.trim().is_empty())
            .map(|s| s.trim().parse::<u32>().unwrap())
            .collect();
        assert_eq!(ids, vec![9515u32, 235, 1234]);
    }

    #[test]
    fn token_decode_query_handles_whitespace() {
        let q = " 9515 , 235 , 1234 ";
        let ids: Vec<u32> = q
            .split(',')
            .filter(|s| !s.trim().is_empty())
            .map(|s| s.trim().parse::<u32>().unwrap())
            .collect();
        assert_eq!(ids, vec![9515u32, 235, 1234]);
    }

    #[test]
    fn token_decode_query_single_id() {
        let q = "9515";
        let ids: Vec<u32> = q
            .split(',')
            .filter(|s| !s.trim().is_empty())
            .map(|s| s.trim().parse::<u32>().unwrap())
            .collect();
        assert_eq!(ids, vec![9515u32]);
    }

    // ── Embed matrix lookup logic ────────────────────────────────────────────

    #[test]
    fn embed_lookup_returns_correct_row() {
        // embed[2] = [0, 0, 1, 0] → after scale=1.0 same
        let mut embed = Array2::<f32>::zeros((4, 4));
        embed[[2, 2]] = 1.0;
        let scale = 1.0f32;

        let tok_id = 2usize;
        let row: Vec<f32> = embed.row(tok_id).iter().map(|&v| v * scale).collect();
        assert_eq!(row, vec![0.0, 0.0, 1.0, 0.0]);
    }

    #[test]
    fn embed_lookup_applies_scale() {
        let mut embed = Array2::<f32>::zeros((4, 4));
        embed[[1, 0]] = 1.0;
        let scale = 2.5f32;

        let row: Vec<f32> = embed.row(1).iter().map(|&v| v * scale).collect();
        assert_eq!(row, vec![2.5, 0.0, 0.0, 0.0]);
    }

    #[test]
    fn embed_lookup_out_of_range_detected() {
        let embed = Array2::<f32>::zeros((8, 4));
        let vocab = embed.shape()[0];
        assert!((8usize >= vocab)); // token_id=8 is OOB for vocab=8
        assert!(7usize < vocab); // token_id=7 is in range
    }

    #[test]
    fn embed_response_shape() {
        // seq_len=3 tokens, hidden=4 → residual is [[f32×4], [f32×4], [f32×4]]
        let seq_len = 3;
        let hidden = 4;
        let h = Array2::<f32>::zeros((seq_len, hidden));
        let residual: Vec<Vec<f32>> = h.rows().into_iter().map(|r| r.to_vec()).collect();
        assert_eq!(residual.len(), seq_len);
        assert!(residual.iter().all(|row| row.len() == hidden));
    }

    // ── Default parameter values ─────────────────────────────────────────────

    #[test]
    fn default_top_k_is_five() {
        assert_eq!(default_top_k(), 5);
    }

    #[test]
    fn default_temperature_is_one() {
        assert!((default_temperature() - 1.0).abs() < 1e-6);
    }
}
