//! `POST /v1/expert/batch` — pre-2026-05-01 multi-expert wire format.
//!
//! Each item carries its own residual; the server runs them in parallel via
//! rayon. Superseded for the common-case `forward_moe` flow by
//! `/v1/experts/layer-batch` (one residual + K (expert_id, weight) pairs),
//! but kept here because:
//!   - the binary `application/x-larql-expert` wire is still emitted by
//!     older clients during rolling upgrades,
//!   - it's the only batch endpoint that supports cross-layer requests in
//!     a single round-trip (e.g. interp tooling).

use std::sync::Arc;

use axum::body::Bytes;
use axum::extract::State;
use axum::http::header;
use axum::response::Response;

use larql_inference::ffn::moe_remote::{
    decode_expert_request, encode_expert_response, ExpertCallItem, ExpertResultItem,
    EXPERT_BINARY_CONTENT_TYPE,
};

use crate::error::ServerError;
use crate::http::JSON_CONTENT_TYPE;
use crate::state::AppState;

use super::single::run_expert;
use super::{BatchExpertRequest, BatchExpertResponse, BatchExpertResult};

#[utoipa::path(
    post,
    path = "/v1/expert/batch",
    tag = "expert",
    request_body(
        content = crate::routes::expert::BatchExpertRequest,
        description = "JSON `{requests:[{layer, expert_id, residual}]}` OR binary \
                       `application/x-larql-expert` wire (see docs/server-spec.md).",
    ),
    responses(
        (status = 200, description = "Per-item expert outputs", body = crate::routes::expert::BatchExpertResponse),
        (status = 400, body = crate::error::ErrorBody),
    ),
)]
pub async fn handle_expert_batch(
    State(state): State<Arc<AppState>>,
    headers: axum::http::HeaderMap,
    body: Bytes,
) -> Result<Response, ServerError> {
    state.bump_requests();
    let start = std::time::Instant::now();

    // Accept both binary (application/x-larql-expert) and JSON.
    let binary = crate::wire::has_content_type(&headers, EXPERT_BINARY_CONTENT_TYPE);

    let items: Vec<ExpertCallItem> = if binary {
        decode_expert_request(&body)
            .ok_or_else(|| ServerError::BadRequest("binary expert request truncated".into()))?
    } else {
        let req: BatchExpertRequest = serde_json::from_slice(&body)
            .map_err(|e| ServerError::BadRequest(format!("JSON parse: {e}")))?;
        req.requests
            .into_iter()
            .map(|r| ExpertCallItem {
                layer: r.layer,
                expert_id: r.expert_id,
                residual: r.residual,
            })
            .collect()
    };

    let result_items = tokio::task::spawn_blocking(move || {
        use rayon::prelude::*;
        items
            .par_iter()
            .map(|item| {
                run_expert(&state, item.layer, item.expert_id, &item.residual).map(|output| {
                    ExpertResultItem {
                        layer: item.layer,
                        expert_id: item.expert_id,
                        output,
                    }
                })
            })
            .collect::<Result<Vec<ExpertResultItem>, ServerError>>()
    })
    .await
    .map_err(|e| ServerError::Internal(e.to_string()))??;

    let latency_ms = (start.elapsed().as_secs_f64() * 1000.0) as f32;

    let response = if binary {
        let body = encode_expert_response(&result_items, latency_ms);
        Response::builder()
            .header(header::CONTENT_TYPE, EXPERT_BINARY_CONTENT_TYPE)
            .body(axum::body::Body::from(body))
            .map_err(|e| ServerError::Internal(e.to_string()))?
    } else {
        let resp = BatchExpertResponse {
            results: result_items
                .into_iter()
                .map(|r| BatchExpertResult {
                    layer: r.layer,
                    expert_id: r.expert_id,
                    output: r.output,
                })
                .collect(),
            latency_ms: latency_ms as f64,
        };
        Response::builder()
            .header(header::CONTENT_TYPE, JSON_CONTENT_TYPE)
            .body(axum::body::Body::from(
                serde_json::to_vec(&resp).map_err(|e| ServerError::Internal(e.to_string()))?,
            ))
            .map_err(|e| ServerError::Internal(e.to_string()))?
    };

    Ok(response)
}
