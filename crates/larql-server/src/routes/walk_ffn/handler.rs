//! Axum HTTP handler for `POST /v1/walk-ffn`. Negotiates binary vs
//! JSON request shape; the binary branch goes through
//! [`super::binary::decode_binary_request`] →
//! [`super::core::run_full_output_core`] → one of the three binary
//! encoders (f32/f16/i8) per `Accept` (ADR-0009). The JSON branch
//! goes through [`super::dispatch::run_walk_ffn`].

use std::sync::Arc;

use axum::extract::State;
use axum::http::{header, StatusCode};
use axum::response::Response;

use crate::error::ServerError;
use crate::state::AppState;

use super::binary::{
    decode_binary_request, encode_binary_output, encode_binary_output_f16, encode_binary_output_i8,
};
use super::core::run_full_output_core;
use super::dispatch::run_walk_ffn;
use super::types::{RifGuard, WalkFfnRequest, BINARY_CT};
use super::validate::{collect_scan_layers, validate_owned, validate_residual};

#[utoipa::path(
    post,
    path = "/v1/walk-ffn",
    tag = "expert",
    request_body(
        content_type = "application/x-larql-ffn",
        description = "Dense-FFN walk. Accepts JSON `WalkFfnRequest` (Content-Type `application/json`) \
                       OR the packed binary `application/x-larql-ffn` wire (requires `full_output = true`). \
                       See `docs/server-spec.md` for the full wire layout.",
    ),
    responses(
        (status = 200, description = "JSON result when the request was JSON",
         content_type = "application/json", body = Vec<u8>),
        (status = 200, description = "Binary packed output when the request was `application/x-larql-ffn`",
         content_type = "application/x-larql-ffn", body = Vec<u8>),
        (status = 400, body = crate::error::ErrorBody),
        (status = 404, body = crate::error::ErrorBody),
    ),
)]
pub async fn handle_walk_ffn(
    State(state): State<Arc<AppState>>,
    request: axum::extract::Request,
) -> Result<Response, ServerError> {
    state.bump_requests();

    // Track active requests for GT6 drain, and bump the per-shard
    // cumulative counter that the grid announce loop diffs to emit
    // HeartbeatMsg.req_per_sec.
    let _rif_guard = state.models.first().map(|m| {
        use std::sync::atomic::Ordering;
        m.requests_in_flight.fetch_add(1, Ordering::Relaxed);
        m.requests_total.fetch_add(1, Ordering::Relaxed);
        RifGuard(m.requests_in_flight.clone())
    });

    let headers = request.headers();
    let is_binary = crate::wire::has_content_type(headers, BINARY_CT);
    let accept = crate::wire::accept_header(headers).map(str::to_owned);

    let body = axum::body::to_bytes(request.into_body(), 64 * 1024 * 1024)
        .await
        .map_err(|e| ServerError::BadRequest(format!("read body: {e}")))?;

    if is_binary {
        let req = decode_binary_request(&body)?;
        if !req.full_output {
            return Err(ServerError::BadRequest(
                "binary wire format requires full_output = true".into(),
            ));
        }

        // Negotiate response content-type (ADR-0009): f16 if client accepts it.
        let resp_ct = crate::wire::preferred_response_ct(accept.as_deref()).to_owned();

        let result = tokio::task::spawn_blocking(move || {
            let model = state
                .model(None)
                .ok_or_else(|| ServerError::NotFound("no model loaded".into()))?;
            validate_residual(&req, model.config.hidden_size)?;
            let scan_layers = collect_scan_layers(&req)?;
            validate_owned(model, &scan_layers)?;
            let start = std::time::Instant::now();
            let out = run_full_output_core(model, &req, &scan_layers, start)?;
            if model.release_mmap_after_request {
                let patched = model.patched.blocking_read();
                patched.base().release_mmap_pages();
            }
            Ok::<_, ServerError>(out)
        })
        .await
        .map_err(|e| ServerError::Internal(e.to_string()))??;

        let bytes = if resp_ct == crate::wire::FFN_F16_CT {
            encode_binary_output_f16(&result)
        } else if resp_ct == crate::wire::FFN_I8_CT {
            encode_binary_output_i8(&result)
        } else {
            encode_binary_output(&result)
        };
        return Ok(Response::builder()
            .status(StatusCode::OK)
            .header(header::CONTENT_TYPE, resp_ct)
            .body(axum::body::Body::from(bytes))
            .unwrap());
    }

    // JSON path — original behaviour preserved.
    let req: WalkFfnRequest = serde_json::from_slice(&body)
        .map_err(|e| ServerError::BadRequest(format!("invalid JSON: {e}")))?;

    let result = tokio::task::spawn_blocking(move || {
        let result = run_walk_ffn(&state, &req)?;
        if let Some(model) = state.model(None) {
            if model.release_mmap_after_request {
                let patched = model.patched.blocking_read();
                patched.base().release_mmap_pages();
            }
        }
        Ok::<_, ServerError>(result)
    })
    .await
    .map_err(|e| ServerError::Internal(e.to_string()))??;

    let json_bytes =
        serde_json::to_vec(&result).map_err(|e| ServerError::Internal(e.to_string()))?;
    Ok(Response::builder()
        .status(StatusCode::OK)
        .header(header::CONTENT_TYPE, "application/json")
        .body(axum::body::Body::from(json_bytes))
        .unwrap())
}
