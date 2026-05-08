//! `POST /v1/experts/multi-layer-batch` — all 30 layers in one request.
//!
//! Receives all layers' routing decisions in a single request.  Tasks run in
//! parallel via rayon (same as the 30-concurrent-HTTP path) but over ONE TCP
//! connection, saving per-request HTTPS overhead (~15 ms × 30 connections).
//! The outer rayon parallelises across layers; each layer's run_experts_cpu_batch
//! uses rayon internally for K experts.  Total parallelism = n_layers × K_experts;
//! moderate oversubscription on 8 cores is acceptable and measurably faster than
//! pure sequential processing.
//!
//! Used by the predispatch path when all shards are HTTP/UDS transport.

use std::sync::Arc;

use axum::body::Bytes;
use axum::extract::State;
use axum::http::header;
use axum::response::Response;

use larql_compute::Q8KActivation;
use larql_inference::ffn::moe_remote::{
    decode_multi_layer_request, decode_multi_layer_request_q8k, encode_multi_layer_response,
    MultiLayerResult, MULTI_LAYER_BATCH_CONTENT_TYPE,
};

use crate::env_flags;
use crate::error::ServerError;
use crate::state::AppState;

use super::cpu::{run_experts_cpu_batch, run_experts_cpu_batch_q8k_prenormed};

#[utoipa::path(
    post,
    path = "/v1/experts/multi-layer-batch",
    tag = "expert",
    request_body(
        content_type = "application/octet-stream",
        description = "N packed layer tasks — each `(layer, residual, expert_ids, weights)`. \
                       Server runs every task in parallel via rayon and returns N per-layer f32 outputs in one response.",
    ),
    responses(
        (status = 200, content_type = "application/x-larql-ffn-multi",
         description = "N per-layer f32 outputs, one per task", body = Vec<u8>),
        (status = 400, body = crate::error::ErrorBody),
    ),
)]
pub async fn handle_experts_multi_layer_batch(
    State(state): State<Arc<AppState>>,
    body: Bytes,
) -> Result<Response, ServerError> {
    state.bump_requests();
    let timing = env_flags::http_timing_enabled();
    let t_start = std::time::Instant::now();

    let tasks = decode_multi_layer_request(&body)
        .ok_or_else(|| ServerError::BadRequest("multi-layer-batch request truncated".into()))?;
    let n_tasks = tasks.len();

    // Parallel processing: rayon par_iter across all layers, same compute
    // shape as 30 concurrent per-layer requests but without per-connection
    // HTTPS overhead.  Arc<AppState> is Send + Sync; par_iter closure is safe.
    let results =
        tokio::task::spawn_blocking(move || -> Result<Vec<MultiLayerResult>, ServerError> {
            use rayon::prelude::*;
            tasks
                .par_iter()
                .map(|task| {
                    let expert_ids: Vec<usize> =
                        task.expert_ids.iter().map(|&e| e as usize).collect();
                    let h2 = run_experts_cpu_batch(
                        &state,
                        task.layer,
                        &task.residual,
                        &expert_ids,
                        &task.weights,
                    )?;
                    Ok(MultiLayerResult {
                        layer: task.layer,
                        h2,
                    })
                })
                .collect::<Result<Vec<_>, ServerError>>()
        })
        .await
        .map_err(|e| ServerError::Internal(e.to_string()))??;

    let latency_us = t_start.elapsed().as_secs_f64() * 1e6;
    let body = encode_multi_layer_response(&results);

    if timing {
        eprintln!("[multi_layer_batch] tasks={n_tasks} total={latency_us:.0}us");
    }

    Response::builder()
        .header(header::CONTENT_TYPE, MULTI_LAYER_BATCH_CONTENT_TYPE)
        .body(axum::body::Body::from(body))
        .map_err(|e| ServerError::Internal(e.to_string()))
}

/// Q8K-prenormed variant: client pre-quantises h_norm, server skips
/// `pre_experts_norm` and `quantize_h_norm_for_q4k` — just the matvec.
/// 4× smaller upload; response is standard f32.
#[utoipa::path(
    post,
    path = "/v1/experts/multi-layer-batch-q8k",
    tag = "expert",
    request_body(
        content_type = "application/octet-stream",
        description = "Same shape as `/v1/experts/multi-layer-batch` but the client has already applied `pre_experts_norm` \
                       and quantised `h_norm` to Q8K, saving ~4× upload bandwidth. Response is standard f32.",
    ),
    responses(
        (status = 200, content_type = "application/x-larql-ffn-multi",
         description = "N per-layer f32 outputs, one per task", body = Vec<u8>),
        (status = 400, body = crate::error::ErrorBody),
    ),
)]
pub async fn handle_experts_multi_layer_batch_q8k(
    State(state): State<Arc<AppState>>,
    body: Bytes,
) -> Result<Response, ServerError> {
    state.bump_requests();
    let timing = env_flags::http_timing_enabled();
    let t_start = std::time::Instant::now();

    let tasks = decode_multi_layer_request_q8k(&body)
        .ok_or_else(|| ServerError::BadRequest("multi-layer-batch-q8k request truncated".into()))?;
    let n_tasks = tasks.len();

    let results = tokio::task::spawn_blocking(move || {
        use rayon::prelude::*;
        tasks
            .par_iter()
            .map(|task| {
                // Reconstruct Q8KActivation from wire fields.
                let q8k = Q8KActivation {
                    qs: task.qs.clone(),
                    d: task.d.clone(),
                    sums: task.sums.clone(),
                };
                let expert_ids: Vec<usize> = task.expert_ids.iter().map(|&e| e as usize).collect();
                let h2 = run_experts_cpu_batch_q8k_prenormed(
                    &state,
                    task.layer,
                    &q8k,
                    &expert_ids,
                    &task.weights,
                )?;
                Ok(MultiLayerResult {
                    layer: task.layer,
                    h2,
                })
            })
            .collect::<Result<Vec<_>, ServerError>>()
    })
    .await
    .map_err(|e| ServerError::Internal(e.to_string()))??;

    let latency_us = t_start.elapsed().as_secs_f64() * 1e6;
    let body = encode_multi_layer_response(&results);

    if timing {
        eprintln!("[multi_layer_batch_q8k] tasks={n_tasks} total={latency_us:.0}us");
    }

    Response::builder()
        .header(header::CONTENT_TYPE, MULTI_LAYER_BATCH_CONTENT_TYPE)
        .body(axum::body::Body::from(body))
        .map_err(|e| ServerError::Internal(e.to_string()))
}
