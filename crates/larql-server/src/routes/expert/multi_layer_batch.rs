//! `POST /v1/experts/multi-layer-batch` — all 30 layers in one request.
//!
//! Receives all layers' routing decisions in a single request.  Tasks run in
//! parallel via rayon (same as the 30-concurrent-HTTP path) but over ONE TCP
//! connection, saving per-request HTTPS overhead (~15 ms × 30 connections).
//! Parallel structure: an outer `par_iter` over tasks × an inner `par_iter`
//! over each task's K experts, all scheduled on the ONE global rayon pool —
//! nesting adds stealable work items to that pool, it does NOT spawn extra
//! threads, so there is no thread oversubscription.  The leaf matvec kernel
//! is single-threaded.  At B=1 only `top_k` work items exist, so the pool
//! underfills: the flat B<4 region of any batch-size curve is thread-fill,
//! not batch efficiency.
//!
//! Wire decode, validation, compute, and response encode all run inside
//! `spawn_blocking` — the tokio reactor thread never does per-MB codec work.
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
    MultiLayerResult, MultiLayerTaskQ8K, MULTI_LAYER_BATCH_CONTENT_TYPE,
};
use larql_inference::ffn::remote::{append_timing_trailer, timing_requested, TIMING_HEADER};

use crate::env_flags;
use crate::error::ServerError;
use crate::state::AppState;

use super::cpu::{
    count_nonzero_weights, run_experts_cpu_batch, run_experts_cpu_batch_q8k_prenormed,
};

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
        (status = 400, description = "Bad request", body = crate::error::ErrorBody),
    ),
)]
pub async fn handle_experts_multi_layer_batch(
    State(state): State<Arc<AppState>>,
    headers: axum::http::HeaderMap,
    body: Bytes,
) -> Result<Response, ServerError> {
    state.bump_requests();
    let _rif_guard = crate::routes::walk_ffn::types::track_model_request(&state);
    let timing = env_flags::http_timing_enabled();
    // Opt-in timing extension (DEC-1A two-scoreboard schema): with
    // `x-larql-timing: 1` the response gains the 8-byte serve_us trailer;
    // without it the bytes are byte-identical to the pre-extension wire.
    // The `Bytes` extractor has already drained the request body, so this
    // clock starts post-receive — upload time stays in the client's
    // transmit term.
    let timing_trailer = timing_requested(headers.get(TIMING_HEADER).and_then(|v| v.to_str().ok()));
    let t_start = std::time::Instant::now();

    // Decode, validation, compute AND response encode all run inside
    // spawn_blocking: the request body can be multiple MB and the codec is
    // per-MB work the tokio reactor thread must never do.
    let (encoded, n_tasks) =
        tokio::task::spawn_blocking(move || -> Result<(Vec<u8>, usize), ServerError> {
            let tasks = decode_multi_layer_request(&body).ok_or_else(|| {
                ServerError::BadRequest("multi-layer-batch request truncated".into())
            })?;
            let n_tasks = tasks.len();

            // Shape gate before any compute (mirrors `single.rs`): every task's
            // residual must match the model's hidden_size, else the kernels compute
            // against a wrong-shaped activation.
            let hidden_size = state.model_or_err(None)?.config.hidden_size;
            for (i, task) in tasks.iter().enumerate() {
                if task.residual.len() != hidden_size {
                    return Err(ServerError::BadRequest(format!(
                        "task {i} (layer {}): residual length {} != hidden_size {hidden_size}",
                        task.layer,
                        task.residual.len()
                    )));
                }
            }

            // Parallel processing: rayon par_iter across all layers, same compute
            // shape as 30 concurrent per-layer requests but without per-connection
            // HTTPS overhead.  Arc<AppState> is Send + Sync; par_iter closure is safe.
            use rayon::prelude::*;
            let results = tasks
                .par_iter()
                .map(|task| {
                    let expert_ids: Vec<usize> =
                        task.expert_ids.iter().map(|&e| e as usize).collect();
                    let n_requested = count_nonzero_weights(&task.weights);
                    let t_layer = std::time::Instant::now();
                    let (h2, n_run) = run_experts_cpu_batch(
                        &state,
                        task.layer,
                        &task.residual,
                        &expert_ids,
                        &task.weights,
                    )?;
                    if let Some(m) = state.model(None) {
                        m.layer_latency_tracker
                            .record(task.layer as u32, t_layer.elapsed().as_secs_f32() * 1000.0);
                    }
                    if n_run != n_requested {
                        // Clients partition tasks per shard ownership — an
                        // unresolvable expert is a genuine error, never a
                        // partial sum to return as a 200.
                        return Err(ServerError::BadRequest(format!(
                            "layer {}: {} of {n_requested} requested experts could \
                             not be resolved on this shard",
                            task.layer,
                            n_requested - n_run
                        )));
                    }
                    Ok(MultiLayerResult {
                        layer: task.layer,
                        h2,
                    })
                })
                .collect::<Result<Vec<_>, ServerError>>()?;

            let mut encoded = encode_multi_layer_response(&results);
            if timing_trailer {
                append_timing_trailer(&mut encoded, t_start.elapsed().as_secs_f32() * 1e6);
            }
            Ok((encoded, n_tasks))
        })
        .await
        .map_err(|e| ServerError::Internal(e.to_string()))??;

    let latency_us = t_start.elapsed().as_secs_f64() * 1e6;

    if timing {
        eprintln!("[multi_layer_batch] tasks={n_tasks} total={latency_us:.0}us");
    }

    Response::builder()
        .header(header::CONTENT_TYPE, MULTI_LAYER_BATCH_CONTENT_TYPE)
        .body(axum::body::Body::from(encoded))
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
        (status = 400, description = "Bad request", body = crate::error::ErrorBody),
    ),
)]
pub async fn handle_experts_multi_layer_batch_q8k(
    State(state): State<Arc<AppState>>,
    headers: axum::http::HeaderMap,
    body: Bytes,
) -> Result<Response, ServerError> {
    state.bump_requests();
    let _rif_guard = crate::routes::walk_ffn::types::track_model_request(&state);
    let timing = env_flags::http_timing_enabled();
    // Same opt-in serve_us trailer as `handle_experts_multi_layer_batch`.
    let timing_trailer = timing_requested(headers.get(TIMING_HEADER).and_then(|v| v.to_str().ok()));
    let t_start = std::time::Instant::now();

    // Decode, validation, compute AND response encode all run inside
    // spawn_blocking — see `handle_experts_multi_layer_batch`.
    let (encoded, n_tasks) =
        tokio::task::spawn_blocking(move || -> Result<(Vec<u8>, usize), ServerError> {
            let tasks = decode_multi_layer_request_q8k(&body).ok_or_else(|| {
                ServerError::BadRequest("multi-layer-batch-q8k request truncated".into())
            })?;
            let n_tasks = tasks.len();

            // Shape gate before any compute (mirrors `single.rs`): the wire `hidden`
            // must match the model's hidden_size — the decoder already rejected
            // non-256-multiple values, this catches wrong-but-aligned ones.
            let hidden_size = state.model_or_err(None)?.config.hidden_size;
            for (i, task) in tasks.iter().enumerate() {
                if task.hidden != hidden_size {
                    return Err(ServerError::BadRequest(format!(
                        "task {i} (layer {}): hidden {} != hidden_size {hidden_size}",
                        task.layer, task.hidden
                    )));
                }
            }

            use rayon::prelude::*;
            // `into_par_iter` (indexed) keeps request order in the collected
            // Vec; tasks are consumed by value so the decoded qs/d/sums Vecs
            // MOVE into Q8KActivation instead of being cloned per task.
            let results = tasks
                .into_par_iter()
                .map(|task| {
                    let MultiLayerTaskQ8K {
                        layer,
                        hidden: _,
                        qs,
                        d,
                        sums,
                        expert_ids,
                        weights,
                    } = task;
                    let q8k = Q8KActivation { qs, d, sums };
                    let expert_ids: Vec<usize> = expert_ids.iter().map(|&e| e as usize).collect();
                    let n_requested = count_nonzero_weights(&weights);
                    let t_layer = std::time::Instant::now();
                    let (h2, n_run) = run_experts_cpu_batch_q8k_prenormed(
                        &state,
                        layer,
                        &q8k,
                        &expert_ids,
                        &weights,
                    )?;
                    if let Some(m) = state.model(None) {
                        m.layer_latency_tracker
                            .record(layer as u32, t_layer.elapsed().as_secs_f32() * 1000.0);
                    }
                    if n_run != n_requested {
                        return Err(ServerError::BadRequest(format!(
                            "layer {layer}: {} of {n_requested} requested experts could \
                             not be resolved on this shard",
                            n_requested - n_run
                        )));
                    }
                    Ok(MultiLayerResult { layer, h2 })
                })
                .collect::<Result<Vec<_>, ServerError>>()?;

            let mut encoded = encode_multi_layer_response(&results);
            if timing_trailer {
                append_timing_trailer(&mut encoded, t_start.elapsed().as_secs_f32() * 1e6);
            }
            Ok((encoded, n_tasks))
        })
        .await
        .map_err(|e| ServerError::Internal(e.to_string()))??;

    let latency_us = t_start.elapsed().as_secs_f64() * 1e6;

    if timing {
        eprintln!("[multi_layer_batch_q8k] tasks={n_tasks} total={latency_us:.0}us");
    }

    Response::builder()
        .header(header::CONTENT_TYPE, MULTI_LAYER_BATCH_CONTENT_TYPE)
        .body(axum::body::Body::from(encoded))
        .map_err(|e| ServerError::Internal(e.to_string()))
}
