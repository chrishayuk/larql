//! `POST /v1/experts/layer-batch[-f16]` — single residual + K (expert_id,
//! weight) pairs for one layer. Server applies pre_experts_norm once,
//! quantises h_norm to Q8_K once, fans out the K expert kernels with the
//! shared activation via `run_experts_cpu_batch`, returns the
//! router-weighted sum.
//!
//! Wire format documented in `larql_inference::ffn::moe_remote` next to
//! `LAYER_BATCH_CONTENT_TYPE`. Replaces the K-residual-copies pattern of
//! `/v1/expert/batch` for the common-case `forward_moe` call where every
//! expert in the layer's top-K shares the same residual.
//!
//! The f16 variant (`-f16`) halves wire bytes — opt-in via
//! `LARQL_MOE_WIRE_F16=1` for LAN deployments where the savings cancel
//! the conversion CPU cost.

use std::sync::{Arc, OnceLock};

use axum::body::Bytes;
use axum::extract::State;
use axum::http::header;
use axum::response::Response;
use tokio::sync::Semaphore;

use larql_inference::ffn::moe_remote::{
    decode_layer_batch_request, decode_layer_batch_request_f16, encode_layer_batch_response,
    encode_layer_batch_response_f16, LAYER_BATCH_CONTENT_TYPE, LAYER_BATCH_F16_CONTENT_TYPE,
};

use crate::env_flags;
use crate::error::ServerError;
use crate::state::AppState;

use super::cpu::run_experts_cpu_batch;

// Limits concurrent `run_experts_cpu_batch` calls to the number of logical
// CPUs on the machine.  Without this, 30 simultaneous predispatch requests
// each try to use rayon's global thread pool, causing ~30× oversubscription
// that balloons server compute from ~4 ms to ~180 ms per token.
//
// With the semaphore: at most N_CORES calls run simultaneously, each using
// rayon efficiently.  Wall time ≈ ceil(30 / N_CORES) × 1 ms per layer —
// ~4 ms on 8 cores vs 180 ms unthrottled.
//
// `LARQL_COMPUTE_CONCURRENCY=N` overrides the auto-detected core count.
fn compute_semaphore() -> &'static Semaphore {
    static SEM: OnceLock<Semaphore> = OnceLock::new();
    SEM.get_or_init(|| {
        let n = std::env::var("LARQL_COMPUTE_CONCURRENCY")
            .ok()
            .and_then(|s| s.parse::<usize>().ok())
            .unwrap_or_else(|| {
                std::thread::available_parallelism()
                    .map(|n| n.get())
                    .unwrap_or(8)
            });
        Semaphore::new(n)
    })
}

#[utoipa::path(
    post,
    path = "/v1/experts/layer-batch",
    tag = "expert",
    request_body(
        content_type = "application/octet-stream",
        description = "Binary wire: one pre-norm residual + K (expert_id, weight) pairs for a single layer. \
                       Router-weighted sum is returned as f32. See `docs/server-spec.md` for the exact packed layout.",
    ),
    responses(
        (status = 200, content_type = "application/x-larql-ffn",
         description = "Weighted-sum f32 output", body = Vec<u8>),
        (status = 400, body = crate::error::ErrorBody),
    ),
)]
pub async fn handle_experts_layer_batch(
    State(state): State<Arc<AppState>>,
    body: Bytes,
) -> Result<Response, ServerError> {
    state.bump_requests();
    // Per-stage timing for HTTP-overhead diagnosis. Enable with
    // `LARQL_HTTP_TIMING=1`. Cached process-wide in `env_flags`.
    let timing = env_flags::http_timing_enabled();
    let t_start = std::time::Instant::now();

    let (layer, residual, expert_ids_u32, expert_weights) = decode_layer_batch_request(&body)
        .ok_or_else(|| ServerError::BadRequest("layer-batch request truncated".into()))?;
    let t_decode = if timing {
        Some(t_start.elapsed())
    } else {
        None
    };

    let expert_ids: Vec<usize> = expert_ids_u32.iter().map(|&e| e as usize).collect();

    let t_spawn_in = std::time::Instant::now();
    // Acquire a compute slot before spawning.  Limits concurrent
    // `run_experts_cpu_batch` calls to N_CORES so rayon is not oversubscribed
    // when many predispatch requests arrive simultaneously.
    let _permit = compute_semaphore()
        .acquire()
        .await
        .map_err(|_| ServerError::Internal("compute semaphore closed".into()))?;
    let (weighted_sum, t_spawn_internal) = tokio::task::spawn_blocking(move || {
        let t_in = std::time::Instant::now();
        let r = run_experts_cpu_batch(&state, layer, &residual, &expert_ids, &expert_weights);
        let t_internal = t_in.elapsed();
        (r, t_internal)
    })
    .await
    .map_err(|e| ServerError::Internal(e.to_string()))?;
    let weighted_sum = weighted_sum?;
    let t_total_compute = t_spawn_in.elapsed();
    let t_spawn_overhead = t_total_compute.saturating_sub(t_spawn_internal);

    let t_encode_in = std::time::Instant::now();
    let latency_ms = (t_start.elapsed().as_secs_f64() * 1000.0) as f32;
    let body = encode_layer_batch_response(&weighted_sum, latency_ms);
    let t_encode = t_encode_in.elapsed();

    let resp = Response::builder()
        .header(header::CONTENT_TYPE, LAYER_BATCH_CONTENT_TYPE)
        .body(axum::body::Body::from(body))
        .map_err(|e| ServerError::Internal(e.to_string()))?;

    if timing {
        eprintln!(
            "[handle_layer_batch] layer={layer} K={} decode={:.0}us \
             spawn_overhead={:.0}us compute={:.0}us encode={:.0}us total={:.0}us",
            expert_ids_u32.len(),
            t_decode.unwrap().as_secs_f64() * 1e6,
            t_spawn_overhead.as_secs_f64() * 1e6,
            t_spawn_internal.as_secs_f64() * 1e6,
            t_encode.as_secs_f64() * 1e6,
            t_start.elapsed().as_secs_f64() * 1e6,
        );
    }

    Ok(resp)
}

#[utoipa::path(
    post,
    path = "/v1/experts/layer-batch-f16",
    tag = "expert",
    request_body(
        content_type = "application/octet-stream",
        description = "Same shape as `/v1/experts/layer-batch` but residual is f16 to cut upload bandwidth by 2×.",
    ),
    responses(
        (status = 200, content_type = "application/x-larql-ffn",
         description = "Weighted-sum f32 output", body = Vec<u8>),
        (status = 400, body = crate::error::ErrorBody),
    ),
)]
pub async fn handle_experts_layer_batch_f16(
    State(state): State<Arc<AppState>>,
    body: Bytes,
) -> Result<Response, ServerError> {
    state.bump_requests();
    let timing = env_flags::http_timing_enabled();
    let t_start = std::time::Instant::now();

    let (layer, residual, expert_ids_u32, expert_weights) =
        decode_layer_batch_request_f16(&body)
            .ok_or_else(|| ServerError::BadRequest("layer-batch-f16 request truncated".into()))?;
    let t_decode = if timing {
        Some(t_start.elapsed())
    } else {
        None
    };

    let expert_ids: Vec<usize> = expert_ids_u32.iter().map(|&e| e as usize).collect();

    let t_spawn_in = std::time::Instant::now();
    let _permit = compute_semaphore()
        .acquire()
        .await
        .map_err(|_| ServerError::Internal("compute semaphore closed".into()))?;
    let (weighted_sum, t_spawn_internal) = tokio::task::spawn_blocking(move || {
        let t_in = std::time::Instant::now();
        let r = run_experts_cpu_batch(&state, layer, &residual, &expert_ids, &expert_weights);
        let t_internal = t_in.elapsed();
        (r, t_internal)
    })
    .await
    .map_err(|e| ServerError::Internal(e.to_string()))?;
    let weighted_sum = weighted_sum?;
    let t_total_compute = t_spawn_in.elapsed();
    let t_spawn_overhead = t_total_compute.saturating_sub(t_spawn_internal);

    let t_encode_in = std::time::Instant::now();
    let latency_ms = (t_start.elapsed().as_secs_f64() * 1000.0) as f32;
    let body = encode_layer_batch_response_f16(&weighted_sum, latency_ms);
    let t_encode = t_encode_in.elapsed();

    let resp = Response::builder()
        .header(header::CONTENT_TYPE, LAYER_BATCH_F16_CONTENT_TYPE)
        .body(axum::body::Body::from(body))
        .map_err(|e| ServerError::Internal(e.to_string()))?;

    if timing {
        eprintln!(
            "[handle_layer_batch_f16] layer={layer} K={} decode={:.0}us \
             spawn_overhead={:.0}us compute={:.0}us encode={:.0}us total={:.0}us",
            expert_ids_u32.len(),
            t_decode.unwrap().as_secs_f64() * 1e6,
            t_spawn_overhead.as_secs_f64() * 1e6,
            t_spawn_internal.as_secs_f64() * 1e6,
            t_encode.as_secs_f64() * 1e6,
            t_start.elapsed().as_secs_f64() * 1e6,
        );
    }

    Ok(resp)
}

#[cfg(test)]
mod layer_batch_wire_tests {
    use larql_inference::ffn::moe_remote::{
        decode_layer_batch_request, decode_layer_batch_request_f16, encode_layer_batch_request,
        encode_layer_batch_request_f16, encode_layer_batch_response,
        encode_layer_batch_response_f16,
    };

    /// Server-side `decode_layer_batch_request` round-trips a request encoded
    /// by the client.  The actual handlers (`handle_experts_layer_batch{,_f16}`)
    /// gate on this returning `Some` — short-circuit-friendly truncation
    /// detection is critical for handler correctness, so we exercise it here.
    #[test]
    fn server_decodes_layer_batch_request_f32() {
        let layer = 7usize;
        let residual: Vec<f32> = (0..256).map(|i| i as f32 * 0.0125).collect();
        let expert_ids: Vec<u32> = vec![1, 5, 23, 42];
        let weights: Vec<f32> = vec![0.4, 0.3, 0.2, 0.1];
        let bytes = encode_layer_batch_request(layer, &residual, &expert_ids, &weights);
        let (l, r, ids, ws) = decode_layer_batch_request(&bytes).expect("decode round-trip");
        assert_eq!(l, layer);
        assert_eq!(r, residual);
        assert_eq!(ids, expert_ids);
        assert_eq!(ws, weights);
    }

    #[test]
    fn server_rejects_truncated_layer_batch_request() {
        let bytes = encode_layer_batch_request(0, &[1.0; 256], &[0u32], &[1.0]);
        for trunc in [0usize, 8, 12, bytes.len() - 1] {
            assert!(
                decode_layer_batch_request(&bytes[..trunc]).is_none(),
                "expected None on {} bytes (full = {})",
                trunc,
                bytes.len()
            );
        }
    }

    #[test]
    fn server_decodes_layer_batch_request_f16() {
        let layer = 11usize;
        let residual: Vec<f32> = (0..256).map(|i| (i as f32 * 0.013).sin() * 5.0).collect();
        let expert_ids: Vec<u32> = vec![3, 17];
        let weights: Vec<f32> = vec![0.6, 0.4];
        let bytes = encode_layer_batch_request_f16(layer, &residual, &expert_ids, &weights);
        let (l, r, ids, ws) =
            decode_layer_batch_request_f16(&bytes).expect("f16 decode round-trip");
        assert_eq!(l, layer);
        assert_eq!(ids, expert_ids);
        assert_eq!(ws, weights);
        assert_eq!(r.len(), residual.len());
        // f16 round-trip → ~3 decimal digits; tolerate 0.1% relative.
        for (a, b) in residual.iter().zip(r.iter()) {
            let tol = (a.abs() * 1e-3).max(1e-3);
            assert!((a - b).abs() < tol, "f16 drift {a} vs {b}");
        }
    }

    /// Response encoders shouldn't panic on edge dims.  Empty (hidden=0)
    /// returns a fixed-size 8-byte header (hidden u32 + latency f32).
    #[test]
    fn server_response_encoders_handle_empty() {
        let bytes_f32 = encode_layer_batch_response(&[], 0.0);
        assert_eq!(bytes_f32.len(), 8);
        let bytes_f16 = encode_layer_batch_response_f16(&[], 0.0);
        assert_eq!(bytes_f16.len(), 8);
    }
}
