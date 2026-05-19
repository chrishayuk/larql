//! Dispatch glue between the parsed request and the FFN backends.
//!
//! [`run_walk_ffn`] is the JSON entry point used by `handle_walk_ffn`
//! when the request came in as `application/json`. It validates the
//! request shape (via [`super::validate`]), then routes to either
//! [`run_full_output`] (full FFN compute, JSON-encoded) or
//! [`run_features_only`] (gate-KNN feature lookup, fast path).

use crate::error::ServerError;
use crate::state::{AppState, LoadedModel};

use super::binary::encode_json_full_output;
use super::core::run_full_output_core;
use super::types::WalkFfnRequest;
use super::validate::{collect_scan_layers, validate_owned, validate_residual};

/// Full FFN compute → JSON envelope. Calls [`run_full_output_core`]
/// then [`encode_json_full_output`].
fn run_full_output(
    model: &LoadedModel,
    req: &WalkFfnRequest,
    scan_layers: &[usize],
    start: std::time::Instant,
) -> Result<serde_json::Value, ServerError> {
    let out = run_full_output_core(model, req, scan_layers, start)?;
    Ok(encode_json_full_output(&out))
}

/// Gate-KNN feature lookup (no FFN compute). Used when
/// `full_output: false` — fastest path for top-K feature lookup.
fn run_features_only(
    model: &LoadedModel,
    req: &WalkFfnRequest,
    scan_layers: &[usize],
    start: std::time::Instant,
) -> Result<serde_json::Value, ServerError> {
    let patched = model.patched.blocking_read();
    let query = larql_vindex::ndarray::Array1::from_vec(req.residual.clone());

    let mut results = Vec::with_capacity(scan_layers.len());
    for &layer in scan_layers {
        let hits = patched.gate_knn(layer, &query, req.top_k);
        let features: Vec<usize> = hits.iter().map(|(f, _)| *f).collect();
        let scores: Vec<f32> = hits
            .iter()
            .map(|(_, s)| (*s * 100.0).round() / 100.0)
            .collect();
        results.push(serde_json::json!({
            "layer": layer,
            "features": features,
            "scores": scores,
        }));
    }

    let latency_ms = start.elapsed().as_secs_f64() * 1000.0;
    let latency_rounded = (latency_ms * 10.0).round() / 10.0;

    if scan_layers.len() == 1 {
        let r = &results[0];
        Ok(serde_json::json!({
            "layer": r["layer"],
            "features": r["features"],
            "scores": r["scores"],
            "latency_ms": latency_rounded,
        }))
    } else {
        Ok(serde_json::json!({
            "results": results,
            "latency_ms": latency_rounded,
        }))
    }
}

/// JSON-entrypoint dispatcher: parse → validate → route to full or
/// features-only path. Returns the response JSON envelope.
pub(crate) fn run_walk_ffn(
    state: &AppState,
    req: &WalkFfnRequest,
) -> Result<serde_json::Value, ServerError> {
    let model = state
        .model(None)
        .ok_or_else(|| ServerError::NotFound("no model loaded".into()))?;

    let hidden = model.config.hidden_size;
    validate_residual(req, hidden)?;

    let scan_layers = collect_scan_layers(req)?;
    validate_owned(model, &scan_layers)?;

    let start = std::time::Instant::now();

    if req.full_output {
        run_full_output(model, req, &scan_layers, start)
    } else {
        run_features_only(model, req, &scan_layers, start)
    }
}
