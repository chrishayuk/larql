//! GET /v1/stats

use std::sync::Arc;

use axum::extract::{Path, State};
use axum::Json;

use crate::error::ServerError;
use crate::state::{AppState, LoadedModel};

fn build_stats(model: &LoadedModel) -> serde_json::Value {
    let config = &model.config;
    let total_features: usize = config.layers.iter().map(|l| l.num_features).sum();
    let features_per_layer = if !config.layers.is_empty() {
        config.layers[0].num_features
    } else {
        0
    };

    let layer_bands = config.layer_bands.as_ref().map(|b| {
        serde_json::json!({
            "syntax": [b.syntax.0, b.syntax.1],
            "knowledge": [b.knowledge.0, b.knowledge.1],
            "output": [b.output.0, b.output.1],
        })
    });

    let has_inference = config.extract_level == larql_vindex::ExtractLevel::Inference
        || config.extract_level == larql_vindex::ExtractLevel::All
        || config.has_model_weights;

    let mode = if model.embed_only {
        "embed-service"
    } else if model.ffn_only {
        "ffn-service"
    } else {
        "full"
    };

    serde_json::json!({
        "model": config.model,
        "family": config.family,
        "mode": mode,
        "layers": config.num_layers,
        "features": total_features,
        "features_per_layer": features_per_layer,
        "hidden_size": config.hidden_size,
        "vocab_size": config.vocab_size,
        "extract_level": config.extract_level.to_string(),
        "dtype": config.dtype.to_string(),
        "layer_bands": layer_bands,
        "loaded": {
            "browse": !model.embed_only,
            "inference": has_inference && !model.infer_disabled,
            "ffn_service": !model.embed_only,
            "embed_service": true,
        },
    })
}

/// Async wrapper for the Q4K cache + W2 surface. The base
/// `build_stats` stays sync so the existing single-/multi-model
/// handlers don't change shape; this overlay merges the `q4k_ffn`
/// block in once we have an `.await`-friendly read guard.
async fn add_q4k_ffn(model: &LoadedModel, mut stats: serde_json::Value) -> serde_json::Value {
    let p = model.patched.read().await;
    let (slots, bytes) = p.base.q4k_ffn_cache_stats();
    let has_fm = p.base.has_down_features_q4k();
    if let Some(obj) = stats.as_object_mut() {
        obj.insert(
            "q4k_ffn".into(),
            serde_json::json!({
                "cache_slots": slots,
                "cache_bytes": bytes,
                "feature_major_down": has_fm,
            }),
        );
    }
    stats
}

#[utoipa::path(
    get,
    path = "/v1/stats",
    tag = "browse",
    responses(
        (status = 200, description = "Model + vindex statistics", body = crate::openapi::schemas::StatsResponse),
        (status = 404, body = crate::error::ErrorBody),
    ),
)]
pub async fn handle_stats(
    State(state): State<Arc<AppState>>,
) -> Result<Json<serde_json::Value>, ServerError> {
    state.bump_requests();
    let model = state.model_or_err(None)?;
    let stats = build_stats(model);
    Ok(Json(add_q4k_ffn(model, stats).await))
}

#[utoipa::path(
    get,
    path = "/v1/{model_id}/stats",
    tag = "browse",
    params(("model_id" = String, Path, description = "Id of a loaded vindex.")),
    responses(
        (status = 200, body = crate::openapi::schemas::StatsResponse),
        (status = 404, body = crate::error::ErrorBody),
    ),
)]
pub async fn handle_stats_multi(
    State(state): State<Arc<AppState>>,
    Path(model_id): Path<String>,
) -> Result<Json<serde_json::Value>, ServerError> {
    state.bump_requests();
    let model = state.model_or_err(Some(&model_id))?;
    let stats = build_stats(model);
    Ok(Json(add_q4k_ffn(model, stats).await))
}
