//! GET /v1/walk — feature scan for a prompt.

use std::sync::Arc;

use axum::extract::{Path, Query, State};
use axum::Json;
use serde::Deserialize;

use crate::error::ServerError;
use crate::state::{elapsed_ms, AppState, LoadedModel};

#[derive(Deserialize, utoipa::IntoParams)]
#[into_params(parameter_in = Query)]
pub struct WalkParams {
    /// Prompt text to scan for features.
    pub prompt: String,
    /// Top-K features per layer.
    #[serde(default = "default_top")]
    pub top: usize,
    /// Restrict scan to these layers — either a range (`"24-33"`) or a comma list (`"14,26,27"`).
    #[serde(default)]
    pub layers: Option<String>,
}

fn default_top() -> usize {
    5
}

/// Parse a layer range string like "24-33" or "14,26,27".
fn parse_layers(s: &str, all: &[usize]) -> Vec<usize> {
    if let Some((start, end)) = s.split_once('-') {
        if let (Ok(s), Ok(e)) = (start.parse::<usize>(), end.parse::<usize>()) {
            return all.iter().copied().filter(|l| *l >= s && *l <= e).collect();
        }
    }
    s.split(',')
        .filter_map(|p| p.trim().parse::<usize>().ok())
        .filter(|l| all.contains(l))
        .collect()
}

fn walk_prompt(model: &LoadedModel, params: &WalkParams) -> Result<serde_json::Value, ServerError> {
    let start = std::time::Instant::now();

    let encoding = model
        .tokenizer
        .encode(params.prompt.as_str(), true)
        .map_err(|e| ServerError::Internal(format!("tokenize error: {e}")))?;
    let token_ids: Vec<u32> = encoding.get_ids().to_vec();

    if token_ids.is_empty() {
        return Err(ServerError::BadRequest("empty prompt".into()));
    }

    let last_tok = *token_ids.last().unwrap();
    let embed_row = model.embeddings.row(last_tok as usize);
    let query = embed_row.mapv(|v| v * model.embed_scale);

    let patched = model.patched.blocking_read();
    let all_layers = patched.loaded_layers();

    let walk_layers: Vec<usize> = match &params.layers {
        Some(s) => parse_layers(s, &all_layers),
        None => all_layers,
    };

    let trace = patched.walk(&query, &walk_layers, params.top);

    let hits: Vec<serde_json::Value> = trace
        .layers
        .iter()
        .flat_map(|(layer, hits)| {
            hits.iter().map(move |hit| {
                let mut h = serde_json::json!({
                    "layer": layer,
                    "feature": hit.feature,
                    "gate_score": (hit.gate_score * 10.0).round() / 10.0,
                    "target": hit.meta.top_token.trim(),
                });
                if let Some(label) = model.probe_labels.get(&(*layer, hit.feature)) {
                    h["relation"] = serde_json::json!(label);
                }
                h
            })
        })
        .collect();

    Ok(serde_json::json!({
        "prompt": params.prompt,
        "hits": hits,
        "latency_ms": elapsed_ms(start),
    }))
}

#[utoipa::path(
    get,
    path = "/v1/walk",
    tag = "browse",
    params(WalkParams),
    responses(
        (status = 200, description = "Top feature hits for the prompt", body = crate::openapi::schemas::WalkResponse),
        (status = 400, body = crate::error::ErrorBody),
        (status = 500, body = crate::error::ErrorBody),
    ),
)]
pub async fn handle_walk(
    State(state): State<Arc<AppState>>,
    Query(params): Query<WalkParams>,
) -> Result<Json<serde_json::Value>, ServerError> {
    state.bump_requests();
    let model = state.model_or_err(None)?.clone();
    let result = tokio::task::spawn_blocking(move || walk_prompt(&model, &params))
        .await
        .map_err(|e| ServerError::Internal(e.to_string()))??;
    Ok(Json(result))
}

#[utoipa::path(
    get,
    path = "/v1/{model_id}/walk",
    tag = "browse",
    params(
        ("model_id" = String, Path, description = "Id of a loaded vindex."),
        WalkParams,
    ),
    responses(
        (status = 200, body = crate::openapi::schemas::WalkResponse),
        (status = 400, body = crate::error::ErrorBody),
        (status = 404, body = crate::error::ErrorBody),
        (status = 500, body = crate::error::ErrorBody),
    ),
)]
pub async fn handle_walk_multi(
    State(state): State<Arc<AppState>>,
    Path(model_id): Path<String>,
    Query(params): Query<WalkParams>,
) -> Result<Json<serde_json::Value>, ServerError> {
    state.bump_requests();
    let model = state.model_or_err(Some(&model_id))?.clone();
    let result = tokio::task::spawn_blocking(move || walk_prompt(&model, &params))
        .await
        .map_err(|e| ServerError::Internal(e.to_string()))??;
    Ok(Json(result))
}
