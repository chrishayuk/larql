//! POST /v1/select — SQL-style edge query.

use std::sync::Arc;

use axum::extract::{Path, State};
use axum::Json;
use serde::Deserialize;

use crate::error::ServerError;
use crate::state::{elapsed_ms, AppState, LoadedModel};

#[derive(Deserialize, utoipa::ToSchema)]
pub struct SelectRequest {
    #[serde(default)]
    pub entity: Option<String>,
    /// Filter by probe-confirmed relation label.
    #[serde(default)]
    pub relation: Option<String>,
    #[serde(default)]
    pub layer: Option<usize>,
    #[serde(default = "default_limit")]
    pub limit: usize,
    #[serde(default)]
    pub min_confidence: Option<f32>,
    #[serde(default)]
    pub order_by: Option<String>,
    #[serde(default = "default_order")]
    pub order: String,
}

fn default_limit() -> usize {
    20
}
fn default_order() -> String {
    "desc".into()
}

fn select_edges(
    model: &LoadedModel,
    req: &SelectRequest,
) -> Result<serde_json::Value, ServerError> {
    let start = std::time::Instant::now();

    let patched = model.patched.blocking_read();
    let all_layers = patched.loaded_layers();

    let scan_layers: Vec<usize> = if let Some(l) = req.layer {
        vec![l]
    } else {
        all_layers
    };

    struct Row {
        layer: usize,
        feature: usize,
        top_token: String,
        c_score: f32,
        relation: Option<String>,
    }

    let mut rows: Vec<Row> = Vec::new();

    for &layer in &scan_layers {
        let num_features = patched.num_features(layer);
        for feat_idx in 0..num_features {
            // Check probe label first (fast filter when relation is specified)
            let relation = model.probe_labels.get(&(layer, feat_idx)).cloned();
            if let Some(ref rel_filter) = req.relation {
                match &relation {
                    Some(r) if r.to_lowercase().contains(&rel_filter.to_lowercase()) => {}
                    _ => continue,
                }
            }

            // Get feature metadata (handles both heap and mmap down_meta)
            if let Some(meta) = patched.feature_meta(layer, feat_idx) {
                if let Some(ref ent) = req.entity {
                    if !meta.top_token.to_lowercase().contains(&ent.to_lowercase()) {
                        continue;
                    }
                }
                if let Some(min_c) = req.min_confidence {
                    if meta.c_score < min_c {
                        continue;
                    }
                }
                rows.push(Row {
                    layer,
                    feature: feat_idx,
                    top_token: meta.top_token.clone(),
                    c_score: meta.c_score,
                    relation,
                });
            }
        }
    }

    let descending = req.order == "desc";
    match req.order_by.as_deref() {
        Some("gate_score") | Some("confidence") | Some("c_score") => {
            rows.sort_by(|a, b| {
                let cmp = a
                    .c_score
                    .partial_cmp(&b.c_score)
                    .unwrap_or(std::cmp::Ordering::Equal);
                if descending {
                    cmp.reverse()
                } else {
                    cmp
                }
            });
        }
        Some("layer") => {
            rows.sort_by(|a, b| {
                let cmp = a.layer.cmp(&b.layer);
                if descending {
                    cmp.reverse()
                } else {
                    cmp
                }
            });
        }
        _ => {
            rows.sort_by(|a, b| {
                let cmp = a
                    .c_score
                    .partial_cmp(&b.c_score)
                    .unwrap_or(std::cmp::Ordering::Equal);
                cmp.reverse()
            });
        }
    }

    let total = rows.len();
    rows.truncate(req.limit);

    let edges: Vec<serde_json::Value> = rows
        .iter()
        .map(|r| {
            let mut edge = serde_json::json!({
                "layer": r.layer,
                "feature": r.feature,
                "target": r.top_token.trim(),
                "c_score": r.c_score,
            });
            if let Some(ref rel) = r.relation {
                edge["relation"] = serde_json::json!(rel);
            }
            edge
        })
        .collect();

    Ok(serde_json::json!({
        "edges": edges,
        "total": total,
        "latency_ms": elapsed_ms(start),
    }))
}

#[utoipa::path(
    post,
    path = "/v1/select",
    tag = "inference",
    request_body = SelectRequest,
    responses(
        (status = 200, description = "Selected edges", body = crate::openapi::schemas::SelectResponse),
        (status = 400, body = crate::error::ErrorBody),
        (status = 500, body = crate::error::ErrorBody),
    ),
)]
pub async fn handle_select(
    State(state): State<Arc<AppState>>,
    Json(req): Json<SelectRequest>,
) -> Result<Json<serde_json::Value>, ServerError> {
    state.bump_requests();
    let model = state.model_or_err(None)?.clone();
    let result = tokio::task::spawn_blocking(move || select_edges(&model, &req))
        .await
        .map_err(|e| ServerError::Internal(e.to_string()))??;
    Ok(Json(result))
}

#[utoipa::path(
    post,
    path = "/v1/{model_id}/select",
    tag = "inference",
    params(("model_id" = String, Path, description = "Id of a loaded vindex.")),
    request_body = SelectRequest,
    responses(
        (status = 200, body = crate::openapi::schemas::SelectResponse),
        (status = 400, body = crate::error::ErrorBody),
        (status = 404, body = crate::error::ErrorBody),
    ),
)]
pub async fn handle_select_multi(
    State(state): State<Arc<AppState>>,
    Path(model_id): Path<String>,
    Json(req): Json<SelectRequest>,
) -> Result<Json<serde_json::Value>, ServerError> {
    state.bump_requests();
    let model = state.model_or_err(Some(&model_id))?.clone();
    let result = tokio::task::spawn_blocking(move || select_edges(&model, &req))
        .await
        .map_err(|e| ServerError::Internal(e.to_string()))??;
    Ok(Json(result))
}
