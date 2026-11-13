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

/// Rows returned when a request omits `limit`.
pub(crate) const DEFAULT_SELECT_LIMIT: usize = 20;

fn default_limit() -> usize {
    DEFAULT_SELECT_LIMIT
}
fn default_order() -> String {
    "desc".into()
}

fn select_edges(
    sessions: &crate::session::SessionManager,
    session_id: Option<&str>,
    model: &LoadedModel,
    req: &SelectRequest,
) -> Result<serde_json::Value, ServerError> {
    crate::routes::limits::within("limit", req.limit, crate::routes::limits::MAX_RESULT_ROWS)
        .map_err(ServerError::BadRequest)?;
    let start = std::time::Instant::now();

    let patched = sessions.view_blocking(model, session_id);
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
        (status = 400, description = "Bad request", body = crate::error::ErrorBody),
        (status = 500, description = "Internal server error", body = crate::error::ErrorBody),
    ),
)]
pub async fn handle_select(
    State(state): State<Arc<AppState>>,
    headers: axum::http::HeaderMap,
    Json(req): Json<SelectRequest>,
) -> Result<Json<serde_json::Value>, ServerError> {
    state.bump_requests();
    let model = state.model_or_err(None)?.clone();
    let result = {
        let session_id = crate::session::extract_session_id(&headers);
        let sessions_of = state.clone();
        tokio::task::spawn_blocking(move || {
            select_edges(&sessions_of.sessions, session_id.as_deref(), &model, &req)
        })
    }
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
        (status = 200, description = "OK", body = crate::openapi::schemas::SelectResponse),
        (status = 400, description = "Bad request", body = crate::error::ErrorBody),
        (status = 404, description = "Not found", body = crate::error::ErrorBody),
    ),
)]
pub async fn handle_select_multi(
    State(state): State<Arc<AppState>>,
    headers: axum::http::HeaderMap,
    Path(model_id): Path<String>,
    Json(req): Json<SelectRequest>,
) -> Result<Json<serde_json::Value>, ServerError> {
    state.bump_requests();
    let model = state.model_or_err(Some(&model_id))?.clone();
    let result = {
        let session_id = crate::session::extract_session_id(&headers);
        let sessions_of = state.clone();
        tokio::task::spawn_blocking(move || {
            select_edges(&sessions_of.sessions, session_id.as_deref(), &model, &req)
        })
    }
    .await
    .map_err(|e| ServerError::Internal(e.to_string()))??;
    Ok(Json(result))
}
