//! GET /v1/relations — list all known relation types (top tokens).

use std::collections::HashMap;
use std::sync::Arc;

use axum::extract::{Path, Query, State};
use axum::Json;
use serde::Deserialize;

use crate::error::ServerError;
use crate::state::{elapsed_ms, AppState, LoadedModel};

/// Content-word filter matching the local executor's `is_content_token`.
fn is_content_token(tok: &str) -> bool {
    let tok = tok.trim();
    if tok.is_empty() || tok.len() > 30 {
        return false;
    }
    // is_readable_token inline
    let readable = tok
        .chars()
        .filter(|c| {
            c.is_ascii_alphanumeric()
                || *c == ' '
                || *c == '-'
                || *c == '\''
                || *c == '.'
                || *c == ','
        })
        .count();
    let total = tok.chars().count();
    if readable * 2 < total || total == 0 {
        return false;
    }
    let chars: Vec<char> = tok.chars().collect();
    if chars.len() < 3 || chars.len() > 25 {
        return false;
    }
    let alpha = chars.iter().filter(|c| c.is_ascii_alphabetic()).count();
    if alpha < chars.len() * 2 / 3 {
        return false;
    }
    for w in chars.windows(2) {
        if w[0].is_ascii_lowercase() && w[1].is_ascii_uppercase() {
            return false;
        }
    }
    if !chars.iter().any(|c| c.is_ascii_alphabetic()) {
        return false;
    }
    let lower = tok.to_lowercase();
    !matches!(
        lower.as_str(),
        "the"
            | "and"
            | "for"
            | "but"
            | "not"
            | "you"
            | "all"
            | "can"
            | "her"
            | "was"
            | "one"
            | "our"
            | "out"
            | "are"
            | "has"
            | "his"
            | "how"
            | "its"
            | "may"
            | "new"
            | "now"
            | "old"
            | "see"
            | "way"
            | "who"
            | "did"
            | "get"
            | "let"
            | "say"
            | "she"
            | "too"
            | "use"
            | "from"
            | "have"
            | "been"
            | "will"
            | "with"
            | "this"
            | "that"
            | "they"
            | "were"
            | "some"
            | "them"
            | "than"
            | "when"
            | "what"
            | "your"
            | "each"
            | "make"
            | "like"
            | "just"
            | "over"
            | "such"
            | "take"
            | "also"
            | "into"
            | "only"
            | "very"
            | "more"
            | "does"
            | "most"
            | "about"
            | "which"
            | "their"
            | "would"
            | "there"
            | "could"
            | "other"
            | "after"
            | "being"
            | "where"
            | "these"
            | "those"
            | "first"
            | "should"
            | "because"
            | "through"
            | "before"
            | "par"
            | "aux"
            | "che"
            | "del"
    )
}

#[derive(Deserialize, Default, utoipa::IntoParams)]
#[into_params(parameter_in = Query)]
pub struct RelationsParams {
    /// Filter by label source (future use).
    #[serde(default)]
    #[allow(dead_code)]
    pub source: Option<String>,
}

fn list_relations(model: &LoadedModel) -> Result<serde_json::Value, ServerError> {
    let start = std::time::Instant::now();

    let patched = model.patched.blocking_read();
    let all_layers = patched.loaded_layers();

    // Scan knowledge band layers (14-27 for Gemma, or use config).
    let bands = crate::band_utils::get_layer_bands(model);

    let scan_layers: Vec<usize> = all_layers
        .iter()
        .copied()
        .filter(|l| *l >= bands.knowledge.0 && *l <= bands.knowledge.1)
        .collect();

    struct TokenInfo {
        count: usize,
        max_score: f32,
        min_layer: usize,
        max_layer: usize,
        original: String,
        examples: Vec<String>,
    }

    let mut tokens: HashMap<String, TokenInfo> = HashMap::new();

    for &layer in &scan_layers {
        let num_features = patched.num_features(layer);
        for feat_idx in 0..num_features {
            if let Some(meta) = patched.feature_meta(layer, feat_idx) {
                let tok = meta.top_token.trim();
                if !is_content_token(tok) {
                    continue;
                }
                if meta.c_score < 0.2 {
                    continue;
                }
                let key = tok.to_lowercase();
                let examples: Vec<String> = meta
                    .top_k
                    .iter()
                    .filter(|t| t.token.trim() != tok && is_content_token(t.token.trim()))
                    .take(3)
                    .map(|t| t.token.trim().to_string())
                    .collect();
                let entry = tokens.entry(key).or_insert(TokenInfo {
                    count: 0,
                    max_score: 0.0,
                    min_layer: layer,
                    max_layer: layer,
                    original: tok.to_string(),
                    examples,
                });
                entry.count += 1;
                if meta.c_score > entry.max_score {
                    entry.max_score = meta.c_score;
                }
                if layer < entry.min_layer {
                    entry.min_layer = layer;
                }
                if layer > entry.max_layer {
                    entry.max_layer = layer;
                }
            }
        }
    }

    let mut sorted: Vec<&TokenInfo> = tokens.values().collect();
    sorted.sort_by(|a, b| b.count.cmp(&a.count));
    sorted.truncate(50);

    let relations: Vec<serde_json::Value> = sorted
        .iter()
        .map(|info| {
            serde_json::json!({
                "name": info.original,
                "count": info.count,
                "max_score": info.max_score,
                "min_layer": info.min_layer,
                "max_layer": info.max_layer,
                "examples": info.examples,
            })
        })
        .collect();

    // Probe-confirmed relation labels (from feature_labels.json)
    let mut probe_relations: HashMap<String, usize> = HashMap::new();
    for label in model.probe_labels.values() {
        *probe_relations.entry(label.clone()).or_insert(0) += 1;
    }
    let mut probe_sorted: Vec<(&String, &usize)> = probe_relations.iter().collect();
    probe_sorted.sort_by(|a, b| b.1.cmp(a.1));
    let probe_list: Vec<serde_json::Value> = probe_sorted
        .iter()
        .map(|(name, count)| serde_json::json!({"name": name, "count": count}))
        .collect();

    Ok(serde_json::json!({
        "relations": relations,
        "probe_relations": probe_list,
        "probe_count": model.probe_labels.len(),
        "total": tokens.len(),
        "latency_ms": elapsed_ms(start),
    }))
}

#[utoipa::path(
    get,
    path = "/v1/relations",
    tag = "browse",
    params(RelationsParams),
    responses(
        (status = 200, description = "Known relation types across the knowledge band", body = crate::openapi::schemas::RelationsResponse),
        (status = 500, body = crate::error::ErrorBody),
    ),
)]
pub async fn handle_relations(
    State(state): State<Arc<AppState>>,
    Query(_params): Query<RelationsParams>,
) -> Result<Json<serde_json::Value>, ServerError> {
    state.bump_requests();
    let model = state.model_or_err(None)?.clone();
    let result = tokio::task::spawn_blocking(move || list_relations(&model))
        .await
        .map_err(|e| ServerError::Internal(e.to_string()))??;
    Ok(Json(result))
}

#[utoipa::path(
    get,
    path = "/v1/{model_id}/relations",
    tag = "browse",
    params(
        ("model_id" = String, Path, description = "Id of a loaded vindex."),
        RelationsParams,
    ),
    responses(
        (status = 200, body = crate::openapi::schemas::RelationsResponse),
        (status = 404, body = crate::error::ErrorBody),
        (status = 500, body = crate::error::ErrorBody),
    ),
)]
pub async fn handle_relations_multi(
    State(state): State<Arc<AppState>>,
    Path(model_id): Path<String>,
    Query(_params): Query<RelationsParams>,
) -> Result<Json<serde_json::Value>, ServerError> {
    state.bump_requests();
    let model = state.model_or_err(Some(&model_id))?.clone();
    let result = tokio::task::spawn_blocking(move || list_relations(&model))
        .await
        .map_err(|e| ServerError::Internal(e.to_string()))??;
    Ok(Json(result))
}
