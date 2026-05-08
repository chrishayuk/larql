//! POST /v1/infer — full forward pass with attention.

use std::sync::Arc;

use axum::extract::{Path, State};
use axum::http::HeaderMap;
use axum::Json;
use serde::Deserialize;

use crate::band_utils::{INFER_MODE_COMPARE, INFER_MODE_DENSE, INFER_MODE_WALK};
use crate::error::ServerError;
use crate::session::extract_session_id;
use crate::state::{elapsed_ms, AppState, LoadedModel};

#[derive(Deserialize, utoipa::ToSchema)]
pub struct InferRequest {
    /// Prompt to run inference on.
    pub prompt: String,
    /// Top-K next-token predictions to return.
    #[serde(default = "default_top")]
    pub top: usize,
    /// Inference mode: `walk` (default), `dense`, or `compare`.
    #[serde(default = "default_mode")]
    pub mode: String,
}

fn default_top() -> usize {
    5
}
fn default_mode() -> String {
    INFER_MODE_WALK.into()
}

fn round_probability(prob: f64) -> f64 {
    (prob * 10000.0).round() / 10000.0
}

fn format_predictions(predictions: &[(String, f64)]) -> Vec<serde_json::Value> {
    predictions
        .iter()
        .map(|(tok, prob)| {
            serde_json::json!({
                "token": tok,
                "probability": round_probability(*prob),
            })
        })
        .collect()
}

fn format_knn_override(
    ovr: &larql_inference::KnnOverride,
    model_top1: Option<&(String, f64)>,
) -> serde_json::Value {
    let mut value = serde_json::json!({
        "token": &ovr.token,
        "cosine": ovr.cosine,
        "layer": ovr.layer,
        "source": "knn_override",
        "stage": "post_logits",
        "materialized": false,
    });
    if let Some((tok, prob)) = model_top1 {
        value["model_top1"] = serde_json::json!({
            "token": tok,
            "probability": round_probability(*prob),
        });
    }
    value
}

fn infer_mode_flags(mode: &str) -> (bool, bool, bool) {
    let is_compare = mode == INFER_MODE_COMPARE;
    let use_walk = mode == INFER_MODE_WALK || is_compare;
    let use_dense = mode == INFER_MODE_DENSE || is_compare;
    (is_compare, use_walk, use_dense)
}

fn run_infer(
    state: &AppState,
    model: &LoadedModel,
    req: &InferRequest,
    session_id: Option<&str>,
) -> Result<serde_json::Value, ServerError> {
    if model.infer_disabled {
        return Err(ServerError::InferenceUnavailable(
            "inference disabled (--no-infer)".into(),
        ));
    }

    if !model.config.has_model_weights
        && model.config.extract_level != larql_vindex::ExtractLevel::Inference
        && model.config.extract_level != larql_vindex::ExtractLevel::All
    {
        return Err(ServerError::InferenceUnavailable(
            "vindex does not contain model weights. Rebuild with --include-weights".into(),
        ));
    }

    let weights_guard = model
        .get_or_load_weights()
        .map_err(ServerError::InferenceUnavailable)?;
    let weights: &larql_inference::ModelWeights = &weights_guard;

    let encoding = model
        .tokenizer
        .encode(req.prompt.as_str(), true)
        .map_err(|e| ServerError::Internal(format!("tokenize error: {e}")))?;
    let token_ids: Vec<u32> = encoding.get_ids().to_vec();

    if token_ids.is_empty() {
        return Err(ServerError::BadRequest("empty prompt".into()));
    }

    let start = std::time::Instant::now();

    let (is_compare, use_walk, use_dense) = infer_mode_flags(&req.mode);

    let mut result = serde_json::Map::new();
    result.insert("prompt".into(), serde_json::json!(req.prompt));

    // Helper: run walk inference against a PatchedVindex.
    let run_walk = |patched: &larql_vindex::PatchedVindex| {
        larql_inference::infer_patched(
            weights,
            &model.tokenizer,
            patched,
            Some(&patched.knn_store),
            &token_ids,
            req.top,
        )
    };

    if use_walk {
        let pred = if let Some(sid) = session_id {
            // Session-scoped: use session's PatchedVindex
            let sessions = state.sessions.sessions_blocking_write();
            if let Some(session) = sessions.get(sid) {
                run_walk(&session.patched)
            } else {
                drop(sessions);
                let patched = model.patched.blocking_read();
                run_walk(&patched)
            }
        } else {
            let patched = model.patched.blocking_read();
            run_walk(&patched)
        };

        let predictions = format_predictions(&pred.predictions);
        if let Some(ovr) = &pred.knn_override {
            result.insert(
                "knn_override".into(),
                format_knn_override(ovr, pred.model_top1.as_ref()),
            );
        }

        if is_compare {
            result.insert(INFER_MODE_WALK.into(), serde_json::json!(predictions));
            result.insert(
                "walk_ms".into(),
                serde_json::json!((pred.walk_ms * 10.0).round() / 10.0),
            );
        } else {
            result.insert("predictions".into(), serde_json::json!(predictions));
            result.insert("mode".into(), serde_json::json!(INFER_MODE_WALK));
        }
    }

    if use_dense {
        let dense_start = std::time::Instant::now();
        let pred = larql_inference::predict(weights, &model.tokenizer, &token_ids, req.top);
        let dense_ms = dense_start.elapsed().as_secs_f64() * 1000.0;

        let predictions = format_predictions(&pred.predictions);

        if is_compare {
            result.insert(INFER_MODE_DENSE.into(), serde_json::json!(predictions));
            result.insert(
                "dense_ms".into(),
                serde_json::json!((dense_ms * 10.0).round() / 10.0),
            );
        } else {
            result.insert("predictions".into(), serde_json::json!(predictions));
            result.insert("mode".into(), serde_json::json!(INFER_MODE_DENSE));
        }
    }

    result.insert("latency_ms".into(), serde_json::json!(elapsed_ms(start)));

    Ok(serde_json::Value::Object(result))
}

#[utoipa::path(
    post,
    path = "/v1/infer",
    tag = "inference",
    request_body = InferRequest,
    responses(
        (status = 200, description = "Next-token predictions", body = crate::openapi::schemas::InferResponse),
        (status = 400, body = crate::error::ErrorBody),
        (status = 503, body = crate::error::ErrorBody, description = "Inference weights unavailable"),
        (status = 500, body = crate::error::ErrorBody),
    ),
)]
pub async fn handle_infer(
    State(state): State<Arc<AppState>>,
    headers: HeaderMap,
    Json(req): Json<InferRequest>,
) -> Result<Json<serde_json::Value>, ServerError> {
    state.bump_requests();
    let model = state.model_or_err(None)?.clone();
    let sid = extract_session_id(&headers);
    let state2 = Arc::clone(&state);
    let result =
        tokio::task::spawn_blocking(move || run_infer(&state2, &model, &req, sid.as_deref()))
            .await
            .map_err(|e| ServerError::Internal(e.to_string()))??;
    Ok(Json(result))
}

#[utoipa::path(
    post,
    path = "/v1/{model_id}/infer",
    tag = "inference",
    params(("model_id" = String, Path, description = "Id of a loaded vindex.")),
    request_body = InferRequest,
    responses(
        (status = 200, body = crate::openapi::schemas::InferResponse),
        (status = 400, body = crate::error::ErrorBody),
        (status = 404, body = crate::error::ErrorBody),
        (status = 503, body = crate::error::ErrorBody),
    ),
)]
pub async fn handle_infer_multi(
    State(state): State<Arc<AppState>>,
    Path(model_id): Path<String>,
    headers: HeaderMap,
    Json(req): Json<InferRequest>,
) -> Result<Json<serde_json::Value>, ServerError> {
    state.bump_requests();
    let model = state.model_or_err(Some(&model_id))?.clone();
    let sid = extract_session_id(&headers);
    let state2 = Arc::clone(&state);
    let result =
        tokio::task::spawn_blocking(move || run_infer(&state2, &model, &req, sid.as_deref()))
            .await
            .map_err(|e| ServerError::Internal(e.to_string()))??;
    Ok(Json(result))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn infer_defaults_match_api_contract() {
        assert_eq!(default_top(), 5);
        assert_eq!(default_mode(), INFER_MODE_WALK);
    }

    #[test]
    fn infer_request_deserializes_defaults() {
        let req: InferRequest = serde_json::from_value(serde_json::json!({
            "prompt": "The capital of France is"
        }))
        .unwrap();
        assert_eq!(req.prompt, "The capital of France is");
        assert_eq!(req.top, 5);
        assert_eq!(req.mode, INFER_MODE_WALK);
    }

    #[test]
    fn infer_request_accepts_dense_and_compare_modes() {
        let dense: InferRequest = serde_json::from_value(serde_json::json!({
            "prompt": "x",
            "top": 2,
            "mode": "dense"
        }))
        .unwrap();
        assert_eq!(dense.top, 2);
        assert_eq!(dense.mode, INFER_MODE_DENSE);

        let compare: InferRequest = serde_json::from_value(serde_json::json!({
            "prompt": "x",
            "mode": "compare"
        }))
        .unwrap();
        assert_eq!(compare.mode, INFER_MODE_COMPARE);
    }

    #[test]
    fn infer_mode_flags_select_expected_paths() {
        assert_eq!(infer_mode_flags(INFER_MODE_WALK), (false, true, false));
        assert_eq!(infer_mode_flags(INFER_MODE_DENSE), (false, false, true));
        assert_eq!(infer_mode_flags(INFER_MODE_COMPARE), (true, true, true));
        assert_eq!(infer_mode_flags("unknown"), (false, false, false));
    }

    #[test]
    fn format_predictions_rounds_probability() {
        let predictions = format_predictions(&[("Paris".into(), 0.123456)]);
        assert_eq!(predictions[0]["token"], "Paris");
        assert_eq!(predictions[0]["probability"], 0.1235);
    }
}
