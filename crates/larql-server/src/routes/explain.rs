//! POST /v1/explain-infer — walk inference with per-layer feature trace.

use std::sync::Arc;

use axum::extract::{Path, State};
use axum::Json;
use serde::Deserialize;

use crate::band_utils::{get_layer_bands, BAND_KNOWLEDGE, BAND_OUTPUT, BAND_SYNTAX};
use crate::error::ServerError;
use crate::state::{elapsed_ms, AppState, LoadedModel};

#[derive(Deserialize, utoipa::ToSchema)]
pub struct ExplainRequest {
    pub prompt: String,
    #[serde(default = "default_top")]
    pub top: usize,
    #[serde(default = "default_per_layer")]
    pub per_layer: usize,
    #[serde(default = "default_band")]
    pub band: String,
    #[serde(default)]
    pub relations_only: bool,
    #[serde(default)]
    pub with_attention: bool,
}

fn default_top() -> usize {
    5
}
fn default_per_layer() -> usize {
    3
}
fn default_band() -> String {
    crate::band_utils::BAND_ALL.into()
}

fn round_probability(prob: f64) -> f64 {
    (prob * 10000.0).round() / 10000.0
}

fn round_gate_score(score: f32) -> f64 {
    ((score as f64) * 10.0).round() / 10.0
}

fn round_attention_weight(weight: f32) -> f64 {
    ((weight as f64) * 1000.0).round() / 1000.0
}

fn layer_range_for_band(bands: &larql_vindex::LayerBands, band: &str) -> Option<(usize, usize)> {
    match band {
        BAND_SYNTAX => Some(bands.syntax),
        BAND_KNOWLEDGE => Some(bands.knowledge),
        BAND_OUTPUT => Some(bands.output),
        _ => None,
    }
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

fn format_attention(attn: &[(String, f32)]) -> Vec<serde_json::Value> {
    attn.iter()
        .map(|(tok, weight)| {
            serde_json::json!({
                "token": tok,
                "weight": round_attention_weight(*weight),
            })
        })
        .collect()
}

fn format_lens(token: &str, probability: f64) -> serde_json::Value {
    serde_json::json!({
        "token": token,
        "probability": round_probability(probability),
    })
}

fn explain_infer(
    model: &LoadedModel,
    req: &ExplainRequest,
) -> Result<serde_json::Value, ServerError> {
    let start = std::time::Instant::now();

    let weights_guard = model
        .get_or_load_weights()
        .map_err(ServerError::InferenceUnavailable)?;
    let weights: &larql_inference::ModelWeights = &weights_guard;
    let encoding = model
        .tokenizer
        .encode(req.prompt.as_str(), true)
        .map_err(|e| ServerError::Internal(format!("tokenize: {e}")))?;
    let token_ids: Vec<u32> = encoding.get_ids().to_vec();

    // Decode tokens for attention display (None for special tokens like BOS/EOS)
    let token_strs: Vec<Option<String>> = if req.with_attention {
        token_ids
            .iter()
            .map(|&id| larql_inference::decode_token(&model.tokenizer, id))
            .collect()
    } else {
        Vec::new()
    };

    let patched = model.patched.blocking_read();
    let walk_ffn = larql_inference::vindex::WalkFfn::new_unlimited_with_trace(weights, &*patched);

    let (predictions_raw, attention_captures, lens_residuals) = if req.with_attention {
        let r = larql_inference::predict_with_ffn_attention(
            weights,
            &model.tokenizer,
            &token_ids,
            req.top,
            &walk_ffn,
        );
        (r.predictions, r.attention, r.residuals)
    } else {
        let r = larql_inference::predict_with_ffn(
            weights,
            &model.tokenizer,
            &token_ids,
            req.top,
            &walk_ffn,
        );
        (r.predictions, Vec::new(), Vec::new())
    };
    let residuals = walk_ffn.take_residuals();
    let model_top1 = predictions_raw.first().cloned();
    let (predictions_raw, knn_override) = larql_inference::apply_knn_override(
        predictions_raw,
        &residuals,
        Some(&patched.knn_store),
        req.top,
    );
    let trace_layers = larql_inference::walk_trace_from_residuals(&residuals, &patched);

    // Build logit lens: layer → (top_token, probability)
    let lens_map: std::collections::HashMap<usize, (String, f64)> = lens_residuals
        .iter()
        .filter_map(|(layer, residual)| {
            let pred = larql_inference::logit_lens_top1(weights, &model.tokenizer, residual)?;
            Some((*layer, pred))
        })
        .collect();

    // Build attention lookup: layer → top attended tokens
    let attention_map: std::collections::HashMap<usize, Vec<(String, f32)>> = {
        let mut map = std::collections::HashMap::new();
        for cap in &attention_captures {
            let n_heads = cap.weights.heads.len();
            if n_heads == 0 || token_strs.is_empty() {
                continue;
            }
            let seq_len = cap.weights.heads[0].len();
            let mut avg = vec![0.0f32; seq_len];
            for head in &cap.weights.heads {
                for (j, &w) in head.iter().enumerate() {
                    avg[j] += w;
                }
            }
            for v in avg.iter_mut() {
                *v /= n_heads as f32;
            }
            let mut pairs: Vec<(String, f32)> = avg
                .iter()
                .copied()
                .enumerate()
                .filter_map(|(j, w)| {
                    let tok = token_strs.get(j)?.as_ref()?;
                    Some((tok.trim().to_string(), w))
                })
                .collect();
            pairs.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
            pairs.truncate(3);
            map.insert(cap.layer, pairs);
        }
        map
    };

    let bands = get_layer_bands(model);
    let layer_range = layer_range_for_band(&bands, &req.band);

    let predictions = format_predictions(&predictions_raw);

    let mut layers = Vec::new();
    for (layer, hits) in &trace_layers {
        if let Some((lo, hi)) = layer_range {
            if *layer < lo || *layer > hi {
                continue;
            }
        }
        // When relations_only, re-sort so positive gates rank first
        let ordered_hits: Vec<_> = if req.relations_only {
            let mut lh: Vec<_> = hits
                .iter()
                .filter(|hit| model.probe_labels.contains_key(&(*layer, hit.feature)))
                .collect();
            lh.sort_by(|a, b| {
                let a_pos = a.gate_score > 0.0;
                let b_pos = b.gate_score > 0.0;
                match (a_pos, b_pos) {
                    (true, false) => std::cmp::Ordering::Less,
                    (false, true) => std::cmp::Ordering::Greater,
                    _ => b
                        .gate_score
                        .abs()
                        .partial_cmp(&a.gate_score.abs())
                        .unwrap_or(std::cmp::Ordering::Equal),
                }
            });
            lh
        } else {
            hits.iter().collect()
        };

        let features: Vec<serde_json::Value> = ordered_hits
            .iter()
            .filter_map(|hit| {
                let relation = model.probe_labels.get(&(*layer, hit.feature)).cloned();
                if req.relations_only && relation.is_none() {
                    return None;
                }
                let top_tokens: Vec<String> = hit
                    .meta
                    .top_k
                    .iter()
                    .take(3)
                    .map(|t| t.token.trim().to_string())
                    .collect();
                Some(serde_json::json!({
                    "feature": hit.feature,
                    "gate_score": round_gate_score(hit.gate_score),
                    "top_token": hit.meta.top_token.trim(),
                    "top_tokens": top_tokens,
                    "relation": relation,
                }))
            })
            .take(req.per_layer)
            .collect();
        if !features.is_empty() {
            let mut layer_obj = serde_json::json!({
                "layer": layer,
                "features": features,
            });
            if let Some(attn) = attention_map.get(layer) {
                layer_obj["attention"] = serde_json::json!(format_attention(attn));
            }
            if let Some((tok, prob)) = lens_map.get(layer) {
                layer_obj["lens"] = format_lens(tok, *prob);
            }
            layers.push(layer_obj);
        }
    }

    let mut body = serde_json::json!({
        "prompt": req.prompt,
        "predictions": predictions,
        "trace": layers,
        "latency_ms": elapsed_ms(start),
    });
    if let Some(ovr) = knn_override {
        body["knn_override"] = serde_json::json!({
            "token": ovr.token,
            "cosine": ovr.cosine,
            "layer": ovr.layer,
            "source": "knn_override",
            "stage": "post_logits",
            "materialized": false,
        });
        if let Some((tok, prob)) = model_top1 {
            body["knn_override"]["model_top1"] = serde_json::json!({
                "token": tok,
                "probability": round_probability(prob),
            });
        }
    }
    Ok(body)
}

#[utoipa::path(
    post,
    path = "/v1/explain-infer",
    tag = "inference",
    request_body = ExplainRequest,
    responses(
        (status = 200, description = "Predictions with per-layer feature traces", body = crate::openapi::schemas::ExplainResponse),
        (status = 400, body = crate::error::ErrorBody),
        (status = 500, body = crate::error::ErrorBody),
    ),
)]
pub async fn handle_explain(
    State(state): State<Arc<AppState>>,
    Json(req): Json<ExplainRequest>,
) -> Result<Json<serde_json::Value>, ServerError> {
    state.bump_requests();
    let model = state.model_or_err(None)?.clone();
    let result = tokio::task::spawn_blocking(move || explain_infer(&model, &req))
        .await
        .map_err(|e| ServerError::Internal(e.to_string()))??;
    Ok(Json(result))
}

#[utoipa::path(
    post,
    path = "/v1/{model_id}/explain-infer",
    tag = "inference",
    params(("model_id" = String, Path, description = "Id of a loaded vindex.")),
    request_body = ExplainRequest,
    responses(
        (status = 200, body = crate::openapi::schemas::ExplainResponse),
        (status = 400, body = crate::error::ErrorBody),
        (status = 404, body = crate::error::ErrorBody),
    ),
)]
pub async fn handle_explain_multi(
    State(state): State<Arc<AppState>>,
    Path(model_id): Path<String>,
    Json(req): Json<ExplainRequest>,
) -> Result<Json<serde_json::Value>, ServerError> {
    state.bump_requests();
    let model = state.model_or_err(Some(&model_id))?.clone();
    let result = tokio::task::spawn_blocking(move || explain_infer(&model, &req))
        .await
        .map_err(|e| ServerError::Internal(e.to_string()))??;
    Ok(Json(result))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn explain_defaults_match_api_contract() {
        assert_eq!(default_top(), 5);
        assert_eq!(default_per_layer(), 3);
        assert_eq!(default_band(), crate::band_utils::BAND_ALL);
    }

    #[test]
    fn explain_request_deserializes_optional_fields() {
        let req: ExplainRequest = serde_json::from_value(serde_json::json!({
            "prompt": "The capital of France is"
        }))
        .unwrap();
        assert_eq!(req.prompt, "The capital of France is");
        assert_eq!(req.top, 5);
        assert_eq!(req.per_layer, 3);
        assert_eq!(req.band, crate::band_utils::BAND_ALL);
        assert!(!req.relations_only);
        assert!(!req.with_attention);
    }

    #[test]
    fn explain_request_accepts_explicit_options() {
        let req: ExplainRequest = serde_json::from_value(serde_json::json!({
            "prompt": "x",
            "top": 2,
            "per_layer": 4,
            "band": "knowledge",
            "relations_only": true,
            "with_attention": true
        }))
        .unwrap();
        assert_eq!(req.top, 2);
        assert_eq!(req.per_layer, 4);
        assert_eq!(req.band, BAND_KNOWLEDGE);
        assert!(req.relations_only);
        assert!(req.with_attention);
    }

    #[test]
    fn layer_range_for_band_maps_named_bands() {
        let bands = larql_vindex::LayerBands {
            syntax: (0, 2),
            knowledge: (3, 7),
            output: (8, 9),
        };
        assert_eq!(layer_range_for_band(&bands, BAND_SYNTAX), Some((0, 2)));
        assert_eq!(layer_range_for_band(&bands, BAND_KNOWLEDGE), Some((3, 7)));
        assert_eq!(layer_range_for_band(&bands, BAND_OUTPUT), Some((8, 9)));
        assert_eq!(
            layer_range_for_band(&bands, crate::band_utils::BAND_ALL),
            None
        );
        assert_eq!(layer_range_for_band(&bands, "unknown"), None);
    }

    #[test]
    fn format_predictions_rounds_probability() {
        let predictions = format_predictions(&[("Paris".into(), 0.123456)]);
        assert_eq!(predictions[0]["token"], "Paris");
        assert_eq!(predictions[0]["probability"], 0.1235);
    }

    #[test]
    fn format_attention_rounds_weight() {
        let attention = format_attention(&[("France".into(), 0.12356)]);
        assert_eq!(attention[0]["token"], "France");
        assert_eq!(attention[0]["weight"], 0.124);
    }

    #[test]
    fn format_lens_rounds_probability() {
        let lens = format_lens("Paris", 0.987654);
        assert_eq!(lens["token"], "Paris");
        assert_eq!(lens["probability"], 0.9877);
    }

    #[test]
    fn score_rounding_matches_response_contract() {
        assert_eq!(round_gate_score(12.34), 12.3);
        assert_eq!(round_attention_weight(0.3336), 0.334);
    }
}
