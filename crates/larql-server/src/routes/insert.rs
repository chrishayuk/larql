//! POST /v1/insert — constellation knowledge insertion.
//!
//! Full trace-guided multi-layer insert: forward pass to capture residuals,
//! use as gate vectors, write down vector overrides with target embedding.
//! Supports session isolation via X-Session-Id header.

use std::sync::Arc;

use axum::extract::{Path, State};
use axum::http::HeaderMap;
use axum::Json;
use serde::Deserialize;

use crate::band_utils::{get_layer_bands, INSERT_MODE_CONSTELLATION, INSERT_MODE_EMBEDDING};
use crate::error::ServerError;
use crate::session::extract_session_id;
use crate::state::{elapsed_ms, AppState, LoadedModel};

#[derive(Clone, Debug, Deserialize)]
pub struct InsertRequest {
    pub entity: String,
    pub relation: String,
    pub target: String,
    #[serde(default)]
    pub layer: Option<usize>,
    #[serde(default = "default_alpha")]
    pub alpha: f32,
    #[serde(default = "default_confidence")]
    pub confidence: f32,
}

#[derive(Clone, Debug, Deserialize)]
pub struct MemoryUpdateBatchRequest {
    /// Optional producer label, e.g. `ferrosa-memory`.
    #[serde(default)]
    pub source: Option<String>,
    /// Knowledge updates to apply as INSERT patch operations.
    pub updates: Vec<InsertRequest>,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn memory_update_batch_deserializes_ferrosa_fact_payload() {
        let req: MemoryUpdateBatchRequest = serde_json::from_value(serde_json::json!({
            "source": "ferrosa-memory",
            "updates": [{
                "entity": "Kimi K2",
                "relation": "learned_fact",
                "target": "vindex patches can represent long term memory",
                "confidence": 0.97,
                "alpha": 0.2
            }]
        }))
        .unwrap();

        assert_eq!(req.source.as_deref(), Some("ferrosa-memory"));
        assert_eq!(req.updates.len(), 1);
        assert_eq!(req.updates[0].entity, "Kimi K2");
        assert_eq!(req.updates[0].relation, "learned_fact");
        assert_eq!(
            req.updates[0].target,
            "vindex patches can represent long term memory"
        );
    }
}

fn default_alpha() -> f32 {
    0.25
}
fn default_confidence() -> f32 {
    0.9
}

/// Compute insert layers and residuals from a forward pass.
/// Needs only read access to the patched vindex.
fn compute_residuals(
    model: &LoadedModel,
    patched: &larql_vindex::PatchedVindex,
    req: &InsertRequest,
    insert_layers: &[usize],
) -> Vec<(usize, Vec<f32>)> {
    if !model.config.has_model_weights {
        return Vec::new();
    }

    let weights_guard = match model.get_or_load_weights() {
        Ok(w) => w,
        Err(_) => return Vec::new(),
    };
    let weights: &larql_inference::ModelWeights = &weights_guard;

    let prompt = format!(
        "The {} of {} is",
        req.relation.replace(['-', '_'], " "),
        req.entity
    );
    let encoding = match model.tokenizer.encode(prompt.as_str(), true) {
        Ok(e) => e,
        Err(_) => return Vec::new(),
    };
    let token_ids: Vec<u32> = encoding.get_ids().to_vec();

    let walk_ffn = larql_inference::vindex::WalkFfn::new_unlimited_with_trace(weights, patched);
    let _result =
        larql_inference::predict_with_ffn(weights, &model.tokenizer, &token_ids, 1, &walk_ffn);

    walk_ffn
        .take_residuals()
        .into_iter()
        .filter(|(layer, _)| insert_layers.contains(layer))
        .collect()
}

/// Insert features into a PatchedVindex. Mutates patched in place.
fn apply_insert(
    model: &LoadedModel,
    patched: &mut larql_vindex::PatchedVindex,
    req: &InsertRequest,
    insert_layers: &[usize],
    residuals: &[(usize, Vec<f32>)],
) -> (usize, bool) {
    let hidden = model.embeddings.shape()[1];
    let mut inserted = 0usize;

    // Target embedding for down vector
    let target_encoding = match model.tokenizer.encode(req.target.as_str(), false) {
        Ok(e) => e,
        Err(_) => return (0, false),
    };
    let target_ids: Vec<u32> = target_encoding.get_ids().to_vec();
    let target_id = target_ids.first().copied().unwrap_or(0);

    let mut target_embed = vec![0.0f32; hidden];
    for &tok in &target_ids {
        let row = model.embeddings.row(tok as usize);
        for j in 0..hidden {
            target_embed[j] += row[j] * model.embed_scale;
        }
    }
    let n = target_ids.len().max(1) as f32;
    for v in &mut target_embed {
        *v /= n;
    }

    let use_constellation = !residuals.is_empty();

    for &layer in insert_layers {
        let feature = match patched.find_free_feature(layer) {
            Some(f) => f,
            None => continue,
        };

        // Gate vector: residual (constellation) or entity embedding (fallback)
        let gate_vec: Vec<f32> =
            if let Some((_, ref residual)) = residuals.iter().find(|(l, _)| *l == layer) {
                let mut gv = residual.clone();
                if let Some(gate_matrix) = patched.base().gate_vectors_at(layer) {
                    let sample = gate_matrix.nrows().min(100);
                    if sample > 0 {
                        let avg_norm: f32 = (0..sample)
                            .map(|i| gate_matrix.row(i).dot(&gate_matrix.row(i)).sqrt())
                            .sum::<f32>()
                            / sample as f32;
                        let res_norm: f32 = gv.iter().map(|v| v * v).sum::<f32>().sqrt();
                        if res_norm > 1e-8 && avg_norm > 0.0 {
                            let scale = avg_norm / res_norm;
                            for v in &mut gv {
                                *v *= scale;
                            }
                        }
                    }
                }
                gv
            } else {
                let enc = match model.tokenizer.encode(req.entity.as_str(), false) {
                    Ok(e) => e,
                    Err(_) => continue,
                };
                let ids = enc.get_ids();
                let mut ev = vec![0.0f32; hidden];
                for &tok in ids {
                    let row = model.embeddings.row(tok as usize);
                    for j in 0..hidden {
                        ev[j] += row[j] * model.embed_scale;
                    }
                }
                let n = ids.len().max(1) as f32;
                for v in &mut ev {
                    *v /= n;
                }
                let norm: f32 = ev.iter().map(|v| v * v).sum::<f32>().sqrt();
                if norm > 1e-8 {
                    for v in &mut ev {
                        *v /= norm;
                    }
                }
                ev
            };

        let down_vec: Vec<f32> = target_embed.iter().map(|v| v * req.alpha).collect();

        let meta = larql_vindex::FeatureMeta {
            top_token: req.target.clone(),
            top_token_id: target_id,
            c_score: req.confidence,
            top_k: vec![larql_models::TopKEntry {
                token: req.target.clone(),
                token_id: target_id,
                logit: req.confidence,
            }],
        };

        patched.insert_feature(layer, feature, gate_vec.clone(), meta);
        patched.set_up_vector(layer, feature, gate_vec);
        patched.set_down_vector(layer, feature, down_vec);
        inserted += 1;
    }

    (inserted, use_constellation)
}

fn run_insert(
    state: &AppState,
    model: &LoadedModel,
    req: &InsertRequest,
    session_id: Option<&str>,
) -> Result<serde_json::Value, ServerError> {
    let start = std::time::Instant::now();

    // Determine insert layers
    let bands = get_layer_bands(model);

    let insert_layers: Vec<usize> = if let Some(l) = req.layer {
        vec![l]
    } else {
        let mid = (bands.knowledge.0 + bands.knowledge.1) / 2;
        (mid..=bands.knowledge.1).collect()
    };

    let (inserted, use_constellation) = if let Some(sid) = session_id {
        // Session-scoped: read from session for residuals, write to session for insert
        let mut sessions = state.sessions.sessions_blocking_write();
        let now = std::time::Instant::now();

        let session = sessions.entry(sid.to_string()).or_insert_with(|| {
            let base = model.patched.blocking_read();
            crate::session::SessionState::new(base.base().clone(), now)
        });
        session.touch(now);

        let residuals = compute_residuals(model, &session.patched, req, &insert_layers);
        apply_insert(model, &mut session.patched, req, &insert_layers, &residuals)
    } else {
        // Global: read from global for residuals, write to global for insert
        let residuals = {
            let patched = model.patched.blocking_read();
            compute_residuals(model, &patched, req, &insert_layers)
        };

        let mut patched = model.patched.blocking_write();
        apply_insert(model, &mut patched, req, &insert_layers, &residuals)
    };

    Ok(serde_json::json!({
        "entity": req.entity,
        "relation": req.relation,
        "target": req.target,
        "inserted": inserted,
        "mode": if use_constellation { INSERT_MODE_CONSTELLATION } else { INSERT_MODE_EMBEDDING },
        "alpha": req.alpha,
        "session": session_id,
        "latency_ms": elapsed_ms(start),
    }))
}

fn run_memory_update_batch(
    state: &AppState,
    model: &LoadedModel,
    req: &MemoryUpdateBatchRequest,
    session_id: Option<&str>,
) -> Result<serde_json::Value, ServerError> {
    let start = std::time::Instant::now();
    let mut results = Vec::with_capacity(req.updates.len());
    let mut inserted_total = 0usize;

    for update in &req.updates {
        let result = run_insert(state, model, update, session_id)?;
        inserted_total += result
            .get("inserted")
            .and_then(serde_json::Value::as_u64)
            .unwrap_or(0) as usize;
        results.push(result);
    }

    Ok(serde_json::json!({
        "source": req.source,
        "updates": req.updates.len(),
        "inserted": inserted_total,
        "results": results,
        "session": session_id,
        "latency_ms": elapsed_ms(start),
    }))
}

pub async fn handle_insert(
    State(state): State<Arc<AppState>>,
    headers: HeaderMap,
    Json(req): Json<InsertRequest>,
) -> Result<Json<serde_json::Value>, ServerError> {
    state.bump_requests();
    let model = Arc::clone(state.model_or_err(None)?);
    let sid = extract_session_id(&headers);
    let state2 = Arc::clone(&state);
    let result =
        tokio::task::spawn_blocking(move || run_insert(&state2, &model, &req, sid.as_deref()))
            .await
            .map_err(|e| ServerError::Internal(e.to_string()))??;
    Ok(Json(result))
}

pub async fn handle_insert_multi(
    State(state): State<Arc<AppState>>,
    Path(model_id): Path<String>,
    headers: HeaderMap,
    Json(req): Json<InsertRequest>,
) -> Result<Json<serde_json::Value>, ServerError> {
    state.bump_requests();
    let model = Arc::clone(state.model_or_err(Some(&model_id))?);
    let sid = extract_session_id(&headers);
    let state2 = Arc::clone(&state);
    let result =
        tokio::task::spawn_blocking(move || run_insert(&state2, &model, &req, sid.as_deref()))
            .await
            .map_err(|e| ServerError::Internal(e.to_string()))??;
    Ok(Json(result))
}

pub async fn handle_memory_update_batch(
    State(state): State<Arc<AppState>>,
    headers: HeaderMap,
    Json(req): Json<MemoryUpdateBatchRequest>,
) -> Result<Json<serde_json::Value>, ServerError> {
    state.bump_requests();
    let model = Arc::clone(state.model_or_err(None)?);
    let sid = extract_session_id(&headers);
    let state2 = Arc::clone(&state);
    let result = tokio::task::spawn_blocking(move || {
        run_memory_update_batch(&state2, &model, &req, sid.as_deref())
    })
    .await
    .map_err(|e| ServerError::Internal(e.to_string()))??;
    Ok(Json(result))
}

pub async fn handle_memory_update_batch_multi(
    State(state): State<Arc<AppState>>,
    Path(model_id): Path<String>,
    headers: HeaderMap,
    Json(req): Json<MemoryUpdateBatchRequest>,
) -> Result<Json<serde_json::Value>, ServerError> {
    state.bump_requests();
    let model = Arc::clone(state.model_or_err(Some(&model_id))?);
    let sid = extract_session_id(&headers);
    let state2 = Arc::clone(&state);
    let result = tokio::task::spawn_blocking(move || {
        run_memory_update_batch(&state2, &model, &req, sid.as_deref())
    })
    .await
    .map_err(|e| ServerError::Internal(e.to_string()))??;
    Ok(Json(result))
}
