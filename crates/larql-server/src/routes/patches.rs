//! POST/GET/DELETE /v1/patches — patch management endpoints.
//!
//! Session-aware: if `X-Session-Id` header is present, patches are scoped
//! to that session. Otherwise, patches go to the global shared state.

use std::sync::Arc;

use axum::extract::{Path, State};
use axum::http::HeaderMap;
use axum::Json;
use serde::Deserialize;

use crate::error::ServerError;
use crate::session::{extract_session_id, PATCH_UNNAMED};
use crate::state::AppState;

const PATCH_INLINE_NAME: &str = "inline-patch";

#[derive(Deserialize)]
pub struct ApplyPatchRequest {
    #[serde(default)]
    pub url: Option<String>,
    #[serde(default)]
    pub patch: Option<larql_vindex::VindexPatch>,
}

/// Resolve a patch from the request body (inline or URL).
fn resolve_patch(
    req: &ApplyPatchRequest,
) -> Result<(larql_vindex::VindexPatch, String), ServerError> {
    if let Some(ref patch) = req.patch {
        let name = req
            .url
            .clone()
            .or_else(|| patch.description.clone())
            .unwrap_or_else(|| PATCH_INLINE_NAME.into());
        return Ok((patch.clone(), name));
    }

    if let Some(ref url) = req.url {
        let path = if larql_vindex::is_hf_path(url) {
            let resolved = larql_vindex::resolve_hf_vindex(url)
                .map_err(|e| ServerError::Internal(format!("failed to resolve HF path: {e}")))?;
            let vlp_path = resolved.join("patch.vlp");
            if vlp_path.exists() {
                vlp_path
            } else {
                return Err(ServerError::BadRequest(format!(
                    "no patch.vlp found at {url}"
                )));
            }
        } else {
            std::path::PathBuf::from(url)
        };
        let patch = larql_vindex::VindexPatch::load(&path)
            .map_err(|e| ServerError::Internal(format!("failed to load patch: {e}")))?;
        return Ok((patch, url.clone()));
    }

    Err(ServerError::BadRequest(
        "must provide 'url' or 'patch' in request body".into(),
    ))
}

/// Synthesise a gate vector from entity embedding when the client didn't provide one.
fn enrich_patch_ops(model: &crate::state::LoadedModel, patch: &mut larql_vindex::VindexPatch) {
    let hidden = model.embeddings.shape()[1];
    for op in &mut patch.operations {
        if let larql_vindex::PatchOp::Insert {
            entity,
            relation,
            feature,
            gate_vector_b64,
            ..
        } = op
        {
            // Synthesise gate vector if missing
            if gate_vector_b64.is_none() {
                let encoding = model.tokenizer.encode(entity.as_str(), false);
                if let Ok(enc) = encoding {
                    let ids = enc.get_ids();
                    if !ids.is_empty() {
                        let mut embed = vec![0.0f32; hidden];
                        for &tok in ids {
                            let row = model.embeddings.row(tok as usize);
                            for j in 0..hidden {
                                embed[j] += row[j] * model.embed_scale;
                            }
                        }
                        let n = ids.len() as f32;
                        for v in &mut embed {
                            *v /= n;
                        }

                        // Normalise the embedding to unit length — gate KNN uses
                        // cosine similarity so magnitude doesn't matter.
                        let embed_norm: f32 = embed.iter().map(|v| v * v).sum::<f32>().sqrt();
                        if embed_norm > 1e-8 {
                            for v in &mut embed {
                                *v /= embed_norm;
                            }
                        }

                        *gate_vector_b64 =
                            Some(larql_vindex::patch::core::encode_gate_vector(&embed));
                    }
                }

                // Assign a feature slot if unset
                if *feature == 0 {
                    // Use a deterministic slot based on layer + entity hash
                    let hash = entity
                        .bytes()
                        .fold(0u64, |acc, b| acc.wrapping_mul(31).wrapping_add(b as u64));
                    *feature = (hash as usize % 10240) + 1;
                }
            }

            // Register the relation label so DESCRIBE shows it
            if let Some(rel) = relation {
                if !rel.is_empty() {
                    // We can't mutate probe_labels directly (it's not behind a lock),
                    // but the patched overlay will store the metadata.
                }
            }
        }
    }
}

async fn apply_patch_to_model(
    state: &AppState,
    model_id: Option<&str>,
    headers: &HeaderMap,
    req: ApplyPatchRequest,
) -> Result<Json<serde_json::Value>, ServerError> {
    let model = state.model_or_err(model_id)?;

    let (mut patch, name) = resolve_patch(&req)?;

    // Enrich INSERT ops with gate vectors if missing
    enrich_patch_ops(model, &mut patch);

    let op_count = patch.operations.len();

    // Session-scoped or global?
    if let Some(sid) = extract_session_id(headers) {
        let (ops, active) = state.sessions.apply_patch(&sid, model, patch).await;
        Ok(Json(serde_json::json!({
            "applied": name,
            "operations": ops,
            "active_patches": active,
            "session": sid,
        })))
    } else {
        let mut patched = model.patched.write().await;
        patched.apply_patch(patch);
        let active = patched.num_patches();
        Ok(Json(serde_json::json!({
            "applied": name,
            "operations": op_count,
            "active_patches": active,
        })))
    }
}

#[utoipa::path(
    post,
    path = "/v1/patches/apply",
    tag = "patches",
    request_body(
        content = crate::openapi::schemas::ApplyPatchBody,
        description = "Provide either a `url` (path / http(s):// / hf://) or an inline `patch` object. \
                       Use the `X-Session-Id` header to scope the apply to a session.",
    ),
    responses(
        (status = 200, description = "Patch applied", body = crate::openapi::schemas::ApplyPatchResponse),
        (status = 400, body = crate::error::ErrorBody),
        (status = 500, body = crate::error::ErrorBody),
    ),
)]
pub async fn handle_apply_patch(
    State(state): State<Arc<AppState>>,
    headers: HeaderMap,
    Json(req): Json<ApplyPatchRequest>,
) -> Result<Json<serde_json::Value>, ServerError> {
    state.bump_requests();
    apply_patch_to_model(&state, None, &headers, req).await
}

#[utoipa::path(
    post,
    path = "/v1/{model_id}/patches/apply",
    tag = "patches",
    params(("model_id" = String, Path, description = "Id of a loaded vindex.")),
    request_body(content = crate::openapi::schemas::ApplyPatchBody),
    responses(
        (status = 200, body = crate::openapi::schemas::ApplyPatchResponse),
        (status = 400, body = crate::error::ErrorBody),
        (status = 404, body = crate::error::ErrorBody),
    ),
)]
pub async fn handle_apply_patch_multi(
    State(state): State<Arc<AppState>>,
    Path(model_id): Path<String>,
    headers: HeaderMap,
    Json(req): Json<ApplyPatchRequest>,
) -> Result<Json<serde_json::Value>, ServerError> {
    state.bump_requests();
    apply_patch_to_model(&state, Some(&model_id), &headers, req).await
}

async fn list_patches_for_model(
    state: &AppState,
    model_id: Option<&str>,
    headers: &HeaderMap,
) -> Result<Json<serde_json::Value>, ServerError> {
    let _model = state.model_or_err(model_id)?;

    if let Some(sid) = extract_session_id(headers) {
        let patches = state.sessions.list_patches(&sid).await;
        return Ok(Json(serde_json::json!({
            "patches": patches,
            "session": sid,
        })));
    }

    let model = state.model(model_id).unwrap();
    let patched = model.patched.read().await;
    let patches: Vec<serde_json::Value> = patched
        .patches
        .iter()
        .map(|p| {
            serde_json::json!({
                "name": p.description.as_deref().unwrap_or(PATCH_UNNAMED),
                "operations": p.operations.len(),
                "base_model": p.base_model,
            })
        })
        .collect();

    Ok(Json(serde_json::json!({ "patches": patches })))
}

#[utoipa::path(
    get,
    path = "/v1/patches",
    tag = "patches",
    responses(
        (status = 200, description = "Active patches for the current session or global state",
         body = crate::openapi::schemas::ListPatchesResponse),
        (status = 404, body = crate::error::ErrorBody),
    ),
)]
pub async fn handle_list_patches(
    State(state): State<Arc<AppState>>,
    headers: HeaderMap,
) -> Result<Json<serde_json::Value>, ServerError> {
    state.bump_requests();
    list_patches_for_model(&state, None, &headers).await
}

#[utoipa::path(
    get,
    path = "/v1/{model_id}/patches",
    tag = "patches",
    params(("model_id" = String, Path, description = "Id of a loaded vindex.")),
    responses(
        (status = 200, body = crate::openapi::schemas::ListPatchesResponse),
        (status = 404, body = crate::error::ErrorBody),
    ),
)]
pub async fn handle_list_patches_multi(
    State(state): State<Arc<AppState>>,
    Path(model_id): Path<String>,
    headers: HeaderMap,
) -> Result<Json<serde_json::Value>, ServerError> {
    state.bump_requests();
    list_patches_for_model(&state, Some(&model_id), &headers).await
}

async fn remove_patch_from_model(
    state: &AppState,
    model_id: Option<&str>,
    headers: &HeaderMap,
    name: &str,
) -> Result<Json<serde_json::Value>, ServerError> {
    if let Some(sid) = extract_session_id(headers) {
        let remaining = state
            .sessions
            .remove_patch(&sid, name)
            .await
            .map_err(ServerError::NotFound)?;
        return Ok(Json(serde_json::json!({
            "removed": name,
            "active_patches": remaining,
            "session": sid,
        })));
    }

    let model = state.model_or_err(model_id)?;

    let mut patched = model.patched.write().await;

    let idx = patched
        .patches
        .iter()
        .position(|p| p.description.as_deref().unwrap_or(PATCH_UNNAMED) == name)
        .ok_or_else(|| ServerError::NotFound(format!("patch '{}' not found", name)))?;

    patched.remove_patch(idx);

    Ok(Json(serde_json::json!({
        "removed": name,
        "active_patches": patched.num_patches(),
    })))
}

#[utoipa::path(
    delete,
    path = "/v1/patches/{name}",
    tag = "patches",
    params(
        ("name" = String, Path, description = "Patch description/name (or `inline-patch` if it was inlined without one)."),
    ),
    responses(
        (status = 200, description = "Patch removed", body = crate::openapi::schemas::RemovePatchResponse),
        (status = 404, body = crate::error::ErrorBody),
    ),
)]
pub async fn handle_remove_patch(
    State(state): State<Arc<AppState>>,
    headers: HeaderMap,
    Path(name): Path<String>,
) -> Result<Json<serde_json::Value>, ServerError> {
    state.bump_requests();
    remove_patch_from_model(&state, None, &headers, &name).await
}

#[utoipa::path(
    delete,
    path = "/v1/{model_id}/patches/{name}",
    tag = "patches",
    params(
        ("model_id" = String, Path, description = "Id of a loaded vindex."),
        ("name" = String, Path, description = "Patch description/name."),
    ),
    responses(
        (status = 200, body = crate::openapi::schemas::RemovePatchResponse),
        (status = 404, body = crate::error::ErrorBody),
    ),
)]
pub async fn handle_remove_patch_multi(
    State(state): State<Arc<AppState>>,
    Path((model_id, name)): Path<(String, String)>,
    headers: HeaderMap,
) -> Result<Json<serde_json::Value>, ServerError> {
    state.bump_requests();
    remove_patch_from_model(&state, Some(&model_id), &headers, &name).await
}
