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
/// File a `hf://` patch repository must carry.
const HF_PATCH_FILE: &str = "patch.vlp";

/// Where `POST /v1/patches {"url": …}` may load a patch from. The default
/// is nowhere: only inline patches are accepted, so a client cannot make
/// the server open files or start hub downloads.
#[derive(Debug, Clone, Default)]
pub struct PatchSources {
    /// Directory local patch paths resolve inside (`--patch-dir`). A path
    /// that escapes it after canonicalisation is refused.
    pub dir: Option<std::path::PathBuf>,
    /// Allow `hf://` references (`--allow-hf-patches`); each one starts a
    /// hub download on the server.
    pub allow_hf: bool,
}

impl PatchSources {
    /// The on-disk patch `url` names, if this policy admits it.
    pub fn resolve(&self, url: &str) -> Result<std::path::PathBuf, ServerError> {
        if larql_vindex::is_hf_path(url) {
            if !self.allow_hf {
                return Err(ServerError::BadRequest(
                    "hf:// patches are disabled on this server (--allow-hf-patches)".into(),
                ));
            }
            let resolved = larql_vindex::resolve_hf_vindex(url).map_err(|e| {
                tracing::warn!("hf patch {url} failed to resolve: {e}");
                ServerError::BadRequest(format!("could not resolve {url}"))
            })?;
            let vlp_path = resolved.join(HF_PATCH_FILE);
            if !vlp_path.exists() {
                return Err(ServerError::BadRequest(format!(
                    "no {HF_PATCH_FILE} found at {url}"
                )));
            }
            return Ok(vlp_path);
        }
        let Some(dir) = &self.dir else {
            return Err(ServerError::BadRequest(
                "patch paths are disabled on this server; send the patch inline \
                 or start it with --patch-dir"
                    .into(),
            ));
        };
        // One message for every refusal below, so a client cannot probe
        // which paths exist.
        let refused = || ServerError::BadRequest(format!("patch {url:?} is not available"));
        let root = dir.canonicalize().map_err(|_| refused())?;
        let path = root.join(url).canonicalize().map_err(|_| refused())?;
        if !path.starts_with(&root) {
            return Err(refused());
        }
        Ok(path)
    }
}

#[derive(Deserialize)]
pub struct ApplyPatchRequest {
    #[serde(default)]
    pub url: Option<String>,
    #[serde(default)]
    pub patch: Option<larql_vindex::VindexPatch>,
}

/// Resolve a patch from the request body (inline or URL).
fn resolve_patch(
    sources: &PatchSources,
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
        let path = sources.resolve(url)?;
        let patch = larql_vindex::VindexPatch::load(&path).map_err(|e| {
            tracing::warn!("patch {} failed to load: {e}", path.display());
            ServerError::BadRequest(format!("patch {url:?} could not be loaded"))
        })?;
        return Ok((patch, url.clone()));
    }

    Err(ServerError::BadRequest(
        "must provide 'url' or 'patch' in request body".into(),
    ))
}

/// Synthesise a gate vector from entity embedding when the client didn't provide one.
///
/// Returns the positions of the INSERT ops that asked the server to choose
/// their slot: no gate vector and `feature == 0`. The slots are chosen later,
/// by [`assign_requested_slots`], on the overlay the patch is applied to.
fn enrich_patch_ops(
    model: &crate::state::LoadedModel,
    patch: &mut larql_vindex::VindexPatch,
) -> Vec<usize> {
    let hidden = model.embeddings.shape()[1];
    let mut requested_slots = Vec::new();
    for (position, op) in patch.operations.iter_mut().enumerate() {
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
                if *feature == 0 {
                    requested_slots.push(position);
                }
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
    requested_slots
}

/// Give each INSERT op at `requested` a slot that is free on `target`, the
/// overlay the patch is about to be applied to, and not given to another op
/// of the same patch. A layer with no free slot refuses the whole patch.
///
/// This used to hash the entity into `1..=10240` whatever the layer's width,
/// and never asked whether the slot was free: two entities could share a
/// slot, and a slot could overwrite a feature the model or another patch had.
fn assign_requested_slots(
    target: &larql_vindex::PatchedVindex,
    patch: &mut larql_vindex::VindexPatch,
    requested: &[usize],
) -> Result<(), larql_vindex::VindexError> {
    let mut claimed: std::collections::HashMap<usize, std::collections::HashSet<usize>> =
        std::collections::HashMap::new();
    // Slots the patch states itself. A requested op's `0` is the request,
    // not a slot, so it claims nothing.
    for (position, op) in patch.operations.iter().enumerate() {
        if let larql_vindex::PatchOp::Insert { layer, feature, .. } = op {
            if !requested.contains(&position) {
                claimed.entry(*layer).or_default().insert(*feature);
            }
        }
    }
    for &position in requested {
        if let Some(larql_vindex::PatchOp::Insert {
            layer,
            feature,
            entity,
            ..
        }) = patch.operations.get_mut(position)
        {
            let taken = claimed.entry(*layer).or_default();
            let slot = target
                .find_free_feature_excluding(*layer, taken)
                .ok_or_else(|| {
                    larql_vindex::VindexError::Parse(format!(
                        "no free feature slot at layer {layer} for the INSERT of `{entity}`"
                    ))
                })?;
            taken.insert(slot);
            *feature = slot;
        }
    }
    Ok(())
}

/// A patch the overlay refused. Nothing was applied, and the client is told
/// so: this used to be answered `"applied"`.
fn refused_patch(e: larql_vindex::VindexError) -> ServerError {
    ServerError::BadRequest(format!("patch not applied: {e}"))
}

async fn apply_patch_to_model(
    state: &AppState,
    model_id: Option<&str>,
    headers: &HeaderMap,
    req: ApplyPatchRequest,
) -> Result<Json<serde_json::Value>, ServerError> {
    let model = state.model_or_err(model_id)?;

    let (mut patch, name) = resolve_patch(&state.patch_sources, &req)?;

    // A patch is written against one model. This route used to apply any
    // patch to any model, including one that names no model at all; local
    // `APPLY PATCH` has always refused both.
    patch
        .check_base_model(&model.config.model)
        .map_err(refused_patch)?;

    // Enrich INSERT ops with gate vectors if missing
    let requested_slots = enrich_patch_ops(&model, &mut patch);

    let op_count = patch.operations.len();

    // Session-scoped or global?
    if let Some(sid) = extract_session_id(headers) {
        let (ops, active) = state
            .sessions
            .apply_patch_with(&sid, &model, patch, |overlay, patch| {
                assign_requested_slots(overlay, patch, &requested_slots)
            })
            .await
            .map_err(refused_patch)?;
        Ok(Json(serde_json::json!({
            "applied": name,
            "operations": ops,
            "active_patches": active,
            "session": sid,
        })))
    } else {
        let mut patched = model.patched.write().await;
        assign_requested_slots(&patched, &mut patch, &requested_slots).map_err(refused_patch)?;
        patched.try_apply_patch(patch).map_err(refused_patch)?;
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
        (status = 400, description = "Bad request", body = crate::error::ErrorBody),
        (status = 500, description = "Internal server error", body = crate::error::ErrorBody),
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
        (status = 200, description = "OK", body = crate::openapi::schemas::ApplyPatchResponse),
        (status = 400, description = "Bad request", body = crate::error::ErrorBody),
        (status = 404, description = "Not found", body = crate::error::ErrorBody),
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

    let model = state.v2_or_unsupported(model_id)?;
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
        (status = 404, description = "Not found", body = crate::error::ErrorBody),
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
        (status = 200, description = "OK", body = crate::openapi::schemas::ListPatchesResponse),
        (status = 404, description = "Not found", body = crate::error::ErrorBody),
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
        (status = 404, description = "Not found", body = crate::error::ErrorBody),
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
        (status = 200, description = "OK", body = crate::openapi::schemas::RemovePatchResponse),
        (status = 404, description = "Not found", body = crate::error::ErrorBody),
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

#[cfg(test)]
mod tests;
