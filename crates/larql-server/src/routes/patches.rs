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

/// Feature-slot space used to hash an entity into a deterministic slot when
/// an INSERT op arrives with `feature == 0`.
///
/// WARNING: this pins Gemma's 10,240-wide FFN (`intermediate_size`) and is
/// wrong for other architectures. The per-layer feature count is not safely
/// reachable here (it lives behind the async `model.patched` lock and this
/// enrichment runs synchronously inside the request handler), so the historic
/// literal is kept — existing Gemma behavior must not change.
const LEGACY_FEATURE_SLOT_SPACE: usize = 10240;

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
                    *feature = (hash as usize % LEGACY_FEATURE_SLOT_SPACE) + 1;
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

    let (mut patch, name) = resolve_patch(&state.patch_sources, &req)?;

    // Enrich INSERT ops with gate vectors if missing
    enrich_patch_ops(&model, &mut patch);

    let op_count = patch.operations.len();

    // Session-scoped or global?
    if let Some(sid) = extract_session_id(headers) {
        let (ops, active) = state.sessions.apply_patch(&sid, &model, patch).await;
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
