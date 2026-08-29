//! Typed container facts — the query protocol's read nouns.
//!
//! `GET /v1/components`, `/v1/representations`, `/v1/provenance`,
//! `/v1/authority`: the same declarations the LQL directory statements
//! print, returned as structured JSON so a client can *render* them
//! rather than re-parse prose. This is the protocol taking shape: any
//! independent VINDEX3 reader could serve these from `index.json` and
//! the system graph alone, which is exactly the litmus test for what
//! belongs on the surface. Read-only; the container is the authority;
//! nothing here is reconstructed from names.

use std::sync::Arc;

use axum::extract::State;
use axum::Json;

use crate::error::ServerError;
use crate::state::{AppState, ServedModel};
use larql_vindex::format::filenames::INDEX_JSON;
use larql_vindex::format::vindex3::index::{ContainerAuthority, Vindex3Index};
use larql_vindex::format::vindex3::inspect::inspect_container;

fn v3_path(state: &AppState) -> Result<std::path::PathBuf, ServerError> {
    match state.served_or_err(None)? {
        ServedModel::V3(model) => Ok(model.path.clone()),
        ServedModel::V2(_) => Err(ServerError::NotFound(
            "container facts require a VINDEX3 binding".into(),
        )),
    }
}

fn read_index(root: &std::path::Path) -> Result<Vindex3Index, ServerError> {
    let text = std::fs::read_to_string(root.join(INDEX_JSON))
        .map_err(|e| ServerError::Internal(format!("read {INDEX_JSON}: {e}")))?;
    serde_json::from_str(&text)
        .map_err(|e| ServerError::Internal(format!("parse {INDEX_JSON}: {e}")))
}

/// `GET /v1/components` — the system graph's census, structured.
pub async fn handle_components(
    State(state): State<Arc<AppState>>,
) -> Result<Json<serde_json::Value>, ServerError> {
    state.bump_requests();
    let path = v3_path(&state)?;
    let inspection = inspect_container(&path, false)
        .map_err(|e| ServerError::Internal(format!("inspect container: {e}")))?;
    Ok(Json(serde_json::json!({
        "components": inspection.components,
        "objects": inspection.graph.objects.len(),
        "edges": inspection.graph.edges.len(),
        "coherent": inspection.is_coherent(),
    })))
}

/// `GET /v1/representations` — the physical directory.
pub async fn handle_representations(
    State(state): State<Arc<AppState>>,
) -> Result<Json<serde_json::Value>, ServerError> {
    state.bump_requests();
    let index = read_index(&v3_path(&state)?)?;
    let entries: Vec<serde_json::Value> = index
        .representations
        .iter()
        .map(|(id, e)| {
            serde_json::json!({
                "id": id,
                "object": e.object,
                "encoding": e.encoding,
                "tensor_count": e.tensor_count,
                "payload_bytes": e.payload_bytes,
                "compiled_from": e.compiled_from,
            })
        })
        .collect();
    Ok(Json(serde_json::json!({ "entries": entries })))
}

/// `GET /v1/provenance` — whole hashes and lineage. Digests are never
/// abbreviated: provenance abbreviated is provenance lost.
pub async fn handle_provenance(
    State(state): State<Arc<AppState>>,
) -> Result<Json<serde_json::Value>, ServerError> {
    state.bump_requests();
    let index = read_index(&v3_path(&state)?)?;
    let entries: Vec<serde_json::Value> = index
        .representations
        .iter()
        .map(|(id, e)| {
            serde_json::json!({
                "id": id,
                "object": e.object,
                "segment": e.segment,
                "payload_sha256": e.payload_sha256,
                "segment_sha256": e.segment_sha256,
                "compiled_from": e.compiled_from,
                "source_representation_digest": e.source_representation_digest,
            })
        })
        .collect();
    Ok(Json(serde_json::json!({
        "authority": match index.authority {
            ContainerAuthority::Canonical => "canonical",
            ContainerAuthority::Derived => "derived",
        },
        "derived_from_model": index.derived_from_model,
        "entries": entries,
    })))
}

/// `GET /v1/authority` — the container's own declaration.
pub async fn handle_authority(
    State(state): State<Arc<AppState>>,
) -> Result<Json<serde_json::Value>, ServerError> {
    state.bump_requests();
    let index = read_index(&v3_path(&state)?)?;
    let profiles: Vec<&str> = index.profiles.iter().map(|p| p.name.as_str()).collect();
    Ok(Json(serde_json::json!({
        "authority": match index.authority {
            ContainerAuthority::Canonical => "canonical",
            ContainerAuthority::Derived => "derived",
        },
        "derived_from_model": index.derived_from_model,
        "profiles": profiles,
    })))
}
