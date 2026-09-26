//! Stateless, artifact-bound V3 layer-prefix RPC. Only mounted privately.
use crate::{
    error::ServerError,
    state::{AppState, ServedModel},
    vindex3::V3Model,
};
use axum::{extract::State, Json};
use larql_router_protocol::vindex3::{Binding, Request, Response};
use std::sync::Arc;

fn shard(state: &AppState) -> Result<Arc<V3Model>, ServerError> {
    match state.served_or_err(None)? {
        ServedModel::V3(model) if model.shard.is_some() => Ok(model),
        _ => Err(ServerError::Unsupported(
            "requires a VINDEX3 CPU binding started with --layers".into(),
        )),
    }
}
pub async fn metadata(State(state): State<Arc<AppState>>) -> Result<Json<Binding>, ServerError> {
    state.bump_requests();
    Ok(Json(shard(&state)?.shard.clone().expect("checked shard")))
}
pub async fn forward(
    State(state): State<Arc<AppState>>,
    Json(request): Json<Request>,
) -> Result<Json<Response>, ServerError> {
    state.bump_requests();
    let model = shard(&state)?;
    let binding = model.shard.as_ref().expect("checked shard");
    if request.binding != *binding {
        return Err(ServerError::BadRequest(
            "layer request does not match this worker's binding".into(),
        ));
    }
    binding
        .validate_rows(&request.rows)
        .map_err(ServerError::BadRequest)?;
    tokio::task::spawn_blocking(move || model.execute_layer_prefix(request).map(Json))
        .await
        .map_err(|e| ServerError::Internal(e.to_string()))?
}
