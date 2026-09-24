//! Private artifact-bound dense FFN operation service.
use crate::{
    error::ServerError,
    state::{AppState, ServedModel},
    vindex3::V3Model,
};
use axum::{extract::State, Json};
use larql_router_protocol::vindex3_ffn::{Binding, Request, Response};
use std::sync::Arc;
fn worker(state: &AppState) -> Result<Arc<V3Model>, ServerError> {
    match state.served_or_err(None)? {
        ServedModel::V3(model) if model.ffn_shard.is_some() => Ok(model),
        _ => Err(ServerError::Unsupported(
            "requires a VINDEX3 CPU --ffn-only --layers worker".into(),
        )),
    }
}
pub async fn metadata(State(state): State<Arc<AppState>>) -> Result<Json<Binding>, ServerError> {
    state.bump_requests();
    Ok(Json(
        worker(&state)?.ffn_shard.clone().expect("checked worker"),
    ))
}
pub async fn forward(
    State(state): State<Arc<AppState>>,
    Json(request): Json<Request>,
) -> Result<Json<Response>, ServerError> {
    state.bump_requests();
    let model = worker(&state)?;
    let binding = model.ffn_shard.as_ref().expect("checked worker");
    if request.binding != *binding {
        return Err(ServerError::BadRequest(
            "dense FFN request does not match this worker's binding".into(),
        ));
    }
    binding
        .validate_row(request.layer, &request.row)
        .map_err(ServerError::BadRequest)?;
    tokio::task::spawn_blocking(move || {
        larql_inference::vindex3::dense_ffn::forward(
            model.runtime.plan(),
            model.runtime.operands(),
            model.runtime.backend(),
            &request.binding,
            request.layer,
            &request.row,
        )
        .map(Json)
        .map_err(|e| ServerError::Internal(e.to_string()))
    })
    .await
    .map_err(|e| ServerError::Internal(e.to_string()))?
}
