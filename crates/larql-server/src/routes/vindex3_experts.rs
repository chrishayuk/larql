use crate::{
    error::ServerError,
    state::{AppState, ServedModel},
    vindex3::V3Model,
};
use axum::{
    extract::{FromRequest, State},
    response::{IntoResponse, Response},
    Json,
};
use larql_router_protocol::vindex3_experts as wire;
use std::sync::Arc;
fn worker(state: &AppState) -> Result<Arc<V3Model>, ServerError> {
    match state.served_or_err(None)? {
        ServedModel::V3(model) if model.expert_wire.is_some() => Ok(model),
        _ => Err(ServerError::Unsupported(
            "requires VINDEX3 --ffn-only --layers ... --experts ... worker".into(),
        )),
    }
}
pub async fn metadata(
    State(state): State<Arc<AppState>>,
) -> Result<Json<wire::Binding>, ServerError> {
    state.bump_requests();
    Ok(Json(
        worker(&state)?
            .expert_wire
            .as_ref()
            .unwrap()
            .execution
            .binding()
            .clone(),
    ))
}
pub async fn open(
    State(state): State<Arc<AppState>>,
    Json(binding): Json<wire::Binding>,
) -> Result<Json<wire::Opened>, ServerError> {
    state.bump_requests();
    let model = worker(&state)?;
    let w = model.expert_wire.as_ref().unwrap();
    if &binding != w.execution.binding() {
        return Err(ServerError::BadRequest(
            "expert open binding mismatch".into(),
        ));
    }
    Ok(Json(wire::Opened {
        version: 1,
        handle: w.handle,
        binding,
    }))
}
pub async fn forward(
    State(state): State<Arc<AppState>>,
    request: axum::extract::Request,
) -> Result<Response, ServerError> {
    if request
        .headers()
        .get(axum::http::header::CONTENT_TYPE)
        .is_none_or(|h| h != wire::CONTENT_TYPE)
    {
        return Err(ServerError::BadRequest(
            "expert binary content type required".into(),
        ));
    }
    let bytes = match axum::body::Bytes::from_request(request, &state).await {
        Ok(b) => b,
        Err(e) => return Ok(e.into_response()),
    };
    state.bump_requests();
    let model = worker(&state)?;
    let w = model.expert_wire.as_ref().unwrap();
    let hidden = w.execution.binding().program.hidden;
    let request = wire::decode_request(&bytes, hidden, w.execution.max_selected())
        .map_err(ServerError::BadRequest)?;
    if request.handle != w.handle {
        return Err(ServerError::BadRequest(
            "unknown or stale expert worker handle".into(),
        ));
    }
    let b = w.execution.binding();
    if !(b.program.start..b.program.end).contains(&request.layer)
        || request
            .experts
            .iter()
            .any(|id| !(b.expert_start..b.expert_end).contains(id))
    {
        return Err(ServerError::BadRequest(
            "layer or expert outside worker ownership".into(),
        ));
    }
    let output = tokio::task::spawn_blocking(move || {
        model.expert_wire.as_ref().unwrap().execution.apply(
            request.layer,
            &request.experts,
            &request.row,
        )
    })
    .await
    .map_err(|e| ServerError::Internal(e.to_string()))?
    .map_err(|e| ServerError::BadRequest(e.to_string()))?;
    let rows: Vec<_> = output.into_iter().map(|r| (r.expert, r.row)).collect();
    let bytes = wire::encode_response(
        request.handle,
        request.sequence,
        request.layer,
        hidden,
        &rows,
    )
    .map_err(ServerError::Internal)?;
    Ok((
        [(axum::http::header::CONTENT_TYPE, wire::CONTENT_TYPE)],
        bytes,
    )
        .into_response())
}
