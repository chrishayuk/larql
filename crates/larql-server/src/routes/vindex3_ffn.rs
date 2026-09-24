//! Private artifact-bound dense FFN operation service.
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
use larql_router_protocol::vindex3_ffn::{Binding, Request, WorkerTiming, PROFILE_HEADER};
use std::sync::Arc;
use std::time::Instant;
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
    request: axum::extract::Request,
) -> Result<Response, ServerError> {
    let profiling = request
        .headers()
        .get(PROFILE_HEADER)
        .is_some_and(|v| v == "1");
    let started = profiling.then(Instant::now);
    let Json(request) = match Json::<Request>::from_request(request, &state).await {
        Ok(request) => request,
        Err(error) => return Ok(error.into_response()),
    };
    let mut timing = WorkerTiming::default();
    if let Some(started) = started {
        timing.decode_ns = started.elapsed().as_nanos() as u64;
    }
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
    let queued = profiling.then(Instant::now);
    let (response, mut timing) = tokio::task::spawn_blocking(move || {
        if let Some(queued) = queued {
            timing.queue_ns = queued.elapsed().as_nanos() as u64;
        }
        let execute = profiling.then(Instant::now);
        let response = larql_inference::vindex3::dense_ffn::forward_timed(
            model.runtime.plan(),
            model.runtime.operands(),
            model.runtime.backend(),
            &request.binding,
            request.layer,
            &request.row,
            profiling.then_some(&mut timing.ffn_ns),
        )
        .map_err(|e| ServerError::Internal(e.to_string()))?;
        if let Some(execute) = execute {
            timing.execute_ns = execute.elapsed().as_nanos() as u64;
        }
        Ok::<_, ServerError>((response, timing))
    })
    .await
    .map_err(|e| ServerError::Internal(e.to_string()))??;
    if !profiling {
        return Ok(Json(response).into_response());
    }
    let encode = Instant::now();
    let body = serde_json::to_vec(&response).map_err(|e| ServerError::Internal(e.to_string()))?;
    timing.encode_ns = encode.elapsed().as_nanos() as u64;
    timing.handler_ns = started.expect("profiling").elapsed().as_nanos() as u64;
    let metadata =
        serde_json::to_string(&timing).map_err(|e| ServerError::Internal(e.to_string()))?;
    Ok((
        [
            (
                axum::http::header::CONTENT_TYPE.as_str(),
                "application/json",
            ),
            (PROFILE_HEADER, metadata.as_str()),
        ],
        body,
    )
        .into_response())
}
