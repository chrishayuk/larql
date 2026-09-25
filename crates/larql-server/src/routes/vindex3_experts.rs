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
    let profiling = request
        .headers()
        .get(wire::PROFILE_HEADER)
        .is_some_and(|h| h == "1");
    let started = profiling.then(std::time::Instant::now);
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
    let mut timing = wire::WorkerTiming::default();
    if let Some(started) = started {
        timing.decode_ns = started.elapsed().as_nanos() as u64;
    }
    let queued = profiling.then(std::time::Instant::now);
    let (output, mut timing) = tokio::task::spawn_blocking(move || {
        if let Some(queued) = queued {
            timing.queue_ns = queued.elapsed().as_nanos() as u64;
        }
        let execute = profiling.then(std::time::Instant::now);
        let output = model
            .expert_wire
            .as_ref()
            .unwrap()
            .execution
            .apply_profiled(
                request.layer,
                &request.experts,
                &request.row,
                profiling.then_some(&mut timing.experts_ns),
            )
            .map_err(|e| ServerError::BadRequest(e.to_string()))?;
        if let Some(execute) = execute {
            timing.execute_ns = execute.elapsed().as_nanos() as u64;
        }
        Ok::<_, ServerError>((output, timing))
    })
    .await
    .map_err(|e| ServerError::Internal(e.to_string()))??;
    let encode = profiling.then(std::time::Instant::now);
    let rows: Vec<_> = output.into_iter().map(|r| (r.expert, r.row)).collect();
    let bytes = wire::encode_response(
        request.handle,
        request.sequence,
        request.layer,
        hidden,
        &rows,
    )
    .map_err(ServerError::Internal)?;
    let mut response = (
        [(axum::http::header::CONTENT_TYPE, wire::CONTENT_TYPE)],
        bytes,
    )
        .into_response();
    if let (Some(started), Some(encode)) = (started, encode) {
        timing.encode_ns = encode.elapsed().as_nanos() as u64;
        timing.handler_ns = started.elapsed().as_nanos() as u64;
        let value =
            serde_json::to_string(&timing).map_err(|e| ServerError::Internal(e.to_string()))?;
        response.headers_mut().insert(
            wire::PROFILE_HEADER,
            value
                .parse()
                .map_err(|e| ServerError::Internal(format!("expert timing header: {e}")))?,
        );
    }
    Ok(response)
}
