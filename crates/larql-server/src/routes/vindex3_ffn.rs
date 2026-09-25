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
use larql_router_protocol::vindex3_ffn::binary::{self, Direction, Opened};
use larql_router_protocol::vindex3_ffn::{Binding, Request, WorkerTiming, PROFILE_HEADER};
use std::sync::Arc;
use std::time::Instant;

/// The full binding is compared once per client open. The handle names this
/// immutable worker instance, not KV, an authorization credential or a cache.
pub async fn open(
    State(state): State<Arc<AppState>>,
    Json(binding): Json<Binding>,
) -> Result<Json<Opened>, ServerError> {
    state.bump_requests();
    let model = worker(&state)?;
    let wire = model
        .ffn_wire
        .as_ref()
        .ok_or_else(|| ServerError::Unsupported("binary FFN worker unavailable".into()))?;
    if &binding != wire.execution.binding() {
        return Err(ServerError::BadRequest("FFN open binding mismatch".into()));
    }
    Ok(Json(Opened {
        version: binary::VERSION,
        handle: wire.handle,
        binding,
    }))
}

pub async fn binary_forward(
    State(state): State<Arc<AppState>>,
    request: axum::extract::Request,
) -> Result<Response, ServerError> {
    let profiling = request
        .headers()
        .get(PROFILE_HEADER)
        .is_some_and(|v| v == "1");
    let started = profiling.then(Instant::now);
    if request
        .headers()
        .get(axum::http::header::CONTENT_TYPE)
        .is_none_or(|v| v != binary::CONTENT_TYPE)
    {
        return Err(ServerError::BadRequest(
            "binary FFN content type required".into(),
        ));
    }
    let bytes = match axum::body::Bytes::from_request(request, &state).await {
        Ok(bytes) => bytes,
        Err(error) => return Ok(error.into_response()),
    };
    state.bump_requests();
    let model = worker(&state)?;
    let wire = model
        .ffn_wire
        .as_ref()
        .ok_or_else(|| ServerError::Unsupported("binary FFN worker unavailable".into()))?;
    let frame = binary::decode(
        &bytes,
        Direction::Request,
        wire.execution.binding().program.hidden,
    )
    .map_err(ServerError::BadRequest)?;
    if frame.handle != wire.handle {
        return Err(ServerError::BadRequest(
            "unknown or stale FFN handle; reopen worker".into(),
        ));
    }
    let range = &wire.execution.binding().program;
    if !(range.start..range.end).contains(&(frame.layer as usize)) {
        return Err(ServerError::BadRequest(
            "layer outside bound FFN worker".into(),
        ));
    }
    let mut timing = WorkerTiming::default();
    if let Some(started) = started {
        timing.decode_ns = started.elapsed().as_nanos() as u64;
    }
    let queued = profiling.then(Instant::now);
    let (row, mut timing) = tokio::task::spawn_blocking(move || {
        if let Some(queued) = queued {
            timing.queue_ns = queued.elapsed().as_nanos() as u64;
        }
        let execute = profiling.then(Instant::now);
        let row = model
            .ffn_wire
            .as_ref()
            .expect("bound worker")
            .execution
            .apply(frame.layer as usize, &frame.row)
            .map_err(|e| ServerError::Internal(e.to_string()))?;
        if let Some(execute) = execute {
            timing.ffn_ns = execute.elapsed().as_nanos() as u64;
            timing.execute_ns = timing.ffn_ns;
        }
        Ok::<_, ServerError>((row, timing))
    })
    .await
    .map_err(|e| ServerError::Internal(e.to_string()))??;
    let encode = profiling.then(Instant::now);
    let body = binary::encode(
        Direction::Response,
        frame.handle,
        frame.sequence,
        frame.layer as usize,
        &row,
    )
    .map_err(ServerError::Internal)?;
    let mut response = (
        [(axum::http::header::CONTENT_TYPE, binary::CONTENT_TYPE)],
        body,
    )
        .into_response();
    if let (Some(started), Some(encode)) = (started, encode) {
        timing.encode_ns = encode.elapsed().as_nanos() as u64;
        timing.handler_ns = started.elapsed().as_nanos() as u64;
        let value =
            serde_json::to_string(&timing).map_err(|e| ServerError::Internal(e.to_string()))?;
        response.headers_mut().insert(
            PROFILE_HEADER,
            value
                .parse()
                .map_err(|e| ServerError::Internal(format!("FFN timing header: {e}")))?,
        );
    }
    Ok(response)
}
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
