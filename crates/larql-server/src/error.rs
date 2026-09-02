//! Error types → HTTP status codes.

use axum::http::StatusCode;
use axum::response::{IntoResponse, Response};
use serde::Serialize;
use utoipa::ToSchema;

/// JSON body returned for every error response.
#[derive(Debug, Serialize, ToSchema)]
pub struct ErrorBody {
    /// Human-readable error message.
    pub error: String,
}

#[derive(Debug, thiserror::Error)]
pub enum ServerError {
    #[error("not found: {0}")]
    NotFound(String),

    #[error("bad request: {0}")]
    BadRequest(String),

    /// The request is well-formed but the server's current state
    /// refuses it — a lifecycle mutation (`POST`/`DELETE
    /// /v1/runtime/model`) that conflicts with what's already
    /// happening (a load/unload in progress, a different model
    /// already bound) or with a static topology invariant
    /// (`docs/runtime-lifecycle-design.md` §7). Distinct from
    /// `BadRequest`: retrying the identical request later, once the
    /// conflicting state has changed, can succeed.
    #[error("conflict: {0}")]
    Conflict(String),

    #[error("inference not available: {0}")]
    #[allow(dead_code)]
    InferenceUnavailable(String),

    #[error("internal error: {0}")]
    Internal(String),

    /// Inference handler exceeded the server-side deadline.  We drop
    /// the in-flight `spawn_blocking` future, log the original
    /// elapsed time, and respond `504 Gateway Timeout` so the
    /// client can decide whether to retry.  The blocking thread
    /// keeps running to completion in the background — we don't
    /// have cooperative cancellation on the inference path — but it
    /// no longer holds up the HTTP handler or the next request.
    #[error("inference timed out: {0}")]
    Timeout(String),

    /// The request names a model that IS bound, but the capability
    /// behind this route is not served for that model's generation —
    /// today, a VINDEX2-only route asked to act on a VINDEX3
    /// container. Distinct from `NotFound`: a loaded-but-unsupported
    /// model must never masquerade as absent, so this is `501 Not
    /// Implemented` with the generation named, never a 404.
    #[error("not supported: {0}")]
    Unsupported(String),
}

impl ServerError {
    /// The message without the variant prefix — for surfaces that carry
    /// their own status vocabulary (WebSocket error frames, gRPC).
    pub fn message(&self) -> &str {
        match self {
            ServerError::NotFound(m)
            | ServerError::BadRequest(m)
            | ServerError::Conflict(m)
            | ServerError::InferenceUnavailable(m)
            | ServerError::Internal(m)
            | ServerError::Timeout(m)
            | ServerError::Unsupported(m) => m,
        }
    }
}

impl IntoResponse for ServerError {
    fn into_response(self) -> Response {
        let (status, message) = match &self {
            ServerError::NotFound(msg) => (StatusCode::NOT_FOUND, msg.clone()),
            ServerError::BadRequest(msg) => (StatusCode::BAD_REQUEST, msg.clone()),
            ServerError::Conflict(msg) => (StatusCode::CONFLICT, msg.clone()),
            ServerError::InferenceUnavailable(msg) => {
                (StatusCode::SERVICE_UNAVAILABLE, msg.clone())
            }
            ServerError::Internal(msg) => (StatusCode::INTERNAL_SERVER_ERROR, msg.clone()),
            ServerError::Timeout(msg) => (StatusCode::GATEWAY_TIMEOUT, msg.clone()),
            ServerError::Unsupported(msg) => (StatusCode::NOT_IMPLEMENTED, msg.clone()),
        };

        (status, axum::Json(ErrorBody { error: message })).into_response()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn every_variant() -> Vec<ServerError> {
        vec![
            ServerError::NotFound("nf".into()),
            ServerError::BadRequest("br".into()),
            ServerError::Conflict("cf".into()),
            ServerError::InferenceUnavailable("iu".into()),
            ServerError::Internal("in".into()),
            ServerError::Timeout("to".into()),
            ServerError::Unsupported("un".into()),
        ]
    }

    /// `message()` is the bare text every variant carries, for the
    /// surfaces that speak their own status vocabulary.
    #[test]
    fn message_is_the_inner_text_of_every_variant() {
        let expected = ["nf", "br", "cf", "iu", "in", "to", "un"];
        for (err, want) in every_variant().into_iter().zip(expected) {
            assert_eq!(err.message(), want, "{err}");
            assert!(err.to_string().ends_with(want), "{err}");
        }
    }

    /// Each variant maps to exactly one status; `Unsupported` is 501, the
    /// code that says "bound, but not this operation" — never 404.
    #[test]
    fn each_variant_maps_to_its_status() {
        let expected = [
            StatusCode::NOT_FOUND,
            StatusCode::BAD_REQUEST,
            StatusCode::CONFLICT,
            StatusCode::SERVICE_UNAVAILABLE,
            StatusCode::INTERNAL_SERVER_ERROR,
            StatusCode::GATEWAY_TIMEOUT,
            StatusCode::NOT_IMPLEMENTED,
        ];
        for (err, want) in every_variant().into_iter().zip(expected) {
            let text = err.message().to_string();
            let response = err.into_response();
            assert_eq!(response.status(), want, "{text}");
        }
    }
}
