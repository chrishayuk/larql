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

    #[error("inference not available: {0}")]
    #[allow(dead_code)]
    InferenceUnavailable(String),

    #[error("internal error: {0}")]
    Internal(String),
}

impl IntoResponse for ServerError {
    fn into_response(self) -> Response {
        let (status, message) = match &self {
            ServerError::NotFound(msg) => (StatusCode::NOT_FOUND, msg.clone()),
            ServerError::BadRequest(msg) => (StatusCode::BAD_REQUEST, msg.clone()),
            ServerError::InferenceUnavailable(msg) => {
                (StatusCode::SERVICE_UNAVAILABLE, msg.clone())
            }
            ServerError::Internal(msg) => (StatusCode::INTERNAL_SERVER_ERROR, msg.clone()),
        };

        (status, axum::Json(ErrorBody { error: message })).into_response()
    }
}
