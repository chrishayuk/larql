//! OpenAI-shaped error envelope for `/v1/embeddings`,
//! `/v1/completions`, and `/v1/chat/completions`.
//!
//! # Why this exists
//!
//! The crate-wide [`crate::error::ServerError`] renders responses as
//! `{"error": "msg"}` (flat) — the right shape for the LARQL paradigm
//! endpoints (`/v1/describe`, `/v1/walk`, `/v1/select`, ...). The
//! OpenAI Python and JS SDKs, however, expect a nested envelope:
//!
//! ```json
//! {
//!   "error": {
//!     "message": "the encoding_format 'foo' is not supported",
//!     "type": "invalid_request_error",
//!     "param": null,
//!     "code": null
//!   }
//! }
//! ```
//!
//! Returning the flat envelope from `/v1/embeddings` etc. makes the
//! Python SDK throw `KeyError: 'message'` on field access; integrators
//! then have to special-case our server, which is exactly the friction
//! the OpenAI parity surface exists to remove.
//!
//! `OpenAIError` is therefore *only* used by the three OpenAI handlers.
//! Internal helpers in those handlers may continue to return
//! [`ServerError`]; the [`From<ServerError>`] impl below transcribes
//! them at the response boundary.
//!
//! # Mapping
//!
//! | `ServerError` variant   | HTTP status | OpenAI `type`              |
//! |-------------------------|-------------|----------------------------|
//! | `BadRequest`            | 400         | `invalid_request_error`    |
//! | `NotFound`              | 404         | `not_found_error`          |
//! | `InferenceUnavailable`  | 503         | `service_unavailable_error`|
//! | `Internal`              | 500         | `server_error`             |
//! | `Timeout`               | 504         | `timeout_error`            |
//! | `Conflict`              | 409         | `conflict_error`           |
//!
//! See `docs/server-spec.md` for the LARQL-vs-OpenAI envelope split.

use axum::http::StatusCode;
use axum::response::{IntoResponse, Response};
use axum::Json;
use serde::Serialize;
use utoipa::ToSchema;

use crate::error::ServerError;

/// OpenAI-shaped error returned by the three OpenAI-compat handlers.
///
/// Renders as `{"error": {"message", "type", "param", "code"}}`.
#[derive(Debug)]
pub struct OpenAIError {
    pub status: StatusCode,
    pub message: String,
    pub error_type: &'static str,
    pub param: Option<String>,
    pub code: Option<String>,
}

impl OpenAIError {
    pub fn invalid_request(message: impl Into<String>) -> Self {
        Self {
            status: StatusCode::BAD_REQUEST,
            message: message.into(),
            error_type: "invalid_request_error",
            param: None,
            code: None,
        }
    }

    pub fn not_found(message: impl Into<String>) -> Self {
        Self {
            status: StatusCode::NOT_FOUND,
            message: message.into(),
            error_type: "not_found_error",
            param: None,
            code: None,
        }
    }

    pub fn service_unavailable(message: impl Into<String>) -> Self {
        Self {
            status: StatusCode::SERVICE_UNAVAILABLE,
            message: message.into(),
            error_type: "service_unavailable_error",
            param: None,
            code: None,
        }
    }

    pub fn server_error(message: impl Into<String>) -> Self {
        Self {
            status: StatusCode::INTERNAL_SERVER_ERROR,
            message: message.into(),
            error_type: "server_error",
            param: None,
            code: None,
        }
    }

    pub fn timeout(message: impl Into<String>) -> Self {
        Self {
            status: StatusCode::GATEWAY_TIMEOUT,
            message: message.into(),
            error_type: "timeout_error",
            param: None,
            code: None,
        }
    }

    /// A bound model whose generation this route does not serve
    /// (a VINDEX3 container on a VINDEX2-only capability): `501`, so
    /// a loaded-but-unsupported model never reads as absent.
    pub fn not_implemented(message: impl Into<String>) -> Self {
        Self {
            status: StatusCode::NOT_IMPLEMENTED,
            message: message.into(),
            error_type: "not_implemented_error",
            param: None,
            code: None,
        }
    }

    pub fn conflict(message: impl Into<String>) -> Self {
        Self {
            status: StatusCode::CONFLICT,
            message: message.into(),
            error_type: "conflict_error",
            param: None,
            code: None,
        }
    }
}

impl From<ServerError> for OpenAIError {
    fn from(e: ServerError) -> Self {
        match e {
            ServerError::BadRequest(m) => OpenAIError::invalid_request(m),
            ServerError::NotFound(m) => OpenAIError::not_found(m),
            ServerError::InferenceUnavailable(m) => OpenAIError::service_unavailable(m),
            ServerError::Internal(m) => OpenAIError::server_error(m),
            ServerError::Timeout(m) => OpenAIError::timeout(m),
            ServerError::Conflict(m) => OpenAIError::conflict(m),
            ServerError::Unsupported(m) => OpenAIError::not_implemented(m),
        }
    }
}

/// Wire shape for the `error` field. Always includes `param` and `code`
/// even when null — some SDKs hard-key on those keys.
///
/// Public so the OpenAPI schema for 4xx/5xx responses on the OpenAI
/// endpoints can reference it.
#[derive(Debug, Serialize, ToSchema)]
pub struct OpenAIErrorPayload {
    pub message: String,
    #[serde(rename = "type")]
    #[schema(rename = "type")]
    pub error_type: String,
    pub param: Option<String>,
    pub code: Option<String>,
}

/// Top-level OpenAI error envelope: `{"error": {message, type, param, code}}`.
#[derive(Debug, Serialize, ToSchema)]
pub struct OpenAIErrorBody {
    pub error: OpenAIErrorPayload,
}

impl IntoResponse for OpenAIError {
    fn into_response(self) -> Response {
        let body = OpenAIErrorBody {
            error: OpenAIErrorPayload {
                message: self.message,
                error_type: self.error_type.to_string(),
                param: self.param,
                code: self.code,
            },
        };
        (self.status, Json(body)).into_response()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use axum::body::to_bytes;

    async fn body_json(resp: Response) -> serde_json::Value {
        let bytes = to_bytes(resp.into_body(), 1 << 20).await.unwrap();
        serde_json::from_slice(&bytes).unwrap()
    }

    #[tokio::test]
    async fn invalid_request_renders_400_with_nested_envelope() {
        let resp = OpenAIError::invalid_request("bad input").into_response();
        assert_eq!(resp.status(), StatusCode::BAD_REQUEST);
        let v = body_json(resp).await;
        assert_eq!(v["error"]["message"], "bad input");
        assert_eq!(v["error"]["type"], "invalid_request_error");
        assert!(v["error"]["param"].is_null());
        assert!(v["error"]["code"].is_null());
    }

    #[tokio::test]
    async fn not_found_renders_404_with_canonical_type() {
        let resp = OpenAIError::not_found("model 'foo' missing").into_response();
        assert_eq!(resp.status(), StatusCode::NOT_FOUND);
        let v = body_json(resp).await;
        assert_eq!(v["error"]["type"], "not_found_error");
    }

    #[tokio::test]
    async fn service_unavailable_renders_503() {
        let resp = OpenAIError::service_unavailable("inference disabled").into_response();
        assert_eq!(resp.status(), StatusCode::SERVICE_UNAVAILABLE);
        let v = body_json(resp).await;
        assert_eq!(v["error"]["type"], "service_unavailable_error");
        assert_eq!(v["error"]["message"], "inference disabled");
    }

    #[tokio::test]
    async fn server_error_renders_500() {
        let resp = OpenAIError::server_error("oops").into_response();
        assert_eq!(resp.status(), StatusCode::INTERNAL_SERVER_ERROR);
        let v = body_json(resp).await;
        assert_eq!(v["error"]["type"], "server_error");
    }

    #[tokio::test]
    async fn conflict_renders_409() {
        let resp = OpenAIError::conflict("a load is already in progress").into_response();
        assert_eq!(resp.status(), StatusCode::CONFLICT);
        let v = body_json(resp).await;
        assert_eq!(v["error"]["type"], "conflict_error");
        assert_eq!(v["error"]["message"], "a load is already in progress");
    }

    #[test]
    fn from_server_error_preserves_status_and_message() {
        let cases = [
            (
                ServerError::BadRequest("br".into()),
                StatusCode::BAD_REQUEST,
                "invalid_request_error",
                "br",
            ),
            (
                ServerError::NotFound("nf".into()),
                StatusCode::NOT_FOUND,
                "not_found_error",
                "nf",
            ),
            (
                ServerError::InferenceUnavailable("iu".into()),
                StatusCode::SERVICE_UNAVAILABLE,
                "service_unavailable_error",
                "iu",
            ),
            (
                ServerError::Internal("oops".into()),
                StatusCode::INTERNAL_SERVER_ERROR,
                "server_error",
                "oops",
            ),
            (
                ServerError::Timeout("to".into()),
                StatusCode::GATEWAY_TIMEOUT,
                "timeout_error",
                "to",
            ),
            (
                ServerError::Conflict("cf".into()),
                StatusCode::CONFLICT,
                "conflict_error",
                "cf",
            ),
        ];
        for (input, want_status, want_type, want_msg) in cases {
            let oe: OpenAIError = input.into();
            assert_eq!(oe.status, want_status);
            assert_eq!(oe.error_type, want_type);
            assert_eq!(oe.message, want_msg);
        }
    }

    #[tokio::test]
    async fn body_includes_param_and_code_keys_even_when_null() {
        // Some SDKs hard-key on these — we always emit them.
        let resp = OpenAIError::invalid_request("x").into_response();
        let v = body_json(resp).await;
        let err_obj = v["error"].as_object().expect("error is an object");
        assert!(err_obj.contains_key("param"));
        assert!(err_obj.contains_key("code"));
    }
}
