//! `GET /v1/responses/{response_id}` and
//! `DELETE /v1/responses/{response_id}` — retrieval and deletion of
//! stored responses.
//!
//! Both operate on the bounded in-memory
//! [`crate::response_store::ResponseStore`]; a response evicted by the
//! FIFO cap (or created with `store: false`) is a 404, the same
//! contract clients already handle for expired server-side state.

use std::sync::Arc;

use axum::extract::{Path, State};
use axum::Json;

use crate::routes::openai::OpenAIError;
use crate::state::AppState;

#[utoipa::path(
    get,
    path = "/v1/responses/{response_id}",
    tag = "openai",
    params(("response_id" = String, Path, description = "Stored response id (`resp_…`)")),
    responses(
        (status = 200, description = "The stored response envelope.",
         body = crate::openapi::schemas::OpenAiResponsesResponse),
        (status = 404, description = "Not found", body = crate::routes::openai::error::OpenAIErrorBody),
    ),
)]
pub async fn handle_get_response(
    State(state): State<Arc<AppState>>,
    Path(response_id): Path<String>,
) -> Result<Json<serde_json::Value>, OpenAIError> {
    state.bump_requests();
    let stored = state
        .responses
        .get(&response_id)
        .ok_or_else(|| OpenAIError::not_found(format!("response '{response_id}' not found")))?;
    Ok(Json(stored.envelope.clone()))
}

#[utoipa::path(
    delete,
    path = "/v1/responses/{response_id}",
    tag = "openai",
    params(("response_id" = String, Path, description = "Stored response id (`resp_…`)")),
    responses(
        (status = 200, description = "Deletion receipt.", body = serde_json::Value),
        (status = 404, description = "Not found", body = crate::routes::openai::error::OpenAIErrorBody),
    ),
)]
pub async fn handle_delete_response(
    State(state): State<Arc<AppState>>,
    Path(response_id): Path<String>,
) -> Result<Json<serde_json::Value>, OpenAIError> {
    state.bump_requests();
    if !state.responses.remove(&response_id) {
        return Err(OpenAIError::not_found(format!(
            "response '{response_id}' not found"
        )));
    }
    Ok(Json(serde_json::json!({
        "id": response_id,
        "object": "response.deleted",
        "deleted": true,
    })))
}
