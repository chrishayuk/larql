//! POST /v1/query — the PUBLIC_EXPLORER statement surface.
//!
//! The body carries one LQL statement; the response carries the
//! session's lines. No simulation and no route-side filtering: the
//! statement runs through the same `Session::execute` every other
//! transport uses, where `CapabilityProfile::PublicExplorer` judges it
//! after parsing and before execution. The HTTP mapping keeps the
//! refusal honest — 403 means nothing failed, the profile does not
//! serve this.

use std::sync::Arc;

use axum::extract::State;
use axum::http::StatusCode;
use axum::Json;
use serde::{Deserialize, Serialize};

use crate::lql_bridge::{LqlBridge, QueryFailure};

/// Longest statement the surface accepts. The grammar has no statement
/// anywhere near this long; anything larger is not a query.
const MAX_STATEMENT_BYTES: usize = 8 * 1024;

#[derive(Debug, Deserialize, utoipa::ToSchema)]
pub struct QueryRequest {
    /// One LQL statement, e.g. `SHOW COMPONENTS;`
    pub statement: String,
}

#[derive(Debug, Serialize, utoipa::ToSchema)]
pub struct QueryResponse {
    /// The capability profile the statement executed under.
    pub profile: String,
    /// The session's output lines, in order.
    pub lines: Vec<String>,
}

#[utoipa::path(
    post,
    path = "/v1/query",
    tag = "browse",
    request_body = QueryRequest,
    responses(
        (status = 200, description = "Statement executed", body = QueryResponse),
        (status = 400, description = "Statement did not parse", body = crate::error::ErrorBody),
        (status = 403, description = "The capability profile refused the statement", body = crate::error::ErrorBody),
        (status = 413, description = "Statement too long", body = crate::error::ErrorBody),
        (status = 422, description = "Execution failed", body = crate::error::ErrorBody),
    ),
)]
pub async fn handle_query(
    State(bridge): State<Arc<LqlBridge>>,
    Json(req): Json<QueryRequest>,
) -> Result<Json<QueryResponse>, (StatusCode, Json<serde_json::Value>)> {
    let err = |status: StatusCode, msg: String| (status, Json(serde_json::json!({ "error": msg })));

    if req.statement.len() > MAX_STATEMENT_BYTES {
        return Err(err(
            StatusCode::PAYLOAD_TOO_LARGE,
            format!("statement exceeds {MAX_STATEMENT_BYTES} bytes"),
        ));
    }

    match bridge.query(req.statement).await {
        Ok(lines) => Ok(Json(QueryResponse {
            profile: bridge.profile.into(),
            lines,
        })),
        Err(QueryFailure::Parse(msg)) => Err(err(StatusCode::BAD_REQUEST, msg)),
        Err(QueryFailure::Refused(msg)) => Err(err(StatusCode::FORBIDDEN, msg)),
        Err(QueryFailure::Execution(msg)) => Err(err(StatusCode::UNPROCESSABLE_ENTITY, msg)),
        Err(QueryFailure::Bridge(msg)) => Err(err(StatusCode::SERVICE_UNAVAILABLE, msg)),
    }
}
