//! `POST /v1/plan` — plan a source without encoding it.
//!
//! Thin: the policy, the cost bound and the cache live in
//! [`crate::plan_service`], and the verdict itself comes from
//! `larql-vindex`. This handler is the HTTP shape and nothing else.

use std::sync::Arc;

use axum::extract::State;
use axum::Json;
use serde::Deserialize;
use utoipa::ToSchema;

use crate::error::ServerError;
use crate::plan_service::PlanService;

#[derive(Debug, Deserialize, ToSchema)]
pub struct PlanRequest {
    /// The artifacts to plan, as `hf://owner/repo[@revision]` or —
    /// where the profile permits it — local checkpoint paths.
    ///
    /// A list, not a string, because a VINDEX3 plan is a *system*
    /// plan: several artifacts can compose into one model, and their
    /// interfaces are part of the verdict. One source is the ordinary
    /// case and is written `["hf://owner/repo"]`. There is deliberately
    /// no singular `source` alias — two spellings of one request is a
    /// second thing to keep in agreement.
    pub sources: Vec<String>,
}

#[utoipa::path(
    post,
    path = "/v1/plan",
    tag = "admin",
    request_body = PlanRequest,
    responses(
        (status = 200, description = "The architecture-support verdict, plus what staging read to reach it", body = serde_json::Value),
        (status = 400, body = crate::error::ErrorBody, description = "No sources, too many, or a source that cannot be read or planned"),
        (status = 403, body = crate::error::ErrorBody, description = "This serving profile will not plan that source form — see GET /v1/capabilities"),
        (status = 409, body = crate::error::ErrorBody, description = "Too many plans already in flight"),
    ),
)]
pub async fn handle_plan(
    State(service): State<Arc<PlanService>>,
    Json(req): Json<PlanRequest>,
) -> Result<Json<serde_json::Value>, ServerError> {
    service.plan(req.sources).await.map(Json)
}
