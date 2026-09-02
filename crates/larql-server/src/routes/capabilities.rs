//! `GET /v1/capabilities` — the Explorer contract's front door.
//!
//! Thin by design. Every judgement lives in
//! [`crate::capabilities`], which derived this answer from the route
//! ledger at router-build time; the handler only serves it. There is
//! nothing to recompute per request — the route table is frozen once
//! axum has it, so a capability that changed between requests would
//! be a lie in one of them.
//!
//! Unauthenticated and mounted on every profile: a client has to be
//! able to ask what a server does *before* it knows whether it can
//! talk to it, and the answer discloses no model, no data and no
//! path — only which of this server's own routes exist.

use std::sync::Arc;

use axum::extract::State;
use axum::Json;

use crate::capabilities::Capabilities;

#[utoipa::path(
    get,
    path = "/v1/capabilities",
    tag = "admin",
    responses(
        (status = 200, description = "What this server will and will not do", body = crate::openapi::schemas::CapabilitiesResponse),
    ),
)]
pub async fn handle_capabilities(State(caps): State<Arc<Capabilities>>) -> Json<serde_json::Value> {
    Json(caps.to_json())
}
