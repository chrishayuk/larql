//! GET /v1/health

use std::sync::Arc;

use axum::extract::State;
use axum::Json;

use crate::band_utils::HEALTH_STATUS_OK;
use crate::state::AppState;

#[utoipa::path(
    get,
    path = "/v1/health",
    tag = "admin",
    responses(
        (status = 200, description = "Server is alive", body = crate::openapi::schemas::HealthResponse),
    ),
)]
pub async fn handle_health(State(state): State<Arc<AppState>>) -> Json<serde_json::Value> {
    state.bump_requests();
    let uptime = state.started_at.elapsed().as_secs();
    let served = state
        .requests_served
        .load(std::sync::atomic::Ordering::Relaxed);

    Json(serde_json::json!({
        "status": HEALTH_STATUS_OK,
        "uptime_seconds": uptime,
        "requests_served": served,
    }))
}
