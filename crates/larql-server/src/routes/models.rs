//! `GET /v1/models` — OpenAI-compatible model listing (N0.5).
//!
//! Response shape conforms to the OpenAI Models API
//! (<https://platform.openai.com/docs/api-reference/models/list>):
//!
//! ```json
//! {
//!   "object": "list",
//!   "data": [
//!     { "id": "<model-id>", "object": "model",
//!       "created": <unix-secs>, "owned_by": "larql",
//!       /* larql-specific extras follow */
//!       "path": "/v1/<model-id>" | "/v1",
//!       "features": <total>, "loaded": true }
//!   ]
//! }
//! ```
//!
//! The OpenAI spec only requires `id`, `object`, `created`, `owned_by`;
//! every other field is an extension that compatible clients ignore.
//! This means existing OpenAI SDKs (`openai.models.list()`) work
//! unmodified, while larql-aware clients still see `path` / `features`
//! / `loaded`.

use std::sync::Arc;
use std::time::{SystemTime, UNIX_EPOCH};

use axum::extract::State;
use axum::Json;

use crate::http::API_PREFIX;
use crate::state::AppState;

const MODEL_OBJECT: &str = "model";
const LIST_OBJECT: &str = "list";
const OWNED_BY: &str = "larql";

/// Returns the boot-time of this server in unix seconds. Used as the
/// `created` field for every loaded model — close enough to the
/// OpenAI semantic ("when this model became available") since `larql`
/// loads its full model set at boot.
fn server_boot_unix_secs(state: &AppState) -> u64 {
    let now_unix = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0);
    let uptime = state.started_at.elapsed().as_secs();
    now_unix.saturating_sub(uptime)
}

#[utoipa::path(
    get,
    path = "/v1/models",
    tag = "browse",
    responses(
        (status = 200, description = "OpenAI-compatible list of loaded models", body = crate::openapi::schemas::ModelsListResponse),
    ),
)]
pub async fn handle_models(State(state): State<Arc<AppState>>) -> Json<serde_json::Value> {
    state.bump_requests();

    let created = server_boot_unix_secs(&state);
    let multi = state.is_multi_model();

    let data: Vec<serde_json::Value> = state
        .models
        .iter()
        .map(|m| {
            let total_features: usize = m.config.layers.iter().map(|l| l.num_features).sum();
            serde_json::json!({
                "id": m.id,
                "object": MODEL_OBJECT,
                "created": created,
                "owned_by": OWNED_BY,
                // larql-specific extras — OpenAI clients ignore these.
                "path": if multi {
                    format!("{}/{}", API_PREFIX, m.id)
                } else {
                    API_PREFIX.to_string()
                },
                "features": total_features,
                "loaded": true,
            })
        })
        .collect();

    Json(serde_json::json!({
        "object": LIST_OBJECT,
        "data": data,
    }))
}
