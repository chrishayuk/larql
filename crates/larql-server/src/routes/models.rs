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
//!       "features": <total>, "loaded": true,
//!       "aliases": ["<served-name>"] /* only when the model has any */ }
//!   ]
//! }
//! ```
//!
//! The OpenAI spec only requires `id`, `object`, `created`, `owned_by`;
//! every other field is an extension that compatible clients ignore.
//! This means existing OpenAI SDKs (`openai.models.list()`) work
//! unmodified, while larql-aware clients still see `path` / `features`
//! / `loaded`.
//!
//! `id` is always the id the container declares, so a client that read it
//! from this listing keeps working. `aliases` lists the other names a
//! request's `model` field may use for the same model (`--model-alias`,
//! which `larql serve <name>` sets to the name it was given).

use std::sync::Arc;
use std::time::{SystemTime, UNIX_EPOCH};

use axum::extract::{Path, State};
use axum::Json;

use crate::http::API_PREFIX;
use crate::routes::openai::OpenAIError;
use crate::state::{AppState, LoadedModel};
use crate::vindex3::V3Model;

const MODEL_OBJECT: &str = "model";
const LIST_OBJECT: &str = "list";
const OWNED_BY: &str = "larql";
/// larql-specific: the other names a model answers to.
const ALIASES_FIELD: &str = "aliases";
/// larql-specific `generation` marker for VINDEX3 containers.
/// Reported as `generation` on a V3 entry. Sourced from the format
/// crate so the API and the container's own marker cannot drift.
const V3_GENERATION: u32 = larql_vindex::format::generation::ContainerGeneration::V3.number();

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
    // One coherent snapshot for the whole listing, rather than reading
    // the V2 and V3 registries as two separate locked reads — see
    // `ModelSet`.
    let snapshot = state.models_snapshot();

    let mut data: Vec<serde_json::Value> = snapshot
        .models
        .iter()
        .map(|m| with_aliases(v2_entry(m, created, multi), snapshot.aliases_of(&m.id)))
        .collect();
    data.extend(
        snapshot
            .v3_models
            .iter()
            .map(|m| with_aliases(v3_entry(m, created, multi), snapshot.aliases_of(&m.id))),
    );

    Json(serde_json::json!({
        "object": LIST_OBJECT,
        "data": data,
    }))
}

/// One list/retrieve entry for a V2 model. Extras beyond the OpenAI
/// contract (`path`, `generation`, `features`, `loaded`) are
/// larql-specific and ignored by compatible clients.
///
/// `generation` is reported for **both** classes. Emitting it only for
/// V3 made V2 the implicit normal case and V3 the special one, so a
/// client could not tell "generation 2" from "this server predates the
/// field" without knowing larql's release history. Discovery states the
/// generation of everything it lists.
fn v2_entry(m: &LoadedModel, created: u64, multi: bool) -> serde_json::Value {
    let total_features: usize = m.config.layers.iter().map(|l| l.num_features).sum();
    serde_json::json!({
        "id": m.id,
        "object": MODEL_OBJECT,
        "created": created,
        "owned_by": OWNED_BY,
        "path": model_path(&m.id, multi),
        "generation": larql_vindex::format::generation::ContainerGeneration::V2.number(),
        "features": total_features,
        "loaded": true,
    })
}

/// One entry for a VINDEX3 runtime (VI3-SERVE-1). `generation` marks
/// the container class; no `features` count — a V3 container is an
/// executable program, not a feature index.
fn v3_entry(m: &V3Model, created: u64, multi: bool) -> serde_json::Value {
    serde_json::json!({
        "id": m.id,
        "object": MODEL_OBJECT,
        "created": created,
        "owned_by": OWNED_BY,
        "path": model_path(&m.id, multi),
        "generation": V3_GENERATION,
        "loaded": true,
    })
}

/// Add the `aliases` extension to an entry, only when there are any — an
/// entry with none keeps exactly its earlier shape.
fn with_aliases(mut entry: serde_json::Value, aliases: Vec<String>) -> serde_json::Value {
    if !aliases.is_empty() {
        entry[ALIASES_FIELD] = serde_json::json!(aliases);
    }
    entry
}

fn model_path(id: &str, multi: bool) -> String {
    if multi {
        format!("{API_PREFIX}/{id}")
    } else {
        API_PREFIX.to_string()
    }
}

#[utoipa::path(
    get,
    path = "/v1/models/{model}",
    tag = "browse",
    params(("model" = String, Path, description = "Model id")),
    responses(
        (status = 200, description = "OpenAI-compatible single-model entry", body = crate::openapi::schemas::ModelEntry),
        (status = 404, body = crate::routes::openai::error::OpenAIErrorBody),
    ),
)]
pub async fn handle_model_retrieve(
    State(state): State<Arc<AppState>>,
    Path(model): Path<String>,
) -> Result<Json<serde_json::Value>, OpenAIError> {
    state.bump_requests();
    let created = server_boot_unix_secs(&state);
    let multi = state.is_multi_model();

    // `served(Some(id))` searches V2 then V3 by exact id (or alias)
    // match — the same order and semantics this handler implemented by
    // hand before; reusing it means one search implementation, not two.
    let snapshot = state.models_snapshot();
    match state.served(Some(&model)) {
        Some(crate::state::ServedModel::V2(m)) => Ok(Json(with_aliases(
            v2_entry(&m, created, multi),
            snapshot.aliases_of(&m.id),
        ))),
        Some(crate::state::ServedModel::V3(m)) => Ok(Json(with_aliases(
            v3_entry(&m, created, multi),
            snapshot.aliases_of(&m.id),
        ))),
        None => Err(OpenAIError::not_found(format!("model '{model}' not found"))),
    }
}
