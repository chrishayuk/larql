//! `GET /v1/sessions`, `GET /v1/sessions/{id}`, `DELETE /v1/sessions/{id}`
//! — the operational view and eviction control plane for runtime session
//! state.
//!
//! # What this surface is, and is not
//!
//! It **observes and terminates** state the runtime already owns. It is
//! deliberately *not* a second authority for modifying model state: there
//! is no create, no rename, no patch editing here. Patches are applied
//! and removed through `/v1/patches` (and LQL); this endpoint reports
//! their identities and can free the whole session. Keeping mutation in
//! one place is what stops the session API and the patch algebra from
//! drifting into two different answers about what a session contains.
//!
//! For the same reason the representation is metadata only — patch names
//! and counts, continuation *availability* and token counts — never
//! overlay contents, KV bytes, tokenizer state, or cache keys.
//!
//! # Deletion
//!
//! `DELETE` is idempotent and strong. It frees the patch overlay, frees
//! every KV continuation the session owns, and retires the session's
//! lease so a generation still in flight cannot re-insert one afterwards.
//! Repeat deletes are not an error: they report `deleted: false` and free
//! nothing. Unrelated sessions are untouched.
//!
//! A delete that lands while a sessioned request is mid-forward-pass
//! waits for that request's read guard and then removes the session; the
//! in-flight request completes safely on the state it already acquired.
//! That is the frozen contract — see the tests in
//! `tests/test_http_sessions.rs`.

use std::sync::Arc;

use axum::extract::{Path, State};
use axum::Json;

use crate::error::ServerError;
use crate::response_kv::SessionContinuation;
use crate::session::SessionSummary;
use crate::state::AppState;

#[cfg(test)]
mod tests;

/// `object` discriminator for one session.
const SESSION_OBJECT: &str = "session";
/// `object` discriminator for the list envelope.
const LIST_OBJECT: &str = "list";
/// `object` discriminator for the deletion receipt.
const DELETED_OBJECT: &str = "session.deleted";

/// The only lifecycle state a client can observe.
///
/// Expiry is applied at read time — an expired session is absent, not
/// listed as `expired` — so `active` is the only value this surface can
/// currently emit. The field exists so the vocabulary has somewhere to
/// grow (a future RSS-budget eviction might want `evicted`) without
/// changing the shape of the object.
const STATE_ACTIVE: &str = "active";

/// Render one session plus its continuation metadata.
fn session_json(summary: &SessionSummary, continuation: &SessionContinuation) -> serde_json::Value {
    serde_json::json!({
        "object": SESSION_OBJECT,
        "id": summary.id,
        "model": summary.model,
        "created_at": summary.created_at,
        "last_used_at": summary.last_used_at,
        "expires_at": summary.expires_at,
        "state": STATE_ACTIVE,
        "patches": {
            "active": summary.patch_names.len(),
            "ids": summary.patch_names,
        },
        "continuation": {
            "available": continuation.available(),
            "input_tokens": continuation.input_tokens,
            "resumptions": summary.resumptions,
            "reused_tokens_total": summary.reused_tokens_total,
        },
    })
}

/// Join a session summary with the continuation state it owns.
fn render(state: &AppState, summary: &SessionSummary) -> serde_json::Value {
    session_json(summary, &state.v3_kv.owned_by(&summary.id))
}

#[utoipa::path(
    get,
    path = "/v1/sessions",
    tag = "sessions",
    responses(
        (status = 200, description = "Live sessions, most recently used first.",
         body = crate::openapi::schemas::SessionListResponse),
    ),
)]
pub async fn handle_list_sessions(State(state): State<Arc<AppState>>) -> Json<serde_json::Value> {
    state.bump_requests();
    let data: Vec<serde_json::Value> = state
        .sessions
        .list()
        .await
        .iter()
        .map(|s| render(&state, s))
        .collect();
    Json(serde_json::json!({ "object": LIST_OBJECT, "data": data }))
}

#[utoipa::path(
    get,
    path = "/v1/sessions/{session_id}",
    tag = "sessions",
    params(("session_id" = String, Path, description = "Value of the `X-Session-Id` header.")),
    responses(
        (status = 200, description = "OK", body = crate::openapi::schemas::SessionResponse),
        (status = 404, description = "Not found", body = crate::error::ErrorBody),
    ),
)]
pub async fn handle_get_session(
    State(state): State<Arc<AppState>>,
    Path(session_id): Path<String>,
) -> Result<Json<serde_json::Value>, ServerError> {
    state.bump_requests();
    let summary = state
        .sessions
        .get(&session_id)
        .await
        .ok_or_else(|| ServerError::NotFound(format!("session '{session_id}' not found")))?;
    Ok(Json(render(&state, &summary)))
}

#[utoipa::path(
    delete,
    path = "/v1/sessions/{session_id}",
    tag = "sessions",
    params(("session_id" = String, Path, description = "Value of the `X-Session-Id` header.")),
    responses(
        (status = 200, description = "Deletion receipt. Idempotent: deleting an \
          absent session reports `deleted: false` rather than 404.",
         body = crate::openapi::schemas::SessionDeletedResponse),
    ),
)]
pub async fn handle_delete_session(
    State(state): State<Arc<AppState>>,
    Path(session_id): Path<String>,
) -> Json<serde_json::Value> {
    state.bump_requests();
    // Order matters: retiring the session (which kills its lease) first
    // means any generation still in flight can no longer insert a
    // continuation, so the purge below cannot be undone behind our back.
    let patches_freed = state.sessions.delete(&session_id).await;
    let continuations_freed = state.v3_kv.drop_owned_by(&session_id);
    Json(serde_json::json!({
        "object": DELETED_OBJECT,
        "id": session_id,
        "deleted": patches_freed.is_some(),
        "patches_freed": patches_freed.unwrap_or(0),
        "continuations_freed": continuations_freed,
    }))
}
