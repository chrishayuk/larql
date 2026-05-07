//! `/v1/attention/*` HTTP routes — session lifecycle.
//!
//! Prefill and decode handlers (POST /v1/attention/prefill,
//! POST /v1/attention/decode) live alongside these once the
//! attention block runner is wired in; this commit covers
//! create/get/delete plus the endpoint stubs that return
//! 501 Not Implemented for the rest.
//!
//! `attention-service-routes` change.

use std::sync::Arc;

use axum::extract::{Path, State};
use axum::http::StatusCode;
use axum::response::IntoResponse;
use axum::Json;
use larql_rotorquant::KvFormat;
use serde::{Deserialize, Serialize};

use crate::attention_session::{AttentionSession, AttentionSessionMap, SessionId, SessionMapError};
use crate::kv_snapshot;
use crate::state::AppState;

// ── Wire types ─────────────────────────────────────────────────────────────

#[derive(Debug, Deserialize)]
pub struct CreateSessionRequest {
    /// Model id this session is bound to. Required.
    pub model_id: String,
    /// Optional KV compression format. `None`/missing ⇒ `"fp32"`.
    /// Accepted: `"fp32"`, `"planar3"`, `"planar4"`, `"iso3"`, `"iso4"`.
    #[serde(default)]
    pub kv_format: Option<String>,
    /// Optional restore: base64-encoded KV snapshot blob.
    /// When provided, the new session's cache is initialised from
    /// the blob (the format is read from the blob's header, not from
    /// `kv_format`).
    #[serde(default)]
    pub restore_from_snapshot: Option<String>,
}

#[derive(Debug, Serialize)]
pub struct CreateSessionResponse {
    pub session_id: String,
    pub layer_range: [u32; 2],
    pub kv_format: &'static str,
}

#[derive(Debug, Serialize)]
pub struct GetSessionResponse {
    pub session_id: String,
    pub model_id: String,
    pub kv_format: &'static str,
    pub seq_len: usize,
    pub prefilled: bool,
    pub num_layers: usize,
}

#[derive(Debug, Serialize)]
pub struct ErrorBody {
    pub error: &'static str,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub detail: Option<String>,
}

// ── Helpers ────────────────────────────────────────────────────────────────

fn map_kv_format(s: &str) -> Result<Option<KvFormat>, String> {
    match s {
        "fp32" => Ok(None),
        "planar3" => Ok(Some(KvFormat::Planar3)),
        "planar4" => Ok(Some(KvFormat::Planar4)),
        "iso3" => Ok(Some(KvFormat::Iso3)),
        "iso4" => Ok(Some(KvFormat::Iso4)),
        other => Err(format!("unknown kv_format: {other}")),
    }
}

fn fmt_to_str(f: Option<KvFormat>) -> &'static str {
    match f {
        None => "fp32",
        Some(KvFormat::Planar3) => "planar3",
        Some(KvFormat::Planar4) => "planar4",
        Some(KvFormat::Iso3) => "iso3",
        Some(KvFormat::Iso4) => "iso4",
    }
}

fn err_response<B: serde::Serialize>(status: StatusCode, body: B) -> (StatusCode, Json<B>) {
    (status, Json(body))
}

fn err_no_session() -> (StatusCode, Json<ErrorBody>) {
    err_response(
        StatusCode::NOT_FOUND,
        ErrorBody {
            error: "no_such_session",
            detail: None,
        },
    )
}

// ── Handlers ───────────────────────────────────────────────────────────────

pub async fn handle_create_session(
    State(state): State<Arc<AppState>>,
    Json(req): Json<CreateSessionRequest>,
) -> impl IntoResponse {
    state.bump_requests();

    let Some(model) = state.model(Some(&req.model_id)) else {
        return err_response(
            StatusCode::NOT_FOUND,
            ErrorBody {
                error: "no_such_model",
                detail: Some(format!("model_id = {}", req.model_id)),
            },
        )
        .into_response();
    };
    let num_layers = model.config.num_layers;

    // Build the session — restore-from-snapshot wins over kv_format.
    let session = if let Some(b64) = &req.restore_from_snapshot {
        match restore_session(&req.model_id, b64) {
            Ok(s) => s,
            Err((status, body)) => return err_response(status, body).into_response(),
        }
    } else {
        let kv_format = match map_kv_format(req.kv_format.as_deref().unwrap_or("fp32")) {
            Ok(v) => v,
            Err(detail) => {
                return err_response(
                    StatusCode::BAD_REQUEST,
                    ErrorBody {
                        error: "kv_format_unknown",
                        detail: Some(detail),
                    },
                )
                .into_response();
            }
        };
        AttentionSession::new(SessionId::new(), &req.model_id, num_layers, kv_format)
    };

    let entry = match state.attention_sessions.insert(session) {
        Ok(e) => e,
        Err(SessionMapError::AtCap { cap }) => {
            return err_response(
                StatusCode::SERVICE_UNAVAILABLE,
                ErrorBody {
                    error: "session_map_full",
                    detail: Some(format!("at cap of {cap}")),
                },
            )
            .into_response();
        }
    };
    let g = entry.read().await;
    let resp = CreateSessionResponse {
        session_id: g.id.as_str().to_string(),
        layer_range: [0, num_layers as u32],
        kv_format: fmt_to_str(g.cache.kv_format),
    };
    Json(resp).into_response()
}

pub async fn handle_get_session(
    State(state): State<Arc<AppState>>,
    Path(id): Path<String>,
) -> impl IntoResponse {
    state.bump_requests();
    let session_id = SessionId(id);
    let Some(entry) = state.attention_sessions.get(&session_id) else {
        return err_no_session().into_response();
    };
    let g = entry.read().await;
    Json(GetSessionResponse {
        session_id: g.id.as_str().to_string(),
        model_id: g.model_id.clone(),
        kv_format: fmt_to_str(g.cache.kv_format),
        seq_len: g.seq_len,
        prefilled: g.prefilled,
        num_layers: g.cache.layers.len(),
    })
    .into_response()
}

pub async fn handle_delete_session(
    State(state): State<Arc<AppState>>,
    Path(id): Path<String>,
) -> impl IntoResponse {
    state.bump_requests();
    let session_id = SessionId(id);
    if state.attention_sessions.remove(&session_id).is_none() {
        return err_no_session().into_response();
    }
    StatusCode::NO_CONTENT.into_response()
}

pub async fn handle_kv_snapshot(
    State(state): State<Arc<AppState>>,
    Json(req): Json<SnapshotRequest>,
) -> impl IntoResponse {
    state.bump_requests();
    let session_id = SessionId(req.session_id);
    let Some(entry) = state.attention_sessions.get(&session_id) else {
        return err_no_session().into_response();
    };
    let g = entry.read().await;
    let bytes = kv_snapshot::serialize(&g.cache);
    use base64::Engine;
    let b64 = base64::engine::general_purpose::STANDARD.encode(&bytes);
    Json(SnapshotResponse {
        session_id: g.id.as_str().to_string(),
        snapshot: b64,
        bytes_len: bytes.len(),
    })
    .into_response()
}

#[derive(Deserialize)]
pub struct SnapshotRequest {
    pub session_id: String,
}

#[derive(Serialize)]
pub struct SnapshotResponse {
    pub session_id: String,
    pub snapshot: String,
    pub bytes_len: usize,
}

// ── /v1/kv-cache/restore ───────────────────────────────────────────────────

#[derive(Deserialize)]
pub struct RestoreRequest {
    pub session_id: String,
    /// Base64-encoded snapshot blob (same byte format as the
    /// snapshot endpoint's `snapshot` field).
    pub snapshot: String,
}

#[derive(Serialize)]
pub struct RestoreResponse {
    pub session_id: String,
    pub seq_len: usize,
    pub num_layers: usize,
}

pub async fn handle_kv_restore(
    State(state): State<Arc<AppState>>,
    Json(req): Json<RestoreRequest>,
) -> impl IntoResponse {
    state.bump_requests();
    let session_id = SessionId(req.session_id);
    let Some(entry) = state.attention_sessions.get(&session_id) else {
        return err_no_session().into_response();
    };

    use base64::Engine;
    let bytes = match base64::engine::general_purpose::STANDARD.decode(&req.snapshot) {
        Ok(b) => b,
        Err(e) => {
            return err_response(
                StatusCode::BAD_REQUEST,
                ErrorBody {
                    error: "snapshot_base64_decode_failed",
                    detail: Some(e.to_string()),
                },
            )
            .into_response();
        }
    };
    let new_cache = match kv_snapshot::deserialize(&bytes) {
        Ok(c) => c,
        Err(e) => {
            let err = match &e {
                kv_snapshot::SnapshotError::UnsupportedVersion { .. } => {
                    "snapshot_version_unsupported"
                }
                kv_snapshot::SnapshotError::MagicMismatch { .. } => "snapshot_magic_mismatch",
                _ => "snapshot_invalid",
            };
            return err_response(
                StatusCode::BAD_REQUEST,
                ErrorBody {
                    error: err,
                    detail: Some(e.to_string()),
                },
            )
            .into_response();
        }
    };

    let mut g = entry.write().await;
    let num_layers = new_cache.layers.len();
    let seq_len = new_cache.next_position;
    g.cache = new_cache;
    g.seq_len = seq_len;
    g.prefilled = seq_len > 0;
    g.touch();

    Json(RestoreResponse {
        session_id: g.id.as_str().to_string(),
        seq_len,
        num_layers,
    })
    .into_response()
}

// ── /v1/kv-cache/free ──────────────────────────────────────────────────────

#[derive(Deserialize)]
pub struct FreeRequest {
    pub session_id: String,
    /// `None` ⇒ free every layer's K/V slot in the cache.
    /// `Some(layer)` ⇒ free that one layer (FP32 + compressed slots).
    #[serde(default)]
    pub layer: Option<u32>,
}

#[derive(Serialize)]
pub struct FreeResponse {
    pub session_id: String,
    pub layers_freed: u32,
}

pub async fn handle_kv_free(
    State(state): State<Arc<AppState>>,
    Json(req): Json<FreeRequest>,
) -> impl IntoResponse {
    state.bump_requests();
    let session_id = SessionId(req.session_id);
    let Some(entry) = state.attention_sessions.get(&session_id) else {
        return err_no_session().into_response();
    };
    let mut g = entry.write().await;
    let total_layers = g.cache.layers.len();
    let layers_freed = match req.layer {
        None => {
            for layer in 0..total_layers {
                g.cache.clear_layer(layer);
            }
            total_layers as u32
        }
        Some(layer) => {
            let layer_us = layer as usize;
            if layer_us >= total_layers {
                return err_response(
                    StatusCode::BAD_REQUEST,
                    ErrorBody {
                        error: "layer_out_of_range",
                        detail: Some(format!("layer {layer} >= num_layers {total_layers}")),
                    },
                )
                .into_response();
            }
            g.cache.clear_layer(layer_us);
            1
        }
    };
    g.touch();
    Json(FreeResponse {
        session_id: g.id.as_str().to_string(),
        layers_freed,
    })
    .into_response()
}

// ── Restore helper ─────────────────────────────────────────────────────────

fn restore_session(
    model_id: &str,
    b64_snapshot: &str,
) -> Result<AttentionSession, (StatusCode, ErrorBody)> {
    use base64::Engine;
    let bytes = base64::engine::general_purpose::STANDARD
        .decode(b64_snapshot)
        .map_err(|e| {
            (
                StatusCode::BAD_REQUEST,
                ErrorBody {
                    error: "snapshot_base64_decode_failed",
                    detail: Some(e.to_string()),
                },
            )
        })?;
    let cache = kv_snapshot::deserialize(&bytes).map_err(|e| {
        let err = match &e {
            kv_snapshot::SnapshotError::UnsupportedVersion { .. } => "snapshot_version_unsupported",
            kv_snapshot::SnapshotError::MagicMismatch { .. } => "snapshot_magic_mismatch",
            _ => "snapshot_invalid",
        };
        (
            StatusCode::BAD_REQUEST,
            ErrorBody {
                error: err,
                detail: Some(e.to_string()),
            },
        )
    })?;
    let mut session = AttentionSession::new(SessionId::new(), model_id, cache.layers.len(), None);
    session.cache = cache;
    session.seq_len = session.cache.next_position;
    session.prefilled = session.seq_len > 0;
    Ok(session)
}

// ── Router builder ─────────────────────────────────────────────────────────

use axum::routing::{delete, get, post};
use axum::Router;

pub fn attention_routes(state: Arc<AppState>) -> Router {
    Router::new()
        .route("/v1/attention/session", post(handle_create_session))
        .route("/v1/attention/session/{id}", get(handle_get_session))
        .route("/v1/attention/session/{id}", delete(handle_delete_session))
        .route("/v1/kv-cache/snapshot", post(handle_kv_snapshot))
        .with_state(state)
}

// Reference suppression for fields we'll need once prefill/decode land.
#[allow(dead_code)]
fn _untouched_helpers() {
    let _ = AttentionSessionMap::new;
}

// ── Tests ──────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::attention_session::AttentionSessionMap;

    fn empty_state() -> Arc<AppState> {
        Arc::new(AppState {
            models: vec![],
            started_at: std::time::Instant::now(),
            requests_served: std::sync::atomic::AtomicU64::new(0),
            api_key: None,
            sessions: crate::session::SessionManager::new(60),
            describe_cache: crate::cache::DescribeCache::new(0),
            attention_sessions: Arc::new(AttentionSessionMap::new(60, 16)),
        })
    }

    #[test]
    fn map_kv_format_recognises_known_formats() {
        assert!(map_kv_format("fp32").unwrap().is_none());
        assert_eq!(map_kv_format("iso3").unwrap(), Some(KvFormat::Iso3));
        assert_eq!(map_kv_format("iso4").unwrap(), Some(KvFormat::Iso4));
        assert_eq!(map_kv_format("planar3").unwrap(), Some(KvFormat::Planar3));
        assert_eq!(map_kv_format("planar4").unwrap(), Some(KvFormat::Planar4));
    }

    #[test]
    fn map_kv_format_rejects_unknown() {
        assert!(map_kv_format("nf4").is_err());
        assert!(map_kv_format("").is_err());
    }

    #[test]
    fn fmt_to_str_round_trips() {
        for s in ["fp32", "planar3", "planar4", "iso3", "iso4"] {
            assert_eq!(fmt_to_str(map_kv_format(s).unwrap()), s);
        }
    }

    #[tokio::test]
    async fn delete_unknown_session_returns_404() {
        let state = empty_state();
        let resp =
            handle_delete_session(State(state), Path("01HM1MISSING0000000000000A".to_string()))
                .await
                .into_response();
        assert_eq!(resp.status(), StatusCode::NOT_FOUND);
    }

    #[tokio::test]
    async fn get_unknown_session_returns_404() {
        let state = empty_state();
        let resp = handle_get_session(State(state), Path("01HM1MISSING0000000000000A".to_string()))
            .await
            .into_response();
        assert_eq!(resp.status(), StatusCode::NOT_FOUND);
    }

    #[tokio::test]
    async fn snapshot_unknown_session_returns_404() {
        let state = empty_state();
        let resp = handle_kv_snapshot(
            State(state),
            Json(SnapshotRequest {
                session_id: "01HM1MISSING0000000000000A".to_string(),
            }),
        )
        .await
        .into_response();
        assert_eq!(resp.status(), StatusCode::NOT_FOUND);
    }

    #[tokio::test]
    async fn restore_unknown_session_returns_404() {
        let state = empty_state();
        let resp = handle_kv_restore(
            State(state),
            Json(RestoreRequest {
                session_id: "01HM1MISSING0000000000000A".to_string(),
                snapshot: String::new(),
            }),
        )
        .await
        .into_response();
        assert_eq!(resp.status(), StatusCode::NOT_FOUND);
    }

    #[tokio::test]
    async fn restore_with_bad_base64_returns_400() {
        let state = empty_state();
        // Insert a session so we get past the 404 path.
        let id = SessionId::new();
        let _ = state
            .attention_sessions
            .insert(AttentionSession::new(id.clone(), "m", 1, None))
            .unwrap();
        let resp = handle_kv_restore(
            State(state),
            Json(RestoreRequest {
                session_id: id.as_str().to_string(),
                snapshot: "@@@not-base64@@@".into(),
            }),
        )
        .await
        .into_response();
        assert_eq!(resp.status(), StatusCode::BAD_REQUEST);
    }

    #[tokio::test]
    async fn restore_with_bad_magic_returns_400() {
        use base64::Engine;
        let state = empty_state();
        let id = SessionId::new();
        let _ = state
            .attention_sessions
            .insert(AttentionSession::new(id.clone(), "m", 1, None))
            .unwrap();
        // 32 bytes of zero — passes the length check but fails magic.
        let bytes = vec![0u8; 64];
        let b64 = base64::engine::general_purpose::STANDARD.encode(&bytes);
        let resp = handle_kv_restore(
            State(state),
            Json(RestoreRequest {
                session_id: id.as_str().to_string(),
                snapshot: b64,
            }),
        )
        .await
        .into_response();
        assert_eq!(resp.status(), StatusCode::BAD_REQUEST);
    }

    #[tokio::test]
    async fn free_unknown_session_returns_404() {
        let state = empty_state();
        let resp = handle_kv_free(
            State(state),
            Json(FreeRequest {
                session_id: "01HM1MISSING0000000000000A".to_string(),
                layer: None,
            }),
        )
        .await
        .into_response();
        assert_eq!(resp.status(), StatusCode::NOT_FOUND);
    }

    #[tokio::test]
    async fn free_layer_out_of_range_returns_400() {
        let state = empty_state();
        let id = SessionId::new();
        let _ = state
            .attention_sessions
            .insert(AttentionSession::new(id.clone(), "m", 4, None))
            .unwrap();
        let resp = handle_kv_free(
            State(state),
            Json(FreeRequest {
                session_id: id.as_str().to_string(),
                layer: Some(99),
            }),
        )
        .await
        .into_response();
        assert_eq!(resp.status(), StatusCode::BAD_REQUEST);
    }

    #[tokio::test]
    async fn free_all_layers_clears_cache() {
        let state = empty_state();
        let id = SessionId::new();
        let mut sess = AttentionSession::new(id.clone(), "m", 3, None);
        sess.cache.layers[0] = Some((
            ndarray::Array2::<f32>::ones((1, 4)),
            ndarray::Array2::<f32>::ones((1, 4)),
        ));
        sess.cache.layers[2] = Some((
            ndarray::Array2::<f32>::ones((1, 4)),
            ndarray::Array2::<f32>::ones((1, 4)),
        ));
        let _ = state.attention_sessions.insert(sess).unwrap();
        let resp = handle_kv_free(
            State(state.clone()),
            Json(FreeRequest {
                session_id: id.as_str().to_string(),
                layer: None,
            }),
        )
        .await
        .into_response();
        assert_eq!(resp.status(), StatusCode::OK);
        // All layer slots are cleared.
        let entry = state.attention_sessions.get(&id).unwrap();
        let g = entry.read().await;
        assert!(g.cache.layers.iter().all(|s| s.is_none()));
    }
}
