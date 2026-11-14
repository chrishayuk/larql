//! The browse routes read what the request's session sees.
//!
//! `select`, `walk`, `relations`, `describe` and `explain-infer` used to read
//! the shared overlay whatever the request's `X-Session-Id`, so a client that
//! patched its session read the model back as if it had not. And `/v1/infer`,
//! the one route that did read the session, read a session's overlay for
//! whichever model the request named, though the overlay was built from the
//! model the session was bound to.
//!
//! Each route now reads through `SessionManager::view_blocking`. The HTTP
//! tests here use the routes whose answer the fixture lets a patch change
//! (`select`, `walk`, `relations`); the rule itself is tested on the manager.

mod common;
use common::*;

use std::sync::Arc;
use std::time::{Duration, Instant};

use axum::http::StatusCode;

const SESSION_HEADER: &str = "x-session-id";
const SESSION: &str = "reads-session";
const OTHER_SESSION: &str = "another-session";
/// The fixture feature `inline_delete_patch` deletes (layer 0, feature 2).
const DELETED_TARGET: &str = "Europe";
/// Features the fixture has, and how many the delete leaves.
const FEATURES: u64 = 3;
const FEATURES_AFTER_DELETE: u64 = FEATURES - 1;

/// A state whose session `SESSION` has deleted `DELETED_TARGET`; the shared
/// model is untouched.
async fn session_with_a_deleted_feature() -> Arc<larql_server::state::AppState> {
    let st = state(vec![model_functional("test")]);
    let resp = post_json_h(
        single_model_router(st.clone()),
        "/v1/patches/apply",
        inline_delete_patch("delete-europe"),
        (SESSION_HEADER, SESSION),
    )
    .await;
    assert_eq!(resp.status(), StatusCode::OK);
    st
}

async fn select_total(st: &Arc<larql_server::state::AppState>, session: Option<&str>) -> u64 {
    let app = single_model_router(st.clone());
    let resp = match session {
        Some(id) => {
            post_json_h(
                app,
                "/v1/select",
                serde_json::json!({}),
                (SESSION_HEADER, id),
            )
            .await
        }
        None => post_json(app, "/v1/select", serde_json::json!({})).await,
    };
    assert_eq!(resp.status(), StatusCode::OK);
    body_json(resp.into_body()).await["total"]
        .as_u64()
        .expect("total")
}

async fn get_text(
    st: &Arc<larql_server::state::AppState>,
    path: &str,
    session: Option<&str>,
) -> String {
    let app = single_model_router(st.clone());
    let resp = match session {
        Some(id) => get_h(app, path, (SESSION_HEADER, id)).await,
        None => get(app, path).await,
    };
    assert_eq!(resp.status(), StatusCode::OK);
    body_json(resp.into_body()).await.to_string()
}

// ── Routes ────────────────────────────────────────────────────────────────

#[tokio::test]
async fn select_reads_the_sessions_overlay() {
    let st = session_with_a_deleted_feature().await;
    assert_eq!(
        select_total(&st, Some(SESSION)).await,
        FEATURES_AFTER_DELETE
    );
    assert_eq!(select_total(&st, None).await, FEATURES, "shared model");
}

#[tokio::test]
async fn a_session_that_never_patched_selects_the_shared_model() {
    let st = session_with_a_deleted_feature().await;
    assert_eq!(select_total(&st, Some(OTHER_SESSION)).await, FEATURES);
}

#[tokio::test]
async fn walk_reads_the_sessions_overlay() {
    let st = session_with_a_deleted_feature().await;
    let path = "/v1/walk?prompt=France&top=3";
    let in_session = get_text(&st, path, Some(SESSION)).await;
    let shared = get_text(&st, path, None).await;
    assert!(!in_session.contains(DELETED_TARGET), "{in_session}");
    assert!(shared.contains(DELETED_TARGET), "{shared}");
}

#[tokio::test]
async fn relations_reads_the_sessions_overlay() {
    let st = session_with_a_deleted_feature().await;
    let in_session = get_text(&st, "/v1/relations", Some(SESSION)).await;
    let shared = get_text(&st, "/v1/relations", None).await;
    assert!(!in_session.contains(DELETED_TARGET), "{in_session}");
    assert!(shared.contains(DELETED_TARGET), "{shared}");
}

#[tokio::test]
async fn describe_answers_a_patched_session_past_the_shared_cache() {
    // The cache holds what the shared overlay answered. A seeded entry stands
    // for it: the shared request is served it, the patched session is not,
    // and the session's answer does not replace it.
    const ENTITY: &str = "France";
    const BAND: &str = "all";
    const LIMIT: usize = 5;
    const MIN_SCORE: f32 = 1.0;
    const CACHE_TTL_SECS: u64 = 60;
    let path =
        format!("/v1/describe?entity={ENTITY}&band={BAND}&limit={LIMIT}&min_score={MIN_SCORE}");
    let seeded = serde_json::json!({"seeded": "the shared answer"});

    let st = state_with_cache(vec![model_functional("test")], CACHE_TTL_SECS);
    let resp = post_json_h(
        single_model_router(st.clone()),
        "/v1/patches/apply",
        inline_delete_patch("delete-europe"),
        (SESSION_HEADER, SESSION),
    )
    .await;
    assert_eq!(resp.status(), StatusCode::OK);
    let key = larql_server::cache::DescribeCache::key("test", ENTITY, BAND, LIMIT, MIN_SCORE);
    st.describe_cache.put(key.clone(), seeded.clone());

    let in_session = get_text(&st, &path, Some(SESSION)).await;
    assert!(!in_session.contains("seeded"), "{in_session}");
    assert_eq!(st.describe_cache.get(&key), Some(seeded.clone()));

    let shared = get_text(&st, &path, None).await;
    assert_eq!(shared, seeded.to_string());
    let never_patched = get_text(&st, &path, Some(OTHER_SESSION)).await;
    assert_eq!(never_patched, seeded.to_string());
}

// ── The rule ──────────────────────────────────────────────────────────────

/// What `view_blocking` gives, as "session" or "shared".
async fn view_kind(
    st: &Arc<larql_server::state::AppState>,
    model_id: &str,
    session: Option<&'static str>,
    at: Instant,
) -> &'static str {
    let st = st.clone();
    let model_id = model_id.to_string();
    tokio::task::spawn_blocking(move || {
        let model = st.model_or_err(Some(&model_id)).expect("model");
        if st
            .sessions
            .view_blocking_at(&model, session, at)
            .is_session()
        {
            "session"
        } else {
            "shared"
        }
    })
    .await
    .expect("view")
}

#[tokio::test]
async fn the_sessions_overlay_is_read_only_when_it_applies() {
    let st = state(vec![model_functional("a"), model_functional("b")]);
    let resp = post_json_h(
        multi_model_router(st.clone()),
        "/v1/a/patches/apply",
        inline_delete_patch("delete-europe"),
        (SESSION_HEADER, SESSION),
    )
    .await;
    assert_eq!(resp.status(), StatusCode::OK);
    let now = Instant::now();
    let past_ttl = now + st.sessions.ttl() + Duration::from_secs(1);

    assert_eq!(view_kind(&st, "a", Some(SESSION), now).await, "session");
    assert_eq!(view_kind(&st, "a", None, now).await, "shared", "no header");
    assert_eq!(
        view_kind(&st, "a", Some(OTHER_SESSION), now).await,
        "shared",
        "unknown session"
    );
    assert_eq!(
        view_kind(&st, "a", Some(SESSION), past_ttl).await,
        "shared",
        "expired session"
    );
    assert_eq!(
        view_kind(&st, "b", Some(SESSION), now).await,
        "shared",
        "a session bound to another model"
    );

    let a = st.model_or_err(Some("a")).expect("a");
    assert!(st.sessions.has_patched_view(&a, Some(SESSION)).await);
    assert!(!st.sessions.has_patched_view(&a, None).await);
    let b = st.model_or_err(Some("b")).expect("b");
    assert!(!st.sessions.has_patched_view(&b, Some(SESSION)).await);
}

#[tokio::test]
async fn a_session_bound_to_another_model_selects_that_models_own_features() {
    let st = state(vec![model_functional("a"), model_functional("b")]);
    let resp = post_json_h(
        multi_model_router(st.clone()),
        "/v1/a/patches/apply",
        inline_delete_patch("delete-europe"),
        (SESSION_HEADER, SESSION),
    )
    .await;
    assert_eq!(resp.status(), StatusCode::OK);

    for (model_id, expected) in [("a", FEATURES_AFTER_DELETE), ("b", FEATURES)] {
        let resp = post_json_h(
            multi_model_router(st.clone()),
            &format!("/v1/{model_id}/select"),
            serde_json::json!({}),
            (SESSION_HEADER, SESSION),
        )
        .await;
        assert_eq!(resp.status(), StatusCode::OK);
        let total = body_json(resp.into_body()).await["total"].as_u64();
        assert_eq!(total, Some(expected), "model {model_id}");
    }
}
