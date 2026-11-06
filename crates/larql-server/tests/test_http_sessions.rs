//! `/v1/sessions` — the operational view and eviction control plane.
//!
//! Two halves. The **surface** half drives the endpoint against the light
//! synthetic model: what a session looks like, what deletion frees, and
//! that expiry makes a session vanish from every observer at once. The
//! **continuation** half needs a real V3 runtime, because only a V3
//! generation retains a KV state to own — it pins the frozen contract
//! that deleting a session frees its continuations, that unrelated
//! sessions are untouched, and that a chain through a deleted session
//! still produces the right answer without resuming.

mod common;
use common::*;

use std::sync::Arc;

use axum::http::StatusCode;

const SESSION_HEADER: &str = "x-session-id";

// ══════════════════════════════════════════════════════════════
// SURFACE — shape, listing, deletion, expiry
// ══════════════════════════════════════════════════════════════

#[tokio::test]
async fn sessions_list_is_empty_before_anything_binds() {
    let app = single_model_router(state(vec![model("test")]));
    let resp = get(app, "/v1/sessions").await;
    assert_eq!(resp.status(), StatusCode::OK);
    let body = body_json(resp.into_body()).await;
    assert_eq!(body["object"], "list");
    assert_eq!(body["data"], serde_json::json!([]));
}

#[tokio::test]
async fn a_patched_session_is_listed_with_its_patch_identities() {
    let st = state(vec![model("test")]);
    let app = single_model_router(Arc::clone(&st));
    let resp = post_json_h(
        app.clone(),
        "/v1/patches/apply",
        inline_delete_patch("alpha"),
        (SESSION_HEADER, "s-1"),
    )
    .await;
    assert_eq!(resp.status(), StatusCode::OK);

    let body = body_json(get(app, "/v1/sessions").await.into_body()).await;
    let listed = body["data"].as_array().expect("data array");
    assert_eq!(listed.len(), 1);
    let session = &listed[0];
    assert_eq!(session["object"], "session");
    assert_eq!(session["id"], "s-1");
    assert_eq!(session["model"], "test");
    assert_eq!(session["state"], "active");
    assert_eq!(session["patches"]["active"], 1);
    assert_eq!(session["patches"]["ids"][0], "alpha");
    // Lifecycle stamps are coherent: expiry sits one TTL past last use.
    let last_used = session["last_used_at"].as_u64().expect("last_used_at");
    let expires = session["expires_at"].as_u64().expect("expires_at");
    assert_eq!(expires, last_used + st.sessions.ttl().as_secs());
    assert!(session["created_at"].as_u64().expect("created_at") <= last_used);
    // Nothing has generated, so there is nothing to continue from.
    assert_eq!(session["continuation"]["available"], false);
    assert_eq!(session["continuation"]["input_tokens"], 0);
}

#[tokio::test]
async fn get_by_id_returns_the_same_object_as_the_listing() {
    let app = single_model_router(state(vec![model("test")]));
    post_json_h(
        app.clone(),
        "/v1/patches/apply",
        inline_delete_patch("alpha"),
        (SESSION_HEADER, "s-1"),
    )
    .await;

    let listed = body_json(get(app.clone(), "/v1/sessions").await.into_body()).await;
    let single = body_json(get(app, "/v1/sessions/s-1").await.into_body()).await;
    assert_eq!(listed["data"][0], single);
}

#[tokio::test]
async fn get_unknown_session_is_404() {
    let app = single_model_router(state(vec![model("test")]));
    let resp = get(app, "/v1/sessions/never-seen").await;
    assert_eq!(resp.status(), StatusCode::NOT_FOUND);
}

#[tokio::test]
async fn delete_frees_the_session_and_repeats_deterministically() {
    let app = single_model_router(state(vec![model("test")]));
    post_json_h(
        app.clone(),
        "/v1/patches/apply",
        inline_delete_patch("alpha"),
        (SESSION_HEADER, "s-1"),
    )
    .await;

    let first = delete(app.clone(), "/v1/sessions/s-1").await;
    assert_eq!(first.status(), StatusCode::OK);
    let receipt = body_json(first.into_body()).await;
    assert_eq!(receipt["object"], "session.deleted");
    assert_eq!(receipt["id"], "s-1");
    assert_eq!(receipt["deleted"], true);
    assert_eq!(receipt["patches_freed"], 1);
    assert_eq!(receipt["continuations_freed"], 0);

    // Idempotent: a repeat is not an error, and frees nothing.
    let second = delete(app.clone(), "/v1/sessions/s-1").await;
    assert_eq!(second.status(), StatusCode::OK);
    let receipt = body_json(second.into_body()).await;
    assert_eq!(receipt["deleted"], false);
    assert_eq!(receipt["patches_freed"], 0);

    assert_eq!(
        get(app.clone(), "/v1/sessions/s-1").await.status(),
        StatusCode::NOT_FOUND
    );
    // The patch went with the session, not just the listing entry.
    let patches = body_json(
        get_h(app, "/v1/patches", (SESSION_HEADER, "s-1"))
            .await
            .into_body(),
    )
    .await;
    assert_eq!(patches["patches"], serde_json::json!([]));
}

#[tokio::test]
async fn deleting_a_session_that_was_never_created_is_still_a_receipt() {
    let app = single_model_router(state(vec![model("test")]));
    let resp = delete(app, "/v1/sessions/never-seen").await;
    assert_eq!(resp.status(), StatusCode::OK);
    let receipt = body_json(resp.into_body()).await;
    assert_eq!(receipt["deleted"], false);
    assert_eq!(receipt["continuations_freed"], 0);
}

#[tokio::test]
async fn deleting_one_session_leaves_the_others_alone() {
    let app = single_model_router(state(vec![model("test")]));
    for sid in ["s-1", "s-2"] {
        post_json_h(
            app.clone(),
            "/v1/patches/apply",
            inline_delete_patch("alpha"),
            (SESSION_HEADER, sid),
        )
        .await;
    }

    delete(app.clone(), "/v1/sessions/s-1").await;
    assert_eq!(
        get(app.clone(), "/v1/sessions/s-2").await.status(),
        StatusCode::OK
    );
    let body = body_json(get(app, "/v1/sessions").await.into_body()).await;
    let ids: Vec<&str> = body["data"]
        .as_array()
        .expect("data array")
        .iter()
        .map(|s| s["id"].as_str().expect("id"))
        .collect();
    assert_eq!(ids, vec!["s-2"]);
}

#[tokio::test]
async fn an_expired_session_vanishes_from_every_observer_at_once() {
    // Drives the same routine the maintenance sweeper drives, through an
    // explicit clock rather than a sleep: the session must disappear from
    // the listing, from GET by id, and from `/v1/stats.server.sessions`,
    // and unrelated sessions must be unaffected.
    let st = state(vec![model("test")]);
    let app = single_model_router(Arc::clone(&st));
    // Anchor before creation: both sessions are born at or after `t0`.
    let t0 = std::time::Instant::now();
    for sid in ["stale", "fresh"] {
        post_json_h(
            app.clone(),
            "/v1/patches/apply",
            inline_delete_patch("alpha"),
            (SESSION_HEADER, sid),
        )
        .await;
    }
    let stats = body_json(get(app.clone(), "/v1/stats").await.into_body()).await;
    assert_eq!(stats["server"]["sessions"]["active"], 2);

    // Refresh "fresh" half a TTL in, while BOTH are still comfortably
    // live so the refresh itself evicts nothing, then sweep one second
    // past "stale"'s deadline.
    let ttl = st.sessions.ttl();
    st.sessions.bind_at("fresh", "test", t0 + ttl / 2).await;
    let sweep_at = t0 + ttl + std::time::Duration::from_secs(1);
    assert_eq!(st.sessions.evict_expired_at(sweep_at).await, 1);

    assert_eq!(
        get(app.clone(), "/v1/sessions/stale").await.status(),
        StatusCode::NOT_FOUND
    );
    let body = body_json(get(app.clone(), "/v1/sessions").await.into_body()).await;
    assert_eq!(body["data"].as_array().expect("data array").len(), 1);
    assert_eq!(body["data"][0]["id"], "fresh");
    let stats = body_json(get(app, "/v1/stats").await.into_body()).await;
    assert_eq!(stats["server"]["sessions"]["active"], 1);
}

#[tokio::test]
async fn sessions_are_reachable_in_multi_model_mode_too() {
    // Sessions are a server-level resource, not a per-model one, so the
    // path is unprefixed in both routers.
    let app = multi_model_router(state(vec![model("a"), model("b")]));
    let resp = get(app, "/v1/sessions").await;
    assert_eq!(resp.status(), StatusCode::OK);
}

// ══════════════════════════════════════════════════════════════
// CONTINUATION OWNERSHIP — needs a real V3 runtime
// ══════════════════════════════════════════════════════════════

mod continuation {
    use std::path::Path;
    use std::sync::Arc;

    use axum::body::Body;
    use axum::http::{header, Request, StatusCode};
    use larql_inference::test_utils::synthetic_tokenizer_json;
    use larql_server::bootstrap::{load_artifact, LoadVindexOptions, LoadedArtifact};
    use larql_server::state::AppState;
    use larql_vindex::format::vindex3::fixtures::{
        encode_fixture_container, miniature_glimmer, G_VOCAB,
    };
    use tower::ServiceExt;

    use super::SESSION_HEADER;

    const NEW_TOKENS: usize = 4;
    /// Vocabulary-covered user turn.
    const USER_INPUT: &str = "[3]";
    /// Directory basename of the fixture container — the model id derives
    /// from it, and it is what a session records as its binding.
    const MODEL_NAME: &str = "sessions-fixture";

    fn v3_container(root: &Path) -> std::path::PathBuf {
        let checkpoint = root.join("checkpoint");
        let container = root.join(MODEL_NAME);
        std::fs::create_dir_all(&checkpoint).unwrap();
        std::fs::create_dir_all(&container).unwrap();
        encode_fixture_container(miniature_glimmer, &checkpoint, &container, MODEL_NAME);
        std::fs::write(
            container.join("tokenizer.json"),
            synthetic_tokenizer_json(G_VOCAB),
        )
        .unwrap();
        container
    }

    fn v3_state(container: &Path, kv_entries: usize) -> Arc<AppState> {
        let v3 = match load_artifact(&container.to_string_lossy(), LoadVindexOptions::default())
            .unwrap()
        {
            LoadedArtifact::V3(m) => Arc::new(*m),
            LoadedArtifact::V2(_) => panic!("a VINDEX3 container must bind as V3"),
        };
        Arc::new(AppState {
            model_set: std::sync::RwLock::new(larql_server::state::ModelSet {
                models: Vec::new(),
                v3_models: vec![v3],
                aliases: Default::default(),
            }),
            router_topology: larql_server::state::RouterTopology::SingleModel,
            lifecycle: std::sync::Mutex::new(larql_server::state::LifecycleState::Idle),
            started_at: std::time::Instant::now(),
            requests_served: std::sync::atomic::AtomicU64::new(0),
            api_key: None,
            sessions: larql_server::session::SessionManager::new(3600),
            describe_cache: larql_server::cache::DescribeCache::new(0),
            infer_timeout: std::time::Duration::from_secs(60),
            patch_sources: Default::default(),
            responses: larql_server::response_store::ResponseStore::new(),
            v3_kv: larql_server::response_kv::ResponseKvCache::new(
                kv_entries,
                larql_server::response_kv::DEFAULT_TTL_SECS,
            ),
            runtime: Arc::new(larql_server::runtime_stats::RuntimeRecorder::new()),
        })
    }

    /// One `/v1/responses` turn, optionally bound to a session.
    async fn respond(
        app: &axum::Router,
        session: Option<&str>,
        body: serde_json::Value,
    ) -> serde_json::Value {
        let mut req = Request::builder()
            .method("POST")
            .uri("/v1/responses")
            .header(header::CONTENT_TYPE, "application/json");
        if let Some(sid) = session {
            req = req.header(SESSION_HEADER, sid);
        }
        let resp = app
            .clone()
            .oneshot(
                req.body(Body::from(serde_json::to_vec(&body).unwrap()))
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(resp.status(), StatusCode::OK, "generation must succeed");
        let bytes = axum::body::to_bytes(resp.into_body(), usize::MAX)
            .await
            .unwrap();
        serde_json::from_slice(&bytes).unwrap()
    }

    fn turn(input: &str) -> serde_json::Value {
        serde_json::json!({
            "model": MODEL_NAME,
            "input": input,
            "max_output_tokens": NEW_TOKENS,
            "temperature": 0.0,
        })
    }

    async fn json_at(app: &axum::Router, path: &str) -> (StatusCode, serde_json::Value) {
        let resp = app
            .clone()
            .oneshot(
                Request::builder()
                    .method("GET")
                    .uri(path)
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        let status = resp.status();
        let bytes = axum::body::to_bytes(resp.into_body(), usize::MAX)
            .await
            .unwrap();
        (
            status,
            serde_json::from_slice(&bytes).unwrap_or(serde_json::Value::Null),
        )
    }

    async fn delete_session(app: &axum::Router, sid: &str) -> serde_json::Value {
        let resp = app
            .clone()
            .oneshot(
                Request::builder()
                    .method("DELETE")
                    .uri(format!("/v1/sessions/{sid}"))
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(resp.status(), StatusCode::OK);
        let bytes = axum::body::to_bytes(resp.into_body(), usize::MAX)
            .await
            .unwrap();
        serde_json::from_slice(&bytes).unwrap()
    }

    #[tokio::test]
    async fn a_generation_binds_its_continuation_to_the_session() {
        let root = tempfile::tempdir().unwrap();
        let container = v3_container(root.path());
        let state = v3_state(&container, 4);
        let app = larql_server::routes::single_model_router(Arc::clone(&state));

        respond(&app, Some("s-1"), turn(USER_INPUT)).await;
        assert_eq!(state.v3_kv.len(), 1, "a V3 turn retains its KV state");

        let (status, session) = json_at(&app, "/v1/sessions/s-1").await;
        assert_eq!(status, StatusCode::OK);
        assert_eq!(
            session["model"], MODEL_NAME,
            "bound to the runtime that ran"
        );
        assert_eq!(session["continuation"]["available"], true);
        assert!(
            session["continuation"]["input_tokens"]
                .as_u64()
                .expect("input_tokens")
                > 0,
            "the resident state has absorbed the turn's prompt"
        );
    }

    #[tokio::test]
    async fn a_generation_without_the_header_owns_nothing() {
        // Continuation caching still happens — it is just governed by
        // capacity and TTL, with no session able to claim or free it.
        let root = tempfile::tempdir().unwrap();
        let container = v3_container(root.path());
        let state = v3_state(&container, 4);
        let app = larql_server::routes::single_model_router(Arc::clone(&state));

        respond(&app, None, turn(USER_INPUT)).await;
        assert_eq!(state.v3_kv.len(), 1);

        let (_, listing) = json_at(&app, "/v1/sessions").await;
        assert_eq!(listing["data"], serde_json::json!([]));
        let receipt = delete_session(&app, "s-1").await;
        assert_eq!(receipt["continuations_freed"], 0);
        assert_eq!(state.v3_kv.len(), 1, "an unowned state is not collateral");
    }

    #[tokio::test]
    async fn deleting_a_session_frees_its_continuations_and_only_its_own() {
        let root = tempfile::tempdir().unwrap();
        let container = v3_container(root.path());
        let state = v3_state(&container, 4);
        let app = larql_server::routes::single_model_router(Arc::clone(&state));

        respond(&app, Some("victim"), turn(USER_INPUT)).await;
        respond(&app, Some("bystander"), turn(USER_INPUT)).await;
        assert_eq!(state.v3_kv.len(), 2);

        let receipt = delete_session(&app, "victim").await;
        assert_eq!(receipt["deleted"], true);
        assert_eq!(receipt["continuations_freed"], 1);
        assert_eq!(state.v3_kv.len(), 1, "the bystander's state survives");

        let (status, _) = json_at(&app, "/v1/sessions/victim").await;
        assert_eq!(status, StatusCode::NOT_FOUND);
        let (status, bystander) = json_at(&app, "/v1/sessions/bystander").await;
        assert_eq!(status, StatusCode::OK);
        assert_eq!(bystander["continuation"]["available"], true);
    }

    /// Control for [`a_chain_through_a_deleted_session_still_answers_without_resuming`]:
    /// on a LIVE session the same chain finds its resident state, so the
    /// deleted arm's `hits == 0` is a consequence of the deletion and not
    /// of the fixture. (Whether the found state then *engages* is a
    /// separate question — this tokenizer renders every prompt as one
    /// `[UNK]`, so it never does; `hits` is the signal here, not
    /// `cached_tokens`.)
    #[tokio::test]
    async fn a_chain_on_a_live_session_finds_its_continuation() {
        let root = tempfile::tempdir().unwrap();
        let container = v3_container(root.path());
        let state = v3_state(&container, 4);
        let app = larql_server::routes::single_model_router(Arc::clone(&state));

        let first = respond(&app, Some("s-1"), turn(USER_INPUT)).await;
        let first_id = first["id"].as_str().expect("response id").to_string();

        let mut chained = turn(USER_INPUT);
        chained["previous_response_id"] = serde_json::Value::String(first_id);
        respond(&app, Some("s-1"), chained).await;

        assert_eq!(state.v3_kv.hits(), 1, "the session's state was resident");
        assert_eq!(state.v3_kv.misses(), 0);
    }

    #[tokio::test]
    async fn a_chain_through_a_deleted_session_still_answers_without_resuming() {
        // The frozen contract: deletion costs the acceleration, never the
        // answer. The conversation itself lives in the response store, so
        // the chain replays it and re-prefills from scratch.
        let root = tempfile::tempdir().unwrap();
        let container = v3_container(root.path());
        let state = v3_state(&container, 4);
        let app = larql_server::routes::single_model_router(Arc::clone(&state));

        let first = respond(&app, Some("s-1"), turn(USER_INPUT)).await;
        let first_id = first["id"].as_str().expect("response id").to_string();
        delete_session(&app, "s-1").await;
        assert_eq!(state.v3_kv.len(), 0);

        let mut chained = turn(USER_INPUT);
        chained["previous_response_id"] = serde_json::Value::String(first_id);
        let second = respond(&app, Some("s-1"), chained).await;

        assert!(
            !second["output"][0]["content"][0]["text"]
                .as_str()
                .expect("output text")
                .is_empty(),
            "the chain still produces an answer"
        );
        assert_eq!(
            second["usage"]["input_tokens_details"]["cached_tokens"], 0,
            "nothing was served from resumed KV"
        );
        assert_eq!(state.v3_kv.hits(), 0, "nothing resident was found");
        assert_eq!(state.v3_kv.misses(), 1);
    }

    #[tokio::test]
    async fn an_expired_session_takes_its_continuation_with_it() {
        // TTL eviction and explicit deletion must agree: the sweeper kills
        // the lease, and the cache's own sweep collects the orphan.
        let root = tempfile::tempdir().unwrap();
        let container = v3_container(root.path());
        let state = v3_state(&container, 4);
        let app = larql_server::routes::single_model_router(Arc::clone(&state));

        respond(&app, Some("s-1"), turn(USER_INPUT)).await;
        assert_eq!(state.v3_kv.len(), 1);

        let sweep_at =
            std::time::Instant::now() + state.sessions.ttl() + std::time::Duration::from_secs(1);
        assert_eq!(state.sessions.evict_expired_at(sweep_at).await, 1);

        let (status, _) = json_at(&app, "/v1/sessions/s-1").await;
        assert_eq!(status, StatusCode::NOT_FOUND);
        assert_eq!(state.v3_kv.evict_expired(), 1, "the orphan is collected");
        assert_eq!(state.v3_kv.len(), 0);
    }
}
