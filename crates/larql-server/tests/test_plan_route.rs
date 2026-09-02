//! `POST /v1/plan` — planning a source before anything is encoded.
//!
//! The load-bearing test is `advertised_plan_sources_match_what_the_
//! endpoint_does`. `GET /v1/capabilities` tells a client which source
//! forms this server will plan; that promise is only worth making if
//! the endpoint keeps it, so the test reads the promise from the
//! report and then checks it against the endpoint's actual answer, per
//! profile. One policy, two readings — the same discipline the
//! capability ledger uses against axum's router.
//!
//! No test here reaches the network. The `hf://` probes are
//! deliberately malformed (`hf://` with no repo), which the spec parser
//! refuses before a client is ever constructed — so an HF source can be
//! taken all the way through the policy gate and stopped at the first
//! step that would cost anything.

mod common;

use std::sync::Arc;

use axum::body::Body;
use axum::http::{Request, StatusCode};
use larql_inference::test_utils::synthetic_tokenizer_json;
use larql_server::capabilities::PLAN_SCHEMA_EXPECTED;
use larql_vindex::format::vindex3::fixtures::{
    encode_fixture_container, miniature_glimmer, G_VOCAB,
};
use tower::ServiceExt;

/// An `hf://` reference that is classified as remote and then refused
/// by the parser — reaching the policy gate without reaching the hub.
const UNREACHABLE_HF: &str = "hf://";

fn v3_container() -> tempfile::TempDir {
    let checkpoint = tempfile::tempdir().unwrap();
    let container = tempfile::tempdir().unwrap();
    encode_fixture_container(
        miniature_glimmer,
        checkpoint.path(),
        container.path(),
        "plan-fixture",
    );
    std::fs::write(
        container.path().join("tokenizer.json"),
        synthetic_tokenizer_json(G_VOCAB),
    )
    .unwrap();
    container
}

/// A real checkpoint on disk — the thing `plan` actually takes.
fn checkpoint() -> tempfile::TempDir {
    let dir = tempfile::tempdir().unwrap();
    miniature_glimmer(dir.path());
    dir
}

fn public_app(container: &std::path::Path) -> axum::Router {
    let artifact = larql_server::bootstrap::load_artifact(
        &container.to_string_lossy(),
        larql_server::bootstrap::LoadVindexOptions::default(),
    )
    .unwrap();
    let v3 = match artifact {
        larql_server::bootstrap::LoadedArtifact::V3(m) => Arc::new(*m),
        larql_server::bootstrap::LoadedArtifact::V2(_) => panic!("fixture must bind as V3"),
    };
    let state = common::state(Vec::new());
    state.model_set.write().unwrap().v3_models = vec![v3];
    let bridge = Arc::new(
        larql_server::lql_bridge::spawn(container, std::time::Duration::from_secs(60)).unwrap(),
    );
    larql_server::routes::public_explorer_router(state, bridge)
}

fn local_app() -> axum::Router {
    larql_server::routes::single_model_router(common::state(Vec::new()))
}

async fn post_plan(app: &axum::Router, sources: &[&str]) -> (StatusCode, serde_json::Value) {
    let req = Request::builder()
        .method("POST")
        .uri("/v1/plan")
        .header("content-type", "application/json")
        .body(Body::from(
            serde_json::json!({ "sources": sources }).to_string(),
        ))
        .unwrap();
    let response = app.clone().oneshot(req).await.unwrap();
    let status = response.status();
    let bytes = axum::body::to_bytes(response.into_body(), usize::MAX)
        .await
        .unwrap();
    (
        status,
        serde_json::from_slice(&bytes).unwrap_or(serde_json::Value::Null),
    )
}

async fn capabilities(app: &axum::Router) -> serde_json::Value {
    let req = Request::builder()
        .uri("/v1/capabilities")
        .body(Body::empty())
        .unwrap();
    let response = app.clone().oneshot(req).await.unwrap();
    let bytes = axum::body::to_bytes(response.into_body(), usize::MAX)
        .await
        .unwrap();
    serde_json::from_slice(&bytes).unwrap()
}

// ── the invariant ───────────────────────────────────────────────────

/// What the report promises about `sources.plan.*` is what the endpoint
/// does. A refusal for policy is 403; anything else (including a 400
/// for a source that cannot be read) means the policy admitted it.
#[tokio::test]
async fn advertised_plan_sources_match_what_the_endpoint_does() {
    let container = v3_container();
    let local_path = checkpoint();
    let local_spec = local_path.path().to_string_lossy().into_owned();

    for (name, app) in [
        ("public_explorer", public_app(container.path())),
        ("single_model", local_app()),
    ] {
        let report = capabilities(&app).await;
        for (key, source) in [
            ("/sources/plan/local", local_spec.as_str()),
            ("/sources/plan/hf", UNREACHABLE_HF),
        ] {
            let advertised = report.pointer(key).unwrap().as_bool().unwrap();
            let (status, body) = post_plan(&app, &[source]).await;
            let refused = status == StatusCode::FORBIDDEN;
            assert_eq!(
                advertised, !refused,
                "{name}: {key} advertises {advertised} but POST /v1/plan answered \
                 {status} for {source} — a client that read the report first would be \
                 surprised by the endpoint. Body: {body}"
            );
        }
    }
}

/// The public surface names what it refuses and why, and points at the
/// report — a refusal that leaves the caller guessing is the failure
/// this whole contract exists to remove.
#[tokio::test]
async fn the_public_refusal_names_the_profile_and_the_capability() {
    let container = v3_container();
    let local = checkpoint();
    let (status, body) = post_plan(
        &public_app(container.path()),
        &[&local.path().to_string_lossy()],
    )
    .await;
    assert_eq!(status, StatusCode::FORBIDDEN);
    let msg = body["error"].as_str().unwrap();
    assert!(msg.contains("public_explorer"), "{msg}");
    assert!(msg.contains("sources.plan.local"), "{msg}");
}

// ── the verdict ─────────────────────────────────────────────────────

#[tokio::test]
async fn a_local_server_plans_a_real_checkpoint() {
    let dir = checkpoint();
    let (status, body) = post_plan(&local_app(), &[&dir.path().to_string_lossy()]).await;
    assert_eq!(status, StatusCode::OK, "{body}");

    // The document is the plan, at the same keys `vindex plan --json`
    // writes it — not a server-shaped wrapper around one.
    assert_eq!(body["schema"], PLAN_SCHEMA_EXPECTED);
    assert!(
        body["planner"].is_object(),
        "the verdict names its judge: {body}"
    );
    assert!(body["artifacts"].is_array());
    assert!(body["admissible"].is_boolean());

    // The verdict names its subject as the caller wrote it.
    assert_eq!(
        body["artifacts"][0]["source"]["path"],
        serde_json::json!(dir.path().to_string_lossy())
    );
}

/// A local checkpoint has no immutable commit, so its verdict is served
/// and never stored. This is the cache rule that matters: a partially
/// immutable verdict is not an immutable verdict.
#[tokio::test]
async fn a_local_verdict_is_served_but_never_cached() {
    let dir = checkpoint();
    let (status, body) = post_plan(&local_app(), &[&dir.path().to_string_lossy()]).await;
    assert_eq!(status, StatusCode::OK, "{body}");
    assert_eq!(body["serving"]["cacheable"], serde_json::json!(false));
    assert_eq!(body["serving"]["cached"], serde_json::json!(false));
    assert_eq!(
        body["serving"]["profile"],
        serde_json::json!("single_model")
    );

    // Asking twice must not turn an uncacheable verdict into a hit.
    let (_, again) = post_plan(&local_app(), &[&dir.path().to_string_lossy()]).await;
    assert_eq!(again["serving"]["cached"], serde_json::json!(false));
}

/// Planning reads headers, so a local checkpoint reports no staging —
/// nothing was staged to answer.
#[tokio::test]
async fn a_local_plan_reports_no_staging() {
    let dir = checkpoint();
    let (_, body) = post_plan(&local_app(), &[&dir.path().to_string_lossy()]).await;
    assert!(
        body.get("staging").is_none(),
        "a local artifact stages nothing: {body}"
    );
}

// ── refusals that are not policy ────────────────────────────────────

#[tokio::test]
async fn an_unreadable_source_is_a_bad_request_not_a_server_error() {
    let (status, body) = post_plan(&local_app(), &["/no/such/checkpoint"]).await;
    assert_eq!(status, StatusCode::BAD_REQUEST, "{body}");
    assert!(
        body["error"].as_str().unwrap().contains("cannot read"),
        "{body}"
    );
}

#[tokio::test]
async fn a_malformed_hf_reference_is_refused_before_the_network() {
    let (status, body) = post_plan(&local_app(), &[UNREACHABLE_HF]).await;
    assert_eq!(status, StatusCode::BAD_REQUEST, "{body}");
}

#[tokio::test]
async fn an_empty_source_list_is_refused() {
    let (status, _) = post_plan(&local_app(), &[]).await;
    assert_eq!(status, StatusCode::BAD_REQUEST);
}

#[tokio::test]
async fn plan_is_mounted_on_every_profile() {
    let container = v3_container();
    for app in [public_app(container.path()), local_app()] {
        let (status, _) = post_plan(&app, &[UNREACHABLE_HF]).await;
        assert_ne!(status, StatusCode::NOT_FOUND, "every profile plans");
    }
}
