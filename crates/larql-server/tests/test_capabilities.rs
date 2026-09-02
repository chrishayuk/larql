//! `GET /v1/capabilities` — and the invariant that makes it worth
//! trusting.
//!
//! The endpoint exists so a client (the vindex3.org Explorer) never
//! infers a server's powers from its hostname. That only helps if the
//! report cannot over-claim, so the load-bearing test here is
//! `advertised_capabilities_match_the_mounted_router`: for every
//! capability, on every profile, it reads the answer twice —
//!
//! 1. from the report the server generated off its mounted-route
//!    ledger, and
//! 2. from axum's own matcher, by sending a method the route does not
//!    accept: **405 means the path is mounted, 404 means it is not**.
//!    The probe never reaches a handler, so it needs no model and has
//!    no side effects, and it cannot be fooled by a handler that
//!    returns 404 for its own reasons.
//!
//! — and asserts they agree. Two independent readings of one fact.

mod common;

use std::sync::Arc;

use axum::body::Body;
use axum::http::{Request, StatusCode};
use larql_inference::test_utils::synthetic_tokenizer_json;
use larql_server::bootstrap::{classify_source, SourceKind};
use larql_server::capabilities::{CAPABILITIES_SCHEMA, ROUTE_CAPABILITIES, V3_BACKENDS};
use larql_vindex::format::vindex3::fixtures::{
    encode_fixture_container, miniature_glimmer, G_VOCAB,
};
use tower::ServiceExt;

/// A method no route in this server accepts, used to ask axum whether
/// a path exists without invoking anything behind it.
const PROBE_METHOD: &str = "PATCH";

fn v3_container() -> tempfile::TempDir {
    let checkpoint = tempfile::tempdir().unwrap();
    let container = tempfile::tempdir().unwrap();
    encode_fixture_container(
        miniature_glimmer,
        checkpoint.path(),
        container.path(),
        "capabilities-fixture",
    );
    std::fs::write(
        container.path().join("tokenizer.json"),
        synthetic_tokenizer_json(G_VOCAB),
    )
    .unwrap();
    container
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

async fn get_json(app: &axum::Router, uri: &str) -> (StatusCode, serde_json::Value) {
    let req = Request::builder()
        .method("GET")
        .uri(uri)
        .body(Body::empty())
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

/// Ask axum whether `path` is mounted, without reaching a handler.
async fn is_mounted(app: &axum::Router, path: &str) -> bool {
    let req = Request::builder()
        .method(PROBE_METHOD)
        .uri(path)
        .body(Body::empty())
        .unwrap();
    let status = app.clone().oneshot(req).await.unwrap().status();
    match status {
        StatusCode::METHOD_NOT_ALLOWED => true,
        StatusCode::NOT_FOUND => false,
        other => panic!(
            "probe of {path} answered {other} — the {PROBE_METHOD} probe only distinguishes \
             mounted (405) from absent (404); some route now accepts {PROBE_METHOD} and the \
             probe needs a different method"
        ),
    }
}

async fn capabilities_of(app: &axum::Router) -> serde_json::Value {
    let (status, body) = get_json(app, "/v1/capabilities").await;
    assert_eq!(status, StatusCode::OK, "capabilities must answer: {body}");
    body
}

/// The three profiles, each as a live router.
async fn all_profiles(container: &std::path::Path) -> Vec<(&'static str, axum::Router)> {
    vec![
        ("public_explorer", public_app(container)),
        (
            "single_model",
            larql_server::routes::single_model_router(common::state(Vec::new())),
        ),
        (
            "multi_model",
            larql_server::routes::multi_model_router(common::state(Vec::new())),
        ),
    ]
}

// ── the invariant ───────────────────────────────────────────────────

#[tokio::test]
async fn advertised_capabilities_match_the_mounted_router() {
    let container = v3_container();
    for (name, app) in all_profiles(container.path()).await {
        let report = capabilities_of(&app).await;
        assert_eq!(report["profile"], name, "profile must name its own router");

        for cap in ROUTE_CAPABILITIES {
            let advertised = report
                .pointer(cap.key)
                .unwrap_or_else(|| panic!("{name}: {} missing from the report", cap.key))
                .as_bool()
                .unwrap_or_else(|| panic!("{name}: {} is not a boolean", cap.key));
            let mounted = is_mounted(&app, cap.route).await;

            if advertised {
                assert!(
                    mounted,
                    "{name} advertises {} but never mounted {} — the report over-claims, \
                     which is exactly the failure this endpoint exists to prevent",
                    cap.key, cap.route
                );
            } else if cap.source.is_none() {
                // With no source conjunct, "not advertised" can only
                // mean "route absent" — so the two readings must agree
                // in both directions, not just the dangerous one.
                assert!(
                    !mounted,
                    "{name} mounted {} but does not advertise {} — the report under-claims",
                    cap.route, cap.key
                );
            }
        }
    }
}

#[tokio::test]
async fn capabilities_lists_itself_and_is_mounted_on_every_profile() {
    let container = v3_container();
    for (name, app) in all_profiles(container.path()).await {
        let report = capabilities_of(&app).await;
        let routes: Vec<&str> = report["routes"]
            .as_array()
            .unwrap()
            .iter()
            .map(|r| r.as_str().unwrap())
            .collect();
        assert!(
            routes.contains(&"/v1/capabilities"),
            "{name}: a server reporting its surface must appear in it: {routes:?}"
        );
        assert_eq!(report["schema"], CAPABILITIES_SCHEMA);
        assert_eq!(report["object"], "capabilities");
        assert_eq!(report["server"]["name"], "larql-server");
    }
}

/// The reported route list is the ledger, so it must agree with axum
/// route for route — not just on the paths the capability table
/// happens to name.
#[tokio::test]
async fn every_reported_route_is_really_mounted() {
    let container = v3_container();
    for (name, app) in all_profiles(container.path()).await {
        let report = capabilities_of(&app).await;
        for route in report["routes"].as_array().unwrap() {
            let path = route.as_str().unwrap();
            // Paths with parameters need a concrete value to probe.
            if path.contains('{') {
                continue;
            }
            assert!(
                is_mounted(&app, path).await,
                "{name} reports {path} as mounted, but axum does not serve it"
            );
        }
    }
}

// ── what today's server actually offers ─────────────────────────────

#[tokio::test]
async fn public_explorer_offers_the_read_surface_and_refuses_the_rest() {
    let container = v3_container();
    let app = public_app(container.path());
    let report = capabilities_of(&app).await;

    for key in [
        "/explorer/models",
        "/explorer/components",
        "/explorer/representations",
        "/explorer/provenance",
        "/explorer/authority",
        "/explorer/query",
    ] {
        assert_eq!(
            report.pointer(key).unwrap(),
            &serde_json::json!(true),
            "{key}"
        );
    }
    // The public surface executes nothing and binds nothing.
    for key in [
        "/runtime/execute",
        "/runtime/lifecycle",
        "/runtime/introspect",
        "/sources/load/local",
        "/sources/load/hf",
    ] {
        assert_eq!(
            report.pointer(key).unwrap(),
            &serde_json::json!(false),
            "{key}"
        );
    }
}

/// The container-facts routes are mounted **only** on the public
/// explorer today. A localhost server therefore cannot answer them,
/// and the Explorer must be told so rather than discovering it with a
/// 404 — this test pins the asymmetry so it is a decision, not a
/// surprise.
#[tokio::test]
async fn a_local_single_model_server_does_not_yet_serve_container_facts() {
    let app = larql_server::routes::single_model_router(common::state(Vec::new()));
    let report = capabilities_of(&app).await;
    for key in [
        "/explorer/components",
        "/explorer/representations",
        "/explorer/provenance",
        "/explorer/authority",
        "/explorer/query",
    ] {
        assert_eq!(
            report.pointer(key).unwrap(),
            &serde_json::json!(false),
            "{key} — if this flipped, the facts routes gained a second mount and the \
             Explorer's server tab can now offer them"
        );
    }
    // What it does offer instead: binding and running a model.
    assert_eq!(
        report.pointer("/runtime/execute").unwrap(),
        &serde_json::json!(true)
    );
    assert_eq!(
        report.pointer("/runtime/lifecycle").unwrap(),
        &serde_json::json!(true)
    );
    assert_eq!(
        report.pointer("/sources/load/local").unwrap(),
        &serde_json::json!(true)
    );
    assert_eq!(
        report.pointer("/sources/load/hf").unwrap(),
        &serde_json::json!(true)
    );
}

/// Step 4 mounted `/v1/plan`, and this is the whole point of deriving
/// the report from the route ledger: the capability flipped because the
/// route appeared, with no edit to `ROUTE_CAPABILITIES`. Encode and
/// residency are still mounted by nobody and still report false.
#[tokio::test]
async fn planning_is_offered_but_encoding_and_residency_are_not() {
    let container = v3_container();
    for (name, app) in all_profiles(container.path()).await {
        let report = capabilities_of(&app).await;
        assert_eq!(
            report.pointer("/sources/plan/hf").unwrap(),
            &serde_json::json!(true),
            "{name}: every profile plans an hf:// source"
        );
        for key in [
            "/sources/encode/local",
            "/sources/encode/hf",
            "/explorer/residency",
        ] {
            assert_eq!(
                report.pointer(key).unwrap(),
                &serde_json::json!(false),
                "{name}: {key}"
            );
        }
    }
}

/// The one capability that differs by profile rather than by route.
/// A public server plans `hf://` and refuses a local path; a localhost
/// server plans either. `tests/test_plan_route.rs` checks that the
/// endpoint actually behaves this way — here we only pin what is
/// advertised.
#[tokio::test]
async fn only_a_local_server_advertises_planning_a_local_path() {
    let container = v3_container();
    let public = capabilities_of(&public_app(container.path())).await;
    assert_eq!(
        public.pointer("/sources/plan/local").unwrap(),
        &serde_json::json!(false),
        "the public surface must not offer to plan a filesystem path"
    );

    let local = capabilities_of(&larql_server::routes::single_model_router(common::state(
        Vec::new(),
    )))
    .await;
    assert_eq!(
        local.pointer("/sources/plan/local").unwrap(),
        &serde_json::json!(true)
    );
}

// ── the two facts that are not route-derived ────────────────────────

/// `sources.load.hf` is true because the resolver classifies `hf://`
/// as an HF reference — not because a table says so. If `is_hf_path`
/// stopped recognising the scheme, the capability would go false on
/// its own.
#[test]
fn source_kinds_come_from_the_resolver() {
    assert_eq!(classify_source("hf://owner/repo"), SourceKind::Hf);
    assert_eq!(
        classify_source("/var/lib/larql/example.vindex3"),
        SourceKind::Local
    );
    assert_eq!(classify_source("./relative.vindex3"), SourceKind::Local);
}

/// The server binds every V3 container on the CPU executor, so
/// `runtime.backends` is `["cpu"]`. `metal-experts` is a VINDEX2 MoE
/// dispatch feature and must not widen this list: advertising Metal
/// would tell the Explorer it can offer a GPU run this server cannot
/// perform. When a real Metal V3 binding lands, this test fails and
/// `V3_BACKENDS` grows with it.
#[test]
fn backends_match_the_v3_binding() {
    let binding = include_str!("../src/vindex3.rs");
    assert!(
        binding.contains("ProductionBackend"),
        "the V3 loader no longer names ProductionBackend — re-derive V3_BACKENDS"
    );
    let metal_bound = binding.contains("MetalBackend");
    assert_eq!(
        metal_bound,
        V3_BACKENDS.contains(&"metal"),
        "V3_BACKENDS says {V3_BACKENDS:?} while the V3 loader {} a Metal backend",
        if metal_bound {
            "binds"
        } else {
            "does not bind"
        }
    );
    assert_eq!(V3_BACKENDS, &["cpu"]);
}
