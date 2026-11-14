//! Every remote LQL request is per-session.
//!
//! A mutation must land in this client's session overlay on the server,
//! never in the model it shares with other clients; a read must be
//! answered against that same overlay. Both depend on one thing the
//! client controls: the session header on the request. Patches the client
//! builds must also declare the served model as their `base_model`, so the
//! server can refuse one written against another model.
//!
//! Each mock below only answers a request carrying the session header (and,
//! for patches, the right `base_model`); a request without it gets
//! mockito's 501 and the statement fails.

use super::*;
use mockito::Matcher;

/// The header naming the client's session on the server.
const SESSION_HEADER: &str = "x-session-id";
/// Shape of the session id `USE REMOTE` mints: `larql-{pid}-{millis}`.
const SESSION_ID_PATTERN: &str = r"^larql-\d+-\d+$";
/// The model `stats_body()` serves.
const SERVED_MODEL: &str = "test-model";
const ENDPOINT_PATCHES: &str = "/v1/patches";
const ENDPOINT_PATCHES_APPLY: &str = "/v1/patches/apply";

fn session_header() -> Matcher {
    Matcher::Regex(SESSION_ID_PATTERN.into())
}

/// Body matcher: an apply request whose inline patch declares the served
/// model.
fn patch_for_served_model() -> Matcher {
    Matcher::PartialJson(serde_json::json!({"patch": {"base_model": SERVED_MODEL}}))
}

/// Write a `.vlp` declaring `base_model`, with one Delete op.
fn write_patch(file_stem: &str, description: &str, base_model: &str) -> std::path::PathBuf {
    let dir = std::env::temp_dir().join(format!(
        "larql_remote_session_{}_{}",
        std::process::id(),
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_nanos())
            .unwrap_or(0)
    ));
    std::fs::create_dir_all(&dir).expect("create patch dir");
    let path = dir.join(format!("{file_stem}.vlp"));
    let patch = larql_vindex::VindexPatch {
        version: 1,
        base_model: base_model.into(),
        base_checksum: None,
        created_at: String::new(),
        description: Some(description.into()),
        author: None,
        tags: vec![],
        operations: vec![larql_vindex::PatchOp::Delete {
            layer: 1,
            feature: 2,
            reason: Some("session test".into()),
        }],
    };
    patch.save(&path).expect("save .vlp");
    path
}

// ── Reads carry the session ──────────────────────────────────────

#[test]
fn describe_sends_the_session_header() {
    let (mut server, mut session) = connect();
    let m = server
        .mock("GET", Matcher::Regex(r"^/v1/describe".into()))
        .match_header(SESSION_HEADER, session_header())
        .with_status(200)
        .with_body(serde_json::json!({"edges": [], "latency_ms": 1.0}).to_string())
        .create();
    ok_joined(&mut session, r#"DESCRIBE "France";"#);
    m.assert();
}

#[test]
fn walk_and_explain_walk_send_the_session_header() {
    let (mut server, mut session) = connect();
    let m = server
        .mock("GET", Matcher::Regex(r"^/v1/walk".into()))
        .match_header(SESSION_HEADER, session_header())
        .with_status(200)
        .with_body(serde_json::json!({"hits": [], "latency_ms": 1.0}).to_string())
        .expect(2)
        .create();
    ok_joined(&mut session, r#"WALK "The capital of France is";"#);
    ok_joined(&mut session, r#"EXPLAIN WALK "The capital of France is";"#);
    m.assert();
}

#[test]
fn explain_infer_sends_the_session_header() {
    let (mut server, mut session) = connect();
    let m = server
        .mock("POST", "/v1/explain-infer")
        .match_header(SESSION_HEADER, session_header())
        .with_status(200)
        .with_body(serde_json::json!({"predictions": [], "trace": []}).to_string())
        .create();
    ok_joined(&mut session, r#"EXPLAIN INFER "The capital of France is";"#);
    m.assert();
}

#[test]
fn select_sends_the_session_header() {
    let (mut server, mut session) = connect();
    let m = server
        .mock("POST", "/v1/select")
        .match_header(SESSION_HEADER, session_header())
        .with_status(200)
        .with_body(serde_json::json!({"edges": [], "total": 0}).to_string())
        .create();
    ok_joined(
        &mut session,
        r#"SELECT * FROM EDGES WHERE entity = "France";"#,
    );
    m.assert();
}

#[test]
fn show_relations_sends_the_session_header() {
    let (mut server, mut session) = connect();
    let m = server
        .mock("GET", Matcher::Regex(r"^/v1/relations".into()))
        .match_header(SESSION_HEADER, session_header())
        .with_status(200)
        .with_body(serde_json::json!({"relations": [], "probe_relations": []}).to_string())
        .create();
    ok_joined(&mut session, "SHOW RELATIONS;");
    m.assert();
}

#[test]
fn stats_after_use_sends_the_session_header() {
    // The USE handshake runs before a session exists; every STATS after it
    // carries one.
    let mut server = mockito::Server::new();
    let handshake = server
        .mock("GET", ENDPOINT_STATS)
        .match_header(SESSION_HEADER, Matcher::Missing)
        .with_status(200)
        .with_body(stats_body())
        .create();
    let mut session = Session::new();
    ok_joined(&mut session, &format!(r#"USE REMOTE "{}";"#, server.url()));
    handshake.assert();

    let stats = server
        .mock("GET", ENDPOINT_STATS)
        .match_header(SESSION_HEADER, session_header())
        .with_status(200)
        .with_body(stats_body())
        .create();
    ok_joined(&mut session, "STATS;");
    stats.assert();
}

// ── Mutations are per-session and name the served model ─────────

#[test]
fn insert_sends_the_session_header() {
    let (mut server, mut session) = connect();
    let m = server
        .mock("POST", "/v1/insert")
        .match_header(SESSION_HEADER, session_header())
        .with_status(200)
        .with_body(serde_json::json!({"inserted": 1, "mode": "knn", "latency_ms": 1.0}).to_string())
        .create();
    ok_joined(
        &mut session,
        r#"INSERT INTO EDGES (entity, relation, target) VALUES ("X", "r", "Y");"#,
    );
    m.assert();
}

#[test]
fn delete_is_per_session_and_declares_the_served_model() {
    let (mut server, mut session) = connect();
    let m = server
        .mock("POST", ENDPOINT_PATCHES_APPLY)
        .match_header(SESSION_HEADER, session_header())
        .match_body(patch_for_served_model())
        .with_status(200)
        .with_body(serde_json::json!({"applied": "DELETE L26 F7", "active_patches": 1}).to_string())
        .create();
    let out = ok_joined(
        &mut session,
        "DELETE FROM EDGES WHERE layer = 26 AND feature = 7;",
    );
    assert!(out.contains("remote session"), "got: {out}");
    m.assert();
}

#[test]
fn update_is_per_session_and_declares_the_served_model() {
    let (mut server, mut session) = connect();
    let m = server
        .mock("POST", ENDPOINT_PATCHES_APPLY)
        .match_header(SESSION_HEADER, session_header())
        .match_body(patch_for_served_model())
        .with_status(200)
        .with_body(serde_json::json!({"applied": "UPDATE L26 F0", "active_patches": 1}).to_string())
        .create();
    ok_joined(
        &mut session,
        r#"UPDATE EDGES SET target = "Madrid" WHERE layer = 26 AND feature = 0;"#,
    );
    m.assert();
}

#[test]
fn delete_and_update_are_refused_when_the_server_names_no_model() {
    let mut server = mockito::Server::new();
    server
        .mock("GET", ENDPOINT_STATS)
        .with_status(200)
        .with_body(serde_json::json!({"layers": 2, "features": 4}).to_string())
        .create();
    let apply = server
        .mock("POST", ENDPOINT_PATCHES_APPLY)
        .expect(0)
        .create();
    let mut session = Session::new();
    let banner = ok_joined(&mut session, &format!(r#"USE REMOTE "{}";"#, server.url()));
    assert!(banner.contains("unknown"), "got: {banner}");

    let err = run(
        &mut session,
        "DELETE FROM EDGES WHERE layer = 0 AND feature = 0;",
    )
    .unwrap_err();
    assert!(err.contains("remote DELETE refused"), "got: {err}");
    assert!(err.contains("named no model"), "got: {err}");
    let err = run(
        &mut session,
        "UPDATE EDGES SET confidence = 0.5 WHERE layer = 0 AND feature = 0;",
    )
    .unwrap_err();
    assert!(err.contains("remote UPDATE refused"), "got: {err}");
    apply.assert();
}

// ── APPLY / SHOW / REMOVE PATCH reach the server ────────────────

#[test]
fn apply_patch_posts_the_patch_to_the_session() {
    let (mut server, mut session) = connect();
    let m = server
        .mock("POST", ENDPOINT_PATCHES_APPLY)
        .match_header(SESSION_HEADER, session_header())
        .match_body(Matcher::PartialJson(serde_json::json!({
            "patch": {"base_model": SERVED_MODEL, "description": "session-patch"}
        })))
        .with_status(200)
        .with_body(
            serde_json::json!({"applied": "session-patch", "operations": 1, "active_patches": 1})
                .to_string(),
        )
        .create();
    let p = write_patch("apply", "session-patch", SERVED_MODEL);
    let out = ok_joined(&mut session, &format!(r#"APPLY PATCH "{}";"#, sql_path(&p)));
    assert!(out.contains("Applied to remote session"), "got: {out}");
    assert!(out.contains("1 patch(es) active"), "got: {out}");
    m.assert();
    let _ = std::fs::remove_file(p);
}

#[test]
fn apply_patch_refused_by_the_server_is_an_error() {
    let (mut server, mut session) = connect();
    server
        .mock("POST", ENDPOINT_PATCHES_APPLY)
        .with_status(400)
        .with_body(r#"{"error":"patch not applied: slot out of range"}"#)
        .create();
    let p = write_patch("refused", "refused-patch", SERVED_MODEL);
    let err = run(&mut session, &format!(r#"APPLY PATCH "{}";"#, sql_path(&p))).unwrap_err();
    assert!(err.contains("400"), "got: {err}");
    assert!(err.contains("patch not applied"), "got: {err}");
    let _ = std::fs::remove_file(p);
}

#[test]
fn apply_patch_for_another_model_is_refused_before_sending() {
    let (mut server, mut session) = connect();
    let apply = server
        .mock("POST", ENDPOINT_PATCHES_APPLY)
        .expect(0)
        .create();
    let p = write_patch("other_model", "foreign-patch", "some-other-model");
    let err = run(&mut session, &format!(r#"APPLY PATCH "{}";"#, sql_path(&p))).unwrap_err();
    assert!(err.contains("some-other-model"), "got: {err}");
    assert!(err.contains(SERVED_MODEL), "got: {err}");
    apply.assert();
    let _ = std::fs::remove_file(p);
}

#[test]
fn show_patches_lists_the_session_patches_from_the_server() {
    let (mut server, mut session) = connect();
    let m = server
        .mock("GET", ENDPOINT_PATCHES)
        .match_header(SESSION_HEADER, session_header())
        .with_status(200)
        .with_body(
            serde_json::json!({
                "patches": [
                    {"name": "patch-A", "operations": 3, "base_model": SERVED_MODEL},
                    {"operations": 1, "base_model": SERVED_MODEL},
                ],
                "session": "s",
            })
            .to_string(),
        )
        .create();
    let out = ok_joined(&mut session, "SHOW PATCHES;");
    assert!(out.contains("Remote session patches"), "got: {out}");
    assert!(out.contains("patch-A"), "got: {out}");
    assert!(out.contains("3 ops"), "got: {out}");
    assert!(out.contains("(unnamed)"), "got: {out}");
    m.assert();
}

#[test]
fn show_patches_with_none_says_so() {
    let (mut server, mut session) = connect();
    server
        .mock("GET", ENDPOINT_PATCHES)
        .match_header(SESSION_HEADER, session_header())
        .with_status(200)
        .with_body(serde_json::json!({"patches": [], "session": "s"}).to_string())
        .create();
    let out = ok_joined(&mut session, "SHOW PATCHES;");
    assert!(
        out.contains("no patches in this remote session"),
        "got: {out}"
    );
}

#[test]
fn remove_patch_deletes_it_from_the_session() {
    let (mut server, mut session) = connect();
    // The name is one percent-encoded path segment.
    let m = server
        .mock("DELETE", "/v1/patches/insert%20Hyrule-%3EHateno")
        .match_header(SESSION_HEADER, session_header())
        .with_status(200)
        .with_body(
            serde_json::json!({"removed": "insert Hyrule->Hateno", "active_patches": 0})
                .to_string(),
        )
        .create();
    let out = ok_joined(&mut session, r#"REMOVE PATCH "insert Hyrule->Hateno";"#);
    assert!(out.contains("Removed remote patch"), "got: {out}");
    m.assert();
}

#[test]
fn remove_unknown_patch_is_an_error_by_name() {
    let (mut server, mut session) = connect();
    server
        .mock("DELETE", "/v1/patches/ghost")
        .match_header(SESSION_HEADER, session_header())
        .with_status(404)
        .with_body(r#"{"error":"patch 'ghost' not found in session"}"#)
        .create();
    let err = run(&mut session, r#"REMOVE PATCH "ghost";"#).unwrap_err();
    assert!(err.contains("no patch named `ghost`"), "got: {err}");
}

#[test]
fn remove_patch_server_failure_is_an_error() {
    let (mut server, mut session) = connect();
    server
        .mock("DELETE", "/v1/patches/broken")
        .with_status(500)
        .with_body("boom")
        .create();
    let err = run(&mut session, r#"REMOVE PATCH "broken";"#).unwrap_err();
    assert!(err.contains("500"), "got: {err}");
}
