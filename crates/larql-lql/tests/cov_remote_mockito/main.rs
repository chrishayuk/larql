//! Integration-link coverage for the REMOTE backend forwarders.
//!
//! The in-crate `#[cfg(test)]` mockito tests already exercise the
//! `remote_*` forwarder bodies, but `cargo llvm-cov --package larql-lql`
//! reports against the *integration-link* build of the library — a line
//! covered only by an in-crate unit test does not always move the
//! summary for `executor/remote/query.rs`, `executor/mod.rs::execute_remote`,
//! or the `Backend::Remote` accessor arms in `executor/backend.rs`.
//!
//! These tests therefore drive the remote path through the **public**
//! surface only: `larql_lql::parser::parse` + `Session::execute`. A
//! `USE "<http url>"` statement flips the session to `Backend::Remote`,
//! after which `execute()` routes every statement through
//! `execute_remote`, which in turn calls the `remote_*` forwarders.
//!
//! Each test stands up a `mockito::Server`, points the session at it via
//! `USE "<url>"`, mocks the relevant `/v1/*` endpoint(s) with canned
//! JSON, parses a real LQL statement, executes it, and asserts on the
//! Ok-vs-Err result and the rendered output shape. Both the success
//! (200 + valid JSON) and failure (4xx/5xx, malformed body, unreachable
//! host) branches are covered so both arms of each forwarder run.
//!
//! Plumbing-only: no real model is loaded. Assertions are on output
//! *shape* (header present, row rendered, error message text), never on
//! semantic model behaviour.

use larql_lql::executor::Session;
use larql_lql::parser;

const ENDPOINT_STATS: &str = "/v1/stats";

/// Canned `/v1/stats` body — the connection probe that `USE "<url>"`
/// runs to confirm the server is reachable and to print the banner.
fn stats_body() -> String {
    serde_json::json!({
        "model": "test-model",
        "family": "llama",
        "layers": 32,
        "features": 4096,
        "hidden_size": 1024,
        "dtype": "f32",
        "extract_level": "all",
        "layer_bands": {"syntax": [0, 9], "knowledge": [10, 20], "output": [21, 31]},
        "loaded": {"browse": true, "inference": false},
    })
    .to_string()
}

/// Stand up a mockito server with a `/v1/stats` mock, then `USE` it so
/// the returned session is `Backend::Remote`. Returns the live server
/// (kept alive by the caller) and the connected session.
fn connect() -> (mockito::ServerGuard, Session) {
    let mut server = mockito::Server::new();
    server
        .mock("GET", ENDPOINT_STATS)
        .with_status(200)
        .with_header("content-type", "application/json")
        .with_body(stats_body())
        // Stats may be re-probed by STATS statements as well as the
        // initial USE handshake.
        .expect_at_least(1)
        .create();

    let url = server.url();
    let mut session = Session::new();
    run(&mut session, &format!(r#"USE REMOTE "{url}";"#)).expect("USE REMOTE");
    (server, session)
}

/// Parse + execute one statement through the public API.
fn run(session: &mut Session, sql: &str) -> Result<Vec<String>, String> {
    let parsed = parser::parse(sql).map_err(|e| format!("parse {sql:?}: {e}"))?;
    session
        .execute(&parsed)
        .map_err(|e| format!("execute {sql:?}: {e}"))
}

/// Execute, asserting Ok, and return the joined output.
fn ok_joined(session: &mut Session, sql: &str) -> String {
    run(session, sql)
        .unwrap_or_else(|e| panic!("expected Ok for {sql:?}: {e}"))
        .join("\n")
}

//    only, knn-override, top_tokens, error) ────────────────────────

/// Escape a path for embedding in an LQL double-quoted string literal.
fn sql_path(p: &std::path::Path) -> String {
    p.display().to_string().replace('\\', "\\\\")
}

mod connection_handshake;
mod explain_infer_trace_predictions_first_at;
mod infer_walk_ffn_compare_knn_override_defa;
mod session_scope;
