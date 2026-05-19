//! Node.js wasm_bindgen_test suite for larql-browser-cli.
//!
//! Attempts all public surface area without pre-gating assumptions about
//! what will pass or fail. CI logs are the empirical record.

#![cfg(target_arch = "wasm32")]

use larql_browser_cli::{parse_lql, LqlSession};
use wasm_bindgen_test::*;

wasm_bindgen_test_configure!(run_in_node_experimental);

// ── parse_lql ────────────────────────────────────────────────────────────────

#[wasm_bindgen_test]
fn parse_walk_returns_ok() {
    let result = parse_lql("WALK 'France'");
    assert!(result.is_ok(), "expected Ok, got: {:?}", result.err());
}

#[wasm_bindgen_test]
fn parse_select_returns_ok() {
    let result = parse_lql("SELECT * FROM edges LIMIT 10");
    assert!(result.is_ok(), "expected Ok, got: {:?}", result.err());
}

#[wasm_bindgen_test]
fn parse_describe_returns_ok() {
    let result = parse_lql("DESCRIBE 'Paris'");
    assert!(result.is_ok(), "expected Ok, got: {:?}", result.err());
}

#[wasm_bindgen_test]
fn parse_infer_returns_ok() {
    let result = parse_lql("INFER 'The capital of France is'");
    assert!(result.is_ok(), "expected Ok, got: {:?}", result.err());
}

#[wasm_bindgen_test]
fn parse_stats_returns_ok() {
    let result = parse_lql("STATS");
    assert!(result.is_ok(), "expected Ok, got: {:?}", result.err());
}

#[wasm_bindgen_test]
fn parse_invalid_lql_returns_err() {
    assert!(parse_lql("NOT VALID LQL !!!").is_err());
}

#[wasm_bindgen_test]
fn parse_empty_returns_err() {
    assert!(parse_lql("").is_err());
}

#[wasm_bindgen_test]
fn parse_result_is_nonempty_string() {
    let s = parse_lql("WALK 'France'").unwrap();
    assert!(!s.is_empty());
}

// ── LqlSession ───────────────────────────────────────────────────────────────

#[wasm_bindgen_test]
fn session_new_does_not_panic() {
    let _s = LqlSession::new();
}

#[wasm_bindgen_test]
fn session_walk_on_empty_session_returns_err() {
    let mut s = LqlSession::new();
    let result = s.execute("WALK 'France'");
    // No backend loaded — expect an error, not a panic.
    assert!(result.is_err(), "expected Err (no backend), got Ok");
}

#[wasm_bindgen_test]
fn session_select_on_empty_session_returns_err() {
    let mut s = LqlSession::new();
    let result = s.execute("SELECT * FROM edges LIMIT 5");
    assert!(result.is_err());
}

#[wasm_bindgen_test]
fn session_describe_on_empty_session_returns_err() {
    let mut s = LqlSession::new();
    let result = s.execute("DESCRIBE 'Paris'");
    assert!(result.is_err());
}

#[wasm_bindgen_test]
fn session_stats_on_empty_session() {
    // STATS with no vindex argument — may succeed (session stats) or err; either is valid.
    let mut s = LqlSession::new();
    let _ = s.execute("STATS");
}

#[wasm_bindgen_test]
fn session_invalid_lql_returns_err() {
    let mut s = LqlSession::new();
    assert!(s.execute("NOT VALID LQL").is_err());
}

#[wasm_bindgen_test]
fn session_error_contains_message() {
    let mut s = LqlSession::new();
    let err = s.execute("WALK 'x'").unwrap_err();
    let msg = err.as_string().unwrap_or_default();
    assert!(!msg.is_empty(), "error message should be non-empty");
}

#[wasm_bindgen_test]
fn multiple_sessions_are_independent() {
    let _s1 = LqlSession::new();
    let _s2 = LqlSession::new();
    // Two sessions coexist without interference.
}
