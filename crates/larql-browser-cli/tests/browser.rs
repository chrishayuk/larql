//! Firefox browser wasm_bindgen_test suite for larql-browser-cli.
//!
//! Same coverage as wasm.rs but run in a real browser environment.
//! Gated by the `browser-tests` feature so the serial (Node.js) workflow
//! does not pull in the browser runner configuration.

#![cfg(all(target_arch = "wasm32", feature = "browser-tests"))]

use larql_browser_cli::{parse_lql, LqlSession};
use wasm_bindgen_test::*;

wasm_bindgen_test_configure!(run_in_browser);

#[wasm_bindgen_test]
fn parse_walk_browser() {
    assert!(parse_lql("WALK 'France'").is_ok());
}

#[wasm_bindgen_test]
fn parse_invalid_browser() {
    assert!(parse_lql("GARBAGE INPUT").is_err());
}

#[wasm_bindgen_test]
fn session_new_browser() {
    let _s = LqlSession::new();
}

#[wasm_bindgen_test]
fn session_execute_no_backend_browser() {
    let mut s = LqlSession::new();
    assert!(s.execute("WALK 'Paris'").is_err());
}
