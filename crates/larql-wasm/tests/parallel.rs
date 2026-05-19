//! Parallel (multi-threaded) integration tests.
//!
//! Only compiled when the `parallel` feature is active.
//! Runs in a headless Firefox browser — wasm-bindgen-rayon spawns Web Workers
//! which require the browser `self` global; Node.js does not expose it.
//!
//! `wasm-pack test crates/larql-wasm --firefox --headless --features parallel`

#![cfg(feature = "parallel")]

use wasm_bindgen_test::*;

use larql_wasm::GraphSession;

/// The thread pool is initialised by the browser runner before each test.
/// Tests verify that the `parallel` feature compiles and the exported symbols
/// are callable under a real multi-threaded wasm environment.
#[wasm_bindgen_test]
fn parallel_session_new_and_edge_count() {
    let s = GraphSession::new();
    assert_eq!(s.edge_count(), 0);
}

#[wasm_bindgen_test]
fn parallel_benchmark_pagerank_runs() {
    let ms = larql_wasm::benchmark_pagerank_parallel(100, 2);
    assert!(ms >= 0.0);
}

#[wasm_bindgen_test]
fn parallel_benchmark_bfs_runs() {
    let ms = larql_wasm::benchmark_bfs_parallel(100, 2);
    assert!(ms >= 0.0);
}
