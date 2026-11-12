//! Unit tests for session identity, lifetime, and the read-only
//! projection. The model-coupled paths (`get_or_create`, `apply_patch`)
//! are exercised end-to-end in `tests/test_unit_state.rs` and
//! `tests/test_http_session.rs`; the HTTP surface in
//! `tests/test_http_sessions.rs`.

mod header;
mod lease;
mod lifecycle;
mod view;

use super::*;
use std::time::{Duration, Instant};

/// Short TTL so expiry can be expressed by advancing an explicit clock.
const TEST_TTL_SECS: u64 = 10;
/// Runtime binding the test sessions are created against.
const TEST_MODEL: &str = "m-test";

fn tiny_index() -> larql_vindex::VectorIndex {
    let hidden = 4;
    let gate = larql_vindex::ndarray::Array2::<f32>::zeros((2, hidden));
    larql_vindex::VectorIndex::new(vec![Some(gate)], vec![None], 1, hidden)
}

/// Bind a session under `id` whose clock reads `created`.
fn insert_session(sm: &SessionManager, id: &str, created: Instant) {
    let mut map = sm.sessions_blocking_write();
    sm.bind_in_guard(&mut map, id, TEST_MODEL, created);
}

/// Give `id` a patch overlay, so patch-bearing projections have something
/// to report without needing a `LoadedModel`.
fn give_overlay(sm: &SessionManager, id: &str, patch_names: &[&str]) {
    let mut map = sm.sessions_blocking_write();
    let session = map.get_mut(id).expect("session bound");
    let overlay = session.overlay_mut(tiny_index);
    for name in patch_names {
        overlay
            .try_apply_patch(named_patch(name))
            .expect("patch applies");
    }
}

fn named_patch(name: &str) -> larql_vindex::VindexPatch {
    larql_vindex::VindexPatch {
        version: 1,
        base_model: TEST_MODEL.into(),
        base_checksum: None,
        created_at: "2026-08-22".into(),
        description: Some(name.to_string()),
        author: None,
        tags: vec![],
        operations: vec![larql_vindex::PatchOp::Delete {
            layer: 0,
            feature: 0,
            reason: None,
        }],
    }
}

fn block_on<F: std::future::Future>(f: F) -> F::Output {
    tokio::runtime::Builder::new_current_thread()
        .build()
        .expect("current-thread runtime")
        .block_on(f)
}
