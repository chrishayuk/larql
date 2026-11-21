//! COMPACT MAJOR persistence (Backend::Vindex.memit_store wiring)
//! TRACE
//! REBALANCE
//! COMPACT MINOR / MAJOR / SHOW COMPACT STATUS
//! SHOW ENTITIES
//! REMOVE PATCH

use super::*;

#[test]
fn memit_store_mut_unavailable_without_backend() {
    let mut session = Session::new();
    assert!(matches!(
        session.memit_store_mut().unwrap_err(),
        LqlError::NoBackend
    ));
}

#[test]
fn memit_store_mut_returns_empty_store_on_fresh_vindex() {
    let (mut session, dir) = vindex_session("memit_empty");
    let store = session
        .memit_store_mut()
        .expect("vindex backend has memit_store");
    assert_eq!(store.num_cycles(), 0);
    assert_eq!(store.total_facts(), 0);
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn show_compact_status_reflects_live_memit_store() {
    // Regression: prior implementation hardcoded "0 facts across 0
    // cycles" regardless of session state. After we add a synthetic
    // cycle, SHOW COMPACT STATUS should report it.
    let (mut session, dir) = vindex_session("compact_status_live");

    // Synthetic vindex has hidden_dim = 4 → MEMIT-supported branch is
    // disabled (requires ≥ 1024). We don't run the live-count check
    // when the L2 line is gated. Skip cleanly in that case.
    let initial = session
        .execute(&parser::parse("SHOW COMPACT STATUS;").unwrap())
        .expect("show compact status");
    let initial_joined = initial.join("\n");
    if initial_joined.contains("not available") {
        let _ = std::fs::remove_dir_all(&dir);
        return;
    }

    // Push a cycle directly through the backend accessor.
    {
        let store = session.memit_store_mut().expect("vindex backend");
        store.add_cycle(
            7,
            vec![larql_vindex::MemitFact {
                entity: "X".into(),
                relation: "y".into(),
                target: "Z".into(),
                key: larql_vindex::ndarray::Array1::zeros(4),
                decomposed_down: larql_vindex::ndarray::Array1::zeros(4),
                reconstruction_cos: 1.0,
            }],
            0.0,
            1.0,
            0.0,
        );
    }

    let after = session
        .execute(&parser::parse("SHOW COMPACT STATUS;").unwrap())
        .expect("show compact status");
    let joined = after.join("\n");
    assert!(
        joined.contains("1 fact(s) across 1 cycle(s)"),
        "expected live MEMIT counts, got: {joined}"
    );

    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn memit_store_persists_added_cycles() {
    // Verifies the wiring change from item #5: facts pushed into the
    // session-level MemitStore survive subsequent accesses. The
    // production COMPACT MAJOR pipeline writes through the same path.
    let (mut session, dir) = vindex_session("memit_persist");
    {
        let store = session.memit_store_mut().expect("vindex backend");
        store.add_cycle(
            33,
            vec![larql_vindex::MemitFact {
                entity: "France".into(),
                relation: "capital".into(),
                target: "Paris".into(),
                key: larql_vindex::ndarray::Array1::zeros(4),
                decomposed_down: larql_vindex::ndarray::Array1::zeros(4),
                reconstruction_cos: 1.0,
            }],
            0.5,
            1.0,
            0.0,
        );
    }
    // Re-borrow to confirm the cycle survived.
    let store = session.memit_store_mut().expect("vindex backend");
    assert_eq!(store.num_cycles(), 1);
    assert_eq!(store.total_facts(), 1);
    let hits = store.lookup("France", "capital");
    assert_eq!(hits.len(), 1);
    assert_eq!(hits[0].target, "Paris");
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn no_backend_trace() {
    let mut session = Session::new();
    let stmt = parser::parse(r#"TRACE "The capital of France is";"#).unwrap();
    assert!(matches!(
        session.execute(&stmt).unwrap_err(),
        LqlError::NoBackend
    ));
}

#[test]
fn trace_on_browse_only_vindex_errors_with_weights_hint() {
    // The synthetic fixture is browse-only; TRACE needs model weights.
    let (mut session, dir) = vindex_session("trace_no_weights");
    let stmt = parser::parse(r#"TRACE "any prompt";"#).unwrap();
    let err = session
        .execute(&stmt)
        .expect_err("TRACE on browse-only vindex should fail");
    match err {
        LqlError::Execution(msg) => {
            assert!(
                msg.contains("TRACE requires model weights"),
                "expected model-weights hint, got: {msg}"
            );
        }
        other => panic!("expected Execution error, got {other:?}"),
    }
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn rebalance_without_backend_is_noop() {
    // REBALANCE short-circuits on empty `installed_edges` BEFORE the
    // backend check (mutation/rebalance.rs:38-43), so it returns Ok
    // with a "no compose-mode installs" message even with no backend.
    // This is the same behaviour as REBALANCE on a fresh vindex.
    let mut session = Session::new();
    let stmt = parser::parse("REBALANCE;").unwrap();
    let out = session
        .execute(&stmt)
        .expect("REBALANCE with empty install set should succeed");
    assert!(
        out.iter()
            .any(|line| line.contains("no compose-mode installs")),
        "expected empty-installs note in: {out:?}"
    );
}

#[test]
fn rebalance_without_compose_installs_is_noop() {
    // With no `installed_edges` registered, REBALANCE returns a
    // single-line note and doesn't touch the overlay.
    let (mut session, dir) = vindex_session("rebalance_empty");
    let stmt = parser::parse("REBALANCE;").unwrap();
    let out = session
        .execute(&stmt)
        .expect("REBALANCE on empty compose set should succeed");
    assert!(
        out.iter()
            .any(|line| line.contains("no compose-mode installs")),
        "expected empty-installs note in: {out:?}"
    );
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn no_backend_compact_minor() {
    let mut session = Session::new();
    let stmt = parser::parse("COMPACT MINOR;").unwrap();
    assert!(matches!(
        session.execute(&stmt).unwrap_err(),
        LqlError::NoBackend
    ));
}

#[test]
fn no_backend_compact_major() {
    // Refused before any work, so before the backend check (Gate B, B6 D2).
    let mut session = Session::new();
    let stmt = parser::parse("COMPACT MAJOR;").unwrap();
    let err = session.execute(&stmt).unwrap_err().to_string();
    assert!(
        err.contains(crate::executor::compact::COMPACT_MAJOR_REFUSAL),
        "{err}"
    );
}

#[test]
fn no_backend_show_compact_status() {
    let mut session = Session::new();
    let stmt = parser::parse("SHOW COMPACT STATUS;").unwrap();
    assert!(matches!(
        session.execute(&stmt).unwrap_err(),
        LqlError::NoBackend
    ));
}

#[test]
fn compact_minor_on_empty_l0_returns_message() {
    let (mut session, dir) = vindex_session("compact_minor_empty");
    let stmt = parser::parse("COMPACT MINOR;").unwrap();
    let out = session
        .execute(&stmt)
        .expect("COMPACT MINOR with empty L0 should succeed");
    assert!(
        out.iter().any(|l| l.contains("L0 is empty")),
        "expected empty-L0 message in: {out:?}"
    );
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn show_compact_status_reports_empty_tiers() {
    let (mut session, dir) = vindex_session("compact_status");
    let stmt = parser::parse("SHOW COMPACT STATUS;").unwrap();
    let out = session
        .execute(&stmt)
        .expect("SHOW COMPACT STATUS should succeed");
    let joined = out.join("\n");
    assert!(joined.contains("L0"), "expected L0 tier: {joined}");
    assert!(joined.contains("L1"), "expected L1 tier: {joined}");
    // The synthetic fixture has 0 overrides; the L0/L1 counts should read 0.
    assert!(
        joined.contains("0 entries") || joined.contains("0 edges"),
        "expected zero counts in: {joined}"
    );
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn no_backend_show_entities() {
    let mut session = Session::new();
    let stmt = parser::parse("SHOW ENTITIES;").unwrap();
    assert!(matches!(
        session.execute(&stmt).unwrap_err(),
        LqlError::NoBackend
    ));
}

#[test]
fn show_entities_scans_synthetic_vindex() {
    // The synthetic fixture seeds content-shaped tokens — SHOW ENTITIES
    // should run cleanly and produce the `Distinct entities …` summary
    // line followed by the tabular header.
    let (mut session, dir) = vindex_session("show_entities_scan");
    let stmt = parser::parse("SHOW ENTITIES LIMIT 20;").unwrap();
    let out = session
        .execute(&stmt)
        .expect("SHOW ENTITIES should succeed");
    let joined = out.join("\n");
    assert!(
        joined.contains("Distinct entities"),
        "expected summary line in: {joined}"
    );
    assert!(
        joined.contains("Entity") && joined.contains("Max Score"),
        "expected tabular header in: {joined}"
    );
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn no_backend_remove_patch() {
    let mut session = Session::new();
    let stmt = parser::parse(r#"REMOVE PATCH "missing.vlp";"#).unwrap();
    assert!(matches!(
        session.execute(&stmt).unwrap_err(),
        LqlError::NoBackend
    ));
}

#[test]
fn remove_patch_unknown_errors_cleanly() {
    let (mut session, dir) = vindex_session("remove_patch_missing");
    let stmt = parser::parse(r#"REMOVE PATCH "never-applied.vlp";"#).unwrap();
    let err = session
        .execute(&stmt)
        .expect_err("REMOVE PATCH should error when no such patch is applied");
    match err {
        LqlError::Execution(msg) => assert!(msg.contains("patch not found")),
        other => panic!("expected Execution error, got {other:?}"),
    }
    let _ = std::fs::remove_dir_all(&dir);
}

/// The overlay accessors refuse a raw-weights binding by pointing at
/// EXTRACT, and an unbound session as having no backend.
#[test]
fn patched_accessors_refuse_without_a_vindex() {
    let mut weights = weight_session();
    let err = weights.require_patched().err().unwrap().to_string();
    assert!(err.contains("requires a vindex"), "{err}");
    let err = weights.require_patched_mut().err().unwrap().to_string();
    assert!(err.contains("mutation requires a vindex"), "{err}");
    assert!(weights.memit_store().is_none());

    let mut unbound = Session::new();
    assert!(matches!(
        unbound.require_patched_mut().err(),
        Some(LqlError::NoBackend)
    ));
    assert!(unbound.memit_store().is_none());
}
