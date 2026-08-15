use super::*;
use crate::movement_ledger::bytes::{OperandMovement, COUNTER_LOCK};
use crate::movement_ledger::Tier;

/// Every surface has a distinct label — the report is unreadable if two
/// surfaces collide.
#[test]
fn surface_labels_are_distinct() {
    let labels: Vec<_> = ALL_SURFACES.iter().map(|s| s.label()).collect();
    for (i, a) in labels.iter().enumerate() {
        assert!(!a.is_empty());
        for b in labels.iter().skip(i + 1) {
            assert_ne!(a, b);
        }
    }
    assert_eq!(labels.len(), ALL_SURFACES.len());
}

/// An instrumented surface starts SILENT, not covered. Coverage is fired
/// evidence; silence is never read as success.
#[test]
fn instrumented_surface_starts_silent_then_becomes_covered() {
    let _g = COUNTER_LOCK.lock().unwrap_or_else(|e| e.into_inner());
    reset_for_test();
    assert_eq!(Surface::MoeExperts.fired(), 0);
    let before: Vec<_> = states();
    assert!(before
        .iter()
        .any(|(s, st)| *s == Surface::MoeExperts && *st == SurfaceState::Silent));

    record(
        Surface::MoeExperts,
        OperandMovement::fully_consumed(10, 10, Tier::Dram),
    );
    assert!(states()
        .iter()
        .any(|(s, st)| *s == Surface::MoeExperts && *st == SurfaceState::Covered(1)));
    reset_for_test();
    crate::movement_ledger::bytes::reset_for_test();
}

/// An uninstrumented surface can never report as covered, however much
/// the run does — that is the honest admission the report depends on.
#[test]
fn uninstrumented_surfaces_never_report_covered() {
    let _g = COUNTER_LOCK.lock().unwrap_or_else(|e| e.into_inner());
    reset_for_test();
    for (s, st) in states() {
        if !s.is_instrumented() {
            assert_eq!(st, SurfaceState::NotInstrumented, "{}", s.label());
        }
    }
}

/// `record` moves the byte counters AND the coverage evidence together —
/// they must not be updatable independently.
#[test]
fn record_updates_bytes_and_coverage_together() {
    let _g = COUNTER_LOCK.lock().unwrap_or_else(|e| e.into_inner());
    reset_for_test();
    crate::movement_ledger::bytes::reset_for_test();

    let before = crate::movement_ledger::bytes::snapshot();
    record(
        Surface::MoeExperts,
        OperandMovement::fully_consumed(100, 128, Tier::Dram),
    );
    let d = before.delta(&crate::movement_ledger::bytes::snapshot());
    assert_eq!(d.physical_touched, 128);
    assert_eq!(d.semantic_requested, 100);
    assert_eq!(Surface::MoeExperts.fired(), 1);

    reset_for_test();
    crate::movement_ledger::bytes::reset_for_test();
}

/// While any surface is uninstrumented the ledger is not complete, and
/// the rendered line must say so in terms a reader cannot miss.
#[test]
fn partial_coverage_renders_an_explicit_warning() {
    let _g = COUNTER_LOCK.lock().unwrap_or_else(|e| e.into_inner());
    reset_for_test();
    assert!(!is_complete(), "not every surface is instrumented yet");

    let out = render();
    assert!(out.contains("NOT instrumented"));
    assert!(
        out.contains("kv-cache"),
        "kv traffic omission must be named"
    );
    assert!(out.contains("PARTIAL"));
    assert!(out.contains("UNDERSTATED"));
    // The one thing that stays valid under partial coverage is named.
    assert!(out.contains("DELTAS"));
}

/// Before anything fires, the covered list says "none" rather than
/// rendering an empty list that reads like completeness.
#[test]
fn empty_coverage_renders_as_none() {
    let _g = COUNTER_LOCK.lock().unwrap_or_else(|e| e.into_inner());
    reset_for_test();
    assert!(render().contains("covered: none"));
}

/// A silent instrumented surface is reported separately from a missing
/// one — "the path did not run" and "there is no bump site" are
/// different facts.
#[test]
fn silent_and_missing_are_reported_separately() {
    let _g = COUNTER_LOCK.lock().unwrap_or_else(|e| e.into_inner());
    reset_for_test();
    let out = render();
    assert!(out.contains("instrumented but SILENT"));
    assert!(out.contains("NOT instrumented"));
    let silent_idx = out.find("SILENT").unwrap();
    let missing_idx = out.find("NOT instrumented").unwrap();
    assert!(
        silent_idx < missing_idx,
        "report order is covered/silent/missing"
    );
}
