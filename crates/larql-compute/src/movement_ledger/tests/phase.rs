use super::*;

/// No scope installed means no phase — a refusal, not a guessed default.
#[test]
fn no_scope_means_no_phase() {
    reset_for_test();
    assert_eq!(current_phase(), None);
}

/// A held scope reports its phase for as long as it is alive.
#[test]
fn scope_reports_its_phase_while_held() {
    reset_for_test();
    let scope = PhaseScope::new(Phase::Decode);
    assert_eq!(current_phase(), Some(Phase::Decode));
    drop(scope);
    assert_eq!(
        current_phase(),
        None,
        "drop restores the prior (empty) state"
    );
}

/// Nested scopes restore the OUTER phase on drop, not `None` — a nested
/// call must not blind whatever the caller was already attributing.
#[test]
fn nested_scope_restores_the_outer_phase_on_drop() {
    reset_for_test();
    let outer = PhaseScope::new(Phase::Prefill);
    assert_eq!(current_phase(), Some(Phase::Prefill));
    {
        let _inner = PhaseScope::new(Phase::Decode);
        assert_eq!(current_phase(), Some(Phase::Decode));
    }
    assert_eq!(
        current_phase(),
        Some(Phase::Prefill),
        "inner scope's drop restores the outer phase, not None"
    );
    drop(outer);
    assert_eq!(current_phase(), None);
}

/// The two phases carry distinct labels — a report that printed the same
/// tag for both would be no report at all.
#[test]
fn labels_are_distinct() {
    assert_ne!(Phase::Prefill.label(), Phase::Decode.label());
}
