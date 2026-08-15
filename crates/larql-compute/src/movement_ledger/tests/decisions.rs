use super::*;
use crate::movement_ledger::bytes::COUNTER_LOCK;
use crate::movement_ledger::Tier;

fn movement(semantic: u64, physical: u64) -> OperandMovement {
    OperandMovement::fully_consumed(semantic, physical, Tier::Dram)
}

/// An untouched window has no skip rate to report. `None`, not `0.0` —
/// "the policy declined every opportunity" and "there were no
/// opportunities" are different facts, and only one of them is evidence
/// about a policy.
#[test]
fn an_unmeasured_window_has_no_rate() {
    let d = DecisionCounts::default();
    assert!(!d.is_measured());
    assert_eq!(d.skip_rate(), None);
    assert_eq!(d.avoided_share(0), None);
}

/// The identity every reader depends on.
#[test]
fn requested_partitions_into_executed_and_skipped() {
    let _g = COUNTER_LOCK.lock().unwrap_or_else(|e| e.into_inner());
    reset();
    for _ in 0..7 {
        record_executed();
    }
    for _ in 0..3 {
        record_skipped(&movement(100, 120));
    }
    let d = snapshot();
    assert_eq!(d.requested, 10);
    assert_eq!(d.executed, 7);
    assert_eq!(d.skipped, 3);
    assert!(d.is_consistent());
    assert_eq!(d.skip_rate(), Some(0.3));
    reset();
}

/// Avoided bytes accumulate from the movement the operation WOULD have
/// generated, both classes kept apart exactly as they are on the byte
/// side.
#[test]
fn avoided_bytes_keep_semantic_and_physical_apart() {
    let _g = COUNTER_LOCK.lock().unwrap_or_else(|e| e.into_inner());
    reset();
    record_skipped(&movement(1_000, 1_200));
    record_skipped(&movement(1_000, 1_200));
    let d = snapshot();
    assert_eq!(d.semantic_avoided, 2_000);
    assert_eq!(d.physical_avoided, 2_400);
    reset();
}

/// `avoided_share` divides by what the canonical arm WOULD have moved,
/// not by what this arm did. Dividing by the touched total alone would
/// report a share above 1.0 as soon as more than half the groups were
/// skipped.
#[test]
fn avoided_share_divides_by_the_canonical_total() {
    let mut d = DecisionCounts::default();
    d.accumulate(&DecisionCounts {
        requested: 4,
        executed: 1,
        skipped: 3,
        semantic_avoided: 300,
        physical_avoided: 300,
    });
    // One group ran, moving 100; three were skipped, avoiding 300.
    assert_eq!(d.avoided_share(100), Some(0.75));
}

/// An instrumentation bug — a site that recorded a decision outside
/// `resolve_expert_group` — is detectable rather than silently absorbed.
#[test]
fn an_inconsistent_count_is_detectable() {
    let d = DecisionCounts {
        requested: 5,
        executed: 2,
        skipped: 2,
        ..Default::default()
    };
    assert!(!d.is_consistent());
}

/// Deltas saturate, so a counter reset between two reads yields zero
/// rather than a huge bogus window — the same contract `ByteMovement`
/// uses, for the same reason.
#[test]
fn delta_saturates_on_a_reset() {
    let later = DecisionCounts {
        requested: 3,
        executed: 3,
        ..Default::default()
    };
    let earlier = DecisionCounts {
        requested: 10,
        executed: 10,
        ..Default::default()
    };
    let d = earlier.delta(&later);
    assert_eq!(d.requested, 0);
    assert_eq!(d.executed, 0);
}

/// A window's delta is the work done inside it, not the process total.
#[test]
fn delta_isolates_a_window() {
    let _g = COUNTER_LOCK.lock().unwrap_or_else(|e| e.into_inner());
    reset();
    record_executed();
    record_skipped(&movement(10, 12));
    let opened = snapshot();

    record_skipped(&movement(10, 12));
    record_executed();
    let closed = snapshot();

    let window = opened.delta(&closed);
    assert_eq!(window.requested, 2);
    assert_eq!(window.executed, 1);
    assert_eq!(window.skipped, 1);
    assert_eq!(window.physical_avoided, 12);
    reset();
}

/// Accumulation folds field by field, so a steady-state total built from
/// a stream of windows matches one built from a single window of the same
/// content.
#[test]
fn accumulate_folds_every_field() {
    let mut total = DecisionCounts::default();
    let one = DecisionCounts {
        requested: 2,
        executed: 1,
        skipped: 1,
        semantic_avoided: 5,
        physical_avoided: 6,
    };
    total.accumulate(&one);
    total.accumulate(&one);
    assert_eq!(
        total,
        DecisionCounts {
            requested: 4,
            executed: 2,
            skipped: 2,
            semantic_avoided: 10,
            physical_avoided: 12,
        }
    );
}
