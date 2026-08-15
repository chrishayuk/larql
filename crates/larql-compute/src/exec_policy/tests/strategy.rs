use super::*;
use crate::movement_ledger::Phase;

/// The two strategies are distinct values, not two spellings of one. A
/// `Skip` that compared equal to `Canonical` would let every dispatch
/// site's `== ExecutionStrategy::Skip` guard silently never fire.
#[test]
fn canonical_and_skip_are_distinct() {
    assert_ne!(ExecutionStrategy::Canonical, ExecutionStrategy::Skip);
    assert!(!ExecutionStrategy::Canonical.is_skip());
    assert!(ExecutionStrategy::Skip.is_skip());
}

/// Labels are what a report and a test failure message print. They must
/// not collide, or a run under a skip policy reads as a canonical one.
#[test]
fn labels_are_distinct_and_stable() {
    assert_eq!(ExecutionStrategy::Canonical.label(), "canonical");
    assert_eq!(ExecutionStrategy::Skip.label(), "skip");
}

/// The site is `Copy`, because it is built once per layer per token on
/// the decode hot path and passed by reference to a policy that may hold
/// nothing. A site that allocated would put an allocation between the
/// router and the expert kernels.
#[test]
fn site_is_copy_and_carries_the_full_address() {
    let site = ExpertGroupSite {
        layer: 21,
        phase: Some(Phase::Decode),
        step: Some(7),
        slots: 4,
    };
    let copied = site;
    assert_eq!(copied, site);
    assert_eq!(copied.layer, 21);
    assert_eq!(copied.slots, 4);
}
