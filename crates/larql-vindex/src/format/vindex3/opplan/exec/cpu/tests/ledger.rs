//! The ledger must count what ran, and reset to nothing.
//!
//! Every test here uses a LOCAL `ProjectionLedger` rather than the
//! process-wide one. The global is shared with every other test in the
//! suite and `reset` is destructive, so a test that reached for it would
//! be racing the thing it was trying to measure — and would pass or fail
//! depending on what else happened to be running.

use super::super::ledger::{ledger, ProjectionLedger};
use super::super::physical::PhysicalProjectionPlan;

/// Every plan the ledger has a slot for. Adding a plan without adding it
/// here would leave the new slot untested, so `all_enumerates_every_plan`
/// checks the two agree in length as well as in content.
const PLANS: [PhysicalProjectionPlan; 8] = [
    PhysicalProjectionPlan::ScalarF32,
    PhysicalProjectionPlan::BlasF32,
    PhysicalProjectionPlan::FusedBf16,
    PhysicalProjectionPlan::FusedQ8,
    PhysicalProjectionPlan::FusedQ4,
    // The integer arms. Each has its OWN slot on purpose: an arm folded
    // into another's counter would let a byte census agree with itself
    // while describing a mixture of two arithmetics.
    PhysicalProjectionPlan::Q8xQ8,
    PhysicalProjectionPlan::Q4xQ8,
    PhysicalProjectionPlan::Bf16xQ8,
];

#[test]
fn each_plan_is_counted_separately() {
    let l = ProjectionLedger::default();
    l.record(PhysicalProjectionPlan::FusedBf16, 1_000, 12);
    l.record(PhysicalProjectionPlan::FusedBf16, 2_000, 12);
    l.record(PhysicalProjectionPlan::BlasF32, 40, 1);

    let fused = l.get(PhysicalProjectionPlan::FusedBf16);
    assert_eq!(fused.calls, 2);
    assert_eq!(fused.bytes, 3_000);
    assert_eq!(fused.slabs, 24);

    let blas = l.get(PhysicalProjectionPlan::BlasF32);
    assert_eq!((blas.calls, blas.bytes, blas.slabs), (1, 40, 1));
    assert_eq!(l.get(PhysicalProjectionPlan::ScalarF32), Default::default());
    assert_eq!(l.total_bytes(), 3_040);
}

/// `all()` enumerates every plan, so a reader cannot silently stop
/// covering one.
#[test]
fn all_enumerates_every_plan() {
    let l = ProjectionLedger::default();
    for (i, plan) in PLANS.iter().enumerate() {
        l.record(*plan, i + 1, 1);
    }
    let seen: Vec<_> = l.all().iter().map(|(p, t)| (*p, t.bytes)).collect();
    let want: Vec<_> = PLANS
        .iter()
        .enumerate()
        .map(|(i, p)| (*p, (i + 1) as u64))
        .collect();
    assert_eq!(
        seen, want,
        "`all` and the test's plan list disagree — a plan with a ledger slot and no test is a          tally nothing checks"
    );
    assert_eq!(l.total_bytes(), (1..=PLANS.len() as u64).sum::<u64>());
}

/// Reset zeroes every plan, not just the one that was busiest.
///
/// A partial reset is the failure that would matter: the CLI resets
/// before the step it prices, so a leftover count would silently fold the
/// weight load and every warm-up step into a per-token number.
#[test]
fn reset_clears_every_plan() {
    let l = ProjectionLedger::default();
    for plan in PLANS {
        l.record(plan, 7, 3);
    }
    assert_eq!(l.total_bytes(), 7 * PLANS.len() as u64);
    l.reset();
    for plan in PLANS {
        assert_eq!(
            l.get(plan),
            Default::default(),
            "{plan:?} survived the reset"
        );
    }
    assert_eq!(l.total_bytes(), 0);
}

/// The process-wide ledger exists and is the same one every time.
#[test]
fn the_shared_ledger_is_one_ledger() {
    assert!(std::ptr::eq(ledger(), ledger()));
}

/// A plan the policy cannot yet produce still has a working slot.
///
/// `FusedQ8` is reachable by OBSERVATION before it is reachable by
/// `choose`, so nothing in a decode writes to its tally yet. Without this
/// the slot would be untested until the day it was first used, which is
/// the worst day to discover it aliases another.
#[test]
fn a_slot_works_before_the_policy_can_reach_it() {
    // `new` rather than `default`: it is what the process static is
    // built from, so a test that only ever used `default` would leave the
    // constructor the shipped ledger actually uses unexercised.
    let l = ProjectionLedger::new();
    assert_eq!(l.total_bytes(), 0, "a fresh ledger has counted nothing");
    l.record(PhysicalProjectionPlan::FusedQ8, 4_096, 6);
    assert_eq!(
        l.get(PhysicalProjectionPlan::FusedQ8),
        crate::format::vindex3::opplan::exec::cpu::PlanTally {
            calls: 1,
            bytes: 4_096,
            slabs: 6,
        }
    );
    for other in PLANS
        .iter()
        .copied()
        .filter(|p| *p != PhysicalProjectionPlan::FusedQ8)
    {
        assert_eq!(
            l.get(other),
            Default::default(),
            "{other:?} was written by a FusedQ8 record — the slots alias"
        );
    }
}
