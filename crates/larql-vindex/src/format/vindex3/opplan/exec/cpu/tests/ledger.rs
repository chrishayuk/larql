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
    l.record(PhysicalProjectionPlan::FusedBf16, 1_000, 12, 0);
    l.record(PhysicalProjectionPlan::FusedBf16, 2_000, 12, 0);
    l.record(PhysicalProjectionPlan::BlasF32, 40, 1, 0);

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
        l.record(*plan, i + 1, 1, 0);
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
        l.record(plan, 7, 3, 0);
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
    l.record(PhysicalProjectionPlan::FusedQ8, 4_096, 6, 0);
    assert_eq!(
        l.get(PhysicalProjectionPlan::FusedQ8),
        crate::format::vindex3::opplan::exec::cpu::PlanTally {
            calls: 1,
            bytes: 4_096,
            slabs: 6,
            // A single-position `record` is one position and not grouped.
            grouped: 0,
            positions: 1,
            nanos: 0,
            nanos_many: 0,
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

/// The realised group width is what turns a disappointing CPU-7C clock
/// into a diagnosis. At 1.0 the eligible projections never grouped and
/// the timing is measuring something else entirely, so the number has to
/// be right in both directions — and it must not divide by zero on a
/// plan that never ran.
#[test]
fn the_group_width_is_positions_per_call_and_zero_when_nothing_ran() {
    use super::super::ledger::Call;

    let l = ProjectionLedger::default();
    assert_eq!(
        l.get(PhysicalProjectionPlan::Q8xQ8).group_width(),
        0.0,
        "a plan with no calls must report 0.0, not divide by zero"
    );

    // Two calls, four positions each: width 4.
    for _ in 0..2 {
        l.record_many(
            PhysicalProjectionPlan::Q8xQ8,
            Call {
                bytes: 100,
                slabs: 1,
                positions: 4,
                grouped: true,
                nanos: 10,
                nanos_many: 10,
            },
        );
    }
    assert_eq!(l.get(PhysicalProjectionPlan::Q8xQ8).group_width(), 4.0);

    // One more call serving a single position drags the mean down — the
    // realised width is an average over calls, not a maximum.
    l.record_many(
        PhysicalProjectionPlan::Q8xQ8,
        Call {
            bytes: 100,
            slabs: 1,
            positions: 1,
            grouped: false,
            nanos: 10,
            nanos_many: 10,
        },
    );
    let t = l.get(PhysicalProjectionPlan::Q8xQ8);
    assert_eq!(t.calls, 3);
    assert_eq!(t.positions, 9);
    assert_eq!(t.group_width(), 3.0);
}

/// `grouped` is the KERNEL's own answer, never inferred from
/// `positions > 1` — the looping default also serves several positions
/// per call, and counting that as grouped would make the ledger agree
/// with the hope rather than with the machine.
#[test]
fn grouped_counts_the_kernels_answer_not_the_position_count() {
    use super::super::ledger::Call;

    let l = ProjectionLedger::default();
    // Four positions, but the kernel says it looped.
    l.record_many(
        PhysicalProjectionPlan::Q8xQ8,
        Call {
            bytes: 10,
            slabs: 1,
            positions: 4,
            grouped: false,
            nanos: 1,
            nanos_many: 1,
        },
    );
    assert_eq!(
        l.get(PhysicalProjectionPlan::Q8xQ8).grouped,
        0,
        "positions > 1 must not imply grouped"
    );

    l.record_many(
        PhysicalProjectionPlan::Q8xQ8,
        Call {
            bytes: 10,
            slabs: 1,
            positions: 4,
            grouped: true,
            nanos: 1,
            nanos_many: 1,
        },
    );
    assert_eq!(l.get(PhysicalProjectionPlan::Q8xQ8).grouped, 1);
}

/// Projection nanoseconds are the NUMERATOR of `g`; the denominator is
/// the step's wall time, which the ledger cannot see. Summing across
/// plans is all it may do, and the multi-position share travels beside
/// the total rather than replacing it.
#[test]
fn projection_nanos_sums_across_plans_and_keeps_the_many_share_apart() {
    use super::super::ledger::Call;

    let l = ProjectionLedger::default();
    l.record_many(
        PhysicalProjectionPlan::Q8xQ8,
        Call {
            bytes: 1,
            slabs: 1,
            positions: 2,
            grouped: true,
            nanos: 500,
            nanos_many: 500,
        },
    );
    l.record_many(
        PhysicalProjectionPlan::FusedBf16,
        Call {
            bytes: 1,
            slabs: 1,
            positions: 1,
            grouped: false,
            nanos: 300,
            nanos_many: 0,
        },
    );

    let (nanos, nanos_many) = l.projection_nanos();
    assert_eq!(nanos, 800, "every plan's time must sum");
    assert_eq!(
        nanos_many, 500,
        "only the multi-position entry contributes to the many share"
    );
    assert!(
        nanos_many < nanos,
        "the share must stay a share, never the whole"
    );
}

/// Time is attributed to the class the ISSUING thread was inside. The
/// guard restores the enclosing class on drop, so nesting cannot leak a
/// class into its parent's total.
#[test]
fn time_lands_in_the_site_the_issuing_thread_declared() {
    use super::super::ledger::{current_site, in_site, Call, Site};

    let l = ProjectionLedger::default();
    let call = |nanos| Call {
        bytes: 1,
        slabs: 1,
        positions: 1,
        grouped: false,
        nanos,
        nanos_many: nanos,
    };

    {
        let _g = in_site(Site::Ffn);
        assert_eq!(current_site(), Site::Ffn);
        l.record_many(PhysicalProjectionPlan::Q8xQ8, call(700));
        {
            let _inner = in_site(Site::Attention);
            l.record_many(PhysicalProjectionPlan::Q8xQ8, call(200));
        }
        // The inner guard dropped: the enclosing class must be restored,
        // or every later call in this scope is misattributed.
        assert_eq!(current_site(), Site::Ffn, "nesting leaked a class");
    }
    assert_eq!(current_site(), Site::Unclassified);

    assert_eq!(l.site(Site::Ffn).nanos, 700);
    assert_eq!(l.site(Site::Attention).nanos, 200);
    assert_eq!(
        l.site(Site::Recurrent),
        Default::default(),
        "a class nothing ran in must stay empty"
    );
}

/// Each class names itself, and the names are distinct — the name is
/// what a `g` breakdown is read from, and two classes sharing one would
/// merge their shares under a single row.
#[test]
fn every_site_has_a_distinct_name() {
    use super::super::ledger::Site;

    let mut names: Vec<&str> = Site::ALL.iter().map(|s| s.name()).collect();
    assert_eq!(names.len(), 4);
    names.sort_unstable();
    names.dedup();
    assert_eq!(names.len(), 4, "two classes share a name");

    // Distinct bits too: the mask is how a class-selective arm is
    // expressed, and a collision would switch two classes at once.
    let mut bits: Vec<u8> = Site::ALL.iter().map(|s| s.bit()).collect();
    bits.sort_unstable();
    bits.dedup();
    assert_eq!(bits.len(), 4, "two classes share a mask bit");
}
