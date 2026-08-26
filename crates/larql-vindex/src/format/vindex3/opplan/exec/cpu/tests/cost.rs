//! The cost model is an instrument, so it is gated like one.
//!
//! It must reproduce the measurement it was built from BEFORE it is
//! allowed to predict a plan nobody has timed. A model that could not
//! recover CPU-4Y's own numbers from CPU-4Y's own byte counts would be
//! predicting something, but not this machine.

use super::super::cost::{
    budget, measured_rate_gbps, predicted_tokens_per_second, SYNTHETIC_TO_REAL,
};
use super::super::ledger::PlanTally;
use super::super::physical::PhysicalProjectionPlan;

/// CPU-4Y's table: bytes per token and the milliseconds they took, in
/// one run on one build.
const CPU_4Y: [(PhysicalProjectionPlan, f64, f64); 4] = [
    (PhysicalProjectionPlan::FusedBf16, 51.20, 420.84),
    (PhysicalProjectionPlan::FusedQ8, 27.20, 332.97),
    (PhysicalProjectionPlan::Q8xQ8, 27.20, 224.75),
    (PhysicalProjectionPlan::Q4xQ8, 14.40, 135.10),
];

fn tally(bytes: u64, calls: u64) -> PlanTally {
    PlanTally {
        calls,
        bytes,
        slabs: calls,
        // The cost model is built from single-position decode, where every
        // call serves exactly one position.
        grouped: 0,
        positions: calls,
        nanos: 0,
        nanos_many: 0,
    }
}

/// **The model reproduces the measurement it was built from.**
#[test]
fn the_cost_model_recovers_the_rung_it_was_fitted_to() {
    for (plan, gb, ms) in CPU_4Y {
        let b = budget(&[(plan, tally((gb * 1e9) as u64, 1))]);
        let err = (b.synthetic_ms - ms).abs() / ms;
        assert!(
            err < 0.01,
            "{plan:?}: model says {:.2} ms where CPU-4Y measured {ms:.2} ({:.2}% off)",
            b.synthetic_ms,
            err * 100.0
        );
    }
}

/// **And it FAILS on known-different input.**
///
/// A model that returned something plausible for any byte count would
/// pass the test above by construction. Halving the bytes of a
/// memory-bound arithmetic must halve its predicted time; if it does
/// not, the model is not reading the bytes.
#[test]
fn the_cost_model_moves_with_the_bytes() {
    let one = budget(&[(PhysicalProjectionPlan::Q4xQ8, tally(14_400_000_000, 1))]);
    let half = budget(&[(PhysicalProjectionPlan::Q4xQ8, tally(7_200_000_000, 1))]);
    let ratio = one.synthetic_ms / half.synthetic_ms;
    assert!(
        (ratio - 2.0).abs() < 1e-6,
        "halving the bytes changed the prediction by {ratio:.4}x, not 2x"
    );
}

/// A mixed plan is priced per arithmetic and summed, which is the whole
/// reason an exception set can be evaluated without benchmarking it.
#[test]
fn a_mixed_plan_is_priced_arithmetic_by_arithmetic() {
    let mixed = budget(&[
        (PhysicalProjectionPlan::Q4xQ8, tally(10_000_000_000, 300)),
        (PhysicalProjectionPlan::Q8xQ8, tally(5_000_000_000, 60)),
        (PhysicalProjectionPlan::FusedBf16, tally(340_000_000, 32)),
    ]);
    let want = 10.0 / measured_rate_gbps(PhysicalProjectionPlan::Q4xQ8) * 1e3
        + 5.0 / measured_rate_gbps(PhysicalProjectionPlan::Q8xQ8) * 1e3
        + 0.34 / measured_rate_gbps(PhysicalProjectionPlan::FusedBf16) * 1e3;
    assert!((mixed.synthetic_ms - want).abs() < 1e-6);
    assert_eq!(mixed.total_bytes, 15_340_000_000);
    assert_eq!(
        mixed.bytes_for(PhysicalProjectionPlan::Q4xQ8),
        10_000_000_000
    );
    // An arithmetic that did not run is absent, not zero: a table of
    // empty rows invites reading them as measurements.
    assert_eq!(mixed.rows.len(), 3);
    assert_eq!(mixed.bytes_for(PhysicalProjectionPlan::ScalarF32), 0);
}

/// The predicted decode speed carries the MEASURED synthetic-to-real
/// correction and the non-projection floor, so a tok/s claim is never
/// the projection alone.
#[test]
fn the_predicted_decode_includes_the_correction_and_the_floor() {
    let b = budget(&[(PhysicalProjectionPlan::Q4xQ8, tally(14_400_000_000, 369))]);
    assert!((b.predicted_ms - b.synthetic_ms * SYNTHETIC_TO_REAL).abs() < 1e-9);

    // CPU-4Y's own projection: ~141 ms real, and a 17-24 ms floor puts
    // the token at 158-166 ms, i.e. 6.0-6.3 tok/s.
    assert!(
        (140.0..143.0).contains(&b.predicted_ms),
        "real projection predicted at {:.1} ms, outside CPU-4Y's ~141",
        b.predicted_ms
    );
    let fast = predicted_tokens_per_second(&b, 17.0);
    let slow = predicted_tokens_per_second(&b, 24.0);
    assert!(
        (5.9..6.4).contains(&fast) && (5.9..6.4).contains(&slow),
        "predicted decode {slow:.2}-{fast:.2} tok/s, outside CPU-4Y's 6.0-6.3"
    );
}
