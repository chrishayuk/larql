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
    let want = 10.0 / measured_rate_gbps(PhysicalProjectionPlan::Q4xQ8).expect("a measured plan")
        * 1e3
        + 5.0 / measured_rate_gbps(PhysicalProjectionPlan::Q8xQ8).expect("a measured plan") * 1e3
        + 0.34 / measured_rate_gbps(PhysicalProjectionPlan::FusedBf16).expect("a measured plan")
            * 1e3;
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

/// Every plan the ledger can tally must be priced. The four arms below
/// are the ones no measured rung exercises — the oracle, the cached
/// BLAS path, Q4-against-f32, and the bf16 control — and an unpriced
/// arm would either not compile or, worse, return some neighbour's
/// rate and make a plan look cheaper than it is.
#[test]
fn every_arithmetic_carries_a_rate() {
    let all = [
        PhysicalProjectionPlan::ScalarF32,
        PhysicalProjectionPlan::BlasF32,
        PhysicalProjectionPlan::FusedBf16,
        PhysicalProjectionPlan::FusedQ8,
        PhysicalProjectionPlan::FusedQ4,
        PhysicalProjectionPlan::Q8xQ8,
        PhysicalProjectionPlan::Q4xQ8,
        PhysicalProjectionPlan::Bf16xQ8,
    ];
    for p in all {
        let r = measured_rate_gbps(p).expect("a measured plan");
        assert!(r > 0.0, "{p:?} has no positive streaming rate");
    }
    // The oracle is the slow literal transcription and the cached BLAS
    // path the fastest; pinning the ordering catches an arm that was
    // given a neighbour's number.
    assert!(
        measured_rate_gbps(PhysicalProjectionPlan::ScalarF32).expect("a measured plan")
            < measured_rate_gbps(PhysicalProjectionPlan::FusedQ4).expect("a measured plan"),
        "the scalar oracle must be the slowest arm"
    );
    assert!(
        measured_rate_gbps(PhysicalProjectionPlan::BlasF32).expect("a measured plan")
            > measured_rate_gbps(PhysicalProjectionPlan::Q8xQ8).expect("a measured plan"),
        "cache-resident sgemv must out-rate the streaming integer kernel"
    );
    // CPU-4A's finding, and the one rate that surprises: Q4 against an
    // f32 activation is SLOWER than Q8, because the kernel was already
    // conversion-bound and the nibble split adds to it.
    assert!(
        measured_rate_gbps(PhysicalProjectionPlan::FusedQ4).expect("a measured plan")
            < measured_rate_gbps(PhysicalProjectionPlan::FusedQ8).expect("a measured plan"),
        "CPU-4A: Q4 x F32 is slower than Q8 x F32, not faster"
    );
    // The control streams bf16 bytes, so it must be priced at the bf16
    // rate exactly — a control priced differently would not be one.
    assert_eq!(
        measured_rate_gbps(PhysicalProjectionPlan::Bf16xQ8).expect("a measured plan"),
        measured_rate_gbps(PhysicalProjectionPlan::FusedBf16).expect("a measured plan"),
        "the A1 control runs the bf16 kernel and must carry its rate"
    );
}

/// An arithmetic the token never ran carries no bytes. Zero rather than
/// a panic or a skipped row, because an exception-set search subtracts
/// `bytes_for` against plans that may legitimately be absent.
#[test]
fn bytes_for_an_absent_arithmetic_is_zero() {
    let b = budget(&[(PhysicalProjectionPlan::Q8xQ8, tally(1_000_000, 4))]);
    assert_eq!(b.bytes_for(PhysicalProjectionPlan::Q8xQ8), 1_000_000);
    assert_eq!(
        b.bytes_for(PhysicalProjectionPlan::FusedQ4),
        0,
        "a plan with no calls must price as zero bytes, not panic"
    );
}

/// A plan that ran nothing is dropped rather than reported at zero: a
/// table of eight arithmetics where six are empty invites reading the
/// empty ones as measurements.
#[test]
fn a_plan_with_no_calls_is_not_a_row() {
    let b = budget(&[
        (PhysicalProjectionPlan::Q8xQ8, tally(1_000_000, 2)),
        (PhysicalProjectionPlan::FusedQ4, tally(0, 0)),
    ]);
    assert_eq!(b.rows.len(), 1, "the zero-call plan must not appear");
    assert_eq!(b.rows[0].plan, PhysicalProjectionPlan::Q8xQ8);
}

/// The floor guards division. A non-positive token time would otherwise
/// divide to infinity and render as a throughput headline.
#[test]
fn a_non_positive_token_time_predicts_zero_not_infinity() {
    let empty = budget(&[]);
    assert_eq!(empty.predicted_ms, 0.0);
    assert_eq!(
        predicted_tokens_per_second(&empty, 0.0),
        0.0,
        "must floor at zero rather than divide by zero"
    );
    assert_eq!(
        predicted_tokens_per_second(&empty, -5.0),
        0.0,
        "a negative floor is nonsense input and must not produce a rate"
    );
}
