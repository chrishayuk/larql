//! `tests` for [`super`].

use super::*;

#[test]
fn a_p99_over_too_few_observations_is_the_maximum() {
    // The arithmetic behind the whole module: nearest-rank p99 of n
    // values is the largest one until n reaches 100.
    for n in [1u64, 5, 46, 99] {
        let s = TailSupport {
            quantile: 0.99,
            observations: n,
        };
        assert!(
            s.expected_tail_observations() < 1.0,
            "{n} observations cannot put one above p99"
        );
    }
    assert!(
        (TailSupport {
            quantile: 0.99,
            observations: 100
        })
        .expected_tail_observations()
            >= 1.0
    );
}

#[test]
fn the_policy_requires_five_hundred_observations_for_a_p99() {
    let p = TailSupportPolicy::route_cal_1();
    assert_eq!(p.required_observations(0.99), 500);
    assert_eq!(p.required_observations(0.95), 100);
    assert_eq!(p.required_observations(0.999), 5000);
}

#[test]
fn the_measured_route_evidence_is_judged_as_route_cal_1_found_it() {
    let p = TailSupportPolicy::route_cal_1();
    // The four-family map's own route evidence, at both scales.
    let diagnostic = p.status(Some(TailSupport {
        quantile: 0.99,
        observations: 46,
    }));
    assert_eq!(
        diagnostic,
        MeasurementStatus::InsufficientTailSupport {
            observations: 46,
            required: 500
        }
    );
    assert!(!diagnostic.is_priceable());
    let authority = p.status(Some(TailSupport {
        quantile: 0.99,
        observations: 1303,
    }));
    assert_eq!(authority, MeasurementStatus::Measured);
    assert!(authority.is_priceable());
}

#[test]
fn nothing_observed_is_distinct_from_thinly_observed() {
    let p = TailSupportPolicy::route_cal_1();
    assert_eq!(p.status(None), MeasurementStatus::NotObserved);
    assert!(!p.status(None).is_priceable());
    // Both are unpriceable, and they are different facts: one candidate
    // moved no routing at all, the other moved some and was not watched
    // long enough to say how much.
    assert_ne!(
        p.status(None),
        p.status(Some(TailSupport {
            quantile: 0.99,
            observations: 46
        }))
    );
}

#[test]
fn the_policy_carries_the_reason_it_exists() {
    // A refusal must be traceable to the decision that refused it,
    // rather than to an edited constant.
    let p = TailSupportPolicy::route_cal_1();
    assert!(p.provenance.contains("ROUTE-CAL-1"));
    assert!(p.min_tail_observations > 0.0);
}

#[test]
fn a_degenerate_quantile_demands_the_impossible_rather_than_dividing_by_zero() {
    let p = TailSupportPolicy::route_cal_1();
    assert_eq!(p.required_observations(1.0), u64::MAX);
    assert_eq!(
        p.status(Some(TailSupport {
            quantile: 1.0,
            observations: u64::MAX
        })),
        MeasurementStatus::InsufficientTailSupport {
            observations: u64::MAX,
            required: u64::MAX
        }
    );
}
