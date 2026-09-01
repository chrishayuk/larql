//! The five properties BS2-F1 has to establish, plus the conformance
//! check that keeps the two schemas honest about the same numbers.

use super::super::constraint::ConstraintVector;
use super::super::measurement::{EvidenceScale, MeasurementStatus, TailSupportPolicy};
use super::super::quality::{
    kimi_logit_balanced_v1, Criterion, LogitEvidence, QualityBank, QualityGate, RoutingEvidence,
    Statistic,
};
use super::super::search_evidence::{SearchCalibrationRegistry, SearchEvidence};
use super::*;

/// A 256-position diagnostic bank with the shape the real Kimi guard
/// bank has: 46 route-change events behind a mixture-mass p99, which is
/// a maximum wearing a percentile's name.
pub(in crate::format::vindex3::represent) fn guard_256() -> QualityBank {
    QualityBank {
        positions: 256,
        logits: LogitEvidence {
            kl_p50: 6.2998e-5,
            kl_p95: 6.8794e-4,
            kl_p99: 2.4762e-3,
            max_logit_delta: 1.1154,
            top1_flips: 2,
            top10_changes: 74,
        },
        routing: RoutingEvidence {
            route_flips: 46,
            positions_with_route_change: 34,
            layers_with_route_change: 7,
            first_layer_with_route_change: Some(20),
            route_margin: None,
            route_weight_mass_moved: None,
        },
        min_covered_mass: Some(0.7814),
        top1_mass_displaced: None,
        top10_mass_displaced: None,
        top10_margin: None,
        top10_candidate_margin: None,
        top10_rank_displacement: None,
        top1_margin: None,
        top1_candidate_margin: None,
    }
}

/// **(1)** The observable the contract does not bound is nonetheless
/// observed — which is the whole of BS2-F1.
#[test]
fn route_flip_rate_is_observed_although_no_gate_bounds_it() {
    let gate = kimi_logit_balanced_v1();
    assert!(
        gate.route_flip_max.is_none(),
        "the contract stays frozen: it does NOT bound route flips"
    );
    let authority = ConstraintVector::of(&gate, &guard_256());
    assert!(
        !authority
            .margins
            .iter()
            .any(|m| m.criterion == Criterion::RouteFlips),
        "and so has no route-flip margin"
    );

    let d = DiagnosticVector::of(&DiagnosticPolicy::bs2_kimi_v1(), &guard_256());
    let r = d
        .reading(Statistic::RouteFlipRate)
        .expect("BS2-F1: the search sees it anyway");
    assert_eq!(r.purpose, ObservationPurpose::SearchEvidence);
    assert_eq!(r.observed, Some(46.0 / 256.0));
}

/// **(2)** Instrumentation cannot move the contract. Adding or removing
/// an observation leaves admission bit-identical.
#[test]
fn changing_the_diagnostic_policy_cannot_change_admission() {
    let gate = kimi_logit_balanced_v1();
    let bank = guard_256();
    let before = ConstraintVector::of(&gate, &bank);

    let mut widened = DiagnosticPolicy::bs2_kimi_v1();
    widened.observations.push(DiagnosticObservation {
        statistic: Statistic::Top10Changes,
        purpose: ObservationPurpose::SearchEvidence,
    });
    let mut narrowed = DiagnosticPolicy::bs2_kimi_v1();
    narrowed.observations.clear();

    for p in [&widened, &narrowed] {
        let _ = DiagnosticVector::of(p, &bank);
        let after = ConstraintVector::of(&gate, &bank);
        assert_eq!(before, after, "the contract does not hear the instrument");
        assert_eq!(before.admissible(), after.admissible());
        assert_eq!(before.sound(), after.sound());
    }
}

/// **(3)** And the converse: a contract criterion is not diagnostic
/// evidence just for being a criterion. The policy is explicit, so a
/// gate that grows a bound grows no observation.
#[test]
fn an_authority_criterion_does_not_become_diagnostic_evidence() {
    let bounded = QualityGate {
        route_flip_max: Some(64),
        ..kimi_logit_balanced_v1()
    };
    let authority = ConstraintVector::of(&bounded, &guard_256());
    assert!(
        authority
            .margins
            .iter()
            .any(|m| m.what == Statistic::RouteFlips),
        "the gate now bounds the COUNT, so a margin exists"
    );

    let d = DiagnosticVector::of(&DiagnosticPolicy::bs2_kimi_v1(), &guard_256());
    assert!(
        d.reading(Statistic::RouteFlips).is_none(),
        "but the policy never asked for the count, so the search does not see it"
    );
    assert!(
        !DiagnosticPolicy::bs2_kimi_v1().observes(Statistic::RouteFlips),
        "observation is declared, never inferred from a bound"
    );
}

/// **(4)** It orders, and it is never a price — at the scale it was
/// calibrated for.
#[test]
fn route_flip_rate_orders_and_is_never_priced() {
    let d = DiagnosticVector::of(&DiagnosticPolicy::bs2_kimi_v1(), &guard_256());
    let r = d.reading(Statistic::RouteFlipRate).expect("observed");
    let policy = TailSupportPolicy::route_cal_1();
    let registry = SearchCalibrationRegistry::route_cal_1();

    // A rate over every position is dense: no thin tail, well measured.
    assert_eq!(r.measurement_status(&policy), MeasurementStatus::Measured);

    let e = r.evidence(&registry, &policy);
    assert!(
        matches!(e, SearchEvidence::OrderingProxy { .. }),
        "well measured and STILL only a proxy, got {e:?}"
    );
    assert!(e.orders());
    assert!(!e.is_priceable(), "a proxy never becomes price");
}

/// **(4b)** The second invariant, from the other side: a contract
/// criterion that this scale cannot support is refused as evidence here
/// while remaining a criterion there.
#[test]
fn an_authority_criterion_may_be_unusable_at_diagnostic_scale() {
    let registry = SearchCalibrationRegistry::route_cal_1();
    let thin = MeasurementStatus::InsufficientTailSupport {
        observations: 46,
        required: 500,
    };
    assert_eq!(
        registry.evidence_for(
            Statistic::RouteMixtureMassP99,
            EvidenceScale::Diagnostic,
            &thin
        ),
        SearchEvidence::Unusable
    );
    assert_eq!(
        registry.evidence_for(
            Statistic::RouteMixtureMassP99,
            EvidenceScale::Authority,
            &MeasurementStatus::Measured
        ),
        SearchEvidence::Direct,
        "the contract judges it unchanged at the scale it was written for"
    );
}

/// The two schemas must never DISAGREE about what a number is. BS2-F2
/// was two vocabularies drifting; two bank extractions would drift the
/// same way, so there is only one, and this pins that it is shared.
#[test]
fn both_schemas_read_the_same_bank_identically() {
    let bank = guard_256();
    let gate = QualityGate {
        route_flip_max: Some(64),
        ..kimi_logit_balanced_v1()
    };
    let authority = ConstraintVector::of(&gate, &bank);
    let d = DiagnosticVector::of(&DiagnosticPolicy::bs2_kimi_v1(), &bank);

    let mut compared = 0;
    for r in &d.readings {
        if let Some(m) = authority.margins.iter().find(|m| m.what == r.statistic) {
            assert_eq!(m.observed, r.observed, "{} value", r.statistic);
            assert_eq!(m.tail_support, r.tail_support, "{} tail", r.statistic);
            compared += 1;
        }
    }
    assert!(compared >= 3, "the schemas do overlap: {compared} shared");
}

/// **The payoff.** What rung 4 can actually rank on, at 256 positions.
///
/// Before BS2-F1 the answer was `kl p99` ALONE — the two mixture-mass
/// percentiles rest on 74 and 46 events and are `Unusable` at this
/// scale, and the one statistic that does survive a small bank was not
/// in the vector at all. Ranking on a single proxy is the dependence
/// the evidence ladder exists to prevent, so this is the property that
/// says the instrumentation surface is now adequate to start.
#[test]
fn the_search_can_order_on_two_independent_dimensions_not_one() {
    let d = DiagnosticVector::of(&DiagnosticPolicy::bs2_kimi_v1(), &guard_256());
    let orderable: Vec<Statistic> = d
        .ordering(
            &SearchCalibrationRegistry::route_cal_1(),
            &TailSupportPolicy::route_cal_1(),
        )
        .into_iter()
        .map(|(r, _)| r.statistic)
        .collect();

    assert_eq!(
        orderable,
        vec![Statistic::KlP99, Statistic::RouteFlipRate],
        "a dense percentile and a count-derived rate — and nothing that \
         needs a tail this bank cannot supply"
    );
    // Both order; neither may be spent.
    for (r, e) in d.ordering(
        &SearchCalibrationRegistry::route_cal_1(),
        &TailSupportPolicy::route_cal_1(),
    ) {
        assert!(e.orders(), "{}", r.statistic);
        assert!(
            !e.is_priceable(),
            "{} must never be priced here",
            r.statistic
        );
    }
}
