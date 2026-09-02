//! R4-F10. The declaration is checked, never trusted, and never inferred.

use super::super::diagnostic::{DiagnosticPolicy, DiagnosticVector};
use super::super::quality::{Distribution, QualityBank};
use super::*;

fn bank(kl: f64, flips: u64) -> QualityBank {
    let mut b = super::super::diagnostic::tests::guard_256();
    b.logits.kl_p99 = kl;
    b.routing.route_flips = flips;
    b
}

fn vector(kl: f64, flips: u64) -> DiagnosticVector {
    DiagnosticVector::of(&DiagnosticPolicy::bs2_kimi_v1(), &bank(kl, flips))
}

/// The head's ground, as measured in iteration 4.
const HEAD: &str = "lm_head is applied after every routing decision";

#[test]
fn undeclared_statistics_participate() {
    let d = ParticipationDeclaration::all_affected();
    assert_eq!(
        d.of(Statistic::RouteFlipRate),
        StatisticParticipation::Affected
    );
    assert!(d.of(Statistic::KlP99).participates());
    assert!(!d.of(Statistic::KlP99).is_structurally_invariant());
    assert!(d.known_zero_spend().is_empty());
}

#[test]
fn a_declared_invariant_does_not_participate_but_is_known() {
    let d = ParticipationDeclaration::all_affected()
        .structurally_invariant(Statistic::RouteFlipRate, HEAD);
    let p = d.of(Statistic::RouteFlipRate);
    assert!(!p.participates(), "an invariant may not rank the pair");
    assert!(p.is_structurally_invariant());
    assert_eq!(d.known_zero_spend(), vec![Statistic::RouteFlipRate]);
    // Declaring one dimension must not touch any other.
    assert!(d.of(Statistic::KlP99).participates());
}

#[test]
fn the_ground_is_carried_for_the_trace() {
    let d = ParticipationDeclaration::all_affected()
        .structurally_invariant(Statistic::RouteFlips, HEAD);
    match d.of(Statistic::RouteFlips) {
        StatisticParticipation::StructurallyInvariant { because } => assert_eq!(because, HEAD),
        other => panic!("expected an invariant, got {other:?}"),
    }
}

#[test]
fn an_unchanged_statistic_verifies() {
    let parent = vector(2.5707e-3, 43);
    // H: kl moved a long way, routing did not move at all.
    let candidate = vector(4.4171e-3, 43);
    let d = ParticipationDeclaration::all_affected()
        .structurally_invariant(Statistic::RouteFlipRate, HEAD)
        .structurally_invariant(Statistic::RouteFlips, HEAD);
    assert_eq!(d.verify(&parent, &candidate), Ok(()));
}

#[test]
fn a_moved_statistic_is_a_violation_not_a_warning() {
    let parent = vector(2.5707e-3, 43);
    // K25 moved routing. Declaring it invariant would DELETE that.
    let candidate = vector(2.5262e-3, 50);
    let d = ParticipationDeclaration::all_affected()
        .structurally_invariant(Statistic::RouteFlipRate, HEAD);
    let err = d.verify(&parent, &candidate).expect_err("must refuse");
    assert_eq!(err.statistic, Statistic::RouteFlipRate);
    assert_eq!(err.parent, Some(43.0 / 256.0));
    assert_eq!(err.candidate, Some(50.0 / 256.0));
    assert!(err.to_string().contains("declared structurally invariant"));
}

#[test]
fn verification_is_bit_exact_not_approximate() {
    let parent = vector(2.5707e-3, 43);
    let mut b = bank(2.5707e-3, 43);
    // One ulp. A structural invariant is the SAME number by the same
    // path; "nearly the same" is a different claim.
    b.logits.kl_p99 = f64::from_bits(b.logits.kl_p99.to_bits() + 1);
    let candidate = DiagnosticVector::of(&DiagnosticPolicy::bs2_kimi_v1(), &b);
    let d = ParticipationDeclaration::all_affected()
        .structurally_invariant(Statistic::KlP99, "hypothetical");
    assert!(d.verify(&parent, &candidate).is_err(), "one ulp is a move");
}

#[test]
fn a_statistic_neither_side_observed_is_vacuously_conformant() {
    // route mixture mass is absent from guard_256 on both sides.
    let parent = vector(2.5707e-3, 43);
    let candidate = vector(4.4171e-3, 43);
    assert!(parent
        .reading(Statistic::RouteMixtureMassP99)
        .is_none_or(|r| r.observed.is_none()));
    let d = ParticipationDeclaration::all_affected()
        .structurally_invariant(Statistic::RouteMixtureMassP99, HEAD);
    assert_eq!(d.verify(&parent, &candidate), Ok(()));
}

#[test]
fn appearing_on_only_one_side_is_a_violation() {
    // Observed on the parent, absent on the candidate. Not "identical",
    // and emphatically not a proven zero.
    let mut pb = bank(2.5707e-3, 43);
    pb.top1_mass_displaced = Some(Distribution {
        count: 4,
        min: 0.01,
        p50: 0.02,
        p95: 0.05,
        p99: 0.058,
        max: 0.0583,
    });
    let parent = DiagnosticVector::of(&DiagnosticPolicy::bs2_kimi_v1(), &pb);
    let candidate = vector(2.5707e-3, 43);
    assert!(parent
        .reading(Statistic::Top1MassDisplaced)
        .is_some_and(|r| r.observed.is_some()));
    let d = ParticipationDeclaration::all_affected()
        .structurally_invariant(Statistic::Top1MassDisplaced, "hypothetical");
    let err = d
        .verify(&parent, &candidate)
        .expect_err("one-sided is not identical");
    assert_eq!(err.statistic, Statistic::Top1MassDisplaced);
    assert_eq!(err.parent, Some(0.0583));
    assert!(err.candidate.is_none());
}

#[test]
fn redeclaring_replaces_rather_than_duplicates() {
    let d = ParticipationDeclaration::all_affected()
        .structurally_invariant(Statistic::RouteFlips, "first")
        .structurally_invariant(Statistic::RouteFlips, "second");
    assert_eq!(d.known_zero_spend(), vec![Statistic::RouteFlips]);
    match d.of(Statistic::RouteFlips) {
        StatisticParticipation::StructurallyInvariant { because } => assert_eq!(because, "second"),
        other => panic!("expected an invariant, got {other:?}"),
    }
}
