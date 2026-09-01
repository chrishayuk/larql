//! `tests` for [`super`].

use super::super::assessment::{CandidateAssessment, EvidenceContext};
use super::super::byte_ledger::{ByteLedger, ScopeBytes};
use super::super::constraint::{ConstraintVector, LimitKind, Margin};
use super::super::execution_cost::{m3max_metal_001, ExecutionCostModel};
use super::super::measurement::{EvidenceScale, TailSupport};
use super::super::quality::Criterion;
use super::super::quality::Statistic;
use super::*;

const KIMI: &str = "Kimi-Linear-48B-A3B-Instruct";
const BASELINE_BYTES: u64 = 5_984_976_896;

fn ledger(fraction: f64, name: &str) -> ByteLedger {
    ByteLedger {
        model: KIMI.into(),
        baseline_representation: "BF16".into(),
        candidate_representation: name.into(),
        scopes: vec![ScopeBytes {
            scope: "whole decoder".into(),
            family: "whole decoder".into(),
            baseline_bytes: BASELINE_BYTES,
            candidate_bytes: BASELINE_BYTES - (BASELINE_BYTES as f64 * fraction) as u64,
        }],
    }
}

fn vector(kl: f64, route: f64) -> ConstraintVector {
    let ceiling = |what: Statistic, criterion, u: f64| Margin {
        criterion,
        what,
        kind: LimitKind::Ceiling,
        limit: 1.0,
        observed: Some(u),
        vacuous: false,
        tail_support: Some(TailSupport {
            quantile: 0.99,
            observations: 46,
        }),
    };
    ConstraintVector {
        gate_id: "kimi-logit-balanced-v1".into(),
        margins: vec![
            ceiling(Statistic::KlP99, Criterion::KlP99, kl),
            ceiling(
                Statistic::RouteMixtureMassP99,
                Criterion::RouteDisplacement,
                route,
            ),
        ],
    }
}

fn candidate(name: &str, removes: f64, kl_after: f64, route_after: f64) -> CandidateAssessment {
    CandidateAssessment::of(
        &EvidenceContext::route_cal_1(EvidenceScale::Diagnostic),
        &ExecutionCostModel::new(vec![m3max_metal_001()]),
        &ledger(0.16, "parent"),
        &ledger(removes, name),
        vector(0.68, 0.83),
        vector(kl_after, route_after),
    )
    .expect("same model")
}

fn flip_rate(parent: f64, cand: f64) -> ProxyObservation {
    ProxyObservation {
        statistic: Statistic::RouteFlipRate,
        for_criterion: Statistic::RouteMixtureMassP99,
        parent,
        candidate: cand,
        evidence: SearchEvidence::OrderingProxy {
            calibration: "ROUTE-CAL-1".into(),
        },
    }
}

// ── The invariant. ──

#[test]
fn an_unpriced_dimension_cannot_create_an_advantage_by_being_invisible() {
    // A buys MORE and costs LESS on the one dimension anybody can
    // price. Its route cost is unpriceable at this scale — and the
    // flip-rate proxy says it moves routing hard.
    let a = PromotionCandidate::new(
        candidate("A dangerous", 0.24, 0.73, 0.95),
        vec![flip_rate(0.16, 0.31)],
    );
    // B buys less, costs marginally more on kl, and its proxy is benign.
    let b = PromotionCandidate::new(
        candidate("B benign", 0.22, 0.74, 0.84),
        vec![flip_rate(0.16, 0.15)],
    );

    // Everything an economics-only ranking can see favours A.
    assert!(
        a.assessment.ranking_score.gpu_ms_saved > b.assessment.ranking_score.gpu_ms_saved,
        "A buys more"
    );
    assert_eq!(
        a.assessment
            .ranking_score
            .cmp_rank(&b.assessment.ranking_score),
        std::cmp::Ordering::Less,
        "and on economics alone A would win"
    );

    // It must not, because the reason its route cost looks like nothing
    // is that nothing could measure it.
    assert_eq!(a.readiness(), PromotionReadiness::ProxyRisky);
    assert_eq!(b.readiness(), PromotionReadiness::ProxySupported);
    assert_eq!(
        b.cmp_rank(&a),
        std::cmp::Ordering::Less,
        "B must promote first"
    );
    assert!(a.why().contains("Elevated"), "{}", a.why());
}

#[test]
fn no_proxy_at_all_is_worse_than_a_warning_proxy() {
    // "Nobody has measured this" ranks below "a proxy says it is bad",
    // because a warning is information and silence is not.
    let risky = PromotionCandidate::new(
        candidate("risky", 0.22, 0.70, 0.90),
        vec![flip_rate(0.16, 0.30)],
    );
    let uninformed = PromotionCandidate::new(candidate("uninformed", 0.22, 0.70, 0.90), vec![]);
    assert_eq!(risky.readiness(), PromotionReadiness::ProxyRisky);
    assert_eq!(uninformed.readiness(), PromotionReadiness::Uninformed);
    assert_eq!(risky.cmp_rank(&uninformed), std::cmp::Ordering::Less);
    assert!(
        uninformed.why().contains("no proxy"),
        "{}",
        uninformed.why()
    );
}

#[test]
fn an_unusable_proxy_carries_no_weight() {
    // ROUTE-CAL-1 marked diagnostic route mass p99 Unusable. Offering it
    // AS a proxy must not launder it into evidence.
    let mut p = flip_rate(0.16, 0.15);
    p.evidence = SearchEvidence::Unusable;
    let c = PromotionCandidate::new(candidate("c", 0.22, 0.70, 0.84), vec![p]);
    assert_eq!(
        c.readiness(),
        PromotionReadiness::Uninformed,
        "an unusable observation is not a benign one"
    );
}

#[test]
fn the_weakest_unpriceable_dimension_decides_not_the_average() {
    // One benign proxy must not offset one missing.
    let a = PromotionCandidate::new(
        candidate("a", 0.22, 0.70, 0.84),
        vec![flip_rate(0.16, 0.15)],
    );
    assert_eq!(a.readiness(), PromotionReadiness::ProxySupported);
    let mut orphan = flip_rate(0.16, 0.15);
    orphan.for_criterion = Statistic::Top10Changes;
    let b = PromotionCandidate::new(candidate("b", 0.22, 0.70, 0.84), vec![orphan]);
    assert_eq!(
        b.readiness(),
        PromotionReadiness::Uninformed,
        "a proxy for a different criterion speaks to nothing here"
    );
}

#[test]
fn a_worthless_move_ranks_last_whatever_its_evidence() {
    let worthless = PromotionCandidate::new(
        candidate("worthless", 0.16, 0.68, 0.83),
        vec![flip_rate(0.16, 0.10)],
    );
    let uninformed = PromotionCandidate::new(candidate("uninformed", 0.22, 0.70, 0.90), vec![]);
    assert_eq!(
        worthless.assessment.ranking_score.class,
        MoveClass::Worthless
    );
    assert_eq!(
        uninformed.cmp_rank(&worthless),
        std::cmp::Ordering::Less,
        "even the least informed real gain beats a move that buys nothing"
    );
}

#[test]
fn the_order_is_total_and_reproducible() {
    let mk = |n: &str| {
        PromotionCandidate::new(candidate(n, 0.22, 0.70, 0.84), vec![flip_rate(0.16, 0.15)])
    };
    let (a, b) = (mk("aaa"), mk("zzz"));
    assert_eq!(a.cmp_rank(&b), std::cmp::Ordering::Less, "identity decides");
    assert_eq!(b.cmp_rank(&a), std::cmp::Ordering::Greater);
    let mut one = [a.clone(), b.clone()];
    let mut two = [b, a];
    one.sort_by(PromotionCandidate::cmp_rank);
    two.sort_by(PromotionCandidate::cmp_rank);
    let names = |v: &[PromotionCandidate]| {
        v.iter()
            .map(|c| c.assessment.candidate_map.clone())
            .collect::<Vec<_>>()
    };
    assert_eq!(names(&one), names(&two));
}

#[test]
fn a_proxy_is_a_direction_and_never_a_magnitude() {
    // The conflation search_evidence exists to prevent: rho +0.991 is
    // good enough to say "worse", not good enough to say "worse by
    // this much of the remaining budget".
    let big = flip_rate(0.16, 0.90);
    let small = flip_rate(0.16, 0.17);
    assert_eq!(big.risk(), ProxyRisk::Elevated);
    assert_eq!(small.risk(), ProxyRisk::Elevated);
    let c = PromotionCandidate::new(candidate("c", 0.22, 0.70, 0.90), vec![big]);
    // No route price appears anywhere in the assessment.
    assert_eq!(c.assessment.ranking_score.scarce_fraction_consumed, None);
    assert_eq!(c.assessment.ranking_score.score, None);
}

/// **(5)** A candidate is CLASSIFIED by the proxy and never priced by
/// it: readiness improves, and no route-flip cost exists to spend.
#[test]
fn the_proxy_classifies_a_candidate_without_pricing_it() {
    use super::super::diagnostic::{DiagnosticPolicy, DiagnosticVector};
    use super::super::measurement::TailSupportPolicy;
    use super::super::search_evidence::SearchCalibrationRegistry;
    let bank = super::super::diagnostic::tests::guard_256();
    let d = DiagnosticVector::of(&DiagnosticPolicy::bs2_kimi_v1(), &bank);
    let r = d.reading(Statistic::RouteFlipRate).expect("observed");
    let evidence = r.evidence(
        &SearchCalibrationRegistry::route_cal_1(),
        &TailSupportPolicy::route_cal_1(),
    );

    let proxy = ProxyObservation {
        statistic: Statistic::RouteFlipRate,
        for_criterion: Statistic::RouteMixtureMassP99,
        parent: 0.180,
        candidate: 0.176,
        evidence,
    };
    let candidate = PromotionCandidate::new(candidate("c", 0.22, 0.70, 0.84), vec![proxy]);
    assert_eq!(
        candidate.readiness(),
        PromotionReadiness::ProxySupported,
        "the search may PREFER on it"
    );
    assert!(
        !candidate
            .assessment
            .marginal
            .costs
            .iter()
            .any(|c| c.what == Statistic::RouteFlipRate || c.what == Statistic::RouteFlips),
        "and there is no route-flip cost anywhere to spend against a budget"
    );
}
