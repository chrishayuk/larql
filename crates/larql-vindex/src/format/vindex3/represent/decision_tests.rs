//! R4-F1. The permutation control is the one that would have caught the
//! original defect; the rest pin the doctrine it exposed.

use super::super::assessment::{CandidateAssessment, EvidenceContext};
use super::super::byte_ledger::{ByteLedger, ScopeBytes};
use super::super::constraint::ConstraintVector;
use super::super::diagnostic::{DiagnosticPolicy, DiagnosticVector};
use super::super::execution_cost::{m3max_metal_001, ExecutionCostModel};
use super::super::measurement::EvidenceScale;
use super::super::promotion::PromotionCandidate;
use super::super::quality::{kimi_logit_balanced_v1, QualityBank};
use super::*;

const BASE: u64 = 13_480_000_000;

fn bank(kl: f64, flips: u64) -> QualityBank {
    let mut b = super::super::diagnostic::tests::guard_256();
    b.logits.kl_p99 = kl;
    b.routing.route_flips = flips;
    b
}

fn ledger(name: &str, bytes: u64) -> ByteLedger {
    ByteLedger {
        model: "Kimi-Linear-48B-A3B-Instruct".into(),
        baseline_representation: "Q8_0".into(),
        candidate_representation: name.into(),
        scopes: vec![ScopeBytes {
            scope: "experts".into(),
            family: "routed experts".into(),
            baseline_bytes: BASE,
            candidate_bytes: bytes,
        }],
    }
}

/// A candidate whose diagnostic bank says `kl`/`flips`, removing `bytes`.
fn candidate(id: &str, kl: f64, flips: u64, bytes: u64) -> SearchCandidate {
    let b = bank(kl, flips);
    let gate = kimi_logit_balanced_v1();
    let parent = bank(2.4762e-3, 46);
    let a = CandidateAssessment::of(
        &EvidenceContext::route_cal_1(EvidenceScale::Diagnostic),
        &ExecutionCostModel::new(vec![m3max_metal_001()]),
        &ledger("parent", BASE),
        &ledger(id, bytes),
        ConstraintVector::of(&gate, &parent),
        ConstraintVector::of(&gate, &b),
    )
    .expect("same model");
    SearchCandidate {
        id: id.into(),
        promotion: PromotionCandidate::new(a, vec![]),
        diagnostic: DiagnosticVector::of(&DiagnosticPolicy::bs2_kimi_v1(), &b),
    }
}

fn reg() -> SearchCalibrationRegistry {
    SearchCalibrationRegistry::route_cal_1()
}
fn pol() -> TailSupportPolicy {
    TailSupportPolicy::route_cal_1()
}

/// The four real Rung-4 candidates, by their measured diagnostics.
fn rung4_set() -> Vec<SearchCandidate> {
    vec![
        candidate("e26", 2.6551e-3, 46, 13_040_000_000),
        candidate("e24", 2.9993e-3, 51, 13_040_000_000),
        candidate("e23", 3.6266e-3, 58, 13_040_000_000),
        candidate("e20", 5.0994e-3, 71, 13_040_000_000),
    ]
}

/// **THE control.** The original defect printed `PROMOTE: e26` for one
/// input order and `PROMOTE: e20` for the reverse, from identical
/// reports, because every `rank_key` tied and a stable sort returned
/// input order. A decision must not depend on call order.
#[test]
fn the_promotion_decision_is_invariant_under_input_permutation() {
    let (r, p) = (reg(), pol());
    let forward = rung4_set();
    let mut reversed = rung4_set();
    reversed.reverse();

    let a = decide_promotion(&forward, &r, &p);
    let b = decide_promotion(&reversed, &r, &p);
    assert_eq!(a, b, "the decision changed when only the ORDER changed");

    match a {
        PromotionDecision::SelectForAuthority { candidate, .. } => assert_eq!(candidate, "e26"),
        other => panic!("expected a promotion, got {other:?}"),
    }
}

/// The pre-registered ordering, recovered from the evidence rather than
/// from the order the candidates were supplied in.
#[test]
fn the_measured_rung4_set_recovers_the_pre_registered_order() {
    let (r, p) = (reg(), pol());
    let c = rung4_set();
    // e26 < e24 < e23 < e20, by dominance and transitively.
    for (better, worse) in [(0, 1), (1, 2), (2, 3), (0, 3)] {
        assert!(
            c[better].proxy_dominates(&c[worse], &r, &p),
            "{} should dominate {}",
            c[better].id,
            c[worse].id
        );
        assert!(
            !c[worse].proxy_dominates(&c[better], &r, &p),
            "dominance must not be symmetric"
        );
    }
}

/// A conflict is REFUSED, not scalarised: there is no basis for trading
/// kl places against routing places.
#[test]
fn conflicting_proxies_are_refused_rather_than_traded_off() {
    let (r, p) = (reg(), pol());
    // a: best kl, worst flips. b: worse kl, best flips.
    let set = vec![
        candidate("a", 2.0e-3, 80, 13_040_000_000),
        candidate("b", 4.0e-3, 40, 13_040_000_000),
    ];
    assert!(!set[0].proxy_dominates(&set[1], &r, &p));
    assert!(!set[1].proxy_dominates(&set[0], &r, &p));
    match decide_promotion(&set, &r, &p) {
        PromotionDecision::Ambiguous {
            candidates, reason, ..
        } => {
            assert_eq!(candidates, vec!["a".to_string(), "b".to_string()]);
            assert_eq!(reason, AmbiguityReason::ConflictingOrderingProxies);
        }
        other => panic!("a conflict must not resolve: {other:?}"),
    }
}

/// Proxies that AGREE the candidates are equal leave physical gain free
/// to separate them — a legitimate later stage, and recorded as such.
#[test]
fn physical_gain_may_separate_what_the_proxies_call_equal() {
    let (r, p) = (reg(), pol());
    let set = vec![
        candidate("small", 3.0e-3, 50, 13_400_000_000),
        candidate("big", 3.0e-3, 50, 12_000_000_000),
    ];
    match decide_promotion(&set, &r, &p) {
        PromotionDecision::SelectForAuthority {
            candidate,
            evidence,
            ..
        } => {
            assert_eq!(candidate, "big", "more bytes removed");
            assert!(evidence.decided_by_physical_gain);
        }
        other => panic!("expected gain to decide: {other:?}"),
    }
}

/// Identical on evidence AND on gain: refuse. Identity must not promote.
#[test]
fn indistinguishable_candidates_are_refused_not_ordered_by_identity() {
    let (r, p) = (reg(), pol());
    let set = vec![
        candidate("aaa", 3.0e-3, 50, 13_040_000_000),
        candidate("zzz", 3.0e-3, 50, 13_040_000_000),
    ];
    match decide_promotion(&set, &r, &p) {
        PromotionDecision::Ambiguous {
            candidates, reason, ..
        } => {
            assert_eq!(candidates, vec!["aaa".to_string(), "zzz".to_string()]);
            assert_eq!(reason, AmbiguityReason::IndistinguishableOnEveryProxy);
        }
        other => panic!("identity must not promote: {other:?}"),
    }
    // ...while DISPLAY may order them deterministically, by identity.
    let shown: Vec<&str> = display_order(&set, &r, &p)
        .iter()
        .map(|c| c.id.as_str())
        .collect();
    let mut rev = set.clone();
    rev.reverse();
    let shown_rev: Vec<&str> = display_order(&rev, &r, &p)
        .iter()
        .map(|c| c.id.as_str())
        .collect();
    assert_eq!(shown, shown_rev, "display order is deterministic");
    assert_eq!(shown, vec!["aaa", "zzz"]);
}

#[test]
fn an_empty_round_promotes_nothing() {
    match decide_promotion(&[], &reg(), &pol()) {
        PromotionDecision::None { reason } => assert_eq!(reason, NoPromotableCandidate::EmptySet),
        other => panic!("{other:?}"),
    }
}

/// The table reads BEST FIRST. Sorting on `cmp_rank` alone put `e20` —
/// the worst candidate — at the top, because every key tied and identity
/// decided.
#[test]
fn the_display_table_reads_best_first_and_is_deterministic() {
    let (r, p) = (reg(), pol());
    let set = rung4_set();
    let ids = |v: &[SearchCandidate]| -> Vec<String> {
        display_order(v, &r, &p)
            .iter()
            .map(|c| c.id.clone())
            .collect()
    };
    let mut rev = set.clone();
    rev.reverse();
    assert_eq!(ids(&set), vec!["e26", "e24", "e23", "e20"]);
    assert_eq!(
        ids(&set),
        ids(&rev),
        "display order is permutation-invariant too"
    );
}

/// **`Uninformed` is eligible for authority, and says what it cannot
/// see.** Gating it out would silently rewrite the doctrine into
/// "diagnostic must predict every authority dimension before authority
/// may run", which at 256 positions is impossible for the mass tails —
/// and authority is the mechanism that resolves them.
///
/// Selection is not prediction: the decision must NAME the dimensions it
/// could not speak to, or a trace reads as confidence.
#[test]
fn uninformed_may_be_selected_for_authority_and_names_what_it_cannot_see() {
    let (r, p) = (reg(), pol());
    let set = rung4_set();
    let winner = &set[0];
    assert_eq!(
        winner.promotion.readiness(),
        PromotionReadiness::Uninformed,
        "the real Rung-4 set is Uninformed — nothing speaks to the mass tails"
    );

    match decide_promotion(&set, &r, &p) {
        PromotionDecision::SelectForAuthority {
            candidate,
            evidence,
            ..
        } => {
            assert_eq!(candidate, "e26");
            assert_eq!(evidence.readiness, PromotionReadiness::Uninformed);
            assert!(
                !evidence.unresolved.is_empty(),
                "an Uninformed selection MUST name the dimensions it cannot see"
            );
            // Exactly the criteria the diagnostic scale is silent on.
            assert!(evidence
                .unresolved
                .contains(&Statistic::Top10MassDisplacedP99));
            assert!(evidence
                .unresolved
                .contains(&Statistic::RouteMixtureMassP99));
            // ...and never the ones that DID decide it.
            for s in &evidence.deciding {
                assert!(
                    !evidence.unresolved.contains(s),
                    "{s} both decided and is unresolved"
                );
            }
        }
        other => panic!("Uninformed must be eligible to measure: {other:?}"),
    }
}
