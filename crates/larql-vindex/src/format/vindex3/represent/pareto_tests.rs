//! **REPRESENT-PARETO-1 preconditions.** Nothing here is the witness.
//!
//! P1 asks whether the two candidates the world table will carry land
//! in the SAME promotion tier. If they do not, stage 1 of
//! `decide_promotion` filters one out before the non-dominated frontier
//! is ever computed, quality never gets a say, and a passing witness
//! would be passing for the wrong reason.
//!
//! The expectation here is the AMENDED one
//! (`docs/represent/forecasts/represent-pareto-1-bank-family-decision.json`).
//! The freeze forecast `Unscorable`/tier 1, reasoning from
//! `Fixture::observed`, which nulls three mass-displacement
//! distributions; the fixture is built on `authority_reading`, which
//! populates them, so every criterion scores and the class is `Priced`.
//! That falsification is recorded and not rewritten.
//!
//! **This file does not test `ranking_score` or `within`.** `within`
//! stops falling back to `gpu_ms_saved` at `Priced`, and it can change
//! without touching the mechanism PARETO-1 claims, because
//! `decide_promotion` compares pairwise and never reads it. What is
//! asserted is only what stage 1 and stage 5 can see.

use super::assessment::MoveClass;
use super::decision::{decide_promotion, AmbiguityReason, PromotionDecision};
use super::measurement::EvidenceScale;
use super::state::fixtures;

/// `gpu_ms_saved` as the FROZEN FACTS determine it, not as a run
/// reports it.
///
/// The cost observation gives `beta = ((10.0 - 8.0)/10.0) / ((32768 -
/// 24576)/32768) = 0.2/0.25 = 0.8`. The parent ledger removes nothing,
/// so its predicted cost is `10.0 * (1 - 0.8*0.0) = 10.0`. Each child
/// removes 8,192 of 32,768 bytes, so `fraction_removed = 0.25` and the
/// predicted cost is `10.0 * (1 - 0.8*0.25) = 8.0`.
///
/// Pinning it is what makes the physical inertness of W1, W2 and C1 a
/// preregistered fact rather than a stipulation: if a later edit to the
/// ledgers or the cost observation breaks the equality, those three
/// worlds silently stop being controlled, and this assertion fails
/// first.
const FROZEN_GPU_MS_SAVED: f64 = 2.0;

/// **P1.** Both candidates buy time, classify the same way, share one
/// tier, and gain exactly the same amount.
#[test]
fn p1_both_candidates_buy_time_and_share_one_promotion_tier() {
    let snap = fixtures::pareto_p1_snapshot();
    let candidates = snap
        .promotion_candidates(EvidenceScale::Authority)
        .expect("the cost model covers this model");

    // Both edges survive assessment: each end carries a reading and a
    // ledger. A move with either missing is skipped, not defaulted.
    assert_eq!(
        candidates.len(),
        2,
        "two measured, ledgered children of one root, got {:?}",
        candidates.iter().map(|c| &c.id).collect::<Vec<_>>()
    );

    // Report every row BEFORE asserting: a first-failure panic names one
    // candidate, and the question P1 asks is about the pair.
    for c in &candidates {
        let s = &c.promotion.assessment.ranking_score;
        println!(
            "P1 row  id={:<12} class={:?} tier={} gpu_ms_saved={}",
            c.id,
            s.class,
            s.class.tier(),
            s.gpu_ms_saved
        );
    }

    for c in &candidates {
        let score = &c.promotion.assessment.ranking_score;
        assert!(
            score.gpu_ms_saved > 0.0,
            "{}: gpu_ms_saved {} must be positive, or the class drops to \
             Worthless and tier 0 ends the round",
            c.id,
            score.gpu_ms_saved
        );
        assert_eq!(
            score.class,
            MoveClass::Priced,
            "{}: the populated bank scores every criterion, so a \
             positive-gain move is Priced; asserting the CLASS is what \
             makes the shared tier a reason rather than a coincidence",
            c.id
        );
        assert_eq!(score.class.tier(), 2, "{}", c.id);
        assert_eq!(
            score.gpu_ms_saved, FROZEN_GPU_MS_SAVED,
            "{}: the frozen ledgers and cost observation determine this \
             exactly; a different value means the physical facts moved",
            c.id
        );
    }

    // Stated separately from the per-candidate pin because THIS is the
    // property W1, W2 and C1 depend on: with physical gain equal, stage
    // 5 cannot separate the candidates either, so only the accepted
    // quality values can.
    assert_eq!(
        candidates[0]
            .promotion
            .assessment
            .ranking_score
            .gpu_ms_saved,
        candidates[1]
            .promotion
            .assessment
            .ranking_score
            .gpu_ms_saved,
        "physical gain must be inert across the pair"
    );
}

/// **P2's repair, observed through the decision.** With ROUTE-CAL-1's
/// registry actually carried, the comparator is no longer blind.
///
/// `comparable()` is private, so the repair is asserted where it is
/// visible: under the DEFAULT registry every statistic is `Unusable` at
/// diagnostic scale and every decision collapses to
/// `NoOrderingEvidence`. Anything else proves at least one statistic
/// now orders.
#[test]
fn p2_repair_the_carried_registry_can_order_something() {
    let snap = fixtures::pareto_p1_snapshot();
    let candidates = snap
        .promotion_candidates(EvidenceScale::Authority)
        .expect("the cost model covers this model");
    let decision = decide_promotion(
        &candidates,
        &snap.config().calibrations,
        &snap.config().tail_support,
    );
    assert!(
        !matches!(
            decision,
            PromotionDecision::Ambiguous {
                reason: AmbiguityReason::NoOrderingEvidence,
                ..
            }
        ),
        "the corrected fixture still cannot order anything: {decision:?}"
    );
}

// ---------------------------------------------------- the five worlds

use super::quality::Statistic;
use super::state::fixtures::ParetoWorld;
use super::state::snapshot::SearchSnapshot;

use fixtures::{PARETO_BETTER as BETTER, PARETO_MIDDLE as MIDDLE, PARETO_WORSE as WORSE};

/// Decide one world, checking the two things every world must satisfy
/// whatever its verdict.
fn decide(world: &ParetoWorld) -> PromotionDecision {
    let snap = world.snapshot();
    let candidates = snap
        .promotion_candidates(EvidenceScale::Authority)
        .expect("the cost model covers this model");
    assert_eq!(candidates.len(), 2, "both edges must survive assessment");

    // Stage 1 must not decide in ANY world, or the frontier is never
    // reached and the world measured nothing.
    assert_eq!(
        candidates[0]
            .promotion
            .assessment
            .ranking_score
            .class
            .tier(),
        candidates[1]
            .promotion
            .assessment
            .ranking_score
            .class
            .tier(),
        "stage 1 separated the candidates; the frontier was never reached"
    );

    let decision = decide_promotion(
        &candidates,
        &snap.config().calibrations,
        &snap.config().tail_support,
    );

    // R4-F1, re-armed at this layer: permuting enumeration order changes
    // neither the decision nor the RECORD.
    let mut reversed = candidates.clone();
    reversed.reverse();
    let permuted = decide_promotion(
        &reversed,
        &snap.config().calibrations,
        &snap.config().tail_support,
    );
    assert_eq!(
        format!("{decision:?}"),
        format!("{permuted:?}"),
        "input order reached the decision or its record"
    );

    decision
}

/// **W1 — quality alone selects A.**
#[test]
fn w1_quality_alone_selects_a() {
    match decide(&ParetoWorld::inert(BETTER, WORSE)) {
        PromotionDecision::SelectForAuthority {
            candidate,
            evidence,
        } => {
            assert_eq!(candidate, ParetoWorld::A);
            assert_eq!(evidence.dominated, vec![ParetoWorld::B.to_string()]);
            assert!(
                !evidence.decided_by_physical_gain,
                "physical gain is inert here; a true flag would mean the \
                 economics decided and the quality result is unattributable"
            );
            assert_eq!(
                evidence.deciding,
                vec![Statistic::KlP99, Statistic::RouteFlipRate],
                "both registered ordering proxies separated the pair"
            );
        }
        other => panic!("expected A selected, got {other:?}"),
    }
}

/// **W2 — the same machinery selects B when only the accepted values
/// swap.** This is the claim.
#[test]
fn w2_swapping_only_the_accepted_values_selects_b() {
    match decide(&ParetoWorld::inert(WORSE, BETTER)) {
        PromotionDecision::SelectForAuthority {
            candidate,
            evidence,
        } => {
            assert_eq!(candidate, ParetoWorld::B);
            assert_eq!(evidence.dominated, vec![ParetoWorld::A.to_string()]);
            assert!(!evidence.decided_by_physical_gain);
            assert_eq!(
                evidence.deciding,
                vec![Statistic::KlP99, Statistic::RouteFlipRate]
            );
        }
        other => panic!("expected B selected, got {other:?}"),
    }
}

/// **W1 and W2 differ in NOTHING except the accepted values.**
///
/// Asserted explicitly, because without it the pair of results above is
/// LOOP-1's claim wearing a new name: a different answer obtained by
/// observing a different key set is not value-sensitivity.
#[test]
fn w1_and_w2_hold_the_observed_key_set_identical() {
    let keys = |s: &SearchSnapshot| {
        let mut v: Vec<String> = s
            .facts()
            .measurements
            .keys()
            .map(|k| k.as_str().to_string())
            .collect();
        v.sort();
        v
    };
    let ids = |s: &SearchSnapshot| {
        let mut v: Vec<String> = s
            .promotion_candidates(EvidenceScale::Authority)
            .expect("cost")
            .iter()
            .map(|c| c.id.clone())
            .collect();
        v.sort();
        v
    };
    let gains = |s: &SearchSnapshot| {
        let mut v: Vec<f64> = s
            .promotion_candidates(EvidenceScale::Authority)
            .expect("cost")
            .iter()
            .map(|c| c.promotion.assessment.ranking_score.gpu_ms_saved)
            .collect();
        v.sort_by(f64::total_cmp);
        v
    };

    let w1 = ParetoWorld::inert(BETTER, WORSE).snapshot();
    let w2 = ParetoWorld::inert(WORSE, BETTER).snapshot();

    assert_eq!(keys(&w1), keys(&w2), "the observed key SET must be equal");
    assert_eq!(ids(&w1), ids(&w2), "the same two candidates");
    assert_eq!(gains(&w1), gains(&w2), "the same physical facts");
    assert_eq!(
        gains(&w1),
        vec![FROZEN_GPU_MS_SAVED, FROZEN_GPU_MS_SAVED],
        "and those facts are inert across the pair"
    );
}

/// **C1 — conflicting proxies refuse rather than manufacture a winner.**
///
/// The `AmbiguityReason` is pinned, not just the variant. C1 would
/// receive `Ambiguous` even from a fixture whose comparator was blind,
/// so a test matching only the variant passes while the mechanism it
/// claims to exercise is absent.
#[test]
fn c1_conflicting_proxies_refuse() {
    match decide(&ParetoWorld::inert(
        (BETTER.0, WORSE.1),
        (WORSE.0, BETTER.1),
    )) {
        PromotionDecision::Ambiguous { candidates, reason } => {
            assert_eq!(reason, AmbiguityReason::ConflictingOrderingProxies);
            let mut expected = vec![ParetoWorld::A.to_string(), ParetoWorld::B.to_string()];
            expected.sort();
            assert_eq!(candidates, expected);
        }
        other => panic!("expected a refusal, got {other:?}"),
    }
}

/// **C2 — indistinguishable quality, separated by grounded cost.**
///
/// The empty `deciding` and the true flag are what distinguish this from
/// W1 and W2 in the trace: a reader never has to guess which stage
/// decided.
#[test]
fn c2_equal_quality_is_separated_by_physical_gain() {
    match decide(&ParetoWorld::physically_separated(MIDDLE, MIDDLE)) {
        PromotionDecision::SelectForAuthority {
            candidate,
            evidence,
        } => {
            assert_eq!(candidate, ParetoWorld::A, "A removes twice the bytes");
            assert!(evidence.decided_by_physical_gain);
            assert!(evidence.deciding.is_empty(), "no proxy separated them");
            assert!(evidence.dominated.is_empty());
        }
        other => panic!("expected a physical decision, got {other:?}"),
    }
}

/// **C3 — identity must never break a tie.**
#[test]
fn c3_equal_on_everything_refuses() {
    match decide(&ParetoWorld::inert(MIDDLE, MIDDLE)) {
        PromotionDecision::Ambiguous { candidates, reason } => {
            assert_eq!(reason, AmbiguityReason::IndistinguishableOnEveryProxy);
            let mut expected = vec![ParetoWorld::A.to_string(), ParetoWorld::B.to_string()];
            expected.sort();
            assert_eq!(candidates, expected);
        }
        other => panic!("expected a refusal, got {other:?}"),
    }
}

/// The two rungs of the measurement ladder, as classes.
#[test]
fn the_depth_templates_differ_only_in_what_they_can_price() {
    let (cheap, ..) = fixtures::pareto_candidate_template_at(8);
    let (priced, ..) = fixtures::pareto_candidate_template_at(500);
    println!(
        "ladder  cheap={:?}/{}  priced={:?}/{}",
        cheap.assessment.ranking_score.class,
        cheap.assessment.ranking_score.class.tier(),
        priced.assessment.ranking_score.class,
        priced.assessment.ranking_score.class.tier()
    );
    assert_eq!(cheap.assessment.ranking_score.class, MoveClass::Unscorable);
    assert_eq!(priced.assessment.ranking_score.class, MoveClass::Priced);
    assert!(
        cheap.assessment.ranking_score.class.tier() < priced.assessment.ranking_score.class.tier(),
        "escalation raises the tier — this is the preregistered hazard"
    );
}
