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
        candidates[0].promotion.assessment.ranking_score.gpu_ms_saved,
        candidates[1].promotion.assessment.ranking_score.gpu_ms_saved,
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
