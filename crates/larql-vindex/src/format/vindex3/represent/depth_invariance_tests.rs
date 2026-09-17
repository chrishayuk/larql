//! **REPRESENT-DEPTH-1, against the CURRENT contract.**
//!
//! Frozen at `docs/represent/forecasts/represent-depth-1.json`.
//!
//! > Evidence strength determines which conclusions are licensed; it is
//! > not evidence that a candidate is superior.
//!
//! **These arms were written against the BROKEN contract and asserted
//! the defect.** DEPTH-2 repaired it, every one of them failed, and they
//! are now flipped to assert the contract. That order matters: the
//! repair is checked against a result recorded BEFORE the
//! implementation was chosen, not against a green path invented
//! afterwards.
//!
//! The repair, in one line: stage 1 stopped using `Unscorable` as
//! PRECEDENCE and stage 5 started using it as a QUALIFICATION
//! PRECONDITION.

use super::assessment::MoveClass;
use super::decision::{decide_promotion, AmbiguityReason, PromotionDecision, SearchCandidate};
use super::state::fixtures;
use super::statistic::Statistic;

const CHEAP: u64 = 8;
const PRICED: u64 = 500;

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum Depth {
    Cheap,
    Priced,
}

/// One candidate at a chosen measurement depth. Depth sets the
/// assessment — and therefore the `MoveClass` — and nothing else.
fn candidate(id: &str, depth: Depth, kl: f64, flips: u64) -> SearchCandidate {
    let (promotion, _, _, policy) = match depth {
        Depth::Cheap => fixtures::pareto_candidate_template_at(CHEAP),
        Depth::Priced => fixtures::pareto_candidate_template_at(PRICED),
    };
    fixtures::pareto_candidate(id, &promotion, &policy, kl, flips)
}

fn decide(candidates: &[SearchCandidate]) -> PromotionDecision {
    let (_, registry, tail, _) = fixtures::pareto_candidate_template_at(PRICED);
    let forward = decide_promotion(candidates, &registry, &tail);
    // A6, applied to every arm rather than tested once: this is a
    // PRECEDENCE leak, not an ordering leak, and conflating the two
    // would misname the defect.
    let mut reversed = candidates.to_vec();
    reversed.reverse();
    assert_eq!(
        format!("{forward:?}"),
        format!("{:?}", decide_promotion(&reversed, &registry, &tail)),
        "A6: input order reached the decision — that would be a DIFFERENT defect"
    );
    forward
}

fn classes(candidates: &[SearchCandidate]) -> Vec<(String, MoveClass, u8)> {
    candidates
        .iter()
        .map(|c| {
            let s = &c.promotion.assessment.ranking_score;
            (c.id.clone(), s.class, s.class.tier())
        })
        .collect()
}

/// A dominates B on BOTH proxies: lower kl and fewer route flips.
fn winner_and_loser(a_depth: Depth, b_depth: Depth) -> Vec<SearchCandidate> {
    vec![
        candidate("A", a_depth, 3.0e-3, 1_000),
        candidate("B", b_depth, 4.0e-3, 1_600),
    ]
}

/// A better on kl, B better on flips: neither dominates.
fn conflicting(a_depth: Depth, b_depth: Depth) -> Vec<SearchCandidate> {
    vec![
        candidate("A", a_depth, 3.0e-3, 1_600),
        candidate("B", b_depth, 4.0e-3, 1_000),
    ]
}

/// **G1 — price only the loser, and the loser does NOT win.**
///
/// A dominates B on both proxies. B is measured more deeply. B's extra
/// measurement is still represented — `readiness` says `Uninformed`
/// because A, the winner, is the one that was not priced — but it no
/// longer overrules evidence.
///
/// Before DEPTH-2 this returned `SelectForAuthority{B}` with
/// `dominated: []` and `deciding: []`.
#[test]
fn g1_pricing_only_the_loser_does_not_make_the_loser_win() {
    let equal = winner_and_loser(Depth::Cheap, Depth::Cheap);
    let baseline = decide(&equal);

    let mixed = winner_and_loser(Depth::Cheap, Depth::Priced);
    println!("G1 classes: {:?}", classes(&mixed));
    let decision = decide(&mixed);
    println!("G1 decision: {decision:?}");

    match &decision {
        PromotionDecision::SelectForAuthority {
            candidate,
            evidence,
        } => {
            assert_eq!(candidate, "A", "the candidate the EVIDENCE favours");
            assert_eq!(evidence.dominated, vec!["B".to_string()]);
            assert_eq!(
                evidence.deciding,
                vec![Statistic::KlP99, Statistic::RouteFlipRate],
                "and it wins on the named proxies, not on a tier"
            );
            assert!(!evidence.decided_by_physical_gain);
        }
        other => panic!("expected A on the evidence, got {other:?}"),
    }

    // **The invariant.** Identity, physical facts and observed values
    // all held fixed; only B's depth changed; the decision does not
    // move — evidence record included.
    assert_eq!(
        format!("{baseline:?}"),
        format!("{decision:?}"),
        "measuring one candidate more deeply changed the decision"
    );
}

/// **G2 — depth cannot manufacture a resolution.**
///
/// Before DEPTH-2, pricing either side of a conflict selected that side
/// with empty evidence, and the mirror proved it was depth and not
/// merit.
#[test]
fn g2_a_conflict_survives_asymmetric_depth() {
    let baseline = decide(&conflicting(Depth::Cheap, Depth::Cheap));
    for (a, b) in [
        (Depth::Priced, Depth::Cheap),
        (Depth::Cheap, Depth::Priced),
        (Depth::Priced, Depth::Priced),
    ] {
        let decision = decide(&conflicting(a, b));
        match &decision {
            PromotionDecision::Ambiguous { candidates, reason } => {
                assert_eq!(*reason, AmbiguityReason::ConflictingOrderingProxies);
                assert_eq!(candidates, &vec!["A".to_string(), "B".to_string()]);
            }
            other => panic!("{a:?}/{b:?}: depth manufactured a winner: {other:?}"),
        }
        assert_eq!(
            format!("{baseline:?}"),
            format!("{decision:?}"),
            "{a:?}/{b:?}: the conflict's RECORD moved with depth"
        );
    }
}

/// **G3 — equal proxies with asymmetric depth is `IncompletePricingAuthority`.**
///
/// The proxies tie, so a physical comparison is what would continue. It
/// may not, because one member's behavioural cost was never scored.
/// This is NOT `IndistinguishableOnEveryProxy` — that means "they are
/// the same"; this means "we cannot yet tell" — and it hands control
/// back to the measurement layer with a named reason to spend.
#[test]
fn g3_equal_proxies_with_an_unscored_cost_refuses_to_price() {
    let tied = |a: Depth, b: Depth| {
        vec![
            candidate("A", a, 3.0e-3, 1_000),
            candidate("B", b, 3.0e-3, 1_000),
        ]
    };

    for (a, b) in [(Depth::Priced, Depth::Cheap), (Depth::Cheap, Depth::Priced)] {
        let decision = decide(&tied(a, b));
        println!("G3 {a:?}/{b:?}: {decision:?}");
        match &decision {
            PromotionDecision::Ambiguous { candidates, reason } => {
                assert_eq!(
                    *reason,
                    AmbiguityReason::IncompletePricingAuthority,
                    "one member's cost is unscored; this is not indistinguishability"
                );
                assert_eq!(candidates, &vec!["A".to_string(), "B".to_string()]);
            }
            other => panic!("{a:?}/{b:?}: expected a refusal to price, got {other:?}"),
        }
    }

    // **Both unscored is NOT the same answer**, and the first
    // implementation of this precondition got it wrong. The test is
    // COMPARABILITY, not universal support: a frontier where nobody's
    // cost was scored is equally unsupported, and comparing predicted
    // gains there is pre-DEPTH-2 behaviour that must be left alone.
    // Reading the precondition as "every member must be scorable" broke
    // two existing `decision` tests at UNIFORM depth, which is exactly
    // what G4 forbids. The freeze's own word was the corrective — "not
    // supported COMPARABLY across that frontier".
    match decide(&tied(Depth::Cheap, Depth::Cheap)) {
        PromotionDecision::Ambiguous { reason, .. } => {
            assert_eq!(reason, AmbiguityReason::IndistinguishableOnEveryProxy);
        }
        other => panic!("expected the pre-DEPTH-2 answer, got {other:?}"),
    }

    // Both scored, and the physical stage may legitimately act. Here the
    // gains are equal too, so it correctly reports indistinguishability
    // — the variant G3 must never be confused with.
    match decide(&tied(Depth::Priced, Depth::Priced)) {
        PromotionDecision::Ambiguous { reason, .. } => {
            assert_eq!(
                reason,
                AmbiguityReason::IndistinguishableOnEveryProxy,
                "with every cost scored, the honest answer is 'the same'"
            );
        }
        other => panic!("expected indistinguishability, got {other:?}"),
    }
}

/// **G4 — the uniform-depth control, unchanged.**
///
/// PARETO-1, FRONTIER-SCALE-1 and FRONTIER-EXPLORE-1 all hold every
/// candidate at one depth. Their behaviour must be exactly what it was.
#[test]
fn g4_uniform_depth_behaves_exactly_as_before() {
    for depth in [Depth::Cheap, Depth::Priced] {
        match decide(&winner_and_loser(depth, depth)) {
            PromotionDecision::SelectForAuthority {
                candidate,
                evidence,
            } => {
                assert_eq!(candidate, "A");
                assert_eq!(evidence.dominated, vec!["B".to_string()]);
                assert_eq!(
                    evidence.deciding,
                    vec![Statistic::KlP99, Statistic::RouteFlipRate]
                );
            }
            other => panic!("{depth:?}: {other:?}"),
        }
        match decide(&conflicting(depth, depth)) {
            PromotionDecision::Ambiguous { reason, .. } => {
                assert_eq!(reason, AmbiguityReason::ConflictingOrderingProxies);
            }
            other => panic!("{depth:?}: expected a refusal, got {other:?}"),
        }
    }
}

/// **`Worthless` is still excluded.** Stage 1 became admissibility, not
/// nothing: a move that buys no time is still not promotable.
#[test]
fn g5_stage_1_still_excludes_worthless() {
    let (_, registry, tail, _) = fixtures::pareto_candidate_template_at(PRICED);
    assert!(matches!(
        decide_promotion(&[], &registry, &tail),
        PromotionDecision::None { .. }
    ));
}

/// What stage 5 would actually read, at each rung of the ladder.
#[test]
fn what_the_physical_stage_sees_at_each_depth() {
    for (name, depth) in [("cheap", Depth::Cheap), ("priced", Depth::Priced)] {
        let c = candidate("X", depth, 3.0e-3, 1_000);
        let s = &c.promotion.assessment.ranking_score;
        println!(
            "{name:>7}: class={:?}/{}  gpu_ms_saved={}  readiness={:?}",
            s.class,
            s.class.tier(),
            s.gpu_ms_saved,
            c.promotion.readiness()
        );
        // Unchanged by DEPTH-2, and deliberately so: it is a PREDICTION
        // from `execution_cost::predict`, not measured authority.
        assert_eq!(s.gpu_ms_saved, 2.0);
    }
}
