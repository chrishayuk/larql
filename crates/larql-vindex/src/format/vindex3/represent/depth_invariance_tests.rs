//! **REPRESENT-DEPTH-1, against the CURRENT contract.**
//!
//! Frozen at `docs/represent/forecasts/represent-depth-1.json`.
//!
//! > Evidence strength determines which conclusions are licensed; it is
//! > not evidence that a candidate is superior.
//!
//! Every arm states what the CONTRACT requires and what the current
//! implementation is predicted to do. Where they differ, the test
//! asserts the PREDICTION and names the contract requirement it
//! violates, so the defect is pinned rather than the hope. No remedy is
//! implemented here.

use super::assessment::MoveClass;
use super::decision::{decide_promotion, AmbiguityReason, PromotionDecision, SearchCandidate};
use super::state::fixtures;

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

/// **A3 — THE HEADLINE. Price only the loser, and the loser wins.**
///
/// CONTRACT REQUIRES: A still wins. B being measured harder is not a
/// reason to prefer it.
///
/// CURRENT: B is selected. A candidate that LOST on the evidence wins by
/// having been measured more.
#[test]
fn a3_pricing_only_the_loser_makes_the_loser_win() {
    // Baseline: at equal cheap depth, A wins on the evidence.
    let equal = winner_and_loser(Depth::Cheap, Depth::Cheap);
    match decide(&equal) {
        PromotionDecision::SelectForAuthority {
            candidate,
            evidence,
        } => {
            assert_eq!(candidate, "A", "A dominates on both proxies");
            assert_eq!(evidence.dominated, vec!["B".to_string()]);
            assert!(!evidence.deciding.is_empty(), "and wins ON the evidence");
        }
        other => panic!("expected A to win at equal depth, got {other:?}"),
    }

    // Now price ONLY the loser. Nothing about either candidate's
    // observed values changed.
    let mixed = winner_and_loser(Depth::Cheap, Depth::Priced);
    println!("A3 classes: {:?}", classes(&mixed));
    let decision = decide(&mixed);
    println!("A3 decision: {decision:?}");

    match decision {
        PromotionDecision::SelectForAuthority {
            candidate,
            evidence,
        } => {
            assert_eq!(
                candidate, "B",
                "PINNED DEFECT: the loser wins. The contract requires A."
            );
            assert!(
                evidence.dominated.is_empty() && evidence.deciding.is_empty(),
                "and it wins on NO evidence — nothing was compared"
            );
        }
        other => panic!("expected the defect to fire, got {other:?}"),
    }
}

/// **A4 — price only the winner: the right candidate, the wrong reason.**
///
/// CONTRACT REQUIRES: A wins, `dominated` names B, `deciding` names the
/// separating proxies.
///
/// CURRENT: A wins with EMPTY evidence. The answer is right and the
/// justification is hollow, which is why asserting the evidence and not
/// just the winner is what catches this arm.
#[test]
fn a4_pricing_only_the_winner_hollows_out_the_justification() {
    let mixed = winner_and_loser(Depth::Priced, Depth::Cheap);
    let decision = decide(&mixed);
    println!("A4 decision: {decision:?}");
    match decision {
        PromotionDecision::SelectForAuthority {
            candidate,
            evidence,
        } => {
            assert_eq!(candidate, "A", "the right candidate");
            assert!(
                evidence.dominated.is_empty() && evidence.deciding.is_empty(),
                "PINNED DEFECT: won on tier, not on evidence. The contract \
                 requires `dominated` to name B and `deciding` to name the \
                 separating proxies."
            );
        }
        other => panic!("expected a hollow win, got {other:?}"),
    }
}

/// **A1 / A2 — a conflict is not resolved by depth, and the mirror
/// proves it is depth and not merit.**
///
/// CONTRACT REQUIRES: `Ambiguous{ConflictingOrderingProxies}` in both.
///
/// CURRENT: whichever side was priced is selected, on no evidence.
#[test]
fn a1_a2_pricing_either_side_of_a_conflict_fabricates_a_winner() {
    // Baseline: at equal depth the conflict is refused, correctly.
    match decide(&conflicting(Depth::Cheap, Depth::Cheap)) {
        PromotionDecision::Ambiguous { reason, .. } => {
            assert_eq!(reason, AmbiguityReason::ConflictingOrderingProxies);
        }
        other => panic!("expected a refusal at equal depth, got {other:?}"),
    }

    for (a, b, expected) in [
        (Depth::Priced, Depth::Cheap, "A"),
        (Depth::Cheap, Depth::Priced, "B"),
    ] {
        let decision = decide(&conflicting(a, b));
        println!("A1/A2 priced={expected}: {decision:?}");
        match decision {
            PromotionDecision::SelectForAuthority {
                candidate,
                evidence,
            } => {
                assert_eq!(
                    candidate, expected,
                    "PINNED DEFECT: the priced side is selected. The contract \
                     requires the conflict to stand."
                );
                assert!(evidence.deciding.is_empty());
            }
            other => panic!("expected the defect to fire, got {other:?}"),
        }
    }
}

/// **A5 — the control. Uniform depth is unaffected.**
///
/// This is what makes the defect specific to UNEQUAL depth, and why
/// PARETO-1 and FRONTIER-SCALE-1 remain valid: both hold every candidate
/// at one depth, so stage 1 is a no-op there.
#[test]
fn a5_uniform_depth_behaves_correctly_at_either_rung() {
    for depth in [Depth::Cheap, Depth::Priced] {
        match decide(&winner_and_loser(depth, depth)) {
            PromotionDecision::SelectForAuthority {
                candidate,
                evidence,
            } => {
                assert_eq!(candidate, "A");
                assert_eq!(evidence.dominated, vec!["B".to_string()]);
                assert!(!evidence.deciding.is_empty(), "{depth:?}: won on evidence");
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

/// **The invariant, stated as a test.**
///
/// Identity, physical facts and observed ordering values are all held
/// fixed; only one candidate's measurement depth changes. The decision
/// must not move. It does.
#[test]
fn changing_only_one_candidates_depth_changes_the_decision() {
    let before = decide(&winner_and_loser(Depth::Cheap, Depth::Cheap));
    let after = decide(&winner_and_loser(Depth::Cheap, Depth::Priced));
    assert_ne!(
        format!("{before:?}"),
        format!("{after:?}"),
        "PINNED DEFECT: this assertion is INVERTED. The contract requires \
         these to be EQUAL — nothing changed except how hard B was looked \
         at. When the remedy lands, this test must be rewritten to \
         assert_eq and this comment deleted."
    );
    println!("invariant violated:\n  before {before:?}\n  after  {after:?}");
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
    }
}
