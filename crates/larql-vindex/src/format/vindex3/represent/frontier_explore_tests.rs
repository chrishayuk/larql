//! **REPRESENT-FRONTIER-EXPLORE-1, first wave.**
//!
//! Frozen at
//! `docs/represent/forecasts/represent-frontier-explore-1.json`.
//!
//! FRONTIER-SCALE-1 measured 99.1% refusal at n=128. Nothing here makes
//! that number smaller. `decide_promotion` is untouched and is the sole
//! authority on WHICH candidates are non-dominated — these policies
//! never re-derive dominance, they read the frontier it names.
//!
//! **The invariant the whole rung protects:** an exploration policy may
//! decide WHAT TO MEASURE. Only admitted evidence may change
//! PREFERENCE. Every test below asserts the decision is unchanged by
//! the act of exploring.
//!
//! **Nothing here is production vocabulary.** `AuthorityChoice` lives in
//! this test module until the experiment says which policy is worth
//! having. The freeze records the collision with
//! `PromotionDecision::SelectForAuthority`, which already means "spend
//! measurement here"; promoting a name before the evidence is exactly
//! the ordering this programme avoids.

use super::decision::{decide_promotion, AmbiguityReason, PromotionDecision, SearchCandidate};
use super::state::fixtures;
use super::statistic::Statistic;

/// Why authority was spent here. Never why a candidate is preferred.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Reason {
    RandomFrontier,
    FrontierCoverage,
    CallerObjective,
}

#[derive(Debug, Clone, PartialEq)]
struct AuthorityChoice {
    candidate: String,
    reason: Reason,
}

/// Deterministic xorshift64*, as FRONTIER-SCALE-1 uses. A seeded policy
/// is reproducible; an unseeded one cannot be permutation-tested.
struct Rng(u64);

impl Rng {
    fn new(seed: u64) -> Self {
        Self(seed | 1)
    }
    fn below(&mut self, n: usize) -> usize {
        let mut x = self.0;
        x ^= x >> 12;
        x ^= x << 25;
        x ^= x >> 27;
        self.0 = x;
        (x.wrapping_mul(0x2545_F491_4F6C_DD1D) % n as u64) as usize
    }
}

/// The frontier, as `decide_promotion` names it. Dominance is never
/// recomputed here: a second implementation would be a second opinion.
fn frontier<'a>(
    candidates: &'a [SearchCandidate],
    decision: &PromotionDecision,
) -> Vec<&'a SearchCandidate> {
    let ids: Vec<&str> = match decision {
        PromotionDecision::SelectForAuthority { candidate, .. } => vec![candidate.as_str()],
        PromotionDecision::Ambiguous { candidates, .. } => {
            candidates.iter().map(String::as_str).collect()
        }
        PromotionDecision::None { .. } => vec![],
    };
    let mut out: Vec<&SearchCandidate> = candidates
        .iter()
        .filter(|c| ids.contains(&c.id.as_str()))
        .collect();
    // Canonicalise so a seeded draw is reproducible. Identity ORDERS the
    // set here; it never decides which member is chosen. That is the
    // same licence `display_order` takes and `decide_promotion` refuses.
    out.sort_by(|a, b| a.id.cmp(&b.id));
    out
}

/// A candidate's position in evidence space: the two registered
/// ordering proxies, and nothing physical.
fn evidence_point(c: &SearchCandidate) -> (f64, f64) {
    let read = |s: Statistic| {
        c.diagnostic
            .reading(s)
            .and_then(|r| r.observed)
            .expect("both proxies are observed in these fixtures")
    };
    (read(Statistic::KlP99), read(Statistic::RouteFlipRate))
}

/// **P1 — uniform random over the frontier.**
fn p1_random(frontier: &[&SearchCandidate], rng: &mut Rng) -> AuthorityChoice {
    let pick = rng.below(frontier.len());
    AuthorityChoice {
        candidate: frontier[pick].id.clone(),
        reason: Reason::RandomFrontier,
    }
}

/// **P2 — coverage.** The frontier member furthest, in evidence space,
/// from everything already measured. Reads no physical cost.
///
/// Declared tie-break: among equidistant members, the smaller evidence
/// VECTOR wins. Values, never identity, never list order.
fn p2_coverage(frontier: &[&SearchCandidate], measured: &[(f64, f64)]) -> AuthorityChoice {
    let scale = |(k, r): (f64, f64)| (k * 1.0e6, r * 1.0e4);
    let best = frontier
        .iter()
        .max_by(|a, b| {
            let d = |c: &SearchCandidate| {
                let (x, y) = scale(evidence_point(c));
                measured
                    .iter()
                    .map(|&m| {
                        let (mx, my) = scale(m);
                        ((x - mx).powi(2) + (y - my).powi(2)).sqrt()
                    })
                    .fold(f64::INFINITY, f64::min)
            };
            d(a).total_cmp(&d(b))
                .then_with(|| evidence_point(b).0.total_cmp(&evidence_point(a).0))
                .then_with(|| evidence_point(b).1.total_cmp(&evidence_point(a).1))
        })
        .expect("a non-empty frontier");
    AuthorityChoice {
        candidate: best.id.clone(),
        reason: Reason::FrontierCoverage,
    }
}

/// An externally declared deployment objective.
///
/// **Restricted to quality terms in this wave.** "Minimise measured
/// latency" would read `gpu_ms_saved`, which descends from a beta model
/// with no realization index, so latency objectives wait for
/// REALIZATION-COST-1 exactly as gain-ranked exploration does.
#[derive(Debug, Clone, Copy)]
struct Objective {
    /// Refuse any candidate whose kl p99 exceeds this.
    kl_ceiling: f64,
    /// Among those that pass, minimise this statistic.
    minimise: Statistic,
}

/// **P3 — caller objective.** Not evidence manufacturing a winner: the
/// objective arrives from outside, is recorded as the reason, and
/// changes nothing about what the evidence supports.
fn p3_objective(frontier: &[&SearchCandidate], o: Objective) -> Option<AuthorityChoice> {
    let read = |c: &SearchCandidate, s: Statistic| {
        c.diagnostic.reading(s).and_then(|r| r.observed).unwrap()
    };
    frontier
        .iter()
        .filter(|c| read(c, Statistic::KlP99) <= o.kl_ceiling)
        .min_by(|a, b| {
            read(a, o.minimise)
                .total_cmp(&read(b, o.minimise))
                // Declared tie-break on the OTHER proxy's value.
                .then_with(|| read(a, Statistic::KlP99).total_cmp(&read(b, Statistic::KlP99)))
        })
        .map(|c| AuthorityChoice {
            candidate: c.id.clone(),
            reason: Reason::CallerObjective,
        })
}

/// A conflicting pair: A better on kl, B better on route flips.
type Policies = (
    super::search_evidence::SearchCalibrationRegistry,
    super::measurement::TailSupportPolicy,
    super::diagnostic::DiagnosticPolicy,
    super::promotion::PromotionCandidate,
);

fn conflicting_pair() -> (Vec<SearchCandidate>, Policies) {
    let (promotion, registry, tail, policy) = fixtures::pareto_candidate_template();
    let candidates = vec![
        fixtures::pareto_candidate("A", &promotion, &policy, 3.4000e-3, 1600),
        fixtures::pareto_candidate("B", &promotion, &policy, 3.9000e-3, 1200),
    ];
    (candidates, (registry, tail, policy, promotion))
}

/// **The central preregistration: conflict is not uncertainty.**
///
/// Re-measuring both candidates to ever finer precision never resolves
/// a Pareto conflict, because the conflict is between what the proxies
/// SAY, not between what is known about them. Expected runs to a unique
/// evidence preference is INFINITE.
#[test]
fn measuring_more_precisely_never_resolves_a_conflict() {
    let (base, (registry, tail, policy, promotion)) = conflicting_pair();
    let decision = decide_promotion(&base, &registry, &tail);
    assert!(matches!(
        decision,
        PromotionDecision::Ambiguous {
            reason: AmbiguityReason::ConflictingOrderingProxies,
            ..
        }
    ));

    // 64 successive "re-measurements": the same trade-off, resolved to
    // finer and finer precision. Nothing converges, because there is
    // nothing to converge to.
    for round in 1..=64u32 {
        let eps = 1.0e-9 * round as f64;
        let candidates = vec![
            fixtures::pareto_candidate("A", &promotion, &policy, 3.4000e-3 + eps, 1600),
            fixtures::pareto_candidate("B", &promotion, &policy, 3.9000e-3 - eps, 1200),
        ];
        match decide_promotion(&candidates, &registry, &tail) {
            PromotionDecision::Ambiguous {
                reason: AmbiguityReason::ConflictingOrderingProxies,
                ..
            } => {}
            other => panic!("round {round} converged, which cannot happen: {other:?}"),
        }
    }

    // Exit 1: a NEW candidate that dominates both. The candidate SET
    // changed — not the evidence about the original two.
    let mut widened = base.clone();
    widened.push(fixtures::pareto_candidate(
        "C", &promotion, &policy, 3.0000e-3, 1000,
    ));
    match decide_promotion(&widened, &registry, &tail) {
        PromotionDecision::SelectForAuthority {
            candidate,
            evidence,
        } => {
            assert_eq!(candidate, "C");
            assert_eq!(evidence.dominated, vec!["A".to_string(), "B".to_string()]);
        }
        other => panic!("a dominating candidate must resolve it, got {other:?}"),
    }
}

/// **Exit 2: a caller objective licenses the trade** — and two different
/// objectives choose differently from the SAME frontier, which is the
/// K3-shaped use case in miniature.
#[test]
fn different_caller_objectives_choose_differently_from_one_frontier() {
    let (candidates, (registry, tail, ..)) = conflicting_pair();
    let decision = decide_promotion(&candidates, &registry, &tail);
    let f = frontier(&candidates, &decision);
    assert_eq!(f.len(), 2, "both members are non-dominated");

    let quality_first = p3_objective(
        &f,
        Objective {
            kl_ceiling: 1.0,
            minimise: Statistic::KlP99,
        },
    )
    .expect("something passes");
    let routing_first = p3_objective(
        &f,
        Objective {
            kl_ceiling: 1.0,
            minimise: Statistic::RouteFlipRate,
        },
    )
    .expect("something passes");

    assert_eq!(quality_first.candidate, "A");
    assert_eq!(routing_first.candidate, "B");
    assert_eq!(quality_first.reason, Reason::CallerObjective);

    // A ceiling that excludes B leaves only A, whatever is minimised.
    let constrained = p3_objective(
        &f,
        Objective {
            kl_ceiling: 3.5000e-3,
            minimise: Statistic::RouteFlipRate,
        },
    )
    .expect("A passes the ceiling");
    assert_eq!(constrained.candidate, "A");

    // And the objective changed NOTHING about what the evidence says.
    assert_eq!(decide_promotion(&candidates, &registry, &tail), decision);
}

/// **Tie discipline.** Every policy must survive a permuted frontier.
#[test]
fn no_policy_falls_through_to_list_order() {
    let (candidates, (registry, tail, ..)) = conflicting_pair();
    let decision = decide_promotion(&candidates, &registry, &tail);

    let forward = frontier(&candidates, &decision);
    let mut reversed_input = candidates.clone();
    reversed_input.reverse();
    let reversed = frontier(&reversed_input, &decision);

    // P1: same seed, same choice, whatever order the input arrived in.
    assert_eq!(
        p1_random(&forward, &mut Rng::new(0xA11CE)),
        p1_random(&reversed, &mut Rng::new(0xA11CE))
    );
    // P2: deterministic, and its tie-break is on values.
    let measured = [(3.4000e-3, 1600.0 / 8192.0)];
    assert_eq!(
        p2_coverage(&forward, &measured),
        p2_coverage(&reversed, &measured)
    );
    // P3: likewise.
    let o = Objective {
        kl_ceiling: 1.0,
        minimise: Statistic::KlP99,
    };
    assert_eq!(p3_objective(&forward, o), p3_objective(&reversed, o));
}

/// **P1 is uniform, and P2 covers.**
#[test]
fn random_is_uniform_and_coverage_moves_away_from_what_is_measured() {
    let (promotion, registry, tail, policy) = fixtures::pareto_candidate_template();
    // Four mutually non-dominated candidates: kl improves as flips worsen.
    let candidates: Vec<SearchCandidate> = (0..4)
        .map(|i| {
            fixtures::pareto_candidate(
                &format!("m{i}"),
                &promotion,
                &policy,
                3.0e-3 + i as f64 * 1.0e-4,
                1600 - i as u64 * 100,
            )
        })
        .collect();
    let decision = decide_promotion(&candidates, &registry, &tail);
    let f = frontier(&candidates, &decision);
    assert_eq!(f.len(), 4, "all four are non-dominated");

    let mut counts = [0usize; 4];
    let mut rng = Rng::new(0x5EED);
    for _ in 0..4000 {
        let c = p1_random(&f, &mut rng);
        let i: usize = c.candidate.trim_start_matches('m').parse().unwrap();
        counts[i] += 1;
    }
    println!("P1 uniform over 4 frontier members: {counts:?}");
    for c in counts {
        assert!((c as f64 - 1000.0).abs() < 120.0, "not uniform: {counts:?}");
    }

    // P2: having measured the kl-best corner, coverage must choose the
    // far end rather than its neighbour.
    let chosen = p2_coverage(&f, &[evidence_point(f[0])]);
    assert_eq!(chosen.candidate, "m3");
    assert_eq!(chosen.reason, Reason::FrontierCoverage);
}
