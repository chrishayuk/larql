//! **REPRESENT-TERMINAL-1.** Observed frontier is not the resolved set.
//!
//! Frozen at `docs/represent/forecasts/represent-terminal-1.json`.
//!
//! Three stopping rules as HYPOTHESES, across a noise sweep, under both
//! exploration policies. The success criterion is the CHEAPEST rule that
//! yields policy-independent terminal preference — not the rule that
//! spends least in isolation.
//!
//! **R3's uncertainty box may not rank.** It answers exactly one
//! question — safely excludable, or not — and never orders the
//! candidates inside the unresolved set. A box that ranked would be a
//! new hidden comparator, which is the defect DEPTH-2 removed wearing a
//! different hat.
//!
//! **0/534 is not a prior for R1.** That figure came from conditioning
//! the DEPTH-2 rerun on "both searches priced their winner", which
//! selects for longer-running searches already more likely to have found
//! the truth. R1 is tested here from scratch, as a stopping rule.

use std::sync::OnceLock;

use super::decision::{decide_promotion, PromotionDecision, SearchCandidate};
use super::diagnostic::DiagnosticPolicy;
use super::measurement::TailSupportPolicy;
use super::promotion::PromotionCandidate;
use super::search_evidence::SearchCalibrationRegistry;
use super::state::fixtures;

const N: usize = 16;
const CHEAP: u64 = 8;
const PRICED: u64 = 500;
const BUDGET: u64 = 16_000;
const R: usize = 300;

/// The constant invented for FRONTIER-SPEND-1, calibrated to nothing.
/// The planted front spans 1.4e-3 of kl, so this is ~1.4x the entire
/// spread it perturbs.
const BASE_KL_NOISE: f64 = 1.0e-3;
const BASE_FLIP_NOISE: f64 = 500.0;
/// Multipliers of that constant. It is not privileged; it is 1.0 here.
const NOISE_SWEEP: [f64; 5] = [0.01, 0.1, 0.3, 1.0, 2.0];

struct Rng(u64);

impl Rng {
    fn new(seed: u64) -> Self {
        Self(seed | 1)
    }
    fn next_u64(&mut self) -> u64 {
        let mut x = self.0;
        x ^= x >> 12;
        x ^= x << 25;
        x ^= x >> 27;
        self.0 = x;
        x.wrapping_mul(0x2545_F491_4F6C_DD1D)
    }
    fn signed_unit(&mut self) -> f64 {
        (self.next_u64() % 2_000_001) as f64 / 1_000_000.0 - 1.0
    }
    fn below(&mut self, n: usize) -> usize {
        (self.next_u64() % n as u64) as usize
    }
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum Depth {
    Cheap,
    Priced,
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum Rule {
    WinnerOnly,
    ObservedFrontier,
    CompetitiveSet,
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum Policy {
    Uniform,
    Coverage,
}

#[derive(Clone, Copy)]
struct Truth {
    kl: f64,
    flips: u64,
}

#[derive(Clone, Copy)]
struct Reading {
    kl: f64,
    flips: f64,
    depth: Depth,
}

/// The planted landscape: 15 mutually non-dominated candidates and ONE
/// that dominates them all. `c15` is the answer the search is supposed
/// to find, and the witness that motivated this rung — under a noisy
/// cheap reading it can look dominated and leave the observed frontier.
const TRUE_DOMINATOR: usize = N - 1;

fn truth() -> Vec<Truth> {
    let mut v: Vec<Truth> = (0..N - 1)
        .map(|i| Truth {
            kl: 3.0e-3 + i as f64 * 1.0e-4,
            flips: 1_600 - i as u64 * 20,
        })
        .collect();
    v.push(Truth {
        kl: 2.5e-3,
        flips: 1_000,
    });
    v
}

fn cheap_reading(t: Truth, scale: f64, rng: &mut Rng) -> Reading {
    Reading {
        kl: (t.kl + rng.signed_unit() * BASE_KL_NOISE * scale).max(1.0e-5),
        flips: (t.flips as f64 + rng.signed_unit() * BASE_FLIP_NOISE * scale).max(1.0),
        depth: Depth::Cheap,
    }
}

type Templates = (PromotionCandidate, PromotionCandidate);
type Policies = (
    SearchCalibrationRegistry,
    TailSupportPolicy,
    DiagnosticPolicy,
);

fn ladder() -> &'static (Templates, Policies) {
    static L: OnceLock<(Templates, Policies)> = OnceLock::new();
    L.get_or_init(|| {
        let (cheap, registry, tail, policy) = fixtures::pareto_candidate_template_at(CHEAP);
        let (priced, ..) = fixtures::pareto_candidate_template_at(PRICED);
        ((cheap, priced), (registry, tail, policy))
    })
}

fn build(readings: &[Reading]) -> Vec<SearchCandidate> {
    let ((cheap, priced), (_, _, policy)) = ladder();
    readings
        .iter()
        .enumerate()
        .map(|(i, r)| {
            let promotion = match r.depth {
                Depth::Cheap => cheap,
                Depth::Priced => priced,
            };
            fixtures::pareto_candidate(
                &format!("c{i:02}"),
                promotion,
                policy,
                r.kl,
                r.flips.round() as u64,
            )
        })
        .collect()
}

fn decide(readings: &[Reading]) -> PromotionDecision {
    let (_, (registry, tail, _)) = ladder();
    decide_promotion(&build(readings), registry, tail)
}

fn index_of(id: &str) -> usize {
    id.trim_start_matches('c').parse().unwrap()
}

fn observed_frontier(decision: &PromotionDecision) -> Vec<usize> {
    let mut v: Vec<usize> = match decision {
        PromotionDecision::SelectForAuthority { candidate, .. } => vec![index_of(candidate)],
        PromotionDecision::Ambiguous { candidates, .. } => {
            candidates.iter().map(|c| index_of(c)).collect()
        }
        PromotionDecision::None { .. } => vec![],
    };
    v.sort_unstable();
    v
}

/// **The uncertainty box.** A cheap reading's true value lies within the
/// generator's own bound; a priced reading is exact.
///
/// This function answers ONE question and returns a bool. It must never
/// be used to order anything — see the module header.
fn safely_excluded_by(x: &Reading, leader: &Reading, scale: f64) -> bool {
    let (nk, nf) = (BASE_KL_NOISE * scale, BASE_FLIP_NOISE * scale);
    let (xk, xf) = match x.depth {
        // X at its BEST possible deep value.
        Depth::Cheap => (x.kl - nk, x.flips - nf),
        Depth::Priced => (x.kl, x.flips),
    };
    let (lk, lf) = match leader.depth {
        // The leader at its WORST possible deep value.
        Depth::Cheap => (leader.kl + nk, leader.flips + nf),
        Depth::Priced => (leader.kl, leader.flips),
    };
    // Lower is better on both. X is safely excluded only if the leader
    // still dominates it in that worst-versus-best comparison.
    lk <= xk && lf <= xf && (lk < xk || lf < xf)
}

/// Candidates that could still overturn the leader. Membership only.
fn competitive_set(readings: &[Reading], scale: f64) -> Vec<usize> {
    (0..readings.len())
        .filter(|&i| {
            !(0..readings.len())
                .any(|j| j != i && safely_excluded_by(&readings[i], &readings[j], scale))
        })
        .collect()
}

/// What this rule still requires to be priced before the search may stop.
fn outstanding(
    rule: Rule,
    readings: &[Reading],
    decision: &PromotionDecision,
    scale: f64,
) -> Vec<usize> {
    let required: Vec<usize> = match rule {
        Rule::WinnerOnly => match decision {
            PromotionDecision::SelectForAuthority { candidate, .. } => vec![index_of(candidate)],
            // No unique winner yet: the rule cannot stop, so the whole
            // observed frontier is what it must work on.
            _ => observed_frontier(decision),
        },
        Rule::ObservedFrontier => observed_frontier(decision),
        Rule::CompetitiveSet => competitive_set(readings, scale),
    };
    required
        .into_iter()
        .filter(|&i| readings[i].depth == Depth::Cheap)
        .collect()
}

fn choose(policy: Policy, options: &[usize], readings: &[Reading], rng: &mut Rng) -> usize {
    match policy {
        Policy::Uniform => options[rng.below(options.len())],
        Policy::Coverage => {
            let measured: Vec<(f64, f64)> = readings
                .iter()
                .filter(|r| r.depth == Depth::Priced)
                .map(|r| (r.kl * 1.0e6, r.flips))
                .collect();
            *options
                .iter()
                .max_by(|&&a, &&b| {
                    let d = |i: usize| {
                        let (x, y) = (readings[i].kl * 1.0e6, readings[i].flips);
                        measured
                            .iter()
                            .map(|&(mx, my)| ((x - mx).powi(2) + (y - my).powi(2)).sqrt())
                            .fold(f64::INFINITY, f64::min)
                    };
                    d(a).total_cmp(&d(b))
                        // Declared tie-break on VALUES, never identity.
                        .then_with(|| readings[b].kl.total_cmp(&readings[a].kl))
                        .then_with(|| readings[b].flips.total_cmp(&readings[a].flips))
                })
                .expect("non-empty")
        }
    }
}

struct Outcome {
    winner: Option<usize>,
    priced_runs: u64,
    position_equivalents: u64,
    /// Retained for the trace; the sweep scores winners, not reasons.
    #[allow(dead_code)]
    terminal: &'static str,
    /// The planted dominator left the observed frontier at some point
    /// while it was still only cheaply measured.
    dominator_escaped: bool,
}

fn search(rule: Rule, policy: Policy, scale: f64, seed: u64) -> Outcome {
    let truth = truth();
    let mut rng = Rng::new(seed);
    let mut readings: Vec<Reading> = truth
        .iter()
        .map(|&t| cheap_reading(t, scale, &mut rng))
        .collect();
    let mut spend = N as u64 * CHEAP;
    let mut priced_runs = 0u64;
    let mut dominator_escaped = false;

    loop {
        let decision = decide(&readings);
        let frontier = observed_frontier(&decision);
        if readings[TRUE_DOMINATOR].depth == Depth::Cheap && !frontier.contains(&TRUE_DOMINATOR) {
            dominator_escaped = true;
        }

        let outstanding = outstanding(rule, &readings, &decision, scale);
        if outstanding.is_empty() {
            let winner = match &decision {
                PromotionDecision::SelectForAuthority { candidate, .. } => {
                    Some(index_of(candidate))
                }
                _ => None,
            };
            return Outcome {
                winner,
                priced_runs,
                position_equivalents: spend,
                terminal: if winner.is_some() {
                    "resolved"
                } else {
                    "no-unique-preference"
                },
                dominator_escaped,
            };
        }
        if spend + PRICED > BUDGET {
            return Outcome {
                winner: None,
                priced_runs,
                position_equivalents: spend,
                terminal: "budget-exhausted",
                dominator_escaped,
            };
        }

        let pick = choose(policy, &outstanding, &readings, &mut rng);
        readings[pick] = Reading {
            kl: truth[pick].kl,
            flips: truth[pick].flips as f64,
            depth: Depth::Priced,
        };
        spend += PRICED;
        priced_runs += 1;
    }
}

#[derive(Default)]
struct Row {
    n: usize,
    violations: usize,
    hits: usize,
    misses: usize,
    unresolved: usize,
    priced: u64,
    positions: u64,
    escapes: usize,
}

#[test]
fn the_cheapest_rule_that_makes_terminal_preference_policy_independent() {
    println!(
        "{:>18} {:>7} {:>6} {:>6} {:>6} {:>7} {:>9} {:>8}",
        "rule", "noise", "viol", "hit", "miss", "unres", "priced", "escapes"
    );
    let mut summary: Vec<(Rule, f64, usize, f64)> = Vec::new();

    for rule in [
        Rule::WinnerOnly,
        Rule::ObservedFrontier,
        Rule::CompetitiveSet,
    ] {
        for scale in NOISE_SWEEP {
            let mut row = Row::default();
            for r in 0..R {
                let seed = 0x7E12_0000u64
                    .wrapping_add(r as u64)
                    .wrapping_mul((scale * 1000.0) as u64 + 1);
                let u = search(rule, Policy::Uniform, scale, seed);
                let c = search(rule, Policy::Coverage, scale, seed);
                row.n += 1;
                if u.winner != c.winner {
                    row.violations += 1;
                }
                for o in [&u, &c] {
                    match o.winner {
                        Some(TRUE_DOMINATOR) => row.hits += 1,
                        Some(_) => row.misses += 1,
                        None => row.unresolved += 1,
                    }
                    row.priced += o.priced_runs;
                    row.positions += o.position_equivalents;
                    if o.dominator_escaped {
                        row.escapes += 1;
                    }
                }
            }
            let searches = (row.n * 2) as f64;
            println!(
                "{:>18?} {:>7.2} {:>6} {:>6} {:>6} {:>7} {:>9.2} {:>8}",
                rule,
                scale,
                row.violations,
                row.hits,
                row.misses,
                row.unresolved,
                row.priced as f64 / searches,
                row.escapes
            );
            summary.push((rule, scale, row.violations, row.positions as f64 / searches));
        }
    }

    println!("\nPolicy-independent (0 violations) arms, cheapest first:");
    let mut clean: Vec<_> = summary.iter().filter(|s| s.2 == 0).collect();
    clean.sort_by(|a, b| a.3.total_cmp(&b.3));
    for (rule, scale, _, pos) in &clean {
        println!("  {rule:?} at noise {scale:.2}: {pos:.0} position-equivalents");
    }
    if clean.is_empty() {
        println!("  NONE — no rule achieved policy-independence at any noise level");
    }

    // ---- the findings, pinned ----

    let at = |rule: Rule, scale: f64| {
        summary
            .iter()
            .find(|s| s.0 == rule && s.1 == scale)
            .expect("swept")
    };

    // **R1 and R2 are the same rule on this landscape.** When a unique
    // winner exists the observed frontier IS that winner, so "price the
    // winner" and "price the frontier" name the same set; and when no
    // winner exists R1 cannot stop either. The distinction the freeze
    // drew does not exist operationally.
    for scale in NOISE_SWEEP {
        let (_, _, v1, p1) = at(Rule::WinnerOnly, scale);
        let (_, _, v2, p2) = at(Rule::ObservedFrontier, scale);
        assert_eq!((v1, p1), (v2, p2), "R1 and R2 diverged at noise {scale}");
    }

    // **The cheap rules are policy-independent up to the invented
    // constant, and fail beyond it.**
    assert_eq!(at(Rule::WinnerOnly, 1.00).2, 0);
    assert!(
        at(Rule::WinnerOnly, 2.00).2 > 0,
        "at twice the invented noise the cheap rule must start failing"
    );

    // **The competitive set is policy-independent at every level** — and
    // pays for it by pricing nearly everything.
    for scale in NOISE_SWEEP {
        assert_eq!(
            at(Rule::CompetitiveSet, scale).2,
            0,
            "R3 lost policy-independence at noise {scale}"
        );
    }

    // **The ranking is NOT stable across the sweep**, which is the
    // result: the cheapest sufficient rule depends on a constant nobody
    // calibrated.
    assert!(
        at(Rule::CompetitiveSet, 1.00).3 > at(Rule::WinnerOnly, 1.00).3,
        "at the invented constant R3 must be the more expensive insurance"
    );
}
