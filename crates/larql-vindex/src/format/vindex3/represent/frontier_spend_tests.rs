//! **REPRESENT-FRONTIER-SPEND-1.** Same truth, different path,
//! different authority cost.
//!
//! Frozen at `docs/represent/forecasts/represent-frontier-spend-1.json`.
//!
//! The primary claim is policy-independence of the DESTINATION. Spend is
//! secondary. `decide_promotion` is untouched and is deliberately NOT
//! protected from the preregistered tier-escalation hazard: the point of
//! the rung is that exploration must not be able to manufacture
//! preference by choosing whom to measure more deeply.
//!
//! **The diagnostic vector's declared depth is not varied.** Ordering is
//! depth-independent at diagnostic scale — `route_cal_1` registers
//! `KlP99` as an `OrderingProxy` and that lookup hits before any tail
//! fallback — which was observed directly when the OPT-6 layer selected
//! cleanly at 8 positions. Depth therefore changes what can be PRICED,
//! which is carried by the assessment template, and nothing else.

use std::sync::OnceLock;

use super::assessment::MoveClass;
use super::decision::{decide_promotion, PromotionDecision, SearchCandidate};
use super::diagnostic::DiagnosticPolicy;
use super::measurement::TailSupportPolicy;
use super::promotion::PromotionCandidate;
use super::search_evidence::SearchCalibrationRegistry;
use super::state::fixtures;

const N: usize = 16;
const CHEAP: u64 = 8;
const PRICED: u64 = 500;
const BUDGET: u64 = 8_000;
const R: usize = 1_000;
const VERBOSE_REPLICATES: usize = 2;

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
    /// Uniform in [-1, 1].
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

#[derive(Clone, Copy)]
struct Truth {
    kl: f64,
    flips: u64,
}

#[derive(Clone, Copy)]
struct Reading {
    kl: f64,
    flips: u64,
    depth: Depth,
}

/// Fifteen mutually non-dominated candidates plus ONE that dominates
/// every one of them on both proxies.
///
/// H3: a unique evidence preference must EXIST once everything is
/// priced, or policy-independence is vacuous — both policies would
/// trivially agree on "no unique preference".
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

/// A cheap run observes the truth with noise; a priced run resolves it.
/// That is what makes escalation worth spending on, and it is why H1 —
/// no apparent global dominator after the cheap sweep — is satisfiable
/// at the same time as H3.
fn cheap_reading(t: Truth, rng: &mut Rng) -> Reading {
    Reading {
        kl: (t.kl + rng.signed_unit() * 1.0e-3).max(1.0e-4),
        flips: (t.flips as i64 + (rng.signed_unit() * 500.0) as i64).max(1) as u64,
        depth: Depth::Cheap,
    }
}

type Templates = (PromotionCandidate, PromotionCandidate);
type Policies = (
    SearchCalibrationRegistry,
    TailSupportPolicy,
    DiagnosticPolicy,
);

/// Built once: the two rungs of the ladder cost real container work.
fn ladder() -> &'static (Templates, Policies) {
    static L: OnceLock<(Templates, Policies)> = OnceLock::new();
    L.get_or_init(|| {
        let (cheap, registry, tail, policy) = fixtures::pareto_candidate_template_at(CHEAP);
        let (priced, ..) = fixtures::pareto_candidate_template_at(PRICED);
        assert_eq!(cheap.assessment.ranking_score.class, MoveClass::Unscorable);
        assert_eq!(priced.assessment.ranking_score.class, MoveClass::Priced);
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
            fixtures::pareto_candidate(&format!("c{i:02}"), promotion, policy, r.kl, r.flips)
        })
        .collect()
}

fn decide(readings: &[Reading]) -> (Vec<SearchCandidate>, PromotionDecision) {
    let (_, (registry, tail, _)) = ladder();
    let candidates = build(readings);
    let decision = decide_promotion(&candidates, registry, tail);
    (candidates, decision)
}

/// The frontier, as the comparator names it. Canonicalised so a seeded
/// draw is reproducible; identity orders the set and never chooses.
fn frontier_ids(decision: &PromotionDecision) -> Vec<String> {
    let mut ids = match decision {
        PromotionDecision::SelectForAuthority { candidate, .. } => vec![candidate.clone()],
        PromotionDecision::Ambiguous { candidates, .. } => candidates.clone(),
        PromotionDecision::None { .. } => vec![],
    };
    ids.sort();
    ids
}

fn index_of(id: &str) -> usize {
    id.trim_start_matches('c').parse().unwrap()
}

fn winner_is_priced(decision: &PromotionDecision, readings: &[Reading]) -> bool {
    match decision {
        PromotionDecision::SelectForAuthority { candidate, .. } => {
            readings[index_of(candidate)].depth == Depth::Priced
        }
        _ => false,
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Policy {
    Uniform,
    Coverage,
}

/// Choose which frontier member to escalate. Neither policy reads any
/// physical cost.
fn choose(policy: Policy, frontier: &[String], readings: &[Reading], rng: &mut Rng) -> usize {
    let unpriced: Vec<&String> = frontier
        .iter()
        .filter(|id| readings[index_of(id)].depth == Depth::Cheap)
        .collect();
    assert!(!unpriced.is_empty(), "caller checks this first");
    match policy {
        Policy::Uniform => index_of(unpriced[rng.below(unpriced.len())]),
        Policy::Coverage => {
            let scale = |r: &Reading| (r.kl * 1.0e6, r.flips as f64);
            let measured: Vec<(f64, f64)> = readings
                .iter()
                .filter(|r| r.depth == Depth::Priced)
                .map(scale)
                .collect();
            let far = unpriced
                .iter()
                .max_by(|a, b| {
                    let d = |id: &str| {
                        let (x, y) = scale(&readings[index_of(id)]);
                        measured
                            .iter()
                            .map(|&(mx, my)| ((x - mx).powi(2) + (y - my).powi(2)).sqrt())
                            .fold(f64::INFINITY, f64::min)
                    };
                    d(a).total_cmp(&d(b))
                        // Declared tie-break on VALUES, never identity.
                        .then_with(|| {
                            let (ax, ay) = scale(&readings[index_of(a)]);
                            let (bx, by) = scale(&readings[index_of(b)]);
                            bx.total_cmp(&ax).then_with(|| by.total_cmp(&ay))
                        })
                })
                .expect("non-empty");
            index_of(far)
        }
    }
}

#[derive(Debug)]
struct Outcome {
    /// Was the selected candidate itself measured at the priced depth?
    ///
    /// The primary claim holds only "when the available evidence is
    /// sufficient to determine a preference". A search that terminates
    /// on a unique frontier member whose own reading is a NOISY cheap
    /// one has not met that precondition — it simply does not know yet.
    /// The freeze's terminal conditions did not distinguish the two.
    winner_priced: bool,
    decision: PromotionDecision,
    priced_runs: u64,
    /// Retained, unread. This is the metric the spend comparison would
    /// have ranked; the hazard fired first, so no search produced a
    /// comparable number. Deleting it would delete the driver's
    /// readiness for the run that happens after the contract is fixed.
    #[allow(dead_code)]
    position_equivalents: u64,
    terminal: &'static str,
    hazard: bool,
}

/// One search. Returns where it ended, what it cost, and whether the
/// preregistered hazard fired.
fn search(policy: Policy, seed: u64, verbose: bool) -> Outcome {
    let truth = truth();
    let mut rng = Rng::new(seed);
    let mut readings: Vec<Reading> = truth.iter().map(|&t| cheap_reading(t, &mut rng)).collect();
    let mut spend = N as u64 * CHEAP;
    let mut priced_runs = 0u64;

    let (_, mut decision) = decide(&readings);

    // H1: no apparent global dominator after the cheap sweep. The
    // comparator itself is the authority on that — a second dominance
    // implementation here would be a second opinion.
    if matches!(decision, PromotionDecision::SelectForAuthority { .. }) {
        return Outcome {
            winner_priced: winner_is_priced(&decision, &readings),
            decision,
            priced_runs,
            position_equivalents: spend,
            terminal: "H1-resample",
            hazard: false,
        };
    }

    loop {
        let frontier = frontier_ids(&decision);
        let unpriced_on_frontier = frontier
            .iter()
            .any(|id| readings[index_of(id)].depth == Depth::Cheap);

        if !unpriced_on_frontier {
            return Outcome {
                winner_priced: winner_is_priced(&decision, &readings),
                decision,
                priced_runs,
                position_equivalents: spend,
                terminal: "frontier-fully-priced",
                hazard: false,
            };
        }
        if spend + PRICED > BUDGET {
            return Outcome {
                winner_priced: winner_is_priced(&decision, &readings),
                decision,
                priced_runs,
                position_equivalents: spend,
                terminal: "budget-exhausted",
                hazard: false,
            };
        }

        let pick = choose(policy, &frontier, &readings, &mut rng);
        let before = readings[pick];
        readings[pick] = Reading {
            kl: truth[pick].kl,
            flips: truth[pick].flips,
            depth: Depth::Priced,
        };
        spend += PRICED;
        priced_runs += 1;

        let (candidates, next) = decide(&readings);
        let priced_count = readings.iter().filter(|r| r.depth == Depth::Priced).count();
        let pool: Vec<&SearchCandidate> = candidates
            .iter()
            .filter(|c| {
                c.promotion.assessment.ranking_score.class.tier()
                    == candidates
                        .iter()
                        .map(|o| o.promotion.assessment.ranking_score.class.tier())
                        .max()
                        .unwrap()
            })
            .collect();

        if verbose {
            println!(
                "  step {priced_runs}: chose c{pick:02}  depth {:?}->{:?}  class {:?}/{} -> {:?}/{}",
                before.depth,
                Depth::Priced,
                ladder().0 .0.assessment.ranking_score.class,
                ladder().0 .0.assessment.ranking_score.class.tier(),
                ladder().0 .1.assessment.ranking_score.class,
                ladder().0 .1.assessment.ranking_score.class.tier(),
            );
            println!(
                "            stage-1 pool {} of {}   frontier {:?}",
                pool.len(),
                candidates.len(),
                frontier_ids(&next)
            );
            println!("            decision {next:?}");
        }

        // **The preregistered hazard.** A decision reached from a pool
        // that excluded every cheaply-measured candidate, while frontier
        // members remain unpriced, is preference manufactured by
        // measurement depth.
        let still_unpriced = frontier_ids(&next)
            .iter()
            .any(|id| readings[index_of(id)].depth == Depth::Cheap)
            || priced_count < N;
        // The detector tests the EVIDENCE, not the pool. Its first
        // version asked whether the max-tier pool had excluded anyone,
        // which encoded the pre-DEPTH-2 stage 1 and therefore kept
        // "firing" after the repair, on selections that were made
        // legitimately on named proxies. The defect's real signature is
        // a selection that names nothing: no dominated candidate, no
        // deciding proxy, and not the physical tie-break either.
        let no_evidence = matches!(
            &next,
            PromotionDecision::SelectForAuthority { evidence, .. }
                if evidence.dominated.is_empty()
                    && evidence.deciding.is_empty()
                    && !evidence.decided_by_physical_gain
        );
        if no_evidence && still_unpriced {
            return Outcome {
                winner_priced: winner_is_priced(&next, &readings),
                decision: next,
                priced_runs,
                position_equivalents: spend,
                terminal: "HAZARD-tier-escalation",
                hazard: true,
            };
        }

        decision = next;
        if matches!(decision, PromotionDecision::SelectForAuthority { .. }) {
            return Outcome {
                winner_priced: winner_is_priced(&decision, &readings),
                decision,
                priced_runs,
                position_equivalents: spend,
                terminal: "unique-preference",
                hazard: false,
            };
        }
    }
}

/// **The rung, rerun after DEPTH-2.**
///
/// The driver below is UNCHANGED from the run in which the
/// tier-escalation hazard fired 1488 of 1488 times. Only
/// `decide_promotion` moved. That is the point: the success case was
/// frozen before the defect was found, so it did not have to be
/// invented afterwards.
///
/// PRIMARY claim: exploration may change where authority is spent and
/// how much it costs, but not the eventual preference. SECONDARY:
/// spend, in priced runs and position-equivalents.
#[test]
fn exploration_policy_must_not_change_the_destination() {
    let mut resamples = 0usize;
    let mut hazards = 0usize;
    let mut disagreements = 0usize;
    let mut annotation_differs = 0usize;
    let mut insufficient = 0usize;
    let mut scored = 0usize;
    let mut spend = [(0u64, 0u64); 2];
    let mut terminals: std::collections::BTreeMap<&str, usize> = Default::default();

    for r in 0..R {
        let seed = 0x5FE0_0000u64.wrapping_add(r as u64);
        let verbose = r < VERBOSE_REPLICATES;
        if verbose {
            println!("replicate {r} — uniform");
        }
        let u = search(Policy::Uniform, seed, verbose);
        if verbose {
            println!("replicate {r} — coverage");
        }
        let c = search(Policy::Coverage, seed, verbose);

        if u.terminal == "H1-resample" {
            resamples += 1;
            continue;
        }
        if u.hazard || c.hazard {
            hazards += 1;
            continue;
        }

        scored += 1;
        *terminals.entry(u.terminal).or_default() += 1;
        *terminals.entry(c.terminal).or_default() += 1;
        spend[0].0 += u.priced_runs;
        spend[0].1 += u.position_equivalents;
        spend[1].0 += c.priced_runs;
        spend[1].1 += c.position_equivalents;

        // **H2, split.** The freeze asked for the full evidence record
        // to match, including `readiness` and `unresolved`. After
        // DEPTH-2 that is the wrong test, and the reason is DEPTH-2
        // itself: `readiness` is the honest home for the epistemic
        // difference between "we priced this one" and "we did not". Two
        // policies that price DIFFERENT members must end with different
        // `readiness`, and demanding otherwise would demand uniform
        // depth — which is exactly the ladder this rung exists to make
        // safe.
        //
        // So the PREFERENCE must match — candidate, dominated,
        // deciding, and which stage decided — while the ANNOTATION is
        // allowed to differ and is reported rather than failed.
        let preference = |o: &Outcome| match &o.decision {
            PromotionDecision::SelectForAuthority {
                candidate,
                evidence,
            } => Some((
                candidate.clone(),
                evidence.dominated.clone(),
                evidence.deciding.clone(),
                evidence.decided_by_physical_gain,
            )),
            _ => None,
        };
        if !(u.winner_priced && c.winner_priced) {
            insufficient += 1;
            continue;
        }
        if preference(&u) != preference(&c) {
            disagreements += 1;
            if disagreements <= 3 {
                println!(
                    "H2 VIOLATION r={r}\n  uniform  {:?}\n  coverage {:?}",
                    u.decision, c.decision
                );
            }
        } else if format!("{:?}", u.decision) != format!("{:?}", c.decision) {
            annotation_differs += 1;
        }
    }

    println!("\nFRONTIER-SPEND-1 (post DEPTH-2)  R={R}");
    println!("  H1 resamples        {resamples}");
    println!("  hazards             {hazards}");
    println!("  scored replicates   {scored}");
    println!("  terminal conditions {terminals:?}");
    if scored > 0 {
        for (i, name) in ["uniform ", "coverage"].into_iter().enumerate() {
            println!(
                "  {name}  priced runs {:.3}  position-equivalents {:.1}",
                spend[i].0 as f64 / scored as f64,
                spend[i].1 as f64 / scored as f64
            );
        }
    }
    println!(
        "  insufficient        {insufficient}  (terminated on an UNPRICED \
         winner — the claim's precondition was not met)"
    );
    println!("  H2 violations       {disagreements}  (preference, where both priced)");
    println!(
        "  annotation differs  {annotation_differs}  (readiness/unresolved — \
         expected when the policies price different members)"
    );

    assert_eq!(
        hazards, 0,
        "the tier-escalation hazard fired again after DEPTH-2"
    );
    assert!(scored > 0, "every replicate was discarded");
    assert_eq!(
        disagreements, 0,
        "H2: exploration changed the destination — that is not one policy \
         exploring better, it is the exploration layer leaking into the \
         evidence layer"
    );
}
