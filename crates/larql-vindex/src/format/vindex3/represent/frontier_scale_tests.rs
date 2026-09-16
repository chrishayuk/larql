//! **REPRESENT-FRONTIER-SCALE-1.** Where evidence-only selection stops
//! being able to act.
//!
//! Frozen at
//! `docs/represent/forecasts/represent-frontier-scale-1.json` before
//! this file existed. The independent arm is ANALYTIC — expected
//! frontier size `H_n`, selection probability `1/n` — so it is
//! confirmed, not discovered, and a miss indicts the harness first.
//!
//! **F1 is why one assessment is cloned into every candidate.** The
//! gate requires identical `MoveClass` and tier across the set, so
//! candidates may differ ONLY in their diagnostic vector. Sharing the
//! assessment makes that structural rather than asserted-and-hoped, and
//! it is what keeps 2000 replicates affordable.

use std::collections::BTreeSet;

use super::assessment::MoveClass;
use super::decision::{decide_promotion, AmbiguityReason, PromotionDecision, SearchCandidate};
use super::diagnostic::{DiagnosticPolicy, DiagnosticVector};
use super::measurement::{EvidenceScale, TailSupportPolicy};
use super::participation::ParticipationDeclaration;
use super::promotion::PromotionCandidate;
use super::search_evidence::SearchCalibrationRegistry;
use super::state::fixtures;

/// Replicates per (n, geometry). Frozen with the tolerances.
const R: usize = 2000;
const NS: [usize; 7] = [2, 4, 8, 16, 32, 64, 128];

/// Deterministic xorshift64*. A Monte Carlo gate that flakes is not a
/// gate; the seed is fixed so a pass or fail is reproducible.
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
    fn below(&mut self, n: usize) -> usize {
        (self.next_u64() % n as u64) as usize
    }
    fn shuffle<T>(&mut self, v: &mut [T]) {
        for i in (1..v.len()).rev() {
            v.swap(i, self.below(i + 1));
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Geometry {
    Independent,
    Correlated,
    Antagonistic,
    /// Half the set carries the correlated order, half is reshuffled.
    /// Ranks blended with noise: `key = ALPHA*rank + (1-ALPHA)*u`.
    ///
    /// The first construction shuffled only the TAIL of the flip order,
    /// which left candidate 0 best on both coordinates in every draw.
    /// A global dominator always existed, the selection rate read
    /// 1.000000 at every n, and the arm was indistinguishable from the
    /// perfectly correlated one. It measured nothing. Recorded because
    /// a degenerate generator is invisible in a green test.
    Moderate,
}

/// Blend weight for [`Geometry::Moderate`]. 0 is independent, 1 is
/// perfectly correlated.
const ALPHA: f64 = 0.5;

/// Rank pairs `(kl_rank, flip_rank)`. Only ORDER reaches the
/// comparator, so the geometry is defined on ranks and the values are
/// a monotone embedding of them.
fn rank_pairs(g: Geometry, n: usize, rng: &mut Rng) -> Vec<(usize, usize)> {
    let mut flips: Vec<usize> = (0..n).collect();
    match g {
        Geometry::Correlated => {}
        Geometry::Antagonistic => flips.reverse(),
        Geometry::Independent => rng.shuffle(&mut flips),
        Geometry::Moderate => {
            let mut keyed: Vec<(f64, usize)> = (0..n)
                .map(|i| {
                    let u = (rng.next_u64() % 1_000_000) as f64 / 1_000_000.0 * n as f64;
                    (ALPHA * i as f64 + (1.0 - ALPHA) * u, i)
                })
                .collect();
            keyed.sort_by(|a, b| a.0.total_cmp(&b.0));
            for (rank, (_, original)) in keyed.into_iter().enumerate() {
                flips[original] = rank;
            }
        }
    }
    (0..n).zip(flips).collect()
}

/// The shared assessment, and the policies the comparator reads.
fn template() -> (
    PromotionCandidate,
    SearchCalibrationRegistry,
    TailSupportPolicy,
    DiagnosticPolicy,
) {
    let snap = fixtures::pareto_p1_snapshot();
    let candidates = snap
        .promotion_candidates(EvidenceScale::Authority)
        .expect("the cost model covers this model");
    let config = snap.config();
    (
        candidates[0].promotion.clone(),
        config.calibrations.clone(),
        config.tail_support.clone(),
        config.diagnostic_policy.clone(),
    )
}

/// Build one candidate set. `grid` coarsens the ranks so ties become
/// reachable; `None` is the continuous case, where ties do not occur.
fn build(
    pairs: &[(usize, usize)],
    promotion: &PromotionCandidate,
    policy: &DiagnosticPolicy,
    grid: Option<usize>,
) -> Vec<SearchCandidate> {
    pairs
        .iter()
        .enumerate()
        .map(|(i, &(kl_rank, flip_rank))| {
            let (kl_rank, flip_rank) = match grid {
                Some(g) => (kl_rank % g, flip_rank % g),
                None => (kl_rank, flip_rank),
            };
            let bank = fixtures::authority_reading(
                1.0e-3 + kl_rank as f64 * 1.0e-6,
                1_000 + flip_rank as u64,
            );
            SearchCandidate {
                id: format!("c{i:04}"),
                promotion: promotion.clone(),
                diagnostic: DiagnosticVector::of(policy, &bank),
                participation: ParticipationDeclaration::all_affected(),
            }
        })
        .collect()
}

#[derive(Default, Debug)]
struct Tally {
    rounds: usize,
    selections: usize,
    frontier_total: usize,
    conflicting: usize,
    no_ordering: usize,
    indistinguishable: usize,
    by_physical_gain: usize,
    permutation_violations: usize,
}

impl Tally {
    fn selection_rate(&self) -> f64 {
        self.selections as f64 / self.rounds as f64
    }
    fn frontier_mean(&self) -> f64 {
        self.frontier_total as f64 / self.rounds as f64
    }
}

/// Run one arm, applying every harness falsifier.
fn run(g: Geometry, n: usize, grid: Option<usize>, seed: u64) -> Tally {
    let (promotion, registry, tail, policy) = template();
    let mut rng = Rng::new(seed);
    let mut t = Tally::default();

    for _ in 0..R {
        let pairs = rank_pairs(g, n, &mut rng);
        let candidates = build(&pairs, &promotion, &policy, grid);

        // F1: identical class and tier across the set, asserted BEFORE
        // deciding. Mixed classes would make stage 1 the thing being
        // measured.
        let tiers: BTreeSet<u8> = candidates
            .iter()
            .map(|c| c.promotion.assessment.ranking_score.class.tier())
            .collect();
        assert_eq!(tiers.len(), 1, "F1: stage 1 would decide, not the frontier");
        assert_eq!(
            candidates[0].promotion.assessment.ranking_score.class,
            MoveClass::Priced
        );

        let decision = decide_promotion(&candidates, &registry, &tail);

        // F4: the full decision must survive a permutation.
        let mut shuffled = candidates.clone();
        rng.shuffle(&mut shuffled);
        if format!("{decision:?}") != format!("{:?}", decide_promotion(&shuffled, &registry, &tail))
        {
            t.permutation_violations += 1;
        }

        t.rounds += 1;
        match &decision {
            PromotionDecision::SelectForAuthority { evidence, .. } => {
                t.selections += 1;
                t.frontier_total += 1;
                if evidence.decided_by_physical_gain {
                    t.by_physical_gain += 1;
                }
            }
            PromotionDecision::Ambiguous { candidates, reason } => {
                t.frontier_total += candidates.len();
                match reason {
                    AmbiguityReason::ConflictingOrderingProxies => t.conflicting += 1,
                    AmbiguityReason::NoOrderingEvidence => t.no_ordering += 1,
                    AmbiguityReason::IndistinguishableOnEveryProxy => t.indistinguishable += 1,
                }
            }
            other => panic!("unexpected decision: {other:?}"),
        }
    }

    // F2: NoOrderingEvidence cannot fire with two registered, observed
    // proxies. If it did, the registry is half-configured — the P2
    // defect — and the arm says nothing about the search.
    assert_eq!(
        t.no_ordering, 0,
        "F2: the harness lost its calibration authority"
    );
    // F4, scored.
    assert_eq!(
        t.permutation_violations, 0,
        "F4: input order reached a decision"
    );
    t
}

fn harmonic(n: usize) -> f64 {
    (1..=n).map(|k| 1.0 / k as f64).sum()
}

fn harmonic2(n: usize) -> f64 {
    (1..=n).map(|k| 1.0 / (k * k) as f64).sum()
}

/// **The reference arm.** Confirms the comparator behaves exactly as its
/// definition implies, against tolerances frozen from R.
#[test]
fn independent_proxies_match_the_analytic_forecast() {
    println!(
        "{:>5} {:>9} {:>9} {:>9} {:>9} {:>9}",
        "n", "H_n", "obs_mean", "1/n", "obs_rate", "refusal"
    );
    for (i, n) in NS.into_iter().enumerate() {
        let t = run(Geometry::Independent, n, None, 0x51ED_0000 + i as u64);
        let h = harmonic(n);
        let mean_tol = 1.96 * ((h - harmonic2(n)) / R as f64).sqrt();
        let p = 1.0 / n as f64;
        let rate_tol = 1.96 * (p * (1.0 - p) / R as f64).sqrt();
        println!(
            "{:>5} {:>9.4} {:>9.4} {:>9.6} {:>9.6} {:>8.1}%",
            n,
            h,
            t.frontier_mean(),
            p,
            t.selection_rate(),
            100.0 * (1.0 - t.selection_rate())
        );
        assert!(
            (t.frontier_mean() - h).abs() <= mean_tol,
            "n={n}: frontier mean {} outside H_n {h} +/- {mean_tol}",
            t.frontier_mean()
        );
        assert!(
            (t.selection_rate() - p).abs() <= rate_tol,
            "n={n}: selection rate {} outside 1/n {p} +/- {rate_tol}",
            t.selection_rate()
        );
    }
}

/// **The two deterministic endpoints.** No tolerance: these are exact.
#[test]
fn correlated_and_antagonistic_arms_hit_their_predicted_endpoints() {
    for (i, n) in NS.into_iter().enumerate() {
        let c = run(Geometry::Correlated, n, None, 0x0C00_0000 + i as u64);
        assert_eq!(
            c.selection_rate(),
            1.0,
            "n={n}: perfectly correlated proxies must always select"
        );
        assert_eq!(c.frontier_mean(), 1.0);

        let a = run(Geometry::Antagonistic, n, None, 0x0A17_0000 + i as u64);
        assert_eq!(
            a.selection_rate(),
            0.0,
            "n={n}: rank-reversed proxies leave every candidate non-dominated"
        );
        assert_eq!(a.frontier_mean(), n as f64);
        assert_eq!(a.conflicting, R, "and every round refuses for conflict");
    }
}

/// **The only arm being measured rather than confirmed.** Reported, and
/// asserted only to lie between the endpoints.
#[test]
fn moderate_correlation_is_characterised() {
    println!("{:>5} {:>11} {:>11}", "n", "frontier", "selection");
    for (i, n) in NS.into_iter().enumerate() {
        let t = run(Geometry::Moderate, n, None, 0x000D_0000 + i as u64);
        println!(
            "{:>5} {:>11.4} {:>11.6}",
            n,
            t.frontier_mean(),
            t.selection_rate()
        );
        assert!(t.frontier_mean() >= 1.0 && t.frontier_mean() <= n as f64);
        assert!(t.selection_rate() >= 0.0 && t.selection_rate() <= 1.0);
    }
}

/// A second assessment at the SAME class and tier but a different
/// physical gain, so stage 5 has something to separate.
fn leaner_promotion() -> PromotionCandidate {
    let snap = fixtures::ParetoWorld::physically_separated(
        fixtures::PARETO_MIDDLE,
        fixtures::PARETO_MIDDLE,
    )
    .snapshot();
    let candidates = snap
        .promotion_candidates(EvidenceScale::Authority)
        .expect("cost");
    // The child that removes half as many bytes: the smaller gain.
    candidates
        .iter()
        .min_by(|a, b| {
            a.promotion
                .assessment
                .ranking_score
                .gpu_ms_saved
                .total_cmp(&b.promotion.assessment.ranking_score.gpu_ms_saved)
        })
        .expect("two candidates")
        .promotion
        .clone()
}

/// **The discrete-grid arm.** Under continuous draws the physical
/// tie-break is unreachable by construction, so it is measured only
/// where ties can occur. The grid is declared: ranks are folded modulo
/// 2, so candidates share both coordinates exactly.
///
/// **Reaching stage 5 is not the same as stage 5 deciding.** Every
/// candidate in the arms above shares one cloned assessment, so their
/// `gpu_ms_saved` is identical and `by_gain` never narrows to one —
/// ties reach the physical stage and leave it as
/// `IndistinguishableOnEveryProxy`. To measure the tie-break SELECTING,
/// exactly one candidate must carry a different physical gain at the
/// same class and tier, which is what `leaner_promotion` supplies.
#[test]
fn the_physical_tie_break_is_reachable_only_on_a_discrete_grid() {
    let (promotion, registry, tail, policy) = template();
    let leaner = leaner_promotion();
    assert_eq!(
        promotion.assessment.ranking_score.class, leaner.assessment.ranking_score.class,
        "F1: the two assessments must share a class"
    );
    assert!(
        promotion.assessment.ranking_score.gpu_ms_saved
            > leaner.assessment.ranking_score.gpu_ms_saved,
        "and differ only in physical gain"
    );

    // Continuous: the tie-break is unreachable, and a nonzero rate here
    // would be a sampler artifact rather than a behavioural result.
    let continuous = run(Geometry::Independent, 16, None, 0xC047);
    assert_eq!(continuous.by_physical_gain, 0);

    // Discrete, one assessment: ties are reached and refused.
    let shared = run(Geometry::Independent, 16, Some(2), 0xD15C);
    assert!(shared.indistinguishable > 0, "a coarse grid must tie");
    assert_eq!(
        shared.by_physical_gain, 0,
        "equal gains cannot narrow to one; reaching stage 5 is not deciding"
    );

    // Discrete, one candidate leaner: stage 5 can now select.
    let mut rng = Rng::new(0x5A6E);
    let (mut selected_by_gain, mut rounds) = (0usize, 0usize);
    for _ in 0..R {
        let pairs = rank_pairs(Geometry::Independent, 16, &mut rng);
        let mut candidates = build(&pairs, &promotion, &policy, Some(2));
        for c in candidates.iter_mut().skip(1) {
            c.promotion = leaner.clone();
        }
        let tiers: BTreeSet<u8> = candidates
            .iter()
            .map(|c| c.promotion.assessment.ranking_score.class.tier())
            .collect();
        assert_eq!(tiers.len(), 1, "F1");
        rounds += 1;
        if let PromotionDecision::SelectForAuthority { evidence, .. } =
            decide_promotion(&candidates, &registry, &tail)
        {
            if evidence.decided_by_physical_gain {
                selected_by_gain += 1;
            }
        }
    }
    println!(
        "grid=2 n=16  shared_gain: tied={} decided_by_gain={}  |  one_leaner: decided_by_gain={}/{}",
        shared.indistinguishable, shared.by_physical_gain, selected_by_gain, rounds
    );
    assert!(
        selected_by_gain > 0,
        "with one candidate carrying a distinct gain, stage 5 must sometimes decide"
    );
}
