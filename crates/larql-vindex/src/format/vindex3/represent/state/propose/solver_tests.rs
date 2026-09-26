//! Gate 1: the branch-and-bound core against exhaustive enumeration.
//!
//! The reference below shares nothing with the solver but the ranking
//! rule it is specified by: it enumerates every assignment, filters,
//! sorts and deduplicates. Costs and scores are small integers so ties
//! are common (the tie-break is where a bound written `>=` instead of
//! `>` goes wrong) and score sums are exact in f64.

use super::solver::{rank, solve, Core, Leaf, Opt, Outcome, Var};

/// A deterministic generator, so a failure names a reproducible seed.
struct Lcg(u64);

impl Lcg {
    fn next(&mut self) -> u64 {
        self.0 = self
            .0
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        self.0 >> 33
    }

    fn below(&mut self, n: u64) -> u64 {
        self.next() % n
    }
}

fn random_core(rng: &mut Lcg) -> Core {
    let vars = (0..1 + rng.below(7))
        .map(|_| Var {
            options: (0..1 + rng.below(3) as usize)
                .map(|choice| Opt {
                    choice,
                    cost: rng.below(20),
                    score: rng.below(3) as f64,
                })
                .collect(),
        })
        .collect::<Vec<_>>();
    let base = rng.below(10);
    let max: u64 = base
        + vars
            .iter()
            .map(|v: &Var| v.options.iter().map(|o| o.cost).max().unwrap())
            .sum::<u64>();
    let ceiling = (rng.below(3) == 0).then(|| rng.below(max + 1));
    Core {
        base,
        vars,
        ceiling,
    }
}

/// The caller's judgement, as the proposer makes it: some assignments
/// are excluded outright (no-goods), and several assignments resolve to
/// one state (the key), so only the best of them may be proposed.
fn judge(seed: u64) -> impl FnMut(&[usize]) -> Option<String> {
    move |a: &[usize]| {
        let h = a
            .iter()
            .fold(seed, |h, &c| h.wrapping_mul(31).wrapping_add(c as u64 + 1));
        (h % 7 != 0).then(|| format!("state-{}", h % 11))
    }
}

fn exhaustive(
    core: &Core,
    k: usize,
    accept: &mut dyn FnMut(&[usize]) -> Option<String>,
) -> Vec<Leaf> {
    let mut all = Vec::new();
    let mut idx = vec![0usize; core.vars.len()];
    if core.vars.iter().all(|v| !v.options.is_empty()) {
        loop {
            let opts: Vec<&Opt> = idx
                .iter()
                .zip(&core.vars)
                .map(|(&i, v)| &v.options[i])
                .collect();
            let cost = core.base + opts.iter().map(|o| o.cost).sum::<u64>();
            if core.ceiling.is_none_or(|c| cost <= c) {
                let assignment: Vec<usize> = opts.iter().map(|o| o.choice).collect();
                if let Some(key) = accept(&assignment) {
                    all.push(Leaf {
                        assignment,
                        cost,
                        score: opts.iter().map(|o| o.score).sum(),
                        key,
                    });
                }
            }
            // Odometer over the option indices.
            let mut d = 0;
            loop {
                if d == idx.len() {
                    all.sort_by(rank);
                    let mut seen = std::collections::BTreeSet::new();
                    all.retain(|l| seen.insert(l.key.clone()));
                    all.truncate(k);
                    return all;
                }
                idx[d] += 1;
                if idx[d] < core.vars[d].options.len() {
                    break;
                }
                idx[d] = 0;
                d += 1;
            }
        }
    }
    all
}

#[test]
fn the_k_best_distinct_leaves_equal_exhaustive_enumeration() {
    let mut rng = Lcg(0x0A07_05E9);
    for case in 0..3000 {
        let core = random_core(&mut rng);
        let k = 1 + rng.below(6) as usize;
        let seed = rng.next();
        let expected = exhaustive(&core, k, &mut judge(seed));
        let solved = solve(&core, k, u64::MAX, &mut judge(seed));
        assert_eq!(solved.outcome, Outcome::Complete, "case {case}");
        assert_eq!(solved.leaves, expected, "case {case}: {core:?} k={k}");
        assert_eq!(
            solved.lower_bound,
            expected.first().map(|l| l.cost),
            "case {case}: a complete search's bound is its optimum"
        );
    }
}

#[test]
fn a_truncated_search_bounds_the_optimum_from_below() {
    let mut rng = Lcg(0x07E5_7B0D);
    let mut truncated = 0;
    for case in 0..3000 {
        let core = random_core(&mut rng);
        let seed = rng.next();
        let optimum = exhaustive(&core, 1, &mut judge(seed))
            .first()
            .map(|l| l.cost);
        let limit = 1 + rng.below(12);
        let solved = solve(&core, 3, limit, &mut judge(seed));
        if let Outcome::NodeLimit { explored } = solved.outcome {
            truncated += 1;
            assert!(explored <= limit, "case {case}");
            if let Some(optimum) = optimum {
                let bound = solved.lower_bound.expect("a feasible problem has a bound");
                assert!(
                    bound <= optimum,
                    "case {case}: bound {bound} > optimum {optimum}"
                );
            }
        }
        // Whatever was found is real: accepted, within the ceiling, and
        // costed exactly.
        for leaf in &solved.leaves {
            let cost = core.base
                + leaf
                    .assignment
                    .iter()
                    .zip(&core.vars)
                    .map(|(&c, v)| v.options.iter().find(|o| o.choice == c).unwrap().cost)
                    .sum::<u64>();
            assert_eq!(leaf.cost, cost, "case {case}");
            assert!(core.ceiling.is_none_or(|c| cost <= c), "case {case}");
            assert_eq!(judge(seed)(&leaf.assignment), Some(leaf.key.clone()));
        }
    }
    assert!(
        truncated > 500,
        "the budget must actually bite: {truncated}"
    );
}

#[test]
fn an_empty_domain_or_k_zero_proposes_nothing() {
    let empty = Core {
        base: 0,
        vars: vec![Var { options: vec![] }],
        ceiling: None,
    };
    let solved = solve(&empty, 3, u64::MAX, &mut |_| Some("s".into()));
    assert!(solved.leaves.is_empty());
    assert_eq!(solved.outcome, Outcome::Complete);
    assert_eq!(solved.lower_bound, None);

    let one = Core {
        base: 1,
        vars: vec![Var {
            options: vec![Opt {
                choice: 0,
                cost: 2,
                score: 0.0,
            }],
        }],
        ceiling: None,
    };
    assert!(solve(&one, 0, u64::MAX, &mut |_| Some("s".into()))
        .leaves
        .is_empty());
    let solved = solve(&one, 1, u64::MAX, &mut |_| Some("s".into()));
    assert_eq!(solved.leaves[0].cost, 3, "the base is paid by every leaf");
}

/// Many variables whose options cost the same: every leaf ties on cost,
/// so only the score and assignment tie-breaks order them. Bounding cost
/// alone would explore all 2^60 leaves; the full-tuple bound must settle
/// the K best in a number of nodes linear in the variables.
#[test]
fn equal_cost_variables_settle_without_exploring_every_tie() {
    let vars = (0..60)
        .map(|v| Var {
            options: (0..2)
                .map(|choice| Opt {
                    choice,
                    cost: 5,
                    score: if v % 3 == 0 && choice == 1 { -1.0 } else { 0.0 },
                })
                .collect(),
        })
        .collect();
    let core = Core {
        base: 0,
        vars,
        ceiling: None,
    };
    let solved = solve(&core, 4, 100_000, &mut |a: &[usize]| {
        Some(a.iter().map(|c| c.to_string()).collect())
    });
    assert_eq!(solved.outcome, Outcome::Complete);
    assert_eq!(solved.leaves.len(), 4);
    // Rank 1 takes every score-lowering choice and nothing else.
    let best: Vec<usize> = (0..60).map(|v| usize::from(v % 3 == 0)).collect();
    assert_eq!(solved.leaves[0].assignment, best);
    assert!(solved.leaves.windows(2).all(|w| rank(&w[0], &w[1]).is_lt()));
}
