//! **The generic core: K cheapest distinct leaves by branch-and-bound.**
//!
//! Knows nothing about tensors. A problem is a list of variables, each
//! with a few options carrying an integer cost and a tie-break score; a
//! leaf is one option per variable. The caller's `accept` judges a
//! complete leaf and names the state it resolves to, so exclusions keyed
//! by state (`ExactNoGood`) and "distinct states only" live with the
//! caller while the arithmetic of bounds lives here.
//!
//! Kept free of REPRESENT types so it can be checked against exhaustive
//! enumeration on problems small enough to enumerate (AUTO-REP-1a gate 1).

use std::cmp::Ordering;

/// Relative tolerance below which two score sums are not trusted to
/// differ: far above f64 summation error for any realistic group count.
const SCORE_SLACK: f64 = 1e-9;

/// One option of one variable.
#[derive(Debug, Clone, PartialEq)]
pub(crate) struct Opt {
    /// Index into the caller's choice vocabulary; the leaf reports it.
    pub choice: usize,
    pub cost: u64,
    /// Tie-break only; lower is preferred. Never compared across costs.
    pub score: f64,
}

/// A variable's surviving options, after unary filtering.
#[derive(Debug, Clone, PartialEq)]
pub(crate) struct Var {
    pub options: Vec<Opt>,
}

/// The whole core problem.
#[derive(Debug, Clone)]
pub(crate) struct Core {
    /// Cost every leaf pays regardless of its assignment.
    pub base: u64,
    pub vars: Vec<Var>,
    /// Leaves above this total are infeasible.
    pub ceiling: Option<u64>,
}

/// One accepted leaf.
#[derive(Debug, Clone, PartialEq)]
pub(crate) struct Leaf {
    /// Chosen `Opt::choice` per variable, in the caller's variable order.
    pub assignment: Vec<usize>,
    pub cost: u64,
    pub score: f64,
    /// The state the caller resolved this leaf to.
    pub key: String,
}

/// Whether the search ran to completion.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum Outcome {
    Complete,
    NodeLimit { explored: u64 },
}

/// The K best leaves, and how far the search got.
#[derive(Debug, Clone, PartialEq)]
pub(crate) struct Solved {
    /// Best first, distinct keys.
    pub leaves: Vec<Leaf>,
    pub outcome: Outcome,
    /// No accepted leaf costs less. Equal to the rank-1 cost when the
    /// search completed; `None` when nothing is feasible.
    pub lower_bound: Option<u64>,
}

/// The total order leaves are ranked by: cost, then score, then the
/// assignment itself, so ties never depend on search order.
pub(crate) fn rank(a: &Leaf, b: &Leaf) -> Ordering {
    a.cost
        .cmp(&b.cost)
        .then(a.score.total_cmp(&b.score))
        .then(a.assignment.cmp(&b.assignment))
}

struct Search<'a> {
    core: &'a Core,
    /// Variable indices in branching order.
    order: Vec<usize>,
    /// `suffix_min[d]`: cheapest completion of variables `order[d..]`.
    suffix_min: Vec<u64>,
    /// `suffix_score[d]`: lowest score variables `order[d..]` can add.
    suffix_score: Vec<f64>,
    /// Lowest choice each variable offers, in the caller's order.
    min_choice: Vec<usize>,
    k: usize,
    node_limit: u64,
    nodes: u64,
    best: Vec<Leaf>,
    assignment: Vec<usize>,
    accept: &'a mut dyn FnMut(&[usize]) -> Option<String>,
}

/// The search stopped early; `pending` bounds every unexplored leaf.
struct Aborted {
    pending: Option<u64>,
}

fn min_opt(a: Option<u64>, b: Option<u64>) -> Option<u64> {
    match (a, b) {
        (Some(a), Some(b)) => Some(a.min(b)),
        (a, None) => a,
        (None, b) => b,
    }
}

impl Search<'_> {
    fn infeasible(&self, lb: u64) -> bool {
        self.core.ceiling.is_some_and(|c| lb > c)
    }

    /// Whether no leaf below this node can enter the K best.
    ///
    /// Compares the node's lower bound on the whole ranking tuple with
    /// the K-th leaf. Every leaf below costs at least `lb`; one of equal
    /// cost scores at least `score_lb`; one of equal cost and score has
    /// an assignment no smaller than the node's own choices with every
    /// unassigned variable at its lowest choice. Bounding cost alone
    /// would explore every equal-cost subtree to settle tie-breaks, which
    /// is exponential when many groups cost the same.
    fn dominated(&self, depth: usize, lb: u64, score_lb: f64) -> bool {
        let Some(worst) = self.best.last().filter(|_| self.best.len() == self.k) else {
            return false;
        };
        match lb.cmp(&worst.cost) {
            Ordering::Greater => return true,
            Ordering::Less => return false,
            Ordering::Equal => {}
        }
        // `score_lb` sums in another order than a leaf's score does, so
        // only a clear difference prunes; an inexact near-tie explores.
        let slack = SCORE_SLACK * (1.0 + worst.score.abs());
        if score_lb > worst.score + slack {
            return true;
        }
        if score_lb != worst.score {
            return false;
        }
        let assigned = &self.order[..depth];
        let lex_lb: Vec<usize> = (0..self.assignment.len())
            .map(|v| {
                if assigned.contains(&v) {
                    self.assignment[v]
                } else {
                    self.min_choice[v]
                }
            })
            .collect();
        lex_lb >= worst.assignment
    }

    fn offer(&mut self, leaf: Leaf) {
        if let Some(i) = self.best.iter().position(|l| l.key == leaf.key) {
            if rank(&leaf, &self.best[i]) != Ordering::Less {
                return;
            }
            self.best.remove(i);
        }
        let at = self
            .best
            .iter()
            .position(|l| rank(&leaf, l) == Ordering::Less)
            .unwrap_or(self.best.len());
        self.best.insert(at, leaf);
        self.best.truncate(self.k);
    }

    fn visit(&mut self, depth: usize, partial: u64, score: f64) -> Result<(), Aborted> {
        self.nodes += 1;
        if self.nodes > self.node_limit {
            return Err(Aborted {
                pending: Some(partial + self.suffix_min[depth]),
            });
        }
        if depth == self.order.len() {
            if let Some(key) = (self.accept)(&self.assignment) {
                self.offer(Leaf {
                    assignment: self.assignment.clone(),
                    // `suffix_min[len]` is the base every leaf pays.
                    cost: partial + self.suffix_min[depth],
                    score,
                    key,
                });
            }
            return Ok(());
        }
        let var = self.order[depth];
        let options = &self.core.vars[var].options;
        for (i, opt) in options.iter().enumerate() {
            let lb = partial + opt.cost + self.suffix_min[depth + 1];
            if self.infeasible(lb) {
                continue;
            }
            self.assignment[var] = opt.choice;
            let score_lb = score + opt.score + self.suffix_score[depth + 1];
            if self.dominated(depth + 1, lb, score_lb) {
                continue;
            }
            if let Err(aborted) = self.visit(depth + 1, partial + opt.cost, score + opt.score) {
                // Every later sibling is unexplored too.
                let rest = options[i + 1..]
                    .iter()
                    .map(|o| partial + o.cost + self.suffix_min[depth + 1])
                    .filter(|&lb| !self.infeasible(lb))
                    .min();
                return Err(Aborted {
                    pending: min_opt(aborted.pending, rest),
                });
            }
        }
        Ok(())
    }
}

/// Solve `core` for its `k` best distinct accepted leaves, exploring at
/// most `node_limit` nodes.
pub(crate) fn solve(
    core: &Core,
    k: usize,
    node_limit: u64,
    accept: &mut dyn FnMut(&[usize]) -> Option<String>,
) -> Solved {
    if k == 0 || core.vars.iter().any(|v| v.options.is_empty()) {
        return Solved {
            leaves: Vec::new(),
            outcome: Outcome::Complete,
            lower_bound: None,
        };
    }
    let spread = |v: &Var| {
        let costs = v.options.iter().map(|o| o.cost);
        costs.clone().max().unwrap_or(0) - costs.min().unwrap_or(0)
    };
    // Widest cost spread first: the decisions that move the bound most
    // are taken where they prune the most.
    let mut order: Vec<usize> = (0..core.vars.len()).collect();
    order.sort_by(|&a, &b| {
        spread(&core.vars[b])
            .cmp(&spread(&core.vars[a]))
            .then(a.cmp(&b))
    });
    let mut suffix_min = vec![core.base; order.len() + 1];
    let mut suffix_score = vec![0.0; order.len() + 1];
    for d in (0..order.len()).rev() {
        suffix_score[d] = suffix_score[d + 1]
            + core.vars[order[d]]
                .options
                .iter()
                .map(|o| o.score)
                .min_by(f64::total_cmp)
                .expect("non-empty");
        let cheapest = core.vars[order[d]]
            .options
            .iter()
            .map(|o| o.cost)
            .min()
            .expect("non-empty");
        suffix_min[d] = suffix_min[d + 1] + cheapest;
    }
    // `suffix_min` carries `base` at every depth, so the partial sum
    // starts at zero.
    let mut search = Search {
        core,
        min_choice: core
            .vars
            .iter()
            .map(|v| v.options.iter().map(|o| o.choice).min().expect("non-empty"))
            .collect(),
        order,
        suffix_min,
        suffix_score,
        k,
        node_limit,
        nodes: 0,
        best: Vec::new(),
        assignment: vec![usize::MAX; core.vars.len()],
        accept,
    };
    let result = search.visit(0, 0, 0.0);
    let found = search.best.first().map(|l| l.cost);
    match result {
        Ok(()) => Solved {
            leaves: search.best,
            outcome: Outcome::Complete,
            lower_bound: found,
        },
        Err(aborted) => Solved {
            outcome: Outcome::NodeLimit {
                explored: search.nodes - 1,
            },
            lower_bound: min_opt(found, aborted.pending),
            leaves: search.best,
        },
    }
}
