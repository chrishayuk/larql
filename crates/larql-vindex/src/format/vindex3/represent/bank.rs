//! **Building a [`QualityBank`] from per-position observations.**
//!
//! The bank is TEACHER-FORCED: both arms see the same token prefix at
//! every position, and each short sequence starts from clean recurrent
//! state. That is not a detail — it is what makes the measurement
//! attributable.
//!
//! Let the candidate consume its own generated history instead and a
//! route flip at position 4 changes everything after it, so by position
//! 15 the number being measured is accumulated autoregressive
//! divergence, not the representation's direct effect. The question
//! REPRESENT needs answered is "did quantising this region perturb
//! layer 18 enough to cross a routing boundary", not "twenty tokens
//! later, is the candidate somewhere else entirely". A free-running
//! bank answers the second and is the right instrument for behavioural
//! validation AFTER a representation is selected — a different bank,
//! deliberately not this one.
//!
//! ## Why the baseline is stored compressed
//!
//! The baseline is frozen once and reused against every candidate scope
//! (`ExpertWeight/all`, `ExpertWeight/down`, `.../layers 20..26`, ...),
//! so it has to be cheap to keep: a full logit vector is 640 KiB a
//! position, 5 GB for eight thousand of them. Storing the baseline's
//! top-N ids with their logits, plus its `logsumexp` over the FULL
//! vocabulary, makes KL exact on the mass those N carry — the candidate
//! arm is live and can evaluate its own logits at exactly those ids.
//!
//! The uncovered tail is not hidden: [`PositionObservation::covered_mass`]
//! is recorded per position and the minimum is carried into the bank, so
//! a distribution flat enough for truncation to matter announces itself
//! instead of quietly shrinking the KL.

use serde::{Deserialize, Serialize};

use super::quality::{LogitEvidence, QualityBank, RoutingEvidence};

/// One position, both arms, teacher-forced on the same prefix.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct PositionObservation {
    pub sequence: u32,
    pub position: u32,
    /// The baseline's top-N vocabulary ids, most probable first.
    pub top_ids: Vec<u32>,
    /// Baseline logits at `top_ids`, and its logsumexp over the WHOLE
    /// vocabulary — together these give exact probabilities.
    pub baseline_logits: Vec<f32>,
    pub baseline_logsumexp: f32,
    /// Candidate logits at the SAME ids, and its own full logsumexp.
    pub candidate_logits: Vec<f32>,
    pub candidate_logsumexp: f32,
    /// Each arm's argmax over the full vocabulary — not derived from the
    /// truncation, because the candidate's argmax may lie outside the
    /// baseline's top-N precisely when it matters.
    pub baseline_argmax: u32,
    pub candidate_argmax: u32,
    /// Each arm's top-10 ids in order.
    pub baseline_top10: Vec<u32>,
    pub candidate_top10: Vec<u32>,
    /// Selected expert ids per routed layer, in the router's own order.
    pub baseline_routes: Vec<Vec<u32>>,
    pub candidate_routes: Vec<Vec<u32>>,
}

impl PositionObservation {
    /// Baseline probability mass carried by `top_ids`.
    ///
    /// The KL below is exact on this mass and blind beyond it, so a
    /// caller must be able to see it.
    pub fn covered_mass(&self) -> f64 {
        self.baseline_logits
            .iter()
            .map(|l| ((*l - self.baseline_logsumexp) as f64).exp())
            .sum()
    }

    /// `KL(baseline || candidate)` over the covered ids.
    ///
    /// Baseline-first, deliberately: it weights by what the reference
    /// model actually believes, so mass the candidate invents in the
    /// tail is not what dominates. Both arms are normalised by their own
    /// full-vocabulary `logsumexp`, so this is a divergence between two
    /// genuine distributions rather than between two truncations.
    pub fn kl(&self) -> f64 {
        self.baseline_logits
            .iter()
            .zip(&self.candidate_logits)
            .map(|(b, c)| {
                let log_p = (*b - self.baseline_logsumexp) as f64;
                let log_q = (*c - self.candidate_logsumexp) as f64;
                log_p.exp() * (log_p - log_q)
            })
            .sum()
    }

    pub fn top1_flipped(&self) -> bool {
        self.baseline_argmax != self.candidate_argmax
    }

    /// The top-10 changed if its SET or its order changed — order
    /// matters because a reordering is a real change in what the model
    /// prefers, even when the membership is identical.
    pub fn top10_changed(&self) -> bool {
        self.baseline_top10 != self.candidate_top10
    }

    /// Layers whose selected-expert SET changed, and the total number of
    /// id differences.
    ///
    /// Set-wise, not positionally: the router emits ties in a defined
    /// order, and two arms agreeing on which experts run while ordering
    /// them differently is not a routing change.
    pub fn route_changes(&self) -> (usize, u64) {
        let mut layers = 0usize;
        let mut flips = 0u64;
        for (b, c) in self.baseline_routes.iter().zip(&self.candidate_routes) {
            let (mut bs, mut cs) = (b.clone(), c.clone());
            bs.sort_unstable();
            cs.sort_unstable();
            if bs != cs {
                layers += 1;
                flips += bs.iter().filter(|id| !cs.contains(id)).count() as u64;
            }
        }
        (layers, flips)
    }
}

/// Accumulates observations into a [`QualityBank`].
#[derive(Debug, Default)]
pub struct BankBuilder {
    kls: Vec<f64>,
    max_logit_delta: f64,
    top1_flips: u64,
    top10_changes: u64,
    route_flips: u64,
    positions_with_route_change: u64,
    layers_with_route_change: std::collections::BTreeSet<usize>,
    min_covered_mass: f64,
}

impl BankBuilder {
    pub fn new() -> Self {
        Self {
            min_covered_mass: 1.0,
            ..Self::default()
        }
    }

    pub fn observe(&mut self, o: &PositionObservation) {
        self.kls.push(o.kl());
        let delta = o
            .baseline_logits
            .iter()
            .zip(&o.candidate_logits)
            .map(|(b, c)| (b - c).abs() as f64)
            .fold(0.0f64, f64::max);
        self.max_logit_delta = self.max_logit_delta.max(delta);
        self.top1_flips += u64::from(o.top1_flipped());
        self.top10_changes += u64::from(o.top10_changed());
        let (layers, flips) = o.route_changes();
        self.route_flips += flips;
        if layers > 0 {
            self.positions_with_route_change += 1;
        }
        for (i, (b, c)) in o
            .baseline_routes
            .iter()
            .zip(&o.candidate_routes)
            .enumerate()
        {
            let (mut bs, mut cs) = (b.clone(), c.clone());
            bs.sort_unstable();
            cs.sort_unstable();
            if bs != cs {
                self.layers_with_route_change.insert(i);
            }
        }
        self.min_covered_mass = self.min_covered_mass.min(o.covered_mass());
    }

    /// The smallest baseline mass any position's truncation covered.
    ///
    /// Exposed so a caller can refuse a bank whose KL is blind to too
    /// much of the distribution, rather than discovering it later.
    pub fn min_covered_mass(&self) -> f64 {
        self.min_covered_mass
    }

    /// Nearest-rank percentile: the smallest observed value at or above
    /// which `p` of the sample lies.
    ///
    /// Nearest-rank rather than interpolated because a p99 that reports
    /// a number no position actually produced is a worse instrument for
    /// a tail statistic than one that names a real observation.
    fn percentile(sorted: &[f64], p: f64) -> f64 {
        if sorted.is_empty() {
            return 0.0;
        }
        let rank = (p * sorted.len() as f64).ceil().max(1.0) as usize;
        sorted[rank.min(sorted.len()) - 1]
    }

    pub fn finish(mut self) -> QualityBank {
        self.kls.sort_by(f64::total_cmp);
        QualityBank {
            positions: self.kls.len() as u64,
            logits: LogitEvidence {
                kl_p50: Self::percentile(&self.kls, 0.50),
                kl_p95: Self::percentile(&self.kls, 0.95),
                kl_p99: Self::percentile(&self.kls, 0.99),
                max_logit_delta: self.max_logit_delta,
                top1_flips: self.top1_flips,
                top10_changes: self.top10_changes,
            },
            routing: RoutingEvidence {
                route_flips: self.route_flips,
                positions_with_route_change: self.positions_with_route_change,
                layers_with_route_change: self.layers_with_route_change.len() as u64,
            },
        }
    }
}

#[cfg(test)]
#[path = "bank_tests.rs"]
mod tests;
