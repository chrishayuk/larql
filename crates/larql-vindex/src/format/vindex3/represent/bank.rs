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

use super::quality::{Distribution, LogitEvidence, QualityBank, RoutingEvidence};

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
    /// **How severe each routing change was**, one entry per layer
    /// whose selected SET differed. Empty when nothing changed, and
    /// empty when the runner collected no score evidence.
    ///
    /// Counting changes cannot separate a near-tie swap from an
    /// overturned decision, and those are not the same behavioural
    /// event; this is what tells them apart.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub route_changes: Vec<RouteChange>,
    /// **What the top-10 change actually was**, when this position's
    /// top-10 changed. `None` when it did not.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub top10_change: Option<TopKChange>,
    /// **What the argmax flip actually was**, when the winner changed.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub top1_change: Option<Top1Change>,
}

/// One position's argmax flip, weighed rather than counted.
///
/// The strictest criterion in the contract, and until now the only one
/// with no severity evidence at all — so these two could not be told
/// apart:
///
/// ```text
/// "cat" 0.20001 vs "dog" 0.20000   the winner changes
/// "cat" 0.70    vs "dog" 0.12      the winner changes
/// ```
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct Top1Change {
    /// The BASELINE's logit gap between its own winner and the one the
    /// candidate picked — the margin the baseline held its choice by,
    /// and therefore the margin the perturbation had to cross.
    pub boundary_margin: f32,
    /// The CANDIDATE's gap over that same pair, positive by
    /// construction. Read against `boundary_margin`: two small numbers
    /// mean the pair was indistinguishable to both arms.
    pub candidate_margin_same_ids: f32,
    /// Probability the BASELINE gave up by switching:
    /// `p_base(baseline winner) − p_base(candidate winner)`, over the
    /// full vocabulary. Near zero for a coin-flip between near-equal
    /// candidates; large when a confident choice was overturned.
    pub mass_displaced: f32,
}

/// One position's top-k reordering, weighed rather than counted.
///
/// The count alone cannot say whether `55 top-10 changes` means
/// fifty-five rank-10-vs-11 near-tie swaps or genuine reshuffling, and
/// those are not the same behavioural claim.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct TopKChange {
    /// The BASELINE's rank-k minus rank-(k+1) logit gap: how close the
    /// ordering was to begin with.
    pub boundary_margin: f32,
    /// The CANDIDATE's gap at those SAME two ids — the displacement the
    /// boundary pair actually experienced.
    ///
    /// This is the number the margin alone cannot supply. A worst-case
    /// `max|Δlogit|` somewhere in a 163,840-wide vocabulary says
    /// nothing about what happened to this pair; comparing the two
    /// against each other is what shows whether the perturbation merely
    /// crossed a near-tie.
    pub candidate_margin_same_ids: f32,
    /// Half the L1 between the two arms' top-k probability mass over
    /// the UNION of their ids, each normalised over its own top-k.
    /// 0 = the same mass on the same ids, 1 = disjoint.
    pub mass_displaced: f32,
    /// The furthest any id moved in rank, counting an id that left the
    /// top-k as moving to its rank in the other arm's full ordering.
    pub max_rank_displacement: u32,
}

/// One layer's routing change, weighed rather than counted.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct RouteChange {
    pub layer: u32,
    /// Selection-score gap in the BASELINE arm between the last
    /// selected expert and the best unselected one — the margin the
    /// perturbation had to cross. Near zero means a near-tie.
    ///
    /// From the score the selection ACTUALLY used
    /// (`sigmoid(logit) + correction_bias` on Kimi), never a raw
    /// pre-policy logit: the bias selects and the unbiased score
    /// weighs, so the pre-policy value is the wrong boundary.
    pub boundary_margin: f32,
    /// Fraction of this layer's routed combine mass that moved,
    /// `0.5 * Σ|w_base(e) − w_cand(e)|` over the union of selected
    /// experts with each arm's weights normalised to sum to one. Zero
    /// means the swap was between equally-weighted experts; one means
    /// the mixture was replaced outright.
    pub weight_mass_moved: f32,
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

    /// Weigh every layer whose route changed, from the router's own
    /// selection scores and combine weights.
    ///
    /// `scores[layer][expert]` is the biased selection score; `weights`
    /// is the combine vector the MoE multiplied by, routed entries
    /// first. Returns one entry per CHANGED layer, in depth order.
    #[allow(clippy::too_many_arguments)]
    pub fn weigh_route_changes(
        baseline_routes: &[Vec<u32>],
        candidate_routes: &[Vec<u32>],
        baseline_scores: &[Vec<f32>],
        baseline_weights: &[Vec<f32>],
        candidate_weights: &[Vec<f32>],
    ) -> Vec<RouteChange> {
        let mut out = Vec::new();
        for (layer, (b, c)) in baseline_routes.iter().zip(candidate_routes).enumerate() {
            let (mut bs, mut cs) = (b.clone(), c.clone());
            bs.sort_unstable();
            cs.sort_unstable();
            if bs == cs {
                continue;
            }
            let scores = baseline_scores.get(layer);
            // The margin: the LOWEST selected score against the highest
            // score no selected expert holds. Both come from the
            // baseline, which is the decision the candidate departed
            // from.
            let boundary_margin = scores
                .map(|s| {
                    let lowest_selected = b
                        .iter()
                        .filter_map(|e| s.get(*e as usize))
                        .copied()
                        .fold(f32::INFINITY, f32::min);
                    let best_unselected = s
                        .iter()
                        .enumerate()
                        .filter(|(e, _)| !b.contains(&(*e as u32)))
                        .map(|(_, v)| *v)
                        .fold(f32::NEG_INFINITY, f32::max);
                    (lowest_selected - best_unselected).max(0.0)
                })
                .unwrap_or(f32::NAN);
            out.push(RouteChange {
                layer: layer as u32,
                boundary_margin,
                weight_mass_moved: Self::mass_moved(
                    b,
                    c,
                    baseline_weights.get(layer),
                    candidate_weights.get(layer),
                ),
            });
        }
        out
    }

    /// Half the L1 distance between the two arms' routed mixtures, each
    /// normalised to sum to one — 0 when the same mass sits on
    /// different experts to no effect, 1 when the mixture is replaced.
    fn mass_moved(
        b_ids: &[u32],
        c_ids: &[u32],
        b_w: Option<&Vec<f32>>,
        c_w: Option<&Vec<f32>>,
    ) -> f32 {
        let (Some(b_w), Some(c_w)) = (b_w, c_w) else {
            return f32::NAN;
        };
        // Routed entries only: the shared branch is not routed and its
        // weight is a constant, so including it would dilute every
        // number by the same factor and hide the routed movement.
        let norm = |ids: &[u32], w: &[f32]| -> Vec<(u32, f32)> {
            let take = ids.len().min(w.len());
            let sum: f32 = w[..take].iter().sum();
            if sum == 0.0 {
                return Vec::new();
            }
            ids[..take]
                .iter()
                .zip(&w[..take])
                .map(|(e, v)| (*e, v / sum))
                .collect()
        };
        let (bn, cn) = (norm(b_ids, b_w), norm(c_ids, c_w));
        if bn.is_empty() || cn.is_empty() {
            return f32::NAN;
        }
        let weight_of = |v: &[(u32, f32)], e: u32| {
            v.iter()
                .find(|(id, _)| *id == e)
                .map(|(_, w)| *w)
                .unwrap_or(0.0)
        };
        let mut union: Vec<u32> = bn
            .iter()
            .map(|(e, _)| *e)
            .chain(cn.iter().map(|(e, _)| *e))
            .collect();
        union.sort_unstable();
        union.dedup();
        0.5 * union
            .iter()
            .map(|e| (weight_of(&bn, *e) - weight_of(&cn, *e)).abs())
            .sum::<f32>()
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
    /// Every changed route's severity, kept per event rather than
    /// reduced on arrival — the shape of these is the question.
    route_margins: Vec<f64>,
    route_masses: Vec<f64>,
    top10_margins: Vec<f64>,
    top10_candidate_margins: Vec<f64>,
    top10_masses: Vec<f64>,
    top10_rank_moves: Vec<f64>,
    top1_margins: Vec<f64>,
    top1_candidate_margins: Vec<f64>,
    top1_masses: Vec<f64>,
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
        // Severity, where the runner supplied it. A NaN means the
        // evidence was not collected for that event, and it is dropped
        // rather than folded in as a number.
        for change in &o.route_changes {
            if change.boundary_margin.is_finite() {
                self.route_margins.push(f64::from(change.boundary_margin));
            }
            if change.weight_mass_moved.is_finite() {
                self.route_masses.push(f64::from(change.weight_mass_moved));
            }
        }
        if let Some(t) = &o.top1_change {
            if t.boundary_margin.is_finite() {
                self.top1_margins.push(f64::from(t.boundary_margin));
            }
            if t.candidate_margin_same_ids.is_finite() {
                self.top1_candidate_margins
                    .push(f64::from(t.candidate_margin_same_ids));
            }
            if t.mass_displaced.is_finite() {
                self.top1_masses.push(f64::from(t.mass_displaced));
            }
        }
        if let Some(t) = &o.top10_change {
            if t.boundary_margin.is_finite() {
                self.top10_margins.push(f64::from(t.boundary_margin));
            }
            if t.candidate_margin_same_ids.is_finite() {
                self.top10_candidate_margins
                    .push(f64::from(t.candidate_margin_same_ids));
            }
            if t.mass_displaced.is_finite() {
                self.top10_masses.push(f64::from(t.mass_displaced));
            }
            self.top10_rank_moves
                .push(f64::from(t.max_rank_displacement));
        }
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
                // The set is ordered, so its first element IS the
                // shallowest layer that ever moved.
                first_layer_with_route_change: self
                    .layers_with_route_change
                    .first()
                    .map(|i| *i as u64),
                route_margin: Distribution::of(&mut self.route_margins),
                route_weight_mass_moved: Distribution::of(&mut self.route_masses),
            },
            min_covered_mass: Some(self.min_covered_mass),
            top10_margin: Distribution::of(&mut self.top10_margins),
            top10_candidate_margin: Distribution::of(&mut self.top10_candidate_margins),
            top10_mass_displaced: Distribution::of(&mut self.top10_masses),
            top10_rank_displacement: Distribution::of(&mut self.top10_rank_moves),
            top1_margin: Distribution::of(&mut self.top1_margins),
            top1_candidate_margin: Distribution::of(&mut self.top1_candidate_margins),
            top1_mass_displaced: Distribution::of(&mut self.top1_masses),
        }
    }
}

#[cfg(test)]
#[path = "bank_tests.rs"]
mod tests;
