//! **What one move buys, and what fraction of scarce headroom it costs.**
//!
//! [`super::constraint`] says which behavioural resource is scarce.
//! [`super::execution_cost`] says what a map buys in decode time. This
//! is the bridge: given a parent map and a candidate move, what does
//! the move buy, and what does it consume?
//!
//! The objective BALANCED-SEARCH-2 optimises is unchanged:
//!
//! ```text
//! maximise predicted throughput
//! subject to the behavioural contract passing
//! ```
//!
//! What lives here is the HEURISTIC that decides which move to try
//! next. That is a different thing and must not be mistaken for the
//! objective.
//!
//! **Everything is marginal against the current map, never absolute.**
//! A candidate that leaves routing at 88 % has not "cost 88 %" — the
//! map was already at 83 %, so the move consumed five of the seventeen
//! points that remained, which is 29 % of the budget actually
//! available. Absolute utilisation cannot express that, and a search
//! ranking on it would keep choosing moves that eat the scarce resource
//! while reporting a comfortable-looking number.
//!
//! ```text
//! remaining route headroom before   17 %
//! this move consumes                 5 %
//! fraction of remaining consumed    5/17 = 29 %
//! ```
//!
//! **The vector is preserved underneath the score.** Collapsing to a
//! scalar permanently would throw away the interesting case: a move
//! that costs KL but FREES routing may be unusually attractive when
//! routing binds, and only the full before/after vectors can say so.
//! [`CandidateAssessment::ranking_score`] is derived convenience; the
//! vectors are the evidence.

use serde::{Deserialize, Serialize};

use super::byte_ledger::ByteLedger;
use super::constraint::{ConstraintVector, LimitKind, Margin};
use super::execution_cost::{CostPrediction, CostRefusal, ExecutionCostModel};
pub use super::measurement::EvidenceScale;
use super::measurement::TailSupportPolicy;
use super::quality::Criterion;
use super::quality::Statistic;
use super::search_evidence::{SearchCalibrationRegistry, SearchEvidence};

/// What a move did to one criterion.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct MarginalConstraintCost {
    pub criterion: Criterion,
    /// The gate's own wording for this limit, which is the key: a
    /// criterion can carry several limits that move independently.
    pub what: Statistic,
    /// Utilisation before the move, as a fraction of the limit.
    pub before: Option<f64>,
    pub after: Option<f64>,
    /// `after - before`. Negative means the move FREED headroom.
    pub delta: Option<f64>,
    /// Headroom that existed before the move, `1 - before`.
    pub remaining_before: Option<f64>,
    /// `delta / remaining_before` — the fraction of the budget that was
    /// actually available which this move spent.
    ///
    /// `None` when either side is unmeasured, or when no headroom
    /// remained: spending against an exhausted budget is not a
    /// fraction, and reporting a huge number would let it be averaged.
    /// Negative when the move freed headroom.
    pub fraction_of_remaining_consumed: Option<f64>,
    /// **How this dimension was obtained**, which decides whether its
    /// magnitude may be priced at all. A thin-tailed percentile with a
    /// good ordering correlation is real evidence and is still not a
    /// number to spend a budget against.
    pub search_evidence: SearchEvidence,
}

/// What a search needs in order to know which of its numbers mean what.
#[derive(Debug, Clone)]
pub struct EvidenceContext {
    pub scale: EvidenceScale,
    pub registry: SearchCalibrationRegistry,
    pub tail_policy: TailSupportPolicy,
}

impl EvidenceContext {
    /// The context this programme currently has: ROUTE-CAL-1's
    /// registrations and its five-tail-observation policy.
    pub fn route_cal_1(scale: EvidenceScale) -> Self {
        Self {
            scale,
            registry: SearchCalibrationRegistry::route_cal_1(),
            tail_policy: TailSupportPolicy::route_cal_1(),
        }
    }
}

impl MarginalConstraintCost {
    /// Whether this dimension's magnitude may be turned into a fraction
    /// of a remaining budget.
    pub fn priceable(&self) -> bool {
        self.search_evidence.is_priceable()
    }

    /// Whether it carries evidence usable for ORDERING, which is a
    /// weaker claim than pricing.
    pub fn orders(&self) -> bool {
        self.search_evidence.orders()
    }

    fn between(before: &Margin, after: &Margin, ctx: &EvidenceContext) -> Self {
        let (b, a) = (before.utilisation(), after.utilisation());
        let delta = match (b, a) {
            (Some(b), Some(a)) => Some(a - b),
            _ => None,
        };
        let remaining_before = b.map(|b| 1.0 - b);
        let fraction = match (delta, remaining_before) {
            (Some(d), Some(r)) if r > 0.0 => Some(d / r),
            _ => None,
        };
        // The WEAKER of the two sides governs: a move whose "after" is
        // well supported but whose "before" was not has no trustworthy
        // delta.
        let evidence = {
            let each = [before, after].map(|m| {
                ctx.registry.evidence_for(
                    m.what,
                    ctx.scale,
                    &m.measurement_status(&ctx.tail_policy),
                )
            });
            if each[0].confidence_rank() <= each[1].confidence_rank() {
                each[0].clone()
            } else {
                each[1].clone()
            }
        };
        Self {
            criterion: after.criterion,
            what: after.what,
            before: b,
            after: a,
            delta,
            remaining_before,
            fraction_of_remaining_consumed: fraction,
            search_evidence: evidence,
        }
    }

    /// Whether the move freed headroom on this criterion.
    pub fn frees(&self) -> bool {
        self.delta.is_some_and(|d| d < 0.0)
    }
}

/// Every criterion's marginal cost for one move.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct MarginalConstraintVector {
    pub costs: Vec<MarginalConstraintCost>,
}

impl MarginalConstraintVector {
    /// Pair the two vectors up by the gate's own wording for each
    /// limit. Criteria present in only one are dropped: they cannot
    /// have a before-and-after.
    pub fn between(
        before: &ConstraintVector,
        after: &ConstraintVector,
        ctx: &EvidenceContext,
    ) -> Self {
        let costs = after
            .margins
            .iter()
            .filter(|m| m.kind == LimitKind::Ceiling)
            .filter_map(|a| {
                before
                    .margins
                    .iter()
                    .find(|b| b.what == a.what)
                    .map(|b| MarginalConstraintCost::between(b, a, ctx))
            })
            .collect();
        Self { costs }
    }

    /// Dimensions that cost something but may not be priced — the
    /// candidate's invisible costs. A search that ignored these would
    /// prefer exactly the candidates whose expensive dimension it
    /// cannot see.
    pub fn unpriceable_costs(&self) -> impl Iterator<Item = &MarginalConstraintCost> {
        self.costs.iter().filter(|c| !c.priceable())
    }

    /// **The scarce-resource cost of this move**: the largest fraction
    /// of any criterion's remaining headroom that it consumed.
    ///
    /// `None` when the move consumed no headroom anywhere — every
    /// criterion unchanged or improved. That is a move with no
    /// behavioural price, not a move with a price of zero, and the
    /// ranking treats it accordingly.
    pub fn scarce_fraction_consumed(&self) -> Option<f64> {
        self.scarcest()?.fraction_of_remaining_consumed
    }

    /// The criterion that gave up the largest share of its remaining
    /// headroom. `None` when nothing was consumed.
    pub fn scarcest(&self) -> Option<&MarginalConstraintCost> {
        self.costs
            .iter()
            .filter(|c| c.priceable())
            .filter(|c| c.fraction_of_remaining_consumed.is_some_and(|f| f > 0.0))
            .max_by(|x, y| {
                let (a, b) = (
                    x.fraction_of_remaining_consumed
                        .unwrap_or(f64::NEG_INFINITY),
                    y.fraction_of_remaining_consumed
                        .unwrap_or(f64::NEG_INFINITY),
                );
                a.partial_cmp(&b).unwrap_or(std::cmp::Ordering::Equal)
            })
    }

    /// Criteria this move freed headroom on.
    pub fn freed(&self) -> impl Iterator<Item = &MarginalConstraintCost> {
        self.costs.iter().filter(|c| c.frees())
    }

    /// **A cost exists that could not be scored.** Either a criterion
    /// was already over budget and took more, or its evidence is
    /// missing on one side of the move.
    ///
    /// The distinction this draws is the one a search cannot get wrong:
    /// `fraction_of_remaining_consumed == None` means BOTH "consumed
    /// nothing" and "could not be scored", and those rank at opposite
    /// ends. Anything this returns true for is unrankable, not free.
    pub fn unscorable(&self) -> bool {
        self.costs.iter().any(|c| {
            // Either the arithmetic could not produce a fraction...
            (c.priceable()
                && c.fraction_of_remaining_consumed.is_none()
                && (c.delta.is_none() || c.delta.is_some_and(|d| d > 0.0)))
                // ...or the evidence does not license pricing one.
                || !c.priceable()
        })
    }

    /// A criterion that was already over budget before the move and
    /// took more. Not rankable, and a search must not treat it as
    /// cheap simply because no fraction could be computed.
    pub fn spent_past_an_exhausted_budget(&self) -> bool {
        self.costs.iter().any(|c| {
            c.remaining_before.is_some_and(|r| r <= 0.0) && c.delta.is_some_and(|d| d > 0.0)
        })
    }
}

/// **What kind of move this is**, which decides its rank before any
/// number does.
///
/// These are four genuinely different states, and collapsing any of
/// them to `0.0`, infinity or `NaN` produces a search that quietly
/// prefers the wrong thing. `Unscorable` in particular arrives as the
/// same `None` that `Unpriced` does, and ranks at the opposite end.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum MoveClass {
    /// Buys time and consumes no measurable headroom anywhere.
    Unpriced,
    /// Buys time at a measurable cost.
    Priced,
    /// Buys time, but a cost could not be scored — a criterion was
    /// already over budget, or evidence for one is missing.
    Unscorable,
    /// Buys no time, however cheap it was.
    Worthless,
}

impl MoveClass {
    /// Higher sorts first. Encoded as a method rather than left to the
    /// enum's declaration order so that reordering the variants cannot
    /// silently become search policy.
    pub(super) fn tier(self) -> u8 {
        match self {
            Self::Unpriced => 3,
            Self::Priced => 2,
            Self::Unscorable => 1,
            Self::Worthless => 0,
        }
    }
}

/// How a move ranks: what it buys over what it costs.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct RankingScore {
    pub class: MoveClass,
    /// GPU ms/token this move saves against the parent map.
    pub gpu_ms_saved: f64,
    /// Largest fraction of any criterion's remaining headroom consumed.
    pub scarce_fraction_consumed: Option<f64>,
    /// `gpu_ms_saved / scarce_fraction_consumed`.
    ///
    /// `None` when the move consumed no headroom — such a move is not
    /// infinitely good, it is simply unpriced, and it ranks ahead of
    /// every priced one rather than dividing by zero.
    pub score: Option<f64>,
}

impl RankingScore {
    /// **The ranking policy, stated once.** Higher sorts first.
    ///
    /// ```text
    /// 1. unpriced moves    by physical gain
    /// 2. priced moves      by gain per unit of scarce headroom
    /// 3. unscorable moves  by physical gain
    /// 4. zero-gain moves   by physical gain
    /// ```
    ///
    /// Written down because otherwise `sort_by` becomes the policy by
    /// accident, and a search whose preference order is an emergent
    /// property of a comparator is not reproducible.
    pub fn rank_key(&self) -> (u8, f64, f64) {
        let within = match (self.class, self.score) {
            (MoveClass::Priced, Some(s)) => s,
            _ => self.gpu_ms_saved,
        };
        // Lower consumption breaks a tie on the primary key.
        let frugality = -self.scarce_fraction_consumed.unwrap_or(0.0);
        (self.class.tier(), within, frugality)
    }

    /// Descending order — best move first. A TOTAL order: `total_cmp`
    /// leaves no pair incomparable, so a sort cannot depend on input
    /// order even if a key were ever `NaN`.
    pub fn cmp_rank(&self, other: &Self) -> std::cmp::Ordering {
        let (a, b) = (other.rank_key(), self.rank_key());
        a.0.cmp(&b.0)
            .then_with(|| a.1.total_cmp(&b.1))
            .then_with(|| a.2.total_cmp(&b.2))
    }
}

/// Whether a move's evidence lets it enter a map.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum Admission {
    /// Authority-scale evidence, every criterion met. The only variant
    /// that may enter a final map.
    Earned,
    /// Diagnostic-scale evidence with every ceiling met and the
    /// instrument sound. Rankable and searchable; NOT admissible, and
    /// deliberately a different word so a ranking cannot be quoted as
    /// an admission.
    Estimated,
    /// A criterion is not met, at whatever scale.
    Refused { failures: Vec<String> },
}

/// **One move, assessed: what it buys, what it costs, and on what
/// evidence.**
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct CandidateAssessment {
    pub parent_map: String,
    pub candidate_map: String,
    pub scale: EvidenceScale,

    /// Bytes/token this move removes against the PARENT map.
    pub bytes_removed_marginal: u64,
    /// Absolute prediction for the parent map and for the candidate.
    pub parent_cost: CostPrediction,
    pub candidate_cost: CostPrediction,

    /// The authoritative evidence, kept whole beneath the score.
    pub before: ConstraintVector,
    pub after: ConstraintVector,
    pub marginal: MarginalConstraintVector,

    pub ranking_score: RankingScore,
}

impl CandidateAssessment {
    /// Assess a move from `parent` to `candidate`.
    ///
    /// Both ledgers are ABSOLUTE — each against the same baseline
    /// representation the cost observation was measured against — so
    /// that each can be priced. The marginal quantities are then the
    /// difference between the two predictions, which is what the
    /// ranking wants.
    pub fn of(
        ctx: &EvidenceContext,
        costs: &ExecutionCostModel,
        parent: &ByteLedger,
        candidate: &ByteLedger,
        before: ConstraintVector,
        after: ConstraintVector,
    ) -> Result<Self, CostRefusal> {
        let scale = ctx.scale;
        let parent_cost = costs.predict(parent)?;
        let candidate_cost = costs.predict(candidate)?;
        let marginal = MarginalConstraintVector::between(&before, &after, ctx);
        let gpu_ms_saved = parent_cost.gpu_ms_per_token - candidate_cost.gpu_ms_per_token;
        let scarce = marginal.scarce_fraction_consumed();
        let score = match scarce {
            Some(f) if f > 0.0 => Some(gpu_ms_saved / f),
            _ => None,
        };
        // Order matters: a move that buys nothing is worthless whatever
        // it cost, and an unscorable cost outranks neither.
        let class = if gpu_ms_saved <= 0.0 {
            MoveClass::Worthless
        } else if marginal.unscorable() {
            MoveClass::Unscorable
        } else if score.is_some() {
            MoveClass::Priced
        } else {
            MoveClass::Unpriced
        };
        Ok(Self {
            parent_map: parent.candidate_representation.clone(),
            candidate_map: candidate.candidate_representation.clone(),
            scale,
            bytes_removed_marginal: parent
                .candidate_bytes_per_token()
                .saturating_sub(candidate.candidate_bytes_per_token()),
            parent_cost,
            candidate_cost,
            before,
            after,
            marginal,
            ranking_score: RankingScore {
                class,
                gpu_ms_saved,
                scarce_fraction_consumed: scarce,
                score,
            },
        })
    }

    /// **Total order over moves, best first.** [`RankingScore`] decides,
    /// with the candidate's identity as the final tiebreak so that two
    /// moves of identical economics always sort the same way and a
    /// search trace reproduces exactly.
    pub fn cmp_rank(&self, other: &Self) -> std::cmp::Ordering {
        self.ranking_score
            .cmp_rank(&other.ranking_score)
            .then_with(|| self.candidate_map.cmp(&other.candidate_map))
    }

    /// The scarce resource before the move.
    pub fn binding_before(&self) -> Option<&Margin> {
        self.before.binding()
    }

    /// The scarce resource after it — which may be a different
    /// criterion, and a search must recheck rather than assume.
    pub fn binding_after(&self) -> Option<&Margin> {
        self.after.binding()
    }

    /// Whether this move's evidence lets it enter a map.
    ///
    /// The position floor is scale-dependent — a diagnostic bank is
    /// short BY DEFINITION and failing it says nothing about the
    /// candidate. Coverage is not: a blind diagnostic is still blind,
    /// and a ranking computed from one is worthless in exactly the way
    /// an admission from one would be.
    pub fn admission(&self) -> Admission {
        let mut failures: Vec<String> = self
            .after
            .margins
            .iter()
            .filter(|m| {
                self.scale == EvidenceScale::Authority || m.criterion != Criterion::Positions
            })
            .filter(|m| !m.satisfied())
            .map(|m| m.what.label().to_string())
            .collect();
        failures.sort();
        if !failures.is_empty() {
            return Admission::Refused { failures };
        }
        match self.scale {
            EvidenceScale::Authority => Admission::Earned,
            EvidenceScale::Diagnostic => Admission::Estimated,
        }
    }

    /// The search trace: why this move ranks where it does, in the form
    /// a reader can argue with.
    pub fn describe(&self) -> String {
        let mut out = format!(
            "candidate: {} -> {}\n\nphysical:\n  -{:.0} MB/token\n  \
             predicted {:+.2} ms GPU\n\nbehavioural:\n",
            self.parent_map,
            self.candidate_map,
            self.bytes_removed_marginal as f64 / 1e6,
            -self.ranking_score.gpu_ms_saved,
        );
        let scarcest = self.marginal.scarcest().map(|c| c.what);
        for c in &self.marginal.costs {
            let share = match (c.priceable(), c.fraction_of_remaining_consumed) {
                (true, Some(f)) => format!("{:+.0}% of remaining headroom", 100.0 * f),
                (true, None) => "not scored".into(),
                (false, _) if c.orders() => "NOT PRICEABLE at this scale (ordering proxy)".into(),
                (false, _) => "NOT PRICEABLE at this scale (no usable evidence)".into(),
            };
            let mark = if Some(c.what) == scarcest {
                "   <- scarce resource"
            } else {
                ""
            };
            out.push_str(&format!("  {:<32} {share}{mark}\n", c.what));
        }
        out.push_str(&match self.ranking_score.score {
            Some(s) => format!(
                "\nrank: {:.2} ms per unit of scarce headroom ({:.0}% consumed)\n",
                s,
                100.0 * self.ranking_score.scarce_fraction_consumed.unwrap_or(0.0),
            ),
            None if self.marginal.unpriceable_costs().next().is_some() => format!(
                "\nrank: {:+.2} ms GPU, NOT priced — {} of {} dimensions cannot be priced at \
                 this evidence scale\n",
                self.ranking_score.gpu_ms_saved,
                self.marginal.unpriceable_costs().count(),
                self.marginal.costs.len(),
            ),
            None => format!(
                "\nrank: {:+.2} ms GPU at no behavioural cost\n",
                self.ranking_score.gpu_ms_saved
            ),
        });
        out.push_str(&format!(
            "evidence: {:?}, {:?}\n",
            self.scale,
            self.admission()
        ));
        out
    }
}

#[cfg(test)]
#[path = "assessment_tests.rs"]
mod tests;
