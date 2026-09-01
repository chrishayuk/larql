//! **R4-F1 — proxy evidence orders promotion without becoming price.**
//!
//! Rung 4's first run ranked four one-step moves and printed
//! `PROMOTE: e26`, matching the pre-registered prediction. Reversing the
//! input list printed `PROMOTE: e20`, from identical reports. All four
//! `rank_key`s were byte-identical — `tier=1 within=0.702202900
//! frugality=-0.0` — because `within` falls back to `gpu_ms_saved` for
//! any non-`Priced` class and every one-step move removed the same
//! bytes. `cmp_rank` returned `Equal`, and a stable sort handed back
//! INPUT ORDER.
//!
//! At diagnostic scale nothing is priceable BY DESIGN, so that tie is
//! not an edge case; it is the normal state. Meanwhile both calibrated
//! ordering proxies separated the four candidates monotonically and
//! unanimously. **The evidence existed and the comparator could not see
//! it.**
//!
//! # The invariant
//!
//! > Determinism may order indistinguishable candidates for DISPLAY; it
//! > may never convert indistinguishability into evidence for
//! > PROMOTION.
//!
//! [`display_order`] may use candidate identity as its final tie-break.
//! [`decide_promotion`] is not given identity at all, so it cannot.
//!
//! # Ordinal, never scalar
//!
//! An [`OrderingProxy`] licenses ORDER and not MAGNITUDE. A numeric
//! proxy term added to [`RankingScore`] would smuggle magnitude back in
//! through the comparator — "A is 1.7x better than B" is exactly the
//! claim ROUTE-CAL-1 refused to make. So this layer compares ranks:
//!
//! ```text
//! physical economics   can say how much a move buys
//! proxy evidence       can say A should be tried before B
//! neither says         "A is 1.7x better than B"
//! ```
//!
//! # When proxies disagree
//!
//! There is no empirical basis for how many places of kl are worth one
//! place of routing, so a conflict is REFUSED rather than scalarised —
//! [`PromotionDecision::Ambiguous`] names the frontier and why. The
//! search can then seek more evidence or spend authority under an
//! explicitly different exploration policy, which is a decision someone
//! makes on purpose rather than one a comparator makes by accident.
//!
//! [`OrderingProxy`]: super::search_evidence::SearchEvidence::OrderingProxy
//! [`RankingScore`]: super::assessment::RankingScore

use std::cmp::Ordering;

use serde::{Deserialize, Serialize};

use super::diagnostic::DiagnosticVector;
use super::measurement::TailSupportPolicy;
use super::promotion::{PromotionCandidate, PromotionReadiness};
use super::quality::Statistic;
use super::search_evidence::SearchCalibrationRegistry;

/// One candidate in a search round: what it is, how it was assessed, and
/// what the diagnostic scale actually observed of it.
#[derive(Debug, Clone, PartialEq)]
pub struct SearchCandidate {
    pub id: String,
    pub promotion: PromotionCandidate,
    pub diagnostic: DiagnosticVector,
}

impl SearchCandidate {
    /// The statistics BOTH candidates observed and that ORDER at this
    /// scale. A statistic one of them is silent on cannot rank them, and
    /// one the registry judges `Unusable` here must not.
    fn comparable(
        &self,
        other: &Self,
        registry: &SearchCalibrationRegistry,
        policy: &TailSupportPolicy,
    ) -> Vec<Statistic> {
        self.diagnostic
            .readings
            .iter()
            .filter(|r| {
                r.observed.is_some()
                    && r.evidence(registry, policy).orders()
                    && other.diagnostic.reading(r.statistic).is_some_and(|o| {
                        o.observed.is_some() && o.evidence(registry, policy).orders()
                    })
            })
            .map(|r| r.statistic)
            .collect()
    }

    /// Ordinal comparison on one statistic. `Less` means `self` is
    /// better.
    fn order_on(&self, other: &Self, s: Statistic) -> Option<Ordering> {
        let a = self.diagnostic.reading(s)?.observed?;
        let b = other.diagnostic.reading(s)?.observed?;
        Some(s.order(a, b))
    }

    /// Authority dimensions no diagnostic evidence can speak to: an
    /// unpriceable cost that does not order itself and has no proxy
    /// standing in for it.
    ///
    /// These are what makes a candidate `Uninformed`, and naming them is
    /// the condition on which `Uninformed` may still be measured.
    pub fn unresolved_dimensions(&self) -> Vec<Statistic> {
        let mut out: Vec<Statistic> = self
            .promotion
            .assessment
            .marginal
            .unpriceable_costs()
            .filter(|c| !c.orders() && self.promotion.proxy_for(c.what).is_none())
            .map(|c| c.what)
            .collect();
        out.sort_by_key(|s| s.label());
        out.dedup();
        out
    }

    /// **`self` proxy-dominates `other`**: no worse on every comparable
    /// proxy, and strictly better on at least one.
    ///
    /// Vacuously false when nothing is comparable — an absence of
    /// evidence never dominates.
    pub fn proxy_dominates(
        &self,
        other: &Self,
        registry: &SearchCalibrationRegistry,
        policy: &TailSupportPolicy,
    ) -> bool {
        let shared = self.comparable(other, registry, policy);
        if shared.is_empty() {
            return false;
        }
        let mut strictly_better = false;
        for s in shared {
            match self.order_on(other, s) {
                Some(Ordering::Greater) => return false,
                Some(Ordering::Less) => strictly_better = true,
                _ => {}
            }
        }
        strictly_better
    }
}

/// Why a round could not name one candidate.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum AmbiguityReason {
    /// Proxies point different ways and nothing licenses trading one
    /// against the other.
    ConflictingOrderingProxies,
    /// Nothing on the frontier could be compared at all — the search
    /// knows it does not know, which outranks a confident guess.
    NoOrderingEvidence,
    /// Every comparable proxy says the candidates are equal, and no
    /// physical difference separates them either.
    IndistinguishableOnEveryProxy,
}

/// Why a round had nothing to promote.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum NoPromotableCandidate {
    EmptySet,
    /// Every move buys nothing, whatever its evidence.
    EveryMoveWorthless,
}

/// What justified a promotion. Never "it sorted first".
#[derive(Debug, Clone, PartialEq)]
pub struct PromotionEvidence {
    /// Candidates this one proxy-dominates.
    pub dominated: Vec<String>,
    /// The proxies that did the separating.
    pub deciding: Vec<Statistic>,
    pub readiness: PromotionReadiness,
    /// True when the proxies tied and physical gain broke it — a
    /// legitimate later stage, recorded so a trace never has to guess
    /// which stage decided.
    pub decided_by_physical_gain: bool,
    /// **Authority dimensions the diagnostic scale could not speak to.**
    ///
    /// Named explicitly because selection is not prediction: these are
    /// exactly the criteria on which this candidate might still be
    /// refused, and a trace that omitted them would read as confidence.
    pub unresolved: Vec<Statistic>,
}

/// The outcome of one search round.
#[derive(Debug, Clone, PartialEq)]
pub enum PromotionDecision {
    /// **Selected to MEASURE, not predicted admissible.**
    ///
    /// The candidate earned the next authority run because the evidence
    /// distinguishes it — never because the unresolved dimensions are
    /// assumed benign. `Uninformed` readiness is ELIGIBLE here by
    /// design: authority is the mechanism that resolves dimensions the
    /// diagnostic scale cannot see, so refusing to measure them would
    /// silently rewrite the doctrine into "diagnostic must predict every
    /// authority dimension before authority may run", which at 256
    /// positions is impossible for the mass tails.
    ///
    /// ```text
    /// Uninformed  !=  safe
    /// Uninformed  !=  admissible
    /// Uninformed  ==  diagnostic evidence cannot speak about every
    ///                 authority dimension
    /// ```
    ///
    /// The policy, stated once:
    ///
    /// ```text
    /// Uninformed may be selected for authority IFF
    ///   it is otherwise promotable (its class buys something),
    ///   the missing dimensions are EXPLICIT (`unresolved`),
    ///   and no better-evidenced candidate outranks it (the frontier).
    /// ```
    ///
    /// Never `Uninformed => refuse`, and never
    /// `Uninformed => assume the missing dimensions are free`.
    SelectForAuthority {
        candidate: String,
        evidence: PromotionEvidence,
    },
    Ambiguous {
        candidates: Vec<String>,
        reason: AmbiguityReason,
    },
    None {
        reason: NoPromotableCandidate,
    },
}

/// **Deterministic order for a TABLE, best first.** Identity is the
/// final tie-break, which is fine here and forbidden in
/// [`decide_promotion`].
///
/// Dominance COUNT is used as an ordinal summary so the table reads
/// best-first. Sorting on `cmp_rank` alone put the WORST candidate at
/// the top of the Rung-4 trace — every key tied, so identity decided,
/// and `e20` sorted before `e26`. A summary is legitimate here precisely
/// because display is not evidence: this number never reaches
/// [`decide_promotion`], which compares candidates pairwise.
pub fn display_order<'a>(
    candidates: &'a [SearchCandidate],
    registry: &SearchCalibrationRegistry,
    policy: &TailSupportPolicy,
) -> Vec<&'a SearchCandidate> {
    let dominates = |c: &SearchCandidate| {
        candidates
            .iter()
            .filter(|o| !std::ptr::eq(*o, c) && c.proxy_dominates(o, registry, policy))
            .count()
    };
    let mut out: Vec<&SearchCandidate> = candidates.iter().collect();
    out.sort_by(|a, b| {
        b.promotion
            .assessment
            .ranking_score
            .class
            .tier()
            .cmp(&a.promotion.assessment.ranking_score.class.tier())
            .then_with(|| dominates(b).cmp(&dominates(a)))
            .then_with(|| {
                b.promotion
                    .assessment
                    .ranking_score
                    .gpu_ms_saved
                    .total_cmp(&a.promotion.assessment.ranking_score.gpu_ms_saved)
            })
            .then_with(|| a.id.cmp(&b.id))
    });
    out
}

/// **The staged comparator.** Class, then proxy ordering, then physical
/// gain, then frugality — and never identity.
///
/// Physical gain may separate candidates the proxies called EQUAL. It
/// may not overrule a proxy CONFLICT: that is refused, because trading
/// kl places against routing places is a claim no calibration supports.
pub fn decide_promotion(
    candidates: &[SearchCandidate],
    registry: &SearchCalibrationRegistry,
    policy: &TailSupportPolicy,
) -> PromotionDecision {
    if candidates.is_empty() {
        return PromotionDecision::None {
            reason: NoPromotableCandidate::EmptySet,
        };
    }
    // 1. Promotion class.
    let best_tier = candidates
        .iter()
        .map(|c| c.promotion.assessment.ranking_score.class.tier())
        .max()
        .unwrap_or(0);
    if best_tier == 0 {
        return PromotionDecision::None {
            reason: NoPromotableCandidate::EveryMoveWorthless,
        };
    }
    let pool: Vec<&SearchCandidate> = candidates
        .iter()
        .filter(|c| c.promotion.assessment.ranking_score.class.tier() == best_tier)
        .collect();

    // 2. Proxy ordering: the non-dominated frontier.
    let frontier: Vec<&SearchCandidate> = pool
        .iter()
        .filter(|c| {
            !pool
                .iter()
                .any(|o| !std::ptr::eq(*o, **c) && o.proxy_dominates(c, registry, policy))
        })
        .copied()
        .collect();

    if frontier.len() == 1 {
        let winner = frontier[0];
        let mut dominated: Vec<String> = pool
            .iter()
            .filter(|o| !std::ptr::eq(**o, winner) && winner.proxy_dominates(o, registry, policy))
            .map(|o| o.id.clone())
            .collect();
        // The RECORD is sorted too. The decision was already invariant,
        // but a trace that differs between runs of the same round is not
        // reproducible, and identity is allowed to order a record.
        dominated.sort();
        let deciding = pool
            .iter()
            .find(|o| !std::ptr::eq(**o, winner))
            .map(|o| winner.comparable(o, registry, policy))
            .unwrap_or_default();
        return PromotionDecision::SelectForAuthority {
            candidate: winner.id.clone(),
            evidence: PromotionEvidence {
                dominated,
                deciding,
                readiness: winner.promotion.readiness(),
                decided_by_physical_gain: false,
                unresolved: winner.unresolved_dimensions(),
            },
        };
    }

    // The frontier did not settle it. WHY decides what happens next.
    let mut any_comparable = false;
    let mut any_difference = false;
    for (i, a) in frontier.iter().enumerate() {
        for b in frontier.iter().skip(i + 1) {
            let shared = a.comparable(b, registry, policy);
            if !shared.is_empty() {
                any_comparable = true;
            }
            if shared
                .iter()
                .any(|s| a.order_on(b, *s) != Some(Ordering::Equal))
            {
                any_difference = true;
            }
        }
    }
    let ids = |v: &[&SearchCandidate]| {
        let mut o: Vec<String> = v.iter().map(|c| c.id.clone()).collect();
        o.sort();
        o
    };
    if !any_comparable {
        // Silence, not equality. Refuse.
        return PromotionDecision::Ambiguous {
            candidates: ids(&frontier),
            reason: AmbiguityReason::NoOrderingEvidence,
        };
    }
    if any_difference {
        // The proxies point different ways. Do NOT scalarise.
        return PromotionDecision::Ambiguous {
            candidates: ids(&frontier),
            reason: AmbiguityReason::ConflictingOrderingProxies,
        };
    }

    // 3./4. The proxies AGREE these are equal. Physical evidence may
    // separate them; identity may not.
    let best_gain = frontier
        .iter()
        .map(|c| c.promotion.assessment.ranking_score.gpu_ms_saved)
        .fold(f64::NEG_INFINITY, f64::max);
    let by_gain: Vec<&SearchCandidate> = frontier
        .iter()
        .filter(|c| c.promotion.assessment.ranking_score.gpu_ms_saved == best_gain)
        .copied()
        .collect();
    if by_gain.len() == 1 {
        return PromotionDecision::SelectForAuthority {
            candidate: by_gain[0].id.clone(),
            evidence: PromotionEvidence {
                dominated: Vec::new(),
                deciding: Vec::new(),
                readiness: by_gain[0].promotion.readiness(),
                decided_by_physical_gain: true,
                unresolved: by_gain[0].unresolved_dimensions(),
            },
        };
    }
    PromotionDecision::Ambiguous {
        candidates: ids(&by_gain),
        reason: AmbiguityReason::IndistinguishableOnEveryProxy,
    }
}

#[cfg(test)]
#[path = "decision_tests.rs"]
mod tests;
