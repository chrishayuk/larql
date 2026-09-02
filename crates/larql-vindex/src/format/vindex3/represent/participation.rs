//! **R4-F10 — structural invariance is not ordering evidence.**
//!
//! Rung-4 iteration 4 ranked four one-step moves against the M26 parent
//! and REFUSED to promote: the frontier was `{H, K25}` and the proxies
//! conflicted. K25 was better on kl; H was better on route flip rate.
//!
//! H is `lm_head Q8_0 -> Q6_K`. The head is applied to the final hidden
//! state and feeds no router, so H cannot change a routing decision at
//! all — and it did not. Its whole routing block was BIT-IDENTICAL to
//! the parent's, `route_margin.p50` included, to 17 significant figures
//! over 256 positions. Its flip rate was the PARENT's value carried
//! through a causal path H cannot touch.
//!
//! [`proxy_dominates`] could not tell those apart:
//!
//! ```text
//! candidate improved the route proxy
//! candidate CANNOT INFLUENCE the route proxy
//! ```
//!
//! Both present as "no worse", so an unchanged value entered a dominance
//! test as though it had been won, and blocked a round that had a clean
//! answer on the evidence that actually applied.
//!
//! # The rule
//!
//! > A diagnostic statistic that a candidate cannot causally affect must
//! > not enter proxy dominance as though its unchanged value were
//! > favourable evidence. Structural invariance instead licenses an
//! > exact ZERO SPEND on the corresponding authority dimension, where
//! > that causal relationship is proven.
//!
//! Two questions, kept apart:
//!
//! ```text
//! SEARCH                which candidate should authority MEASURE?
//!                       compare only statistics candidates can affect
//! CONSTRAINT ACCOUNTING what can this action CONSUME?
//!                       structural invariance means exact zero spend
//! ```
//!
//! # Why this is not `Direct`
//!
//! An invariant statistic is not a superior measurement. It is a
//! statement that the candidate cannot generate a measurement-relevant
//! effect at all. Calling it `Direct` would say the opposite — that its
//! zero is an unusually trustworthy number — and would put it straight
//! back into pricing, which is the failure BS2-F2 already was.
//!
//! # DECLARED, never inferred
//!
//! Participation comes from the ACTION's position in the computation
//! graph, never from noticing a zero delta. A zero delta is what a
//! structural invariant looks like, not evidence that one exists: K25's
//! flips could coincide with the parent's on some other bank and would
//! still be a real routing participant. Inferring invariance from an
//! observation would let a coincidence silently delete a dimension from
//! the comparison — the [`Statistic`] equivalent of the adjacency
//! reasoning that `UnsupportedComponent` is forbidden to use.
//!
//! The default is [`StatisticParticipation::Affected`]: an undeclared
//! statistic PARTICIPATES. Exclusion must be asked for.
//!
//! # And then CHECKED
//!
//! A declaration that is never checked is an assumption wearing a type.
//! [`ParticipationDeclaration::verify`] requires every statistic
//! declared invariant to be BIT-IDENTICAL between parent and candidate,
//! and refuses otherwise — a declaration that would have hidden a real
//! effect is a hard error, not a warning.
//!
//! [`proxy_dominates`]: super::decision::SearchCandidate::proxy_dominates

use std::collections::BTreeMap;

use serde::{Deserialize, Serialize};

use super::diagnostic::DiagnosticVector;
use super::statistic::Statistic;

/// How a candidate's action relates CAUSALLY to one statistic.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum StatisticParticipation {
    /// The action can causally affect this statistic. The default, and
    /// the conservative reading: a participant's evidence counts.
    Affected,
    /// The action provably CANNOT affect this statistic, because of
    /// where it sits in the computation graph.
    ///
    /// Excluded from candidate-vs-candidate ordinal dominance. Still
    /// exact knowledge for accounting: zero spend on this dimension.
    StructurallyInvariant {
        /// The causal ground, for the trace. This never joins on
        /// anything — it explains, and [`ParticipationDeclaration::verify`]
        /// is what actually checks the claim.
        because: String,
    },
}

impl StatisticParticipation {
    /// May this statistic rank the candidate against another?
    pub fn participates(&self) -> bool {
        matches!(self, Self::Affected)
    }

    /// Is the candidate's spend on this dimension known to be exactly
    /// zero, rather than merely unmeasured?
    pub fn is_structurally_invariant(&self) -> bool {
        matches!(self, Self::StructurallyInvariant { .. })
    }
}

/// What a candidate's action declares it can and cannot affect.
///
/// Absent means [`StatisticParticipation::Affected`] — the checked
/// default. Only invariance is recorded, because only invariance
/// removes evidence from a comparison.
#[derive(Debug, Clone, PartialEq, Eq, Default, Serialize, Deserialize)]
pub struct ParticipationDeclaration {
    invariant: BTreeMap<Statistic, String>,
}

/// A declared structural invariant that the measurement CONTRADICTS.
///
/// The candidate moved a statistic it declared it could not move, so the
/// declaration would have deleted a real effect from the comparison.
#[derive(Debug, Clone, PartialEq)]
pub struct ParticipationViolation {
    pub statistic: Statistic,
    pub because: String,
    pub parent: Option<f64>,
    pub candidate: Option<f64>,
}

impl std::fmt::Display for ParticipationViolation {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "{} was declared structurally invariant ({}) but moved: parent {:?} -> candidate {:?}",
            self.statistic, self.because, self.parent, self.candidate
        )
    }
}

impl std::error::Error for ParticipationViolation {}

impl ParticipationDeclaration {
    /// Everything participates. What an action declares when it sits
    /// upstream of every statistic the search reads.
    pub fn all_affected() -> Self {
        Self::default()
    }

    /// Declare that this action cannot affect `statistic`, and why.
    pub fn structurally_invariant(mut self, statistic: Statistic, because: &str) -> Self {
        self.invariant.insert(statistic, because.to_string());
        self
    }

    /// How this action relates to `statistic`.
    pub fn of(&self, statistic: Statistic) -> StatisticParticipation {
        match self.invariant.get(&statistic) {
            Some(because) => StatisticParticipation::StructurallyInvariant {
                because: because.clone(),
            },
            None => StatisticParticipation::Affected,
        }
    }

    /// The dimensions on which this action's spend is exactly zero.
    ///
    /// **Not "unresolved".** These are the one class of authority
    /// dimension a diagnostic round can speak to with certainty, and a
    /// trace that filed them under uncertainty would understate what the
    /// search knows.
    pub fn known_zero_spend(&self) -> Vec<Statistic> {
        self.invariant.keys().copied().collect()
    }

    /// Every declared invariant must be BIT-IDENTICAL between parent and
    /// candidate.
    ///
    /// Bit equality, not tolerance: a structural invariant is the same
    /// number arriving by the same path, and anything else is a
    /// different claim. NaN compares equal to NaN here for the same
    /// reason — identical bits are identical evidence.
    pub fn verify(
        &self,
        parent: &DiagnosticVector,
        candidate: &DiagnosticVector,
    ) -> Result<(), ParticipationViolation> {
        for (&statistic, because) in &self.invariant {
            let before = parent.reading(statistic).and_then(|r| r.observed);
            let after = candidate.reading(statistic).and_then(|r| r.observed);
            let identical = match (before, after) {
                (None, None) => true,
                (Some(a), Some(b)) => a.to_bits() == b.to_bits(),
                _ => false,
            };
            if !identical {
                return Err(ParticipationViolation {
                    statistic,
                    because: because.clone(),
                    parent: before,
                    candidate: after,
                });
            }
        }
        Ok(())
    }
}

#[cfg(test)]
#[path = "participation_tests.rs"]
mod tests;
