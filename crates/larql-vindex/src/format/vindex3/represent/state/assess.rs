//! **What can be derived about an eligible candidate from facts already
//! held.**
//!
//! ```text
//! EligibleCandidate → assess() → Assessment → SearchPolicy → ordered frontier
//! ```
//!
//! [`Assessment`] carries ingredients and no scalar of record. That is
//! deliberate and it is the lesson of 1d: `CandidateAssessment` holds a
//! `ranking_score`, and a score is a CONCLUSION — persisting one is
//! exactly why promotion could not be replayed from stored facts. A
//! score here is computed on demand under a named rule
//! ([`Assessment::score`]) and never stored.
//!
//! # Sign convention
//!
//! `physical_delta` keeps the sign it has everywhere else in this
//! module: **negative removes bytes**. Ordering therefore prefers the
//! most negative value, and no layer flips the sign on the way past —
//! a sign flip between the accounting and the comparator is the shape
//! of R4-F7's mistake at a different altitude.
//!
//! # What is deliberately not here
//!
//! No information-gain model. A 40 MB experiment that would resolve
//! whether a whole family of moves is state-dependent may well deserve
//! to run before a 600 MB candidate, and *experimental value* is a real
//! quantity distinct from *physical value* — but nothing in this
//! programme has measured it, and inventing it here would put a
//! heuristic where the registered semantics belong. The types leave room
//! for a second rule; they do not guess at one.

use serde::{Deserialize, Serialize};

use super::super::compile::hash_bytes;
use super::super::constraint::ConstraintVector;
use super::candidate::Candidate;
use super::identity::RepresentationStateId;
use super::key::MeasurementKey;
use super::realization::{LogicalBytes, RealizationId};
use super::surface::{FIELD, SECTION};
use super::transition::Action;

/// The canonical-form version every ranking id is computed under.
pub const RANKING_SEMANTICS_ID_VERSION: &str = "ranking-semantics-id/v1";

/// **Which registered rule orders candidates.**
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub struct RankingSemanticsId(String);

impl RankingSemanticsId {
    pub fn as_str(&self) -> &str {
        &self.0
    }

    pub fn short(&self) -> &str {
        &self.0[..self.0.len().min(12)]
    }
}

impl std::fmt::Display for RankingSemanticsId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.0)
    }
}

/// The rules this programme has actually registered.
///
/// One variant, because one rule has actually been used. Rung 5 chose
/// what to build and measure by physical prize — its neighbourhoods rank
/// `−M26 +E24` at −431,777,920 B against `−M26 +K24` at −2,091,136 B and
/// spend the run on the prize — and no other pre-measurement ordering
/// has been registered. A second variant is a decision someone makes on
/// purpose, not a gap to fill speculatively.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum RankingRule {
    /// Order by the exact physical prize, most bytes removed first.
    ///
    /// The only quantity available before an experiment has been run
    /// that is *exact*. Diagnostic evidence cannot order a candidate
    /// that has not been measured, and R5-F4/R5-F9 forbid reading a
    /// parent's diagnostic across to a child's authority.
    PhysicalPrizeFirst,
}

impl RankingRule {
    pub fn name(self) -> &'static str {
        match self {
            Self::PhysicalPrizeFirst => "physical-prize-first",
        }
    }
}

/// The ordering rule in force, and why.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct RankingSemantics {
    pub rule: RankingRule,
    /// Where the rule comes from. Provenance — **excluded from the id**,
    /// for the reason [`super::instrument`] states: a reworded
    /// justification must not split a scientific record.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub provenance: Option<String>,
}

impl RankingSemantics {
    pub fn new(rule: RankingRule) -> Self {
        Self {
            rule,
            provenance: None,
        }
    }

    pub fn because(mut self, provenance: impl Into<String>) -> Self {
        self.provenance = Some(provenance.into());
        self
    }

    /// **The complete order, stated once.**
    ///
    /// Every element after the first exists so that no answer can depend
    /// on insertion order, map iteration order, thread completion order
    /// or an accident of vocabulary traversal. Replay needs one answer,
    /// not a usually-stable one.
    pub fn tie_break_chain(&self) -> &'static [&'static str] {
        &[
            "registered rule",
            "greater physical improvement",
            "canonical child state id",
            "canonical child realization id",
            "canonical action identity",
        ]
    }

    pub fn id(&self) -> RankingSemanticsId {
        let input = format!(
            "{RANKING_SEMANTICS_ID_VERSION}{SECTION}rule={}{FIELD}tie_breaks={}",
            self.rule.name(),
            self.tie_break_chain().join(",")
        );
        RankingSemanticsId(hash_bytes(input.as_bytes()))
    }
}

/// A score, derived on demand and never stored.
///
/// Ordered so that **less is better**, matching `physical_delta`'s sign:
/// the best candidate removes the most bytes and therefore has the most
/// negative delta.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub struct Score(i64);

impl Score {
    pub fn get(self) -> i64 {
        self.0
    }
}

/// What a state's standing against the contract is, where an
/// observation of it exists.
///
/// A trait because deriving it needs the gate, which belongs to the
/// snapshot rather than to a candidate — and because a policy that
/// reached for a gate itself would be a second place the contract is
/// applied.
pub trait ParentStanding {
    /// `None` where the state carries no reading. Absence is a fact.
    fn of(&self, state: &RepresentationStateId) -> Option<ConstraintVector>;
}

/// Nothing has been measured. For a first round, and for tests about
/// ordering rather than about evidence.
#[derive(Debug, Clone, Copy, Default)]
pub struct NothingMeasured;

impl ParentStanding for NothingMeasured {
    fn of(&self, _state: &RepresentationStateId) -> Option<ConstraintVector> {
        None
    }
}

/// **What is derivable about one eligible candidate.**
///
/// Ingredients only. Everything here is either an exact computation over
/// stored facts or a `None` saying the fact is absent — there is no
/// estimate, no prior and no judgement.
#[derive(Debug, Clone, PartialEq)]
pub struct Assessment {
    pub action: Action,
    pub parent_state: RepresentationStateId,
    pub child_state: RepresentationStateId,
    pub child_realization: RealizationId,
    /// Negative removes bytes. Computed by the generator from two
    /// footprints; never asserted by an action.
    pub physical_delta: i64,
    pub child_bytes: LogicalBytes,
    /// The experiment this candidate would run.
    pub intended_key: MeasurementKey,
    /// Readings already held of the child's PHYSICAL state, under other
    /// banks, scales or instruments. An escalation is visible here.
    pub prior_observations: Vec<MeasurementKey>,
    /// The PARENT's standing against the contract, where an observation
    /// of it exists. `None` means the parent is unmeasured — a fact, not
    /// a zero.
    ///
    /// Kept whole rather than reduced to a number: the binding margin,
    /// the headroom and which criterion is scarce are all questions a
    /// reader may want, and a scalar would answer none of them.
    pub parent_standing: Option<ConstraintVector>,
}

impl Assessment {
    /// Assess a candidate against facts already held.
    ///
    /// `parent_standing` is supplied by the caller because deriving it
    /// needs the gate, and the gate belongs to the snapshot rather than
    /// to the candidate.
    pub fn of(candidate: &Candidate, parent_standing: Option<ConstraintVector>) -> Self {
        Self {
            action: candidate.action.clone(),
            parent_state: candidate.parent_state.clone(),
            child_state: candidate.child.physical_id().clone(),
            child_realization: candidate.child.realization_id().clone(),
            physical_delta: candidate.physical_delta,
            child_bytes: candidate.child.logical_bytes(),
            intended_key: candidate.intended_key.clone(),
            prior_observations: candidate.prior_observations.clone(),
            parent_standing,
        }
    }

    /// The score under a named rule. Computed here, stored nowhere.
    pub fn score(&self, semantics: &RankingSemantics) -> Score {
        match semantics.rule {
            RankingRule::PhysicalPrizeFirst => Score(self.physical_delta),
        }
    }

    /// Whether this candidate would escalate a state already observed
    /// under some other experimental context.
    pub fn is_escalation(&self) -> bool {
        !self.prior_observations.is_empty()
    }
}
