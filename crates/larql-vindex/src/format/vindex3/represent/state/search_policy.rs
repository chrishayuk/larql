//! **Which unresolved experiment has the highest priority?**
//!
//! That question, and not *which candidate can replace the incumbent* —
//! those are different, and a candidate can rank first for measurement
//! precisely because it is uncertain while being nowhere near
//! promotable. So:
//!
//! ```text
//! CandidateSet → Assessment → BestFirst        experiment selection
//! Measurements → SearchEvidence → decide_promotion   admissibility
//! ```
//!
//! `decide_promotion` stays downstream and unweakened. It already
//! refuses to scalarise disagreeing proxies, and making ranking easier
//! is not a reason to relax that.
//!
//! # Search orders states; measurement orders experiments
//!
//! Two realizations of one physical state have different future action
//! spaces, so both belong on the frontier. Their *immediate experiment*
//! can still be the same one:
//!
//! ```text
//! A --action x--> C (realization r1) ┐
//!                                    ├─ one MeasurementKey
//! B --action y--> C (realization r2) ┘
//!
//! search:       r1 ≠ r2, both kept
//! measurement:  one experiment, run once
//! ```
//!
//! [`MeasurementOpportunity`] is that grouping, and it is the payoff of
//! the whole 1a-1c identity model: when the observation lands, **both**
//! realizations inherit it, because 1c keys evidence on the physical
//! state while 1b keeps the realizations apart.
//!
//! # Ruling 3
//!
//! ```text
//! 0 eligible  → EXHAUSTED
//! 1 eligible  → SELECT it; the diagnostic is RECORDED and CANNOT VETO
//! > 1         → the registered rule chooses which is measured FIRST
//! ```
//!
//! The middle line is a ruling, not an optimisation: with one
//! opportunity "what next?" is already answered, and an escalation rule
//! that consulted a diagnostic there would be promoting the diagnostic
//! into an admissibility screen — the accident that was withdrawn.
//!
//! # What this stage does not do
//!
//! It does not order the *authority escalation* of
//! diagnostically-measured candidates, and it does not run promotion.
//! Both go through `CandidateAssessment`, which needs two facts a
//! snapshot does not yet hold — a per-state [`ByteLedger`] and an
//! `ExecutionCostModel`. Those are inputs to add, not conclusions to
//! invent, and until they exist this layer says so rather than
//! approximating them.
//!
//! [`ByteLedger`]: super::super::byte_ledger::ByteLedger

use std::collections::BTreeMap;

use serde::{Deserialize, Serialize};

use super::assess::{Assessment, ParentStanding, RankingSemantics};
use super::candidate::CandidateSet;
use super::identity::RepresentationStateId;
use super::key::MeasurementKey;

/// One experiment, and every eligible candidate that would run it.
#[derive(Debug, Clone, PartialEq)]
pub struct MeasurementOpportunity {
    /// The experiment. One key, one run.
    pub key: MeasurementKey,
    pub state: RepresentationStateId,
    /// Every route to it, in the policy's order. More than one is
    /// normal: two realizations can reach one physical state.
    pub candidates: Vec<Assessment>,
}

impl MeasurementOpportunity {
    /// The best physical prize among the routes to this experiment.
    ///
    /// Most negative wins, matching the sign convention. The routes
    /// differ in how they were reached and not in what the experiment
    /// costs, so the opportunity takes the best of them.
    pub fn physical_delta(&self) -> i64 {
        self.candidates
            .iter()
            .map(|a| a.physical_delta)
            .min()
            .expect("an opportunity holds at least one candidate")
    }

    /// The route the policy ranked first.
    pub fn leading(&self) -> &Assessment {
        self.candidates
            .first()
            .expect("an opportunity holds at least one candidate")
    }

    pub fn routes(&self) -> usize {
        self.candidates.len()
    }
}

/// What the policy decided to do next.
#[derive(Debug, Clone, PartialEq)]
pub enum Selection {
    /// Nothing eligible. The neighbourhood is closed.
    Exhausted,
    /// Ruling 3's middle line: exactly one opportunity, so there is
    /// nothing to rank and it is selected.
    Sole(Box<MeasurementOpportunity>),
    /// More than one, ordered by the registered rule.
    Ranked {
        chosen: Box<MeasurementOpportunity>,
        considered: usize,
    },
}

/// What a [`Selection`] IS, without its payload.
///
/// The allocation layer downstream decides what to buy from the SHAPE of
/// a selection and never from the opportunity inside it. Naming that
/// keeps `allocation` independent of how an opportunity is built, so it
/// can be reasoned about — and tested — without resolving a state.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum SelectionShape {
    /// Nothing eligible.
    Exhausted,
    /// Exactly one, so there is nothing to rank.
    Sole,
    /// More than one, ordered by the registered rule.
    Ranked,
}

impl Selection {
    /// This selection's shape, discarding the payload.
    pub fn shape(&self) -> SelectionShape {
        match self {
            Self::Exhausted => SelectionShape::Exhausted,
            Self::Sole(_) => SelectionShape::Sole,
            Self::Ranked { .. } => SelectionShape::Ranked,
        }
    }

    pub fn opportunity(&self) -> Option<&MeasurementOpportunity> {
        match self {
            Self::Exhausted => None,
            Self::Sole(o) => Some(o),
            Self::Ranked { chosen, .. } => Some(chosen),
        }
    }

    pub fn is_exhausted(&self) -> bool {
        matches!(self, Self::Exhausted)
    }
}

/// **The policy.** Deliberately small: identity, comparability,
/// admissibility, deduplication and physical accounting were all settled
/// beneath it, so what is left is an ordering.
#[derive(Debug, Clone, PartialEq)]
pub struct BestFirst {
    pub semantics: RankingSemantics,
}

impl BestFirst {
    pub fn new(semantics: RankingSemantics) -> Self {
        Self { semantics }
    }

    /// Assess every eligible candidate in `set`.
    ///
    /// `standing` answers what the parent's contract standing is, where
    /// an observation of it exists; the policy does not reach for a gate
    /// itself.
    pub fn assess(&self, set: &CandidateSet, standing: &dyn ParentStanding) -> Vec<Assessment> {
        set.eligible()
            .map(|c| Assessment::of(c, standing.of(&c.parent_state)))
            .collect()
    }

    /// **The total order, applied.**
    ///
    /// The chain is [`RankingSemantics::tie_break_chain`], and every
    /// element after the first is there so no answer depends on the
    /// order candidates arrived in.
    pub fn order(&self, mut assessments: Vec<Assessment>) -> Vec<Assessment> {
        assessments.sort_by(|a, b| self.compare(a, b));
        assessments
    }

    /// How two candidates compare. `Less` means `a` is measured first.
    pub fn compare(&self, a: &Assessment, b: &Assessment) -> std::cmp::Ordering {
        a.score(&self.semantics)
            .cmp(&b.score(&self.semantics))
            .then_with(|| a.physical_delta.cmp(&b.physical_delta))
            .then_with(|| a.child_state.as_str().cmp(b.child_state.as_str()))
            .then_with(|| {
                a.child_realization
                    .as_str()
                    .cmp(b.child_realization.as_str())
            })
            .then_with(|| a.action.identity().cmp(&b.action.identity()))
    }

    /// Group ordered candidates into experiments.
    ///
    /// One [`MeasurementKey`] is one opportunity however many routes
    /// reach it, so an experiment is never scheduled twice in a round.
    pub fn opportunities(&self, ordered: Vec<Assessment>) -> Vec<MeasurementOpportunity> {
        let mut by_key: BTreeMap<MeasurementKey, MeasurementOpportunity> = BTreeMap::new();
        for assessment in ordered {
            by_key
                .entry(assessment.intended_key.clone())
                .or_insert_with(|| MeasurementOpportunity {
                    key: assessment.intended_key.clone(),
                    state: assessment.child_state.clone(),
                    candidates: Vec::new(),
                })
                .candidates
                .push(assessment);
        }
        let mut opportunities: Vec<MeasurementOpportunity> = by_key.into_values().collect();
        // Grouping went through a map keyed by digest, so the order it
        // yields is a property of the digests and not of the rule. Sort
        // the opportunities by their leading route, which is.
        opportunities.sort_by(|a, b| self.compare(a.leading(), b.leading()));
        opportunities
    }

    /// **What to measure next**, from a candidate set.
    pub fn select(&self, set: &CandidateSet, standing: &dyn ParentStanding) -> Selection {
        let ordered = self.order(self.assess(set, standing));
        let mut opportunities = self.opportunities(ordered);
        match opportunities.len() {
            0 => Selection::Exhausted,
            1 => Selection::Sole(Box::new(opportunities.remove(0))),
            considered => Selection::Ranked {
                chosen: Box::new(opportunities.remove(0)),
                considered,
            },
        }
    }
}
