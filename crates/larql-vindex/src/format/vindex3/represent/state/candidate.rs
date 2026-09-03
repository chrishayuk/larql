//! **What experiments are legitimately available from this state?**
//!
//! That is the whole question. Not *which is promising*, not *which
//! looks sensitive*, not *which the last round liked*. The generator
//! enumerates, resolves, prices, and partitions — and every candidate
//! that disappears carries one sanctioned mechanical reason for having
//! disappeared.
//!
//! ```text
//! realization
//!     ↓ enumerate legal transformations
//!     ↓ resolve the child realization
//!     ↓ derive the physical state
//!     ↓ price it, from the footprint oracle
//!     ↓ apply ONLY registered pre-measurement pruning
//! CandidateSet
//! ```
//!
//! # The three prunes, and the fourth that is not held
//!
//! Ruling 1 states the complete list, and it is deliberately short:
//!
//! ```text
//! 1  identical MeasurementKey observed            dedup
//! 2  not physically better than an available map  physical dominance
//! 3  structurally impossible map                  structural
//! 4  a PROVED monotonicity theorem                NOT CURRENTLY HELD
//! ```
//!
//! [`PreMeasurementPrune`] therefore has **three** variants. The fourth
//! is absent by construction rather than present-and-unused, so adding
//! it later is a visible schema change and not a quiet flag flip.
//!
//! **Behavioural-superset pruning is not on the list and must not slip
//! onto it.** Ruling 1 is explicit: authority refusal attaches to the
//! MEASURED MAP, is not upward-closed under action-set inclusion, and
//! "more low precision" is not a behavioural partial order. The
//! programme holds evidence against it at every level — R4-F7 on sign,
//! R4-F2 on magnitude, R5-F4 on scale (N=3, both directions), R5-F9 on
//! ordering, R5-F7's 2.47× between two states. The line to hold:
//!
//! ```text
//! "this cannot produce a valid physical experiment"   → prune
//! "this probably will not teach us anything"          → NOT a prune
//! ```
//!
//! The second belongs to assessment and ranking, downstream, where it
//! can be argued with. Here it would be indistinguishable from the
//! first.
//!
//! # Nothing disappears silently
//!
//! [`CandidateSet::census`] counts every disposition and
//! [`Census::conserves`] asserts the conservation law:
//!
//! ```text
//! enumerated = eligible + already observed + dominated + structural
//! ```
//!
//! That is what turns "no fifth prune" from a promise into something a
//! test can check — and later, when an agent asks *why aren't we
//! exploring E24*, the answer comes from this partition rather than from
//! a language model reconstructing a rationale.
//!
//! # An action never asserts its own saving
//!
//! R5-F5 read a footprint column as a saving and overstated an expert
//! revert by 3.39×. So no [`MapEdit`] carries a byte figure. The
//! generator applies the edit, resolves the result, asks the
//! [`Footprint`] oracle what the child costs, and computes the delta
//! from two footprints it holds. Dominance then operates on that.

use std::collections::BTreeSet;

use serde::{Deserialize, Serialize};

use super::super::compiler::SourceIdentity;
use super::super::map::PrecisionMap;
use super::super::measurement::EvidenceScale;
use super::action_space::ActionVocabulary;
use super::evidence_bank::EvidenceBankId;
use super::graph::TransitionPolicy;
use super::identity::{RepresentationState, RepresentationStateId};
use super::instrument::InstrumentSemanticsId;
use super::key::{MeasurementKey, MeasurementRegistry};
use super::realization::{LogicalBytes, RealizationId, ResolvedState};
use super::resolved::LayoutAdmission;
use super::surface::TensorSurface;
use super::transition::Action;
use crate::error::VindexError;

/// What a state's logical footprint is.
///
/// Supplied, never derived here. [`super::super::byte_ledger`] states
/// why scopes are supplied rather than inferred from geometry, and
/// inferring a footprint inside a candidate generator would be the same
/// mistake with more leverage.
pub trait Footprint {
    fn logical_bytes(&self, state: &RepresentationState) -> LogicalBytes;
}

/// **The experiment a candidate would run.**
///
/// A state is not measured or unmeasured; it is measured *under a bank,
/// at a scale, by an instrument*. 1c proved the distinction is load
/// bearing — `diagnostic(child)` and `authority(child)` are two
/// experiments on one physical state — so the generator asks about an
/// intent and never about a state alone.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct MeasurementIntent {
    pub bank: EvidenceBankId,
    pub scale: EvidenceScale,
    pub instrument: InstrumentSemanticsId,
}

impl MeasurementIntent {
    pub fn new(
        bank: EvidenceBankId,
        scale: EvidenceScale,
        instrument: InstrumentSemanticsId,
    ) -> Self {
        Self {
            bank,
            scale,
            instrument,
        }
    }

    pub fn key_for(&self, state: &RepresentationStateId) -> MeasurementKey {
        MeasurementKey::new(state, &self.bank, self.scale, &self.instrument)
    }
}

/// A move that may legitimately be measured.
#[derive(Debug, Clone, PartialEq)]
pub struct Candidate {
    /// The realization it departs from — not merely the physical state,
    /// because the action space is a property of the decisions.
    pub parent_realization: RealizationId,
    pub parent_state: RepresentationStateId,
    /// The applied-edit set this candidate would hold.
    pub applied: BTreeSet<String>,
    pub action: Action,
    pub child: ResolvedState,
    /// Computed from two footprints. Negative means bytes were removed.
    pub physical_delta: i64,
    /// The experiment this candidate would run.
    pub intended_key: MeasurementKey,
    /// Readings already held of this PHYSICAL state under other
    /// contexts. Not a duplicate — an escalation, and the reason the
    /// generator reports it rather than pruning on it.
    pub prior_observations: Vec<MeasurementKey>,
}

/// The only reasons a candidate may vanish before it is measured.
#[derive(Debug, Clone, PartialEq)]
pub enum PreMeasurementPrune {
    /// Ruling 1 §1 — this exact experiment is already on the record.
    AlreadyObserved { key: MeasurementKey },
    /// Ruling 1 §2 — the transition policy refuses it on bytes.
    PhysicallyDominated {
        parent_bytes: LogicalBytes,
        child_bytes: LogicalBytes,
    },
    /// Ruling 1 §3 — the move produces no distinct physical experiment,
    /// or no map at all.
    StructurallyInvalid { why: String },
}

impl PreMeasurementPrune {
    /// The registered category, for the census and for reports.
    pub fn category(&self) -> &'static str {
        match self {
            Self::AlreadyObserved { .. } => "already-observed",
            Self::PhysicallyDominated { .. } => "physically-dominated",
            Self::StructurallyInvalid { .. } => "structurally-invalid",
        }
    }
}

/// What became of one enumerated move.
#[derive(Debug, Clone, PartialEq)]
pub enum CandidateDisposition {
    Eligible(Box<Candidate>),
    Pruned {
        action: Action,
        /// Absent when the move produced no map to identify.
        child_state: Option<RepresentationStateId>,
        reason: PreMeasurementPrune,
    },
}

impl CandidateDisposition {
    pub fn action(&self) -> &Action {
        match self {
            Self::Eligible(c) => &c.action,
            Self::Pruned { action, .. } => action,
        }
    }
}

/// Counts, and the conservation law over them.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct Census {
    pub enumerated: usize,
    pub eligible: usize,
    pub already_observed: usize,
    pub physically_dominated: usize,
    pub structurally_invalid: usize,
}

impl Census {
    /// **Nothing disappears silently.**
    pub fn conserves(&self) -> bool {
        self.enumerated
            == self.eligible
                + self.already_observed
                + self.physically_dominated
                + self.structurally_invalid
    }
}

impl std::fmt::Display for Census {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "generated {:>6}\nstructural invalid {:>6}\nphysical dominance {:>6}\n\
             already measured {:>6}\neligible {:>6}",
            self.enumerated,
            self.structurally_invalid,
            self.physically_dominated,
            self.already_observed,
            self.eligible
        )
    }
}

/// Every move available from one state, and what became of each.
#[derive(Debug, Clone, PartialEq, Default)]
pub struct CandidateSet {
    dispositions: Vec<CandidateDisposition>,
}

impl CandidateSet {
    pub fn dispositions(&self) -> &[CandidateDisposition] {
        &self.dispositions
    }

    pub fn eligible(&self) -> impl Iterator<Item = &Candidate> {
        self.dispositions.iter().filter_map(|d| match d {
            CandidateDisposition::Eligible(c) => Some(c.as_ref()),
            CandidateDisposition::Pruned { .. } => None,
        })
    }

    pub fn pruned(&self) -> impl Iterator<Item = (&Action, &PreMeasurementPrune)> {
        self.dispositions.iter().filter_map(|d| match d {
            CandidateDisposition::Pruned { action, reason, .. } => Some((action, reason)),
            CandidateDisposition::Eligible(_) => None,
        })
    }

    pub fn len(&self) -> usize {
        self.dispositions.len()
    }

    pub fn is_empty(&self) -> bool {
        self.dispositions.is_empty()
    }

    pub fn census(&self) -> Census {
        let mut census = Census {
            enumerated: self.dispositions.len(),
            ..Census::default()
        };
        for disposition in &self.dispositions {
            match disposition {
                CandidateDisposition::Eligible(_) => census.eligible += 1,
                CandidateDisposition::Pruned { reason, .. } => match reason {
                    PreMeasurementPrune::AlreadyObserved { .. } => census.already_observed += 1,
                    PreMeasurementPrune::PhysicallyDominated { .. } => {
                        census.physically_dominated += 1
                    }
                    PreMeasurementPrune::StructurallyInvalid { .. } => {
                        census.structurally_invalid += 1
                    }
                },
            }
        }
        census
    }
}

/// **The deterministic candidate generator.**
///
/// Holds only what enumeration and lawful pruning need. It has no
/// ranking input on purpose: there is nothing here for a preference to
/// attach to.
pub struct Generator<'a> {
    pub model: &'a SourceIdentity,
    pub surface: &'a TensorSurface,
    /// The map every applied set is layered onto.
    pub base_map: &'a PrecisionMap,
    pub vocabulary: &'a ActionVocabulary,
    pub layout: &'a dyn LayoutAdmission,
    pub footprint: &'a dyn Footprint,
    pub policy: TransitionPolicy,
    pub measurements: &'a MeasurementRegistry,
}

impl Generator<'_> {
    /// Resolve one applied set into a priced realization.
    pub fn realize(&self, applied: &BTreeSet<String>) -> Result<ResolvedState, VindexError> {
        let map = self.vocabulary.map_for(self.base_map, applied)?;
        let state = RepresentationState::resolve(self.model, self.surface, &map, self.layout);
        let bytes = self.footprint.logical_bytes(&state);
        Ok(ResolvedState::new(state, bytes))
    }

    /// **Every move available from `applied`, partitioned.**
    ///
    /// The universe is rung 5's frame: every unapplied edit as an
    /// addition, and every (applied, unapplied) pair as a 1-out/1-in
    /// exchange. Enumeration is over the whole vocabulary and never over
    /// the last round's leftovers — R5-F6.
    pub fn candidates(
        &self,
        applied: &BTreeSet<String>,
        intent: &MeasurementIntent,
    ) -> Result<CandidateSet, VindexError> {
        let parent = self.realize(applied)?;
        let unapplied: Vec<&str> = self
            .vocabulary
            .names()
            .filter(|n| !applied.contains(*n))
            .collect();

        let mut moves: Vec<(Action, BTreeSet<String>)> = Vec::new();
        for add in &unapplied {
            let mut next = applied.clone();
            next.insert((*add).to_string());
            moves.push((Action::new(format!("+{add}")).adding([*add]), next));
        }
        for remove in applied {
            for add in &unapplied {
                let mut next = applied.clone();
                next.remove(remove);
                next.insert((*add).to_string());
                moves.push((
                    Action::new(format!("−{remove} +{add}"))
                        .removing([remove.as_str()])
                        .adding([*add]),
                    next,
                ));
            }
        }

        let mut dispositions = Vec::with_capacity(moves.len());
        for (action, next) in moves {
            dispositions.push(self.dispose(&parent, action, next, intent)?);
        }
        Ok(CandidateSet { dispositions })
    }

    /// Resolve one move and decide its disposition.
    fn dispose(
        &self,
        parent: &ResolvedState,
        action: Action,
        applied: BTreeSet<String>,
        intent: &MeasurementIntent,
    ) -> Result<CandidateDisposition, VindexError> {
        let child = self.realize(&applied)?;
        let child_state = child.physical_id().clone();

        // §3 structural. A move that changes no decision at all produces
        // no distinct experiment — there is nothing for an instrument to
        // observe that the parent has not already shown.
        if child.realization_id() == parent.realization_id() {
            return Ok(CandidateDisposition::Pruned {
                action,
                child_state: Some(child_state),
                reason: PreMeasurementPrune::StructurallyInvalid {
                    why: "the move changes no resolved decision".into(),
                },
            });
        }

        // §2 physical dominance, delegated to the transition policy so
        // the generator and the graph cannot disagree about what an
        // admissible edge is.
        let physical_delta = child.logical_bytes().delta_from(parent.logical_bytes());
        if !self.policy.admits(physical_delta) {
            return Ok(CandidateDisposition::Pruned {
                action,
                child_state: Some(child_state),
                reason: PreMeasurementPrune::PhysicallyDominated {
                    parent_bytes: parent.logical_bytes(),
                    child_bytes: child.logical_bytes(),
                },
            });
        }

        // §1 dedup, on the INTENDED experiment and never on the state
        // alone: escalating a measured state from diagnostic to
        // authority is the ladder, not a repeat.
        let intended_key = intent.key_for(&child_state);
        if self.measurements.contains(&intended_key) {
            return Ok(CandidateDisposition::Pruned {
                action,
                child_state: Some(child_state),
                reason: PreMeasurementPrune::AlreadyObserved { key: intended_key },
            });
        }

        let prior_observations: Vec<MeasurementKey> = self
            .measurements
            .of_state(&child_state)
            .map(|(k, _)| k.clone())
            .collect();

        Ok(CandidateDisposition::Eligible(Box::new(Candidate {
            parent_realization: parent.realization_id().clone(),
            parent_state: parent.physical_id().clone(),
            applied,
            action,
            child,
            physical_delta,
            intended_key,
            prior_observations,
        })))
    }
}
