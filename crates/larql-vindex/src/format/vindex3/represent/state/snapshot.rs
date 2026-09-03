//! **The snapshot holds facts and configuration. Every conclusion is
//! derived.**
//!
//! ```text
//! STORED — facts and configuration
//!     schema, objective, gate, tail-support policy, search semantics
//!     the state graph      (which states exist, how they were reached)
//!     the measurements     (what was observed of them)
//!
//! NEVER STORED — conclusions
//!     admissible / refused        chosen candidate      candidate rank
//!     binding constraint          the frontier          "best map"
//!     promotion decision          an agent's recommendation
//! ```
//!
//! The invariant, and the reason this stage exists:
//!
//! > **Delete every derived conclusion, deserialise the factual state,
//! > run the deterministic optimiser, and recover the same conclusion.**
//!
//! That is what removes the experiment ledger — and an operator's
//! memory — as the authority for a search. A snapshot that stored its
//! own verdicts would prove serialisation and nothing else.
//!
//! # The frontier is a projection, not a record
//!
//! A stored frontier is a second authority that can drift from the graph
//! and measurements it was computed from. [`SearchSnapshot::frontier`]
//! recomputes it every time. Caching is a later optimisation; the replay
//! gate must pass without one.
//!
//! Likewise `admissible`, `sound` and `binding`: all three are
//! [`ConstraintVector`] questions about one observation and one gate, and
//! all three are recomputed.
//!
//! # Semantic replay, not implementation replay
//!
//! What must reproduce is the eligible set and the conclusions, not the
//! order a `BTreeMap` happened to yield. Swapping a container must not
//! invalidate a scientific record. So the derived views return sets and
//! adjudications keyed by identity, and where an order is exposed it is
//! by a stated key — never by insertion.
//!
//! # What 1d does NOT replay, and why that is principled
//!
//! [`decide_promotion`] takes a `SearchCandidate`, which carries a
//! `PromotionCandidate` holding an `assessment.ranking_score` — a
//! CONCLUSION. Persisting one to make promotion replayable would be
//! exactly the cheat this stage exists to forbid. Re-deriving it instead
//! needs the assessment layer wired to the graph, which is the candidate
//! generator's job at stage 2. Until then a snapshot replays the
//! CONTRACT chain — margins, binding constraint, admissibility, refusal
//! and its reasons — and does not claim to replay promotion ordering.
//!
//! [`decide_promotion`]: super::super::decision::decide_promotion

use std::collections::BTreeMap;

use serde::{Deserialize, Serialize};

use super::super::constraint::{ConstraintVector, Margin};
use super::super::measurement::{EvidenceScale, TailSupportPolicy};
use super::super::quality::QualityGate;
use super::assess::ParentStanding;
use super::graph::RepresentationStateGraph;
use super::identity::RepresentationStateId;
use super::key::{MeasurementKey, MeasurementRegistry};
use super::realization::LogicalBytes;
use super::semantics::{SearchSemantics, SearchSemanticsId};
use crate::error::VindexError;

/// The snapshot format version.
pub const SNAPSHOT_SCHEMA: &str = "represent-search-snapshot/v1";

/// What the search is trying to do.
///
/// One variant, because one objective has actually been searched
/// against. Measured throughput joins when residency enters the state
/// and the transition policy stops being monotone in bytes; inventing
/// the variant now would be inventing the accounting behind it.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum Objective {
    /// Fewest logical bytes that still satisfies the contract.
    MinimiseLogicalBytes,
}

/// **A derived verdict on one observation.** Never stored.
#[derive(Debug, Clone, PartialEq)]
pub struct Adjudication {
    key: MeasurementKey,
    constraints: ConstraintVector,
}

impl Adjudication {
    /// Which experiment this is a verdict on.
    pub fn key(&self) -> &MeasurementKey {
        &self.key
    }

    pub fn constraints(&self) -> &ConstraintVector {
        &self.constraints
    }

    /// Whether every criterion — ceiling and floor — is met.
    pub fn admissible(&self) -> bool {
        self.constraints.admissible()
    }

    /// Whether the measurement itself is sound: every floor cleared. A
    /// blind instrument passes every ceiling perfectly.
    pub fn sound(&self) -> bool {
        self.constraints.sound()
    }

    /// The scarce resource — the ceiling closest to its limit.
    pub fn binding(&self) -> Option<&Margin> {
        self.constraints.binding()
    }

    /// Every criterion this observation failed, in the gate's own order.
    pub fn failures(&self) -> Vec<&Margin> {
        self.constraints
            .margins
            .iter()
            .filter(|m| !m.satisfied())
            .collect()
    }
}

impl ParentStanding for SearchSnapshot {
    /// A state's standing at AUTHORITY scale, where such a reading
    /// exists.
    ///
    /// Authority and not "whatever was measured": a diagnostic reading
    /// prices nothing against the contract, and a policy handed one as
    /// though it did would be reading a diagnostic as authority — the
    /// inference R5-F4 and R5-F9 closed.
    fn of(&self, state: &RepresentationStateId) -> Option<ConstraintVector> {
        self.frontier()
            .into_iter()
            .find(|e| &e.state == state)?
            .adjudications
            .into_iter()
            .find(|a| a.key().scale() == EvidenceScale::Authority)
            .map(|a| a.constraints().clone())
    }
}

/// One state's standing, derived.
#[derive(Debug, Clone, PartialEq)]
pub struct FrontierEntry {
    pub state: RepresentationStateId,
    pub logical_bytes: LogicalBytes,
    /// Every adjudication held for this state, one per observation.
    pub adjudications: Vec<Adjudication>,
}

impl FrontierEntry {
    /// Admitted only on an AUTHORITY reading. A diagnostic reading that
    /// passes is not an admission — the whole ladder rests on that.
    pub fn admitted(&self) -> bool {
        self.adjudications
            .iter()
            .any(|a| a.key.scale() == EvidenceScale::Authority && a.admissible())
    }

    /// Refused where an authority reading failed the contract.
    pub fn refused(&self) -> bool {
        self.adjudications
            .iter()
            .any(|a| a.key.scale() == EvidenceScale::Authority && !a.admissible())
    }

    pub fn measured_at(&self, scale: EvidenceScale) -> bool {
        self.adjudications.iter().any(|a| a.key.scale() == scale)
    }
}

/// **The persisted scientific state of a search.**
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SearchSnapshot {
    schema: String,
    objective: Objective,
    /// The frozen behavioural contract conclusions are drawn against.
    gate: QualityGate,
    tail_support: TailSupportPolicy,
    /// The rules the original conclusions were drawn under, so a later
    /// replay can tell a changed procedure from changed data.
    semantics: SearchSemantics,
    graph: RepresentationStateGraph,
    measurements: MeasurementRegistry,
}

impl SearchSnapshot {
    pub fn new(
        objective: Objective,
        gate: QualityGate,
        tail_support: TailSupportPolicy,
        semantics: SearchSemantics,
        graph: RepresentationStateGraph,
        measurements: MeasurementRegistry,
    ) -> Self {
        Self {
            schema: SNAPSHOT_SCHEMA.into(),
            objective,
            gate,
            tail_support,
            semantics,
            graph,
            measurements,
        }
    }

    /// Refuse a snapshot written under another schema.
    pub fn check_schema(&self) -> Result<(), VindexError> {
        if self.schema != SNAPSHOT_SCHEMA {
            return Err(VindexError::Parse(format!(
                "snapshot schema is `{}` but this build reads `{SNAPSHOT_SCHEMA}` — a stored \
                 search state whose canonical forms have moved must be recognisably stale, \
                 not silently reinterpreted",
                self.schema
            )));
        }
        Ok(())
    }

    pub fn schema(&self) -> &str {
        &self.schema
    }

    pub fn objective(&self) -> Objective {
        self.objective
    }

    pub fn gate(&self) -> &QualityGate {
        &self.gate
    }

    pub fn tail_support(&self) -> &TailSupportPolicy {
        &self.tail_support
    }

    pub fn semantics(&self) -> &SearchSemantics {
        &self.semantics
    }

    /// The rules this snapshot's conclusions were originally drawn
    /// under. A replay whose own semantics differ is answering a
    /// different question, and this is what lets it say so.
    pub fn semantics_id(&self) -> SearchSemanticsId {
        self.semantics.id()
    }

    pub fn graph(&self) -> &RepresentationStateGraph {
        &self.graph
    }

    pub fn measurements(&self) -> &MeasurementRegistry {
        &self.measurements
    }

    // ---------------------------------------------------------- derived

    /// **The verdict on one observation**, computed from the stored
    /// reading and the stored gate. `None` when no such experiment was
    /// run — a miss, which is a fact about the record and not a failure.
    pub fn adjudicate(&self, key: &MeasurementKey) -> Option<Adjudication> {
        let observation = self.measurements.get(key)?;
        Some(Adjudication {
            key: key.clone(),
            constraints: ConstraintVector::of(&self.gate, observation),
        })
    }

    /// **The frontier, recomputed.** One entry per state the graph
    /// holds, in state-id order — a stated order, never insertion order.
    pub fn frontier(&self) -> Vec<FrontierEntry> {
        let mut by_state: BTreeMap<&RepresentationStateId, Vec<Adjudication>> = BTreeMap::new();
        for node in self.graph.nodes() {
            by_state.entry(node.physical_id()).or_default();
        }
        for (key, observation) in self
            .measurements
            .keys()
            .filter_map(|k| self.measurements.get(k).map(|observation| (k, observation)))
        {
            // A measurement of a state this graph does not hold belongs
            // to another search; it is not this frontier's business.
            if let Some(entry) = by_state.get_mut(key.state()) {
                entry.push(Adjudication {
                    key: key.clone(),
                    constraints: ConstraintVector::of(&self.gate, observation),
                });
            }
        }
        by_state
            .into_iter()
            .map(|(state, adjudications)| FrontierEntry {
                state: state.clone(),
                logical_bytes: self
                    .graph
                    .node(state)
                    .expect("every key came from the graph")
                    .logical_bytes(),
                adjudications,
            })
            .collect()
    }

    /// States with an authority reading that satisfies the contract,
    /// **cheapest first** — the objective, applied.
    pub fn admitted(&self) -> Vec<FrontierEntry> {
        let mut admitted: Vec<FrontierEntry> = self
            .frontier()
            .into_iter()
            .filter(|e| e.admitted())
            .collect();
        admitted.sort_by(|a, b| {
            a.logical_bytes
                .cmp(&b.logical_bytes)
                .then_with(|| a.state.as_str().cmp(b.state.as_str()))
        });
        admitted
    }

    /// States the graph holds that carry no reading at `scale`.
    ///
    /// A fact query, deliberately unordered by desirability: WHICH of
    /// these to spend an authority run on is a search policy's decision
    /// and this stage has no policy.
    pub fn unmeasured_at(&self, scale: EvidenceScale) -> Vec<RepresentationStateId> {
        self.frontier()
            .into_iter()
            .filter(|e| !e.measured_at(scale))
            .map(|e| e.state)
            .collect()
    }
}
