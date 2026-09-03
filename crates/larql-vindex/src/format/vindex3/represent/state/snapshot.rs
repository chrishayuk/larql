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

use std::collections::BTreeSet;

use super::super::assessment::{CandidateAssessment, EvidenceContext};
use super::super::byte_ledger::ByteLedger;
use super::super::constraint::{ConstraintVector, Margin};
use super::super::decision::SearchCandidate;
use super::super::diagnostic::{DiagnosticPolicy, DiagnosticVector};
use super::super::execution_cost::{CostRefusal, ExecutionCostModel};
use super::super::map::PrecisionMap;
use super::super::measurement::{EvidenceScale, TailSupportPolicy};
use super::super::participation::ParticipationDeclaration;
use super::super::promotion::PromotionCandidate;
use super::super::quality::QualityBank;
use super::super::quality::QualityGate;
use super::super::search_evidence::SearchCalibrationRegistry;
use super::action_space::ActionVocabulary;
use super::assess::{ParentStanding, RankingSemantics};
use super::candidate::{Footprint, Generator, MeasurementIntent};
use super::graph::RepresentationStateGraph;
use super::identity::RepresentationStateId;
use super::key::{MeasurementKey, MeasurementRegistry};
use super::realization::LogicalBytes;
use super::resolved::LayoutAdmission;
use super::search_policy::{BestFirst, Selection};
use super::semantics::{SearchSemantics, SearchSemanticsId};
use super::surface::TensorSurface;
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

/// **What the search is over**: the model's surface and the moves that
/// exist at all.
///
/// The vocabulary belongs here and not in the policy: R5-F6 was a
/// vocabulary failure, not a ranking failure, and no policy can recover
/// a state whose move was never declared.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SearchSpace {
    pub surface: TensorSurface,
    /// The map every applied set is layered onto.
    pub base_map: PrecisionMap,
    pub vocabulary: ActionVocabulary,
}

/// **How facts become conclusions.** Every field is a rule or a
/// threshold; none is an outcome.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SearchConfig {
    pub objective: Objective,
    /// The frozen behavioural contract conclusions are drawn against.
    pub gate: QualityGate,
    pub tail_support: TailSupportPolicy,
    /// What this programme has learned about its own instruments — the
    /// `SearchEvidence` ladder's registrations.
    pub calibrations: SearchCalibrationRegistry,
    /// Which statistics a diagnostic reads, and for what purpose.
    pub diagnostic_policy: DiagnosticPolicy,
    /// The rules the original conclusions were drawn under, so a later
    /// replay can tell a changed procedure from changed data.
    pub semantics: SearchSemantics,
    pub ranking: RankingSemantics,
}

/// **What has been observed, and what it costs.**
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SearchFacts {
    pub graph: RepresentationStateGraph,
    pub measurements: MeasurementRegistry,
    /// Per-token reads, per state. NOT [`LogicalBytes`], which is a
    /// whole-map footprint — the two are different quantities and the
    /// newtype exists to keep them apart.
    pub byte_ledgers: BTreeMap<RepresentationStateId, ByteLedger>,
    /// Measured execution observations, each carrying its machine,
    /// device, backend and compiler commit. The COST is not stored:
    /// `ExecutionCostModel::predict` derives it, and its `status()`
    /// refuses to call the model calibrated until beta has been shown
    /// across separated breadths.
    pub execution_cost: ExecutionCostModel,
}

/// **The persisted scientific state of a search.**
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SearchSnapshot {
    schema: String,
    space: SearchSpace,
    config: SearchConfig,
    facts: SearchFacts,
}

impl SearchSnapshot {
    pub fn new(space: SearchSpace, config: SearchConfig, facts: SearchFacts) -> Self {
        Self {
            schema: SNAPSHOT_SCHEMA.into(),
            space,
            config,
            facts,
        }
    }

    pub fn space(&self) -> &SearchSpace {
        &self.space
    }

    pub fn config(&self) -> &SearchConfig {
        &self.config
    }

    pub fn facts(&self) -> &SearchFacts {
        &self.facts
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
        self.config.objective
    }

    pub fn gate(&self) -> &QualityGate {
        &self.config.gate
    }

    pub fn tail_support(&self) -> &TailSupportPolicy {
        &self.config.tail_support
    }

    pub fn semantics(&self) -> &SearchSemantics {
        &self.config.semantics
    }

    /// The rules this snapshot's conclusions were originally drawn
    /// under. A replay whose own semantics differ is answering a
    /// different question, and this is what lets it say so.
    pub fn semantics_id(&self) -> SearchSemanticsId {
        self.config.semantics.id()
    }

    pub fn graph(&self) -> &RepresentationStateGraph {
        &self.facts.graph
    }

    pub fn measurements(&self) -> &MeasurementRegistry {
        &self.facts.measurements
    }

    /// The per-token ledger for one state, where it is held.
    pub fn ledger(&self, state: &RepresentationStateId) -> Option<&ByteLedger> {
        self.facts.byte_ledgers.get(state)
    }

    // ---------------------------------------------------------- derived

    /// **The verdict on one observation**, computed from the stored
    /// reading and the stored gate. `None` when no such experiment was
    /// run — a miss, which is a fact about the record and not a failure.
    pub fn adjudicate(&self, key: &MeasurementKey) -> Option<Adjudication> {
        let observation = self.facts.measurements.get(key)?;
        Some(Adjudication {
            key: key.clone(),
            constraints: ConstraintVector::of(&self.config.gate, observation),
        })
    }

    /// **The frontier, recomputed.** One entry per state the graph
    /// holds, in state-id order — a stated order, never insertion order.
    pub fn frontier(&self) -> Vec<FrontierEntry> {
        let mut by_state: BTreeMap<&RepresentationStateId, Vec<Adjudication>> = BTreeMap::new();
        for node in self.facts.graph.nodes() {
            by_state.entry(node.physical_id()).or_default();
        }
        for (key, observation) in self.facts.measurements.keys().filter_map(|k| {
            self.facts
                .measurements
                .get(k)
                .map(|observation| (k, observation))
        }) {
            // A measurement of a state this graph does not hold belongs
            // to another search; it is not this frontier's business.
            if let Some(entry) = by_state.get_mut(key.state()) {
                entry.push(Adjudication {
                    key: key.clone(),
                    constraints: ConstraintVector::of(&self.config.gate, observation),
                });
            }
        }
        by_state
            .into_iter()
            .map(|(state, adjudications)| FrontierEntry {
                state: state.clone(),
                logical_bytes: self
                    .facts
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

    // ------------------------------------------------ the derivation chain

    /// **The action space, from stored facts.**
    ///
    /// `layout` and `footprint` are supplied because they are CODE — a
    /// layout rule and a pricing routine — not data. Everything the
    /// generator reads that is a fact or a rule comes from the snapshot:
    /// the model from the graph, the surface, base map and vocabulary
    /// from the space, the transition policy from the graph, and the
    /// measurement registry from the facts.
    pub fn generator<'a>(
        &'a self,
        layout: &'a dyn LayoutAdmission,
        footprint: &'a dyn Footprint,
    ) -> Generator<'a> {
        Generator {
            model: self.facts.graph.model(),
            surface: &self.space.surface,
            base_map: &self.space.base_map,
            vocabulary: &self.space.vocabulary,
            layout,
            footprint,
            policy: self.facts.graph.policy(),
            measurements: &self.facts.measurements,
        }
    }

    /// The ordering policy this snapshot's conclusions are drawn under.
    pub fn best_first(&self) -> BestFirst {
        BestFirst::new(self.config.ranking.clone())
    }

    /// **What to measure next**, derived end to end from stored facts.
    pub fn next_experiment(
        &self,
        applied: &BTreeSet<String>,
        intent: &MeasurementIntent,
        layout: &dyn LayoutAdmission,
        footprint: &dyn Footprint,
    ) -> Result<Selection, VindexError> {
        let set = self
            .generator(layout, footprint)
            .candidates(applied, intent)?;
        Ok(self.best_first().select(&set, self))
    }

    /// **Promotion, derived from stored facts.**
    ///
    /// Builds one [`SearchCandidate`] per measured move and hands the set
    /// to [`decide_promotion`], which is not reimplemented here: it
    /// already refuses to scalarise disagreeing proxies, and making this
    /// convenient is not a reason to weaken it.
    ///
    /// **Promotion reads the graph's EDGES, not a candidate set.**
    ///
    /// A promotion candidate is a move that has been built and measured;
    /// the generator prunes exactly those as `AlreadyObserved`, because
    /// its question is what to try NEXT. Feeding it eligible candidates
    /// would ask which unmeasured move should replace the incumbent,
    /// which is not a question any evidence can answer.
    ///
    /// A move is only a promotion candidate when BOTH ends carry a
    /// reading at `scale` and both carry a per-token ledger — the
    /// marginal quantities are a difference between two predictions, and
    /// a difference with one end missing is not a smaller number, it is
    /// no number. Such a move is skipped, not defaulted.
    pub fn promotion_candidates(
        &self,
        scale: EvidenceScale,
    ) -> Result<Vec<SearchCandidate>, CostRefusal> {
        let ctx = EvidenceContext {
            scale,
            registry: self.config.calibrations.clone(),
            tail_policy: self.config.tail_support.clone(),
        };
        let mut candidates = Vec::new();
        for edge in self.facts.graph.edges() {
            let (Some(parent_bank), Some(child_bank)) = (
                self.reading_of(edge.parent(), scale),
                self.reading_of(edge.child(), scale),
            ) else {
                continue;
            };
            let (Some(parent_ledger), Some(child_ledger)) =
                (self.ledger(edge.parent()), self.ledger(edge.child()))
            else {
                continue;
            };
            let assessment = CandidateAssessment::of(
                &ctx,
                &self.facts.execution_cost,
                parent_ledger,
                child_ledger,
                ConstraintVector::of(&self.config.gate, parent_bank),
                ConstraintVector::of(&self.config.gate, child_bank),
            )?;
            candidates.push(SearchCandidate {
                id: edge.action().label.clone(),
                // No proxies: a proxy observation is a registered
                // finding about an instrument, and none has been
                // recorded for these statistics. An invented one would
                // be exactly the magnitude claim ROUTE-CAL-1 refused.
                promotion: PromotionCandidate::new(assessment, Vec::new()),
                diagnostic: DiagnosticVector::of(&self.config.diagnostic_policy, child_bank),
                participation: ParticipationDeclaration::all_affected(),
            });
        }
        Ok(candidates)
    }

    /// The reading held of one state at one scale, under any bank or
    /// instrument this snapshot holds.
    fn reading_of<'a>(
        &'a self,
        state: &'a RepresentationStateId,
        scale: EvidenceScale,
    ) -> Option<&'a QualityBank> {
        self.facts
            .measurements
            .of_state(state)
            .find(|(k, _)| k.scale() == scale)
            .map(|(_, bank)| bank)
    }
}
