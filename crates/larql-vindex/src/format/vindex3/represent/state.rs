//! **The search state: what a representation IS, independent of how it
//! was reached.**
//!
//! [`super::map::PrecisionMap`] is a *policy* — a default encoding, the
//! roles it applies to, and ordered exceptions. That is the right shape
//! for a program someone writes and a container carries, and it is the
//! wrong shape for a search graph's identity, in both directions:
//!
//! ```text
//! two different rule sets, identical resolved decisions   SAME state
//! one rule set, two models                                DIFFERENT states
//! ```
//!
//! A digest over the rule text gets both wrong. It splits a state that a
//! redundant or shadowed exception wrote differently, and it merges two
//! models that happen to share a policy. Splitting is the expensive
//! failure: the search re-measures a state it has already refused, and
//! every path that reaches a map by a different route looks novel. The
//! exchange search already reached the same final map by different
//! histories, which is what makes this the first thing to settle.
//!
//! # The invariant
//!
//! > **If two maps cause VINDEX3 to present exactly the same
//! > representation decision for every tensor of the same model surface,
//! > they are the same search state.**
//!
//! Everything in this module is in service of that sentence, and the
//! boundaries follow from it:
//!
//! ```text
//! IN   model identity          the same map on two models is two states
//! IN   tensor surface          adding, removing or reshaping a tensor
//!                              is a different model surface even when
//!                              every surviving decision is unchanged
//! IN   effective decisions     what the container will actually present
//!
//! OUT  rule syntax             ordering that changes no decision,
//!                              shadowed exceptions, redundant defaults,
//!                              the map's `name`, the recipe that built it
//! OUT  evidence                a measurement DESCRIBES a state; it is
//!                              not part of one
//! OUT  search history          different paths to the same decisions
//!                              converge to one node — that is the point
//! OUT  the behavioural contract  a representation is the same
//!                              representation when evaluated under
//!                              another contract. Contract, bank and
//!                              execution belong to the MEASUREMENT key,
//!                              not to the state.
//! ```
//!
//! That last exclusion is the one worth defending. Folding
//! `balanced-v1` into the digest would identify an *experimental
//! context* rather than a representation, and the same physical bytes
//! measured under a second contract would arrive as an unrelated state
//! with no shared physical accounting.
//!
//! # Effective, not declared
//!
//! A map may say "compile this" and the storage layout refuse it:
//! NVFP4 stores 2-D matrices whose `k` is a multiple of 16, and
//! `represent`'s compiler carries anything else verbatim whatever the
//! policy said. So the resolved decision is what the layout will
//! actually admit, not what the rules assert — see
//! [`resolved::ResolvedEncoding`].
//!
//! A layout-refused tensor and a protected tensor present the same
//! bytes, so they are the same *state*; they are different *facts*, and
//! the decision vector keeps them apart for reports and for action
//! generation (unprotecting is a move, un-refusing is not). Identity
//! collapses them; explanation does not.

pub mod accounting;
pub mod action_space;
pub mod assess;
pub mod bind;
pub mod candidate;
pub mod evidence_bank;
pub mod footprint;
pub mod graph;
pub mod identity;
pub mod instrument;
pub mod key;
pub mod realization;
pub mod resolved;
pub mod search_policy;
pub mod semantics;
pub mod snapshot;
pub mod surface;
pub mod transition;

pub use accounting::{
    read_source_storage, PhysicalAccountingFacts, PhysicalAccountingSemantics,
    PhysicalAccountingSemanticsId, SourceDType, SourceStorageFact, TensorIdentity,
    PHYSICAL_ACCOUNTING_PROCEDURE,
};
pub use action_space::{ActionVocabulary, MapEdit};
pub use assess::{
    Assessment, NothingMeasured, ParentStanding, RankingRule, RankingSemantics, RankingSemanticsId,
    Score, RANKING_SEMANTICS_ID_VERSION,
};
pub use bind::{AccountingBindError, AccountingIncomplete, BoundPhysicalAccounting};
pub use candidate::{
    Candidate, CandidateDisposition, CandidateSet, Census, Footprint, Generator, MeasurementIntent,
    PreMeasurementPrune,
};
pub use evidence_bank::{EvidenceBank, EvidenceBankId, EVIDENCE_BANK_ID_VERSION};
pub use footprint::{CompiledBytes, FootprintError, PackCompiledBytes, SurfaceFootprint};
pub use graph::{RepresentationStateGraph, StateNode, TransitionPolicy};
pub use identity::{RepresentationState, RepresentationStateId, STATE_ID_VERSION};
pub use instrument::{InstrumentSemantics, InstrumentSemanticsId, INSTRUMENT_SEMANTICS_ID_VERSION};
pub use key::{MeasurementKey, MeasurementRegistry, MEASUREMENT_KEY_VERSION};
pub use realization::{LogicalBytes, RealizationId, ResolvedState};
pub use resolved::{
    resolve, LayoutAdmission, NoLayoutConstraint, PackLayoutAdmission, ResolvedDecision,
    ResolvedDecisionVector, ResolvedEncoding, SOURCE_PRECISION,
};
pub use search_policy::{BestFirst, MeasurementOpportunity, Selection};
pub use semantics::{SearchSemantics, SearchSemanticsId, SEARCH_SEMANTICS_ID_VERSION};
pub use snapshot::{
    Adjudication, FrontierEntry, Objective, SearchConfig, SearchFacts, SearchSnapshot, SearchSpace,
    SNAPSHOT_SCHEMA,
};
pub use surface::{SurfaceTensor, TensorSurface};
pub use transition::{Action, Provenance, Transition};

#[cfg(test)]
mod candidate_tests;
/// The Rung 5 record, shared by every test that needs a real search.
#[cfg(any(test, feature = "test-utils"))]
pub mod fixtures;
#[cfg(test)]
mod graph_tests;
#[cfg(test)]
mod key_tests;
#[cfg(test)]
mod replay_tests;
#[cfg(test)]
mod search_policy_tests;
#[cfg(test)]
mod snapshot_tests;
#[cfg(test)]
pub(crate) mod tests;
