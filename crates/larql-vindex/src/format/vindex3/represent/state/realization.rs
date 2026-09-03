//! **Physical identity is not search-context identity.**
//!
//! Stage 1a collapses a protected tensor and a layout-refused tensor
//! into one [`RepresentationStateId`], and that is correct: they present
//! the same bytes, so a measurement of one is a measurement of the
//! other. It is also, on its own, an information-loss bug waiting to
//! happen, because the two are not *action*-equivalent:
//!
//! ```text
//!                  physical equivalence
//!                          │
//!                 RepresentationStateId
//!                    ↑                ↑
//!          realization A        realization B
//!          X = Source           X = LayoutRefused
//!                    │                │
//!          "unprotect X" is    nothing can compile X;
//!          a legal move        the layout refuses it
//! ```
//!
//! So the graph holds two keys and keeps them straight:
//!
//! ```text
//! RepresentationStateId   what is PRESENTED   → evidence, MeasurementKey
//! RealizationId           what is DECIDED     → the action generator
//! ```
//!
//! **Evidence may deduplicate more aggressively than search may.** A
//! node is one physical state and may carry several realizations; a
//! measurement belongs to the node, an action space belongs to a
//! realization. Merging the second into the first would let an
//! optimizer believe it can remove a structural refusal.
//!
//! The invariant tying them together — *same realization implies same
//! physical state* — holds by construction, because
//! [`ResolvedDecisionVector::canonical_full`] refines
//! [`ResolvedDecisionVector::canonical`], and is tested rather than
//! merely asserted here.

use serde::{Deserialize, Serialize};

use super::super::compile::hash_bytes;
use super::identity::{RepresentationState, RepresentationStateId, STATE_ID_VERSION};
use super::surface::SECTION;

/// **The whole-map logical footprint of one state.**
///
/// A newtype, and not a bare `u64`, because this programme has already
/// paid for confusing three different byte quantities:
///
/// ```text
/// logical footprint   what a MAP costs           this type
/// per-token read      what a DECODER reads       ByteLedger
/// saving              a DIFFERENCE of footprints Transition::physical_delta
/// ```
///
/// R5-F5 read a footprint column as a saving and overstated an expert
/// revert by 3.39×, in the direction that makes expert in-moves look
/// prunable. A delta is therefore never supplied — it is computed, by
/// [`Self::delta_from`], from two footprints the graph holds.
///
/// Supplied by the caller from the byte ledger, never derived here:
/// [`super::super::byte_ledger`] states why scopes are supplied rather
/// than inferred, and inferring a footprint from geometry would be the
/// same mistake one level down.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub struct LogicalBytes(u64);

impl LogicalBytes {
    pub fn new(bytes: u64) -> Self {
        Self(bytes)
    }

    pub fn get(self) -> u64 {
        self.0
    }

    /// This footprint minus `parent`'s. Negative means bytes were
    /// removed, which is the direction a search wants.
    ///
    /// `i64` is wide enough by three orders of magnitude for any model
    /// that fits on a disk, so the cast cannot wrap in practice; it is
    /// written as one expression so there is a single place to check.
    pub fn delta_from(self, parent: LogicalBytes) -> i64 {
        self.0 as i64 - parent.0 as i64
    }
}

impl std::fmt::Display for LogicalBytes {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{} B", self.0)
    }
}

/// **What a realization IS**: the full decision facts, not merely what
/// they present.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub struct RealizationId(String);

impl RealizationId {
    pub fn as_str(&self) -> &str {
        &self.0
    }

    pub fn short(&self) -> &str {
        &self.0[..self.0.len().min(12)]
    }
}

impl std::fmt::Display for RealizationId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.0)
    }
}

/// One realization: a resolved state, its two identities, and what it
/// costs.
///
/// The costs and the identities travel together deliberately. A node
/// that held the id and looked its footprint up elsewhere could be
/// asked about a state whose bytes nobody had priced, and would have to
/// answer with a default.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ResolvedState {
    realization_id: RealizationId,
    state: RepresentationState,
    logical_bytes: LogicalBytes,
}

impl ResolvedState {
    /// Identify a resolved state as a realization.
    pub fn new(state: RepresentationState, logical_bytes: LogicalBytes) -> Self {
        let input = format!(
            "{STATE_ID_VERSION}{SECTION}realization{SECTION}physical={}{SECTION}decisions={}",
            state.id(),
            state.decisions().canonical_full()
        );
        Self {
            realization_id: RealizationId(hash_bytes(input.as_bytes())),
            state,
            logical_bytes,
        }
    }

    /// What this state PRESENTS — the evidence and `MeasurementKey` key.
    pub fn physical_id(&self) -> &RepresentationStateId {
        self.state.id()
    }

    /// What this state DECIDES — the action generator's key.
    pub fn realization_id(&self) -> &RealizationId {
        &self.realization_id
    }

    pub fn state(&self) -> &RepresentationState {
        &self.state
    }

    pub fn logical_bytes(&self) -> LogicalBytes {
        self.logical_bytes
    }
}
