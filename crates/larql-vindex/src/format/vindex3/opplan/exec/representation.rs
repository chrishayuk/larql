//! How continuation state is physically held between uses.
//!
//! [`continuation`](super::continuation) established the division of
//! labour: geometry says **what must persist**, the operator says **how it
//! changes**. This module adds the third axis — a representation says
//! **how the persisting state is physically held**, and what getting it
//! back promises.
//!
//! The claim being made first-class is that a [`ContinuationState`] is a
//! *semantic* object with more than one possible physical form. The
//! materialized form — every KV row and every recurrent buffer resident in
//! memory — is one representation, not the definition. A boundary
//! checkpoint that re-derives rows on demand, a compressed store, a
//! quantised store: each would be another representation of the *same*
//! continuation, distinguished by what it costs to hold and what its
//! reconstruction guarantees.
//!
//! ```text
//! canonical      the ContinuationState the geometry defines
//! stored form    what a representation actually retains  (store)
//! reconstruction how the canonical state comes back      (reconstruct)
//! correctness    what reconstruction promises            (contract)
//! ```
//!
//! Only the materialized representation exists. That is deliberate: this
//! rung introduces the seam with the boring arm — the one every current
//! consumer already uses implicitly — so that the first non-trivial
//! representation lands as *an alternative under an existing contract*
//! rather than as an engine special case. The seam is not allowed to grow
//! a variant before that consumer exists, for the same reason
//! [`RecurrentGeometry`](super::continuation::RecurrentGeometry) refused
//! to learn DeltaNet's semantics.

use super::continuation::{ContinuationState, LayerContinuationGeometry};

/// What reconstruction promises, relative to the canonical state.
///
/// One variant, and that is a recorded fact rather than a placeholder: the
/// materialized representation round-trips bit-for-bit, and no built
/// representation claims anything weaker. A bounded contract needs a
/// metric, a budget unit and a composition rule — the Residual ABI
/// programme owns that metric (horizon-T predictive KL, with a spacing
/// constraint, not a magnitude budget) — and inventing the unit here,
/// before the first non-exact representation exists to be measured in it,
/// would be choosing a precision no consumer asked for.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CorrectnessContract {
    /// Reconstruction yields the canonical state bit-for-bit — every KV
    /// row, every recurrent cell, and the logical position.
    Exact,
}

/// One physical representation of a component's continuation state.
///
/// Deliberately knows nothing about layers, operators, or devices: which
/// layers carry KV and which carry a recurrence is the geometry's
/// knowledge, and how a buffer updates is the operator's. A
/// representation's whole vocabulary is hold, give back, and cost.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ContinuationRepresentation {
    /// The canonical state itself, resident in memory. Storage is the
    /// state; reconstruction is identity; the contract is trivially
    /// exact. This is the control arm every alternative is judged
    /// against.
    Materialized,
}

/// A continuation state in a representation's stored form.
///
/// A separate type from [`ContinuationState`] even though today's only
/// variant wraps one unchanged, because the whole point of the seam is
/// that a stored form and a canonical state are different things: an
/// executor consumes the canonical state and must not learn to read
/// stored forms, and a store holds the stored form and must not hand it
/// out un-reconstructed.
#[derive(Debug, Clone, PartialEq)]
pub enum StoredContinuation {
    Materialized(ContinuationState),
}

impl StoredContinuation {
    /// The representation this stored form belongs to.
    pub fn representation(&self) -> ContinuationRepresentation {
        match self {
            Self::Materialized(_) => ContinuationRepresentation::Materialized,
        }
    }
}

/// A representation plus its promise: the unit a caller reasons about.
///
/// The plan is the authority on the round trip. A caller that wants its
/// continuation held some way asks the plan to [`store`](Self::store) it,
/// gets it back through [`reconstruct`](Self::reconstruct), and reads what
/// that round trip promised from [`contract`](Self::contract) — never
/// from knowledge of the representation's internals.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ContinuationPlan {
    representation: ContinuationRepresentation,
}

impl ContinuationPlan {
    /// The control arm: hold the canonical state itself.
    pub fn materialized() -> Self {
        Self {
            representation: ContinuationRepresentation::Materialized,
        }
    }

    pub fn representation(&self) -> ContinuationRepresentation {
        self.representation
    }

    /// What this plan's round trip promises.
    pub fn contract(&self) -> CorrectnessContract {
        match self.representation {
            ContinuationRepresentation::Materialized => CorrectnessContract::Exact,
        }
    }

    /// Put a canonical state into this plan's stored form.
    pub fn store(&self, state: ContinuationState) -> StoredContinuation {
        match self.representation {
            ContinuationRepresentation::Materialized => StoredContinuation::Materialized(state),
        }
    }

    /// Get the canonical state back from a stored form.
    ///
    /// Refuses a stored form belonging to a different representation
    /// rather than converting it: a conversion path between stored forms
    /// would let state silently cross contracts, and the caller holding a
    /// mismatched pair has already lost track of which promise applies.
    pub fn reconstruct(&self, stored: StoredContinuation) -> Result<ContinuationState, String> {
        match (self.representation, stored) {
            (ContinuationRepresentation::Materialized, StoredContinuation::Materialized(state)) => {
                Ok(state)
            }
        }
    }

    /// Elements this plan actually retains after `positions` steps.
    ///
    /// This is the axis on which representations differ economically, so
    /// it is arithmetic on the plan, not a label. The materialized answer
    /// IS the geometry's answer — KV layers scale with `positions`,
    /// recurrent layers do not — because holding the canonical state
    /// means paying exactly what the geometry says must persist. An
    /// alternative representation earns its existence by answering less
    /// here while keeping its contract.
    pub fn stored_elements_at(
        &self,
        geometry: &[LayerContinuationGeometry],
        positions: usize,
    ) -> usize {
        match self.representation {
            ContinuationRepresentation::Materialized => {
                geometry.iter().map(|g| g.elements_at(positions)).sum()
            }
        }
    }
}
