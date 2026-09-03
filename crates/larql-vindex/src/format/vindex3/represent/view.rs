//! **The read-only facade: the optimiser renders its own truth.**
//!
//! Stage 4 of the physical optimiser. The theorem it exists to hold:
//!
//! > **Anything an agent can learn through this facade is already
//! > derivable from the deterministic optimiser substrate.**
//!
//! Everything here is a projection. No view orders anything the
//! optimiser did not order, computes a verdict the contract did not
//! draw, or prices a byte a footprint did not price. Where the
//! substrate already derives `Serialize` the view renders the
//! substrate's own type, because a reshaping of a [`Margin`] is a
//! second `Margin` that can disagree with the first. Where it does not
//! — [`Adjudication`] and [`FrontierEntry`] withhold `Serialize` on
//! purpose, being derived verdicts that stage 1d forbids storing — each
//! rendered field names the call that produced it, and
//! [`origin::walk`] checks that the list is complete in both
//! directions.
//!
//! # Why the facade is a type and not a set of functions
//!
//! [`OptimizerView`] holds the snapshot privately. A transport is given
//! the view and never the [`SearchSnapshot`], so the "derive nothing in
//! transport" rule is a matter of what is reachable rather than of what
//! a reviewer noticed. Serving one of these over MCP should be
//! dispatch and serialisation and nothing else.
//!
//! # What is deliberately absent
//!
//! ```text
//! record      apply      expand      promote      accept_candidate
//! ```
//!
//! Not "not yet": the search is a deterministic optimiser and the
//! evidence system decides what is true. An agent chooses which
//! question to ask. It gets no vote on what the answer means, and
//! `accept_candidate(reason = "AI judgement")` must never exist.
//!
//! One of the seven refuses. [`next_experiment`] explains why, and the
//! refusal is a finding about the substrate rather than a gap in the
//! facade.
//!
//! [`Margin`]: super::constraint::Margin
//! [`Adjudication`]: super::state::snapshot::Adjudication
//! [`FrontierEntry`]: super::state::snapshot::FrontierEntry

pub mod compare;
pub mod current;
pub mod describe;
pub mod evidence;
pub mod explain;
pub mod frontier;
pub mod next_experiment;
pub mod origin;

pub use compare::{Comparison, NotHeld};
pub use current::{Current, ScaleGap};
pub use describe::Describe;
pub use evidence::{EvidenceReport, Observation};
pub use explain::Explanation;
pub use frontier::{AdjudicationView, Frontier, StateStanding};
pub use next_experiment::{Missing, NextExperiment, Refusal};
pub use origin::{Coverage, Origin, Rendered};

use super::state::snapshot::SearchSnapshot;
use super::state::RepresentationStateId;

/// **The seven questions an agent may ask.**
///
/// The whole read surface, and the only handle a transport is given.
#[derive(Debug, Clone, Copy)]
pub struct OptimizerView<'a> {
    snapshot: &'a SearchSnapshot,
}

impl<'a> OptimizerView<'a> {
    pub fn new(snapshot: &'a SearchSnapshot) -> Self {
        Self { snapshot }
    }

    /// What this search IS — model, contract, and every rule a
    /// conclusion is drawn under.
    pub fn describe(&self) -> Describe {
        Describe::of(self.snapshot)
    }

    /// Where it stands: what has been built, what has been admitted,
    /// and what is still dark.
    pub fn current(&self) -> Current {
        Current::of(self.snapshot)
    }

    /// Every state's standing, recomputed from the readings and the
    /// frozen gate.
    pub fn frontier(&self) -> Frontier {
        Frontier {
            states: self
                .snapshot
                .frontier()
                .iter()
                .map(StateStanding::of)
                .collect(),
            admitted: self
                .snapshot
                .admitted()
                .iter()
                .map(StateStanding::of)
                .collect(),
        }
    }

    /// One state: what it is, and — separately — how it was reached.
    /// `None` when the graph does not hold it.
    pub fn explain(&self, state: &RepresentationStateId) -> Option<Explanation> {
        Explanation::of(self.snapshot, state)
    }

    /// Two states side by side, with the physical difference between
    /// them and the edge that joins them where one exists.
    pub fn compare(
        &self,
        left: &RepresentationStateId,
        right: &RepresentationStateId,
    ) -> Result<Comparison, NotHeld> {
        Comparison::of(self.snapshot, left, right)
    }

    /// The measurement record — raw banks beside their verdicts — and
    /// the rules for reading it.
    pub fn evidence(&self, state: Option<&RepresentationStateId>) -> EvidenceReport {
        EvidenceReport::of(self.snapshot, state)
    }

    /// What to measure next, which this record cannot say. The refusal
    /// names what is missing.
    pub fn next_experiment(&self) -> NextExperiment {
        NextExperiment::of(self.snapshot)
    }
}

#[cfg(test)]
mod tests;
