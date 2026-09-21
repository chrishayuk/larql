//! **What to run next, and whether this build can run it.**
//!
//! [`super::super::view::next_experiment::NextExperiment`] answers three
//! questions an agent needs kept apart — here is the experiment, there
//! is nothing left, I cannot tell you. Actuation adds a fourth, and it
//! is genuinely different from the other three:
//!
//! ```text
//! Ready           the record selected an experiment and this build can
//!                 state it as a request
//! Exhausted       every move is observed or pruned
//! NotSelectable   the record could not say WHAT to measure
//! NotPreparable   it said what, and it cannot say HOW
//! ```
//!
//! Collapsing the last two would be the same error as collapsing
//! `Exhausted` into `Unavailable`: they are fixed by different actions
//! at different layers. `NotSelectable` means the record lacks pricing
//! authority — read the container's segment table again. `NotPreparable`
//! means it lacks protocol authority, or this build lacks a gate or a
//! layout policy the record names; re-reading a container fixes one of
//! those and cannot fix the other.
//!
//! # Derived from the record and nothing else
//!
//! [`PreparedExperiment::of`] takes no arguments beyond the snapshot,
//! for 4b's reason: the question — which applied set, which corpus,
//! which scale, which instrument, which gate — is stored, so the answer
//! is a property of the record rather than of whatever a caller wired
//! in. It calls `SearchSnapshot::next_experiment`, which itself injects
//! nothing.
//!
//! # Two doors, and only one of them is the authorised path
//!
//! ```text
//! PreparedExperiment::of(record)          the record's own selection,
//!                                         authorised end to end
//!
//! MeasurementRequest::of(record, key, …)  restate THIS experiment,
//!                                         truthfully
//! ```
//!
//! The second exists because a caller may legitimately hold several
//! already-authorised keys at once — an `ExecutionBatch` is exactly
//! that — and each needs a request. It is not a selection back door and
//! could not be one: it takes the key it is given and refuses unless the
//! map presents that key's state, so a caller may choose a different
//! experiment and can never be handed a request that misstates the one
//! it asked for. Choosing stays the policy's business; truthfulness is
//! this layer's.

use super::super::state::search_policy::Selection;
use super::super::state::snapshot::SearchSnapshot;
use super::request::{MeasurementRequest, RequestRefusal};

/// A selected experiment this build can state as a request.
#[derive(Debug, Clone, PartialEq)]
pub struct Ready {
    /// The experiment, restated so something can perform it.
    pub request: MeasurementRequest,
    /// The best physical prize among the routes to it. Negative removes
    /// bytes, which is the direction the objective wants.
    pub physical_delta: i64,
    /// How many routes reach this one experiment. More than one is
    /// normal: two realizations can reach one physical state, and they
    /// share the measurement.
    pub routes: usize,
    /// How many opportunities the policy ordered. `1` when there was
    /// nothing to rank.
    pub considered: usize,
}

/// **What to run next.**
#[derive(Debug, Clone, PartialEq)]
pub enum PreparedExperiment {
    Ready(Box<Ready>),
    /// The optimiser could answer and there is nothing left to measure.
    Exhausted,
    /// The optimiser could not select an experiment. The substrate's own
    /// message, unedited — the reasons are already distinct there.
    NotSelectable {
        detail: String,
    },
    /// It selected one and this build cannot state it as a request.
    NotPreparable(RequestRefusal),
}

impl PreparedExperiment {
    /// Derive the answer from the record and nothing else.
    pub fn of(snapshot: &SearchSnapshot) -> Self {
        let selection = match snapshot.next_experiment() {
            Ok(selection) => selection,
            Err(e) => {
                return Self::NotSelectable {
                    detail: e.to_string(),
                }
            }
        };
        let considered = match &selection {
            Selection::Ranked { considered, .. } => *considered,
            _ => 1,
        };
        let Some(opportunity) = selection.opportunity() else {
            return Self::Exhausted;
        };

        // The LEADING route, and not any route that happens to reach the
        // state: the policy has already ordered them, and picking a
        // different one here would be this layer overruling the ranking
        // it was handed. Every route reaches the same physical state, so
        // the choice does not change what is measured — but it would
        // change which provenance the run is attributed to.
        let leading = opportunity.leading();
        match MeasurementRequest::of(snapshot, &opportunity.key, &leading.applied) {
            Ok(request) => Self::Ready(Box::new(Ready {
                request,
                physical_delta: opportunity.physical_delta(),
                routes: opportunity.routes(),
                considered,
            })),
            Err(refusal) => Self::NotPreparable(refusal),
        }
    }

    /// The request, where there is one.
    pub fn request(&self) -> Option<&MeasurementRequest> {
        match self {
            Self::Ready(ready) => Some(&ready.request),
            _ => None,
        }
    }

    /// Whether the record can be acted on right now.
    pub fn is_ready(&self) -> bool {
        matches!(self, Self::Ready(_))
    }
}
