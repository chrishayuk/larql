//! **Two states, side by side.**
//!
//! Co-location and two substrate calls, and deliberately nothing else.
//!
//! In particular this view does NOT report whether the two readings are
//! comparable. Two observations under different banks, scales or
//! instruments are different experiments — 1c exists because that
//! distinction is load bearing — and a facade that pronounced them
//! comparable would be making an evidence judgement it has no standing
//! to make. Both keys are rendered whole; what they license is the
//! evidence system's question, not the transport's.

use serde::Serialize;

use super::super::state::snapshot::SearchSnapshot;
use super::super::state::{RepresentationStateId, Transition};
use super::frontier::StateStanding;
use super::origin::{Origin, Rendered};

/// Two states' standings, with the physical difference between them.
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct Comparison {
    pub left: StateStanding,
    pub right: StateStanding,
    /// The right state's footprint minus the left's. Negative means the
    /// right removes bytes, which is the direction the objective wants.
    ///
    /// Physics composes exactly, so this subtraction is sound. The
    /// behavioural difference is NOT rendered beside it and must never
    /// be: R4-F3 and R4-F11 both closed on a composed behavioural
    /// prediction, once by 51 %.
    pub physical_delta: i64,
    /// The edge from left to right, where the graph holds one. Its
    /// delta was computed by the graph from two footprints when the
    /// move was made, and it is the only difference here that carries
    /// an action and a provenance.
    pub transition: Option<Transition>,
}

/// Which of the two states the record does not hold.
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct NotHeld {
    pub states: Vec<RepresentationStateId>,
}

impl Comparison {
    /// `Err` naming the states the graph does not hold, so a caller is
    /// told which of its two ids was wrong rather than that something
    /// was.
    pub fn of(
        snapshot: &SearchSnapshot,
        left: &RepresentationStateId,
        right: &RepresentationStateId,
    ) -> Result<Self, NotHeld> {
        let frontier = snapshot.frontier();
        let standing = |id: &RepresentationStateId| frontier.iter().find(|e| &e.state == id);
        let (Some(l), Some(r)) = (standing(left), standing(right)) else {
            return Err(NotHeld {
                states: [left, right]
                    .into_iter()
                    .filter(|id| standing(id).is_none())
                    .cloned()
                    .collect(),
            });
        };
        Ok(Self {
            physical_delta: r.logical_bytes.delta_from(l.logical_bytes),
            transition: snapshot
                .graph()
                .outgoing(left)
                .into_iter()
                .find(|t| t.child() == right)
                .cloned(),
            left: StateStanding::of(l),
            right: StateStanding::of(r),
        })
    }
}

impl Rendered for Comparison {
    fn origins() -> Vec<Origin> {
        let mut origins = vec![
            Origin::new("physical_delta", "LogicalBytes::delta_from()"),
            Origin::new("transition", "RepresentationStateGraph::outgoing()"),
        ];
        origins.extend(StateStanding::origins().iter().map(|o| o.under("left")));
        origins.extend(StateStanding::origins().iter().map(|o| o.under("right")));
        origins
    }
}
