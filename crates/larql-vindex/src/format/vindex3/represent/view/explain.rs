//! **One state: what it is, and how it was reached.**
//!
//! The two are separate questions and this view keeps them apart, which
//! is the reason provenance lives on incoming EDGES and never on the
//! node:
//!
//! ```text
//! A --[Q6 E24]-------------------→ C
//! B --[exchange K24→K25, +E24]---→ C
//! ```
//!
//! One `C`, several explanations of how it was reached. A node that
//! carried its own provenance would have to pick one of them, and the
//! experiment ledger currently makes that distinction by hand.

use serde::Serialize;

use super::super::byte_ledger::ByteLedger;
use super::super::state::realization::{LogicalBytes, RealizationId};
use super::super::state::snapshot::SearchSnapshot;
use super::super::state::{RepresentationStateId, Transition};
use super::frontier::StateStanding;
use super::origin::{Origin, Rendered};

/// One state's identity, discovery paths and standing.
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct Explanation {
    pub state: RepresentationStateId,
    pub logical_bytes: LogicalBytes,
    /// The realizations that present this same state. More than one
    /// means two different DECISION sets that VINDEX3 would present
    /// identically — measurement-equivalent, and not action-equivalent.
    pub realizations: Vec<RealizationId>,
    /// Every edge that arrives here, each carrying its own action and
    /// its own provenance.
    pub reached_by: Vec<Transition>,
    pub leads_to: Vec<Transition>,
    /// What has been observed of it, and what the contract makes of
    /// that. `None` when the graph holds the state but the frontier
    /// does not — which cannot happen, and is reported rather than
    /// asserted.
    pub standing: Option<StateStanding>,
    /// Per-token reads, where the record holds them. NOT the footprint.
    pub ledger: Option<ByteLedger>,
}

impl Explanation {
    /// `None` when the graph does not hold this state — a fact about
    /// the record, not a failure.
    pub fn of(snapshot: &SearchSnapshot, state: &RepresentationStateId) -> Option<Self> {
        let graph = snapshot.graph();
        let node = graph.node(state)?;
        Some(Self {
            state: state.clone(),
            logical_bytes: node.logical_bytes(),
            realizations: node
                .realizations()
                .map(|r| r.realization_id().clone())
                .collect(),
            reached_by: graph.incoming(state).into_iter().cloned().collect(),
            leads_to: graph.outgoing(state).into_iter().cloned().collect(),
            standing: snapshot
                .frontier()
                .iter()
                .find(|e| &e.state == state)
                .map(StateStanding::of),
            ledger: snapshot.ledger(state).cloned(),
        })
    }
}

impl Rendered for Explanation {
    fn origins() -> Vec<Origin> {
        let mut origins = vec![
            Origin::new("state", "StateNode::physical_id()"),
            Origin::new("logical_bytes", "StateNode::logical_bytes()"),
            Origin::new("realizations", "StateNode::realizations()"),
            Origin::new("reached_by", "RepresentationStateGraph::incoming()"),
            Origin::new("leads_to", "RepresentationStateGraph::outgoing()"),
            Origin::new("ledger", "SearchSnapshot::ledger()"),
        ];
        origins.extend(StateStanding::origins().iter().map(|o| o.under("standing")));
        origins
    }
}
