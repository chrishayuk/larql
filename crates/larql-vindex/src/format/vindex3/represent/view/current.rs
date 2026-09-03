//! **Where the search stands.**
//!
//! The incumbent is the head of the optimiser's own admitted ordering
//! and not a pick made here — [`SearchSnapshot::admitted`] applies the
//! objective and orders cheapest first, and this view reports position
//! zero of that list. A facade that chose its own best map would be a
//! second authority, which is the whole thing stage 4 must not become.

use serde::Serialize;

use super::super::measurement::EvidenceScale;
use super::super::state::snapshot::SearchSnapshot;
use super::super::state::RepresentationStateId;
use super::frontier::StateStanding;
use super::origin::{Origin, Rendered};

/// States the record holds no reading of, at one scale.
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct ScaleGap {
    pub scale: EvidenceScale,
    /// Deliberately unordered by desirability: WHICH of these deserves
    /// a run is a search policy's decision, and a fact query must not
    /// pre-empt it.
    pub states: Vec<RepresentationStateId>,
}

impl ScaleGap {
    /// One gap per scale, cheapest evidence first.
    pub fn all(snapshot: &SearchSnapshot) -> Vec<Self> {
        EvidenceScale::ALL
            .into_iter()
            .map(|scale| Self {
                scale,
                states: snapshot.unmeasured_at(scale),
            })
            .collect()
    }

    pub fn origins() -> Vec<Origin> {
        vec![
            Origin::new("scale", "EvidenceScale::ALL"),
            Origin::new("states", "SearchSnapshot::unmeasured_at(scale)"),
        ]
    }
}

/// What has been built, what has been admitted, and what is still dark.
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct Current {
    /// The realization the graph was rooted at.
    pub root: RepresentationStateId,
    pub states: usize,
    pub transitions: usize,
    /// Checked, not assumed from the policy.
    pub acyclic: bool,
    /// The cheapest state carrying an authority reading that satisfies
    /// the contract, or `None` when nothing has been admitted.
    pub incumbent: Option<StateStanding>,
    /// Every admitted state, cheapest first.
    pub admitted: Vec<StateStanding>,
    pub unmeasured: Vec<ScaleGap>,
}

impl Current {
    pub fn of(snapshot: &SearchSnapshot) -> Self {
        let graph = snapshot.graph();
        let admitted: Vec<StateStanding> =
            snapshot.admitted().iter().map(StateStanding::of).collect();
        Self {
            root: graph.root().clone(),
            states: graph.len(),
            transitions: graph.edge_count(),
            acyclic: graph.is_acyclic(),
            incumbent: admitted.first().cloned(),
            admitted,
            unmeasured: ScaleGap::all(snapshot),
        }
    }
}

impl Rendered for Current {
    fn origins() -> Vec<Origin> {
        let mut origins = vec![
            Origin::new("root", "RepresentationStateGraph::root()"),
            Origin::new("states", "RepresentationStateGraph::len()"),
            Origin::new("transitions", "RepresentationStateGraph::edge_count()"),
            Origin::new("acyclic", "RepresentationStateGraph::is_acyclic()"),
        ];
        origins.extend(
            StateStanding::origins()
                .iter()
                .map(|o| o.under("incumbent")),
        );
        origins.extend(
            StateStanding::origins()
                .iter()
                .map(|o| o.under("admitted[]")),
        );
        origins.extend(ScaleGap::origins().iter().map(|o| o.under("unmeasured[]")));
        origins
    }
}
