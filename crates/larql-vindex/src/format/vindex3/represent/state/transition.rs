//! **Edges own how you got there. Nodes do not.**
//!
//! ```text
//! A ── +E24 ──────────────────┐
//!                             ▼
//!                             C
//!                             ▲
//! B ── K24→K25, then +E24 ────┘
//! ```
//!
//! One `C` where the physical state is one physical state, and both
//! discoveries survive. Baking either recipe into the node would make an
//! arbitrary one of them read as *the* recipe — which is what R5-F3
//! caught the hard way, when `P − K25 + H` turned out to be a map
//! already measured and rejected under a different name.
//!
//! # What may travel on an edge, and what may not
//!
//! ```text
//! MAY   the action, for REPRODUCTION
//! MAY   the physical delta, COMPUTED from two footprints
//! MAY   provenance: round, rank, script, session, an agent's rationale
//!
//! MUST NOT   reach state identity
//! MUST NOT   enter any prediction, ranking term or cost
//! ```
//!
//! The second prohibition is rung 5's, stated when per-candidate
//! provenance was first emitted: the action lists are recorded because
//! reproducing the map needs them *and for no other purpose*. So
//! [`Action`] carries no ordering and no score, and [`Provenance`] is
//! narrative — a place to put "rung5/N3, candidate U1, ranked first,
//! diagnostic 2.1720e-3" without inviting a comparator to read it.

use serde::{Deserialize, Serialize};

use super::identity::RepresentationStateId;
use super::realization::RealizationId;
use super::surface::{FIELD, RECORD};

/// What was applied to reach a child state.
///
/// Deliberately descriptive and deliberately inert: no `Ord`, no score,
/// no cost. Two transitions are the same transition when their parent,
/// child and action identity agree, and nothing else about an action is
/// consulted by anything.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Action {
    /// A short name for the move, e.g. `"+E24"`, `"K24→K25"`.
    pub label: String,
    /// Admitted actions this move removed, by their programme names.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub removed: Vec<String>,
    /// Actions this move added.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub added: Vec<String>,
}

impl Action {
    pub fn new(label: impl Into<String>) -> Self {
        Self {
            label: label.into(),
            removed: Vec::new(),
            added: Vec::new(),
        }
    }

    pub fn removing(mut self, actions: impl IntoIterator<Item = impl Into<String>>) -> Self {
        self.removed = actions.into_iter().map(Into::into).collect();
        self
    }

    pub fn adding(mut self, actions: impl IntoIterator<Item = impl Into<String>>) -> Self {
        self.added = actions.into_iter().map(Into::into).collect();
        self
    }

    /// What makes two applications of "the same move" the same edge.
    ///
    /// The removed and added lists are sorted here and nowhere else: an
    /// exchange written `-M26 +E24` and one written `+E24 -M26` are the
    /// same move, while the map's own exception ORDER — which is
    /// semantic — is a property of the resulting state and was already
    /// settled by the physical digest.
    pub fn identity(&self) -> String {
        let mut removed = self.removed.clone();
        let mut added = self.added.clone();
        removed.sort();
        added.sort();
        format!(
            "{}{FIELD}-{}{FIELD}+{}",
            self.label,
            removed.join(","),
            added.join(",")
        )
    }
}

/// Why this edge is in the graph — discovery context, never state.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Provenance {
    /// Who or what discovered it: a rung, a script, a session, an agent.
    pub by: String,
    /// Anything worth reading later. Round number, candidate rank,
    /// diagnostic reading, an agent's rationale — all belong here, as
    /// text, precisely so that no comparator can compute on them.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub note: Option<String>,
}

impl Provenance {
    pub fn new(by: impl Into<String>) -> Self {
        Self {
            by: by.into(),
            note: None,
        }
    }

    pub fn noting(mut self, note: impl Into<String>) -> Self {
        self.note = Some(note.into());
        self
    }
}

/// One edge: a move from a parent state to a child state.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Transition {
    parent: RepresentationStateId,
    child: RepresentationStateId,
    /// The child realization this move produced. Two moves can reach one
    /// physical state through different realizations, and the edge
    /// records which one it actually built.
    child_realization: RealizationId,
    action: Action,
    /// Computed from the two footprints, never supplied. Negative means
    /// bytes were removed.
    physical_delta: i64,
    /// Every discovery of this edge, deduplicated.
    provenance: Vec<Provenance>,
}

impl Transition {
    pub(crate) fn new(
        parent: RepresentationStateId,
        child: RepresentationStateId,
        child_realization: RealizationId,
        action: Action,
        physical_delta: i64,
        provenance: Provenance,
    ) -> Self {
        Self {
            parent,
            child,
            child_realization,
            action,
            physical_delta,
            provenance: vec![provenance],
        }
    }

    pub fn parent(&self) -> &RepresentationStateId {
        &self.parent
    }

    pub fn child(&self) -> &RepresentationStateId {
        &self.child
    }

    pub fn child_realization(&self) -> &RealizationId {
        &self.child_realization
    }

    pub fn action(&self) -> &Action {
        &self.action
    }

    pub fn physical_delta(&self) -> i64 {
        self.physical_delta
    }

    pub fn provenance(&self) -> &[Provenance] {
        &self.provenance
    }

    /// What makes this edge this edge.
    ///
    /// The child REALIZATION is in the key, not merely the physical
    /// child: a move that reaches the same bytes through a different set
    /// of decisions is a different move, and collapsing the two would
    /// undo the separation [`super::realization`] exists to keep.
    pub(crate) fn identity(&self) -> String {
        format!(
            "{}{RECORD}{}{RECORD}{}{RECORD}{}",
            self.parent,
            self.child,
            self.child_realization,
            self.action.identity()
        )
    }

    /// Record another discovery of this same edge.
    ///
    /// Re-discovering an edge under provenance already recorded is a
    /// no-op — a replayed round must not inflate the record. Two
    /// genuinely different discoverers of one edge is real information
    /// and is kept.
    pub(crate) fn observe(&mut self, provenance: Provenance) {
        if !self.provenance.contains(&provenance) {
            self.provenance.push(provenance);
        }
    }
}
