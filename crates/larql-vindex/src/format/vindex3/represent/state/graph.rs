//! **The state graph: physical nodes, realization facets, provenance on
//! the edges.**
//!
//! ```text
//! Physical identity        RepresentationStateId   → evidence, MeasurementKey
//!         │
//!         ▼
//! Resolved realization     full decision facts     → the action generator
//!         │
//!         ▼
//! State node               one per PHYSICAL state, holding its realizations
//!         │
//!  transformation edges    parent, child, action, delta, provenance
//!         ▼
//! State node
//! ```
//!
//! # Why this is not called a DAG
//!
//! Acyclicity here is a **theorem under a policy**, not a property of
//! the domain, and the distinction is worth a type rather than a
//! comment.
//!
//! Under [`TransitionPolicy::StrictlyImprovingPhysical`] every admitted
//! edge strictly decreases [`LogicalBytes`]. A cycle would require a
//! strictly decreasing sequence of `u64` returning to its start, so
//! there are no cycles: the structure is a DAG, and
//! [`RepresentationStateGraph::is_acyclic`] can only ever confirm it.
//! That policy is what rung 5 already enforces — its neighbourhoods
//! pruned `−E26 + H` at +1.39 GB and `−K25 + M23` at +2,091,136 B for
//! being physically worse, and Ruling 1 lists physical dominance as one
//! of the four legitimate pre-measurement prunes.
//!
//! It is nonetheless a policy and not a law, and the programme's own
//! roadmap says when it breaks: once residency joins the state and the
//! objective becomes measured tok/s, a move that *adds* logical bytes
//! while freeing unified memory for resident experts is exactly the kind
//! of move the search must be able to make. At that point transitions
//! stop being monotone in bytes, transpositions and cycles become
//! reachable, and the canonical structure is a general graph with the
//! DAG living in the search overlay instead.
//!
//! So the policy is recorded on the graph, checked on every insertion,
//! and named in the serialised form — a graph built under one policy is
//! recognisably not a graph built under the other.
//!
//! # One graph, one model
//!
//! A map's physical prize is a property of the model it is resolved
//! against, so a graph that mixed models would hold nodes whose costs
//! could not be compared. The root fixes the model and the tensor
//! surface; every later state is checked against both.

use std::collections::{BTreeMap, BTreeSet};

use serde::{Deserialize, Serialize};

use super::super::compiler::SourceIdentity;
use super::identity::RepresentationStateId;
use super::realization::{LogicalBytes, RealizationId, ResolvedState};
use super::transition::{Action, Provenance, Transition};
use crate::error::VindexError;

/// What the graph will admit as an edge.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum TransitionPolicy {
    /// Every edge must strictly reduce logical bytes. Acyclicity
    /// follows; the structure is a DAG.
    StrictlyImprovingPhysical,
    /// Any edge is admitted. Use when the objective is no longer bytes —
    /// the structure is then a general graph and the caller owns
    /// termination.
    Unconstrained,
}

impl TransitionPolicy {
    /// Whether an edge with this delta is admitted.
    ///
    /// Public because the candidate generator prunes on physical
    /// dominance by asking THIS, rather than by reimplementing the
    /// comparison — a generator that offered candidates the graph would
    /// then refuse, or pruned ones it would have taken, would be two
    /// answers to one question.
    pub fn admits(self, physical_delta: i64) -> bool {
        match self {
            Self::StrictlyImprovingPhysical => physical_delta < 0,
            Self::Unconstrained => true,
        }
    }

    /// Whether acyclicity is guaranteed by construction under this
    /// policy, rather than merely observed so far.
    pub fn guarantees_acyclic(self) -> bool {
        matches!(self, Self::StrictlyImprovingPhysical)
    }
}

/// One physical state, and every realization of it seen so far.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct StateNode {
    physical_id: RepresentationStateId,
    logical_bytes: LogicalBytes,
    realizations: BTreeMap<RealizationId, ResolvedState>,
}

impl StateNode {
    pub fn physical_id(&self) -> &RepresentationStateId {
        &self.physical_id
    }

    pub fn logical_bytes(&self) -> LogicalBytes {
        self.logical_bytes
    }

    /// Every realization of this physical state, in id order.
    ///
    /// More than one is normal, not exceptional: a tensor held at source
    /// precision by a protection and one held there by a layout refusal
    /// present identical bytes and admit different moves.
    pub fn realizations(&self) -> impl Iterator<Item = &ResolvedState> {
        self.realizations.values()
    }

    pub fn realization(&self, id: &RealizationId) -> Option<&ResolvedState> {
        self.realizations.get(id)
    }

    pub fn realization_count(&self) -> usize {
        self.realizations.len()
    }

    /// Fold a realization in, refusing one that disagrees about bytes.
    fn absorb(&mut self, state: ResolvedState) -> Result<(), VindexError> {
        if state.logical_bytes() != self.logical_bytes {
            return Err(VindexError::Parse(format!(
                "state {} is already priced at {} but this realization reports {} — one \
                 physical state presents one set of bytes, so one of the two footprints is \
                 wrong",
                self.physical_id.short(),
                self.logical_bytes,
                state.logical_bytes()
            )));
        }
        self.realizations
            .entry(state.realization_id().clone())
            .or_insert(state);
        Ok(())
    }
}

/// **The search state graph.**
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct RepresentationStateGraph {
    policy: TransitionPolicy,
    model: SourceIdentity,
    surface_identity: String,
    root: RepresentationStateId,
    nodes: BTreeMap<RepresentationStateId, StateNode>,
    /// Keyed by [`Transition::identity`], so re-discovering an edge
    /// merges into it rather than appending a duplicate.
    edges: BTreeMap<String, Transition>,
}

impl RepresentationStateGraph {
    /// Start a graph at `root`, which fixes the model and the surface.
    pub fn new(policy: TransitionPolicy, root: ResolvedState) -> Self {
        let physical_id = root.physical_id().clone();
        let model = root.state().model().clone();
        let surface_identity = root.state().surface_identity().to_string();
        let node = StateNode {
            physical_id: physical_id.clone(),
            logical_bytes: root.logical_bytes(),
            realizations: BTreeMap::from([(root.realization_id().clone(), root)]),
        };
        Self {
            policy,
            model,
            surface_identity,
            root: physical_id.clone(),
            nodes: BTreeMap::from([(physical_id, node)]),
            edges: BTreeMap::new(),
        }
    }

    /// **Apply `action` at `parent`, arriving at `child`.**
    ///
    /// Returns the child's physical id. Every refusal names what it
    /// checked rather than reporting a generic failure, because each one
    /// is a different bug in the caller.
    pub fn apply(
        &mut self,
        parent: &RepresentationStateId,
        action: Action,
        child: ResolvedState,
        provenance: Provenance,
    ) -> Result<RepresentationStateId, VindexError> {
        let parent_node = self.nodes.get(parent).ok_or_else(|| {
            VindexError::Parse(format!(
                "no state {} in this graph — an edge cannot begin at a state the graph has \
                 never held",
                parent.short()
            ))
        })?;
        self.check_same_model(&child)?;

        let physical_delta = child
            .logical_bytes()
            .delta_from(parent_node.logical_bytes());
        if !self.policy.admits(physical_delta) {
            return Err(VindexError::Parse(format!(
                "transition `{}` moves {} logical bytes and the graph's policy is {:?} — the \
                 policy is what makes this structure acyclic, so admitting this edge would \
                 silently change what the graph IS",
                action.label, physical_delta, self.policy
            )));
        }

        let child_id = child.physical_id().clone();
        let realization_id = child.realization_id().clone();
        match self.nodes.get_mut(&child_id) {
            Some(existing) => existing.absorb(child)?,
            None => {
                self.nodes.insert(
                    child_id.clone(),
                    StateNode {
                        physical_id: child_id.clone(),
                        logical_bytes: child.logical_bytes(),
                        realizations: BTreeMap::from([(realization_id.clone(), child)]),
                    },
                );
            }
        }

        let edge = Transition::new(
            parent.clone(),
            child_id.clone(),
            realization_id,
            action,
            physical_delta,
            provenance.clone(),
        );
        match self.edges.get_mut(&edge.identity()) {
            Some(existing) => existing.observe(provenance),
            None => {
                self.edges.insert(edge.identity(), edge);
            }
        }
        Ok(child_id)
    }

    fn check_same_model(&self, child: &ResolvedState) -> Result<(), VindexError> {
        if child.state().model() != &self.model {
            return Err(VindexError::Parse(
                "this state belongs to a different container — a map's physical prize is a \
                 property of the model it resolves against, so one graph holds one model"
                    .into(),
            ));
        }
        if child.state().surface_identity() != self.surface_identity {
            return Err(VindexError::Parse(format!(
                "this state was resolved against surface {} but the graph's root used {} — \
                 the same bytes under a different enumerated surface is a different search \
                 problem",
                &child.state().surface_identity()[..12],
                &self.surface_identity[..12]
            )));
        }
        Ok(())
    }

    pub fn policy(&self) -> TransitionPolicy {
        self.policy
    }

    pub fn model(&self) -> &SourceIdentity {
        &self.model
    }

    pub fn surface_identity(&self) -> &str {
        &self.surface_identity
    }

    pub fn root(&self) -> &RepresentationStateId {
        &self.root
    }

    pub fn node(&self, id: &RepresentationStateId) -> Option<&StateNode> {
        self.nodes.get(id)
    }

    pub fn nodes(&self) -> impl Iterator<Item = &StateNode> {
        self.nodes.values()
    }

    pub fn len(&self) -> usize {
        self.nodes.len()
    }

    pub fn is_empty(&self) -> bool {
        // A graph always holds its root.
        self.nodes.is_empty()
    }

    pub fn edges(&self) -> impl Iterator<Item = &Transition> {
        self.edges.values()
    }

    pub fn edge_count(&self) -> usize {
        self.edges.len()
    }

    /// Every edge that arrives at `id` — the several explanations of how
    /// one state was reached.
    pub fn incoming(&self, id: &RepresentationStateId) -> Vec<&Transition> {
        self.edges.values().filter(|e| e.child() == id).collect()
    }

    pub fn outgoing(&self, id: &RepresentationStateId) -> Vec<&Transition> {
        self.edges.values().filter(|e| e.parent() == id).collect()
    }

    /// Whether the graph contains no directed cycle.
    ///
    /// Under [`TransitionPolicy::StrictlyImprovingPhysical`] this can
    /// only return `true` — it is the theorem, checked. It is a real
    /// question only for an `Unconstrained` graph.
    pub fn is_acyclic(&self) -> bool {
        let mut done: BTreeSet<&RepresentationStateId> = BTreeSet::new();
        let mut path: BTreeSet<&RepresentationStateId> = BTreeSet::new();
        self.nodes
            .keys()
            .all(|id| Self::visit(self, id, &mut done, &mut path))
    }

    fn visit<'a>(
        &'a self,
        id: &'a RepresentationStateId,
        done: &mut BTreeSet<&'a RepresentationStateId>,
        path: &mut BTreeSet<&'a RepresentationStateId>,
    ) -> bool {
        if done.contains(id) {
            return true;
        }
        if !path.insert(id) {
            return false;
        }
        // Walk the edge's own child id rather than looking it up: `apply`
        // inserts the child node before the edge, so a dangling child
        // cannot exist, and a lookup here would need a `None` arm that no
        // input can reach — an untestable branch guarding an impossible
        // state.
        let ok = self
            .outgoing(id)
            .into_iter()
            .all(|e| self.visit(e.child(), done, path));
        path.remove(id);
        done.insert(id);
        ok
    }
}
