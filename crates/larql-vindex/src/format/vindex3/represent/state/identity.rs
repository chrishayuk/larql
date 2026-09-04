//! **`RepresentationStateId` — the dedup key the search graph is built
//! on.**
//!
//! ```text
//! RepresentationStateId = H( model_identity,
//!                            tensor_surface_identity,
//!                            effective_decision_vector )
//! ```
//!
//! Three inputs, each answering a question the other two cannot:
//!
//! * **model identity** — the SEMANTIC half of [`SourceIdentity`]:
//!   the graph by content, every representation's authority, and the
//!   rest of the validated index. The same map over two models is two
//!   states, and the graph level is what catches byte-identical
//!   payloads under different semantics.
//! * **surface identity** — a digest of every `(object, tensor, role,
//!   shape)`. Adding, removing, renaming or reshaping a tensor is a
//!   different surface even when every surviving decision is unchanged.
//! * **effective decisions** — what is presented, per tensor, with
//!   protected and layout-refused collapsed to source precision.
//!
//! The model identity already covers a great deal of what the surface
//! covers, and both are kept: the semantic identity states a
//! container's own declared facts, while the surface is what REPRESENT
//! *enumerated* from them under one classification. A change to role classification moves the
//! surface without moving a single byte on disk, and that is a genuinely
//! different search problem — the set of maps that would compile a
//! tensor changed.
//!
//! # What is NOT read: how the index was serialised
//!
//! v1 folded in `hash_bytes(index.json)`, so an index carrying
//! identical values in a different serialisation was a different
//! state. That is a FALSE SPLIT — the same physical reality arriving
//! with none of its own evidence — and removing it is why this is v2.
//! [`SourceIdentity::artifact`] still records those bytes, as
//! provenance, and no digest here reads it.
//!
//! ```text
//! presentation bytes changed only     SAME physical state
//! header/storage reality changed      DIFFERENT physical state
//! payload reality changed             DIFFERENT physical state
//! graph reality changed               DIFFERENT physical state
//! ```
//!
//! # Versioned, because the canonical form is a promise
//!
//! Every digest input is prefixed with [`STATE_ID_VERSION`]. A stored
//! DAG that outlives a change to the canonical form must be recognisably
//! stale rather than silently colliding with new states, and the
//! alternative — a bare hash whose meaning drifts — is how a persisted
//! search state starts answering questions about a scheme it was not
//! built under.

use serde::{Deserialize, Serialize};

use super::super::compile::hash_bytes;
use super::super::compiler::SourceIdentity;
use super::super::map::PrecisionMap;
use super::resolved::{resolve, LayoutAdmission, ResolvedDecisionVector};
use super::surface::{TensorSurface, SECTION};

/// The canonical-form version every state id is computed under.
pub const STATE_ID_VERSION: &str = "represent-state-id/v2";

/// **What a representation state IS.** Opaque by construction: it is a
/// digest and there is nothing to read out of it, which is the point —
/// a consumer that wants to know what a state decides asks the decision
/// vector, not the key.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub struct RepresentationStateId(String);

impl RepresentationStateId {
    pub fn as_str(&self) -> &str {
        &self.0
    }

    /// First twelve hex characters, for reports and log lines.
    pub fn short(&self) -> &str {
        &self.0[..self.0.len().min(12)]
    }
}

impl std::fmt::Display for RepresentationStateId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.0)
    }
}

/// **A node's immutable truth.**
///
/// Deliberately carries no measurement, no contract, no visit count and
/// no history. Evidence describes a state and mutable search statistics
/// are the policy's business; both are keyed BY the id and neither is
/// part of it. Keeping them out is what lets two paths that reach the
/// same decisions converge on one node.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct RepresentationState {
    id: RepresentationStateId,
    model: SourceIdentity,
    surface_identity: String,
    decisions: ResolvedDecisionVector,
}

impl RepresentationState {
    /// Resolve `map` against `surface` and identify the result.
    ///
    /// The map itself is not retained. It is one witness rule set among
    /// however many resolve identically, and keeping it on the node
    /// would invite exactly the confusion this type exists to prevent —
    /// a reader treating one arbitrary recipe as *the* recipe. The DAG's
    /// incoming edges are where recipes belong.
    pub fn resolve(
        model: &SourceIdentity,
        surface: &TensorSurface,
        map: &PrecisionMap,
        layout: &dyn LayoutAdmission,
    ) -> Self {
        Self::from_decisions(model, surface, resolve(map, surface, layout))
    }

    /// Identify an already-resolved decision vector.
    ///
    /// For a caller that read the decisions out of a built container
    /// rather than resolving a map — the check that a compiled artifact
    /// is the state the search believes it measured.
    pub fn from_decisions(
        model: &SourceIdentity,
        surface: &TensorSurface,
        decisions: ResolvedDecisionVector,
    ) -> Self {
        let surface_identity = surface.identity();
        let id = RepresentationStateId(hash_bytes(
            digest_input(model, &surface_identity, &decisions).as_bytes(),
        ));
        Self {
            id,
            model: model.clone(),
            surface_identity,
            decisions,
        }
    }

    pub fn id(&self) -> &RepresentationStateId {
        &self.id
    }

    pub fn model(&self) -> &SourceIdentity {
        &self.model
    }

    pub fn surface_identity(&self) -> &str {
        &self.surface_identity
    }

    pub fn decisions(&self) -> &ResolvedDecisionVector {
        &self.decisions
    }
}

/// The exact bytes the state digest is taken over.
///
/// Written out as one function so the canonical form is stated in a
/// single place a reader can check against the doc comment, rather than
/// implied by the order of calls at three call sites.
///
/// The model is written by
/// [`SourceSemanticIdentity::canonical`](super::super::source_identity::SourceSemanticIdentity::canonical)
/// and not restated here: a second canonicalisation of the same facts
/// is a second authority that can disagree with the first.
fn digest_input(
    model: &SourceIdentity,
    surface_identity: &str,
    decisions: &ResolvedDecisionVector,
) -> String {
    format!(
        "{STATE_ID_VERSION}{SECTION}model={}{SECTION}surface={surface_identity}{SECTION}decisions={}",
        model.semantic.canonical(),
        decisions.canonical()
    )
}
