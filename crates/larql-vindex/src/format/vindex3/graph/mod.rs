//! The VINDEX3 system graph — components, logical objects, representations,
//! and cross-component interface edges (V3-G2).
//!
//! A modern release is not a weights file; it is a *system*: a target model,
//! a perception tower physically embedded in the same checkpoint, a drafter
//! artifact consuming the target's hidden states through a declared tap
//! interface. The graph is the schema level where that structure lives:
//!
//! ```text
//!               SystemGraph
//!                    │
//!       ┌────────────┴────────────┐
//!       │                         │
//!  Component (role, topology)  HiddenStateEdge
//!       │                         (logical flow)
//!  LogicalObject  ←────────────── implemented by tensors,
//!       │                         but the edge is NOT the tensor
//!  Representation
//!       │
//!  SourceBinding (physical tensor names — never the identity)
//! ```
//!
//! Two identity rules are load-bearing:
//!
//! - **Object identity is conceptual.** `draft.target_feature_projector`,
//!   not `encoder.fc`. Physical tensor names live in [`SourceBinding`]s so
//!   the same logical operand can later carry several materialisations
//!   (bf16, q4, transposed-Metal) without unwinding its id.
//! - **The edge is not the implementing tensor.** A drafter's tap interface
//!   is logical flow between components; the fusion projector that
//!   implements its consumer side is a tensor object. The graph carries
//!   both, separately, linked by id.

pub mod build;
pub mod complete;
pub mod component;
pub mod edge;
pub mod object;
pub mod policy;
pub mod roles;
pub mod surface;

#[cfg(test)]
mod tests;

use serde::{Deserialize, Serialize};

pub use build::{build_from_inventories, BuiltGraph, IncompleteSurface, UnplacedGroup};
pub use complete::{execution_completeness, CompletenessDefect};
pub use component::{Component, ComponentRole};
pub use edge::HiddenStateEdge;
pub use object::{most_specific_owner, LogicalObject, ObjectKind, Representation, SourceBinding};
pub use policy::AttentionLayerPolicy;
pub use roles::{NormPlacement, OperandRole};
pub use surface::ExecutionSurface;

/// Current system-graph schema. Bump on any breaking change.
///
/// v2: [`Component`] carries an [`ExecutionSurface`] (V3-G5a) — the
/// deletion invariant's missing half. A v1 graph deserialises with
/// `execution: None`, which [`execution_completeness`] reports as
/// incomplete.
///
/// v3: the execution surface distinguishes an *absent* operation from one
/// declared as the identity, and records the post-norm epsilon as a
/// judgment ([`larql_models::config::PostNormEps`]) rather than a bare
/// number. A v2 graph cannot be reinterpreted into v3: it wrote
/// `post_norm_eps` unconditionally, so a recorded value equal to
/// `norm_eps` is ambiguous between "declared distinct" and "shares" —
/// exactly the distinction v3 exists to keep. Such containers are
/// refused and must be re-encoded, never silently upgraded.
///
/// v4: the execution surface carries the judged *embedding
/// normalisation* ([`larql_models::config::EmbeddingNorm`]). A v3 graph
/// deserialises with `embedding_norm: None`, which under absence ≠
/// identity is the definite claim "this model has no such operation" —
/// wrong for any family that does, and silent because the norm is
/// weightless and no operand contradicts it. Refuse and re-encode.
///
/// v5: the norm surface carries a complete
/// [`larql_models::config::NormSpec`] per *site* (pre, post, final)
/// instead of a model-scope kind/epsilon/offset. Muse-Glimmer proved
/// twice that norm facts are per-site: its post-norms use a different
/// epsilon (1e-8 vs 1e-5) and its final norm a different weight offset
/// (0.0 vs 1.0, centred layers vs an ordinary final norm). A v4 graph
/// records one offset for every site, which is simply wrong for any
/// such family and unrecoverable from the graph alone.
pub const GRAPH_SCHEMA: u32 = 5;

/// The complete executable-system description.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SystemGraph {
    /// Always [`GRAPH_SCHEMA`].
    pub schema: u32,
    /// Components, sorted by id.
    pub components: Vec<Component>,
    /// Logical objects, sorted by id.
    pub objects: Vec<LogicalObject>,
    /// Cross-component hidden-state interfaces.
    pub edges: Vec<HiddenStateEdge>,
}

/// A structural defect in a graph — returned by [`SystemGraph::validate`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum GraphDefect {
    DuplicateComponentId(String),
    DuplicateObjectId(String),
    /// An object names a component the graph does not carry.
    ObjectOrphaned {
        object: String,
        component: String,
    },
    /// An edge endpoint names a missing component or object.
    EdgeOrphaned {
        detail: String,
    },
    /// An object carries no source binding — nothing implements it.
    ObjectUnbound(String),
}

impl SystemGraph {
    /// Structural validation: ids unique, every reference resolvable,
    /// every object physically bound.
    pub fn validate(&self) -> Vec<GraphDefect> {
        let mut defects = Vec::new();
        let mut component_ids = std::collections::BTreeSet::new();
        for component in &self.components {
            if !component_ids.insert(component.id.as_str()) {
                defects.push(GraphDefect::DuplicateComponentId(component.id.clone()));
            }
        }
        let mut object_ids = std::collections::BTreeSet::new();
        for object in &self.objects {
            if !object_ids.insert(object.id.as_str()) {
                defects.push(GraphDefect::DuplicateObjectId(object.id.clone()));
            }
            if !component_ids.contains(object.component.as_str()) {
                defects.push(GraphDefect::ObjectOrphaned {
                    object: object.id.clone(),
                    component: object.component.clone(),
                });
            }
            if object.source_bindings.is_empty() {
                defects.push(GraphDefect::ObjectUnbound(object.id.clone()));
            }
        }
        for edge in &self.edges {
            if !component_ids.contains(edge.producer_component.as_str()) {
                defects.push(GraphDefect::EdgeOrphaned {
                    detail: format!("producer component `{}`", edge.producer_component),
                });
            }
            if !component_ids.contains(edge.consumer_component.as_str()) {
                defects.push(GraphDefect::EdgeOrphaned {
                    detail: format!("consumer component `{}`", edge.consumer_component),
                });
            }
            if !object_ids.contains(edge.consumer_object.as_str()) {
                defects.push(GraphDefect::EdgeOrphaned {
                    detail: format!("consumer object `{}`", edge.consumer_object),
                });
            }
        }
        defects
    }
}
