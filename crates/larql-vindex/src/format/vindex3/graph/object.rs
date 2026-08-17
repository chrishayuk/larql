//! Logical objects, their representations, and physical source bindings.

use serde::{Deserialize, Serialize};

/// What a logical object *is* in the execution graph. The vocabulary is
/// architectural, not familial — any decoder has a stack, most have an
/// embedding and a head; a perception component has a tower and an adapter;
/// a drafter has a feature projector implementing its tap interface.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ObjectKind {
    /// Token embedding table.
    Embedding,
    /// The per-layer transformer stack (attention + FFN + norms).
    DecoderStack,
    /// The final norm before the output head.
    FinalNorm,
    /// The vocabulary projection.
    OutputHead,
    /// A perception encoder stack (incl. its own norms/patch machinery).
    PerceptionTower,
    /// The projection(s) adapting perception output into the consumer.
    PerceptionAdapter,
    /// The fusion/projection implementing the consumer side of a
    /// hidden-state interface (e.g. a drafter's tap-fusion projector).
    FeatureProjector,
    /// The routed experts of a decoder stack — every layer's expert bank
    /// as one logical object, bound per layer (`model.layers.{L}.mlp.
    /// experts`), carved out of the stack by binding specificity (see
    /// [`most_specific_owner`]). Its own object because it is the unit
    /// residency, representation choice and remote placement address; the
    /// stack keeps the router, which is dense.
    ExpertBank,
}

impl ObjectKind {
    /// Snake-case name, doubling as the object-id suffix.
    pub fn name(self) -> &'static str {
        match self {
            Self::Embedding => "embedding",
            Self::DecoderStack => "decoder_stack",
            Self::FinalNorm => "final_norm",
            Self::OutputHead => "output_head",
            Self::PerceptionTower => "perception_tower",
            Self::PerceptionAdapter => "perception_adapter",
            Self::FeatureProjector => "feature_projector",
            Self::ExpertBank => "expert_bank",
        }
    }
}

/// One logical operand of the execution graph.
///
/// Identity is conceptual (`draft.target_feature_projector`); everything
/// physical hangs off [`SourceBinding`]s and [`Representation`]s, so a
/// second materialisation never changes what the object *is*.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LogicalObject {
    /// `{component}.{kind}` — stable, conceptual, unique in the graph.
    pub id: String,
    /// Component this object belongs to (a [`super::Component::id`]).
    pub component: String,
    pub kind: ObjectKind,
    /// Physical tensors implementing this object, by name prefix. Multiple
    /// bindings are normal — a tower binds its layers, pre/post norms and
    /// patch machinery as one logical stack.
    pub source_bindings: Vec<SourceBinding>,
    /// Materialisations of this object. At minimum the canonical one.
    pub representations: Vec<Representation>,
}

/// Where an object's bytes physically come from.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct SourceBinding {
    /// Artifact (inventory) name holding the tensors.
    pub artifact: String,
    /// Tensor-name prefix within that artifact (`model.vision_tower.layers`).
    pub tensor_prefix: String,
    pub tensors: usize,
    pub bytes: u64,
}

/// One materialisation of a logical object.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Representation {
    /// Storage encoding, verbatim from the source (`BF16`, `Q6_K`, …).
    pub encoding: String,
    pub fidelity: Fidelity,
}

/// Whether a representation is the reference or an approximation of it.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Fidelity {
    /// Bit-authoritative: what the release ships as ground truth.
    Canonical,
    /// A lossy re-encoding selected by a profile.
    Approximate,
}

impl SourceBinding {
    /// Whether `source_name` is this binding's prefix itself or lies under
    /// it at a segment boundary (`model.layers` owns `model.layers.0.x`,
    /// not `model.layers_extra.0`).
    pub fn covers(&self, source_name: &str) -> bool {
        source_name == self.tensor_prefix
            || source_name
                .strip_prefix(&self.tensor_prefix)
                .is_some_and(|rest| rest.starts_with('.'))
    }
}

/// The object owning `source_name`: among every binding of every object
/// that covers the name, the one with the **longest** prefix wins. This is
/// the single membership rule of the graph — it lets an expert bank bound
/// at `model.layers.3.mlp.experts` carve its tensors out of a stack bound
/// at `model.layers` with no exclusion lists, and it is what encode, verify
/// and the operand tables all consult so none can disagree.
pub fn most_specific_owner<'a>(
    objects: impl IntoIterator<Item = &'a LogicalObject>,
    source_name: &str,
) -> Option<&'a LogicalObject> {
    objects
        .into_iter()
        .filter_map(|object| {
            object
                .source_bindings
                .iter()
                .filter(|b| b.covers(source_name))
                .map(|b| b.tensor_prefix.len())
                .max()
                .map(|depth| (depth, object))
        })
        .max_by_key(|(depth, _)| *depth)
        .map(|(_, object)| object)
}
