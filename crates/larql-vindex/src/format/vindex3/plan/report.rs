//! Output types for the semantic representability plan (`larql vindex3 plan`).
//!
//! A plan is a statement over one or more inspected artifacts of **exactly
//! why** the current VINDEX3 container schema can or cannot faithfully encode
//! them. Findings are typed twice:
//!
//! - **category** — what kind of statement this is: representable,
//!   value-mismatched between authorities, unrepresented, or a declared
//!   cross-component interface.
//! - **semantic class** — how much losing it would matter. `consumed` in the
//!   inventory is *not* sufficient: a key can be consumed and still resolve
//!   to a different value than the checkpoint declared, which is why
//!   mismatch findings compare values, not statuses.
//!
//! The verdict is fail-closed: any mismatched/unrepresented finding whose
//! class is execution-, tensor-, or interface-semantic — or whose key the
//! registry has never seen (`unknown`) — makes the plan inadmissible.

use serde::{Deserialize, Serialize};

use super::carriage::Carriage;
use crate::format::vindex3::graph::SystemGraph;

/// Current plan schema. Bump on any breaking change to these types.
///
/// v2: the plan carries the built [`SystemGraph`] — representability is
/// defined as "the graph builder placed it", and the graph is the proof.
/// Interfaces are reported as resolved edges rather than candidate guesses.
///
/// v3: findings about config keys carry a [`Carriage`] stage, and the
/// census reports every declared key rather than only the unconsumed
/// ones — so `unrepresented: N` is a count against a stated denominator
/// instead of a lower bound. Adds the `training_only` and `alias`
/// semantic classes.
pub const PLAN_SCHEMA: u32 = 3;

/// The whole-system plan over every artifact given.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SystemPlan {
    /// Always [`PLAN_SCHEMA`].
    pub schema: u32,
    /// One entry per inspected artifact, in input order.
    pub artifacts: Vec<ArtifactPlan>,
    /// Cross-component interfaces resolved into graph edges.
    pub interfaces: Vec<InterfacePlan>,
    /// True iff no blocking finding exists anywhere in the system.
    ///
    /// **Model completeness**, not executability. A container can be
    /// inadmissible here and still run text generation perfectly — see
    /// [`capabilities`](Self::capabilities).
    pub admissible: bool,
    /// Per-capability execution admissibility, each judged on its own
    /// dependency closure.
    ///
    /// Defaulted on deserialisation so plans written before capability
    /// scoping existed still read, and skipped when empty so those plans
    /// serialise byte-identically.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub capabilities: Vec<super::capability::CapabilityStatus>,
    pub summary: PlanSummary,
    /// The system graph the builder produced — what G3 encodes.
    pub graph: SystemGraph,
}

/// Counts over every finding in the system.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct PlanSummary {
    pub representable: usize,
    pub mismatched: usize,
    pub unrepresented: usize,
    pub interfaces: usize,
    /// Findings that make the plan inadmissible (see [`Finding::blocks`]).
    pub blocking: usize,
}

/// The plan for one physical artifact (one inventory).
///
/// Deliberately named *artifact*, not *component*: the Glimmer target
/// checkpoint physically carries both the text model and the vision tower.
/// Logical components live inside findings' `component` field; the physical
/// file boundary is never authoritative.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ArtifactPlan {
    /// Artifact name (the inspected directory's stem).
    pub name: String,
    pub model_type: String,
    pub findings: Vec<Finding>,
}

/// One statement about one subject.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Finding {
    pub category: FindingCategory,
    pub class: SemanticClass,
    /// Logical component the subject belongs to (`text`, `vision`, `root`).
    pub component: String,
    /// What the finding is about: a config key path, a tensor group, or a
    /// topology aspect.
    pub subject: String,
    /// Value the checkpoint declares, when the finding compares values.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub declared: Option<serde_json::Value>,
    /// Value this build resolves, when the finding compares values.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub resolved: Option<serde_json::Value>,
    /// How far VINDEX3 carries the subject past the parser, for findings
    /// about a config key. `None` for findings about tensors, topology or
    /// interfaces, where carriage is not the question being asked.
    ///
    /// This is the field that keeps `consumed` from being read as
    /// `represented`: a key can be parsed and go no further, and the plan
    /// now says which.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub carriage: Option<Carriage>,
    pub detail: String,
}

impl Finding {
    /// Whether this finding makes the plan inadmissible.
    ///
    /// Representable findings never block. Interface findings always carry
    /// [`SemanticClass::InterfaceSemantic`], so the class test covers them.
    pub fn blocks(&self) -> bool {
        !matches!(self.category, FindingCategory::Representable) && self.class.is_critical()
    }
}

/// What kind of statement a finding makes.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum FindingCategory {
    /// The schema has a faithful home for this subject.
    Representable,
    /// Declared and resolved authorities disagree on the value.
    Mismatched,
    /// The subject has no home in the current schema.
    Unrepresented,
    /// A declared cross-component dependency.
    Interface,
}

/// How much losing or corrupting the subject would matter.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum SemanticClass {
    /// Reviewed and safe to drop (registry-listed, justified per entry).
    IgnoredSafe,
    /// Identity/training facts inert for a forward pass.
    MetadataOnly,
    /// Declared for training and inert at inference — an auxiliary-loss
    /// coefficient, a router-logit output switch. Distinct from
    /// [`Self::MetadataOnly`]: these describe the *forward pass of
    /// training*, so the reason they are safe to drop is that inference
    /// does not run that path, not that they are identity strings.
    TrainingOnly,
    /// A redundant spelling of a fact the checkpoint declares elsewhere
    /// and a parser reads. Safe only while the canonical key is present
    /// and agrees — the registry names it and the gate checks both, so
    /// `alias` cannot become a way to silence a key.
    Alias,
    /// Changes what a forward pass computes (norm eps, rope, scales…).
    ExecutionSemantic,
    /// Describes stored operands (shapes, widths, component topology).
    TensorSemantic,
    /// Declares a cross-component contract (taps, token protocol).
    InterfaceSemantic,
    /// A model component has been **positively identified** from the
    /// checkpoint semantics, and this build has no implementation for it.
    ///
    /// Blocking, exactly like [`Self::Unknown`] — the difference is not
    /// severity but knowledge:
    ///
    /// ```text
    /// Unknown               nobody has established what this means yet
    /// UnsupportedComponent  we know what component this configures,
    ///                       and this build does not implement it
    /// ```
    ///
    /// which is the difference between uncertainty and known engineering
    /// work. Nine keys naming one absent component is one job; reporting
    /// them as nine anonymous unknowns says nothing about how much.
    ///
    /// Deliberately distinct from every neighbouring idea, because the
    /// engineering implication differs in each case. This is **not** a
    /// parser chore, a spelling alias, a checked default, or an inactive
    /// declaration. It means *there is machinery missing*.
    ///
    /// **The registration rule is positive evidence of component
    /// ownership, never plausible adjacency** — see
    /// [`UNSUPPORTED_COMPONENT_KEYS`](super::semantics::UNSUPPORTED_COMPONENT_KEYS).
    /// A prefix match or a regex is a discovery tool and must not become
    /// the authority; that is precisely how `indexer_rope_interleave`
    /// ended up filed under general RoPE, where acting on it would have
    /// re-paired the whole model's rotary.
    UnsupportedComponent,
    /// Not in the registry — nobody has judged it, so it blocks.
    Unknown,
}

impl SemanticClass {
    /// Classes that make a non-representable finding blocking.
    pub fn is_critical(self) -> bool {
        matches!(
            self,
            Self::ExecutionSemantic
                | Self::TensorSemantic
                | Self::InterfaceSemantic
                | Self::UnsupportedComponent
                | Self::Unknown
        )
    }
}

/// A declared cross-component interface, resolved into a graph edge.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InterfacePlan {
    /// Producing component (graph id, e.g. `target`).
    pub producer_component: String,
    /// Tapped layers, in declaration order.
    pub producer_layers: Vec<usize>,
    /// Consuming component (graph id, e.g. `draft`).
    pub consumer_component: String,
    /// Logical object implementing consumption (`draft.feature_projector`).
    pub consumer_object: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub block_size: Option<usize>,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn representable_findings_never_block() {
        let finding = Finding {
            category: FindingCategory::Representable,
            class: SemanticClass::ExecutionSemantic,
            component: "text".into(),
            subject: "x".into(),
            declared: None,
            resolved: None,
            carriage: None,
            detail: String::new(),
        };
        assert!(!finding.blocks());
    }

    #[test]
    fn critical_classes_block_outside_representable() {
        for class in [
            SemanticClass::ExecutionSemantic,
            SemanticClass::TensorSemantic,
            SemanticClass::InterfaceSemantic,
            SemanticClass::Unknown,
        ] {
            let finding = Finding {
                category: FindingCategory::Unrepresented,
                class,
                component: "text".into(),
                subject: "x".into(),
                declared: None,
                resolved: None,
                carriage: None,
                detail: String::new(),
            };
            assert!(finding.blocks(), "{class:?} must block");
        }
    }

    #[test]
    fn benign_classes_do_not_block() {
        for class in [SemanticClass::IgnoredSafe, SemanticClass::MetadataOnly] {
            let finding = Finding {
                category: FindingCategory::Unrepresented,
                class,
                component: "root".into(),
                subject: "x".into(),
                declared: None,
                resolved: None,
                carriage: None,
                detail: String::new(),
            };
            assert!(!finding.blocks(), "{class:?} must not block");
        }
    }

    #[test]
    fn classes_serialise_snake_case() {
        assert_eq!(
            serde_json::to_string(&SemanticClass::ExecutionSemantic).unwrap(),
            "\"execution_semantic\""
        );
        assert_eq!(
            serde_json::to_string(&FindingCategory::Unrepresented).unwrap(),
            "\"unrepresented\""
        );
    }
}
