//! Closure defects and the op-plan outcome.

use super::super::graph::{ObjectKind, OperandRole};
use serde::Serialize;

#[allow(unused_imports)]
use super::*;

/// Why a plan could not be built — each variant names the exact operand
/// or fact, so a refusal is a work item, not a mystery.
#[derive(Debug, Clone, PartialEq, Serialize)]
pub enum ClosureDefect {
    /// The component has no (complete) execution surface.
    MissingSurface { component: String },
    /// The primary text component declares a vocabulary (a head surface)
    /// but has neither an output-head tensor nor a declared reuse of the
    /// embedding: it cannot produce a logit (#743).
    NoOutputHead { component: String },
    /// The component has no per-layer attention policy table.
    MissingAttentionTable { component: String },
    /// A stack tensor no operand role classifies.
    UnclassifiedOperand { object: String, tensor: String },
    /// An operand classified into a role that lives in another object
    /// kind — an expert operand in the stack, a router in the bank.
    MisplacedOperand {
        object: String,
        tensor: String,
        belongs_in: ObjectKind,
    },
    /// An operand exists whose op the surface does not carry — the
    /// container physically requires a primitive its semantics lack.
    OperandImpliesAbsentOp {
        object: String,
        tensor: String,
        required_primitive: String,
    },
    /// An op the surface/placement implies has no operand.
    MissingOperand { layer: usize, role: OperandRole },
    /// The declared FFN schedule and the operand evidence disagree for a
    /// layer. The surface declares which layers are routed
    /// (`dense_prefix_layers`, else every layer of a MoE surface); the
    /// expert bank and router operands are evidence that the declaration
    /// is honoured, never the authority. A routed layer whose bank is
    /// absent used to plan as dense with no defect — a missing expert
    /// bank must not quietly turn a MoE layer into an MLP — and a
    /// declared-dense prefix layer with routed operands is the same
    /// disagreement the other way round.
    FfnIdentityMismatch {
        layer: usize,
        declared: FfnIdentity,
        evidence: FfnIdentity,
    },
    /// Two tensors classified into the same role of the same layer.
    DuplicateOperand { layer: usize, role: OperandRole },
    /// An operand's stored shape contradicts the surface's geometry.
    GeometryMismatch {
        tensor: String,
        expected: Vec<usize>,
        actual: Vec<usize>,
    },
    /// A non-stack executable object with an unexpected tensor estate.
    ObjectShape { object: String, detail: String },
    /// The per-layer dense-FFN width declaration cannot be honoured: it
    /// covers the wrong number of layers, or names a width of zero or
    /// wider than the component's dense width. Refused before any layer
    /// is shaped against it — a width nobody can hold must not become a
    /// silently uniform plan.
    FfnWidthDeclaration { component: String, detail: String },
    /// Two declared facts that cannot both hold. Refused rather than
    /// resolved in favour of either: choosing one would execute a program
    /// the checkpoint's other declaration contradicts.
    ContradictoryDeclaration { component: String, detail: String },
    /// The structure requires a semantic fact nothing has established.
    ///
    /// Distinct from [`Self::MissingOperand`]: no tensor is absent, and
    /// the plan would build. It would simply execute a value nobody
    /// judged — the failure mode where an identity or inherited default
    /// is numerically plausible but semantically unfounded, so the
    /// program looks executable and is quietly wrong.
    UnjudgedSemantic {
        component: String,
        /// The fact, named as the surface names it.
        fact: String,
        /// The structure that makes it load-bearing.
        required_by: String,
    },
    /// The surface states the fact exactly, and this build has no
    /// lowering for it.
    ///
    /// The opposite of [`Self::UnjudgedSemantic`] and the distinction
    /// matters: there, nothing established the value and executing would
    /// be guessing; here the value is established and the OP SET is what
    /// is missing. Recognising a shape is not implementing it, and the
    /// two must not read alike — a reader who cannot tell them apart
    /// cannot tell "we do not know what this model does" from "we know
    /// exactly what it does and cannot yet do it".
    ///
    /// Refusing is the point. Every alternative lowering of a shape this
    /// build does not implement is numerically plausible and
    /// semantically wrong, and produces fluent output rather than an
    /// error — the same reason `MoeRouterKind::Sigmoid` and
    /// `PositionPolicy::Relative` refuse at their own match arms.
    UnimplementedSemantic {
        component: String,
        /// The fact, named as the surface names it.
        fact: String,
        /// What the container can already state about it, so a reader
        /// knows the gap is in execution and not in representation.
        representable_as: String,
    },
}

impl std::fmt::Display for ClosureDefect {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::FfnWidthDeclaration { component, detail } => write!(
                f,
                "component {component}: per-layer FFN width declaration refused: {detail}"
            ),
            Self::ContradictoryDeclaration { component, detail } => {
                write!(f, "component {component}: contradictory declaration: {detail}")
            }
            Self::MissingSurface { component } => {
                write!(f, "component `{component}` has no complete execution surface")
            }
            Self::NoOutputHead { component } => write!(
                f,
                "component `{component}` is the primary text model and declares a vocabulary, but has \
                 no output-head tensor and does not reuse its embedding: it cannot produce a logit"
            ),
            Self::UnimplementedSemantic {
                component,
                fact,
                representable_as,
            } => write!(
                f,
                "component `{component}`: {fact} is representable ({representable_as}) and this                  build has no lowering for it — refused rather than lowered as a shape it is not"
            ),
            Self::MissingAttentionTable { component } => {
                write!(f, "component `{component}` has no per-layer attention policy table")
            }
            Self::UnclassifiedOperand { object, tensor } => {
                write!(f, "unclassified executable operand: {object}/{tensor}")
            }
            Self::FfnIdentityMismatch {
                layer,
                declared,
                evidence,
            } => write!(
                f,
                "layer {layer}: the surface declares a {} FFN but the operands are those of a {} \
                 one — a missing expert bank does not make a routed layer dense, and stray \
                 routed operands do not make a dense-prefix layer routed",
                declared.name(),
                evidence.name()
            ),
            Self::MisplacedOperand {
                object,
                tensor,
                belongs_in,
            } => write!(
                f,
                "misplaced operand: {object}/{tensor} belongs in the {} object",
                belongs_in.name()
            ),
            Self::OperandImpliesAbsentOp {
                object,
                tensor,
                required_primitive,
            } => write!(
                f,
                "unrepresented executable operand: {object}/{tensor} — required primitive: {required_primitive}"
            ),
            Self::MissingOperand { layer, role } => {
                write!(f, "layer {layer}: no operand for role {role:?}")
            }
            Self::DuplicateOperand { layer, role } => {
                write!(f, "layer {layer}: two operands claim role {role:?}")
            }
            Self::GeometryMismatch {
                tensor,
                expected,
                actual,
            } => write!(
                f,
                "geometry mismatch: `{tensor}` is {actual:?}, surface implies {expected:?}"
            ),
            Self::ObjectShape { object, detail } => write!(f, "object `{object}`: {detail}"),
            Self::UnjudgedSemantic {
                component,
                fact,
                required_by,
            } => write!(
                f,
                "component `{component}`: {fact} is not judged, and {required_by} requires it"
            ),
        }
    }
}

/// The planning outcome: the plan exists **only** when closure holds.
#[derive(Debug, Serialize)]
pub struct OpPlanOutcome {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub plan: Option<ComponentOpPlan>,
    pub defects: Vec<ClosureDefect>,
}

impl OpPlanOutcome {
    pub fn closed(&self) -> bool {
        self.defects.is_empty()
    }
}
