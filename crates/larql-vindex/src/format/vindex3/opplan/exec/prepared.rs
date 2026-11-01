//! Operands lowered into the backend's execution form, once.
//!
//! A [`ComponentOpPlan`] names its operands; it does not hold them.
//! Turning those names into arithmetic-ready weights — widening,
//! re-quantising to the backend's declared format, and handing the
//! backend a chance to place them on a device — is the expensive step,
//! and it is *model-shaped*, not request-shaped.
//!
//! Before this module both traversals loaded operands as they went:
//! [`DecodeSession`](super::decode::DecodeSession) built its own set at
//! construction, and the batch traversal called `store.load(...)` per
//! layer (per *position*, for norms). A server that batch-prefills and
//! then decodes therefore materialised the whole model twice per
//! request — measured at 3.8 s + 3.3 s against 0.13 s of actual decode
//! on a 3 B container.
//!
//! [`PreparedOperands`] is that state made explicit. It is deliberately
//! *not* a cache inside the operand loader: residency is a fact about a
//! served model, and hiding it behind a memoised loader would leave
//! device placement, accounting, and slicing with nowhere to live.
//!
//! # Composition with the operand seam
//!
//! Preparation resolves through an [`OperandSource`], not the bare
//! store, so a prepared image is "the **effective** operands for this
//! source" — base representation plus whatever overlay it carries.
//! That keeps the two seams orthogonal and in the right order:
//!
//! ```text
//! base representation + overlay → OperandSource → PreparedOperands → executor
//! ```
//!
//! An image is therefore immutable *for the source it was prepared
//! from*: a session composing new edits prepares its own view rather
//! than mutating the shared one, so one image can serve every
//! concurrent request that shares its overlay.
//!
//! # Slicing
//!
//! Preparation takes an [`ExecutionSlice`] because a VINDEX3 component
//! is not only ever executed whole. A shard that owns layers 10–19, an
//! attention-only node, or an expert server all want *part* of the same
//! plan prepared, and none of them should pay for operands they will
//! never execute. `Full` is the common case; the variants below are the
//! seam the decoupled surfaces grow from, and preparation refuses a
//! slice the plan cannot satisfy rather than silently preparing less.

use super::backend::{MatrixClass, NormCall, PlanBackend};
use super::hyper_connection::{SiteWeights, HC_SCALE_LEN};
use super::operands::OperandSource;
use crate::error::VindexError;
use crate::format::vindex3::opplan::planned::{Operation, PlannedOperand};
use larql_models::config::NormType;

use super::super::{AttnResSiteOp, ComponentOpPlan, HcSiteOp, LayerPlan, NormOp};
use super::attention_residual;
use larql_models::config::{HyperConnection, HyperConnectionWeights};

mod accessors;
mod census;
mod mixers;
mod mla;
mod observation;
mod residual_sites;
mod selection;
pub use census::*;
pub(super) use mixers::*;
pub use mla::*;
pub use observation::*;
pub(super) use residual_sites::*;
pub use selection::*;

mod loading;
mod residency;

/// Which part of a component's program to prepare.
///
/// The plan is the authority for what exists; a slice says which of it
/// this process is responsible for executing. Preparing a slice loads
/// only that slice's operands.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum ExecutionSlice {
    /// Embedding, every layer, final norm and head — a whole model.
    Full,
    /// Local stack with dense FFN matrices placed at an external provider.
    DenseFfnCoordinator,
    /// Local attention, router and endpoints, with expert transforms placed remotely.
    RoutedExpertCoordinator,
    /// Only expert rows in half-open layer and expert ranges.
    RoutedExperts {
        start: usize,
        end: usize,
        expert_start: usize,
        expert_end: usize,
    },
    /// Dense FFN matrices only, prepared by `PreparedDenseFfns`.
    DenseFfns { start: usize, end: usize },
    /// Only token embedding and final norm/head, for a distributed coordinator.
    /// No transformer layer operands are loaded.
    Endpoints,
    /// Layers `[start, end)` of the stack and nothing else: no
    /// embedding, no final norm, no head. Hidden states in, hidden
    /// states out — the shape a layer-range shard executes.
    LayerRange { start: usize, end: usize },
    /// Embedding, layers `[0, end)`, then the component's **own** final
    /// norm and output head — a reduced-depth model that still speaks
    /// the target's token vocabulary.
    ///
    /// Not a [`Self::LayerRange`] with the ends bolted on. A shard is a
    /// hidden-state transform and composes with other shards; this is a
    /// complete model that happens to be shallower, and it owns both
    /// ends precisely so its logits are comparable to the target's. The
    /// distinction is the whole point: a drafter is a different semantic
    /// object, not a slice with options.
    ///
    /// `Draft { end: plan.layers.len() }` must be observationally
    /// identical to [`Self::Full`] — that equivalence is the gate the
    /// variant has to pass before any reduced depth is believed.
    ///
    /// Prefix-only by construction. Selecting a *scattered* subset of a
    /// hybrid stack is not merely a coarser approximation: omitted
    /// recurrent layers own state transitions that later layers consume,
    /// so `{0, 8, 16, ...}` is not yet a defined program. That is a
    /// separate rung, and this variant deliberately cannot express it.
    Draft { end: usize },
}

impl ExecutionSlice {
    /// The layer indices this slice covers, as a half-open range.
    pub fn layers(&self, plan: &ComponentOpPlan) -> std::ops::Range<usize> {
        match self {
            Self::Full | Self::DenseFfnCoordinator | Self::RoutedExpertCoordinator => {
                0..plan.layers.len()
            }
            Self::Endpoints => 0..0,
            Self::LayerRange { start, end }
            | Self::DenseFfns { start, end }
            | Self::RoutedExperts { start, end, .. } => *start..*end,
            Self::Draft { end } => 0..*end,
        }
    }

    /// Whether the slice carries the stack's ends — embedding on the
    /// way in, final norm and output head on the way out.
    pub fn is_whole_stack(&self) -> bool {
        matches!(
            self,
            Self::Full
                | Self::DenseFfnCoordinator
                | Self::RoutedExpertCoordinator
                | Self::Endpoints
                | Self::Draft { .. }
        )
    }

    pub(super) fn contains(&self, plan: &ComponentOpPlan, operand: &PlannedOperand) -> bool {
        let in_layer = operand
            .layer
            .is_some_and(|l| self.layers(plan).contains(&l));
        let ffn = operand.operation == Operation::Project(MatrixClass::FfnProjection);
        match self {
            Self::RoutedExperts { .. } => {
                in_layer && operand.operation == Operation::ExpertBankSlice
            }
            Self::RoutedExpertCoordinator => !matches!(
                operand.operation,
                Operation::ExpertBankSlice | Operation::ExpertProject { .. }
            ),
            Self::DenseFfns { .. } => in_layer && ffn,
            Self::DenseFfnCoordinator => !ffn,
            _ => in_layer || (operand.layer.is_none() && self.is_whole_stack()),
        }
    }

    /// Refuse a slice the plan cannot satisfy. A shard asked for layers
    /// the model does not have is a deployment error, and preparing
    /// "as much as exists" would serve a silently wrong submodel — the
    /// same failure the V3 load options used to have.
    pub(super) fn validate(&self, plan: &ComponentOpPlan) -> Result<(), VindexError> {
        if matches!(self, Self::DenseFfnCoordinator | Self::DenseFfns { .. }) {
            super::dense_ffn::validate_plan(plan)?;
        }
        if matches!(
            self,
            Self::RoutedExpertCoordinator | Self::RoutedExperts { .. }
        ) {
            super::routed_experts::validate_plan(plan)?;
        }
        if let Self::RoutedExperts {
            start,
            end,
            expert_start,
            expert_end,
        } = self
        {
            if start >= end
                || *end > plan.layers.len()
                || expert_start >= expert_end
                || plan.layers[*start..*end].iter().any(|l| {
                    l.ffn
                        .as_ref()
                        .and_then(|f| f.routed())
                        .is_none_or(|r| *expert_end > r.experts)
                })
            {
                return Err(VindexError::Parse(
                    "routed worker layer or expert range outside plan".into(),
                ));
            }
        }
        if let Self::Draft { end } = self {
            if *end == 0 {
                return Err(VindexError::Parse(
                    "a draft slice must execute at least one layer — `Draft { end: 0 }` is the \
                     embedding and head with nothing between them"
                        .to_string(),
                ));
            }
            if *end > plan.layers.len() {
                return Err(VindexError::Parse(format!(
                    "draft slice of depth {end} is deeper than component `{}`, which has {} layers",
                    plan.component,
                    plan.layers.len()
                )));
            }
            return Ok(());
        }
        let (Self::LayerRange { start, end } | Self::DenseFfns { start, end }) = self else {
            return Ok(());
        };
        if start >= end {
            return Err(VindexError::Parse(format!(
                "execution slice {start}..{end} is empty — a slice must cover at least one layer"
            )));
        }
        if *end > plan.layers.len() {
            return Err(VindexError::Parse(format!(
                "execution slice {start}..{end} is outside component `{}`, which has {} layers",
                plan.component,
                plan.layers.len()
            )));
        }
        Ok(())
    }
}

/// One norm site's weight, held resident beside the op that names it.
pub(super) struct PreparedNorm {
    op: NormOp,
    weight: Vec<f32>,
}

impl PreparedNorm {
    fn load(op: &NormOp, store: OperandSource<'_>) -> Result<Self, VindexError> {
        Ok(Self {
            op: op.clone(),
            weight: store.load(&op.weight)?,
        })
    }

    pub(super) fn apply<B: PlanBackend + ?Sized>(&self, backend: &B, x: &[f32]) -> Vec<f32> {
        backend.norm(NormCall {
            kind: self.op.kind,
            x,
            weight: &self.weight,
            weight_offset: self.op.weight_offset,
            eps: self.op.eps,
        })
    }

    /// The norm's kind, so a consumer decomposing a write through it can
    /// refuse a kind whose linearisation it does not know.
    pub(super) fn kind(&self) -> NormType {
        self.op.kind
    }

    /// The learned weight, before the offset.
    pub(super) fn weight(&self) -> &[f32] {
        &self.weight
    }

    /// The offset added to the weight (`1.0` on a Gemma-style `(1 + w)`
    /// norm, `0.0` otherwise).
    pub(super) fn weight_offset(&self) -> f32 {
        self.op.weight_offset
    }

    pub(super) fn eps(&self) -> f64 {
        self.op.eps
    }
}

/// Why a whole-stack image cannot be prepared over a hyper-connected
/// component that declares no head object (GLM-5.3-Flash ships none and
/// its `mhc` is unexplained): there is no declared reduction from the
/// bundle to one vector before the final norm, and this build does not
/// invent one. A layer-range image runs the layers without one.
const HC_HEADLESS_WHOLE_STACK: &str = "declares the hyper-connection residual topology and no \
     hyper_connection_head object: a whole-stack image has no declared reduction from the bundle \
     to one vector before the final norm, and this build does not invent one. A layer-range image \
     runs the layers without a head";

/// A per-layer output scalar has one judged meaning — multiply the
/// `[hidden]` residual after the FFN add — and no hyper-connected
/// checkpoint declares one. Applied to a bundle it is unjudged.
const HC_WITH_LAYER_SCALE: &str = "carries a layer scale under the hyper-connection residual \
     topology; a scalar applied to a bundle of streams is unjudged, and no hyper-connected \
     checkpoint declares one";

/// One hyper-connection site's three operands, resident as f32 glue.
///
/// Glue, not matrix traffic, in this wave: stage one's mix projection
/// runs through the reference matvec in f32 (`hyper_connection::mix_projection`),
/// no backend format class describes a `[(2 + hc)·hc, hc·hidden]`
/// operand, and the residency census counts it beside the norms. A
/// backend-formatted mix projection is a later performance rung.
pub(super) struct PreparedHcSite {
    mix_fn: Vec<f32>,
    base: Vec<f32>,
    scale: Vec<f32>,
}

impl PreparedHcSite {
    fn load(
        op: &HcSiteOp,
        store: OperandSource<'_>,
        hc: HyperConnection,
        hidden: usize,
        what: &str,
    ) -> Result<Self, VindexError> {
        let mix_fn = store.load(&op.mix_fn)?;
        let base = store.load(&op.base)?;
        let scale = store.load(&op.scale)?;
        // Closure checked these shapes at plan time; the loaded lengths
        // are checked again so a store answering with a different tensor
        // cannot reach the stages, whose asserts are debug-only in spirit.
        let mix_rows = HyperConnectionWeights::mix_rows_for(hc.streams);
        let expect = |name: &str, got: usize, want: usize| {
            if got == want {
                Ok(())
            } else {
                Err(VindexError::Parse(format!(
                    "{what}: {name} holds {got} values, the declared geometry needs {want}"
                )))
            }
        };
        expect("mix_fn", mix_fn.len(), mix_rows * hc.streams * hidden)?;
        expect("base", base.len(), mix_rows)?;
        expect("scale", scale.len(), HC_SCALE_LEN)?;
        Ok(Self {
            mix_fn,
            base,
            scale,
        })
    }

    pub(super) fn weights(&self) -> SiteWeights<'_> {
        SiteWeights {
            mix_fn: &self.mix_fn,
            base: &self.base,
            scale: &self.scale,
        }
    }

    fn glue_bytes(&self) -> usize {
        std::mem::size_of_val(&self.mix_fn[..])
            + std::mem::size_of_val(&self.base[..])
            + std::mem::size_of_val(&self.scale[..])
    }
}

/// The two sites one hyper-connected layer wraps its sublayers in.
pub(super) struct PreparedHyperConnection {
    pub(super) attention: PreparedHcSite,
    pub(super) ffn: PreparedHcSite,
}

/// Why a whole-stack image cannot be prepared over an attention-residual
/// component that owns no exit object.
///
/// Unlike the hyper-connection head — which GLM-5.3-Flash declines to
/// ship, so its absence is a checkpoint's choice — the exit reduction is
/// REQUIRED by this declaration: the stack's last layer leaves a prefix
/// and a snapshot history, and something has to collapse them before the
/// final norm. The plan report refuses such a component one step
/// earlier, by the exit's own name; this states the same fact where the
/// operands are read.
const ATTN_RES_EXITLESS_WHOLE_STACK: &str = "declares the attention-residual topology and owns no \
     attention_residual_exit object: a whole-stack image has no declared reduction from the \
     snapshot history to the one vector the final norm reads, and the declaration requires one";

/// A per-layer output scalar has one judged meaning — multiply the
/// `[hidden]` residual after the FFN add — and the topology carries a
/// history beside that residual which the scalar says nothing about.
/// No attention-residual checkpoint declares one.
const ATTN_RES_WITH_LAYER_SCALE: &str = "carries a layer scale under the attention-residual \
     residual topology; whether it also scales the snapshot history is unjudged, and no \
     attention-residual checkpoint declares one";

/// One attention-residual site's operand pair, resident as f32 glue —
/// two `[hidden]` vectors, counted beside the norms for the same reason
/// the hyper-connection sites are.
pub(super) struct PreparedAttnResSite {
    norm: Vec<f32>,
    proj: Vec<f32>,
}

impl PreparedAttnResSite {
    fn load(
        op: &AttnResSiteOp,
        store: OperandSource<'_>,
        hidden: usize,
        what: &str,
    ) -> Result<Self, VindexError> {
        let norm = store.load(&op.norm)?;
        let proj = store.load(&op.proj)?;
        // Closure checked `[hidden]` and `[1, hidden]` at plan time; the
        // loaded lengths are checked again so a store answering with a
        // different tensor cannot reach the reduction.
        for (name, got) in [("norm", norm.len()), ("proj", proj.len())] {
            if got != hidden {
                return Err(VindexError::Parse(format!(
                    "{what}: {name} holds {got} values, the component's width is {hidden}"
                )));
            }
        }
        Ok(Self { norm, proj })
    }

    pub(super) fn pair(&self) -> attention_residual::SitePair<'_> {
        attention_residual::SitePair {
            norm: &self.norm,
            proj: &self.proj,
        }
    }

    fn glue_bytes(&self) -> usize {
        std::mem::size_of_val(&self.norm[..]) + std::mem::size_of_val(&self.proj[..])
    }
}

/// One layer's two attention-residual sites.
pub(super) struct PreparedAttentionResidual {
    pub(super) attention: PreparedAttnResSite,
    pub(super) ffn: PreparedAttnResSite,
}

impl PreparedAttentionResidual {
    fn for_layer(
        layer: &LayerPlan,
        declared: bool,
        hidden: usize,
        store: OperandSource<'_>,
    ) -> Result<Option<Self>, VindexError> {
        match (&layer.attention_residual, declared) {
            (None, false) => Ok(None),
            (Some(sites), true) => {
                if layer.layer_scale.is_some() {
                    return Err(VindexError::Parse(format!(
                        "layer {} {ATTN_RES_WITH_LAYER_SCALE}",
                        layer.layer
                    )));
                }
                let l = layer.layer;
                Ok(Some(Self {
                    attention: PreparedAttnResSite::load(
                        &sites.attention,
                        store,
                        hidden,
                        &format!("layer {l}'s attention-residual attention site"),
                    )?,
                    ffn: PreparedAttnResSite::load(
                        &sites.ffn,
                        store,
                        hidden,
                        &format!("layer {l}'s attention-residual mlp site"),
                    )?,
                }))
            }
            (Some(_), false) => Err(VindexError::Parse(format!(
                "layer {} carries attention-residual sites but the component declares no block \
                 size; the op plan never produces this",
                layer.layer
            ))),
            (None, true) => Err(VindexError::Parse(format!(
                "layer {} carries no attention-residual sites under a component that declares \
                 the topology; closure requires all four operands on every layer",
                layer.layer
            ))),
        }
    }

    fn glue_bytes(&self) -> usize {
        self.attention.glue_bytes() + self.ffn.glue_bytes()
    }
}

/// The stack's exit reduction: the same operation as a site's, run once
/// over the whole snapshot history before the final norm.
pub(super) struct PreparedAttnResExit {
    site: PreparedAttnResSite,
    norm_eps: f64,
}

// `pub(super)` so the executor's own tests reach the seams it defines.
#[cfg(test)]
pub(super) mod tests;
