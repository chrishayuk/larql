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

use super::accounting::{
    declared_resident_for, expectations, reconcile, BlockGeometry, Bound, Expectation, Observed,
    Reconciliation, ResidencyBudget, ResourceLedger,
};
use super::backend::{MatrixClass, NormCall, PlanBackend, WeightFormat, WeightSlice};
use super::experts::FfnOperands;
use super::hyper_connection::{HeadWeights, SiteWeights, HC_HEAD_SCALE_LEN, HC_SCALE_LEN};
use super::operands::{OperandSource, SourceStamp};
use super::realization::{
    realization_residency, RealizationId, RealizationRecord, RepresentationFacts, SelectionReason,
    SelectionRefusals,
};
use super::weights::{load_weight, LoadedWeight};
use super::AttentionOperands;
use crate::error::VindexError;
use crate::format::vindex3::opplan::planned::Operation;
use crate::format::vindex3::represent::codec::CodecRegistry;
use crate::format::vindex3::represent::nvfp4_pack::CodecIdentity;

use super::super::conv_qkv::ConvQkvOp;
use super::super::{
    AttnResSiteOp, ComponentOpPlan, GatedDeltaOp, HcSiteOp, HyperConnectionLayerOp, KdaOp,
    LayerAttention, LayerPlan, Mamba2Op, MlaOp, NormOp, OperandRef, OutputOp,
};
use super::attention_residual;
use larql_models::config::{HyperConnection, HyperConnectionWeights, ResidualTopology};

/// Which part of a component's program to prepare.
///
/// The plan is the authority for what exists; a slice says which of it
/// this process is responsible for executing. Preparing a slice loads
/// only that slice's operands.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum ExecutionSlice {
    /// Embedding, every layer, final norm and head — a whole model.
    Full,
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
            Self::Full => 0..plan.layers.len(),
            Self::LayerRange { start, end } => *start..*end,
            Self::Draft { end } => 0..*end,
        }
    }

    /// Whether the slice carries the stack's ends — embedding on the
    /// way in, final norm and output head on the way out.
    pub fn is_whole_stack(&self) -> bool {
        matches!(self, Self::Full | Self::Draft { .. })
    }

    /// Refuse a slice the plan cannot satisfy. A shard asked for layers
    /// the model does not have is a deployment error, and preparing
    /// "as much as exists" would serve a silently wrong submodel — the
    /// same failure the V3 load options used to have.
    pub(super) fn validate(&self, plan: &ComponentOpPlan) -> Result<(), VindexError> {
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
        let Self::LayerRange { start, end } = self else {
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

impl PreparedAttnResExit {
    /// Present only on a whole-stack image of an attention-residual
    /// component, and REQUIRED there: see [`ATTN_RES_EXITLESS_WHOLE_STACK`].
    fn load(
        plan: &ComponentOpPlan,
        hidden: usize,
        store: OperandSource<'_>,
    ) -> Result<Self, VindexError> {
        let Some(op) = &plan.attention_residual_exit else {
            return Err(VindexError::Parse(format!(
                "component `{}` {ATTN_RES_EXITLESS_WHOLE_STACK}",
                plan.component
            )));
        };
        Ok(Self {
            site: PreparedAttnResSite::load(
                &AttnResSiteOp {
                    norm: op.norm.clone(),
                    proj: op.proj.clone(),
                },
                store,
                hidden,
                "the attention-residual exit",
            )?,
            // The exit's RMSNorm is constructed from the component's
            // `rms_norm_eps`, exactly as every site's is, so it is ONE
            // component value — read through the same derivation the
            // hyper-connection head uses, which refuses a stack whose
            // layers disagree rather than picking one of them. Taking
            // the first layer's would have been a silent choice on
            // exactly the plan that needed a loud one.
            norm_eps: component_norm_eps(plan)?,
        })
    }

    pub(super) fn pair(&self) -> attention_residual::SitePair<'_> {
        self.site.pair()
    }

    /// The component's declared norm epsilon, which the exit reduction
    /// scores at.
    pub(super) fn norm_eps(&self) -> f64 {
        self.norm_eps
    }
}

impl PreparedHyperConnection {
    /// The layer's sites under the component's topology — present
    /// exactly when both agree, and a plan where they disagree is one
    /// the builder never produced.
    fn for_layer(
        layer: &LayerPlan,
        topology: Option<HyperConnection>,
        hidden: usize,
        store: OperandSource<'_>,
    ) -> Result<Option<Self>, VindexError> {
        match (&layer.hyper_connection, topology) {
            (None, None) => Ok(None),
            (Some(sites), Some(hc)) => {
                if layer.layer_scale.is_some() {
                    return Err(VindexError::Parse(format!(
                        "layer {} {HC_WITH_LAYER_SCALE}",
                        layer.layer
                    )));
                }
                Ok(Some(Self::load(sites, hc, hidden, layer.layer, store)?))
            }
            (Some(_), None) => Err(VindexError::Parse(format!(
                "layer {} carries hyper-connection sites but the component declares a single \
                 residual stream; the op plan never produces this",
                layer.layer
            ))),
            (None, Some(_)) => Err(VindexError::Parse(format!(
                "the component declares the hyper-connection topology but layer {} carries no \
                 sites; closure requires them on every layer",
                layer.layer
            ))),
        }
    }

    fn load(
        sites: &HyperConnectionLayerOp,
        hc: HyperConnection,
        hidden: usize,
        layer: usize,
        store: OperandSource<'_>,
    ) -> Result<Self, VindexError> {
        Ok(Self {
            attention: PreparedHcSite::load(
                &sites.attention,
                store,
                hc,
                hidden,
                &format!("layer {layer} attention site"),
            )?,
            ffn: PreparedHcSite::load(
                &sites.ffn,
                store,
                hc,
                hidden,
                &format!("layer {layer} ffn site"),
            )?,
        })
    }

    fn glue_bytes(&self) -> usize {
        self.attention.glue_bytes() + self.ffn.glue_bytes()
    }
}

/// The head's own reduction operands (a different operation from a
/// site's: one row per stream, one scalar, no Sinkhorn), plus the norm
/// epsilon its mix projection runs at.
pub(super) struct PreparedHcHead {
    reduce_fn: Vec<f32>,
    base: Vec<f32>,
    scale: f32,
    norm_eps: f64,
}

impl PreparedHcHead {
    /// Present only on a whole-stack image of a hyper-connected
    /// component, and REQUIRED there: see [`HC_HEADLESS_WHOLE_STACK`].
    fn load(
        plan: &ComponentOpPlan,
        hc: HyperConnection,
        hidden: usize,
        store: OperandSource<'_>,
    ) -> Result<Self, VindexError> {
        let Some(op) = &plan.hyper_connection_head else {
            return Err(VindexError::Parse(format!(
                "component `{}` {HC_HEADLESS_WHOLE_STACK}",
                plan.component
            )));
        };
        let reduce_fn = store.load(&op.reduce_fn)?;
        let base = store.load(&op.base)?;
        let scale = store.load(&op.scale)?;
        if reduce_fn.len() != hc.streams * hc.streams * hidden || base.len() != hc.streams {
            return Err(VindexError::Parse(format!(
                "component `{}`: the hyper-connection head's operands do not hold the head's \
                 geometry ([{}, {}] and [{}])",
                plan.component,
                hc.streams,
                hc.streams * hidden,
                hc.streams
            )));
        }
        let scale = match scale[..] {
            [scale] => scale,
            ref other => {
                return Err(VindexError::Parse(format!(
                    "component `{}`: the hyper-connection head's scale holds {} values; the \
                     head reads exactly {HC_HEAD_SCALE_LEN}",
                    plan.component,
                    other.len()
                )))
            }
        };
        Ok(Self {
            reduce_fn,
            base,
            scale,
            norm_eps: component_norm_eps(plan)?,
        })
    }

    pub(super) fn weights(&self) -> HeadWeights<'_> {
        HeadWeights {
            reduce_fn: &self.reduce_fn,
            base: &self.base,
            scale: self.scale,
        }
    }

    /// The component's declared norm epsilon — stage one's `norm_eps`
    /// for the head's mix projection.
    pub(super) fn norm_eps(&self) -> f64 {
        self.norm_eps
    }

    fn glue_bytes(&self) -> usize {
        std::mem::size_of_val(&self.reduce_fn[..]) + std::mem::size_of_val(&self.base[..])
    }
}

/// The component's ONE declared norm epsilon, read from the layers that
/// carry it. The head's mix projection runs at the component's
/// `rms_norm_eps`, which the plan carries per layer as a single
/// component fact; layers that disagree are a plan this build has not
/// judged, so the derivation refuses rather than picking one.
fn component_norm_eps(plan: &ComponentOpPlan) -> Result<f64, VindexError> {
    let mut layers = plan.layers.iter();
    let Some(first) = layers.next() else {
        return Err(VindexError::Parse(format!(
            "component `{}` has no layers to read a norm epsilon from",
            plan.component
        )));
    };
    let eps = first.declared_norm_eps;
    if let Some(other) = layers.find(|l| l.declared_norm_eps != eps) {
        return Err(VindexError::Parse(format!(
            "component `{}`: layer {} declares norm eps {} where layer {} declares {}; the \
             hyper-connection head needs one component value",
            plan.component, other.layer, other.declared_norm_eps, first.layer, eps
        )));
    }
    Ok(eps)
}

/// One layer's operands, lowered into the backend's execution form.
pub(super) struct PreparedLayer {
    /// `None` under post-norm placement — the sublayer reads the raw
    /// residual and the wrap norm applies to its output instead.
    pub(super) pre_attention: Option<PreparedNorm>,
    pub(super) attention: PreparedAttention,
    pub(super) post_attention: Option<PreparedNorm>,
    /// Absent on a mixer-only (Mamba2) layer — the plan carries no FFN
    /// program there, so there is nothing to prepare and nothing to run.
    pub(super) pre_ffn: Option<PreparedNorm>,
    pub(super) ffn: Option<FfnOperands>,
    pub(super) post_ffn: Option<PreparedNorm>,
    /// The layer's output scalar, when the plan carries one.
    pub(super) layer_scale: Option<f32>,
    /// The two Sinkhorn sites, present exactly when the component
    /// declares the topology (wave 19a).
    pub(super) hyper_connection: Option<PreparedHyperConnection>,
    /// The two attention-residual sites, present exactly when the
    /// component declares the topology.
    pub(super) attention_residual: Option<PreparedAttentionResidual>,
}

impl PreparedLayer {
    /// This layer's norm weights — f32 glue, counted so the census adds
    /// up to the whole image rather than to the parts that were easy.
    /// The hyper-connection sites count here too, by the decision that
    /// made them f32 glue.
    fn glue_bytes(&self) -> usize {
        let norm = |n: &PreparedNorm| std::mem::size_of_val(&n.weight[..]);
        self.pre_attention.as_ref().map_or(0, norm)
            + self.pre_ffn.as_ref().map_or(0, norm)
            + self.post_attention.as_ref().map_or(0, norm)
            + self.post_ffn.as_ref().map_or(0, norm)
            + self
                .attention_residual
                .as_ref()
                .map_or(0, PreparedAttentionResidual::glue_bytes)
            + self
                .hyper_connection
                .as_ref()
                .map_or(0, PreparedHyperConnection::glue_bytes)
    }
}

/// Which attention-class operator a prepared layer holds operands for.
///
/// An enum, not `Option<AttentionOperands>` and not "softmax unless
/// proven otherwise": a layer runs exactly one operator, and the
/// alternative spellings both make "I could not tell" indistinguishable
/// from "it is softmax". Qwen3.8 is 48 layers where that difference is
/// the whole model.
///
/// Chosen from the op plan's `LayerAttention`, which the op builder
/// derived from operand EVIDENCE — so the operands loaded here and the
/// operator dispatched later cannot disagree.
pub(super) enum PreparedAttention {
    Softmax(Box<AttentionOperands>),
    GatedDelta(Box<GatedDeltaOperands>),
    Mamba2(Box<Mamba2Operands>),
    ConvQkv(Box<ConvQkvOperands>),
    Kda(Box<KdaOperands>),
    Mla(Box<MlaOperands>),
}

impl PreparedAttention {
    /// Every bound operand, paired by the loader that bound it — refusing
    /// a plan whose attention is a different program from the prepared one.
    pub(super) fn bound<'a>(
        &'a self,
        attention: &'a LayerAttention,
    ) -> Result<Vec<Bound<'a>>, VindexError> {
        Ok(match (self, attention) {
            (Self::Softmax(ops), LayerAttention::Softmax(op)) => ops.bound(op),
            (Self::GatedDelta(ops), LayerAttention::GatedDelta(op)) => ops.bound(op),
            (Self::Mamba2(ops), LayerAttention::Mamba2(op)) => ops.bound(op),
            (Self::ConvQkv(ops), LayerAttention::ConvQkv(op)) => ops.bound(op),
            (Self::Kda(ops), LayerAttention::Kda(op)) => ops.bound(op),
            (Self::Mla(ops), LayerAttention::Mla(op)) => ops.bound(op),
            _ => {
                return Err(VindexError::Parse(
                    "the prepared attention and the plan's attention are different programs"
                        .to_string(),
                ))
            }
        })
    }

    /// Every matrix this attention holds resident — what a pinned
    /// projection realization is checked against.
    pub(super) fn matrices(&self) -> Vec<&LoadedWeight> {
        match self {
            Self::Softmax(ops) => ops.loaded_matrices(),
            Self::GatedDelta(ops) => ops.loaded_matrices().to_vec(),
            Self::Mamba2(ops) => ops.loaded_matrices().to_vec(),
            Self::ConvQkv(ops) => ops.loaded_matrices().to_vec(),
            Self::Kda(ops) => ops.loaded_matrices().to_vec(),
            Self::Mla(ops) => ops.loaded_matrices().to_vec(),
        }
    }

    /// Matrix operands for device placement.
    ///
    /// A recurrence contributes none: its nine operands are elementwise
    /// glue and a depthwise convolution, not the matrix traffic a device
    /// backend holds resident — and no device backend runs this operator
    /// yet, so placing them would reserve memory nothing reads.
    fn weight_slices(&self) -> Vec<WeightSlice<'_>> {
        match self {
            Self::Softmax(ops) => ops.weight_slices(),
            Self::GatedDelta(_)
            | Self::Mamba2(_)
            | Self::ConvQkv(_)
            | Self::Kda(_)
            | Self::Mla(_) => Vec::new(),
        }
    }
}

/// The nine operands a Gated DeltaNet layer reads, loaded once.
///
/// The five projections carry a `LoadedWeight` and the four glue
/// operands a `Vec<f32>`, which is the split the measurements draw: 11.1
/// GB of matrix against 6 MB of convolution kernel, gate bias and norm.
pub(super) struct GatedDeltaOperands {
    pub(super) op: GatedDeltaOp,
    in_proj_qkv: LoadedWeight,
    in_proj_a: LoadedWeight,
    in_proj_b: LoadedWeight,
    in_proj_z: LoadedWeight,
    out_proj: LoadedWeight,
    conv1d: Vec<f32>,
    a_log: Vec<f32>,
    dt_bias: Vec<f32>,
    norm: Vec<f32>,
    norm_eps: f32,
}

impl GatedDeltaOperands {
    pub(super) fn bound<'a>(&'a self, op: &'a GatedDeltaOp) -> Vec<Bound<'a>> {
        vec![
            Bound::one(&op.in_proj_qkv, &self.in_proj_qkv),
            Bound::one(&op.in_proj_a, &self.in_proj_a),
            Bound::one(&op.in_proj_b, &self.in_proj_b),
            Bound::one(&op.in_proj_z, &self.in_proj_z),
            Bound::one(&op.out_proj, &self.out_proj),
        ]
    }

    fn load(
        op: &GatedDeltaOp,
        store: OperandSource<'_>,
        format: FormatFor<'_>,
        norm_eps: f32,
    ) -> Result<Self, VindexError> {
        // Per operand, and the answers differ WITHIN this layer: at
        // Qwen3.8's shapes `in_proj_qkv` is 105 MB and stays compact
        // while `in_proj_a` is 0.5 MB and does not. A single format for
        // the layer could not express that, and the version of this that
        // loaded everything f32 is what left 48 of 64 layers widened.
        let matrix = |r: &OperandRef| load_weight(store, r, format(r)?);
        let glue = |r: &OperandRef| store.load(r);
        Ok(Self {
            op: op.clone(),
            in_proj_qkv: matrix(&op.in_proj_qkv)?,
            in_proj_a: matrix(&op.in_proj_a)?,
            in_proj_b: matrix(&op.in_proj_b)?,
            in_proj_z: matrix(&op.in_proj_z)?,
            out_proj: matrix(&op.out_proj)?,
            conv1d: glue(&op.conv1d)?,
            a_log: glue(&op.a_log)?,
            dt_bias: glue(&op.dt_bias)?,
            norm: glue(&op.norm)?,
            norm_eps,
        })
    }

    /// The five matrices, for residency ACCOUNTING — not for device
    /// placement, which [`PreparedAttention::weight_slices`] still
    /// declines to offer for a recurrence no device kernel runs.
    pub(super) fn loaded_matrices(&self) -> [&LoadedWeight; 5] {
        [
            &self.in_proj_qkv,
            &self.in_proj_a,
            &self.in_proj_b,
            &self.in_proj_z,
            &self.out_proj,
        ]
    }

    /// The four f32 operands that are not matrix traffic.
    pub(super) fn glue_bytes(&self) -> usize {
        [&self.conv1d, &self.a_log, &self.dt_bias, &self.norm]
            .iter()
            .map(|v| std::mem::size_of_val(&v[..]))
            .sum()
    }

    pub(super) fn weights(&self) -> Result<super::gated_delta::GatedDeltaWeights<'_>, VindexError> {
        // Geometry from the op, never from the slice length: a resident
        // slab is page-padded and can be longer than the matrix.
        Ok(super::gated_delta::GatedDeltaWeights {
            in_proj_qkv: matrix_rows(&self.in_proj_qkv, &self.op.in_proj_qkv)?,
            in_proj_a: matrix_rows(&self.in_proj_a, &self.op.in_proj_a)?,
            in_proj_b: matrix_rows(&self.in_proj_b, &self.op.in_proj_b)?,
            in_proj_z: matrix_rows(&self.in_proj_z, &self.op.in_proj_z)?,
            out_proj: matrix_rows(&self.out_proj, &self.op.out_proj)?,
            conv1d: &self.conv1d,
            a_log: &self.a_log,
            dt_bias: &self.dt_bias,
            norm: &self.norm,
            norm_eps: self.norm_eps,
        })
    }
}

/// The nine operands a Mamba2 layer reads, loaded once.
///
/// The two projections carry a `LoadedWeight`; the seven glue operands
/// are f32 — the same matrix/glue split the delta operands draw, at this
/// family's shapes (a 6448×1536 fused projection against kilobytes of
/// conv taps and per-head scalars).
pub(super) struct Mamba2Operands {
    pub(super) op: Mamba2Op,
    in_proj: LoadedWeight,
    out_proj: LoadedWeight,
    conv1d: Vec<f32>,
    conv1d_bias: Option<Vec<f32>>,
    a_log: Vec<f32>,
    d: Vec<f32>,
    dt_bias: Vec<f32>,
    norm: Option<Vec<f32>>,
    norm_eps: f32,
}

impl Mamba2Operands {
    pub(super) fn bound<'a>(&'a self, op: &'a Mamba2Op) -> Vec<Bound<'a>> {
        vec![
            Bound::one(&op.in_proj, &self.in_proj),
            Bound::one(&op.out_proj, &self.out_proj),
        ]
    }

    fn load(
        op: &Mamba2Op,
        store: OperandSource<'_>,
        format: FormatFor<'_>,
    ) -> Result<Self, VindexError> {
        let matrix = |r: &OperandRef| load_weight(store, r, format(r)?);
        let glue = |r: &OperandRef| store.load(r);
        Ok(Self {
            op: op.clone(),
            in_proj: matrix(&op.in_proj)?,
            out_proj: matrix(&op.out_proj)?,
            conv1d: glue(&op.conv1d)?,
            conv1d_bias: op.conv1d_bias.as_ref().map(glue).transpose()?,
            a_log: glue(&op.a_log)?,
            d: glue(&op.d)?,
            dt_bias: glue(&op.dt_bias)?,
            norm: op
                .gated_norm
                .as_ref()
                .map(|n| glue(&n.weight))
                .transpose()?,
            // The epsilon travels with the gated norm's own NormOp; a
            // mixer with `rms_norm: false` has no norm and the value is
            // never read.
            norm_eps: op.gated_norm.as_ref().map_or(0.0, |n| n.eps as f32),
        })
    }

    /// The two matrices, for residency accounting.
    pub(super) fn loaded_matrices(&self) -> [&LoadedWeight; 2] {
        [&self.in_proj, &self.out_proj]
    }

    /// The f32 operands that are not matrix traffic.
    pub(super) fn glue_bytes(&self) -> usize {
        let opt = |v: &Option<Vec<f32>>| v.as_ref().map_or(0, |v| std::mem::size_of_val(&v[..]));
        [&self.conv1d, &self.a_log, &self.d, &self.dt_bias]
            .iter()
            .map(|v| std::mem::size_of_val(&v[..]))
            .sum::<usize>()
            + opt(&self.conv1d_bias)
            + opt(&self.norm)
    }

    pub(super) fn weights(&self) -> Result<super::mamba2::Mamba2Weights<'_>, VindexError> {
        Ok(super::mamba2::Mamba2Weights {
            in_proj: matrix_rows(&self.in_proj, &self.op.in_proj)?,
            out_proj: matrix_rows(&self.out_proj, &self.op.out_proj)?,
            conv1d: &self.conv1d,
            conv1d_bias: self.conv1d_bias.as_deref(),
            a_log: &self.a_log,
            d: &self.d,
            dt_bias: &self.dt_bias,
            norm: self.norm.as_deref(),
            norm_eps: self.norm_eps,
        })
    }
}

/// The four operands a conv-QKV attention layer reads, loaded once —
/// the same matrix/glue split as the mixer's: two dense projections
/// against kilobytes of conv taps.
pub(super) struct ConvQkvOperands {
    pub(super) op: super::super::conv_qkv::ConvQkvOp,
    in_proj: LoadedWeight,
    out_proj: LoadedWeight,
    conv1d: Vec<f32>,
    conv1d_bias: Option<Vec<f32>>,
}

impl ConvQkvOperands {
    pub(super) fn bound<'a>(&'a self, op: &'a ConvQkvOp) -> Vec<Bound<'a>> {
        vec![
            Bound::one(&op.in_proj, &self.in_proj),
            Bound::one(&op.out_proj, &self.out_proj),
        ]
    }

    fn load(
        op: &super::super::conv_qkv::ConvQkvOp,
        store: OperandSource<'_>,
        format: FormatFor<'_>,
    ) -> Result<Self, VindexError> {
        let matrix = |r: &OperandRef| load_weight(store, r, format(r)?);
        let glue = |r: &OperandRef| store.load(r);
        Ok(Self {
            op: op.clone(),
            in_proj: matrix(&op.in_proj)?,
            out_proj: matrix(&op.out_proj)?,
            conv1d: glue(&op.conv1d)?,
            conv1d_bias: op.conv1d_bias.as_ref().map(glue).transpose()?,
        })
    }

    /// The two matrices, for residency accounting.
    pub(super) fn loaded_matrices(&self) -> [&LoadedWeight; 2] {
        [&self.in_proj, &self.out_proj]
    }

    /// The f32 operands that are not matrix traffic.
    pub(super) fn glue_bytes(&self) -> usize {
        std::mem::size_of_val(&self.conv1d[..])
            + self
                .conv1d_bias
                .as_ref()
                .map_or(0, |v| std::mem::size_of_val(&v[..]))
    }

    pub(super) fn weights(&self) -> Result<super::conv_qkv::ConvQkvWeights<'_>, VindexError> {
        Ok(super::conv_qkv::ConvQkvWeights {
            in_proj: matrix_rows(&self.in_proj, &self.op.in_proj)?,
            out_proj: matrix_rows(&self.out_proj, &self.op.out_proj)?,
            conv1d: &self.conv1d,
            conv1d_bias: self.conv1d_bias.as_deref(),
        })
    }
}

/// The fifteen operands a KDA layer reads, loaded once.
///
/// The same matrix/glue split every recurrence here draws, at KDA's own
/// proportions: four wide projections (q, k, v and the output, the whole
/// of this layer's matrix traffic) against fifteen kilobytes of
/// convolution taps, low-rank gate factors, decay parameters and the
/// gated norm's weight. The gate factorisations are matrices too, but at
/// `[rank, hidden]` and `[width, rank]` they are three orders of
/// magnitude smaller than the four, and the executor consumes them f32 —
/// so they load as glue, which is what they cost.
pub(super) struct KdaOperands {
    pub(super) op: super::super::KdaOp,
    q_proj: LoadedWeight,
    k_proj: LoadedWeight,
    v_proj: LoadedWeight,
    o_proj: LoadedWeight,
    q_conv1d: Vec<f32>,
    k_conv1d: Vec<f32>,
    v_conv1d: Vec<f32>,
    f_a_proj: Vec<f32>,
    f_b_proj: Vec<f32>,
    g_a_proj: Vec<f32>,
    g_b_proj: Vec<f32>,
    b_proj: Vec<f32>,
    a_log: Vec<f32>,
    dt_bias: Vec<f32>,
    o_norm: Vec<f32>,
    norm_eps: f32,
}

impl KdaOperands {
    pub(super) fn bound<'a>(&'a self, op: &'a KdaOp) -> Vec<Bound<'a>> {
        vec![
            Bound::one(&op.q_proj, &self.q_proj),
            Bound::one(&op.k_proj, &self.k_proj),
            Bound::one(&op.v_proj, &self.v_proj),
            Bound::one(&op.out_proj, &self.o_proj),
        ]
    }

    fn load(
        op: &super::super::KdaOp,
        store: OperandSource<'_>,
        format: FormatFor<'_>,
        norm_eps: f32,
    ) -> Result<Self, VindexError> {
        let matrix = |r: &OperandRef| load_weight(store, r, format(r)?);
        let glue = |r: &OperandRef| store.load(r);
        Ok(Self {
            op: op.clone(),
            q_proj: matrix(&op.q_proj)?,
            k_proj: matrix(&op.k_proj)?,
            v_proj: matrix(&op.v_proj)?,
            o_proj: matrix(&op.out_proj)?,
            q_conv1d: glue(&op.q_conv1d)?,
            k_conv1d: glue(&op.k_conv1d)?,
            v_conv1d: glue(&op.v_conv1d)?,
            f_a_proj: glue(&op.f_a_proj)?,
            f_b_proj: glue(&op.f_b_proj)?,
            g_a_proj: glue(&op.g_a_proj)?,
            g_b_proj: glue(&op.g_b_proj)?,
            b_proj: glue(&op.b_proj)?,
            a_log: glue(&op.a_log)?,
            dt_bias: glue(&op.dt_bias)?,
            o_norm: glue(&op.o_norm)?,
            norm_eps,
        })
    }

    /// The four matrices, for residency accounting.
    pub(super) fn loaded_matrices(&self) -> [&LoadedWeight; 4] {
        [&self.q_proj, &self.k_proj, &self.v_proj, &self.o_proj]
    }

    /// The f32 operands that are not matrix traffic.
    pub(super) fn glue_bytes(&self) -> usize {
        [
            &self.q_conv1d,
            &self.k_conv1d,
            &self.v_conv1d,
            &self.f_a_proj,
            &self.f_b_proj,
            &self.g_a_proj,
            &self.g_b_proj,
            &self.b_proj,
            &self.a_log,
            &self.dt_bias,
            &self.o_norm,
        ]
        .iter()
        .map(|v| std::mem::size_of_val(&v[..]))
        .sum()
    }

    pub(super) fn weights(&self) -> Result<super::kda::KdaWeights<'_>, VindexError> {
        Ok(super::kda::KdaWeights {
            q_proj: matrix_rows(&self.q_proj, &self.op.q_proj)?,
            k_proj: matrix_rows(&self.k_proj, &self.op.k_proj)?,
            v_proj: matrix_rows(&self.v_proj, &self.op.v_proj)?,
            o_proj: matrix_rows(&self.o_proj, &self.op.out_proj)?,
            q_conv1d: &self.q_conv1d,
            k_conv1d: &self.k_conv1d,
            v_conv1d: &self.v_conv1d,
            f_a_proj: &self.f_a_proj,
            f_b_proj: &self.f_b_proj,
            g_a_proj: &self.g_a_proj,
            g_b_proj: &self.g_b_proj,
            b_proj: &self.b_proj,
            a_log: &self.a_log,
            dt_bias: &self.dt_bias,
            o_norm: &self.o_norm,
            norm_eps: self.norm_eps,
            gate_rank: self.op.gate_rank,
        })
    }
}

/// The five operands an MLA layer reads, loaded once — plus the one
/// epsilon its own latent norm runs at.
///
/// That epsilon is why this struct exists in this form. It is not the
/// layer's `rms_norm_eps`: `kv_a_layernorm` takes its class default
/// (`1e-6` against the layer's `1e-5`), and until lift 2 the container
/// could not carry that at all — it lived as a constant inside a
/// family-shaped loader, where deleting the checkpoint could not restore
/// it. Loading REFUSES a container that carries no judged value rather
/// than borrowing the layer's: a norm at the wrong epsilon computes a
/// different function with every shape still closing.
pub(super) struct MlaOperands {
    pub(super) op: super::super::MlaOp,
    q_proj: LoadedWeight,
    kv_a_proj: LoadedWeight,
    kv_b_proj: LoadedWeight,
    o_proj: LoadedWeight,
    kv_a_norm: Vec<f32>,
    kv_a_norm_eps: f64,
}

impl MlaOperands {
    pub(super) fn bound<'a>(&'a self, op: &'a MlaOp) -> Vec<Bound<'a>> {
        vec![
            Bound::one(&op.q_proj, &self.q_proj),
            Bound::one(&op.kv_a_proj, &self.kv_a_proj),
            Bound::one(&op.kv_b_proj, &self.kv_b_proj),
            Bound::one(&op.out_proj, &self.o_proj),
        ]
    }

    fn load(
        op: &super::super::MlaOp,
        store: OperandSource<'_>,
        format: FormatFor<'_>,
    ) -> Result<Self, VindexError> {
        let matrix = |r: &OperandRef| load_weight(store, r, format(r)?);
        let kv_a_norm_eps = op.kv_a_norm_eps.ok_or_else(|| {
            VindexError::Parse(
                "this MLA layer carries no epsilon for its latent norm (`kv_a_layernorm`), \
                 which is NOT the layer's `rms_norm_eps` on any judged checkpoint; refusing \
                 to substitute one"
                    .to_string(),
            )
        })?;
        Ok(Self {
            op: op.clone(),
            q_proj: matrix(&op.q_proj)?,
            kv_a_proj: matrix(&op.kv_a_proj)?,
            kv_b_proj: matrix(&op.kv_b_proj)?,
            o_proj: matrix(&op.out_proj)?,
            kv_a_norm: store.load(&op.kv_a_norm)?,
            kv_a_norm_eps,
        })
    }

    /// The four matrices, for residency accounting.
    pub(super) fn loaded_matrices(&self) -> [&LoadedWeight; 4] {
        [&self.q_proj, &self.kv_a_proj, &self.kv_b_proj, &self.o_proj]
    }

    /// The one f32 operand that is not matrix traffic.
    pub(super) fn glue_bytes(&self) -> usize {
        std::mem::size_of_val(&self.kv_a_norm[..])
    }

    pub(super) fn weights(&self) -> Result<super::mla::MlaWeights<'_>, VindexError> {
        Ok(super::mla::MlaWeights {
            q_proj: matrix_rows(&self.q_proj, &self.op.q_proj)?,
            kv_a_proj: matrix_rows(&self.kv_a_proj, &self.op.kv_a_proj)?,
            kv_b_proj: matrix_rows(&self.kv_b_proj, &self.op.kv_b_proj)?,
            o_proj: matrix_rows(&self.o_proj, &self.op.out_proj)?,
            kv_a_norm: &self.kv_a_norm,
            kv_a_norm_eps: self.kv_a_norm_eps,
        })
    }
}

/// A resident matrix as row ranges, cut to the geometry the op declares.
///
/// The geometry comes from the OP and never from the slice length: a
/// resident slab is page-padded, so `len / in_dim` can exceed the number
/// of rows the matrix has.
fn matrix_rows<'a>(
    w: &'a LoadedWeight,
    r: &OperandRef,
) -> Result<super::cpu::WeightRows<'a>, VindexError> {
    let (out_dim, in_dim) = two_dims(r)?;
    w.slice().rows(out_dim, in_dim)
}

/// A matrix operand's `[out, in]` geometry.
///
/// Fails closed on anything else: a projection is two-dimensional, and a
/// caller that inferred `out_dim` from a slice length instead would read
/// page padding as extra rows.
fn two_dims(r: &OperandRef) -> Result<(usize, usize), VindexError> {
    match r.shape.as_slice() {
        [out_dim, in_dim] => Ok((*out_dim, *in_dim)),
        other => Err(VindexError::Parse(format!(
            "operand `{}` has shape {other:?}; a dense projection is `[out, in]`",
            r.tensor
        ))),
    }
}

/// Resolves the load format for ONE matrix operand.
///
/// A function rather than a value because the question is now per matrix
/// and not per class: a layer hands its q/k/v/o — or its five delta
/// projections — to the same resolver and can get different answers, which
/// is what lets a `48 x 5120` gate stay f32 inside a stack whose `10240 x
/// 5120` projections do not.
pub(super) type FormatFor<'a> = &'a dyn Fn(&OperandRef) -> Result<WeightFormat, VindexError>;

/// Resolve, admit and select a realization for every planned operand
/// `slice` of `plan` will load — BEFORE any byte is read.
///
/// The backend is handed the planned operation and the registry's facts
/// for the stored dtype, never the bytes and never a label to match on.
/// Every refusal is collected, so the caller sees the whole problem and
/// not its first symptom. An operand the container does not hold is
/// skipped here: the loader refuses it by name, as it always has.
pub fn select_realizations<B: PlanBackend + ?Sized>(
    plan: &ComponentOpPlan,
    store: OperandSource<'_>,
    backend: &B,
    slice: &ExecutionSlice,
) -> Result<Vec<RealizationRecord>, VindexError> {
    Ok(select_records(plan, store, backend, slice)?
        .into_iter()
        .map(|(record, _)| record)
        .collect())
}

/// [`select_realizations`], then held against `budget`: while the plan's
/// PHYSICAL working set or per-token touch exceeds it, the record with
/// the largest resident saving among its own candidates is re-selected
/// to the cheaper-resident one the backend had considered, with reason
/// [`SelectionReason::BudgetPolicy`]; when no candidate can bring the
/// plan inside, the whole preparation is refused BEFORE any payload byte
/// with the irreducible deficit and the alternatives that were tried.
/// Under [`ResidencyBudget::UNBOUNDED`] this is exactly
/// [`select_realizations`].
pub fn select_realizations_within<B: PlanBackend + ?Sized>(
    plan: &ComponentOpPlan,
    store: OperandSource<'_>,
    backend: &B,
    slice: &ExecutionSlice,
    budget: &ResidencyBudget,
) -> Result<Vec<RealizationRecord>, VindexError> {
    let mut selected = select_records(plan, store, backend, slice)?;
    let geometry = BlockGeometry::executor();
    let stored_len = |op: &OperandRef| store.stored_len(op);
    let mut switches: Vec<String> = Vec::new();
    loop {
        let records: Vec<RealizationRecord> = selected.iter().map(|(r, _)| r.clone()).collect();
        let priced = expectations(&records, stored_len, geometry);
        let ledger = ResourceLedger::aggregate(&priced);
        let deficit = budget.deficit(&ledger);
        if deficit.is_zero() {
            return Ok(records);
        }
        // The best switch: the largest resident saving any record can make
        // by moving to another of ITS OWN candidates.
        let mut best: Option<(usize, RealizationId, u64)> = None;
        for (i, (record, facts)) in selected.iter().enumerate() {
            let current = record.selection.realization;
            let now = declared_resident_for(
                &record.planned,
                current,
                record.selection.residency,
                geometry,
            );
            for candidate in &record.selection.candidates {
                if *candidate == current {
                    continue;
                }
                let then = declared_resident_for(
                    &record.planned,
                    *candidate,
                    realization_residency(facts, *candidate),
                    geometry,
                );
                if then < now {
                    let saving = now - then;
                    if best.is_none_or(|(_, _, s)| saving > s) {
                        best = Some((i, *candidate, saving));
                    }
                }
            }
        }
        let Some((i, candidate, saving)) = best else {
            return Err(VindexError::Parse(budget_refusal(
                budget, &ledger, &deficit, &priced, &switches,
            )));
        };
        let (record, facts) = &mut selected[i];
        switches.push(format!(
            "`{}` {} → {} (saves {:.2} GB resident)",
            record.planned.operand.tensor,
            record.selection.realization.name(),
            candidate.name(),
            saving as f64 / 1e9
        ));
        record.selection.residency = realization_residency(facts, candidate);
        record.selection.realization = candidate;
        record.selection.reason = SelectionReason::BudgetPolicy;
    }
}

/// The refusal a budget produces: what the plan demands, what the budget
/// allows, the irreducible deficit, the largest committed operands that
/// have nowhere cheaper to go, and every alternative already taken.
fn budget_refusal(
    budget: &ResidencyBudget,
    ledger: &ResourceLedger,
    deficit: &super::accounting::BudgetDeficit,
    priced: &[Expectation],
    switches: &[String],
) -> String {
    const GB: f64 = 1e9;
    const LARGEST: usize = 5;
    let mut out = format!(
        "the plan cannot be prepared within the residency budget before any payload byte: \
         physical working set {:.2} GB (resident {:.2} + staging peak {:.2} + page-in per \
         token {:.2}) against {}; touch per token {:.2} GB against {}; irreducible deficit: \
         {:.2} GB physical, {:.2} GB per token",
        ledger.physical_working_set() as f64 / GB,
        ledger.resident as f64 / GB,
        ledger.transient_peak as f64 / GB,
        ledger.page_in_per_token as f64 / GB,
        budget
            .physical_bytes
            .map(|b| format!("{:.2} GB", b as f64 / GB))
            .unwrap_or_else(|| "no physical limit".to_string()),
        ledger.touch_per_token as f64 / GB,
        budget
            .throughput
            .map(|t| format!("{:.2} GB per token", t.bytes_per_token() as f64 / GB))
            .unwrap_or_else(|| "no throughput limit".to_string()),
        deficit.physical as f64 / GB,
        deficit.touch_per_token as f64 / GB,
    );
    let mut largest: Vec<&Expectation> = priced
        .iter()
        .filter(|e| e.resources().resident > 0)
        .collect();
    largest.sort_by_key(|e| std::cmp::Reverse(e.resources().resident));
    out.push_str("; largest committed operands with no cheaper candidate: ");
    out.push_str(
        &largest
            .iter()
            .take(LARGEST)
            .map(|e| {
                format!(
                    "`{}` {} {:.2} GB",
                    e.operand.tensor,
                    e.realization.name(),
                    e.resources().resident as f64 / GB
                )
            })
            .collect::<Vec<_>>()
            .join(", "),
    );
    if switches.is_empty() {
        out.push_str("; no alternative was cheaper");
    } else {
        out.push_str(&format!(
            "; alternatives already taken ({}): {}",
            switches.len(),
            switches.join(", ")
        ));
    }
    out
}

/// Every planned operand in `slice` with its representation facts and
/// the backend's selection, or the complete list of refusals.
fn select_records<B: PlanBackend + ?Sized>(
    plan: &ComponentOpPlan,
    store: OperandSource<'_>,
    backend: &B,
    slice: &ExecutionSlice,
) -> Result<Vec<(RealizationRecord, RepresentationFacts)>, VindexError> {
    let registry = store.registry();
    let whole = slice.is_whole_stack();
    let range = slice.layers(plan);
    let mut records = Vec::new();
    let mut refusals = Vec::new();
    for planned in plan.planned_operands() {
        let in_scope = match planned.layer {
            Some(layer) => range.contains(&layer),
            None => whole,
        };
        if !in_scope {
            continue;
        }
        let Some(stored) = store.store().stored_dtype(&planned.operand) else {
            continue;
        };
        let mut facts = RepresentationFacts::resolve_declared(
            registry,
            stored,
            planned.declared_representation,
        );
        let label = facts.label.clone();
        if store.is_overridden(&planned.operand) {
            facts = facts.overlaid();
        }
        let provider = facts.registered.as_ref().map(|r| r.identity.clone());
        match backend.select(&planned, &facts) {
            Ok(selection) => records.push((
                RealizationRecord {
                    representation: label.to_string(),
                    provider,
                    planned,
                    selection,
                },
                facts,
            )),
            Err(refusal) => refusals.push(*refusal),
        }
    }
    if refusals.is_empty() {
        Ok(records)
    } else {
        Err(VindexError::Parse(SelectionRefusals(refusals).to_string()))
    }
}

/// The representation pinned for `op` under `operation`.
///
/// An operand with no record is either absent from the container — the
/// loader refuses it by name a moment later, so any answer here is
/// unread — or one the plan's own view failed to list, which is a
/// disagreement between `planned_operands()` and the loader and is
/// refused as such rather than defaulted.
fn pinned_format(
    records: &[RealizationRecord],
    store: OperandSource<'_>,
    op: &OperandRef,
    operation: Operation,
) -> Result<WeightFormat, VindexError> {
    if let Some(record) = records.iter().find(|r| {
        r.planned.operand.object == op.object
            && r.planned.operand.tensor == op.tensor
            && r.planned.operation == operation
    }) {
        return Ok(record.selection.realization.format());
    }
    if store.store().stored_dtype(op).is_none() {
        return Ok(WeightFormat::F32);
    }
    Err(VindexError::Parse(format!(
        "tensor `{}` ({}): the loader resolves an operand the plan's own view does not list — \
         planned_operands() and the loader disagree",
        op.tensor,
        operation.name()
    )))
}

/// The resident form pinned for layer `index`'s bank: a packed bank's
/// one slice pin, or the one form every matrix of a per-expert bank was
/// pinned to — a bank whose experts were pinned differently is a plan
/// the loader cannot bind, and is refused by name.
fn bank_pin(records: &[RealizationRecord], index: usize) -> Result<WeightFormat, VindexError> {
    let mut pins = records
        .iter()
        .filter(|r| {
            r.planned.layer == Some(index)
                && matches!(
                    r.planned.operation,
                    Operation::ExpertBankSlice | Operation::ExpertProject { .. }
                )
        })
        .map(|r| r.selection.realization.format());
    let Some(first) = pins.next() else {
        return Err(VindexError::Parse(format!(
            "layer {index}: a routed FFN with no pinned bank realization — planned_operands() \
             and the loader disagree"
        )));
    };
    if let Some(other) = pins.find(|f| *f != first) {
        return Err(VindexError::Parse(format!(
            "layer {index}: the bank's experts were pinned to {first:?} and {other:?}; one bank \
             binds in one form"
        )));
    }
    Ok(first)
}

/// A component's operands, lowered once for a given slice and backend.
///
/// Immutable for its lifetime: this is the canonical base model. A
/// session that carries an overlay composes *over* these operands
/// rather than mutating them, so one prepared image can serve every
/// concurrent request on the model.
pub struct PreparedOperands {
    /// The registry this image was prepared through — the store's — and
    /// the one execution re-checks its providers against.
    registry: &'static CodecRegistry,
    /// Which effective source this image was compiled from.
    stamp: SourceStamp,
    slice: ExecutionSlice,
    hidden: usize,
    /// Present only for a slice that carries the stack's input end.
    embed_table: Option<Vec<f32>>,
    /// Plan index of `layers[0]`, so a sliced image can still address
    /// the plan's per-layer ops and the KV state's layer rows.
    first_layer: usize,
    layers: Vec<PreparedLayer>,
    final_norm: Option<PreparedNorm>,
    output: Option<(OutputOp, LoadedWeight)>,
    /// One pinned realization per planned operand this image executes,
    /// in the plan's order — the record the trace reads.
    realizations: Vec<RealizationRecord>,
    /// The component's residual topology, carried so the traversal reads
    /// the stream count from the image it executes.
    topology: ResidualTopology,
    /// The head's reduction — present only on a whole-stack image of a
    /// hyper-connected component.
    hyper_connection_head: Option<PreparedHcHead>,
    /// The exit reduction — present only on a whole-stack image of an
    /// attention-residual component, and required there.
    attention_residual_exit: Option<PreparedAttnResExit>,
}

impl PreparedOperands {
    /// Lower `slice` of `plan`'s operands into `backend`'s execution
    /// form, and give the backend its chance to place them (device
    /// residency). Every operand this slice needs is loaded here, and
    /// none of it is loaded again.
    pub fn load<'s, B: PlanBackend + ?Sized>(
        plan: &ComponentOpPlan,
        store: impl Into<OperandSource<'s>>,
        backend: &B,
        slice: ExecutionSlice,
    ) -> Result<Self, VindexError> {
        Self::load_within(plan, store, backend, slice, &ResidencyBudget::UNBOUNDED)
    }

    /// [`Self::load`] under a residency budget: the pins are chosen so the
    /// plan's physical working set and per-token touch fit `budget`, or
    /// the preparation is refused before any payload byte with the
    /// deficit and the alternatives considered — see
    /// [`select_realizations_within`].
    pub fn load_within<'s, B: PlanBackend + ?Sized>(
        plan: &ComponentOpPlan,
        store: impl Into<OperandSource<'s>>,
        backend: &B,
        slice: ExecutionSlice,
        budget: &ResidencyBudget,
    ) -> Result<Self, VindexError> {
        let store = store.into();
        slice.validate(plan)?;
        // **Every declared residual topology is traversable here.**
        // Single-stream always was; hyper-connections joined it in wave
        // 19 when the bundle was witnessed on both the decode step and
        // the batch traversal; attention residuals join it now, their
        // decode (2a) and batch (2b) traversals each witnessed against a
        // Torch oracle transcribed from the reference. The authority
        // this used to consult — `ResidualTopology::unimplemented_reason`
        // — is deleted rather than left answering `None`, so there is no
        // dead refusal here for a reader to consult and conclude from.
        // A topology that cannot be traversed again must bring both the
        // authority and its readers back together.
        //
        // What a hyper-connected image still cannot be is said below by
        // name — a whole-stack image with no declared head reduction, a
        // layer scale under the topology — and the plan report reads the
        // same facts, so a plan it calls executable is one prepared here.
        Self::load_validated(plan, store, backend, slice, budget)
    }

    /// The loader proper, past slice validation. Every operand the slice
    /// needs is loaded here, and none of it is loaded again.
    fn load_validated<B: PlanBackend + ?Sized>(
        plan: &ComponentOpPlan,
        store: OperandSource<'_>,
        backend: &B,
        slice: ExecutionSlice,
        budget: &ResidencyBudget,
    ) -> Result<Self, VindexError> {
        let stamp = store.stamp();
        let whole = slice.is_whole_stack();
        // **Select before any operand is loaded.** Every planned operand
        // this slice executes is resolved against the registry, admitted,
        // and pinned to one realization — or the whole plan is refused
        // with every reason — before a byte of any of them is read.
        let realizations = select_realizations_within(plan, store, backend, &slice, budget)?;
        let embedding = plan.embedding.as_ref().ok_or_else(|| {
            VindexError::Parse(format!(
                "component `{}` has no embedding op — external hidden-state input is a later rung",
                plan.component
            ))
        })?;
        let hidden = embedding.table.shape[1];
        let embed_table = if whole {
            Some(store.load(&embedding.table)?)
        } else {
            None
        };
        let topology = plan.residual_topology;
        let hyper_connection = match topology {
            ResidualTopology::HyperConnection(hc) => Some(hc),
            // Neither of the others is a bundle. An attention-residual
            // plan never reaches this loader at all — `load` refuses it
            // above — and the arm answers what is true of the topology
            // rather than restating that refusal.
            ResidualTopology::SingleStream | ResidualTopology::AttentionResidual { .. } => None,
        };
        // The other topology's declaration, as a flag: its sites need no
        // parameter from it (the pair's geometry closes over the width
        // alone), only the fact that the component declares it.
        let attention_residual = matches!(topology, ResidualTopology::AttentionResidual { .. });

        // The loaders ask by operand and class; the answer is the pin.
        let pinned = |op: &OperandRef, operation: Operation| {
            pinned_format(&realizations, store, op, operation)
        };
        let attention_format =
            |op: &OperandRef| pinned(op, Operation::Project(MatrixClass::AttentionProjection));
        let ffn_format =
            |op: &OperandRef| pinned(op, Operation::Project(MatrixClass::FfnProjection));
        let shared_format = |op: &OperandRef| pinned(op, Operation::SharedExpertProject);
        let head_format = |op: &OperandRef| pinned(op, Operation::OutputHead);

        let range = slice.layers(plan);
        let first_layer = range.start;
        let mut layers = Vec::with_capacity(range.len());
        for index in range.clone() {
            let layer = &plan.layers[index];
            // A packed bank's realization is pinned per layer; a layer
            // with no bank never reads the value, and gets the widened
            // form so nothing compact is implied.
            let bank_format = match layer.ffn.as_ref().and_then(|f| f.routed()) {
                Some(_) => bank_pin(&realizations, index)?,
                None => WeightFormat::F32,
            };
            layers.push(PreparedLayer {
                // Absent under post-norm placement: the sublayer reads
                // the raw residual there. `None` is the program, and the
                // executor skips the site rather than applying identity.
                pre_attention: match &layer.pre_attention_norm {
                    Some(op) => Some(PreparedNorm::load(op, store)?),
                    None => None,
                },
                // The operator is decided here, from the plan, and the
                // operands follow it. No layer is prepared as softmax by
                // default.
                attention: match &layer.attention {
                    LayerAttention::Softmax(op) => PreparedAttention::Softmax(Box::new(
                        AttentionOperands::load(op, store, &attention_format)?,
                    )),
                    // No executor exists for this operator yet. The
                    // operands are bound and the geometry is stated, but
                    // binding is not running: preparing a KDA layer as
                    // anything else would execute the wrong recurrence on
                    // correctly-bound tensors, which is the failure the
                    // separate variant exists to make impossible.
                    // The layer's own pre-attention norm epsilon: KDA's
                    // gated output norm is built as
                    // `FusedRMSNormGated(head_dim, eps=config.rms_norm_eps)`
                    // in the checkpoint's own modeling code, the same
                    // value the layer norms use — unlike MLA's latent
                    // norm below, which is exactly why that one is
                    // carried per-op and this one is not.
                    LayerAttention::Kda(op) => PreparedAttention::Kda(Box::new(KdaOperands::load(
                        op,
                        store,
                        &attention_format,
                        layer.declared_norm_eps as f32,
                    )?)),
                    // Same posture as KDA above: represented, not
                    // executable. MLA's operands are bound and its
                    // geometry is stated, but no executor consumes them.
                    LayerAttention::Mla(op) => PreparedAttention::Mla(Box::new(MlaOperands::load(
                        op,
                        store,
                        &attention_format,
                    )?)),
                    LayerAttention::ConvQkv(op) => PreparedAttention::ConvQkv(Box::new(
                        ConvQkvOperands::load(op, store, &attention_format)?,
                    )),
                    LayerAttention::Mamba2(op) => PreparedAttention::Mamba2(Box::new(
                        Mamba2Operands::load(op, store, &attention_format)?,
                    )),
                    LayerAttention::GatedDelta(op) => {
                        PreparedAttention::GatedDelta(Box::new(GatedDeltaOperands::load(
                            op,
                            store,
                            &attention_format,
                            layer.declared_norm_eps as f32,
                        )?))
                    }
                },
                post_attention: layer
                    .post_attention_norm
                    .as_ref()
                    .map(|op| PreparedNorm::load(op, store))
                    .transpose()?,
                // Absent on a mixer-only layer: the plan carries no FFN
                // program there, and preparing one would fabricate work
                // the executor must then skip.
                pre_ffn: layer
                    .pre_ffn_norm
                    .as_ref()
                    .map(|op| PreparedNorm::load(op, store))
                    .transpose()?,
                ffn: layer
                    .ffn
                    .as_ref()
                    .map(|ffn| {
                        FfnOperands::load(ffn, store, &ffn_format, bank_format, &shared_format)
                    })
                    .transpose()?,
                post_ffn: layer
                    .post_ffn_norm
                    .as_ref()
                    .map(|op| PreparedNorm::load(op, store))
                    .transpose()?,
                layer_scale: layer
                    .layer_scale
                    .as_ref()
                    .map(|op| store.load(op).and_then(|v| super::layer_scalar_of(&v)))
                    .transpose()?,
                hyper_connection: PreparedHyperConnection::for_layer(
                    layer,
                    hyper_connection,
                    hidden,
                    store,
                )?,
                attention_residual: PreparedAttentionResidual::for_layer(
                    layer,
                    attention_residual,
                    hidden,
                    store,
                )?,
            });
        }

        let final_norm = if whole {
            plan.final_norm
                .as_ref()
                .map(|op| PreparedNorm::load(op, store))
                .transpose()?
        } else {
            None
        };
        let output = if whole {
            plan.output
                .as_ref()
                .map(|op| {
                    Ok::<_, VindexError>((
                        op.clone(),
                        load_weight(store, &op.projection, head_format(&op.projection)?)?,
                    ))
                })
                .transpose()?
        } else {
            None
        };

        // The head's reduction belongs to the stack's END: a whole-stack
        // image of a hyper-connected component must carry one, and a
        // layer-range image must not consult one — the per-layer contract
        // needs no head (GLM-5.3-Flash has none to offer).
        let hyper_connection_head = match (whole, hyper_connection) {
            (true, Some(hc)) => Some(PreparedHcHead::load(plan, hc, hidden, store)?),
            _ => None,
        };
        // The exit reduction belongs to the stack's END for the same
        // reason the head's does — and unlike the head it is REQUIRED
        // under its declaration, so a whole-stack image without one
        // refuses here rather than running a stack whose history nothing
        // collapses. A layer-range image must not consult one: its
        // output IS the history it hands on.
        let attention_residual_exit = match (whole, attention_residual) {
            (true, true) => Some(PreparedAttnResExit::load(plan, hidden, store)?),
            _ => None,
        };
        let prepared = Self {
            stamp,
            slice,
            hidden,
            embed_table,
            first_layer,
            layers,
            final_norm,
            output,
            registry: store.registry(),
            realizations,
            topology,
            hyper_connection_head,
            attention_residual_exit,
        };
        // **The executor runs what was pinned.** Every resident matrix
        // holds the representation its record named, checked here so a
        // loader that drifted from the selector cannot hand the executor
        // bytes the plan never pinned.
        prepared.verify_pins()?;
        prepared.place(backend);
        Ok(prepared)
    }

    /// Hand every matrix operand to the backend once, so a device
    /// backend can hold the model resident for this image's lifetime.
    fn place<B: PlanBackend + ?Sized>(&self, backend: &B) {
        let mut weights: Vec<WeightSlice<'_>> = Vec::new();
        for layer in &self.layers {
            weights.extend(layer.attention.weight_slices());
            if let Some(ffn) = &layer.ffn {
                weights.extend(ffn.weight_slices());
            }
        }
        if let Some((_, projection)) = &self.output {
            weights.push(projection.slice());
        }
        backend.prepare(&weights);
    }

    /// **What this image actually occupies, by site and representation.**
    ///
    /// Site by site rather than one total, because a total cannot fail
    /// usefully. The claim CPU-2A makes is not "the model is smaller" but
    /// "every streaming matrix kept the checkpoint's own bytes" — and a
    /// single number is satisfied just as well by a stack that halved its
    /// FFN and left 11 GB of recurrence widened.
    ///
    /// The embedding table is the one f32 population that is EXPECTED:
    /// decode gathers a single row from it per token, so it is residency
    /// without traffic, and no kernel here consumes a compact one.
    /// One pinned realization per planned operand this image executes:
    /// the representation resolved, the candidates considered, the one
    /// selected, its reason and its declared residency.
    /// The registry this image was prepared through.
    pub fn registry(&self) -> &'static CodecRegistry {
        self.registry
    }

    pub fn realizations(&self) -> &[RealizationRecord] {
        &self.realizations
    }

    #[cfg(test)]
    pub(super) fn realizations_mut(&mut self) -> &mut Vec<RealizationRecord> {
        &mut self.realizations
    }

    /// Every resident matrix holds the representation its pinned
    /// realization named — checked per layer and site as a multiset, so
    /// the executor, which reads its kernel off the resident bytes, can
    /// run nothing the plan did not pin. A packed bank's per-expert slices
    /// are a different realization and are checked by the bank loader.
    pub fn verify_pins(&self) -> Result<(), VindexError> {
        let pinned =
            |layer: Option<usize>, want: &dyn Fn(Operation) -> bool| -> Vec<WeightFormat> {
                let mut out: Vec<WeightFormat> = self
                    .realizations
                    .iter()
                    .filter(|r| r.planned.layer == layer && want(r.planned.operation))
                    .map(|r| r.selection.realization.format())
                    .collect();
                out.sort_by_key(|f| format!("{f:?}"));
                out
            };
        let observed = |weights: Vec<&LoadedWeight>| -> Vec<WeightFormat> {
            let mut out: Vec<WeightFormat> = weights.iter().map(|w| w.format()).collect();
            out.sort_by_key(|f| format!("{f:?}"));
            out
        };
        let mismatch = |site: String, pinned: Vec<WeightFormat>, resident: Vec<WeightFormat>| {
            VindexError::Parse(format!(
                "{site}: the resident representations {resident:?} are not the pinned \
                 realizations {pinned:?} — the loader drifted from the selector"
            ))
        };
        for (offset, layer) in self.layers.iter().enumerate() {
            let index = self.first_layer + offset;
            let attention = pinned(Some(index), &|o| {
                o == Operation::Project(MatrixClass::AttentionProjection)
            });
            let resident = observed(layer.attention.matrices());
            if attention != resident {
                return Err(mismatch(
                    format!("layer {index} attention"),
                    attention,
                    resident,
                ));
            }
            let ffn = pinned(Some(index), &|o| {
                o == Operation::Project(MatrixClass::FfnProjection)
            });
            let resident = observed(
                layer
                    .ffn
                    .as_ref()
                    .map(|f| f.dense_matrices())
                    .unwrap_or_default(),
            );
            if ffn != resident {
                return Err(mismatch(format!("layer {index} ffn"), ffn, resident));
            }
        }
        let head = pinned(None, &|o| o == Operation::OutputHead);
        let resident = observed(self.output.iter().map(|(_, w)| w).collect());
        if head != resident {
            return Err(mismatch("output head".to_string(), head, resident));
        }
        Ok(())
    }

    /// Every planned operand paired with the object(s) the loader bound
    /// for it — the OBSERVATION side of the accounting, read off the
    /// resident objects and never off a declaration.
    pub fn bound(&self, plan: &ComponentOpPlan) -> Result<Vec<Observed>, VindexError> {
        let mut out = Vec::new();
        if let (Some(embedding), Some(table)) = (&plan.embedding, &self.embed_table) {
            out.push(Observed {
                operand: embedding.table.clone(),
                operation: Operation::Embed,
                layer: None,
                format: WeightFormat::F32,
                resident_bytes: std::mem::size_of_val(&table[..]) as u64,
                mapped_bytes: 0,
                allocations: 0,
            });
        }
        for (offset, prepared) in self.layers.iter().enumerate() {
            let index = self.first_layer + offset;
            let layer = plan.layers.get(index).ok_or_else(|| {
                VindexError::Parse(format!("layer {index}: prepared but not in the plan"))
            })?;
            for bound in prepared.attention.bound(&layer.attention)? {
                out.push(bound.observed(
                    Operation::Project(MatrixClass::AttentionProjection),
                    Some(index),
                )?);
            }
            if let (Some(ffn), Some(op)) = (&prepared.ffn, &layer.ffn) {
                // The loader names the operation it bound each object for.
                for (operation, bound) in ffn.bound(op)? {
                    out.push(bound.observed(operation, Some(index))?);
                }
            }
        }
        if let Some((op, weight)) = &self.output {
            out.push(Bound::one(&op.projection, weight).observed(Operation::OutputHead, None)?);
        }
        Ok(out)
    }

    /// What every pinned realization DECLARES it costs, priced from the
    /// pin, the codec's declared residency, `geometry`, and the container's
    /// recorded lengths — the EXPECTATION side, which reads no object.
    pub fn expectations(
        &self,
        store: OperandSource<'_>,
        geometry: BlockGeometry,
    ) -> Vec<Expectation> {
        expectations(&self.realizations, |op| store.stored_len(op), geometry)
    }

    /// Bind the declarations AGAINST the observations: every pin meets
    /// exactly one resident object in the pinned representation holding
    /// the declared bytes, and nothing resident is unpinned.
    pub fn reconcile(
        &self,
        plan: &ComponentOpPlan,
        store: OperandSource<'_>,
    ) -> Result<Reconciliation, VindexError> {
        reconcile(
            &self.expectations(store, BlockGeometry::executor()),
            &self.bound(plan)?,
        )
    }

    /// The providers this image was prepared against, by representation
    /// label, with the identity each resolved to. Since 3d every record
    /// names a registered codec — a packed bank's carrier label resolves
    /// to the codec the plan declares — so a `None` here is an overlay
    /// edit's f32-space fact, never a label a loader judged for itself.
    pub fn providers(&self) -> Vec<(String, Option<CodecIdentity>)> {
        let mut out: Vec<(String, Option<CodecIdentity>)> = Vec::new();
        for r in &self.realizations {
            if !out.iter().any(|(label, _)| *label == r.representation) {
                out.push((r.representation.clone(), r.provider.clone()));
            }
        }
        out
    }

    /// Refuse to execute this image against a registry that no longer
    /// resolves every provider it was prepared with to the same identity
    /// — a provider that disappeared or changed invalidates the
    /// preparation; nothing falls back.
    pub fn ensure_providers_in(&self, registry: &CodecRegistry) -> Result<(), VindexError> {
        let describe = |identity: &Option<CodecIdentity>| {
            identity
                .as_ref()
                .map(|i| format!("{} r{}", i.family, i.revision))
                .unwrap_or_else(|| "no registered codec".to_string())
        };
        for (label, prepared) in self.providers() {
            let now = registry.by_label(&label).map(|c| c.identity());
            if now != prepared {
                return Err(VindexError::Parse(format!(
                    "representation `{label}` was prepared against {} and the registry now \
                     offers {}; re-prepare rather than execute a pin whose provider changed",
                    describe(&prepared),
                    describe(&now)
                )));
            }
        }
        Ok(())
    }

    pub fn residency_census(&self) -> ResidencyCensus {
        let mut census = ResidencyCensus::default();
        if let Some(table) = &self.embed_table {
            census.embedding.widened_f32 += std::mem::size_of_val(&table[..]);
        }
        for layer in &self.layers {
            match &layer.attention {
                PreparedAttention::Softmax(ops) => {
                    for w in ops.loaded_matrices() {
                        census.attention.add(w);
                    }
                }
                PreparedAttention::GatedDelta(ops) => {
                    for w in ops.loaded_matrices() {
                        census.delta.add(w);
                    }
                    census.glue.widened_f32 += ops.glue_bytes();
                }
                PreparedAttention::Mamba2(ops) => {
                    for w in ops.loaded_matrices() {
                        census.delta.add(w);
                    }
                    census.glue.widened_f32 += ops.glue_bytes();
                }
                PreparedAttention::ConvQkv(ops) => {
                    // Attention matrix traffic — the block attends, and
                    // its fused QKV/out projections are what a device
                    // backend would hold resident on that site.
                    for w in ops.loaded_matrices() {
                        census.attention.add(w);
                    }
                    census.glue.widened_f32 += ops.glue_bytes();
                }
                // KDA is a recurrence: its four wide projections are
                // counted where the other recurrences' are, so a
                // hybrid's census still separates "the model attends"
                // from "the model recurs".
                PreparedAttention::Kda(ops) => {
                    for w in ops.loaded_matrices() {
                        census.delta.add(w);
                    }
                    census.glue.widened_f32 += ops.glue_bytes();
                }
                // MLA attends — over a compressed cache, but it attends.
                PreparedAttention::Mla(ops) => {
                    for w in ops.loaded_matrices() {
                        census.attention.add(w);
                    }
                    census.glue.widened_f32 += ops.glue_bytes();
                }
            }
            if let Some(ffn) = &layer.ffn {
                for w in ffn.loaded_matrices() {
                    census.ffn.add(w);
                }
            }
            census.glue.widened_f32 += layer.glue_bytes();
        }
        if let Some(norm) = &self.final_norm {
            census.glue.widened_f32 += std::mem::size_of_val(&norm.weight[..]);
        }
        if let Some(head) = &self.hyper_connection_head {
            census.glue.widened_f32 += head.glue_bytes();
        }
        if let Some((_, projection)) = &self.output {
            census.head.add(projection);
        }
        census
    }

    /// Where this image's allocations landed. See [`AllocationCensus`].
    /// The image's mappings, and how much of them is physically resident
    /// at this moment: address space summed once per bound region, pages
    /// resident as the OS reports them now. Cheap enough to ask between
    /// tokens, which is what a residency curve is.
    pub fn mapped_residency(&self) -> MappedResidency {
        let mut out = MappedResidency::default();
        let mut add = |w: &LoadedWeight| {
            let mapped = w.mapped_bytes() as u64;
            if mapped > 0 {
                out.mapped_bytes += mapped;
                out.resident_bytes += w.resident_bytes() as u64;
                out.regions += 1;
            }
        };
        for layer in &self.layers {
            match &layer.attention {
                PreparedAttention::Softmax(ops) => {
                    ops.loaded_matrices().iter().for_each(|w| add(w))
                }
                PreparedAttention::GatedDelta(ops) => {
                    ops.loaded_matrices().iter().for_each(|w| add(w))
                }
                PreparedAttention::Mamba2(ops) => ops.loaded_matrices().iter().for_each(|w| add(w)),
                PreparedAttention::ConvQkv(ops) => {
                    ops.loaded_matrices().iter().for_each(|w| add(w))
                }
                PreparedAttention::Kda(ops) => ops.loaded_matrices().iter().for_each(|w| add(w)),
                PreparedAttention::Mla(ops) => ops.loaded_matrices().iter().for_each(|w| add(w)),
            }
            if let Some(ffn) = &layer.ffn {
                ffn.loaded_matrices().iter().for_each(|w| add(w));
            }
        }
        if let Some((_, projection)) = &self.output {
            add(projection);
        }
        out
    }

    pub fn allocation_census(&self) -> AllocationCensus {
        let mut census = AllocationCensus::default();
        let mut add = |w: &LoadedWeight| {
            for (address, bytes) in w.allocations() {
                census.add(address, bytes);
            }
        };
        for layer in &self.layers {
            match &layer.attention {
                PreparedAttention::Softmax(ops) => {
                    ops.loaded_matrices().iter().for_each(|w| add(w))
                }
                PreparedAttention::GatedDelta(ops) => {
                    ops.loaded_matrices().iter().for_each(|w| add(w))
                }
                PreparedAttention::Mamba2(ops) => ops.loaded_matrices().iter().for_each(|w| add(w)),
                PreparedAttention::ConvQkv(ops) => {
                    ops.loaded_matrices().iter().for_each(|w| add(w))
                }
                PreparedAttention::Kda(ops) => ops.loaded_matrices().iter().for_each(|w| add(w)),
                PreparedAttention::Mla(ops) => ops.loaded_matrices().iter().for_each(|w| add(w)),
            }
            if let Some(ffn) = &layer.ffn {
                ffn.loaded_matrices().iter().for_each(|w| add(w));
            }
        }
        if let Some((_, projection)) = &self.output {
            add(projection);
        }
        census
    }

    /// The slice this image was prepared for.
    pub fn slice(&self) -> &ExecutionSlice {
        &self.slice
    }

    /// The effective source this image was compiled from.
    pub fn source_stamp(&self) -> SourceStamp {
        self.stamp
    }

    /// Whether this image still describes `source`.
    ///
    /// False after any overlay mutation, and for a different store or a
    /// different override set. A caller that has the source in hand
    /// should ask before reusing a cached image; one that does not
    /// (the serve path, which holds only its own image) is safe by
    /// ownership — it has nothing else to confuse it with.
    pub fn is_current_for(&self, source: &OperandSource<'_>) -> bool {
        self.stamp == source.stamp()
    }

    /// [`Self::is_current_for`] as a refusal, for callers that would
    /// otherwise execute a stale image.
    pub fn ensure_current_for(&self, source: &OperandSource<'_>) -> Result<(), VindexError> {
        if self.is_current_for(source) {
            return Ok(());
        }
        Err(VindexError::Parse(
            "this prepared image was compiled from a different effective operand source — \
             the overlay changed, or it belongs to another container. Re-prepare rather than \
             executing a stale compilation of the model."
                .to_string(),
        ))
    }

    /// Hidden width, read from the plan's embedding op.
    pub fn hidden(&self) -> usize {
        self.hidden
    }

    /// How many layers this image can execute.
    pub fn layer_count(&self) -> usize {
        self.layers.len()
    }

    /// Whether this image carries an output head (only a whole-stack
    /// slice does).
    pub fn has_output(&self) -> bool {
        self.output.is_some()
    }

    pub(super) fn embed_table(&self) -> Option<&[f32]> {
        self.embed_table.as_deref()
    }

    pub(super) fn first_layer(&self) -> usize {
        self.first_layer
    }

    /// The plan-layer indices this prepared set will execute, as a
    /// half-open range.
    ///
    /// Public because the identity of the work is part of a run's
    /// provenance: a caller banking logits has to be able to record —
    /// and a comparison to assert — which layers actually ran, and
    /// neither can be inferred from a layer count.
    pub fn executed_layers(&self) -> std::ops::Range<usize> {
        self.first_layer..self.first_layer + self.layers.len()
    }

    pub(super) fn layers(&self) -> &[PreparedLayer] {
        &self.layers
    }

    pub(super) fn final_norm(&self) -> Option<&PreparedNorm> {
        self.final_norm.as_ref()
    }

    pub(super) fn output(&self) -> Option<&(OutputOp, LoadedWeight)> {
        self.output.as_ref()
    }

    /// The declared hyper-connection topology this image was prepared
    /// under, `None` for the single stream.
    pub(super) fn hyper_connection(&self) -> Option<HyperConnection> {
        match self.topology {
            ResidualTopology::HyperConnection(hc) => Some(hc),
            ResidualTopology::SingleStream | ResidualTopology::AttentionResidual { .. } => None,
        }
    }

    /// Whether this image holds hyper-connection site operands — i.e.
    /// whether the residual it executes is a bundle.
    pub fn carries_hyper_connection(&self) -> bool {
        self.hyper_connection().is_some()
    }

    pub(super) fn hyper_connection_head(&self) -> Option<&PreparedHcHead> {
        self.hyper_connection_head.as_ref()
    }

    /// The declared block period, `None` on every other topology. The
    /// traversal reads it to decide which layers carry the boundary
    /// event, and it is the ONE declared fact the schedule needs.
    pub(super) fn attention_residual_block_size(&self) -> Option<usize> {
        match self.topology {
            ResidualTopology::AttentionResidual { block_size } => Some(block_size),
            ResidualTopology::SingleStream | ResidualTopology::HyperConnection(_) => None,
        }
    }

    pub(super) fn attention_residual_exit(&self) -> Option<&PreparedAttnResExit> {
        self.attention_residual_exit.as_ref()
    }

    /// Whether this image holds attention-residual site operands — i.e.
    /// whether the residual it executes carries a snapshot history.
    pub fn carries_attention_residual(&self) -> bool {
        self.attention_residual_block_size().is_some()
    }
}

/// Where a prepared image's allocations LAND, as distinct from how many
/// bytes they hold.
///
/// CPU-PERF-1 found the isolated kernel harness predicts real bf16
/// projection to +0.7% and misses real Q8 by 7.9%, and CPU-PERF-2 ruled
/// out machine state. What is left is the resident representation itself,
/// and the two formats differ in more than bytes: bf16 lands in
/// page-aligned `AlignedBytes`, one allocation per matrix, while Q8 uses
/// ordinary heap vectors and TWO allocations per matrix.
///
/// This measures that difference before anything is changed on the
/// strength of it — a large `Vec` may already receive a page-aligned VM
/// region, in which case "align it" would be an intervention with nothing
/// to intervene on.
#[derive(Default, Clone, Copy, Debug)]
pub struct AllocationCensus {
    pub allocations: usize,
    pub page_aligned: usize,
    /// The coarsest alignment every allocation shares, in bytes.
    pub common_alignment: usize,
    pub bytes: usize,
}

impl AllocationCensus {
    fn add(&mut self, address: usize, bytes: usize) {
        self.allocations += 1;
        self.bytes += bytes;
        if address.is_multiple_of(super::weights::DEVICE_PAGE_ALIGN) {
            self.page_aligned += 1;
        }
        let align = 1usize << address.trailing_zeros().min(30);
        self.common_alignment = if self.allocations == 1 {
            align
        } else {
            self.common_alignment.min(align)
        };
    }
}

/// A prepared image's mappings at one moment: their address space and the
/// pages of it physically resident — the two figures a mapping must never
/// be reported as one of.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct MappedResidency {
    pub regions: usize,
    pub mapped_bytes: u64,
    pub resident_bytes: u64,
}

/// One site's bytes, split by whether the loader widened — and, apart
/// from both, what it MAPPED: address space over the container's own
/// segment, resident only as touched, never counted as committed.
#[derive(Default, Clone, Copy, Debug)]
pub struct SiteResidency {
    /// Bytes held as f32 — doubled, when the checkpoint stored bf16.
    pub widened_f32: usize,
    /// Bytes held exactly as the checkpoint holds them, committed.
    pub compact: usize,
    /// Bytes bound as a mapping of the container's segment.
    pub mapped: usize,
}

impl SiteResidency {
    fn add(&mut self, w: &LoadedWeight) {
        let mapped = w.mapped_bytes();
        if mapped > 0 {
            self.mapped += mapped;
        } else if w.is_widened_f32() {
            self.widened_f32 += w.resident_bytes();
        } else {
            self.compact += w.resident_bytes();
        }
    }

    /// Committed bytes: what the process holds, mappings excluded.
    pub fn total(&self) -> usize {
        self.widened_f32 + self.compact
    }
}

/// Where a prepared image's bytes are, and in which representation.
#[derive(Default, Clone, Copy, Debug)]
pub struct ResidencyCensus {
    pub embedding: SiteResidency,
    pub attention: SiteResidency,
    pub delta: SiteResidency,
    pub ffn: SiteResidency,
    pub head: SiteResidency,
    /// Norms, biases, the depthwise convolution, gate biases — always
    /// f32, and small enough that widening them costs nothing worth
    /// recovering.
    pub glue: SiteResidency,
}

impl ResidencyCensus {
    /// Every site, in the order a decode reads them.
    pub fn sites(&self) -> [(&'static str, SiteResidency); 6] {
        [
            ("embedding", self.embedding),
            ("attention", self.attention),
            ("delta", self.delta),
            ("ffn", self.ffn),
            ("head", self.head),
            ("glue", self.glue),
        ]
    }

    pub fn total(&self) -> usize {
        self.sites().iter().map(|(_, s)| s.total()).sum()
    }

    pub fn widened_f32(&self) -> usize {
        self.sites().iter().map(|(_, s)| s.widened_f32).sum()
    }

    /// Address space held as mappings, across every site.
    pub fn mapped(&self) -> usize {
        self.sites().iter().map(|(_, s)| s.mapped).sum()
    }

    pub fn compact(&self) -> usize {
        self.sites().iter().map(|(_, s)| s.compact).sum()
    }
}
