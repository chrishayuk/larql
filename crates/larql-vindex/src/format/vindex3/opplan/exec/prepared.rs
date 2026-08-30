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

use super::backend::{
    MatrixClass, MatrixOperand, NormCall, PlanBackend, WeightFormat, WeightSlice,
};
use super::experts::FfnOperands;
use super::operands::{OperandSource, SourceStamp};
use super::weights::{load_weight, LoadedWeight};
use super::AttentionOperands;
use crate::error::VindexError;

use super::super::{ComponentOpPlan, GatedDeltaOp, LayerAttention, NormOp, OperandRef, OutputOp};

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
    fn validate(&self, plan: &ComponentOpPlan) -> Result<(), VindexError> {
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

/// One layer's operands, lowered into the backend's execution form.
pub(super) struct PreparedLayer {
    pub(super) pre_attention: PreparedNorm,
    pub(super) attention: PreparedAttention,
    pub(super) post_attention: Option<PreparedNorm>,
    pub(super) pre_ffn: PreparedNorm,
    pub(super) ffn: FfnOperands,
    pub(super) post_ffn: Option<PreparedNorm>,
    /// The layer's output scalar, when the plan carries one.
    pub(super) layer_scale: Option<f32>,
}

impl PreparedLayer {
    /// This layer's norm weights — f32 glue, counted so the census adds
    /// up to the whole image rather than to the parts that were easy.
    fn glue_bytes(&self) -> usize {
        let norm = |n: &PreparedNorm| std::mem::size_of_val(&n.weight[..]);
        norm(&self.pre_attention)
            + norm(&self.pre_ffn)
            + self.post_attention.as_ref().map_or(0, norm)
            + self.post_ffn.as_ref().map_or(0, norm)
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
}

impl PreparedAttention {
    /// Matrix operands for device placement.
    ///
    /// A recurrence contributes none: its nine operands are elementwise
    /// glue and a depthwise convolution, not the matrix traffic a device
    /// backend holds resident — and no device backend runs this operator
    /// yet, so placing them would reserve memory nothing reads.
    fn weight_slices(&self) -> Vec<WeightSlice<'_>> {
        match self {
            Self::Softmax(ops) => ops.weight_slices(),
            Self::GatedDelta(_) => Vec::new(),
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
        let matrix = |r: &OperandRef| load_weight(store, r, format(r));
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
pub(super) type FormatFor<'a> = &'a dyn Fn(&OperandRef) -> WeightFormat;

/// Ask the backend about one operand, with the physical facts attached.
///
/// The backend still cannot reach the bytes — it is told the element
/// count and what the checkpoint holds, and answers with a
/// representation. Handing it the `OperandRef` would let it resolve
/// operands by name, which is the one thing the seam forbids.
fn resolve<B: PlanBackend + ?Sized>(
    backend: &B,
    store: OperandSource<'_>,
    class: MatrixClass,
    op: &OperandRef,
) -> WeightFormat {
    backend.weight_format(MatrixOperand {
        class,
        elements: op.shape.iter().product(),
        stored_bf16: store.is_stored_bf16(op),
    })
}

/// A component's operands, lowered once for a given slice and backend.
///
/// Immutable for its lifetime: this is the canonical base model. A
/// session that carries an overlay composes *over* these operands
/// rather than mutating them, so one prepared image can serve every
/// concurrent request on the model.
pub struct PreparedOperands {
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
        let store = store.into();
        slice.validate(plan)?;
        let stamp = store.stamp();
        let whole = slice.is_whole_stack();
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

        // One resolver per matrix class, each answering per operand.
        let attention_format =
            |op: &OperandRef| resolve(backend, store, MatrixClass::AttentionProjection, op);
        let ffn_format = |op: &OperandRef| resolve(backend, store, MatrixClass::FfnProjection, op);
        // The bank is asked once, for the class: it is one stored tensor
        // holding every expert's matrix, so there is no per-matrix size to
        // answer about.
        let bank_format = backend.weight_format(MatrixOperand {
            class: MatrixClass::RoutedExpertBank,
            elements: 0,
            stored_bf16: false,
        });
        let head_format = |op: &OperandRef| resolve(backend, store, MatrixClass::OutputHead, op);

        let range = slice.layers(plan);
        let first_layer = range.start;
        let mut layers = Vec::with_capacity(range.len());
        for layer in &plan.layers[range] {
            layers.push(PreparedLayer {
                pre_attention: PreparedNorm::load(&layer.pre_attention_norm, store)?,
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
                    LayerAttention::Kda(_) => {
                        return Err(VindexError::Parse(
                            "KDA (Kimi Delta Attention) layers are represented but not \
                             executable: no executor exists for this operator"
                                .to_string(),
                        ))
                    }
                    // Same posture as KDA above: represented, not
                    // executable. MLA's operands are bound and its
                    // geometry is stated, but no executor consumes them.
                    LayerAttention::Mla(_) => {
                        return Err(VindexError::Parse(
                            "MLA (Multi-Latent Attention) layers are represented but not \
                             executable: no executor exists for this operator"
                                .to_string(),
                        ))
                    }
                    LayerAttention::GatedDelta(op) => {
                        PreparedAttention::GatedDelta(Box::new(GatedDeltaOperands::load(
                            op,
                            store,
                            &attention_format,
                            layer.pre_attention_norm.eps as f32,
                        )?))
                    }
                },
                post_attention: layer
                    .post_attention_norm
                    .as_ref()
                    .map(|op| PreparedNorm::load(op, store))
                    .transpose()?,
                pre_ffn: PreparedNorm::load(&layer.pre_ffn_norm, store)?,
                ffn: FfnOperands::load(&layer.ffn, store, &ffn_format, bank_format)?,
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
                        load_weight(store, &op.projection, head_format(&op.projection))?,
                    ))
                })
                .transpose()?
        } else {
            None
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
        };
        prepared.place(backend);
        Ok(prepared)
    }

    /// Hand every matrix operand to the backend once, so a device
    /// backend can hold the model resident for this image's lifetime.
    fn place<B: PlanBackend + ?Sized>(&self, backend: &B) {
        let mut weights: Vec<WeightSlice<'_>> = Vec::new();
        for layer in &self.layers {
            weights.extend(layer.attention.weight_slices());
            weights.extend(layer.ffn.weight_slices());
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
            }
            for w in layer.ffn.loaded_matrices() {
                census.ffn.add(w);
            }
            census.glue.widened_f32 += layer.glue_bytes();
        }
        if let Some(norm) = &self.final_norm {
            census.glue.widened_f32 += std::mem::size_of_val(&norm.weight[..]);
        }
        if let Some((_, projection)) = &self.output {
            census.head.add(projection);
        }
        census
    }

    /// Where this image's allocations landed. See [`AllocationCensus`].
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
            }
            layer.ffn.loaded_matrices().iter().for_each(|w| add(w));
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

/// One site's resident bytes, split by whether the loader widened.
#[derive(Default, Clone, Copy, Debug)]
pub struct SiteResidency {
    /// Bytes held as f32 — doubled, when the checkpoint stored bf16.
    pub widened_f32: usize,
    /// Bytes held exactly as the checkpoint holds them.
    pub compact: usize,
}

impl SiteResidency {
    fn add(&mut self, w: &LoadedWeight) {
        if w.is_widened_f32() {
            self.widened_f32 += w.resident_bytes();
        } else {
            self.compact += w.resident_bytes();
        }
    }

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

    pub fn compact(&self) -> usize {
        self.sites().iter().map(|(_, s)| s.compact).sum()
    }
}
