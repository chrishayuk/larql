//! The backend seam (V3-G5b-3b): what *executes* a plan, versus what the
//! plan *means*.
//!
//! One [`ComponentOpPlan`](super::super::ComponentOpPlan), one interpreter,
//! many backends. The interpreter in [`super`] owns every decision that is
//! semantics — operation ordering, residual ordering, layer traversal,
//! whether an optional operation exists at all, and how position and span
//! policy dispatch. A [`PlanBackend`] owns only the numerical realisation
//! of work it is handed.
//!
//! **Nothing in this file mentions a model family, and nothing in it takes
//! a plan type.** Backends receive primitives, judged enums, and *already
//! loaded* weight slices — never a `LayerPlan`, an `OperandRef`, or the
//! `OperandStore`. That is deliberate and load-bearing: a backend that
//! could resolve its own operands by name, or read the layer structure,
//! could quietly grow into a second implementation of the model and
//! disagree with the IR while still passing. It cannot reach the bytes,
//! so it cannot reinterpret them.
//!
//! The corollary for anyone adding a method: if a backend needs to ask
//! *whether* to do something, the seam is in the wrong place. It should
//! only ever be told what to compute.

use larql_models::config::{
    Activation, AttentionGateSpec, AttentionSinkSpec, ExpertRoutingPolicy, GateUpLayout,
    MoeRouterKind, NormType, ParameterFreeQkNorm, PositionPolicy, QkNormScope,
};

use super::super::super::graph::policy::AttentionSpan;
use super::cpu::WeightRows;
use super::quantise::SUM_BLOCK;
use super::realization::{RepresentationFacts, Selection, SelectionRefusal};
use crate::error::VindexError;
use crate::format::vindex3::opplan::planned::PlannedOperand;
use crate::format::vindex3::represent::kquant::KQuant;

/// The numerical representation a backend wants matrix operands in.
///
/// Asked once by the interpreter (a capability, like [`PlanBackend::name`],
/// not a per-call decision): the interpreter loads every matrix operand in
/// the declared format and the backend receives what it asked for. Norm
/// and QK-norm weights and the embedding table are always f32 — they are
/// elementwise glue, not matrix traffic, and narrowing them buys nothing.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum WeightFormat {
    /// Widened f32 — the constitutional representation.
    F32,
    /// Stored bf16, kept EXACTLY as the checkpoint holds it.
    ///
    /// Not a conversion and not a quantisation: bf16 is the top 16 bits
    /// of the f32 it denotes, so a consumer widens with `(bits as u32) <<
    /// 16` — no rounding, no table, no loss. Declaring this removes the
    /// artificial F32 materialisation (107.6 GB resident against a 53.8
    /// GB checkpoint) rather than introducing a new numeric format.
    ///
    /// Only worth declaring for matrices large enough to STREAM. A
    /// cache-resident matrix has no RAM traffic to halve, and the
    /// measured `48 x 5120` case runs 3.8x faster through BLAS f32 — see
    /// `exec::cpu::kernels::FusedBf16`.
    Bf16,
    /// Symmetric int8, one f32 scale per [`Q8_BLOCK`] elements along the
    /// input axis.
    ///
    /// **The first LOSSY residency format.** `Bf16` keeps the
    /// checkpoint's own bytes and changes no value; this one quantises at
    /// load and the model it decodes is not quite the model that was
    /// stored. Everything about it is therefore judged on logits, KL, a
    /// trajectory and recurrent-state drift, not on residency alone.
    ///
    /// Blocked along the input axis so a kernel accumulates a block and
    /// scales once. 8.5 bits/weight with the scales counted.
    ///
    /// Worth declaring only where the BF16 image is too big for cache —
    /// measured, `1024 x 5120` runs 0.81x through Q8 because at 10.5 MB
    /// it is already L2-resident and the extra unpacking is pure cost.
    Q8,
    /// Symmetric int4, two codes per byte, one f32 scale per
    /// [`Q4_BLOCK`](crate::format::vindex3::opplan::exec::quantise::Q4_BLOCK)
    /// elements along the input axis. 4.5 bits/weight.
    ///
    /// **The second lossy residency format, and by far the larger
    /// perturbation.** Its step is `peak / 7` against Q8's `peak / 127` —
    /// 18.1x coarser at the same block — so it is not "Q8 with fewer
    /// bytes", it is a different numerical proposition that has to earn
    /// its place on logits, KL, a trajectory and recurrent-state drift.
    ///
    /// Worth declaring only where the arithmetic can consume it. CPU-4A
    /// measured Q4 against f32 activations at 1.08x — SLOWER than Q8 —
    /// because the kernel was already conversion-bound and Q4 adds a
    /// nibble split on top; CPU-4Y measured the same bytes against an
    /// int8 activation at 3.12x. The format is only ever as good as the
    /// domain it is multiplied in.
    Q4,
    /// IEEE 754 half, little-endian. Exactly representable from stored
    /// bf16 for all normal-range values (bf16's 7 mantissa bits fit in
    /// f16's 10); conversion fails closed on overflow. A device backend
    /// declares this so weights can stay resident in half the bytes.
    F16,
    /// OCP microscaling 4-bit float: e2m1 codes two-per-byte plus one
    /// e8m0 scale per 32-element group, in separate streams. A lossy
    /// realisation — quantised at load, judged by the parity gates —
    /// that quarters the bytes every decoded token must read.
    Mxfp4,
    /// The same e2m1 elements under a different scale geometry: 16-element
    /// groups with **E4M3** scales, plus one f32 per matrix. 4.5 bpw
    /// against MXFP4's 4.25.
    ///
    /// Present as its own format rather than a parameter of [`Self::Mxfp4`]
    /// because the difference is the point: E8M0 forces a group's scale to
    /// a power of two, and a weight-reconstruction sweep over Muse-Glimmer
    /// with an equal-bit-budget control (E8M0 at group 16) found the group
    /// size worth nothing and the scale format worth 1.27x in relative RMS
    /// and 1.7x in worst-element error.
    Nvfp4,
    /// A stored ggml K-quant pack — Q8_0, Q6_K or Q4_K — kept as the
    /// container holds it and executed in place by the codec's kernel.
    ///
    /// Like [`Self::Bf16`] and unlike [`Self::Q8`], NOT a conversion: the
    /// resident bytes are the stored bytes. The lossy step, if any,
    /// happened when `vindex represent` compiled the pack, and it is the
    /// artifact under measurement, not this loader's doing. Which codec
    /// is a property of the bytes, carried with them — the format names
    /// the family, the operand's stored dtype names the member.
    ///
    /// Only ever declared for an operand the container holds as a
    /// K-quant; a backend asking for it over anything else is refused at
    /// load rather than served a manufactured pack.
    KQuant,
}

/// Which matrix a format question is about. Formats are declared per
/// class because the classes have different numerical stakes: the
/// output head feeds logits directly, attention feeds the softmax, and
/// the FFN is the bulk of the bytes.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum MatrixClass {
    AttentionProjection,
    FfnProjection,
    OutputHead,
    /// One stored tensor holding every expert's matrix, sliced at load.
    ///
    /// Its own class because it is not one matrix: the bank is split into
    /// `experts` matrices and may be quantised on the way, so the path has
    /// already widened to f32 by the time a format could be applied. A
    /// question about "how big is this matrix" has no answer here, which
    /// is exactly why answering it as an `FfnProjection` would be wrong.
    RoutedExpertBank,
}

/// A backend's declared format per matrix class.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct WeightFormats {
    pub attention: WeightFormat,
    pub ffn: WeightFormat,
    pub head: WeightFormat,
}

impl WeightFormats {
    /// The same format everywhere.
    pub fn uniform(format: WeightFormat) -> Self {
        Self {
            attention: format,
            ffn: format,
            head: format,
        }
    }

    pub fn for_class(&self, class: MatrixClass) -> WeightFormat {
        match class {
            MatrixClass::AttentionProjection => self.attention,
            // A device backend places the bank exactly as it places any
            // other FFN matrix — the distinction the class draws is about
            // the HOST load path, not about device residency.
            MatrixClass::FfnProjection | MatrixClass::RoutedExpertBank => self.ffn,
            MatrixClass::OutputHead => self.head,
        }
    }
}

/// One matrix operand, in the representation the backend declared.
///
/// An `F16` slice may be longer than the matrix needs (page-padded for
/// zero-copy device wrapping); geometry always travels separately.
#[derive(Clone, Copy)]
pub enum WeightSlice<'a> {
    F32(&'a [f32]),
    /// Stored bf16 code units, still compact.
    Bf16(&'a [u16]),
    /// Symmetric int8 codes and their per-block f32 scales.
    Q8 {
        codes: &'a [i8],
        scales: &'a [f32],
        /// Per-`SUM_BLOCK` code sums; empty where no arm consumes them.
        sums: &'a [i16],
        block: usize,
    },
    /// Symmetric int4 codes packed two per byte, and their per-block f32
    /// scales. Byte `j` of a block holds elements `j` and `j + block/2`.
    Q4 {
        packed: &'a [u8],
        scales: &'a [f32],
        block: usize,
    },
    /// Little-endian IEEE f16 bytes.
    F16(&'a [u8]),
    /// MXFP4: packed e2m1 codes (`[n, k/32, 16]`, lo nibble first) and
    /// e8m0 scales (`[n, k/32]`) as two streams.
    Mxfp4 {
        packed: &'a [u8],
        scales: &'a [u8],
    },
    /// NVFP4: packed e2m1 codes (`[n, k/16, 8]`, lo nibble first), E4M3
    /// group scales (`[n, k/16]`), and the single f32 both scale levels
    /// are expressed relative to.
    Nvfp4 {
        packed: &'a [u8],
        scales: &'a [u8],
        tensor_scale: f32,
    },
    /// A stored ggml K-quant block stream, still compact, with the codec
    /// that names its layout. Scales live inside the blocks, so this is
    /// one stream, not two.
    KQuant {
        blocks: &'a [u8],
        codec: KQuant,
    },
}

impl<'a> WeightSlice<'a> {
    /// The f32 view a CPU backend computes with. A backend that declared
    /// `F32` can never legitimately receive `F16`, so this is fail-closed
    /// evidence of an interpreter bug, not a conversion point.
    /// The stored bf16 code units, when that is what was loaded.
    ///
    /// Deliberately NOT a widening accessor. A `Bf16` variant whose only
    /// consumer called `as_f32()` would give a tidy type and zero
    /// benefit: the whole point is that the compact bytes reach a kernel
    /// still compact.
    pub fn as_bf16(&self) -> Result<&'a [u16], VindexError> {
        match self {
            WeightSlice::Bf16(w) => Ok(w),
            _ => Err(VindexError::Parse(
                "backend declared bf16 weights but was handed another format — interpreter \
                 loaded the wrong representation"
                    .to_string(),
            )),
        }
    }

    /// The row-range view a CPU kernel consumes, cut to the matrix's
    /// real geometry.
    ///
    /// **The truncation is load-bearing.** A resident slice may be LONGER
    /// than `out_dim * in_dim`: `AlignedBytes` pads every allocation up to
    /// the device page so a Metal buffer can wrap it zero-copy, and the
    /// padding is zeros. A kernel handed the whole slice would compute
    /// `len / in_dim` rows — more rows than the matrix has — and the
    /// executor would partition the wrong total across its workers.
    ///
    /// Qwen3.8 cannot show this. Every one of its matrices happens to be
    /// an exact multiple of the 16 KiB page, so the padded and logical
    /// lengths coincide and a version of this that forgot to truncate
    /// would decode the model perfectly. The gate uses a shape that is
    /// not a page multiple for exactly that reason.
    pub fn rows(&self, out_dim: usize, in_dim: usize) -> Result<WeightRows<'a>, VindexError> {
        let want = out_dim * in_dim;
        let short = |have: usize| {
            VindexError::Parse(format!(
                "a {out_dim} x {in_dim} projection needs {want} weights but only {have} are                  resident"
            ))
        };
        match self {
            WeightSlice::F32(w) => w
                .get(..want)
                .map(WeightRows::F32)
                .ok_or_else(|| short(w.len())),
            WeightSlice::Bf16(w) => w
                .get(..want)
                .map(WeightRows::Bf16)
                .ok_or_else(|| short(w.len())),
            WeightSlice::Q8 {
                codes,
                scales,
                sums,
                block,
            } => {
                let per_row = in_dim.div_ceil(*block);
                // The index is cut to the same geometry as the codes, or
                // stays empty. A partially-sliced index would pair a row
                // with another row's sums and still return finite numbers.
                let per_sum = in_dim.div_ceil(SUM_BLOCK);
                let cut = if sums.is_empty() {
                    Some(&sums[..0])
                } else {
                    sums.get(..out_dim * per_sum)
                };
                match (codes.get(..want), scales.get(..out_dim * per_row), cut) {
                    (Some(codes), Some(scales), Some(sums)) => Ok(WeightRows::Q8 {
                        codes,
                        scales,
                        sums,
                        block: *block,
                    }),
                    _ => Err(short(codes.len())),
                }
            }
            WeightSlice::Q4 {
                packed,
                scales,
                block,
            } => {
                let per_row = in_dim.div_ceil(*block);
                // Two codes to the byte, so the code stream is HALF the
                // element count. Asking for `want` bytes here would demand
                // twice the matrix and reject every legitimate operand.
                match (packed.get(..want / 2), scales.get(..out_dim * per_row)) {
                    (Some(packed), Some(scales)) => Ok(WeightRows::Q4 {
                        packed,
                        scales,
                        block: *block,
                    }),
                    _ => Err(short(packed.len() * 2)),
                }
            }
            WeightSlice::Nvfp4 {
                packed,
                scales,
                tensor_scale,
            } => {
                // Groups run along the input axis and the group size is
                // the format's, not a policy's: `k/16` scale bytes and
                // `k/2` code bytes per row.
                const GROUP: usize = 16;
                if !in_dim.is_multiple_of(GROUP) {
                    return Err(VindexError::Parse(format!(
                        "NVFP4 slab: in_dim={in_dim} is not a multiple of the {GROUP}-element \
                         group, so this pack does not describe these rows"
                    )));
                }
                let groups_per_row = in_dim / GROUP;
                match (
                    packed.get(..want / 2),
                    scales.get(..out_dim * groups_per_row),
                ) {
                    (Some(packed), Some(scales)) => Ok(WeightRows::Nvfp4 {
                        packed,
                        scales,
                        tensor_scale: *tensor_scale,
                    }),
                    _ => Err(short(packed.len() * 2)),
                }
            }
            WeightSlice::KQuant { blocks, codec } => {
                // The stride is the codec's: blocks run along the row,
                // and a width off the block grid describes no rows.
                let Some(per_row) = codec.row_bytes(in_dim) else {
                    return Err(VindexError::Parse(format!(
                        "{} slab: in_dim={in_dim} is not a whole number of {}-element blocks, \
                         so this pack does not describe these rows",
                        codec.name, codec.elements_per_block
                    )));
                };
                // EXACT, not a prefix cut like the arms above. Those
                // tolerate a longer slice because `AlignedBytes` pads to
                // the device page; a K-quant is plain owned bytes bound at
                // load against the codec's own plan of the shape, so any
                // length but the matrix means the codec or the geometry is
                // wrong. The LONGER direction is the dangerous one: Q6_K
                // bytes read as Q4_K are longer than Q4_K wants, would pass
                // a prefix cut, and would then be walked at a 144-byte
                // stride over 210-byte blocks — finite, plausible, wrong.
                let want_bytes = out_dim * per_row;
                if blocks.len() < want_bytes {
                    return Err(short(blocks.len() / per_row * in_dim));
                }
                if blocks.len() > want_bytes {
                    return Err(VindexError::Parse(format!(
                        "{} slab: {} bytes describe more than a {out_dim} x {in_dim} matrix's \
                         {want_bytes} — the bytes are not this codec's, or not this shape's",
                        codec.name,
                        blocks.len()
                    )));
                }
                Ok(WeightRows::KQuant {
                    blocks,
                    codec: *codec,
                })
            }
            other => Err(VindexError::Parse(format!(
                "no CPU projection kernel consumes {} weights — the backend declared a \
                 representation only a device can run, so this refuses rather than converting \
                 mid-decode",
                other.representation()
            ))),
        }
    }

    /// This slice's representation, for diagnostics. Never dispatched on
    /// — a backend that branched on the name instead of the variant would
    /// be one `match` away from silently accepting a format it cannot run.
    pub fn representation(&self) -> &'static str {
        match self {
            WeightSlice::F32(_) => "f32",
            WeightSlice::Bf16(_) => "bf16",
            WeightSlice::Q8 { .. } => "q8",
            WeightSlice::Q4 { .. } => "q4",
            WeightSlice::F16(_) => "f16",
            WeightSlice::Mxfp4 { .. } => "mxfp4",
            WeightSlice::Nvfp4 { .. } => "nvfp4",
            WeightSlice::KQuant { codec, .. } => codec.name,
        }
    }

    pub fn as_f32(&self) -> Result<&'a [f32], VindexError> {
        match self {
            WeightSlice::F32(w) => Ok(w),
            WeightSlice::Bf16(_)
            | WeightSlice::Q8 { .. }
            | WeightSlice::Q4 { .. }
            | WeightSlice::F16(_)
            | WeightSlice::Mxfp4 { .. }
            | WeightSlice::Nvfp4 { .. }
            | WeightSlice::KQuant { .. } => Err(VindexError::Parse(
                "backend declared f32 weights but was handed another format — interpreter \
                 loaded the wrong representation"
                    .to_string(),
            )),
        }
    }
}

/// One normalisation, fully resolved.
///
/// `weight` empty means a parameter-free application (statistic only) —
/// the interpreter decides that from the plan, never the backend.
pub struct NormCall<'a> {
    pub kind: NormType,
    pub x: &'a [f32],
    pub weight: &'a [f32],
    pub weight_offset: f32,
    pub eps: f64,
}

/// One `[out, in]` row-major projection applied to one vector.
pub struct ProjectCall<'a> {
    pub weight: WeightSlice<'a>,
    pub out_dim: usize,
    pub in_dim: usize,
    pub x: &'a [f32],
}

/// QK normalisation weights and scope, when the plan binds them.
pub struct QkNormCall<'a> {
    pub scope: QkNormScope,
    pub weight_offset: f32,
    pub q_weight: &'a [f32],
    pub k_weight: &'a [f32],
}

/// The attention output gate, when the surface judged one.
pub struct GateCall<'a> {
    pub spec: AttentionGateSpec,
    pub weight: WeightSlice<'a>,
}

/// The judged attention-sink semantics plus the per-query-head logits,
/// f32 like every other elementwise operand.
pub struct SinkCall<'a> {
    pub spec: AttentionSinkSpec,
    /// `num_q_heads` logits.
    pub logits: &'a [f32],
}

/// The additive projection biases, all four present or none — closure
/// guarantees the pairing with the surface's `attention_bias`. Each is
/// one value per output row of its projection.
pub struct BiasCall<'a> {
    pub q: &'a [f32],
    pub k: &'a [f32],
    pub v: &'a [f32],
    pub o: &'a [f32],
}

/// One attention operation over a whole sequence, fully resolved.
///
/// `inputs` are the attention *inputs* — already normalised by the
/// interpreter — because the judged gate reads that same vector, and
/// handing the backend one operand for both uses removes any chance of
/// the two drifting apart.
/// What a whole-sequence attention pass produces.
///
/// `outputs[p]` is position `p`'s attention output post
/// output-projection; `keys[p]` / `values[p]` are the conditioned rows
/// for that position — the rows a [`KvState`](super::kv::KvState)
/// provider caches, in the same form [`PlanBackend::attention_step`]
/// returns.
///
/// Positions are the sequence's own, starting at zero: a batched pass
/// conditions position `p` as the `p`-th token, so it cannot express a
/// prefill resuming part-way through a sequence. That is why the
/// executor still steps when extending a populated provider.
pub struct AttentionOut {
    pub outputs: Vec<Vec<f32>>,
    pub keys: Vec<Vec<f32>>,
    pub values: Vec<Vec<f32>>,
}

pub struct AttentionCall<'a> {
    pub inputs: &'a [Vec<f32>],
    pub hidden: usize,
    pub num_q_heads: usize,
    pub num_kv_heads: usize,
    pub head_dim: usize,
    pub w_q: WeightSlice<'a>,
    pub w_k: WeightSlice<'a>,
    pub w_v: WeightSlice<'a>,
    pub w_o: WeightSlice<'a>,
    pub qk_norm: Option<QkNormCall<'a>>,
    pub parameter_free_qk_norm: ParameterFreeQkNorm,
    /// Epsilon for both QK-norm forms; rides with the layer's norm
    /// surface because neither form declares its own.
    pub qk_norm_eps: f64,
    /// `None` = no query-scale operation, never an invented 1.0.
    pub query_scale: Option<f64>,
    pub score_scale: f64,
    pub logit_softcapping: Option<f32>,
    pub position: PositionPolicy,
    pub span: AttentionSpan,
    pub window: Option<usize>,
    pub gate: Option<GateCall<'a>>,
    /// Q/K/V/O biases: Q and K added right after projection (before
    /// QK-norm and rope), V before caching, O after the output
    /// projection. `None` = the op has no biases.
    pub bias: Option<BiasCall<'a>>,
    /// Attention sinks; `None` = ordinary softmax.
    pub sinks: Option<SinkCall<'a>>,
}

/// One feed-forward operation over one vector, fully resolved.
///
/// `gate` present means gated; absent means standard. Again the
/// interpreter reads that from the plan.
pub struct FfnCall<'a> {
    pub x: &'a [f32],
    pub hidden: usize,
    pub intermediate: usize,
    pub gate: Option<WeightSlice<'a>>,
    pub up: WeightSlice<'a>,
    pub down: WeightSlice<'a>,
    pub activation: Activation,
    /// How `gate` combines with `up`. Every backend must honour it or
    /// refuse: computing `activation(gate) * up` for a `ClampedGlu` plan
    /// runs a different model.
    pub gate_policy: larql_models::ExpertGatePolicy,
}

/// The same dense feed-forward operation over SEVERAL positions.
///
/// One weight traversal per projection where the backend has a kernel for
/// it, rather than one per position. Deliberately a separate call rather
/// than `FfnCall` with a slice of inputs: a backend that has no
/// multi-position kernel should keep consuming `FfnCall` unchanged, and
/// the default [`PlanBackend::ffn_many`] gives it exactly that.
///
/// The activation stays PER POSITION. Only the projections group.
pub struct FfnManyCall<'a> {
    pub xs: &'a [&'a [f32]],
    pub hidden: usize,
    pub intermediate: usize,
    pub gate: Option<WeightSlice<'a>>,
    pub up: WeightSlice<'a>,
    pub down: WeightSlice<'a>,
    pub activation: Activation,
    pub gate_policy: larql_models::ExpertGatePolicy,
}

/// One routed feed-forward operation over one vector, fully resolved:
/// the router in f32 (glue-sized), every expert's projections in the
/// backend's declared FFN format, and every judged semantic as an
/// argument. The backend routes, runs the selected experts and combines
/// — nothing here is re-derived from the plan.
pub struct RoutedFfnCall<'a> {
    pub x: &'a [f32],
    pub hidden: usize,
    /// Per-expert intermediate width.
    pub intermediate: usize,
    pub experts: usize,
    pub top_k: usize,
    pub router_kind: MoeRouterKind,
    pub routing_policy: ExpertRoutingPolicy,
    pub activation: Activation,
    pub gate_policy: larql_models::ExpertGatePolicy,
    /// Router logits matrix `[experts, hidden]`, row-major.
    pub router: &'a [f32],
    /// Additive router bias `[experts]`.
    pub router_bias: Option<&'a [f32]>,
    /// The experts' matrices, in the shape their bank stores them.
    pub weights: ExpertSlices<'a>,
    /// Fused gate/up bias, `[experts · 2·intermediate]` flat, in the
    /// operand's own row layout. Packed banks only.
    pub gate_up_bias: Option<&'a [f32]>,
    /// Down bias, `[experts · hidden]` flat. Packed banks only.
    pub down_bias: Option<&'a [f32]>,
    /// What the router reads. Every family but Gemma 4 routes on the same
    /// vector the experts consume (`x`); Gemma 4's router reads the RAW
    /// post-attention residual and conditions it itself. `None` = `x`.
    pub router_input: Option<&'a [f32]>,
    /// `MoeRouterKind::Gemma4Hybrid` conditioning, present iff the plan
    /// carries it: `router_input` is RMS-normalised without a weight
    /// (`router_norm_eps`), multiplied by `router_scale` `[hidden]` and by
    /// `hidden^-0.5` before the projection; the renormalised top-k weights
    /// are multiplied by `router_per_expert_scale[selected]`.
    pub router_scale: Option<&'a [f32]>,
    pub router_per_expert_scale: Option<&'a [f32]>,
    pub router_norm_eps: Option<f64>,
}

/// How a routed layer's expert matrices are handed to a backend: the
/// shape the bank STORES them in, never converted to the other.
pub enum ExpertSlices<'a> {
    /// A packed bank: one fused `[2·intermediate, hidden]` gate/up per
    /// expert, split by the call's `gate_up_layout`, and one
    /// `[hidden, intermediate]` down.
    Fused {
        gate_up: &'a [WeightSlice<'a>],
        down: &'a [WeightSlice<'a>],
        /// How each expert's fused rows split into gate and up — a
        /// property of the fused operand, so it travels with it.
        layout: GateUpLayout,
    },
    /// A per-expert bank: gate `[intermediate, hidden]`, up
    /// `[intermediate, hidden]` and down `[hidden, intermediate]` as three
    /// whole matrices per expert — each one the stored operand it is.
    Separate {
        gate: &'a [WeightSlice<'a>],
        up: &'a [WeightSlice<'a>],
        down: &'a [WeightSlice<'a>],
    },
}

/// One position's attention against interpreter-owned K/V state — the
/// decode step.
///
/// `op.inputs` holds exactly one row: this position's already-normalised
/// attention input. `keys`/`values` are the post-norm, post-rope K and V
/// rows of every earlier position, exactly as this backend returned them
/// from earlier steps — the interpreter owns the cache; the backend owns
/// only the arithmetic of one step.
pub struct AttentionStepCall<'a> {
    /// The resolved attention operation, identical in meaning to the
    /// batch call — one struct so the two paths cannot drift apart in
    /// what they carry.
    pub op: AttentionCall<'a>,
    /// Absolute position of the row in `op.inputs`.
    pub position: usize,
    /// Cached K rows for positions `0..position`.
    pub keys: &'a [Vec<f32>],
    /// Cached V rows for positions `0..position`.
    pub values: &'a [Vec<f32>],
}

/// One position's projected, conditioned (Q, K, V) — the intermediate
/// every backend's projection helper produces.
pub type ProjectedQkv = (Vec<f32>, Vec<f32>, Vec<f32>);

/// What one decode step returns: this position's K and V rows (for the
/// interpreter to append to its cache) and the attention output
/// (post gate, post output-projection).
pub struct AttentionStepOut {
    pub key: Vec<f32>,
    pub value: Vec<f32>,
    pub output: Vec<f32>,
}

/// The numerical realisation of a plan's operations.
///
/// Every method is total over its arguments: the caller has already
/// decided the operation happens. A backend may fail on work it cannot
/// perform (an unimplemented QK-norm scope, a device error), but it may
/// not decline work on semantic grounds — that judgment was made before
/// the call.
///
/// `Sync` because the interpreter issues per-position calls from
/// worker threads. Positions are independent through every operation
/// (attention reads other positions' K/V but never writes them), so
/// this parallelism reorders nothing within any one position's
/// arithmetic — results stay bit-identical to a serial execution.
/// What a backend spent inside its own dispatch calls, for attributing a
/// token's latency between device work and the interpreter's glue.
///
/// Exists because "the part that does not scale with weight bytes" is not
/// automatically submission overhead: the elementwise glue (norms, RoPE,
/// softmax over the KV cache, activations, residuals) is also a fixed
/// per-token cost, and optimising the wrong one of the two is free to
/// look like progress on a fit that cannot tell them apart.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct DispatchStats {
    /// Wall nanoseconds inside device dispatch calls — submission,
    /// device execution, and the wait, together.
    pub device_nanos: u64,
    /// Device submissions made (one per command buffer).
    pub submissions: u64,
}

pub trait PlanBackend: Sync {
    /// A name for diagnostics and parity reports. Not dispatched on.
    fn name(&self) -> &str;

    /// Cumulative device-dispatch accounting, when the backend keeps it.
    /// `None` for backends with no device to account for.
    fn dispatch_stats(&self) -> Option<DispatchStats> {
        None
    }

    /// The realization this backend takes for one planned operand, given
    /// what the registry declares for its stored representation.
    ///
    /// Chosen from candidates the backend derives from those declarations,
    /// or refused naming every candidate considered. Asked per operand at
    /// preparation, BEFORE any byte is read, and pinned in the prepared
    /// plan: the executor runs what was pinned or refuses. The default is
    /// the reference oracle — the literal f32 transcription — so a backend
    /// that says nothing inherits the arithmetic that defines correctness.
    fn select(
        &self,
        operand: &PlannedOperand,
        facts: &RepresentationFacts,
    ) -> Result<Selection, Box<SelectionRefusal>> {
        super::realization::reference_selection(operand, facts)
    }

    /// How this backend performs a Gated DeltaNet layer's dense
    /// projections.
    ///
    /// Defaults to the literal scalar transcription, so a backend that
    /// says nothing gets the reference arithmetic rather than inheriting
    /// somebody else's. The recurrence itself is not selectable — only
    /// the five matrix products around it.
    fn dense_projector(&self) -> &dyn super::gated_delta::DenseProjections {
        &super::gated_delta::ScalarProjections
    }

    /// Residency hint before a decode run: every matrix operand the
    /// session will read, already loaded. Computes nothing and must
    /// change no number — a backend may warm caches or wire device
    /// memory, or ignore it entirely. The default does nothing.
    fn prepare(&self, _weights: &[WeightSlice<'_>]) {}

    /// Look up one embedding row, applying the scale operation when the
    /// plan carries one. `scale` `None` = no such operation, so the row
    /// is returned unscaled rather than multiplied by an identity.
    fn embed(&self, table: &[f32], hidden: usize, token: u32, scale: Option<f32>) -> Vec<f32>;

    fn norm(&self, call: NormCall<'_>) -> Vec<f32>;

    /// Fallible for the same reason as [`Self::attention`]: a device
    /// backend may be unable to perform the work, and it must say so
    /// rather than borrow another backend's arithmetic.
    fn project(&self, call: ProjectCall<'_>) -> Result<Vec<f32>, VindexError>;

    /// Attention over the whole sequence.
    ///
    /// Returns the conditioned K/V rows alongside the outputs because
    /// the realisation already computes them: it must, to attend at
    /// all. Discarding them is what forced a caller that wanted a
    /// populated K/V cache down [`Self::attention_step`] instead —
    /// coupling "I want KV" to "run attention one position at a time"
    /// (V3-SERVE-2).
    ///
    /// The rows must be the same rows [`Self::attention_step`] would
    /// produce for the same position and input; both realisations of a
    /// backend answer for one program, and the attention-parity gates
    /// pin them together.
    fn attention(&self, call: AttentionCall<'_>) -> Result<AttentionOut, VindexError>;

    /// One position's attention against cached K/V — the decode step.
    ///
    /// Must realise exactly the arithmetic its own [`Self::attention`]
    /// applies to a single position: the decode-vs-batch parity tests
    /// pin the two paths together per backend, and a backend may not
    /// borrow another backend's step to fill the gap.
    fn attention_step(&self, call: AttentionStepCall<'_>) -> Result<AttentionStepOut, VindexError>;

    /// Fallible for the same reason as [`Self::attention`]: a backend
    /// with no kernel for a judged variant must say so, not borrow
    /// another backend's arithmetic to fill the gap.
    fn ffn(&self, call: FfnCall<'_>) -> Result<Vec<f32>, VindexError>;

    /// The dense FFN over several positions at once.
    ///
    /// Default is the loop it replaces, so every backend keeps working
    /// untouched. Overriding it is a claim about SCHEDULE only: each
    /// position keeps its own activation and its own arithmetic, and the
    /// results must be indistinguishable from calling [`Self::ffn`] once
    /// per position.
    fn ffn_many(&self, call: FfnManyCall<'_>) -> Result<Vec<Vec<f32>>, VindexError> {
        call.xs
            .iter()
            .map(|x| {
                self.ffn(FfnCall {
                    x,
                    hidden: call.hidden,
                    intermediate: call.intermediate,
                    gate: call.gate,
                    up: call.up,
                    down: call.down,
                    activation: call.activation,
                    gate_policy: call.gate_policy,
                })
            })
            .collect()
    }

    /// The routed FFN — a mixture of experts. Required of every backend
    /// for the same reason as [`Self::ffn`]: a backend without the
    /// arithmetic must refuse, never borrow it.
    /// Multiply one hidden row by a scalar in place — Gemma 4's
    /// `layer_scalar` on the whole layer output. Elementwise glue like
    /// [`Self::residual_add`]; a backend overrides only to keep the row on
    /// its device.
    fn scale_row(&self, row: &mut [f32], scale: f32) {
        for value in row {
            *value *= scale;
        }
    }

    fn routed_ffn(&self, call: RoutedFfnCall<'_>) -> Result<Vec<f32>, VindexError>;

    /// Vocabulary projection plus the head's optional multiplier and
    /// softcap, in that order.
    fn output_head(
        &self,
        projection: WeightSlice<'_>,
        vocab: usize,
        hidden: usize,
        x: &[f32],
        multiplier: Option<f64>,
        softcapping: Option<f32>,
    ) -> Result<Vec<f32>, VindexError>;

    /// Add `delta` into `acc` elementwise — the residual write.
    ///
    /// A method rather than a loop in the interpreter because residual
    /// accumulation order is exactly the kind of thing a fused production
    /// kernel wants to own, and because a backend that reassociates it
    /// should have to say so.
    fn residual_add(&self, acc: &mut [f32], delta: &[f32]);
}
