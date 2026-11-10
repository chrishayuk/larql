//! Weight slices and the per-op call records a backend receives.

use super::super::super::super::graph::policy::AttentionSpan;
use super::super::cpu::WeightRows;
use super::super::quantise::SUM_BLOCK;
use crate::error::VindexError;
use larql_models::config::{
    Activation, AttentionGateSpec, AttentionSinkSpec, NormType, ParameterFreeQkNorm,
    PositionPolicy, QkNormScope,
};

#[allow(unused_imports)]
use super::*;

impl<'a> WeightSlice<'a> {
    /// Materialise the effective f32 matrix represented by this resident
    /// slice. This is an offline evidence path, not an execution path: GW-0B
    /// uses it to express contribution addresses in the exact weight image
    /// the selected kernel consumed.
    pub fn decode_f32(&self, out_dim: usize, in_dim: usize) -> Result<Vec<f32>, VindexError> {
        let rows = self.rows(out_dim, in_dim)?;
        match rows {
            WeightRows::F32(values) => Ok(values.to_vec()),
            WeightRows::Bf16(values) => Ok(values
                .iter()
                .map(|bits| f32::from_bits(u32::from(*bits) << 16))
                .collect()),
            WeightRows::Q8 {
                codes,
                scales,
                block,
                ..
            } => {
                let blocks_per_row = in_dim.div_ceil(block);
                Ok((0..out_dim * in_dim)
                    .map(|index| {
                        let row = index / in_dim;
                        let column = index % in_dim;
                        f32::from(codes[index]) * scales[row * blocks_per_row + column / block]
                    })
                    .collect())
            }
            _ => Err(VindexError::Parse(
                "offline dense-FFN attribution cannot materialise this resident weight form"
                    .to_string(),
            )),
        }
    }

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
                activation,
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
                        activation: *activation,
                    }),
                    _ => Err(short(packed.len() * 2)),
                }
            }
            WeightSlice::KQuant {
                blocks,
                codec,
                activation,
            } => {
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
                    activation: *activation,
                })
            }
            WeightSlice::Fp8Block {
                codes,
                scales,
                block_rows,
                block_cols,
                scale_cols,
            } => {
                // One byte per element, so the declared geometry and the
                // stored length must agree exactly. A slab that was
                // merely LONG ENOUGH would put row 1 at the right offset
                // and every scale at the wrong one.
                let want = out_dim * in_dim;
                if codes.len() != want {
                    return Err(VindexError::Parse(format!(
                        "fine-grained FP8 slab: {} E4M3 bytes do not describe a \
                         {out_dim} x {in_dim} matrix's {want}",
                        codes.len()
                    )));
                }
                Ok(WeightRows::Fp8Block {
                    codes,
                    scales,
                    block_rows: *block_rows,
                    block_cols: *block_cols,
                    scale_cols: *scale_cols,
                    // A whole matrix starts at the top of its first tile.
                    // Only `WeightRows::slice_rows` produces any other
                    // offset, and it derives it.
                    row_in_tile: 0,
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
    pub fn representation(&self) -> &'a str {
        match self {
            WeightSlice::F32(_) => "f32",
            WeightSlice::Bf16(_) => "bf16",
            WeightSlice::Q8 { .. } => "q8",
            WeightSlice::Q4 { .. } => "q4",
            WeightSlice::F16(_) => "f16",
            WeightSlice::Mxfp4 { .. } => "mxfp4",
            WeightSlice::Nvfp4 { .. } => "nvfp4",
            WeightSlice::KQuant { codec, .. } => codec.name,
            WeightSlice::Fp8Block { .. } => "fp8-block",
            WeightSlice::CodecOwned { label, .. } => label,
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
            | WeightSlice::KQuant { .. }
            | WeightSlice::Fp8Block { .. }
            | WeightSlice::CodecOwned { .. } => Err(VindexError::Parse(
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

/// The additive projection biases. Q/K/V are always present together;
/// the output bias is present under `attention_bias` and absent under
/// `qkv_bias` — closure guarantees each pairing with its declaration.
/// Each is one value per output row of its projection.
pub struct BiasCall<'a> {
    pub q: &'a [f32],
    pub k: &'a [f32],
    pub v: &'a [f32],
    pub o: Option<&'a [f32]>,
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
/// for that position — the rows a [`KvState`](super::super::kv::KvState)
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
    /// How RoPE cos/sin are constructed (PREC-DIAG-1 E0b). A per-run
    /// diagnostic choice, not a plan fact; [`RopeAngles::Larql`] by default.
    ///
    /// [`RopeAngles::Larql`]: super::super::rope_angles::RopeAngles::Larql
    pub rope_angles: super::super::rope_angles::RopeAngles,
}

impl AttentionCall<'_> {
    /// The retention authority for this call: the span and window it
    /// carries were copied from its plan op, and the policy that reads
    /// them is [`HistoryRange::of_span`](super::super::kv::HistoryRange::of_span).
    pub fn history(&self) -> super::super::kv::HistoryRange {
        super::super::kv::HistoryRange::of_span(self.span, self.window)
    }
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
