//! Loading a layer's FFN operands — dense or routed — in the backend's
//! declared format, and building the resolved call.
//!
//! The routed case binds a **packed expert bank**: every expert's
//! projections live in one operand (`[experts, rows, k]`, MXFP4 blocks
//! plus a scales stream, or unquantised BF16). Per expert, the loader
//! either binds the stored bytes as they are — an MXFP4 expert for a
//! backend that declared MXFP4 is a copy into aligned memory and nothing
//! else — or converts through f32 to the format the backend asked for,
//! exactly as `load_weight` does for a dense matrix. One resolution path,
//! so the batch executor and the decode session cannot drift.

use larql_models::config::ExpertFormat;
use larql_models::quant::mxfp4::{dequantize_expert, MXFP4_GROUP_BYTES, MXFP4_GROUP_ELEMS};

use super::backend::{FfnCall, NormCall, RoutedFfnCall, WeightFormat, WeightSlice};
use super::narrow::{bf16_bytes_to_f16, f32_bytes_to_f16};
use super::operands::{widen, OperandSource};
use super::realization::RepresentationFacts;
use super::weights::{load_weight, quantize_mxfp4, quantize_nvfp4, AlignedBytes, LoadedWeight};
use crate::error::VindexError;
use crate::format::vindex3::opplan::{
    ExpertBank, FfnOp, LayerFfn, NormOp, OperandRef, PackedProjection, RoutedFfnOp,
};

/// Stored dtype of MXFP4 block and scale streams.
const DTYPE_U8: &str = "U8";
/// Stored dtype of an unquantised packed bank.
const DTYPE_BF16: &str = "BF16";
/// Gate and up: the two branches sharing one fused operand.
const FUSED_BRANCHES: usize = larql_models::quant::mxfp4::FUSED_HALVES;

/// A layer's FFN operands, loaded once in the backend's declared format.
pub(super) enum FfnOperands {
    Dense(DenseOperands),
    Routed(RoutedOperands),
    /// Gemma 4: both, plus the three branch norms (f32 glue) — see
    /// [`super::HybridFfnOp`] for the program. Boxed: it is the sum of
    /// the other two plus three norms, several-fold the dense variant.
    Hybrid(Box<HybridOperands>),
}

/// A hybrid layer's operands: both branches and their norms.
pub(super) struct HybridOperands {
    dense: DenseOperands,
    routed: RoutedOperands,
    pre_experts_norm: LoadedNormWeight,
    post_dense_norm: LoadedNormWeight,
    post_experts_norm: LoadedNormWeight,
}

/// A dense layer's three (or two) matrices.
pub(super) struct DenseOperands {
    gate: Option<LoadedWeight>,
    up: LoadedWeight,
    down: LoadedWeight,
}

/// A norm's weight, loaded once beside the op that names it.
pub(super) struct LoadedNormWeight {
    op: NormOp,
    weight: Vec<f32>,
}

impl LoadedNormWeight {
    fn load(op: &NormOp, store: OperandSource<'_>) -> Result<Self, VindexError> {
        Ok(Self {
            op: op.clone(),
            weight: store.load(&op.weight)?,
        })
    }

    fn apply<B: super::backend::PlanBackend + ?Sized>(&self, backend: &B, x: &[f32]) -> Vec<f32> {
        backend.norm(NormCall {
            kind: self.op.kind,
            x,
            weight: &self.weight,
            weight_offset: self.op.weight_offset,
            eps: self.op.eps,
        })
    }
}

/// A routed layer's operands: router (f32 glue) and per-expert matrices,
/// plus Gemma 4's router conditioning when the op carries it.
pub(super) struct RoutedOperands {
    router: Vec<f32>,
    router_bias: Option<Vec<f32>>,
    router_scale: Option<Vec<f32>>,
    router_per_expert_scale: Option<Vec<f32>>,
    router_norm_eps: Option<f64>,
    gate_up: Vec<LoadedWeight>,
    gate_up_bias: Option<Vec<f32>>,
    down: Vec<LoadedWeight>,
    down_bias: Option<Vec<f32>>,
}

impl FfnOperands {
    pub(super) fn load(
        ffn: &LayerFfn,
        store: OperandSource<'_>,
        format: super::prepared::FormatFor<'_>,
        bank: WeightFormat,
    ) -> Result<Self, VindexError> {
        match ffn {
            LayerFfn::Dense(op) => Ok(Self::Dense(DenseOperands::load(op, store, format)?)),
            LayerFfn::Routed(op) => Ok(Self::Routed(RoutedOperands::load(op, store, bank)?)),
            LayerFfn::Hybrid(op) => Ok(Self::Hybrid(Box::new(HybridOperands {
                dense: DenseOperands::load(&op.dense, store, format)?,
                routed: RoutedOperands::load(&op.routed, store, bank)?,
                pre_experts_norm: LoadedNormWeight::load(&op.pre_experts_norm, store)?,
                post_dense_norm: LoadedNormWeight::load(&op.post_dense_norm, store)?,
                post_experts_norm: LoadedNormWeight::load(&op.post_experts_norm, store)?,
            }))),
        }
    }

    /// The dense projections only — what a pinned projection realization
    /// is checked against. A bank's per-expert slices are a different
    /// realization and are not projections of the plan's operands.
    pub(super) fn dense_matrices(&self) -> Vec<&LoadedWeight> {
        match self {
            Self::Dense(d) => d.loaded_matrices(),
            Self::Routed(_) => Vec::new(),
            Self::Hybrid(h) => h.dense.loaded_matrices(),
        }
    }

    /// Every matrix operand, for residency accounting.
    pub(super) fn loaded_matrices(&self) -> Vec<&LoadedWeight> {
        match self {
            Self::Dense(dense) => dense.loaded_matrices(),
            Self::Routed(routed) => routed.loaded_matrices(),
            Self::Hybrid(hybrid) => {
                let mut all = hybrid.dense.loaded_matrices();
                all.extend(hybrid.routed.loaded_matrices());
                all
            }
        }
    }

    /// Every matrix operand, for residency preparation.
    pub(super) fn weight_slices(&self) -> Vec<WeightSlice<'_>> {
        match self {
            Self::Dense(dense) => dense.weight_slices(),
            Self::Routed(routed) => routed.weight_slices(),
            Self::Hybrid(hybrid) => {
                let mut slices = hybrid.dense.weight_slices();
                slices.extend(hybrid.routed.weight_slices());
                slices
            }
        }
    }

    /// Run this layer's FFN over one normalised vector on `backend` — the
    /// dense-only and routed-only shapes, which read one input.
    pub(super) fn apply<B: super::backend::PlanBackend + ?Sized>(
        &self,
        ffn: &LayerFfn,
        backend: &B,
        x: &[f32],
        hidden: usize,
    ) -> Result<Vec<f32>, VindexError> {
        match (self, ffn) {
            (Self::Dense(dense), LayerFfn::Dense(op)) => dense.apply(op, backend, x, hidden),
            (Self::Routed(routed), LayerFfn::Routed(op)) => routed.apply(op, backend, x, x, hidden),
            _ => Err(VindexError::Parse(
                "FFN operands were loaded for a different op kind than the plan carries"
                    .to_string(),
            )),
        }
    }

    /// The whole FFN block from the post-attention residual up to — not
    /// including — the layer's post-FFN norm and residual add. Both
    /// drivers (batch and decode) call this, so the hybrid program lives
    /// in exactly one place:
    ///
    /// ```text
    /// dense/routed:  ffn(pre_ffn_normed)
    /// hybrid:        post_dense_norm(dense(pre_ffn_normed))
    ///              + post_experts_norm(routed(pre_experts_norm(residual), router ← residual))
    /// ```
    ///
    /// `pre_ffn_normed` is the layer's pre-FFN norm of `residual`, produced
    /// by the caller (it is also what the judged gate reads).
    pub(super) fn apply_from_residual<B: super::backend::PlanBackend + ?Sized>(
        &self,
        ffn: &LayerFfn,
        backend: &B,
        residual: &[f32],
        pre_ffn_normed: &[f32],
        hidden: usize,
    ) -> Result<Vec<f32>, VindexError> {
        match (self, ffn) {
            (Self::Hybrid(hybrid), LayerFfn::Hybrid(op)) => {
                let dense_out = hybrid
                    .dense
                    .apply(&op.dense, backend, pre_ffn_normed, hidden)?;
                let dense_out = hybrid.post_dense_norm.apply(backend, &dense_out);
                let expert_input = hybrid.pre_experts_norm.apply(backend, residual);
                let experts_out =
                    hybrid
                        .routed
                        .apply(&op.routed, backend, &expert_input, residual, hidden)?;
                let experts_out = hybrid.post_experts_norm.apply(backend, &experts_out);
                Ok(dense_out
                    .iter()
                    .zip(&experts_out)
                    .map(|(d, e)| d + e)
                    .collect())
            }
            (Self::Hybrid(_), _) | (_, LayerFfn::Hybrid(_)) => Err(VindexError::Parse(
                "FFN operands were loaded for a different op kind than the plan carries"
                    .to_string(),
            )),
            _ => self.apply(ffn, backend, pre_ffn_normed, hidden),
        }
    }

    /// [`Self::apply_from_residual`] over several positions at once.
    ///
    /// Only the DENSE arm groups. A routed or hybrid FFN selects experts
    /// per position, so its weight traversal is not shared between them
    /// and grouping it is a different rung with its own question — those
    /// arms keep the per-position program, which is the same arithmetic
    /// they ran before.
    pub(super) fn apply_from_residual_many<B: super::backend::PlanBackend + ?Sized>(
        &self,
        ffn: &LayerFfn,
        backend: &B,
        residuals: &[&[f32]],
        pre_ffn_normed: &[&[f32]],
        hidden: usize,
    ) -> Result<Vec<Vec<f32>>, VindexError> {
        match (self, ffn) {
            (Self::Dense(dense), LayerFfn::Dense(op)) => {
                dense.apply_many(op, backend, pre_ffn_normed, hidden)
            }
            _ => residuals
                .iter()
                .zip(pre_ffn_normed)
                .map(|(residual, normed)| {
                    self.apply_from_residual(ffn, backend, residual, normed, hidden)
                })
                .collect(),
        }
    }
}

impl DenseOperands {
    fn load(
        op: &FfnOp,
        store: OperandSource<'_>,
        format: super::prepared::FormatFor<'_>,
    ) -> Result<Self, VindexError> {
        Ok(Self {
            gate: match &op.gate {
                Some(gate) => Some(load_weight(store, gate, format(gate)?)?),
                None => None,
            },
            up: load_weight(store, &op.up, format(&op.up)?)?,
            down: load_weight(store, &op.down, format(&op.down)?)?,
        })
    }

    pub(super) fn loaded_matrices(&self) -> Vec<&LoadedWeight> {
        let mut all = vec![&self.up, &self.down];
        if let Some(gate) = &self.gate {
            all.push(gate);
        }
        all
    }

    fn weight_slices(&self) -> Vec<WeightSlice<'_>> {
        let mut slices = vec![self.up.slice(), self.down.slice()];
        if let Some(gate) = &self.gate {
            slices.push(gate.slice());
        }
        slices
    }

    fn apply<B: super::backend::PlanBackend + ?Sized>(
        &self,
        op: &FfnOp,
        backend: &B,
        x: &[f32],
        hidden: usize,
    ) -> Result<Vec<f32>, VindexError> {
        backend.ffn(FfnCall {
            x,
            hidden,
            intermediate: op.intermediate_size,
            gate: self.gate.as_ref().map(LoadedWeight::slice),
            up: self.up.slice(),
            down: self.down.slice(),
            activation: op.activation,
            gate_policy: op.gate_policy,
        })
    }

    fn apply_many<B: super::backend::PlanBackend + ?Sized>(
        &self,
        op: &FfnOp,
        backend: &B,
        xs: &[&[f32]],
        hidden: usize,
    ) -> Result<Vec<Vec<f32>>, VindexError> {
        backend.ffn_many(super::backend::FfnManyCall {
            xs,
            hidden,
            intermediate: op.intermediate_size,
            gate: self.gate.as_ref().map(LoadedWeight::slice),
            up: self.up.slice(),
            down: self.down.slice(),
            activation: op.activation,
            gate_policy: op.gate_policy,
        })
    }
}

impl RoutedOperands {
    /// Every expert matrix, for residency accounting. The router itself
    /// is f32 glue and is counted with the norms.
    fn loaded_matrices(&self) -> Vec<&LoadedWeight> {
        self.gate_up.iter().chain(&self.down).collect()
    }

    fn weight_slices(&self) -> Vec<WeightSlice<'_>> {
        self.gate_up
            .iter()
            .chain(&self.down)
            .map(LoadedWeight::slice)
            .collect()
    }

    /// The routed FFN over `x` (what the experts consume), routing on
    /// `router_input` (the same vector for every family but Gemma 4).
    fn apply<B: super::backend::PlanBackend + ?Sized>(
        &self,
        op: &RoutedFfnOp,
        backend: &B,
        x: &[f32],
        router_input: &[f32],
        hidden: usize,
    ) -> Result<Vec<f32>, VindexError> {
        let gate_up: Vec<WeightSlice<'_>> = self.gate_up.iter().map(LoadedWeight::slice).collect();
        let down: Vec<WeightSlice<'_>> = self.down.iter().map(LoadedWeight::slice).collect();
        backend.routed_ffn(RoutedFfnCall {
            x,
            hidden,
            intermediate: op.expert_intermediate_size,
            experts: op.experts,
            top_k: op.top_k,
            router_kind: op.router_kind,
            routing_policy: op.routing_policy,
            activation: op.activation,
            gate_policy: op.gate_policy,
            gate_up_layout: op.gate_up_layout.ok_or_else(|| {
                VindexError::Parse(
                    "routed FFN op carries no gate_up layout; closure requires one".to_string(),
                )
            })?,
            router: &self.router,
            router_bias: self.router_bias.as_deref(),
            gate_up: &gate_up,
            gate_up_bias: self.gate_up_bias.as_deref(),
            down: &down,
            down_bias: self.down_bias.as_deref(),
            router_input: (!std::ptr::eq(router_input, x)).then_some(router_input),
            router_scale: self.router_scale.as_deref(),
            router_per_expert_scale: self.router_per_expert_scale.as_deref(),
            router_norm_eps: self.router_norm_eps,
        })
    }

    fn load(
        op: &RoutedFfnOp,
        store: OperandSource<'_>,
        format: WeightFormat,
    ) -> Result<Self, VindexError> {
        // `ExpertBank::PerExpert` (Kimi Linear, Mixtral, DeepSeek): closure
        // plans the operand set (`opplan::build`), but no executor
        // consumes it yet — refuse loudly rather than misread `experts`
        // separate tensors as one packed bank of one.
        let ExpertBank::Packed { gate_up, down } = &op.bank else {
            return Err(VindexError::Parse(
                "routed FFN op carries a per-expert (unfused) bank; execution for \
                 `ExpertFormat::PerExpert` is not built yet"
                    .to_string(),
            ));
        };
        let hidden = op.router.shape.get(1).copied().unwrap_or(0);
        let inter = op.expert_intermediate_size;
        // The bank first: its geometry is DECLARED — `k` follows from the
        // router's declared width — and a stray width refuses on the
        // declaration, before any operand's bytes are read.
        let gate_up_bank = load_packed(store, gate_up, op, FUSED_BRANCHES * inter, hidden, format)?;
        let down_bank = load_packed(store, down, op, hidden, inter, format)?;
        Ok(Self {
            router: store.load(&op.router)?,
            router_bias: op.router_bias.as_ref().map(|b| store.load(b)).transpose()?,
            router_scale: op
                .router_scale
                .as_ref()
                .map(|s| store.load(s))
                .transpose()?,
            router_per_expert_scale: op
                .router_per_expert_scale
                .as_ref()
                .map(|s| store.load(s))
                .transpose()?,
            router_norm_eps: op.router_norm_eps,
            gate_up: gate_up_bank,
            gate_up_bias: gate_up.bias.as_ref().map(|b| store.load(b)).transpose()?,
            down: down_bank,
            down_bias: down.bias.as_ref().map(|b| store.load(b)).transpose()?,
        })
    }
}

/// Load one packed projection as `experts` matrices of `[rows, k]` in
/// `format`.
fn load_packed(
    store: OperandSource<'_>,
    projection: &PackedProjection,
    op: &RoutedFfnOp,
    rows: usize,
    k: usize,
    format: WeightFormat,
) -> Result<Vec<LoadedWeight>, VindexError> {
    let name = projection.weights.tensor.as_str();
    // Declared geometry before bytes: a `k` the group cannot tile is a
    // fact about the op, and refusing it costs no read.
    if op.expert_format == ExpertFormat::PackedMxfp4 && !k.is_multiple_of(MXFP4_GROUP_ELEMS) {
        return Err(VindexError::Parse(format!(
            "`{name}`: k={k} is not a multiple of the MXFP4 group"
        )));
    }
    require_row_access(store, &projection.weights)?;
    let raw = store.load_raw(&projection.weights)?;
    match op.expert_format {
        ExpertFormat::PackedMxfp4 => {
            let scales_ref = projection.scales.as_ref().ok_or_else(|| {
                VindexError::Parse(format!(
                    "`{name}`: MXFP4 expert projection carries no scales operand"
                ))
            })?;
            let scales = store.load_raw(scales_ref)?;
            expect_dtype(&raw.dtype, DTYPE_U8, name)?;
            expect_dtype(&scales.dtype, DTYPE_U8, &scales_ref.tensor)?;
            let groups = k / MXFP4_GROUP_ELEMS;
            let block_stride = rows * groups * MXFP4_GROUP_BYTES;
            let scale_stride = rows * groups;
            expect_len(raw.bytes.len(), op.experts * block_stride, name)?;
            expect_len(
                scales.bytes.len(),
                op.experts * scale_stride,
                &scales_ref.tensor,
            )?;
            (0..op.experts)
                .map(|e| {
                    let packed = &raw.bytes[e * block_stride..(e + 1) * block_stride];
                    let scale = &scales.bytes[e * scale_stride..(e + 1) * scale_stride];
                    match format {
                        // Native: the stored bytes are the operand.
                        WeightFormat::Mxfp4 => Ok(LoadedWeight::Mxfp4 {
                            packed: AlignedBytes::from_bytes(packed),
                            scales: AlignedBytes::from_bytes(scale),
                        }),
                        // Everything else converts through f32, exactly as
                        // a dense matrix would.
                        other => {
                            let values = dequantize_expert(packed, scale, rows, groups)
                                .map_err(|e| VindexError::Parse(format!("`{name}`: {e}")))?;
                            from_f32(values, rows, k, other, name)
                        }
                    }
                })
                .collect()
        }
        ExpertFormat::PackedBF16 => {
            expect_dtype(&raw.dtype, DTYPE_BF16, name)?;
            let stride = rows * k * 2;
            expect_len(raw.bytes.len(), op.experts * stride, name)?;
            (0..op.experts)
                .map(|e| {
                    let bytes = &raw.bytes[e * stride..(e + 1) * stride];
                    match format {
                        WeightFormat::F16 => Ok(LoadedWeight::F16(bf16_bytes_to_f16(bytes, name)?)),
                        other => from_f32(widen(DTYPE_BF16, bytes, name)?, rows, k, other, name),
                    }
                })
                .collect()
        }
        // Unreachable via `RoutedOperands::load` (it refuses `ExpertBank::
        // PerExpert` before this is called), kept as the function's own
        // invariant rather than trusting the one caller never to drift.
        ExpertFormat::PerExpert => Err(VindexError::Parse(format!(
            "`{name}`: per-expert tensors are not a packed projection"
        ))),
    }
}

/// One expert's `[rows, k]` f32 matrix, converted to `format`.
fn from_f32(
    values: Vec<f32>,
    rows: usize,
    k: usize,
    format: WeightFormat,
    name: &str,
) -> Result<LoadedWeight, VindexError> {
    match format {
        WeightFormat::F32 => Ok(LoadedWeight::F32(super::weights::staged::StagedF32::stage(
            values,
        )?)),
        WeightFormat::F16 => {
            let bytes: Vec<u8> = values.iter().flat_map(|v| v.to_le_bytes()).collect();
            Ok(LoadedWeight::F16(f32_bytes_to_f16(&bytes, name)?))
        }
        // A packed expert bank arrives already widened to f32, so the
        // CPU compact formats have no stored bytes to keep here — the
        // same reason `Bf16` is refused below. Naming them explicitly
        // rather than falling through keeps the refusal a decision.
        WeightFormat::Q4 => Err(VindexError::Parse(format!(
            "expert bank `{name}` cannot be made q4-resident: the bank is widened to f32 on \
             the way in, so there is nothing compact left to keep"
        ))),
        WeightFormat::KQuant => Err(VindexError::Parse(format!(
            "expert bank `{name}` cannot bind a stored K-quant: the bank is widened to f32 on \
             the way in, so the stored blocks are no longer what is being bound"
        ))),
        WeightFormat::Mxfp4 => quantize_mxfp4(&values, rows, k, name),
        WeightFormat::Nvfp4 => quantize_nvfp4(&values, rows, k, name),
        // This path has already widened to f32 (packed expert banks
        // arrive that way), and narrowing back would ROUND — bf16
        // residency means the stored bytes are the resident bytes, and
        // there are no stored bytes left here to keep. Refuse rather
        // than quietly return something the format does not promise.
        WeightFormat::Bf16 | WeightFormat::Q8 => Err(VindexError::Parse(format!(
            "tensor `{name}`: compact residency needs the stored bytes, and this expert path \
             has already widened to f32"
        ))),
    }
}

/// A packed bank is sliced per expert — arbitrary rows of the stored
/// bytes. Ask the representation whether it can be addressed that way
/// BEFORE its bytes are read, so a sequential codec is refused by its
/// access class, which is the planner's question, and not by dtype name
/// after the read was paid for.
///
/// Only a registered codec can answer. The MXFP4 bank's `U8` streams are
/// a dialect this loader judges itself below, and an unregistered label
/// keeps taking that path unchanged.
fn require_row_access(store: OperandSource<'_>, operand: &OperandRef) -> Result<(), VindexError> {
    let Some(dtype) = store.store().stored_dtype(operand) else {
        return Ok(());
    };
    // The same admission the selector made before this loader was
    // reached — one derivation, kept here so a direct caller gets the
    // same refusal rather than a read followed by a dtype-name refusal.
    RepresentationFacts::resolve(dtype).admit_row_slicing()?;
    Ok(())
}

fn expect_dtype(found: &str, expected: &str, name: &str) -> Result<(), VindexError> {
    if found == expected {
        Ok(())
    } else {
        Err(VindexError::Parse(format!(
            "`{name}`: expected stored dtype {expected}, found {found}"
        )))
    }
}

fn expect_len(found: usize, expected: usize, name: &str) -> Result<(), VindexError> {
    if found == expected {
        Ok(())
    } else {
        Err(VindexError::Parse(format!(
            "`{name}`: {found} stored bytes, expected {expected} for the declared expert geometry"
        )))
    }
}
