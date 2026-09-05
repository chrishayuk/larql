//! Loading a layer's FFN operands — dense or routed — in the backend's
//! declared format, and building the resolved call.
//!
//! The routed case binds a **packed expert bank**: every expert's
//! projections live in one operand (`[experts, rows, k]`), bound ONCE as
//! its codec's named streams over the whole `[experts × rows, k]` region.
//! The codec — the container's label when that names one, else the codec
//! the plan's declared layout carries (a packed MXFP4 bank is stored as
//! two `U8` streams) — validates the streams against that geometry and
//! decodes each expert as a row range of it, which the loader converts to
//! the format the backend asked for exactly as `load_weight` does for a
//! dense matrix. A backend that declared the bank's own format gets each
//! expert's stored rows copied into aligned memory and nothing else. One
//! resolution path, so the batch executor and the decode session cannot
//! drift, and no dtype name is judged here: the codec is.

use larql_models::quant::mxfp4::{MXFP4_GROUP_BYTES, MXFP4_GROUP_ELEMS};

use super::accounting::Bound;
use super::backend::{
    ExpertSlices, FfnCall, MatrixClass, NormCall, RoutedFfnCall, WeightFormat, WeightSlice,
};
use super::narrow::f32_bytes_to_f16;
use super::operands::OperandSource;
use super::realization::RepresentationFacts;
use super::weights::{
    load_weight, quantize_mxfp4, quantize_nvfp4, AlignedBytes, LoadedWeight, MappedForm,
};
use crate::error::VindexError;
use crate::format::vindex3::opplan::planned::{declared_bank_representation, Operation};
use crate::format::vindex3::opplan::{
    ExpertBank, FfnOp, LayerFfn, NormOp, OperandRef, PackedProjection, RoutedFfnOp,
};
use crate::format::vindex3::represent::codec::codecs::lyrw2::bind_region;
use crate::format::vindex3::represent::codec::codecs::mxfp4::DTYPE_MXFP4;
use crate::format::vindex3::represent::codec::streams::{GROUP_SCALES, VALUES};
use crate::format::vindex3::represent::codec::RepresentationExtent;

/// Gate and up: the two branches sharing one fused operand.
const FUSED_BRANCHES: usize = larql_models::quant::mxfp4::FUSED_HALVES;

/// A layer's FFN operands, loaded once in the backend's declared format.
pub(super) enum FfnOperands {
    /// Every variant boxed: the routed operands carry a bank and a shared
    /// branch, the hybrid both programs, and the dense one is then the odd
    /// one out — one pointer each keeps the enum the size of a word.
    Dense(Box<DenseOperands>),
    Routed(Box<RoutedOperands>),
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

/// A routed layer's operands: router (f32 glue), the experts' matrices in
/// the shape their bank stores them, the shared branch when the plan
/// carries one, plus Gemma 4's router conditioning when the op carries it.
pub(super) struct RoutedOperands {
    router: Vec<f32>,
    router_bias: Option<Vec<f32>>,
    router_scale: Option<Vec<f32>>,
    router_per_expert_scale: Option<Vec<f32>>,
    router_norm_eps: Option<f64>,
    experts: ExpertMatrices,
    gate_up_bias: Option<Vec<f32>>,
    down_bias: Option<Vec<f32>>,
    /// The always-active shared expert: three whole projections under
    /// the dense-FFN program, summed unscaled onto the routed output
    /// (`KimiSparseMoeBlock.forward`). The op the loader built for it is
    /// kept beside the operands so `apply` and `bound` read one program.
    shared: Option<(FfnOp, DenseOperands)>,
}

/// The experts' matrices, in the shape their bank stores them.
enum ExpertMatrices {
    /// A packed bank sliced per expert at load: one fused gate/up and one
    /// down per expert, in the backend's declared form.
    Fused {
        gate_up: Vec<LoadedWeight>,
        down: Vec<LoadedWeight>,
    },
    /// A per-expert bank bound as mapped regions of the stored bytes —
    /// one physical mapping per object, one region per matrix, nothing
    /// copied.
    Separate {
        gate: Vec<LoadedWeight>,
        up: Vec<LoadedWeight>,
        down: Vec<LoadedWeight>,
    },
}

/// Every matrix's slice, in order.
fn slices(w: &[LoadedWeight]) -> Vec<WeightSlice<'_>> {
    w.iter().map(LoadedWeight::slice).collect()
}

impl ExpertMatrices {
    fn all(&self) -> Vec<&LoadedWeight> {
        match self {
            Self::Fused { gate_up, down } => gate_up.iter().chain(down).collect(),
            Self::Separate { gate, up, down } => gate.iter().chain(up).chain(down).collect(),
        }
    }
}

impl FfnOperands {
    pub(super) fn load(
        ffn: &LayerFfn,
        store: OperandSource<'_>,
        format: super::prepared::FormatFor<'_>,
        bank: WeightFormat,
        shared: super::prepared::FormatFor<'_>,
    ) -> Result<Self, VindexError> {
        match ffn {
            LayerFfn::Dense(op) => Ok(Self::Dense(Box::new(DenseOperands::load(
                op, store, format,
            )?))),
            LayerFfn::Routed(op) => Ok(Self::Routed(Box::new(RoutedOperands::load(
                op, store, bank, shared,
            )?))),
            LayerFfn::Hybrid(op) => Ok(Self::Hybrid(Box::new(HybridOperands {
                dense: DenseOperands::load(&op.dense, store, format)?,
                routed: RoutedOperands::load(&op.routed, store, bank, shared)?,
                pre_experts_norm: LoadedNormWeight::load(&op.pre_experts_norm, store)?,
                post_dense_norm: LoadedNormWeight::load(&op.post_dense_norm, store)?,
                post_experts_norm: LoadedNormWeight::load(&op.post_experts_norm, store)?,
            }))),
        }
    }

    /// Every bound operand of this FFN, each under the OPERATION the
    /// loader bound it for — the dense projections one each, a packed
    /// bank as one operand over its per-expert objects, a per-expert bank
    /// one region per matrix, the shared branch as three projections.
    /// The loader names the operation because only it knows which object
    /// it bound for what; the accounting must not guess from counts.
    pub(super) fn bound<'a>(
        &'a self,
        ffn: &'a LayerFfn,
    ) -> Result<Vec<(Operation, Bound<'a>)>, VindexError> {
        let dense = |bounds: Vec<Bound<'a>>| {
            bounds
                .into_iter()
                .map(|b| (Operation::Project(MatrixClass::FfnProjection), b))
        };
        match (self, ffn) {
            (Self::Dense(d), LayerFfn::Dense(op)) => Ok(dense(d.bound(op)).collect()),
            (Self::Routed(r), LayerFfn::Routed(op)) => Ok(r.bound(op)),
            (Self::Hybrid(h), LayerFfn::Hybrid(op)) => {
                let mut out: Vec<_> = dense(h.dense.bound(&op.dense)).collect();
                out.extend(h.routed.bound(&op.routed));
                Ok(out)
            }
            _ => Err(VindexError::Parse(
                "the prepared FFN and the plan's FFN are different programs".to_string(),
            )),
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

    /// Each projection paired with the operand it binds.
    pub(super) fn bound<'a>(&'a self, op: &'a FfnOp) -> Vec<Bound<'a>> {
        let mut out = Vec::new();
        if let (Some(gate), Some(weight)) = (&op.gate, &self.gate) {
            out.push(Bound::one(gate, weight));
        }
        out.push(Bound::one(&op.up, &self.up));
        out.push(Bound::one(&op.down, &self.down));
        out
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
    /// The two banks, each one operand over its per-expert objects. A
    /// per-expert bank is refused before it is loaded, so it binds nothing.
    pub(super) fn bound<'a>(&'a self, op: &'a RoutedFfnOp) -> Vec<(Operation, Bound<'a>)> {
        let mut out: Vec<(Operation, Bound<'a>)> = match (&op.bank, &self.experts) {
            (
                ExpertBank::Packed { gate_up, down },
                ExpertMatrices::Fused {
                    gate_up: g,
                    down: d,
                },
            ) => vec![
                (
                    Operation::ExpertBankSlice,
                    Bound {
                        operand: &gate_up.weights,
                        weights: g.iter().collect(),
                    },
                ),
                (
                    Operation::ExpertBankSlice,
                    Bound {
                        operand: &down.weights,
                        weights: d.iter().collect(),
                    },
                ),
            ],
            // One planned operand per expert matrix, one mapped region
            // each: multiplicity for touch, the mapping shared underneath.
            (
                ExpertBank::PerExpert { gate, up, down },
                ExpertMatrices::Separate {
                    gate: g,
                    up: u,
                    down: d,
                },
            ) => {
                let bank = Operation::ExpertProject {
                    experts: op.experts,
                    top_k: op.top_k,
                };
                gate.iter()
                    .zip(g)
                    .chain(up.iter().zip(u))
                    .chain(down.iter().zip(d))
                    .map(|(operand, weight)| (bank, Bound::one(operand, weight)))
                    .collect()
            }
            // The loader binds the shape the plan declares; the other
            // pairing cannot be constructed.
            (ExpertBank::Packed { .. }, ExpertMatrices::Separate { .. })
            | (ExpertBank::PerExpert { .. }, ExpertMatrices::Fused { .. }) => Vec::new(),
        };
        if let Some((ffn, dense)) = &self.shared {
            out.extend(
                dense
                    .bound(ffn)
                    .into_iter()
                    .map(|b| (Operation::SharedExpertProject, b)),
            );
        }
        out
    }

    /// Every expert matrix, for residency accounting. The router itself
    /// is f32 glue and is counted with the norms.
    fn loaded_matrices(&self) -> Vec<&LoadedWeight> {
        let mut out = self.experts.all();
        if let Some((_, dense)) = &self.shared {
            out.extend(dense.loaded_matrices());
        }
        out
    }

    fn weight_slices(&self) -> Vec<WeightSlice<'_>> {
        self.loaded_matrices()
            .into_iter()
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
        let (gate_up, down, gate, up);
        let weights = match &self.experts {
            ExpertMatrices::Fused {
                gate_up: g,
                down: d,
            } => {
                gate_up = slices(g);
                down = slices(d);
                ExpertSlices::Fused {
                    gate_up: &gate_up,
                    down: &down,
                    layout: op.gate_up_layout.ok_or_else(|| {
                        VindexError::Parse(
                            "routed FFN op carries a packed bank and no gate_up layout; closure \
                             requires one"
                                .to_string(),
                        )
                    })?,
                }
            }
            ExpertMatrices::Separate {
                gate: g,
                up: u,
                down: d,
            } => {
                gate = slices(g);
                up = slices(u);
                down = slices(d);
                ExpertSlices::Separate {
                    gate: &gate,
                    up: &up,
                    down: &down,
                }
            }
        };
        let mut routed = backend.routed_ffn(RoutedFfnCall {
            x,
            hidden,
            intermediate: op.expert_intermediate_size,
            experts: op.experts,
            top_k: op.top_k,
            router_kind: op.router_kind,
            routing_policy: op.routing_policy,
            activation: op.activation,
            gate_policy: op.gate_policy,
            router: &self.router,
            router_bias: self.router_bias.as_deref(),
            weights,
            gate_up_bias: self.gate_up_bias.as_deref(),
            down_bias: self.down_bias.as_deref(),
            router_input: (!std::ptr::eq(router_input, x)).then_some(router_input),
            router_scale: self.router_scale.as_deref(),
            router_per_expert_scale: self.router_per_expert_scale.as_deref(),
            router_norm_eps: self.router_norm_eps,
        })?;
        // `y = moe(x) + shared_experts(x)` — the always-active branch is a
        // dense FFN over the same input, summed unscaled: composed here,
        // once, for every backend. A gated branch is refused at selection.
        if let Some((ffn, dense)) = &self.shared {
            let shared = dense.apply(ffn, backend, x, hidden)?;
            for (acc, v) in routed.iter_mut().zip(&shared) {
                *acc += v;
            }
        }
        Ok(routed)
    }

    fn load(
        op: &RoutedFfnOp,
        store: OperandSource<'_>,
        format: WeightFormat,
        shared_format: super::prepared::FormatFor<'_>,
    ) -> Result<Self, VindexError> {
        let hidden = op.router.shape.get(1).copied().unwrap_or(0);
        let inter = op.expert_intermediate_size;
        // The bank first: its geometry is DECLARED — `k` follows from the
        // router's declared width — and a stray width refuses on the
        // declaration, before any operand's bytes are read.
        let (experts, gate_up_bias, down_bias) = match &op.bank {
            ExpertBank::Packed { gate_up, down } => (
                ExpertMatrices::Fused {
                    gate_up: load_packed(
                        store,
                        gate_up,
                        op,
                        FUSED_BRANCHES * inter,
                        hidden,
                        format,
                    )?,
                    down: load_packed(store, down, op, hidden, inter, format)?,
                },
                gate_up.bias.as_ref().map(|b| store.load(b)).transpose()?,
                down.bias.as_ref().map(|b| store.load(b)).transpose()?,
            ),
            // A per-expert bank is never read: each matrix is a region of
            // its object's one mapping, in the form the pin declares, and a
            // pin that is not a mapped form is the plan and the loader
            // disagreeing.
            ExpertBank::PerExpert { gate, up, down } => {
                let form = MappedForm::of(format).ok_or_else(|| {
                    VindexError::Parse(format!(
                        "a per-expert bank pinned to {format:?}: only a mapped stored form \
                         (bf16, f32) binds a bank; planned_operands() and the loader disagree"
                    ))
                })?;
                let map = |operands: &[OperandRef], rows: usize, k: usize| {
                    operands
                        .iter()
                        .map(|operand| {
                            let region = store
                                .store()
                                .map_region(operand, (rows * k * form.width()) as u64)?;
                            Ok(LoadedWeight::Mapped { region, form })
                        })
                        .collect::<Result<Vec<_>, VindexError>>()
                };
                (
                    ExpertMatrices::Separate {
                        gate: map(gate, inter, hidden)?,
                        up: map(up, inter, hidden)?,
                        down: map(down, hidden, inter)?,
                    },
                    None,
                    None,
                )
            }
        };
        let shared = match &op.shared {
            Some(shared) => {
                let ffn = FfnOp {
                    intermediate_size: shared.intermediate_size,
                    activation: shared.activation,
                    gate_policy: shared.gate_policy,
                    gate: Some(shared.gate.clone()),
                    up: shared.up.clone(),
                    down: shared.down.clone(),
                };
                let dense = DenseOperands::load(&ffn, store, shared_format)?;
                Some((ffn, dense))
            }
            None => None,
        };
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
            experts,
            gate_up_bias,
            down_bias,
            shared,
        })
    }
}

/// Load one packed projection as `experts` matrices of `[rows, k]` in
/// `format`: the bank bound once through its codec, each expert decoded
/// as a row range of it.
fn load_packed(
    store: OperandSource<'_>,
    projection: &PackedProjection,
    op: &RoutedFfnOp,
    rows: usize,
    k: usize,
    format: WeightFormat,
) -> Result<Vec<LoadedWeight>, VindexError> {
    let name = projection.weights.tensor.as_str();
    let facts = bank_facts(store, op, &projection.weights)?;
    let Some(registered) = facts.registered.as_ref() else {
        return Err(VindexError::Parse(format!(
            "`{name}`: `{}` names no registered codec, so the bank cannot be decoded",
            facts.label
        )));
    };
    // Declared geometry before bytes: a `k` the codec's row alignment
    // cannot tile is a fact about the op, and refusing it costs no read.
    if !registered.capabilities.admits_k(k) {
        return Err(VindexError::Parse(format!(
            "`{name}`: k={k} is not a whole number of `{}` row alignments of {} elements",
            facts.label, registered.capabilities.row_align_elems
        )));
    }
    // The same admission the selector made before this loader was
    // reached — one derivation, so a direct caller gets the same refusal
    // rather than a read followed by a decode-time one.
    facts.admit_row_slicing()?;
    let codec = store.registry().resolve(&facts.label, name)?;
    let raw = store.load_raw(&projection.weights)?;
    let scales = projection
        .scales
        .as_ref()
        .map(|scales_ref| store.load_raw(scales_ref))
        .transpose()?;
    // The whole bank is one region of `experts × rows` rows: the codec
    // validates every stream's length against that geometry, and a codec
    // that declares no scales stream refuses a partner it cannot consume.
    let bank_shape = [op.experts * rows, k];
    let operands = bind_region(
        codec,
        &bank_shape,
        &raw.bytes,
        scales.as_ref().map(|s| s.bytes.as_slice()),
        name,
    )?;
    (0..op.experts)
        .map(|e| {
            let expert_rows = e * rows..(e + 1) * rows;
            match format {
                // Native: the bank IS MXFP4 — the codec bound it as
                // MXFP4's two streams — so each expert's rows are one
                // contiguous slab of each stream, copied as they are.
                WeightFormat::Mxfp4 if facts.label == DTYPE_MXFP4 => {
                    let groups = k / MXFP4_GROUP_ELEMS;
                    let code_row = groups * MXFP4_GROUP_BYTES;
                    let codes = operands.stream_of_len(
                        VALUES,
                        expert_rows.end * code_row,
                        DTYPE_MXFP4,
                        name,
                    )?;
                    let scales = operands.stream_of_len(
                        GROUP_SCALES,
                        expert_rows.end * groups,
                        DTYPE_MXFP4,
                        name,
                    )?;
                    Ok(LoadedWeight::Mxfp4 {
                        packed: AlignedBytes::from_bytes(
                            &codes[expert_rows.start * code_row..expert_rows.end * code_row],
                        ),
                        scales: AlignedBytes::from_bytes(
                            &scales[expert_rows.start * groups..expert_rows.end * groups],
                        ),
                    })
                }
                // Everything else decodes through the codec and converts
                // exactly as a dense matrix would.
                other => {
                    let mut values = vec![0.0f32; rows * k];
                    codec.decode_rows(
                        &operands,
                        &bank_shape,
                        expert_rows,
                        RepresentationExtent::TERMINAL,
                        &mut values,
                        name,
                    )?;
                    from_f32(values, rows, k, other, name)
                }
            }
        })
        .collect()
}

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

/// What a packed bank IS: the same resolution the selector made before
/// this loader was reached — the container's label when it names a codec,
/// else the codec the plan's declared layout carries — so a direct caller
/// is refused exactly as the selector would have refused it.
fn bank_facts(
    store: OperandSource<'_>,
    op: &RoutedFfnOp,
    operand: &OperandRef,
) -> Result<RepresentationFacts, VindexError> {
    let registry = store.registry();
    let declared = declared_bank_representation(op.expert_format);
    match (store.store().stored_dtype(operand), declared) {
        (Some(stored), declared) => Ok(RepresentationFacts::resolve_declared(
            registry, stored, declared,
        )),
        (None, Some(declared)) => Ok(RepresentationFacts::resolve_in(registry, declared)),
        (None, None) => Err(VindexError::Parse(format!(
            "`{}`: the bank is neither stored under a label nor declared by the plan",
            operand.tensor
        ))),
    }
}
