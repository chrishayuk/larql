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

use super::backend::{NormCall, WeightSlice};
use super::operands::OperandSource;
use super::weights::LoadedWeight;
use crate::error::VindexError;
use crate::format::vindex3::opplan::{FfnOp, NormOp};

/// The routed-aggregate norm's semantics, transcribed from the class the
/// reference constructs (`KimiRMSNorm`) rather than carried on the op:
/// they are facts of that norm class, not of the checkpoint. RMS — no
/// mean subtraction — with nothing added to the learned scale.
///
/// The EPSILON is the opposite kind of fact and is carried on the op, for
/// the reason the rung exists: two neighbouring norms in this same family
/// run at a class default ten times away, so the value is the
/// checkpoint's to state and only these two are the class's.
const LATENT_NORM_KIND: larql_models::config::NormType = larql_models::config::NormType::RmsNorm;
const NO_WEIGHT_OFFSET: f32 = 0.0;

mod dense;
mod loading;
mod routed;
pub(super) use loading::*;

/// Gate and up: the two branches sharing one fused operand.
const FUSED_BRANCHES: usize = larql_models::quant::mxfp4::FUSED_HALVES;

/// **Enter the routed bottleneck.** `[width, hidden] · x -> [width]`.
///
/// Through the BACKEND, not through CPU glue. The projection is a whole
/// dense matrix per token, so a device arm that ran it here on the host
/// would be a silent CPU stage inside a run reported as executing on the
/// device — the wrapper is the one part of this operator that would be
/// easiest to leave behind, because everything downstream of it still
/// closes on shape.
///
/// Separated from [`RoutedOperands::apply`] so the oracle parity test
/// drives the arithmetic execution actually runs, rather than a second
/// copy of it written in the test — a self-consistent pair proves the
/// two agree, never that either is right.
pub(super) fn enter_latent<B: super::backend::PlanBackend + ?Sized>(
    backend: &B,
    down: WeightSlice<'_>,
    x: &[f32],
    width: usize,
    hidden: usize,
) -> Result<Vec<f32>, VindexError> {
    backend.project(super::backend::ProjectCall {
        weight: down,
        out_dim: width,
        in_dim: hidden,
        x,
    })
}

/// **Leave it.** Optional RMSNorm on the WEIGHTED AGGREGATE, then
/// `[hidden, width] · v -> [hidden]`.
///
/// The norm sits here, between summation and expansion, and its epsilon
/// arrives with its weight rather than being read from the layer. Both
/// are the rung's substance: normalising per expert, before the
/// weighting, or after the up-projection are three different models, and
/// this is the operator's one placement fact that no shape can catch.
///
/// Both steps go through the backend for the reason [`enter_latent`]
/// does.
pub(super) fn exit_latent<B: super::backend::PlanBackend + ?Sized>(
    backend: &B,
    aggregate: Vec<f32>,
    norm: Option<(&[f32], f64)>,
    up: WeightSlice<'_>,
    hidden: usize,
    width: usize,
) -> Result<Vec<f32>, VindexError> {
    let normed = match norm {
        Some((weight, eps)) => backend.norm(NormCall {
            kind: LATENT_NORM_KIND,
            x: &aggregate,
            weight,
            bias: &[],
            weight_offset: NO_WEIGHT_OFFSET,
            eps,
        }),
        None => aggregate,
    };
    backend.project(super::backend::ProjectCall {
        weight: up,
        out_dim: hidden,
        in_dim: width,
        x: &normed,
    })
}

/// A layer's FFN operands, loaded once in the backend's declared format.
pub(super) enum FfnOperands {
    External {
        layer: usize,
        provider: Option<std::sync::Arc<dyn super::dense_ffn::DenseFfnProvider>>,
    },
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
    /// The declared up/down biases, f32 (elementwise glue, not matrices).
    up_bias: Option<Vec<f32>>,
    down_bias: Option<Vec<f32>>,
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
            bias: &[],
            weight_offset: self.op.weight_offset,
            eps: self.op.eps,
        })
    }
}

/// A routed layer's operands: router (f32 glue), the experts' matrices in
/// the shape their bank stores them, the shared branch when the plan
/// carries one, plus Gemma 4's router conditioning when the op carries it.
pub(super) struct RoutedOperands {
    placement: Option<(
        usize,
        Option<std::sync::Arc<dyn super::routed_experts::RoutedExpertProvider>>,
    )>,
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
    /// The shared branch's scalar output gate and its `[1, hidden]`
    /// weight, when the family runs one (Qwen MoE:
    /// `sigmoid(shared_expert_gate(x)) * shared(x)`). Loaded as the f32
    /// row its realization pins and bound like any planned operand, so the
    /// ledger that reconciles preparation names it. `None` sums the branch
    /// unscaled (DeepSeek / Kimi).
    shared_gate: Option<(larql_models::config::SharedExpertGateSpec, LoadedWeight)>,
    /// The latent bottleneck's two projections and its optional norm
    /// weight, when the plan carries the wrapper.
    latent: Option<LatentOperands>,
}

/// The latent routed branch's loaded operands.
pub(super) struct LatentOperands {
    /// `[width, hidden]`, in the backend's declared format — the same
    /// `LoadedWeight` a dense projection binds, because that is what
    /// these are. Loading them as f32 glue instead would have been the
    /// quiet way to make the wrapper unrepresentable on a device.
    down: LoadedWeight,
    /// `[hidden, width]`.
    up: LoadedWeight,
    /// The RMSNorm weight `[width]` and its epsilon, together — the
    /// epsilon is the norm's own, not the layer's, and separating them
    /// is how the wrong one gets used. f32 glue like every other norm
    /// weight in this file.
    norm: Option<(Vec<f32>, f64)>,
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
        /// How the selected experts' pages are brought in per token.
        access: super::realization::MappedAccess,
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
            Self::Separate { gate, up, down, .. } => gate.iter().chain(up).chain(down).collect(),
        }
    }
}

#[cfg(test)]
mod tests;
