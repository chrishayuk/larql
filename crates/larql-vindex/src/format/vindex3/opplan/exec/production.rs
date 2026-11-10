//! The production backend: the same plan, realised by `larql-compute`.
//!
//! Deliberately boring. Every method maps to the most direct existing
//! production kernel that preserves the resolved operation — no fusion,
//! no special-casing, no optimisation. The only claim being made at this
//! rung is that one semantic IR drives two numerical implementations;
//! speed comes later, under two fixed correctness anchors.
//!
//! **It binds the real kernels, not lookalikes.** `matmul_vec` is public
//! precisely so a VINDEX3 backend can call *that* function rather than
//! reimplement a similar loop — binding the real one is the difference
//! between proving kernel binding works and proving two similar loops
//! agree.
//!
//! **It fails closed.** Where `larql-compute` has no kernel for a judged
//! variant, this returns an error naming what is missing. Falling back to
//! the reference's arithmetic would make the two backends agree by
//! sharing code, which is exactly the agreement that proves nothing.

use larql_models::config::{mrope_axis_table, Activation, PositionPolicy, RotaryFrequencyBasis};
use ndarray::Array2;

use larql_compute::cpu::ops::geglu::{geglu_silu_alloc, silu};
use larql_compute::ffn::gelu_tanh;
use larql_compute::residual::{rms_norm_heads_no_weight_eps, rms_norm_qk_eps};

use super::backend::{AttentionCall, Nvfp4Activation, QkNormCall};
use super::cpu::physical::KQuantExecution;
use super::kernels::{mrope_rotate_scaled, rope_rotate, rope_rotate_scaled};
use super::timing::{timed, OpClass};
use larql_compute::attention::rope::{
    rope_freq_plan, rope_freq_plan_proportional, RopeFreqScaling,
};

/// The whole head rotates: `PositionPolicy` carries no partial-rotary
/// fraction (no family through this path declares one).
const FULL_ROTARY: f64 = 1.0;
/// No position divisor (`rope_freq_plan` treats 0 as 1).
const NO_POSITION_DIVISOR: f64 = 1.0;
use crate::error::VindexError;
use rayon::prelude::*;

mod helpers;
mod plan_backend;
mod select;
pub(super) use helpers::*;
pub(crate) use select::*;

/// Name reported by [`PlanBackend::name`].
const NAME: &str = "production-larql-compute";
/// The provider's family ([`PlanBackend::identity`]): the CPU executor
/// over `larql-compute`'s kernels. Revision 1 is the arithmetic this
/// module binds today; it moves when the same pin would compute a
/// different number, never for a faster kernel computing the same one.
pub const IDENTITY_FAMILY: &str = "cpu-production";
pub const IDENTITY_REVISION: u32 = 1;

/// The Q8_K-activation provider's name and family (Q8K-ACT-1). A distinct
/// identity, because the same pin computes different numbers under it:
/// a prepared image records which provider qualified its pins, and an
/// image prepared for one must not execute under the other.
const Q8K_NAME: &str = "production-larql-compute-q8k";
pub const Q8K_IDENTITY_FAMILY: &str = "cpu-production-q8k";
pub const Q8K_IDENTITY_REVISION: u32 = 1;

/// The NVFP4 x Q8 provider's name and family (NVFP4-Q8-1), distinct for
/// Q8K-ACT-1's reason: the same NVFP4 pin computes different numbers
/// against a Q8 activation, so an image prepared under one provider must
/// never execute under the other.
const NVFP4_Q8_NAME: &str = "production-larql-compute-nvfp4-q8";
pub const NVFP4_Q8_IDENTITY_FAMILY: &str = "cpu-production-nvfp4-q8";
pub const NVFP4_Q8_IDENTITY_REVISION: u32 = 1;

/// `larql-compute` realisation of every plan operation.
#[derive(Debug, Default, Clone, Copy)]
pub struct ProductionBackend {
    /// How stored K-quants execute. `None` is the shipped provider: the
    /// process arm ([`kquant_execution`], `direct` unless widened). `Some`
    /// is a provider constructed for one arm, under its own identity.
    kquant: Option<KQuantExecution>,
    /// Which activation a stored NVFP4 pack runs against. `F32` is the
    /// shipped provider; `Q8` is NVFP4-Q8-1's, under its own identity.
    nvfp4: Nvfp4Activation,
}

impl ProductionBackend {
    pub fn new() -> Self {
        Self {
            kquant: None,
            nvfp4: Nvfp4Activation::F32,
        }
    }

    /// The Q8_K-activation provider (Q8K-ACT-1): stored Q4_K / Q6_K
    /// blocks run against an activation quantised to Q8_K. Everything
    /// else is this backend's ordinary arithmetic.
    pub fn q8k_activation() -> Self {
        Self {
            kquant: Some(KQuantExecution::DirectQ8k),
            nvfp4: Nvfp4Activation::F32,
        }
    }

    /// The NVFP4 x Q8 provider (NVFP4-Q8-1): stored NVFP4 packs run
    /// against an activation quantised to Q8, one scale per group.
    /// Everything else is this backend's ordinary arithmetic.
    pub fn nvfp4_q8_activation() -> Self {
        Self {
            kquant: None,
            nvfp4: Nvfp4Activation::Q8,
        }
    }

    fn is_q8k(&self) -> bool {
        self.kquant == Some(KQuantExecution::DirectQ8k)
    }

    fn is_nvfp4_q8(&self) -> bool {
        self.nvfp4 == Nvfp4Activation::Q8
    }
}

/// The FFN's elementwise middle, shared by the one-position and
/// many-position arms so the two cannot drift into different arithmetic.
///
/// The activation ONLY. The three projections around it are timed by the
/// executor, and a timer that spanned them would make this class the
/// whole FFN.
fn ffn_activation(
    gate: Option<&[f32]>,
    up: &[f32],
    activation: Activation,
    policy: larql_models::ExpertGatePolicy,
) -> Result<Vec<f32>, VindexError> {
    let _t = timed(OpClass::FfnActivation);
    // A gate POLICY that is not plain gating owns the whole combine, and
    // the nonlinearity beside it is inert. Handled before the activation
    // match so the two facts cannot be applied at once.
    if let larql_models::ExpertGatePolicy::SituGlu { beta, linear_beta } = policy {
        let Some(gate) = gate else {
            return Err(VindexError::Parse(
                "SiTU-GLU is a gated combine and this FFN has no gate projection; refusing \
                 rather than computing it on the up branch alone"
                    .to_string(),
            ));
        };
        let rule = larql_compute::MoeGateRule::SituGlu { beta, linear_beta };
        return Ok(gate
            .iter()
            .zip(up)
            .map(|(g, u)| rule.combine(*g, *u))
            .collect());
    }
    match gate {
        Some(gate) => match activation {
            Activation::Silu => Ok(geglu_silu_alloc(gate, up)),
            // The served Gemma gate/up kernel (tanh-approximated GELU on
            // the gate, times up).
            Activation::GeluTanh => Ok(gate
                .iter()
                .zip(up)
                .map(|(g, u)| gelu_tanh(*g) * u)
                .collect()),
            other => Err(unsupported_activation("gated", other)),
        },
        None => match activation {
            Activation::Silu => Ok(up.iter().map(|u| silu(*u)).collect()),
            Activation::GeluTanh => Ok(up.iter().map(|u| gelu_tanh(*u)).collect()),
            other => Err(unsupported_activation("ungated", other)),
        },
    }
}

/// Refuse an activation `larql-compute` has no kernel for.
///
/// Naming what is missing, rather than silently reusing the reference's
/// scalar loop: two backends that share arithmetic agree by construction,
/// and that agreement is exactly what this rung must not manufacture.
pub(super) fn unsupported_activation(shape: &str, activation: Activation) -> VindexError {
    VindexError::Parse(format!(
        "no production {shape}-FFN kernel for activation {activation:?} — refusing rather \
         than borrowing the reference backend's arithmetic"
    ))
}

/// The gate policy every backend here honours today. A `ClampedGlu` plan
/// (GPT-OSS's `swiglu_limit`) is carried by the container and refused
/// until A-9.3 executes it — computing `activation(gate) * up` for it
/// would run a different model without saying so.
pub(super) fn require_executable_gate(
    backend: &str,
    policy: larql_models::ExpertGatePolicy,
) -> Result<(), VindexError> {
    match policy {
        larql_models::ExpertGatePolicy::Gated => Ok(()),
        // K3-ACT-1: both CPU-glue backends compute SiTU elementwise
        // through `MoeGateRule::combine` — the same authority the routed
        // path already uses — so admitting it here is a statement about
        // what they execute, not a relaxation of what they check.
        larql_models::ExpertGatePolicy::SituGlu { .. } => Ok(()),
        larql_models::ExpertGatePolicy::ClampedGlu { limit, alpha } => {
            Err(VindexError::Parse(format!(
                "the {backend} backend does not execute ExpertGatePolicy::ClampedGlu {{ limit: \
             {limit}, alpha: {alpha} }} yet (A-9.3); refusing rather than applying plain \
             gating to a clamped-GLU FFN"
            )))
        }
        larql_models::ExpertGatePolicy::ClampedGated { limit } => Err(VindexError::Parse(format!(
            "the {backend} backend does not execute ExpertGatePolicy::ClampedGated {{ limit: \
             {limit} }} yet; refusing rather than applying plain gating to a CLAMPED FFN, \
             whose clamp is one-sided on the gate and symmetric on the up branch"
        ))),
    }
}

/// Wrap one vector as a `[1, n]` matrix for the row-wise norm kernels.
pub(super) fn as_row(x: &[f32]) -> Array2<f32> {
    Array2::from_shape_vec((1, x.len()), x.to_vec()).expect("row shape matches length")
}

/// Take the single row back out.
pub(super) fn from_row(m: Array2<f32>) -> Vec<f32> {
    m.into_raw_vec_and_offset().0
}

/// Apply Q/K normalisation to one projection in place.
///
/// Head geometry is passed to the kernel rather than sliced here, so the
/// production path exercises the production reduction over `head_dim`.
pub(super) fn qk_norm_in_place(
    values: &mut [f32],
    weight: Option<(&[f32], f32)>,
    parameter_free: bool,
    num_heads: usize,
    head_dim: usize,
    scope: larql_models::config::QkNormScope,
    eps: f64,
) {
    if let Some((w, offset)) = weight {
        let normed = rms_norm_qk_eps(&as_row(values), w, num_heads, head_dim, offset, scope, eps);
        values.copy_from_slice(&from_row(normed));
    }
    if parameter_free {
        let normed = rms_norm_heads_no_weight_eps(&as_row(values), num_heads, head_dim, eps);
        values.copy_from_slice(&from_row(normed));
    }
}

/// Q/K normalisation, query scale and position encoding for one
/// position's already-projected Q/K, in the judged order — the CPU glue
/// applied identically by the production and device backends after
/// their own projection arithmetic.
pub(super) fn condition_qk_in_place(
    call: &AttentionCall<'_>,
    position: usize,
    q: &mut [f32],
    k: &mut [f32],
) -> Result<(), VindexError> {
    condition_qk_in_place_keeping(call, position, q, k, None)
}

/// [`condition_qk_in_place`], copying Q and K into `keep` immediately
/// before position encoding when asked (PREC-DIAG-1 E0a). The copy reads
/// the buffers; the rotation then runs on them exactly as it does with
/// `keep = None`.
pub(super) fn condition_qk_in_place_keeping(
    call: &AttentionCall<'_>,
    position: usize,
    q: &mut [f32],
    k: &mut [f32],
    keep: Option<&mut super::observe::UnpositionedQk>,
) -> Result<(), VindexError> {
    let head_dim = call.head_dim;
    let qk_weight = call.qk_norm.as_ref().map(
        |QkNormCall {
             weight_offset,
             q_weight,
             k_weight,
             scope,
         }| (*scope, *weight_offset, *q_weight, *k_weight),
    );
    let (scope, offset, q_w, k_w) = match qk_weight {
        Some((scope, offset, q_w, k_w)) => (scope, offset, Some(q_w), Some(k_w)),
        None => (larql_models::config::QkNormScope::PerHead, 0.0, None, None),
    };
    // Two leaves, not one: QK normalisation and position encoding are
    // different operations that happen to be adjacent, and "conditioning
    // cost 9 ms" would not say which to look at.
    let norm = timed(OpClass::Norm);
    qk_norm_in_place(
        q,
        q_w.map(|w| (w, offset)),
        call.parameter_free_qk_norm.q,
        call.num_q_heads,
        head_dim,
        scope,
        call.qk_norm_eps,
    );
    qk_norm_in_place(
        k,
        k_w.map(|w| (w, offset)),
        call.parameter_free_qk_norm.k,
        call.num_kv_heads,
        head_dim,
        scope,
        call.qk_norm_eps,
    );

    if let Some(query_scale) = call.query_scale {
        for value in q.iter_mut() {
            *value *= query_scale as f32;
        }
    }
    drop(norm);
    if let Some(keep) = keep {
        keep.keep(q, k);
    }

    let _t = timed(OpClass::Rope);
    // PREC-DIAG-1 E0b: under `RopeAngles::HfF32` Q and K are rotated here
    // and the match below sees `None`; by default it sees the plan's
    // policy and runs exactly as before.
    let policy = super::rope_angles::rotate_or_defer(
        call.rope_angles,
        call.position,
        head_dim,
        position,
        q,
        k,
    )?;
    match policy {
        PositionPolicy::Rope { theta } => {
            for head in q.chunks_exact_mut(head_dim) {
                rope_rotate(head, position, theta);
            }
            for head in k.chunks_exact_mut(head_dim) {
                rope_rotate(head, position, theta);
            }
        }
        // YaRN through the served rope planner: the same ramp and
        // amplitude the production forward applies (full rotary width, no
        // position divisor — what `PositionPolicy::Yarn` carries).
        PositionPolicy::Yarn { theta, scaling } => {
            let plan = rope_freq_plan(
                head_dim,
                FULL_ROTARY,
                theta,
                NO_POSITION_DIVISOR,
                RopeFreqScaling::Yarn(scaling),
            );
            let amplitude = plan.amplitude as f32;
            for head in q.chunks_exact_mut(head_dim) {
                rope_rotate_scaled(head, position, &plan.inv_freq, amplitude);
            }
            for head in k.chunks_exact_mut(head_dim) {
                rope_rotate_scaled(head, position, &plan.inv_freq, amplitude);
            }
        }
        // Linear through the served rope planner: the position divisor
        // the planner has always taken, at full rotary width, unscaled
        // frequencies, unit amplitude. The one arm that passes a divisor
        // other than `NO_POSITION_DIVISOR` — Gemma 3's global layers.
        PositionPolicy::Linear { theta, factor } => {
            let plan = rope_freq_plan(head_dim, FULL_ROTARY, theta, factor, RopeFreqScaling::None);
            let amplitude = plan.amplitude as f32;
            for head in q.chunks_exact_mut(head_dim) {
                rope_rotate_scaled(head, position, &plan.inv_freq, amplitude);
            }
            for head in k.chunks_exact_mut(head_dim) {
                rope_rotate_scaled(head, position, &plan.inv_freq, amplitude);
            }
        }
        // Llama-3 through the same served rope planner: wavelength-band
        // frequencies at full rotary width, unit amplitude. The planner
        // has implemented this since before the container could express
        // it — the gap this arm closes was carriage, not mathematics.
        PositionPolicy::Llama3 { theta, scaling } => {
            let plan = rope_freq_plan(
                head_dim,
                FULL_ROTARY,
                theta,
                NO_POSITION_DIVISOR,
                RopeFreqScaling::Llama3(scaling),
            );
            let amplitude = plan.amplitude as f32;
            for head in q.chunks_exact_mut(head_dim) {
                rope_rotate_scaled(head, position, &plan.inv_freq, amplitude);
            }
            for head in k.chunks_exact_mut(head_dim) {
                rope_rotate_scaled(head, position, &plan.inv_freq, amplitude);
            }
        }
        // Declared, and no backend rotates for it. Refusing is the only
        // honest arm: doing nothing would run the model with no position
        // information at all, which is a wrong answer that produces
        // plausible text. The plan blocks such a stack, so this is
        // unreachable through the supported path.
        PositionPolicy::Relative { d_rel, extent } => {
            return Err(VindexError::Parse(format!(
                "relative position (d_rel {d_rel}, extent {extent}) is represented but not \
                 executable: no backend implements it"
            )))
        }
        PositionPolicy::None => {}
        // Partial rotary through the served planners: the proportional
        // (head-width) plan is head-sized with zero pairs above the
        // fraction, applied over the whole head; the plain (rotary-width)
        // plan is prefix-sized and applied to the prefix as its own block.
        PositionPolicy::PartialRope {
            theta,
            rotary_fraction,
            basis,
        } => match basis {
            RotaryFrequencyBasis::HeadWidth => {
                let plan = rope_freq_plan_proportional(head_dim, rotary_fraction, theta);
                for head in q.chunks_exact_mut(head_dim) {
                    rope_rotate_scaled(head, position, &plan.inv_freq, plan.amplitude as f32);
                }
                for head in k.chunks_exact_mut(head_dim) {
                    rope_rotate_scaled(head, position, &plan.inv_freq, plan.amplitude as f32);
                }
            }
            RotaryFrequencyBasis::RotaryWidth => {
                let plan = rope_freq_plan(
                    head_dim,
                    rotary_fraction,
                    theta,
                    NO_POSITION_DIVISOR,
                    RopeFreqScaling::None,
                );
                let width = plan.inv_freq.len() * 2;
                for head in q.chunks_exact_mut(head_dim) {
                    rope_rotate_scaled(&mut head[..width], position, &plan.inv_freq, 1.0);
                }
                for head in k.chunks_exact_mut(head_dim) {
                    rope_rotate_scaled(&mut head[..width], position, &plan.inv_freq, 1.0);
                }
            }
        },
        // Multi-axis rotary on the served frequency plan. The prefix
        // block and its frequencies are exactly the plain partial
        // rotary's; only which position each slot reads differs, and on
        // the interpreter's scalar position the grid is `(p, p, p)`.
        PositionPolicy::MRope {
            theta,
            rotary_fraction,
            basis,
            section,
            interleaved,
        } => match basis {
            RotaryFrequencyBasis::RotaryWidth => {
                let plan = rope_freq_plan(
                    head_dim,
                    rotary_fraction,
                    theta,
                    NO_POSITION_DIVISOR,
                    RopeFreqScaling::None,
                );
                let width = plan.inv_freq.len() * 2;
                let axes = mrope_axis_table(section, interleaved, plan.inv_freq.len());
                let grid = [position, position, position];
                for head in q.chunks_exact_mut(head_dim) {
                    mrope_rotate_scaled(&mut head[..width], grid, &axes, &plan.inv_freq, 1.0);
                }
                for head in k.chunks_exact_mut(head_dim) {
                    mrope_rotate_scaled(&mut head[..width], grid, &axes, &plan.inv_freq, 1.0);
                }
            }
            RotaryFrequencyBasis::HeadWidth => {
                return Err(VindexError::Parse(
                    "M-RoPE with a head-width frequency basis is unjudged; no checkpoint \
                     declares it and the section-to-dimension mapping is undefined"
                        .to_string(),
                ))
            }
        },
    }
    Ok(())
}

/// The parameter-free V norm (Gemma 4 `v_norm`) on one position's raw
/// value projection, per head, through the served kernel — shared glue
/// for the production and device backends, applied right after the
/// projection biases and before V is cached.
pub(super) fn condition_v_in_place(call: &AttentionCall<'_>, v: &mut [f32]) {
    let _t = timed(OpClass::Norm);
    if call.parameter_free_qk_norm.v {
        let normed = rms_norm_heads_no_weight_eps(
            &as_row(v),
            call.num_kv_heads,
            call.head_dim,
            call.qk_norm_eps,
        );
        v.copy_from_slice(&from_row(normed));
    }
}

/// The Q/K/V projection biases, added right after projection — before
/// [`condition_qk_in_place`] reads Q/K and before V is cached. Shared
/// glue, so the production and device backends place them identically.
pub(super) fn add_projection_biases(
    call: &AttentionCall<'_>,
    q: &mut [f32],
    k: &mut [f32],
    v: &mut [f32],
) {
    if let Some(bias) = &call.bias {
        add_bias_in_place(q, bias.q);
        add_bias_in_place(k, bias.k);
        add_bias_in_place(v, bias.v);
    }
}
