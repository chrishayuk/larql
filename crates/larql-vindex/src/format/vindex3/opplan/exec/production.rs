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

use larql_models::config::{
    mrope_axis_table, Activation, AttentionSinkSpec, ExpertRoutingPolicy, GateActivation,
    GateCombine, GatePlacement, GateSource, GateUpBranch, MoeRouterKind, NormType, PositionPolicy,
    RotaryFrequencyBasis,
};
use ndarray::Array2;

use larql_compute::attention::softmax::{softmax_in_place, softmax_in_place_f32};
use larql_compute::cpu::ops::geglu::{geglu_silu_alloc, silu};
use larql_compute::cpu::ops::moe::math::matmul_vec;
use larql_compute::ffn::expert_weight::router;
use larql_compute::ffn::gelu_tanh;
use larql_compute::residual::{
    layer_norm_eps, rms_norm_eps, rms_norm_heads_no_weight_eps, rms_norm_qk_eps,
};
use larql_compute::MoeGateRule;
use larql_models::config::GateUpLayout;

use super::super::super::graph::policy::AttentionSpan;
use super::backend::{
    AttentionCall, AttentionOut, AttentionStepCall, AttentionStepOut, ExpertSlices, FfnCall,
    FfnManyCall, GateCall, NormCall, PlanBackend, ProjectCall, ProjectedQkv, QkNormCall,
    RoutedFfnCall, WeightFormat, WeightSlice,
};
use super::cpu::physical::{
    kquant_execution, project_matrix, project_matrix_many, ExecutorProjections, KQuantExecution,
};
use super::cpu::PhysicalProjectionPlan;
use super::kernels::{
    gather_fused_half, mrope_rotate_scaled, rope_rotate, rope_rotate_scaled, sigmoid, FusedHalf,
};
use super::prefetch;
use super::realization::{
    class_of, common_selection, cpu_projection_candidates, realization_residency, RealizationForm,
    RealizationId, RefusalKind, RepresentationFacts, Selection, SelectionReason, SelectionRefusal,
};
use super::routing_trace;
use super::stages::{stage, Stage};
use super::timing::{timed, OpClass};
use crate::format::vindex3::opplan::planned::PlannedOperand;
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

/// Name reported by [`PlanBackend::name`].
const NAME: &str = "production-larql-compute";

/// `larql-compute` realisation of every plan operation.
#[derive(Debug, Default, Clone, Copy)]
pub struct ProductionBackend;

impl ProductionBackend {
    pub fn new() -> Self {
        Self
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

    let _t = timed(OpClass::Rope);
    match call.position {
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

/// The output-projection bias, added after `w_o`.
pub(super) fn add_output_bias(call: &AttentionCall<'_>, out: &mut [f32]) {
    if let Some(bias) = &call.bias {
        add_bias_in_place(out, bias.o);
    }
}

/// `x[i] += b[i]`; a length mismatch is a geometry bug closure refuses,
/// so it panics rather than pads.
fn add_bias_in_place(x: &mut [f32], b: &[f32]) {
    assert_eq!(
        x.len(),
        b.len(),
        "bias length must equal the projection's rows"
    );
    for (x, b) in x.iter_mut().zip(b) {
        *x += b;
    }
}

/// Gate and up: the two branches sharing one fused operand.
pub(super) const FUSED_BRANCHES: usize = larql_models::quant::mxfp4::FUSED_HALVES;

/// The vector the router projects: `x` for every family but Gemma 4,
/// whose router reads the raw residual conditioned by a scale-less RMS
/// norm (served `rms_norm_no_weight`), the learned `router.scale` and
/// `hidden^-0.5` — the served `moe_router_input` arithmetic under HF's
/// input choice. Every conditioning operand must be present.
pub(super) fn router_input(call: &RoutedFfnCall<'_>) -> Result<Vec<f32>, VindexError> {
    if call.router_kind != MoeRouterKind::Gemma4Hybrid {
        return Ok(call.x.to_vec());
    }
    let missing = |what: &str| {
        VindexError::Parse(format!(
            "Gemma4Hybrid router without its {what}; the plan must carry it"
        ))
    };
    let router_scale = call.router_scale.ok_or_else(|| missing("router scale"))?;
    let eps = call
        .router_norm_eps
        .ok_or_else(|| missing("router norm eps"))?;
    let residual = call.router_input.unwrap_or(call.x);
    // The served scale-less RMS norm, the whole vector as one "head".
    let normed = rms_norm_heads_no_weight_eps(&as_row(residual), 1, call.hidden, eps);
    let mut conditioned: Vec<f32> = normed.iter().copied().collect();
    let root_hidden_inv = (call.hidden as f32).powf(-0.5);
    for (v, s) in conditioned.iter_mut().zip(router_scale) {
        *v *= s * root_hidden_inv;
    }
    Ok(conditioned)
}

/// Route one token through the served selection rule
/// (`larql-compute`'s `router::select`) over the router logits — shared
/// glue, so the production and device backends select identically and
/// exactly as the served path does. Gemma 4 selects with the served
/// renormalised-softmax rule and then applies its per-expert scale to the
/// selected weights (served `moe_route_from_router_input`'s
/// `RenormalizedSoftmax` + `PerExpert` arms).
pub(super) fn select_experts(
    call: &RoutedFfnCall<'_>,
    logits: &mut [f32],
) -> Result<Vec<(usize, f32)>, VindexError> {
    if call.router_kind == MoeRouterKind::Gemma4Hybrid {
        let per_expert = call.router_per_expert_scale.ok_or_else(|| {
            VindexError::Parse(
                "Gemma4Hybrid router without its per-expert scale; the plan must carry it"
                    .to_string(),
            )
        })?;
        let mut selected = router::select(
            logits,
            call.top_k,
            ExpertRoutingPolicy::NormalisedOverSelected,
        );
        for (e, w) in &mut selected {
            *w *= per_expert[*e];
        }
        return Ok(selected);
    }
    let mut selected = if call.router_kind == MoeRouterKind::Sigmoid {
        sigmoid_select(logits, call.router_bias, call.top_k, call.routing_policy)
    } else {
        if let Some(bias) = call.router_bias {
            for (l, b) in logits.iter_mut().zip(bias) {
                *l += b;
            }
        }
        router::select(logits, call.top_k, call.routing_policy)
    };
    if call.branch_scale != 1.0 {
        for (_, w) in &mut selected {
            *w *= call.branch_scale;
        }
    }
    Ok(selected)
}

/// The reference's renormalisation guard: `weights / (sum + 1e-20)`, so a
/// selection whose scores all underflow divides by something.
const SIGMOID_RENORM_EPS: f32 = 1e-20;

/// The sigmoid router (DeepSeek-V3, Kimi, GLM-5.3-Flash): every expert's
/// score is `sigmoid(logit)`, independent of the others; the correction
/// bias moves which experts are SELECTED and never what they WEIGH; the
/// selected raw scores are the weights, renormalised to sum to one under
/// [`ExpertRoutingPolicy::NormalisedOverSelected`] and kept raw otherwise.
/// Ties rank by first index, as `torch.topk` does.
pub(super) fn sigmoid_select(
    logits: &[f32],
    bias: Option<&[f32]>,
    top_k: usize,
    policy: ExpertRoutingPolicy,
) -> Vec<(usize, f32)> {
    let scores: Vec<f32> = logits.iter().map(|&l| sigmoid(l)).collect();
    let keys: Vec<f32> = match bias {
        Some(bias) => scores.iter().zip(bias).map(|(s, b)| s + b).collect(),
        None => scores.clone(),
    };
    let mut ranked: Vec<usize> = (0..logits.len()).collect();
    // A stable sort on the key keeps equal keys in index order.
    ranked.sort_by(|&a, &b| {
        keys[b]
            .partial_cmp(&keys[a])
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    ranked.truncate(top_k.min(logits.len()));
    let mut selected: Vec<(usize, f32)> = ranked.iter().map(|&e| (e, scores[e])).collect();
    if policy == ExpertRoutingPolicy::NormalisedOverSelected && selected.len() > 1 {
        let sum = selected.iter().map(|(_, w)| w).sum::<f32>() + SIGMOID_RENORM_EPS;
        for (_, w) in &mut selected {
            *w /= sum;
        }
    }
    selected
}

/// One selected expert's inner activation from its fused gate/up output
/// (bias already added): rows read through the declared layout, combined
/// by the served gate rule.
pub(super) fn expert_inner(
    call: &RoutedFfnCall<'_>,
    layout: GateUpLayout,
    fused: &[f32],
) -> Vec<f32> {
    let rule = MoeGateRule::from_arch(call.gate_policy, call.activation);
    (0..call.intermediate)
        .map(|i| {
            let g = fused[layout.row(GateUpBranch::Gate, i, call.intermediate)];
            let u = fused[layout.row(GateUpBranch::Up, i, call.intermediate)];
            rule.combine(g, u)
        })
        .collect()
}

/// `x[i] += bias[expert-th row]` for a per-expert bias stored flat.
pub(super) fn add_expert_bias(x: &mut [f32], bias: Option<&[f32]>, expert: usize) {
    if let Some(bias) = bias {
        let rows = x.len();
        add_bias_in_place(x, &bias[expert * rows..(expert + 1) * rows]);
    }
}

/// One query position's scores, softmax and weighted-V aggregation —
/// the production softmax kernel over whatever K/V storage the caller
/// abstracts through `key_of`/`value_of`. Shared by the production and
/// device backends (the device deliberately runs production glue so a
/// divergence is attributable to device matmul arithmetic alone); the
/// gate and output projections stay with each backend's own matmuls.
pub(super) fn aggregate_heads<'k>(
    call: &AttentionCall<'_>,
    position: usize,
    query: &[f32],
    key_of: impl Fn(usize) -> &'k [f32],
    value_of: impl Fn(usize) -> &'k [f32],
) -> Vec<f32> {
    let head_dim = call.head_dim;
    let q_rows = call.num_q_heads * head_dim;
    let group = call.num_q_heads / call.num_kv_heads;
    // Exhaustive over the span vocabulary on purpose: a `_` arm would let
    // the next span kind mean "whole prefix" without anyone deciding that,
    // which is the defect `layer_types` already suffered once.
    let start = match (call.span, call.window) {
        (AttentionSpan::Sliding, Some(window)) => (position + 1).saturating_sub(window),
        // A sliding layer with no declared window has no bound to apply.
        (AttentionSpan::Sliding, None) | (AttentionSpan::Full, _) => 0,
        // A spatial window's extent is not a position count, so no
        // sequence bound follows from it. No generic op lowers a
        // perception component today; when one does, it needs the
        // component's own geometry here rather than this fallthrough.
        (AttentionSpan::Windowed, _) => 0,
    };
    let _t = timed(OpClass::AttentionCore);
    let mut concat = vec![0.0f32; q_rows];
    for q_head in 0..call.num_q_heads {
        let kv_head = q_head / group;
        let q_slice = &query[q_head * head_dim..(q_head + 1) * head_dim];
        let mut scores: Vec<f32> = (start..=position)
            .map(|key_position| {
                let k_slice = &key_of(key_position)[kv_head * head_dim..(kv_head + 1) * head_dim];
                let dot: f32 = q_slice.iter().zip(k_slice).map(|(a, b)| a * b).sum();
                let scaled = dot * call.score_scale as f32;
                match call.logit_softcapping {
                    Some(cap) => cap * (scaled / cap).tanh(),
                    None => scaled,
                }
            })
            .collect();
        match &call.sinks {
            // The served path's own sink softmax; exhaustive on the judged
            // semantics so a new variant must be implemented before it
            // can execute here.
            Some(sinks) => {
                let AttentionSinkSpec::SoftmaxDenominator = sinks.spec;
                softmax_in_place(&mut scores, Some(sinks.logits[q_head]));
            }
            None => softmax_in_place_f32(&mut scores),
        }
        let head_out = &mut concat[q_head * head_dim..(q_head + 1) * head_dim];
        for (offset, key_position) in (start..=position).enumerate() {
            let v_slice = &value_of(key_position)[kv_head * head_dim..(kv_head + 1) * head_dim];
            let weight = scores[offset];
            for (acc, v) in head_out.iter_mut().zip(v_slice) {
                *acc += weight * v;
            }
        }
    }
    concat
}

/// One position's projections, plus the gate half when it came out of
/// the same product.
///
/// Private to this backend: `ProjectedQkv` is the shared seam type and
/// the reference backend deliberately still projects the gate a second
/// time — it is the literal transcription, and the oracle's value is that
/// it does the obvious thing. That the two agree to 4e-7 is what licenses
/// this sharing.
pub(super) struct ProjectedAttention {
    pub(super) qkv: ProjectedQkv,
    /// The gate's values when the plan said they are the other half of
    /// the query projection. `None` for a gate with its own operand, and
    /// for a layer with no gate at all.
    pub(super) gate: Option<Vec<f32>>,
}

impl ProductionBackend {
    /// One position's Q/K/V projections through the production matvec,
    /// conditioned by the shared glue.
    pub(super) fn project_position(
        call: &AttentionCall<'_>,
        position: usize,
        pre: &[f32],
    ) -> Result<ProjectedAttention, VindexError> {
        let head_dim = call.head_dim;
        let q_rows = call.num_q_heads * head_dim;
        let kv_rows = call.num_kv_heads * head_dim;
        // A fused query/gate projection is `2 · head_dim` per head with
        // the halves INTERLEAVED; the first `q_rows` rows are not the
        // queries. See `gather_fused_half`.
        let fused_gate = matches!(
            call.gate.as_ref().map(|g| g.spec.source),
            Some(GateSource::FusedQueryProjection)
        );
        // **Both halves come out of one product.**
        //
        // `FusedQueryProjection` says the gate IS the other half of this
        // projection, over this same activation. Projecting again to
        // collect it read Qwen3.8's `12288 x 5120` q_proj a second time
        // per layer — 2.01 GB/token, 3.8% of every token's traffic, for a
        // vector already computed and discarded.
        //
        // The gather is per HEAD, not a contiguous range: head 0's query
        // rows, then head 0's gate rows, then head 1's. Taking the first
        // or second `q_rows` would have the right shape and the wrong
        // tensor.
        let (mut q, gate) = if fused_gate {
            let full = project_matrix(&call.w_q, pre, q_rows * 2, call.hidden)?;
            (
                gather_fused_half(&full, call.num_q_heads, head_dim, FusedHalf::Query),
                Some(gather_fused_half(
                    &full,
                    call.num_q_heads,
                    head_dim,
                    FusedHalf::Gate,
                )),
            )
        } else {
            (project_matrix(&call.w_q, pre, q_rows, call.hidden)?, None)
        };
        let mut k = project_matrix(&call.w_k, pre, kv_rows, call.hidden)?;
        let mut v = project_matrix(&call.w_v, pre, kv_rows, call.hidden)?;
        add_projection_biases(call, &mut q, &mut k, &mut v);
        condition_v_in_place(call, &mut v);
        condition_qk_in_place(call, position, &mut q, &mut k)?;
        Ok(ProjectedAttention {
            qkv: (q, k, v),
            gate,
        })
    }

    /// Aggregation plus this backend's own gate and output matmuls.
    #[allow(clippy::too_many_arguments)]
    pub(super) fn attend_position<'k>(
        call: &AttentionCall<'_>,
        position: usize,
        query: &[f32],
        key_of: impl Fn(usize) -> &'k [f32],
        value_of: impl Fn(usize) -> &'k [f32],
        gate_input: &[f32],
        projected_gate: Option<&[f32]>,
    ) -> Result<Vec<f32>, VindexError> {
        let q_rows = call.num_q_heads * call.head_dim;
        let mut concat = aggregate_heads(call, position, query, key_of, value_of);

        if let Some(GateCall { spec, weight }) = &call.gate {
            // Exhaustive on the judged semantics, same as the
            // reference: a new variant must be implemented before it
            // can execute on this backend either.
            let GateActivation::Sigmoid = spec.activation;
            let GateCombine::ElementwiseMultiply = spec.combine;
            let GatePlacement::AfterAggregationBeforeOutputProjection = spec.placement;
            let gate_values = match (spec.source, projected_gate) {
                // Already computed: the projection that produced the
                // queries produced these in the same pass.
                (GateSource::FusedQueryProjection, Some(values)) => values.to_vec(),
                // A fused gate with nothing handed over — the batched
                // path before it threads one through, or a caller that
                // reached here another way. Correct, and reads the
                // operand a second time; the ledger shows it as an extra
                // call rather than hiding it.
                (GateSource::FusedQueryProjection, None) => {
                    let full = project_matrix(weight, gate_input, q_rows * 2, call.hidden)?;
                    gather_fused_half(&full, call.num_q_heads, call.head_dim, FusedHalf::Gate)
                }
                // Its own matrix over its own activation: nothing to
                // share, and sharing would be wrong.
                (GateSource::AttentionInput, _) => {
                    project_matrix(weight, gate_input, q_rows, call.hidden)?
                }
            };
            let _t = timed(OpClass::OutputGate);
            for (c, g) in concat.iter_mut().zip(&gate_values) {
                *c *= 1.0 / (1.0 + (-g).exp());
            }
        }

        let mut out = project_matrix(&call.w_o, &concat, call.hidden, q_rows)?;
        add_output_bias(call, &mut out);
        Ok(out)
    }
}

/// The CPU executor's own re-quantised resident forms, offered as
/// candidates for a float source it knows how to narrow — which a codec
/// declares by naming the direct bf16 kernel.
const REQUANTISE: [PhysicalProjectionPlan; 4] = [
    PhysicalProjectionPlan::FusedQ8,
    PhysicalProjectionPlan::Q8xQ8,
    PhysicalProjectionPlan::Q4xQ8,
    PhysicalProjectionPlan::FusedQ4,
];

/// [`ProductionBackend::select`]'s decision, with the K-quant execution
/// arm passed in rather than read from the environment — so both arms are
/// testable in one process without touching it.
///
/// **The policy, over a derived candidate set.** The candidates come from
/// the codec's declarations and the executor's own compact forms; the
/// ladder below only ORDERS them, and can answer nothing that is not a
/// candidate. A compiled NVFP4 pack outranks everything; a stored K-quant
/// runs in place or widens by the arm; a float source goes to the size
/// policy, which keeps a large bf16 image compact and widens a small one;
/// a codec with no direct realization decodes, and says so.
pub(crate) fn select_cpu(
    operand: &PlannedOperand,
    facts: &RepresentationFacts,
    kquant: KQuantExecution,
) -> Result<Selection, Box<SelectionRefusal>> {
    use RealizationForm::{Decode, Direct, Requantise};
    if let Some(common) = common_selection(operand, facts, WeightFormat::F32) {
        return common;
    }
    let refuse = |kind, considered| {
        Box::new(SelectionRefusal {
            operand: operand.operand.clone(),
            operation: operand.operation,
            representation: facts.label.clone(),
            requested: operand.access,
            kind,
            considered,
        })
    };
    let Some(class) = class_of(operand.operation) else {
        return Err(refuse(RefusalKind::MissingRealization, vec![]));
    };
    if facts.registered.is_none() {
        return Err(refuse(RefusalKind::UnregisteredRepresentation, vec![]));
    }
    let candidates = cpu_projection_candidates(facts, PhysicalProjectionPlan::BlasF32, &REQUANTISE);
    let has = |form: RealizationForm| candidates.iter().any(|c| c.form == form);
    let decode = RealizationId::cpu(Decode(PhysicalProjectionPlan::BlasF32));
    let pick = |id: RealizationId, reason: SelectionReason| {
        Ok(Selection {
            realization: id,
            residency: realization_residency(facts, id),
            reason,
            candidates: candidates.clone(),
        })
    };
    if has(Direct(PhysicalProjectionPlan::FusedNvfp4)) {
        return pick(
            RealizationId::cpu(Direct(PhysicalProjectionPlan::FusedNvfp4)),
            SelectionReason::DirectDeclared,
        );
    }
    if has(Direct(PhysicalProjectionPlan::FusedKQuant)) {
        return match kquant {
            KQuantExecution::Direct => pick(
                RealizationId::cpu(Direct(PhysicalProjectionPlan::FusedKQuant)),
                SelectionReason::DirectDeclared,
            ),
            KQuantExecution::Widen => pick(decode, SelectionReason::ArmPrefersDecode),
        };
    }
    // The size policy is asked whether a bf16 image is worth keeping
    // compact — and the fact it is asked about is the codec DECLARING the
    // direct bf16 kernel, not a dtype the loader compared.
    let bf16_kernel_declared = has(Direct(PhysicalProjectionPlan::FusedBf16));
    let plan = PhysicalProjectionPlan::choose_for(
        Some(class),
        operand.logical_elements,
        bf16_kernel_declared,
    );
    let form = if has(Direct(plan)) {
        Direct(plan)
    } else if plan.format() == WeightFormat::F32 {
        Decode(plan)
    } else {
        Requantise(plan)
    };
    if !has(form) {
        return Err(refuse(
            RefusalKind::MissingRealization,
            candidates
                .iter()
                .map(|c| {
                    (
                        *c,
                        "not the resident form the size policy chose".to_string(),
                    )
                })
                .collect(),
        ));
    }
    let reason = match form {
        _ if facts.overlaid => SelectionReason::OverlaidEdit,
        Direct(_) if !bf16_kernel_declared => SelectionReason::DirectDeclared,
        Decode(_) if !bf16_kernel_declared => SelectionReason::NoDirectRealization,
        _ => SelectionReason::SizePolicy,
    };
    pick(RealizationId::cpu(form), reason)
}

impl PlanBackend for ProductionBackend {
    fn dense_projector(&self) -> &dyn super::gated_delta::DenseProjections {
        &ExecutorProjections
    }

    /// **The policy.** One decision per matrix, producing the resident
    /// form here and the kernel at [`project_rows`] — see
    /// [`PhysicalProjectionPlan`] and [`select_cpu`].
    fn select(
        &self,
        operand: &PlannedOperand,
        facts: &RepresentationFacts,
    ) -> Result<Selection, Box<SelectionRefusal>> {
        select_cpu(operand, facts, kquant_execution())
    }

    fn name(&self) -> &str {
        NAME
    }

    fn embed(&self, table: &[f32], hidden: usize, token: u32, scale: Option<f32>) -> Vec<f32> {
        let _t = timed(OpClass::Embed);
        let row = &table[token as usize * hidden..(token as usize + 1) * hidden];
        match scale {
            Some(scale) => row.iter().map(|v| v * scale).collect(),
            None => row.to_vec(),
        }
    }

    fn norm(&self, call: NormCall<'_>) -> Vec<f32> {
        let _t = timed(OpClass::Norm);
        let weight = (!call.weight.is_empty()).then(|| call.weight.to_vec());
        let normed = match call.kind {
            NormType::RmsNorm => rms_norm_eps(
                &as_row(call.x),
                weight.as_ref(),
                call.weight_offset,
                call.eps,
            ),
            // The production layer-norm kernel takes a bias; the plan
            // carries none, so it is absent rather than zeroed.
            NormType::LayerNorm => layer_norm_eps(&as_row(call.x), weight.as_ref(), None, call.eps),
        };
        from_row(normed)
    }

    fn project(&self, call: ProjectCall<'_>) -> Result<Vec<f32>, VindexError> {
        project_matrix(&call.weight, call.x, call.out_dim, call.in_dim)
    }

    fn attention(&self, call: AttentionCall<'_>) -> Result<AttentionOut, VindexError> {
        // Positions are independent, so projection runs in parallel with
        // each position's arithmetic untouched — bit-identical to the
        // serial order.
        let projected: Vec<ProjectedAttention> = call
            .inputs
            .par_iter()
            .enumerate()
            .map(|(position, pre)| Self::project_position(&call, position, pre))
            .collect::<Result<_, VindexError>>()?;
        let mut queries = Vec::with_capacity(projected.len());
        let mut keys = Vec::with_capacity(projected.len());
        let mut values = Vec::with_capacity(projected.len());
        // The gate halves travel with their positions: the batched path
        // shares the projection exactly as the step path does, so the two
        // do not differ in how many times they read `w_q`.
        let mut gates: Vec<Option<Vec<f32>>> = Vec::with_capacity(projected.len());
        for ProjectedAttention {
            qkv: (q, k, v),
            gate,
        } in projected
        {
            queries.push(q);
            keys.push(k);
            values.push(v);
            gates.push(gate);
        }

        // Each query position reads every position's K/V but writes only
        // its own output row — parallel over queries, arithmetic intact.
        let outputs: Vec<Vec<f32>> = queries
            .par_iter()
            .enumerate()
            .map(|(position, query)| {
                Self::attend_position(
                    &call,
                    position,
                    query,
                    |p| keys[p].as_slice(),
                    |p| values[p].as_slice(),
                    &call.inputs[position],
                    gates[position].as_deref(),
                )
            })
            .collect::<Result<_, VindexError>>()?;
        Ok(AttentionOut {
            outputs,
            keys,
            values,
        })
    }

    fn attention_step(&self, step: AttentionStepCall<'_>) -> Result<AttentionStepOut, VindexError> {
        let call = &step.op;
        let pre = &call.inputs[0];
        let ProjectedAttention {
            qkv: (q, k, v),
            gate,
        } = Self::project_position(call, step.position, pre)?;
        let output = Self::attend_position(
            call,
            step.position,
            &q,
            |p| {
                if p == step.position {
                    k.as_slice()
                } else {
                    step.keys[p].as_slice()
                }
            },
            |p| {
                if p == step.position {
                    v.as_slice()
                } else {
                    step.values[p].as_slice()
                }
            },
            pre,
            gate.as_deref(),
        )?;
        Ok(AttentionStepOut {
            key: k,
            value: v,
            output,
        })
    }

    fn ffn(&self, call: FfnCall<'_>) -> Result<Vec<f32>, VindexError> {
        require_executable_gate("production", call.gate_policy)?;
        let up = project_matrix(&call.up, call.x, call.intermediate, call.hidden)?;
        let gate = match call.gate {
            Some(w) => Some(project_matrix(&w, call.x, call.intermediate, call.hidden)?),
            None => None,
        };
        let inner = ffn_activation(gate.as_deref(), &up, call.activation, call.gate_policy)?;
        project_matrix(&call.down, &inner, call.hidden, call.intermediate)
    }

    /// **CPU-7C2.** The dense FFN over several positions, with each
    /// projection taken as ONE weight traversal.
    ///
    /// The activation stays per position — it is elementwise, it is small
    /// against the projections, and grouping it would be a change to the
    /// arithmetic rather than to the schedule.
    ///
    /// Note what is NOT here: no `par_iter` over positions. Rows own the
    /// machine and positions live inside the row traversal. The previous
    /// shape ran positions in parallel and each of them re-entered the
    /// executor, where `caller_owns_the_machine` collapsed every
    /// projection to a single worker — CPU-7C1 measured that as
    /// `slabs/call` 5.03 -> 2.81 and a 42% loss against serial decode.
    fn ffn_many(&self, call: FfnManyCall<'_>) -> Result<Vec<Vec<f32>>, VindexError> {
        require_executable_gate("production", call.gate_policy)?;
        let ups = project_matrix_many(&call.up, call.xs, call.intermediate, call.hidden)?;
        let gates = match &call.gate {
            Some(w) => Some(project_matrix_many(
                w,
                call.xs,
                call.intermediate,
                call.hidden,
            )?),
            None => None,
        };
        let inners: Vec<Vec<f32>> = (0..call.xs.len())
            .map(|p| {
                ffn_activation(
                    gates.as_ref().map(|g| g[p].as_slice()),
                    &ups[p],
                    call.activation,
                    call.gate_policy,
                )
            })
            .collect::<Result<_, _>>()?;
        let refs: Vec<&[f32]> = inners.iter().map(Vec::as_slice).collect();
        project_matrix_many(&call.down, &refs, call.hidden, call.intermediate)
    }

    fn routed_ffn(&self, call: RoutedFfnCall<'_>) -> Result<Vec<f32>, VindexError> {
        let selected = {
            let _stage = stage(Stage::Router);
            let routed_input = router_input(&call)?;
            let mut logits = matmul_vec(&routed_input, call.router, call.experts, call.hidden);
            select_experts(&call, &mut logits)?
        };
        routing_trace::record(&selected);
        if let ExpertSlices::Separate {
            gate,
            up,
            down,
            access,
        } = &call.weights
        {
            // The selected experts' pages, ahead of the loop that reads
            // them — the access realization, timed apart from the loop
            // so a fault moved is a fault moved, not a fault removed.
            let _prefetch = stage(Stage::Prefetch);
            let ranges: Vec<prefetch::Range> = selected
                .iter()
                .flat_map(|(e, _)| [&gate[*e], &up[*e], &down[*e]])
                .filter_map(|w| match w {
                    WeightSlice::Bf16(rows) => Some(prefetch::Range::of(rows)),
                    WeightSlice::F32(rows) => Some(prefetch::Range::of(rows)),
                    _ => None,
                })
                .collect();
            let parallelism = super::cpu::shared().map(|e| e.workers()).unwrap_or(1);
            prefetch::prefetch(*access, &ranges, parallelism);
        }
        let _stage = stage(Stage::RoutedExperts);
        let mut out = vec![0.0f32; call.hidden];
        match call.weights {
            ExpertSlices::Fused {
                gate_up,
                down,
                layout,
            } => {
                let two_inter = FUSED_BRANCHES * call.intermediate;
                for (expert, weight) in selected {
                    let mut fused =
                        matmul_vec(call.x, gate_up[expert].as_f32()?, two_inter, call.hidden);
                    add_expert_bias(&mut fused, call.gate_up_bias, expert);
                    let inner = expert_inner(&call, layout, &fused);
                    let mut expert_out = matmul_vec(
                        &inner,
                        down[expert].as_f32()?,
                        call.hidden,
                        call.intermediate,
                    );
                    add_expert_bias(&mut expert_out, call.down_bias, expert);
                    for (acc, v) in out.iter_mut().zip(&expert_out) {
                        *acc += weight * v;
                    }
                }
            }
            // A per-expert bank: each selected expert's three whole
            // matrices run through the SAME production projection
            // kernels a dense FFN uses — bf16 in place, f32 through BLAS
            // — so the bank stays in its stored form. No bias layout is
            // defined for separate experts, and none is planned; one
            // arriving here is a plan the executor does not know.
            ExpertSlices::Separate { gate, up, down, .. } => {
                if call.gate_up_bias.is_some() || call.down_bias.is_some() {
                    return Err(VindexError::Parse(
                        "a per-expert bank carries no expert bias; the call declares one"
                            .to_string(),
                    ));
                }
                let rule = MoeGateRule::from_arch(call.gate_policy, call.activation);
                for (expert, weight) in selected {
                    let g = project_matrix(&gate[expert], call.x, call.intermediate, call.hidden)?;
                    let u = project_matrix(&up[expert], call.x, call.intermediate, call.hidden)?;
                    let inner: Vec<f32> = g
                        .iter()
                        .zip(&u)
                        .map(|(g, u)| rule.combine(*g, *u))
                        .collect();
                    let expert_out =
                        project_matrix(&down[expert], &inner, call.hidden, call.intermediate)?;
                    for (acc, v) in out.iter_mut().zip(&expert_out) {
                        *acc += weight * v;
                    }
                }
            }
        }
        Ok(out)
    }

    fn output_head(
        &self,
        projection: super::backend::WeightSlice<'_>,
        vocab: usize,
        hidden: usize,
        x: &[f32],
        multiplier: Option<f64>,
        softcapping: Option<f32>,
    ) -> Result<Vec<f32>, VindexError> {
        let mut logits = project_matrix(&projection, x, vocab, hidden)?;
        // The vocabulary pass only — 248320 elements of multiplier and
        // softcap, which is real work and nothing to do with the matmul
        // that produced them.
        let _t = timed(OpClass::Logits);
        for logit in &mut logits {
            if let Some(multiplier) = multiplier {
                *logit *= multiplier as f32;
            }
            if let Some(cap) = softcapping {
                *logit = cap * (*logit / cap).tanh();
            }
        }
        Ok(logits)
    }

    fn residual_add(&self, acc: &mut [f32], delta: &[f32]) {
        let _t = timed(OpClass::Residual);
        for (a, b) in acc.iter_mut().zip(delta) {
            *a += b;
        }
    }
}
