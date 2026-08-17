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
    Activation, AttentionSinkSpec, GateActivation, GateCombine, GatePlacement, GateSource,
    GateUpBranch, MoeRouterKind, NormType, PositionPolicy,
};
use ndarray::Array2;

use larql_compute::attention::softmax::{softmax_in_place, softmax_in_place_f32};
use larql_compute::cpu::ops::geglu::{geglu_silu_alloc, silu};
use larql_compute::cpu::ops::moe::math::matmul_vec;
use larql_compute::ffn::expert_weight::router;
use larql_compute::residual::{
    layer_norm_eps, rms_norm_eps, rms_norm_heads_no_weight_eps, rms_norm_qk_eps,
};
use larql_compute::MoeGateRule;

use super::super::super::graph::policy::AttentionSpan;
use super::backend::{
    AttentionCall, AttentionStepCall, AttentionStepOut, FfnCall, GateCall, NormCall, PlanBackend,
    ProjectCall, ProjectedQkv, QkNormCall, RoutedFfnCall,
};
use super::kernels::{rope_rotate, rope_rotate_scaled};
use larql_compute::attention::rope::{rope_freq_plan, RopeFreqScaling};

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
pub(super) fn require_plain_gate(
    backend: &str,
    policy: larql_models::ExpertGatePolicy,
) -> Result<(), VindexError> {
    match policy {
        larql_models::ExpertGatePolicy::Gated => Ok(()),
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
        PositionPolicy::None => {}
    }
    Ok(())
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

/// Route one token through the served selection rule
/// (`larql-compute`'s `router::select`) over the router logits — shared
/// glue, so the production and device backends select identically and
/// exactly as the served path does. Exhaustive on the router kind: Gemma
/// 4's per-expert scale has its own arithmetic and must be implemented
/// before it can execute here.
pub(super) fn select_experts(
    call: &RoutedFfnCall<'_>,
    logits: &mut [f32],
) -> Result<Vec<(usize, f32)>, VindexError> {
    match call.router_kind {
        MoeRouterKind::TopKSoftmax | MoeRouterKind::TopKThenSoftmax => {}
        MoeRouterKind::Gemma4Hybrid => {
            return Err(VindexError::Parse(
                "MoeRouterKind::Gemma4Hybrid carries a per-expert scale this backend does not \
                 execute; refusing rather than routing with the plain rule"
                    .to_string(),
            ))
        }
    }
    if let Some(bias) = call.router_bias {
        for (l, b) in logits.iter_mut().zip(bias) {
            *l += b;
        }
    }
    Ok(router::select(logits, call.top_k, call.routing_policy))
}

/// One selected expert's inner activation from its fused gate/up output
/// (bias already added): rows read through the declared layout, combined
/// by the served gate rule.
pub(super) fn expert_inner(call: &RoutedFfnCall<'_>, fused: &[f32]) -> Vec<f32> {
    let rule = MoeGateRule::from_arch(call.gate_policy, call.activation);
    (0..call.intermediate)
        .map(|i| {
            let g = fused[call
                .gate_up_layout
                .row(GateUpBranch::Gate, i, call.intermediate)];
            let u = fused[call
                .gate_up_layout
                .row(GateUpBranch::Up, i, call.intermediate)];
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

impl ProductionBackend {
    /// One position's Q/K/V projections through the production matvec,
    /// conditioned by the shared glue.
    fn project_position(
        call: &AttentionCall<'_>,
        position: usize,
        pre: &[f32],
    ) -> Result<ProjectedQkv, VindexError> {
        let head_dim = call.head_dim;
        let q_rows = call.num_q_heads * head_dim;
        let kv_rows = call.num_kv_heads * head_dim;
        let mut q = matmul_vec(pre, call.w_q.as_f32()?, q_rows, call.hidden);
        let mut k = matmul_vec(pre, call.w_k.as_f32()?, kv_rows, call.hidden);
        let mut v = matmul_vec(pre, call.w_v.as_f32()?, kv_rows, call.hidden);
        add_projection_biases(call, &mut q, &mut k, &mut v);
        condition_qk_in_place(call, position, &mut q, &mut k)?;
        Ok((q, k, v))
    }

    /// Aggregation plus this backend's own gate and output matmuls.
    fn attend_position<'k>(
        call: &AttentionCall<'_>,
        position: usize,
        query: &[f32],
        key_of: impl Fn(usize) -> &'k [f32],
        value_of: impl Fn(usize) -> &'k [f32],
        gate_input: &[f32],
    ) -> Result<Vec<f32>, VindexError> {
        let q_rows = call.num_q_heads * call.head_dim;
        let mut concat = aggregate_heads(call, position, query, key_of, value_of);

        if let Some(GateCall { spec, weight }) = &call.gate {
            // Exhaustive on the judged semantics, same as the
            // reference: a new variant must be implemented before it
            // can execute on this backend either.
            let GateSource::AttentionInput = spec.source;
            let GateActivation::Sigmoid = spec.activation;
            let GateCombine::ElementwiseMultiply = spec.combine;
            let GatePlacement::AfterAggregationBeforeOutputProjection = spec.placement;
            let gate_values = matmul_vec(gate_input, weight.as_f32()?, q_rows, call.hidden);
            for (c, g) in concat.iter_mut().zip(&gate_values) {
                *c *= 1.0 / (1.0 + (-g).exp());
            }
        }

        let mut out = matmul_vec(&concat, call.w_o.as_f32()?, call.hidden, q_rows);
        add_output_bias(call, &mut out);
        Ok(out)
    }
}

impl PlanBackend for ProductionBackend {
    fn name(&self) -> &str {
        NAME
    }

    fn embed(&self, table: &[f32], hidden: usize, token: u32, scale: Option<f32>) -> Vec<f32> {
        let row = &table[token as usize * hidden..(token as usize + 1) * hidden];
        match scale {
            Some(scale) => row.iter().map(|v| v * scale).collect(),
            None => row.to_vec(),
        }
    }

    fn norm(&self, call: NormCall<'_>) -> Vec<f32> {
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
        Ok(matmul_vec(
            call.x,
            call.weight.as_f32()?,
            call.out_dim,
            call.in_dim,
        ))
    }

    fn attention(&self, call: AttentionCall<'_>) -> Result<Vec<Vec<f32>>, VindexError> {
        // Positions are independent, so projection runs in parallel with
        // each position's arithmetic untouched — bit-identical to the
        // serial order.
        let projected: Vec<(Vec<f32>, Vec<f32>, Vec<f32>)> = call
            .inputs
            .par_iter()
            .enumerate()
            .map(|(position, pre)| Self::project_position(&call, position, pre))
            .collect::<Result<_, VindexError>>()?;
        let mut queries = Vec::with_capacity(projected.len());
        let mut keys = Vec::with_capacity(projected.len());
        let mut values = Vec::with_capacity(projected.len());
        for (q, k, v) in projected {
            queries.push(q);
            keys.push(k);
            values.push(v);
        }

        // Each query position reads every position's K/V but writes only
        // its own output row — parallel over queries, arithmetic intact.
        queries
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
                )
            })
            .collect()
    }

    fn attention_step(&self, step: AttentionStepCall<'_>) -> Result<AttentionStepOut, VindexError> {
        let call = &step.op;
        let pre = &call.inputs[0];
        let (q, k, v) = Self::project_position(call, step.position, pre)?;
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
        )?;
        Ok(AttentionStepOut {
            key: k,
            value: v,
            output,
        })
    }

    fn ffn(&self, call: FfnCall<'_>) -> Result<Vec<f32>, VindexError> {
        require_plain_gate("production", call.gate_policy)?;
        let up = matmul_vec(call.x, call.up.as_f32()?, call.intermediate, call.hidden);
        let inner: Vec<f32> = match call.gate {
            Some(gate_weight) => {
                let gate = matmul_vec(
                    call.x,
                    gate_weight.as_f32()?,
                    call.intermediate,
                    call.hidden,
                );
                match call.activation {
                    Activation::Silu => geglu_silu_alloc(&gate, &up),
                    other => return Err(unsupported_activation("gated", other)),
                }
            }
            None => match call.activation {
                Activation::Silu => up.iter().map(|u| silu(*u)).collect(),
                other => return Err(unsupported_activation("ungated", other)),
            },
        };
        Ok(matmul_vec(
            &inner,
            call.down.as_f32()?,
            call.hidden,
            call.intermediate,
        ))
    }

    fn routed_ffn(&self, call: RoutedFfnCall<'_>) -> Result<Vec<f32>, VindexError> {
        let mut logits = matmul_vec(call.x, call.router, call.experts, call.hidden);
        let selected = select_experts(&call, &mut logits)?;
        let two_inter = FUSED_BRANCHES * call.intermediate;
        let mut out = vec![0.0f32; call.hidden];
        for (expert, weight) in selected {
            let mut fused = matmul_vec(
                call.x,
                call.gate_up[expert].as_f32()?,
                two_inter,
                call.hidden,
            );
            add_expert_bias(&mut fused, call.gate_up_bias, expert);
            let inner = expert_inner(&call, &fused);
            let mut expert_out = matmul_vec(
                &inner,
                call.down[expert].as_f32()?,
                call.hidden,
                call.intermediate,
            );
            add_expert_bias(&mut expert_out, call.down_bias, expert);
            for (acc, v) in out.iter_mut().zip(&expert_out) {
                *acc += weight * v;
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
        let mut logits = matmul_vec(x, projection.as_f32()?, vocab, hidden);
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
        for (a, b) in acc.iter_mut().zip(delta) {
            *a += b;
        }
    }
}
