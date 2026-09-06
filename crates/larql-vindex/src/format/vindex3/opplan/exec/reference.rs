//! The reference backend: naive f32, the semantic anchor.
//!
//! Shares **nothing** with `larql-compute`'s kernels. That is the whole
//! point of it — a reference that called the production kernels would
//! agree with them by construction, and the agreement would prove
//! nothing. Plain loops, row-major `[out, in]` weights, no BLAS, no
//! SIMD, no fusion.
//!
//! When the production backend disagrees with this one, this one is
//! right about *meaning* and may well be wrong about speed. Divergence
//! is a bug in the production backend or a hole in the seam, never a
//! licence to change what the plan means.

use larql_models::config::{
    AttentionSinkSpec, ExpertRoutingPolicy, GateActivation, GateCombine, GatePlacement, GateSource,
    GateUpBranch, MoeRouterKind, QkNormScope,
};

use super::super::super::graph::policy::AttentionSpan;
use super::backend::{
    AttentionCall, AttentionOut, AttentionStepCall, AttentionStepOut, ExpertSlices, FfnCall,
    GateCall, NormCall, PlanBackend, ProjectCall, ProjectedQkv, QkNormCall, RoutedFfnCall,
    WeightSlice,
};
use super::kernels::{
    activate, gather_fused_half_mutated, llama3_frequencies, matvec, mrope_rotate, norm,
    partial_rotary_frequencies, partial_rotary_slice, rope_rotate, rope_rotate_scaled, sigmoid,
    softcap, softmax, softmax_with_sink, yarn_frequencies, FusedHalf, GateMutation,
};
use crate::error::VindexError;
use larql_models::config::NormType;
use larql_models::config::{PositionPolicy, RotaryFrequencyBasis};
use rayon::prelude::*;

/// Name reported by [`PlanBackend::name`].
const NAME: &str = "reference-f32";

/// Naive f32 realisation of every plan operation.
#[derive(Debug, Default, Clone, Copy)]
pub struct ReferenceBackend;

impl ReferenceBackend {
    pub fn new() -> Self {
        Self
    }

    /// Q/K normalisation: weighted per-head when the plan binds weights,
    /// parameter-free when the surface judged it. Both may apply.
    fn apply_qk_norm(
        call: &AttentionCall<'_>,
        q: &mut [f32],
        k: &mut [f32],
    ) -> Result<(), VindexError> {
        let head_dim = call.head_dim;
        let eps = call.qk_norm_eps;
        if let Some(QkNormCall {
            scope,
            weight_offset,
            q_weight,
            k_weight,
        }) = &call.qk_norm
        {
            match scope {
                QkNormScope::PerHead => {
                    for head in q.chunks_exact_mut(head_dim) {
                        let normed = norm(NormType::RmsNorm, head, q_weight, *weight_offset, eps);
                        head.copy_from_slice(&normed);
                    }
                    for head in k.chunks_exact_mut(head_dim) {
                        let normed = norm(NormType::RmsNorm, head, k_weight, *weight_offset, eps);
                        head.copy_from_slice(&normed);
                    }
                }
                QkNormScope::FullProjection => {
                    return Err(VindexError::Parse(
                        "full-projection QK norm has no judged reference execution yet".to_string(),
                    ));
                }
            }
        }
        if call.parameter_free_qk_norm.q {
            for head in q.chunks_exact_mut(head_dim) {
                let normed = norm(NormType::RmsNorm, head, &[], 0.0, eps);
                head.copy_from_slice(&normed);
            }
        }
        if call.parameter_free_qk_norm.k {
            for head in k.chunks_exact_mut(head_dim) {
                let normed = norm(NormType::RmsNorm, head, &[], 0.0, eps);
                head.copy_from_slice(&normed);
            }
        }
        Ok(())
    }

    /// One position's Q/K/V projections with QK normalisation, query
    /// scale and position encoding applied in the judged order — the
    /// arithmetic both the batch path and the decode step share, so the
    /// two cannot disagree about a single position.
    /// Whether this layer's gate is sourced from the query projection —
    /// the one case where `w_q` is wider than the attention width.
    fn fused_query_gate(call: &AttentionCall<'_>) -> bool {
        matches!(
            call.gate.as_ref().map(|g| g.spec.source),
            Some(GateSource::FusedQueryProjection)
        )
    }

    fn project_position(
        call: &AttentionCall<'_>,
        position: usize,
        pre: &[f32],
    ) -> Result<ProjectedQkv, VindexError> {
        Self::project_position_inner(call, position, pre, GateMutation::None)
    }

    /// [`Self::project_position`] with a deliberate defect in the fused
    /// query/gate split. The public path above always passes
    /// [`GateMutation::None`]; the mutation table drives THIS function,
    /// so the table exercises the shipped implementation.
    pub(super) fn project_position_inner(
        call: &AttentionCall<'_>,
        position: usize,
        pre: &[f32],
        gate_mutation: GateMutation,
    ) -> Result<ProjectedQkv, VindexError> {
        let head_dim = call.head_dim;
        let q_rows = call.num_q_heads * head_dim;
        let kv_rows = call.num_kv_heads * head_dim;
        // A fused query projection carries `2 · head_dim` rows per head,
        // query and gate INTERLEAVED. Taking the first `q_rows` would
        // read the query of head 0, the gate of head 0, the query of
        // head 1 … and call the result "the queries" — right shape,
        // wrong tensor. The halves are gathered per head instead.
        let mut q = if Self::fused_query_gate(call) {
            let full = matvec(call.w_q.as_f32()?, q_rows * 2, call.hidden, pre);
            gather_fused_half_mutated(
                &full,
                call.num_q_heads,
                head_dim,
                FusedHalf::Query,
                gate_mutation,
            )
        } else {
            matvec(call.w_q.as_f32()?, q_rows, call.hidden, pre)
        };
        let mut k = matvec(call.w_k.as_f32()?, kv_rows, call.hidden, pre);
        let mut v = matvec(call.w_v.as_f32()?, kv_rows, call.hidden, pre);
        // Biases belong to the projections: added before anything reads
        // the projected values (QK-norm, rope, the cache).
        if let Some(bias) = &call.bias {
            add_in_place(&mut q, bias.q);
            add_in_place(&mut k, bias.k);
            add_in_place(&mut v, bias.v);
        }

        Self::apply_qk_norm(call, &mut q, &mut k)?;
        // The parameter-free V norm (Gemma 4 `v_norm`): per head, no
        // weight, the same epsilon — applied to the raw value projection
        // (on a K≡V layer that is the raw K projection, before its norm
        // and rotation, which is why V is taken before either).
        if call.parameter_free_qk_norm.v {
            for head in v.chunks_exact_mut(head_dim) {
                let normed = norm(NormType::RmsNorm, head, &[], 0.0, call.qk_norm_eps);
                head.copy_from_slice(&normed);
            }
        }
        if let Some(query_scale) = call.query_scale {
            for value in &mut q {
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
            // YaRN: the ramped frequency blend AND the amplitude on
            // cos/sin, from the reference transcription of the block the
            // container carries.
            PositionPolicy::Yarn { theta, scaling } => {
                let (inv_freq, amplitude) = yarn_frequencies(&scaling, head_dim, theta);
                for head in q.chunks_exact_mut(head_dim) {
                    rope_rotate_scaled(head, position, &inv_freq, amplitude);
                }
                for head in k.chunks_exact_mut(head_dim) {
                    rope_rotate_scaled(head, position, &inv_freq, amplitude);
                }
            }
            // Llama-3, transcribed: wavelength-band frequencies at unit
            // amplitude. `UNIT_AMPLITUDE` is passed explicitly rather
            // than defaulted, so an amplitude arriving here later has to
            // be written down rather than inherited.
            PositionPolicy::Llama3 { theta, scaling } => {
                const UNIT_AMPLITUDE: f32 = 1.0;
                let inv_freq = llama3_frequencies(&scaling, head_dim, theta);
                for head in q.chunks_exact_mut(head_dim) {
                    rope_rotate_scaled(head, position, &inv_freq, UNIT_AMPLITUDE);
                }
                for head in k.chunks_exact_mut(head_dim) {
                    rope_rotate_scaled(head, position, &inv_freq, UNIT_AMPLITUDE);
                }
            }
            // See `production.rs`: no backend rotates for a relative
            // scheme, and silently skipping position is a wrong answer
            // that still produces fluent output.
            PositionPolicy::Relative { d_rel, extent } => {
                return Err(VindexError::Parse(format!(
                    "relative position (d_rel {d_rel}, extent {extent}) is represented but not \
                     executable: no backend implements it"
                )))
            }
            PositionPolicy::None => {}
            // Partial rotary, transcribed: head-width basis is the full
            // rotate-half table with the top frequencies zero; rotary-width
            // basis rotates the prefix as its own block.
            PositionPolicy::PartialRope {
                theta,
                rotary_fraction,
                basis,
            } => match basis {
                RotaryFrequencyBasis::HeadWidth => {
                    let inv_freq = partial_rotary_frequencies(head_dim, rotary_fraction, theta);
                    for head in q.chunks_exact_mut(head_dim) {
                        rope_rotate_scaled(head, position, &inv_freq, 1.0);
                    }
                    for head in k.chunks_exact_mut(head_dim) {
                        rope_rotate_scaled(head, position, &inv_freq, 1.0);
                    }
                }
                RotaryFrequencyBasis::RotaryWidth => {
                    let width = partial_rotary_slice(head_dim, rotary_fraction);
                    for head in q.chunks_exact_mut(head_dim) {
                        rope_rotate(&mut head[..width], position, theta);
                    }
                    for head in k.chunks_exact_mut(head_dim) {
                        rope_rotate(&mut head[..width], position, theta);
                    }
                }
            },
            // Multi-axis rotary. The interpreter holds one scalar
            // position, so the grid is `(p, p, p)` — a text sequence,
            // where the axis assignment provably selects equal values.
            // The assignment still runs: see `mrope_rotate`.
            PositionPolicy::MRope {
                theta,
                rotary_fraction,
                basis,
                section,
                interleaved,
            } => {
                let width =
                    match basis {
                        RotaryFrequencyBasis::RotaryWidth => {
                            partial_rotary_slice(head_dim, rotary_fraction)
                        }
                        // No judged checkpoint pairs a head-width basis with
                        // M-RoPE; refusing beats guessing which block the
                        // sections index.
                        RotaryFrequencyBasis::HeadWidth => return Err(VindexError::Parse(
                            "M-RoPE with a head-width frequency basis is unjudged; no checkpoint \
                             declares it and the section-to-dimension mapping is undefined"
                                .to_string(),
                        )),
                    };
                let grid = [position, position, position];
                for head in q.chunks_exact_mut(head_dim) {
                    mrope_rotate(&mut head[..width], grid, theta, section, interleaved);
                }
                for head in k.chunks_exact_mut(head_dim) {
                    mrope_rotate(&mut head[..width], grid, theta, section, interleaved);
                }
            }
        }
        Ok((q, k, v))
    }

    /// One query position's scores, softmax, weighted-V aggregation,
    /// gate, and output projection. `key_of`/`value_of` abstract where
    /// K/V rows live (the batch path's local vectors, or the decode
    /// step's interpreter-owned cache plus the fresh row).
    #[allow(clippy::too_many_arguments)]
    fn attend_position<'k>(
        call: &AttentionCall<'_>,
        position: usize,
        query: &[f32],
        key_of: impl Fn(usize) -> &'k [f32],
        value_of: impl Fn(usize) -> &'k [f32],
        gate_input: &[f32],
    ) -> Result<Vec<f32>, VindexError> {
        Self::attend_position_inner(
            call,
            position,
            query,
            key_of,
            value_of,
            gate_input,
            GateMutation::None,
        )
    }

    /// [`Self::attend_position`] with a deliberate defect in the gate
    /// stage. See [`Self::project_position_inner`].
    #[allow(clippy::too_many_arguments)]
    pub(super) fn attend_position_inner<'k>(
        call: &AttentionCall<'_>,
        position: usize,
        query: &[f32],
        key_of: impl Fn(usize) -> &'k [f32],
        value_of: impl Fn(usize) -> &'k [f32],
        gate_input: &[f32],
        gate_mutation: GateMutation,
    ) -> Result<Vec<f32>, VindexError> {
        let head_dim = call.head_dim;
        let q_rows = call.num_q_heads * head_dim;
        let group = call.num_q_heads / call.num_kv_heads;
        // Span: which key positions this query may attend to. Exhaustive
        // over the vocabulary so a new span kind forces a decision here
        // instead of silently meaning "whole prefix".
        let start = match (call.span, call.window) {
            (AttentionSpan::Sliding, Some(window)) => (position + 1).saturating_sub(window),
            (AttentionSpan::Sliding, None) | (AttentionSpan::Full, _) => 0,
            // A spatial window bounds a region, not a position range; no
            // generic op lowers a perception component today.
            (AttentionSpan::Windowed, _) => 0,
        };
        let mut concat = vec![0.0f32; q_rows];
        for q_head in 0..call.num_q_heads {
            let kv_head = q_head / group;
            let q_slice = &query[q_head * head_dim..(q_head + 1) * head_dim];
            let mut scores: Vec<f32> = (start..=position)
                .map(|key_position| {
                    let k_slice =
                        &key_of(key_position)[kv_head * head_dim..(kv_head + 1) * head_dim];
                    let mut dot = 0.0f32;
                    for (a, b) in q_slice.iter().zip(k_slice) {
                        dot += a * b;
                    }
                    let mut score = dot * call.score_scale as f32;
                    if let Some(cap) = call.logit_softcapping {
                        score = softcap(score, cap);
                    }
                    score
                })
                .collect();
            match &call.sinks {
                // Exhaustive on the judged semantics: a new variant must
                // be implemented here before it can execute.
                Some(sinks) => {
                    let AttentionSinkSpec::SoftmaxDenominator = sinks.spec;
                    softmax_with_sink(&mut scores, sinks.logits[q_head]);
                }
                None => softmax(&mut scores),
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

        if let Some(GateCall { spec, weight }) = &call.gate {
            // Exhaustive on the judged semantics: a new variant must
            // be implemented here before it can execute.
            let GateActivation::Sigmoid = spec.activation;
            let GateCombine::ElementwiseMultiply = spec.combine;
            let GatePlacement::AfterAggregationBeforeOutputProjection = spec.placement;
            let gate_values = match spec.source {
                GateSource::AttentionInput => {
                    matvec(weight.as_f32()?, q_rows, call.hidden, gate_input)
                }
                // The gate rows live inside the query projection, so the
                // "gate weight" IS that projection and the gate is its
                // per-head second half. Recomputed from the same input
                // the query half was projected from — the reference
                // backend pays a second matvec rather than threading a
                // value through the call, which keeps this readable
                // beside the HF source it transcribes.
                GateSource::FusedQueryProjection => {
                    let full = matvec(weight.as_f32()?, q_rows * 2, call.hidden, gate_input);
                    let mut gate = gather_fused_half_mutated(
                        &full,
                        call.num_q_heads,
                        head_dim,
                        FusedHalf::Gate,
                        gate_mutation,
                    );
                    // The gate slice sees NEITHER the query norm nor the
                    // rotary — it is not a query. Both are mutations here
                    // rather than absent code, so a refactor that starts
                    // feeding the gate through either is caught by a
                    // number instead of by review.
                    if gate_mutation == GateMutation::GateGetsQNorm {
                        if let Some(qk) = &call.qk_norm {
                            for head in gate.chunks_exact_mut(head_dim) {
                                let normed = norm(
                                    NormType::RmsNorm,
                                    head,
                                    qk.q_weight,
                                    qk.weight_offset,
                                    call.qk_norm_eps,
                                );
                                head.copy_from_slice(&normed);
                            }
                        }
                    }
                    if gate_mutation == GateMutation::GateGetsRoPe {
                        for head in gate.chunks_exact_mut(head_dim) {
                            if let Some(theta) = call.position.rope_theta() {
                                rope_rotate(head, position, theta);
                            }
                        }
                    }
                    gate
                }
            };
            let activate_gate = |g: f32| match gate_mutation {
                // `silu(g)` is what `output_gate_type: "swish"` would mean
                // if it owned this gate. HF computes `sigmoid(g)`.
                GateMutation::SiluGate => g * sigmoid(g),
                _ => sigmoid(g),
            };
            if gate_mutation != GateMutation::NoGate
                && gate_mutation != GateMutation::GateAfterOProj
            {
                for (c, g) in concat.iter_mut().zip(&gate_values) {
                    *c *= activate_gate(*g);
                }
            }
            if gate_mutation == GateMutation::GateAfterOProj {
                let mut out = matvec(call.w_o.as_f32()?, call.hidden, q_rows, &concat);
                for (o, g) in out.iter_mut().zip(&gate_values) {
                    *o *= activate_gate(*g);
                }
                if let Some(bias) = &call.bias {
                    add_in_place(&mut out, bias.o);
                }
                return Ok(out);
            }
        }

        let mut out = matvec(call.w_o.as_f32()?, call.hidden, q_rows, &concat);
        if let Some(bias) = &call.bias {
            add_in_place(&mut out, bias.o);
        }
        Ok(out)
    }
}

/// Gate and up: the two branches sharing one fused operand.
const FUSED_BRANCHES: usize = larql_models::quant::mxfp4::FUSED_HALVES;

/// The judged expert selection, in the literal form: rank the logits
/// (ties to the lower index, as `torch.topk`), keep `top_k`, and weight
/// them by a softmax whose denominator the routing policy chooses — every
/// expert (`SoftmaxThenSelect`) or the selected ones only
/// (`NormalisedOverSelected`; GPT-OSS's top-k-then-softmax is that same
/// number). Gemma 4 (`Gemma4Hybrid`) is its own rule, transcribed from
/// `Gemma4TextRouter.forward`: the router input is the raw residual,
/// RMS-normalised without a weight, times `scale`, times `hidden^-0.5`;
/// softmax over every expert; top-k; the selected weights renormalised to
/// sum to one; then each multiplied by `per_expert_scale[expert]`.
pub(super) fn select_experts_reference(
    call: &RoutedFfnCall<'_>,
) -> Result<Vec<(usize, f32)>, VindexError> {
    let mut selected = if call.router_kind == MoeRouterKind::Gemma4Hybrid {
        select_experts_gemma4_reference(call)?
    } else if call.router_kind == MoeRouterKind::Sigmoid {
        select_experts_sigmoid_reference(call)
    } else {
        select_experts_softmax_reference(call)
    };
    // `routed_scaling_factor`: the reference multiplies the routed sum;
    // multiplying each weight is the same sum.
    for (_, w) in &mut selected {
        *w *= call.branch_scale;
    }
    Ok(selected)
}

/// `weights / (weights.sum() + 1e-20)` — the reference's literal.
const RENORM_EPS: f32 = 1e-20;

/// DeepSeek-V3's `MoEGate` with `scoring_func = "sigmoid"`, literal: scores
/// are sigmoids of the logits; `topk` runs over `scores +
/// e_score_correction_bias`; the gathered weights are the UNCORRECTED
/// scores, renormalised when `norm_topk_prob`.
fn select_experts_sigmoid_reference(call: &RoutedFfnCall<'_>) -> Vec<(usize, f32)> {
    let logits = matvec(call.router, call.experts, call.hidden, call.x);
    let scores: Vec<f32> = logits.iter().map(|l| 1.0 / (1.0 + (-l).exp())).collect();
    let mut ranked: Vec<(usize, f32)> = scores
        .iter()
        .enumerate()
        .map(|(e, &s)| (e, s + call.router_bias.map_or(0.0, |b| b[e])))
        .collect();
    ranked.sort_by(|a, b| b.1.total_cmp(&a.1).then(a.0.cmp(&b.0)));
    let k = call.top_k.min(ranked.len());
    let mut selected: Vec<(usize, f32)> =
        ranked[..k].iter().map(|&(e, _)| (e, scores[e])).collect();
    if call.routing_policy == ExpertRoutingPolicy::NormalisedOverSelected && k > 1 {
        let denominator = selected.iter().map(|(_, w)| w).sum::<f32>() + RENORM_EPS;
        for (_, w) in &mut selected {
            *w /= denominator;
        }
    }
    selected
}

/// The softmax routers, literal (see [`select_experts_reference`]).
fn select_experts_softmax_reference(call: &RoutedFfnCall<'_>) -> Vec<(usize, f32)> {
    let mut logits = matvec(call.router, call.experts, call.hidden, call.x);
    if let Some(bias) = call.router_bias {
        for (l, b) in logits.iter_mut().zip(bias) {
            *l += b;
        }
    }
    let mut ranked: Vec<(usize, f32)> = logits.iter().copied().enumerate().collect();
    // Stable, so equal logits keep index order.
    ranked.sort_by(|a, b| b.1.total_cmp(&a.1));
    let k = call.top_k.min(ranked.len());
    let max = ranked.first().map_or(0.0, |r| r.1);
    let denominator: f32 = match call.routing_policy {
        ExpertRoutingPolicy::SoftmaxThenSelect => ranked.iter().map(|r| (r.1 - max).exp()).sum(),
        ExpertRoutingPolicy::NormalisedOverSelected => {
            ranked.iter().take(k).map(|r| (r.1 - max).exp()).sum()
        }
    };
    ranked
        .into_iter()
        .take(k)
        .map(|(e, l)| (e, (l - max).exp() / denominator))
        .collect()
}

/// Gemma 4's router, literal (see [`select_experts_reference`]). Every
/// conditioning operand must be present — the plan pairs them with the
/// kind, and a missing one here is a broken plan, refused.
fn select_experts_gemma4_reference(
    call: &RoutedFfnCall<'_>,
) -> Result<Vec<(usize, f32)>, VindexError> {
    let missing = |what: &str| {
        VindexError::Parse(format!(
            "Gemma4Hybrid router without its {what}; the plan must carry it"
        ))
    };
    let router_scale = call.router_scale.ok_or_else(|| missing("router scale"))?;
    let per_expert = call
        .router_per_expert_scale
        .ok_or_else(|| missing("per-expert scale"))?;
    let eps = call
        .router_norm_eps
        .ok_or_else(|| missing("router norm eps"))?;
    let residual = call.router_input.unwrap_or(call.x);
    // Scale-less RMS norm: x / sqrt(mean(x²) + eps), in f32 as HF does.
    let mean_sq = residual.iter().map(|v| v * v).sum::<f32>() / residual.len() as f32;
    let inv_rms = 1.0 / (mean_sq + eps as f32).sqrt();
    let root_hidden = (call.hidden as f32).sqrt();
    let conditioned: Vec<f32> = residual
        .iter()
        .zip(router_scale)
        .map(|(v, s)| v * inv_rms * s / root_hidden)
        .collect();
    let logits = matvec(call.router, call.experts, call.hidden, &conditioned);
    let mut ranked: Vec<(usize, f32)> = logits.iter().copied().enumerate().collect();
    ranked.sort_by(|a, b| b.1.total_cmp(&a.1));
    let k = call.top_k.min(ranked.len());
    let max = ranked.first().map_or(0.0, |r| r.1);
    let all: f32 = ranked.iter().map(|r| (r.1 - max).exp()).sum();
    let selected: Vec<(usize, f32)> = ranked
        .into_iter()
        .take(k)
        .map(|(e, l)| (e, (l - max).exp() / all))
        .collect();
    let selected_sum: f32 = selected.iter().map(|(_, w)| w).sum();
    Ok(selected
        .into_iter()
        .map(|(e, w)| (e, w / selected_sum * per_expert[e]))
        .collect())
}

/// One expert's gate/up combine under the judged policy, transcribed from
/// the family definitions: plain gating is `activation(gate) · up`;
/// GPT-OSS's clamped GLU clamps gate above and up both ways at `limit`,
/// scales the sigmoid argument by `alpha` and adds one to up.
/// The exact f32 image of a slice the reference can transcribe: f32 as
/// it is, bf16 widened bit-exactly. Any other form needs a decode the
/// reference does not perform, and is refused by name.
fn widen_reference<'a>(
    slice: &WeightSlice<'a>,
) -> Result<std::borrow::Cow<'a, [f32]>, VindexError> {
    match slice {
        WeightSlice::F32(w) => Ok(std::borrow::Cow::Borrowed(w)),
        WeightSlice::Bf16(w) => Ok(std::borrow::Cow::Owned(
            w.iter()
                .map(|b| f32::from_bits(u32::from(*b) << 16))
                .collect(),
        )),
        other => Err(VindexError::Parse(format!(
            "the reference transcribes f32 and bf16 experts; a {} expert was handed to it",
            slice_form(other)
        ))),
    }
}

/// The stored form a slice is in, named for a refusal — never its bytes.
fn slice_form(slice: &WeightSlice<'_>) -> &'static str {
    match slice {
        WeightSlice::F32(_) => "f32",
        WeightSlice::Bf16(_) => "bf16",
        WeightSlice::F16(_) => "f16",
        WeightSlice::Q8 { .. } => "q8",
        WeightSlice::Q4 { .. } => "q4",
        WeightSlice::Mxfp4 { .. } => "mxfp4",
        WeightSlice::Nvfp4 { .. } => "nvfp4",
        WeightSlice::KQuant { .. } => "k-quant",
    }
}

fn combine_gate_up_reference(
    policy: larql_models::ExpertGatePolicy,
    activation: larql_models::config::Activation,
    g: f32,
    u: f32,
) -> f32 {
    match policy {
        larql_models::ExpertGatePolicy::Gated => activate(activation, g) * u,
        larql_models::ExpertGatePolicy::ClampedGlu { limit, alpha } => {
            let g = g.min(limit);
            let u = u.clamp(-limit, limit);
            (u + 1.0) * (g * sigmoid(alpha * g))
        }
    }
}

/// `x[i] += b[i]`; a bias of the wrong length is a geometry bug closure
/// should have refused, so it panics rather than pads.
fn add_in_place(x: &mut [f32], b: &[f32]) {
    assert_eq!(
        x.len(),
        b.len(),
        "bias length must equal the projection's rows"
    );
    for (x, b) in x.iter_mut().zip(b) {
        *x += b;
    }
}

impl PlanBackend for ReferenceBackend {
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
        norm(call.kind, call.x, call.weight, call.weight_offset, call.eps)
    }

    fn project(&self, call: ProjectCall<'_>) -> Result<Vec<f32>, VindexError> {
        Ok(matvec(
            call.weight.as_f32()?,
            call.out_dim,
            call.in_dim,
            call.x,
        ))
    }

    fn attention(&self, call: AttentionCall<'_>) -> Result<AttentionOut, VindexError> {
        self.attention_mutated(call, GateMutation::None)
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
        super::production::require_plain_gate("reference", call.gate_policy)?;
        let up = matvec(call.up.as_f32()?, call.intermediate, call.hidden, call.x);
        let inner: Vec<f32> = match call.gate {
            Some(gate_weight) => {
                let gate = matvec(
                    gate_weight.as_f32()?,
                    call.intermediate,
                    call.hidden,
                    call.x,
                );
                gate.iter()
                    .zip(&up)
                    .map(|(g, u)| activate(call.activation, *g) * u)
                    .collect()
            }
            None => up.iter().map(|u| activate(call.activation, *u)).collect(),
        };
        Ok(matvec(
            call.down.as_f32()?,
            call.hidden,
            call.intermediate,
            &inner,
        ))
    }

    /// The routed FFN, stated literally: router logits, the judged
    /// selection rule, each selected expert's fused gate/up read through
    /// the declared row layout, the judged gate policy, its down
    /// projection, and the weighted sum. Shares nothing with the served
    /// MoE path — plain loops over widened f32 operands.
    fn routed_ffn(&self, call: RoutedFfnCall<'_>) -> Result<Vec<f32>, VindexError> {
        let selected = select_experts_reference(&call)?;
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
                        matvec(gate_up[expert].as_f32()?, two_inter, call.hidden, call.x);
                    if let Some(bias) = call.gate_up_bias {
                        for (f, b) in fused
                            .iter_mut()
                            .zip(&bias[expert * two_inter..(expert + 1) * two_inter])
                        {
                            *f += b;
                        }
                    }
                    let inner: Vec<f32> = (0..call.intermediate)
                        .map(|i| {
                            let g = fused[layout.row(GateUpBranch::Gate, i, call.intermediate)];
                            let u = fused[layout.row(GateUpBranch::Up, i, call.intermediate)];
                            combine_gate_up_reference(call.gate_policy, call.activation, g, u)
                        })
                        .collect();
                    let mut expert_out = matvec(
                        down[expert].as_f32()?,
                        call.hidden,
                        call.intermediate,
                        &inner,
                    );
                    if let Some(bias) = call.down_bias {
                        for (o, b) in expert_out
                            .iter_mut()
                            .zip(&bias[expert * call.hidden..(expert + 1) * call.hidden])
                        {
                            *o += b;
                        }
                    }
                    for (acc, v) in out.iter_mut().zip(&expert_out) {
                        *acc += weight * v;
                    }
                }
            }
            // A per-expert bank, transcribed literally: each matrix
            // widened to f32 exactly (bf16 is the top half of the f32 it
            // denotes), three matvecs, the declared combine.
            ExpertSlices::Separate { gate, up, down, .. } => {
                if call.gate_up_bias.is_some() || call.down_bias.is_some() {
                    return Err(VindexError::Parse(
                        "a per-expert bank carries no expert bias; the call declares one"
                            .to_string(),
                    ));
                }
                for (expert, weight) in selected {
                    let g = matvec(
                        &widen_reference(&gate[expert])?,
                        call.intermediate,
                        call.hidden,
                        call.x,
                    );
                    let u = matvec(
                        &widen_reference(&up[expert])?,
                        call.intermediate,
                        call.hidden,
                        call.x,
                    );
                    let inner: Vec<f32> = g
                        .iter()
                        .zip(&u)
                        .map(|(g, u)| {
                            combine_gate_up_reference(call.gate_policy, call.activation, *g, *u)
                        })
                        .collect();
                    let expert_out = matvec(
                        &widen_reference(&down[expert])?,
                        call.hidden,
                        call.intermediate,
                        &inner,
                    );
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
        let mut logits = matvec(projection.as_f32()?, vocab, hidden, x);
        for logit in &mut logits {
            if let Some(multiplier) = multiplier {
                *logit *= multiplier as f32;
            }
            if let Some(cap) = softcapping {
                *logit = softcap(*logit, cap);
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

impl ReferenceBackend {
    /// [`PlanBackend::attention`] with a deliberate defect in the fused
    /// query/gate path.
    ///
    /// The trait method is the only production caller and always passes
    /// [`GateMutation::None`], so QW-3.5C's mutation table drives the
    /// SHIPPED implementation rather than a transcription of it.
    pub(super) fn attention_mutated(
        &self,
        call: AttentionCall<'_>,
        gate_mutation: GateMutation,
    ) -> Result<AttentionOut, VindexError> {
        // Projections per position, with QK normalisation, query scale
        // and position encoding applied in the judged order. Positions
        // are independent, so they run in parallel with each position's
        // arithmetic untouched — bit-identical to the serial order.
        let projected: Vec<(Vec<f32>, Vec<f32>, Vec<f32>)> = call
            .inputs
            .par_iter()
            .enumerate()
            .map(|(position, pre)| {
                Self::project_position_inner(&call, position, pre, gate_mutation)
            })
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
        let outputs: Vec<Vec<f32>> = queries
            .par_iter()
            .enumerate()
            .map(|(position, query)| {
                Self::attend_position_inner(
                    &call,
                    position,
                    query,
                    |p| keys[p].as_slice(),
                    |p| values[p].as_slice(),
                    &call.inputs[position],
                    gate_mutation,
                )
            })
            .collect::<Result<_, VindexError>>()?;
        Ok(AttentionOut {
            outputs,
            keys,
            values,
        })
    }
}
