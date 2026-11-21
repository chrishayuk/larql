//! Reference routing helpers and the reference `PlanBackend` impl.

use super::super::backend::{
    AttentionCall, AttentionOut, AttentionStepCall, AttentionStepOut, ExpertSlices, FfnCall,
    NormCall, PlanBackend, ProjectCall, RoutedFfnCall, WeightSlice,
};
use super::super::kernels::{activate, matvec, norm, sigmoid, softcap, GateMutation};
use super::super::lowering::LoweringIdentity;
use super::super::observe::AttentionHeadRecord;
use crate::error::VindexError;
use larql_models::config::{ExpertRoutingPolicy, GateUpBranch, MoeRouterKind};

#[allow(unused_imports)]
use super::*;

/// Gate and up: the two branches sharing one fused operand.
pub(super) const FUSED_BRANCHES: usize = larql_models::quant::mxfp4::FUSED_HALVES;

/// The judged expert selection, in the literal form: rank the logits
/// (ties to the lower index, as `torch.topk`), keep `top_k`, and weight
/// them by a softmax whose denominator the routing policy chooses — every
/// expert (`SoftmaxThenSelect`) or the selected ones only
/// (`NormalisedOverSelected`; GPT-OSS's top-k-then-softmax is that same
/// number). Gemma 4 (`TopKRenormScaled`) is its own rule, transcribed from
/// `Gemma4TextRouter.forward`: the router input is the raw residual,
/// RMS-normalised without a weight, times `scale`, times `hidden^-0.5`;
/// softmax over every expert; top-k; the selected weights renormalised to
/// sum to one; then each multiplied by `per_expert_scale[expert]`.
pub(in super::super) fn select_experts_reference(
    call: &RoutedFfnCall<'_>,
) -> Result<Vec<(usize, f32)>, VindexError> {
    let mut selected = if call.router_kind == MoeRouterKind::TopKRenormScaled {
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
pub(super) const RENORM_EPS: f32 = 1e-20;

/// DeepSeek-V3's `MoEGate` with `scoring_func = "sigmoid"`, literal: scores
/// are sigmoids of the logits; `topk` runs over `scores +
/// e_score_correction_bias`; the gathered weights are the UNCORRECTED
/// scores, renormalised when `norm_topk_prob`.
pub(super) fn select_experts_sigmoid_reference(call: &RoutedFfnCall<'_>) -> Vec<(usize, f32)> {
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
pub(super) fn select_experts_softmax_reference(call: &RoutedFfnCall<'_>) -> Vec<(usize, f32)> {
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
pub(super) fn select_experts_gemma4_reference(
    call: &RoutedFfnCall<'_>,
) -> Result<Vec<(usize, f32)>, VindexError> {
    let missing = |what: &str| {
        VindexError::Parse(format!(
            "TopKRenormScaled router without its {what}; the plan must carry it"
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
pub(super) fn widen_reference<'a>(
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
pub(super) fn slice_form<'a>(slice: &WeightSlice<'a>) -> &'a str {
    match slice {
        WeightSlice::F32(_) => "f32",
        WeightSlice::Bf16(_) => "bf16",
        WeightSlice::F16(_) => "f16",
        WeightSlice::Q8 { .. } => "q8",
        WeightSlice::Q4 { .. } => "q4",
        WeightSlice::Mxfp4 { .. } => "mxfp4",
        WeightSlice::Nvfp4 { .. } => "nvfp4",
        WeightSlice::KQuant { .. } => "k-quant",
        WeightSlice::Fp8Block { .. } => "fine-grained fp8",
        WeightSlice::CodecOwned { label, .. } => label,
    }
}

pub(super) fn combine_gate_up_reference(
    policy: larql_models::ExpertGatePolicy,
    activation: larql_models::config::Activation,
    g: f32,
    u: f32,
) -> f32 {
    match policy {
        larql_models::ExpertGatePolicy::Gated
            if super::super::diag_c::get().silu_d
                && activation == larql_models::config::Activation::Silu =>
        {
            larql_compute::cpu::ops::geglu::silu(g) * u
        }
        larql_models::ExpertGatePolicy::Gated => activate(activation, g) * u,
        larql_models::ExpertGatePolicy::ClampedGlu { limit, alpha } => {
            let g = g.min(limit);
            let u = u.clamp(-limit, limit);
            (u + 1.0) * (g * sigmoid(alpha * g))
        }
        // Delegated, not transcribed a third time. SiTU has exactly one
        // formula and `MoeGateRule::combine` is where it lives — unlike
        // the `Gated` arm above, which deliberately differs from the
        // production tier (exact erf-GELU here, the tanh approximation
        // there) and therefore has to be spelled out.
        larql_models::ExpertGatePolicy::SituGlu { beta, linear_beta } => {
            larql_compute::MoeGateRule::SituGlu { beta, linear_beta }.combine(g, u)
        }
        // GLM-5.3-Flash: the SAME clamp, then the ordinary gated
        // product. No `alpha`, no `(u + 1)`.
        larql_models::ExpertGatePolicy::ClampedGated { limit } => {
            let g = g.min(limit);
            let u = u.clamp(-limit, limit);
            activate(activation, g) * u
        }
    }
}

/// `x[i] += b[i]`; a bias of the wrong length is a geometry bug closure
/// should have refused, so it panics rather than pads.
pub(super) fn add_in_place(x: &mut [f32], b: &[f32]) {
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

    fn identity(&self) -> LoweringIdentity {
        LoweringIdentity::new(IDENTITY_FAMILY, IDENTITY_REVISION)
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
        Self::attention_step_tapped(step, None, None)
    }

    fn serves_attention_heads(&self) -> bool {
        true
    }

    fn attention_step_observed(
        &self,
        step: AttentionStepCall<'_>,
        tap: &mut dyn FnMut(AttentionHeadRecord<'_>),
    ) -> Result<AttentionStepOut, VindexError> {
        Self::attention_step_tapped(step, Some(tap), None)
    }

    fn serves_head_intervention(&self) -> bool {
        true
    }

    fn attention_step_intervened(
        &self,
        step: AttentionStepCall<'_>,
        tap: Option<&mut dyn FnMut(AttentionHeadRecord<'_>)>,
        head_intervene: &mut super::super::backend::HeadIntervene<'_>,
    ) -> Result<AttentionStepOut, VindexError> {
        Self::attention_step_tapped(step, tap, Some(head_intervene))
    }

    fn ffn(&self, call: FfnCall<'_>) -> Result<Vec<f32>, VindexError> {
        self.ffn_observed(call, &mut |_| {})
    }

    fn serves_ffn_down_input(&self) -> bool {
        true
    }

    fn ffn_observed(
        &self,
        call: FfnCall<'_>,
        tap: &mut dyn FnMut(&[f32]),
    ) -> Result<Vec<f32>, VindexError> {
        super::super::production::require_executable_gate("reference", call.gate_policy)?;
        use super::super::diag_c::{matvec_at, Site};
        let up = matvec_at(Site::Up, call.up.as_f32()?, call.intermediate, call.hidden, call.x);
        let inner: Vec<f32> = match call.gate {
            Some(gate_weight) => {
                let gate = matvec_at(
                    Site::Gate,
                    gate_weight.as_f32()?,
                    call.intermediate,
                    call.hidden,
                    call.x,
                );
                // Through the same combine the routed path uses, so this
                // backend has ONE answer for what a gate policy means
                // rather than a dense answer and a routed one.
                gate.iter()
                    .zip(&up)
                    .map(|(g, u)| {
                        combine_gate_up_reference(call.gate_policy, call.activation, *g, *u)
                    })
                    .collect()
            }
            None => up.iter().map(|u| activate(call.activation, *u)).collect(),
        };
        tap(&inner);
        Ok(matvec_at(
            Site::Down,
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
        projection: super::super::backend::WeightSlice<'_>,
        vocab: usize,
        hidden: usize,
        x: &[f32],
        multiplier: Option<f64>,
        softcapping: Option<f32>,
    ) -> Result<Vec<f32>, VindexError> {
        let mut logits = super::super::diag_c::matvec_at(
            super::super::diag_c::Site::Head,
            projection.as_f32()?,
            vocab,
            hidden,
            x,
        );
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
    pub(in super::super) fn attention_mutated(
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
