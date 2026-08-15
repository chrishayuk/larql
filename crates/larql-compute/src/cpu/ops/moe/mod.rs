//! CPU-side MoE (Mixture-of-Experts) forward pass for hybrid models (Gemma 4 26B A4B).
//!
//! Called when a layer has `is_hybrid_moe() == true`. Computes the expert block
//! in parallel with the dense FFN and returns the expert contribution for summation.
//!
//! Module layout:
//! - [`math`]    — numeric primitives (rms_norm, softmax, top-k, bf16 dequant, matmul)
//! - [`expert`]  — per-expert gated-FFN execution (used by the remote-shard path)
//! - [`forward`] — full block: router → top-k → weighted sum of expert outputs
//!
//! Expert weights are stored as packed BF16: [num_experts, out_dim, in_dim].
//! We dequantize only the selected top-k expert slices on demand.

mod cache;
mod expert;
pub mod expert_override;
mod forward;
pub mod latent_mask;
pub mod math;
mod within_expert;

pub use crate::cpu::ops::q4k_q8k_dot::{quantize_x_to_q8k, Q8KActivation};
pub use expert::{
    pre_experts_norm, quantize_h_norm_for_q4k, run_single_expert, run_single_expert_into,
    run_single_expert_kq_q8k_into, run_single_expert_kq_q8k_parallel_into,
    run_single_expert_q4k_q8k_into, run_single_expert_with_norm, ExpertScratch,
};
pub use forward::cpu_moe_forward;
pub use math::{matmul_vec as moe_score_experts, softmax as moe_softmax};
pub use within_expert::{
    is_active as within_expert_active, set_current_layer, set_routing, ExpertFeatureSelector,
    WithinExpertRouting,
};

use crate::{
    MoeExpertScalePolicy, MoeInputSource, MoeLayerWeights, MoePostExpertNormPolicy,
    MoeRouterNormPolicy, MoeTopKWeightPolicy,
};

/// Process-wide cached snapshot of `LARQL_DISABLE_Q4K_DIRECT`.
///
/// `cpu_moe_forward` (per layer) and `run_single_expert` (per expert
/// per layer) used to read this env every call. On Gemma 4 26B-A4B
/// (40 layers × top_k 4-8) that's ~120-300 `getenv` syscalls per token.
/// Resolving once globally settles them all.
///
/// Trade-off: the env var is process-bound — flip it before the first
/// MoE forward call, not at runtime. There is no production caller
/// that toggles this mid-run; the var is a kernel-debug A/B switch.
pub(crate) fn q4k_direct_disabled() -> bool {
    use std::sync::OnceLock;
    static CACHE: OnceLock<bool> = OnceLock::new();
    *CACHE.get_or_init(|| crate::options::env_flag(crate::options::ENV_DISABLE_Q4K_DIRECT))
}

pub fn moe_expert_input(
    h: &[f32],
    moe: &MoeLayerWeights<'_>,
    norm_offset: f32,
    eps: f32,
) -> Vec<f32> {
    match moe.routing_policy.expert_input {
        MoeInputSource::Residual => h.to_vec(),
        MoeInputSource::PreExpertsNorm => math::rms_norm(h, moe.pre_experts_norm, eps, norm_offset),
    }
}

pub fn moe_router_input(
    h: &[f32],
    expert_input: &[f32],
    moe: &MoeLayerWeights<'_>,
    norm_offset: f32,
    eps: f32,
) -> Vec<f32> {
    let router_base = match moe.routing_policy.router_input {
        MoeInputSource::Residual => h,
        MoeInputSource::PreExpertsNorm => expert_input,
    };

    let router_in_normed = match moe.routing_policy.router_norm {
        MoeRouterNormPolicy::None => router_base.to_vec(),
        MoeRouterNormPolicy::Learned => {
            if moe.router_norm.is_empty() {
                router_base.to_vec()
            } else {
                math::rms_norm(router_base, moe.router_norm, eps, norm_offset)
            }
        }
        MoeRouterNormPolicy::ParameterFree => math::rms_norm_no_weight(router_base, eps),
        MoeRouterNormPolicy::LearnedOrParameterFree => {
            if !moe.router_norm.is_empty() {
                math::rms_norm(router_base, moe.router_norm, eps, norm_offset)
            } else if moe.router_norm_parameter_free {
                math::rms_norm_no_weight(router_base, eps)
            } else {
                router_base.to_vec()
            }
        }
    };

    let mut router_in: Vec<f32> = if !moe.router_scale.is_empty() {
        router_in_normed
            .iter()
            .zip(moe.router_scale.iter())
            .map(|(a, b)| a * b)
            .collect()
    } else {
        router_in_normed
    };
    if moe.router_input_scalar != 1.0 {
        for v in &mut router_in {
            *v *= moe.router_input_scalar;
        }
    }
    router_in
}

/// Router projection: `logits[e] = dot(router_in, router_proj[e]) + bias[e]`.
///
/// Router bias joins the logits BEFORE softmax/selection — it changes
/// which experts win, not just their weights, so adding it later would
/// be a different (wrong) router. Reference: `ExpertWeightFfn::
/// router_logits`, which the f32 tier pins against `transformers`.
///
/// This is the served tier's routing oracle: GPU router implementations
/// gate their parity against exactly this function, so route selection
/// changes must land here, not in a copy.
pub fn moe_router_logits(
    router_in: &[f32],
    router_proj: &[f32],
    router_bias: &[f32],
    num_experts: usize,
) -> Vec<f32> {
    let hidden = router_in.len();
    let mut logits = math::matmul_vec(router_in, router_proj, num_experts, hidden);
    if !router_bias.is_empty() {
        assert_eq!(
            router_bias.len(),
            num_experts,
            "router bias length does not match the expert count — this bias \
             does not describe this router"
        );
        for (l, b) in logits.iter_mut().zip(router_bias) {
            *l += b;
        }
    }
    logits
}

pub fn moe_route_from_router_input(
    router_in: &[f32],
    moe: &MoeLayerWeights<'_>,
) -> (Vec<usize>, Vec<f32>) {
    let num_experts = moe.num_experts;
    let top_k_val = moe.top_k;

    let mut logits = moe_router_logits(router_in, moe.router_proj, moe.router_bias, num_experts);
    math::softmax(&mut logits);
    let (indices, mut weights) = math::top_k(&logits, top_k_val);

    if moe.routing_policy.selected_weight == MoeTopKWeightPolicy::RenormalizedSoftmax {
        let sum: f32 = weights.iter().sum();
        if sum > 0.0 {
            for w in &mut weights {
                *w /= sum;
            }
        }
    }

    if moe.routing_policy.expert_scale == MoeExpertScalePolicy::PerExpert
        && !moe.router_per_expert_scale.is_empty()
    {
        for (i, &ei) in indices.iter().enumerate() {
            if ei < moe.router_per_expert_scale.len() {
                weights[i] *= moe.router_per_expert_scale[ei];
            }
        }
    }

    // The served tier's route decision boundary. The reference tier records
    // its own selections where it already knows the layer; this is the other
    // implementation, and leaving it unobserved is what made a routing trace
    // come back empty on a model that was visibly generating.
    crate::moe_route_observe::observe(&indices);

    (indices, weights)
}

pub fn moe_post_expert_output(
    expert_out: &[f32],
    moe: &MoeLayerWeights<'_>,
    norm_offset: f32,
    eps: f32,
) -> Vec<f32> {
    match moe.routing_policy.post_expert_norm {
        MoePostExpertNormPolicy::None => expert_out.to_vec(),
        MoePostExpertNormPolicy::RmsNorm => {
            math::rms_norm(expert_out, moe.post_experts_norm, eps, norm_offset)
        }
    }
}

/// CPU router: returns `(top_k_indices, selected_weights)` for the given
/// hidden state. Used by GPU dispatch paths that route on CPU but run expert
/// FFNs on GPU. Mirrors the policy-driven routing logic in
/// `forward::cpu_moe_forward`.
pub fn cpu_moe_route(
    h: &[f32],
    moe: &crate::MoeLayerWeights<'_>,
    eps: f32,
) -> (Vec<usize>, Vec<f32>) {
    let expert_input = moe_expert_input(h, moe, 0.0, eps);
    let router_in = moe_router_input(h, &expert_input, moe, 0.0, eps);
    moe_route_from_router_input(&router_in, moe)
}

#[cfg(test)]
mod tests;
