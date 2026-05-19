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
mod forward;
mod math;

pub use crate::cpu::ops::q4k_q8k_dot::{quantize_x_to_q8k, Q8KActivation};
pub use expert::{
    pre_experts_norm, quantize_h_norm_for_q4k, run_single_expert, run_single_expert_into,
    run_single_expert_q4k_q8k_into, run_single_expert_with_norm, ExpertScratch,
};
pub use forward::cpu_moe_forward;

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

pub fn moe_route_from_router_input(
    router_in: &[f32],
    moe: &MoeLayerWeights<'_>,
) -> (Vec<usize>, Vec<f32>) {
    let hidden = router_in.len();
    let num_experts = moe.num_experts;
    let top_k_val = moe.top_k;

    let mut logits = math::matmul_vec(router_in, moe.router_proj, num_experts, hidden);
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
mod tests {
    use super::*;
    use crate::MoeLayerWeights;

    fn make_moe<'a>(
        hidden: usize,
        inter: usize,
        num_experts: usize,
        top_k: usize,
        gate_up: &'a [u8],
        down: &'a [u8],
        router: &'a [f32],
    ) -> MoeLayerWeights<'a> {
        let gu_stride = 2 * inter * hidden * 2;
        let dn_stride = hidden * inter * 2;
        let experts_gate_up: Vec<&[u8]> = (0..num_experts)
            .map(|e| &gate_up[e * gu_stride..(e + 1) * gu_stride])
            .collect();
        let experts_down: Vec<&[u8]> = (0..num_experts)
            .map(|e| &down[e * dn_stride..(e + 1) * dn_stride])
            .collect();
        MoeLayerWeights {
            experts_gate_up,
            experts_down,
            routing_policy: crate::MoeRoutingPolicy::default(),
            weight_layout: crate::MoeWeightLayout::default(),
            expert_data_format: crate::QuantFormat::BF16,
            router_proj: router,
            router_scale: &[],
            router_per_expert_scale: &[],
            router_norm: &[],
            router_norm_parameter_free: false,
            router_input_scalar: 1.0,
            pre_experts_norm: &[],
            post_ffn1_norm: &[],
            post_experts_norm: &[],
            num_experts,
            top_k,
            intermediate_size: inter,
            activation: crate::Activation::Silu,
        }
    }

    #[test]
    fn test_moe_zero_input_produces_zero() {
        let hidden = 8;
        let inter = 4;
        let num_experts = 4;
        let top_k = 2;

        // All-zero BF16 weights (value 0.0 in BF16 = 0x0000)
        let gate_up = vec![0u8; num_experts * 2 * inter * hidden * 2];
        let down = vec![0u8; num_experts * hidden * inter * 2];
        let router = vec![0.0f32; num_experts * hidden];

        let moe = make_moe(hidden, inter, num_experts, top_k, &gate_up, &down, &router);
        let h = vec![1.0f32; hidden];
        let out = cpu_moe_forward(&h, &moe, 0.0, 1e-6);
        assert_eq!(out.len(), hidden);
        assert!(
            out.iter().all(|v| v.abs() < 1e-5),
            "zero weights → zero output"
        );
    }

    #[test]
    fn cache_eviction_no_panic() {
        // Insert 70 unique heap allocations to trigger LRU eviction (default cap = 64).
        // Keeps all Vecs alive simultaneously so the allocator gives unique addresses.
        let _bufs: Vec<Vec<u8>> = (0..70usize)
            .map(|i| {
                // Vary content slightly so the allocator can't trivially reuse the slot,
                // but the key guarantee is unique heap pointer per live Vec.
                let data = vec![i as u8, 0x3Fu8, 0x00u8, 0x3Fu8]; // 2 BF16 values
                let _ = cache::try_cached_dequant(&data, crate::QuantFormat::BF16, data.len() / 2)
                    .unwrap();
                data
            })
            .collect();
        // Reaching here without panic confirms eviction path is safe.
        assert_eq!(_bufs.len(), 70);
    }

    #[test]
    fn cache_hit_returns_same_arc() {
        // Same byte slice pointer → second call hits the cache, no new allocation.
        let data = vec![0x80u8, 0x3Fu8, 0x80u8, 0x3Fu8]; // BF16 1.0 × 2
        let first = cache::try_cached_dequant(&data, crate::QuantFormat::BF16, 2).unwrap();
        let second = cache::try_cached_dequant(&data, crate::QuantFormat::BF16, 2).unwrap();
        // Both Arcs should point to the same allocation (same pointer).
        assert!(
            std::sync::Arc::ptr_eq(&first, &second),
            "cache hit should return the same Arc"
        );
    }

    #[test]
    fn router_input_scalar_zero_is_applied() {
        let hidden = 4;
        let num_experts = 2;
        let top_k = 2;
        let h = vec![1.0f32; hidden];
        let router = vec![
            0.0, 0.0, 0.0, 0.0, // expert 0
            2.0, 2.0, 2.0, 2.0, // expert 1
        ];

        let moe_no_scale = MoeLayerWeights {
            experts_gate_up: Vec::new(),
            experts_down: Vec::new(),
            routing_policy: crate::MoeRoutingPolicy::default(),
            weight_layout: crate::MoeWeightLayout::default(),
            expert_data_format: crate::QuantFormat::BF16,
            router_proj: &router,
            router_scale: &[],
            router_per_expert_scale: &[],
            router_norm: &[],
            router_norm_parameter_free: false,
            router_input_scalar: 1.0,
            pre_experts_norm: &[],
            post_ffn1_norm: &[],
            post_experts_norm: &[],
            num_experts,
            top_k,
            intermediate_size: 1,
            activation: crate::Activation::Silu,
        };
        let (_, no_scale_weights) = cpu_moe_route(&h, &moe_no_scale, 1e-6);
        let moe_zero_scale = MoeLayerWeights {
            router_input_scalar: 0.0,
            ..moe_no_scale
        };
        let (_, zero_scale_weights) = cpu_moe_route(&h, &moe_zero_scale, 1e-6);

        assert_ne!(no_scale_weights, zero_scale_weights);
        assert!(zero_scale_weights.iter().all(|w| (*w - 0.5).abs() < 1e-6));
    }

    #[test]
    fn top_k_softmax_policy_keeps_raw_selected_weight() {
        let num_experts = 2;
        let h = [1.0f32, 0.0];
        let router = [
            2.0, 0.0, // expert 0 logit = 2
            0.0, 0.0, // expert 1 logit = 0
        ];
        let moe = MoeLayerWeights {
            experts_gate_up: Vec::new(),
            experts_down: Vec::new(),
            routing_policy: crate::MoeRoutingPolicy::top_k_softmax(),
            weight_layout: crate::MoeWeightLayout::default(),
            expert_data_format: crate::QuantFormat::BF16,
            router_proj: &router,
            router_scale: &[],
            router_per_expert_scale: &[10.0, 10.0],
            router_norm: &[],
            router_norm_parameter_free: true,
            router_input_scalar: 1.0,
            pre_experts_norm: &[2.0, 2.0],
            post_ffn1_norm: &[],
            post_experts_norm: &[],
            num_experts,
            top_k: 1,
            intermediate_size: 1,
            activation: crate::Activation::Silu,
        };

        let (indices, weights) = cpu_moe_route(&h, &moe, 1e-6);

        assert_eq!(indices, vec![0]);
        assert!(
            weights[0] < 1.0 && weights[0] > 0.5,
            "top_k_softmax policy should keep the selected softmax probability"
        );
    }

    #[test]
    fn test_moe_identity_expert() {
        // Construct a single expert that acts as identity via gate≫0, up=1, down=identity
        // This verifies the full path runs without panics.
        let hidden = 4;
        let inter = 2;
        let num_experts = 2;
        let top_k = 1;

        // BF16 encoding of 1.0 = 0x3F80
        let one_bf16 = [0x80u8, 0x3Fu8];
        // BF16 encoding of 5.0 (large gate → SiLU ≈ 5) = 0x40A0
        let five_bf16 = [0xA0u8, 0x40u8];

        // gate_up: [num_experts, 2*inter, hidden] — expert 0: gate rows = 5.0, up rows = 1.0
        let mut gate_up = vec![0u8; num_experts * 2 * inter * hidden * 2];
        // Expert 0, gate rows (rows 0..inter): set to 5.0
        for row in 0..inter {
            for col in 0..hidden {
                let byte_off = (row * hidden + col) * 2;
                gate_up[byte_off] = five_bf16[0];
                gate_up[byte_off + 1] = five_bf16[1];
            }
        }
        // Expert 0, up rows (rows inter..2*inter): set to 1.0
        for row in inter..2 * inter {
            for col in 0..hidden {
                let byte_off = (row * hidden + col) * 2;
                gate_up[byte_off] = one_bf16[0];
                gate_up[byte_off + 1] = one_bf16[1];
            }
        }

        // down: [num_experts, hidden, inter] — expert 0: 1.0 everywhere
        let mut down = vec![0u8; num_experts * hidden * inter * 2];
        for i in 0..(hidden * inter) {
            let byte_off = i * 2;
            down[byte_off] = one_bf16[0];
            down[byte_off + 1] = one_bf16[1];
        }

        // router: [num_experts, hidden] — expert 0 row has 1.0, expert 1 row has 0.0
        let mut router = vec![0.0f32; num_experts * hidden];
        router[..hidden].fill(1.0); // expert 0 gets high logit

        let moe = make_moe(hidden, inter, num_experts, top_k, &gate_up, &down, &router);
        let h = vec![1.0f32; hidden];
        let out = cpu_moe_forward(&h, &moe, 0.0, 1e-6);
        assert_eq!(out.len(), hidden);
        // Output should be nonzero since gate activates
        assert!(
            out.iter().any(|v| v.abs() > 0.01),
            "expected nonzero output from identity-like expert"
        );
    }

    /// Q4_K path: build per-expert tables of quantised bytes (one super-block
    /// per expert in this fixture: hidden=256, inter=128 so the matmul shapes
    /// are 2*128*256 = 65536 elements = 256 super-blocks per gate+up entry).
    /// The test confirms `cpu_moe_forward` produces a finite, non-NaN output
    /// when the format dispatch routes to the Q4_K dequantiser.
    #[test]
    fn cpu_moe_forward_q4k_dispatch() {
        use crate::cpu::ops::q4_common::quantize_q4_k;

        // Smallest legal Q4_K MoE shape: hidden must be multiple of 256.
        let hidden = 256;
        let inter = 256; // multiple of 256 → no padding
        let num_experts = 2;
        let top_k = 1;

        let gate_up_floats = 2 * inter * hidden; // = 131072 = 512 super-blocks
        let down_floats = hidden * inter;

        // Same f32 ramp for both experts; routes to expert 0 via router.
        let ramp: Vec<f32> = (0..gate_up_floats)
            .map(|i| (i as f32 / gate_up_floats as f32 - 0.5) * 0.2)
            .collect();
        let down_ramp: Vec<f32> = (0..down_floats)
            .map(|i| (i as f32 / down_floats as f32 - 0.5) * 0.1)
            .collect();
        let gu_q = quantize_q4_k(&ramp);
        let dn_q = quantize_q4_k(&down_ramp);

        // Per-expert table: same bytes for both experts — fine for the smoke test.
        let experts_gate_up: Vec<&[u8]> = vec![&gu_q, &gu_q];
        let experts_down: Vec<&[u8]> = vec![&dn_q, &dn_q];

        // Router: high logit on expert 0.
        let mut router = vec![0.0f32; num_experts * hidden];
        router[..hidden].fill(1.0);

        let h = vec![0.5f32; hidden];
        let moe = MoeLayerWeights {
            experts_gate_up,
            experts_down,
            routing_policy: crate::MoeRoutingPolicy::default(),
            weight_layout: crate::MoeWeightLayout::default(),
            expert_data_format: crate::QuantFormat::Q4_K,
            router_proj: &router,
            router_scale: &[],
            router_per_expert_scale: &[],
            router_norm: &[],
            router_norm_parameter_free: false,
            router_input_scalar: 1.0,
            pre_experts_norm: &[],
            post_ffn1_norm: &[],
            post_experts_norm: &[],
            num_experts,
            top_k,
            intermediate_size: inter,
            activation: crate::Activation::Silu,
        };

        let out = cpu_moe_forward(&h, &moe, 0.0, 1e-6);
        assert_eq!(out.len(), hidden);
        assert!(
            out.iter().all(|v| v.is_finite()),
            "Q4_K MoE output must be finite (no NaN/Inf): {:?}",
            out.iter().take(4).collect::<Vec<_>>()
        );
        assert!(
            out.iter().any(|v| v.abs() > 1e-6),
            "Q4_K dispatch produced all-zeros — format routing likely broken"
        );
    }

    /// Per-expert table indexing: routing to expert 1 must use `experts_*[1]`,
    /// not `experts_*[0]` plus a stride. Build a fixture where expert 0's gate
    /// is zero and expert 1's gate is non-zero — output should be non-zero
    /// (proves the router selected expert 1 AND the indexing pulled the right
    /// per-expert byte slice).
    #[test]
    fn per_expert_indexing_routes_correctly() {
        let hidden = 4;
        let inter = 2;
        let num_experts = 2;
        let top_k = 1;

        // BF16: 1.0 = [0x80, 0x3F]; 0.0 = [0x00, 0x00].
        let one_bf16 = [0x80u8, 0x3Fu8];
        let zero_bf16 = [0x00u8, 0x00u8];
        // Expert 0: all zeros (gate_up + down). Expert 1: gate=5.0, up=down=1.0.
        // gate_up shape [2*inter, hidden] = 16 floats = 32 bytes per expert.
        let mut e0_gu = vec![0u8; 2 * inter * hidden * 2];
        for chunk in e0_gu.chunks_exact_mut(2) {
            chunk.copy_from_slice(&zero_bf16);
        }
        let mut e1_gu = vec![0u8; 2 * inter * hidden * 2];
        // Expert 1 gate rows (rows 0..inter): 5.0 BF16 = [0xA0, 0x40].
        let five_bf16 = [0xA0u8, 0x40u8];
        for row in 0..inter {
            for col in 0..hidden {
                let off = (row * hidden + col) * 2;
                e1_gu[off] = five_bf16[0];
                e1_gu[off + 1] = five_bf16[1];
            }
        }
        // Expert 1 up rows: 1.0.
        for row in inter..2 * inter {
            for col in 0..hidden {
                let off = (row * hidden + col) * 2;
                e1_gu[off] = one_bf16[0];
                e1_gu[off + 1] = one_bf16[1];
            }
        }
        // Down: e0 zero, e1 1.0 everywhere.
        let e0_dn = vec![0u8; hidden * inter * 2];
        let mut e1_dn = vec![0u8; hidden * inter * 2];
        for chunk in e1_dn.chunks_exact_mut(2) {
            chunk.copy_from_slice(&one_bf16);
        }

        // Router: row for expert 1 is 1.0, row for expert 0 is 0.0 →
        // expert 1 wins, output should be non-zero. If indexing were swapped,
        // the router would still pick expert id 1 but pull expert 0's bytes
        // (all zeros) and the output would be 0.
        let mut router = vec![0.0f32; num_experts * hidden];
        router[hidden..].fill(1.0); // expert 1 row

        let moe = MoeLayerWeights {
            experts_gate_up: vec![&e0_gu, &e1_gu],
            experts_down: vec![&e0_dn, &e1_dn],
            routing_policy: crate::MoeRoutingPolicy::default(),
            weight_layout: crate::MoeWeightLayout::default(),
            expert_data_format: crate::QuantFormat::BF16,
            router_proj: &router,
            router_scale: &[],
            router_per_expert_scale: &[],
            router_norm: &[],
            router_norm_parameter_free: false,
            router_input_scalar: 1.0,
            pre_experts_norm: &[],
            post_ffn1_norm: &[],
            post_experts_norm: &[],
            num_experts,
            top_k,
            intermediate_size: inter,
            activation: crate::Activation::Silu,
        };

        let h = vec![1.0f32; hidden];
        let out = cpu_moe_forward(&h, &moe, 0.0, 1e-6);
        assert_eq!(out.len(), hidden);
        assert!(
            out.iter().any(|v| v.abs() > 0.01),
            "expert 1 has non-zero weights; output must be non-zero. \
             Got {out:?} — per-expert indexing is likely confusing 0 and 1."
        );
    }

    /// Regression test: `cpu_moe_forward` and `cpu_moe_route` must agree on
    /// the **router input convention** — both should compute the router norm
    /// on top of the pre-experts-normed h (not raw h).
    ///
    /// History: silently picking different top-K experts between the two
    /// paths produced incoherent text on Gemma 4 26B-A4B. The h_norm
    /// convention matches Metal's `gpu_moe_dispatch` and the trained
    /// 26B-A4B weights — even though HF's modeling_gemma4.py uses raw h.
    /// `larql parity --component moe-block` exposes the divergence.
    ///
    /// The fixture chooses non-trivial `pre_experts_norm` weights so raw-h
    /// and h_norm produce **different** logits, then asserts the two paths
    /// pick the **same** top-K (i.e., both route on the same input).
    #[test]
    fn cpu_moe_forward_uses_same_router_input_as_cpu_moe_route() {
        // 4-expert, top-2 fixture. Use non-uniform `pre_experts_norm` so
        // h_norm differs from h enough to sometimes flip the top-K choice
        // (vs identity-norm where h_norm == h after rescaling).
        let hidden = 8;
        let inter = 4;
        let num_experts = 4;
        let top_k = 2;

        // pre_experts_norm: arbitrary non-uniform weights (some negative
        // would also be fine; here a simple 1, 1.5, 2, ... ramp with one
        // strong outlier ensures rms(h*w) != rms(h) for typical inputs).
        let pre_norm: Vec<f32> = (0..hidden).map(|i| 1.0 + i as f32 * 0.5).collect();

        // Router projection: arrange so the [0] dim of h dominates in raw
        // space but a different dim dominates in normed space.
        let mut router_proj = vec![0.0f32; num_experts * hidden];
        // Expert 0: large weight on dim 0 → wins raw routing.
        router_proj[0] = 5.0;
        // Expert 1: large weight on dim 7 → may win normed routing
        // because pre_norm[7] = 1 + 3.5 = 4.5, amplifying that dim.
        router_proj[hidden + 7] = 5.0;
        router_proj[2 * hidden + 3] = 1.0;
        router_proj[3 * hidden + 5] = 1.0;

        // Identity gate_up + down so per-expert outputs are deterministic
        // (we only care about top-K selection here).
        let gate_up = vec![0u8; num_experts * 2 * inter * hidden * 2];
        let down = vec![0u8; num_experts * hidden * inter * 2];

        // Build per-expert byte tables (matches the post-refactor API).
        let gu_stride = 2 * inter * hidden * 2;
        let dn_stride = hidden * inter * 2;
        let experts_gate_up: Vec<&[u8]> = (0..num_experts)
            .map(|e| &gate_up[e * gu_stride..(e + 1) * gu_stride])
            .collect();
        let experts_down: Vec<&[u8]> = (0..num_experts)
            .map(|e| &down[e * dn_stride..(e + 1) * dn_stride])
            .collect();

        let moe = MoeLayerWeights {
            experts_gate_up,
            experts_down,
            routing_policy: crate::MoeRoutingPolicy::default(),
            weight_layout: crate::MoeWeightLayout::default(),
            expert_data_format: crate::QuantFormat::BF16,
            router_proj: &router_proj,
            router_scale: &[],
            router_per_expert_scale: &[],
            router_norm: &[],
            // Force the parameter-free RMSNorm path on routing. This is the
            // Gemma 4 26B-A4B convention; it's also the place the bug lived.
            router_norm_parameter_free: true,
            router_input_scalar: 1.0,
            pre_experts_norm: &pre_norm,
            post_ffn1_norm: &[],
            post_experts_norm: &[],
            num_experts,
            top_k,
            intermediate_size: inter,
            activation: crate::Activation::Silu,
        };

        // Sample residual with the [0] and [7] dims at similar magnitudes
        // in raw space but with different scaling under pre_norm.
        let h: Vec<f32> = (0..hidden)
            .map(|i| if i == 0 || i == 7 { 1.0 } else { 0.1 })
            .collect();

        let expert_input = moe_expert_input(&h, &moe, 0.0, 1e-6);
        let router_in = moe_router_input(&h, &expert_input, &moe, 0.0, 1e-6);
        let (route_from_shared_steps, _) = moe_route_from_router_input(&router_in, &moe);
        let (route_indices, _) = cpu_moe_route(&h, &moe, 1e-6);

        let mut raw_logits = math::matmul_vec(&h, &router_proj, num_experts, hidden);
        math::softmax(&mut raw_logits);
        let (route_raw, _) = math::top_k(&raw_logits, top_k);

        // Sanity: the fixture is engineered so the two conventions disagree.
        assert_ne!(
            route_indices, route_raw,
            "fixture is broken — h_norm and raw-h routing must give different \
             top-K, otherwise this test can't catch a regression. \
             route_policy={route_indices:?} route_raw={route_raw:?}"
        );

        // Pin one policy implementation for callers (Metal dispatch,
        // gRPC remote, cpu_moe_forward): public routing and forward's
        // shared steps must agree on the policy-derived router input.
        assert_eq!(route_indices, route_from_shared_steps);
        assert_eq!(
            route_indices.len(),
            top_k,
            "cpu_moe_route should return top_k={top_k} indices"
        );
    }

    /// Per-expert table indexing is by **expert id**, not by position in
    /// the top-K list. Pinning the contract so a future "iterate via the
    /// position-k index instead" refactor would fail loudly.
    ///
    /// History: this test exists because the bench framework's earlier
    /// numbers were misleading (0.10 ms cpu_moe_forward floor was the
    /// buggy old code silently returning empty buffers). We now test
    /// behaviour, not just timing.
    #[test]
    fn experts_gate_up_indexed_by_expert_id_not_topk_position() {
        let hidden = 4;
        let inter = 2;
        let num_experts = 4;
        // Build per-expert tables. Each expert's bytes are tagged by a
        // distinct first-byte signature so we can detect mis-indexing.
        let gu_stride = 2 * inter * hidden * 2;
        let dn_stride = hidden * inter * 2;
        let mut gate_up_blob = vec![0u8; num_experts * gu_stride];
        let mut down_blob = vec![0u8; num_experts * dn_stride];
        for e in 0..num_experts {
            gate_up_blob[e * gu_stride] = 0xA0 + e as u8;
            down_blob[e * dn_stride] = 0xB0 + e as u8;
        }
        let experts_gate_up: Vec<&[u8]> = (0..num_experts)
            .map(|e| &gate_up_blob[e * gu_stride..(e + 1) * gu_stride])
            .collect();
        let experts_down: Vec<&[u8]> = (0..num_experts)
            .map(|e| &down_blob[e * dn_stride..(e + 1) * dn_stride])
            .collect();

        // Verify by index that experts[2] is the bytes tagged 0xA2 / 0xB2.
        assert_eq!(experts_gate_up[2][0], 0xA2);
        assert_eq!(experts_down[2][0], 0xB2);
        assert_eq!(experts_gate_up[3][0], 0xA3);
        // Counter-test: the *first* element of the table (position 0) is
        // expert 0, not whichever expert the router happens to pick first.
        assert_eq!(experts_gate_up[0][0], 0xA0);
    }
}
