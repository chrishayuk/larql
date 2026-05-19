extern crate blas_src;

use larql_compute::cpu::ops::moe::cpu_moe_forward;
use larql_compute::MoeLayerWeights;
use larql_compute::{cpu_backend, default_backend, Activation};

// ── lib.rs entry points ──────────────────────────────────────────────────────

#[test]
fn cpu_backend_name_is_nonempty() {
    assert!(!cpu_backend().name().is_empty());
}

#[test]
fn cpu_backend_device_info_is_nonempty() {
    assert!(!cpu_backend().device_info().is_empty());
}

#[test]
fn default_backend_name_is_nonempty() {
    assert!(!default_backend().name().is_empty());
}

#[test]
fn cpu_backend_is_dyn_compatible() {
    let _: Box<dyn larql_compute::ComputeBackend> = cpu_backend();
}

// ── MoE forward — router norm variants ──────────────────────────────────────

fn bf16_fill(len: usize, val: f32) -> Vec<u8> {
    let hi = (val.to_bits() >> 16) as u16;
    let b = hi.to_le_bytes();
    let mut v = vec![0u8; len * 2];
    for i in 0..len {
        v[i * 2] = b[0];
        v[i * 2 + 1] = b[1];
    }
    v
}

fn bf16_expert_tables<'a>(
    gate_up: &'a [u8],
    down: &'a [u8],
    num_experts: usize,
    inter: usize,
    hidden: usize,
) -> (Vec<&'a [u8]>, Vec<&'a [u8]>) {
    let gu_stride = 2 * inter * hidden * 2;
    let dn_stride = hidden * inter * 2;
    let experts_gate_up = (0..num_experts)
        .map(|e| &gate_up[e * gu_stride..(e + 1) * gu_stride])
        .collect();
    let experts_down = (0..num_experts)
        .map(|e| &down[e * dn_stride..(e + 1) * dn_stride])
        .collect();
    (experts_gate_up, experts_down)
}

#[allow(clippy::too_many_arguments)]
fn make_moe_weights<'a>(
    hidden: usize,
    inter: usize,
    num_experts: usize,
    top_k: usize,
    gate_up: &'a [u8],
    down: &'a [u8],
    router: &'a [f32],
    router_norm: &'a [f32],
    router_norm_parameter_free: bool,
) -> MoeLayerWeights<'a> {
    let (experts_gate_up, experts_down) =
        bf16_expert_tables(gate_up, down, num_experts, inter, hidden);
    MoeLayerWeights {
        experts_gate_up,
        experts_down,
        routing_policy: larql_compute::MoeRoutingPolicy::default(),
        weight_layout: larql_compute::MoeWeightLayout::default(),
        router_proj: router,
        router_scale: &[],
        router_per_expert_scale: &[],
        router_norm,
        router_norm_parameter_free,
        router_input_scalar: 1.0,
        pre_experts_norm: &[],
        post_ffn1_norm: &[],
        post_experts_norm: &[],
        num_experts,
        top_k,
        intermediate_size: inter,
        activation: Activation::Silu,
        expert_data_format: larql_compute::QuantFormat::BF16,
    }
}

#[test]
fn moe_parameter_free_router_norm_runs_without_panic() {
    // Exercises the `rms_norm_no_weight` code path in forward.rs
    let hidden = 8;
    let inter = 4;
    let num_experts = 4;
    let top_k = 2;

    let gate_up = bf16_fill(num_experts * 2 * inter * hidden, 1.0);
    let down = bf16_fill(num_experts * hidden * inter, 1.0);
    // Non-zero router so experts can be selected
    let router: Vec<f32> = (0..num_experts * hidden)
        .map(|i| if i < hidden { 1.0 } else { 0.1 })
        .collect();

    let moe = make_moe_weights(
        hidden,
        inter,
        num_experts,
        top_k,
        &gate_up,
        &down,
        &router,
        &[],  // empty router_norm → triggers parameter_free path
        true, // router_norm_parameter_free = true
    );
    let h = vec![1.0f32; hidden];
    let out = cpu_moe_forward(&h, &moe, 0.0, 1e-6);
    assert_eq!(out.len(), hidden);
}

#[test]
fn moe_learned_router_norm_runs_without_panic() {
    // Exercises the learned `router_norm` code path (non-empty router_norm slice)
    let hidden = 8;
    let inter = 4;
    let num_experts = 4;
    let top_k = 2;

    let gate_up = bf16_fill(num_experts * 2 * inter * hidden, 1.0);
    let down = bf16_fill(num_experts * hidden * inter, 1.0);
    let router: Vec<f32> = (0..num_experts * hidden)
        .map(|i| if i < hidden { 1.0 } else { 0.1 })
        .collect();
    let router_norm = vec![1.0f32; hidden];

    let moe = make_moe_weights(
        hidden,
        inter,
        num_experts,
        top_k,
        &gate_up,
        &down,
        &router,
        &router_norm,
        false,
    );
    let h = vec![1.0f32; hidden];
    let out = cpu_moe_forward(&h, &moe, 0.0, 1e-6);
    assert_eq!(out.len(), hidden);
}

#[test]
fn moe_per_expert_scale_applied() {
    // Verify that per_expert_scale changes the output magnitude.
    let hidden = 8;
    let inter = 4;
    let num_experts = 4;
    let top_k = 1;

    let gate_up = bf16_fill(num_experts * 2 * inter * hidden, 1.0);
    let down = bf16_fill(num_experts * hidden * inter, 1.0);
    let router: Vec<f32> = (0..num_experts * hidden)
        .map(|i| if i < hidden { 1.0 } else { 0.0 })
        .collect();
    let h = vec![1.0f32; hidden];
    let (experts_gate_up, experts_down) =
        bf16_expert_tables(&gate_up, &down, num_experts, inter, hidden);

    // Without per-expert scale
    let moe_no_scale = MoeLayerWeights {
        experts_gate_up: experts_gate_up.clone(),
        experts_down: experts_down.clone(),
        routing_policy: larql_compute::MoeRoutingPolicy::default(),
        weight_layout: larql_compute::MoeWeightLayout::default(),
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
        activation: Activation::Silu,
        expert_data_format: larql_compute::QuantFormat::BF16,
    };
    let out_no_scale = cpu_moe_forward(&h, &moe_no_scale, 0.0, 1e-6);

    // With per-expert scale = [2.0, 1.0, 1.0, 1.0] (expert 0 gets 2× weight)
    let per_expert_scale = vec![2.0f32, 1.0, 1.0, 1.0];
    let moe_scaled = MoeLayerWeights {
        experts_gate_up,
        experts_down,
        routing_policy: larql_compute::MoeRoutingPolicy::default(),
        weight_layout: larql_compute::MoeWeightLayout::default(),
        router_proj: &router,
        router_scale: &[],
        router_per_expert_scale: &per_expert_scale,
        router_norm: &[],
        router_norm_parameter_free: false,
        router_input_scalar: 1.0,
        pre_experts_norm: &[],
        post_ffn1_norm: &[],
        post_experts_norm: &[],
        num_experts,
        top_k,
        intermediate_size: inter,
        activation: Activation::Silu,
        expert_data_format: larql_compute::QuantFormat::BF16,
    };
    let out_scaled = cpu_moe_forward(&h, &moe_scaled, 0.0, 1e-6);

    assert_eq!(out_no_scale.len(), hidden);
    assert_eq!(out_scaled.len(), hidden);
    // Scaled output should differ from unscaled (expert 0 weight doubled)
    let max_diff: f32 = out_no_scale
        .iter()
        .zip(&out_scaled)
        .map(|(a, b)| (a - b).abs())
        .fold(0.0, f32::max);
    assert!(
        max_diff > 1e-6,
        "per_expert_scale should change output; max_diff={max_diff}"
    );
}

#[test]
fn moe_router_scale_vector_applied() {
    // Exercises the `!moe.router_scale.is_empty()` branch in forward.rs
    let hidden = 8;
    let inter = 4;
    let num_experts = 4;
    let top_k = 1;

    let gate_up = bf16_fill(num_experts * 2 * inter * hidden, 1.0);
    let down = bf16_fill(num_experts * hidden * inter, 1.0);
    let router: Vec<f32> = (0..num_experts * hidden)
        .map(|i| if i < hidden { 1.0 } else { 0.0 })
        .collect();
    let router_scale = vec![1.0f32; hidden]; // scale each hidden dim by 1 (neutral)
    let h = vec![1.0f32; hidden];
    let (experts_gate_up, experts_down) =
        bf16_expert_tables(&gate_up, &down, num_experts, inter, hidden);

    let moe = MoeLayerWeights {
        experts_gate_up,
        experts_down,
        routing_policy: larql_compute::MoeRoutingPolicy::default(),
        weight_layout: larql_compute::MoeWeightLayout::default(),
        router_proj: &router,
        router_scale: &router_scale, // non-empty → enters the scale branch
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
        activation: Activation::Silu,
        expert_data_format: larql_compute::QuantFormat::BF16,
    };
    let out = cpu_moe_forward(&h, &moe, 0.0, 1e-6);
    assert_eq!(out.len(), hidden);
}

#[test]
fn moe_router_input_scalar_nonunit() {
    // Exercises the `router_input_scalar != 1.0` branch in forward.rs.
    let hidden = 8;
    let inter = 4;
    let num_experts = 4;
    let top_k = 1;

    let gate_up = bf16_fill(num_experts * 2 * inter * hidden, 1.0);
    let down = bf16_fill(num_experts * hidden * inter, 1.0);
    let router: Vec<f32> = (0..num_experts * hidden)
        .map(|i| if i < hidden { 1.0 } else { 0.0 })
        .collect();
    let h = vec![1.0f32; hidden];
    let (experts_gate_up, experts_down) =
        bf16_expert_tables(&gate_up, &down, num_experts, inter, hidden);

    // scalar = 0.5 → router input scaled down before projection
    let moe_scalar = MoeLayerWeights {
        experts_gate_up,
        experts_down,
        routing_policy: larql_compute::MoeRoutingPolicy::default(),
        weight_layout: larql_compute::MoeWeightLayout::default(),
        router_proj: &router,
        router_scale: &[],
        router_per_expert_scale: &[],
        router_norm: &[],
        router_norm_parameter_free: false,
        router_input_scalar: 0.5, // non-unit → enters the scaling branch
        pre_experts_norm: &[],
        post_ffn1_norm: &[],
        post_experts_norm: &[],
        num_experts,
        top_k,
        intermediate_size: inter,
        activation: Activation::Silu,
        expert_data_format: larql_compute::QuantFormat::BF16,
    };
    let out = cpu_moe_forward(&h, &moe_scalar, 0.0, 1e-6);
    assert_eq!(out.len(), hidden);
}

#[test]
fn moe_empty_router_proj_returns_zeros() {
    let hidden = 8;
    let moe = MoeLayerWeights {
        experts_gate_up: Vec::new(),
        experts_down: Vec::new(),
        routing_policy: larql_compute::MoeRoutingPolicy::default(),
        weight_layout: larql_compute::MoeWeightLayout::default(),
        router_proj: &[], // empty → early return
        router_scale: &[],
        router_per_expert_scale: &[],
        router_norm: &[],
        router_norm_parameter_free: false,
        router_input_scalar: 1.0,
        pre_experts_norm: &[],
        post_ffn1_norm: &[],
        post_experts_norm: &[],
        num_experts: 4,
        top_k: 2,
        intermediate_size: 4,
        activation: Activation::Silu,
        expert_data_format: larql_compute::QuantFormat::BF16,
    };
    let h = vec![1.0f32; hidden];
    let out = cpu_moe_forward(&h, &moe, 0.0, 1e-6);
    assert_eq!(out.len(), hidden);
    assert!(
        out.iter().all(|v| *v == 0.0),
        "empty router_proj should produce all-zero output"
    );
}

#[test]
fn moe_zero_num_experts_returns_zeros() {
    // Exercises the num_experts == 0 early-return in forward.rs line 41.
    let hidden = 8;
    let moe = MoeLayerWeights {
        experts_gate_up: Vec::new(),
        experts_down: Vec::new(),
        routing_policy: larql_compute::MoeRoutingPolicy::default(),
        weight_layout: larql_compute::MoeWeightLayout::default(),
        router_proj: &[1.0f32], // non-empty so we don't hit that guard
        router_scale: &[],
        router_per_expert_scale: &[],
        router_norm: &[],
        router_norm_parameter_free: false,
        router_input_scalar: 1.0,
        pre_experts_norm: &[],
        post_ffn1_norm: &[],
        post_experts_norm: &[],
        num_experts: 0, // triggers the early return
        top_k: 2,
        intermediate_size: 4,
        activation: Activation::Silu,
        expert_data_format: larql_compute::QuantFormat::BF16,
    };
    let h = vec![1.0f32; hidden];
    let out = cpu_moe_forward(&h, &moe, 0.0, 1e-6);
    assert_eq!(out, vec![0.0f32; hidden]);
}

#[test]
fn moe_zero_top_k_or_intermediate_returns_zeros() {
    let hidden = 8;
    let router = vec![1.0f32; hidden * 2];
    let gate_up = bf16_fill(2 * 2 * hidden, 1.0);
    let down = bf16_fill(2 * hidden, 1.0);
    let (experts_gate_up, experts_down) = bf16_expert_tables(&gate_up, &down, 1, 2, hidden);
    let h = vec![1.0f32; hidden];

    let zero_top_k = MoeLayerWeights {
        experts_gate_up: experts_gate_up.clone(),
        experts_down: experts_down.clone(),
        routing_policy: larql_compute::MoeRoutingPolicy::default(),
        weight_layout: larql_compute::MoeWeightLayout::default(),
        router_proj: &router,
        router_scale: &[],
        router_per_expert_scale: &[],
        router_norm: &[],
        router_norm_parameter_free: false,
        router_input_scalar: 1.0,
        pre_experts_norm: &[],
        post_ffn1_norm: &[],
        post_experts_norm: &[],
        num_experts: 2,
        top_k: 0,
        intermediate_size: 2,
        activation: Activation::Silu,
        expert_data_format: larql_compute::QuantFormat::BF16,
    };
    assert_eq!(
        cpu_moe_forward(&h, &zero_top_k, 0.0, 1e-6),
        vec![0.0; hidden]
    );

    let zero_intermediate = MoeLayerWeights {
        top_k: 1,
        intermediate_size: 0,
        ..zero_top_k
    };
    assert_eq!(
        cpu_moe_forward(&h, &zero_intermediate, 0.0, 1e-6),
        vec![0.0; hidden]
    );
}

#[test]
fn moe_missing_selected_expert_tables_are_skipped() {
    let hidden = 8;
    let inter = 2;
    let num_experts = 4;
    let top_k = 1;
    let gate_up = bf16_fill(2 * inter * hidden, 1.0);
    let down = bf16_fill(hidden * inter, 1.0);
    let (experts_gate_up, experts_down) = bf16_expert_tables(&gate_up, &down, 1, inter, hidden);
    let mut router = vec![0.0f32; num_experts * hidden];
    for v in &mut router[3 * hidden..4 * hidden] {
        *v = 10.0;
    }
    let moe = MoeLayerWeights {
        experts_gate_up,
        experts_down,
        routing_policy: larql_compute::MoeRoutingPolicy::default(),
        weight_layout: larql_compute::MoeWeightLayout::default(),
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
        activation: Activation::Silu,
        expert_data_format: larql_compute::QuantFormat::BF16,
    };
    let h = vec![1.0f32; hidden];

    assert_eq!(cpu_moe_forward(&h, &moe, 0.0, 1e-6), vec![0.0; hidden]);
}

#[test]
fn moe_post_experts_norm_branch_runs() {
    let hidden = 8;
    let inter = 4;
    let num_experts = 2;
    let top_k = 1;
    let gate_up = bf16_fill(num_experts * 2 * inter * hidden, 1.0);
    let down = bf16_fill(num_experts * hidden * inter, 1.0);
    let router: Vec<f32> = (0..num_experts * hidden)
        .map(|i| if i < hidden { 1.0 } else { 0.0 })
        .collect();
    let post_norm = vec![1.0f32; hidden];
    let (experts_gate_up, experts_down) =
        bf16_expert_tables(&gate_up, &down, num_experts, inter, hidden);
    let moe = MoeLayerWeights {
        experts_gate_up,
        experts_down,
        routing_policy: larql_compute::MoeRoutingPolicy::default(),
        weight_layout: larql_compute::MoeWeightLayout::default(),
        router_proj: &router,
        router_scale: &[],
        router_per_expert_scale: &[],
        router_norm: &[],
        router_norm_parameter_free: false,
        router_input_scalar: 1.0,
        pre_experts_norm: &[],
        post_ffn1_norm: &[],
        post_experts_norm: &post_norm,
        num_experts,
        top_k,
        intermediate_size: inter,
        activation: Activation::Silu,
        expert_data_format: larql_compute::QuantFormat::BF16,
    };

    let out = cpu_moe_forward(&vec![1.0f32; hidden], &moe, 0.0, 1e-6);

    assert_eq!(out.len(), hidden);
    assert!(out.iter().all(|v| v.is_finite()));
}

#[test]
fn moe_gelu_tanh_activation_in_forward() {
    // Exercises the GeluTanh arm of the match in the rayon closure (forward.rs line 157).
    let hidden = 8;
    let inter = 4;
    let num_experts = 4;
    let top_k = 1;

    let gate_up = bf16_fill(num_experts * 2 * inter * hidden, 1.0);
    let down = bf16_fill(num_experts * hidden * inter, 1.0);
    let router: Vec<f32> = (0..num_experts * hidden)
        .map(|i| if i < hidden { 1.0 } else { 0.0 })
        .collect();
    let (experts_gate_up, experts_down) =
        bf16_expert_tables(&gate_up, &down, num_experts, inter, hidden);

    let moe = MoeLayerWeights {
        experts_gate_up,
        experts_down,
        routing_policy: larql_compute::MoeRoutingPolicy::default(),
        weight_layout: larql_compute::MoeWeightLayout::default(),
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
        activation: Activation::GeluTanh, // exercises the GeluTanh arm
        expert_data_format: larql_compute::QuantFormat::BF16,
    };
    let h = vec![1.0f32; hidden];
    let out = cpu_moe_forward(&h, &moe, 0.0, 1e-6);
    assert_eq!(out.len(), hidden);
    assert!(
        out.iter().any(|v| v.abs() > 1e-4),
        "GeluTanh forward should produce nonzero output"
    );
}

// ── Metal: prefill_kquant with MoE layers ────────────────────────────────────────
//
// Integration tests for the batched MoE prefill path introduced in
// 2026-04-26. They call through the public `DecodeBackend::prefill_kquant` API
// so they exercise the full `dispatch_full_pipeline` + `moe_fn` callback
// chain without reaching into private internals.

#[cfg(target_os = "macos")]
mod moe_prefill_integration {
    use larql_compute::backend::DecodeBackend;
    use larql_compute::pipeline::*;
    use larql_compute::MoeLayerWeights;
    use larql_compute_metal::MetalBackend;

    /// Minimal Q4_K weight buffer: one super-block (144 bytes) per row,
    /// all scales = 1.0 (f16 0x3C00), all nibbles = 0.
    fn synth_q4k(rows: usize, cols: usize) -> Vec<u8> {
        let blocks = cols.div_ceil(256);
        let mut v = vec![0u8; rows * blocks * 144];
        for b in 0..rows * blocks {
            v[b * 144 + 1] = 0x3C; // d = f16(1.0) hi byte
        }
        v
    }

    fn layer<'a>(
        q4k: &'a [u8],
        norm: &'a [f32],
        moe: Option<MoeLayerWeights<'a>>,
    ) -> FullPipelineLayer<'a> {
        let q4w = || QuantWeight {
            data: q4k,
            scales: None,
            format: QuantFormat::Q4_K,
        };
        FullPipelineLayer {
            wq: q4w(),
            wk: q4w(),
            wv: q4w(),
            wo: q4w(),
            gate: q4w(),
            up: q4w(),
            down: q4w(),
            input_norm: norm,
            post_attn_norm: norm,
            pre_ffn_norm: None,
            post_ffn_norm: None,
            input_norm_bias: None,
            post_attn_norm_bias: None,
            norm_offset: 1.0,
            qk_norm_offset: 0.0,
            eps: 1e-6,
            has_post_norms: false,
            norm_type: NormType::RmsNorm,
            ffn_type: FfnType::Gated,
            activation: Activation::Silu,
            attn_scale: 0.125,
            head_dim: 64,
            num_q_heads: 4,
            num_kv_heads: 4,
            rope_base: 10000.0,
            rotary_dim: 0,
            sliding_window: 0,
            has_v_norm: false,
            layer_scalar: 0.0,
            q_norm_weight: None,
            k_norm_weight: None,
            ffn_up_bias: None,
            ffn_down_bias: None,
            moe,
            ffn_is_remote: false,
            moe_combined_output_norm: false,
            moe_outer_post_norm: None,
            kv_shared_source: None,
            residual_multiplier: 1.0,
            ple_input_gate: None,
            ple_projection: None,
            ple_post_norm: None,
        }
    }

    fn null_moe(inter: usize) -> MoeLayerWeights<'static> {
        // num_experts=0 → cpu_moe_forward returns zeros immediately.
        // Sufficient to exercise the callback path without real expert weights.
        MoeLayerWeights {
            experts_gate_up: Vec::new(),
            experts_down: Vec::new(),
            routing_policy: larql_compute::MoeRoutingPolicy::default(),
            weight_layout: larql_compute::MoeWeightLayout::default(),
            router_proj: &[],
            router_scale: &[],
            router_per_expert_scale: &[],
            router_norm: &[],
            router_norm_parameter_free: false,
            router_input_scalar: 1.0,
            pre_experts_norm: &[],
            post_ffn1_norm: &[],
            post_experts_norm: &[],
            num_experts: 0,
            top_k: 1,
            intermediate_size: inter,
            activation: Activation::Silu,
            expert_data_format: larql_compute::QuantFormat::BF16,
        }
    }

    /// `prefill_kquant` on a model with MoE layers returns a vec of the right
    /// length and finite values. Exercises the batched-commit path end-to-end.
    #[test]
    fn prefill_q4_with_moe_returns_correct_shape() {
        let Some(metal) = MetalBackend::new() else {
            return;
        };
        let hidden = 256usize;
        let inter = 256usize;
        let seq_len = 3usize;
        let q4k = synth_q4k(hidden.max(inter), hidden);
        let norm = vec![1.0f32; hidden];
        let layers = vec![
            layer(&q4k, &norm, None),
            layer(&q4k, &norm, Some(null_moe(inter))),
            layer(&q4k, &norm, None),
        ];
        let x = vec![0.0f32; seq_len * hidden];
        let out = metal.prefill_kquant(&layers, &x, hidden, inter, seq_len, false, 0.0);
        let out = out.expect("prefill_kquant must return Some on Metal");
        assert_eq!(
            out.len(),
            seq_len * hidden,
            "output length must be seq_len × hidden"
        );
        assert!(
            out.iter().all(|v| v.is_finite()),
            "output must be finite (no NaN/Inf)"
        );
    }

    /// Variant of [`layer`] with V-norm enabled and learned QK-norm
    /// weights populated — drives the `has_v_norm` + `use_qk_norm`
    /// branches of `dispatch_full_pipeline` (lines 320-381 of
    /// `ops/full_pipeline/dispatch.rs`).
    fn layer_with_qk_v_norms<'a>(
        q4k: &'a [u8],
        norm: &'a [f32],
        head_dim_norm: &'a [f32],
    ) -> FullPipelineLayer<'a> {
        let mut base = layer(q4k, norm, None);
        base.has_v_norm = true;
        base.q_norm_weight = Some(head_dim_norm);
        base.k_norm_weight = Some(head_dim_norm);
        base
    }

    /// `prefill_kquant` with every layer carrying V-norm + learned QK-norm
    /// weights — exercises the prerope QK-norm + parameter-free V-norm
    /// dispatch branches in `ops/full_pipeline/dispatch.rs`.
    #[test]
    fn prefill_q4_with_qk_norm_and_v_norm_branches() {
        let Some(metal) = MetalBackend::new() else {
            return;
        };
        let hidden = 256usize;
        let inter = 256usize;
        let seq_len = 2usize;
        let head_dim = 64usize;
        let q4k = synth_q4k(hidden.max(inter), hidden);
        let norm = vec![1.0f32; hidden];
        let head_norm = vec![1.0f32; head_dim];
        let layers: Vec<_> = (0..3)
            .map(|_| layer_with_qk_v_norms(&q4k, &norm, &head_norm))
            .collect();
        let x = vec![0.01f32; seq_len * hidden];
        // `use_qk_norm = true` drives the `applied_prerope_qk_norm`
        // dispatch branch at `dispatch.rs:353`.
        let out = metal
            .prefill_kquant(&layers, &x, hidden, inter, seq_len, true, 0.0)
            .expect("prefill_kquant must return Some on Metal");
        assert_eq!(out.len(), seq_len * hidden);
        assert!(out.iter().all(|v| v.is_finite()));
    }

    /// Q4_KF QKV format — drives the fused Q4_KF QKV path
    /// (`stages.rs` lines 80-87) and the matching shader dispatch.
    /// Q4_KF is the llama.cpp-port pre-baked-scales format.
    #[test]
    fn prefill_q4_with_q4kf_qkv_format() {
        let Some(metal) = MetalBackend::new() else {
            return;
        };
        let hidden = 256usize;
        let inter = 256usize;
        let seq_len = 1usize;
        let q4k = synth_q4k(hidden.max(inter), hidden);
        let norm = vec![1.0f32; hidden];
        // Build a layer where Q/K/V are Q4_KF but gate/up/down stay Q4_K.
        let q4kf = QuantWeight {
            data: &q4k,
            scales: None,
            format: QuantFormat::Q4_KF,
        };
        let q4w = || QuantWeight {
            data: q4k.as_slice(),
            scales: None,
            format: QuantFormat::Q4_K,
        };
        let layers = vec![FullPipelineLayer {
            wq: q4kf,
            wk: q4kf,
            wv: q4kf,
            wo: q4w(),
            gate: q4w(),
            up: q4w(),
            down: q4w(),
            input_norm: &norm,
            post_attn_norm: &norm,
            pre_ffn_norm: None,
            post_ffn_norm: None,
            input_norm_bias: None,
            post_attn_norm_bias: None,
            norm_offset: 1.0,
            qk_norm_offset: 0.0,
            eps: 1e-6,
            has_post_norms: false,
            norm_type: NormType::RmsNorm,
            ffn_type: FfnType::Gated,
            activation: Activation::Silu,
            attn_scale: 0.125,
            head_dim: 64,
            num_q_heads: 4,
            num_kv_heads: 4,
            rope_base: 10000.0,
            rotary_dim: 0,
            sliding_window: 0,
            has_v_norm: false,
            layer_scalar: 0.0,
            q_norm_weight: None,
            k_norm_weight: None,
            ffn_up_bias: None,
            ffn_down_bias: None,
            moe: None,
            ffn_is_remote: false,
            moe_combined_output_norm: false,
            moe_outer_post_norm: None,
            kv_shared_source: None,
            residual_multiplier: 1.0,
            ple_input_gate: None,
            ple_projection: None,
            ple_post_norm: None,
        }];
        let x = vec![0.01f32; seq_len * hidden];
        let out = metal
            .prefill_kquant(&layers, &x, hidden, inter, seq_len, false, 0.0)
            .expect("prefill_kquant must return Some on Metal");
        assert_eq!(out.len(), seq_len * hidden);
        assert!(out.iter().all(|v| v.is_finite()));
    }

    /// Mixed QKV formats (Q4_K Q+K, Q6_K V — the Gemma 4 31B convention)
    /// drives the `all_same_format == false` fallback at
    /// `stages.rs` line 94 and the per-projection encode path
    /// (lines 142-180).
    #[test]
    fn prefill_q4_with_mixed_qkv_formats() {
        let Some(metal) = MetalBackend::new() else {
            return;
        };
        let hidden = 256usize;
        let inter = 256usize;
        let seq_len = 1usize;
        let q4k = synth_q4k(hidden.max(inter), hidden);
        let norm = vec![1.0f32; hidden];
        let q4_view = |fmt| QuantWeight {
            data: q4k.as_slice(),
            scales: None,
            format: fmt,
        };
        let layers = vec![FullPipelineLayer {
            wq: q4_view(QuantFormat::Q4_K),
            wk: q4_view(QuantFormat::Q4_K),
            wv: q4_view(QuantFormat::Q6_K),
            wo: q4_view(QuantFormat::Q4_K),
            gate: q4_view(QuantFormat::Q4_K),
            up: q4_view(QuantFormat::Q4_K),
            down: q4_view(QuantFormat::Q4_K),
            input_norm: &norm,
            post_attn_norm: &norm,
            pre_ffn_norm: None,
            post_ffn_norm: None,
            input_norm_bias: None,
            post_attn_norm_bias: None,
            norm_offset: 1.0,
            qk_norm_offset: 0.0,
            eps: 1e-6,
            has_post_norms: false,
            norm_type: NormType::RmsNorm,
            ffn_type: FfnType::Gated,
            activation: Activation::Silu,
            attn_scale: 0.125,
            head_dim: 64,
            num_q_heads: 4,
            num_kv_heads: 4,
            rope_base: 10000.0,
            rotary_dim: 0,
            sliding_window: 0,
            has_v_norm: false,
            layer_scalar: 0.0,
            q_norm_weight: None,
            k_norm_weight: None,
            ffn_up_bias: None,
            ffn_down_bias: None,
            moe: None,
            ffn_is_remote: false,
            moe_combined_output_norm: false,
            moe_outer_post_norm: None,
            kv_shared_source: None,
            residual_multiplier: 1.0,
            ple_input_gate: None,
            ple_projection: None,
            ple_post_norm: None,
        }];
        let x = vec![0.01f32; seq_len * hidden];
        let out = metal
            .prefill_kquant(&layers, &x, hidden, inter, seq_len, false, 0.0)
            .expect("mixed-format prefill returns Some");
        assert_eq!(out.len(), seq_len * hidden);
        assert!(out.iter().all(|v| v.is_finite()));
    }

    /// Q8_0 QKV format drives the fused-Q8-QKV branch in
    /// `ops/full_pipeline/stages.rs` lines 204-227 + the `q8_qkv_proj`
    /// shader dispatch.  Production Q8 attention path.
    #[test]
    fn prefill_q4_with_q8_0_qkv_drives_fused_q8_qkv_path() {
        let Some(metal) = MetalBackend::new() else {
            return;
        };
        let hidden = 256usize;
        let inter = 256usize;
        let seq_len = 1usize;
        let q4k = synth_q4k(hidden.max(inter), hidden);
        let norm = vec![1.0f32; hidden];
        let num_q_heads = 4usize;
        let num_kv_heads = 4usize;
        let head_dim = 64usize;
        let q_dim = num_q_heads * head_dim;
        let kv_dim = num_kv_heads * head_dim;
        // Q8_0 weights + per-row scales.
        let wq_q8: Vec<u8> = vec![1u8; q_dim * hidden];
        let wq_scales: Vec<f32> = vec![0.01f32; q_dim];
        let wk_q8: Vec<u8> = vec![1u8; kv_dim * hidden];
        let wk_scales: Vec<f32> = vec![0.01f32; kv_dim];
        let wv_q8: Vec<u8> = vec![1u8; kv_dim * hidden];
        let wv_scales: Vec<f32> = vec![0.01f32; kv_dim];

        let q4w = |fmt: QuantFormat| QuantWeight {
            data: q4k.as_slice(),
            scales: None,
            format: fmt,
        };
        let layers = vec![FullPipelineLayer {
            wq: QuantWeight {
                data: &wq_q8,
                scales: Some(&wq_scales),
                format: QuantFormat::Q8_0,
            },
            wk: QuantWeight {
                data: &wk_q8,
                scales: Some(&wk_scales),
                format: QuantFormat::Q8_0,
            },
            wv: QuantWeight {
                data: &wv_q8,
                scales: Some(&wv_scales),
                format: QuantFormat::Q8_0,
            },
            wo: q4w(QuantFormat::Q4_K),
            gate: q4w(QuantFormat::Q4_K),
            up: q4w(QuantFormat::Q4_K),
            down: q4w(QuantFormat::Q4_K),
            input_norm: &norm,
            post_attn_norm: &norm,
            pre_ffn_norm: None,
            post_ffn_norm: None,
            input_norm_bias: None,
            post_attn_norm_bias: None,
            norm_offset: 1.0,
            qk_norm_offset: 0.0,
            eps: 1e-6,
            has_post_norms: false,
            norm_type: NormType::RmsNorm,
            ffn_type: FfnType::Gated,
            activation: Activation::Silu,
            attn_scale: 0.125,
            head_dim,
            num_q_heads,
            num_kv_heads,
            rope_base: 10000.0,
            rotary_dim: 0,
            sliding_window: 0,
            has_v_norm: false,
            layer_scalar: 0.0,
            q_norm_weight: None,
            k_norm_weight: None,
            ffn_up_bias: None,
            ffn_down_bias: None,
            moe: None,
            ffn_is_remote: false,
            moe_combined_output_norm: false,
            moe_outer_post_norm: None,
            kv_shared_source: None,
            residual_multiplier: 1.0,
            ple_input_gate: None,
            ple_projection: None,
            ple_post_norm: None,
        }];
        let x = vec![0.01f32; seq_len * hidden];
        let out = metal
            .prefill_kquant(&layers, &x, hidden, inter, seq_len, false, 0.0)
            .expect("prefill_kquant must return Some on Metal");
        assert_eq!(out.len(), seq_len * hidden);
        assert!(out.iter().all(|v| v.is_finite()));
    }

    /// `LARQL_METAL_DUMP_LAYERS=<dir>` drives the dump helpers in
    /// `ops/full_pipeline/dump.rs` (`dump_h_embed`, `dump_layer0_q_after_stage`,
    /// `dump_layer_snapshots`).
    #[test]
    fn prefill_q4_with_metal_dump_layers_env_drives_dump_helpers() {
        let Some(metal) = MetalBackend::new() else {
            return;
        };
        let tmp = std::env::temp_dir().join("larql-cm-metal-dump-test");
        let _ = std::fs::create_dir_all(&tmp);
        let path = tmp.to_str().unwrap().to_string();
        let path_static: &'static str = Box::leak(path.into_boxed_str());
        let saved = std::env::var_os("LARQL_METAL_DUMP_LAYERS");
        // SAFETY: env vars are process-global; the make-target run is
        // single-threaded for this test's scope.  We restore at end.
        unsafe {
            std::env::set_var("LARQL_METAL_DUMP_LAYERS", path_static);
        }

        let hidden = 256usize;
        let inter = 256usize;
        let seq_len = 1usize;
        let q4k = synth_q4k(hidden.max(inter), hidden);
        let norm = vec![1.0f32; hidden];
        let layers = vec![layer(&q4k, &norm, None)];
        let x = vec![0.01f32; seq_len * hidden];
        let out = metal
            .prefill_kquant(&layers, &x, hidden, inter, seq_len, false, 0.0)
            .expect("prefill_kquant must return Some on Metal");
        assert_eq!(out.len(), seq_len * hidden);

        unsafe {
            match saved {
                Some(v) => std::env::set_var("LARQL_METAL_DUMP_LAYERS", v),
                None => std::env::remove_var("LARQL_METAL_DUMP_LAYERS"),
            }
        }
        let _ = std::fs::remove_dir_all(&tmp);
    }

    /// `prefill_kquant_with_head_replacement` exercises the
    /// `PipelineIntervention` hooks in `dispatch_full_pipeline`:
    /// `capture + zero target head` at hook A (dispatch.rs:455-495) and
    /// `replacement_delta` add at hook B (dispatch.rs:541-560).
    #[test]
    fn prefill_q4_with_head_replacement_drives_intervention_hooks() {
        let Some(metal) = MetalBackend::new() else {
            return;
        };
        let hidden = 256usize;
        let inter = 256usize;
        let seq_len = 2usize;
        let q4k = synth_q4k(hidden.max(inter), hidden);
        let norm = vec![1.0f32; hidden];
        let layers = vec![layer(&q4k, &norm, None), layer(&q4k, &norm, None)];
        let x = vec![0.01f32; seq_len * hidden];

        let replacement_delta = vec![0.1f32; seq_len * hidden];
        // Target layer 1, head 0 — exercises both hook A (capture +
        // zero) and hook B (delta add).
        let out = metal
            .prefill_kquant_with_head_replacement(
                &layers,
                &x,
                hidden,
                inter,
                seq_len,
                false,
                0.0,
                /* target_layer */ 1,
                /* target_head */ 0,
                &replacement_delta,
            )
            .expect("prefill_kquant_with_head_replacement returns Some on Metal");
        assert_eq!(out.len(), seq_len * hidden);
        assert!(out.iter().all(|v| v.is_finite()));
    }

    /// MoE + head replacement falls back to plain `prefill_kquant` since
    /// the intervention path doesn't support MoE layers (dispatch.rs
    /// `has_moe` early-out at line 473-477).
    #[test]
    fn prefill_q4_with_head_replacement_falls_back_when_moe_present() {
        let Some(metal) = MetalBackend::new() else {
            return;
        };
        let hidden = 256usize;
        let inter = 256usize;
        let seq_len = 1usize;
        let q4k = synth_q4k(hidden.max(inter), hidden);
        let norm = vec![1.0f32; hidden];
        let layers = vec![layer(&q4k, &norm, Some(null_moe(inter)))];
        let x = vec![0.0f32; seq_len * hidden];
        let delta = vec![0.0f32; seq_len * hidden];
        let out = metal
            .prefill_kquant_with_head_replacement(
                &layers, &x, hidden, inter, seq_len, false, 0.0, 0, 0, &delta,
            )
            .expect("MoE fallback still returns Some");
        assert_eq!(out.len(), seq_len * hidden);
    }

    /// `prefill_kquant` on an all-MoE model (every layer has MoE) uses the
    /// per-layer commit path. Result shape and finiteness are the minimum bar;
    /// the benchmark verifies correctness vs. the baseline.
    #[test]
    fn prefill_q4_all_moe_layers_returns_correct_shape() {
        let Some(metal) = MetalBackend::new() else {
            return;
        };
        let hidden = 256usize;
        let inter = 256usize;
        let seq_len = 4usize;
        let q4k = synth_q4k(hidden.max(inter), hidden);
        let norm = vec![1.0f32; hidden];
        let layers: Vec<_> = (0..4)
            .map(|_| layer(&q4k, &norm, Some(null_moe(inter))))
            .collect();
        let x = vec![0.0f32; seq_len * hidden];
        let out = metal
            .prefill_kquant(&layers, &x, hidden, inter, seq_len, false, 0.0)
            .expect("prefill_kquant must return Some on Metal");
        assert_eq!(out.len(), seq_len * hidden);
        assert!(out.iter().all(|v| v.is_finite()));
    }

    /// `prefill_kquant` without MoE (original path) is unaffected by the new
    /// callback infrastructure — same shape and finiteness contract.
    #[test]
    fn prefill_q4_no_moe_unaffected() {
        let Some(metal) = MetalBackend::new() else {
            return;
        };
        let hidden = 256usize;
        let inter = 256usize;
        let seq_len = 2usize;
        let q4k = synth_q4k(hidden.max(inter), hidden);
        let norm = vec![1.0f32; hidden];
        let layers = vec![layer(&q4k, &norm, None), layer(&q4k, &norm, None)];
        let x = vec![0.0f32; seq_len * hidden];
        let out = metal
            .prefill_kquant(&layers, &x, hidden, inter, seq_len, false, 0.0)
            .expect("prefill_kquant must return Some on Metal");
        assert_eq!(out.len(), seq_len * hidden);
        assert!(out.iter().all(|v| v.is_finite()));
    }
}
