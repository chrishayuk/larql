//! Bias and gate-rule threading through the quantised expert paths.
//!
//! The scalar combine math is pinned against PyTorch in
//! `ffn::expert_weight::gate`; these tests pin the *plumbing* — that each
//! bias reaches its projection (gate vs up de-interleaved correctly, down
//! added exactly once) — which the math tests cannot see. Every expectation
//! is hand-derived from identity weights, so a swapped or dropped bias
//! changes the numbers.

use super::super::f32::run_single_expert;
use crate::{Activation, ExpertMlp, MoeGateRule};

const HIDDEN: usize = 2;
const INTER: usize = 2;

/// BF16 bytes for an identity-structured gate_up + down pair:
/// gate = up = down = I₂, so `gate_out == up_out == h` and the down matmul
/// passes the activation through unchanged.
fn identity_expert_bytes() -> (Vec<u8>, Vec<u8>) {
    let one = 1.0f32.to_bits() >> 16; // BF16 of 1.0
    let bf16 = |v: f32| -> [u8; 2] {
        let bits = (v.to_bits() >> 16) as u16;
        bits.to_le_bytes()
    };
    let _ = one;
    // gate_up: [2*INTER, HIDDEN] rows — gate rows first, then up rows.
    let mut gate_up = Vec::with_capacity(2 * INTER * HIDDEN * 2);
    for row in 0..2 * INTER {
        for col in 0..HIDDEN {
            let v = if row % INTER == col { 1.0 } else { 0.0 };
            gate_up.extend_from_slice(&bf16(v));
        }
    }
    // down: [HIDDEN, INTER] identity.
    let mut down = Vec::with_capacity(HIDDEN * INTER * 2);
    for row in 0..HIDDEN {
        for col in 0..INTER {
            let v = if row == col { 1.0 } else { 0.0 };
            down.extend_from_slice(&bf16(v));
        }
    }
    (gate_up, down)
}

fn silu(x: f32) -> f32 {
    x / (1.0 + (-x).exp())
}

/// Gate and up biases are stored fused-interleaved (`[g0, u0, g1, u1]`).
/// With identity weights the expected output is directly computable, and
/// the biases are chosen asymmetric so swapping gate/up (reading odd rows
/// as gate) or mis-striding the row changes the result.
#[test]
fn fused_bias_row_reaches_gate_and_up_de_interleaved() {
    let (gate_up, down) = identity_expert_bytes();
    let h = [1.0f32, 2.0];
    // gate biases: [0.5, -0.25]; up biases: [3.0, 7.0] — interleaved.
    let fused_bias = [0.5f32, 3.0, -0.25, 7.0];
    let mlp = ExpertMlp {
        rule: MoeGateRule::Gated(Activation::Silu),
        gate_up_bias: &fused_bias,
        down_bias: &[],
    };
    let out = run_single_expert(&h, &gate_up, &down, INTER, crate::QuantFormat::BF16, mlp);

    let expect = [
        silu(1.0 + 0.5) * (1.0 + 3.0),
        silu(2.0 - 0.25) * (2.0 + 7.0),
    ];
    for (o, e) in out.iter().zip(expect) {
        assert!((o - e).abs() < 1e-4, "got {out:?}, expected {expect:?}");
    }

    // Control: the same biases read under the WRONG convention (first half
    // gate, second half up — the §4.7 mixture bug, one tensor over) give a
    // different answer, so this test discriminates the layouts.
    let wrong = [
        silu(1.0 + 0.5) * (1.0 - 0.25),
        silu(2.0 + 3.0) * (2.0 + 7.0),
    ];
    assert!(
        (out[0] - wrong[0]).abs() > 1e-3 || (out[1] - wrong[1]).abs() > 1e-3,
        "fixture cannot distinguish interleaved from contiguous bias halves"
    );
}

/// The down bias is added once, after the down matmul, un-weighted by the
/// router (weighting happens in the caller). Identity weights make the
/// expectation exact.
#[test]
fn down_bias_is_added_after_the_down_matmul() {
    let (gate_up, down) = identity_expert_bytes();
    let h = [1.0f32, 2.0];
    let down_bias = [10.0f32, -20.0];
    let mlp = ExpertMlp {
        rule: MoeGateRule::Gated(Activation::Silu),
        gate_up_bias: &[],
        down_bias: &down_bias,
    };
    let out = run_single_expert(&h, &gate_up, &down, INTER, crate::QuantFormat::BF16, mlp);
    let expect = [silu(1.0) * 1.0 + 10.0, silu(2.0) * 2.0 - 20.0];
    for (o, e) in out.iter().zip(expect) {
        assert!((o - e).abs() < 1e-4, "got {out:?}, expected {expect:?}");
    }
}

/// The ClampedGlu rule flows through the quantised path: at `up = 0` the
/// GLU passes through where SwiGLU annihilates — the single most
/// consequential difference (`docs/k3-funnel.md` §4.7.1 findings 2-4).
#[test]
fn clamped_glu_rule_reaches_the_quantised_path() {
    let (gate_up, down) = identity_expert_bytes();
    // h = [2, 0]: under identity weights, gate_out = up_out = h.
    let h = [2.0f32, 0.0];
    let mlp = ExpertMlp {
        rule: MoeGateRule::ClampedGlu {
            limit: 7.0,
            alpha: 1.702,
        },
        gate_up_bias: &[],
        down_bias: &[],
    };
    let out = run_single_expert(&h, &gate_up, &down, INTER, crate::QuantFormat::BF16, mlp);

    let sigmoid = |x: f32| 1.0 / (1.0 + (-x).exp());
    // (u + 1) * g * sigmoid(g * alpha); element 1 has g = u = 0 → exactly 0.
    let expect0 = (2.0 + 1.0) * (2.0 * sigmoid(2.0 * 1.702));
    assert!(
        (out[0] - expect0).abs() < 1e-4,
        "got {}, want {expect0}",
        out[0]
    );
    assert_eq!(out[1], 0.0);

    // Discriminating control: plain SiLU·up gives (silu(2)·2, 0) — element 0
    // differs by more than any BF16 rounding could.
    let gated = run_single_expert(
        &h,
        &gate_up,
        &down,
        INTER,
        crate::QuantFormat::BF16,
        ExpertMlp::gated(Activation::Silu),
    );
    assert!((out[0] - gated[0]).abs() > 0.5);
}

/// `MoeLayerWeights::expert_mlp` hands each expert its OWN bias rows.
#[test]
fn expert_mlp_slices_by_expert_id() {
    let gate_up_bias = [
        1.0f32, 2.0, 3.0, 4.0, /* expert 1 */ 5.0, 6.0, 7.0, 8.0,
    ];
    let down_bias = [0.1f32, 0.2, /* expert 1 */ 0.3, 0.4];
    let moe = crate::MoeLayerWeights {
        expert_scales: crate::MoeExpertScales::Inline,
        fused_row_layout: crate::MoeFusedRowLayout::ContiguousHalves,
        experts_gate_up: Vec::new(),
        experts_down: Vec::new(),
        routing_policy: crate::MoeRoutingPolicy::top_k_then_softmax(),
        weight_layout: crate::MoeWeightLayout::default(),
        expert_data_format: crate::QuantFormat::BF16,
        router_proj: &[0.0; 4], // 2 experts × hidden 2
        router_bias: &[],
        experts_gate_up_bias: &gate_up_bias,
        experts_down_bias: &down_bias,
        router_scale: &[],
        router_per_expert_scale: &[],
        router_norm: &[],
        router_norm_parameter_free: false,
        router_input_scalar: 1.0,
        pre_experts_norm: &[],
        post_ffn1_norm: &[],
        post_experts_norm: &[],
        num_experts: 2,
        top_k: 1,
        intermediate_size: 2,
        gate_rule: MoeGateRule::Gated(Activation::Silu),
    };
    let e1 = moe.expert_mlp(1);
    assert_eq!(e1.gate_up_bias, &[5.0, 6.0, 7.0, 8.0]);
    assert_eq!(e1.down_bias, &[0.3, 0.4]);
    // De-interleave: gate takes even fused indices (j=1 → index 2), up odd.
    assert_eq!(e1.gate_bias(1), 7.0);
    assert_eq!(e1.up_bias(0), 6.0);
    assert_eq!(e1.up_bias(1), 8.0);
    // Empty tables answer 0.0 for every expert — the no-bias architectures.
    let none = crate::MoeLayerWeights {
        experts_gate_up_bias: &[],
        experts_down_bias: &[],
        ..moe
    };
    assert_eq!(none.expert_mlp(1).gate_bias(0), 0.0);
}

#[test]
#[should_panic(expected = "too short for expert")]
fn a_short_bias_table_panics_rather_than_reusing_expert_zero() {
    let gate_up_bias = [1.0f32; 4]; // one expert's worth
    let moe = crate::MoeLayerWeights {
        expert_scales: crate::MoeExpertScales::Inline,
        fused_row_layout: crate::MoeFusedRowLayout::ContiguousHalves,
        experts_gate_up: Vec::new(),
        experts_down: Vec::new(),
        routing_policy: crate::MoeRoutingPolicy::top_k_then_softmax(),
        weight_layout: crate::MoeWeightLayout::default(),
        expert_data_format: crate::QuantFormat::BF16,
        router_proj: &[0.0; 4],
        router_bias: &[],
        experts_gate_up_bias: &gate_up_bias,
        experts_down_bias: &[],
        router_scale: &[],
        router_per_expert_scale: &[],
        router_norm: &[],
        router_norm_parameter_free: false,
        router_input_scalar: 1.0,
        pre_experts_norm: &[],
        post_ffn1_norm: &[],
        post_experts_norm: &[],
        num_experts: 2,
        top_k: 1,
        intermediate_size: 2,
        gate_rule: MoeGateRule::Gated(Activation::Silu),
    };
    let _ = moe.expert_mlp(1);
}

/// The row-parallel expert variant computes the same numbers as the serial
/// form — same kernels, different schedule — pinned on a Q6_K fixture at a
/// block-multiple width so both take the integer path. Rows are
/// independent, so equality here is exact, not a tolerance.
#[test]
fn parallel_expert_variant_matches_serial_exactly() {
    use super::super::super::{
        quantize_h_norm_for_q4k, run_single_expert_kq_q8k_into,
        run_single_expert_kq_q8k_parallel_into, ExpertScratch,
    };
    use crate::cpu::ops::q4_common::quantize_q6_k;

    const H: usize = 256;
    const I: usize = 256;
    let fill = |n: usize, seed: u32| -> Vec<f32> {
        (0..n)
            .map(|i| {
                let h = (i as u32).wrapping_mul(2654435761).wrapping_add(seed);
                ((h >> 16) as f32 / 65536.0) - 0.5
            })
            .collect()
    };
    let mut gu = fill(I * H, 1);
    gu.extend(fill(I * H, 2));
    let gu_q = quantize_q6_k(&gu);
    let dn_q = quantize_q6_k(&fill(H * I, 3));
    let h = fill(H, 9);
    let q8k = quantize_h_norm_for_q4k(&h).expect("block-multiple width");
    let mlp = ExpertMlp {
        rule: MoeGateRule::ClampedGlu {
            limit: 7.0,
            alpha: 1.702,
        },
        gate_up_bias: &[],
        down_bias: &[],
    };

    let mut s1 = ExpertScratch::new(H, I, I);
    let serial = run_single_expert_kq_q8k_into(
        &mut s1,
        &q8k,
        &gu_q,
        &dn_q,
        I,
        crate::QuantFormat::Q6_K,
        mlp,
    )
    .to_vec();
    let mut s2 = ExpertScratch::new(H, I, I);
    let parallel = run_single_expert_kq_q8k_parallel_into(
        &mut s2,
        &q8k,
        &gu_q,
        &dn_q,
        I,
        crate::QuantFormat::Q6_K,
        mlp,
    )
    .to_vec();
    assert_eq!(serial, parallel, "schedules must not change the numbers");
    assert!(serial.iter().any(|v| v.abs() > 1e-4));
}

/// The parallel variant refuses a short gate+up slab by name, like the
/// serial one; it used to hand back a zeroed expert output.
#[test]
#[should_panic(expected = "run_single_expert_kq_q8k_parallel_into: refused operands")]
fn parallel_expert_variant_refuses_a_short_slab() {
    use super::super::super::{
        quantize_h_norm_for_q4k, run_single_expert_kq_q8k_parallel_into, ExpertScratch,
    };
    use crate::cpu::ops::q4_common::quantize_q6_k;

    const H: usize = 256;
    const I: usize = 256;
    let gu_q = quantize_q6_k(&vec![0.25f32; 2 * I * H]);
    let dn_q = quantize_q6_k(&vec![0.25f32; H * I]);
    let q8k = quantize_h_norm_for_q4k(&vec![1.0f32; H]).expect("block-multiple width");
    let mlp = ExpertMlp {
        rule: MoeGateRule::ClampedGlu {
            limit: 7.0,
            alpha: 1.702,
        },
        gate_up_bias: &[],
        down_bias: &[],
    };
    let mut s3 = ExpertScratch::new(H, I, I);
    let _ = run_single_expert_kq_q8k_parallel_into(
        &mut s3,
        &q8k,
        &gu_q[..10],
        &dn_q,
        I,
        crate::QuantFormat::Q6_K,
        mlp,
    );
}
