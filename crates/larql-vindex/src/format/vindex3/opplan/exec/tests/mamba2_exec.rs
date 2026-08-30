//! Unit gates for the Mamba2 reference operator: the recurrence checked
//! against hand arithmetic at degenerate sizes, the dt clamp against its
//! declared bounds, and the convolution's causality by impulse.
//!
//! The end-to-end witness (`opplan/tests/mamba2.rs`) proves the operator
//! composes through the generic path; these pin the pieces a composed
//! run could get wrong in compensating pairs.

use larql_models::config::{Activation, DtBound, Mamba2Geometry};

use crate::format::vindex3::opplan::exec::continuation::RecurrentState;
use crate::format::vindex3::opplan::exec::gated_delta::ScalarProjections;
use crate::format::vindex3::opplan::exec::mamba2::{
    layer_forward_with, state_geometry, Mamba2Weights, CONV_HISTORY, SSM_STATE,
};
use crate::format::vindex3::opplan::{Mamba2Op, OperandRef};

/// A degenerate mixer where every stage is hand-computable: one head,
/// head_dim 1, state 1, one group, kernel 1 (the conv is a 1-tap scale),
/// hidden 1 — so in_proj emits [z, xBC(3), dt] = 5 rows.
fn tiny_op(kernel: usize) -> Mamba2Op {
    let geometry = Mamba2Geometry::read(&serde_json::json!({
        "state_size": 1, "num_heads": 1, "head_dim": 1, "expand": 1,
        "conv_kernel": kernel, "n_groups": 1, "chunk_size": 4,
        "time_step_limit": [0.0, "Infinity"],
        "rms_norm": false, "use_bias": false, "use_conv_bias": false
    }))
    .unwrap();
    let operand = |name: &str, shape: Vec<usize>| OperandRef {
        object: "t.decoder_stack".into(),
        tensor: name.into(),
        dtype: "F32".into(),
        shape,
    };
    Mamba2Op {
        geometry,
        activation: Activation::Silu,
        residual_in_fp32: None,
        in_proj: operand("0.mixer.in_proj.weight", vec![5, 1]),
        conv1d: operand("0.mixer.conv1d.weight", vec![3, 1, kernel]),
        conv1d_bias: None,
        a_log: operand("0.mixer.A_log", vec![1]),
        d: operand("0.mixer.D", vec![1]),
        dt_bias: operand("0.mixer.dt_bias", vec![1]),
        gated_norm: None,
        out_proj: operand("0.mixer.out_proj.weight", vec![1, 1]),
    }
}

fn silu(x: f32) -> f32 {
    x / (1.0 + (-x).exp())
}

fn softplus(x: f32) -> f32 {
    (1.0 + x.exp()).ln()
}

fn state_for(op: &Mamba2Op) -> RecurrentState {
    RecurrentState::zeros(&state_geometry(op))
}

/// **The recurrence, against hand arithmetic.** One head, one channel,
/// identity projections and a 1-tap conv, so every stage collapses to a
/// scalar chain that a reader can recompute on paper:
///
/// ```text
/// z = x = b = c = raw dt = u          (identity in_proj rows)
/// conv(x) = silu(u)                   (1-tap conv, tap 1.0)
/// dt = softplus(u + 0)
/// s' = s·exp(dt·(−e^0)) + dt·b·x      (A_log 0 ⇒ A = −1)
/// y  = s'·c + D·x = s'·c              (D = 0)
/// out = y · silu(z)                   (no norm; out_proj = 1)
/// ```
#[test]
fn the_tiny_recurrence_matches_hand_arithmetic() {
    let op = tiny_op(1);
    let in_proj = vec![1.0f32; 5]; // every row projects hidden [u] to u
    let out_proj = vec![1.0f32];
    let conv1d = vec![1.0f32, 1.0, 1.0]; // 3 channels × 1 tap
    let (a_log, d, dt_bias) = (vec![0.0f32], vec![0.0f32], vec![0.0f32]);
    let w = Mamba2Weights {
        in_proj: crate::format::vindex3::opplan::exec::cpu::WeightRows::F32(&in_proj),
        out_proj: crate::format::vindex3::opplan::exec::cpu::WeightRows::F32(&out_proj),
        conv1d: &conv1d,
        conv1d_bias: None,
        a_log: &a_log,
        d: &d,
        dt_bias: &dt_bias,
        norm: None,
        norm_eps: 0.0,
    };
    let mut state = state_for(&op);
    let inputs = vec![vec![0.7f32], vec![-0.3f32]];
    let planes = layer_forward_with(&op, &w, &inputs, &mut state, &ScalarProjections);

    let mut s = 0.0f32;
    for (t, u) in [0.7f32, -0.3].into_iter().enumerate() {
        let xbc = silu(u); // x, b and c share the value: identity rows + 1-tap conv
        let dt = softplus(u);
        s = s * (-dt).exp() + dt * xbc * xbc; // A = −exp(0) = −1
        let y = s * xbc; // + D·x with D = 0
        let expected = y * silu(u); // gate z = u; no norm; out_proj 1
        assert!(
            (planes.output[t][0] - expected).abs() < 1e-6,
            "position {t}: {} vs hand {expected}",
            planes.output[t][0]
        );
    }
    // The advanced state is the hand-computed one, exactly.
    assert!((state.buffer(SSM_STATE).cells()[0] - s).abs() < 1e-6);
}

/// **The dt clamp honours its declared bounds — and an unbounded side
/// clamps nothing.** A finite ceiling caps the discretised timestep;
/// the released `[0.0, Infinity]` declaration leaves softplus alone.
#[test]
fn the_dt_clamp_honours_declared_bounds() {
    let mut capped = tiny_op(1);
    capped.geometry.dt_limit_max = DtBound::Finite(0.25);
    let uncapped = tiny_op(1);

    let in_proj = vec![1.0f32; 5];
    let out_proj = vec![1.0f32];
    let conv1d = vec![1.0f32, 1.0, 1.0];
    let (a_log, d, dt_bias) = (vec![0.0f32], vec![0.0f32], vec![0.0f32]);
    let w = |_: ()| Mamba2Weights {
        in_proj: crate::format::vindex3::opplan::exec::cpu::WeightRows::F32(&in_proj),
        out_proj: crate::format::vindex3::opplan::exec::cpu::WeightRows::F32(&out_proj),
        conv1d: &conv1d,
        conv1d_bias: None,
        a_log: &a_log,
        d: &d,
        dt_bias: &dt_bias,
        norm: None,
        norm_eps: 0.0,
    };
    // A large input drives softplus(dt) ≈ 3.0 — far above the cap.
    let inputs = vec![vec![3.0f32]];
    let mut s1 = state_for(&capped);
    let out_capped = layer_forward_with(&capped, &w(()), &inputs, &mut s1, &ScalarProjections);
    let mut s2 = state_for(&uncapped);
    let out_uncapped = layer_forward_with(&uncapped, &w(()), &inputs, &mut s2, &ScalarProjections);
    assert_ne!(
        out_capped.output[0][0], out_uncapped.output[0][0],
        "a finite ceiling must change the computation"
    );
    // Hand value under the cap: dt = 0.25 exactly.
    let xbc = silu(3.0f32);
    let dt = 0.25f32;
    let s = dt * xbc * xbc * (0.0f32); // decay applies to zero state
    let s = s + dt * xbc * xbc;
    let expected = s * xbc * silu(3.0f32);
    assert!((out_capped.output[0][0] - expected).abs() < 1e-6);
}

/// **The convolution is causal.** With a 2-tap kernel, an impulse at
/// position 1 must not reach position 0's output — and position 1 must
/// see position 0 through the window.
#[test]
fn the_convolution_is_causal_by_impulse() {
    let op = tiny_op(2);
    let in_proj = vec![1.0f32; 5];
    let out_proj = vec![1.0f32];
    // taps [older, current] per channel — weight the OLDER tap so any
    // future leak would be loud.
    let conv1d = vec![0.5f32, 1.0, 0.5, 1.0, 0.5, 1.0];
    let (a_log, d, dt_bias) = (vec![0.0f32], vec![0.0f32], vec![0.0f32]);
    let w = Mamba2Weights {
        in_proj: crate::format::vindex3::opplan::exec::cpu::WeightRows::F32(&in_proj),
        out_proj: crate::format::vindex3::opplan::exec::cpu::WeightRows::F32(&out_proj),
        conv1d: &conv1d,
        conv1d_bias: None,
        a_log: &a_log,
        d: &d,
        dt_bias: &dt_bias,
        norm: None,
        norm_eps: 0.0,
    };
    // Baseline: [u, 0]. Perturbed: [u, IMPULSE]. Position 0's output
    // must be bit-identical — nothing may see its own future.
    let mut s_base = state_for(&op);
    let base = layer_forward_with(
        &op,
        &w,
        &[vec![0.9f32], vec![0.0f32]],
        &mut s_base,
        &ScalarProjections,
    );
    let mut s_pert = state_for(&op);
    let pert = layer_forward_with(
        &op,
        &w,
        &[vec![0.9f32], vec![5.0f32]],
        &mut s_pert,
        &ScalarProjections,
    );
    assert_eq!(base.output[0][0], pert.output[0][0], "no future leak");
    assert_ne!(base.output[1][0], pert.output[1][0]);
    // And the durable history rolled: the newest slot holds the newest
    // PRE-convolution input (x channel of position 1 = 5.0 through the
    // identity projection).
    let history = s_pert.buffer(CONV_HISTORY);
    assert_eq!(history.cells()[1], 5.0, "channel 0, newest slot");
}
