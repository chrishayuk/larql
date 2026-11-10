//! Production-backend arms the fixtures never select: the SiTU-GLU and
//! clamped-gated FFN policies, the ungated tanh-GELU FFN, and the two
//! position policies that are represented but not executable.

use crate::format::vindex3::graph::policy::AttentionSpan;
use crate::format::vindex3::opplan::exec::backend::{
    AttentionCall, FfnCall, PlanBackend, WeightSlice,
};
use crate::format::vindex3::opplan::exec::production::{condition_qk_in_place, ProductionBackend};
use larql_models::config::{Activation, ParameterFreeQkNorm, PositionPolicy, RotaryFrequencyBasis};
use larql_models::ExpertGatePolicy;

/// Every width in the tiny FFN: identity-shaped so the arithmetic is
/// checkable by hand.
const WIDTH: usize = 2;
const IDENTITY: [f32; WIDTH * WIDTH] = [1.0, 0.0, 0.0, 1.0];
const X: [f32; WIDTH] = [0.5, -1.5];
/// SiTU-GLU's declared sharpness, and the clamp a clamped FFN declares.
const SITU_BETA: f32 = 1.0;
const CLAMP_LIMIT: f32 = 10.0;
/// One attention head for the position-policy probes.
const HEAD_DIM: usize = 4;
const THETA: f64 = 10_000.0;
const FULL_ROTARY: f64 = 1.0;
const MROPE_SECTION: [usize; 3] = [1, 1, 0];
const RELATIVE_D_REL: usize = 4;
const RELATIVE_EXTENT: usize = 16;
const EPS: f64 = 1e-6;

fn ffn(gate: bool, activation: Activation, policy: ExpertGatePolicy) -> FfnCall<'static> {
    FfnCall {
        x: &X,
        hidden: WIDTH,
        intermediate: WIDTH,
        gate: gate.then_some(WeightSlice::F32(&IDENTITY)),
        up: WeightSlice::F32(&IDENTITY),
        down: WeightSlice::F32(&IDENTITY),
        activation,
        gate_policy: policy,
    }
}

/// SiTU-GLU owns the whole combine: it runs on a gated FFN, through the
/// shared `MoeGateRule` authority, and refuses an FFN with no gate
/// rather than computing on the up branch alone.
#[test]
fn situ_glu_combines_a_gated_ffn_and_refuses_an_ungated_one() {
    let backend = ProductionBackend::new();
    let policy = ExpertGatePolicy::SituGlu {
        beta: SITU_BETA,
        linear_beta: None,
    };
    let rule = larql_compute::MoeGateRule::SituGlu {
        beta: SITU_BETA,
        linear_beta: None,
    };
    let out = backend.ffn(ffn(true, Activation::Silu, policy)).unwrap();
    // Identity projections: the combine IS the output.
    let want: Vec<f32> = X.iter().map(|v| rule.combine(*v, *v)).collect();
    assert_eq!(out, want);

    let err = backend
        .ffn(ffn(false, Activation::Silu, policy))
        .unwrap_err()
        .to_string();
    assert!(err.contains("SiTU-GLU") && err.contains("no gate"), "{err}");
}

/// A clamped-gated FFN is refused by name: plain gating would drop a
/// clamp that is one-sided on the gate and symmetric on the up branch.
#[test]
fn a_clamped_gated_ffn_is_refused_by_name() {
    let err = ProductionBackend::new()
        .ffn(ffn(
            true,
            Activation::Silu,
            ExpertGatePolicy::ClampedGated { limit: CLAMP_LIMIT },
        ))
        .unwrap_err()
        .to_string();
    assert!(
        err.contains("ClampedGated") && err.contains("production"),
        "{err}"
    );
}

/// The ungated tanh-GELU FFN is `gelu_tanh(up)` through the down
/// projection — here the identity.
#[test]
fn an_ungated_tanh_gelu_ffn_applies_the_activation_to_the_up_branch() {
    let out = ProductionBackend::new()
        .ffn(ffn(false, Activation::GeluTanh, ExpertGatePolicy::Gated))
        .unwrap();
    let want: Vec<f32> = X
        .iter()
        .map(|v| larql_compute::ffn::gelu_tanh(*v))
        .collect();
    for (got, want) in out.iter().zip(&want) {
        assert!((got - want).abs() <= f32::EPSILON, "{got} vs {want}");
    }
}

fn attention<'a>(
    inputs: &'a [Vec<f32>],
    w: &'a [f32],
    position: PositionPolicy,
) -> AttentionCall<'a> {
    AttentionCall {
        inputs,
        hidden: HEAD_DIM,
        num_q_heads: 1,
        num_kv_heads: 1,
        head_dim: HEAD_DIM,
        w_q: WeightSlice::F32(w),
        w_k: WeightSlice::F32(w),
        w_v: WeightSlice::F32(w),
        w_o: WeightSlice::F32(w),
        qk_norm: None,
        parameter_free_qk_norm: ParameterFreeQkNorm {
            q: false,
            k: false,
            v: false,
        },
        qk_norm_eps: EPS,
        query_scale: None,
        score_scale: 1.0,
        logit_softcapping: None,
        position,
        span: AttentionSpan::Full,
        window: None,
        gate: None,
        bias: None,
        sinks: None,
        rope_angles: Default::default(),
    }
}

/// A position policy that is represented but has no judged rotation is
/// refused at conditioning time, never skipped — skipping would run the
/// model with no position information at all.
#[test]
fn unexecutable_position_policies_are_refused_at_conditioning() {
    static W: [f32; HEAD_DIM * HEAD_DIM] = [0.0; HEAD_DIM * HEAD_DIM];
    let inputs = vec![vec![0.0f32; HEAD_DIM]];
    let cases = [
        (
            PositionPolicy::Relative {
                d_rel: RELATIVE_D_REL,
                extent: RELATIVE_EXTENT,
            },
            "relative position",
        ),
        (
            PositionPolicy::MRope {
                theta: THETA,
                rotary_fraction: FULL_ROTARY,
                basis: RotaryFrequencyBasis::HeadWidth,
                section: MROPE_SECTION,
                interleaved: false,
            },
            "head-width frequency basis",
        ),
    ];
    for (position, reason) in cases {
        let call = attention(&inputs, &W, position);
        let (mut q, mut k) = (vec![1.0f32; HEAD_DIM], vec![1.0f32; HEAD_DIM]);
        let err = condition_qk_in_place(&call, 0, &mut q, &mut k)
            .unwrap_err()
            .to_string();
        assert!(err.contains(reason), "{err}");
    }
}
