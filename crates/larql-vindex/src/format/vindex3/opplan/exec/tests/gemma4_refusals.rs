//! V3-F0 witness 3, G4.2: the partial rotary EXECUTES on the CPU backends
//! and agrees across them — the reference transcribes HF, the production
//! backend goes through the served planners — and each basis is its own
//! rotation (the control that tells the two bases, and a plain rotary,
//! apart).

use larql_models::config::{ParameterFreeQkNorm, PositionPolicy, RotaryFrequencyBasis};

use super::lcg_values;
use crate::format::vindex3::graph::policy::AttentionSpan;
use crate::format::vindex3::opplan::exec::backend::{AttentionCall, PlanBackend, WeightSlice};
use crate::format::vindex3::opplan::exec::production::ProductionBackend;
use crate::format::vindex3::opplan::exec::reference::ReferenceBackend;

const HEAD_DIM: usize = 16;
const EPS: f64 = 1e-5;
const POSITIONS: usize = 8;
/// `lcg_values` is ±0.05; scale to O(1) so rotation differences are not
/// drowned by the attention aggregation.
const INPUT_GAIN: f32 = 20.0;
const ROTARY_FRACTION: f64 = 0.25;
/// A small base so every pair rotates by O(1) within eight positions —
/// the distinctness control must not hide behind angles of 1e-3 rad.
const THETA: f64 = 10.0;
/// Reference naive loops vs the served planner + the same rotate kernel:
/// f32 reassociation only.
const PARITY: f32 = 1e-5;
/// Two different rotations of the same inputs must differ by far more
/// than parity noise, relative to the output scale.
const DISTINCT: f32 = 1e-3;

fn call<'a>(inputs: &'a [Vec<f32>], w: &'a [f32], position: PositionPolicy) -> AttentionCall<'a> {
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
        score_scale: 1.0 / (HEAD_DIM as f64).sqrt(),
        logit_softcapping: None,
        position,
        span: AttentionSpan::Full,
        window: None,
        gate: None,
        bias: None,
        sinks: None,
    }
}

fn max_abs_diff(a: &[Vec<f32>], b: &[Vec<f32>]) -> f32 {
    a.iter()
        .zip(b)
        .flat_map(|(x, y)| x.iter().zip(y).map(|(p, q)| (p - q).abs()))
        .fold(0.0, f32::max)
}

fn max_abs(a: &[Vec<f32>]) -> f32 {
    a.iter().flatten().fold(0.0, |m, v| m.max(v.abs()))
}

/// Largest elementwise difference, relative to the larger output's scale.
fn relative_diff(a: &[Vec<f32>], b: &[Vec<f32>]) -> f32 {
    max_abs_diff(a, b) / max_abs(a).max(max_abs(b))
}

/// Both partial-rotary bases run on the reference and production
/// backends and agree; each differs from a plain rotary at the same base
/// and from the other basis — three distinct rotations, as HF defines
/// them.
#[test]
fn partial_rotary_bases_execute_at_parity_and_are_distinct_rotations() {
    let inputs: Vec<Vec<f32>> = (0..POSITIONS)
        .map(|p| {
            lcg_values(HEAD_DIM, p as u64 + 1)
                .into_iter()
                .map(|v| v * INPUT_GAIN)
                .collect()
        })
        .collect();
    let w = lcg_values(HEAD_DIM * HEAD_DIM, 7);
    let reference = ReferenceBackend::new();
    let production = ProductionBackend::new();
    let policies = [
        PositionPolicy::PartialRope {
            theta: THETA,
            rotary_fraction: ROTARY_FRACTION,
            basis: RotaryFrequencyBasis::HeadWidth,
        },
        PositionPolicy::PartialRope {
            theta: THETA,
            rotary_fraction: ROTARY_FRACTION,
            basis: RotaryFrequencyBasis::RotaryWidth,
        },
        PositionPolicy::Rope { theta: THETA },
    ];
    let outputs: Vec<Vec<Vec<f32>>> = policies
        .iter()
        .map(|&policy| {
            let r = reference
                .attention(call(&inputs, &w, policy))
                .unwrap_or_else(|e| panic!("reference {policy:?}: {e}"));
            let p = production
                .attention(call(&inputs, &w, policy))
                .unwrap_or_else(|e| panic!("production {policy:?}: {e}"));
            let diff = relative_diff(&r, &p);
            assert!(diff < PARITY, "{policy:?}: reference vs production {diff}");
            r
        })
        .collect();
    let (head_width, rotary_width, plain) = (&outputs[0], &outputs[1], &outputs[2]);
    assert!(
        relative_diff(head_width, plain) > DISTINCT,
        "head-width ≠ plain: {}",
        relative_diff(head_width, plain)
    );
    assert!(
        relative_diff(rotary_width, plain) > DISTINCT,
        "rotary-width ≠ plain: {}",
        relative_diff(rotary_width, plain)
    );
    assert!(
        relative_diff(head_width, rotary_width) > DISTINCT,
        "the two bases are different rotations: {}",
        relative_diff(head_width, rotary_width)
    );
}
