//! The linear rope arm EXECUTES through both backends, not just the
//! kernel underneath it.
//!
//! `linear_frequencies` is a one-line transcription; what this gates is
//! whether `ReferenceBackend::attention` and `ProductionBackend::attention`
//! reach it with the right head width, rotate BOTH `q` and `k`, apply no
//! amplitude, and — the control that makes the pair a gate — rotate
//! something other than a plain rope. An arm that ignored the factor and
//! rotated plain would pass parity perfectly, both backends being wrong
//! together; the distinctness test below is what stops it. Same shape as
//! `llama3_rope`.
//!
//! Gemma 3's own numbers: global base 1e6, `factor: 8.0`. Every frequency
//! is divided, so unlike Llama-3's band scaling the effect is large at
//! any sequence length; 64 positions separate the two rotations by far
//! more than the reference-vs-production reassociation floor.

use larql_models::config::{ParameterFreeQkNorm, PositionPolicy};

use super::lcg_values;
use crate::format::vindex3::graph::policy::AttentionSpan;
use crate::format::vindex3::opplan::exec::backend::{AttentionCall, PlanBackend, WeightSlice};
use crate::format::vindex3::opplan::exec::production::ProductionBackend;
use crate::format::vindex3::opplan::exec::reference::ReferenceBackend;

const HEAD_DIM: usize = 64;
/// Gemma 3's global-layer base.
const THETA: f64 = 1_000_000.0;
/// Gemma 3's declared `rope_scaling.factor`.
const FACTOR: f64 = 8.0;
/// A divisor of one is the plain rotary by definition.
const UNIT_FACTOR: f64 = 1.0;

const EPS: f64 = 1e-5;
const POSITIONS: usize = 64;
/// `lcg_values` is ±0.05; scale to O(1) so rotation differences are not
/// drowned by the attention aggregation.
const INPUT_GAIN: f32 = 20.0;
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

/// Run one policy on both backends, require they agree, and return the
/// reference output for the caller's own comparisons.
fn agreed(position: PositionPolicy) -> Vec<Vec<f32>> {
    let inputs: Vec<Vec<f32>> = (0..POSITIONS)
        .map(|p| {
            lcg_values(HEAD_DIM, p as u64 + 1)
                .into_iter()
                .map(|v| v * INPUT_GAIN)
                .collect()
        })
        .collect();
    let w = lcg_values(HEAD_DIM * HEAD_DIM, 7);
    let reference = ReferenceBackend::new()
        .attention(call(&inputs, &w, position))
        .unwrap_or_else(|e| panic!("reference {position:?}: {e}"))
        .outputs;
    let production = ProductionBackend::new()
        .attention(call(&inputs, &w, position))
        .unwrap_or_else(|e| panic!("production {position:?}: {e}"))
        .outputs;
    let diff = relative_diff(&reference, &production);
    assert!(
        diff < PARITY,
        "{position:?}: reference vs production {diff}"
    );
    reference
}

/// The reference transcription and the served planner's position
/// divisor are two implementations of the same block; through a whole
/// attention call at Gemma 3's own numbers they must not disagree.
#[test]
fn the_gemma3_global_geometry_executes_at_parity() {
    agreed(PositionPolicy::Linear {
        theta: THETA,
        factor: FACTOR,
    });
}

/// The control: the factor reaches the rotation. Without this, the
/// parity test passes for an arm that dropped the divisor on both sides.
#[test]
fn the_factor_reaches_the_rotation_and_is_not_a_plain_rope() {
    let scaled = agreed(PositionPolicy::Linear {
        theta: THETA,
        factor: FACTOR,
    });
    let plain = agreed(PositionPolicy::Rope { theta: THETA });
    let diff = relative_diff(&scaled, &plain);
    assert!(
        diff > DISTINCT,
        "linear scaling did not move the rotation: {diff}"
    );
}

/// The other bound: a unit divisor IS the plain rotary. Pins that the
/// arm scales positions and nothing else — an amplitude or a band ramp
/// smuggled into it would separate the two here.
#[test]
fn a_unit_factor_is_the_plain_rope() {
    let unit = agreed(PositionPolicy::Linear {
        theta: THETA,
        factor: UNIT_FACTOR,
    });
    let plain = agreed(PositionPolicy::Rope { theta: THETA });
    let diff = relative_diff(&unit, &plain);
    assert!(diff < PARITY, "a unit factor changed the rotation: {diff}");
}
