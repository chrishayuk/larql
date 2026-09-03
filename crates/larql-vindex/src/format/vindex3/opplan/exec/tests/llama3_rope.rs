//! The Llama-3 rope arm EXECUTES through the interpreter, not just the
//! kernel underneath it.
//!
//! `kernels::llama3_tests` already holds `llama3_frequencies` to the
//! served planner's `apply_llama3_inv_freq`. That proves the frequency
//! table. It says nothing about whether `ReferenceBackend::attention`
//! ever reaches the table: the arm could pass the wrong `head_dim`,
//! rotate only `q`, or apply an amplitude Llama-3 does not specify, and
//! every kernel test would still be green.
//!
//! So this drives the two independent backends through a whole attention
//! call and requires them to agree — the same shape `gemma4_refusals`
//! uses for the partial-rotary bases.
//!
//! # The sequence length is load-bearing
//!
//! Llama-3.2's block only divides the bands whose wavelength exceeds
//! `original / low_freq_factor` = 8192, and those rotate slowly enough
//! that over a short sequence a scaled rope and a plain one are
//! indistinguishable. This was measured, not assumed: at the eight
//! positions the neighbouring fixtures use, an arm mutated to ignore the
//! block and rotate plain still cleared the parity assertion below. A
//! fixture built that way would pass for exactly the behaviour
//! `PositionPolicy::Llama3` exists to stop.
//!
//! Separation against sequence length, same inputs, measured on this
//! geometry — and against it the reference-vs-production noise floor,
//! which is flat in the sequence length because it is f32 reassociation
//! and nothing else:
//!
//! | positions | llama3 vs plain rope | reference vs production |
//! | --------- | -------------------- | ----------------------- |
//! | 128       | 1.0e-4               | 1.7e-7                  |
//! | 512       | 3.6e-4               | 1.7e-7                  |
//! | 2048      | 1.3e-3               | 1.7e-7                  |
//!
//! Hence 512 and a threshold of 1e-5: roughly the geometric mean of the
//! two, 36x below the effect it must see and 57x above the noise it must
//! not trip on. The effect is far smaller than the 0.25 rad gap the two
//! frequency tables differ by at this length because the softmax
//! aggregation dilutes it.

use larql_models::config::{ParameterFreeQkNorm, PositionPolicy};

use super::lcg_values;
use crate::format::vindex3::graph::policy::AttentionSpan;
use crate::format::vindex3::opplan::exec::backend::{AttentionCall, PlanBackend, WeightSlice};
use crate::format::vindex3::opplan::exec::production::ProductionBackend;
use crate::format::vindex3::opplan::exec::reference::ReferenceBackend;

/// Llama-3.2-1B's own head width, base and scaling block. At this
/// geometry all three bands are populated — 15 half-pairs left alone, 3
/// in the blend, 14 divided — so one fixture runs every branch of
/// `llama3_frequencies`.
const HEAD_DIM: usize = 64;
const THETA: f64 = 500000.0;

fn scaling() -> larql_models::Llama3RopeScaling {
    larql_models::Llama3RopeScaling {
        factor: 32.0,
        low_freq_factor: 1.0,
        high_freq_factor: 4.0,
        original_max_position_embeddings: 8192.0,
    }
}

const EPS: f64 = 1e-5;
/// Long enough for the divided bands to separate from an unscaled rope.
/// See the module header — at eight this whole file proves nothing.
const POSITIONS: usize = 512;
/// `lcg_values` is ±0.05; scale to O(1) so rotation differences are not
/// drowned by the attention aggregation.
const INPUT_GAIN: f32 = 20.0;
/// Reference naive loops vs the served planner + the same rotate kernel:
/// f32 reassociation only. Measured at 1.7e-7.
const PARITY: f32 = 1e-5;
/// Two different rotations of the same inputs must differ by far more
/// than parity noise, relative to the output scale. Measured at 3.6e-4.
const DISTINCT: f32 = 1e-5;

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

/// The reference transcription and the served planner are two
/// independent implementations of the same block, and the reference is
/// only worth having if it can disagree. Through a whole attention call,
/// on the geometry Llama-3.2-1B actually declares, it must not.
#[test]
fn the_shipped_llama3_geometry_executes_at_parity() {
    agreed(PositionPolicy::Llama3 {
        theta: THETA,
        scaling: scaling(),
    });
}

/// The control. Without it the parity test above passes for an arm that
/// ignored the block and rotated plain — both backends would be wrong
/// together and agree perfectly. This is what makes the pair a gate
/// rather than a smoke test.
#[test]
fn the_scaling_reaches_the_rotation_and_is_not_a_plain_rope() {
    let scaled = agreed(PositionPolicy::Llama3 {
        theta: THETA,
        scaling: scaling(),
    });
    let plain = agreed(PositionPolicy::Rope { theta: THETA });
    let diff = relative_diff(&scaled, &plain);
    assert!(
        diff > DISTINCT,
        "llama3 scaling did not move the rotation: {diff}"
    );
}
