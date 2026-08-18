//! V3-F0 witness 3, G4.0: the semantics Gemma 4 declares are CARRIED by
//! the container and REFUSED by every executor until G4.2 executes them.
//! Each refusal is typed, names the rung, and fires before any bytes
//! move — running the plain path would be a different model.

use larql_models::config::{ParameterFreeQkNorm, PositionPolicy, RotaryFrequencyBasis};

use super::lcg_values;
use crate::format::vindex3::graph::policy::AttentionSpan;
use crate::format::vindex3::opplan::exec::backend::{AttentionCall, PlanBackend, WeightSlice};
use crate::format::vindex3::opplan::exec::production::ProductionBackend;
use crate::format::vindex3::opplan::exec::reference::ReferenceBackend;

const HEAD_DIM: usize = 8;
const EPS: f64 = 1e-5;
const POSITIONS: usize = 2;
const ROTARY_FRACTION: f64 = 0.25;
const THETA: f64 = 1_000_000.0;
/// The rung the refusal must name.
const RUNG: &str = "G4.2";

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

/// A proportional partial rotary is refused by the reference and
/// production backends alike, naming the policy and the rung — and the
/// same call with plain rotary runs, so it is the policy being refused.
#[test]
fn a_partial_rotary_is_refused_typed_by_both_cpu_backends() {
    let inputs: Vec<Vec<f32>> = (0..POSITIONS)
        .map(|p| lcg_values(HEAD_DIM, p as u64 + 1))
        .collect();
    let w = lcg_values(HEAD_DIM * HEAD_DIM, 7);
    let partial = PositionPolicy::PartialRope {
        theta: THETA,
        rotary_fraction: ROTARY_FRACTION,
        basis: RotaryFrequencyBasis::HeadWidth,
    };
    for (name, backend) in [
        (
            "reference",
            Box::new(ReferenceBackend::new()) as Box<dyn PlanBackend>,
        ),
        ("production", Box::new(ProductionBackend::new())),
    ] {
        let err = backend
            .attention(call(&inputs, &w, partial))
            .expect_err("PartialRope must be refused");
        let message = err.to_string();
        assert!(message.contains("PartialRope"), "{name}: {message}");
        assert!(message.contains(RUNG), "{name}: {message}");
        // Control: the plain rotary at the same base runs.
        backend
            .attention(call(&inputs, &w, PositionPolicy::Rope { theta: THETA }))
            .unwrap_or_else(|e| panic!("{name}: plain rope must run: {e}"));
    }
}
