//! Three arms of the reference backend that no fixture reached: the two
//! position/QK-norm forms it refuses by declaration, and the spatial
//! window span it treats as the whole prefix. Witnessed on the same
//! hand-built one-head call the partial-rotary tests use. Tests only.

use larql_models::config::{ParameterFreeQkNorm, PositionPolicy, QkNormScope};

use super::lcg_values;
use crate::format::vindex3::graph::policy::AttentionSpan;
use crate::format::vindex3::opplan::exec::backend::{
    AttentionCall, PlanBackend, QkNormCall, WeightSlice,
};
use crate::format::vindex3::opplan::exec::reference::ReferenceBackend;

const HEAD_DIM: usize = 16;
const EPS: f64 = 1e-5;
const POSITIONS: usize = 4;
/// The RMS-norm gain offset Gemma-style norms declare; the value is
/// immaterial to a refusal that happens before any weight is read.
const WEIGHT_OFFSET: f32 = 1.0;

fn inputs() -> Vec<Vec<f32>> {
    (0..POSITIONS)
        .map(|p| lcg_values(HEAD_DIM, p as u64 + 1))
        .collect()
}

fn call<'a>(
    inputs: &'a [Vec<f32>],
    w: &'a [f32],
    position: PositionPolicy,
    span: AttentionSpan,
    qk_norm: Option<QkNormCall<'a>>,
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
        qk_norm,
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
        span,
        window: None,
        gate: None,
        bias: None,
        sinks: None,
    }
}

#[test]
fn the_reference_refuses_a_relative_position_scheme_by_name() {
    let inputs = inputs();
    let w = lcg_values(HEAD_DIM * HEAD_DIM, 7);
    let err = ReferenceBackend::new()
        .attention(call(
            &inputs,
            &w,
            PositionPolicy::Relative {
                d_rel: 32,
                extent: 128,
            },
            AttentionSpan::Full,
            None,
        ))
        .err()
        .expect("a relative scheme is represented but not executable")
        .to_string();
    assert!(
        err.contains("relative position (d_rel 32, extent 128) is represented but not executable"),
        "{err}"
    );
}

#[test]
fn the_reference_refuses_a_full_projection_qk_norm_it_has_not_judged() {
    let inputs = inputs();
    let w = lcg_values(HEAD_DIM * HEAD_DIM, 7);
    let gain = vec![1.0f32; HEAD_DIM];
    let err = ReferenceBackend::new()
        .attention(call(
            &inputs,
            &w,
            PositionPolicy::None,
            AttentionSpan::Full,
            Some(QkNormCall {
                scope: QkNormScope::FullProjection,
                weight_offset: WEIGHT_OFFSET,
                q_weight: &gain,
                k_weight: &gain,
            }),
        ))
        .err()
        .expect("the full-projection scope has no judged reference execution")
        .to_string();
    assert!(
        err.contains("full-projection QK norm has no judged reference execution yet"),
        "{err}"
    );
}

#[test]
fn a_spatial_window_span_attends_over_the_whole_prefix() {
    let inputs = inputs();
    let w = lcg_values(HEAD_DIM * HEAD_DIM, 7);
    let backend = ReferenceBackend::new();
    let windowed = backend
        .attention(call(
            &inputs,
            &w,
            PositionPolicy::None,
            AttentionSpan::Windowed,
            None,
        ))
        .unwrap()
        .outputs;
    let full = backend
        .attention(call(
            &inputs,
            &w,
            PositionPolicy::None,
            AttentionSpan::Full,
            None,
        ))
        .unwrap()
        .outputs;
    assert_eq!(
        windowed, full,
        "no generic op lowers a perception window today; the span reads the whole prefix"
    );
}
