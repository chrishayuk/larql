//! SG-EDIT-1 I5 (B): the H1-only replay path, by two independent checks.
//!
//! 1. A replay with the natural value rows reproduces the captured `ctx_h`
//!    bit for bit, for every layer, query head and position, on the
//!    production backend. The replay shares production's softmax
//!    (`larql_compute::attention::softmax::softmax_in_place_f32`); the
//!    reference backend is an independent transcription (`kernels::softmax`,
//!    an f32 sum and a divide), so there the replay is only close. The H1-only
//!    path is therefore admitted on the production backend alone.
//! 2. A head `Replace` with that replay leaves the whole run unchanged:
//!    every carrier write and every step's logits. Check 1 alone would not
//!    do — the post-attention norm couples heads — so the executor's own
//!    downstream arithmetic is what is compared.
//!
//! Then the counterfactual itself: replacing one source row changes only
//! the replaced head's replay, and replacing a row with itself is exact.

use std::collections::BTreeMap;

use super::decode::fixture;
use super::golden::{G_LAYERS, G_TOKENS};
use super::i5_support::{assert_runs_identical, run, run_plain, same_bits, Recorder};
use crate::format::vindex3::fixtures::{G_HEAD_DIM, G_Q_HEADS};
use crate::format::vindex3::opplan::exec::backend::PlanBackend;
use crate::format::vindex3::opplan::exec::intervene::InterventionPlan;
use crate::format::vindex3::opplan::exec::intervene_heads::HeadInterventionPlan;
use crate::format::vindex3::opplan::exec::observe::{StepEvent, StepObserver, SublayerSite};
use crate::format::vindex3::opplan::exec::production::ProductionBackend;
use crate::format::vindex3::opplan::exec::reference::ReferenceBackend;
use crate::format::vindex3::opplan::exec::source_head::SourceHeadCapture;

fn every_head_address() -> Vec<(usize, usize, usize)> {
    let mut addresses = Vec::new();
    for layer in 0..G_LAYERS {
        for head in 0..G_Q_HEADS {
            for position in 0..G_TOKENS.len() {
                addresses.push((layer, head, position));
            }
        }
    }
    addresses
}

fn captured<B: PlanBackend>(backend: &B) -> (Vec<Vec<f32>>, Recorder) {
    let (_dir, plan, store) = fixture();
    let mut recorder = Recorder {
        heads: Some(SourceHeadCapture::at(every_head_address())),
        ..Recorder::default()
    };
    let logits = run_plain(&plan, &store, backend, &G_TOKENS, &mut recorder);
    (logits, recorder)
}

/// The relative L2 within which the replay matches the reference
/// backend's independently transcribed kernel: a few f32 roundings on a
/// unit-scale head.
const REFERENCE_KERNEL_RELATIVE_L2: f64 = 1e-6;

fn relative_l2(a: &[f32], b: &[f32]) -> f64 {
    let error: f64 = a
        .iter()
        .zip(b)
        .map(|(&x, &y)| (f64::from(x) - f64::from(y)).powi(2))
        .sum::<f64>()
        .sqrt();
    let norm: f64 = b.iter().map(|&y| f64::from(y).powi(2)).sum::<f64>().sqrt();
    if norm == 0.0 {
        error
    } else {
        error / norm
    }
}

#[test]
fn a_natural_replay_reproduces_ctx_on_the_production_backend() {
    let (_dir, plan, _store) = fixture();
    let (_, recorder) = captured(&ProductionBackend::new());
    let heads = recorder.heads.unwrap();
    assert_eq!(heads.captured(), every_head_address().len());
    for (layer, head, position) in every_head_address() {
        let source = heads.get(layer, head, position).unwrap();
        let replayed = source.replay(&plan).unwrap();
        assert!(
            same_bits(&replayed, &source.ctx),
            "production: layer {layer} head {head} position {position} replay != ctx_h"
        );
    }
}

#[test]
fn the_replay_is_only_close_to_the_reference_kernel() {
    let (_dir, plan, _store) = fixture();
    let (_, recorder) = captured(&ReferenceBackend::new());
    let heads = recorder.heads.unwrap();
    let mut any_differs = false;
    for (layer, head, position) in every_head_address() {
        let source = heads.get(layer, head, position).unwrap();
        let replayed = source.replay(&plan).unwrap();
        any_differs |= !same_bits(&replayed, &source.ctx);
        let distance = relative_l2(&replayed, &source.ctx);
        assert!(
            distance < REFERENCE_KERNEL_RELATIVE_L2,
            "reference: layer {layer} head {head} position {position} at {distance}"
        );
    }
    // Recorded so a kernel change that makes them agree is noticed and the
    // admission note above revisited.
    assert!(
        any_differs,
        "the reference kernel now matches the replay bit for bit"
    );
}

fn replace_with_replay_is_the_no_op<B: PlanBackend>(backend: &B, label: &str) {
    let (_dir, plan, store) = fixture();
    // Only meaningful where check 1 holds bit for bit: the production
    // backend (see the module note).
    let (baseline_logits, baseline) = captured(backend);
    let heads = baseline.heads.as_ref().unwrap();
    let last = G_TOKENS.len() - 1;
    for layer in 0..G_LAYERS {
        for head in 0..G_Q_HEADS {
            let source = heads.get(layer, head, last).unwrap();
            let replace = source.replace_with(source.replay(&plan).unwrap()).unwrap();
            let head_plan = HeadInterventionPlan::none().with(replace).unwrap();
            let mut recorder = Recorder::default();
            let logits = run(
                &plan,
                &store,
                backend,
                &G_TOKENS,
                &mut recorder,
                &InterventionPlan::none(),
                &head_plan,
            );
            assert_runs_identical(
                &format!("{label} layer {layer} head {head}"),
                (&baseline_logits, &baseline.writes),
                (&logits, &recorder.writes),
            );
        }
    }
}

#[test]
fn replacing_ctx_with_its_replay_leaves_the_run_unchanged_on_the_production_backend() {
    replace_with_replay_is_the_no_op(&ProductionBackend::new(), "production");
}

#[test]
fn a_replaced_source_row_moves_only_the_replaced_heads_replay() {
    let (_dir, plan, store) = fixture();
    let backend = ProductionBackend::new();
    let (baseline_logits, baseline) = captured(&backend);
    let heads = baseline.heads.as_ref().unwrap();
    let (layer, head, position) = (G_LAYERS - 1, 1, G_TOKENS.len() - 1);
    let source = heads.get(layer, head, position).unwrap();
    let row = *source.sources().start();

    // Replacing a row with itself is exact.
    let itself = BTreeMap::from([(row, source.values[0].clone())]);
    assert!(same_bits(
        &source.replay_with_values(&plan, &itself).unwrap(),
        &source.ctx
    ));

    // A shifted row changes this head's replay ...
    let shifted: Vec<f32> = source.values[0].iter().map(|v| v + 1.0).collect();
    let shift = BTreeMap::from([(row, shifted)]);
    let counterfactual = source.replay_with_values(&plan, &shift).unwrap();
    assert!(!same_bits(&counterfactual, &source.ctx));

    // ... and, applied, changes the layer's attention write while every
    // other head of the KV group keeps its natural ctx_h (it never saw the
    // shifted row).
    let head_plan = HeadInterventionPlan::none()
        .with(source.replace_with(counterfactual).unwrap())
        .unwrap();
    let mut recorder = Recorder {
        heads: Some(SourceHeadCapture::at(
            (0..G_Q_HEADS).map(|h| (layer, h, position)),
        )),
        ..Recorder::default()
    };
    let logits = run(
        &plan,
        &store,
        &backend,
        &G_TOKENS,
        &mut recorder,
        &InterventionPlan::none(),
        &head_plan,
    );
    let key = (layer, SublayerSite::Attention, position);
    assert!(!same_bits(&recorder.writes[&key], &baseline.writes[&key]));
    assert!(!same_bits(&logits[position], &baseline_logits[position]));
    let after = recorder.heads.unwrap();
    for other in (0..G_Q_HEADS).filter(|&h| h != head) {
        assert!(same_bits(
            &after.get(layer, other, position).unwrap().ctx,
            &heads.get(layer, other, position).unwrap().ctx
        ));
    }
}

#[test]
fn the_replay_refuses_rows_it_cannot_place() {
    let (_dir, plan, _store) = fixture();
    let (_, recorder) = captured(&ProductionBackend::new());
    let heads = recorder.heads.unwrap();
    let source = heads.get(0, 0, 0).unwrap();
    let outside = BTreeMap::from([(source.position + 1, vec![0.0; G_HEAD_DIM])]);
    let err = source.replay_with_values(&plan, &outside).unwrap_err();
    assert!(err.to_string().contains("is not a source"), "{err}");
    let narrow = BTreeMap::from([(source.position, vec![0.0; G_HEAD_DIM - 1])]);
    let err = source.replay_with_values(&plan, &narrow).unwrap_err();
    assert!(err.to_string().contains("width"), "{err}");
    let mut off_plan = source.clone();
    off_plan.layer = G_LAYERS;
    let err = off_plan.replay(&plan).unwrap_err();
    assert!(err.to_string().contains("no softmax attention"), "{err}");
}

#[test]
fn the_capture_asks_only_where_it_was_declared() {
    let mut empty = SourceHeadCapture::at([]);
    assert!(!empty.wants_attention_heads());
    empty.event(StepEvent::Embedded { position: 0 });
    let capture = SourceHeadCapture::at([(1, 2, 3)]);
    assert!(capture.wants_attention_heads());
    assert!(capture.wants_attention_heads_at(1, 3));
    assert!(!capture.wants_attention_heads_at(1, 2));
    assert!(!capture.wants_attention_heads_at(0, 3));
    assert!(capture.get(1, 2, 3).is_none());
    assert_eq!(capture.captured(), 0);
}
