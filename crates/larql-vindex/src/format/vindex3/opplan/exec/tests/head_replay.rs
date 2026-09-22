//! The head replay is the actual prepared W_O + norm + residual path.

use crate::format::vindex3::fixtures::{dense_f32_model, encode_fixture_container};
use crate::format::vindex3::inspect::inspect_container;
use crate::format::vindex3::opplan::exec::{
    decode::DecodeSession,
    head_replay::{replay_attention_heads, replay_attention_mixture, replay_softmax_source_head},
    observe::{AttentionHeadRecord, CarrierWriteRecord, StepEvent, StepObserver, SublayerSite},
    operands::OperandStore,
    prepared::{ExecutionSlice, PreparedOperands},
    production::ProductionBackend,
};
use crate::format::vindex3::opplan::{plan_component_ops, ComponentOpPlan};

fn fixture() -> (tempfile::TempDir, ComponentOpPlan, OperandStore) {
    let checkpoint = tempfile::tempdir().unwrap();
    let container = tempfile::tempdir().unwrap();
    encode_fixture_container(
        dense_f32_model,
        checkpoint.path(),
        container.path(),
        "head-replay",
    );
    let inspection = inspect_container(container.path(), false).unwrap();
    let outcome = plan_component_ops(&inspection, container.path(), "target").unwrap();
    assert!(outcome.closed(), "defects: {:?}", outcome.defects);
    let store = OperandStore::open(container.path(), &inspection).unwrap();
    (container, outcome.plan.unwrap(), store)
}

#[derive(Default)]
struct Capture {
    position: usize,
    entering: Option<Vec<f32>>,
    heads: Vec<Vec<f32>>,
    raw: Option<Vec<f32>>,
    delta: Option<Vec<f32>>,
    after: Option<Vec<f32>>,
    source: Option<(Vec<f32>, Vec<Vec<f32>>, Vec<Vec<f32>>, Vec<f32>)>,
}

impl StepObserver for Capture {
    fn event(&mut self, event: StepEvent) {
        if let StepEvent::Embedded { position } = event {
            self.position = position;
        }
    }

    fn wants_attention_heads(&self) -> bool {
        true
    }

    fn entering_carrier(&mut self, position: usize, values: &[f32]) {
        if position == self.position {
            self.entering = Some(values.to_vec());
        }
    }

    fn attention_head(&mut self, layer: usize, record: AttentionHeadRecord<'_>) {
        if layer == 0 && record.position == self.position {
            if record.head == 0 {
                self.source = Some((
                    record.query.to_vec(),
                    record.source_keys.to_vec(),
                    record.source_values.to_vec(),
                    record.values.to_vec(),
                ));
            }
            self.heads.push(record.values.to_vec());
        }
    }

    fn attention_output(&mut self, layer: usize, position: usize, values: &[f32]) {
        if layer == 0 && position == self.position {
            self.raw = Some(values.to_vec());
        }
    }

    fn carrier_write(&mut self, record: CarrierWriteRecord<'_>) {
        if record.layer == 0
            && record.site == SublayerSite::Attention
            && record.position == self.position
        {
            self.delta = Some(record.delta.to_vec());
            self.after = Some(record.after.to_vec());
        }
    }
}

fn relative_l2(a: &[f32], b: &[f32]) -> f64 {
    let error = a
        .iter()
        .zip(b)
        .map(|(&a, &b)| f64::from(a - b).powi(2))
        .sum::<f64>()
        .sqrt();
    let norm = a
        .iter()
        .map(|&value| f64::from(value).powi(2))
        .sum::<f64>()
        .sqrt();
    if norm == 0.0 {
        error
    } else {
        error / norm
    }
}

#[test]
fn replay_reconstructs_the_observed_raw_norm_and_residual_path() {
    let (_dir, plan, store): (_, _, OperandStore) = fixture();
    let backend = ProductionBackend::new();
    let prepared = PreparedOperands::load(&plan, &store, &backend, ExecutionSlice::Full).unwrap();
    let mut session = DecodeSession::new(&plan, &store, &backend).unwrap();
    let mut capture = Capture::default();
    session.step_observed(3, &mut capture).unwrap();

    let replay = replay_attention_heads(
        &plan,
        &prepared,
        &backend,
        0,
        capture.entering.as_deref().unwrap(),
        &capture.heads,
    )
    .unwrap();
    let fast = replay_attention_mixture(
        &plan,
        &prepared,
        &backend,
        0,
        capture.entering.as_deref().unwrap(),
        &capture.heads,
    )
    .unwrap();
    assert_eq!(
        replay.contributions.len(),
        plan.layers[0].attention.softmax().unwrap().num_q_heads
    );
    assert!(replay.raw_reconstruction_relative_l2 < 1e-5);
    assert!(
        relative_l2(
            capture.raw.as_deref().unwrap(),
            &replay.raw_attention_output
        ) < 1e-6
    );
    assert!(relative_l2(capture.delta.as_deref().unwrap(), &replay.applied_delta) < 1e-6);
    assert!(relative_l2(capture.after.as_deref().unwrap(), &replay.carrier_after) < 1e-6);
    assert_eq!(fast.raw_attention_output, replay.raw_attention_output);
    assert_eq!(fast.applied_delta, replay.applied_delta);
    assert_eq!(fast.carrier_after, replay.carrier_after);
    let (query, keys, values, natural) = capture.source.unwrap();
    let op = plan.layers[0].attention.softmax().unwrap();
    let source =
        replay_softmax_source_head(&query, &keys, &values, op.score_scale, op.logit_softcapping)
            .unwrap();
    assert_eq!(source.values, natural);
}
