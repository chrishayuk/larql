//! Shared harness for the SG-EDIT-1 I5 acceptance tests: one observer that
//! records every single-stream carrier write and forwards to the captures
//! under test, and one driver that steps the golden tokens through a fresh
//! session with interventions armed.

use std::collections::BTreeMap;

use crate::format::vindex3::opplan::exec::backend::PlanBackend;
use crate::format::vindex3::opplan::exec::decode::DecodeSession;
use crate::format::vindex3::opplan::exec::down_input_capture::DownInputCapture;
use crate::format::vindex3::opplan::exec::intervene::{CarrierCapture, InterventionPlan};
use crate::format::vindex3::opplan::exec::intervene_heads::HeadInterventionPlan;
use crate::format::vindex3::opplan::exec::kv::RowKvState;
use crate::format::vindex3::opplan::exec::observe::{
    AttentionHeadRecord, CarrierWriteRecord, StepEvent, StepObserver, SublayerSite,
};
use crate::format::vindex3::opplan::exec::operands::OperandStore;
use crate::format::vindex3::opplan::exec::source_head::SourceHeadCapture;
use crate::format::vindex3::opplan::ComponentOpPlan;

pub(super) type WriteKey = (usize, SublayerSite, usize);

/// Every carrier write's `after`, plus whichever captures a test arms.
#[derive(Default)]
pub(super) struct Recorder {
    pub(super) writes: BTreeMap<WriteKey, Vec<f32>>,
    pub(super) carriers: Option<CarrierCapture>,
    pub(super) down: Option<DownInputCapture>,
    pub(super) heads: Option<SourceHeadCapture>,
}

impl StepObserver for Recorder {
    fn event(&mut self, event: StepEvent) {
        if let Some(down) = &mut self.down {
            down.observe_event(&event);
        }
    }

    fn wants_ffn_down_input(&self, layer: usize) -> bool {
        self.down.as_ref().is_some_and(|d| d.wants_layer(layer))
    }

    fn ffn_down_input(&mut self, layer: usize, values: &[f32]) {
        if let Some(down) = &mut self.down {
            down.observe_down_input(layer, values);
        }
    }

    fn wants_attention_heads(&self) -> bool {
        self.heads
            .as_ref()
            .is_some_and(StepObserver::wants_attention_heads)
    }

    fn wants_attention_heads_at(&self, layer: usize, position: usize) -> bool {
        self.heads
            .as_ref()
            .is_some_and(|h| h.wants_at(layer, position))
    }

    fn attention_head(&mut self, layer: usize, record: AttentionHeadRecord<'_>) {
        if let Some(heads) = &mut self.heads {
            heads.observe(layer, &record);
        }
    }

    fn carrier_write(&mut self, record: CarrierWriteRecord<'_>) {
        self.writes.insert(
            (record.layer, record.site, record.position),
            record.after.to_vec(),
        );
        if let Some(carriers) = &mut self.carriers {
            carriers.carrier_write(record);
        }
    }
}

/// Step `tokens` through a fresh session with both plans armed on every
/// step, returning each step's logits.
pub(super) fn run<B: PlanBackend>(
    plan: &ComponentOpPlan,
    store: &OperandStore,
    backend: &B,
    tokens: &[u32],
    recorder: &mut Recorder,
    interventions: &InterventionPlan,
    head_interventions: &HeadInterventionPlan,
) -> Vec<Vec<f32>> {
    let mut session =
        DecodeSession::new(plan, store, backend, Box::new(RowKvState::default())).unwrap();
    tokens
        .iter()
        .map(|&token| {
            session
                .step_intervened(token, recorder, interventions, head_interventions)
                .unwrap()
                .logits
                .unwrap()
        })
        .collect()
}

/// [`run`] with nothing armed.
pub(super) fn run_plain<B: PlanBackend>(
    plan: &ComponentOpPlan,
    store: &OperandStore,
    backend: &B,
    tokens: &[u32],
    recorder: &mut Recorder,
) -> Vec<Vec<f32>> {
    run(
        plan,
        store,
        backend,
        tokens,
        recorder,
        &InterventionPlan::none(),
        &HeadInterventionPlan::none(),
    )
}

/// Bitwise equality of two f32 vectors (distinguishes `-0.0` from `+0.0`).
pub(super) fn same_bits(a: &[f32], b: &[f32]) -> bool {
    a.len() == b.len() && a.iter().zip(b).all(|(x, y)| x.to_bits() == y.to_bits())
}

/// Bitwise equality of two runs: every write and every step's logits.
pub(super) fn assert_runs_identical(
    label: &str,
    (logits_a, writes_a): (&[Vec<f32>], &BTreeMap<WriteKey, Vec<f32>>),
    (logits_b, writes_b): (&[Vec<f32>], &BTreeMap<WriteKey, Vec<f32>>),
) {
    assert_eq!(
        writes_a.keys().collect::<Vec<_>>(),
        writes_b.keys().collect::<Vec<_>>(),
        "{label}: the runs wrote different addresses"
    );
    for (key, after) in writes_a {
        assert!(
            same_bits(after, &writes_b[key]),
            "{label}: carrier at {key:?} differs"
        );
    }
    assert_eq!(
        logits_a.len(),
        logits_b.len(),
        "{label}: step counts differ"
    );
    for (step, (a, b)) in logits_a.iter().zip(logits_b).enumerate() {
        assert!(same_bits(a, b), "{label}: logits differ at step {step}");
    }
}
