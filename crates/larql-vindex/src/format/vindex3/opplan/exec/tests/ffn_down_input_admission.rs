//! CAL-1.1's down-input tap is admitted before the token executes, as the
//! per-head request is: a backend that cannot serve it refuses with no
//! layer run, no KV written and nothing observed. Tests only.

use std::sync::Arc;

use super::decode::fixture;
use super::golden::{G_LAYERS, G_TOKENS};
use crate::error::VindexError;
use crate::format::vindex3::opplan::exec::backend::{
    AttentionCall, AttentionOut, AttentionStepCall, AttentionStepOut, FfnCall, NormCall,
    PlanBackend, ProjectCall, RoutedFfnCall, WeightSlice,
};
use crate::format::vindex3::opplan::exec::decode::DecodeSession;
use crate::format::vindex3::opplan::exec::intervene::InterventionPlan;
use crate::format::vindex3::opplan::exec::intervene_heads::HeadInterventionPlan;
use crate::format::vindex3::opplan::exec::kv::RowKvState;
use crate::format::vindex3::opplan::exec::observe::{InputSite, StepEvent, StepObserver};
use crate::format::vindex3::opplan::exec::production::ProductionBackend;
use crate::format::vindex3::opplan::exec::reference::ReferenceBackend;

/// The last layer: every earlier layer would have run and written KV by
/// the time a mid-step refusal reached it.
const TAPPED_LAYER: usize = G_LAYERS - 1;

/// Asks for the down input at one layer and counts everything it is sent.
#[derive(Default)]
struct DownInputProbe {
    events: usize,
    operand_inputs: usize,
    captures: usize,
}

impl StepObserver for DownInputProbe {
    fn event(&mut self, _event: StepEvent) {
        self.events += 1;
    }

    fn wants_ffn_down_input(&self, layer: usize) -> bool {
        layer == TAPPED_LAYER
    }

    fn ffn_down_input(&mut self, layer: usize, values: &[f32]) {
        assert_eq!(layer, TAPPED_LAYER);
        assert!(!values.is_empty());
        self.captures += 1;
    }

    fn operand_input(&mut self, _layer: usize, _site: InputSite, _values: &[f32]) {
        self.operand_inputs += 1;
    }
}

/// Reference arithmetic behind a backend that declares nothing about the
/// down-input tap: `serves_ffn_down_input` and `ffn_observed` are the
/// trait's defaults.
struct NoDownInputBackend(ReferenceBackend);

impl PlanBackend for NoDownInputBackend {
    fn name(&self) -> &str {
        "no-down-input"
    }

    fn identity(&self) -> crate::format::vindex3::opplan::exec::lowering::LoweringIdentity {
        self.0.identity()
    }

    fn embed(&self, table: &[f32], hidden: usize, token: u32, scale: Option<f32>) -> Vec<f32> {
        self.0.embed(table, hidden, token, scale)
    }

    fn norm(&self, call: NormCall<'_>) -> Vec<f32> {
        self.0.norm(call)
    }

    fn project(&self, call: ProjectCall<'_>) -> Result<Vec<f32>, VindexError> {
        self.0.project(call)
    }

    fn attention(&self, call: AttentionCall<'_>) -> Result<AttentionOut, VindexError> {
        self.0.attention(call)
    }

    fn attention_step(&self, call: AttentionStepCall<'_>) -> Result<AttentionStepOut, VindexError> {
        self.0.attention_step(call)
    }

    fn ffn(&self, call: FfnCall<'_>) -> Result<Vec<f32>, VindexError> {
        self.0.ffn(call)
    }

    fn routed_ffn(&self, call: RoutedFfnCall<'_>) -> Result<Vec<f32>, VindexError> {
        self.0.routed_ffn(call)
    }

    fn output_head(
        &self,
        projection: WeightSlice<'_>,
        vocab: usize,
        hidden: usize,
        x: &[f32],
        multiplier: Option<f64>,
        softcapping: Option<f32>,
    ) -> Result<Vec<f32>, VindexError> {
        self.0
            .output_head(projection, vocab, hidden, x, multiplier, softcapping)
    }

    fn residual_add(&self, acc: &mut [f32], delta: &[f32]) {
        self.0.residual_add(acc, delta)
    }
}

const REFUSAL: &str = "FFN down-input capture is not served by the no-down-input backend";

#[test]
fn backends_declare_whether_they_serve_the_down_input() {
    assert!(ReferenceBackend::new().serves_ffn_down_input());
    assert!(ProductionBackend::new().serves_ffn_down_input());
    assert!(Arc::new(ReferenceBackend::new()).serves_ffn_down_input());
    assert!(!NoDownInputBackend(ReferenceBackend::new()).serves_ffn_down_input());
}

#[test]
fn an_unserved_down_input_request_refuses_before_the_token_executes() {
    let (_c, plan, store) = fixture();
    let backend = NoDownInputBackend(ReferenceBackend::new());
    let mut session =
        DecodeSession::new(&plan, &store, &backend, Box::new(RowKvState::default())).unwrap();
    let mut probe = DownInputProbe::default();
    let err = match session.step_observed(G_TOKENS[0], &mut probe) {
        Ok(_) => panic!("an unserved down-input request must refuse"),
        Err(e) => e.to_string(),
    };
    assert!(err.contains(REFUSAL), "{err}");
    assert_eq!(
        (probe.events, probe.operand_inputs, probe.captures),
        (0, 0, 0),
        "nothing is observed before the refusal"
    );
    assert_eq!(session.position(), 0, "no position is consumed");

    // No KV was written: the refused session's next step is a fresh
    // session's first step, bit for bit.
    let reused = session.step(G_TOKENS[0]).unwrap().logits;
    let mut fresh =
        DecodeSession::new(&plan, &store, &backend, Box::new(RowKvState::default())).unwrap();
    let expected = fresh.step(G_TOKENS[0]).unwrap().logits;
    assert_eq!(
        reused
            .iter()
            .flatten()
            .map(|v| v.to_bits())
            .collect::<Vec<_>>(),
        expected
            .iter()
            .flatten()
            .map(|v| v.to_bits())
            .collect::<Vec<_>>()
    );
}

#[test]
fn an_unserved_down_input_request_refuses_the_intervened_step_too() {
    let (_c, plan, store) = fixture();
    let backend = NoDownInputBackend(ReferenceBackend::new());
    let mut session =
        DecodeSession::new(&plan, &store, &backend, Box::new(RowKvState::default())).unwrap();
    let mut probe = DownInputProbe::default();
    let err = match session.step_intervened(
        G_TOKENS[0],
        &mut probe,
        &InterventionPlan::none(),
        &HeadInterventionPlan::none(),
    ) {
        Ok(_) => panic!("an unserved down-input request must refuse"),
        Err(e) => e.to_string(),
    };
    assert!(err.contains(REFUSAL), "{err}");
    assert_eq!((probe.events, probe.operand_inputs), (0, 0));
    assert_eq!(session.position(), 0);
}

#[test]
fn a_served_down_input_request_is_captured_once_per_step() {
    let (_c, plan, store) = fixture();
    let backend = ReferenceBackend::new();
    let mut session =
        DecodeSession::new(&plan, &store, &backend, Box::new(RowKvState::default())).unwrap();
    let mut probe = DownInputProbe::default();
    for &token in G_TOKENS.iter() {
        session.step_observed(token, &mut probe).unwrap();
    }
    assert_eq!(probe.captures, G_TOKENS.len());
}
