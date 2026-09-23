//! A cached single-stream handoff must preserve the canonical tail exactly.
use super::decode::fixture;
use crate::format::vindex3::fixtures::G_TOKENS;
use crate::format::vindex3::opplan::exec::{
    decode::DecodeSession,
    intervene::InterventionPlan,
    intervene_heads::HeadInterventionPlan,
    kv::RowKvState,
    observe::{CarrierWriteRecord, NoopObserver, StepEvent, StepObserver, SublayerSite},
    payload_prefix::PayloadPrefix,
    prepared::{ExecutionSlice, PreparedOperands},
    reference::ReferenceBackend,
};

#[derive(Default)]
struct Carrier {
    boundary: Vec<f32>,
    last: Vec<f32>,
}
impl StepObserver for Carrier {
    fn event(&mut self, _: StepEvent) {}
    fn carrier_write(&mut self, r: CarrierWriteRecord<'_>) {
        self.last = r.after.to_vec();
        if let Some(scale) = r.layer_scale {
            for v in &mut self.last {
                *v *= scale;
            }
        }
        if r.layer == 0 && r.site == SublayerSite::Ffn {
            self.boundary = self.last.clone();
        }
    }
}

#[test]
fn carrier_entry_matches_full_decode_and_refuses_bad_shape_before_advancing() {
    let (_dir, plan, store) = fixture();
    let backend = ReferenceBackend;
    let full = PreparedOperands::load(&plan, &store, &backend, ExecutionSlice::Full).unwrap();
    let tail = PreparedOperands::load(
        &plan,
        &store,
        &backend,
        ExecutionSlice::LayerRange {
            start: 1,
            end: plan.layers.len(),
        },
    )
    .unwrap();
    let mut prefix = RowKvState::default();
    {
        let mut session =
            DecodeSession::over_prepared(&plan, &full, &backend, &mut prefix).unwrap();
        for &t in &G_TOKENS[..G_TOKENS.len() - 1] {
            session.step_observed(t, &mut NoopObserver).unwrap();
        }
    }
    let mut original = prefix.clone();
    let mut observer = Carrier::default();
    let expected = DecodeSession::over_prepared(&plan, &full, &backend, &mut original)
        .unwrap()
        .step_observed(*G_TOKENS.last().unwrap(), &mut observer)
        .unwrap()
        .logits
        .unwrap();
    let mut resumed = DecodeSession::over_prepared(&plan, &tail, &backend, &mut prefix).unwrap();
    let position = resumed.position();
    assert!(resumed
        .step_from_carrier_intervened(
            &[f32::NAN],
            &mut NoopObserver,
            &InterventionPlan::none(),
            &HeadInterventionPlan::none(),
        )
        .is_err());
    assert_eq!(resumed.position(), position);
    let mut final_carrier = Carrier::default();
    let result = resumed
        .step_from_carrier_intervened(
            &observer.boundary,
            &mut final_carrier,
            &InterventionPlan::none(),
            &HeadInterventionPlan::none(),
        )
        .unwrap();
    assert!(result.logits.is_none());
    assert_eq!(
        expected,
        full.head_logits(&backend, &final_carrier.last)
            .unwrap()
            .unwrap()
    );
}

#[test]
fn payload_prefix_has_no_vocabulary_exit_and_preserves_lower_carriers() {
    let (_dir, plan, store) = fixture();
    let backend = ReferenceBackend;
    let full = PreparedOperands::load(&plan, &store, &backend, ExecutionSlice::Full).unwrap();
    let prefix = PayloadPrefix::prepare(&plan, &store, &backend, 1).unwrap();
    let carriers = prefix.carriers(&G_TOKENS, &backend).unwrap();
    let mut kv = RowKvState::default();
    let mut session = DecodeSession::over_prepared(&plan, &full, &backend, &mut kv).unwrap();
    for (&token, expected) in G_TOKENS.iter().zip(carriers) {
        let mut observer = Carrier::default();
        session.step_observed(token, &mut observer).unwrap();
        assert_eq!(expected, observer.boundary);
    }
}
