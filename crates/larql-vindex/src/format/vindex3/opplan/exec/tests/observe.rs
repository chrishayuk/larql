//! LQL-2 TRACE gates at the executor: observation is observational.
//!
//! Two claims: the observed step computes exactly what the unobserved
//! step computes (bit-for-bit — an observer can never fork the
//! semantics), and the event stream mirrors the plan's own structure
//! (every executed sublayer appears, in execution order).

use super::golden::{G_LAYERS, G_TOKENS, G_VOCAB};
use crate::format::vindex3::opplan::exec::decode::DecodeSession;
use crate::format::vindex3::opplan::exec::observe::{
    CarrierForm, RecordingObserver, StepEvent, SublayerSite,
};
use crate::format::vindex3::opplan::exec::reference::ReferenceBackend;

#[test]
fn an_observed_step_is_bit_identical_to_an_unobserved_one() {
    let (_c, plan, store) = super::decode::fixture();
    let backend = ReferenceBackend::new();

    let mut plain = DecodeSession::new(
        &plan,
        &store,
        &backend,
        Box::new(crate::format::vindex3::opplan::exec::kv::RowKvState::default()),
    )
    .unwrap();
    let mut observed = DecodeSession::new(
        &plan,
        &store,
        &backend,
        Box::new(crate::format::vindex3::opplan::exec::kv::RowKvState::default()),
    )
    .unwrap();
    let mut recorder = RecordingObserver::default();
    for &token in G_TOKENS.iter() {
        let a = plain.step(token).unwrap().logits;
        let b = observed.step_observed(token, &mut recorder).unwrap().logits;
        assert_eq!(a, b, "observation changed the arithmetic");
    }
}

#[test]
fn the_event_stream_mirrors_the_plans_structure_in_execution_order() {
    let (_c, plan, store) = super::decode::fixture();
    let backend = ReferenceBackend::new();
    let mut session = DecodeSession::new(
        &plan,
        &store,
        &backend,
        Box::new(crate::format::vindex3::opplan::exec::kv::RowKvState::default()),
    )
    .unwrap();
    let mut recorder = RecordingObserver::default();
    session.step_observed(G_TOKENS[0], &mut recorder).unwrap();

    // Per layer: the attention write, the attention boundary, the FFN
    // write, the FFN boundary — a write precedes the boundary that
    // closes its sublayer (V3-OBS-1).
    let mut expected = vec![StepEvent::Embedded { position: 0 }];
    for layer in 0..G_LAYERS {
        expected.push(StepEvent::CarrierWrite {
            layer,
            site: SublayerSite::Attention,
            carrier: CarrierForm::Single,
        });
        expected.push(StepEvent::AttentionDone { layer });
        expected.push(StepEvent::CarrierWrite {
            layer,
            site: SublayerSite::Ffn,
            carrier: CarrierForm::Single,
        });
        expected.push(StepEvent::FfnDone { layer });
    }
    expected.push(StepEvent::Logits { vocab: G_VOCAB });
    assert_eq!(recorder.events, expected);
    assert_eq!(
        plan.layers.len(),
        G_LAYERS,
        "expected stream covers every layer"
    );
}
