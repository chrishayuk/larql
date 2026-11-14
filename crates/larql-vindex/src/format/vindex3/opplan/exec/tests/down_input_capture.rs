//! SG-EDIT-1 I5 (E): the positioned `down_proj`-input capture names the
//! right position, and capturing changes nothing — every carrier write and
//! every step's logits are bit-identical with and without it, on both CPU
//! backends. The capture routes the FFN through the backend's observed
//! entry point, so this is a check on the executor, not a formality.

use super::decode::fixture;
use super::golden::{G_LAYERS, G_TOKENS};
use super::i5_support::{assert_runs_identical, run_plain, same_bits, Recorder};
use crate::format::vindex3::fixtures::G_FFN;
use crate::format::vindex3::opplan::exec::backend::PlanBackend;
use crate::format::vindex3::opplan::exec::down_input_capture::DownInputCapture;
use crate::format::vindex3::opplan::exec::intervene::vector_sha256;
use crate::format::vindex3::opplan::exec::observe::{StepEvent, StepObserver};
use crate::format::vindex3::opplan::exec::production::ProductionBackend;
use crate::format::vindex3::opplan::exec::reference::ReferenceBackend;

fn every_address() -> Vec<(usize, usize)> {
    (0..G_LAYERS)
        .flat_map(|layer| (0..G_TOKENS.len()).map(move |p| (layer, p)))
        .collect()
}

fn capture_is_bit_identical<B: PlanBackend>(backend: &B, label: &str) {
    let (_dir, plan, store) = fixture();
    let mut plain = Recorder::default();
    let plain_logits = run_plain(&plan, &store, backend, &G_TOKENS, &mut plain);
    let mut captured = Recorder {
        down: Some(DownInputCapture::at(every_address())),
        ..Recorder::default()
    };
    let captured_logits = run_plain(&plan, &store, backend, &G_TOKENS, &mut captured);
    assert_runs_identical(
        label,
        (&plain_logits, &plain.writes),
        (&captured_logits, &captured.writes),
    );
    let down = captured.down.unwrap();
    assert_eq!(down.captured(), G_LAYERS * G_TOKENS.len());
    assert!(down.missing().is_empty());
    for (layer, position) in every_address() {
        let values = down.get(layer, position).unwrap();
        assert_eq!(values.len(), G_FFN, "{label}: down input width");
        assert_eq!(down.sha256(layer, position).unwrap(), vector_sha256(values));
    }
}

#[test]
fn capturing_the_down_input_is_bit_identical_on_the_reference_backend() {
    capture_is_bit_identical(&ReferenceBackend::new(), "reference");
}

#[test]
fn capturing_the_down_input_is_bit_identical_on_the_production_backend() {
    capture_is_bit_identical(&ProductionBackend::new(), "production");
}

#[test]
fn a_single_address_capture_matches_the_full_capture_at_that_position() {
    let (_dir, plan, store) = fixture();
    let backend = ProductionBackend::new();
    let mut full = Recorder {
        down: Some(DownInputCapture::at(every_address())),
        ..Recorder::default()
    };
    run_plain(&plan, &store, &backend, &G_TOKENS, &mut full);
    let full = full.down.unwrap();
    let target = (G_LAYERS - 1, 2);
    let mut single = Recorder {
        down: Some(DownInputCapture::at([target])),
        ..Recorder::default()
    };
    run_plain(&plan, &store, &backend, &G_TOKENS, &mut single);
    let single = single.down.unwrap();
    assert_eq!(single.captured(), 1);
    assert!(same_bits(
        single.get(target.0, target.1).unwrap(),
        full.get(target.0, target.1).unwrap()
    ));
    // Different positions carry different inputs: the position is not
    // stuck at one value.
    assert!(!same_bits(
        full.get(target.0, 0).unwrap(),
        full.get(target.0, target.1).unwrap()
    ));
}

#[test]
fn an_unreached_address_is_reported_missing_not_filled() {
    let (_dir, plan, store) = fixture();
    let beyond = G_TOKENS.len();
    let mut recorder = Recorder {
        down: Some(DownInputCapture::at([(0, 0), (0, beyond)])),
        ..Recorder::default()
    };
    run_plain(
        &plan,
        &store,
        &ProductionBackend::new(),
        &G_TOKENS,
        &mut recorder,
    );
    let down = recorder.down.unwrap();
    assert_eq!(down.missing(), vec![(0, beyond)]);
    assert!(down.get(0, beyond).is_none());
    assert!(down.sha256(0, beyond).is_none());
}

#[test]
fn the_capture_is_a_standalone_observer_too() {
    let mut capture = DownInputCapture::at([(1, 4)]);
    assert!(capture.wants_ffn_down_input(1));
    assert!(!capture.wants_ffn_down_input(0));
    capture.event(StepEvent::Embedded { position: 3 });
    capture.ffn_down_input(1, &[1.0, 2.0]);
    assert_eq!(capture.captured(), 0, "position 3 is not declared");
    capture.event(StepEvent::Embedded { position: 4 });
    capture.ffn_down_input(1, &[1.0, 2.0]);
    assert_eq!(capture.get(1, 4), Some(&[1.0, 2.0][..]));
}

#[test]
#[should_panic(expected = "before the step's `Embedded` event")]
fn a_down_input_before_any_position_is_a_caller_error() {
    let mut capture = DownInputCapture::at([(0, 0)]);
    capture.observe_down_input(0, &[1.0]);
}
