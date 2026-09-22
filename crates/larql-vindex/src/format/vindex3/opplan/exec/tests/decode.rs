//! Decode-vs-batch parity: the two traversals realise one program.
//!
//! A [`DecodeSession`] feeds tokens one at a time through
//! `attention_step` against its own K/V cache; the batch path computes
//! every position in one pass. Per position the arithmetic is the same
//! operations on the same values in the same order, so the final
//! logits must agree **bit-for-bit** per backend — any gap means the
//! step path re-derived something (a rope position, a mask start, a
//! norm placement) instead of inheriting it.
//!
//! The miniature fixture's sliding window (3, over 5 positions)
//! genuinely truncates from position 3, so this parity also pins the
//! step path's masking — the cache may hold a position the span must
//! exclude, and only the span logic keeps it out.

use super::golden::{miniature_glimmer, G_TOKENS};
use crate::format::vindex3::encode::encode_system;
use crate::format::vindex3::inspect::inspect_container;
use crate::format::vindex3::opplan::exec::backend::{PlanBackend, WeightFormat};
use crate::format::vindex3::opplan::exec::decode::{DecodeObserver, DecodeSession};
use crate::format::vindex3::opplan::exec::device::DevicePlanBackend;
use crate::format::vindex3::opplan::exec::execute_plan;
use crate::format::vindex3::opplan::exec::operands::OperandStore;
use crate::format::vindex3::opplan::exec::production::ProductionBackend;
use crate::format::vindex3::opplan::exec::reference::ReferenceBackend;
use crate::format::vindex3::opplan::{plan_component_ops, ComponentOpPlan};

fn fixture() -> (tempfile::TempDir, ComponentOpPlan, OperandStore) {
    let dir = tempfile::tempdir().unwrap();
    miniature_glimmer(dir.path());
    let inventory = larql_models::inventory::build_inventory(dir.path()).unwrap();
    let container = tempfile::tempdir().unwrap();
    encode_system(&[("mini-glimmer".to_string(), inventory)], container.path()).unwrap();
    let inspection = inspect_container(container.path(), false).unwrap();
    let outcome = plan_component_ops(&inspection, container.path(), "target").unwrap();
    assert!(outcome.closed(), "defects: {:?}", outcome.defects);
    let plan = outcome.plan.unwrap();
    let store = OperandStore::open(container.path(), &inspection).unwrap();
    (container, plan, store)
}

/// Step every fixture token through a fresh session and return the last
/// position's logits.
fn decode_logits<B: PlanBackend>(
    plan: &ComponentOpPlan,
    store: &OperandStore,
    backend: &B,
) -> Vec<f32> {
    let mut session = DecodeSession::new(plan, store, backend).unwrap();
    let mut last = None;
    for &token in G_TOKENS.iter() {
        last = session.step(token).unwrap().logits;
    }
    assert_eq!(session.position(), G_TOKENS.len());
    last.expect("plan carries an output head")
}

fn assert_bit_exact<B: PlanBackend>(backend: &B) {
    let (_c, plan, store) = fixture();
    let batch = execute_plan(&plan, &store, &G_TOKENS, backend).unwrap();
    let stepped = decode_logits(&plan, &store, backend);
    assert_eq!(
        batch.logits.as_deref(),
        Some(stepped.as_slice()),
        "{}: decode-session logits differ from the batch traversal",
        backend.name()
    );
}

#[test]
fn reference_decode_matches_the_batch_traversal_bit_for_bit() {
    assert_bit_exact(&ReferenceBackend::new());
}

#[test]
fn production_decode_matches_the_batch_traversal_bit_for_bit() {
    assert_bit_exact(&ProductionBackend::new());
}

#[test]
fn mxfp4_device_decode_matches_its_own_batch_traversal_bit_for_bit() {
    // The 32-aligned fixture: MXFP4's group constraint correctly
    // refuses the awkward hidden-12 miniature.
    let (_c, plan, store) = super::device::aligned_fixture();
    let backend = DevicePlanBackend::new(
        super::device::LoopDevice,
        "loop-device-mxfp4-decode",
        WeightFormat::Mxfp4,
    );
    let batch = execute_plan(&plan, &store, &G_TOKENS, &backend).unwrap();
    let stepped = decode_logits(&plan, &store, &backend);
    assert_eq!(batch.logits.as_deref(), Some(stepped.as_slice()));
}

#[test]
fn f16_device_decode_matches_its_own_batch_traversal_bit_for_bit() {
    // Same loop device the seam tests use; the point here is that the
    // step path and the batch path convert and consume the *same* f16
    // bytes, so even the lossy realisation self-agrees exactly.
    assert_bit_exact(&DevicePlanBackend::new(
        super::device::LoopDevice,
        "loop-device-f16-decode",
        WeightFormat::F16,
    ));
}

#[test]
fn reset_starts_an_independent_sequence_without_reloading_operands() {
    let (_c, plan, store) = fixture();
    let backend = ReferenceBackend::new();
    let mut session = DecodeSession::new(&plan, &store, &backend).unwrap();
    let mut first = None;
    for &token in G_TOKENS.iter() {
        first = session.step(token).unwrap().logits;
    }
    session.reset();
    assert_eq!(session.position(), 0);
    let mut second = None;
    for &token in G_TOKENS.iter() {
        second = session.step(token).unwrap().logits;
    }
    assert_eq!(first, second);
}

#[derive(Default)]
struct TapCounter {
    layer_inputs: Vec<usize>,
    attention_inputs: Vec<usize>,
    attention_outputs: Vec<usize>,
    post_attention: Vec<usize>,
    ffn_calls: Vec<usize>,
    ffn_outputs: Vec<usize>,
    post_layers: Vec<usize>,
}

impl DecodeObserver for TapCounter {
    fn layer_input(
        &mut self,
        layer: usize,
        _residual: &[f32],
    ) -> Result<(), crate::error::VindexError> {
        self.layer_inputs.push(layer);
        Ok(())
    }

    fn attention_input(
        &mut self,
        layer: usize,
        _input: &[f32],
    ) -> Result<(), crate::error::VindexError> {
        self.attention_inputs.push(layer);
        Ok(())
    }

    fn attention_output(
        &mut self,
        layer: usize,
        _output: &[f32],
    ) -> Result<(), crate::error::VindexError> {
        self.attention_outputs.push(layer);
        Ok(())
    }

    fn post_attention(
        &mut self,
        layer: usize,
        _residual: &[f32],
    ) -> Result<(), crate::error::VindexError> {
        self.post_attention.push(layer);
        Ok(())
    }

    fn ffn_call(
        &mut self,
        layer: usize,
        _call: &crate::format::vindex3::opplan::exec::backend::FfnCall<'_>,
    ) -> Result<(), crate::error::VindexError> {
        self.ffn_calls.push(layer);
        Ok(())
    }

    fn ffn_output(
        &mut self,
        layer: usize,
        _output: &[f32],
    ) -> Result<(), crate::error::VindexError> {
        self.ffn_outputs.push(layer);
        Ok(())
    }

    fn post_layer(
        &mut self,
        layer: usize,
        _residual: &[f32],
    ) -> Result<(), crate::error::VindexError> {
        self.post_layers.push(layer);
        Ok(())
    }
}

#[test]
fn observed_step_has_one_read_only_tap_per_layer() {
    let (_c, plan, store) = fixture();
    let backend = ReferenceBackend::new();
    let mut session = DecodeSession::new(&plan, &store, &backend).unwrap();
    let baseline = session.step(G_TOKENS[0]).unwrap().logits;
    session.reset();
    let mut taps = TapCounter::default();
    let observed = session
        .step_observed(G_TOKENS[0], &mut taps)
        .unwrap()
        .logits;
    assert_eq!(baseline, observed, "observer changed execution");
    assert_eq!(
        taps.layer_inputs,
        (0..plan.layers.len()).collect::<Vec<_>>()
    );
    assert_eq!(taps.attention_inputs, taps.layer_inputs);
    assert_eq!(taps.attention_outputs, taps.layer_inputs);
    assert_eq!(taps.post_attention, taps.layer_inputs);
    assert_eq!(taps.ffn_calls, (0..plan.layers.len()).collect::<Vec<_>>());
    assert_eq!(taps.ffn_outputs, taps.layer_inputs);
    assert_eq!(taps.post_layers, taps.layer_inputs);
}
