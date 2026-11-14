//! **C3 of CONTINUATION-PLUGIN-1: a one-shot forward has no hidden
//! continuation provider.**
//!
//! Before C3 the one-shot traversal manufactured a `row/v1` state for any
//! stack with state beyond softmax attention. Now it refuses without
//! caller state, the `_in` forms take the caller's selection, and
//! `requires_continuation` is the one rule for when state is needed.
//! These tests pin each of those in this crate. The callers that reach
//! them (CLI, inference) cover them in their own crates, which this
//! crate's coverage does not count.

use crate::error::VindexError;
use crate::format::vindex3::fixtures::{dense_f32_model, encode_fixture_container};
use crate::format::vindex3::inspect::inspect_container;
use crate::format::vindex3::opplan::exec::continuation::{
    plan_continuation_geometry, LatentKvRows, LayerContinuationGeometry, RecurrentState,
};
use crate::format::vindex3::opplan::exec::kv::{
    ContinuationError, ContinuationProvider, LayerKvGeometry, RowKvState,
};
use crate::format::vindex3::opplan::exec::kv_view::KvView;
use crate::format::vindex3::opplan::exec::operands::OperandStore;
use crate::format::vindex3::opplan::exec::prepared::{ExecutionSlice, PreparedOperands};
use crate::format::vindex3::opplan::exec::reference::ReferenceBackend;
use crate::format::vindex3::opplan::exec::{
    execute_plan_streaming, execute_plan_streaming_in, execute_prepared_streaming_in,
    requires_continuation, PlaneEvent,
};
use crate::format::vindex3::opplan::{plan_component_ops, ComponentOpPlan};
use std::sync::{Arc, Mutex};

use super::draft_slice::hybrid;

const TOKENS: [u32; 3] = [1, 2, 3];

fn dense() -> (tempfile::TempDir, ComponentOpPlan, OperandStore) {
    let checkpoint = tempfile::tempdir().unwrap();
    let container = tempfile::tempdir().unwrap();
    encode_fixture_container(
        dense_f32_model,
        checkpoint.path(),
        container.path(),
        "dense",
    );
    let inspection = inspect_container(container.path(), false).unwrap();
    let plan = plan_component_ops(&inspection, container.path(), "target")
        .unwrap()
        .plan
        .unwrap();
    let store = OperandStore::open(container.path(), &inspection).unwrap();
    (container, plan, store)
}

fn ignore(_: PlaneEvent) -> Result<(), VindexError> {
    Ok(())
}

#[test]
fn requires_continuation_is_true_exactly_when_a_layer_keeps_state_beyond_softmax() {
    let (_d, dense_plan, _) = dense();
    let (_h, hybrid_plan, _) = hybrid();
    assert!(!requires_continuation(&dense_plan));
    assert!(requires_continuation(&hybrid_plan));
}

#[test]
fn a_one_shot_forward_without_state_still_runs_a_softmax_stack() {
    let (_d, plan, store) = dense();
    let out = execute_plan_streaming(
        &plan,
        &store,
        &TOKENS,
        &ReferenceBackend::new(),
        None,
        &mut ignore,
    )
    .unwrap();
    assert!(out.logits.is_some());
}

#[test]
fn a_one_shot_forward_without_state_refuses_a_stateful_stack_naming_the_layer() {
    let (_h, plan, store) = hybrid();
    let first = plan
        .layers
        .iter()
        .position(|l| l.attention.softmax().is_none())
        .expect("the hybrid keeps state");
    let err = execute_plan_streaming(
        &plan,
        &store,
        &TOKENS,
        &ReferenceBackend::new(),
        None,
        &mut ignore,
    )
    .unwrap_err()
    .to_string();
    assert!(err.contains(&format!("layer {first}")), "{err}");
    assert!(err.contains("`_in` form"), "{err}");
}

/// A caller-chosen provider that forwards every call to a real one and,
/// when dropped, records one recurrent layer's cells where the test can
/// still read them.
///
/// The `_in` forms CONSUME their provider (RESIDUAL-BUS-2): a one-shot
/// traversal leaves nothing to continue from. So a test can no longer
/// inspect the provider afterwards, and this is how it still sees what
/// the traversal ran in. The record is written only on drop, so its
/// presence also proves the provider was consumed.
struct Spy {
    inner: RowKvState,
    layer: usize,
    cells: Arc<Mutex<Option<Vec<f32>>>>,
}

impl ContinuationProvider for Spy {
    fn prepare(&mut self, layers: &[LayerKvGeometry]) {
        self.inner.prepare(layers)
    }
    fn append(&mut self, layer: usize, key: Vec<f32>, value: Vec<f32>) {
        self.inner.append(layer, key, value)
    }
    fn rows(&self, layer: usize) -> KvView<'_> {
        self.inner.rows(layer)
    }
    fn prepare_layer(&mut self, layer: usize) {
        self.inner.prepare_layer(layer)
    }
    fn position(&self) -> usize {
        self.inner.position()
    }
    fn set_position(&mut self, position: usize) {
        self.inner.set_position(position)
    }
    fn invalidate(&mut self, reason: &str) {
        self.inner.invalidate(reason)
    }
    fn invalidated(&self) -> Option<&str> {
        self.inner.invalidated()
    }
    fn prepare_continuation(
        &mut self,
        layers: &[LayerContinuationGeometry],
    ) -> Result<(), ContinuationError> {
        self.inner.prepare_continuation(layers)
    }
    fn recurrent_state(&mut self, layer: usize) -> Result<&mut RecurrentState, ContinuationError> {
        self.inner.recurrent_state(layer)
    }
    fn latent_state(&mut self, layer: usize) -> Result<&mut LatentKvRows, ContinuationError> {
        self.inner.latent_state(layer)
    }
}

impl Drop for Spy {
    fn drop(&mut self) {
        let cells = self
            .inner
            .recurrent_state(self.layer)
            .ok()
            .map(|state| state.buffer(0).cells().to_vec());
        *self.cells.lock().unwrap() = cells;
    }
}

#[test]
fn the_in_forms_run_a_stateful_stack_over_the_callers_state_and_agree() {
    let (_h, plan, store) = hybrid();
    let backend = ReferenceBackend::new();

    let via_store = execute_plan_streaming_in(
        &plan,
        &store,
        &TOKENS,
        &backend,
        None,
        &mut ignore,
        Box::new(RowKvState::default()),
    )
    .unwrap();

    let geometry = plan_continuation_geometry(&plan).unwrap();
    let layer = geometry
        .iter()
        .position(|g| g.recurrent().is_some())
        .expect("a recurrent layer");
    let cells = Arc::new(Mutex::new(None));
    let spy = Spy {
        inner: RowKvState::default(),
        layer,
        cells: Arc::clone(&cells),
    };
    let ops = PreparedOperands::load(&plan, &store, &backend, ExecutionSlice::Full).unwrap();
    let via_prepared = execute_prepared_streaming_in(
        &plan,
        &ops,
        &TOKENS,
        &backend,
        None,
        &mut ignore,
        Box::new(spy),
    )
    .unwrap();

    assert!(via_store.logits.is_some());
    assert_eq!(via_store.logits, via_prepared.logits);
    // Recorded on drop: the traversal consumed the caller's provider...
    let cells = cells
        .lock()
        .unwrap()
        .take()
        .expect("the provider was consumed, and its recurrence recorded");
    // ...and the recurrence ran in it: it moved.
    assert!(
        cells.iter().any(|v| *v != 0.0),
        "the caller's state never moved"
    );
}

#[test]
fn a_one_shot_forward_refuses_state_that_has_already_advanced() {
    let (_h, plan, store) = hybrid();
    let backend = ReferenceBackend::new();
    let ops = PreparedOperands::load(&plan, &store, &backend, ExecutionSlice::Full).unwrap();
    let mut advanced = RowKvState::default();
    advanced.set_position(TOKENS.len());
    let advanced = Box::new(advanced);
    let err =
        execute_prepared_streaming_in(&plan, &ops, &TOKENS, &backend, None, &mut ignore, advanced)
            .unwrap_err()
            .to_string();
    assert!(
        err.contains(&format!("already at position {}", TOKENS.len())),
        "{err}"
    );
    assert!(err.contains("resuming is prefill's job"), "{err}");
}
