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
use crate::format::vindex3::opplan::exec::continuation::plan_continuation_geometry;
use crate::format::vindex3::opplan::exec::kv::{KvState, RowKvState};
use crate::format::vindex3::opplan::exec::operands::OperandStore;
use crate::format::vindex3::opplan::exec::prepared::{ExecutionSlice, PreparedOperands};
use crate::format::vindex3::opplan::exec::reference::ReferenceBackend;
use crate::format::vindex3::opplan::exec::{
    execute_plan_streaming, execute_plan_streaming_in, execute_prepared_streaming_in,
    requires_continuation, PlaneEvent,
};
use crate::format::vindex3::opplan::{plan_component_ops, ComponentOpPlan};

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

#[test]
fn the_in_forms_run_a_stateful_stack_over_the_callers_state_and_agree() {
    let (_h, plan, store) = hybrid();
    let backend = ReferenceBackend::new();

    let mut loaded = RowKvState::default();
    let via_store = execute_plan_streaming_in(
        &plan,
        &store,
        &TOKENS,
        &backend,
        None,
        &mut ignore,
        &mut loaded,
    )
    .unwrap();

    let ops = PreparedOperands::load(&plan, &store, &backend, ExecutionSlice::Full).unwrap();
    let mut prepared = RowKvState::default();
    let via_prepared = execute_prepared_streaming_in(
        &plan,
        &ops,
        &TOKENS,
        &backend,
        None,
        &mut ignore,
        &mut prepared,
    )
    .unwrap();

    assert!(via_store.logits.is_some());
    assert_eq!(via_store.logits, via_prepared.logits);
    // The caller's state is what the recurrence ran in: it moved.
    let geometry = plan_continuation_geometry(&plan).unwrap();
    let layer = geometry
        .iter()
        .position(|g| g.recurrent().is_some())
        .expect("a recurrent layer");
    let cells = prepared
        .recurrent_state(layer)
        .unwrap()
        .buffer(0)
        .cells()
        .to_vec();
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
    let err = execute_prepared_streaming_in(
        &plan,
        &ops,
        &TOKENS,
        &backend,
        None,
        &mut ignore,
        &mut advanced,
    )
    .unwrap_err()
    .to_string();
    assert!(
        err.contains(&format!("already at position {}", TOKENS.len())),
        "{err}"
    );
    assert!(err.contains("resuming is prefill's job"), "{err}");
}
