//! [`execute_streaming_position_exits`]: one traversal, observed. The planes it forwards are the traversal's own,
//! its output is `execute_plan`'s, and its exit at each position is the per-position exit
//! [`execute_slice_position_logits`] computes, with the vector the head read beside it.

use super::golden::{miniature_glimmer, G_TOKENS};
use crate::format::vindex3::encode::encode_system;
use crate::format::vindex3::inspect::inspect_container;
use crate::format::vindex3::opplan::exec::hyper_connection::Bundle;
use crate::format::vindex3::opplan::exec::operands::OperandStore;
use crate::format::vindex3::opplan::exec::position_exits::same_hidden;
use crate::format::vindex3::opplan::exec::prepared::ExecutionSlice;
use crate::format::vindex3::opplan::exec::reference::ReferenceBackend;
use crate::format::vindex3::opplan::exec::{
    execute_plan, execute_slice_position_logits, execute_streaming_position_exits, FinalState,
    PlaneEvent,
};
use crate::format::vindex3::opplan::{plan_component_ops, ComponentOpPlan};

fn fixture() -> (tempfile::TempDir, ComponentOpPlan, OperandStore) {
    let dir = tempfile::tempdir().unwrap();
    miniature_glimmer(dir.path());
    let inventory = larql_models::inventory::build_inventory(dir.path()).unwrap();
    let container = tempfile::tempdir().unwrap();
    encode_system(&[("mini-glimmer".to_string(), inventory)], container.path()).unwrap();
    let inspection = inspect_container(container.path(), false).unwrap();
    let outcome = plan_component_ops(&inspection, container.path(), "target").unwrap();
    let plan = outcome.plan.unwrap();
    let store = OperandStore::open(container.path(), &inspection).unwrap();
    (container, plan, store)
}

fn bits(values: &[f32]) -> Vec<u32> {
    values.iter().map(|v| v.to_bits()).collect()
}

#[test]
fn the_planes_output_and_exits_are_the_traversals_own() {
    let (_c, plan, store) = fixture();
    let backend = ReferenceBackend::new();
    let positions: Vec<usize> = (0..G_TOKENS.len()).rev().collect();
    let mut planes = Vec::new();
    let observed = execute_streaming_position_exits(
        &plan,
        &store,
        &G_TOKENS,
        &backend,
        None,
        &positions,
        &mut |event| {
            if let PlaneEvent::Layer { trace, .. } = event {
                planes.push(trace.post_layer.rows().to_vec());
            }
            Ok(())
        },
    )
    .unwrap();
    let control = execute_plan(&plan, &store, &G_TOKENS, &backend).unwrap();
    assert_eq!(planes.len(), control.layers.len());
    for (seen, want) in planes.iter().zip(&control.layers) {
        assert_eq!(seen.as_slice(), want.post_layer.rows());
    }
    assert_eq!(
        bits(observed.output.logits.as_ref().unwrap()),
        bits(control.logits.as_ref().unwrap())
    );
    let logits = execute_slice_position_logits(
        &plan,
        &store,
        &G_TOKENS,
        &backend,
        ExecutionSlice::Full,
        None,
        &positions,
    )
    .unwrap();
    assert_eq!(observed.exits.len(), positions.len());
    for (exit, want) in observed.exits.iter().zip(&logits) {
        assert_eq!(bits(&exit.logits), bits(want));
    }
    // Requested last-first: the first exit is the last position's, the traversal's own final row.
    assert_eq!(
        bits(&observed.exits[0].hidden),
        bits(control.final_hidden())
    );
}

#[test]
fn with_no_position_requested_nothing_is_exited() {
    let (_c, plan, store) = fixture();
    let backend = ReferenceBackend::new();
    let mut layers = 0usize;
    let observed = execute_streaming_position_exits(
        &plan,
        &store,
        &G_TOKENS,
        &backend,
        None,
        &[],
        &mut |event| {
            if matches!(event, PlaneEvent::Layer { .. }) {
                layers += 1;
            }
            Ok(())
        },
    )
    .unwrap();
    assert!(observed.exits.is_empty());
    assert_eq!(layers, plan.layers.len());
    assert!(observed.output.logits.is_some());
}

#[test]
fn a_position_outside_the_sequence_is_refused_before_anything_runs() {
    let (_c, plan, store) = fixture();
    let backend = ReferenceBackend::new();
    let mut events = 0usize;
    let err = execute_streaming_position_exits(
        &plan,
        &store,
        &G_TOKENS,
        &backend,
        None,
        &[G_TOKENS.len()],
        &mut |_| {
            events += 1;
            Ok(())
        },
    )
    .unwrap_err();
    assert!(err.to_string().contains("outside"), "{err}");
    assert_eq!(events, 0);
}

#[test]
fn the_last_hidden_must_be_the_traversals_own_exit_state() {
    let exit = FinalState::Hidden(vec![1.0, -0.0]);
    same_hidden(&[1.0, -0.0], &exit).unwrap();
    for wrong in [&[1.0, 0.0][..], &[1.0][..], &[1.0, -0.0, 2.0][..]] {
        let err = same_hidden(wrong, &exit).unwrap_err().to_string();
        assert!(err.contains("final-normed row differs"), "{err}");
    }
    // A run that ended on a bundle has no [hidden] exit to agree with.
    let bundle = Bundle::replicate(&[1.0], 2);
    assert!(same_hidden(&[1.0], &FinalState::Bundle(bundle)).is_err());
}
