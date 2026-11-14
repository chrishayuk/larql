//! Per-position logits from one causal prefill
//! ([`execute_slice_position_logits`]) against the per-prefix method it
//! replaces: on the reference backend, each position's row is BIT-EXACT
//! the last row of a fresh prefill of that prefix, because causal
//! attention at position `p` reads keys `0..=p` only and the interpreter
//! sums them in the same order either way.

use super::golden::{miniature_glimmer, G_TOKENS};
use crate::format::vindex3::encode::encode_system;
use crate::format::vindex3::inspect::inspect_container;
use crate::format::vindex3::opplan::exec::operands::OperandStore;
use crate::format::vindex3::opplan::exec::position_logits::{exit_rows, same_as_exit};
use crate::format::vindex3::opplan::exec::prepared::ExecutionSlice;
use crate::format::vindex3::opplan::exec::reference::ReferenceBackend;
use crate::format::vindex3::opplan::exec::{execute_plan, execute_slice_position_logits, Plane};
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

#[test]
fn every_position_equals_a_fresh_prefill_of_its_prefix_bit_for_bit() {
    let (_c, plan, store) = fixture();
    let backend = ReferenceBackend::new();
    let positions: Vec<usize> = (0..G_TOKENS.len()).collect();
    let rows = execute_slice_position_logits(
        &plan,
        &store,
        &G_TOKENS,
        &backend,
        ExecutionSlice::Full,
        None,
        &positions,
    )
    .unwrap();
    for &p in &positions {
        let control = execute_plan(&plan, &store, &G_TOKENS[..=p], &backend)
            .unwrap()
            .logits
            .unwrap();
        assert!(
            rows[p]
                .iter()
                .zip(&control)
                .all(|(a, b)| a.to_bits() == b.to_bits()),
            "position {p} differs from its own prefix's prefill"
        );
    }
}

#[test]
fn a_position_outside_the_sequence_is_refused() {
    let (_c, plan, store) = fixture();
    let backend = ReferenceBackend::new();
    let err = execute_slice_position_logits(
        &plan,
        &store,
        &G_TOKENS,
        &backend,
        ExecutionSlice::Full,
        None,
        &[G_TOKENS.len()],
    )
    .unwrap_err();
    assert!(err.to_string().contains("outside"), "{err}");
}

#[test]
fn only_a_single_stream_plane_has_per_position_rows() {
    assert_eq!(
        exit_rows(Some(Plane::Rows(vec![vec![1.0]]))).unwrap(),
        vec![vec![1.0]]
    );
    assert!(exit_rows(Some(Plane::Bundles(Vec::new()))).is_err());
    assert!(exit_rows(Some(Plane::Histories(Vec::new()))).is_err());
    assert!(exit_rows(None).is_err());
}

#[test]
fn the_last_row_must_be_the_traversals_own_exit() {
    assert!(same_as_exit(&[1.0, 2.0], Some(&[1.0, 2.0])).is_ok());
    assert!(same_as_exit(&[1.0, 2.0], Some(&[1.0, 2.5])).is_err());
    assert!(same_as_exit(&[1.0], None).is_err());
}

/// Two quiet NaNs with different payloads, and the two zeros: equal or
/// unequal as numbers in ways that are not how their bits compare.
const NAN_PAYLOAD_A: u32 = 0x7fc0_0001;
const NAN_PAYLOAD_B: u32 = 0x7fc0_0002;

/// The self-check is on bit patterns, as "bit for bit" says: it decides
/// whether a per-position exit is refused, so each case is pinned.
#[test]
fn the_exit_self_check_compares_bit_patterns() {
    use crate::format::vindex3::opplan::exec::position_logits::same_as_exit;
    let nan_a = f32::from_bits(NAN_PAYLOAD_A);
    let nan_b = f32::from_bits(NAN_PAYLOAD_B);
    let refused = |row: &[f32], exit: Option<&[f32]>| {
        let error = same_as_exit(row, exit).unwrap_err().to_string();
        assert!(
            error.contains("differ from the traversal's own exit"),
            "{error}"
        );
    };

    // The same bits are the same exit, finite or not.
    same_as_exit(&[1.5, -2.0], Some(&[1.5, -2.0])).unwrap();
    same_as_exit(&[1.5, nan_a], Some(&[1.5, nan_a])).unwrap();
    same_as_exit(&[f32::INFINITY, -0.0], Some(&[f32::INFINITY, -0.0])).unwrap();
    same_as_exit(&[], Some(&[])).unwrap();

    // NaNs of different payloads are different exits.
    refused(&[1.5, nan_a], Some(&[1.5, nan_b]));
    // The zeros are equal as numbers and are not the same bits.
    refused(&[0.0], Some(&[-0.0]));
    refused(&[-0.0], Some(&[0.0]));
    // A different value, a different length either way, and no exit at all.
    refused(&[1.5, -2.0], Some(&[1.5, -2.5]));
    refused(&[1.5], Some(&[1.5, -2.0]));
    refused(&[1.5, -2.0], Some(&[1.5]));
    refused(&[1.5], None);
}
