//! Multi-axis rotary on a text sequence is plain rotary.
//!
//! The interpreter holds one scalar position, so M-RoPE's grid is
//! `(p, p, p)`: every section reads the same position, and the rotation
//! each pair receives is the one partial rotary over the same width would
//! give it. That equality is the witness — the section assignment still
//! runs, and a wrong section-to-pair mapping would not change the answer
//! only because every axis agrees, which is exactly the text case the
//! interpreter claims.

use crate::format::vindex3::fixtures::{dense_f32_model, encode_fixture_container, DENSE_HEAD_DIM};
use crate::format::vindex3::inspect::inspect_container;
use crate::format::vindex3::opplan::exec::backend::PlanBackend;
use crate::format::vindex3::opplan::exec::decode::DecodeSession;
use crate::format::vindex3::opplan::exec::operands::OperandStore;
use crate::format::vindex3::opplan::exec::production::ProductionBackend;
use crate::format::vindex3::opplan::exec::reference::ReferenceBackend;
use crate::format::vindex3::opplan::{plan_component_ops, ComponentOpPlan};
use larql_models::config::{PositionPolicy, RotaryFrequencyBasis};

const TOKENS: [u32; 4] = [3, 17, 60, 0];
const THETA: f64 = 10_000.0;
/// Every dimension rotates, so the width is the whole head.
const FULL_ROTARY: f64 = 1.0;
/// Temporal, height and width pairs: they must cover `head_dim / 2`.
const SECTION: [usize; 3] = [2, 1, 1];
/// Two arithmetic orders of the same rotation.
const TOLERANCE: f32 = 1e-5;

const _: () = assert!(
    SECTION[0] + SECTION[1] + SECTION[2] == DENSE_HEAD_DIM / 2,
    "the sections must cover every rotary pair"
);

fn fixture() -> (
    tempfile::TempDir,
    tempfile::TempDir,
    ComponentOpPlan,
    OperandStore,
) {
    let checkpoint = tempfile::tempdir().unwrap();
    let container = tempfile::tempdir().unwrap();
    encode_fixture_container(
        dense_f32_model,
        checkpoint.path(),
        container.path(),
        "mrope",
    );
    let inspection = inspect_container(container.path(), false).unwrap();
    let outcome = plan_component_ops(&inspection, container.path(), "target").unwrap();
    assert!(outcome.closed(), "defects: {:?}", outcome.defects);
    let store = OperandStore::open(container.path(), &inspection).unwrap();
    (checkpoint, container, outcome.plan.unwrap(), store)
}

fn with_position(plan: &ComponentOpPlan, position: PositionPolicy) -> ComponentOpPlan {
    let mut plan = plan.clone();
    for layer in &mut plan.layers {
        layer.attention.softmax_mut().unwrap().position = position;
    }
    plan
}

fn mrope(basis: RotaryFrequencyBasis, interleaved: bool) -> PositionPolicy {
    PositionPolicy::MRope {
        theta: THETA,
        rotary_fraction: FULL_ROTARY,
        basis,
        section: SECTION,
        interleaved,
    }
}

fn decode<B: PlanBackend>(
    plan: &ComponentOpPlan,
    store: &OperandStore,
    backend: &B,
) -> Result<Vec<Vec<f32>>, crate::error::VindexError> {
    let mut session = DecodeSession::new(plan, store, backend)?;
    TOKENS
        .iter()
        .map(|&t| Ok(session.step(t)?.logits.unwrap()))
        .collect()
}

fn max_abs(a: &[Vec<f32>], b: &[Vec<f32>]) -> f32 {
    a.iter()
        .flatten()
        .zip(b.iter().flatten())
        .map(|(x, y)| (x - y).abs())
        .fold(0.0, f32::max)
}

#[test]
fn text_sequence_mrope_is_partial_rotary_over_the_same_width() {
    let (_c, _k, plan, store) = fixture();
    let backend = ReferenceBackend::new();
    let rope = with_position(
        &plan,
        PositionPolicy::PartialRope {
            theta: THETA,
            rotary_fraction: FULL_ROTARY,
            basis: RotaryFrequencyBasis::RotaryWidth,
        },
    );
    let expected = decode(&rope, &store, &backend).unwrap();
    for interleaved in [false, true] {
        let multi = with_position(&plan, mrope(RotaryFrequencyBasis::RotaryWidth, interleaved));
        let actual = decode(&multi, &store, &backend).unwrap();
        let error = max_abs(&actual, &expected);
        assert!(error <= TOLERANCE, "interleaved={interleaved}: {error}");
    }
}

#[test]
fn reference_and_production_agree_on_mrope() {
    let (_c, _k, plan, store) = fixture();
    let multi = with_position(&plan, mrope(RotaryFrequencyBasis::RotaryWidth, false));
    let reference = decode(&multi, &store, &ReferenceBackend::new()).unwrap();
    let production = decode(&multi, &store, &ProductionBackend::new()).unwrap();
    let error = max_abs(&reference, &production);
    assert!(error <= TOLERANCE, "{error}");
}

#[test]
fn the_reference_refuses_mrope_over_a_head_width_basis() {
    let (_c, _k, plan, store) = fixture();
    let multi = with_position(&plan, mrope(RotaryFrequencyBasis::HeadWidth, false));
    assert!(decode(&multi, &store, &ReferenceBackend::new()).is_err());
}
