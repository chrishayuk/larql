//! Rung 3a over the routed miniature: a packed bank is two expert-slice
//! operations requiring row access, a per-expert bank is a set of whole
//! projections, and the production loader's census holds exactly the bytes
//! the view lists for the FFN site. Lives beside the routed fixture because
//! that fixture is scoped here.

use super::fixture::{routed_fixture, EXPERTS};
use crate::format::vindex3::opplan::exec::backend::MatrixClass;
use crate::format::vindex3::opplan::exec::prepared::{ExecutionSlice, PreparedOperands};
use crate::format::vindex3::opplan::exec::production::ProductionBackend;
use crate::format::vindex3::opplan::planned::Operation;
use crate::format::vindex3::opplan::{ExpertBank, LayerFfn, OperandRef, SharedExpertOp};
use crate::format::vindex3::represent::codec::RequiredAccess;

const F32_WIDTH: usize = std::mem::size_of::<f32>();

#[test]
fn a_packed_bank_is_two_row_random_slices_and_the_census_agrees() {
    let fixture = routed_fixture();
    let plan = &fixture.plan;
    let slices: Vec<_> = plan
        .planned_operands()
        .into_iter()
        .filter(|p| p.operation == Operation::ExpertBankSlice)
        .collect();
    let routed_layers = plan
        .layers
        .iter()
        .filter(|l| l.ffn.as_ref().is_some_and(|f| f.routed().is_some()))
        .count();
    assert!(routed_layers > 0);
    assert_eq!(
        slices.len(),
        2 * routed_layers,
        "gate/up and down per routed layer"
    );
    assert!(slices.iter().all(|s| s.access == RequiredAccess::RowRandom));
    // The bank's stored operands, exactly — never the router or a bias.
    let ExpertBank::Packed { gate_up, down } = &fixture.op.bank else {
        panic!("the miniature packs its bank");
    };
    let named: Vec<&str> = slices.iter().map(|s| s.operand.tensor.as_str()).collect();
    assert!(named.contains(&gate_up.weights.tensor.as_str()));
    assert!(named.contains(&down.weights.tensor.as_str()));
    assert!(!named.contains(&fixture.op.router.tensor.as_str()));

    let census = PreparedOperands::load(
        plan,
        &fixture.store,
        &ProductionBackend::new(),
        ExecutionSlice::Full,
    )
    .unwrap()
    .residency_census();
    let ffn_bytes: usize = plan
        .planned_operands()
        .iter()
        .filter(|p| {
            matches!(
                p.operation,
                Operation::ExpertBankSlice | Operation::Project(MatrixClass::FfnProjection)
            )
        })
        .map(|p| p.elements() * F32_WIDTH)
        .sum();
    assert_eq!(census.ffn.widened_f32 + census.ffn.compact, ffn_bytes);
}

#[test]
fn a_per_expert_bank_is_a_set_of_whole_projections() {
    let fixture = routed_fixture();
    let mut plan = fixture.plan.clone();
    let synthetic = |name: &str| OperandRef {
        object: fixture.op.router.object.clone(),
        tensor: name.to_string(),
        dtype: "F32".into(),
        shape: vec![8, 4],
    };
    let per_expert = ExpertBank::PerExpert {
        gate: (0..EXPERTS)
            .map(|e| synthetic(&format!("experts.{e}.gate")))
            .collect(),
        up: (0..EXPERTS)
            .map(|e| synthetic(&format!("experts.{e}.up")))
            .collect(),
        down: (0..EXPERTS)
            .map(|e| synthetic(&format!("experts.{e}.down")))
            .collect(),
    };
    let mut op = fixture.op.clone();
    op.bank = per_expert;
    plan.layers[0].ffn = Some(LayerFfn::Routed(Box::new(op)));
    let listed: Vec<_> = plan
        .planned_operands()
        .into_iter()
        .filter(|p| p.operand.tensor.starts_with("experts."))
        .collect();
    assert_eq!(listed.len(), 3 * EXPERTS);
    assert!(listed.iter().all(
        |p| p.operation == Operation::Project(MatrixClass::FfnProjection)
            && p.access == RequiredAccess::Sequential
    ));
}

/// The plan executes a shared expert beside the routed ones, so the view
/// lists its three projections. The CPU loader does not bind them today —
/// only the Metal stack does — and this test pins that gap in bytes: the
/// view exceeds the CPU census by exactly the shared expert. When a CPU
/// loader binds it, this assertion is what changes, deliberately.
#[test]
fn a_shared_expert_is_three_projections_the_cpu_loader_does_not_yet_bind() {
    let fixture = routed_fixture();
    let mut plan = fixture.plan.clone();
    let mut op = fixture.op.clone();
    let synthetic = |name: &str, shape: Vec<usize>| OperandRef {
        object: op.router.object.clone(),
        tensor: name.to_string(),
        dtype: "F32".into(),
        shape,
    };
    let hidden = op.router.shape[1];
    let inter = 16;
    op.shared = Some(SharedExpertOp {
        intermediate_size: inter,
        activation: op.activation,
        gate_policy: op.gate_policy,
        gate: synthetic("shared.gate", vec![inter, hidden]),
        up: synthetic("shared.up", vec![inter, hidden]),
        down: synthetic("shared.down", vec![hidden, inter]),
        branch_gate: None,
    });
    plan.layers[0].ffn = Some(LayerFfn::Routed(Box::new(op)));
    let planned = plan.planned_operands();
    let shared: Vec<_> = planned
        .iter()
        .filter(|p| p.operand.tensor.starts_with("shared."))
        .collect();
    assert_eq!(shared.len(), 3);
    assert!(shared
        .iter()
        .all(|p| p.operation == Operation::Project(MatrixClass::FfnProjection)));
    let shared_bytes: usize = shared.iter().map(|p| p.elements() * F32_WIDTH).sum();
    assert_eq!(shared_bytes, 3 * inter * hidden * F32_WIDTH);

    // The synthetic operands are not in the container, and the CPU loader
    // never asks for them: preparation succeeds and the census is short by
    // exactly the shared expert.
    let census = PreparedOperands::load(
        &plan,
        &fixture.store,
        &ProductionBackend::new(),
        ExecutionSlice::Full,
    )
    .expect("the CPU loader ignores the shared expert")
    .residency_census();
    let ffn_planned: usize = planned
        .iter()
        .filter(|p| {
            matches!(
                p.operation,
                Operation::ExpertBankSlice | Operation::Project(MatrixClass::FfnProjection)
            )
        })
        .map(|p| p.elements() * F32_WIDTH)
        .sum();
    assert_eq!(
        ffn_planned - (census.ffn.widened_f32 + census.ffn.compact),
        shared_bytes
    );
}
