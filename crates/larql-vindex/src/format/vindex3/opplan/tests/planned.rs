//! Rung 3a: the plan's own statement of what it executes through a
//! representation, and what each operation requires — checked against the
//! loader that binds those operands, byte for byte.

use std::collections::BTreeSet;

use super::conv_qkv::miniature_hybrid;
use super::encoded_fixture;
use crate::format::vindex3::fixtures::{
    dense_f32_model_with, encode_fixture_container, hybrid_lllf_f32_model, HeadStorage,
};
use crate::format::vindex3::fixtures_kimi::hybrid_kda_mla_f32_model;
use crate::format::vindex3::inspect::inspect_container;
use crate::format::vindex3::opplan::exec::backend::MatrixClass;
use crate::format::vindex3::opplan::exec::operands::OperandStore;
use crate::format::vindex3::opplan::exec::prepared::{ExecutionSlice, PreparedOperands};
use crate::format::vindex3::opplan::exec::reference::ReferenceBackend;
use crate::format::vindex3::opplan::planned::{Operation, PlannedOperand};
use crate::format::vindex3::opplan::{
    plan_component_ops, ComponentOpPlan, LayerAttention, LayerFfn,
};
use crate::format::vindex3::represent::codec::{RepresentationExtent, RequiredAccess};
use larql_models::config::GateSource;

const F32_WIDTH: usize = std::mem::size_of::<f32>();

/// A plan and the store it was planned over, sources kept alive.
struct Planned {
    _src: tempfile::TempDir,
    _container: tempfile::TempDir,
    plan: ComponentOpPlan,
    store: OperandStore,
}

fn planned(write: impl FnOnce(&std::path::Path), name: &str, component: &str) -> Planned {
    let src = tempfile::tempdir().unwrap();
    let container = tempfile::tempdir().unwrap();
    encode_fixture_container(write, src.path(), container.path(), name);
    let inspection = inspect_container(container.path(), false).unwrap();
    let plan = plan_component_ops(&inspection, container.path(), component)
        .unwrap()
        .plan
        .unwrap_or_else(|| panic!("{name} plans"));
    let store = OperandStore::open(container.path(), &inspection).unwrap();
    Planned {
        _src: src,
        _container: container,
        plan,
        store,
    }
}

/// Every fixture whose attention variants together cover the view's arms:
/// softmax with an output gate (Glimmer), gated delta (LLLF), Mamba2 and
/// conv-QKV (the OuteAI miniature), KDA and MLA (the Kimi miniature).
fn every_plan() -> Vec<(&'static str, Planned)> {
    let glimmer = {
        let fixture = encoded_fixture();
        let plan = plan_component_ops(&fixture.inspection, &fixture.root, "target")
            .unwrap()
            .plan
            .unwrap();
        let store = OperandStore::open(&fixture.root, &fixture.inspection).unwrap();
        Planned {
            _src: fixture._target_dir,
            _container: fixture.container,
            plan,
            store,
        }
    };
    vec![
        ("glimmer", glimmer),
        ("lllf", planned(hybrid_lllf_f32_model, "hybrid", "target")),
        ("mamba2attn", planned(miniature_hybrid, "oute", "target")),
        (
            "kda-mla",
            planned(hybrid_kda_mla_f32_model, "kimi", "target"),
        ),
    ]
}

fn key(p: &PlannedOperand) -> (String, String, &'static str) {
    (
        p.operand.object.clone(),
        p.operand.tensor.clone(),
        p.operation.name(),
    )
}

#[test]
fn every_operation_declares_its_access_and_every_extent_is_terminal() {
    for (name, fixture) in every_plan() {
        let planned = fixture.plan.planned_operands();
        assert!(!planned.is_empty(), "{name}");
        for p in &planned {
            assert_eq!(
                p.access,
                p.operation.access(),
                "{name}: {}",
                p.operand.tensor
            );
            assert_eq!(p.extent, RepresentationExtent::TERMINAL, "{name}");
            let expected = match p.operation {
                Operation::Embed | Operation::ExpertBankSlice => RequiredAccess::RowRandom,
                Operation::Project(_) | Operation::OutputHead => RequiredAccess::Sequential,
            };
            assert_eq!(p.access, expected, "{name}: {}", p.operation.name());
        }
        let distinct: BTreeSet<_> = planned.iter().map(key).collect();
        assert_eq!(
            distinct.len(),
            planned.len(),
            "{name}: an operand is listed twice for one operation"
        );
    }
}

/// The count is derived from the plan's structure, variant by variant, so
/// the mapping rule is pinned; the bytes-level check below is what proves
/// the rule matches the loader.
#[test]
fn each_variant_lists_exactly_its_matrices_in_loader_order() {
    for (name, fixture) in every_plan() {
        let plan = &fixture.plan;
        let planned = plan.planned_operands();
        let mut expected = usize::from(plan.embedding.is_some());
        let mut seen = BTreeSet::new();
        for layer in &plan.layers {
            let per_attention = match &layer.attention {
                LayerAttention::Softmax(op) => {
                    seen.insert("softmax");
                    4 + usize::from(
                        op.output_gate
                            .as_ref()
                            .is_some_and(|g| g.spec.source != GateSource::FusedQueryProjection),
                    )
                }
                LayerAttention::GatedDelta(_) => {
                    seen.insert("gated-delta");
                    5
                }
                LayerAttention::Mamba2(_) => {
                    seen.insert("mamba2");
                    2
                }
                LayerAttention::ConvQkv(_) => {
                    seen.insert("conv-qkv");
                    2
                }
                LayerAttention::Kda(_) => {
                    seen.insert("kda");
                    4
                }
                LayerAttention::Mla(_) => {
                    seen.insert("mla");
                    4
                }
            };
            let per_ffn = match &layer.ffn {
                None => 0,
                Some(LayerFfn::Dense(op)) => 2 + usize::from(op.gate.is_some()),
                Some(LayerFfn::Routed(_)) => 2,
                Some(LayerFfn::Hybrid(op)) => 2 + usize::from(op.dense.gate.is_some()) + 2,
            };
            expected += per_attention + per_ffn;
        }
        expected += usize::from(plan.output.is_some());
        assert_eq!(planned.len(), expected, "{name}");
        assert_eq!(
            planned.first().map(|p| p.operation),
            Some(Operation::Embed),
            "{name}"
        );
        assert_eq!(
            planned.last().map(|p| p.operation),
            Some(Operation::OutputHead),
            "{name}"
        );
        eprintln!("{name}: {} planned operands over {:?}", planned.len(), seen);
    }
    // Together the fixtures reach every arm, or this test proves less than
    // it looks like it does.
    let reached: BTreeSet<&str> = every_plan()
        .iter()
        .flat_map(|(_, f)| {
            f.plan.layers.iter().map(|l| match &l.attention {
                LayerAttention::Softmax(_) => "softmax",
                LayerAttention::GatedDelta(_) => "gated-delta",
                LayerAttention::Mamba2(_) => "mamba2",
                LayerAttention::ConvQkv(_) => "conv-qkv",
                LayerAttention::Kda(_) => "kda",
                LayerAttention::Mla(_) => "mla",
            })
        })
        .collect();
    assert_eq!(
        reached,
        ["softmax", "gated-delta", "mamba2", "conv-qkv", "kda", "mla"]
            .into_iter()
            .collect()
    );
}

/// The reference loader widens every matrix to f32. So the bytes the census
/// holds at the matrix sites must be exactly four per planned element —
/// nothing the loader binds is missing from the view, and nothing in the
/// view is a matrix the loader does not bind. Glue is the census's own
/// site and is not planned.
#[test]
fn the_view_accounts_for_every_matrix_byte_the_reference_loader_makes_resident() {
    for (name, fixture) in every_plan() {
        let planned = fixture.plan.planned_operands();
        let census = PreparedOperands::load(
            &fixture.plan,
            &fixture.store,
            &ReferenceBackend::new(),
            ExecutionSlice::Full,
        )
        .unwrap_or_else(|e| panic!("{name}: {e}"))
        .residency_census();
        let bytes = |op: fn(&PlannedOperand) -> bool| -> usize {
            planned
                .iter()
                .filter(|p| op(p))
                .map(|p| p.elements() * F32_WIDTH)
                .sum()
        };
        assert_eq!(
            census.compact(),
            0,
            "{name}: the reference backend widens everything"
        );
        assert_eq!(
            census.embedding.widened_f32,
            bytes(|p| p.operation == Operation::Embed),
            "{name}: embedding"
        );
        assert_eq!(
            census.head.widened_f32,
            bytes(|p| p.operation == Operation::OutputHead),
            "{name}: head"
        );
        assert_eq!(
            census.ffn.widened_f32,
            bytes(|p| matches!(
                p.operation,
                Operation::Project(MatrixClass::FfnProjection) | Operation::ExpertBankSlice
            )),
            "{name}: ffn"
        );
        // Attention and recurrence projections are one operation class in
        // the view and two sites in the census (the census separates "the
        // model attends" from "the model recurs"); together they must match.
        assert_eq!(
            census.attention.widened_f32 + census.delta.widened_f32,
            bytes(|p| p.operation == Operation::Project(MatrixClass::AttentionProjection)),
            "{name}: attention + delta"
        );
    }
}

#[test]
fn a_tied_head_is_two_operations_over_one_operand() {
    let fixture = planned(
        |d| dense_f32_model_with(d, HeadStorage::Tied),
        "tied",
        "target",
    );
    let planned = fixture.plan.planned_operands();
    let embed = planned
        .iter()
        .find(|p| p.operation == Operation::Embed)
        .unwrap();
    let head = planned
        .iter()
        .find(|p| p.operation == Operation::OutputHead)
        .unwrap();
    assert_eq!(
        embed.operand.object, head.operand.object,
        "one stored object"
    );
    assert_ne!(
        embed.access, head.access,
        "two operations, two requirements"
    );
    // And the loader binds it twice — the census says so in bytes.
    let census = PreparedOperands::load(
        &fixture.plan,
        &fixture.store,
        &ReferenceBackend::new(),
        ExecutionSlice::Full,
    )
    .unwrap()
    .residency_census();
    assert_eq!(census.embedding.widened_f32, census.head.widened_f32);
    assert_eq!(census.head.widened_f32, head.elements() * F32_WIDTH);
}

/// The view is computed, never serialised: the plan a container carries and
/// `vindex3 plan` prints is the same bytes it was before this rung.
#[test]
fn the_plan_s_wire_shape_carries_none_of_this() {
    for (name, fixture) in every_plan() {
        let json = serde_json::to_string(&fixture.plan).unwrap();
        for absent in [
            "\"planned_operands\"",
            "\"operation\":",
            "\"required_access\"",
            "\"extent\":",
        ] {
            assert!(
                !json.contains(absent),
                "{name}: the wire shape grew a `{absent}` field"
            );
        }
        assert!(!fixture.plan.planned_operands().is_empty(), "{name}");
    }
}
