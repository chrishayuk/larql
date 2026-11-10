//! The lowered session's gate binding is checked by SOURCE. A fused-query gate (Qwen3.5/3.8) must be refused
//! by name: the encoder would otherwise bind `q_proj` as an ordinary gate matrix and run silently wrong.

use super::super::construct::refuse_unlowered_gate_source;
use crate::format::vindex3::encode::encode_system_unenforced as encode_system;
use crate::format::vindex3::fixtures::gated_q_f32_model;
use crate::format::vindex3::inspect::inspect_container;
use crate::format::vindex3::opplan::{plan_component_ops, ComponentOpPlan};
use larql_models::config::GateSource;

fn fused_gate_plan() -> ComponentOpPlan {
    let dir = tempfile::tempdir().unwrap();
    gated_q_f32_model(dir.path());
    let inventory = larql_models::inventory::build_inventory(dir.path()).unwrap();
    let container = tempfile::tempdir().unwrap();
    encode_system(&[("gated".to_string(), inventory)], container.path()).unwrap();
    let inspection = inspect_container(container.path(), false).unwrap();
    let outcome = plan_component_ops(&inspection, container.path(), "target").unwrap();
    outcome.plan.expect("the fused-gate fixture plans")
}

#[test]
fn a_gate_read_from_the_fused_query_projection_is_refused_by_name() {
    let plan = fused_gate_plan();
    let err = refuse_unlowered_gate_source(&plan).unwrap_err().to_string();
    assert!(err.contains("FusedQueryProjection"), "{err}");
    assert!(err.contains("layer 0"), "{err}");
}

/// The control: the same plan with its gate re-declared as reading the attention input (the form the
/// encoder does bind) passes this check, so the refusal above is about the source and nothing else.
#[test]
fn the_same_gate_declared_on_the_attention_input_passes() {
    let mut plan = fused_gate_plan();
    for layer in &mut plan.layers {
        if let Some(gate) = layer
            .attention
            .softmax_mut()
            .and_then(|a| a.output_gate.as_mut())
        {
            gate.spec.source = GateSource::AttentionInput;
        }
    }
    refuse_unlowered_gate_source(&plan).expect("an attention-input gate is the bound form");
}

/// A plan with no gate at all is unaffected.
#[test]
fn a_plan_without_a_gate_passes() {
    let mut plan = fused_gate_plan();
    for layer in &mut plan.layers {
        if let Some(a) = layer.attention.softmax_mut() {
            a.output_gate = None;
        }
    }
    refuse_unlowered_gate_source(&plan).expect("no gate, nothing to bind");
}
