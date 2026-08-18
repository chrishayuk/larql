//! V3-F0 witness 3, G4.0: operand closure honours the per-layer K≡V
//! judgment both ways — the full layer needs no V operand and refuses a
//! stray one; the sliding layers still need theirs. (The hybrid FFN is
//! G4.1's rung; its unjudged-semantic defect is expected here and is not
//! what these tests read.)

use crate::format::vindex3::encode::encode_system;
use crate::format::vindex3::graph::OperandRole;
use crate::format::vindex3::inspect::inspect_container;
use crate::format::vindex3::opplan::{plan_component_ops, ClosureDefect, OpPlanOutcome};
use crate::format::vindex3::plan::tests_support::{gemma4_shaped_target_with, GEMMA4_FULL_LAYER};

const SLIDING_LAYER: usize = 0;

fn plan_variant(mutate_tensors: impl FnOnce(&mut Vec<(String, Vec<usize>)>)) -> OpPlanOutcome {
    let dir = tempfile::tempdir().unwrap();
    let inventory = gemma4_shaped_target_with(dir.path(), |_| {}, mutate_tensors);
    let named = vec![("gemma4-artifact".to_string(), inventory)];
    let out = tempfile::tempdir().unwrap();
    encode_system(&named, out.path()).unwrap();
    let inspection = inspect_container(out.path(), false).unwrap();
    plan_component_ops(&inspection, out.path(), "target").unwrap()
}

fn missing_v(outcome: &OpPlanOutcome, layer: usize) -> bool {
    outcome.defects.iter().any(|d| {
        matches!(d, ClosureDefect::MissingOperand { layer: l, role: OperandRole::AttnV } if *l == layer)
    })
}

/// The full layer ships no `v_proj` and closure does not ask for one;
/// every sliding layer's V is still required.
#[test]
fn a_k_eq_v_layer_needs_no_v_operand_and_a_sliding_layer_still_does() {
    let outcome = plan_variant(|_| {});
    assert!(
        !missing_v(&outcome, GEMMA4_FULL_LAYER),
        "{:?}",
        outcome.defects
    );
    assert!(!missing_v(&outcome, SLIDING_LAYER), "{:?}", outcome.defects);

    let outcome = plan_variant(|tensors| {
        tensors.retain(|(name, _)| {
            name != &format!("model.language_model.layers.{SLIDING_LAYER}.self_attn.v_proj.weight")
        });
    });
    assert!(missing_v(&outcome, SLIDING_LAYER), "{:?}", outcome.defects);
}

/// A `v_proj` on the K≡V layer is a stray: an operand whose op the layer
/// does not carry, named as the value projection.
#[test]
fn a_v_operand_on_a_k_eq_v_layer_is_a_stray() {
    let hidden = 64;
    let kv_rows = 24;
    let outcome = plan_variant(|tensors| {
        tensors.push((
            format!("model.language_model.layers.{GEMMA4_FULL_LAYER}.self_attn.v_proj.weight"),
            vec![kv_rows, hidden],
        ));
    });
    assert!(
        outcome.defects.iter().any(|d| matches!(
            d,
            ClosureDefect::OperandImpliesAbsentOp { tensor, required_primitive, .. }
                if tensor.contains(&format!("{GEMMA4_FULL_LAYER}.self_attn.v_proj"))
                    && required_primitive.contains("value projection")
        )),
        "{:?}",
        outcome.defects
    );
}
