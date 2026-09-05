//! QW-1 refusal integrity: a plan containing Gated DeltaNet layers is
//! refused BEFORE any layer output is committed.
//!
//! The "before" is the whole point. An error raised after some layers had
//! already executed would still be architecturally wrong: it would prove
//! the runtime can partially realise a model whose semantics it cannot
//! complete, and a caller holding those planes has no way to know they
//! describe 16 of 64 layers rather than the model.
//!
//! This becomes load-bearing the moment the builder emits real
//! `GatedDelta` layers, which it now does for `linear_attention` layers.
//! Qwen3.8-27B is 48 of 64 such layers, so "run the softmax ones" is not
//! a degraded mode — it is a different model.

use super::*;
use crate::format::vindex3::encode::encode_system;
use crate::format::vindex3::inspect::inspect_container;
use crate::format::vindex3::opplan::exec::operands::OperandStore;
use crate::format::vindex3::opplan::exec::reference::ReferenceBackend;
use crate::format::vindex3::opplan::exec::{execute_plan, execute_plan_streaming, PlaneEvent};
use crate::format::vindex3::opplan::{
    plan_component_ops, GatedDeltaOp, LayerAttention, OperandRef,
};

/// Qwen3.8's real geometry, so the op under test is the shape that ships.
const KEY_HEADS: usize = 16;
const VALUE_HEADS: usize = 48;
const KEY_HEAD_DIM: usize = 128;
const VALUE_HEAD_DIM: usize = 128;
const CONV_KERNEL: usize = 4;

fn operand(name: &str) -> OperandRef {
    OperandRef {
        object: "target.decoder_stack".into(),
        tensor: name.into(),
        dtype: "F32".into(),
        shape: vec![1],
    }
}

fn gated_delta() -> GatedDeltaOp {
    GatedDeltaOp {
        num_key_heads: KEY_HEADS,
        num_value_heads: VALUE_HEADS,
        key_head_dim: KEY_HEAD_DIM,
        value_head_dim: VALUE_HEAD_DIM,
        conv_kernel: CONV_KERNEL,
        state_dtype: None,
        in_proj_qkv: operand("0.linear_attn.in_proj_qkv.weight"),
        in_proj_a: operand("0.linear_attn.in_proj_a.weight"),
        in_proj_b: operand("0.linear_attn.in_proj_b.weight"),
        in_proj_z: operand("0.linear_attn.in_proj_z.weight"),
        conv1d: operand("0.linear_attn.conv1d.weight"),
        a_log: operand("0.linear_attn.A_log"),
        dt_bias: operand("0.linear_attn.dt_bias"),
        norm: operand("0.linear_attn.norm.weight"),
        out_proj: operand("0.linear_attn.out_proj.weight"),
    }
}

/// A wholly-softmax plan from the dense fixture.
fn softmax_plan() -> (
    tempfile::TempDir,
    crate::format::vindex3::opplan::ComponentOpPlan,
    OperandStore,
) {
    let dir = tempfile::tempdir().unwrap();
    dense_f32_model(dir.path());
    let inventory = larql_models::inventory::build_inventory(dir.path()).unwrap();
    let container = tempfile::tempdir().unwrap();
    encode_system(&[("dense".to_string(), inventory)], container.path()).unwrap();
    let inspection = inspect_container(container.path(), false).unwrap();
    let plan = plan_component_ops(&inspection, container.path(), "target")
        .unwrap()
        .plan
        .unwrap();
    let store = OperandStore::open(container.path(), &inspection).unwrap();
    (container, plan, store)
}

/// A hybrid plan: the softmax stack with its LAST layer swapped for a
/// recurrence, mirroring what the builder now produces on Qwen3.8.
///
/// Last, on purpose. If the executor refused only when it reached the
/// offending layer, every earlier layer would already have run — the
/// failure this file exists to catch. Verified: bypassing the up-front
/// refusal makes `a_hybrid_plan_is_refused_before_any_layer_output` fail
/// with `["embedded", "layer 0"]`, while the one-shot test still passes,
/// because "an error occurred" is a weaker claim than "nothing ran".
fn hybrid_plan() -> (
    tempfile::TempDir,
    crate::format::vindex3::opplan::ComponentOpPlan,
    OperandStore,
) {
    let (container, mut plan, store) = softmax_plan();
    let last = plan.layers.len() - 1;
    plan.layers[last].attention = LayerAttention::GatedDelta(Box::new(gated_delta()));
    (container, plan, store)
}

#[test]
fn a_hybrid_plan_is_refused_before_any_layer_output() {
    let (_container, plan, store) = hybrid_plan();
    let backend = ReferenceBackend;

    let mut events: Vec<String> = Vec::new();
    let result =
        execute_plan_streaming(&plan, &store, &[1u32, 2, 3], &backend, None, &mut |event| {
            events.push(match event {
                PlaneEvent::Embedded(_) => "embedded".to_string(),
                PlaneEvent::Layer { index, .. } => format!("layer {index}"),
                PlaneEvent::HyperConnectionSite(_) => "site".to_string(),
            });
            Ok(())
        });

    let err = result.expect_err("a plan naming operands the container lacks must not execute");
    // The claim this test has always made, unchanged: nothing is emitted
    // before the refusal. The recurrence is in the LAST layer, so a lazy
    // refusal would have committed three layers first.
    assert!(
        events.is_empty(),
        "refused only AFTER committing output: {events:?}"
    );
    // What CHANGED at QW-3.6b: a recurrence is no longer refused for
    // being a recurrence — the traversal runs them. This fixture injects
    // a `GatedDelta` op into a plan whose container holds no
    // `linear_attn.*` tensors, so what is now wrong with it is the
    // disagreement between plan and operands, and the refusal names that.
    // The recurrence-execution claims moved to
    // `tests::hybrid_traversal`, which runs a real encoded hybrid stack.
    let message = err.to_string();
    assert!(
        message.contains("in_proj_qkv"),
        "the refusal must name the operand it could not find, not a downstream \
         symptom; got: {message}"
    );
}

/// The same refusal on the non-streaming entry point, which is what most
/// callers use.
#[test]
fn the_one_shot_entry_point_refuses_too() {
    let (_container, plan, store) = hybrid_plan();
    let err = execute_plan(&plan, &store, &[1u32, 2, 3], &ReferenceBackend)
        .expect_err("a plan naming operands the container lacks must not execute");
    assert!(err.to_string().contains("in_proj_qkv"), "{err}");
}

/// The regression half: an unmodified softmax plan still runs. Without
/// this, a refusal that fired on everything would pass the tests above.
#[test]
fn a_pure_softmax_plan_is_untouched() {
    let (_container, plan, store) = softmax_plan();
    execute_plan(&plan, &store, &[1u32, 2, 3], &ReferenceBackend)
        .expect("a wholly-softmax plan must still execute");
}
