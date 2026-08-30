//! A represented-but-unexecutable operator must **refuse**, not fall
//! through to another one.
//!
//! KDA is fully described by the IR — every operand bound, every dimension
//! stated — and no executor consumes it. Both places that turn a plan into
//! something runnable therefore have to say so. Silently preparing such a
//! layer as softmax, or sizing it as a KV cache, would run the wrong
//! operator on correctly-bound tensors: the failure the separate variant
//! exists to make impossible, reappearing one layer down.
//!
//! The fixture is the Gated DeltaNet hybrid with one layer's attention
//! swapped for a KDA op. Swapping rather than encoding a KDA container
//! keeps this test about the refusal and not about KDA admission.

use crate::format::vindex3::opplan::exec::continuation::plan_continuation_geometry;
use crate::format::vindex3::opplan::{KdaOp, LayerAttention, OperandRef};

use super::hybrid_traversal::hybrid_plan_for_tests;

const HEADS: usize = 2;
const HEAD_DIM: usize = 4;

fn stub_operand() -> OperandRef {
    OperandRef {
        object: "target.decoder_stack".to_string(),
        tensor: "0.self_attn.q_proj.weight".to_string(),
        dtype: "F32".to_string(),
        shape: vec![HEADS * HEAD_DIM, 8],
    }
}

fn stub_kda() -> KdaOp {
    let o = stub_operand;
    KdaOp {
        num_heads: HEADS,
        head_dim: HEAD_DIM,
        conv_kernel: 4,
        gate_rank: 4,
        gate_lower_bound: Some(-5.0),
        q_proj: o(),
        k_proj: o(),
        v_proj: o(),
        q_conv1d: o(),
        k_conv1d: o(),
        v_conv1d: o(),
        f_a_proj: o(),
        f_b_proj: o(),
        g_a_proj: o(),
        g_b_proj: o(),
        b_proj: o(),
        a_log: o(),
        dt_bias: o(),
        o_norm: o(),
        out_proj: o(),
    }
}

/// Continuation planning refuses, and names the state it would have had to
/// size — so the refusal carries the fact a future executor needs rather
/// than only a complaint.
#[test]
fn continuation_planning_refuses_a_kda_layer() {
    let (_container, mut plan, _store) = hybrid_plan_for_tests();
    plan.layers[0].attention = LayerAttention::Kda(Box::new(stub_kda()));

    let err =
        plan_continuation_geometry(&plan).expect_err("a KDA layer has no declared state precision");
    assert!(err.contains("KDA"), "{err}");
    assert!(
        err.contains(&stub_kda().state_elements().to_string()),
        "the refusal must name the state it could not size: {err}"
    );
    assert!(
        err.contains("precision"),
        "and why it refused rather than what it lacks: {err}"
    );
}

/// The unmodified fixture still plans, so the test above is about KDA and
/// not about the fixture having become unplannable.
#[test]
fn the_same_stack_without_a_kda_layer_still_plans() {
    let (_container, plan, _store) = hybrid_plan_for_tests();
    assert!(
        plan_continuation_geometry(&plan).is_ok(),
        "the Gated DeltaNet fixture must still plan"
    );
}

/// Preparing operands refuses a KDA layer too.
///
/// The second door to the same wrong answer: continuation planning sizes
/// state, this binds weights, and a layer that slipped past either would
/// be executed by whichever operator the loader happened to build.
#[test]
fn preparing_operands_refuses_a_kda_layer() {
    use crate::format::vindex3::opplan::exec::prepared::{ExecutionSlice, PreparedOperands};
    use crate::format::vindex3::opplan::exec::reference::ReferenceBackend;

    let (_container, mut plan, store) = hybrid_plan_for_tests();

    // The unmodified fixture prepares — so the refusal below is about KDA.
    PreparedOperands::load(&plan, &store, &ReferenceBackend, ExecutionSlice::Full)
        .expect("the Gated DeltaNet fixture prepares");

    plan.layers[0].attention = LayerAttention::Kda(Box::new(stub_kda()));
    let message =
        match PreparedOperands::load(&plan, &store, &ReferenceBackend, ExecutionSlice::Full) {
            Ok(_) => panic!("no executor consumes a KDA layer, so this must refuse"),
            Err(err) => err.to_string(),
        };
    assert!(message.contains("KDA"), "{message}");
    assert!(
        message.contains("not") && message.contains("executable"),
        "the refusal must say represented-but-not-executable: {message}"
    );
}
