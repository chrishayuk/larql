//! What a KDA layer RETAINS, and at what precision — lift 2's F5.
//!
//! This file used to pin two refusals: continuation planning could not
//! size a KDA layer, and preparation would not bind one. Neither was ever
//! about a missing kernel — `exec::kda` has been parity-proven against
//! the banked oracle since P3d — but about a fact the schema could not
//! state: the precision to hold the recurrence at, which no checkpoint
//! declares. Lift 2 answers it from the reference (`fla`'s
//! `naive_recurrent_kda` holds fp32) and records the judgment in one
//! place, so what remains to pin here is the ANSWER.
//!
//! Execution end to end is the miniature witness in
//! `opplan/tests/kda_mla_exec.rs`; this file stays at the planning seam.
//!
//! The fixture is the Gated DeltaNet hybrid with one layer's attention
//! swapped for a KDA op. Swapping rather than encoding a KDA container
//! keeps this test about the geometry and not about KDA admission.

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

/// Continuation planning ANSWERS for a KDA layer, and says exactly what
/// it retains — lift 2's F5, on the side the schema could not state
/// before: four buffers, the `Dk × Dv` matrix per head and the three
/// convolution windows, all at the precision the reference computes at.
///
/// The precision is the whole point of pinning it here. No checkpoint
/// declares one, so this asserts a JUDGMENT — `fla`'s
/// `naive_recurrent_kda` holds fp32 — and a build that silently sized the
/// state at the model's bulk dtype would fail this line rather than
/// running the recurrence at a precision its author never chose.
#[test]
fn continuation_planning_sizes_a_kda_layer() {
    use crate::format::vindex3::opplan::exec::continuation::{
        LayerContinuationGeometry, StateInitialization,
    };
    use larql_models::inventory::report::RecurrentStateDtype;

    let (_container, mut plan, _store) = hybrid_plan_for_tests();
    plan.layers[0].attention = LayerAttention::Kda(Box::new(stub_kda()));

    let geometry = plan_continuation_geometry(&plan).expect("KDA's state geometry is stateable");
    let LayerContinuationGeometry::Recurrent(state) = &geometry[0] else {
        panic!(
            "a KDA layer keeps a recurrence, not rows: {:?}",
            geometry[0]
        );
    };
    assert_eq!(state.buffers.len(), 4, "the matrix and three conv windows");
    assert_eq!(state.buffers[0].shape, vec![HEADS, HEAD_DIM, HEAD_DIM]);
    assert_eq!(
        state.buffers[0].shape.iter().product::<usize>(),
        stub_kda().state_elements(),
        "the sized state must be the one the op declares"
    );
    for window in &state.buffers[1..] {
        assert_eq!(
            window.shape,
            vec![HEADS * HEAD_DIM, 3],
            "one conv window is `kernel - 1` deep: the current input is not history"
        );
    }
    for buffer in &state.buffers {
        assert_eq!(
            buffer.dtype,
            RecurrentStateDtype::Float32,
            "the reference recurrence is fp32; sizing it otherwise runs a different operator"
        );
        assert_eq!(buffer.initialization, StateInitialization::Zeros);
    }
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
