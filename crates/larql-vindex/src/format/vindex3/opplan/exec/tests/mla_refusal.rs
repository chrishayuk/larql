//! A represented-but-unexecutable operator must **refuse**, not fall
//! through to another one — MLA's own instance of [`super::kda_refusal`].
//!
//! MLA is fully described by the IR — every operand bound, every
//! dimension stated, including a real (if compressed) KV cache width —
//! and no executor consumes it yet. Both places that turn a plan into
//! something runnable therefore have to say so, the same as for KDA.
//!
//! The fixture is the Gated DeltaNet hybrid with one layer's attention
//! swapped for an MLA op. Swapping rather than encoding an MLA container
//! keeps this test about the refusal and not about MLA admission.

use crate::format::vindex3::opplan::exec::continuation::plan_continuation_geometry;
use crate::format::vindex3::opplan::{LayerAttention, MlaOp, OperandRef};

use super::hybrid_traversal::hybrid_plan_for_tests;

const NUM_HEADS: usize = 2;
const KV_LORA_RANK: usize = 8;
const QK_NOPE_HEAD_DIM: usize = 4;
const QK_ROPE_HEAD_DIM: usize = 2;
const V_HEAD_DIM: usize = 4;

fn stub_operand() -> OperandRef {
    OperandRef {
        object: "target.decoder_stack".to_string(),
        tensor: "0.self_attn.q_proj.weight".to_string(),
        dtype: "F32".to_string(),
        shape: vec![NUM_HEADS * (QK_NOPE_HEAD_DIM + QK_ROPE_HEAD_DIM), 8],
    }
}

fn stub_mla() -> MlaOp {
    let o = stub_operand;
    MlaOp {
        num_heads: NUM_HEADS,
        kv_lora_rank: KV_LORA_RANK,
        qk_nope_head_dim: QK_NOPE_HEAD_DIM,
        qk_rope_head_dim: QK_ROPE_HEAD_DIM,
        v_head_dim: V_HEAD_DIM,
        q_proj: o(),
        kv_a_proj: o(),
        kv_b_proj: o(),
        kv_a_norm: o(),
        out_proj: o(),
    }
}

/// Continuation planning refuses, and names the compressed cache width it
/// would have had to size — so the refusal carries the fact a future
/// executor needs rather than only a complaint.
#[test]
fn continuation_planning_refuses_an_mla_layer() {
    let (_container, mut plan, _store) = hybrid_plan_for_tests();
    plan.layers[0].attention = LayerAttention::Mla(Box::new(stub_mla()));

    let err =
        plan_continuation_geometry(&plan).expect_err("no continuation geometry exists for MLA yet");
    assert!(err.contains("MLA"), "{err}");
    assert!(
        err.contains(&stub_mla().compressed_kv_width().to_string()),
        "the refusal must name the compressed width it could not size: {err}"
    );
}

/// Preparing operands refuses an MLA layer too.
///
/// The second door to the same wrong answer: continuation planning sizes
/// the cache, this binds weights, and a layer that slipped past either
/// would be executed by whichever operator the loader happened to build.
#[test]
fn preparing_operands_refuses_an_mla_layer() {
    use crate::format::vindex3::opplan::exec::prepared::{ExecutionSlice, PreparedOperands};
    use crate::format::vindex3::opplan::exec::reference::ReferenceBackend;

    let (_container, mut plan, store) = hybrid_plan_for_tests();

    // The unmodified fixture prepares — so the refusal below is about MLA.
    PreparedOperands::load(&plan, &store, &ReferenceBackend, ExecutionSlice::Full)
        .expect("the Gated DeltaNet fixture prepares");

    plan.layers[0].attention = LayerAttention::Mla(Box::new(stub_mla()));
    let message =
        match PreparedOperands::load(&plan, &store, &ReferenceBackend, ExecutionSlice::Full) {
            Ok(_) => panic!("no executor consumes an MLA layer, so this must refuse"),
            Err(err) => err.to_string(),
        };
    assert!(message.contains("MLA"), "{message}");
    assert!(
        message.contains("not") && message.contains("executable"),
        "the refusal must say represented-but-not-executable: {message}"
    );
}
