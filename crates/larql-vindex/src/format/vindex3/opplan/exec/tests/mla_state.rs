//! What an MLA layer RETAINS, and the one fact it refuses without —
//! lift 2's F5 and F6, at the planning seam.
//!
//! MLA's cache was always describable arithmetically (`kv_lora_rank +
//! rope` per position); what the schema lacked was a state SPECIES for
//! it — one row per position, not a K/V pair — and a home for the
//! epsilon its latent norm runs at. The drill called that epsilon the
//! one judged semantic the container could not carry (F6). Both are
//! carried now, and the refusal that remains is the honest one: a
//! container that carries NO epsilon is refused rather than lent the
//! layer\'s own.
//!
//! Execution end to end is the miniature witness in
//! `opplan/tests/kda_mla_exec.rs`; this file stays at the planning seam.
//!
//! The fixture is the Gated DeltaNet hybrid with one layer\'s attention
//! swapped for an MLA op. Swapping rather than encoding an MLA container
//! keeps this test about the geometry and not about MLA admission.

use crate::format::vindex3::opplan::exec::continuation::plan_continuation_geometry;
use crate::format::vindex3::opplan::{LayerAttention, MlaOp, MlaQueryProjection, OperandRef};

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
        output_gate: None,
        num_heads: NUM_HEADS,
        kv_lora_rank: KV_LORA_RANK,
        qk_nope_head_dim: QK_NOPE_HEAD_DIM,
        qk_rope_head_dim: QK_ROPE_HEAD_DIM,
        v_head_dim: V_HEAD_DIM,
        query: MlaQueryProjection::Direct { q_proj: o() },
        kv_a_proj: o(),
        kv_b_proj: o(),
        kv_a_norm: o(),
        out_proj: o(),
        kv_a_norm_eps: Some(1e-6),
    }
}

/// Continuation planning ANSWERS for an MLA layer, in the third state
/// species lift 2 introduced: ONE row per position, of the compressed
/// width — never a K/V pair.
///
/// The negative half is the load-bearing one. `kv()` must stay `None`
/// here: it is the projection every KV-only provider takes, and a latent
/// layer answering it would size a cache of two decompressed rows the
/// model never keeps.
#[test]
fn continuation_planning_sizes_an_mla_layer() {
    use crate::format::vindex3::opplan::exec::continuation::LayerContinuationGeometry;

    let (_container, mut plan, _store) = hybrid_plan_for_tests();
    plan.layers[0].attention = LayerAttention::Mla(Box::new(stub_mla()));

    let geometry = plan_continuation_geometry(&plan).expect("MLA's cache is stateable");
    assert_eq!(
        geometry[0],
        LayerContinuationGeometry::LatentKv(
            crate::format::vindex3::opplan::exec::continuation::LayerLatentKvGeometry {
                width: KV_LORA_RANK + QK_ROPE_HEAD_DIM,
            }
        )
    );
    assert_eq!(
        geometry[0].elements_at(7),
        stub_mla().compressed_kv_width() * 7,
        "one row per position, not two"
    );
    assert!(
        geometry[0].kv().is_none() && geometry[0].kv_side().is_none(),
        "a KV-only provider must not be able to serve this layer by projection"
    );
    assert!(geometry[0].recurrent().is_none(), "MLA is not a recurrence");
}

/// **An MLA layer with no carried epsilon is REFUSED at preparation** —
/// before a single matrix is bound, and naming the fact.
///
/// The fail-open this closes is specific and was real: `kv_a_layernorm`
/// runs at `1e-6` on the judged checkpoint while every other norm in the
/// layer runs at `1e-5`, and the executor that held that constant could
/// not be handed it by any container. Substituting the layer epsilon
/// here would compute a different function with every shape still
/// closing — the exact signature of a defect no gate catches — so the
/// absence has to stop preparation instead.
#[test]
fn preparing_an_mla_layer_without_its_norm_epsilon_refuses() {
    use crate::format::vindex3::opplan::exec::prepared::{ExecutionSlice, PreparedOperands};
    use crate::format::vindex3::opplan::exec::reference::ReferenceBackend;

    let (_container, mut plan, store) = hybrid_plan_for_tests();

    // The unmodified fixture prepares — so the refusal below is about MLA.
    PreparedOperands::load(&plan, &store, &ReferenceBackend, ExecutionSlice::Full)
        .expect("the Gated DeltaNet fixture prepares");

    let mut unjudged = stub_mla();
    unjudged.kv_a_norm_eps = None;
    plan.layers[0].attention = LayerAttention::Mla(Box::new(unjudged));
    let message =
        match PreparedOperands::load(&plan, &store, &ReferenceBackend, ExecutionSlice::Full) {
            Ok(_) => panic!("an MLA layer with no judged latent-norm epsilon must refuse"),
            Err(err) => err.to_string(),
        };
    assert!(
        message.contains("kv_a_layernorm") && message.contains("rms_norm_eps"),
        "the refusal must name the norm and the value it will NOT substitute: {message}"
    );
    // And it refuses BEFORE binding: the stub operands do not exist in
    // this fixture, so a message about a missing tensor would mean the
    // epsilon was checked too late to be the reason.
    assert!(
        !message.contains("no tensor"),
        "the epsilon must be checked before any operand is bound: {message}"
    );
}
