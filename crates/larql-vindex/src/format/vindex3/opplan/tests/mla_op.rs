//! `MlaOp` states everything needed to reconstruct Multi-Latent Attention,
//! and the variant that holds it never answers for another operator.
//!
//! The geometry is Kimi Linear's real one: `hidden 2304, 32 heads,
//! kv_lora_rank 512, qk_nope_head_dim 128, qk_rope_head_dim 64,
//! v_head_dim 128` — confirmed against the real checkpoint's tensor
//! shapes (`q_proj [6144, 2304]`, `kv_a_proj_with_mqa [576, 2304]`,
//! `kv_b_proj [8192, 512]`, `o_proj [2304, 4096]`), not invented.

use crate::format::vindex3::opplan::{LayerAttention, MlaOp, OperandRef};

const HIDDEN: usize = 2304;
const NUM_HEADS: usize = 32;
const KV_LORA_RANK: usize = 512;
const QK_NOPE_HEAD_DIM: usize = 128;
const QK_ROPE_HEAD_DIM: usize = 64;
const V_HEAD_DIM: usize = 128;

fn operand(name: &str, shape: Vec<usize>) -> OperandRef {
    OperandRef {
        object: "target.decoder_stack".to_string(),
        tensor: format!("3.self_attn.{name}"),
        dtype: "BF16".to_string(),
        shape,
    }
}

/// An MLA op at Kimi Linear's real geometry, every operand shaped exactly
/// as the checkpoint stores it.
fn mla_op() -> MlaOp {
    let q_head_dim = QK_NOPE_HEAD_DIM + QK_ROPE_HEAD_DIM;
    MlaOp {
        num_heads: NUM_HEADS,
        kv_lora_rank: KV_LORA_RANK,
        qk_nope_head_dim: QK_NOPE_HEAD_DIM,
        qk_rope_head_dim: QK_ROPE_HEAD_DIM,
        v_head_dim: V_HEAD_DIM,
        q_proj: operand("q_proj.weight", vec![NUM_HEADS * q_head_dim, HIDDEN]),
        kv_a_proj: operand(
            "kv_a_proj_with_mqa.weight",
            vec![KV_LORA_RANK + QK_ROPE_HEAD_DIM, HIDDEN],
        ),
        kv_b_proj: operand(
            "kv_b_proj.weight",
            vec![NUM_HEADS * (QK_NOPE_HEAD_DIM + V_HEAD_DIM), KV_LORA_RANK],
        ),
        kv_a_norm: operand("kv_a_layernorm.weight", vec![KV_LORA_RANK]),
        out_proj: operand("o_proj.weight", vec![HIDDEN, NUM_HEADS * V_HEAD_DIM]),
        kv_a_norm_eps: Some(1e-6),
    }
}

/// The completeness rule: every dimension the operator needs is on the
/// op, and the real checkpoint's operand shapes close against it exactly.
#[test]
fn the_geometry_closes_against_every_operand_at_kimis_real_widths() {
    let op = mla_op();
    assert_eq!(op.q_head_dim(), 192, "128 nope + 64 rope");
    assert_eq!(op.q_proj.shape, vec![NUM_HEADS * 192, HIDDEN]);
    assert_eq!(op.q_proj.shape, vec![6144, 2304], "the real q_proj shape");
    assert_eq!(
        op.kv_a_proj.shape,
        vec![576, 2304],
        "kv_lora_rank + qk_rope_head_dim = 512 + 64"
    );
    assert_eq!(
        op.kv_b_proj.shape,
        vec![8192, 512],
        "32 heads * (128 nope + 128 v) = 8192, against kv_lora_rank"
    );
    assert_eq!(op.kv_a_norm.shape, vec![512]);
    assert_eq!(
        op.out_proj.shape,
        vec![2304, 4096],
        "32 heads * 128 v_head_dim = 4096"
    );
    assert_eq!(
        op.compressed_kv_width(),
        576,
        "the cache carries the compressed latent, not the decompressed pair"
    );
    // The asymmetry that makes this operator distinct from softmax: the
    // query/key side and the value side answer different widths.
    assert_ne!(op.q_head_dim(), op.v_head_dim);
}

/// An MLA layer must not answer as any other operator.
#[test]
fn the_mla_variant_never_answers_for_another_operator() {
    let attention = LayerAttention::Mla(Box::new(mla_op()));

    assert!(attention.mla().is_some());
    assert!(
        attention.kda().is_none(),
        "the other non-softmax operator must not be handed an MLA op"
    );
    assert!(
        attention.gated_delta().is_none(),
        "a recurrence must not be handed an MLA op"
    );
    assert!(
        attention.softmax().is_none(),
        "MLA has no softmax op — its operands do not share that contract"
    );

    // The reverse direction: `.mla()` itself must refuse every OTHER
    // variant, not just be correct when called on its own.
    let kda_stub = LayerAttention::Kda(Box::new(crate::format::vindex3::opplan::KdaOp {
        num_heads: 1,
        head_dim: 1,
        conv_kernel: 1,
        gate_rank: 1,
        gate_lower_bound: None,
        q_proj: operand("q_proj.weight", vec![1, 1]),
        k_proj: operand("k_proj.weight", vec![1, 1]),
        v_proj: operand("v_proj.weight", vec![1, 1]),
        q_conv1d: operand("q_conv1d.weight", vec![1, 1, 1]),
        k_conv1d: operand("k_conv1d.weight", vec![1, 1, 1]),
        v_conv1d: operand("v_conv1d.weight", vec![1, 1, 1]),
        f_a_proj: operand("f_a_proj.weight", vec![1, 1]),
        f_b_proj: operand("f_b_proj.weight", vec![1, 1]),
        g_a_proj: operand("g_a_proj.weight", vec![1, 1]),
        g_b_proj: operand("g_b_proj.weight", vec![1, 1]),
        b_proj: operand("b_proj.weight", vec![1, 1]),
        a_log: operand("A_log", vec![1]),
        dt_bias: operand("dt_bias", vec![1]),
        o_norm: operand("o_norm.weight", vec![1]),
        out_proj: operand("o_proj.weight", vec![1, 1]),
    }));
    assert!(
        kda_stub.mla().is_none(),
        "`.mla()` must refuse a KDA layer, not just answer correctly for its own"
    );
    assert_eq!(
        attention.declared_name(),
        crate::format::vindex3::graph::policy::AttentionSpan::Full.declared_name(),
        "MLA has no `layer_types` spelling of its own"
    );
    assert_eq!(
        attention.recurrent_state_elements(),
        None,
        "MLA keeps a per-position cache, not a fixed recurrent state — \
         the same answer a softmax layer gives"
    );
}

/// `softmax_mut` must refuse an MLA layer too — the mutable accessor is a
/// second door to the same wrong answer.
#[test]
fn the_mutable_softmax_accessor_also_refuses() {
    let mut attention = LayerAttention::Mla(Box::new(mla_op()));
    assert!(attention.softmax_mut().is_none());
}

/// The compressed cache is far smaller than a decompressed K/V pair would
/// be — the entire point of the operator, stated as a number rather than
/// left for a reader to derive.
#[test]
fn the_compressed_cache_is_much_smaller_than_a_decompressed_kv_pair() {
    let op = mla_op();
    let decompressed = op.num_heads * (op.qk_nope_head_dim + op.v_head_dim);
    assert_eq!(decompressed, 8192);
    assert_eq!(op.compressed_kv_width(), 576);
    assert!(op.compressed_kv_width() * 14 < decompressed);
}
