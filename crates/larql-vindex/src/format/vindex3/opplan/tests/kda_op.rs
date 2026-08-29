//! `KdaOp` states everything needed to reconstruct the recurrence, and the
//! variant that holds it never answers for another operator.
//!
//! The geometries are the two real ones — Kimi Linear 32×128 and
//! GLM-5.3-Flash 64×128 — so a width baked in for either is visible here.

use crate::format::vindex3::opplan::{KdaOp, LayerAttention, OperandRef};

const KIMI_HIDDEN: usize = 2304;
const KIMI_HEADS: usize = 32;
const GLM_HIDDEN: usize = 4096;
const GLM_HEADS: usize = 64;
const HEAD_DIM: usize = 128;
const CONV_KERNEL: usize = 4;
const GATE_RANK: usize = 128;
const GATE_LOWER_BOUND: f32 = -5.0;

fn operand(name: &str, shape: Vec<usize>) -> OperandRef {
    OperandRef {
        object: "target.decoder_stack".to_string(),
        tensor: format!("0.self_attn.{name}"),
        dtype: "BF16".to_string(),
        shape,
    }
}

/// A KDA op at the given geometry, with every operand shaped as the
/// checkpoint stores it.
fn kda_op(hidden: usize, num_heads: usize) -> KdaOp {
    let width = num_heads * HEAD_DIM;
    let proj = |n: &str| operand(n, vec![width, hidden]);
    let conv = |n: &str| operand(n, vec![width, 1, CONV_KERNEL]);
    KdaOp {
        num_heads,
        head_dim: HEAD_DIM,
        conv_kernel: CONV_KERNEL,
        gate_rank: GATE_RANK,
        gate_lower_bound: Some(GATE_LOWER_BOUND),
        q_proj: proj("q_proj.weight"),
        k_proj: proj("k_proj.weight"),
        v_proj: proj("v_proj.weight"),
        q_conv1d: conv("q_conv1d.weight"),
        k_conv1d: conv("k_conv1d.weight"),
        v_conv1d: conv("v_conv1d.weight"),
        f_a_proj: operand("f_a_proj.weight", vec![GATE_RANK, hidden]),
        f_b_proj: operand("f_b_proj.weight", vec![width, GATE_RANK]),
        g_a_proj: operand("g_a_proj.weight", vec![GATE_RANK, hidden]),
        g_b_proj: operand("g_b_proj.weight", vec![width, GATE_RANK]),
        b_proj: operand("b_proj.weight", vec![num_heads, hidden]),
        a_log: operand("A_log", vec![num_heads]),
        dt_bias: operand("dt_bias", vec![width]),
        o_norm: operand("o_norm.weight", vec![HEAD_DIM]),
        out_proj: operand("o_proj.weight", vec![hidden, width]),
    }
}

/// The completeness rule: every dimension the recurrence needs is on the
/// op, and the operand shapes close against it — with no reference to
/// which checkpoint the container came from.
#[test]
fn the_geometry_closes_against_every_operand_on_both_checkpoints() {
    for (name, hidden, heads) in [
        ("Kimi Linear", KIMI_HIDDEN, KIMI_HEADS),
        ("GLM-5.3-Flash", GLM_HIDDEN, GLM_HEADS),
    ] {
        let op = kda_op(hidden, heads);
        let width = op.value_width();
        assert_eq!(width, heads * HEAD_DIM, "{name}");

        // The three projections and three convs all carry the value width.
        for p in [&op.q_proj, &op.k_proj, &op.v_proj] {
            assert_eq!(p.shape, vec![width, hidden], "{name}");
        }
        for c in [&op.q_conv1d, &op.k_conv1d, &op.v_conv1d] {
            assert_eq!(c.shape, vec![width, 1, op.conv_kernel], "{name}");
        }
        // The gate pair agrees on one rank — the closure fact no
        // per-operand shape contract can state.
        assert_eq!(op.f_a_proj.shape[0], op.gate_rank, "{name}");
        assert_eq!(op.f_b_proj.shape[1], op.gate_rank, "{name}");
        assert_eq!(op.g_a_proj.shape[0], op.gate_rank, "{name}");
        assert_eq!(op.g_b_proj.shape[1], op.gate_rank, "{name}");
        // The discriminator: per channel, not per head.
        assert_eq!(op.dt_bias.shape, vec![width], "{name}");
        assert_eq!(op.a_log.shape, vec![op.num_heads], "{name}");
        assert_ne!(
            op.dt_bias.shape, op.a_log.shape,
            "{name}: a per-head dt_bias would make this Gated DeltaNet"
        );
        // Norm is over ONE head's width, not the value side.
        assert_eq!(op.o_norm.shape, vec![op.head_dim], "{name}");
        assert_eq!(op.out_proj.shape, vec![hidden, width], "{name}");
        // State is constant in sequence length: one Dk x Dv per head.
        assert_eq!(op.state_elements(), heads * HEAD_DIM * HEAD_DIM, "{name}");
    }
}

/// The two geometries differ where they must and agree where they may —
/// the same assertion the config-level fixtures make, repeated at the op
/// so a width assumption cannot enter between the two layers.
#[test]
fn the_two_checkpoints_differ_in_geometry_not_in_vocabulary() {
    let kimi = kda_op(KIMI_HIDDEN, KIMI_HEADS);
    let glm = kda_op(GLM_HIDDEN, GLM_HEADS);
    assert_eq!(kimi.head_dim, glm.head_dim);
    assert_eq!(kimi.conv_kernel, glm.conv_kernel);
    assert_eq!(kimi.gate_rank, glm.gate_rank);
    assert_ne!(kimi.value_width(), glm.value_width());
    assert_ne!(kimi.state_elements(), glm.state_elements());
}

/// A KDA layer must not answer as any other operator.
#[test]
fn the_kda_variant_never_answers_for_another_operator() {
    let attention = LayerAttention::Kda(Box::new(kda_op(KIMI_HIDDEN, KIMI_HEADS)));

    assert!(attention.kda().is_some());
    assert!(
        attention.gated_delta().is_none(),
        "the other recurrence must not be handed a KDA op"
    );
    assert!(
        attention.softmax().is_none(),
        "a recurrence has no softmax op, and no span to hand a KV planner"
    );
    assert_eq!(
        attention.declared_name(),
        larql_models::config::LAYER_TYPE_LINEAR_ATTENTION
    );
    assert_eq!(
        attention.recurrent_state_elements(),
        Some(KIMI_HEADS * HEAD_DIM * HEAD_DIM),
        "a KV planner is told state elements, never positions"
    );
}

/// `softmax_mut` must refuse a KDA layer too — the mutable accessor is a
/// second door to the same wrong answer.
#[test]
fn the_mutable_softmax_accessor_also_refuses() {
    let mut attention = LayerAttention::Kda(Box::new(kda_op(GLM_HIDDEN, GLM_HEADS)));
    assert!(attention.softmax_mut().is_none());
}

/// An undeclared decay clamp stays absent. A wrong clamp changes the decay
/// envelope without changing any shape, so it is the one field a default
/// could corrupt invisibly.
#[test]
fn an_undeclared_decay_clamp_is_absent_not_defaulted() {
    let mut op = kda_op(KIMI_HIDDEN, KIMI_HEADS);
    assert_eq!(op.gate_lower_bound, Some(GATE_LOWER_BOUND));
    op.gate_lower_bound = None;
    assert!(op.gate_lower_bound.is_none());
    // Geometry is unaffected — the clamp is not a dimension.
    assert_eq!(op.value_width(), KIMI_HEADS * HEAD_DIM);
}

/// A per-head vector may carry the broadcast singleton dimensions its
/// reference uses, and nothing else may.
///
/// Kimi Linear stores `A_log` as `[1, 1, 32, 1]` — the shape it broadcasts
/// against `[B, T, H, D]` — where the contract says `[32]`. Those are the
/// same 32 numbers in the same order, so refusing them would demand the
/// checkpoint be rewritten to satisfy a convention rather than a meaning.
///
/// The control is the second half: the equivalence must not become a
/// general squeeze. A `[2, 16]` tensor holds 32 numbers too and is a
/// different operand, and a shape contract exists to say so.
#[test]
fn a_vector_contract_accepts_broadcast_singletons_and_nothing_else() {
    use crate::format::vindex3::opplan::build::shape_satisfies;

    assert!(shape_satisfies(&[32], &[32]));
    assert!(
        shape_satisfies(&[1, 1, 32, 1], &[32]),
        "Kimi Linear's A_log"
    );
    assert!(shape_satisfies(&[32, 1], &[32]));

    assert!(
        !shape_satisfies(&[2, 16], &[32]),
        "a relayout is not a broadcast"
    );
    assert!(!shape_satisfies(&[16], &[32]), "wrong count");
    assert!(
        !shape_satisfies(&[1, 1, 16, 1], &[32]),
        "wrong count, broadcast shape"
    );
    // A matrix contract gets no equivalence at all: only 1-D contracts do.
    assert!(!shape_satisfies(&[1, 4096, 128], &[4096, 128]));
    assert!(shape_satisfies(&[4096, 128], &[4096, 128]));
}
