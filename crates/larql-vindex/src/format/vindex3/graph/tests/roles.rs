//! Operand-role vocabulary gates: exact classification, fail-closed
//! placement evidence.

use crate::format::vindex3::graph::policy::LayerOperator;
use crate::format::vindex3::graph::roles::{
    classify_stack_tensor, classify_stack_tensor_on, norm_placement_evidence, NormPlacement,
    OperandRole,
};

#[test]
fn classification_is_exact_not_fuzzy() {
    assert_eq!(
        classify_stack_tensor("3.self_attn.q_proj.weight"),
        Some((3, OperandRole::AttnQ))
    );
    assert_eq!(
        classify_stack_tensor("0.pre_feedforward_layernorm.weight"),
        Some((0, OperandRole::PreFfnNorm))
    );
    // A-9.1: the projection biases and sinks are judged roles.
    assert_eq!(
        classify_stack_tensor("0.self_attn.q_proj.bias"),
        Some((0, OperandRole::AttnQBias))
    );
    assert_eq!(
        classify_stack_tensor("7.self_attn.o_proj.bias"),
        Some((7, OperandRole::AttnOBias))
    );
    assert_eq!(
        classify_stack_tensor("2.self_attn.sinks"),
        Some((2, OperandRole::AttnSinks))
    );
    // A new upstream spelling classifies as nothing — it must block, not
    // fuzzy-match into the wrong op: a bias on a projection no row names,
    // a sink tensor under another spelling.
    assert_eq!(classify_stack_tensor("0.self_attn.gate_proj.bias"), None);
    assert_eq!(classify_stack_tensor("0.self_attn.sink.weight"), None);
    assert_eq!(
        classify_stack_tensor("0.self_attn.q_projection.weight"),
        None
    );
    // Non-layer-shaped names are not stack operands.
    assert_eq!(classify_stack_tensor("norm.weight"), None);
    assert_eq!(classify_stack_tensor("fc.weight"), None);
}

/// Kimi Linear's fixed (non-indexed) MoE spellings — router weight, its
/// bias-correction tensor kept as a SEPARATE role (biased scores choose
/// ids, unbiased scores weight them — the two must never collapse into
/// one operand), and the shared expert's three projections.
#[test]
fn kimi_moe_fixed_spellings_classify() {
    assert_eq!(
        classify_stack_tensor("1.block_sparse_moe.gate.weight"),
        Some((1, OperandRole::MoeRouterWeight))
    );
    assert_eq!(
        classify_stack_tensor("1.block_sparse_moe.gate.e_score_correction_bias"),
        Some((1, OperandRole::MoeRouterBias))
    );
    assert_eq!(
        classify_stack_tensor("1.block_sparse_moe.shared_experts.gate_proj.weight"),
        Some((1, OperandRole::SharedExpertGate))
    );
    assert_eq!(
        classify_stack_tensor("1.block_sparse_moe.shared_experts.up_proj.weight"),
        Some((1, OperandRole::SharedExpertUp))
    );
    assert_eq!(
        classify_stack_tensor("1.block_sparse_moe.shared_experts.down_proj.weight"),
        Some((1, OperandRole::SharedExpertDown))
    );
}

/// The indexed per-expert vocabulary: `w1`/`w3`/`w2` → gate/up/down (the
/// mapping `modeling_kimi.py` states, not alphabetic order), carrying the
/// expert id on the role itself so 256 experts' worth of the same
/// suffix-minus-index do not collide in one layer's operand map.
#[test]
fn per_expert_indexed_operands_classify_with_their_expert_id() {
    for op in [
        LayerOperator::Softmax,
        // Layer-blind: the indexed vocabulary carries no attention
        // spelling to collide with, so the operator must not matter.
        LayerOperator::Kda,
    ] {
        assert_eq!(
            classify_stack_tensor_on("3.block_sparse_moe.experts.0.w1.weight", op),
            Some((3, OperandRole::PerExpertGate(0)))
        );
        assert_eq!(
            classify_stack_tensor_on("3.block_sparse_moe.experts.255.w3.weight", op),
            Some((3, OperandRole::PerExpertUp(255)))
        );
        assert_eq!(
            classify_stack_tensor_on("3.block_sparse_moe.experts.42.w2.weight", op),
            Some((3, OperandRole::PerExpertDown(42)))
        );
    }
}

/// Structural evidence, not a substring guess: a non-digit index segment,
/// an unrecognised leaf, and a bare `experts.` with no index all refuse
/// rather than classify a plausible-looking wrong role.
#[test]
fn per_expert_classification_requires_real_structural_evidence() {
    // Not a decimal expert id.
    assert_eq!(
        classify_stack_tensor("3.block_sparse_moe.experts.abc.w1.weight"),
        None
    );
    // A leaf this family does not declare.
    assert_eq!(
        classify_stack_tensor("3.block_sparse_moe.experts.0.w4.weight"),
        None
    );
    // No index segment at all.
    assert_eq!(
        classify_stack_tensor("3.block_sparse_moe.experts.w1.weight"),
        None
    );
    // No leaf after the index either — the prefix strips clean but there
    // is nothing further to split on.
    assert_eq!(classify_stack_tensor("3.block_sparse_moe.experts.5"), None);
    // An all-digit index too large for a `u16` expert count refuses
    // rather than wrapping or panicking.
    assert_eq!(
        classify_stack_tensor("3.block_sparse_moe.experts.99999999.w1.weight"),
        None
    );
    // Expert 1 and expert 10 do not collide on a substring match.
    assert_eq!(
        classify_stack_tensor("3.block_sparse_moe.experts.10.w1.weight"),
        Some((3, OperandRole::PerExpertGate(10)))
    );
    assert_eq!(
        classify_stack_tensor("3.block_sparse_moe.experts.1.w1.weight"),
        Some((3, OperandRole::PerExpertGate(1)))
    );
}

#[test]
fn placement_evidence_reads_two_and_four_norm_estates() {
    let four = [
        "0.input_layernorm.weight",
        "0.post_attention_layernorm.weight",
        "0.pre_feedforward_layernorm.weight",
        "0.post_feedforward_layernorm.weight",
    ];
    assert_eq!(
        norm_placement_evidence(four.iter().copied()),
        Ok(NormPlacement::PrePost)
    );
    let two = [
        "0.input_layernorm.weight",
        "0.post_attention_layernorm.weight",
    ];
    assert_eq!(
        norm_placement_evidence(two.iter().copied()),
        Ok(NormPlacement::PreOnly)
    );
}

#[test]
fn placement_evidence_refuses_partial_and_absent_estates() {
    // Three of four: neither judged placement — refuse, naming the flags.
    let mixed = [
        "0.input_layernorm.weight",
        "0.post_attention_layernorm.weight",
        "0.pre_feedforward_layernorm.weight",
    ];
    let err = norm_placement_evidence(mixed.iter().copied()).unwrap_err();
    assert!(err.contains("neither two-norm nor four-norm"), "{err}");
    assert!(err.contains("pre_ffn true"), "{err}");

    let none = ["0.self_attn.q_proj.weight"];
    let err = norm_placement_evidence(none.iter().copied()).unwrap_err();
    assert!(err.contains("no per-layer norm operands"), "{err}");
}
