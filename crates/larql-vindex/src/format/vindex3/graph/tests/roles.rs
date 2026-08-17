//! Operand-role vocabulary gates: exact classification, fail-closed
//! placement evidence.

use crate::format::vindex3::graph::roles::{
    classify_stack_tensor, norm_placement_evidence, NormPlacement, OperandRole,
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
