//! The bias declarations' operand arms: norm, FFN and attention-output
//! biases are required exactly under their declarations and refused by
//! name otherwise.

use super::*;

#[test]
fn norm_biases_are_required_beside_every_trunk_norm_only_when_declared() {
    let declared = LayerOps {
        placement: NormPlacement::PreOnly,
        norm_bias: true,
        ..base_ops()
    };
    let roles = required_roles(&declared);
    for role in [
        OperandRole::PreAttentionNormBias,
        OperandRole::PostAttentionNormBias,
    ] {
        assert!(roles.contains(&role), "{role:?}");
        assert_eq!(absent_op(role, &declared), None, "{role:?}");
    }
    // A two-norm layer has no FFN-site norms, so no FFN-site biases.
    assert!(!roles.contains(&OperandRole::PreFfnNormBias));
    assert_eq!(
        absent_op(OperandRole::PreFfnNormBias, &declared),
        Some("four-norm placement")
    );
    let four = LayerOps {
        placement: NormPlacement::PrePost,
        norm_bias: true,
        ..base_ops()
    };
    let roles = required_roles(&four);
    assert!(roles.contains(&OperandRole::PreFfnNormBias));
    assert!(roles.contains(&OperandRole::PostFfnNormBias));

    let silent = base_ops();
    assert!(!required_roles(&silent).contains(&OperandRole::PreAttentionNormBias));
    for role in [
        OperandRole::PreAttentionNormBias,
        OperandRole::PostAttentionNormBias,
        OperandRole::PreFfnNormBias,
        OperandRole::PostFfnNormBias,
    ] {
        assert_eq!(
            absent_op(role, &silent),
            Some("a norm bias (declared `norm.bias`)"),
            "{role:?}"
        );
    }
    assert_eq!(norm_bias_role(OperandRole::FfnUp), None);
}

#[test]
fn ffn_biases_follow_the_declaration_and_the_layers_ffn_kind() {
    let declared = LayerOps {
        gated_ffn: false,
        ffn_bias: true,
        ..base_ops()
    };
    let roles = required_roles(&declared);
    assert!(roles.contains(&OperandRole::FfnUpBias) && roles.contains(&OperandRole::FfnDownBias));
    assert_eq!(absent_op(OperandRole::FfnUpBias, &declared), None);
    assert_eq!(
        absent_op(OperandRole::FfnDownBias, &base_ops()),
        Some("an FFN projection bias (declared `ffn.bias`)")
    );
    let routed = LayerOps {
        routed: true,
        hybrid: false,
        ffn_bias: true,
        moe: Some(moe(
            MoeRouterKind::TopKSoftmax,
            false,
            ExpertFormat::PackedMxfp4,
        )),
        ..base_ops()
    };
    assert_eq!(
        absent_op(OperandRole::FfnUpBias, &routed),
        Some("dense FFN (this layer is routed)")
    );
}

#[test]
fn an_output_only_bias_requires_the_output_bias_and_refuses_q_k_v() {
    let ops = LayerOps {
        output_bias: true,
        ..base_ops()
    };
    let roles = required_roles(&ops);
    assert!(roles.contains(&OperandRole::AttnOBias));
    assert!(!roles.contains(&OperandRole::AttnQBias));
    assert_eq!(absent_op(OperandRole::AttnOBias, &ops), None);
    assert!(absent_op(OperandRole::AttnQBias, &ops).is_some());
    assert!(absent_op(OperandRole::AttnOBias, &base_ops())
        .unwrap()
        .contains("`output_bias`"));
}
