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

/// The OLMo-2 / OLMo-3 / EXAONE-4 estate: both wrap norms, neither
/// pre-norm. Its whole discriminator against a two-norm Llama stack is
/// `post_feedforward_layernorm` — the families share the
/// `post_attention_layernorm` SPELLING and mean opposite things by it
/// (there, the norm on the attention output; in Llama, the norm on the
/// FFN input), so the placement is read from which norms EXIST rather
/// than from what any one of them is called.
#[test]
fn placement_evidence_reads_the_post_norm_estate() {
    let post = [
        "0.post_attention_layernorm.weight",
        "0.post_feedforward_layernorm.weight",
    ];
    assert_eq!(
        norm_placement_evidence(post.iter().copied()),
        Ok(NormPlacement::PostOnly)
    );

    // The falsifier, and the reason the discriminator is the FFN norm: a
    // two-norm Llama stack carries a tensor of the same name and must
    // keep reading as `PreOnly`.
    let llama = [
        "0.input_layernorm.weight",
        "0.post_attention_layernorm.weight",
    ];
    assert_eq!(
        norm_placement_evidence(llama.iter().copied()),
        Ok(NormPlacement::PreOnly)
    );

    // And a post-attention norm ALONE is still no judged placement —
    // one wrap norm describes no complete stack.
    let half = ["0.post_attention_layernorm.weight"];
    let err = norm_placement_evidence(half.iter().copied()).unwrap_err();
    assert!(err.contains("neither two-norm nor four-norm"), "{err}");
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

/// LFM2's spelling of the two-norm estate resolves to the placement it
/// actually is, and the mapping that looks wrong is the one that is
/// right: `ffn_norm` binds to `PostAttentionNorm`, because in a two-norm
/// layer that role IS the pre-FFN norm and keeps the historical name.
#[test]
fn the_lfm2_norm_dialect_resolves_to_the_two_norm_placement() {
    let lfm2 = ["0.operator_norm.weight", "0.ffn_norm.weight"];
    assert_eq!(
        norm_placement_evidence(lfm2.iter().copied()),
        Ok(NormPlacement::PreOnly)
    );

    // The falsifier for the mapping choice: binding `ffn_norm` to the
    // honestly-named `PreFfnNorm` would read the estate as a PARTIAL
    // four-norm stack and refuse it. That is what this arrangement
    // avoids, and the assertion states the shape it would have produced.
    let as_pre_ffn = [
        "0.operator_norm.weight",
        "0.pre_feedforward_layernorm.weight",
    ];
    let err = norm_placement_evidence(as_pre_ffn.iter().copied()).unwrap_err();
    assert!(err.contains("neither two-norm nor four-norm"), "{err}");

    // And one norm alone is still no judged placement.
    let half = ["0.operator_norm.weight"];
    assert!(norm_placement_evidence(half.iter().copied()).is_err());
}

/// Wave 18: the six Sinkhorn hyper-connection site operands classify by
/// their exact bare leaf, on every layer operator — a KDA layer and an
/// MLA layer of the same GLM-5.3-Flash stack each carry all six — and
/// nothing that merely contains `hc_` does.
#[test]
fn hyper_connection_site_operands_classify_exactly_and_layer_blind() {
    for op in [
        LayerOperator::Softmax,
        LayerOperator::Kda,
        LayerOperator::Mla,
    ] {
        assert_eq!(
            classify_stack_tensor_on("0.hc_attn_fn", op),
            Some((0, OperandRole::HcAttnMixFn)),
            "{op:?}"
        );
        assert_eq!(
            classify_stack_tensor_on("7.hc_attn_base", op),
            Some((7, OperandRole::HcAttnBase))
        );
        assert_eq!(
            classify_stack_tensor_on("42.hc_attn_scale", op),
            Some((42, OperandRole::HcAttnScale))
        );
        assert_eq!(
            classify_stack_tensor_on("0.hc_ffn_fn", op),
            Some((0, OperandRole::HcFfnMixFn))
        );
        assert_eq!(
            classify_stack_tensor_on("0.hc_ffn_base", op),
            Some((0, OperandRole::HcFfnBase))
        );
        assert_eq!(
            classify_stack_tensor_on("0.hc_ffn_scale", op),
            Some((0, OperandRole::HcFfnScale))
        );
    }
    // Near misses refuse rather than fuzzy-match. Each is a real spelling
    // from a real checkpoint that is NOT a Sinkhorn site operand.
    // A `.weight` suffix the checkpoints do not write.
    assert_eq!(classify_stack_tensor("0.hc_attn_fn.weight"), None);
    // Hy4-preview's Sinkhorn-free site (HC-PREPOST).
    assert_eq!(classify_stack_tensor("0.hc_attn_layer.hc_pre.hc_fn"), None);
    // Kimi-K3's AttnRes operands — a different residual topology.
    assert_eq!(
        classify_stack_tensor_on("0.self_attention_res_proj.weight", LayerOperator::Kda),
        None
    );
    assert_eq!(
        classify_stack_tensor_on("0.self_attention_res_norm.weight", LayerOperator::Kda),
        None
    );
    // The head is not a stack operand, even layer-prefixed.
    assert_eq!(classify_stack_tensor("0.hc_head_fn"), None);
}

/// The head's three operands classify by their BARE DeepSeek-V4 names and
/// nothing else: Hy4-preview's `model.hc_head.hc_head_fn` spells a
/// topology this build has not judged and must not bind.
#[test]
fn hyper_connection_head_operands_classify_by_bare_name_only() {
    use crate::format::vindex3::graph::roles::{
        classify_hyper_connection_head_tensor, is_hyper_connection_head_group, HcHeadOperand,
    };
    assert_eq!(
        classify_hyper_connection_head_tensor("hc_head_fn"),
        Some(HcHeadOperand::ReduceFn)
    );
    assert_eq!(
        classify_hyper_connection_head_tensor("hc_head_base"),
        Some(HcHeadOperand::Base)
    );
    assert_eq!(
        classify_hyper_connection_head_tensor("hc_head_scale"),
        Some(HcHeadOperand::Scale)
    );
    assert_eq!(
        classify_hyper_connection_head_tensor("model.hc_head.hc_head_fn"),
        None
    );
    assert_eq!(
        classify_hyper_connection_head_tensor("hc_head_fn.weight"),
        None
    );
    assert_eq!(classify_hyper_connection_head_tensor("hc_attn_fn"), None);

    // The builder's placement question reads the same table.
    assert!(is_hyper_connection_head_group("hc_head_fn"));
    assert!(is_hyper_connection_head_group("hc_head_base"));
    assert!(is_hyper_connection_head_group("hc_head_scale"));
    assert!(!is_hyper_connection_head_group("model.hc_head"));
    assert!(!is_hyper_connection_head_group("mtp"));
    assert!(!is_hyper_connection_head_group("hc_head"));
}

/// **K3-ATTNRES-1: the four site operands classify under the DECLARATION
/// and under nothing else.**
///
/// Two halves, and the second is the whole point. A checkpoint that
/// declares `attn_res_block_size` gets the four roles on every operator
/// — K3 carries them on its KDA layers and its MLA layers alike. A
/// checkpoint that merely ships the four spellings gets nothing, on
/// every operator and under every other topology, so no component can
/// acquire a residual programme from its tensor names.
#[test]
fn attention_residual_site_operands_classify_only_under_the_declaration() {
    use crate::format::vindex3::graph::roles::classify_stack_tensor_under;
    use larql_models::config::{HyperConnection, ResidualTopology};

    let declared = ResidualTopology::AttentionResidual { block_size: 12 };
    let expected = [
        (
            "self_attention_res_norm.weight",
            OperandRole::AttnResAttentionNorm,
        ),
        (
            "self_attention_res_proj.weight",
            OperandRole::AttnResAttentionProj,
        ),
        ("mlp_res_norm.weight", OperandRole::AttnResMlpNorm),
        ("mlp_res_proj.weight", OperandRole::AttnResMlpProj),
    ];
    for op in [
        LayerOperator::Softmax,
        LayerOperator::Kda,
        LayerOperator::Mla,
    ] {
        for (leaf, role) in expected {
            assert_eq!(
                classify_stack_tensor_under(&format!("7.{leaf}"), op, declared),
                Some((7, role)),
                "{leaf} on {op:?}"
            );
        }
    }

    // Half two: without the declaration, nothing. Every other topology,
    // and the operator-only classifier the norm-placement readers use.
    let hc = ResidualTopology::HyperConnection(HyperConnection {
        streams: 4,
        sinkhorn_iters: 20,
        sinkhorn_eps: 1e-6,
    });
    for topology in [ResidualTopology::SingleStream, hc] {
        for op in [
            LayerOperator::Softmax,
            LayerOperator::Kda,
            LayerOperator::Mla,
        ] {
            for (leaf, _) in expected {
                assert_eq!(
                    classify_stack_tensor_under(&format!("7.{leaf}"), op, topology),
                    None,
                    "{leaf} on {op:?} under {topology:?}"
                );
                assert_eq!(classify_stack_tensor_on(&format!("7.{leaf}"), op), None);
            }
        }
    }

    // The exit pair is not a stack operand, even layer-prefixed, and even
    // under the declaration: it is the stack's END, not a site.
    for leaf in ["output_attn_res_norm.weight", "output_attn_res_proj.weight"] {
        assert_eq!(
            classify_stack_tensor_under(&format!("0.{leaf}"), LayerOperator::Kda, declared),
            None,
            "{leaf}"
        );
    }
    // Near misses refuse rather than fuzzy-match.
    for leaf in [
        "self_attention_res_norm",
        "mlp_res_proj",
        "res_norm.weight",
        "self_attn_res_norm.weight",
    ] {
        assert_eq!(
            classify_stack_tensor_under(&format!("0.{leaf}"), LayerOperator::Kda, declared),
            None,
            "{leaf}"
        );
    }
}

/// The op plan's one recogniser of these spellings WITHOUT the
/// declaration, used to tell a stray from an unjudged name — and it
/// answers about the same four suffixes the role table holds, so the two
/// cannot drift.
#[test]
fn the_bare_site_spellings_are_recognised_for_the_refusal_only() {
    use crate::format::vindex3::graph::roles::is_attention_residual_site_operand;
    for leaf in [
        "self_attention_res_norm.weight",
        "self_attention_res_proj.weight",
        "mlp_res_norm.weight",
        "mlp_res_proj.weight",
    ] {
        assert!(is_attention_residual_site_operand(&format!("3.{leaf}")));
    }
    // Not layer-shaped, so not a site operand of any layer.
    assert!(!is_attention_residual_site_operand(
        "self_attention_res_norm.weight"
    ));
    // Nothing else, including the exit pair and the hyper-connection
    // sites.
    for leaf in [
        "output_attn_res_norm.weight",
        "hc_attn_fn",
        "input_layernorm.weight",
    ] {
        assert!(!is_attention_residual_site_operand(&format!("3.{leaf}")));
    }
}

/// The exit's two operands classify by their object-relative spelling
/// only, and the builder's placement vocabulary reads the SAME leaves —
/// so a name the graph places is a name the op plan can classify, and a
/// drift in either direction fails here.
#[test]
fn the_exit_pair_classifies_and_places_from_one_vocabulary() {
    use crate::format::vindex3::graph::roles::{
        classify_attention_residual_exit_tensor, AttentionResidualExitOperand,
        ATTENTION_RESIDUAL_EXIT_LEAVES,
    };
    assert_eq!(
        classify_attention_residual_exit_tensor("output_attn_res_norm.weight"),
        Some(AttentionResidualExitOperand::Norm)
    );
    assert_eq!(
        classify_attention_residual_exit_tensor("output_attn_res_proj.weight"),
        Some(AttentionResidualExitOperand::Proj)
    );
    // The artifact-global name never survives into the container, so the
    // classifier must not accept one — it would mean the strip rule
    // changed underneath it.
    assert_eq!(
        classify_attention_residual_exit_tensor("language_model.model.output_attn_res_norm.weight"),
        None
    );
    assert_eq!(
        classify_attention_residual_exit_tensor("output_attn_res_norm"),
        None
    );
    assert_eq!(
        classify_attention_residual_exit_tensor("mlp_res_norm.weight"),
        None
    );

    // One vocabulary, two consumers: every leaf the builder matches has
    // exactly one classifier row under it, and no row exists for a leaf
    // the builder would never place.
    assert_eq!(ATTENTION_RESIDUAL_EXIT_LEAVES.len(), 2);
    for leaf in ATTENTION_RESIDUAL_EXIT_LEAVES {
        assert!(
            classify_attention_residual_exit_tensor(&format!("{leaf}.weight")).is_some(),
            "{leaf}"
        );
    }
}

/// The site operands are not norms: norm-placement evidence must read
/// straight past them. A hyper-connected two-norm stack is still a
/// two-norm stack.
#[test]
fn hyper_connection_operands_do_not_disturb_norm_placement_evidence() {
    let names = [
        "0.input_layernorm.weight",
        "0.post_attention_layernorm.weight",
        "0.hc_attn_fn",
        "0.hc_attn_base",
        "0.hc_attn_scale",
        "0.hc_ffn_fn",
        "0.hc_ffn_base",
        "0.hc_ffn_scale",
    ];
    assert_eq!(
        norm_placement_evidence(names.iter().copied()),
        Ok(NormPlacement::PreOnly)
    );
}
