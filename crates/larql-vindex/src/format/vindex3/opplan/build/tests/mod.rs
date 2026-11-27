use super::super::super::graph::surface::MoeLatent;
use super::*;

fn base_ops() -> LayerOps {
    LayerOps {
        placement: NormPlacement::PrePost,
        gated_ffn: true,
        mamba2: None,
        conv_qkv: None,
        hyper_connection: false,
        attention_residual: false,
        output_gate: false,
        kda_full_rank_gate: false,
        mla_output_gate: false,
        mla_q_lora: false,
        attention_bias: false,
        qkv_bias: false,
        output_bias: false,
        norm_bias: false,
        ffn_bias: false,
        sinks: false,
        routed: false,
        hybrid: false,
        moe: None,
        v_from_k: false,
        // Softmax by default: these fixtures predate the hybrid
        // ladder, and a recurrent default would silently retarget
        // every one of them at the operator they were not written for.
        operator: LayerOperator::Softmax,
    }
}

fn moe(router_kind: MoeRouterKind, router_bias: bool, expert_format: ExpertFormat) -> MoeSurface {
    MoeSurface {
        branch_scale: None,
        dense_prefix_layers: None,
        experts: 8,
        top_k: 2,
        expert_intermediate_size: 64,
        router_kind,
        routing_policy: larql_models::config::ExpertRoutingPolicy::SoftmaxThenSelect,
        router_bias,
        expert_format,
        gate_up_layout: Some(larql_models::config::GateUpLayout::ContiguousHalves),
        shared_experts: 0,
        shared_expert_intermediate_size: None,
        shared_expert_gate: None,
        hybrid: false,
        latent: None,
    }
}

// ── absent_op: hybrid / routed-FFN / MoE exclusions ──────────────

#[test]
fn dense_ffn_roles_absent_on_a_routed_non_hybrid_layer() {
    let ops = LayerOps {
        routed: true,
        hybrid: false,
        moe: Some(moe(
            MoeRouterKind::TopKSoftmax,
            true,
            ExpertFormat::PackedMxfp4,
        )),
        ..base_ops()
    };
    for role in [
        OperandRole::FfnGate,
        OperandRole::FfnUp,
        OperandRole::FfnDown,
    ] {
        assert_eq!(
            absent_op(role, &ops),
            Some("dense FFN (this layer is routed)"),
            "{role:?}"
        );
    }
}

#[test]
fn gemma4_router_conditioning_absent_unless_the_router_kind_says_so() {
    let non_gemma4 = LayerOps {
        routed: true,
        moe: Some(moe(
            MoeRouterKind::TopKSoftmax,
            true,
            ExpertFormat::PackedMxfp4,
        )),
        ..base_ops()
    };
    for role in [
        OperandRole::MoeRouterScale,
        OperandRole::MoeRouterPerExpertScale,
    ] {
        assert_eq!(
            absent_op(role, &non_gemma4),
            Some("Gemma 4 router conditioning (router kind gemma4_top_k_softmax)"),
            "{role:?}"
        );
    }
    let gemma4 = LayerOps {
        routed: true,
        moe: Some(moe(
            MoeRouterKind::TopKRenormScaled,
            true,
            ExpertFormat::PackedMxfp4,
        )),
        ..base_ops()
    };
    assert_eq!(
        absent_op(OperandRole::MoeRouterScale, &gemma4),
        None,
        "a declared Gemma 4 router must not be reported absent"
    );
}

#[test]
fn hybrid_branch_norms_absent_on_a_non_hybrid_layer() {
    let ops = base_ops();
    for role in [
        OperandRole::PreExpertsNorm,
        OperandRole::PostDenseFfnNorm,
        OperandRole::PostExpertsNorm,
    ] {
        assert_eq!(
            absent_op(role, &ops),
            Some("hybrid dense+routed FFN (judged semantics)"),
            "{role:?}"
        );
    }
    let hybrid = LayerOps {
        hybrid: true,
        ..base_ops()
    };
    assert_eq!(absent_op(OperandRole::PreExpertsNorm, &hybrid), None);
}

#[test]
fn router_bias_absent_when_the_judgment_declares_none() {
    let no_bias = LayerOps {
        routed: true,
        moe: Some(moe(
            MoeRouterKind::TopKSoftmax,
            false,
            ExpertFormat::PackedMxfp4,
        )),
        ..base_ops()
    };
    assert_eq!(
        absent_op(OperandRole::MoeRouterBias, &no_bias),
        Some("router bias (declared by the routed-FFN judgment)")
    );
    let with_bias = LayerOps {
        routed: true,
        moe: Some(moe(
            MoeRouterKind::TopKSoftmax,
            true,
            ExpertFormat::PackedMxfp4,
        )),
        ..base_ops()
    };
    assert_eq!(absent_op(OperandRole::MoeRouterBias, &with_bias), None);
}

#[test]
fn expert_scale_streams_absent_when_the_format_carries_none() {
    let unsplit = LayerOps {
        routed: true,
        moe: Some(moe(
            MoeRouterKind::TopKSoftmax,
            true,
            ExpertFormat::PerExpert,
        )),
        ..base_ops()
    };
    for role in [
        OperandRole::ExpertGateUpScales,
        OperandRole::ExpertDownScales,
    ] {
        assert_eq!(
            absent_op(role, &unsplit),
            Some("a scaled expert format (this format carries no separate scales)"),
            "{role:?}"
        );
    }
    let split = LayerOps {
        routed: true,
        moe: Some(moe(
            MoeRouterKind::TopKSoftmax,
            true,
            ExpertFormat::PackedMxfp4,
        )),
        ..base_ops()
    };
    assert_eq!(absent_op(OperandRole::ExpertGateUpScales, &split), None);
}

// ── expected_shape: Gated DeltaNet + MoE geometry ────────────────

fn base_geometry(linear: Option<LinearAttentionSurface>) -> StackGeometry {
    StackGeometry {
        kda: None,
        mla: None,
        mamba2: None,
        conv_qkv: None,
        hyper_connection: None,
        hidden: 64,
        q_rows: 32,
        kv_rows: 16,
        intermediate: 128,
        head_dim: 8,
        num_q_heads: 4,
        num_kv_heads: 2,
        qk_scope: larql_models::config::QkNormScope::PerHead,
        // Ordinary width: a fused query/gate projection is twice this,
        // and the fixtures that exercise that say so themselves.
        q_proj_rows: 32,
        linear,
    }
}

fn linear_surface() -> LinearAttentionSurface {
    // Qwen3.8's own geometry — see gated_delta.rs's state_elements()
    // test for why real numbers, not placeholders.
    LinearAttentionSurface {
        key_heads: 16,
        key_head_dim: 128,
        value_heads: 48,
        value_head_dim: 128,
        conv_kernel: 4,
        state_dtype: Some(larql_models::inventory::report::RecurrentStateDtype::Float32),
    }
}

#[test]
fn full_projection_qk_norm_shape_is_unpinned() {
    let g = StackGeometry {
        qk_scope: larql_models::config::QkNormScope::FullProjection,
        ..base_geometry(None)
    };
    assert_eq!(expected_shape(OperandRole::AttnQNorm, &g, None), None);
}

#[test]
fn linear_attention_shapes_follow_the_recurrence_geometry_not_the_softmax_fields() {
    let l = linear_surface();
    let g = base_geometry(Some(l));
    assert_eq!(
        expected_shape(OperandRole::LinearAttnInProjQkv, &g, None),
        Some(vec![l.qkv_channels(), g.hidden])
    );
    assert_eq!(
        expected_shape(OperandRole::LinearAttnInProjA, &g, None),
        Some(vec![l.value_heads, g.hidden])
    );
    assert_eq!(
        expected_shape(OperandRole::LinearAttnInProjB, &g, None),
        Some(vec![l.value_heads, g.hidden])
    );
    assert_eq!(
        expected_shape(OperandRole::LinearAttnInProjZ, &g, None),
        Some(vec![l.value_width(), g.hidden])
    );
    assert_eq!(
        expected_shape(OperandRole::LinearAttnConv1d, &g, None),
        Some(vec![l.qkv_channels(), 1, l.conv_kernel])
    );
    assert_eq!(
        expected_shape(OperandRole::LinearAttnALog, &g, None),
        Some(vec![l.value_heads])
    );
    assert_eq!(
        expected_shape(OperandRole::LinearAttnDtBias, &g, None),
        Some(vec![l.value_heads])
    );
    assert_eq!(
        expected_shape(OperandRole::LinearAttnNorm, &g, None),
        Some(vec![l.value_head_dim])
    );
    assert_eq!(
        expected_shape(OperandRole::LinearAttnOutProj, &g, None),
        Some(vec![g.hidden, l.value_width()])
    );
}

#[test]
fn linear_attention_operands_have_no_shape_contract_without_a_declared_recurrence() {
    // `linear` absent while such an operand exists is a refusal, not
    // a waiver — every LinearAttn* role must fall through to `None`
    // via the `linear?` short-circuit, never invent a shape from the
    // softmax fields.
    let g = base_geometry(None);
    for role in [
        OperandRole::LinearAttnInProjQkv,
        OperandRole::LinearAttnInProjA,
        OperandRole::LinearAttnInProjZ,
        OperandRole::LinearAttnConv1d,
        OperandRole::LinearAttnALog,
        OperandRole::LinearAttnNorm,
        OperandRole::LinearAttnOutProj,
    ] {
        assert_eq!(expected_shape(role, &g, None), None, "{role:?}");
    }
}

#[test]
fn moe_router_and_expert_shapes_follow_the_judged_geometry() {
    let g = base_geometry(None);
    let m = moe(MoeRouterKind::TopKSoftmax, true, ExpertFormat::PerExpert);
    assert_eq!(
        expected_shape(OperandRole::MoeRouterPerExpertScale, &g, Some(&m)),
        Some(vec![m.experts])
    );
    assert_eq!(
        expected_shape(OperandRole::MoeRouterWeight, &g, Some(&m)),
        Some(vec![m.experts, g.hidden])
    );
    assert_eq!(
        expected_shape(OperandRole::MoeRouterBias, &g, Some(&m)),
        Some(vec![m.experts])
    );
    // PerExpert/PackedBF16 keep the unpacked [experts, rows, k] shape
    // and a bare [experts, rows] scales stream (packed_shape/
    // scales_shape's non-MXFP4 arm).
    assert_eq!(
        expected_shape(OperandRole::ExpertGateUp, &g, Some(&m)),
        Some(vec![
            m.experts,
            FUSED_BRANCHES * m.expert_intermediate_size,
            g.hidden
        ])
    );
    assert_eq!(
        expected_shape(OperandRole::ExpertGateUpBias, &g, Some(&m)),
        Some(vec![m.experts, FUSED_BRANCHES * m.expert_intermediate_size])
    );
    assert_eq!(
        expected_shape(OperandRole::ExpertDown, &g, Some(&m)),
        Some(vec![m.experts, g.hidden, m.expert_intermediate_size])
    );
    assert_eq!(
        expected_shape(OperandRole::ExpertDownBias, &g, Some(&m)),
        Some(vec![m.experts, g.hidden])
    );
    let split = moe(MoeRouterKind::TopKSoftmax, true, ExpertFormat::PackedMxfp4);
    assert_eq!(
        expected_shape(OperandRole::ExpertGateUpScales, &g, Some(&split)),
        Some(scales_shape(
            &split,
            FUSED_BRANCHES * split.expert_intermediate_size,
            g.hidden
        ))
    );
    assert_eq!(
        expected_shape(OperandRole::ExpertDownScales, &g, Some(&m)),
        Some(scales_shape(&m, g.hidden, m.expert_intermediate_size))
    );
}

/// **K3-LATENTMOE-1, D8 — the single width authority, and the reason
/// this test exists at all.**
///
/// A latent width can be made to look carried by giving it a home on
/// the surface. That moves a blocker count and changes nothing: the
/// expert bank would still be sized from the component's `hidden`,
/// so the op plan would refuse K3's real bank on shape while the plan
/// reported the width as carried. That is HOLLOW CARRIAGE — the plan
/// claiming a fact the operand plane contradicts — and it is the
/// specific failure this rung was frozen to avoid.
///
/// So every routed-bank contract is checked against the DECLARED
/// latent width, and each assertion is paired with a negative arm
/// proving it would have failed had that consumer kept reading
/// `hidden`. Without those `assert_ne!`s the test would pass just as
/// happily against the old code.
///
/// `ExpertDownBias` is here although D8's enumeration omitted it: its
/// contract describes the same latent output axis, and reconnaissance
/// missed it only because K3 ships no expert bias operand. Recorded
/// as a deviation in the execution notes.
#[test]
fn every_routed_bank_contract_reads_the_declared_latent_width() {
    let g = base_geometry(None);
    let mut m = moe(MoeRouterKind::TopKSoftmax, true, ExpertFormat::PackedMxfp4);
    // Hostile on purpose, exactly as the oracle's geometry is: the
    // latent width is neither `hidden` nor `hidden / 2` nor the
    // expert intermediate width, so no accidental derivation passes.
    let latent = 40;
    assert_ne!(latent, g.hidden);
    assert_ne!(latent, g.hidden / 2);
    assert_ne!(latent, m.expert_intermediate_size);
    m.latent = Some(MoeLatent {
        width: latent,
        norm: None,
    });

    // The authority itself, and the uniform form it must fall back to.
    assert_eq!(m.routed_expert_input_width(g.hidden), latent);
    assert_eq!(
        moe(MoeRouterKind::TopKSoftmax, true, ExpertFormat::PackedMxfp4)
            .routed_expert_input_width(g.hidden),
        g.hidden,
        "no declaration must still size the bank from hidden"
    );

    let inter = m.expert_intermediate_size;
    let fused = FUSED_BRANCHES * inter;

    // Packed gate/up and its scales: the `k` axis is the experts'
    // INPUT width. Packed down and its scales: the ROW axis is the
    // output width.
    for (role, want, hidden_arm) in [
        (
            OperandRole::ExpertGateUp,
            packed_shape(&m, fused, latent),
            packed_shape(&m, fused, g.hidden),
        ),
        (
            OperandRole::ExpertGateUpScales,
            scales_shape(&m, fused, latent),
            scales_shape(&m, fused, g.hidden),
        ),
        (
            OperandRole::ExpertDown,
            packed_shape(&m, latent, inter),
            packed_shape(&m, g.hidden, inter),
        ),
        (
            OperandRole::ExpertDownScales,
            scales_shape(&m, latent, inter),
            scales_shape(&m, g.hidden, inter),
        ),
    ] {
        let got = expected_shape(role, &g, Some(&m));
        assert_eq!(got, Some(want), "{role:?} must size from the latent width");
        assert_ne!(
            got,
            Some(hidden_arm),
            "{role:?} would have passed on `hidden` too — the arm proves nothing"
        );
    }

    // Per-expert storage reaches the same axes by a different route,
    // and must not be able to disagree with the packed one.
    assert_eq!(
        expected_shape(OperandRole::PerExpertGate(0), &g, Some(&m)),
        Some(vec![inter, latent])
    );
    assert_eq!(
        expected_shape(OperandRole::PerExpertUp(3), &g, Some(&m)),
        Some(vec![inter, latent])
    );
    assert_eq!(
        expected_shape(OperandRole::PerExpertDown(7), &g, Some(&m)),
        Some(vec![latent, inter])
    );
    assert_ne!(
        expected_shape(OperandRole::PerExpertDown(7), &g, Some(&m)),
        Some(vec![g.hidden, inter])
    );

    // D8's missing consumer. Same latent output axis as `ExpertDown`.
    assert_eq!(
        expected_shape(OperandRole::ExpertDownBias, &g, Some(&m)),
        Some(vec![m.experts, latent])
    );
    assert_ne!(
        expected_shape(OperandRole::ExpertDownBias, &g, Some(&m)),
        Some(vec![m.experts, g.hidden])
    );

    // And the two that must NOT move: the router reads the
    // un-projected block input, and the gate/up bias indexes the
    // fused intermediate rows, not the input width.
    assert_eq!(
        expected_shape(OperandRole::MoeRouterWeight, &g, Some(&m)),
        Some(vec![m.experts, g.hidden]),
        "the router reads hidden — routing on the latent is a different model"
    );
    assert_eq!(
        expected_shape(OperandRole::ExpertGateUpBias, &g, Some(&m)),
        Some(vec![m.experts, fused])
    );
}

// ── Attention residuals (K3-ATTNRES-1) ───────────────────────────

/// The four site operands are required on every transformer layer of
/// a component that declares the period, and on no other component.
/// Presence and requirement come from the same flag, so they cannot
/// desync — the `required_roles` half of what `absent_op` states
/// below.
#[test]
fn attention_residual_sites_are_required_exactly_under_the_declaration() {
    let sites = [
        OperandRole::AttnResAttentionNorm,
        OperandRole::AttnResAttentionProj,
        OperandRole::AttnResMlpNorm,
        OperandRole::AttnResMlpProj,
    ];
    let declared = LayerOps {
        attention_residual: true,
        ..base_ops()
    };
    let required = required_roles(&declared);
    for role in sites {
        assert!(required.contains(&role), "{role:?}");
    }
    let plain = required_roles(&base_ops());
    for role in sites {
        assert!(!plain.contains(&role), "{role:?}");
    }
}

/// A one-sublayer block under the declaration has no attention and
/// FFN sites for the topology's pairs to sit at, and this build has
/// judged no attention-residual form of one — so the operands are
/// strays there, named for the layer kind they would need. On an
/// ordinary transformer layer under the same declaration they are
/// consumed.
///
/// There is no single-stream arm to test beside this one, and its
/// absence is deliberate: without the declaration these spellings
/// never become roles at all
/// ([`classify_stack_tensor_under`](crate::format::vindex3::graph::roles::classify_stack_tensor_under)),
/// so `absent_op` is never asked about them and a guard here would
/// be dead.
#[test]
fn attention_residual_sites_on_a_one_sublayer_block_are_strays() {
    let sites = [
        OperandRole::AttnResAttentionNorm,
        OperandRole::AttnResAttentionProj,
        OperandRole::AttnResMlpNorm,
        OperandRole::AttnResMlpProj,
    ];
    for operator in [LayerOperator::Mamba2, LayerOperator::ConvQkvAttention] {
        let mixer = LayerOps {
            attention_residual: true,
            operator,
            ..base_ops()
        };
        for role in sites {
            assert_eq!(
                absent_op(role, &mixer),
                Some(ATTN_RES_SITE_ON_MIXER_LAYER),
                "{role:?} on {operator:?}"
            );
        }
    }
    let transformer = LayerOps {
        attention_residual: true,
        ..base_ops()
    };
    for role in sites {
        assert_eq!(absent_op(role, &transformer), None, "{role:?}");
    }
}

/// The pair's geometry closes over the component's width alone: the
/// declared period parameterises the snapshot SCHEDULE and no
/// operand's shape, so nothing here reads it. The asymmetry between
/// the two halves is the contract.
#[test]
fn attention_residual_shapes_close_over_the_width_and_not_the_period() {
    let g = base_geometry(None);
    for role in [
        OperandRole::AttnResAttentionNorm,
        OperandRole::AttnResMlpNorm,
    ] {
        assert_eq!(
            expected_shape(role, &g, None),
            Some(vec![g.hidden]),
            "{role:?}"
        );
    }
    for role in [
        OperandRole::AttnResAttentionProj,
        OperandRole::AttnResMlpProj,
    ] {
        assert_eq!(
            expected_shape(role, &g, None),
            Some(vec![1, g.hidden]),
            "{role:?}"
        );
    }
}

/// The exit's operand-level invariant, stated for a container whose
/// graph was edited rather than built: the builder places this object
/// only under the declaration, so closure asked about it on a
/// single-stream component names what the pair would require instead
/// of binding it. Unreachable through the builder, and checked here
/// so the refusal is not merely asserted in a comment.
#[test]
fn the_exit_pair_on_an_undeclared_component_names_what_it_requires() {
    let object = LogicalObject {
        id: "target.attention_residual_exit".to_string(),
        component: "target".to_string(),
        kind: ObjectKind::AttentionResidualExit,
        source_bindings: Vec::new(),
        representations: Vec::new(),
    };
    let tensor = |name: &str, shape: Vec<usize>| SegmentTensor {
        name: name.to_string(),
        dtype: "BF16".to_string(),
        shape,
        offset: 0,
        len: 0,
    };
    let tensors = vec![
        tensor("output_attn_res_norm.weight", vec![64]),
        tensor("output_attn_res_proj.weight", vec![1, 64]),
    ];

    let mut defects = Vec::new();
    attention_residual_exit_closure(&object, &tensors, false, 64, &mut defects);
    assert_eq!(defects.len(), 2, "{defects:?}");
    assert!(
        defects.iter().all(|d| matches!(
            d,
            ClosureDefect::OperandImpliesAbsentOp {
                required_primitive,
                ..
            } if required_primitive == ATTN_RES_EXIT_WITHOUT_DECLARATION
        )),
        "{defects:?}"
    );

    // Under the declaration the same pair closes silently.
    let mut declared = Vec::new();
    attention_residual_exit_closure(&object, &tensors, true, 64, &mut declared);
    assert!(declared.is_empty(), "{declared:?}");

    // ...and a half-shipped pair names the missing operand.
    let mut half = Vec::new();
    attention_residual_exit_closure(&object, &tensors[..1], true, 64, &mut half);
    assert!(
        half.iter().any(|d| matches!(
            d,
            ClosureDefect::ObjectShape { detail, .. } if detail.contains("Proj")
        )),
        "{half:?}"
    );
}

mod object_closures;
