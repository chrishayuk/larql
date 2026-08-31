//! The name table is pinned because three of its entries fail silently,
//! and the constructor invariant is tested because the type system
//! cannot express "only on unquantised sources" on its own.

use super::*;

const TYPE_F32: u32 = 0;
const TYPE_BF16: u32 = 30;
const TYPE_NVFP4: u32 = 40;

/// Every name llama.cpp binds, pinned. A regression here does not
/// error — it produces a file that loads and misreads.
#[test]
fn the_tensor_name_table_is_exactly_what_llama_cpp_binds() {
    let expected: &[(&str, &str)] = &[
        ("query", "blk.7.attn_q.weight"),
        ("key", "blk.7.attn_k.weight"),
        ("value", "blk.7.attn_v.weight"),
        ("output", "blk.7.attn_output.weight"),
        ("fused recurrent q|k|v", "blk.7.attn_qkv.weight"),
        ("output-gate projection", "blk.7.attn_gate.weight"),
        ("causal conv over q|k|v", "blk.7.ssm_conv1d.weight"),
        ("log decay", "blk.7.ssm_a"),
        ("timestep bias", "blk.7.ssm_dt.bias"),
        ("decay projection", "blk.7.ssm_alpha.weight"),
        ("write-strength projection", "blk.7.ssm_beta.weight"),
        ("gated norm", "blk.7.ssm_norm.weight"),
        ("output projection", "blk.7.ssm_out.weight"),
        ("ffn gate", "blk.7.ffn_gate.weight"),
        ("ffn up", "blk.7.ffn_up.weight"),
        ("ffn down", "blk.7.ffn_down.weight"),
    ];
    for (role, want) in expected {
        assert_eq!(
            qwen35_tensor_name(role, 7).as_deref(),
            Some(*want),
            "role `{role}`"
        );
    }
}

/// The three that fail quietly, called out on their own so a future
/// "tidy up the suffixes" commit has to argue with a test.
#[test]
fn the_three_naming_hazards_are_deliberate() {
    assert_eq!(
        qwen35_tensor_name("log decay", 0).as_deref(),
        Some("blk.0.ssm_a"),
        "ssm_a carries NO .weight suffix — adding one fails to bind"
    );
    assert_eq!(
        qwen35_tensor_name("timestep bias", 0).as_deref(),
        Some("blk.0.ssm_dt.bias"),
        "ssm_dt is a .bias, not a .weight"
    );
    assert_eq!(
        qwen35_tensor_name("query", 0).as_deref(),
        Some("blk.0.attn_q.weight"),
        "attn_q is the FUSED Q + output gate at double width; there is no separate gate tensor"
    );
    assert!(
        qwen35_tensor_name("attention output gate", 0).is_none(),
        "a separate full-attention gate has no target — it lives inside attn_q"
    );
}

#[test]
fn the_model_scope_surfaces_carry_no_layer_index() {
    assert_eq!(qwen35_global_name("embedding"), Some("token_embd.weight"));
    assert_eq!(qwen35_global_name("final norm"), Some("output_norm.weight"));
    assert_eq!(qwen35_global_name("output head"), Some("output.weight"));
    assert_eq!(qwen35_global_name("ffn down"), None, "not a global surface");
}

#[test]
fn an_unknown_role_has_no_target_rather_than_a_guessed_one() {
    assert!(qwen35_tensor_name("something new", 3).is_none());
    assert!(qwen35_global_name("something new").is_none());
}

/// **The planner's own invariant.** Preflight is policy-level defence;
/// this is the local one, and it holds even if a caller skips preflight
/// entirely.
#[test]
fn a_value_transform_on_a_quantised_source_is_an_illegal_plan() {
    let err = LoweredTensorPlan::new(
        "layer.0.ffn.down",
        "blk.0.ffn_down.weight",
        RepresentationKind::Nvfp4,
        TYPE_NVFP4,
        vec![5120, 17408],
        vec![],
        vec![ValueTransform::ApplyWeightOffset(1.0)],
        None,
    )
    .expect_err("arithmetic on a block-quantised source must refuse");

    let msg = err.to_string();
    assert!(msg.contains("changes values"), "{msg}");
    assert!(
        msg.contains("re-quantising") && msg.contains("nobody measured"),
        "the refusal must say what the damage would be: {msg}"
    );

    // The same for the log-decay materialisation.
    assert!(LoweredTensorPlan::new(
        "layer.0.mixer.log_decay",
        "blk.0.ssm_a",
        RepresentationKind::Nvfp4,
        TYPE_NVFP4,
        vec![5120, 17408],
        vec![],
        vec![ValueTransform::MaterializeLogDecay],
        None,
    )
    .is_err());
}

/// Layout transforms are fine on quantised sources — that is the whole
/// point of establishing the group-boundary invariant first.
#[test]
fn layout_transforms_are_legal_on_a_quantised_source() {
    let plan = LoweredTensorPlan::new(
        "layer.0.mixer.out_proj",
        "blk.0.ssm_out.weight",
        RepresentationKind::Nvfp4,
        TYPE_NVFP4,
        vec![5120, 6144],
        vec![LayoutTransform::ReorderVColumnsByGroups {
            key_heads: 16,
            v_per_k: 3,
            groups_per_head: 8,
        }],
        vec![],
        Some("blk.0.ssm_out.scale".into()),
    )
    .expect("permuting whole groups moves bytes, not values");
    assert_eq!(plan.scale_tensor.as_deref(), Some("blk.0.ssm_out.scale"));
}

/// And value transforms are legal where the source stores numbers.
#[test]
fn value_transforms_are_legal_on_unquantised_sources() {
    assert!(LoweredTensorPlan::new(
        "layer.0.mixer.log_decay",
        "blk.0.ssm_a",
        RepresentationKind::F32,
        TYPE_F32,
        vec![5120, 17408],
        vec![],
        vec![ValueTransform::MaterializeLogDecay],
        None,
    )
    .is_ok());

    assert!(LoweredTensorPlan::new(
        "layer.0.norm.pre",
        "blk.0.attn_norm.weight",
        RepresentationKind::Bf16,
        TYPE_BF16,
        vec![5120, 17408],
        vec![],
        // The offset comes from the graph, not from a literal chosen here.
        vec![ValueTransform::ApplyWeightOffset(1.0)],
        None,
    )
    .is_ok());
}

/// The gated norm takes no offset, and that is a fact from the graph
/// rather than a target-side exception: the linear-attention surface
/// declares none, so the planner is handed an empty transform list.
#[test]
fn the_gated_norm_receives_no_offset() {
    let plan = LoweredTensorPlan::new(
        "layer.0.mixer.gated_norm",
        "blk.0.ssm_norm.weight",
        RepresentationKind::Bf16,
        TYPE_BF16,
        vec![5120, 17408],
        vec![],
        vec![],
        None,
    )
    .unwrap();
    assert!(
        plan.value.is_empty(),
        "llama.cpp exempts this norm, and here the exemption is the graph declaring no offset"
    );
}

/// **Value transforms must not move a dimension.** Geometry is derived
/// by folding the source shape through the LAYOUT transforms only, so a
/// value transform cannot influence it even by mistake — the type
/// system keeps it out of the fold rather than a comment asking nicely.
#[test]
fn value_transforms_do_not_change_geometry() {
    let with_value = LoweredTensorPlan::new(
        "layer.0.mixer.log_decay",
        "blk.0.ssm_a",
        RepresentationKind::F32,
        TYPE_F32,
        vec![48],
        vec![],
        vec![ValueTransform::MaterializeLogDecay],
        None,
    )
    .unwrap();
    assert_eq!(with_value.source_shape, vec![48]);
    assert_eq!(
        with_value.target_shape,
        vec![48],
        "materialising -exp changes numbers, not cardinality"
    );
}

/// The reorders permute within an axis, so they preserve dims. Only the
/// squeeze changes rank, and only for a singleton.
#[test]
fn layout_transforms_have_declared_shape_effects() {
    let reordered = LoweredTensorPlan::new(
        "layer.0.mixer.out_proj",
        "blk.0.ssm_out.weight",
        RepresentationKind::Nvfp4,
        TYPE_NVFP4,
        vec![5120, 6144],
        vec![LayoutTransform::ReorderVColumnsByGroups {
            key_heads: 16,
            v_per_k: 3,
            groups_per_head: 8,
        }],
        vec![],
        None,
    )
    .unwrap();
    assert_eq!(
        reordered.target_shape,
        vec![5120, 6144],
        "a permutation preserves dims"
    );

    let squeezed = LoweredTensorPlan::new(
        "layer.0.mixer.conv",
        "blk.0.ssm_conv1d.weight",
        RepresentationKind::Bf16,
        TYPE_BF16,
        vec![10240, 1, 4],
        vec![LayoutTransform::SqueezeSingletonAxis { axis: 1 }],
        vec![],
        None,
    )
    .unwrap();
    assert_eq!(squeezed.target_shape, vec![10240, 4]);

    // And a real channel axis refuses at plan time, not at write time.
    assert!(LoweredTensorPlan::new(
        "layer.0.mixer.conv",
        "blk.0.ssm_conv1d.weight",
        RepresentationKind::Bf16,
        TYPE_BF16,
        vec![10240, 2, 4],
        vec![LayoutTransform::SqueezeSingletonAxis { axis: 1 }],
        vec![],
        None,
    )
    .is_err());
}
