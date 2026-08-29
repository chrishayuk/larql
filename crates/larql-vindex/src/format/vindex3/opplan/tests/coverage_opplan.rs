//! Op-plan builder arms not reached by the plan/closure gates.
//!
//! The [`LayerFfn`] projections are exclusive views — a dense layer has
//! no routed op and a routed layer has no dense op — and the misplaced-
//! operand defect renders the object kind it belongs in. Neither is
//! observable through the dense fixtures the sibling gates encode.

use larql_models::config::{
    Activation, ExpertFormat, ExpertRoutingPolicy, GateUpLayout, MoeRouterKind, NormType,
};
use larql_models::ExpertGatePolicy;

use super::encoded_fixture;
use crate::format::vindex3::graph::ObjectKind;
use crate::format::vindex3::opplan::{
    plan_component_ops, ClosureDefect, ExpertBank, FfnOp, GatedDeltaOp, HybridFfnOp,
    LayerAttention, LayerFfn, NormOp, OperandRef, PackedProjection, RoutedFfnOp,
};

/// Geometry of the hand-built routed op — small, but every dimension is
/// distinct so an accessor returning the wrong field would be visible.
const ROUTED_EXPERTS: usize = 4;
const ROUTED_TOP_K: usize = 2;
const ROUTED_HIDDEN: usize = 32;
const ROUTED_INTER: usize = 48;
const STACK_OBJECT: &str = "target.decoder_stack";
const BANK_OBJECT: &str = "target.expert_bank";
const ROUTER_TENSOR: &str = "0.mlp.router.weight";
const GATE_UP_TENSOR: &str = "0.mlp.experts.gate_up_proj";
const DOWN_TENSOR: &str = "0.mlp.experts.down_proj";
const F32_DTYPE: &str = "F32";

fn operand(object: &str, tensor: &str, shape: Vec<usize>) -> OperandRef {
    OperandRef {
        object: object.to_string(),
        tensor: tensor.to_string(),
        dtype: F32_DTYPE.to_string(),
        shape,
    }
}

/// A routed FFN op exactly as the builder would emit it for a packed
/// (unbiased) mixture — the shape the accessors are asked about. Packed
/// rather than `ExpertFormat::PerExpert`: the tensor names below are one
/// fused `gate_up_proj`/`down_proj` per bank, which is what a packed
/// format ships — `ExpertBank::PerExpert`'s own construction is covered in
/// `moe_spellings.rs`/the real Kimi-shaped fixtures, not here.
fn routed_layer() -> LayerFfn {
    LayerFfn::Routed(Box::new(RoutedFfnOp {
        experts: ROUTED_EXPERTS,
        top_k: ROUTED_TOP_K,
        expert_intermediate_size: ROUTED_INTER,
        router_kind: MoeRouterKind::TopKSoftmax,
        routing_policy: ExpertRoutingPolicy::SoftmaxThenSelect,
        activation: Activation::Silu,
        gate_policy: ExpertGatePolicy::Gated,
        expert_format: ExpertFormat::PackedBF16,
        gate_up_layout: Some(GateUpLayout::ContiguousHalves),
        router: operand(
            STACK_OBJECT,
            ROUTER_TENSOR,
            vec![ROUTED_EXPERTS, ROUTED_HIDDEN],
        ),
        router_bias: None,
        bank: ExpertBank::Packed {
            gate_up: Box::new(PackedProjection {
                weights: operand(
                    BANK_OBJECT,
                    GATE_UP_TENSOR,
                    vec![ROUTED_EXPERTS, 2 * ROUTED_INTER, ROUTED_HIDDEN],
                ),
                scales: None,
                bias: None,
            }),
            down: Box::new(PackedProjection {
                weights: operand(
                    BANK_OBJECT,
                    DOWN_TENSOR,
                    vec![ROUTED_EXPERTS, ROUTED_HIDDEN, ROUTED_INTER],
                ),
                scales: None,
                bias: None,
            }),
        },
        shared: None,
        router_scale: None,
        router_per_expert_scale: None,
        router_norm_eps: None,
    }))
}

/// A routed layer answers `routed()` with its own op and `dense()` with
/// nothing — the two projections are exclusive, never a lossy view of
/// each other.
#[test]
fn a_routed_layer_exposes_its_routed_op_and_no_dense_op() {
    let layer = routed_layer();
    let routed = layer.routed().expect("routed layer carries a routed op");
    assert_eq!(routed.experts, ROUTED_EXPERTS);
    assert_eq!(routed.top_k, ROUTED_TOP_K);
    assert_eq!(routed.expert_intermediate_size, ROUTED_INTER);
    assert_eq!(routed.router.tensor, ROUTER_TENSOR);
    let ExpertBank::Packed { gate_up, .. } = &routed.bank else {
        panic!("routed_layer() always builds a packed bank");
    };
    assert_eq!(gate_up.weights.object, BANK_OBJECT);
    assert!(
        layer.dense().is_none(),
        "a routed layer must not present a dense op"
    );
}

/// The planned dense fixture: every layer answers `dense()` and none
/// answers `routed()` — the accessor reads the variant, not a default.
#[test]
fn a_planned_dense_layer_exposes_no_routed_op() {
    let fixture = encoded_fixture();
    let plan = plan_component_ops(&fixture.inspection, &fixture.root, "target")
        .unwrap()
        .plan
        .unwrap();
    assert!(!plan.layers.is_empty());
    for layer in &plan.layers {
        assert!(
            layer.ffn.routed().is_none(),
            "layer {}: dense plan presented a routed op",
            layer.layer
        );
        assert!(layer.ffn.dense().is_some(), "layer {}", layer.layer);
    }
}

/// A misplaced-operand defect names the tensor, the object it was found
/// in, and the kind of object it belongs in — by the kind's own name,
/// so the report is a work item without a lookup.
#[test]
fn a_misplaced_operand_defect_renders_where_the_operand_belongs() {
    let defect = ClosureDefect::MisplacedOperand {
        object: STACK_OBJECT.to_string(),
        tensor: GATE_UP_TENSOR.to_string(),
        belongs_in: ObjectKind::ExpertBank,
    };
    let rendered = defect.to_string();
    assert_eq!(
        rendered,
        format!(
            "misplaced operand: {STACK_OBJECT}/{GATE_UP_TENSOR} belongs in the {} object",
            ObjectKind::ExpertBank.name()
        )
    );
    assert!(rendered.contains("expert_bank"), "{rendered}");
}

// ── LayerFfn::Hybrid — the third, dual-branch projection ────────────────
//
// `routed_layer()` above pins the Routed/Dense exclusivity; a hybrid
// layer needs its own hand-built op because no fixture in this crate
// encodes a hybrid checkpoint yet (Qwen3.8-shaped, not this crate's
// dense/routed test models) — same reasoning as the routed op above.

fn norm(tensor: &str) -> NormOp {
    NormOp {
        kind: NormType::RmsNorm,
        eps: 1e-6,
        weight_offset: 0.0,
        weight: operand(STACK_OBJECT, tensor, vec![ROUTED_HIDDEN]),
    }
}

fn hybrid_layer() -> LayerFfn {
    LayerFfn::Hybrid(Box::new(HybridFfnOp {
        dense: FfnOp {
            intermediate_size: ROUTED_INTER,
            activation: Activation::Silu,
            gate_policy: ExpertGatePolicy::Gated,
            gate: Some(operand(STACK_OBJECT, "0.mlp.dense.gate_proj", vec![])),
            up: operand(STACK_OBJECT, "0.mlp.dense.up_proj", vec![]),
            down: operand(STACK_OBJECT, "0.mlp.dense.down_proj", vec![]),
        },
        routed: match routed_layer() {
            LayerFfn::Routed(op) => *op,
            _ => unreachable!("routed_layer() always returns LayerFfn::Routed"),
        },
        pre_experts_norm: norm("0.mlp.pre_experts_norm"),
        post_dense_norm: norm("0.mlp.post_dense_norm"),
        post_experts_norm: norm("0.mlp.post_experts_norm"),
    }))
}

/// A hybrid layer answers `hybrid()` with its own op and neither
/// `dense()` nor `routed()` — the two branches are reached only through
/// the combined view, the same "no lossy partial read" rule as
/// [`a_routed_layer_exposes_its_routed_op_and_no_dense_op`].
#[test]
fn a_hybrid_layer_exposes_its_hybrid_op_and_neither_dense_nor_routed() {
    let layer = hybrid_layer();
    let hybrid = layer.hybrid().expect("hybrid layer carries a hybrid op");
    assert_eq!(hybrid.dense.activation, Activation::Silu);
    assert_eq!(hybrid.routed.experts, ROUTED_EXPERTS);
    assert!(layer.dense().is_none(), "hybrid must not answer dense()");
    assert!(layer.routed().is_none(), "hybrid must not answer routed()");
}

// ── LayerAttention::GatedDelta — the linear-attention branch ────────────
//
// Every plan/closure fixture in this crate is softmax-only (no
// crate-local fixture declares a `linear_attention` layer type yet —
// Qwen3.8's DeltaNet ladder is tracked separately), so `softmax()`'s
// `GatedDelta => None` arm, `gated_delta()`'s `Some` arm, and
// `declared_name()`'s `GatedDelta` arm all need a hand-built layer the
// same way the routed/hybrid FFN ops above do.

fn gated_delta_op() -> GatedDeltaOp {
    // Qwen3.8's own linear-layer geometry — see gated_delta.rs's own
    // state_elements() test for why real numbers, not placeholders.
    GatedDeltaOp {
        num_key_heads: 16,
        num_value_heads: 48,
        key_head_dim: 128,
        value_head_dim: 128,
        conv_kernel: 4,
        state_dtype: Some(larql_models::inventory::report::RecurrentStateDtype::Float32),
        in_proj_qkv: operand(STACK_OBJECT, "0.linear_attn.in_proj_qkv", vec![]),
        in_proj_a: operand(STACK_OBJECT, "0.linear_attn.in_proj_a", vec![]),
        in_proj_b: operand(STACK_OBJECT, "0.linear_attn.in_proj_b", vec![]),
        in_proj_z: operand(STACK_OBJECT, "0.linear_attn.in_proj_z", vec![]),
        conv1d: operand(STACK_OBJECT, "0.linear_attn.conv1d", vec![]),
        a_log: operand(STACK_OBJECT, "0.linear_attn.a_log", vec![]),
        dt_bias: operand(STACK_OBJECT, "0.linear_attn.dt_bias", vec![]),
        norm: operand(STACK_OBJECT, "0.linear_attn.norm", vec![]),
        out_proj: operand(STACK_OBJECT, "0.linear_attn.out_proj", vec![]),
    }
}

/// A DeltaNet layer answers `gated_delta()` with its own op and neither
/// `softmax()` nor `softmax_mut()` — the KV-shaped view a softmax
/// backend needs simply does not exist for a layer with no per-position
/// key/value to retain (see gated_delta.rs's module doc). Its declared
/// name is the checkpoint's own `layer_types` spelling, not the softmax
/// span's.
#[test]
fn a_gated_delta_layer_exposes_its_op_and_no_softmax_op() {
    let mut layer = LayerAttention::GatedDelta(Box::new(gated_delta_op()));
    assert!(
        layer.gated_delta().is_some(),
        "gated-delta layer must answer gated_delta()"
    );
    assert!(layer.softmax().is_none(), "must not answer softmax()");
    assert!(
        layer.softmax_mut().is_none(),
        "must not answer softmax_mut()"
    );
    assert_eq!(
        layer.declared_name(),
        larql_models::config::LAYER_TYPE_LINEAR_ATTENTION
    );
}

/// The mirror image: every softmax layer the real planner produces
/// answers `gated_delta()` with nothing, and its declared name comes
/// from the attention span (not the linear-attention constant above) —
/// pinned against the real dense fixture so this can't silently agree
/// with a hand-built stub instead of what the planner actually emits.
#[test]
fn a_planned_softmax_layer_exposes_no_gated_delta_op() {
    let fixture = encoded_fixture();
    let plan = plan_component_ops(&fixture.inspection, &fixture.root, "target")
        .unwrap()
        .plan
        .unwrap();
    assert!(!plan.layers.is_empty());
    for layer in &plan.layers {
        assert!(
            layer.attention.gated_delta().is_none(),
            "layer {}: softmax plan presented a gated-delta op",
            layer.layer
        );
        assert_ne!(
            layer.attention.declared_name(),
            larql_models::config::LAYER_TYPE_LINEAR_ATTENTION,
            "layer {}",
            layer.layer
        );
    }
}
