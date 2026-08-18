//! Op-plan builder arms not reached by the plan/closure gates.
//!
//! The [`LayerFfn`] projections are exclusive views — a dense layer has
//! no routed op and a routed layer has no dense op — and the misplaced-
//! operand defect renders the object kind it belongs in. Neither is
//! observable through the dense fixtures the sibling gates encode.

use larql_models::config::{
    Activation, ExpertFormat, ExpertRoutingPolicy, GateUpLayout, MoeRouterKind,
};
use larql_models::ExpertGatePolicy;

use super::encoded_fixture;
use crate::format::vindex3::graph::ObjectKind;
use crate::format::vindex3::opplan::{
    plan_component_ops, ClosureDefect, LayerFfn, OperandRef, PackedProjection, RoutedFfnOp,
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

/// A routed FFN op exactly as the builder would emit it for a per-expert
/// (unpacked, unbiased) mixture — the shape the accessors are asked about.
fn routed_layer() -> LayerFfn {
    LayerFfn::Routed(Box::new(RoutedFfnOp {
        experts: ROUTED_EXPERTS,
        top_k: ROUTED_TOP_K,
        expert_intermediate_size: ROUTED_INTER,
        router_kind: MoeRouterKind::TopKSoftmax,
        routing_policy: ExpertRoutingPolicy::SoftmaxThenSelect,
        activation: Activation::Silu,
        gate_policy: ExpertGatePolicy::Gated,
        expert_format: ExpertFormat::PerExpert,
        gate_up_layout: Some(GateUpLayout::ContiguousHalves),
        router: operand(
            STACK_OBJECT,
            ROUTER_TENSOR,
            vec![ROUTED_EXPERTS, ROUTED_HIDDEN],
        ),
        router_bias: None,
        gate_up: PackedProjection {
            weights: operand(
                BANK_OBJECT,
                GATE_UP_TENSOR,
                vec![ROUTED_EXPERTS, 2 * ROUTED_INTER, ROUTED_HIDDEN],
            ),
            scales: None,
            bias: None,
        },
        down: PackedProjection {
            weights: operand(
                BANK_OBJECT,
                DOWN_TENSOR,
                vec![ROUTED_EXPERTS, ROUTED_HIDDEN, ROUTED_INTER],
            ),
            scales: None,
            bias: None,
        },
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
    assert_eq!(routed.gate_up.weights.object, BANK_OBJECT);
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
