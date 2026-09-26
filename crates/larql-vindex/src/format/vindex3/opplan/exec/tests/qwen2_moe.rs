//! **Qwen2-MoE through the generic plan**: Q/K/V-only attention bias
//! (`qkv_bias`) and a shared expert under its own sigmoid gate.
//!
//! Closure holds each declaration to the shipped operands in both
//! directions, the two executors agree, and each of the two semantics is
//! shown to be load-bearing by removing it and watching the logits move.
//! Parity against an independent implementation is the committed
//! conformance fixture's job; these pin the mechanism.

use crate::format::vindex3::encode::encode_system;
use crate::format::vindex3::fixtures_qwen::{
    miniature_qwen2_moe, miniature_qwen2_moe_with, BiasDeclaration, QwenMoeForm, QWEN_LAYERS,
    QWEN_TOKENS,
};
use crate::format::vindex3::inspect::inspect_container;
use crate::format::vindex3::opplan::exec::backend::PlanBackend;
use crate::format::vindex3::opplan::exec::execute_plan;
use crate::format::vindex3::opplan::exec::operands::OperandStore;
use crate::format::vindex3::opplan::exec::production::ProductionBackend;
use crate::format::vindex3::opplan::exec::reference::ReferenceBackend;
use crate::format::vindex3::opplan::{plan_component_ops, ClosureDefect, LayerFfn, OpPlanOutcome};

const COMPONENT: &str = "target";
/// Reference and production run the same arithmetic in different orders.
const BACKEND_TOLERANCE: f32 = 1e-5;
/// A semantic that matters moves some logit by far more than rounding.
const LOAD_BEARING: f32 = 1e-3;

fn encoded(form: QwenMoeForm) -> (tempfile::TempDir, tempfile::TempDir) {
    let checkpoint = tempfile::tempdir().unwrap();
    miniature_qwen2_moe_with(checkpoint.path(), form);
    let container = tempfile::tempdir().unwrap();
    encode(checkpoint.path(), container.path()).unwrap();
    (checkpoint, container)
}

fn encode(checkpoint: &std::path::Path, container: &std::path::Path) -> Result<(), String> {
    let inventory = larql_models::inventory::build_inventory(checkpoint).unwrap();
    encode_system(&[(COMPONENT.to_string(), inventory)], container)
        .map(|_| ())
        .map_err(|e| e.to_string())
}

/// The encode refusal for `form`: encoding holds operand closure, so a
/// container that would not close is never written.
fn refusal(form: QwenMoeForm) -> String {
    let checkpoint = tempfile::tempdir().unwrap();
    miniature_qwen2_moe_with(checkpoint.path(), form);
    let container = tempfile::tempdir().unwrap();
    encode(checkpoint.path(), container.path()).expect_err("a container that does not close")
}

fn closure(container: &std::path::Path) -> OpPlanOutcome {
    let inspection = inspect_container(container, false).unwrap();
    plan_component_ops(&inspection, container, COMPONENT).unwrap()
}

fn last_logits<B: PlanBackend>(form: QwenMoeForm, backend: &B) -> Vec<f32> {
    let (_checkpoint, container) = encoded(form);
    let outcome = closure(container.path());
    assert!(outcome.closed(), "defects: {:?}", outcome.defects);
    let plan = outcome.plan.unwrap();
    let inspection = inspect_container(container.path(), false).unwrap();
    let store = OperandStore::open(container.path(), &inspection).unwrap();
    execute_plan(&plan, &store, &QWEN_TOKENS, backend)
        .unwrap()
        .logits
        .unwrap()
}

fn max_abs_diff(a: &[f32], b: &[f32]) -> f32 {
    assert_eq!(a.len(), b.len());
    a.iter()
        .zip(b)
        .map(|(x, y)| (x - y).abs())
        .fold(0.0, f32::max)
}

#[test]
fn a_qwen2_moe_plan_biases_q_k_v_and_gates_its_shared_expert() {
    for declaration in [BiasDeclaration::QkvBias, BiasDeclaration::Silent] {
        let (_checkpoint, container) = encoded(QwenMoeForm {
            declaration,
            ..Default::default()
        });
        let outcome = closure(container.path());
        assert!(
            outcome.closed(),
            "{declaration:?}: defects {:?}",
            outcome.defects
        );
        let plan = outcome.plan.unwrap();
        assert_eq!(plan.layers.len(), QWEN_LAYERS);
        for layer in &plan.layers {
            let attn = layer.attention.softmax().expect("softmax attention");
            assert!(
                attn.q_bias.is_some() && attn.k_bias.is_some() && attn.v_bias.is_some(),
                "{declaration:?}: Q/K/V biases bound"
            );
            assert!(attn.o_bias.is_none(), "{declaration:?}: output unbiased");
            let Some(LayerFfn::Routed(op)) = &layer.ffn else {
                panic!("routed layer");
            };
            let shared = op.shared.as_ref().expect("shared expert");
            assert!(shared.branch_gate.is_some(), "the shared branch is gated");
        }
    }
}

#[test]
fn the_two_executors_agree_on_a_qwen2_moe_forward() {
    let form = QwenMoeForm::default();
    let reference = last_logits(form, &ReferenceBackend);
    let production = last_logits(form, &ProductionBackend::new());
    let drift = max_abs_diff(&reference, &production);
    assert!(drift < BACKEND_TOLERANCE, "backends drift by {drift}");
}

#[test]
fn the_q_k_v_biases_and_the_branch_gate_are_each_load_bearing() {
    let backend = ProductionBackend::new();
    let plain = last_logits(QwenMoeForm::default(), &backend);
    for (what, form) in [
        (
            "zeroed Q/K/V biases",
            QwenMoeForm {
                bias_values: false,
                ..Default::default()
            },
        ),
        (
            "a zeroed branch gate",
            QwenMoeForm {
                zero_branch_gate: true,
                ..Default::default()
            },
        ),
    ] {
        let moved = max_abs_diff(&plain, &last_logits(form, &backend));
        assert!(
            moved > LOAD_BEARING,
            "{what} moved the logits by only {moved}: the semantic is not executing"
        );
    }
}

#[test]
fn an_output_bias_under_qkv_bias_is_refused_by_name() {
    let err = refusal(QwenMoeForm {
        output_bias: true,
        ..Default::default()
    });
    assert_eq!(
        err.matches("OperandImpliesAbsentOp").count(),
        QWEN_LAYERS,
        "{err}"
    );
    assert!(
        err.contains("o_proj.bias") && err.contains("`qkv_bias` biases Q/K/V only"),
        "{err}"
    );
}

#[test]
fn declaring_both_bias_forms_is_a_contradiction_not_a_choice() {
    let err = refusal(QwenMoeForm {
        declaration: BiasDeclaration::Both,
        ..Default::default()
    });
    assert!(err.contains("ContradictoryDeclaration"), "{err}");
    assert!(
        err.contains("`attention_bias`") && err.contains("`qkv_bias`"),
        "{err}"
    );
    let defect = ClosureDefect::ContradictoryDeclaration {
        component: COMPONENT.into(),
        detail: "x".into(),
    };
    assert!(defect.to_string().contains("contradictory declaration"));
}

#[test]
fn q_k_v_bias_operands_without_any_declaration_refuse() {
    // Silent Qwen2 is admitted by its family default; an explicit
    // `qkv_bias: false` beside shipped bias tensors must refuse them.
    let checkpoint = tempfile::tempdir().unwrap();
    miniature_qwen2_moe(checkpoint.path());
    let config_path = checkpoint.path().join("config.json");
    let mut config: serde_json::Value =
        serde_json::from_str(&std::fs::read_to_string(&config_path).unwrap()).unwrap();
    config["qkv_bias"] = serde_json::json!(false);
    std::fs::write(&config_path, config.to_string()).unwrap();
    let container = tempfile::tempdir().unwrap();
    let err = encode(checkpoint.path(), container.path()).expect_err("undeclared bias operands");
    assert_eq!(
        err.matches("OperandImpliesAbsentOp").count(),
        3 * QWEN_LAYERS,
        "{err}"
    );
    assert!(
        err.contains("declared `attention_bias` or `qkv_bias`"),
        "{err}"
    );
}

#[test]
fn the_branch_scale_is_the_sigmoid_of_one_logit_over_the_block_input() {
    use crate::format::vindex3::opplan::exec::experts::shared_branch_scale;
    use larql_models::config::{
        GateActivation, GateCombine, SharedExpertGateSource, SharedExpertGateSpec,
    };
    let spec = SharedExpertGateSpec {
        source: SharedExpertGateSource::HiddenStateToScalar,
        activation: GateActivation::Sigmoid,
        combine: GateCombine::ElementwiseMultiply,
    };
    let x = [1.0, -2.0, 0.5];
    assert_eq!(shared_branch_scale(&spec, &[0.0; 3], &x).unwrap(), 0.5);
    // w · x = 2 - 2 + 1 = 1.
    let scale = shared_branch_scale(&spec, &[2.0, 1.0, 2.0], &x).unwrap();
    assert!((scale - 1.0 / (1.0 + (-1.0f32).exp())).abs() < f32::EPSILON);
    let err = shared_branch_scale(&spec, &[1.0; 2], &x)
        .unwrap_err()
        .to_string();
    assert!(err.contains("2 weights for a 3-wide input"), "{err}");
}
