//! A-9.1 gates: attention sinks and Q/K/V/O projection biases travel
//! checkpoint → surface → operand closure → `AttentionCall` → every
//! backend, and each is **load-bearing** — the same discipline that
//! caught YaRN's two loss mechanisms in A-9.0.
//!
//! ```text
//! parity     golden (denominator sink, plain adds) ≡ reference ≡ production ≡ device
//! causal     perturb q/k/v/o bias, perturb sinks  → output moves (5 positive controls)
//! absence    the bias-free plan serialises exactly as before; closure refuses a
//!            bias operand nobody declared, a sink operand nobody judged, and a
//!            declaration nothing backs (fail-closed both ways)
//! ```
//!
//! Glimmer's judged family carries neither, so the sink judgment is added
//! to the *persisted graph* — the container is the authority the
//! executor reads (the `controls` module's rule), and a judgment absent
//! from it must refuse.

use super::golden::{
    executor_trace_from, golden_forward, max_abs, miniature_glimmer, miniature_glimmer_with,
    MiniatureExtras, BIAS_SUFFIXES, G_LAYERS, G_TOKENS, SINKS_SUFFIX,
};
use crate::format::vindex3::encode::{encode_system, SYSTEM_GRAPH_JSON};
use crate::format::vindex3::graph::OperandRole;
use crate::format::vindex3::inspect::inspect_container;
use crate::format::vindex3::opplan::exec::backend::WeightFormat;
use crate::format::vindex3::opplan::exec::device::DevicePlanBackend;
use crate::format::vindex3::opplan::exec::operands::OperandStore;
use crate::format::vindex3::opplan::exec::production::ProductionBackend;
use crate::format::vindex3::opplan::exec::reference::ReferenceBackend;
use crate::format::vindex3::opplan::exec::{execute_plan, ExecutionTrace};
use crate::format::vindex3::opplan::{plan_component_ops, ClosureDefect, OpPlanOutcome};

/// Reassociation-only tolerance between independent f32 transcriptions.
const GOLDEN_TOLERANCE: f32 = 1e-5;
/// A perturbed operand must move the layer that consumes it by far more
/// than fp noise: 100× the parity tolerance (measured deltas sit near
/// 1e-1 on the 12-wide fixture). The logits only have to move above the
/// parity tolerance — the miniature's head compresses hard
/// (`output_multiplier` 0.196, softcap 20), so end-to-end deltas are
/// small even for a large layer-level change; the site is where the
/// causal claim is made, the logits prove it propagates.
const CAUSAL_FLOOR: f32 = 1e-3;

/// Encode `dir` and hand back the container.
fn encoded(dir: &std::path::Path) -> tempfile::TempDir {
    let inventory = larql_models::inventory::build_inventory(dir).unwrap();
    let container = tempfile::tempdir().unwrap();
    encode_system(&[("mini-glimmer".to_string(), inventory)], container.path()).unwrap();
    container
}

/// Write the sink judgment into the persisted surface — the fact Glimmer's
/// family does not carry, stated where the executor reads it.
fn judge_sinks(container: &std::path::Path) {
    let path = container.join(SYSTEM_GRAPH_JSON);
    let mut graph: serde_json::Value =
        serde_json::from_str(&std::fs::read_to_string(&path).unwrap()).unwrap();
    let target = graph["components"]
        .as_array_mut()
        .unwrap()
        .iter_mut()
        .find(|c| c["id"] == "target")
        .unwrap();
    target["execution"]["attention"]["sinks"] = serde_json::json!("softmax_denominator");
    std::fs::write(&path, graph.to_string()).unwrap();
}

/// A fixture carrying both extras, encoded and judged.
fn full_fixture(perturb: Option<&'static str>) -> (tempfile::TempDir, tempfile::TempDir) {
    let dir = tempfile::tempdir().unwrap();
    miniature_glimmer_with(
        dir.path(),
        MiniatureExtras {
            attention_bias: true,
            sinks: true,
            perturb,
        },
    );
    let container = encoded(dir.path());
    judge_sinks(container.path());
    (dir, container)
}

fn closure(container: &std::path::Path) -> OpPlanOutcome {
    let inspection = inspect_container(container, false).unwrap();
    plan_component_ops(&inspection, container, "target").unwrap()
}

// ── Parity: the golden semantics through every backend ──

#[test]
fn biases_and_sinks_match_the_independent_golden_semantics() {
    let (dir, container) = full_fixture(None);
    let golden = golden_forward(dir.path());
    let executed = executor_trace_from(container.path());
    for layer in 0..G_LAYERS {
        let attn = max_abs(
            &executed.layers[layer].post_attention,
            &golden.layers[layer].post_attention,
        );
        assert!(
            attn < GOLDEN_TOLERANCE,
            "layer {layer} post_attention {attn:e}"
        );
        let post = max_abs(
            &executed.layers[layer].post_layer,
            &golden.layers[layer].post_layer,
        );
        assert!(post < GOLDEN_TOLERANCE, "layer {layer} post_layer {post:e}");
    }
    let worst = executed
        .logits
        .unwrap()
        .iter()
        .zip(&golden.logits)
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f32, f32::max);
    assert!(worst < GOLDEN_TOLERANCE, "logits {worst:e}");
}

fn assert_traces_agree(a: &ExecutionTrace, b: &ExecutionTrace, label: &str) {
    for (index, (da, db)) in a.layers.iter().zip(&b.layers).enumerate() {
        let delta = max_abs(&da.post_layer, &db.post_layer);
        assert!(delta < GOLDEN_TOLERANCE, "{label}: layer {index} {delta:e}");
    }
    let logits = max_abs(
        std::slice::from_ref(a.logits.as_ref().unwrap()),
        std::slice::from_ref(b.logits.as_ref().unwrap()),
    );
    assert!(logits < GOLDEN_TOLERANCE, "{label}: logits {logits:e}");
}

#[test]
fn every_backend_agrees_on_biases_and_sinks() {
    let (_dir, container) = full_fixture(None);
    let inspection = inspect_container(container.path(), false).unwrap();
    let outcome = closure(container.path());
    assert!(outcome.closed(), "{:?}", outcome.defects);
    let plan = outcome.plan.unwrap();
    let store = OperandStore::open(container.path(), &inspection).unwrap();
    let reference = execute_plan(&plan, &store, &G_TOKENS, &ReferenceBackend::new()).unwrap();
    let production = execute_plan(&plan, &store, &G_TOKENS, &ProductionBackend::new()).unwrap();
    let device = DevicePlanBackend::new(
        super::device::LoopDevice,
        "loop-device-sinks-bias",
        WeightFormat::F32,
    );
    let on_device = execute_plan(&plan, &store, &G_TOKENS, &device).unwrap();
    assert_traces_agree(&production, &reference, "production vs reference");
    assert_traces_agree(&on_device, &reference, "device vs reference");
}

// ── Causal: each operand moves the output ──

#[test]
fn each_bias_and_the_sinks_are_load_bearing() {
    let (_dir, baseline) = full_fixture(None);
    let base = executor_trace_from(baseline.path());
    for suffix in BIAS_SUFFIXES.iter().copied().chain([SINKS_SUFFIX]) {
        let (_dir, container) = full_fixture(Some(suffix));
        let moved = executor_trace_from(container.path());
        let delta = max_abs(
            std::slice::from_ref(base.logits.as_ref().unwrap()),
            std::slice::from_ref(moved.logits.as_ref().unwrap()),
        );
        let attn0 = max_abs(
            &base.layers[0].post_attention,
            &moved.layers[0].post_attention,
        );
        eprintln!("perturb {suffix}: layer0 post_attention {attn0:e}, logits {delta:e}");
        assert!(
            attn0 > CAUSAL_FLOOR,
            "perturbing `{suffix}` moved layer 0's attention output by only {attn0:e} — the \
             operand is carried but not executed"
        );
        assert!(
            delta > GOLDEN_TOLERANCE,
            "perturbing `{suffix}` moved layer 0 by {attn0:e} but the logits by only {delta:e}"
        );
    }
}

// ── Absence: nothing changes for a model without them, and closure is
//    fail-closed in both directions ──

#[test]
fn a_bias_free_plan_serialises_without_the_new_fields() {
    let dir = tempfile::tempdir().unwrap();
    miniature_glimmer(dir.path());
    let container = encoded(dir.path());
    let outcome = closure(container.path());
    let plan = serde_json::to_value(outcome.plan.unwrap()).unwrap();
    let text = plan.to_string();
    for key in ["q_bias", "k_bias", "v_bias", "o_bias", "\"sinks\""] {
        assert!(!text.contains(key), "bias-free plan carries `{key}`");
    }
    let surface = std::fs::read_to_string(container.path().join(SYSTEM_GRAPH_JSON)).unwrap();
    assert!(!surface.contains("attention_bias") && !surface.contains("\"sinks\""));
}

#[test]
fn bias_operands_without_a_declaration_refuse() {
    // Tensors present, `attention_bias` undeclared: the operands imply an
    // op the surface does not carry.
    let dir = tempfile::tempdir().unwrap();
    miniature_glimmer_with(
        dir.path(),
        MiniatureExtras {
            attention_bias: true,
            ..Default::default()
        },
    );
    let mut config: serde_json::Value =
        serde_json::from_str(&std::fs::read_to_string(dir.path().join("config.json")).unwrap())
            .unwrap();
    config.as_object_mut().unwrap().remove("attention_bias");
    std::fs::write(dir.path().join("config.json"), config.to_string()).unwrap();
    let container = encoded(dir.path());
    let outcome = closure(container.path());
    assert!(!outcome.closed());
    let implied = outcome
        .defects
        .iter()
        .filter(|d| matches!(d, ClosureDefect::OperandImpliesAbsentOp { tensor, .. } if tensor.contains("_proj.bias")))
        .count();
    assert_eq!(implied, 4 * G_LAYERS, "{:?}", outcome.defects);
}

#[test]
fn a_declaration_without_bias_operands_refuses() {
    let dir = tempfile::tempdir().unwrap();
    miniature_glimmer(dir.path());
    let mut config: serde_json::Value =
        serde_json::from_str(&std::fs::read_to_string(dir.path().join("config.json")).unwrap())
            .unwrap();
    config["attention_bias"] = serde_json::json!(true);
    std::fs::write(dir.path().join("config.json"), config.to_string()).unwrap();
    let container = encoded(dir.path());
    let outcome = closure(container.path());
    assert!(!outcome.closed());
    for role in [
        OperandRole::AttnQBias,
        OperandRole::AttnKBias,
        OperandRole::AttnVBias,
        OperandRole::AttnOBias,
    ] {
        let missing = outcome
            .defects
            .iter()
            .filter(|d| matches!(d, ClosureDefect::MissingOperand { role: r, .. } if *r == role))
            .count();
        assert_eq!(missing, G_LAYERS, "{role:?}: {:?}", outcome.defects);
    }
}

#[test]
fn a_sink_operand_without_a_judgment_refuses() {
    let dir = tempfile::tempdir().unwrap();
    miniature_glimmer_with(
        dir.path(),
        MiniatureExtras {
            sinks: true,
            ..Default::default()
        },
    );
    let container = encoded(dir.path());
    // Not judged: the operand implies an op the surface does not carry.
    let outcome = closure(container.path());
    assert!(!outcome.closed());
    let implied = outcome
        .defects
        .iter()
        .filter(|d| matches!(d, ClosureDefect::OperandImpliesAbsentOp { tensor, .. } if tensor.ends_with(SINKS_SUFFIX)))
        .count();
    assert_eq!(implied, G_LAYERS, "{:?}", outcome.defects);
    // Judged: it closes, and the plan carries the sinks per layer.
    judge_sinks(container.path());
    let outcome = closure(container.path());
    assert!(outcome.closed(), "{:?}", outcome.defects);
    let plan = outcome.plan.unwrap();
    assert!(plan.layers.iter().all(|l| l.attention.sinks.is_some()));
}

#[test]
fn a_sink_judgment_without_the_operand_refuses() {
    let dir = tempfile::tempdir().unwrap();
    miniature_glimmer(dir.path());
    let container = encoded(dir.path());
    judge_sinks(container.path());
    let outcome = closure(container.path());
    assert!(!outcome.closed());
    let missing = outcome
        .defects
        .iter()
        .filter(|d| {
            matches!(
                d,
                ClosureDefect::MissingOperand {
                    role: OperandRole::AttnSinks,
                    ..
                }
            )
        })
        .count();
    assert_eq!(missing, G_LAYERS, "{:?}", outcome.defects);
}
