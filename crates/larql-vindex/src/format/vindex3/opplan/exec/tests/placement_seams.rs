//! **The seams expert and FFN placement cut through.**
//!
//! Routed placement is admitted only for the stacks it was built for, a
//! selected expert is resolved out of its bank without changing any
//! matrix, and a backend that cannot run a relocated expert refuses by
//! name rather than computing something else. The `Arc` forwarding a
//! runtime shares its backend through must carry each of these calls to
//! the backend it wraps — an arm that fell through to the trait default
//! would turn a supported placement into a refusal.

use std::sync::Arc;

use larql_models::config::{ExpertRoutingPolicy, GateUpLayout, MoeRouterKind};
use larql_models::{Activation, ExpertGatePolicy};

use crate::error::VindexError;
use crate::format::vindex3::fixtures::{encode_fixture_container, miniature_glimmer};
use crate::format::vindex3::fixtures_routed::miniature_gpt_oss;
use crate::format::vindex3::inspect::inspect_container;
use crate::format::vindex3::opplan::exec::backend::{
    ExpertSlices, PlanBackend, RoutedFfnCall, WeightSlice,
};
use crate::format::vindex3::opplan::exec::prepared::ExecutionSlice;
use crate::format::vindex3::opplan::exec::production::ProductionBackend;
use crate::format::vindex3::opplan::exec::realization::MappedAccess;
use crate::format::vindex3::opplan::exec::reference::ReferenceBackend;
use crate::format::vindex3::opplan::exec::requirements::required_objects;
use crate::format::vindex3::opplan::exec::routed_experts::{
    ExpertOutput, ExpertTransformCall, ExpertWeights, RoutedExpertProvider,
};
use crate::format::vindex3::opplan::{plan_component_ops, ComponentOpPlan, LayerFfn};

const COMPONENT: &str = "target";

/// The smallest expert: one hidden unit, one intermediate unit, so a fused
/// gate/up is two values and a down is one.
const HIDDEN: usize = 1;
const INTER: usize = 1;
const EXPERTS: usize = 2;
const TOP_K: usize = 1;

fn plan(write: impl FnOnce(&std::path::Path)) -> ComponentOpPlan {
    let checkpoint = tempfile::tempdir().unwrap();
    let container = tempfile::tempdir().unwrap();
    encode_fixture_container(write, checkpoint.path(), container.path(), COMPONENT);
    let inspection = inspect_container(container.path(), false).unwrap();
    let outcome = plan_component_ops(&inspection, container.path(), COMPONENT).unwrap();
    assert!(outcome.closed(), "defects: {:?}", outcome.defects);
    outcome.plan.unwrap()
}

fn routed_plan() -> ComponentOpPlan {
    plan(|dir| miniature_gpt_oss(dir, false))
}

fn routed_refusal(plan: &ComponentOpPlan) -> String {
    required_objects(plan, &ExecutionSlice::RoutedExpertCoordinator)
        .unwrap_err()
        .to_string()
}

#[test]
fn routed_placement_is_admitted_only_for_the_stack_it_was_built_for() {
    let admitted = routed_plan();
    assert!(required_objects(&admitted, &ExecutionSlice::RoutedExpertCoordinator).is_ok());

    let dense = plan(miniature_glimmer);
    assert!(routed_refusal(&dense).contains("routed FFNs on every layer"));

    let mut empty = admitted.clone();
    empty.layers.clear();
    assert!(routed_refusal(&empty).contains("single-stream softmax stack"));

    // One operator field outside the admitted envelope is enough.
    let mut no_selection = admitted.clone();
    let Some(LayerFfn::Routed(op)) = &mut no_selection.layers[0].ffn else {
        panic!("routed fixture")
    };
    op.top_k = 0;
    assert!(routed_refusal(&no_selection).contains("packed MXFP4 experts"));
}

/// Two fused experts over `WeightSlice::F32`, and the call that carries them.
struct Bank {
    gate_up: Vec<Vec<f32>>,
    down: Vec<Vec<f32>>,
}

impl Bank {
    fn new() -> Self {
        Self {
            gate_up: (0..EXPERTS).map(|e| vec![0.5 + e as f32, 0.25]).collect(),
            down: (0..EXPERTS).map(|e| vec![1.0 + e as f32]).collect(),
        }
    }
}

fn call<'a>(
    x: &'a [f32],
    router: &'a [f32],
    gate_up: &'a [WeightSlice<'a>],
    down: &'a [WeightSlice<'a>],
) -> RoutedFfnCall<'a> {
    RoutedFfnCall {
        x,
        hidden: HIDDEN,
        intermediate: INTER,
        experts: EXPERTS,
        top_k: TOP_K,
        router_kind: MoeRouterKind::default(),
        routing_policy: ExpertRoutingPolicy::SoftmaxThenSelect,
        branch_scale: 1.0,
        activation: Activation::Silu,
        gate_policy: ExpertGatePolicy::Gated,
        router,
        router_bias: None,
        weights: ExpertSlices::Fused {
            gate_up,
            down,
            layout: GateUpLayout::Interleaved,
        },
        gate_up_bias: None,
        down_bias: None,
        router_input: None,
        router_scale: None,
        router_per_expert_scale: None,
        router_norm_eps: None,
    }
}

fn slices(values: &[Vec<f32>]) -> Vec<WeightSlice<'_>> {
    values.iter().map(|v| WeightSlice::F32(v)).collect()
}

const X: [f32; HIDDEN] = [0.75];
const ROUTER: [f32; EXPERTS * HIDDEN] = [1.0, -1.0];

#[test]
fn a_selected_expert_resolves_only_inside_its_bank() {
    let bank = Bank::new();
    let (gate_up, down) = (slices(&bank.gate_up), slices(&bank.down));
    let fused = call(&X, &ROUTER, &gate_up, &down);
    assert!(fused.expert_transform(EXPERTS - 1).is_ok());
    let err = fused.expert_transform(EXPERTS).err().unwrap().to_string();
    assert!(err.contains("outside bank"), "{err}");

    // A per-expert bank stores no expert bias; a call claiming one is a
    // caller error, not a bias to drop.
    let bias = [0.0; EXPERTS * HIDDEN];
    let mut separate = call(&X, &ROUTER, &gate_up, &down);
    separate.weights = ExpertSlices::Separate {
        gate: &gate_up,
        up: &gate_up,
        down: &down,
        access: MappedAccess::default(),
    };
    separate.down_bias = Some(&bias);
    let err = separate.expert_transform(0).err().unwrap().to_string();
    assert!(err.contains("carries no expert bias"), "{err}");
}

fn transform<'a>(x: &'a [f32], bank: &'a [WeightSlice<'a>; 2]) -> ExpertTransformCall<'a> {
    ExpertTransformCall {
        x,
        hidden: HIDDEN,
        intermediate: INTER,
        activation: Activation::Silu,
        gate_policy: ExpertGatePolicy::Gated,
        weights: ExpertWeights::Fused {
            gate_up: bank[0],
            down: bank[1],
            layout: GateUpLayout::Interleaved,
            gate_up_bias: None,
            down_bias: None,
        },
    }
}

/// A provider that must never be reached: the backends under test refuse
/// placement before asking it for anything.
struct Unreachable;
impl RoutedExpertProvider for Unreachable {
    fn apply(&self, _: usize, _: &[f32], _: &[usize]) -> Result<Vec<ExpertOutput>, VindexError> {
        unreachable!("a backend that refuses placement must not call its provider")
    }
}

fn outcome(result: Result<Vec<f32>, VindexError>) -> Result<Vec<u32>, String> {
    result
        .map(|row| row.iter().map(|v| v.to_bits()).collect())
        .map_err(|e| e.to_string())
}

#[test]
fn a_shared_backend_forwards_every_placement_call_to_the_backend_it_wraps() {
    let bank = Bank::new();
    let (gate_up, down) = (slices(&bank.gate_up), slices(&bank.down));
    let expert = [
        WeightSlice::F32(&bank.gate_up[0]),
        WeightSlice::F32(&bank.down[0]),
    ];

    let production = ProductionBackend::new();
    let shared = Arc::new(ProductionBackend::new());
    let direct = outcome(production.expert_transform(transform(&X, &expert)));
    assert!(direct.is_ok(), "{direct:?}");
    assert_eq!(
        outcome(shared.expert_transform(transform(&X, &expert))),
        direct
    );
    assert_eq!(
        outcome(shared.routed_ffn(call(&X, &ROUTER, &gate_up, &down))),
        outcome(production.routed_ffn(call(&X, &ROUTER, &gate_up, &down)))
    );

    // The reference backend runs whole routed layers but relocates
    // nothing: both placement entry points refuse, naming the provider,
    // directly and through the share.
    let reference = Arc::new(ReferenceBackend);
    for backend in [&ReferenceBackend as &dyn PlanBackend, &reference] {
        let err = outcome(backend.expert_transform(transform(&X, &expert))).unwrap_err();
        assert!(
            err.contains("does not support selected expert transforms"),
            "{err}"
        );
        let err =
            outcome(backend.routed_ffn_placed(call(&X, &ROUTER, &gate_up, &down), 0, &Unreachable))
                .unwrap_err();
        assert!(
            err.contains("does not support routed expert placement"),
            "{err}"
        );
        assert!(err.contains(backend.name()), "{err}");
    }
}

#[test]
fn offline_attribution_materialises_bf16_and_refuses_a_block_scaled_form() {
    let values = [1.5_f32, -2.0];
    let bits: Vec<u16> = values.iter().map(|v| (v.to_bits() >> 16) as u16).collect();
    assert_eq!(WeightSlice::Bf16(&bits).decode_f32(1, 2).unwrap(), values);

    // FP8 blocks are a CPU projection form attribution does not unpack:
    // the rows resolve, and materialisation refuses by name.
    let codes = [0_u8; 2];
    let scales = [1.0_f32];
    let fp8 = |codes| WeightSlice::Fp8Block {
        codes,
        scales: &scales,
        block_rows: 1,
        block_cols: 2,
        scale_cols: 1,
    };
    assert_eq!(fp8(&codes).representation(), "fp8-block");
    let err = fp8(&codes).decode_f32(1, 2).unwrap_err().to_string();
    assert!(err.contains("cannot materialise"), "{err}");
    // One byte per element: a slab merely long enough is not this shape.
    let err = fp8(&[0_u8; 3]).rows(1, 2).err().unwrap().to_string();
    assert!(err.contains("do not describe a 1 x 2 matrix"), "{err}");
}
