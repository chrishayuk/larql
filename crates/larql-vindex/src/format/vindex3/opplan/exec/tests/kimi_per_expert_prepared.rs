//! K3-RESIDENCY-VERTICAL-1, V1: a PER-EXPERT bank with a shared expert
//! executes through the prepared plan — bound as a mapping of the stored
//! bytes, never read or copied, selected and pinned like every other
//! operand, reconciled exactly, and computing the same numbers on the
//! production kernels as the reference's literal transcription.
//!
//! The subject is the bytes-backed Kimi-shaped miniature
//! (`fixtures_kimi::kimi_per_expert_moe_f32_model`): plain softmax
//! attention so the plan executes today, one dense prefix layer, two
//! routed layers of four experts (top-2) plus a shared expert. Its twin,
//! with one expert's bytes replaced by another's, is the control that
//! proves the executor read the expert it selected.

use std::path::Path;

use super::super::accounting::BlockGeometry;
use super::super::backend::{ExpertSlices, RoutedFfnCall, WeightFormat};
use super::super::execute_plan;
use super::super::operands::OperandStore;
use super::super::prepared::{ExecutionSlice, PreparedOperands};
use super::super::production::ProductionBackend;
use super::super::realization::{RealizationForm, SelectionReason};
use super::super::reference::ReferenceBackend;
use crate::format::vindex3::fixtures::encode_fixture_container;
use crate::format::vindex3::fixtures_kimi::{
    kimi_per_expert_moe_f32_model, kimi_per_expert_moe_f32_model_routing,
    kimi_per_expert_moe_f32_model_with, KimiRouting, MOE_DENSE_PREFIX, MOE_EXPERTS, MOE_LAYERS,
    MOE_TOP_K,
};
use crate::format::vindex3::inspect::inspect_container;
use crate::format::vindex3::opplan::planned::Operation;
use crate::format::vindex3::opplan::{plan_component_ops, ComponentOpPlan};

/// Prompt over the miniature's 64-token vocabulary.
const TOKENS: [u32; 4] = [3, 17, 28, 11];
/// Naive-loop vs BLAS/f32 on a 32-wide model: reassociation only.
const TOLERANCE: f32 = 2e-5;
/// A wrong expert's bytes move the logits far above that.
const CAUSAL_FLOOR: f32 = 1e-3;
const ROUTED_LAYERS: usize = MOE_LAYERS - MOE_DENSE_PREFIX;
const MATRICES_PER_EXPERT: usize = 3;
const SHARED_PROJECTIONS: usize = 3;

struct Subject {
    _src: tempfile::TempDir,
    container: tempfile::TempDir,
}

impl Subject {
    fn build(write: fn(&Path)) -> Self {
        let src = tempfile::tempdir().unwrap();
        let container = tempfile::tempdir().unwrap();
        encode_fixture_container(write, src.path(), container.path(), "kimi-moe");
        Self {
            _src: src,
            container,
        }
    }

    fn open(&self) -> (ComponentOpPlan, OperandStore) {
        let inspection = inspect_container(self.container.path(), false).unwrap();
        let plan = plan_component_ops(&inspection, self.container.path(), "target")
            .unwrap()
            .plan
            .expect("the per-expert miniature plans");
        let store = OperandStore::open(self.container.path(), &inspection).unwrap();
        (plan, store)
    }
}

fn twin_with_expert_one_as_zero(dir: &Path) {
    kimi_per_expert_moe_f32_model_with(dir, |e| if e == 1 { 0 } else { e as u64 });
}

fn max_abs(a: &[f32], b: &[f32]) -> f32 {
    a.iter()
        .zip(b)
        .map(|(x, y)| (x - y).abs())
        .fold(0.0, f32::max)
}

/// The plan lists every expert matrix under the bank-scoped operation
/// and the shared expert under its own — multiplicity preserved.
#[test]
fn the_plan_lists_the_bank_per_matrix_and_the_shared_expert() {
    let subject = Subject::build(kimi_per_expert_moe_f32_model);
    let (plan, _store) = subject.open();
    let planned = plan.planned_operands();
    let bank = Operation::ExpertProject {
        experts: MOE_EXPERTS,
        top_k: MOE_TOP_K,
    };
    assert_eq!(
        planned.iter().filter(|p| p.operation == bank).count(),
        ROUTED_LAYERS * MOE_EXPERTS * MATRICES_PER_EXPERT
    );
    assert_eq!(
        planned
            .iter()
            .filter(|p| p.operation == Operation::SharedExpertProject)
            .count(),
        ROUTED_LAYERS * SHARED_PROJECTIONS
    );
}

/// Preparation binds the bank as a mapping — one object mapped, one
/// region per matrix, no payload byte of the bank read — pins every
/// expert to the mapped stored form, and reconciles exactly: the bank's
/// declared residency IS its stored bytes, with no padding.
#[test]
fn the_bank_is_bound_once_as_a_mapping_and_reconciles_exactly() {
    let subject = Subject::build(kimi_per_expert_moe_f32_model);
    let (plan, store) = subject.open();
    let loads_before = store.load_count();
    let ops = PreparedOperands::load(
        &plan,
        &store,
        &ProductionBackend::new(),
        ExecutionSlice::Full,
    )
    .unwrap();
    let expert_regions = ROUTED_LAYERS * MOE_EXPERTS * MATRICES_PER_EXPERT;
    assert_eq!(
        store.mapped_objects(),
        1,
        "one physical binding: the expert bank"
    );
    assert_eq!(store.mapped_regions(), expert_regions);
    // The bank's object was bound, not read: every load the loader made
    // came from the decoder stack, the embedding and the head.
    let loads = (store.load_count() - loads_before) as usize;
    let non_bank = plan.planned_operands().len() - expert_regions;
    assert!(
        loads <= non_bank + plan.layers.len() * 4,
        "loads {loads} exceed the non-bank operands plus their norms ({non_bank})"
    );
    let bank: Vec<_> = ops
        .realizations()
        .iter()
        .filter(|r| matches!(r.planned.operation, Operation::ExpertProject { .. }))
        .collect();
    assert_eq!(bank.len(), expert_regions);
    for r in &bank {
        assert_eq!(
            r.selection.realization.form,
            RealizationForm::MappedStored {
                format: WeightFormat::F32
            },
            "{}",
            r.planned.operand.tensor
        );
        assert_eq!(r.selection.reason, SelectionReason::BankMappedAsStored);
    }
    let expected = ops.expectations((&store).into(), BlockGeometry::executor());
    let is_bank = |op: &Operation| matches!(op, Operation::ExpertProject { .. });
    let bank_declared: u64 = expected
        .iter()
        .filter(|e| is_bank(&e.operation))
        .map(|e| e.declared_resident)
        .sum();
    let bank_stored: u64 = expected
        .iter()
        .filter(|e| is_bank(&e.operation))
        .map(|e| e.stored_bytes)
        .sum();
    assert_eq!(
        bank_declared, bank_stored,
        "mapped as stored: declared == stored"
    );
    assert!(expected
        .iter()
        .filter(|e| is_bank(&e.operation))
        .all(|e| e.staging == 0));
    let reconciled = ops.reconcile(&plan, (&store).into()).unwrap();
    assert_eq!(reconciled.matched, expected.len());
    // Every observed object of the bank is the mapping itself: resident
    // exactly the stored bytes, no padded allocation of the loader's own.
    let observed = ops.bound(&plan).unwrap();
    let bank_observed: Vec<_> = observed.iter().filter(|o| is_bank(&o.operation)).collect();
    assert_eq!(bank_observed.len(), expert_regions);
    assert!(bank_observed
        .iter()
        .all(|o| o.allocations == 0 && o.format == WeightFormat::F32));
    let bank_mapped: u64 = bank_observed.iter().map(|o| o.mapped_bytes).sum();
    assert_eq!(
        bank_mapped, bank_stored,
        "mapped address space == stored bytes"
    );
    assert_eq!(reconciled.mapped, bank_stored);
    // The pages resident are a fact of the moment: at most the mapping,
    // and reported apart from it.
    assert!(reconciled.mapped_resident <= reconciled.mapped);
    let bank_resident: u64 = bank_observed.iter().map(|o| o.resident_bytes).sum();
    assert_eq!(bank_resident, reconciled.mapped_resident);
}

/// The production kernels over the mapped bank compute what the
/// reference's literal transcription computes — and a twin container in
/// which expert 1 holds expert 0's bytes computes something else, so the
/// expert that was selected is the expert that was read.
#[test]
fn production_matches_the_reference_over_the_mapped_bank_and_the_twin_differs() {
    let subject = Subject::build(kimi_per_expert_moe_f32_model);
    let (plan, store) = subject.open();
    let production = execute_plan(&plan, &store, &TOKENS, &ProductionBackend::new()).unwrap();
    let reference = execute_plan(&plan, &store, &TOKENS, &ReferenceBackend).unwrap();
    let (p, r) = (
        production.logits.as_ref().expect("head"),
        reference.logits.as_ref().expect("head"),
    );
    let delta = max_abs(p, r);
    assert!(delta < TOLERANCE, "production vs reference: {delta:e}");
    assert!(
        p.iter().any(|v| v.abs() > 0.0),
        "the logits are not identically zero"
    );

    let twin = Subject::build(twin_with_expert_one_as_zero);
    let (twin_plan, twin_store) = twin.open();
    let twinned =
        execute_plan(&twin_plan, &twin_store, &TOKENS, &ProductionBackend::new()).unwrap();
    let moved = max_abs(p, twinned.logits.as_ref().expect("head"));
    assert!(
        moved > CAUSAL_FLOOR,
        "a wrong expert's bytes must move the logits: {moved:e}"
    );
}

// ── The routed scale acts on the routed sum, and on nothing else ───────

fn declared_routing(dir: &Path) {
    kimi_per_expert_moe_f32_model_routing(dir, |e| e as u64, KimiRouting::DECLARED);
}
fn unscaled_routing(dir: &Path) {
    kimi_per_expert_moe_f32_model_routing(
        dir,
        |e| e as u64,
        KimiRouting {
            routed_scaling_factor: None,
            ..KimiRouting::DECLARED
        },
    );
}
fn no_shared_expert(dir: &Path) {
    kimi_per_expert_moe_f32_model_routing(
        dir,
        |e| e as u64,
        KimiRouting {
            shared_experts: 0,
            ..KimiRouting::DECLARED
        },
    );
}
fn unscaled_and_no_shared_expert(dir: &Path) {
    kimi_per_expert_moe_f32_model_routing(
        dir,
        |e| e as u64,
        KimiRouting {
            routed_scaling_factor: None,
            shared_experts: 0,
        },
    );
}

/// The FFN's contribution at the first routed layer: what the layer
/// added to the residual after attention, per position.
fn routed_layer_ffn(write: fn(&Path)) -> Vec<Vec<f32>> {
    let subject = Subject::build(write);
    let (plan, store) = subject.open();
    let trace = execute_plan(&plan, &store, &TOKENS, &ProductionBackend::new()).unwrap();
    let layer = &trace.layers[MOE_DENSE_PREFIX];
    layer
        .post_layer
        .try_rows()
        .unwrap()
        .iter()
        .zip(layer.post_attention.try_rows().unwrap())
        .map(|(out, residual)| out.iter().zip(residual).map(|(o, r)| o - r).collect())
        .collect()
}

fn combine(a: &[Vec<f32>], b: &[Vec<f32>], f: impl Fn(f32, f32) -> f32) -> Vec<Vec<f32>> {
    a.iter()
        .zip(b)
        .map(|(x, y)| x.iter().zip(y).map(|(p, q)| f(*p, *q)).collect())
        .collect()
}

fn assert_rows_close(got: &[Vec<f32>], want: &[Vec<f32>], what: &str) {
    let scale = want.iter().flatten().fold(0.0f32, |m, v| m.max(v.abs()));
    assert!(
        scale > CAUSAL_FLOOR,
        "{what}: the expected side is ~0 ({scale:e}), the check is vacuous"
    );
    for (g, w) in got.iter().zip(want) {
        let delta = max_abs(g, w);
        assert!(
            delta <= TOLERANCE * scale,
            "{what}: off by {delta:e} against magnitude {scale:e}"
        );
    }
}

/// Four containers that differ in one declaration each — the routed
/// scale present or absent, the shared expert present or absent — and
/// the algebra between their first routed layer's FFN outputs: the scale
/// multiplies the routed sum alone, and the shared expert's contribution
/// is the same whether or not the routed branch is scaled. No backend
/// is compared with itself here; the witness is the invariant.
#[test]
fn the_branch_scale_multiplies_the_routed_sum_and_leaves_the_shared_expert_alone() {
    let scaled_with_shared = routed_layer_ffn(declared_routing);
    let unscaled_with_shared = routed_layer_ffn(unscaled_routing);
    let scaled_routed_only = routed_layer_ffn(no_shared_expert);
    let unscaled_routed_only = routed_layer_ffn(unscaled_and_no_shared_expert);
    let scale = KimiRouting::DECLARED.routed_scaling_factor.unwrap() as f32;

    // The scale is a multiplier on the routed sum.
    let expected_scaled = combine(&unscaled_routed_only, &unscaled_routed_only, |v, _| {
        v * scale
    });
    assert_rows_close(&scaled_routed_only, &expected_scaled, "scale × routed sum");

    // The shared expert's contribution does not depend on the routed scale.
    let shared_under_scale = combine(&scaled_with_shared, &scaled_routed_only, |a, b| a - b);
    let shared_unscaled = combine(&unscaled_with_shared, &unscaled_routed_only, |a, b| a - b);
    assert_rows_close(
        &shared_under_scale,
        &shared_unscaled,
        "shared expert under the scale",
    );
}

// ── The refusals around a mapping, by name ─────────────────────────────

/// A region is bound only for an object the store holds, a tensor the
/// segment names, and a length the declared geometry implies; every
/// other request is refused by name, and a second region of the same
/// object shares its one mapping.
#[test]
fn a_region_is_refused_by_name_for_the_wrong_object_tensor_or_length() {
    use crate::format::vindex3::opplan::OperandRef;
    let subject = Subject::build(kimi_per_expert_moe_f32_model);
    let (plan, store) = subject.open();
    let expert = plan
        .planned_operands()
        .into_iter()
        .find(|p| matches!(p.operation, Operation::ExpertProject { .. }))
        .expect("a routed layer plans experts")
        .operand;
    let bytes = (expert.shape.iter().product::<usize>() * std::mem::size_of::<f32>()) as u64;
    let region = store.map_region(&expert, bytes).unwrap();
    assert_eq!(region.len(), bytes);
    let again = store.map_region(&expert, bytes).unwrap();
    assert_eq!(again.len(), bytes);
    assert_eq!(store.mapped_objects(), 1, "one mapping serves both regions");
    assert_eq!(store.mapped_regions(), 2);

    let err = store
        .map_region(&expert, bytes + 4)
        .unwrap_err()
        .to_string();
    assert!(err.contains("declared geometry implies"), "{err}");
    let missing = OperandRef {
        tensor: "no.such.tensor".into(),
        ..expert.clone()
    };
    let err = store.map_region(&missing, bytes).unwrap_err().to_string();
    assert!(err.contains("no tensor `no.such.tensor`"), "{err}");
    let elsewhere = OperandRef {
        object: "target.no_such_object".into(),
        ..expert
    };
    let err = store.map_region(&elsewhere, bytes).unwrap_err().to_string();
    assert!(err.contains("no segment for object"), "{err}");
    assert_eq!(store.mapped_regions(), 2, "a refusal binds nothing");
}

/// The mapping's two figures are read apart: its address space is every
/// region's length, its resident pages are what the OS says now — never
/// more than the address space, and a prepared image with nothing mapped
/// reports nothing. The decode session reads the same figures through
/// the image it borrows.
#[test]
fn the_mapping_reports_its_address_space_apart_from_its_resident_pages() {
    use super::super::decode::DecodeSession;
    use super::super::kv::RowKvState;
    let subject = Subject::build(kimi_per_expert_moe_f32_model);
    let (plan, store) = subject.open();
    let backend = ProductionBackend::new();
    let ops = PreparedOperands::load(&plan, &store, &backend, ExecutionSlice::Full).unwrap();
    let residency = ops.mapped_residency();
    assert_eq!(residency.regions, store.mapped_regions());
    assert_eq!(
        residency.mapped_bytes,
        ops.residency_census().mapped() as u64,
        "the address space is the census's mapped bytes"
    );
    assert!(residency.mapped_bytes > 0);
    assert!(
        residency.resident_bytes <= residency.mapped_bytes,
        "resident {} beyond the mapping {}",
        residency.resident_bytes,
        residency.mapped_bytes
    );
    let mut kv = RowKvState::default();
    let session = DecodeSession::over_prepared(&plan, &ops, &backend, &mut kv).unwrap();
    let through_session = session.mapped_residency();
    assert_eq!(through_session.regions, residency.regions);
    assert_eq!(through_session.mapped_bytes, residency.mapped_bytes);

    // Nothing mapped: the dense stack binds no bank.
    let dense = Subject::build(crate::format::vindex3::fixtures::dense_f32_model);
    let (plan, store) = dense.open();
    let ops = PreparedOperands::load(&plan, &store, &backend, ExecutionSlice::Full).unwrap();
    assert_eq!(ops.mapped_residency(), Default::default());
}

/// A routed call over `experts` per-expert matrices, at hand-checkable
/// sizes, with whatever bias the test wants to put on it.
fn separate_call<'a>(
    x: &'a [f32],
    router: &'a [f32],
    weights: ExpertSlices<'a>,
    down_bias: Option<&'a [f32]>,
    inter: usize,
    experts: usize,
) -> RoutedFfnCall<'a> {
    use larql_models::config::{ExpertRoutingPolicy, MoeRouterKind};
    use larql_models::{Activation, ExpertGatePolicy};
    RoutedFfnCall {
        x,
        hidden: x.len(),
        intermediate: inter,
        experts,
        top_k: 1,
        router_kind: MoeRouterKind::TopKThenSoftmax,
        routing_policy: ExpertRoutingPolicy::NormalisedOverSelected,
        branch_scale: 1.0,
        activation: Activation::Silu,
        gate_policy: ExpertGatePolicy::Gated,
        router,
        router_bias: None,
        weights,
        gate_up_bias: None,
        down_bias,
        router_input: None,
        router_scale: None,
        router_per_expert_scale: None,
        router_norm_eps: None,
    }
}

/// Both CPU backends refuse a per-expert call that carries an expert
/// bias — no layout is defined for one — and the reference refuses a
/// stored form it cannot transcribe, naming it.
#[test]
fn a_separate_expert_call_is_refused_where_it_cannot_be_executed() {
    use super::super::backend::{ExpertSlices, PlanBackend, WeightSlice};
    let inter = 2;
    let experts = 2;
    let x = vec![0.5f32, -0.25, 1.0, 0.75];
    let hidden = x.len();
    let router = vec![0.1f32; experts * hidden];
    let gate_f32 = vec![0.01f32; inter * hidden];
    let down_f32 = vec![0.02f32; hidden * inter];
    let gate = vec![WeightSlice::F32(&gate_f32); experts];
    let down = vec![WeightSlice::F32(&down_f32); experts];
    let codes = vec![0i8; inter * hidden];
    let scales = vec![1.0f32; 1];
    let sums = vec![0i16; 0];
    let q8 = vec![
        WeightSlice::Q8 {
            codes: &codes,
            scales: &scales,
            sums: &sums,
            block: inter * hidden,
        };
        experts
    ];
    let bias = vec![0.0f32; experts * hidden];
    let f32_experts = || ExpertSlices::Separate {
        gate: &gate,
        up: &gate,
        down: &down,
    };
    // Executes, on both, once nothing is refusable.
    let production = ProductionBackend::new()
        .routed_ffn(separate_call(
            &x,
            &router,
            f32_experts(),
            None,
            inter,
            experts,
        ))
        .unwrap();
    let reference = ReferenceBackend
        .routed_ffn(separate_call(
            &x,
            &router,
            f32_experts(),
            None,
            inter,
            experts,
        ))
        .unwrap();
    assert_eq!(production.len(), hidden);
    assert!(max_abs(&production, &reference) < TOLERANCE);
    // A bias on a per-expert bank is a plan neither executor knows.
    for backend in [
        &ProductionBackend::new() as &dyn PlanBackend,
        &ReferenceBackend,
    ] {
        let err = backend
            .routed_ffn(separate_call(
                &x,
                &router,
                f32_experts(),
                Some(&bias),
                inter,
                experts,
            ))
            .unwrap_err()
            .to_string();
        assert!(err.contains("carries no expert bias"), "{err}");
    }
    // The reference transcribes f32 and bf16 only.
    let err = ReferenceBackend
        .routed_ffn(separate_call(
            &x,
            &router,
            ExpertSlices::Separate {
                gate: &q8,
                up: &q8,
                down: &down,
            },
            None,
            inter,
            experts,
        ))
        .unwrap_err()
        .to_string();
    assert!(err.contains("q8 expert"), "{err}");
}

/// The reference names every stored form it cannot transcribe, and
/// transcribes bf16 exactly: a bf16 per-expert bank computes on the
/// reference what the production bf16 kernel computes over the same
/// mapped-form slices.
#[test]
fn the_reference_widens_bf16_experts_exactly_and_names_every_other_form() {
    use super::super::backend::{ExpertSlices, PlanBackend, WeightSlice};
    use larql_models::quant::half::f32_to_bf16;
    let inter = 2;
    let experts = 2;
    let x = vec![0.5f32, -0.25, 1.0, 0.75];
    let hidden = x.len();
    let router = vec![0.1f32; experts * hidden];
    let gate_f32: Vec<f32> = (0..inter * hidden).map(|i| 0.01 * i as f32).collect();
    let down_f32: Vec<f32> = (0..hidden * inter)
        .map(|i| 0.02 * i as f32 - 0.05)
        .collect();
    let gate_bf16: Vec<u16> = gate_f32.iter().map(|v| f32_to_bf16(*v)).collect();
    let down_bf16: Vec<u16> = down_f32.iter().map(|v| f32_to_bf16(*v)).collect();
    let gate = vec![WeightSlice::Bf16(&gate_bf16); experts];
    let down = vec![WeightSlice::Bf16(&down_bf16); experts];
    let bf16 = || ExpertSlices::Separate {
        gate: &gate,
        up: &gate,
        down: &down,
    };
    let production = ProductionBackend::new()
        .routed_ffn(separate_call(&x, &router, bf16(), None, inter, experts))
        .unwrap();
    let reference = ReferenceBackend
        .routed_ffn(separate_call(&x, &router, bf16(), None, inter, experts))
        .unwrap();
    assert!(production.iter().any(|v| v.abs() > 0.0));
    assert!(max_abs(&production, &reference) < TOLERANCE);

    // Every other stored form is refused by name.
    let packed = vec![0u8; inter * hidden];
    let scales_f32 = vec![1.0f32; 1];
    let scales_u8 = vec![0u8; 1];
    let down_f32_slices = vec![WeightSlice::F32(&down_f32); experts];
    let forms: Vec<(Vec<WeightSlice<'_>>, &str)> = vec![
        (
            vec![
                WeightSlice::Q4 {
                    packed: &packed,
                    scales: &scales_f32,
                    block: inter * hidden,
                };
                experts
            ],
            "q4",
        ),
        (vec![WeightSlice::F16(&packed); experts], "f16"),
        (
            vec![
                WeightSlice::Mxfp4 {
                    packed: &packed,
                    scales: &scales_u8,
                };
                experts
            ],
            "mxfp4",
        ),
        (
            vec![
                WeightSlice::Nvfp4 {
                    packed: &packed,
                    scales: &scales_u8,
                    tensor_scale: 1.0,
                };
                experts
            ],
            "nvfp4",
        ),
    ];
    for (slices, name) in &forms {
        let err = ReferenceBackend
            .routed_ffn(separate_call(
                &x,
                &router,
                ExpertSlices::Separate {
                    gate: slices,
                    up: slices,
                    down: &down_f32_slices,
                },
                None,
                inter,
                experts,
            ))
            .unwrap_err()
            .to_string();
        assert!(err.contains(&format!("a {name} expert")), "{name}: {err}");
    }
}
