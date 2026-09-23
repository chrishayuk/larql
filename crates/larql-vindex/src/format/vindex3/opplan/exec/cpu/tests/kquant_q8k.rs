//! Q8K-ACT-1's arm at the executor: the Q8_K-activation plan is observed
//! off the resident binding, runs exactly the kernel it declares (gate
//! F0, `docs/q8k-act-1.md`), is selected only by the provider built for
//! it, and leaves every other arm's answer unchanged.

use larql_compute::cpu::ops::q4k_q8k_dot::{q4k_q8k_matvec_parallel, quantize_x_to_q8k};

use super::super::arithmetic::{AccumulatorRep, ActivationRep, ScaleSpan, WeightRep};
use super::super::ledger::ledger;
use super::super::physical::{KQuantExecution, PhysicalProjectionPlan, Q8K_ACTIVATION_BLOCK};
use super::super::projector::WeightRows;
use crate::format::vindex3::opplan::exec::backend::{
    KQuantActivation, MatrixClass, PlanBackend, WeightFormat,
};
use crate::format::vindex3::opplan::exec::production::{select_cpu, ProductionBackend};
use crate::format::vindex3::opplan::exec::realization::{
    RealizationForm, RealizationId, RepresentationFacts,
};
use crate::format::vindex3::opplan::planned::{Operation, PlannedOperand};
use crate::format::vindex3::opplan::OperandRef;
use crate::format::vindex3::represent::codec::RepresentationExtent;
use crate::format::vindex3::represent::kquant::{KQuant, Q4_K, Q6_K, Q8_0};

/// Two super-blocks per row, so a scale read at the wrong block shows.
const ROWS: usize = 5;
const IN_DIM: usize = 512;

fn weights() -> Vec<f32> {
    (0..ROWS * IN_DIM)
        .map(|i| ((i % 31) as f32 - 15.0) * 0.02 * (1.0 + (i / IN_DIM) as f32))
        .collect()
}

/// An activation with an outlier channel per block, the regime C0 found
/// on real Gemma inputs — a smooth vector would hide the block scale.
fn activation() -> Vec<f32> {
    (0..IN_DIM)
        .map(|i| {
            if i % 97 == 0 {
                7.5
            } else {
                ((i % 11) as f32 - 5.0) / 13.0
            }
        })
        .collect()
}

fn rows(blocks: &[u8], codec: KQuant, activation: KQuantActivation) -> WeightRows<'_> {
    WeightRows::KQuant {
        blocks,
        codec,
        activation,
    }
}

/// F0: the arm's kernel, reached by observation, is bit-identical to the
/// kernel it declares — `quantize_x_to_q8k` then `q4k_q8k_matvec_parallel`
/// — on the same input. Anything else would mean the plan name and the
/// arithmetic had come apart.
#[test]
fn f0_the_q8k_plan_runs_exactly_the_declared_kernel() {
    let x = activation();
    for codec in [Q4_K, Q6_K] {
        let blocks = codec.encode(&weights(), "t").expect("encode");
        let resident = rows(&blocks, codec, KQuantActivation::Q8k);
        let plan = PhysicalProjectionPlan::for_resident(resident, IN_DIM);
        assert_eq!(
            plan,
            PhysicalProjectionPlan::FusedKQuantQ8k,
            "{}",
            codec.name
        );
        assert_eq!(plan.format(), WeightFormat::KQuantQ8k, "{}", codec.name);

        let mut arm = vec![0.0f32; ROWS];
        plan.kernel().project_rows(resident, &x, &mut arm);

        let mut declared = vec![0.0f32; ROWS];
        let q = quantize_x_to_q8k(&x);
        q4k_q8k_matvec_parallel(&mut declared, &q, &blocks, ROWS, IN_DIM, codec.name)
            .expect("declared kernel");
        let bits = |v: &[f32]| v.iter().map(|f| f.to_bits()).collect::<Vec<_>>();
        assert_eq!(
            bits(&arm),
            bits(&declared),
            "{} is not the declared kernel",
            codec.name
        );

        // And it is not the f32 arm by accident: on an outlier-bearing
        // activation the two arithmetics must differ somewhere.
        let mut f32_arm = vec![0.0f32; ROWS];
        PhysicalProjectionPlan::FusedKQuant.kernel().project_rows(
            rows(&blocks, codec, KQuantActivation::F32),
            &x,
            &mut f32_arm,
        );
        assert_ne!(
            bits(&arm),
            bits(&f32_arm),
            "{}: the activation form changed nothing",
            codec.name
        );
    }
}

/// The executor's own dispatch — the path a decode takes — lands on the
/// codec's Q8_K kernel.
///
/// This does NOT read the process-wide ledger. An earlier form asserted
/// the `FusedKQuantQ8k` call count rose across the call, and raced: the
/// ledger is global, tests run in parallel, and any of them may
/// `reset()` it between the two reads. That the call is booked under its
/// own plan is pinned without the global instrument — F0 asserts
/// `for_resident` observes `FusedKQuantQ8k`, and `ledger.rs` pins that the
/// plan has a slot of its own.
#[test]
fn the_executor_dispatch_reaches_the_q8k_kernel() {
    let x = activation();
    let blocks = Q4_K.encode(&weights(), "t").expect("encode");
    let out =
        super::super::physical::project_rows(rows(&blocks, Q4_K, KQuantActivation::Q8k), &x, ROWS)
            .expect("executor");
    let direct = Q4_K
        .gemv_q8k(&blocks, &x, ROWS, IN_DIM)
        .expect("codec kernel");
    assert_eq!(out, direct);
}

/// The arithmetic the plan reports is the one it runs: stored K-quant
/// weights, an int8 activation at Q8_K's 256-wide scale, integer blocks.
#[test]
fn the_plan_reports_a_q8k_activation() {
    let a = PhysicalProjectionPlan::FusedKQuantQ8k.arithmetic();
    assert_eq!(a.weight, WeightRep::KQuant);
    assert_eq!(
        a.activation,
        ActivationRep::Q8 {
            span: ScaleSpan::Block(Q8K_ACTIVATION_BLOCK)
        }
    );
    assert_eq!(a.accumulator, AccumulatorRep::I32);
    assert!(ledger()
        .all()
        .iter()
        .any(|(p, _)| *p == PhysicalProjectionPlan::FusedKQuantQ8k));
}

/// Q8_0 has no Q8_K kernel: the codec says so, rather than answering
/// with a kernel that reads the wrong layout.
#[test]
fn only_q4k_and_q6k_have_a_q8k_kernel() {
    assert!(Q4_K.has_q8k_gemv() && Q6_K.has_q8k_gemv());
    assert!(!Q8_0.has_q8k_gemv());
    let blocks = Q8_0.encode(&weights(), "t").expect("encode");
    assert_eq!(Q8_0.gemv_q8k(&blocks, &activation(), ROWS, IN_DIM), None);
    // Wrong geometry is refused, not read at the wrong stride.
    let q4 = Q4_K.encode(&weights(), "t").expect("encode");
    assert_eq!(Q4_K.gemv_q8k(&q4, &activation(), ROWS + 1, IN_DIM), None);
}

fn planned() -> PlannedOperand {
    let operation = Operation::Project(MatrixClass::FfnProjection);
    PlannedOperand {
        operand: OperandRef {
            object: "target.decoder_stack".into(),
            tensor: "0.mlp.up_proj.weight".into(),
            dtype: String::new(),
            shape: vec![17408, 5120],
        },
        operation,
        access: operation.access(),
        extent: RepresentationExtent::BASE,
        layer: Some(0),
        declared_representation: None,
        logical_elements: 17408 * 5120,
    }
}

/// Only the Q8_K arm selects the new realization, only for members that
/// have the kernel; the direct arm's answer is unchanged by the codec now
/// declaring a second acceleration.
#[test]
fn only_the_q8k_arm_selects_it_and_q8_0_keeps_its_realization() {
    let direct_kquant =
        RealizationId::cpu(RealizationForm::Direct(PhysicalProjectionPlan::FusedKQuant));
    let q8k = RealizationId::cpu(RealizationForm::Direct(
        PhysicalProjectionPlan::FusedKQuantQ8k,
    ));
    for label in ["Q4_K", "Q6_K"] {
        let facts = RepresentationFacts::resolve(label);
        let d = select_cpu(&planned(), &facts, KQuantExecution::Direct).unwrap();
        assert_eq!(d.realization, direct_kquant, "{label} direct");
        let a = select_cpu(&planned(), &facts, KQuantExecution::DirectQ8k).unwrap();
        assert_eq!(a.realization, q8k, "{label} q8k");
        assert_eq!(a.realization.format(), WeightFormat::KQuantQ8k);
        assert!(
            d.candidates.contains(&q8k),
            "{label}: the codec declares it"
        );
    }
    let facts = RepresentationFacts::resolve("Q8_0");
    let a = select_cpu(&planned(), &facts, KQuantExecution::DirectQ8k).unwrap();
    assert_eq!(a.realization, direct_kquant, "Q8_0 has no Q8_K kernel");
}

/// The provider built for the arm answers to its own identity and name,
/// so an image prepared under one can never execute under the other; the
/// shipped provider is untouched.
#[test]
fn the_q8k_provider_has_its_own_identity() {
    let shipped = ProductionBackend::new();
    let q8k = ProductionBackend::q8k_activation();
    assert_ne!(shipped.identity(), q8k.identity());
    assert_ne!(shipped.name(), q8k.name());
    assert_eq!(ProductionBackend::default().identity(), shipped.identity());
    let facts = RepresentationFacts::resolve("Q4_K");
    assert_eq!(
        q8k.select(&planned(), &facts).unwrap().realization.format(),
        WeightFormat::KQuantQ8k
    );
}
