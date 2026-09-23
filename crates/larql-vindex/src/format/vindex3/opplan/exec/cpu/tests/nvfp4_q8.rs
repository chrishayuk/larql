//! NVFP4-Q8-1's witnesses (`docs/nvfp4-q8-1.md`): the weight side is
//! exact, the kernel computes what its definition says, the activation's
//! quantisation stays inside its block bound, and the arm is its own plan
//! under its own provider identity.

use larql_models::quant::nvfp4::{
    dequantize_into, Nvfp4Matrix, NVFP4_GROUP_BYTES, NVFP4_GROUP_ELEMS,
};

use super::super::arithmetic::{
    plans_possible_for, AccumulatorRep, ActivationRep, ScaleSpan, WeightRep,
};
use super::super::integer::quantise_activation_blocked;
use super::super::kernels::{e4m3_steps, FusedNvfp4};
use super::super::nvfp4_q8::{
    nvfp4_group_codes, nvfp4_q8_row_portable, FusedNvfp4Q8, NVFP4_Q8_ACTIVATION_BLOCK,
};
use super::super::physical::{KQuantExecution, PhysicalProjectionPlan};
use super::super::projector::{DenseProjector, WeightRows};
use crate::format::vindex3::opplan::exec::backend::{
    MatrixClass, Nvfp4Activation, PlanBackend, WeightFormat,
};
use crate::format::vindex3::opplan::exec::production::{select_cpu_with, ProductionBackend};
use crate::format::vindex3::opplan::exec::realization::{
    RealizationForm, RealizationId, RepresentationFacts,
};
use crate::format::vindex3::opplan::planned::{Operation, PlannedOperand};
use crate::format::vindex3::opplan::OperandRef;
use crate::format::vindex3::represent::codec::RepresentationExtent;

/// Shapes from one group to a real Gemma 3 4B width, with odd row counts
/// so a slab cut at the wrong row shows.
const SHAPES: [(usize, usize); 6] = [(1, 16), (3, 32), (5, 48), (7, 112), (4, 2560), (3, 8192)];
const TENSOR_SCALE: f32 = 0.0371;
/// The seed the quantisation bound is recorded on.
const BOUND_SEED: u32 = 0x5eed_0001;

/// A deterministic byte stream that visits every value; E4M3's NaN scales
/// are skipped, since a NaN row says nothing about agreement.
fn bytes(n: usize, seed: u32, skip_nan_scales: bool) -> Vec<u8> {
    let mut state = seed.wrapping_mul(2_654_435_761).wrapping_add(1);
    (0..n)
        .map(|_| {
            state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
            let b = (state >> 24) as u8;
            if skip_nan_scales && b & 0x7f == 0x7f {
                0x38
            } else {
                b
            }
        })
        .collect()
}

/// An activation with an outlier channel every 97 elements — the regime
/// real residuals are in, where a per-group scale matters.
fn activations(k: usize) -> Vec<f32> {
    (0..k)
        .map(|i| {
            if i % 97 == 0 {
                7.5
            } else {
                ((i * 37 % 101) as f32 - 50.0) / 13.0
            }
        })
        .collect()
}

struct Case {
    n: usize,
    k: usize,
    packed: Vec<u8>,
    scales: Vec<u8>,
    x: Vec<f32>,
}

fn case(n: usize, k: usize, seed: u32) -> Case {
    let groups = k / NVFP4_GROUP_ELEMS;
    Case {
        n,
        k,
        packed: bytes(n * groups * NVFP4_GROUP_BYTES, seed ^ (n * k) as u32, false),
        scales: bytes(n * groups, seed ^ (n + k) as u32, true),
        x: activations(k),
    }
}

fn rows(c: &Case, activation: Nvfp4Activation) -> WeightRows<'_> {
    WeightRows::Nvfp4 {
        packed: &c.packed,
        scales: &c.scales,
        tensor_scale: TENSOR_SCALE,
        activation,
    }
}

fn reference_weights(c: &Case) -> Vec<f32> {
    let matrix = Nvfp4Matrix {
        packed: c.packed.clone(),
        scales: c.scales.clone(),
        tensor_scale: TENSOR_SCALE,
    };
    let mut w = vec![0.0f32; c.n * c.k];
    dequantize_into(&matrix, c.n, c.k, &mut w).expect("reference geometry");
    w
}

/// The activation the kernel sees, widened back: code × its group scale.
fn dequantised_activation(x: &[f32]) -> Vec<f32> {
    let (qx, xs) = quantise_activation_blocked(x, NVFP4_Q8_ACTIVATION_BLOCK);
    qx.iter()
        .enumerate()
        .map(|(i, &q)| q as f32 * xs[i / NVFP4_Q8_ACTIVATION_BLOCK])
        .collect()
}

/// **Witness 1 — the weight side is exact.** The int8 table decode times
/// the group's halved scale is the reference decoder's value, bit for bit,
/// for every element of every shape.
#[test]
fn the_int8_decode_times_scale_is_the_reference_decode_exactly() {
    let steps = e4m3_steps();
    for (n, k) in SHAPES {
        let c = case(n, k, 1);
        let reference = reference_weights(&c);
        let groups = k / NVFP4_GROUP_ELEMS;
        for row in 0..n {
            for g in 0..groups {
                let at = row * groups + g;
                let codes = nvfp4_group_codes(&c.packed[at * NVFP4_GROUP_BYTES..]);
                let half_step = 0.5 * TENSOR_SCALE * steps[c.scales[at] as usize];
                for (e, &code) in codes.iter().enumerate() {
                    let want = reference[row * k + g * NVFP4_GROUP_ELEMS + e];
                    let got = code as f32 * half_step;
                    // Exact by value. The one bit pattern that can differ is
                    // E2M1's -0 (code 8): an int8 has no negative zero, and
                    // no dot product can observe the sign of a zero term.
                    assert!(
                        got == want,
                        "[{n},{k}] row {row} group {g} element {e}: {got} vs {want}"
                    );
                    if want != 0.0 {
                        assert_eq!(got.to_bits(), want.to_bits(), "[{n},{k}] {row}/{g}/{e}");
                    }
                }
            }
        }
    }
}

/// Σ|w·x| over one row: the scale a reassociation error is judged against,
/// so cancellation cannot make the tolerance vanish.
fn magnitude(w: &[f32], x: &[f32]) -> f32 {
    w.iter().zip(x).map(|(a, b)| (a * b).abs()).sum()
}

/// **Witness 2 — kernel parity.** The kernel and its portable definition
/// agree with the f32 dot of the reference-decoded weights against the
/// DEQUANTISED activation, to f32 reassociation. That isolates the kernel
/// from the quantisation: any error here is the kernel's.
#[test]
fn the_kernel_computes_its_definition_against_the_quantised_activation() {
    for (n, k) in SHAPES {
        let c = case(n, k, 2);
        let w = reference_weights(&c);
        let xq = dequantised_activation(&c.x);
        let (qx, xs) = quantise_activation_blocked(&c.x, NVFP4_Q8_ACTIVATION_BLOCK);
        let mut got = vec![0.0f32; n];
        FusedNvfp4Q8.project_rows(rows(&c, Nvfp4Activation::Q8), &c.x, &mut got);
        let groups = k / NVFP4_GROUP_ELEMS;
        for row in 0..n {
            let wr = &w[row * k..(row + 1) * k];
            let want: f32 = wr.iter().zip(&xq).map(|(a, b)| a * b).sum();
            let portable = nvfp4_q8_row_portable(
                &c.packed[row * groups * NVFP4_GROUP_BYTES..][..groups * NVFP4_GROUP_BYTES],
                &c.scales[row * groups..][..groups],
                TENSOR_SCALE,
                e4m3_steps(),
                &qx,
                &xs,
            );
            let tol = magnitude(wr, &xq) * 1e-6 + 1e-6;
            for (what, value) in [("kernel", got[row]), ("portable", portable)] {
                assert!(
                    (value - want).abs() <= tol,
                    "[{n},{k}] row {row}: {what} {value} vs {want} (tol {tol})"
                );
            }
        }
    }
}

/// **Witness 3 — the quantisation bound.** Against `FusedNvfp4` on the
/// same bytes with the f32 activation, each row's difference is within
/// the Q8 rounding bound Σ_g Σ_i |w_i| · xs_g / 2 (plus reassociation), and
/// it is NOT zero: the arm did quantise.
#[test]
fn the_q8_activation_stays_inside_its_block_bound() {
    let mut worst_relative = 0.0f32;
    let mut any_difference = false;
    for (n, k) in SHAPES {
        let c = case(n, k, BOUND_SEED);
        let w = reference_weights(&c);
        let (_, xs) = quantise_activation_blocked(&c.x, NVFP4_Q8_ACTIVATION_BLOCK);
        let mut q8 = vec![0.0f32; n];
        FusedNvfp4Q8.project_rows(rows(&c, Nvfp4Activation::Q8), &c.x, &mut q8);
        let mut f32_arm = vec![0.0f32; n];
        FusedNvfp4.project_rows(rows(&c, Nvfp4Activation::F32), &c.x, &mut f32_arm);
        for row in 0..n {
            let wr = &w[row * k..(row + 1) * k];
            let bound: f32 = wr
                .iter()
                .enumerate()
                .map(|(i, a)| a.abs() * xs[i / NVFP4_Q8_ACTIVATION_BLOCK] * 0.5)
                .sum();
            let slack = magnitude(wr, &c.x) * 2e-6 + 1e-6;
            let delta = (q8[row] - f32_arm[row]).abs();
            assert!(
                delta <= bound + slack,
                "[{n},{k}] row {row}: |Δ| {delta} exceeds the Q8 bound {bound}"
            );
            any_difference |= delta > 0.0;
            let scale = magnitude(wr, &c.x);
            if scale > 0.0 {
                worst_relative = worst_relative.max(delta / scale);
            }
        }
    }
    assert!(any_difference, "the Q8 arm computed the f32 arm's numbers");
    // Recorded, not judged: the bound above is the judgement.
    eprintln!("NVFP4-Q8-1 seed {BOUND_SEED:#x}: worst |Δ| / Σ|w·x| = {worst_relative:e}");
}

fn planned() -> PlannedOperand {
    let operation = Operation::Project(MatrixClass::FfnProjection);
    PlannedOperand {
        operand: OperandRef {
            object: "target.decoder_stack".into(),
            tensor: "0.mlp.up_proj.weight".into(),
            dtype: String::new(),
            shape: vec![10240, 2560],
        },
        operation,
        access: operation.access(),
        extent: RepresentationExtent::BASE,
        layer: Some(0),
        declared_representation: None,
        logical_elements: 10240 * 2560,
    }
}

/// **Witness 4a — plan enumeration.** The binding decides the plan, the
/// plan reports the arithmetic it runs, and every enumeration lists it.
#[test]
fn the_q8_binding_is_its_own_plan_and_is_enumerated() {
    let c = case(3, 32, 4);
    let q8 = PhysicalProjectionPlan::for_resident(rows(&c, Nvfp4Activation::Q8), c.k);
    let f32_plan = PhysicalProjectionPlan::for_resident(rows(&c, Nvfp4Activation::F32), c.k);
    assert_eq!(q8, PhysicalProjectionPlan::FusedNvfp4Q8);
    assert_eq!(f32_plan, PhysicalProjectionPlan::FusedNvfp4);
    assert_eq!(q8.format(), WeightFormat::Nvfp4Q8);
    let a = q8.arithmetic();
    assert_eq!(a.weight, WeightRep::Nvfp4);
    assert_eq!(
        a.activation,
        ActivationRep::Q8 {
            span: ScaleSpan::Block(NVFP4_Q8_ACTIVATION_BLOCK)
        }
    );
    assert_eq!(a.accumulator, AccumulatorRep::I32);
    assert_eq!(
        NVFP4_Q8_ACTIVATION_BLOCK % NVFP4_GROUP_ELEMS,
        0,
        "an activation block is a whole number of NVFP4 groups"
    );
    assert!(plans_possible_for(WeightRep::Nvfp4).contains(&q8));
}

/// **Witness 4b — selection and identity.** Only the provider built for
/// the arm selects the Q8 realization; the shipped provider's answer is
/// unchanged; and the two answer to different identities, so an image
/// prepared under one never executes under the other.
#[test]
fn only_the_nvfp4_q8_provider_selects_it_under_its_own_identity() {
    let facts = RepresentationFacts::resolve("NVFP4");
    let f32_id = RealizationId::cpu(RealizationForm::Direct(PhysicalProjectionPlan::FusedNvfp4));
    let q8_id = RealizationId::cpu(RealizationForm::Direct(
        PhysicalProjectionPlan::FusedNvfp4Q8,
    ));
    let shipped = select_cpu_with(
        &planned(),
        &facts,
        KQuantExecution::Direct,
        Nvfp4Activation::F32,
    )
    .unwrap();
    assert_eq!(shipped.realization, f32_id);
    assert!(shipped.candidates.contains(&q8_id), "the codec declares it");

    let arm = ProductionBackend::nvfp4_q8_activation();
    let chosen = arm.select(&planned(), &facts).unwrap();
    assert_eq!(chosen.realization, q8_id);
    assert_eq!(chosen.realization.format(), WeightFormat::Nvfp4Q8);
    assert_eq!(
        ProductionBackend::new()
            .select(&planned(), &facts)
            .unwrap()
            .realization,
        f32_id
    );

    let others = [
        ProductionBackend::new(),
        ProductionBackend::q8k_activation(),
    ];
    for other in others {
        assert_ne!(arm.identity(), other.identity());
        assert_ne!(arm.name(), other.name());
    }
}
