//! Tests for [`super::super::arithmetic`] — the rendering of an
//! arithmetic, and the constraint residency puts on it.
//!
//! `Display` is not cosmetic here. `Q4[64] x Q8[16] -> I32 -> F32` is
//! the form every CPU claim in `docs/cpu-execution-roadmap.md` is quoted
//! in, and each term changes the answer: CPU-5 turned on the block size
//! alone (Q8[64] vs Q8[16] activation), and a renderer that dropped the
//! block would make the candidate and its control read identically in a
//! bench table.

use super::super::arithmetic::{
    plans_possible_for, AccumulatorRep, ActivationRep, Arithmetic, ScaleSpan, WeightRep,
};
use super::super::physical::PhysicalProjectionPlan;

/// The block is the term CPU-5 turned on, so it has to survive
/// rendering. `Q8` and `Q8[16]` naming the same arithmetic in a table is
/// how a candidate gets mistaken for its own control.
#[test]
fn a_weight_rendering_carries_its_block() {
    assert_eq!(WeightRep::F32.to_string(), "F32");
    assert_eq!(WeightRep::Bf16.to_string(), "BF16");
    assert_eq!(WeightRep::Q8 { block: 64 }.to_string(), "Q8[64]");
    assert_eq!(WeightRep::Q4 { block: 32 }.to_string(), "Q4[32]");
    assert_eq!(WeightRep::Nvfp4.to_string(), "NVFP4");
    assert_eq!(WeightRep::KQuant.to_string(), "KQUANT");
    assert_ne!(
        WeightRep::Q8 { block: 64 }.to_string(),
        WeightRep::Q8 { block: 16 }.to_string(),
        "two block sizes must not render alike"
    );
}

/// Tensor and block spans are the distinction CPU-5 exists to make — a
/// per-tensor activation scale leaves roughly two effective bits on a
/// heavy-tailed residual stream — so they must never render the same.
#[test]
fn an_activation_rendering_distinguishes_tensor_from_block() {
    assert_eq!(ActivationRep::F32.to_string(), "F32");
    assert_eq!(
        ActivationRep::Q8 {
            span: ScaleSpan::Tensor
        }
        .to_string(),
        "Q8[tensor]"
    );
    assert_eq!(
        ActivationRep::Q8 {
            span: ScaleSpan::Block(16)
        }
        .to_string(),
        "Q8[16]"
    );
    assert_ne!(
        ActivationRep::Q8 {
            span: ScaleSpan::Tensor
        }
        .to_string(),
        ActivationRep::Q8 {
            span: ScaleSpan::Block(16)
        }
        .to_string(),
        "a per-tensor scale and a per-block one are different arithmetic"
    );
}

#[test]
fn an_accumulator_renders_its_width() {
    assert_eq!(AccumulatorRep::F32.to_string(), "F32");
    assert_eq!(AccumulatorRep::I32.to_string(), "I32");
}

/// The whole triple, in the form a claim quotes.
#[test]
fn the_full_arithmetic_renders_every_term() {
    let a = Arithmetic {
        weight: WeightRep::Q4 { block: 64 },
        activation: ActivationRep::Q8 {
            span: ScaleSpan::Block(16),
        },
        accumulator: AccumulatorRep::I32,
    };
    assert_eq!(a.to_string(), "Q4[64] x Q8[16] -> I32 -> F32");
}

/// F32 residency admits exactly one plan, and it is the BLAS one — the
/// scalar oracle is reachable only by a backend that declares it. A
/// reference path selectable off the resident bytes would stop being a
/// reference, which is the invariant this pins.
#[test]
fn f32_residency_admits_only_the_blas_plan() {
    let plans = plans_possible_for(WeightRep::F32);
    assert_eq!(plans, &[PhysicalProjectionPlan::BlasF32]);
    assert!(
        !plans.contains(&PhysicalProjectionPlan::ScalarF32),
        "the scalar oracle must not be reachable by observing residency"
    );
}

/// Every representation admits at least one plan, and no representation
/// admits the oracle. An empty list would mean resident bytes no kernel
/// can consume.
#[test]
fn every_representation_admits_a_runnable_plan_and_never_the_oracle() {
    for rep in [
        WeightRep::F32,
        WeightRep::Bf16,
        WeightRep::Q8 { block: 64 },
        WeightRep::Q4 { block: 64 },
        WeightRep::Nvfp4,
        WeightRep::KQuant,
    ] {
        let plans = plans_possible_for(rep);
        assert!(!plans.is_empty(), "{rep} admits no plan at all");
        assert!(
            !plans.contains(&PhysicalProjectionPlan::ScalarF32),
            "{rep} must not make the oracle selectable"
        );
    }
}
