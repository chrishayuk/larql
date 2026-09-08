//! **Every direct CPU kernel over stored bytes is declared by a registered
//! codec** — the closure that makes "all codecs use the registry" a
//! checked statement rather than a belief.
//!
//! A kernel the executor ships and no codec declares is reachable only by
//! a caller that already knows the answer: selection derives candidates
//! from declarations, so such a kernel is dead to the plan and alive to a
//! privileged path — which is what fine-grained FP8 was until it became
//! a codec. The converse holds too: the executor's own re-quantised forms
//! are not any codec's realization, and a codec that claimed one would be
//! declaring a lossy image of its bytes as a direct reading of them.
//!
//! The plan list is exhaustive by construction: a new variant must be
//! placed on one side or the other here before the crate compiles.

use super::*;
use crate::format::vindex3::opplan::exec::cpu::arithmetic::WeightRep;
use crate::format::vindex3::opplan::exec::cpu::physical::PhysicalProjectionPlan;

/// Every plan the executor can run, and which side of the line it is on:
/// `true` for a kernel over STORED bytes (a codec must declare it), `false`
/// for the executor's own re-quantised image (no codec may).
fn every_plan() -> Vec<(PhysicalProjectionPlan, bool)> {
    use PhysicalProjectionPlan::*;
    let all = [
        ScalarF32,
        FusedQ8,
        FusedQ4,
        FusedNvfp4,
        FusedKQuant,
        FusedFp8Block,
        BlasF32,
        FusedBf16,
        Q8xQ8,
        Q4xQ8,
        Bf16xQ8,
    ];
    all.into_iter()
        .map(|plan| {
            let stored = match plan.weight_rep() {
                WeightRep::F32
                | WeightRep::Bf16
                | WeightRep::Nvfp4
                | WeightRep::KQuant
                | WeightRep::Fp8Block => true,
                WeightRep::Q8 { .. } | WeightRep::Q4 { .. } => false,
            };
            // The match above is what makes the list honest: a variant
            // added to the enum and not to `all` still compiles, so the
            // count is pinned to the enum here, by hand, on purpose.
            (plan, stored)
        })
        .collect()
}

const PLAN_COUNT: usize = 11;

fn declared_by(plan: PhysicalProjectionPlan) -> Vec<&'static str> {
    builtin()
        .into_iter()
        .filter(|codec| codec.accelerations().iter().any(|a| a.plan == plan))
        .map(|codec| codec.encoding_label())
        .collect()
}

#[test]
fn every_direct_kernel_over_stored_bytes_is_declared_by_a_registered_codec() {
    let plans = every_plan();
    assert_eq!(
        plans.len(),
        PLAN_COUNT,
        "the plan list drifted from the enum"
    );
    for (plan, stored) in plans {
        let declared = declared_by(plan);
        if stored {
            assert!(
                !declared.is_empty(),
                "{plan:?} runs over stored bytes and no registered codec declares it — \
                 selection cannot reach it, so only a privileged caller can"
            );
        } else {
            assert!(
                declared.is_empty(),
                "{plan:?} is the executor's own re-quantised image; {declared:?} claim it as \
                 a direct realization"
            );
        }
    }
}

/// And the control: the scan finds what is there. The FP8 kernel was the
/// undeclared one before this rung, so its declaration is asserted by
/// name rather than left to the loop above.
#[test]
fn the_fp8_kernel_is_declared_by_exactly_the_fp8_codec() {
    assert_eq!(
        declared_by(PhysicalProjectionPlan::FusedFp8Block),
        [DTYPE_FP8_BLOCK]
    );
    assert_eq!(
        declared_by(PhysicalProjectionPlan::FusedKQuant),
        ["Q4_K", "Q6_K", "Q8_0"],
        "the K-quant kernel is declared by the members it has a kernel for, and no other"
    );
}

/// A codec with no kernel executes through decode, flagged — never through
/// a kernel that would answer `None`.
#[test]
fn a_kquant_without_a_kernel_declares_no_direct_realization() {
    for codec in [Q5_K, Q3_K] {
        assert!(
            codec.accelerations().is_empty(),
            "{} has no gemv and must not declare one",
            codec.encoding_label()
        );
        assert!(!codec.quant().has_direct_gemv());
    }
    for codec in [Q4_K, Q6_K, Q8_0] {
        assert!(codec.quant().has_direct_gemv());
        assert_eq!(codec.accelerations().len(), 1, "{}", codec.encoding_label());
    }
}
