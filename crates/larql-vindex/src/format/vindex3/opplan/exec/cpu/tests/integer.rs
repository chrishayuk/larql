//! CPU-5 instrument controls: the integer-domain kernels, before any arm
//! of the quality gate is allowed to mean anything.
//!
//! Nothing here is a quality result. These are the gates the
//! pre-registration (`bench/prompts/quality-bank-1/CPU5-Q4Q8-QUALITY.md`)
//! requires to pass FIRST, because an arm run through an unproven kernel
//! measures the kernel and not the format.
//!
//! Each kernel is judged against the format's own portable DEFINITION and
//! never against the original f32 weights. At 4.5 bits the quantiser's
//! error is large enough to hide almost any kernel bug inside a tolerance
//! chosen for it.

use super::super::integer::{quantise_activation, Bf16xQ8, Q4xQ8, Q8xQ8};
use super::super::kernels::FusedBf16;
use super::super::physical::{arithmetic_arm, ArithmeticArm};
use super::super::projector::{DenseProjector, WeightRows};
use crate::format::vindex3::fixtures::lcg_values;
use crate::format::vindex3::opplan::exec::quantise::{
    quantise_q4_for_test, quantise_q8_for_test, Q4_BLOCK, Q8_BLOCK,
};
use crate::format::vindex3::opplan::exec::weights::LoadedWeight;

/// Shapes that exercise a single block, several blocks, and the real
/// model's input width.
const SHAPES: [usize; 3] = [Q8_BLOCK, Q8_BLOCK * 3, 5120];

fn q8_parts(w: &[f32], in_dim: usize) -> (Vec<i8>, Vec<f32>) {
    match quantise_q8_for_test(w, in_dim) {
        LoadedWeight::Q8 { codes, scales, .. } => (codes, scales),
        _ => panic!("the q8 quantiser must produce the q8 variant"),
    }
}

fn q4_parts(w: &[f32], in_dim: usize) -> (Vec<u8>, Vec<f32>) {
    match quantise_q4_for_test(w, in_dim) {
        LoadedWeight::Q4 { packed, scales } => (packed, scales),
        _ => panic!("the q4 quantiser must produce the q4 variant"),
    }
}

/// **The Q8 x Q8 kernel computes what the format denotes.**
///
/// Bit-identity, not a tolerance. The block sums are INTEGER and integer
/// addition is exact and associative, so the vectorised lane-reduction
/// and the sequential loop must reach the same i32; the only float in
/// either is one multiply-add per block, in the same block order. A
/// difference here would be a real defect, never rounding.
#[test]
fn the_q8_q8_kernel_computes_what_the_format_denotes() {
    const OUT: usize = 5;
    for in_dim in SHAPES {
        let w = lcg_values(OUT * in_dim, 21);
        let x = lcg_values(in_dim, 22);
        let (codes, scales) = q8_parts(&w, in_dim);
        let act = quantise_activation(&x);
        let per_row = in_dim.div_ceil(Q8_BLOCK);

        let mut got = vec![0.0f32; OUT];
        Q8xQ8.project_rows(
            WeightRows::Q8 {
                codes: &codes,
                scales: &scales,
                sums: &[],
                block: Q8_BLOCK,
            },
            &x,
            &mut got,
        );

        for o in 0..OUT {
            let want = act.scale
                * super::super::integer::q8_row_portable(
                    &codes[o * in_dim..(o + 1) * in_dim],
                    &scales[o * per_row..(o + 1) * per_row],
                    &act.codes,
                    in_dim,
                    Q8_BLOCK,
                );
            assert_eq!(
                got[o].to_bits(),
                want.to_bits(),
                "q8xq8 row {o} at in_dim {in_dim}: {} vs definition {}",
                got[o],
                want
            );
        }
    }
}

/// **The Q4 x Q8 kernel computes what the format denotes.**
///
/// Same bit-identity argument, plus the packing: byte `j` carries element
/// `j` low and `j + half` high, so a kernel that read adjacent nibbles
/// would pair every weight with the wrong activation and still return
/// finite, plausible numbers.
#[test]
fn the_q4_q8_kernel_computes_what_the_format_denotes() {
    const OUT: usize = 5;
    for in_dim in SHAPES {
        let w = lcg_values(OUT * in_dim, 23);
        let x = lcg_values(in_dim, 24);
        let (packed, scales) = q4_parts(&w, in_dim);
        let act = quantise_activation(&x);
        let per_row = in_dim.div_ceil(Q4_BLOCK);

        let mut got = vec![0.0f32; OUT];
        Q4xQ8.project_rows(
            WeightRows::Q4 {
                packed: &packed,
                scales: &scales,
                block: Q4_BLOCK,
            },
            &x,
            &mut got,
        );

        for o in 0..OUT {
            let want = act.scale
                * super::super::integer::q4_row_portable(
                    &packed[o * (in_dim / 2)..(o + 1) * (in_dim / 2)],
                    &scales[o * per_row..(o + 1) * per_row],
                    &act.codes,
                    in_dim,
                    Q4_BLOCK,
                );
            assert_eq!(
                got[o].to_bits(),
                want.to_bits(),
                "q4xq8 row {o} at in_dim {in_dim}: {} vs definition {}",
                got[o],
                want
            );
        }
    }
}

/// **The control arm changes the ACTIVATION and nothing else.**
///
/// If the activation already lies on the int8 grid its round trip is
/// lossless, so `Bf16xQ8` must reproduce the EXACT kernel bit for bit —
/// same weights, same summation order, same everything. That is what
/// makes A1 a control: every difference it reports on a real activation
/// is the activation's rounding and nothing else.
#[test]
fn the_activation_control_changes_the_activation_and_nothing_else() {
    const OUT: usize = 4;
    const IN: usize = 256;
    let w = lcg_values(OUT * IN, 25);
    let bits: Vec<u16> = w.iter().map(|v| (v.to_bits() >> 16) as u16).collect();

    // An activation already on the grid: 127 levels of a chosen peak, so
    // `round(x / (peak/127))` is exact for every element.
    let peak = 0.25f32;
    let step = peak / 127.0;
    let x: Vec<f32> = (0..IN)
        .map(|i| ((i as i32 % 255) - 127) as f32 * step)
        .collect();
    assert!(
        x.iter().any(|v| *v < 0.0) && x.iter().any(|v| *v > 0.0),
        "the fixture must span both signs or it cannot exercise the clamp"
    );

    let mut exact = vec![0.0f32; OUT];
    FusedBf16.project_rows(WeightRows::Bf16(&bits), &x, &mut exact);
    let mut control = vec![0.0f32; OUT];
    Bf16xQ8.project_rows(WeightRows::Bf16(&bits), &x, &mut control);

    for o in 0..OUT {
        assert_eq!(
            control[o].to_bits(),
            exact[o].to_bits(),
            "A1 row {o}: {} vs the exact kernel {}",
            control[o],
            exact[o]
        );
    }
}

/// And the control is NOT vacuous: on an ordinary activation it moves.
///
/// A control that reproduced the exact answer whatever it was handed
/// would pass the test above by doing nothing at all.
#[test]
fn the_activation_control_does_move_on_an_ordinary_activation() {
    const OUT: usize = 8;
    const IN: usize = 512;
    let w = lcg_values(OUT * IN, 29);
    let bits: Vec<u16> = w.iter().map(|v| (v.to_bits() >> 16) as u16).collect();
    let x = lcg_values(IN, 30);

    let mut exact = vec![0.0f32; OUT];
    FusedBf16.project_rows(WeightRows::Bf16(&bits), &x, &mut exact);
    let mut control = vec![0.0f32; OUT];
    Bf16xQ8.project_rows(WeightRows::Bf16(&bits), &x, &mut control);

    assert!(
        exact.iter().zip(&control).any(|(a, b)| a != b),
        "the A1 control reproduced the exact kernel on a real activation, so it is quantising \
         nothing and would report a clean bill for any weight format"
    );
}

/// **The instrument must FAIL on known-different input.**
///
/// A gate that only ever passes proves nothing. Q4 and Q8 over the SAME
/// weights and the SAME activation must disagree, and by roughly the
/// ratio of their quantisation steps — `peak/7` against `peak/127`. If
/// these two arms agreed, the arm switch would not be reaching the
/// arithmetic at all.
#[test]
fn the_q4_and_q8_arms_disagree_by_about_their_step_ratio() {
    const OUT: usize = 64;
    const IN: usize = 1024;
    let w = lcg_values(OUT * IN, 26);
    let x = lcg_values(IN, 27);
    let (codes, q8_scales) = q8_parts(&w, IN);
    let (packed, q4_scales) = q4_parts(&w, IN);

    let mut q8 = vec![0.0f32; OUT];
    Q8xQ8.project_rows(
        WeightRows::Q8 {
            codes: &codes,
            scales: &q8_scales,
            sums: &[],
            block: Q8_BLOCK,
        },
        &x,
        &mut q8,
    );
    let mut q4 = vec![0.0f32; OUT];
    Q4xQ8.project_rows(
        WeightRows::Q4 {
            packed: &packed,
            scales: &q4_scales,
            block: Q4_BLOCK,
        },
        &x,
        &mut q4,
    );

    let rms = |v: &[f32]| (v.iter().map(|a| (a * a) as f64).sum::<f64>() / v.len() as f64).sqrt();
    let diff: Vec<f32> = q4.iter().zip(&q8).map(|(a, b)| a - b).collect();
    let rel = rms(&diff) / rms(&q8);

    assert!(
        rel > 1e-2,
        "q4 and q8 arms agree to {rel:.3e} — the arm switch is not reaching the arithmetic"
    );
    assert!(
        rel < 1.0,
        "q4 differs from q8 by {rel:.3e}, which is not quantisation but a defect"
    );
}

/// The activation quantiser is bounded by half its own step, and its
/// scale is derived from the peak it must represent.
#[test]
fn the_activation_quantiser_is_bounded_by_half_a_step() {
    let x = lcg_values(4096, 28);
    let act = quantise_activation(&x);
    let peak = x.iter().fold(0.0f32, |m, v| m.max(v.abs()));

    assert!(
        (act.scale - peak / 127.0).abs() <= f32::EPSILON * peak.max(1.0),
        "the activation scale must be peak/127"
    );
    for (i, v) in x.iter().enumerate() {
        let back = act.codes[i] as f32 * act.scale;
        assert!(
            (back - v).abs() <= act.scale * 0.5 + f32::EPSILON,
            "element {i} reconstructs to {back} from {v}, past half a step"
        );
    }
}

/// A zero activation must not divide by zero, and must reconstruct
/// exactly.
#[test]
fn a_zero_activation_quantises_without_dividing_by_zero() {
    let act = quantise_activation(&vec![0.0f32; 128]);
    assert_eq!(act.scale, 1.0, "the zero vector takes the sentinel scale");
    assert!(act.codes.iter().all(|c| *c == 0));
}

/// The arm is the DEFAULT unless the environment names one, and an
/// unrecognised value is the default rather than a fourth regime.
///
/// Read through the same accessor the loader and the executor use, so
/// this pins the value both of them see. It does not set the variable:
/// the arm is resolved once per process on purpose, and a test that
/// mutated it would be asserting about its own ordering.
#[test]
fn the_default_arithmetic_arm_is_the_float_activation() {
    if std::env::var(super::super::physical::ARITHMETIC_ARM_ENV).is_ok() {
        // A deliberately-armed process is running some other measurement;
        // asserting the default here would fail for the right reason and
        // tell nobody anything.
        return;
    }
    assert_eq!(arithmetic_arm(), ArithmeticArm::FloatActivation);
}

/// **The A1 control must cover EXACTLY the operands the arms cover.**
///
/// bf16 bytes are ambiguous: an operand is resident as bf16 either
/// because its image fits L2 and the policy kept it exact, or because
/// the A1 control swapped a streaming Q8 operand back. Observation alone
/// cannot separate those, and an earlier version of this arm did not
/// try — so A1 quantised the activation on the cache-resident operands
/// too, became a LARGER perturbation than the arm it exists to explain,
/// and read worse than Q8 x Q8 while holding exact weights.
///
/// This pins the population rather than the numbers: a cache-resident
/// bf16 operand stays exact under every arm, and only a streaming one
/// joins the control.
#[test]
fn the_activation_control_covers_only_the_streaming_operands() {
    use super::super::physical::{compact_threshold_bytes, PhysicalProjectionPlan, BF16_BYTES};

    // Sized either side of the boundary the policy itself reads.
    let boundary = compact_threshold_bytes() / BF16_BYTES;
    let in_dim = 64usize;
    for (elements, streaming) in [(in_dim * 2, false), (boundary + in_dim, true)] {
        let bits = vec![0u16; elements];
        let observed = PhysicalProjectionPlan::for_resident(WeightRows::Bf16(&bits), in_dim);
        // Under the DEFAULT arm nothing joins the control at any size.
        assert_eq!(
            observed,
            PhysicalProjectionPlan::FusedBf16,
            "a bf16 operand of {elements} elements (streaming={streaming}) must stay exact \
             under the default arm"
        );
    }
}

/// **Exact weights must beat quantised weights under the same activation.**
///
/// A1 holds the checkpoint's own values and A3 holds an 8-bit image of
/// them; the activation error is identical. Their output errors are
/// therefore `e_act` and `e_act + dw.x` with the two terms independent,
/// so A1 can never be the worse arm. If it is, the arms are not seeing
/// the same activation and the control is not a control.
#[test]
fn the_exact_weight_arm_beats_the_quantised_one_under_one_activation() {
    const OUT: usize = 32;
    const IN: usize = 1024;
    let w = lcg_values(OUT * IN, 41);
    // A residual-stream-shaped activation: a few outlier channels tens of
    // times the RMS, which is the regime the real model is in.
    let mut x = lcg_values(IN, 42);
    for (i, v) in x.iter_mut().enumerate() {
        if i % 137 == 0 {
            *v *= 40.0;
        }
    }
    let bits: Vec<u16> = w.iter().map(|v| (v.to_bits() >> 16) as u16).collect();
    let exact_w: Vec<f32> = bits
        .iter()
        .map(|b| f32::from_bits((*b as u32) << 16))
        .collect();
    let (codes, scales) = q8_parts(&exact_w, IN);

    // The truth: exact weights, exact activation.
    let truth: Vec<f32> = (0..OUT)
        .map(|o| {
            exact_w[o * IN..(o + 1) * IN]
                .iter()
                .zip(&x)
                .map(|(a, b)| a * b)
                .sum::<f32>()
        })
        .collect();

    let mut a1 = vec![0.0f32; OUT];
    Bf16xQ8.project_rows(WeightRows::Bf16(&bits), &x, &mut a1);
    let mut a3 = vec![0.0f32; OUT];
    Q8xQ8.project_rows(
        WeightRows::Q8 {
            codes: &codes,
            scales: &scales,
            sums: &[],
            block: Q8_BLOCK,
        },
        &x,
        &mut a3,
    );

    let err = |v: &[f32]| {
        let n: f64 = v
            .iter()
            .zip(&truth)
            .map(|(a, t)| ((a - t) as f64).powi(2))
            .sum();
        let d: f64 = truth.iter().map(|t| (*t as f64).powi(2)).sum();
        (n / d).sqrt()
    };
    let (e1, e3) = (err(&a1), err(&a3));
    assert!(
        e1 <= e3,
        "A1 (exact weights) {e1:.4e} is WORSE than A3 (q8 weights) {e3:.4e} under one \
         activation — the two arms are not quantising the activation the same way"
    );
}

/// **Residency CONSTRAINS which plans are possible; it does not DETERMINE
/// which one executes.**
///
/// The invariant this whole rung produced. Before integer activations it
/// was false in a useful way — bytes implied a kernel — and every reader
/// could rely on that. It is false now: identical Q8 bytes are consumed
/// by a widening f32 GEMV and by `SDOT`, at 83.4 and 118.0 GB/s and with
/// different numerics.
///
/// Pinned as a test because the tempting "simplification" is to infer
/// arithmetic from residency again, and it would pass every parity gate
/// on a machine running the default arm.
#[test]
fn residency_constrains_the_plan_without_determining_it() {
    use super::super::arithmetic::{plans_possible_for, WeightRep};
    use super::super::physical::PhysicalProjectionPlan;

    // Every compact representation admits MORE THAN ONE plan. If any of
    // these collapsed to one, residency would determine execution again.
    for rep in [
        WeightRep::Bf16,
        WeightRep::Q8 { block: Q8_BLOCK },
        WeightRep::Q4 { block: Q4_BLOCK },
    ] {
        assert!(
            plans_possible_for(rep).len() > 1,
            "{rep:?} admits only one plan — residency would determine arithmetic"
        );
    }

    // And whatever the arm, the plan actually chosen is one of the plans
    // that representation makes possible: the constraint is real.
    let bits = vec![0u16; 64 * 4];
    let observed = PhysicalProjectionPlan::for_resident(WeightRows::Bf16(&bits), 64);
    assert!(
        plans_possible_for(WeightRep::Bf16).contains(&observed),
        "the executor chose {observed:?}, which bf16 residency does not admit"
    );
}

/// The plan reports the arithmetic in the form a claim has to quote.
///
/// Every term changes the answer, so a number reported as "Q4" alone is
/// under-described: `Q4[64] x Q8[tensor]` and `Q4[64] x Q8[64]` differ by
/// an order of magnitude in logit error.
#[test]
fn a_plan_describes_its_arithmetic_including_the_activation_geometry() {
    use super::super::arithmetic::{AccumulatorRep, ActivationRep, ScaleSpan};
    use super::super::physical::PhysicalProjectionPlan;

    let a = PhysicalProjectionPlan::Q4xQ8.arithmetic();
    assert_eq!(a.accumulator, AccumulatorRep::I32);
    assert!(matches!(a.activation, ActivationRep::Q8 { .. }));

    let shown = format!("{a}");
    assert!(
        shown.starts_with("Q4[64] x Q8[") && shown.ends_with("-> I32 -> F32"),
        "unexpected description `{shown}`"
    );

    // The exact kernels name an f32 activation and an f32 accumulator,
    // so nothing reads as integer arithmetic that is not.
    let e = PhysicalProjectionPlan::FusedBf16.arithmetic();
    assert_eq!(e.activation, ActivationRep::F32);
    assert_eq!(e.accumulator, AccumulatorRep::F32);
    assert_eq!(format!("{e}"), "BF16 x F32 -> F32 -> F32");

    // And the span is carried, not implied.
    match PhysicalProjectionPlan::Q8xQ8.arithmetic().activation {
        ActivationRep::Q8 { span } => {
            assert!(matches!(span, ScaleSpan::Tensor | ScaleSpan::Block(_)))
        }
        other => panic!("integer arm reported activation {other:?}"),
    }
}

/// **A restored class falls back to Q8 in the SAME arithmetic domain.**
///
/// A rescue rung must move exactly one variable: the weight bits. If
/// restoring a class also dropped it back to an f32 activation, the rung
/// would move the weight format AND the arithmetic together, and its
/// result would license nothing about either — which is precisely how
/// CPU-4A concluded that Q4 was dead when it had only shown that
/// Q4 x F32 was.
#[test]
fn a_restored_class_keeps_the_integer_activation() {
    use super::super::physical::{PhysicalProjectionPlan, Q4Classes};
    use crate::format::vindex3::opplan::exec::backend::MatrixClass;

    // FFN goes to Q4; attention and the head are restored.
    let only_ffn = Q4Classes {
        attention: false,
        ffn: true,
        head: false,
    };
    assert!(only_ffn.admits(MatrixClass::FfnProjection));
    assert!(!only_ffn.admits(MatrixClass::AttentionProjection));
    assert!(!only_ffn.admits(MatrixClass::OutputHead));

    // The bank is never a Q4 candidate under any set: it is widened to
    // f32 on the way in and has no compact bytes to keep.
    assert!(!Q4Classes::ALL.admits(MatrixClass::RoutedExpertBank));

    // And Q8 bytes under a Q4 arm run through SDOT, not through the
    // widening f32 kernel — same activation, same accumulator.
    let codes = vec![0i8; 64 * 4];
    let scales = vec![1.0f32; 4];
    let observed = PhysicalProjectionPlan::for_resident(
        WeightRows::Q8 {
            codes: &codes,
            scales: &scales,
            sums: &[],
            block: Q8_BLOCK,
        },
        64,
    );
    // Under the default arm this is the widening kernel; under either
    // integer arm it is SDOT. Both are legal; neither is inferred from
    // the bytes.
    assert!(
        matches!(
            observed,
            PhysicalProjectionPlan::FusedQ8 | PhysicalProjectionPlan::Q8xQ8
        ),
        "q8 residency produced {observed:?}"
    );
}

/// Blanket Q4 is the DEFAULT of a Q4 arm, so an arm run without an
/// exception set is the hypothesis rather than a silent recipe.
#[test]
fn an_unset_exception_list_is_blanket_q4_not_a_quiet_recipe() {
    use super::super::physical::{q4_classes, Q4Classes, Q4_CLASSES_ENV};

    if std::env::var(Q4_CLASSES_ENV).is_ok() {
        return; // a deliberately-scoped process is running another rung
    }
    assert_eq!(
        q4_classes(),
        Q4Classes::ALL,
        "an unset {Q4_CLASSES_ENV} must mean blanket Q4, not an implicit exception set"
    );
}

/// A residual-stream-shaped activation: a few channels tens of times the
/// RMS, which is the regime the real model is in (peak/rms 28-36 at
/// depth).
fn outlier_activation(n: usize, seed: u64) -> Vec<f32> {
    let mut x = lcg_values(n, seed);
    for (i, v) in x.iter_mut().enumerate() {
        if i % 173 == 0 {
            *v *= 45.0;
        }
    }
    x
}

/// **The sub-blocked Q4 row computes what the format denotes.**
///
/// Against a scalar definition written here, because the packing is the
/// part that can go wrong: byte `j` carries element `j` low and
/// `j + block/2` high, so a sub-block that took adjacent nibbles would
/// pair every weight with the wrong activation and still return finite,
/// plausible numbers.
#[test]
fn the_subblocked_q4_row_computes_what_the_format_denotes() {
    use super::super::integer::q4_row_subblocked;
    const IN: usize = 512;
    let w = lcg_values(IN, 51);
    let (packed, wscales) = q4_parts(&w, IN);
    let x = outlier_activation(IN, 52);

    for ablock in [16usize, 32] {
        let (qx, ascales) = super::super::integer::quantise_activation_blocked(&x, ablock);
        let per_weight = Q4_BLOCK / ablock;
        let folded: Vec<f32> = ascales
            .iter()
            .enumerate()
            .map(|(s, a)| wscales[s / per_weight] * *a)
            .collect();

        // The definition: decode every weight from its nibble, multiply
        // by the reconstructed activation, sum.
        let mut want = 0.0f64;
        #[allow(clippy::needless_range_loop)]
        // The index IS the subject here: it selects a nibble, a byte and
        // a scale by three different divisions, and iterating `qx`
        // would hide the one relationship the test exists to pin.
        for i in 0..IN {
            let b = i / Q4_BLOCK;
            let off = i % Q4_BLOCK;
            let half = Q4_BLOCK / 2;
            let (byte, high) = if off < half {
                (packed[b * (Q4_BLOCK / 2) + off], false)
            } else {
                (packed[b * (Q4_BLOCK / 2) + off - half], true)
            };
            let code = if high {
                (byte >> 4) as i32 - 8
            } else {
                (byte & 0x0f) as i32 - 8
            };
            let s = i / ablock;
            want += (code as f64) * (qx[i] as f64) * (folded[s] as f64);
        }
        let got = q4_row_subblocked(&packed, &folded, &qx, IN, Q4_BLOCK, ablock);
        let rel = ((got as f64 - want) / want.abs().max(1e-12)).abs();
        assert!(
            rel < 1e-5,
            "ablock {ablock}: kernel {got} vs definition {want} (rel {rel:.2e})"
        );
    }
}

/// **With equal sub-block scales, sub-blocking reduces to whole-block.**
///
/// The reduction that has to hold: if every activation sub-block inside a
/// weight block carries the SAME scale, then scaling per sub-block and
/// scaling per block describe the same arithmetic, and a generalisation
/// that disagreed there would be a second implementation rather than a
/// generalisation.
///
/// `ablock == block` is deliberately NOT tested here — a sub-block would
/// span both nibble runs, which is out of this kernel's contract and is
/// why the caller routes that case to `q4_row`. The contract is asserted
/// in the kernel, and the routing is checked below.
#[test]
fn subblocking_reduces_to_whole_block_when_the_scales_agree() {
    use super::super::integer::{q4_row_portable, q4_row_subblocked};
    const IN: usize = 256;
    let w = lcg_values(IN, 53);
    let (packed, wscales) = q4_parts(&w, IN);
    let x = outlier_activation(IN, 54);
    let (qx, _) = super::super::integer::quantise_activation_blocked(&x, Q4_BLOCK);

    let ablock = Q4_BLOCK / 2;
    let per_weight = Q4_BLOCK / ablock;
    // One scale per weight block, repeated across its sub-blocks.
    let folded_sub: Vec<f32> = (0..IN / ablock).map(|s| wscales[s / per_weight]).collect();
    let folded_whole: Vec<f32> = wscales.to_vec();

    let sub = q4_row_subblocked(&packed, &folded_sub, &qx, IN, Q4_BLOCK, ablock);
    let whole = q4_row_portable(&packed, &folded_whole, &qx, IN, Q4_BLOCK);
    let rel = ((sub - whole) as f64 / (whole as f64).abs().max(1e-12)).abs();
    assert!(
        rel < 1e-6,
        "sub-blocked {sub} vs whole-block {whole} (rel {rel:.2e})"
    );
}

/// The whole-block case is routed away from the sub-blocked kernel, and
/// the kernel refuses it rather than reading past a block's bytes.
#[test]
#[should_panic(expected = "q4 sub-blocking needs ablock")]
fn the_subblocked_kernel_refuses_a_straddling_block() {
    use super::super::integer::q4_row_subblocked;
    let packed = vec![0u8; 32];
    let folded = vec![1.0f32; 1];
    let qx = vec![1i8; 64];
    q4_row_subblocked(&packed, &folded, &qx, 64, Q4_BLOCK, Q4_BLOCK);
}

/// **The mechanism claim, made falsifiable: a finer activation block must
/// REDUCE error on an outlier-laden activation.**
///
/// This is the whole justification for spending a rung on the activation
/// rather than on the weight format. An outlier channel in a block of 64
/// crushes 63 neighbours; in a block of 16 it crushes 15. If the error
/// did NOT fall with the block, the diagnosis would be wrong and the
/// activation programme would be chasing the wrong variable.
///
/// Stated as a strict ordering rather than a threshold, because the
/// magnitude is what the bank measures and a number pinned here would be
/// a second, weaker claim about the same thing.
#[test]
fn a_finer_activation_block_reduces_error_on_an_outlier_activation() {
    const IN: usize = 1024;
    let x = outlier_activation(IN, 55);
    let peak = x.iter().fold(0.0f32, |m, v| m.max(v.abs()));
    let rms = (x.iter().map(|v| (*v as f64).powi(2)).sum::<f64>() / IN as f64).sqrt();
    assert!(
        peak as f64 / rms > 10.0,
        "the fixture must actually be heavy-tailed or it tests nothing"
    );

    let err = |ablock: usize| {
        let (qx, ascales) = super::super::integer::quantise_activation_blocked(&x, ablock);
        let se: f64 = x
            .iter()
            .zip(&qx)
            .enumerate()
            .map(|(i, (v, q))| {
                let back = *q as f64 * ascales[i / ablock] as f64;
                (back - *v as f64).powi(2)
            })
            .sum();
        (se / IN as f64).sqrt()
    };

    let (e64, e32, e16) = (err(64), err(32), err(16));
    assert!(
        e16 < e32 && e32 < e64,
        "finer activation blocks did not reduce reconstruction error: \
         64 -> {e64:.3e}, 32 -> {e32:.3e}, 16 -> {e16:.3e}"
    );
    // And the per-tensor scale must be the worst of all — the arm CPU-5
    // measured at KL 0.00061 with exact weights.
    let act = quantise_activation(&x);
    let tensor: f64 = (x
        .iter()
        .zip(&act.codes)
        .map(|(v, c)| (*c as f64 * act.scale as f64 - *v as f64).powi(2))
        .sum::<f64>()
        / IN as f64)
        .sqrt();
    assert!(
        tensor > e64,
        "per-tensor {tensor:.3e} vs block-64 {e64:.3e}"
    );
}

/// **The asymmetric kernel computes what the format denotes.**
///
/// Bit-identity again: both block sums are integer and exact, and the
/// two float operations per block happen in the same order, so the
/// vectorised path and the definition cannot legitimately differ.
#[test]
fn the_asymmetric_q8_kernel_computes_what_the_format_denotes() {
    use super::super::integer::{q8_row_asym_portable, quantise_activation_asymmetric};
    const OUT: usize = 4;
    for in_dim in [64usize, 192, 1024] {
        let w = lcg_values(OUT * in_dim, 61);
        let x = outlier_activation(in_dim, 62);
        let (codes, wscales) = q8_parts(&w, in_dim);
        let ablock = 16usize;
        let (qx, ascales, amids) = quantise_activation_asymmetric(&x, ablock);
        let per_row = in_dim.div_ceil(Q8_BLOCK);
        let per_weight = Q8_BLOCK / ablock;

        for o in 0..OUT {
            let ws = &wscales[o * per_row..(o + 1) * per_row];
            let fs: Vec<f32> = ascales
                .iter()
                .enumerate()
                .map(|(b, a)| ws[b / per_weight] * *a)
                .collect();
            let fm: Vec<f32> = amids
                .iter()
                .enumerate()
                .map(|(b, m)| ws[b / per_weight] * *m)
                .collect();
            let row = &codes[o * in_dim..(o + 1) * in_dim];
            // The EXACT path explicitly: K3 is the process default and
            // reassociates by design, so the denotation gate has to name
            // the implementation it is pinning.
            let got = super::super::integer::q8_row_asym_exact(row, &fs, &fm, &qx, in_dim, ablock);
            let want = q8_row_asym_portable(row, &fs, &fm, &qx, in_dim, ablock);
            assert_eq!(
                got.to_bits(),
                want.to_bits(),
                "asym row {o} at in_dim {in_dim}: {got} vs definition {want}"
            );
        }
    }
}

/// **A constant block reconstructs EXACTLY**, which a symmetric code
/// cannot do.
///
/// The clearest statement of what the offset buys: a block whose values
/// are all the same has zero span, so every code is zero and the offset
/// alone carries it. Symmetric coding has to represent that value as a
/// multiple of `peak/127` and rounds it.
#[test]
fn an_offset_code_reconstructs_a_constant_block_exactly() {
    use super::super::integer::quantise_activation_asymmetric;
    let x = vec![0.7314f32; 32];
    let (codes, scales, mids) = quantise_activation_asymmetric(&x, 16);
    assert!(codes.iter().all(|c| *c == 0));
    for (b, m) in mids.iter().enumerate() {
        assert_eq!(*m, x[0], "block {b} offset {m} is not the constant itself");
        assert_eq!(scales[b], 1.0, "a zero-span block takes the sentinel scale");
    }
}

/// **The offset must EARN its place: it beats the symmetric code on a
/// one-sided block, and does not lose on a balanced one.**
///
/// The mechanism claim behind the whole rung, made falsifiable. If a
/// per-block offset did not reduce reconstruction error where blocks are
/// off-centre, there would be nothing to build.
#[test]
fn an_offset_code_beats_the_symmetric_one_where_blocks_are_off_centre() {
    use super::super::integer::{quantise_activation_asymmetric, quantise_activation_blocked};
    const N: usize = 1024;
    const BLK: usize = 16;

    let err = |x: &[f32], asym: bool| -> f64 {
        let se: f64 = if asym {
            let (c, s, m) = quantise_activation_asymmetric(x, BLK);
            x.iter()
                .enumerate()
                .map(|(i, v)| {
                    let r = c[i] as f64 * s[i / BLK] as f64 + m[i / BLK] as f64;
                    (r - *v as f64).powi(2)
                })
                .sum()
        } else {
            let (c, s) = quantise_activation_blocked(x, BLK);
            x.iter()
                .enumerate()
                .map(|(i, v)| {
                    let r = c[i] as f64 * s[i / BLK] as f64;
                    (r - *v as f64).powi(2)
                })
                .sum()
        };
        (se / x.len() as f64).sqrt()
    };

    // One-sided: every block strictly positive, so symmetric wastes half
    // its range on a sign that never occurs.
    let one_sided: Vec<f32> = lcg_values(N, 63).iter().map(|v| v.abs() + 1.0).collect();
    let (a, sym) = (err(&one_sided, true), err(&one_sided, false));
    assert!(
        a < sym * 0.75,
        "on a one-sided activation the offset code should be clearly better: \
         asym {a:.3e} vs sym {sym:.3e}"
    );

    // Balanced and heavy-tailed — the real regime. It must not LOSE.
    let real = outlier_activation(N, 64);
    let (a2, sym2) = (err(&real, true), err(&real, false));
    assert!(
        a2 <= sym2 * 1.01,
        "on a balanced activation the offset code must not lose: \
         asym {a2:.3e} vs sym {sym2:.3e}"
    );
}

/// **CPU5-K1 is BIT-IDENTICAL to the kernel it replaces.**
///
/// The whole licence for skipping another 69-prompt bank run. K1 reads
/// `SUM(q)` from a precomputed index instead of recomputing it with a
/// second `SDOT`; an i32 sum of i16 sub-sums taken in order is the same
/// integer the reduction produced, and no float operation changes. So
/// the arithmetic that passed the quality gates is the arithmetic that
/// runs — asserted, not argued.
#[test]
fn the_indexed_asymmetric_row_is_bit_identical_to_the_recomputing_one() {
    use super::super::integer::{q8_row_asym_indexed, quantise_activation_asymmetric};
    use crate::format::vindex3::opplan::exec::quantise::{quantise_q8_indexed_for_test, SUM_BLOCK};

    const OUT: usize = 4;
    for in_dim in [64usize, 192, 1024, 5120] {
        let w = lcg_values(OUT * in_dim, 71);
        let x = outlier_activation(in_dim, 72);
        let LoadedWeight::Q8 {
            codes,
            scales,
            sums,
        } = quantise_q8_indexed_for_test(&w, in_dim)
        else {
            panic!("the indexed quantiser must produce the q8 variant");
        };
        assert!(!sums.is_empty(), "the index must actually be built");
        assert_eq!(
            sums.len(),
            OUT * in_dim.div_ceil(SUM_BLOCK),
            "one sum per {SUM_BLOCK} codes per row"
        );

        let per_row = in_dim.div_ceil(Q8_BLOCK);
        let per_sum = in_dim.div_ceil(SUM_BLOCK);
        for ablock in [16usize, 32, 64] {
            let (qx, ascales, amids) = quantise_activation_asymmetric(&x, ablock);
            let per_weight = Q8_BLOCK / ablock;
            for o in 0..OUT {
                let ws = &scales[o * per_row..(o + 1) * per_row];
                let fs: Vec<f32> = ascales
                    .iter()
                    .enumerate()
                    .map(|(b, a)| ws[b / per_weight] * *a)
                    .collect();
                let fm: Vec<f32> = amids
                    .iter()
                    .enumerate()
                    .map(|(b, m)| ws[b / per_weight] * *m)
                    .collect();
                let row = &codes[o * in_dim..(o + 1) * in_dim];
                let recomputed =
                    super::super::integer::q8_row_asym_exact(row, &fs, &fm, &qx, in_dim, ablock);
                let indexed = q8_row_asym_indexed(
                    row,
                    &fs,
                    &fm,
                    &qx,
                    &sums[o * per_sum..(o + 1) * per_sum],
                    in_dim,
                    ablock,
                );
                assert_eq!(
                    indexed.to_bits(),
                    recomputed.to_bits(),
                    "in_dim {in_dim}, ablock {ablock}, row {o}: indexed {indexed} vs \
                     recomputed {recomputed}"
                );
            }
        }
    }
}

/// The index is EXACT and fits i16, which is what makes it one bit per
/// weight rather than two.
#[test]
fn the_code_sum_index_is_exact_and_fits_i16() {
    use crate::format::vindex3::opplan::exec::quantise::{quantise_q8_indexed_for_test, SUM_BLOCK};
    const IN: usize = 320;
    let w = lcg_values(IN, 73);
    let LoadedWeight::Q8 { codes, sums, .. } = quantise_q8_indexed_for_test(&w, IN) else {
        panic!("expected q8");
    };
    for (b, s) in sums.iter().enumerate() {
        let lo = b * SUM_BLOCK;
        let hi = (lo + SUM_BLOCK).min(IN);
        let want: i32 = codes[lo..hi].iter().map(|c| *c as i32).sum();
        assert_eq!(*s as i32, want, "sum block {b}");
        assert!(want.abs() <= SUM_BLOCK as i32 * 127);
    }
}

/// A symmetric arm builds NO index — it has no use for one, and paying
/// ~1 bit/weight of residency and traffic for it would slow the arm that
/// is currently fastest.
#[test]
fn the_symmetric_path_carries_no_index() {
    let LoadedWeight::Q8 { sums, .. } = quantise_q8_for_test(&lcg_values(256, 74), 64) else {
        panic!("expected q8");
    };
    assert!(
        sums.is_empty(),
        "the symmetric quantiser built an index nothing reads"
    );
}

/// **The K2-vs-K3 control: a bug detector, NOT an acceptance substitute.**
///
/// K3 reassociates the sum of already-computed block contributions, so a
/// difference is expected — at the ROUNDING level. This pins the size of
/// it. A reading near 1e-7 says the reassociation is behaving as a
/// reassociation; a reading near 1e-3 says the kernel is computing
/// something else, and catches that before a 50-minute bank run rather
/// than after.
///
/// The frozen quality gates are re-established on the full bank
/// regardless of what this says. A tolerance passed here licenses
/// nothing about the model.
#[test]
fn k3_moves_the_row_only_at_the_rounding_level() {
    use super::super::integer::{
        q8_row_asym_exact, q8_row_asym_k3, quantise_activation_asymmetric,
    };
    const OUT: usize = 8;
    const IN: usize = 5120;
    let w = lcg_values(OUT * IN, 81);
    let x = outlier_activation(IN, 82);
    let (codes, wscales) = q8_parts(&w, IN);
    let ablock = 16usize;
    let (qx, ascales, amids) = quantise_activation_asymmetric(&x, ablock);
    let per_row = IN.div_ceil(Q8_BLOCK);
    let per_weight = Q8_BLOCK / ablock;

    let mut worst = 0.0f64;
    for o in 0..OUT {
        let ws = &wscales[o * per_row..(o + 1) * per_row];
        let fs: Vec<f32> = ascales
            .iter()
            .enumerate()
            .map(|(b, a)| ws[b / per_weight] * *a)
            .collect();
        let fm: Vec<f32> = amids
            .iter()
            .enumerate()
            .map(|(b, m)| ws[b / per_weight] * *m)
            .collect();
        let row = &codes[o * IN..(o + 1) * IN];
        let k2 = q8_row_asym_exact(row, &fs, &fm, &qx, IN, ablock);
        let k3 = q8_row_asym_k3(row, &fs, &fm, &qx, IN);
        let rel = ((k3 - k2) as f64 / (k2 as f64).abs().max(1e-9)).abs();
        worst = worst.max(rel);
    }
    assert!(
        worst < 1e-4,
        "K3 moves the row by {worst:.3e} relative — that is not reassociation, it is a \
         different computation"
    );
    // And it must not be a no-op dressed as an optimisation: over 5120
    // f32 accumulations SOME difference is expected, and exact agreement
    // would mean the K3 path never ran.
    assert!(
        worst > 0.0,
        "K3 agreed with K2 exactly, so the K3 kernel did not run"
    );
}

/// **K4 is BIT-IDENTICAL to K3**, which is what lets one Bank-1 run
/// cover both.
///
/// K4 replaces four correction `SDOT`s with one 64-bit load of the
/// precomputed sums. Those hold exactly the integers the `SDOT`s
/// produce, so `vcvtq_f32_s32` sees the same lanes and no float
/// operation changes. Unlike K3-vs-K2 this is an equality, not a
/// tolerance — and it is asserted rather than argued, because K1 already
/// demonstrated that an index can be plumbed wrongly while still
/// returning finite, plausible numbers.
#[test]
fn k4_is_bit_identical_to_k3() {
    use super::super::integer::{q8_row_asym_k3, q8_row_asym_k4, quantise_activation_asymmetric};
    use crate::format::vindex3::opplan::exec::quantise::{quantise_q8_indexed_for_test, SUM_BLOCK};

    const OUT: usize = 6;
    // 5120 is the real width; 80 exercises a group-of-four tail.
    for in_dim in [80usize, 256, 5120] {
        let w = lcg_values(OUT * in_dim, 91);
        let x = outlier_activation(in_dim, 92);
        let LoadedWeight::Q8 {
            codes,
            scales,
            sums,
        } = quantise_q8_indexed_for_test(&w, in_dim)
        else {
            panic!("expected q8");
        };
        let ablock = SUM_BLOCK;
        let (qx, ascales, amids) = quantise_activation_asymmetric(&x, ablock);
        let per_row = in_dim.div_ceil(Q8_BLOCK);
        let per_sum = in_dim.div_ceil(SUM_BLOCK);
        let per_weight = Q8_BLOCK / ablock;

        for o in 0..OUT {
            let ws = &scales[o * per_row..(o + 1) * per_row];
            let fs: Vec<f32> = ascales
                .iter()
                .enumerate()
                .map(|(b, a)| ws[b / per_weight] * *a)
                .collect();
            let fm: Vec<f32> = amids
                .iter()
                .enumerate()
                .map(|(b, m)| ws[b / per_weight] * *m)
                .collect();
            let row = &codes[o * in_dim..(o + 1) * in_dim];
            let k3 = q8_row_asym_k3(row, &fs, &fm, &qx, in_dim);
            let k4 = q8_row_asym_k4(
                row,
                &fs,
                &fm,
                &qx,
                &sums[o * per_sum..(o + 1) * per_sum],
                in_dim,
            );
            assert_eq!(
                k4.to_bits(),
                k3.to_bits(),
                "in_dim {in_dim}, row {o}: K4 {k4} vs K3 {k3}"
            );
        }
    }
}

/// **K5 is BIT-IDENTICAL to K3**, for both arms.
///
/// K5 builds the folded scale in a register (`ws * ascale[b]`) where K3
/// built it in a per-row buffer. Multiplying two f32 gives the same f32
/// whether the result goes through memory first, and every operation
/// after that is unchanged — so no new numerical evidence is needed
/// beyond the Bank-1 run K3 already requires.
///
/// Asserted for the ASYMMETRIC arm (the candidate) and the SYMMETRIC one
/// (the control), because K5 changes both.
#[test]
fn k5_is_bit_identical_to_k3_on_both_arms() {
    use super::super::integer::{
        q8_row_asym_k3, q8_row_k3_register, quantise_activation_asymmetric,
        quantise_activation_blocked,
    };
    const OUT: usize = 6;
    for in_dim in [64usize, 256, 5120] {
        let w = lcg_values(OUT * in_dim, 101);
        let x = outlier_activation(in_dim, 102);
        let (codes, wscales) = q8_parts(&w, in_dim);
        let ablock = 16usize;
        let per_row = in_dim.div_ceil(Q8_BLOCK);
        let per_weight = Q8_BLOCK / ablock;

        // --- asymmetric: the candidate ---
        let (qx, ascales, amids) = quantise_activation_asymmetric(&x, ablock);
        for o in 0..OUT {
            let ws = &wscales[o * per_row..(o + 1) * per_row];
            let fs: Vec<f32> = ascales
                .iter()
                .enumerate()
                .map(|(b, a)| ws[b / per_weight] * *a)
                .collect();
            let fm: Vec<f32> = amids
                .iter()
                .enumerate()
                .map(|(b, m)| ws[b / per_weight] * *m)
                .collect();
            let row = &codes[o * in_dim..(o + 1) * in_dim];
            let k3 = q8_row_asym_k3(row, &fs, &fm, &qx, in_dim);
            let k5 = q8_row_k3_register(row, ws, &ascales, Some(&amids), &qx, in_dim);
            assert_eq!(
                k5.to_bits(),
                k3.to_bits(),
                "asym in_dim {in_dim} row {o}: K5 {k5} vs K3 {k3}"
            );
        }

        // --- symmetric: the control ---
        let (qxs, sscales) = quantise_activation_blocked(&x, ablock);
        for o in 0..OUT {
            let ws = &wscales[o * per_row..(o + 1) * per_row];
            let fs: Vec<f32> = sscales
                .iter()
                .enumerate()
                .map(|(b, a)| ws[b / per_weight] * *a)
                .collect();
            let row = &codes[o * in_dim..(o + 1) * in_dim];
            let k3 = super::super::integer::q8_row_k3_sym(row, &fs, &qxs, in_dim);
            let k5 = q8_row_k3_register(row, ws, &sscales, None, &qxs, in_dim);
            assert_eq!(
                k5.to_bits(),
                k3.to_bits(),
                "sym in_dim {in_dim} row {o}: K5 {k5} vs K3 {k3}"
            );
        }
    }
}
