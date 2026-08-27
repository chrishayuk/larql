//! The plan must pair a format with a kernel that consumes it, and the
//! executor's observation must land on the loader's decision.

use super::super::physical::{
    compact_threshold_bytes, project_matrix, project_rows, project_rows_many, ExecutorProjections,
    PhysicalProjectionPlan, BF16_BYTES, F32_BYTES,
};
use super::super::projector::WeightRows;
use crate::format::vindex3::opplan::exec::backend::{WeightFormat, WeightSlice};
use crate::format::vindex3::opplan::exec::gated_delta::DenseProjections;
use crate::format::vindex3::opplan::exec::quantise::{quantise_q8_for_test, Q8_BLOCK};
use crate::format::vindex3::opplan::exec::weights::LoadedWeight;

/// Every matrix Qwen3.8-27B decodes through, from the container's own
/// tensor table: `(name, elements)`.
///
/// The whole population rather than a sample, because the claim the
/// policy makes is about the model's residency — and a residency claim
/// that skipped a class would be a claim about part of a model.
///
/// **All thirteen are stored BF16.** The container reports one encoding
/// for the decoder stack, so nothing here is separated by what the
/// checkpoint holds; the two populations below are separated by SIZE
/// alone. A table that marked the delta gates "not stored bf16" would
/// pass the same assertions while testing nothing — the gates would be
/// f32-resident because of the checkpoint, and the threshold could be
/// any number at all.
const REAL_MATRICES: &[(&str, usize)] = &[
    ("mlp.gate_proj", 17408 * 5120),
    ("mlp.up_proj", 17408 * 5120),
    ("mlp.down_proj", 5120 * 17408),
    ("linear_attn.in_proj_qkv", 10240 * 5120),
    ("linear_attn.in_proj_z", 6144 * 5120),
    ("linear_attn.out_proj", 5120 * 6144),
    ("self_attn.q_proj", 12288 * 5120),
    ("self_attn.o_proj", 5120 * 6144),
    ("self_attn.k_proj", 1024 * 5120),
    ("self_attn.v_proj", 1024 * 5120),
    ("linear_attn.in_proj_a", 48 * 5120),
    ("linear_attn.in_proj_b", 48 * 5120),
    ("output_head", 248320 * 5120),
];

/// The stored encoding of every one of them, per the container index.
const STORED_BF16: bool = true;

/// A slab in the plan's OWN format, so a mispairing cannot be papered
/// over by the test choosing the representation the kernel wanted.
struct Slab {
    f32s: Vec<f32>,
    bf16: Vec<u16>,
    codes: Vec<i8>,
    scales: Vec<f32>,
}

fn slab(plan: PhysicalProjectionPlan, elements: usize, in_dim: usize) -> Slab {
    let mut s = Slab {
        f32s: Vec::new(),
        bf16: Vec::new(),
        codes: Vec::new(),
        scales: Vec::new(),
    };
    match plan.format() {
        WeightFormat::F32 => s.f32s = vec![0.5f32; elements],
        WeightFormat::Bf16 => s.bf16 = vec![0x3f00u16; elements],
        WeightFormat::Q8 => {
            s.codes = vec![64i8; elements];
            s.scales = vec![0.01f32; (elements / in_dim) * in_dim.div_ceil(Q8_BLOCK)];
        }
        other => panic!("no CPU plan declares {other:?}"),
    }
    s
}

fn rows<'a>(plan: PhysicalProjectionPlan, s: &'a Slab) -> WeightRows<'a> {
    match plan.format() {
        WeightFormat::F32 => WeightRows::F32(&s.f32s),
        WeightFormat::Bf16 => WeightRows::Bf16(&s.bf16),
        WeightFormat::Q8 => WeightRows::Q8 {
            codes: &s.codes,
            scales: &s.scales,
            sums: &[],
            block: Q8_BLOCK,
        },
        other => panic!("no CPU plan declares {other:?}"),
    }
}

/// **The load-bearing invariant.** What the loader made resident is what
/// the executor observes.
///
/// This is the whole reason the plan is one value: if `choose` and
/// `for_resident` could disagree about a matrix, a BF16-resident weight
/// could be handed to a kernel expecting f32 — and the failure mode is
/// not a wrong answer but 100 MB read as garbage.
#[test]
fn the_observation_lands_on_the_decision() {
    for (name, elements) in REAL_MATRICES.iter().copied() {
        let chosen = PhysicalProjectionPlan::choose(elements, STORED_BF16);
        // A one-row stand-in: the round trip is about representation, and
        // allocating 1.3 G elements to prove it would measure the
        // allocator.
        let s = slab(chosen, 8, 8);
        let observed = PhysicalProjectionPlan::for_resident(rows(chosen, &s), 8);
        assert_eq!(
            observed, chosen,
            "`{name}`: the executor observed {observed:?} where the loader chose {chosen:?} — \
             one matrix, two derivations, and they disagree"
        );
    }
}

/// Each plan's kernel actually consumes each plan's format.
///
/// The kernels panic on the wrong representation, so a mispaired variant
/// fails here loudly rather than at decode on a real container.
#[test]
fn every_plan_runs_its_own_format() {
    let x = vec![1.0f32; Q8_BLOCK];
    for plan in [
        PhysicalProjectionPlan::ScalarF32,
        PhysicalProjectionPlan::BlasF32,
        PhysicalProjectionPlan::FusedBf16,
        PhysicalProjectionPlan::FusedQ8,
    ] {
        let s = slab(plan, Q8_BLOCK * 2, Q8_BLOCK);
        let mut out = vec![0.0f32; 2];
        plan.kernel().project_rows(rows(plan, &s), &x, &mut out);
        assert!(
            out.iter().all(|v| v.is_finite() && *v != 0.0),
            "{plan:?} produced nothing from its own declared format"
        );
    }
}

/// The oracle is chosen by IDENTITY, not by representation.
///
/// `for_resident` is total over what a CPU kernel can hold, and f32 has
/// two kernels: the production `BlasF32` and the reference `ScalarF32`.
/// It answers `BlasF32`, and that is not an omission — the reference
/// backend declares its plan because of what it IS, so nothing ever asks
/// the bytes which of the two it wanted. Asserting the asymmetry here
/// stops a later reader "fixing" it by making the observation guess.
#[test]
fn the_oracle_is_not_reachable_by_observation() {
    let f = vec![0.5f32; 8];
    assert_eq!(
        PhysicalProjectionPlan::for_resident(WeightRows::F32(&f), 8),
        PhysicalProjectionPlan::BlasF32
    );
    let at = compact_threshold_bytes() / F32_BYTES;
    for elements in [1, at - 1, at, at * 64] {
        for stored in [false, true] {
            assert_ne!(
                PhysicalProjectionPlan::choose(elements, stored),
                PhysicalProjectionPlan::ScalarF32,
                "the policy must never route production through the oracle"
            );
        }
    }
}

/// An f32 checkpoint never reaches the compact kernel, however large.
///
/// The alternative would be to narrow at load to hit the threshold, which
/// would ROUND — the policy would be quantising a model while reporting a
/// residency win.
#[test]
fn a_checkpoint_without_stored_bf16_stays_f32() {
    let huge = 1_000 * compact_threshold_bytes() / F32_BYTES;
    assert_eq!(
        PhysicalProjectionPlan::choose(huge, false),
        PhysicalProjectionPlan::BlasF32
    );
    assert_eq!(
        PhysicalProjectionPlan::choose(huge, true),
        PhysicalProjectionPlan::FusedQ8
    );
}

/// **The real model's THREE populations.**
///
/// One boundary per format, each the point where the ALTERNATIVE's image
/// stops fitting cache — the f32 image for bf16, the bf16 image for Q8.
/// Qwen3.8 puts real matrices on every side of both, so a policy that
/// answered uniformly would be wrong three different ways.
#[test]
fn the_real_model_splits_into_three_populations() {
    let plan_of = |elements| PhysicalProjectionPlan::choose(elements, STORED_BF16);
    let named = |want| {
        REAL_MATRICES
            .iter()
            .filter(|(_, e)| plan_of(*e) == want)
            .map(|(n, _)| *n)
            .collect::<Vec<_>>()
    };
    let q8 = named(PhysicalProjectionPlan::FusedQ8);
    let bf16 = named(PhysicalProjectionPlan::FusedBf16);
    let blas = named(PhysicalProjectionPlan::BlasF32);

    // The measured crossovers, not a restatement of the rule: `1024 x
    // 5120` runs 0.81x through Q8 because its bf16 image is 10.5 MB and
    // already L2-resident, and `48 x 5120` runs 3.8x faster through BLAS
    // for the same reason one format further up.
    assert_eq!(
        bf16,
        vec!["self_attn.k_proj", "self_attn.v_proj"],
        "the streaming/cache-resident boundary moved"
    );
    assert_eq!(
        blas,
        vec!["linear_attn.in_proj_a", "linear_attn.in_proj_b"],
        "the tiny delta gates must stay f32"
    );
    assert_eq!(
        q8.len(),
        REAL_MATRICES.len() - bf16.len() - blas.len(),
        "every matrix must land in exactly one population: {q8:?}"
    );
    assert!(q8.contains(&"output_head"));
    assert!(q8.contains(&"mlp.gate_proj"));
}

/// Each boundary is bracketed on both sides, at its own alternative's
/// byte width.
#[test]
fn both_boundaries_are_bracketed() {
    let l2 = compact_threshold_bytes();
    let f32_edge = l2 / F32_BYTES;
    let bf16_edge = l2 / BF16_BYTES;
    assert_eq!(
        PhysicalProjectionPlan::choose(f32_edge - 1, true),
        PhysicalProjectionPlan::BlasF32,
        "below the f32 boundary the widened image still fits cache"
    );
    assert_eq!(
        PhysicalProjectionPlan::choose(f32_edge, true),
        PhysicalProjectionPlan::FusedBf16
    );
    assert_eq!(
        PhysicalProjectionPlan::choose(bf16_edge - 1, true),
        PhysicalProjectionPlan::FusedBf16,
        "below the bf16 boundary there is no traffic left for Q8 to halve, and its extra \
         unpacking is pure cost"
    );
    assert_eq!(
        PhysicalProjectionPlan::choose(bf16_edge, true),
        PhysicalProjectionPlan::FusedQ8
    );
}

/// A projection runs through the executor under its own plan, whichever
/// representation it is resident as, and every representation agrees on
/// the answer to within what its own format costs.
///
/// bf16 must agree with f32 to summation order, because bf16 widens
/// exactly. Q8 must NOT: it is a lossy format and an assertion that it
/// matched to 1e-5 would either be testing nothing or be about to fail on
/// a checkpoint with wider blocks. Its tolerance is stated as what
/// symmetric int8 costs.
#[test]
fn every_representation_projects_to_its_own_accuracy() {
    const OUT: usize = 24;
    const IN: usize = Q8_BLOCK * 2;
    let f: Vec<f32> = (0..OUT * IN)
        .map(|i| {
            let v = (i as f32 * 0.013).sin();
            f32::from_bits(v.to_bits() & 0xffff_0000)
        })
        .collect();
    let b: Vec<u16> = f.iter().map(|v| (v.to_bits() >> 16) as u16).collect();
    let x: Vec<f32> = (0..IN).map(|i| (i as f32 * 0.07).cos()).collect();

    let widened = project_matrix(&WeightSlice::F32(&f), &x, OUT, IN).unwrap();
    let compact = project_matrix(&WeightSlice::Bf16(&b), &x, OUT, IN).unwrap();
    let gated = ExecutorProjections.project(WeightRows::Bf16(&b), &x, OUT);
    assert_eq!(
        compact, gated,
        "the delta seam and the plan seam must agree exactly"
    );

    let rel = |a: &[f32], want: &[f32]| {
        let (mut num, mut den) = (0.0f64, 0.0f64);
        for (p, q) in a.iter().zip(want) {
            num += (*p as f64 - *q as f64).powi(2);
            den += (*q as f64).powi(2);
        }
        (num / den.max(f64::MIN_POSITIVE)).sqrt()
    };
    assert!(rel(&compact, &widened) < 1e-5, "bf16 widens exactly");

    let LoadedWeight::Q8 { codes, scales, .. } = quantise_q8_for_test(&f, IN) else {
        panic!("the quantiser returns q8");
    };
    let q8 = project_matrix(
        &WeightSlice::Q8 {
            codes: &codes,
            scales: &scales,
            sums: &[],
            block: Q8_BLOCK,
        },
        &x,
        OUT,
        IN,
    )
    .unwrap();
    // Derived, not fitted. Uniform quantisation error is `step/sqrt(12)`
    // with `step = peak/127`; against weights whose RMS is roughly
    // `peak/2` that is `2 / (127 * sqrt(12))` = 4.5e-3, and a dot of
    // random-sign terms preserves the ratio because numerator and
    // denominator both grow as sqrt(N). 1.5e-2 is that with 3x headroom
    // for a block whose peak sits well above its typical weight — still
    // orders of magnitude tighter than a broken kernel would manage.
    assert!(
        rel(&q8, &widened) < 1.5e-2,
        "q8 moved {:.2e}, which is more than symmetric int8 costs",
        rel(&q8, &widened)
    );
}

/// A representation no CPU kernel runs refuses, and names itself.
#[test]
fn a_device_only_representation_refuses_by_name() {
    let err = project_matrix(&WeightSlice::F16(&[0u8; 64]), &[1.0f32; 4], 4, 4)
        .expect_err("no CPU kernel consumes f16")
        .to_string();
    assert!(err.contains("f16"), "{err}");
}

/// The threshold is a real cache size, whatever machine reads it.
#[test]
fn the_threshold_is_a_plausible_cache_size() {
    let bytes = compact_threshold_bytes();
    assert!(
        (1 << 20..=1 << 30).contains(&bytes),
        "{bytes} is not a plausible L2 size — a threshold this far out would put every matrix \
         on one side"
    );
}

/// CPU-7 regression: zero positions must not reach a kernel.
///
/// An empty prompt reaches prefill with no rows, and every kernel below
/// reads the input width off `xs[0]` — the executor, the stationary
/// sweep and the integer kernel alike. Before the guard this panicked
/// with `index out of bounds: the len is 0 but the index is 0`, inside
/// a prefill whose contract is to report "produced no logits" as an
/// ordinary error. A panic there is not a worse error message; it takes
/// the server process with it.
///
/// The guard returns before the executor pool is touched, so this holds
/// even where no pool has been stood up.
#[test]
fn projecting_zero_positions_yields_no_rows_instead_of_panicking() {
    let codes = [0i8; 64];
    let scales = [1.0f32];
    let rows = WeightRows::Q8 {
        codes: &codes,
        scales: &scales,
        sums: &[],
        block: 64,
    };
    let out = project_rows_many(rows, &[], 4).expect("zero positions is not an error");
    assert!(
        out.is_empty(),
        "zero positions must produce zero rows, got {}",
        out.len()
    );
}

/// Every plan states the triple it realises. This is the table a CPU
/// claim is quoted from, and the terms are not cosmetic: the accumulator
/// separates exact integer accumulation from float rounding at every
/// add, and the activation separates the candidate from its own control.
#[test]
fn every_plan_states_the_arithmetic_it_realises() {
    use crate::format::vindex3::opplan::exec::cpu::arithmetic::{
        AccumulatorRep, ActivationRep, WeightRep,
    };

    let all = [
        PhysicalProjectionPlan::ScalarF32,
        PhysicalProjectionPlan::BlasF32,
        PhysicalProjectionPlan::FusedBf16,
        PhysicalProjectionPlan::FusedQ8,
        PhysicalProjectionPlan::FusedQ4,
        PhysicalProjectionPlan::Q8xQ8,
        PhysicalProjectionPlan::Q4xQ8,
        PhysicalProjectionPlan::Bf16xQ8,
    ];
    for p in all {
        // Renders without panicking, and names every term.
        let s = p.arithmetic().to_string();
        assert!(s.contains(" x "), "{p:?} rendered as {s:?}");
        assert!(s.ends_with("-> F32"), "{p:?} must return f32: {s:?}");
    }

    // The float arms read the activation unquantised.
    for p in [
        PhysicalProjectionPlan::ScalarF32,
        PhysicalProjectionPlan::BlasF32,
        PhysicalProjectionPlan::FusedBf16,
        PhysicalProjectionPlan::FusedQ8,
        PhysicalProjectionPlan::FusedQ4,
    ] {
        assert_eq!(
            p.arithmetic().activation,
            ActivationRep::F32,
            "{p:?} must not quantise the activation"
        );
        assert_eq!(p.arithmetic().accumulator, AccumulatorRep::F32);
    }

    // Both oracles hold f32 weights; only the BLAS one is selectable.
    assert_eq!(
        PhysicalProjectionPlan::ScalarF32.arithmetic().weight,
        WeightRep::F32
    );
    assert_eq!(
        PhysicalProjectionPlan::BlasF32.arithmetic().weight,
        WeightRep::F32
    );

    // The integer arms accumulate exactly.
    for p in [PhysicalProjectionPlan::Q8xQ8, PhysicalProjectionPlan::Q4xQ8] {
        assert_eq!(
            p.arithmetic().accumulator,
            AccumulatorRep::I32,
            "{p:?} is an integer kernel and must accumulate in i32"
        );
    }

    // The A1 control is the one arm that keeps EXACT weights while
    // quantising the activation — that separation is its whole job, and
    // an arm that also quantised the weight would stop being a control.
    let ctrl = PhysicalProjectionPlan::Bf16xQ8.arithmetic();
    assert_eq!(ctrl.weight, WeightRep::Bf16);
    assert_eq!(ctrl.accumulator, AccumulatorRep::F32);
    assert!(
        matches!(ctrl.activation, ActivationRep::Q8 { .. }),
        "the control must quantise the activation and nothing else"
    );
}

/// Gated DeltaNet's five projections go through the SAME executor as
/// every other matrix. The seam exists so they cannot quietly acquire
/// their own kernel, so it is asserted against the free functions rather
/// than against a recomputed expectation.
#[test]
fn the_delta_projection_seam_runs_the_same_executor() {
    const OUT: usize = 6;
    const IN: usize = 96;
    let w: Vec<f32> = (0..OUT * IN)
        .map(|i| (i % 17) as f32 * 0.03 - 0.2)
        .collect();
    let bits: Vec<u16> = w.iter().map(|v| (v.to_bits() >> 16) as u16).collect();
    let x: Vec<f32> = (0..IN).map(|i| (i % 11) as f32 * 0.07 - 0.3).collect();
    let y: Vec<f32> = (0..IN).map(|i| (i % 5) as f32 * 0.11 - 0.2).collect();

    let one = ExecutorProjections.project(WeightRows::Bf16(&bits), &x, OUT);
    let direct = project_rows(WeightRows::Bf16(&bits), &x, OUT).unwrap();
    assert_eq!(one, direct, "the seam must not be a second kernel");

    let xs: Vec<&[f32]> = vec![&x, &y];
    let many = ExecutorProjections.project_many(WeightRows::Bf16(&bits), &xs, OUT);
    assert_eq!(many.len(), 2);
    assert_eq!(
        many[0], one,
        "position 0 of a group must equal the single-position answer"
    );
    let direct_many = project_rows_many(WeightRows::Bf16(&bits), &xs, OUT).unwrap();
    assert_eq!(many, direct_many);

    // bf16 has no stationary row, so the seam must say so rather than
    // claiming a schedule it cannot run.
    assert!(!ExecutorProjections.is_weight_stationary(WeightRows::Bf16(&bits), IN, 2));
}

/// An operand past the split threshold is partitioned across workers,
/// and the partition must not change the answer. Rows are deliberately
/// not a multiple of the worker count, so the last chunk is short — the
/// shape that catches a partition computing `len / in_dim` rows instead
/// of its own slice's.
#[test]
fn a_large_operand_splits_across_workers_without_changing_the_answer() {
    // Past MIN_SPLIT_BYTES (4 MiB) so the pooled path is taken.
    const OUT: usize = 2049;
    const IN: usize = 2048;
    let w: Vec<f32> = (0..OUT * IN)
        .map(|i| ((i * 31 % 251) as f32 - 125.0) / 400.0)
        .collect();
    let lw = quantise_q8_for_test(&w, IN);
    let rows = lw.slice().rows(OUT, IN).unwrap();

    let x: Vec<f32> = (0..IN)
        .map(|i| ((i * 7 % 97) as f32 - 48.0) / 90.0)
        .collect();
    let y: Vec<f32> = (0..IN)
        .map(|i| ((i * 13 % 89) as f32 - 44.0) / 80.0)
        .collect();

    let single = project_rows(rows, &x, OUT).unwrap();
    assert_eq!(single.len(), OUT);

    let xs: Vec<&[f32]> = vec![&x, &y];
    let many = project_rows_many(rows, &xs, OUT).unwrap();
    assert_eq!(many.len(), 2);
    assert_eq!(many[0].len(), OUT);
    assert_eq!(
        many[0], single,
        "splitting an operand across workers changed position 0's answer"
    );

    let single_y = project_rows(rows, &y, OUT).unwrap();
    assert_eq!(
        many[1], single_y,
        "splitting an operand across workers changed position 1's answer"
    );
}

/// A plan names the representation the loader must make resident, the
/// representation it is resident AS, and the kernel that consumes it.
/// The three have to agree: a plan whose `format` and `weight_rep`
/// disagreed would have the loader prepare bytes its own kernel cannot
/// read, and the failure would surface as a decode error far from here.
#[test]
fn a_plan_agrees_with_itself_about_format_kernel_and_residency() {
    use crate::format::vindex3::opplan::exec::cpu::arithmetic::WeightRep;

    let cases = [
        (PhysicalProjectionPlan::ScalarF32, WeightFormat::F32),
        (PhysicalProjectionPlan::BlasF32, WeightFormat::F32),
        (PhysicalProjectionPlan::FusedBf16, WeightFormat::Bf16),
        (PhysicalProjectionPlan::Bf16xQ8, WeightFormat::Bf16),
        (PhysicalProjectionPlan::FusedQ8, WeightFormat::Q8),
        (PhysicalProjectionPlan::Q8xQ8, WeightFormat::Q8),
        (PhysicalProjectionPlan::FusedQ4, WeightFormat::Q4),
        (PhysicalProjectionPlan::Q4xQ8, WeightFormat::Q4),
    ];

    for (plan, want) in cases {
        assert_eq!(plan.format(), want, "{plan:?} asks for the wrong residency");

        // `weight_rep` is the arithmetic's own answer; it must describe
        // the same bytes `format` asked the loader for.
        let rep = plan.weight_rep();
        let consistent = matches!(
            (want, rep),
            (WeightFormat::F32, WeightRep::F32)
                | (WeightFormat::Bf16, WeightRep::Bf16)
                | (WeightFormat::Q8, WeightRep::Q8 { .. })
                | (WeightFormat::Q4, WeightRep::Q4 { .. })
        );
        assert!(
            consistent,
            "{plan:?} loads {want:?} but computes over {rep} — the loader would \
             prepare bytes the kernel cannot read"
        );

        // Every plan has a kernel, and it declares its own threading.
        let k = plan.kernel();
        let _ = k.parallelism();
    }

    // The two f32 plans share a residency but differ in everything else.
    // Not asserted by comparing kernel addresses: both kernels are
    // zero-sized, and references to ZSTs may legitimately share one — a
    // pointer comparison here would be testing the allocator, not the
    // dispatch. That the oracle is unreachable by observation is pinned
    // by `the_oracle_is_not_reachable_by_observation`; what belongs here
    // is that they are priced as different arithmetic.
    assert_ne!(
        super::super::cost::measured_rate_gbps(PhysicalProjectionPlan::ScalarF32),
        super::super::cost::measured_rate_gbps(PhysicalProjectionPlan::BlasF32),
        "the oracle and the BLAS path must not be priced alike"
    );
}
