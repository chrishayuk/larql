//! CPU-3A: does a fused Q8 consumer beat the BF16 one, and does it
//! compute what the format denotes?
//!
//! Two separate questions, deliberately not mixed. Whether Q8 is a good
//! enough REPRESENTATION of a checkpoint is CPU-3B and needs logits, a
//! trajectory, continuation state and PARITY-FLOOR-1. Nothing here
//! touches that. Here the format is a given and the only claims are
//! mechanical: the kernel realises `code * scale` faithfully, and it is
//! or is not faster.
//!
//! The bench reports **time per matrix** rather than GB/s, because GB/s
//! is the metric that makes a good Q8 kernel look bad: half the bytes at
//! a lower rate is still less time, and a rate comparison would hide
//! that. Both byte rates are printed — stored bytes and the f32 the
//! weights DENOTE — so neither can be quoted without the other.

use super::super::executor::CpuExecutor;
use super::super::kernels::{
    q4_block_dot_portable, q8_block_dot_portable, q8_dot, FusedQ4, FusedQ8,
};
use super::super::physical::PhysicalProjectionPlan;
use super::super::projector::{DenseProjector, WeightRows};
use crate::format::vindex3::fixtures::lcg_values;

/// Elements per scale. Every real Qwen3.8 `in_dim` (5120, 6144, 17408) is
/// a multiple of it, so the model itself never exercises a tail — which
/// is exactly why the tail has its own test.
const BLOCK: usize = 64;

/// Symmetric per-block int8: `q = round(w / scale)`, `scale = max|w| /
/// 127`.
///
/// The simplest quantiser that can be stated in one line, chosen because
/// CPU-3A is about the KERNEL. Choosing a good quantiser is CPU-3B's
/// problem and it will be measured on logits, not here.
fn quantise(weights: &[f32], in_dim: usize, block: usize) -> (Vec<i8>, Vec<f32>) {
    let per_row = in_dim.div_ceil(block);
    let rows = weights.len() / in_dim;
    let mut codes = vec![0i8; weights.len()];
    let mut scales = vec![0.0f32; rows * per_row];
    for r in 0..rows {
        for b in 0..per_row {
            let lo = r * in_dim + b * block;
            let hi = (lo + block).min((r + 1) * in_dim);
            let peak = weights[lo..hi].iter().fold(0.0f32, |m, w| m.max(w.abs()));
            let scale = if peak > 0.0 { peak / 127.0 } else { 1.0 };
            scales[r * per_row + b] = scale;
            for i in lo..hi {
                codes[i] = (weights[i] / scale).round().clamp(-127.0, 127.0) as i8;
            }
        }
    }
    (codes, scales)
}

/// **The kernel realises the format.**
///
/// Against a scalar definition of `sum over blocks of scale * sum(code *
/// x)`, not against the original f32 weights: the quantiser's error is
/// the FORMAT's, and folding it in here would let a broken kernel hide
/// inside a tolerance chosen for quantisation noise.
#[test]
fn the_q8_kernel_computes_what_the_format_denotes() {
    const OUT: usize = 9;
    for in_dim in [BLOCK, BLOCK * 3, 5120] {
        let w = lcg_values(OUT * in_dim, 5);
        let (codes, scales) = quantise(&w, in_dim, BLOCK);
        let x = lcg_values(in_dim, 6);
        let mut got = vec![0.0f32; OUT];
        FusedQ8.project_rows(
            WeightRows::Q8 {
                codes: &codes,
                scales: &scales,
                sums: &[],
                block: BLOCK,
            },
            &x,
            &mut got,
        );
        let per_row = in_dim.div_ceil(BLOCK);
        for (o, value) in got.iter().enumerate() {
            let mut want = 0.0f32;
            for b in 0..per_row {
                let lo = b * BLOCK;
                let hi = (lo + BLOCK).min(in_dim);
                want += scales[o * per_row + b]
                    * q8_block_dot_portable(&codes[o * in_dim + lo..o * in_dim + hi], &x[lo..hi]);
            }
            let tol = want.abs() * 1e-5 + 1e-4;
            assert!(
                (value - want).abs() <= tol,
                "row {o} at in_dim {in_dim}: {value} against the format's {want}"
            );
        }
    }
}

/// A block that does not divide `in_dim` is handled, and the tail is not
/// silently dropped.
///
/// No real Qwen3.8 shape has a tail — 5120, 6144 and 17408 are all
/// multiples of 64 — so the model could not catch a kernel that walked
/// whole blocks and stopped. The awkward shape is the instrument.
#[test]
fn a_ragged_final_block_is_not_dropped() {
    let in_dim = BLOCK * 2 + 5;
    let w = lcg_values(in_dim, 7);
    let (codes, scales) = quantise(&w, in_dim, BLOCK);
    assert_eq!(scales.len(), 3, "the tail must get its own scale");
    let x = lcg_values(in_dim, 8);

    let full = q8_dot(&codes, &scales, BLOCK, &x);
    // The same dot with the tail's codes zeroed: if the kernel ignored
    // the ragged block these would agree, and the test would be asserting
    // nothing at all.
    let mut truncated = codes.clone();
    for c in truncated[BLOCK * 2..].iter_mut() {
        *c = 0;
    }
    let without = q8_dot(&truncated, &scales, BLOCK, &x);
    assert!(
        (full - without).abs() > 1e-6,
        "the ragged tail contributed nothing, so the kernel is walking whole blocks only"
    );
}

/// The NEON block dot and the portable one agree.
///
/// On aarch64 the portable version is dead code, so the claim that it is
/// "the definition" would otherwise be tested by shipping x86 a wrong
/// answer.
#[test]
fn the_portable_and_neon_block_dots_agree() {
    for len in [1usize, 7, 16, 17, 64, 65] {
        let codes: Vec<i8> = (0..len).map(|i| ((i * 37) % 255) as i8).collect();
        let x = lcg_values(len, 21);
        let portable = q8_block_dot_portable(&codes, &x);
        let mut got = vec![0.0f32; 1];
        FusedQ8.project_rows(
            WeightRows::Q8 {
                codes: &codes,
                scales: &[1.0],
                sums: &[],
                block: len.max(1),
            },
            &x,
            &mut got,
        );
        let magnitude: f32 = codes
            .iter()
            .zip(&x)
            .map(|(c, v)| (*c as f32 * v).abs())
            .sum();
        assert!(
            (got[0] - portable).abs() <= 1e-6 * magnitude.max(1.0),
            "len {len}: {} against {portable}",
            got[0]
        );
    }
}

/// Slicing rows must cut the scales with the codes.
///
/// The executor partitions a projection across workers by ROWS. A cut
/// that moved the codes and not the scales would hand a worker the right
/// weights under a different row's scale — finite, plausible, wrong — and
/// only on multi-worker shapes.
#[test]
fn slicing_rows_cuts_the_scales_too() {
    const OUT: usize = 8;
    let in_dim = BLOCK * 2;
    let w = lcg_values(OUT * in_dim, 12);
    let (codes, scales) = quantise(&w, in_dim, BLOCK);
    let rows = WeightRows::Q8 {
        codes: &codes,
        scales: &scales,
        sums: &[],
        block: BLOCK,
    };
    let x = lcg_values(in_dim, 13);

    let mut whole = vec![0.0f32; OUT];
    FusedQ8.project_rows(rows, &x, &mut whole);
    for (start, want) in whole.iter().enumerate() {
        let cut = rows.slice_rows(in_dim, start, 1);
        assert_eq!(cut.rows(in_dim), 1);
        let mut one = vec![0.0f32; 1];
        FusedQ8.project_rows(cut, &x, &mut one);
        assert_eq!(
            one[0], *want,
            "row {start} changed value when sliced out — the scales did not travel with it"
        );
    }
}

/// Symmetric int4, two codes per byte, `j` and `j + half` sharing one.
///
/// Biased by 8 so a nibble is unsigned 0..15 — the kernel's unbias is one
/// vector subtract rather than a sign-extension from four bits.
pub(super) fn quantise_q4_for_test(
    weights: &[f32],
    in_dim: usize,
    block: usize,
) -> (Vec<u8>, Vec<f32>) {
    let per_row = in_dim.div_ceil(block);
    let rows = weights.len() / in_dim;
    let mut packed = vec![0u8; weights.len() / 2];
    let mut scales = vec![0.0f32; rows * per_row];
    for r in 0..rows {
        for b in 0..per_row {
            let lo = b * block;
            let hi = (lo + block).min(in_dim);
            let src = &weights[r * in_dim + lo..r * in_dim + hi];
            let peak = src.iter().fold(0.0f32, |m, w| m.max(w.abs()));
            // 7 and not 8: symmetric, so the negative extreme is unused.
            let scale = if peak > 0.0 { peak / 7.0 } else { 1.0 };
            scales[r * per_row + b] = scale;
            let half = (hi - lo) / 2;
            let base = (r * in_dim + lo) / 2;
            for j in 0..half {
                let code = |v: f32| ((v / scale).round().clamp(-8.0, 7.0) as i32 + 8) as u8;
                packed[base + j] = code(src[j]) | (code(src[j + half]) << 4);
            }
        }
    }
    (packed, scales)
}

/// The Q4 kernel realises what the Q4 format denotes.
///
/// Against the portable definition, not the original weights: at 4.5
/// bits the quantiser's error is large enough to hide almost any kernel
/// bug inside a tolerance chosen for it.
#[test]
fn the_q4_kernel_computes_what_the_format_denotes() {
    const OUT: usize = 5;
    for in_dim in [BLOCK, BLOCK * 3, 5120] {
        let w = lcg_values(OUT * in_dim, 15);
        let (packed, scales) = quantise_q4_for_test(&w, in_dim, BLOCK);
        let x = lcg_values(in_dim, 16);
        let mut got = vec![0.0f32; OUT];
        FusedQ4.project_rows(
            WeightRows::Q4 {
                packed: &packed,
                scales: &scales,
                block: BLOCK,
            },
            &x,
            &mut got,
        );
        let per_row = in_dim.div_ceil(BLOCK);
        let bytes_per_row = in_dim / 2;
        for (o, value) in got.iter().enumerate() {
            let mut want = 0.0f32;
            for b in 0..per_row {
                let (lo, hi) = (b * BLOCK, ((b + 1) * BLOCK).min(in_dim));
                want += scales[o * per_row + b]
                    * q4_block_dot_portable(
                        &packed[o * bytes_per_row + lo / 2..o * bytes_per_row + hi / 2],
                        &x[lo..hi],
                    );
            }
            assert!(
                (value - want).abs() <= want.abs() * 1e-5 + 1e-4,
                "row {o} at in_dim {in_dim}: {value} against the format's {want}"
            );
        }
    }
}

/// Q4 rows slice with their scales, and at HALF a byte per weight.
///
/// The byte offset is `row * in_dim / 2`, not `row * in_dim`. A slice
/// that forgot the packing would hand a worker the rows of a different
/// part of the matrix — plausible numbers, wrong weights, and only when
/// the executor partitions.
#[test]
fn slicing_q4_rows_halves_the_byte_offset() {
    const OUT: usize = 8;
    let in_dim = BLOCK * 2;
    let w = lcg_values(OUT * in_dim, 17);
    let (packed, scales) = quantise_q4_for_test(&w, in_dim, BLOCK);
    let rows = WeightRows::Q4 {
        packed: &packed,
        scales: &scales,
        block: BLOCK,
    };
    assert_eq!(rows.rows(in_dim), OUT);
    let x = lcg_values(in_dim, 18);
    let mut whole = vec![0.0f32; OUT];
    FusedQ4.project_rows(rows, &x, &mut whole);
    for (start, want) in whole.iter().enumerate() {
        let mut one = vec![0.0f32; 1];
        FusedQ4.project_rows(rows.slice_rows(in_dim, start, 1), &x, &mut one);
        assert_eq!(one[0], *want, "row {start} moved when sliced out");
    }
}

/// **The comparison.** BF16 against Q8 against Q4, on the shapes a token
/// runs.
///
/// Measured through `CpuExecutor` with the SHIPPED kernels, in the same
/// binary and the same harness whose BF16 arm reproduces the model's
/// projection cost to -3.9% (`projection_cost`). A ratio from any other
/// harness would not license a claim about LARQL — CPU-2D spent a rung
/// learning that.
///
/// Time per matrix, not GB/s: half the bytes at a lower rate is still
/// less time, and a rate comparison hides exactly the thing being asked.
///
/// ```text
/// QW_Q8_BENCH=1 cargo test --release exec::cpu::tests::q8 -- --nocapture
/// ```
#[test]
fn bf16_against_q8_against_q4_on_the_real_shapes() {
    if std::env::var("QW_Q8_BENCH").is_err() {
        eprintln!("SKIP format comparison: set QW_Q8_BENCH=1");
        return;
    }
    use std::time::Instant;
    let exec = CpuExecutor::new().unwrap();
    println!(
        "\n  BF16 / Q8 / Q4 (block {BLOCK}), {} workers — TIME per matrix.\n",
        exec.workers()
    );
    println!(
        "  {:<22} {:>6} {:>9} {:>9} {:>9} {:>8} {:>8}",
        "projection", "calls", "bf16 ms", "q8 ms", "q4 ms", "q8 vs", "q4 vs"
    );

    let mut ms = [0.0f64; 3];
    let mut gb = [0.0f64; 3];
    for (name, out_dim, in_dim, calls) in super::projection_cost::COMPACT.iter().copied() {
        let f32w = lcg_values(out_dim * in_dim, 11);
        let bf: Vec<u16> = f32w.iter().map(|v| (v.to_bits() >> 16) as u16).collect();
        let (codes, q8_scales) = quantise(&f32w, in_dim, BLOCK);
        let (packed, q4_scales) = quantise_q4_for_test(&f32w, in_dim, BLOCK);
        let x = lcg_values(in_dim, 22);
        let arms = [
            WeightRows::Bf16(&bf),
            WeightRows::Q8 {
                codes: &codes,
                scales: &q8_scales,
                sums: &[],
                block: BLOCK,
            },
            WeightRows::Q4 {
                packed: &packed,
                scales: &q4_scales,
                block: BLOCK,
            },
        ];
        let iters = (3_000_000_000.0 / (out_dim * in_dim) as f64).clamp(3.0, 100.0) as usize;

        let mut sink = 0.0f32;
        let mut each = [0.0f64; 3];
        for (i, rows) in arms.iter().copied().enumerate() {
            let plan = PhysicalProjectionPlan::for_resident(rows, in_dim);
            let mut run = || sink += exec.project(plan.kernel(), rows, &x, out_dim)[0];
            run();
            let t = Instant::now();
            for _ in 0..iters {
                run();
            }
            each[i] = t.elapsed().as_secs_f64() / iters as f64;
            ms[i] += each[i] * calls as f64 * 1e3;
            gb[i] += rows.bytes() as f64 * calls as f64 / 1e9;
        }
        std::hint::black_box(sink);
        println!(
            "  {name:<22} {calls:>6} {:>9.3} {:>9.3} {:>9.3} {:>7.2}x {:>7.2}x",
            each[0] * 1e3,
            each[1] * 1e3,
            each[2] * 1e3,
            each[0] / each[1],
            each[0] / each[2],
        );
    }

    println!("  {:-<80}", "");
    let bits = |i: usize| gb[i] / gb[0] * 16.0;
    for (i, name) in ["bf16", "q8", "q4"].iter().enumerate() {
        println!(
            "  {name:<6} {:>8.2} ms/token {:>8.2} GB/token {:>7.1} GB/s stored {:>6.2} bits/w \
             {:>6.2}x",
            ms[i],
            gb[i],
            gb[i] / (ms[i] / 1e3),
            bits(i),
            ms[0] / ms[i],
        );
    }
    // The whole question of the rung, stated where a reader cannot miss
    // it: is the next representation worth making real?
    println!(
        "\n  Q8 -> Q4 buys {:.2}x on projections; a token would go {:.0} -> {:.0} ms \
         (+{:.0} ms non-projection).\n",
        ms[1] / ms[2],
        ms[1] + 23.5,
        ms[2] + 23.5,
        23.5
    );
}
