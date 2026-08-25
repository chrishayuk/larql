//! CPU-4X: what is int8 weight x int8 ACTIVATION actually worth?
//!
//! CPU-3A and CPU-4A between them established that this execution family
//! is conversion-bound: Q8 left the memory-bound regime at 83.6 GB/s
//! stored against bf16's 121.7, and Q4 — half the bytes again — came back
//! 20% SLOWER because it does strictly more unpacking per weight. Fewer
//! weight bits against f32 activations has stopped paying.
//!
//! The remaining lever is the arithmetic DOMAIN. `SDOT` multiplies int8
//! by int8 into an i32 accumulator in one instruction, so the widen chain
//! that Q8 spends its time on disappears entirely — but the activation
//! has to be quantised first, and that is a cost the f32 path does not
//! pay.
//!
//! **Mechanics only.** Nothing here claims the arithmetic is accurate
//! enough to use; that is a separate and larger numerical decision than
//! weight quantisation, and it gets its own rung with its own gates. The
//! question is narrower: is this a 1.2x idea or a 3x idea?
//!
//! Three costs are timed SEPARATELY, because they have different fixes. A
//! single total would hide whether the activation quantiser or the GEMV
//! is the problem:
//!
//! ```text
//!   quantise    f32 activation -> int8 + scale
//!   gemv        int8 x int8 -> i32, via SDOT
//!   rescale     i32 -> f32, one multiply per output row
//! ```
//!
//! ```text
//! QW_SDOT=1 cargo test --release exec::cpu::tests::sdot -- --nocapture
//! ```

use super::super::executor::CpuExecutor;
use super::super::kernels::FusedQ8;
use super::super::projector::{DenseProjector, WeightRows};
use crate::format::vindex3::fixtures::lcg_values;

const BLOCK: usize = 64;

/// The go/no-go line for the projection class, in ms per token.
///
/// Below this, integer activations open a regime (roughly 4.5 tok/s or
/// better) and the quality programme is justified. Above it, SDOT is
/// another invasive numerical representation for a modest gain, and the
/// conventional CPU programme is finished at Q8's 2.66 tok/s.
const GO_THRESHOLD_MS: f64 = 200.0;

/// What the shipped Q8 x f32 path costs on the same token, from the
/// harness whose bf16 arm reproduces the model to -3.9%.
const Q8_F32_MS: f64 = 325.21;

fn quantise_weights(weights: &[f32], in_dim: usize) -> (Vec<i8>, Vec<f32>) {
    let per_row = in_dim.div_ceil(BLOCK);
    let rows = weights.len() / in_dim;
    let mut codes = vec![0i8; weights.len()];
    let mut scales = vec![0.0f32; rows * per_row];
    for r in 0..rows {
        for b in 0..per_row {
            let lo = r * in_dim + b * BLOCK;
            let hi = (lo + BLOCK).min((r + 1) * in_dim);
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

/// **Cost 1.** One activation vector to symmetric int8, one scale.
///
/// One scale for the whole vector rather than per block: the activation
/// is read once per projection and its scale multiplies out at the end,
/// so a blocked activation scale would buy accuracy the weights' own
/// blocking already provides and cost a multiply per block.
fn quantise_activation(x: &[f32], out: &mut [i8]) -> f32 {
    let peak = x.iter().fold(0.0f32, |m, v| m.max(v.abs()));
    let scale = if peak > 0.0 { peak / 127.0 } else { 1.0 };
    let inv = 1.0 / scale;
    for (dst, v) in out.iter_mut().zip(x) {
        *dst = (v * inv).round().clamp(-127.0, 127.0) as i8;
    }
    scale
}

/// **Cost 2.** One row's integer dot, block-scaled.
///
/// The whole point: no widen chain. `SDOT` consumes sixteen int8 pairs
/// and accumulates four i32 lanes per instruction, so a block of 64 is
/// four instructions rather than four loads, twelve widening steps and
/// four FMAs.
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "dotprod")]
unsafe fn sdot_row(codes: &[i8], scales: &[f32], qx: &[i8], in_dim: usize) -> f32 {
    use std::arch::aarch64::*;
    let mut acc = 0.0f32;
    for (b, scale) in scales.iter().enumerate() {
        let lo = b * BLOCK;
        let hi = (lo + BLOCK).min(in_dim);
        let mut lanes = vdupq_n_s32(0);
        let mut i = lo;
        while i + 16 <= hi {
            lanes = vdotq_s32(
                lanes,
                vld1q_s8(codes.as_ptr().add(i)),
                vld1q_s8(qx.as_ptr().add(i)),
            );
            i += 16;
        }
        let mut sum = vaddvq_s32(lanes);
        while i < hi {
            sum += *codes.get_unchecked(i) as i32 * *qx.get_unchecked(i) as i32;
            i += 1;
        }
        acc += scale * sum as f32;
    }
    acc
}

/// **CPU-4Y.** Q4 weights against a Q8 activation.
///
/// CPU-4A killed Q4 x F32 — half the bytes, 20% SLOWER, because the
/// kernel was already conversion-bound and Q4 adds a nibble split on top.
/// But SDOT then showed that integer arithmetic removes the conversion
/// tax entirely and puts Q8 back on the memory wall at 118 GB/s.
///
/// So the two levers are coupled, and CPU-4A tested one alone. Q4's 14.4
/// GB/token against a ~120 GB/s wall is a ~120 ms floor; the question is
/// only how much of that the nibble unpack gives back.
///
/// The unpack feeds SDOT directly: mask and shift a 16-byte load into two
/// int8 vectors, unbias by 8, and dot each against its half of the
/// activation. No widening, no float anywhere in the inner loop.
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "dotprod")]
unsafe fn q4_sdot_row(packed: &[u8], scales: &[f32], qx: &[i8], in_dim: usize) -> f32 {
    use std::arch::aarch64::*;
    let mask = vdupq_n_u8(0x0f);
    let bias = vdupq_n_s8(8);
    let mut acc = 0.0f32;
    for (b, scale) in scales.iter().enumerate() {
        let lo = b * BLOCK;
        let hi = (lo + BLOCK).min(in_dim);
        let half = (hi - lo) / 2;
        let base = packed.as_ptr().add(lo / 2);
        let xbase = qx.as_ptr().add(lo);
        let mut lanes = vdupq_n_s32(0);
        let mut j = 0usize;
        while j + 16 <= half {
            let raw = vld1q_u8(base.add(j));
            // Byte `j` carries element `j` low and `j + half` high, so one
            // load yields two CONTIGUOUS runs rather than 32 interleaved
            // elements — see `WeightRows::Q4`.
            let low = vsubq_s8(vreinterpretq_s8_u8(vandq_u8(raw, mask)), bias);
            let high = vsubq_s8(vreinterpretq_s8_u8(vshrq_n_u8(raw, 4)), bias);
            lanes = vdotq_s32(lanes, low, vld1q_s8(xbase.add(j)));
            lanes = vdotq_s32(lanes, high, vld1q_s8(xbase.add(j + half)));
            j += 16;
        }
        let mut sum = vaddvq_s32(lanes);
        while j < half {
            let byte = *packed.get_unchecked(lo / 2 + j);
            sum += ((byte & 0x0f) as i32 - 8) * *qx.get_unchecked(lo + j) as i32;
            sum += ((byte >> 4) as i32 - 8) * *qx.get_unchecked(lo + j + half) as i32;
            j += 1;
        }
        acc += scale * sum as f32;
    }
    acc
}

/// Portable counterpart to the aarch64 intrinsic above — same nibble
/// unpack, unbias, multiply-accumulate, just without SIMD. Exists so
/// `q4_sdot_row` is a real symbol on every architecture, not only
/// aarch64: `q8_float_against_q8_integer` below calls it unconditionally
/// (behind a runtime `QW_SDOT=1` opt-in, not a `cfg`), so the crate must
/// still *compile* it everywhere even though the benchmark itself only
/// measures something meaningful on aarch64. This is the exact scalar
/// definition `the_q4_integer_kernel_computes_what_the_format_denotes`
/// checks the intrinsic version against, given the shared name here —
/// not a stub, the real computation.
///
/// `unsafe` to match the intrinsic version's signature, so the call
/// sites' `unsafe { q4_sdot_row(...) }` stays correct (not an
/// unnecessary-unsafe lint) on every target this compiles for.
#[cfg(not(target_arch = "aarch64"))]
unsafe fn q4_sdot_row(packed: &[u8], scales: &[f32], qx: &[i8], in_dim: usize) -> f32 {
    let mut acc = 0.0f32;
    for (b, scale) in scales.iter().enumerate() {
        let lo = b * BLOCK;
        let hi = (lo + BLOCK).min(in_dim);
        let half = (hi - lo) / 2;
        let mut sum = 0i32;
        for j in 0..half {
            let byte = packed[lo / 2 + j];
            sum += ((byte & 0x0f) as i32 - 8) * qx[lo + j] as i32;
            sum += ((byte >> 4) as i32 - 8) * qx[lo + j + half] as i32;
        }
        acc += scale * sum as f32;
    }
    acc
}

/// The portable definition, and what runs where `dotprod` is absent.
fn sdot_row_portable(codes: &[i8], scales: &[f32], qx: &[i8], in_dim: usize) -> f32 {
    let mut acc = 0.0f32;
    for (b, scale) in scales.iter().enumerate() {
        let lo = b * BLOCK;
        let hi = (lo + BLOCK).min(in_dim);
        let sum: i32 = (lo..hi).map(|i| codes[i] as i32 * qx[i] as i32).sum();
        acc += scale * sum as f32;
    }
    acc
}

/// One projection's worth of integer GEMV over a row range.
fn sdot_rows(codes: &[i8], scales: &[f32], qx: &[i8], in_dim: usize, out: &mut [f32]) {
    let per_row = in_dim.div_ceil(BLOCK);
    for (o, slot) in out.iter_mut().enumerate() {
        let row = &codes[o * in_dim..(o + 1) * in_dim];
        let row_scales = &scales[o * per_row..(o + 1) * per_row];
        #[cfg(target_arch = "aarch64")]
        {
            if std::arch::is_aarch64_feature_detected!("dotprod") {
                // SAFETY: guarded by the feature check; all indexing
                // stays inside the row the caller sliced.
                *slot = unsafe { sdot_row(row, row_scales, qx, in_dim) };
                continue;
            }
        }
        *slot = sdot_row_portable(row, row_scales, qx, in_dim);
    }
}

/// The integer path agrees with the float one to what int8 activations
/// cost — and NOT more closely, which is the finding, not a defect.
///
/// Quantising the activation is a second lossy step on top of the
/// weights'. An assertion that the two agreed tightly would be asserting
/// that activation quantisation is free, which is exactly the claim this
/// rung is not making.
#[test]
fn the_integer_path_agrees_to_what_int8_activations_cost() {
    const OUT: usize = 16;
    const IN: usize = 512;
    let w = lcg_values(OUT * IN, 31);
    let (codes, scales) = quantise_weights(&w, IN);
    let x = lcg_values(IN, 32);

    let mut float_out = vec![0.0f32; OUT];
    FusedQ8.project_rows(
        WeightRows::Q8 {
            codes: &codes,
            scales: &scales,
            sums: &[],
            block: BLOCK,
        },
        &x,
        &mut float_out,
    );

    let mut qx = vec![0i8; IN];
    let a_scale = quantise_activation(&x, &mut qx);
    let mut int_out = vec![0.0f32; OUT];
    sdot_rows(&codes, &scales, &qx, IN, &mut int_out);
    for v in int_out.iter_mut() {
        *v *= a_scale;
    }

    let (mut num, mut den) = (0.0f64, 0.0f64);
    for (a, b) in int_out.iter().zip(&float_out) {
        num += (*a as f64 - *b as f64).powi(2);
        den += (*b as f64).powi(2);
    }
    let rel = (num / den.max(f64::MIN_POSITIVE)).sqrt();
    assert!(
        rel < 5e-2,
        "the integer path moved {rel:.2e}, which is more than int8 activations cost — that is a \
         kernel defect rather than a representation cost"
    );
    assert!(
        rel > 1e-9,
        "the integer path agreed EXACTLY, so it is not quantising the activation at all"
    );
}

/// The NEON `SDOT` row and the portable definition agree.
#[test]
fn the_portable_and_sdot_rows_agree() {
    const IN: usize = BLOCK * 3 + 7;
    let w = lcg_values(IN, 41);
    let (codes, scales) = quantise_weights(&w, IN);
    let x = lcg_values(IN, 42);
    let mut qx = vec![0i8; IN];
    quantise_activation(&x, &mut qx);

    let portable = sdot_row_portable(&codes, &scales, &qx, IN);
    let mut got = vec![0.0f32; 1];
    sdot_rows(&codes, &scales, &qx, IN, &mut got);
    let magnitude: f32 = codes
        .iter()
        .zip(&qx)
        .map(|(c, q)| (*c as f32 * *q as f32).abs())
        .sum::<f32>()
        * scales.iter().fold(0.0f32, |m, s| m.max(*s));
    assert!(
        (got[0] - portable).abs() <= 1e-5 * magnitude.max(1.0),
        "{} against {portable}",
        got[0]
    );
}

/// One arm, timed after a warm pass.
fn time(iters: usize, mut f: impl FnMut()) -> f64 {
    use std::time::Instant;
    f();
    let t = Instant::now();
    for _ in 0..iters {
        f();
    }
    t.elapsed().as_secs_f64() / iters as f64
}

/// Q4 x Q8 realises what the Q4 format denotes, against a portable
/// definition of the same integer arithmetic.
#[test]
fn the_q4_integer_kernel_computes_what_the_format_denotes() {
    #[cfg(target_arch = "aarch64")]
    {
        if !std::arch::is_aarch64_feature_detected!("dotprod") {
            return;
        }
        const IN: usize = BLOCK * 3;
        let w = lcg_values(IN, 51);
        let (packed, scales) = super::q8::quantise_q4_for_test(&w, IN, BLOCK);
        let x = lcg_values(IN, 52);
        let mut qx = vec![0i8; IN];
        quantise_activation(&x, &mut qx);

        // The definition: unbiased nibble times quantised activation,
        // summed per block and scaled once.
        let mut want = 0.0f32;
        for (b, scale) in scales.iter().enumerate() {
            let (lo, hi) = (b * BLOCK, (b + 1) * BLOCK);
            let half = (hi - lo) / 2;
            let mut sum = 0i32;
            for j in 0..half {
                let byte = packed[lo / 2 + j];
                sum += ((byte & 0x0f) as i32 - 8) * qx[lo + j] as i32;
                sum += ((byte >> 4) as i32 - 8) * qx[lo + j + half] as i32;
            }
            want += scale * sum as f32;
        }
        // SAFETY: `dotprod` checked above.
        let got = unsafe { q4_sdot_row(&packed, &scales, &qx, IN) };
        assert!(
            (got - want).abs() <= want.abs() * 1e-5 + 1e-4,
            "{got} against the format's {want}"
        );
    }
}

/// **The measurement.** Q8 x F32 against Q8 x Q8, three costs apart.
#[test]
fn q8_float_against_q8_integer() {
    if std::env::var("QW_SDOT").is_err() {
        eprintln!("SKIP q8_float_against_q8_integer: set QW_SDOT=1");
        return;
    }
    #[cfg(target_arch = "aarch64")]
    if !std::arch::is_aarch64_feature_detected!("dotprod") {
        println!(
            "\n  SDOT unavailable on this machine; the integer arm would be the scalar\n  \
                  fallback and the comparison would measure nothing.\n"
        );
        return;
    }
    let exec = CpuExecutor::new().unwrap();
    let workers = exec.workers();
    let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(workers)
        .build()
        .unwrap();
    println!(
        "\n  Q8 x F32 against Q8 x Q8 (SDOT), {workers} workers.\n  \
         Three costs timed apart — they have different fixes.\n"
    );
    println!(
        "  {:<22} {:>6} {:>10} {:>9} {:>9} {:>9} {:>8}",
        "projection", "calls", "q8xf32", "q8xq8", "q4xq8", "q8 gain", "q4/q8"
    );

    let (mut float_ms, mut quant_ms, mut gemv_ms, mut rescale_ms) = (0.0, 0.0, 0.0, 0.0);
    let mut q4_ms = 0.0f64;
    let mut q4_gb = 0.0f64;
    for (name, out_dim, in_dim, calls) in super::projection_cost::COMPACT.iter().copied() {
        let f32w_for_q4 = lcg_values(out_dim * in_dim, 11);
        let w = &f32w_for_q4;
        let (codes, scales) = quantise_weights(w, in_dim);
        let x = lcg_values(in_dim, 22);
        let rows = WeightRows::Q8 {
            codes: &codes,
            scales: &scales,
            sums: &[],
            block: BLOCK,
        };
        let iters = (3_000_000_000.0 / (out_dim * in_dim) as f64).clamp(3.0, 100.0) as usize;
        let f32_each = time(iters, || {
            std::hint::black_box(exec.project(&FusedQ8, rows, &x, out_dim)[0]);
        });
        let mut qx = vec![0i8; in_dim];
        let quant_each = time(iters, || {
            std::hint::black_box(quantise_activation(&x, &mut qx));
        });
        let a_scale = quantise_activation(&x, &mut qx);
        let mut out = vec![0.0f32; out_dim];
        let per = out_dim.div_ceil(workers);
        let per_row = in_dim.div_ceil(BLOCK);
        let gemv_each = time(iters, || {
            pool.install(|| {
                use rayon::prelude::*;
                out.par_chunks_mut(per).enumerate().for_each(|(i, slot)| {
                    let start = i * per;
                    sdot_rows(
                        &codes[start * in_dim..(start + slot.len()) * in_dim],
                        &scales[start * per_row..(start + slot.len()) * per_row],
                        &qx,
                        in_dim,
                        slot,
                    );
                });
            });
        });
        let rescale_each = time(iters, || {
            for v in out.iter_mut() {
                *v *= a_scale;
            }
        });

        // CPU-4Y: the same integer arithmetic over HALF the bytes.
        let (packed, q4_scales) = super::q8::quantise_q4_for_test(&f32w_for_q4, in_dim, BLOCK);
        let q4_each = time(iters, || {
            pool.install(|| {
                use rayon::prelude::*;
                out.par_chunks_mut(per).enumerate().for_each(|(i, slot)| {
                    let start = i * per;
                    let bytes_per_row = in_dim / 2;
                    for (o, cell) in slot.iter_mut().enumerate() {
                        let r = start + o;
                        // SAFETY: `dotprod` checked before this bench ran.
                        *cell = unsafe {
                            q4_sdot_row(
                                &packed[r * bytes_per_row..(r + 1) * bytes_per_row],
                                &q4_scales[r * per_row..(r + 1) * per_row],
                                &qx,
                                in_dim,
                            )
                        };
                    }
                });
            });
        });
        q4_ms += q4_each * calls as f64 * 1e3;
        q4_gb += (packed.len() + q4_scales.len() * 4) as f64 * calls as f64 / 1e9;
        std::hint::black_box(&out);

        let c = calls as f64;
        let integer = (quant_each + gemv_each + rescale_each) * c * 1e3;
        float_ms += f32_each * c * 1e3;
        quant_ms += quant_each * c * 1e3;
        gemv_ms += gemv_each * c * 1e3;
        rescale_ms += rescale_each * c * 1e3;
        println!(
            "  {name:<22} {calls:>6} {:>10.2} {:>9.2} {:>9.2} {:>9.2} {:>8.2}",
            f32_each * c * 1e3,
            gemv_each * c * 1e3,
            q4_each * c * 1e3,
            f32_each * c * 1e3 / integer,
            gemv_each / q4_each,
        );
    }

    let integer_ms = quant_ms + gemv_ms + rescale_ms;
    println!("  {:-<80}", "");
    println!(
        "  {:<22} {:>10.2} ms/token   <- control",
        "Q8 x F32", float_ms
    );
    println!(
        "  {:<22} {:>10.2} ms/token   ({:.2} quant + {:.2} gemv + {:.2} rescale)",
        "Q8 x Q8 (SDOT)", integer_ms, quant_ms, gemv_ms, rescale_ms
    );
    println!("  {:<22} {:>10.2}x", "speedup", float_ms / integer_ms);
    println!(
        "  {:<22} {:>10.2} ms/token   {:.2} GB   {:.1} GB/s   {:.2}x over Q8xQ8",
        "Q4 x Q8 (SDOT)",
        q4_ms,
        q4_gb,
        q4_gb / (q4_ms / 1e3),
        gemv_ms / q4_ms,
    );
    // Pre-registered before the run, so the number is read against a
    // scale rather than talked into one.
    println!(
        "  {:<22} {}",
        "CPU-4Y verdict",
        match q4_ms {
            m if m < 135.0 => "SPECTACULAR — essentially at the roofline",
            m if m < 155.0 => "EXCELLENT",
            m if m < 180.0 => "STRONG",
            m if m < 210.0 => "USEFUL",
            m if m < 230.0 => "MARGINAL over Q8 x Q8",
            _ => "FAILED — unpack defeated the byte reduction again",
        }
    );

    // The control has to reproduce the number this is being judged
    // against, or the comparison is between two different harnesses.
    let drift = (float_ms - Q8_F32_MS) / Q8_F32_MS * 100.0;
    println!("\n  control drift from the shipped Q8 path: {drift:+.1}%");
    println!(
        "  go/no-go: integer projection {:.0} ms against a {GO_THRESHOLD_MS:.0} ms line — {}",
        integer_ms,
        if integer_ms < GO_THRESHOLD_MS {
            "GO, activation quality work is justified"
        } else {
            "NO, another representation for a modest gain"
        }
    );
    println!();
}
