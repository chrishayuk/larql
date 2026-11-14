//! Q4 rows and the dense integer projectors.

use super::super::arithmetic::ScaleSpan;
use super::super::kernels::FusedBf16;
use super::super::projector::{CpuParallelism, DenseProjector, WeightRows};
use crate::format::vindex3::opplan::exec::quantise::SUM_BLOCK;

#[allow(unused_imports)]
use super::*;

/// One Q4 row where the activation is scaled FINER than the weights.
///
/// The nibble layout is a property of the weight block — byte `j` carries
/// element `j` low and `j + block/2` high — so this walks weight blocks
/// to unpack and activation sub-blocks to scale.
///
/// A sub-block never straddles the two nibble runs: `ablock` divides
/// `block` and is at most `block/2` whenever `per_weight > 1`, so a
/// sub-block lies wholly in the low run or wholly in the high run, and
/// `ablock` elements of a run come from exactly `ablock` bytes.
pub(in super::super) fn q4_row_subblocked(
    packed: &[u8],
    folded: &[f32],
    qx: &[i8],
    in_dim: usize,
    block: usize,
    ablock: usize,
) -> f32 {
    // **Enforced, not assumed.** A sub-block spanning both nibble runs
    // would read past the block's bytes; the caller routes `ablock ==
    // block` to `q4_row`, and an assert is what keeps that routing a
    // requirement rather than a convention someone can quietly break.
    assert!(
        ablock * 2 <= block && block.is_multiple_of(ablock),
        "q4 sub-blocking needs ablock ({ablock}) to divide block ({block}) and be at most \
         half of it; ablock == block is the whole-block kernel's case"
    );
    let mut acc = 0.0f32;
    let mut sub = 0usize;
    let mut lo = 0usize;
    while lo < in_dim {
        let hi = (lo + block).min(in_dim);
        let half = (hi - lo) / 2;
        let mut off = 0usize;
        while off < hi - lo {
            let want = ablock.min(hi - lo - off);
            // Which nibble run this sub-block lives in, and where its
            // bytes start within the weight block.
            let (byte0, high) = if off < half {
                (lo / 2 + off, false)
            } else {
                (lo / 2 + off - half, true)
            };
            let mut sum = 0i32;
            for j in 0..want {
                let byte = packed[byte0 + j];
                let code = if high {
                    (byte >> 4) as i32 - Q4_BIAS
                } else {
                    (byte & NIBBLE) as i32 - Q4_BIAS
                };
                sum += code * qx[lo + off + j] as i32;
            }
            acc += folded[sub] * sum as f32;
            sub += 1;
            off += ablock;
        }
        lo = hi;
    }
    acc
}

/// One Q8 row against a quantised activation, vectorised where possible.
#[inline]
pub(in super::super) fn q8_row(
    codes: &[i8],
    scales: &[f32],
    qx: &[i8],
    in_dim: usize,
    block: usize,
) -> f32 {
    #[cfg(target_arch = "aarch64")]
    if has_dotprod() {
        // SAFETY: both guarded by the runtime feature check.
        return unsafe {
            if block == SDOT_LANES && !bit_identical_only() {
                q8_row_b16_vector_sdot(codes, scales, None, qx, in_dim)
            } else {
                q8_row_sdot(codes, scales, qx, in_dim, block)
            }
        };
    }
    q8_row_portable(codes, scales, qx, in_dim, block)
}

/// One Q4 row against a quantised activation, vectorised where possible.
#[inline]
pub(in super::super) fn q4_row(
    packed: &[u8],
    scales: &[f32],
    qx: &[i8],
    in_dim: usize,
    block: usize,
) -> f32 {
    #[cfg(target_arch = "aarch64")]
    if has_dotprod() {
        // SAFETY: guarded by the runtime feature check.
        return unsafe { q4_row_sdot(packed, scales, qx, in_dim, block) };
    }
    q4_row_portable(packed, scales, qx, in_dim, block)
}

/// **Q8 weights x Q8 activation -> i32 -> f32.**
///
/// Same 27.20 GB/token as `FusedQ8` and 118.0 GB/s against bf16's 121.7:
/// SDOT does not make Q8 fast, it makes Q8 stop being SLOW. Which is also
/// why it stops at 224.75 ms — Q8 still reads 27.2 GB, and at the memory
/// wall that is ~225 ms however good the arithmetic is.
pub struct Q8xQ8;

impl DenseProjector for Q8xQ8 {
    fn parallelism(&self) -> CpuParallelism {
        CpuParallelism::ExternalPool
    }

    fn is_weight_stationary(&self, weight: WeightRows<'_>, in_dim: usize, n: usize) -> bool {
        super::super::stationary::supports(weight, in_dim, n)
    }

    /// CPU-7C. One weight traversal, `n` positions — see
    /// [`super::super::stationary`] for the invariant and what would break it.
    fn project_rows_many(
        &self,
        weight_rows: WeightRows<'_>,
        xs: &[&[f32]],
        out: &mut [f32],
        n: usize,
    ) {
        if super::super::stationary::supports(weight_rows, xs[0].len(), n) {
            super::super::stationary::project_rows_many(weight_rows, xs, out, n);
        } else {
            super::super::projector::project_rows_looped(self, weight_rows, xs, out, n);
        }
    }

    fn project_rows(&self, weight_rows: WeightRows<'_>, x: &[f32], out: &mut [f32]) {
        self.project_rows_under(weight_rows, x, out, activation_scaling(), activation_code());
    }
}

impl Q8xQ8 {
    /// [`DenseProjector::project_rows`] under an EXPLICIT activation
    /// scaling and code rather than the process's `OnceLock`-resolved
    /// ones. The trait method is exactly this with the process values; a
    /// test reaches every arm in one process through here, without
    /// setting an environment the rest of the suite shares.
    pub(in super::super) fn project_rows_under(
        &self,
        weight_rows: WeightRows<'_>,
        x: &[f32],
        out: &mut [f32],
        scaling: ScaleSpan,
        code: ActivationCode,
    ) {
        let WeightRows::Q8 {
            codes,
            scales,
            sums,
            block,
        } = weight_rows
        else {
            panic!("the q8 x q8 kernel consumes q8 weights only");
        };
        let in_dim = x.len();
        let per_row = in_dim.div_ceil(block);
        match scaling {
            ScaleSpan::Tensor => {
                let act = quantise_activation(x);
                for (o, slot) in out.iter_mut().enumerate() {
                    let row = &codes[o * in_dim..(o + 1) * in_dim];
                    let row_scales = &scales[o * per_row..(o + 1) * per_row];
                    // One scale for the vector multiplies out ONCE per
                    // row, after the block sum.
                    *slot = act.scale * q8_row(row, row_scales, &act.codes, in_dim, block);
                }
            }
            // Q8 needs no sub-block machinery at all: folding the weight
            // scale into each ACTIVATION block's scale turns a finer
            // activation into the same loop over smaller blocks.
            ScaleSpan::Block(ablock) => {
                let per_weight = block / ablock;
                // **CPU5-K5.** No folded buffer at all: the activation's
                // scales are row-invariant and the weight scale is
                // constant within a group of four blocks, so both fold in
                // registers. Bit-identical to the buffered path.
                if ablock == SDOT_LANES && per_weight == PER_WEIGHT_B16 && !bit_identical_only() {
                    #[cfg(target_arch = "aarch64")]
                    if has_dotprod() {
                        let asym = matches!(code, ActivationCode::Asymmetric);
                        let (qx, act_scales, act_mids) = if asym {
                            let (c, s, m) = quantise_activation_asymmetric(x, ablock);
                            (c, s, Some(m))
                        } else {
                            let (c, s) = quantise_activation_blocked(x, ablock);
                            (c, s, None)
                        };
                        for (o, slot) in out.iter_mut().enumerate() {
                            let row = &codes[o * in_dim..(o + 1) * in_dim];
                            let ws = &scales[o * per_row..(o + 1) * per_row];
                            // SAFETY: guarded by the runtime feature check;
                            // every slice is cut to this row's geometry.
                            *slot = unsafe {
                                q8_row_b16_register_sdot(
                                    row,
                                    ws,
                                    &act_scales,
                                    act_mids.as_deref(),
                                    &qx,
                                    in_dim,
                                )
                            };
                        }
                        return;
                    }
                }
                match code {
                    ActivationCode::Symmetric => {
                        let (qx, act_scales) = quantise_activation_blocked(x, ablock);
                        let mut folded = Vec::with_capacity(act_scales.len());
                        for (o, slot) in out.iter_mut().enumerate() {
                            let row = &codes[o * in_dim..(o + 1) * in_dim];
                            fold_scales(
                                &scales[o * per_row..(o + 1) * per_row],
                                &act_scales,
                                per_weight,
                                &mut folded,
                            );
                            *slot = q8_row(row, &folded, &qx, in_dim, ablock);
                        }
                    }
                    ActivationCode::Asymmetric => {
                        let (qx, act_scales, act_mids) = quantise_activation_asymmetric(x, ablock);
                        let mut fs = Vec::with_capacity(act_scales.len());
                        let mut fm = Vec::with_capacity(act_mids.len());
                        for (o, slot) in out.iter_mut().enumerate() {
                            let row = &codes[o * in_dim..(o + 1) * in_dim];
                            let ws = &scales[o * per_row..(o + 1) * per_row];
                            fold_scales(ws, &act_scales, per_weight, &mut fs);
                            fold_scales(ws, &act_mids, per_weight, &mut fm);
                            // The index is used when the loader built one
                            // and skipped otherwise, so a container
                            // without it still runs — slower, same answer.
                            *slot = if sums.is_empty() {
                                q8_row_asym(row, &fs, &fm, &qx, in_dim, ablock)
                            } else {
                                let per_sum = in_dim.div_ceil(SUM_BLOCK);
                                let idx = &sums[o * per_sum..(o + 1) * per_sum];
                                q8_row_asym_with_index(row, &fs, &fm, &qx, idx, in_dim, ablock)
                            };
                        }
                    }
                }
            }
        }
    }
}

/// **Q4 weights x Q8 activation -> i32 -> f32.** The CPU-4Y frontier.
///
/// 14.40 GB/token at 106.6 GB/s — 3.12x the bf16 baseline and 1.66x
/// Q8 x Q8, because at four bits there are finally bytes to save against
/// a wall the arithmetic no longer keeps it away from.
///
/// **Lossy twice over**, and the two are not the same size. The weight
/// step is `peak / 7` against Q8's `peak / 127`; the activation step is
/// `max|x| / 127`. Which of them dominates is exactly what CPU-5's
/// arms are for, and it is not assumed here.
pub struct Q4xQ8;

impl DenseProjector for Q4xQ8 {
    fn parallelism(&self) -> CpuParallelism {
        CpuParallelism::ExternalPool
    }

    fn project_rows(&self, weight_rows: WeightRows<'_>, x: &[f32], out: &mut [f32]) {
        self.project_rows_under(weight_rows, x, out, activation_scaling());
    }
}

impl Q4xQ8 {
    /// [`DenseProjector::project_rows`] under an EXPLICIT activation
    /// scaling — see [`Q8xQ8::project_rows_under`] for why it exists.
    pub(in super::super) fn project_rows_under(
        &self,
        weight_rows: WeightRows<'_>,
        x: &[f32],
        out: &mut [f32],
        scaling: ScaleSpan,
    ) {
        let WeightRows::Q4 {
            packed,
            scales,
            block,
        } = weight_rows
        else {
            panic!("the q4 x q8 kernel consumes q4 weights only");
        };
        let in_dim = x.len();
        let per_row = in_dim.div_ceil(block);
        let bytes_per_row = in_dim / 2;
        match scaling {
            ScaleSpan::Tensor => {
                let act = quantise_activation(x);
                for (o, slot) in out.iter_mut().enumerate() {
                    let row = &packed[o * bytes_per_row..(o + 1) * bytes_per_row];
                    let row_scales = &scales[o * per_row..(o + 1) * per_row];
                    *slot = act.scale * q4_row(row, row_scales, &act.codes, in_dim, block);
                }
            }
            ScaleSpan::Block(ablock) => {
                let (qx, act_scales) = quantise_activation_blocked(x, ablock);
                let per_weight = block / ablock;
                let mut folded = Vec::with_capacity(act_scales.len());
                for (o, slot) in out.iter_mut().enumerate() {
                    let row = &packed[o * bytes_per_row..(o + 1) * bytes_per_row];
                    fold_scales(
                        &scales[o * per_row..(o + 1) * per_row],
                        &act_scales,
                        per_weight,
                        &mut folded,
                    );
                    // Q4 cannot simply walk smaller blocks the way Q8
                    // can: a byte carries element `j` and `j + block/2`,
                    // so the PACKING is tied to the weight block even
                    // when the scaling is not.
                    *slot = if per_weight == 1 {
                        q4_row(row, &folded, &qx, in_dim, block)
                    } else {
                        q4_row_subblocked(row, &folded, &qx, in_dim, block, ablock)
                    };
                }
            }
        }
    }
}

/// **bf16 weights (EXACT) x Q8 activation.** The control arm.
///
/// Never chosen for speed — it reads full-width weights and then throws
/// precision away on the activation alone. That is the point: Q4 x Q8
/// moves two things at once, and without an arm that moves ONLY the
/// activation a failure cannot be attributed. CPU-4A already cost this
/// ladder a wrong conclusion by testing one coupled lever in isolation.
///
/// **It reconstructs the activation and then defers to [`FusedBf16`]**,
/// rather than running its own dot. Two reasons, and the second is the
/// important one:
///
/// - it runs at the exact kernel's speed, so the control is affordable
///   over a whole bank rather than only over a fixture;
/// - it inherits the exact kernel's SUMMATION ORDER, so the only thing
///   separating this arm from the reference is the activation. A
///   hand-written sequential dot sat ~1e-7 away from `FusedBf16` on
///   reassociation alone, which is small against quantisation but is
///   still a second difference in an arm whose whole job is to have
///   exactly one.
pub struct Bf16xQ8;

impl DenseProjector for Bf16xQ8 {
    fn parallelism(&self) -> CpuParallelism {
        // Whatever the exact kernel wants, since that is what runs.
        FusedBf16.parallelism()
    }

    fn project_rows(&self, weight_rows: WeightRows<'_>, x: &[f32], out: &mut [f32]) {
        self.project_rows_under(weight_rows, x, out, activation_scaling(), activation_code());
    }
}

impl Bf16xQ8 {
    /// [`DenseProjector::project_rows`] under an EXPLICIT activation
    /// scaling and code — see [`Q8xQ8::project_rows_under`].
    pub(in super::super) fn project_rows_under(
        &self,
        weight_rows: WeightRows<'_>,
        x: &[f32],
        out: &mut [f32],
        scaling: ScaleSpan,
        code: ActivationCode,
    ) {
        if !matches!(weight_rows, WeightRows::Bf16(_)) {
            panic!("the bf16 x q8 control kernel consumes bf16 weights only");
        }
        // Reconstructed ONCE per call, not once per row.
        let rx: Vec<f32> = match scaling {
            ScaleSpan::Tensor => {
                let act = quantise_activation(x);
                act.codes.iter().map(|c| *c as f32 * act.scale).collect()
            }
            ScaleSpan::Block(block) => match code {
                // The control blocks on the SAME boundaries the weight
                // formats use, so A1 and A4 differ only in the weights.
                ActivationCode::Symmetric => {
                    let (qx, act_scales) = quantise_activation_blocked(x, block);
                    qx.iter()
                        .enumerate()
                        .map(|(i, c)| *c as f32 * act_scales[i / block])
                        .collect()
                }
                ActivationCode::Asymmetric => {
                    let (qx, act_scales, act_mids) = quantise_activation_asymmetric(x, block);
                    qx.iter()
                        .enumerate()
                        .map(|(i, c)| *c as f32 * act_scales[i / block] + act_mids[i / block])
                        .collect()
                }
            },
        };
        FusedBf16.project_rows(weight_rows, &rx, out);
    }
}
