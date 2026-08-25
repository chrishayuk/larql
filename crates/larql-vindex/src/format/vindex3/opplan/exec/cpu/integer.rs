//! Integer-domain projections: int8 activations, int32 accumulators.
//!
//! Every kernel beside this one multiplies a compact weight by an **f32
//! activation**. CPU-4X measured what that costs and CPU-4Y measured what
//! removing it buys:
//!
//! ```text
//! BF16 x F32   420.84 ms/token   51.20 GB   121.7 GB/s   1.00x
//! Q8   x F32   332.97 ms         27.20 GB    83.4 GB/s   1.26x
//! Q8   x Q8    224.75 ms         27.20 GB   118.0 GB/s   1.87x
//! Q4   x Q8    135.10 ms         14.40 GB   106.6 GB/s   3.12x
//! ```
//!
//! **The regimes alternate.** bf16 is memory-bound; Q8 against f32 leaves
//! that regime and becomes conversion-bound, which is why halving the
//! bytes again to Q4 x f32 came back 20% SLOWER; integer arithmetic
//! removes the conversion and puts the format back on the memory wall,
//! at which point Q4's halved traffic pays again. Compressing weights
//! buys nothing until the arithmetic can consume them, and native
//! arithmetic buys nothing once bytes are the limit.
//!
//! ## What these kernels are FOR
//!
//! They are the numerical instrument for CPU-5's quality gate, and they
//! are the deployment path — deliberately the same code. Scoring a
//! *simulation* of Q4 x Q8 and then shipping a different kernel would
//! qualify something nobody runs.
//!
//! ## Why the activation is quantised INSIDE the kernel
//!
//! The activation is a property of the CALL, not of the weight slab, so
//! it wants to be quantised once in [`super::executor::CpuExecutor::project`]
//! and shared by every worker. It is not, yet, because
//! [`DenseProjector`] names only a weight representation and teaching
//! [`super::PhysicalProjectionPlan`] to name an ACTIVATION and an
//! ACCUMULATOR is a real change to what a physical plan means.
//!
//! Re-quantising per worker is therefore waste, and it is bounded waste:
//! `in_dim` operations against a slab of `out_dim/workers * in_dim`, and
//! CPU-4X priced the whole activation quantiser at **0.33% of the integer
//! path**. It is also numerically FREE — every worker sees the same full
//! `x` and so derives the same scale and the same codes, which is why the
//! row partition cannot change an answer.
//!
//! When the plan learns to say `Q4 x Q8 -> I32 -> F32`, the quantiser
//! moves up to the executor and these kernels take a
//! [`QuantisedActivation`] instead of making one.

use super::arithmetic::ScaleSpan;
use super::kernels::FusedBf16;
use super::projector::{CpuParallelism, DenseProjector, WeightRows};
use crate::format::vindex3::opplan::exec::quantise::SUM_BLOCK;

/// The largest magnitude an int8 activation code may represent.
///
/// 127 and not 128, symmetric, for the same reason the weight quantisers
/// use it: the negative extreme would give one direction a level the
/// other lacks.
const ACT_MAX: f32 = 127.0;

/// Elements per `SDOT` instruction: sixteen int8 pairs into four i32
/// lanes.
///
/// Gated: off aarch64 the portable definitions run and nothing consumes
/// this, which `-D warnings` treats as an error on the CI targets — the
/// exact way cfg-gated code has broken this build before.
#[cfg(target_arch = "aarch64")]
const SDOT_LANES: usize = 16;

/// The int4 bias that makes a signed code an unsigned nibble.
const Q4_BIAS: i32 = 8;

/// Low nibble mask.
const NIBBLE: u8 = 0x0f;

/// One activation vector as symmetric int8, plus the scale that restores
/// it.
///
/// **One scale for the whole vector**, not one per block. The activation
/// is read once per projection and its scale multiplies out at the very
/// end, so a blocked activation scale would buy accuracy the weights'
/// own blocking already provides and cost a multiply per block.
pub struct QuantisedActivation {
    pub codes: Vec<i8>,
    pub scale: f32,
}

/// `scale = max|x| / 127`, `code = round(x / scale)`.
///
/// A zero vector would divide by zero; 1.0 keeps its codes at zero and
/// the vector reconstructs exactly.
pub fn quantise_activation(x: &[f32]) -> QuantisedActivation {
    let peak = x.iter().fold(0.0f32, |m, v| m.max(v.abs()));
    let scale = if peak > 0.0 { peak / ACT_MAX } else { 1.0 };
    let inv = 1.0 / scale;
    let codes = x
        .iter()
        .map(|v| (v * inv).round().clamp(-ACT_MAX, ACT_MAX) as i8)
        .collect();
    QuantisedActivation { codes, scale }
}

/// The same rule, once per `block` elements along the input axis.
///
/// Blocks never straddle the vector's end: a short final block takes its
/// own peak rather than borrowing a neighbour's scale.
pub fn quantise_activation_blocked(x: &[f32], block: usize) -> (Vec<i8>, Vec<f32>) {
    let blocks = x.len().div_ceil(block);
    let mut codes = vec![0i8; x.len()];
    let mut scales = vec![0.0f32; blocks];
    for (b, (scale_slot, chunk)) in scales.iter_mut().zip(x.chunks(block)).enumerate() {
        let peak = chunk.iter().fold(0.0f32, |m, v| m.max(v.abs()));
        let scale = if peak > 0.0 { peak / ACT_MAX } else { 1.0 };
        *scale_slot = scale;
        let inv = 1.0 / scale;
        for (i, v) in chunk.iter().enumerate() {
            codes[b * block + i] = (v * inv).round().clamp(-ACT_MAX, ACT_MAX) as i8;
        }
    }
    (codes, scales)
}

/// Names the activation's scale block, independently of the weight's.
pub const ACT_BLOCK_ENV: &str = "LARQL_CPU_ACT_BLOCK";

/// The smallest activation block one `SDOT` can fill.
const SDOT_MIN: usize = 16;

/// **How many elements share one activation scale.**
///
/// Independent of the weight block, and cheap in a way the weight block
/// is not. A weight scale is paid once per block PER ROW, so halving the
/// weight block costs half a bit on every weight in the model. The
/// activation is ONE VECTOR: at `in_dim` 5120 its scales are 80 floats at
/// block 64 and 320 at block 16 — 320 B against 1.3 KB, set beside 14.4
/// GB of weights per token. Under a millionth of the traffic either way.
///
/// So the only real cost of a finer activation block is arithmetic: one
/// extra float multiply-add per sub-block against `block` integer MACs,
/// and `SDOT` itself is untouched.
///
/// **This asymmetry is the whole reason the activation is worth fixing
/// before the weights are.** CPU-5 measured a blocked-Q8[64] activation
/// against EXACT weights at KL 0.00061 bits/token, 3.8x the entire
/// accepted cost of Q8 weight quantisation — so the activation, not the
/// weight format, is what a Q4 x Q8 plan is spending its budget on.
///
/// Must DIVIDE the weight block and fill at least one `SDOT`. A value
/// that does neither is refused rather than rounded, because a
/// mismatched geometry pairs weights with another block's scale and
/// still returns finite, plausible numbers.
pub fn activation_block() -> usize {
    static BLOCK: std::sync::OnceLock<usize> = std::sync::OnceLock::new();
    *BLOCK.get_or_init(|| {
        let weight_block = crate::format::vindex3::opplan::exec::quantise::Q8_BLOCK;
        let want = std::env::var(ACT_BLOCK_ENV)
            .ok()
            .and_then(|v| v.trim().parse::<usize>().ok())
            .unwrap_or(weight_block);
        assert!(
            want >= SDOT_MIN && want <= weight_block && weight_block.is_multiple_of(want),
            "{ACT_BLOCK_ENV}={want} must divide the weight block {weight_block} and be at \
             least {SDOT_MIN}"
        );
        want
    })
}

/// Whether the activation code carries a per-block OFFSET as well as a
/// scale.
///
/// Symmetric coding centres every block on zero and spends half its range
/// on whichever sign the block does not use. Measured on real residual
/// blocks, the step a per-block offset would save is
/// `2 * peak / (max - min)` — 1.0 for a balanced block, 2.0 for a
/// one-sided one:
///
/// ```text
/// layer   blk16 gain   frac of blocks > 1.2x
///   000      1.175           34.4%
///   016      1.307           63.1%
///   024      1.360           68.8%
/// ```
///
/// KL goes as the SQUARE of the step, so ~1.3x in step is ~1.7x in
/// logit KL — against the 1.6% that `Q8 x Q8[16]` missed G1 by.
///
/// **It costs arithmetic and no traffic.** Reconstructing
/// `x = c * scale + mid` turns the dot into
/// `scale * SUM(w*c) + mid * SUM(w)`, and the weight codes are already
/// in registers for the `SDOT`, so the second term is one more reduction
/// over data that has been loaded either way. On a path already at the
/// memory wall (121.0 GB/s) that is close to free.
#[derive(Clone, Copy, PartialEq, Eq, Debug, Default)]
pub enum ActivationCode {
    /// Codes centred on zero; one scale per block.
    #[default]
    Symmetric,
    /// One scale AND one offset per block.
    Asymmetric,
}

/// Opts into the CPU5-K1 weight-code index. Off by default: it was
/// measured SLOWER than recomputing the sums.
pub const WEIGHT_INDEX_ENV: &str = "LARQL_CPU_WEIGHT_INDEX";

/// Whether to build and consume the weight-code index. See
/// [`WEIGHT_INDEX_ENV`].
pub fn weight_index_enabled() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| {
        matches!(
            std::env::var(WEIGHT_INDEX_ENV)
                .ok()
                .as_deref()
                .map(str::trim),
            Some("1") | Some("true")
        )
    })
}

/// Names the activation code. `symmetric` (default) or `asymmetric`.
pub const ACT_CODE_ENV: &str = "LARQL_CPU_ACT_CODE";

/// The activation code, resolved once per process.
pub fn activation_code() -> ActivationCode {
    static CODE: std::sync::OnceLock<ActivationCode> = std::sync::OnceLock::new();
    *CODE.get_or_init(
        || match std::env::var(ACT_CODE_ENV).ok().as_deref().map(str::trim) {
            Some("asymmetric") => ActivationCode::Asymmetric,
            _ => ActivationCode::Symmetric,
        },
    )
}

/// One activation vector as asymmetric int8: per-block scale AND offset.
///
/// `mid = (max + min) / 2`, `scale = (max - min) / 255`, and
/// `code = round((x - mid) / scale)` lands in `-128..=127` by
/// construction. A constant block has `max == min`; its sentinel scale
/// keeps every code at zero and the block reconstructs EXACTLY from the
/// offset alone, which a symmetric code cannot do.
pub fn quantise_activation_asymmetric(x: &[f32], block: usize) -> (Vec<i8>, Vec<f32>, Vec<f32>) {
    let blocks = x.len().div_ceil(block);
    let mut codes = vec![0i8; x.len()];
    let mut scales = vec![0.0f32; blocks];
    let mut mids = vec![0.0f32; blocks];
    for (b, chunk) in x.chunks(block).enumerate() {
        let (mut lo, mut hi) = (f32::INFINITY, f32::NEG_INFINITY);
        for v in chunk {
            lo = lo.min(*v);
            hi = hi.max(*v);
        }
        let mid = 0.5 * (hi + lo);
        let span = hi - lo;
        let scale = if span > 0.0 { span / ASYM_LEVELS } else { 1.0 };
        scales[b] = scale;
        mids[b] = mid;
        let inv = 1.0 / scale;
        for (i, v) in chunk.iter().enumerate() {
            codes[b * block + i] = ((*v - mid) * inv).round().clamp(-128.0, 127.0) as i8;
        }
    }
    (codes, scales, mids)
}

/// Levels an asymmetric int8 code spans: `-128..=127` is 255 steps.
const ASYM_LEVELS: f32 = 255.0;

/// **The DEFINITION** of an asymmetric Q8 x Q8 row.
///
/// `x = c * scale + mid` per block, so the row is
/// `SUM_b [ scale_b * SUM(w*c) + mid_b * SUM(w) ]`, with both scales
/// pre-multiplied by the weight's own block scale.
pub(super) fn q8_row_asym_portable(
    codes: &[i8],
    fold_scale: &[f32],
    fold_mid: &[f32],
    qx: &[i8],
    in_dim: usize,
    block: usize,
) -> f32 {
    let mut acc = 0.0f32;
    for (b, (s, m)) in fold_scale.iter().zip(fold_mid).enumerate() {
        let lo = b * block;
        let hi = (lo + block).min(in_dim);
        if lo >= hi {
            break;
        }
        let mut dot = 0i32;
        let mut wsum = 0i32;
        for i in lo..hi {
            dot += codes[i] as i32 * qx[i] as i32;
            wsum += codes[i] as i32;
        }
        acc += s * dot as f32 + m * wsum as f32;
    }
    acc
}

/// The same through `SDOT`. The weight sum is one more dot, against a
/// vector of ones — the codes are already loaded, so it buys the offset
/// term without touching memory again.
///
/// # Safety
/// Requires `dotprod`, checked by [`has_dotprod`].
#[cfg(target_arch = "aarch64")]
/// `dotprod` intrinsics are stable since 1.98; see the note on
/// [`q8_row_sdot`].
#[allow(clippy::incompatible_msrv)]
#[target_feature(enable = "dotprod")]
unsafe fn q8_row_asym_sdot(
    codes: &[i8],
    fold_scale: &[f32],
    fold_mid: &[f32],
    qx: &[i8],
    in_dim: usize,
    block: usize,
) -> f32 {
    use std::arch::aarch64::*;
    let ones = vdupq_n_s8(1);
    let mut acc = 0.0f32;
    for (b, (s, m)) in fold_scale.iter().zip(fold_mid).enumerate() {
        let lo = b * block;
        let hi = (lo + block).min(in_dim);
        if lo >= hi {
            break;
        }
        let mut dot_lanes = vdupq_n_s32(0);
        let mut sum_lanes = vdupq_n_s32(0);
        let mut i = lo;
        while i + SDOT_LANES <= hi {
            let w = vld1q_s8(codes.as_ptr().add(i));
            dot_lanes = vdotq_s32(dot_lanes, w, vld1q_s8(qx.as_ptr().add(i)));
            sum_lanes = vdotq_s32(sum_lanes, w, ones);
            i += SDOT_LANES;
        }
        let mut dot = vaddvq_s32(dot_lanes);
        let mut wsum = vaddvq_s32(sum_lanes);
        while i < hi {
            let w = *codes.get_unchecked(i) as i32;
            dot += w * *qx.get_unchecked(i) as i32;
            wsum += w;
            i += 1;
        }
        acc += s * dot as f32 + m * wsum as f32;
    }
    acc
}

/// The indexed asymmetric row: K4's vector index load at block 16, K1's
/// scalar-index row otherwise.
#[inline]
fn q8_row_asym_with_index(
    codes: &[i8],
    fold_scale: &[f32],
    fold_mid: &[f32],
    qx: &[i8],
    sums: &[i16],
    in_dim: usize,
    block: usize,
) -> f32 {
    #[cfg(target_arch = "aarch64")]
    if has_dotprod() && block == SDOT_LANES && !bit_identical_only() {
        // SAFETY: guarded by the runtime feature check.
        return unsafe { q8_row_b16_indexed_sdot(codes, fold_scale, fold_mid, qx, sums, in_dim) };
    }
    q8_row_asym_indexed(codes, fold_scale, fold_mid, qx, sums, in_dim, block)
}

/// **CPU5-K1.** The same row, with the weight sums READ rather than
/// recomputed.
///
/// `SUM(q)` depends only on the weight block, so recomputing it every
/// token costs a second `SDOT` and a second integer reduction per block.
/// The index costs one bit per weight (`i16` per 16 codes, exact because
/// `16 * 127 = 2032`), i.e. ~12% more compact traffic.
///
/// **Bit-identical to [`q8_row_asym`] by construction**: an i32 sum of
/// i16 sub-sums taken in order is the same integer the reduction would
/// have produced, and no float operation changes.
///
/// Written as a whole ROW rather than as a per-block helper on purpose.
/// A first version called a `q8_block_dot(&codes[lo..hi], &qx[lo..hi])`
/// per block and measured 1105 ms against the 757 ms it was meant to
/// beat — the slicing, bounds checks and call boundary per block cost
/// more than the `SDOT` it removed. At 320 blocks a row, per-block
/// abstraction is the thing being optimised away.
pub(super) fn q8_row_asym_indexed(
    codes: &[i8],
    fold_scale: &[f32],
    fold_mid: &[f32],
    qx: &[i8],
    sums: &[i16],
    in_dim: usize,
    block: usize,
) -> f32 {
    #[cfg(target_arch = "aarch64")]
    if has_dotprod() {
        // SAFETY: guarded by the runtime feature check.
        return unsafe {
            q8_row_asym_indexed_sdot(codes, fold_scale, fold_mid, qx, sums, in_dim, block)
        };
    }
    q8_row_asym_indexed_portable(codes, fold_scale, fold_mid, qx, sums, in_dim, block)
}

/// The portable definition of the indexed row.
fn q8_row_asym_indexed_portable(
    codes: &[i8],
    fold_scale: &[f32],
    fold_mid: &[f32],
    qx: &[i8],
    sums: &[i16],
    in_dim: usize,
    block: usize,
) -> f32 {
    let per_block = block / SUM_BLOCK;
    let mut acc = 0.0f32;
    for (b, (s, m)) in fold_scale.iter().zip(fold_mid).enumerate() {
        let lo = b * block;
        let hi = (lo + block).min(in_dim);
        if lo >= hi {
            break;
        }
        let mut dot = 0i32;
        for i in lo..hi {
            dot += codes[i] as i32 * qx[i] as i32;
        }
        let mut wsum = 0i32;
        for k in 0..per_block {
            wsum += sums[b * per_block + k] as i32;
        }
        acc += s * dot as f32 + m * wsum as f32;
    }
    acc
}

/// # Safety
/// Requires `dotprod`, checked by [`has_dotprod`].
#[cfg(target_arch = "aarch64")]
/// `dotprod` intrinsics are stable since 1.98; see [`q8_row_sdot`].
#[allow(clippy::incompatible_msrv)]
#[target_feature(enable = "dotprod")]
unsafe fn q8_row_asym_indexed_sdot(
    codes: &[i8],
    fold_scale: &[f32],
    fold_mid: &[f32],
    qx: &[i8],
    sums: &[i16],
    in_dim: usize,
    block: usize,
) -> f32 {
    use std::arch::aarch64::*;
    let per_block = block / SUM_BLOCK;
    let mut acc = 0.0f32;
    for (b, (s, m)) in fold_scale.iter().zip(fold_mid).enumerate() {
        let lo = b * block;
        let hi = (lo + block).min(in_dim);
        if lo >= hi {
            break;
        }
        let mut lanes = vdupq_n_s32(0);
        let mut i = lo;
        while i + SDOT_LANES <= hi {
            lanes = vdotq_s32(
                lanes,
                vld1q_s8(codes.as_ptr().add(i)),
                vld1q_s8(qx.as_ptr().add(i)),
            );
            i += SDOT_LANES;
        }
        let mut dot = vaddvq_s32(lanes);
        while i < hi {
            dot += *codes.get_unchecked(i) as i32 * *qx.get_unchecked(i) as i32;
            i += 1;
        }
        let mut wsum = 0i32;
        for k in 0..per_block {
            wsum += *sums.get_unchecked(b * per_block + k) as i32;
        }
        acc += s * dot as f32 + m * wsum as f32;
    }
    acc
}

/// **CPU5-K2.** Four block-16 reductions, batched.
///
/// At `block == 16` there is exactly ONE `SDOT` per block, so the
/// cross-lane `vaddvq_s32` after it has no independent work to hide its
/// latency behind. That is why cost is SUPERLINEAR in block count —
/// 80 blocks 266 ms, 160 blocks 298 ms, 320 blocks 484 ms — an
/// instruction-level-parallelism collapse rather than extra arithmetic.
///
/// Four blocks reduce together with three pairwise adds instead of four
/// cross-lane reductions:
///
/// ```text
/// vpaddq(d0,d1) -> [d0a+d0b, d0c+d0d, d1a+d1b, d1c+d1d]
/// vpaddq(d2,d3) -> likewise
/// vpaddq(  ,  ) -> [SUM d0, SUM d1, SUM d2, SUM d3]
/// ```
///
/// **Bit-identical**: every rearrangement is in i32, where these sums are
/// exact, and the four float multiply-adds still happen in block order.
///
/// # Safety
/// Requires `dotprod`, checked by [`has_dotprod`].
#[cfg(target_arch = "aarch64")]
#[allow(clippy::incompatible_msrv)]
#[target_feature(enable = "dotprod")]
unsafe fn q8_row_asym_b16_sdot(
    codes: &[i8],
    fold_scale: &[f32],
    fold_mid: &[f32],
    qx: &[i8],
    in_dim: usize,
) -> f32 {
    use std::arch::aarch64::*;
    let ones = vdupq_n_s8(1);
    let blocks = in_dim / SDOT_LANES;
    let mut acc = 0.0f32;
    let mut b = 0usize;
    while b + 4 <= blocks {
        let i0 = b * SDOT_LANES;
        let w0 = vld1q_s8(codes.as_ptr().add(i0));
        let w1 = vld1q_s8(codes.as_ptr().add(i0 + SDOT_LANES));
        let w2 = vld1q_s8(codes.as_ptr().add(i0 + 2 * SDOT_LANES));
        let w3 = vld1q_s8(codes.as_ptr().add(i0 + 3 * SDOT_LANES));
        let z = vdupq_n_s32(0);
        let d0 = vdotq_s32(z, w0, vld1q_s8(qx.as_ptr().add(i0)));
        let d1 = vdotq_s32(z, w1, vld1q_s8(qx.as_ptr().add(i0 + SDOT_LANES)));
        let d2 = vdotq_s32(z, w2, vld1q_s8(qx.as_ptr().add(i0 + 2 * SDOT_LANES)));
        let d3 = vdotq_s32(z, w3, vld1q_s8(qx.as_ptr().add(i0 + 3 * SDOT_LANES)));
        let s0 = vdotq_s32(z, w0, ones);
        let s1 = vdotq_s32(z, w1, ones);
        let s2 = vdotq_s32(z, w2, ones);
        let s3 = vdotq_s32(z, w3, ones);
        let dv = vpaddq_s32(vpaddq_s32(d0, d1), vpaddq_s32(d2, d3));
        let sv = vpaddq_s32(vpaddq_s32(s0, s1), vpaddq_s32(s2, s3));
        acc += fold_scale[b] * vgetq_lane_s32(dv, 0) as f32
            + fold_mid[b] * vgetq_lane_s32(sv, 0) as f32;
        acc += fold_scale[b + 1] * vgetq_lane_s32(dv, 1) as f32
            + fold_mid[b + 1] * vgetq_lane_s32(sv, 1) as f32;
        acc += fold_scale[b + 2] * vgetq_lane_s32(dv, 2) as f32
            + fold_mid[b + 2] * vgetq_lane_s32(sv, 2) as f32;
        acc += fold_scale[b + 3] * vgetq_lane_s32(dv, 3) as f32
            + fold_mid[b + 3] * vgetq_lane_s32(sv, 3) as f32;
        b += 4;
    }
    while b < blocks {
        let lo = b * SDOT_LANES;
        let w = vld1q_s8(codes.as_ptr().add(lo));
        let d = vaddvq_s32(vdotq_s32(vdupq_n_s32(0), w, vld1q_s8(qx.as_ptr().add(lo))));
        let sm = vaddvq_s32(vdotq_s32(vdupq_n_s32(0), w, ones));
        acc += fold_scale[b] * d as f32 + fold_mid[b] * sm as f32;
        b += 1;
    }
    let done = blocks * SDOT_LANES;
    if done < in_dim {
        let (mut d, mut sm) = (0i32, 0i32);
        for i in done..in_dim {
            let w = *codes.get_unchecked(i) as i32;
            d += w * *qx.get_unchecked(i) as i32;
            sm += w;
        }
        acc += fold_scale[blocks] * d as f32 + fold_mid[blocks] * sm as f32;
    }
    acc
}

/// **CPU5-K3.** Block-16 rows accumulated in the VECTOR domain.
///
/// K2 batched the integer reductions but still crossed from vector to
/// scalar registers twice per block — eight `vgetq_lane_s32` and eight
/// scalar float operations per group of four blocks. At 320 blocks a row
/// that crossing is what remains of the block-16 pathology.
///
/// Here the packed i32 block sums are converted in place, multiplied by
/// vectors of scales (and offsets), and accumulated into a four-lane
/// float accumulator — **one horizontal reduction per ROW** instead of
/// per block.
///
/// **NOT bit-identical**, by design and uniquely on this ladder. K1 and
/// K2 preserved the arithmetic exactly; this reassociates the sum of
/// already-computed block contributions, so the numbers move at the
/// rounding level and the frozen quality gates must be re-established on
/// the full bank rather than inherited.
///
/// # Safety
/// Requires `dotprod`, checked by [`has_dotprod`].
#[cfg(target_arch = "aarch64")]
#[allow(clippy::incompatible_msrv)]
#[target_feature(enable = "dotprod")]
unsafe fn q8_row_b16_vector_sdot(
    codes: &[i8],
    fold_scale: &[f32],
    fold_mid: Option<&[f32]>,
    qx: &[i8],
    in_dim: usize,
) -> f32 {
    use std::arch::aarch64::*;
    let ones = vdupq_n_s8(1);
    let blocks = in_dim / SDOT_LANES;
    let mut acc_v = vdupq_n_f32(0.0);
    let mut b = 0usize;
    while b + 4 <= blocks {
        let i0 = b * SDOT_LANES;
        let z = vdupq_n_s32(0);
        let w0 = vld1q_s8(codes.as_ptr().add(i0));
        let w1 = vld1q_s8(codes.as_ptr().add(i0 + SDOT_LANES));
        let w2 = vld1q_s8(codes.as_ptr().add(i0 + 2 * SDOT_LANES));
        let w3 = vld1q_s8(codes.as_ptr().add(i0 + 3 * SDOT_LANES));
        let d0 = vdotq_s32(z, w0, vld1q_s8(qx.as_ptr().add(i0)));
        let d1 = vdotq_s32(z, w1, vld1q_s8(qx.as_ptr().add(i0 + SDOT_LANES)));
        let d2 = vdotq_s32(z, w2, vld1q_s8(qx.as_ptr().add(i0 + 2 * SDOT_LANES)));
        let d3 = vdotq_s32(z, w3, vld1q_s8(qx.as_ptr().add(i0 + 3 * SDOT_LANES)));
        let dv = vpaddq_s32(vpaddq_s32(d0, d1), vpaddq_s32(d2, d3));
        // Stay in the vector domain: no lane extract, no scalar float.
        acc_v = vfmaq_f32(
            acc_v,
            vld1q_f32(fold_scale.as_ptr().add(b)),
            vcvtq_f32_s32(dv),
        );
        if let Some(mid) = fold_mid {
            let s0 = vdotq_s32(z, w0, ones);
            let s1 = vdotq_s32(z, w1, ones);
            let s2 = vdotq_s32(z, w2, ones);
            let s3 = vdotq_s32(z, w3, ones);
            let sv = vpaddq_s32(vpaddq_s32(s0, s1), vpaddq_s32(s2, s3));
            acc_v = vfmaq_f32(acc_v, vld1q_f32(mid.as_ptr().add(b)), vcvtq_f32_s32(sv));
        }
        b += 4;
    }
    let mut acc = vaddvq_f32(acc_v);
    // Whole blocks below a group of four, then any ragged remainder.
    while b < blocks {
        let lo = b * SDOT_LANES;
        let w = vld1q_s8(codes.as_ptr().add(lo));
        let d = vaddvq_s32(vdotq_s32(vdupq_n_s32(0), w, vld1q_s8(qx.as_ptr().add(lo))));
        acc += fold_scale[b] * d as f32;
        if let Some(mid) = fold_mid {
            acc += mid[b] * vaddvq_s32(vdotq_s32(vdupq_n_s32(0), w, ones)) as f32;
        }
        b += 1;
    }
    let done = blocks * SDOT_LANES;
    if done < in_dim {
        let (mut d, mut sm) = (0i32, 0i32);
        for i in done..in_dim {
            let w = *codes.get_unchecked(i) as i32;
            d += w * *qx.get_unchecked(i) as i32;
            sm += w;
        }
        acc += fold_scale[blocks] * d as f32;
        if let Some(mid) = fold_mid {
            acc += mid[blocks] * sm as f32;
        }
    }
    acc
}

/// The bit-identical asymmetric row, whatever the process arm.
///
/// Tests need BOTH implementations reachable in one binary: K3
/// reassociates, and a control taken across two builds would be arguing
/// with a compiler as much as with the change.
#[cfg(test)]
pub(super) fn q8_row_asym_exact(
    codes: &[i8],
    fold_scale: &[f32],
    fold_mid: &[f32],
    qx: &[i8],
    in_dim: usize,
    block: usize,
) -> f32 {
    #[cfg(target_arch = "aarch64")]
    if has_dotprod() {
        // SAFETY: guarded by the runtime feature check.
        return unsafe {
            if block == SDOT_LANES {
                q8_row_asym_b16_sdot(codes, fold_scale, fold_mid, qx, in_dim)
            } else {
                q8_row_asym_sdot(codes, fold_scale, fold_mid, qx, in_dim, block)
            }
        };
    }
    q8_row_asym_portable(codes, fold_scale, fold_mid, qx, in_dim, block)
}

/// The K3 row, reachable from a test whatever the process arm.
#[cfg(test)]
pub(super) fn q8_row_asym_k3(
    codes: &[i8],
    fold_scale: &[f32],
    fold_mid: &[f32],
    qx: &[i8],
    in_dim: usize,
) -> f32 {
    #[cfg(target_arch = "aarch64")]
    if has_dotprod() {
        // SAFETY: guarded by the runtime feature check.
        return unsafe { q8_row_b16_vector_sdot(codes, fold_scale, Some(fold_mid), qx, in_dim) };
    }
    q8_row_asym_portable(codes, fold_scale, fold_mid, qx, in_dim, SDOT_LANES)
}

/// **CPU5-K4.** K3's vector accumulation, with the weight sums LOADED
/// as a vector instead of recomputed.
///
/// K1 removed the same four correction `SDOT`s and LOST 111 ms, because
/// it fetched the index as 320 scalar `i16` reads a row — a third stream
/// of tiny dependent loads. Under K3's four-block geometry the same four
/// sums are one 64-bit load and one widen:
///
/// ```text
/// K3:  4 useful SDOT + 4 correction SDOT + 6 vpaddq + 2 cvt + 2 vfma
/// K4:  4 useful SDOT               + 3 vpaddq + 2 cvt + 2 vfma
///                                  + 1 vld1_s16 + 1 vmovl_s16
/// ```
///
/// **Bit-identical to K3**, and that is the point rather than a bonus:
/// the index holds exactly the integers the correction `SDOT`s produce,
/// so `vcvtq_f32_s32` sees the same lanes and every float operation is
/// unchanged. One Bank-1 run therefore covers both.
///
/// Requires `sums` blocked on [`SUM_BLOCK`] with the activation blocked
/// identically, so a group of four activation blocks is four consecutive
/// sums.
///
/// # Safety
/// Requires `dotprod`, checked by [`has_dotprod`].
#[cfg(target_arch = "aarch64")]
#[allow(clippy::incompatible_msrv)]
#[target_feature(enable = "dotprod")]
unsafe fn q8_row_b16_indexed_sdot(
    codes: &[i8],
    fold_scale: &[f32],
    fold_mid: &[f32],
    qx: &[i8],
    sums: &[i16],
    in_dim: usize,
) -> f32 {
    use std::arch::aarch64::*;
    let blocks = in_dim / SDOT_LANES;
    let mut acc_v = vdupq_n_f32(0.0);
    let mut b = 0usize;
    while b + 4 <= blocks {
        let i0 = b * SDOT_LANES;
        let z = vdupq_n_s32(0);
        let d0 = vdotq_s32(
            z,
            vld1q_s8(codes.as_ptr().add(i0)),
            vld1q_s8(qx.as_ptr().add(i0)),
        );
        let d1 = vdotq_s32(
            z,
            vld1q_s8(codes.as_ptr().add(i0 + SDOT_LANES)),
            vld1q_s8(qx.as_ptr().add(i0 + SDOT_LANES)),
        );
        let d2 = vdotq_s32(
            z,
            vld1q_s8(codes.as_ptr().add(i0 + 2 * SDOT_LANES)),
            vld1q_s8(qx.as_ptr().add(i0 + 2 * SDOT_LANES)),
        );
        let d3 = vdotq_s32(
            z,
            vld1q_s8(codes.as_ptr().add(i0 + 3 * SDOT_LANES)),
            vld1q_s8(qx.as_ptr().add(i0 + 3 * SDOT_LANES)),
        );
        let dv = vpaddq_s32(vpaddq_s32(d0, d1), vpaddq_s32(d2, d3));
        acc_v = vfmaq_f32(
            acc_v,
            vld1q_f32(fold_scale.as_ptr().add(b)),
            vcvtq_f32_s32(dv),
        );
        // The four correction SDOTs, replaced by one 64-bit load.
        let sv = vmovl_s16(vld1_s16(sums.as_ptr().add(b)));
        acc_v = vfmaq_f32(
            acc_v,
            vld1q_f32(fold_mid.as_ptr().add(b)),
            vcvtq_f32_s32(sv),
        );
        b += 4;
    }
    let mut acc = vaddvq_f32(acc_v);
    // **The tail must associate exactly as K3's does.** K3 adds the
    // scale term and the offset term in two separate accumulations;
    // folding them into one `acc += A + B` here is a different rounding,
    // and the bit-identity gate caught precisely that on a shape with a
    // ragged group of four.
    while b < blocks {
        let lo = b * SDOT_LANES;
        let w = vld1q_s8(codes.as_ptr().add(lo));
        let d = vaddvq_s32(vdotq_s32(vdupq_n_s32(0), w, vld1q_s8(qx.as_ptr().add(lo))));
        acc += fold_scale[b] * d as f32;
        acc += fold_mid[b] * *sums.get_unchecked(b) as f32;
        b += 1;
    }
    let done = blocks * SDOT_LANES;
    if done < in_dim {
        let (mut d, mut sm) = (0i32, 0i32);
        for i in done..in_dim {
            let w = *codes.get_unchecked(i) as i32;
            d += w * *qx.get_unchecked(i) as i32;
            sm += w;
        }
        acc += fold_scale[blocks] * d as f32;
        acc += fold_mid[blocks] * sm as f32;
    }
    acc
}

/// The K4 row, reachable from a test whatever the process arm.
#[cfg(test)]
pub(super) fn q8_row_asym_k4(
    codes: &[i8],
    fold_scale: &[f32],
    fold_mid: &[f32],
    qx: &[i8],
    sums: &[i16],
    in_dim: usize,
) -> f32 {
    #[cfg(target_arch = "aarch64")]
    if has_dotprod() {
        // SAFETY: guarded by the runtime feature check.
        return unsafe { q8_row_b16_indexed_sdot(codes, fold_scale, fold_mid, qx, sums, in_dim) };
    }
    q8_row_asym_indexed_portable(codes, fold_scale, fold_mid, qx, sums, in_dim, SDOT_LANES)
}

/// **CPU5-K5.** K3's vector accumulation with the folded scales built in
/// REGISTERS instead of in a per-row buffer.
///
/// `fold_scales` materialised a 320-entry `f32` array per output row —
/// two of them for the asymmetric arm — and the kernel then read them
/// back. That is ~320 vector load/store operations against ~320 `SDOT`s
/// per row, roughly doubling the inner loop's op count. It is not DRAM
/// traffic (2560 B lives in L1); it is load/store port pressure, which
/// is why K4 removing vector ALU work changed nothing.
///
/// The buffers are avoidable because `ascale` and `amid` are
/// ROW-INVARIANT — they describe the activation, not the row — and
/// because at `ablock` 16 with `block` 64 a group of four activation
/// blocks is exactly one weight block, so the weight scale is a constant
/// within a group:
///
/// ```text
/// was:  fold[b] = wscale[b/4] * ascale[b], stored, reloaded
/// K5:   vmulq_n_f32(vld1q_f32(&ascale[b]), wscale[b/4])
/// ```
///
/// **Bit-identical to K3 and K4**: `ws * ascale[b]` is the same f32
/// whether it goes through memory first or not, and every subsequent
/// operation is unchanged.
///
/// # Safety
/// Requires `dotprod`, checked by [`has_dotprod`]. `wscales` must cover
/// `in_dim / (SDOT_LANES * PER_WEIGHT_B16)` blocks.
#[cfg(target_arch = "aarch64")]
#[allow(clippy::incompatible_msrv)]
#[target_feature(enable = "dotprod")]
unsafe fn q8_row_b16_register_sdot(
    codes: &[i8],
    wscales: &[f32],
    ascales: &[f32],
    amids: Option<&[f32]>,
    qx: &[i8],
    in_dim: usize,
) -> f32 {
    use std::arch::aarch64::*;
    let ones = vdupq_n_s8(1);
    let blocks = in_dim / SDOT_LANES;
    let mut acc_v = vdupq_n_f32(0.0);
    let mut b = 0usize;
    while b + PER_WEIGHT_B16 <= blocks {
        let i0 = b * SDOT_LANES;
        let z = vdupq_n_s32(0);
        let w0 = vld1q_s8(codes.as_ptr().add(i0));
        let w1 = vld1q_s8(codes.as_ptr().add(i0 + SDOT_LANES));
        let w2 = vld1q_s8(codes.as_ptr().add(i0 + 2 * SDOT_LANES));
        let w3 = vld1q_s8(codes.as_ptr().add(i0 + 3 * SDOT_LANES));
        let d0 = vdotq_s32(z, w0, vld1q_s8(qx.as_ptr().add(i0)));
        let d1 = vdotq_s32(z, w1, vld1q_s8(qx.as_ptr().add(i0 + SDOT_LANES)));
        let d2 = vdotq_s32(z, w2, vld1q_s8(qx.as_ptr().add(i0 + 2 * SDOT_LANES)));
        let d3 = vdotq_s32(z, w3, vld1q_s8(qx.as_ptr().add(i0 + 3 * SDOT_LANES)));
        let dv = vpaddq_s32(vpaddq_s32(d0, d1), vpaddq_s32(d2, d3));
        // ONE broadcast multiply where a 320-entry buffer used to be.
        let ws = *wscales.get_unchecked(b / PER_WEIGHT_B16);
        let scale_v = vmulq_n_f32(vld1q_f32(ascales.as_ptr().add(b)), ws);
        acc_v = vfmaq_f32(acc_v, scale_v, vcvtq_f32_s32(dv));
        if let Some(mid) = amids {
            let s0 = vdotq_s32(z, w0, ones);
            let s1 = vdotq_s32(z, w1, ones);
            let s2 = vdotq_s32(z, w2, ones);
            let s3 = vdotq_s32(z, w3, ones);
            let sv = vpaddq_s32(vpaddq_s32(s0, s1), vpaddq_s32(s2, s3));
            let mid_v = vmulq_n_f32(vld1q_f32(mid.as_ptr().add(b)), ws);
            acc_v = vfmaq_f32(acc_v, mid_v, vcvtq_f32_s32(sv));
        }
        b += PER_WEIGHT_B16;
    }
    let mut acc = vaddvq_f32(acc_v);
    while b < blocks {
        let lo = b * SDOT_LANES;
        let w = vld1q_s8(codes.as_ptr().add(lo));
        let ws = *wscales.get_unchecked(b / PER_WEIGHT_B16);
        let d = vaddvq_s32(vdotq_s32(vdupq_n_s32(0), w, vld1q_s8(qx.as_ptr().add(lo))));
        acc += ws * *ascales.get_unchecked(b) * d as f32;
        if let Some(mid) = amids {
            acc +=
                ws * *mid.get_unchecked(b) * vaddvq_s32(vdotq_s32(vdupq_n_s32(0), w, ones)) as f32;
        }
        b += 1;
    }
    let done = blocks * SDOT_LANES;
    if done < in_dim {
        let (mut d, mut sm) = (0i32, 0i32);
        for i in done..in_dim {
            let w = *codes.get_unchecked(i) as i32;
            d += w * *qx.get_unchecked(i) as i32;
            sm += w;
        }
        let ws = *wscales.get_unchecked(blocks / PER_WEIGHT_B16);
        acc += ws * *ascales.get_unchecked(blocks) * d as f32;
        if let Some(mid) = amids {
            acc += ws * *mid.get_unchecked(blocks) * sm as f32;
        }
    }
    acc
}

/// Activation blocks inside one weight block at `ablock` 16, `block` 64.
/// The value that makes the weight scale a constant within a group.
const PER_WEIGHT_B16: usize = 4;

/// The K5 row, reachable from a test whatever the process arm.
#[cfg(test)]
pub(super) fn q8_row_k3_register(
    codes: &[i8],
    wscales: &[f32],
    ascales: &[f32],
    amids: Option<&[f32]>,
    qx: &[i8],
    in_dim: usize,
) -> f32 {
    #[cfg(target_arch = "aarch64")]
    if has_dotprod() {
        // SAFETY: guarded by the runtime feature check.
        return unsafe { q8_row_b16_register_sdot(codes, wscales, ascales, amids, qx, in_dim) };
    }
    unimplemented!("K5 has no portable arm; the gate runs on aarch64")
}

/// The K3 SYMMETRIC row, reachable from a test whatever the process arm.
#[cfg(test)]
pub(super) fn q8_row_k3_sym(codes: &[i8], fold_scale: &[f32], qx: &[i8], in_dim: usize) -> f32 {
    #[cfg(target_arch = "aarch64")]
    if has_dotprod() {
        // SAFETY: guarded by the runtime feature check.
        return unsafe { q8_row_b16_vector_sdot(codes, fold_scale, None, qx, in_dim) };
    }
    q8_row_portable(codes, fold_scale, qx, in_dim, SDOT_LANES)
}

/// Opts OUT of CPU5-K3, back to the bit-identical K2 kernels.
///
/// Exists so the two can be compared in ONE binary: K3 reassociates, and
/// a numerical control across two builds would be arguing with a
/// compiler as much as with the change.
pub const K2_ONLY_ENV: &str = "LARQL_CPU_BIT_IDENTICAL";

/// Whether to stay on the bit-identical K2 kernels.
pub fn bit_identical_only() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| {
        matches!(
            std::env::var(K2_ONLY_ENV).ok().as_deref().map(str::trim),
            Some("1") | Some("true")
        )
    })
}

/// One asymmetric Q8 row, vectorised where possible.
#[inline]
pub(super) fn q8_row_asym(
    codes: &[i8],
    fold_scale: &[f32],
    fold_mid: &[f32],
    qx: &[i8],
    in_dim: usize,
    block: usize,
) -> f32 {
    #[cfg(target_arch = "aarch64")]
    if has_dotprod() {
        // SAFETY: both guarded by the runtime feature check.
        return unsafe {
            if block == SDOT_LANES && !bit_identical_only() {
                q8_row_b16_vector_sdot(codes, fold_scale, Some(fold_mid), qx, in_dim)
            } else if block == SDOT_LANES {
                q8_row_asym_b16_sdot(codes, fold_scale, fold_mid, qx, in_dim)
            } else {
                q8_row_asym_sdot(codes, fold_scale, fold_mid, qx, in_dim, block)
            }
        };
    }
    q8_row_asym_portable(codes, fold_scale, fold_mid, qx, in_dim, block)
}

/// The scale a block's integer sum is multiplied by: the weight's block
/// scale times the activation's.
///
/// One entry per ACTIVATION block. Where the activation block is finer
/// than the weight block, consecutive entries share a weight scale —
/// exact, because a weight scale is constant across its own block by
/// construction.
fn fold_scales(weight_scales: &[f32], act_scales: &[f32], per_weight: usize, into: &mut Vec<f32>) {
    into.clear();
    into.extend(
        act_scales
            .iter()
            .enumerate()
            .map(|(s, a)| weight_scales[s / per_weight] * *a),
    );
}

/// Whether the integer arms scale the activation per tensor or per block.
///
/// Read from the same environment string as the arm, with a `b` suffix,
/// and resolved once per process for the same reason: a bank run is one
/// process per arm and a value that could change mid-decode would make
/// the resulting distribution describe no single representation.
pub fn activation_scaling() -> ScaleSpan {
    static SCALING: std::sync::OnceLock<ScaleSpan> = std::sync::OnceLock::new();
    *SCALING.get_or_init(|| {
        match std::env::var(super::physical::ARITHMETIC_ARM_ENV)
            .ok()
            .as_deref()
            .map(str::trim)
        {
            // Blocked on the WEIGHTS' boundaries, so the two scales fold
            // into one multiply per block and `SDOT` is untouched.
            Some("bf16xq8b") | Some("q8xq8b") | Some("q4xq8b") => {
                ScaleSpan::Block(activation_block())
            }
            _ => ScaleSpan::Tensor,
        }
    })
}

/// Whether this machine has the `dotprod` extension `SDOT` needs.
///
/// Baseline on every Apple M-series part, but not on all aarch64, so the
/// portable definition is a real fallback and not dead code.
#[cfg(target_arch = "aarch64")]
#[inline]
fn has_dotprod() -> bool {
    std::arch::is_aarch64_feature_detected!("dotprod")
}

/// **The DEFINITION** of a Q8 x Q8 row, in portable Rust.
///
/// The integer sum is EXACT — no rounding happens inside a block at all —
/// so the only floating-point in the whole row is one multiply and one
/// add per block. That is what makes the vectorised path bit-comparable
/// rather than merely close.
pub(super) fn q8_row_portable(
    codes: &[i8],
    scales: &[f32],
    qx: &[i8],
    in_dim: usize,
    block: usize,
) -> f32 {
    let mut acc = 0.0f32;
    for (b, scale) in scales.iter().enumerate() {
        let lo = b * block;
        let hi = (lo + block).min(in_dim);
        if lo >= hi {
            break;
        }
        let mut sum = 0i32;
        for i in lo..hi {
            sum += codes[i] as i32 * qx[i] as i32;
        }
        acc += scale * sum as f32;
    }
    acc
}

/// **The DEFINITION** of a Q4 x Q8 row, in portable Rust.
///
/// Byte `j` of a block carries element `j` in its low nibble and element
/// `j + half` in its high nibble, so a block is two CONTIGUOUS runs of
/// the activation rather than one interleaved one.
pub(super) fn q4_row_portable(
    packed: &[u8],
    scales: &[f32],
    qx: &[i8],
    in_dim: usize,
    block: usize,
) -> f32 {
    let mut acc = 0.0f32;
    for (b, scale) in scales.iter().enumerate() {
        let lo = b * block;
        let hi = (lo + block).min(in_dim);
        if lo >= hi {
            break;
        }
        let half = (hi - lo) / 2;
        let mut sum = 0i32;
        for j in 0..half {
            let byte = packed[lo / 2 + j];
            sum += ((byte & NIBBLE) as i32 - Q4_BIAS) * qx[lo + j] as i32;
            sum += ((byte >> 4) as i32 - Q4_BIAS) * qx[lo + j + half] as i32;
        }
        acc += scale * sum as f32;
    }
    acc
}

/// `SDOT`: sixteen int8 pairs into four i32 lanes, one instruction.
///
/// The widen chain Q8 x f32 spends its time on disappears entirely —
/// no `s8 -> s16 -> s32 -> f32` per element, just a load and a dot.
///
/// # Safety
/// Requires `dotprod`, checked by [`has_dotprod`]. Every access stays
/// inside the slices the row geometry describes.
#[cfg(target_arch = "aarch64")]
/// `dotprod` intrinsics are stable since 1.98 and the workspace still
/// declares `rust-version = "1.88"`. The toolchain is PINNED to 1.98 in
/// `rust-toolchain.toml` — deliberately, so local lints are CI's lints —
/// so the declared floor is already below what anything here builds
/// with. Allowed at the function rather than raised workspace-wide,
/// because bumping the manifest's MSRV is a policy decision about every
/// crate and not a side effect of adding a kernel.
#[allow(clippy::incompatible_msrv)]
#[target_feature(enable = "dotprod")]
unsafe fn q8_row_sdot(codes: &[i8], scales: &[f32], qx: &[i8], in_dim: usize, block: usize) -> f32 {
    use std::arch::aarch64::*;
    let mut acc = 0.0f32;
    for (b, scale) in scales.iter().enumerate() {
        let lo = b * block;
        let hi = (lo + block).min(in_dim);
        if lo >= hi {
            break;
        }
        let mut lanes = vdupq_n_s32(0);
        let mut i = lo;
        while i + SDOT_LANES <= hi {
            lanes = vdotq_s32(
                lanes,
                vld1q_s8(codes.as_ptr().add(i)),
                vld1q_s8(qx.as_ptr().add(i)),
            );
            i += SDOT_LANES;
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

/// Q4 x Q8 through `SDOT`: mask and shift one 16-byte load into two int8
/// vectors, unbias by 8, dot each against its half of the activation.
///
/// No widening and no float anywhere in the inner loop.
///
/// # Safety
/// Requires `dotprod`, checked by [`has_dotprod`]. Every access stays
/// inside the slices the row geometry describes.
#[cfg(target_arch = "aarch64")]
/// `dotprod` intrinsics are stable since 1.98 and the workspace still
/// declares `rust-version = "1.88"`. The toolchain is PINNED to 1.98 in
/// `rust-toolchain.toml` — deliberately, so local lints are CI's lints —
/// so the declared floor is already below what anything here builds
/// with. Allowed at the function rather than raised workspace-wide,
/// because bumping the manifest's MSRV is a policy decision about every
/// crate and not a side effect of adding a kernel.
#[allow(clippy::incompatible_msrv)]
#[target_feature(enable = "dotprod")]
unsafe fn q4_row_sdot(
    packed: &[u8],
    scales: &[f32],
    qx: &[i8],
    in_dim: usize,
    block: usize,
) -> f32 {
    use std::arch::aarch64::*;
    let mask = vdupq_n_u8(NIBBLE);
    let bias = vdupq_n_s8(Q4_BIAS as i8);
    let mut acc = 0.0f32;
    for (b, scale) in scales.iter().enumerate() {
        let lo = b * block;
        let hi = (lo + block).min(in_dim);
        if lo >= hi {
            break;
        }
        let half = (hi - lo) / 2;
        let base = packed.as_ptr().add(lo / 2);
        let xbase = qx.as_ptr().add(lo);
        let mut lanes = vdupq_n_s32(0);
        let mut j = 0usize;
        while j + SDOT_LANES <= half {
            let raw = vld1q_u8(base.add(j));
            let low = vsubq_s8(vreinterpretq_s8_u8(vandq_u8(raw, mask)), bias);
            let high = vsubq_s8(vreinterpretq_s8_u8(vshrq_n_u8(raw, 4)), bias);
            lanes = vdotq_s32(lanes, low, vld1q_s8(xbase.add(j)));
            lanes = vdotq_s32(lanes, high, vld1q_s8(xbase.add(j + half)));
            j += SDOT_LANES;
        }
        let mut sum = vaddvq_s32(lanes);
        while j < half {
            let byte = *packed.get_unchecked(lo / 2 + j);
            sum += ((byte & NIBBLE) as i32 - Q4_BIAS) * *qx.get_unchecked(lo + j) as i32;
            sum += ((byte >> 4) as i32 - Q4_BIAS) * *qx.get_unchecked(lo + j + half) as i32;
            j += 1;
        }
        acc += scale * sum as f32;
    }
    acc
}

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
pub(super) fn q4_row_subblocked(
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
fn q8_row(codes: &[i8], scales: &[f32], qx: &[i8], in_dim: usize, block: usize) -> f32 {
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
fn q4_row(packed: &[u8], scales: &[f32], qx: &[i8], in_dim: usize, block: usize) -> f32 {
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

    fn project_rows(&self, weight_rows: WeightRows<'_>, x: &[f32], out: &mut [f32]) {
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
        match activation_scaling() {
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
                        let asym = matches!(activation_code(), ActivationCode::Asymmetric);
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
                match activation_code() {
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
        match activation_scaling() {
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
        if !matches!(weight_rows, WeightRows::Bf16(_)) {
            panic!("the bf16 x q8 control kernel consumes bf16 weights only");
        }
        // Reconstructed ONCE per call, not once per row.
        let rx: Vec<f32> = match activation_scaling() {
            ScaleSpan::Tensor => {
                let act = quantise_activation(x);
                act.codes.iter().map(|c| *c as f32 * act.scale).collect()
            }
            ScaleSpan::Block(block) => match activation_code() {
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
