//! Q4_K weight × Q8_K activation matrix-vector product.
//!
//! The hot path for CPU MoE on Gemma 4 26B-A4B.  Reads 144-byte Q4_K
//! super-blocks straight from the mmapped vindex (no f32 dequant cache),
//! quantises the activation once per call to Q8_K, and accumulates an
//! integer dot product per sub-block.  Math is mathematically equivalent
//! to `q4_common::q4k_matvec_into` (within Q8 quantisation noise on the
//! activation side), but avoids walking ~5.7 GB of f32 weights per token
//! at Gemma 4 26B-A4B sizes — DRAM pressure drops ~4×.
//!
//! Per llama.cpp `ggml_vec_dot_q4_K_q8_K`:
//!
//! ```text
//! per super-block (256 elements, 8 sub-blocks of 32):
//!   d_w    = f16_to_f32(block.d)        (per super-block weight scale)
//!   dmin_w = f16_to_f32(block.dmin)     (per super-block weight min-scale)
//!   d_y    = q8k.d                      (per super-block activation scale)
//!   for sb in 0..8:
//!     sc[sb] (u8 [0..63]), mn[sb] (u8 [0..63])  unpacked from the 12-byte header
//!     dot_sb = Σ_{i in 0..32} q4_nibble[i] * y_q[i]            (i32)
//!     sum_sb = Σ_{i in 0..32} y_q[i]                            (i16, precomputed)
//!     sum1 += sc[sb] * dot_sb
//!     sum2 += mn[sb] * sum_sb
//!   acc += d_w * d_y * sum1 - dmin_w * d_y * sum2
//! out[r] = acc
//! ```
//!
//! Inner kernel uses NEON `sdot` (ARMv8.2-A SDOT instruction, available on
//! Apple M1+ and most modern aarch64 chips) when compiled for `aarch64`;
//! falls back to a scalar reference otherwise.  Both paths share the
//! Q8_K activation quantiser and the per-super-block aggregation math —
//! only the inner i8×i8 → i32 dot differs.

use crate::cpu::ops::q4_common::f16_to_f32;
use larql_models::quant::ggml::{Q4_K_BLOCK_BYTES, Q4_K_BLOCK_ELEMS};

/// Q4_K super-block layout: 144 bytes per 256 values.
const BLOCK_BYTES: usize = Q4_K_BLOCK_BYTES;
/// Number of f32 / i8 elements per Q4_K (and Q8_K) super-block.
const ELEMS_PER_BLOCK: usize = Q4_K_BLOCK_ELEMS;
/// Number of 32-element sub-blocks per super-block.
const SUBBLOCKS_PER_BLOCK: usize = 8;
/// Sub-block size (matches Q4_K's per-32 nibble groups).
const SUBBLOCK_SIZE: usize = 32;

/// Quantised activation in Q8_K layout, one entry per super-block of `x`.
///
/// `qs` packs all super-blocks contiguously: `qs[sb * 256 .. (sb+1) * 256]`
/// is the i8 sub-block stream for super-block `sb`.  `d[sb]` is the f32
/// scale.  `sums[sb * 8 + s]` is the i32 sum of the 32 i8 values in
/// sub-block `s` of super-block `sb` — precomputed once because every
/// row of the matrix needs it for the `mins` term.
pub struct Q8KActivation {
    pub qs: Vec<i8>,
    pub d: Vec<f32>,
    pub sums: Vec<i16>,
}

impl Q8KActivation {
    pub fn n_blocks(&self) -> usize {
        self.d.len()
    }

    /// Allocate an empty Q8KActivation sized for at least `cols` floats.
    /// Used to pre-allocate a reusable buffer in `ExpertScratch` so the
    /// per-expert `quantize_x_to_q8k_into` call doesn't re-allocate at
    /// production sizes.  Rounds `cols` up to the next 256-multiple so
    /// callers don't need to know about Q8_K's super-block geometry —
    /// `quantize_x_to_q8k_into` will resize anyway if the actual input
    /// length differs.
    pub fn with_capacity(cols: usize) -> Self {
        let n_blocks = cols.div_ceil(ELEMS_PER_BLOCK);
        Self {
            qs: vec![0i8; n_blocks * ELEMS_PER_BLOCK],
            d: vec![0.0f32; n_blocks],
            sums: vec![0i16; n_blocks * SUBBLOCKS_PER_BLOCK],
        }
    }
}

/// In-place version of `quantize_x_to_q8k`.  Resizes the output's buffers
/// to match `x.len()` (no-op if already correct), then quantises into
/// them.  Use this from hot paths where the caller owns a long-lived
/// `Q8KActivation` (e.g., per-rayon-thread scratch) so the per-expert
/// activation quantisation doesn't pay an allocator round-trip.
pub fn quantize_x_to_q8k_into(out: &mut Q8KActivation, x: &[f32]) {
    debug_assert_eq!(x.len() % ELEMS_PER_BLOCK, 0);
    let n_blocks = x.len() / ELEMS_PER_BLOCK;
    if out.d.len() != n_blocks {
        out.qs.resize(n_blocks * ELEMS_PER_BLOCK, 0);
        out.d.resize(n_blocks, 0.0);
        out.sums.resize(n_blocks * SUBBLOCKS_PER_BLOCK, 0);
    }

    for sb in 0..n_blocks {
        let base = sb * ELEMS_PER_BLOCK;
        let block = &x[base..base + ELEMS_PER_BLOCK];
        let amax = block.iter().fold(0.0f32, |a, &v| a.max(v.abs()));
        let scale = if amax > 0.0 { amax / 127.0 } else { 0.0 };
        let inv = if scale > 0.0 { 1.0 / scale } else { 0.0 };
        out.d[sb] = scale;

        for s in 0..SUBBLOCKS_PER_BLOCK {
            let off = base + s * SUBBLOCK_SIZE;
            let qoff = sb * ELEMS_PER_BLOCK + s * SUBBLOCK_SIZE;
            let mut acc: i32 = 0;
            for j in 0..SUBBLOCK_SIZE {
                let q = (x[off + j] * inv).round().clamp(-127.0, 127.0) as i8;
                out.qs[qoff + j] = q;
                acc += q as i32;
            }
            out.sums[sb * SUBBLOCKS_PER_BLOCK + s] = acc as i16;
        }
    }
}

/// Quantise an activation vector to Q8_K.  `x.len()` must be a multiple of
/// 256.  Per super-block: find absmax, scale by `127 / absmax` (the
/// llama.cpp convention for Q8_K — symmetric int8 with the full
/// `[-127, 127]` range), and store `d = absmax / 127` so reconstruction
/// is `x ≈ d * q`.  Per sub-block of 32: precompute the i32 sum of the
/// quantised values for the dmin term in the matvec.
pub fn quantize_x_to_q8k(x: &[f32]) -> Q8KActivation {
    debug_assert_eq!(x.len() % ELEMS_PER_BLOCK, 0);
    let n_blocks = x.len() / ELEMS_PER_BLOCK;
    let mut qs = vec![0i8; n_blocks * ELEMS_PER_BLOCK];
    let mut d = vec![0.0f32; n_blocks];
    let mut sums = vec![0i16; n_blocks * SUBBLOCKS_PER_BLOCK];

    for sb in 0..n_blocks {
        let base = sb * ELEMS_PER_BLOCK;
        let block = &x[base..base + ELEMS_PER_BLOCK];
        let amax = block.iter().fold(0.0f32, |a, &v| a.max(v.abs()));
        let scale = if amax > 0.0 { amax / 127.0 } else { 0.0 };
        let inv = if scale > 0.0 { 1.0 / scale } else { 0.0 };
        d[sb] = scale;

        for s in 0..SUBBLOCKS_PER_BLOCK {
            let off = base + s * SUBBLOCK_SIZE;
            let qoff = sb * ELEMS_PER_BLOCK + s * SUBBLOCK_SIZE;
            let mut acc: i32 = 0;
            for j in 0..SUBBLOCK_SIZE {
                let q = (x[off + j] * inv).round().clamp(-127.0, 127.0) as i8;
                qs[qoff + j] = q;
                acc += q as i32;
            }
            sums[sb * SUBBLOCKS_PER_BLOCK + s] = acc as i16;
        }
    }

    Q8KActivation { qs, d, sums }
}

/// Unpack the 12 packed scale/min bytes at the start of a Q4_K super-block
/// into 8 6-bit scales + 8 6-bit mins.  Matches llama.cpp's
/// `get_scale_min_k4` (and `q4_common::dequantize_q4_k` / `q4k_matvec.rs`).
#[inline(always)]
fn unpack_scales_mins(p: &[u8]) -> ([u8; 8], [u8; 8]) {
    let mut scales = [0u8; 8];
    let mut mins = [0u8; 8];
    for j in 0..4 {
        scales[j] = p[j] & 0x3F;
        mins[j] = p[j + 4] & 0x3F;
        scales[j + 4] = (p[j + 8] & 0x0F) | ((p[j] >> 6) << 4);
        mins[j + 4] = (p[j + 8] >> 4) | ((p[j + 4] >> 6) << 4);
    }
    (scales, mins)
}

/// Scalar reference: `out = W · x` where `W` is `rows × cols` Q4_K and `x`
/// has been pre-quantised to Q8_K.  Mathematically equivalent (within Q8
/// quantisation noise on `x`) to `q4_common::q4k_matvec_into`.
///
/// This is the correctness oracle for the NEON implementation below — both
/// must produce bit-identical output given the same `(W, q8k_x)`.
pub fn q4k_q8k_matvec_scalar(
    out: &mut [f32],
    q8k_x: &Q8KActivation,
    w: &[u8],
    rows: usize,
    cols: usize,
) {
    debug_assert_eq!(out.len(), rows);
    debug_assert_eq!(q8k_x.qs.len(), cols);
    debug_assert_eq!(cols % ELEMS_PER_BLOCK, 0);
    if rows == 0 || cols == 0 {
        for v in out.iter_mut() {
            *v = 0.0;
        }
        return;
    }
    let n_blocks = cols / ELEMS_PER_BLOCK;
    let row_bytes = n_blocks * BLOCK_BYTES;
    if w.len() < rows * row_bytes {
        for v in out.iter_mut() {
            *v = 0.0;
        }
        return;
    }

    for (r, out_slot) in out.iter_mut().enumerate().take(rows) {
        let row_base = r * row_bytes;
        let mut acc = 0.0f32;
        for sb in 0..n_blocks {
            let block = &w[row_base + sb * BLOCK_BYTES..row_base + (sb + 1) * BLOCK_BYTES];
            let d_w = f16_to_f32(u16::from_le_bytes([block[0], block[1]]));
            let dmin_w = f16_to_f32(u16::from_le_bytes([block[2], block[3]]));
            let (scales, mins) = unpack_scales_mins(&block[4..16]);
            // 16 = 2 (d) + 2 (dmin) + 12 (packed scales/mins).
            // The remaining BLOCK_BYTES-16 = 128 bytes are nibble-packed quants.
            let quants = &block[16..BLOCK_BYTES];

            let q8_base = sb * ELEMS_PER_BLOCK;
            let q8_qs = &q8k_x.qs[q8_base..q8_base + ELEMS_PER_BLOCK];
            let q8_sums = &q8k_x.sums[sb * SUBBLOCKS_PER_BLOCK..(sb + 1) * SUBBLOCKS_PER_BLOCK];
            let d_y = q8k_x.d[sb];

            // sum1 = Σ_sb scales[sb] · dot_int(q4_nibbles, q8_y)
            // sum2 = Σ_sb mins[sb]   · sum(q8_y in this sb)
            let mut sum1: i32 = 0;
            let mut sum2: i32 = 0;
            for g in 0..4 {
                let sb_lo = 2 * g;
                let sb_hi = 2 * g + 1;
                let chunk = &quants[g * 32..(g + 1) * 32];
                let y_lo = &q8_qs[sb_lo * SUBBLOCK_SIZE..(sb_lo + 1) * SUBBLOCK_SIZE];
                let y_hi = &q8_qs[sb_hi * SUBBLOCK_SIZE..(sb_hi + 1) * SUBBLOCK_SIZE];

                let mut dot_lo: i32 = 0;
                let mut dot_hi: i32 = 0;
                for l in 0..32 {
                    let byte = chunk[l];
                    let q_lo = (byte & 0x0F) as i32;
                    let q_hi = ((byte >> 4) & 0x0F) as i32;
                    dot_lo += q_lo * y_lo[l] as i32;
                    dot_hi += q_hi * y_hi[l] as i32;
                }
                sum1 += scales[sb_lo] as i32 * dot_lo + scales[sb_hi] as i32 * dot_hi;
                sum2 += mins[sb_lo] as i32 * q8_sums[sb_lo] as i32
                    + mins[sb_hi] as i32 * q8_sums[sb_hi] as i32;
            }
            acc += d_w * d_y * sum1 as f32 - dmin_w * d_y * sum2 as f32;
        }
        *out_slot = acc;
    }
}

/// SDOT (signed 8-bit dot-product, accumulate-into-i32x4) wrapper.
///
/// Computes `acc + Σ_{lane=0..16} a[lane] * b[lane]`, returning an `int32x4_t`
/// where each i32 lane holds the sum of 4 i8 × i8 products.  One ARMv8.2-A
/// `SDOT` instruction; M1+ supports it natively (the `dotprod` target
/// feature is enabled by default for `aarch64-apple-darwin`).
///
/// Implemented via inline asm because `core::arch::aarch64::vdotq_s32` is
/// still gated behind the unstable `stdarch_neon_dotprod` feature on Rust
/// 1.91 (issue rust-lang/rust#117224).  The asm form is stable today.
#[cfg(all(target_arch = "aarch64", target_feature = "neon"))]
#[inline(always)]
unsafe fn sdot_acc(
    acc: std::arch::aarch64::int32x4_t,
    a: std::arch::aarch64::int8x16_t,
    b: std::arch::aarch64::int8x16_t,
) -> std::arch::aarch64::int32x4_t {
    let result: std::arch::aarch64::int32x4_t;
    unsafe {
        core::arch::asm!(
            "sdot {0:v}.4s, {1:v}.16b, {2:v}.16b",
            inlateout(vreg) acc => result,
            in(vreg) a,
            in(vreg) b,
            options(pure, nomem, nostack, preserves_flags),
        );
    }
    result
}

/// Software prefetch hint — bring the cache line containing `ptr` into
/// L1 ahead of an upcoming read. Emits an aarch64 `PRFM PLDL1KEEP` so
/// the data is fetched but tagged as keep-in-L1 (good for hot loops
/// that revisit nearby addresses).
///
/// M3 Max's hardware prefetcher handles linear sequential reads
/// well, but the Q4_K matvec stride (144 bytes per super-block, then
/// jumps to the next row) isn't a simple stride pattern. Explicit
/// hints close ~5-15% of the per-core gap to llama.cpp on these
/// kernels (which has the same hints in its hand-asm path).
#[cfg(target_arch = "aarch64")]
#[inline(always)]
#[allow(dead_code)] // kept for future re-enablement on harder access patterns; see DIAGNOSIS-2026-05-16-thread-scaling.md
unsafe fn prefetch_l1_keep(ptr: *const u8) {
    unsafe {
        core::arch::asm!(
            "prfm pldl1keep, [{0}]",
            in(reg) ptr,
            options(nostack, readonly, preserves_flags),
        );
    }
}

/// NEON-accelerated `q4k_q8k_matvec` for `aarch64`.  Inner kernel uses
/// `SDOT` (16 i8 × i8 → 4 i32 lanes per instruction) for the integer dot
/// products against the Q8_K activation.  Per-row work per super-block:
/// load 32-byte nibble chunk, mask low / shift high, two SDOT calls per
/// half (16 lanes each), add into per-row f32 accumulator.
#[cfg(all(target_arch = "aarch64", target_feature = "neon"))]
pub fn q4k_q8k_matvec_neon(
    out: &mut [f32],
    q8k_x: &Q8KActivation,
    w: &[u8],
    rows: usize,
    cols: usize,
) {
    use std::arch::aarch64::*;

    debug_assert_eq!(out.len(), rows);
    debug_assert_eq!(q8k_x.qs.len(), cols);
    debug_assert_eq!(cols % ELEMS_PER_BLOCK, 0);
    if rows == 0 || cols == 0 {
        for v in out.iter_mut() {
            *v = 0.0;
        }
        return;
    }
    let n_blocks = cols / ELEMS_PER_BLOCK;
    let row_bytes = n_blocks * BLOCK_BYTES;
    if w.len() < rows * row_bytes {
        for v in out.iter_mut() {
            *v = 0.0;
        }
        return;
    }

    // Mask vector for low-nibble extraction (broadcast 0x0F across 16 lanes).
    let mask_lo = unsafe { vdupq_n_u8(0x0F) };

    // No software prefetch: tested 2026-05-16 with `prfm pldl1keep`
    // hints at per-row and per-super-block granularity. Both regressed
    // single-thread throughput on M3 Max (5.5 vs 5.7 tok/s baseline).
    // The hardware prefetcher handles both the in-row Q4_K stride and
    // the row-to-row jump well enough that software hints compete for
    // L1 fill bandwidth without delivering new data. Kept the
    // `prefetch_l1_keep` helper for future re-enablement on harder
    // access patterns.
    for (r, out_slot) in out.iter_mut().enumerate().take(rows) {
        let row_base = r * row_bytes;
        let mut acc = 0.0f32;
        for sb in 0..n_blocks {
            let block = &w[row_base + sb * BLOCK_BYTES..row_base + (sb + 1) * BLOCK_BYTES];
            let d_w = f16_to_f32(u16::from_le_bytes([block[0], block[1]]));
            let dmin_w = f16_to_f32(u16::from_le_bytes([block[2], block[3]]));
            let (scales, mins) = unpack_scales_mins(&block[4..16]);
            let quants_ptr = block[16..].as_ptr();

            let q8_base = sb * ELEMS_PER_BLOCK;
            let q8_qs_ptr = q8k_x.qs[q8_base..q8_base + ELEMS_PER_BLOCK].as_ptr();
            let q8_sums = &q8k_x.sums[sb * SUBBLOCKS_PER_BLOCK..(sb + 1) * SUBBLOCKS_PER_BLOCK];
            let d_y = q8k_x.d[sb];

            // sum1 = Σ_sb scales[sb] · dot_int(q4_nibbles, q8_y) (i32)
            // sum2 = Σ_sb mins[sb]   ·  Σ q8_y in this sb        (i32)
            //
            // Vector-running accumulator: keep the i32x4 partial sums
            // across all 4 groups in `sum1_v`, only horizontal-reduce
            // once per super-block instead of once per group. Each
            // group's lo/hi partial dot is scaled (vmulq_n_s32) and
            // added into `sum1_v` via vector mla. Eliminates the
            // 4-per-super-block `vaddvq_s32` + scalar mul chain that
            // forced a forced retire of the prior group's SDOTs.
            //
            // Independent SDOT pairs: instead of chaining
            //   acc = sdot(prev, lo1, y_lo1)
            // (which serialises on `prev` at 4-cycle latency), issue
            // both SDOTs into separate destination registers and
            // combine via vaddq_s32. Drops per-half latency from
            // 8 cycles → ~5 cycles on M3's OoO scheduler.
            let zero_v = unsafe { vdupq_n_s32(0) };
            let mut sum1_v = unsafe { vdupq_n_s32(0) };
            let mut sum2_acc: i32 = 0;

            for g in 0..4 {
                let sb_lo = 2 * g;
                let sb_hi = 2 * g + 1;
                // Paired load: 32 nibble bytes in one `ld1.2d` instead
                // of two `ldr`. Same total bandwidth but a single
                // pipeline slot and a clearer hint to the memory
                // subsystem.
                let nibs_pair = unsafe { vld1q_u8_x2(quants_ptr.add(g * 32)) };
                let nib0 = nibs_pair.0;
                let nib1 = nibs_pair.1;

                // Low nibbles → sub-block 2g, high nibbles → sub-block 2g+1.
                let lo0 = unsafe { vreinterpretq_s8_u8(vandq_u8(nib0, mask_lo)) };
                let lo1 = unsafe { vreinterpretq_s8_u8(vandq_u8(nib1, mask_lo)) };
                let hi0 = unsafe { vreinterpretq_s8_u8(vshrq_n_u8(nib0, 4)) };
                let hi1 = unsafe { vreinterpretq_s8_u8(vshrq_n_u8(nib1, 4)) };

                // Paired loads of the activation halves: 32 bytes
                // for each sub-block (lo + hi). Two `ld1.2d` total.
                let y_lo_pair = unsafe { vld1q_s8_x2(q8_qs_ptr.add(sb_lo * SUBBLOCK_SIZE)) };
                let y_hi_pair = unsafe { vld1q_s8_x2(q8_qs_ptr.add(sb_hi * SUBBLOCK_SIZE)) };
                let y_lo0 = y_lo_pair.0;
                let y_lo1 = y_lo_pair.1;
                let y_hi0 = y_hi_pair.0;
                let y_hi1 = y_hi_pair.1;

                // Independent SDOT pairs: 4 SDOTs into 4 destination
                // registers (no inter-SDOT data dependency), then sum
                // pairs with vaddq.
                let dlo0 = unsafe { sdot_acc(zero_v, lo0, y_lo0) };
                let dlo1 = unsafe { sdot_acc(zero_v, lo1, y_lo1) };
                let dhi0 = unsafe { sdot_acc(zero_v, hi0, y_hi0) };
                let dhi1 = unsafe { sdot_acc(zero_v, hi1, y_hi1) };
                let dlo_acc = unsafe { vaddq_s32(dlo0, dlo1) };
                let dhi_acc = unsafe { vaddq_s32(dhi0, dhi1) };

                // Scale and accumulate into running i32x4. The two
                // vmulq_n_s32 + two vaddq_s32 per group adds ~3 cycles
                // but saves the forced `vaddvq + scalar mul + scalar
                // add` chain (which serialised group g+1 behind it).
                let scaled_lo = unsafe { vmulq_n_s32(dlo_acc, scales[sb_lo] as i32) };
                let scaled_hi = unsafe { vmulq_n_s32(dhi_acc, scales[sb_hi] as i32) };
                sum1_v = unsafe { vaddq_s32(sum1_v, vaddq_s32(scaled_lo, scaled_hi)) };

                // `sum2` stays scalar — the input here is the
                // precomputed Q8_K sums, so no SDOT involved.
                sum2_acc += mins[sb_lo] as i32 * q8_sums[sb_lo] as i32
                    + mins[sb_hi] as i32 * q8_sums[sb_hi] as i32;
            }
            let sum1 = unsafe { vaddvq_s32(sum1_v) };
            acc += d_w * d_y * sum1 as f32 - dmin_w * d_y * sum2_acc as f32;
        }
        *out_slot = acc;
    }
}

/// Two-row variant of `q4k_q8k_matvec_neon`: processes a pair of output rows
/// per inner loop iteration, sharing the activation Q8_K loads.
///
/// Per super-block: load activation halves once, decode both rows' headers,
/// then emit 16 SDOTs (8 per row) instead of 8 sequential ones.  The doubled
/// in-flight SDOT pressure gives the OoO scheduler more independent work to
/// hide DRAM-load latency on the Q4_K weight stream — the bottleneck the
/// 2026-05-01 profile pinned as the remaining ~70% of per-call time.
///
/// The activation load amortisation is small in raw bytes (256 i8 per
/// super-block, hot in L1) but moves the inner-loop bottleneck from
/// "scheduler stall while waiting for the next nibble byte" toward "SDOT
/// throughput limited" — which is what we want, because SDOT pipes can
/// run two-wide on Apple Silicon.
///
/// Tail handling: if `rows` is odd, the final row falls back to the
/// single-row kernel.  Production matvec dims (`inter=704`, `hidden=2816`)
/// are even so this is a no-op there.
#[cfg(all(target_arch = "aarch64", target_feature = "neon"))]
pub fn q4k_q8k_matvec_neon_2row(
    out: &mut [f32],
    q8k_x: &Q8KActivation,
    w: &[u8],
    rows: usize,
    cols: usize,
) {
    use std::arch::aarch64::*;

    debug_assert_eq!(out.len(), rows);
    debug_assert_eq!(q8k_x.qs.len(), cols);
    debug_assert_eq!(cols % ELEMS_PER_BLOCK, 0);
    if rows == 0 || cols == 0 {
        for v in out.iter_mut() {
            *v = 0.0;
        }
        return;
    }
    let n_blocks = cols / ELEMS_PER_BLOCK;
    let row_bytes = n_blocks * BLOCK_BYTES;
    if w.len() < rows * row_bytes {
        for v in out.iter_mut() {
            *v = 0.0;
        }
        return;
    }

    let mask_lo = unsafe { vdupq_n_u8(0x0F) };

    // Pair-of-rows loop: process rows (r, r+1) together.
    let pairs = rows / 2;
    for p in 0..pairs {
        let r0 = 2 * p;
        let r1 = 2 * p + 1;
        let r0_base = r0 * row_bytes;
        let r1_base = r1 * row_bytes;
        let mut acc0 = 0.0f32;
        let mut acc1 = 0.0f32;
        for sb in 0..n_blocks {
            let b0 = &w[r0_base + sb * BLOCK_BYTES..r0_base + (sb + 1) * BLOCK_BYTES];
            let b1 = &w[r1_base + sb * BLOCK_BYTES..r1_base + (sb + 1) * BLOCK_BYTES];
            let d0 = f16_to_f32(u16::from_le_bytes([b0[0], b0[1]]));
            let dmin0 = f16_to_f32(u16::from_le_bytes([b0[2], b0[3]]));
            let d1 = f16_to_f32(u16::from_le_bytes([b1[0], b1[1]]));
            let dmin1 = f16_to_f32(u16::from_le_bytes([b1[2], b1[3]]));
            let (sc0, mn0) = unpack_scales_mins(&b0[4..16]);
            let (sc1, mn1) = unpack_scales_mins(&b1[4..16]);
            let q0 = b0[16..].as_ptr();
            let q1 = b1[16..].as_ptr();

            let q8_base = sb * ELEMS_PER_BLOCK;
            let q8_qs_ptr = q8k_x.qs[q8_base..q8_base + ELEMS_PER_BLOCK].as_ptr();
            let q8_sums = &q8k_x.sums[sb * SUBBLOCKS_PER_BLOCK..(sb + 1) * SUBBLOCKS_PER_BLOCK];
            let d_y = q8k_x.d[sb];

            let mut s1_0: i32 = 0;
            let mut s2_0: i32 = 0;
            let mut s1_1: i32 = 0;
            let mut s2_1: i32 = 0;

            for grp in 0..4 {
                let sb_lo = 2 * grp;
                let sb_hi = 2 * grp + 1;
                // Activation halves shared across both rows.
                let y_lo0 = unsafe { vld1q_s8(q8_qs_ptr.add(sb_lo * SUBBLOCK_SIZE)) };
                let y_lo1 = unsafe { vld1q_s8(q8_qs_ptr.add(sb_lo * SUBBLOCK_SIZE + 16)) };
                let y_hi0 = unsafe { vld1q_s8(q8_qs_ptr.add(sb_hi * SUBBLOCK_SIZE)) };
                let y_hi1 = unsafe { vld1q_s8(q8_qs_ptr.add(sb_hi * SUBBLOCK_SIZE + 16)) };

                // Row-0 nibble bytes for this 32-byte group.
                let n0a = unsafe { vld1q_u8(q0.add(grp * 32)) };
                let n0b = unsafe { vld1q_u8(q0.add(grp * 32 + 16)) };
                let lo0a = unsafe { vreinterpretq_s8_u8(vandq_u8(n0a, mask_lo)) };
                let lo0b = unsafe { vreinterpretq_s8_u8(vandq_u8(n0b, mask_lo)) };
                let hi0a = unsafe { vreinterpretq_s8_u8(vshrq_n_u8(n0a, 4)) };
                let hi0b = unsafe { vreinterpretq_s8_u8(vshrq_n_u8(n0b, 4)) };

                // Row-1 nibble bytes.
                let n1a = unsafe { vld1q_u8(q1.add(grp * 32)) };
                let n1b = unsafe { vld1q_u8(q1.add(grp * 32 + 16)) };
                let lo1a = unsafe { vreinterpretq_s8_u8(vandq_u8(n1a, mask_lo)) };
                let lo1b = unsafe { vreinterpretq_s8_u8(vandq_u8(n1b, mask_lo)) };
                let hi1a = unsafe { vreinterpretq_s8_u8(vshrq_n_u8(n1a, 4)) };
                let hi1b = unsafe { vreinterpretq_s8_u8(vshrq_n_u8(n1b, 4)) };

                // 16 SDOTs total: 8 per row.  Issue them with the two
                // rows interleaved at the inter-iteration level so the
                // OoO scheduler can dispatch from either stream when one
                // is stalled on a load.
                let zero = unsafe { vdupq_n_s32(0) };
                let dlo_0 = unsafe {
                    let a = sdot_acc(zero, lo0a, y_lo0);
                    sdot_acc(a, lo0b, y_lo1)
                };
                let dlo_1 = unsafe {
                    let a = sdot_acc(zero, lo1a, y_lo0);
                    sdot_acc(a, lo1b, y_lo1)
                };
                let dhi_0 = unsafe {
                    let a = sdot_acc(zero, hi0a, y_hi0);
                    sdot_acc(a, hi0b, y_hi1)
                };
                let dhi_1 = unsafe {
                    let a = sdot_acc(zero, hi1a, y_hi0);
                    sdot_acc(a, hi1b, y_hi1)
                };
                let dot_lo_0 = unsafe { vaddvq_s32(dlo_0) };
                let dot_hi_0 = unsafe { vaddvq_s32(dhi_0) };
                let dot_lo_1 = unsafe { vaddvq_s32(dlo_1) };
                let dot_hi_1 = unsafe { vaddvq_s32(dhi_1) };

                s1_0 += sc0[sb_lo] as i32 * dot_lo_0 + sc0[sb_hi] as i32 * dot_hi_0;
                s2_0 += mn0[sb_lo] as i32 * q8_sums[sb_lo] as i32
                    + mn0[sb_hi] as i32 * q8_sums[sb_hi] as i32;
                s1_1 += sc1[sb_lo] as i32 * dot_lo_1 + sc1[sb_hi] as i32 * dot_hi_1;
                s2_1 += mn1[sb_lo] as i32 * q8_sums[sb_lo] as i32
                    + mn1[sb_hi] as i32 * q8_sums[sb_hi] as i32;
            }
            acc0 += d0 * d_y * s1_0 as f32 - dmin0 * d_y * s2_0 as f32;
            acc1 += d1 * d_y * s1_1 as f32 - dmin1 * d_y * s2_1 as f32;
        }
        out[r0] = acc0;
        out[r1] = acc1;
    }

    // Tail: odd row count → process the last row via the single-row kernel.
    if rows % 2 == 1 {
        let r = rows - 1;
        let mut tail_out = [0.0f32; 1];
        let row_w = &w[r * row_bytes..(r + 1) * row_bytes];
        q4k_q8k_matvec_neon(&mut tail_out, q8k_x, row_w, 1, cols);
        out[r] = tail_out[0];
    }
}

/// Public entry point: dispatches to NEON on aarch64, scalar elsewhere.
/// Caller pre-quantises `x` once via `quantize_x_to_q8k` (cost is amortised
/// across all rows of the same matvec, and across all K active experts that
/// share `h_norm`).
pub fn q4k_q8k_matvec_into(
    out: &mut [f32],
    q8k_x: &Q8KActivation,
    w: &[u8],
    rows: usize,
    cols: usize,
) {
    #[cfg(all(target_arch = "aarch64", target_feature = "neon"))]
    {
        // 2-row variant tried 2026-05-01 — bit-exact (`q8k_matvec_2row_matches_single_row_bit_exact`)
        // but bench-neutral on M3 Max: per-thread is BW-bound on the
        // per-row Q4_K weight stream (1.1 MB at 82 µs ≈ 14 GB/s), and
        // sharing the small activation Q8K (256 B) across 2 rows didn't
        // free real DRAM bandwidth.  Kept as `q4k_q8k_matvec_neon_2row`
        // for future hardware where ILP may dominate over BW.
        q4k_q8k_matvec_neon(out, q8k_x, w, rows, cols);
        return;
    }
    #[cfg(target_arch = "x86_64")]
    if is_x86_feature_detected!("avx2") {
        // SAFETY: runtime check guarantees AVX2 availability.
        unsafe { q4k_q8k_matvec_avx2(out, q8k_x, w, rows, cols) };
        return;
    }
    #[allow(unreachable_code)]
    q4k_q8k_matvec_scalar(out, q8k_x, w, rows, cols);
}

/// AVX2 Q4_K × Q8_K matvec for x86_64.
///
/// `vpmaddubsw` (unsigned×signed 8-bit → adjacent-pair-summed 16-bit) replaces
/// 32 scalar multiplies per 32-element group.  `vpmaddwd` widens to 32-bit.
/// On AMD EPYC / Intel Haswell+ this is ~12–16× faster than the scalar path.
///
/// Bit-equivalence with the scalar reference is verified in unit tests below.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn q4k_q8k_matvec_avx2(
    out: &mut [f32],
    q8k_x: &Q8KActivation,
    w: &[u8],
    rows: usize,
    cols: usize,
) {
    use std::arch::x86_64::*;

    if rows == 0 || cols == 0 || w.len() < rows * (cols / ELEMS_PER_BLOCK) * BLOCK_BYTES {
        for v in out.iter_mut() {
            *v = 0.0;
        }
        return;
    }

    let n_blocks = cols / ELEMS_PER_BLOCK;
    let row_bytes = n_blocks * BLOCK_BYTES;
    let lo_mask = _mm256_set1_epi8(0x0F);
    let ones_epi16 = _mm256_set1_epi16(1);

    for (r, out_slot) in out.iter_mut().enumerate().take(rows) {
        let row_base = r * row_bytes;
        let mut acc = 0.0f32;

        for sb in 0..n_blocks {
            let block = &w[row_base + sb * BLOCK_BYTES..row_base + (sb + 1) * BLOCK_BYTES];
            let d_w = f16_to_f32(u16::from_le_bytes([block[0], block[1]]));
            let dmin_w = f16_to_f32(u16::from_le_bytes([block[2], block[3]]));
            let (scales, mins) = unpack_scales_mins(&block[4..16]);
            // 16 = 2 (d) + 2 (dmin) + 12 (packed scales/mins).
            // The remaining BLOCK_BYTES-16 = 128 bytes are nibble-packed quants.
            let quants = &block[16..BLOCK_BYTES];
            let q8_base = sb * ELEMS_PER_BLOCK;
            let q8_qs = &q8k_x.qs[q8_base..q8_base + ELEMS_PER_BLOCK];
            let q8_sums = &q8k_x.sums[sb * SUBBLOCKS_PER_BLOCK..(sb + 1) * SUBBLOCKS_PER_BLOCK];
            let d_y = q8k_x.d[sb];

            let mut sum1: i32 = 0;
            let mut sum2: i32 = 0;

            for g in 0..4 {
                let sb_lo = 2 * g;
                let sb_hi = 2 * g + 1;

                // Load 32 Q4 bytes → separate low nibbles (u8 0-15) and high nibbles.
                let q4 = _mm256_loadu_si256(quants.as_ptr().add(g * 32) as *const __m256i);
                let lo_nibbles = _mm256_and_si256(q4, lo_mask);
                let hi_nibbles = _mm256_and_si256(_mm256_srli_epi16(q4, 4), lo_mask);

                // Load 32 Q8 activation bytes for each sub-block half.
                let y_lo =
                    _mm256_loadu_si256(q8_qs.as_ptr().add(sb_lo * SUBBLOCK_SIZE) as *const __m256i);
                let y_hi =
                    _mm256_loadu_si256(q8_qs.as_ptr().add(sb_hi * SUBBLOCK_SIZE) as *const __m256i);

                // vpmaddubsw: (u8 × i8) → adjacent-pair-summed i16 (32 → 16 values).
                // vpmaddwd with all-ones: i16 pair-sum → i32 (16 → 8 values).
                let dot_lo = hsum_i32x8(_mm256_madd_epi16(
                    _mm256_maddubs_epi16(lo_nibbles, y_lo),
                    ones_epi16,
                ));
                let dot_hi = hsum_i32x8(_mm256_madd_epi16(
                    _mm256_maddubs_epi16(hi_nibbles, y_hi),
                    ones_epi16,
                ));

                sum1 += scales[sb_lo] as i32 * dot_lo + scales[sb_hi] as i32 * dot_hi;
                sum2 += mins[sb_lo] as i32 * q8_sums[sb_lo] as i32
                    + mins[sb_hi] as i32 * q8_sums[sb_hi] as i32;
            }
            acc += d_w * d_y * sum1 as f32 - dmin_w * d_y * sum2 as f32;
        }
        *out_slot = acc;
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn hsum_i32x8(v: std::arch::x86_64::__m256i) -> i32 {
    use std::arch::x86_64::*;
    let lo = _mm256_castsi256_si128(v);
    let hi = _mm256_extracti128_si256(v, 1);
    let v128 = _mm_add_epi32(lo, hi);
    let v64 = _mm_add_epi32(v128, _mm_srli_si128(v128, 8));
    let v32 = _mm_add_epi32(v64, _mm_srli_si128(v64, 4));
    _mm_cvtsi128_si32(v32)
}

/// Fused gate+up matvec: produce two output vectors from two weight matrices
/// against the SAME pre-quantised Q8_K activation in one pass.  Each
/// super-block of `q8k_x` is loaded once and SDOT'd against both `gate_w`
/// and `up_w` per row — gate and up SDOTs interleave on the OoO engine,
/// hiding cross-instruction latency that the back-to-back independent
/// `q4k_q8k_matvec_into` calls couldn't.
///
/// Caller layouts: `gate_w.len() == up_w.len() == rows * (cols / 256) * 144`,
/// `gate_out.len() == up_out.len() == rows`.
pub fn q4k_q8k_gate_up_into(
    gate_out: &mut [f32],
    up_out: &mut [f32],
    q8k_x: &Q8KActivation,
    gate_w: &[u8],
    up_w: &[u8],
    rows: usize,
    cols: usize,
) {
    #[cfg(all(target_arch = "aarch64", target_feature = "neon"))]
    {
        q4k_q8k_gate_up_neon(gate_out, up_out, q8k_x, gate_w, up_w, rows, cols);
        return;
    }
    #[allow(unreachable_code)]
    {
        // Scalar fallback: just call the existing single-matvec path twice.
        q4k_q8k_matvec_scalar(gate_out, q8k_x, gate_w, rows, cols);
        q4k_q8k_matvec_scalar(up_out, q8k_x, up_w, rows, cols);
    }
}

#[cfg(all(target_arch = "aarch64", target_feature = "neon"))]
pub fn q4k_q8k_gate_up_neon(
    gate_out: &mut [f32],
    up_out: &mut [f32],
    q8k_x: &Q8KActivation,
    gate_w: &[u8],
    up_w: &[u8],
    rows: usize,
    cols: usize,
) {
    use std::arch::aarch64::*;

    debug_assert_eq!(gate_out.len(), rows);
    debug_assert_eq!(up_out.len(), rows);
    debug_assert_eq!(q8k_x.qs.len(), cols);
    debug_assert_eq!(cols % ELEMS_PER_BLOCK, 0);
    if rows == 0 || cols == 0 {
        for v in gate_out.iter_mut() {
            *v = 0.0;
        }
        for v in up_out.iter_mut() {
            *v = 0.0;
        }
        return;
    }
    let n_blocks = cols / ELEMS_PER_BLOCK;
    let row_bytes = n_blocks * BLOCK_BYTES;
    if gate_w.len() < rows * row_bytes || up_w.len() < rows * row_bytes {
        for v in gate_out.iter_mut() {
            *v = 0.0;
        }
        for v in up_out.iter_mut() {
            *v = 0.0;
        }
        return;
    }

    let mask_lo = unsafe { vdupq_n_u8(0x0F) };

    for r in 0..rows {
        let row_base = r * row_bytes;
        let mut acc_g = 0.0f32;
        let mut acc_u = 0.0f32;
        for sb in 0..n_blocks {
            let g_block = &gate_w[row_base + sb * BLOCK_BYTES..row_base + (sb + 1) * BLOCK_BYTES];
            let u_block = &up_w[row_base + sb * BLOCK_BYTES..row_base + (sb + 1) * BLOCK_BYTES];
            let d_g = f16_to_f32(u16::from_le_bytes([g_block[0], g_block[1]]));
            let dmin_g = f16_to_f32(u16::from_le_bytes([g_block[2], g_block[3]]));
            let d_u = f16_to_f32(u16::from_le_bytes([u_block[0], u_block[1]]));
            let dmin_u = f16_to_f32(u16::from_le_bytes([u_block[2], u_block[3]]));
            let (sc_g, mn_g) = unpack_scales_mins(&g_block[4..16]);
            let (sc_u, mn_u) = unpack_scales_mins(&u_block[4..16]);
            let q_g = g_block[16..].as_ptr();
            let q_u = u_block[16..].as_ptr();

            let q8_base = sb * ELEMS_PER_BLOCK;
            let q8_qs_ptr = q8k_x.qs[q8_base..q8_base + ELEMS_PER_BLOCK].as_ptr();
            let q8_sums = &q8k_x.sums[sb * SUBBLOCKS_PER_BLOCK..(sb + 1) * SUBBLOCKS_PER_BLOCK];
            let d_y = q8k_x.d[sb];

            let mut s1_g: i32 = 0;
            let mut s2_g: i32 = 0;
            let mut s1_u: i32 = 0;
            let mut s2_u: i32 = 0;

            for grp in 0..4 {
                let sb_lo = 2 * grp;
                let sb_hi = 2 * grp + 1;
                // Activation halves shared between gate and up.
                let y_lo0 = unsafe { vld1q_s8(q8_qs_ptr.add(sb_lo * SUBBLOCK_SIZE)) };
                let y_lo1 = unsafe { vld1q_s8(q8_qs_ptr.add(sb_lo * SUBBLOCK_SIZE + 16)) };
                let y_hi0 = unsafe { vld1q_s8(q8_qs_ptr.add(sb_hi * SUBBLOCK_SIZE)) };
                let y_hi1 = unsafe { vld1q_s8(q8_qs_ptr.add(sb_hi * SUBBLOCK_SIZE + 16)) };

                let gnib0 = unsafe { vld1q_u8(q_g.add(grp * 32)) };
                let gnib1 = unsafe { vld1q_u8(q_g.add(grp * 32 + 16)) };
                let glo0 = unsafe { vreinterpretq_s8_u8(vandq_u8(gnib0, mask_lo)) };
                let glo1 = unsafe { vreinterpretq_s8_u8(vandq_u8(gnib1, mask_lo)) };
                let ghi0 = unsafe { vreinterpretq_s8_u8(vshrq_n_u8(gnib0, 4)) };
                let ghi1 = unsafe { vreinterpretq_s8_u8(vshrq_n_u8(gnib1, 4)) };

                let unib0 = unsafe { vld1q_u8(q_u.add(grp * 32)) };
                let unib1 = unsafe { vld1q_u8(q_u.add(grp * 32 + 16)) };
                let ulo0 = unsafe { vreinterpretq_s8_u8(vandq_u8(unib0, mask_lo)) };
                let ulo1 = unsafe { vreinterpretq_s8_u8(vandq_u8(unib1, mask_lo)) };
                let uhi0 = unsafe { vreinterpretq_s8_u8(vshrq_n_u8(unib0, 4)) };
                let uhi1 = unsafe { vreinterpretq_s8_u8(vshrq_n_u8(unib1, 4)) };

                // 8 SDOTs per group, gate / up issued back-to-back so the
                // OoO engine can dispatch them on different ports.
                let zero = unsafe { vdupq_n_s32(0) };
                let g_dlo = unsafe {
                    let a = sdot_acc(zero, glo0, y_lo0);
                    sdot_acc(a, glo1, y_lo1)
                };
                let u_dlo = unsafe {
                    let a = sdot_acc(zero, ulo0, y_lo0);
                    sdot_acc(a, ulo1, y_lo1)
                };
                let g_dhi = unsafe {
                    let a = sdot_acc(zero, ghi0, y_hi0);
                    sdot_acc(a, ghi1, y_hi1)
                };
                let u_dhi = unsafe {
                    let a = sdot_acc(zero, uhi0, y_hi0);
                    sdot_acc(a, uhi1, y_hi1)
                };

                let g_dot_lo = unsafe { vaddvq_s32(g_dlo) };
                let g_dot_hi = unsafe { vaddvq_s32(g_dhi) };
                let u_dot_lo = unsafe { vaddvq_s32(u_dlo) };
                let u_dot_hi = unsafe { vaddvq_s32(u_dhi) };

                s1_g += sc_g[sb_lo] as i32 * g_dot_lo + sc_g[sb_hi] as i32 * g_dot_hi;
                s2_g += mn_g[sb_lo] as i32 * q8_sums[sb_lo] as i32
                    + mn_g[sb_hi] as i32 * q8_sums[sb_hi] as i32;
                s1_u += sc_u[sb_lo] as i32 * u_dot_lo + sc_u[sb_hi] as i32 * u_dot_hi;
                s2_u += mn_u[sb_lo] as i32 * q8_sums[sb_lo] as i32
                    + mn_u[sb_hi] as i32 * q8_sums[sb_hi] as i32;
            }
            acc_g += d_g * d_y * s1_g as f32 - dmin_g * d_y * s2_g as f32;
            acc_u += d_u * d_y * s1_u as f32 - dmin_u * d_y * s2_u as f32;
        }
        gate_out[r] = acc_g;
        up_out[r] = acc_u;
    }
}

// ── Q6_K × Q8_K matvec ───────────────────────────────────────────────────────
//
// Q6_K super-block: 210 bytes per 256 values.
//   [0..128]   128 bytes: ql — lo4 bits packed 2 per byte (nibble-packed)
//   [128..192]  64 bytes: qh — hi2 bits packed 4 per byte (2 bits each)
//   [192..208]  16 bytes: scales — one int8 per 16 elements
//   [208..210]   2 bytes: d — f16 super-block scale
//
// Element i: raw6 = (ql[i/2] >> 4*(i&1)) & 0xF | (((qh[i/4] >> 2*(i%4)) & 3) << 4)
//            w[i] = d * scales[i/16] * (raw6 - 32)
//
// Dot product with Q8_K activation `q8k`:
//   out[r] = Σ_blocks d_w * d_y * Σ_{g=0..15} scales[g] * dot_g
//   where dot_g = Σ_{i in g*16..(g+1)*16} (raw6[i] - 32) * q8k_q[i]
//
// The -(raw6 - 32) sign matches llama.cpp's `ggml_vec_dot_q6_K_q8_K`.
// No `mins` term (Q6_K doesn't have per-group mins — it's symmetric around 32).

/// Q6_K super-block size in bytes (re-export of the wire-format constant).
const Q6K_BLOCK_BYTES: usize = larql_models::quant::ggml::Q6_K_BLOCK_BYTES;

/// Scalar reference: Q6_K weights × Q8_K activation matvec.
/// Correctness oracle for the NEON implementation below.
pub fn q6k_q8k_matvec_scalar(
    out: &mut [f32],
    q8k_x: &Q8KActivation,
    w: &[u8],
    rows: usize,
    cols: usize,
) {
    debug_assert_eq!(cols % ELEMS_PER_BLOCK, 0);
    let n_blocks = cols / ELEMS_PER_BLOCK;
    let row_bytes = n_blocks * Q6K_BLOCK_BYTES;
    for v in out.iter_mut() {
        *v = 0.0;
    }
    if rows == 0 || cols == 0 || w.len() < rows * row_bytes {
        return;
    }
    for (r, out_r) in out.iter_mut().enumerate().take(rows) {
        let row_base = r * row_bytes;
        let mut acc = 0.0f32;
        for sb in 0..n_blocks {
            let block = &w[row_base + sb * Q6K_BLOCK_BYTES..];
            let ql = &block[0..128];
            let qh = &block[128..192];
            let sc = &block[192..208]; // 16 × int8
            let d_w = f16_to_f32(u16::from_le_bytes([block[208], block[209]]));
            let d_y = q8k_x.d[sb];
            let q8_base = sb * ELEMS_PER_BLOCK;
            let q8_qs = &q8k_x.qs[q8_base..q8_base + ELEMS_PER_BLOCK];

            let mut sum1: i32 = 0;
            for (g, scale_byte) in sc.iter().enumerate().take(16usize) {
                // 16-element group g, using scale sc[g].
                let scale = *scale_byte as i8 as i32;
                let mut dot_g: i32 = 0;
                for k in 0..16usize {
                    let i = g * 16 + k;
                    let lo4 = if i & 1 == 0 {
                        (ql[i / 2] & 0x0F) as i32
                    } else {
                        ((ql[i / 2] >> 4) & 0x0F) as i32
                    };
                    let hi2 = ((qh[i / 4] >> (2 * (i % 4))) & 0x03) as i32;
                    let raw6 = lo4 | (hi2 << 4);
                    let w_i = raw6 - 32;
                    dot_g += w_i * q8_qs[i] as i32;
                }
                sum1 += scale * dot_g;
            }
            acc += d_w * d_y * sum1 as f32;
        }
        *out_r = acc;
    }
}

/// NEON-accelerated Q6_K × Q8_K matvec for `aarch64`.
///
/// Per 16-element scale group:
/// 1. Vectorised dequant: 8 ql bytes → lo4[16] via nibble-unpack + vzip.
///    4 qh bytes → hi2[16] via byte-replicate + vshlq_s8 + mask.
///    raw6 = lo4 | (hi2 << 4); signed = raw6 - 32 → int8.
/// 2. One SDOT over the 16 int8 weight × int8 activation products.
/// 3. scale * dot_g accumulated into sum1.
///
/// Final: acc += d_w * d_y * sum1.
#[cfg(all(target_arch = "aarch64", target_feature = "neon"))]
pub fn q6k_q8k_matvec_neon(
    out: &mut [f32],
    q8k_x: &Q8KActivation,
    w: &[u8],
    rows: usize,
    cols: usize,
) {
    use std::arch::aarch64::*;

    debug_assert_eq!(cols % ELEMS_PER_BLOCK, 0);
    let n_blocks = cols / ELEMS_PER_BLOCK;
    let row_bytes = n_blocks * Q6K_BLOCK_BYTES;
    for v in out.iter_mut() {
        *v = 0.0;
    }
    if rows == 0 || cols == 0 || w.len() < rows * row_bytes {
        return;
    }

    // Shift-right pattern for hi2 extraction: 0, -2, -4, -6 repeated 4×.
    // vshlq_s8 with negative b shifts right: out[i] = a[i] >> (-b[i]).
    const SHIFT_RIGHT: [i8; 16] = [0, -2, -4, -6, 0, -2, -4, -6, 0, -2, -4, -6, 0, -2, -4, -6];
    let shift_v = unsafe { vld1q_s8(SHIFT_RIGHT.as_ptr()) };
    let mask_0f = unsafe { vdupq_n_u8(0x0F) };
    let mask_03 = unsafe { vdupq_n_u8(0x03) };
    let sub32 = unsafe { vdupq_n_s8(32) };

    // No software prefetch — see q4k_q8k_matvec_neon for the rationale.
    for (r, out_r) in out.iter_mut().enumerate().take(rows) {
        let row_base = r * row_bytes;
        let mut acc = 0.0f32;
        for sb in 0..n_blocks {
            let block = &w[row_base + sb * Q6K_BLOCK_BYTES..];
            let ql_base = block.as_ptr();
            let qh_base = unsafe { block.as_ptr().add(128) };
            let sc_base = unsafe { block.as_ptr().add(192) as *const i8 };
            let d_w = f16_to_f32(u16::from_le_bytes([block[208], block[209]]));
            let d_y = q8k_x.d[sb];
            let q8_base = sb * ELEMS_PER_BLOCK;
            let q8_ptr = q8k_x.qs.as_ptr();

            let mut sum1: i32 = 0;

            for g in 0..16usize {
                // Scale group g covers elements g*16..(g+1)*16.
                // ql bytes for group g: ql[g*8..(g+1)*8] (8 bytes → 16 nibbles).
                // qh bytes for group g: qh[g*4..(g+1)*4] (4 bytes → 16 × 2-bit).
                let ql_g = unsafe { ql_base.add(g * 8) };
                let qh_g = unsafe { qh_base.add(g * 4) };
                let q8_g = unsafe { q8_ptr.add(q8_base + g * 16) };
                let scale = unsafe { *sc_base.add(g) as i32 };

                // ── Lo4 extraction (8 ql bytes → 16 uint4 values, in element order) ──
                // ql_v[j] holds lo4 of element 2j (low nibble) and 2j+1 (high nibble).
                let ql_v = unsafe { vld1_u8(ql_g) };
                let lo4_even = unsafe { vand_u8(ql_v, vget_low_u8(mask_0f)) }; // elements 0,2,4,...,14
                let lo4_odd = unsafe { vshr_n_u8(ql_v, 4) }; // elements 1,3,5,...,15
                                                             // Interleave to restore element order: [e0,e1,e2,...,e15].
                let zip = unsafe { vzip_u8(lo4_even, lo4_odd) };
                let lo4_v = unsafe { vcombine_u8(zip.0, zip.1) }; // uint8x16_t

                // ── Hi2 extraction (4 qh bytes → 16 uint2 values) ──
                // Each qh byte j holds hi2 for elements 4j+0..4j+3 in bits 0-1,2-3,4-5,6-7.
                // Build a 16-byte vector with each qh byte replicated 4 times, then
                // shift right by [0,2,4,6, 0,2,4,6, ...] and mask to 2 bits.
                let (q0, q1, q2, q3) = unsafe {
                    (
                        (*qh_g) as u32 * 0x01010101u32,
                        (*qh_g.add(1)) as u32 * 0x01010101u32,
                        (*qh_g.add(2)) as u32 * 0x01010101u32,
                        (*qh_g.add(3)) as u32 * 0x01010101u32,
                    )
                };
                let qh_rep: uint8x16_t = unsafe {
                    vreinterpretq_u8_u32(vcombine_u32(
                        vreinterpret_u32_u64(vcreate_u64((q0 as u64) | ((q1 as u64) << 32))),
                        vreinterpret_u32_u64(vcreate_u64((q2 as u64) | ((q3 as u64) << 32))),
                    ))
                };
                // Variable right-shift then mask to 2 bits.
                let hi2_v = unsafe {
                    vandq_u8(
                        vreinterpretq_u8_s8(vshlq_s8(vreinterpretq_s8_u8(qh_rep), shift_v)),
                        mask_03,
                    )
                };

                // ── Combine → signed int8 weight values ──
                // raw6 = lo4 | (hi2 << 4) ∈ [0..63]; signed = raw6 - 32 ∈ [-32..31].
                let hi2_shifted = unsafe { vshlq_n_u8(hi2_v, 4) };
                let combined = unsafe { vorrq_u8(lo4_v, hi2_shifted) };
                let q6_raw: int8x16_t = unsafe { vsubq_s8(vreinterpretq_s8_u8(combined), sub32) };

                // ── SDOT: 16 × (q6_raw[i] * q8k[i]) → 4 partial i32 sums ──
                let q8_v = unsafe { vld1q_s8(q8_g) };
                let dot_v = unsafe { sdot_acc(vdupq_n_s32(0), q6_raw, q8_v) };
                let dot = unsafe { vaddvq_s32(dot_v) };

                sum1 += scale * dot;
            }

            acc += d_w * d_y * sum1 as f32;
        }
        *out_r = acc;
    }
}

/// Public entry point: dispatches to NEON on aarch64, scalar elsewhere.
/// `w` is a Q6_K weight matrix of `rows` rows × `cols` columns.
/// `q8k_x` is the pre-quantised activation vector (`cols` elements).
pub fn q6k_q8k_matvec_into(
    out: &mut [f32],
    q8k_x: &Q8KActivation,
    w: &[u8],
    rows: usize,
    cols: usize,
) {
    #[cfg(all(target_arch = "aarch64", target_feature = "neon"))]
    {
        q6k_q8k_matvec_neon(out, q8k_x, w, rows, cols);
        return;
    }
    #[allow(unreachable_code)]
    q6k_q8k_matvec_scalar(out, q8k_x, w, rows, cols);
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cpu::ops::q4_common::{q4k_matvec_into, quantize_q4_k, quantize_q6_k};

    /// Q8_K round-trip should reconstruct within 0.5% of absmax (1 LSB on
    /// the 127-step scale).  Sums must equal the literal i32 sums of the
    /// quantised values per sub-block.
    #[test]
    fn q8k_quantize_round_trip_within_quant_step() {
        let x: Vec<f32> = (0..256).map(|i| (i as f32 / 128.0 - 1.0) * 5.0).collect();
        let q = quantize_x_to_q8k(&x);
        assert_eq!(q.qs.len(), 256);
        assert_eq!(q.d.len(), 1);
        assert_eq!(q.sums.len(), 8);

        let amax = x.iter().fold(0.0f32, |a, &v| a.max(v.abs()));
        let step = amax / 127.0;
        for (xv, qv) in x.iter().zip(q.qs.iter()) {
            let recon = q.d[0] * (*qv as f32);
            assert!(
                (xv - recon).abs() < step.max(1e-6),
                "x={xv} recon={recon} step={step}"
            );
        }
        // Sums match the literal sums per sub-block.
        for s in 0..8 {
            let actual: i32 = q.qs[s * 32..(s + 1) * 32].iter().map(|&v| v as i32).sum();
            assert_eq!(actual as i16, q.sums[s]);
        }
    }

    /// Q8_K of all-zeros should produce zero scale + all-zero sums.
    #[test]
    fn q8k_zero_input_clean() {
        let x = vec![0.0f32; 256];
        let q = quantize_x_to_q8k(&x);
        assert_eq!(q.d[0], 0.0);
        assert!(q.qs.iter().all(|&v| v == 0));
        assert!(q.sums.iter().all(|&v| v == 0));
    }

    /// Scalar Q4_K×Q8_K matches the f32-cached path within Q8 quant noise.
    /// Same Q4_K-quantised weights and same f32 activation; one path runs
    /// the f32 dot `q4_common::q4k_matvec_into`, the other quantises x to
    /// Q8_K and runs the integer-dot reference.  Difference should be on
    /// the order of `‖w‖ · ε_q8 · ‖x‖`, well below 1e-3 for typical inputs.
    #[test]
    fn q8k_matvec_matches_f32_cached_within_q8_noise() {
        // Single super-block, single row matrix.
        let cols = 256;
        let rows = 4;
        let x: Vec<f32> = (0..cols).map(|i| (i as f32 * 0.013).sin()).collect();
        let w_f32: Vec<f32> = (0..rows * cols)
            .map(|i| (i as f32 * 0.007).cos() * 0.5)
            .collect();
        let w_q4 = quantize_q4_k(&w_f32);
        assert_eq!(w_q4.len(), rows * 144);

        let mut out_f32 = vec![0.0f32; rows];
        q4k_matvec_into(&mut out_f32, &x, &w_q4, rows, cols);

        let q8 = quantize_x_to_q8k(&x);
        let mut out_q8 = vec![0.0f32; rows];
        q4k_q8k_matvec_scalar(&mut out_q8, &q8, &w_q4, rows, cols);

        // Q8 quantisation step on x is amax/127; downstream noise per
        // output element is ~‖w_row‖₁ · step.  For typical sin-ramp inputs
        // that comes out in the 1e-2 range; tolerate 5e-2 to leave headroom
        // for f16 scale conversion error in d/dmin.
        for r in 0..rows {
            let diff = (out_f32[r] - out_q8[r]).abs();
            assert!(
                diff < 5e-2,
                "row {r}: f32={} q8={} diff={diff}",
                out_f32[r],
                out_q8[r]
            );
        }
    }

    /// Multi-block matrix: hidden=512 = 2 super-blocks per row.  Stresses
    /// the per-super-block aggregation (`acc += ...` summed over 2+ blocks).
    #[test]
    fn q8k_matvec_multi_block_within_noise() {
        let cols = 512; // 2 super-blocks
        let rows = 16;
        let x: Vec<f32> = (0..cols).map(|i| (i as f32 * 0.011).cos() * 2.0).collect();
        let w_f32: Vec<f32> = (0..rows * cols)
            .map(|i| (i as f32 * 0.009).sin() * 0.3)
            .collect();
        let w_q4 = quantize_q4_k(&w_f32);

        let mut out_f32 = vec![0.0f32; rows];
        q4k_matvec_into(&mut out_f32, &x, &w_q4, rows, cols);

        let q8 = quantize_x_to_q8k(&x);
        let mut out_q8 = vec![0.0f32; rows];
        q4k_q8k_matvec_scalar(&mut out_q8, &q8, &w_q4, rows, cols);

        for r in 0..rows {
            let diff = (out_f32[r] - out_q8[r]).abs();
            assert!(
                diff < 8e-2,
                "row {r}: f32={} q8={} diff={diff}",
                out_f32[r],
                out_q8[r]
            );
        }
    }

    /// NEON kernel must be bit-identical to the scalar Q8_K reference on
    /// aarch64 — both implement the same i32 dot math.  Different inputs
    /// from the noise tests above to catch byte-ordering / lane-mapping
    /// bugs that happen to vanish on regular ramps.
    #[cfg(all(target_arch = "aarch64", target_feature = "neon"))]
    #[test]
    fn q8k_matvec_neon_matches_scalar_bit_exact() {
        let cols = 1024; // 4 super-blocks — exercises sb-loop + g-loop
        let rows = 7; // odd row count — exercises tail handling
                      // Use a non-symmetric, non-monotonic input so any lane/byte-swap
                      // bug can't accidentally produce the right sum.
        let x: Vec<f32> = (0..cols)
            .map(|i| {
                let f = i as f32;
                ((f * 0.0173).sin() * 1.7 + (f * 0.041).cos() * 0.9) * 1.3
            })
            .collect();
        let w_f32: Vec<f32> = (0..rows * cols)
            .map(|i| {
                let f = i as f32;
                ((f * 0.013).cos() * 0.4 - (f * 0.027).sin() * 0.2) * 0.6
            })
            .collect();
        let w_q4 = quantize_q4_k(&w_f32);
        let q8 = quantize_x_to_q8k(&x);

        let mut out_scalar = vec![0.0f32; rows];
        let mut out_neon = vec![0.0f32; rows];
        q4k_q8k_matvec_scalar(&mut out_scalar, &q8, &w_q4, rows, cols);
        q4k_q8k_matvec_neon(&mut out_neon, &q8, &w_q4, rows, cols);

        for r in 0..rows {
            assert_eq!(
                out_scalar[r].to_bits(),
                out_neon[r].to_bits(),
                "row {r}: scalar={} neon={} diff={}",
                out_scalar[r],
                out_neon[r],
                (out_scalar[r] - out_neon[r]).abs()
            );
        }
    }

    /// `quantize_x_to_q8k_into` must produce the same `qs`, `d`, `sums` as
    /// the allocating `quantize_x_to_q8k` for any well-sized input — both
    /// also handle resize correctly when reused across different sizes.
    #[test]
    fn q8k_in_place_matches_alloc_version() {
        let x: Vec<f32> = (0..512).map(|i| (i as f32 * 0.013).sin() * 3.0).collect();
        let alloc_q = quantize_x_to_q8k(&x);

        let mut buf = Q8KActivation::with_capacity(512);
        quantize_x_to_q8k_into(&mut buf, &x);

        assert_eq!(buf.qs, alloc_q.qs);
        assert_eq!(buf.d, alloc_q.d);
        assert_eq!(buf.sums, alloc_q.sums);

        // Resize-on-reuse: quantise smaller input into the same buffer.
        let x2: Vec<f32> = (0..256).map(|i| (i as f32 * 0.021).cos()).collect();
        let alloc_q2 = quantize_x_to_q8k(&x2);
        quantize_x_to_q8k_into(&mut buf, &x2);
        assert_eq!(buf.qs.len(), 256);
        assert_eq!(buf.d.len(), 1);
        assert_eq!(buf.sums.len(), 8);
        assert_eq!(buf.qs, alloc_q2.qs);
        assert_eq!(buf.d, alloc_q2.d);
        assert_eq!(buf.sums, alloc_q2.sums);
    }

    /// 2-row matvec must produce bit-exact outputs equal to the single-row
    /// kernel for the same input — the dot math is identical, only the
    /// instruction scheduling differs.  Test on both even and odd row
    /// counts so the tail-handling path is exercised.
    #[cfg(all(target_arch = "aarch64", target_feature = "neon"))]
    #[test]
    fn q8k_matvec_2row_matches_single_row_bit_exact() {
        for &rows in &[2usize, 4, 7, 11, 16, 17] {
            let cols = 1024;
            let x: Vec<f32> = (0..cols)
                .map(|i| (i as f32 * 0.0173).sin() * 1.7 + (i as f32 * 0.041).cos() * 0.9)
                .collect();
            let w_f32: Vec<f32> = (0..rows * cols)
                .map(|i| (i as f32 * 0.013).cos() * 0.4 - (i as f32 * 0.027).sin() * 0.2)
                .collect();
            let w_q4 = quantize_q4_k(&w_f32);
            let q8 = quantize_x_to_q8k(&x);

            let mut out_single = vec![0.0f32; rows];
            let mut out_2row = vec![0.0f32; rows];
            q4k_q8k_matvec_neon(&mut out_single, &q8, &w_q4, rows, cols);
            q4k_q8k_matvec_neon_2row(&mut out_2row, &q8, &w_q4, rows, cols);

            for r in 0..rows {
                assert_eq!(
                    out_single[r].to_bits(),
                    out_2row[r].to_bits(),
                    "rows={rows} r={r}: single={} 2row={} diff={}",
                    out_single[r],
                    out_2row[r],
                    (out_single[r] - out_2row[r]).abs()
                );
            }
        }
    }

    /// Fused gate+up must produce bit-exact outputs equal to two separate
    /// matvec calls — both compile down to the same i32 dot math; only the
    /// instruction interleaving differs.
    #[test]
    fn q8k_gate_up_fused_matches_separate_matvecs() {
        let cols = 1024;
        let rows = 11;
        let x: Vec<f32> = (0..cols)
            .map(|i| (i as f32 * 0.0151).sin() * 1.4 + (i as f32 * 0.029).cos() * 0.7)
            .collect();
        let g_f32: Vec<f32> = (0..rows * cols)
            .map(|i| (i as f32 * 0.011).cos() * 0.4 - (i as f32 * 0.027).sin() * 0.2)
            .collect();
        let u_f32: Vec<f32> = (0..rows * cols)
            .map(|i| (i as f32 * 0.013).sin() * 0.3 + (i as f32 * 0.041).cos() * 0.5)
            .collect();
        let g_w = quantize_q4_k(&g_f32);
        let u_w = quantize_q4_k(&u_f32);
        let q8 = quantize_x_to_q8k(&x);

        let mut g_sep = vec![0.0f32; rows];
        let mut u_sep = vec![0.0f32; rows];
        q4k_q8k_matvec_into(&mut g_sep, &q8, &g_w, rows, cols);
        q4k_q8k_matvec_into(&mut u_sep, &q8, &u_w, rows, cols);

        let mut g_fused = vec![0.0f32; rows];
        let mut u_fused = vec![0.0f32; rows];
        q4k_q8k_gate_up_into(&mut g_fused, &mut u_fused, &q8, &g_w, &u_w, rows, cols);

        for r in 0..rows {
            assert_eq!(
                g_sep[r].to_bits(),
                g_fused[r].to_bits(),
                "gate row {r}: sep={} fused={}",
                g_sep[r],
                g_fused[r]
            );
            assert_eq!(
                u_sep[r].to_bits(),
                u_fused[r].to_bits(),
                "up row {r}: sep={} fused={}",
                u_sep[r],
                u_fused[r]
            );
        }
    }

    /// Empty / degenerate dims should produce zeros without panic.
    #[test]
    fn q8k_matvec_zero_dims_returns_zero() {
        let q = Q8KActivation {
            qs: vec![],
            d: vec![],
            sums: vec![],
        };
        let mut out = vec![1.0f32; 4];
        q4k_q8k_matvec_scalar(&mut out, &q, &[], 4, 0);
        assert!(out.iter().all(|&v| v == 0.0));
    }

    /// Misaligned col count (not a multiple of 256) should fail safely
    /// (leave caller-visible zeros, like the scalar `q4k_matvec_into`).
    #[test]
    fn q8k_matvec_short_weight_buffer_returns_zero() {
        let cols = 256;
        let rows = 2;
        let x = vec![0.5f32; cols];
        let q = quantize_x_to_q8k(&x);
        let w = vec![0u8; 144]; // only enough for 1 row, but rows=2
        let mut out = vec![1.0f32; rows];
        q4k_q8k_matvec_scalar(&mut out, &q, &w, rows, cols);
        assert!(out.iter().all(|&v| v == 0.0));
    }

    #[test]
    fn q6k_q8k_matvec_matches_q6k_f32_dispatch_within_noise() {
        let cols = 512;
        let rows = 5;
        let x: Vec<f32> = (0..cols).map(|i| (i as f32 * 0.017).sin() * 1.5).collect();
        let w_f32: Vec<f32> = (0..rows * cols)
            .map(|i| (i as f32 * 0.006).cos() * 0.7)
            .collect();
        let w_q6 = quantize_q6_k(&w_f32);

        let f32_path = crate::cpu::ops::q6k_matvec::dispatch(&w_q6, &x, rows, cols);
        let q8 = quantize_x_to_q8k(&x);
        let mut q8_path = vec![0.0f32; rows];
        q6k_q8k_matvec_scalar(&mut q8_path, &q8, &w_q6, rows, cols);

        for r in 0..rows {
            let diff = (f32_path[r] - q8_path[r]).abs();
            assert!(
                diff < 1.2e-1,
                "row {r}: f32={} q8={} diff={diff}",
                f32_path[r],
                q8_path[r]
            );
        }
    }

    #[test]
    fn q6k_q8k_public_entrypoint_matches_scalar() {
        let cols = 256;
        let rows = 3;
        let x: Vec<f32> = (0..cols).map(|i| (i as f32 * 0.031).cos()).collect();
        let w_f32: Vec<f32> = (0..rows * cols)
            .map(|i| (i as f32 * 0.011).sin() * 0.4)
            .collect();
        let w_q6 = quantize_q6_k(&w_f32);
        let q8 = quantize_x_to_q8k(&x);
        let mut scalar = vec![0.0f32; rows];
        let mut dispatched = vec![0.0f32; rows];

        q6k_q8k_matvec_scalar(&mut scalar, &q8, &w_q6, rows, cols);
        q6k_q8k_matvec_into(&mut dispatched, &q8, &w_q6, rows, cols);

        for (a, b) in scalar.iter().zip(dispatched.iter()) {
            assert_eq!(a.to_bits(), b.to_bits());
        }
    }

    #[test]
    fn q6k_q8k_zero_dims_and_short_weights_zero_output() {
        let q = Q8KActivation::with_capacity(0);
        let mut out = vec![1.0f32; 4];
        q6k_q8k_matvec_scalar(&mut out, &q, &[], 4, 0);
        assert_eq!(out, vec![0.0f32; 4]);

        let x = vec![1.0f32; 256];
        let q = quantize_x_to_q8k(&x);
        let mut out = vec![1.0f32; 2];
        q6k_q8k_matvec_scalar(&mut out, &q, &vec![0u8; 210], 2, 256);
        assert_eq!(out, vec![0.0f32; 2]);
    }

    /// AVX2 must produce bit-identical output to the scalar reference.
    #[cfg(target_arch = "x86_64")]
    #[test]
    fn q8k_matvec_avx2_matches_scalar() {
        if !is_x86_feature_detected!("avx2") {
            return; // Skip on hardware without AVX2.
        }
        let cols = 1024;
        let rows = 7;
        let x: Vec<f32> = (0..cols)
            .map(|i| {
                let f = i as f32;
                ((f * 0.0173).sin() * 1.7 + (f * 0.041).cos() * 0.9) * 1.3
            })
            .collect();
        let w_f32: Vec<f32> = (0..rows * cols)
            .map(|i| {
                let f = i as f32;
                ((f * 0.013).cos() * 0.4 - (f * 0.027).sin() * 0.2) * 0.6
            })
            .collect();
        let w_q4 = quantize_q4_k(&w_f32);
        let q8 = quantize_x_to_q8k(&x);

        let mut out_scalar = vec![0.0f32; rows];
        let mut out_avx2 = vec![0.0f32; rows];
        q4k_q8k_matvec_scalar(&mut out_scalar, &q8, &w_q4, rows, cols);
        unsafe { q4k_q8k_matvec_avx2(&mut out_avx2, &q8, &w_q4, rows, cols) };

        for r in 0..rows {
            assert_eq!(
                out_scalar[r].to_bits(),
                out_avx2[r].to_bits(),
                "row {r}: scalar={} avx2={} diff={}",
                out_scalar[r],
                out_avx2[r],
                (out_scalar[r] - out_avx2[r]).abs()
            );
        }
    }
}
