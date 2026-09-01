#[cfg(all(target_arch = "aarch64", target_feature = "neon"))]
use super::common::{
    unpack_scales_mins, BLOCK_BYTES, ELEMS_PER_BLOCK, SUBBLOCKS_PER_BLOCK, SUBBLOCK_SIZE,
};
#[cfg(all(target_arch = "aarch64", target_feature = "neon"))]
use super::q8k_activation::Q8KActivation;
#[cfg(all(target_arch = "aarch64", target_feature = "neon"))]
use crate::cpu::ops::q4_common::f16_to_f32;
use crate::cpu::ops::KernelShapeError;

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
pub(super) unsafe fn sdot_acc(
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
) -> Result<(), KernelShapeError> {
    use std::arch::aarch64::*;

    KernelShapeError::check(
        "q4k_q8k_matvec_neon",
        out.len(),
        rows,
        q8k_x.qs.len(),
        cols,
        w.len(),
        ELEMS_PER_BLOCK,
        BLOCK_BYTES,
    )?;
    if rows == 0 || cols == 0 {
        out.fill(0.0);
        return Ok(());
    }
    let n_blocks = cols / ELEMS_PER_BLOCK;
    let row_bytes = n_blocks * BLOCK_BYTES;

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
    Ok(())
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
) -> Result<(), KernelShapeError> {
    use std::arch::aarch64::*;

    KernelShapeError::check(
        "q4k_q8k_matvec_neon_2row",
        out.len(),
        rows,
        q8k_x.qs.len(),
        cols,
        w.len(),
        ELEMS_PER_BLOCK,
        BLOCK_BYTES,
    )?;
    if rows == 0 || cols == 0 {
        out.fill(0.0);
        return Ok(());
    }
    let n_blocks = cols / ELEMS_PER_BLOCK;
    let row_bytes = n_blocks * BLOCK_BYTES;

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
        q4k_q8k_matvec_neon(&mut tail_out, q8k_x, row_w, 1, cols)?;
        out[r] = tail_out[0];
    }
    Ok(())
}
