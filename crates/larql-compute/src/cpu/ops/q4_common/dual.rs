use super::f16_to_f32;
use crate::cpu::ops::KernelShapeError;

/// Fused two-weight Q4_K matvec sharing one input vector.
///
/// `out_a[N] = W_a[N, K] · x[K]`, `out_b[N] = W_b[N, K] · x[K]`.
/// Both weight matrices must have identical `(rows, cols)`. The decode
/// step's gate+up projections fit this contract exactly: same shape
/// `[intermediate, hidden]`, same `h_in` row.
///
/// Win vs two sequential `q4k_matvec_into` calls:
/// * `sum_x` is precomputed once (saves 0.1% per call, negligible)
/// * The expensive part: each rayon worker decodes both W_a and W_b
///   for its row range against the same `x`. `x` (10 KB for Gemma 3
///   4B hidden=2560) stays hot in L1 across both decodes — a
///   sequential pair re-streams it from L2/L3.
/// * Weight reads are independent and dominate bandwidth (~30 MB
///   total for 8192-row Q4_K). Total bandwidth doesn't change; just
///   x re-stream.
///
/// Measured savings: ~3-5% step on Gemma 3 4B's gate+up pair.
pub fn q4k_dual_matvec_into(
    out_a: &mut [f32],
    out_b: &mut [f32],
    x: &[f32],
    w_a: &[u8],
    w_b: &[u8],
    rows: usize,
    cols: usize,
) -> Result<(), KernelShapeError> {
    const BLOCK_BYTES: usize = 144;
    const ELEMS_PER_BLOCK: usize = 256;
    KernelShapeError::check(
        "q4k_dual_matvec_into (a)",
        out_a.len(),
        rows,
        x.len(),
        cols,
        w_a.len(),
        ELEMS_PER_BLOCK,
        BLOCK_BYTES,
    )?;
    KernelShapeError::check(
        "q4k_dual_matvec_into (b)",
        out_b.len(),
        rows,
        x.len(),
        cols,
        w_b.len(),
        ELEMS_PER_BLOCK,
        BLOCK_BYTES,
    )?;
    if rows == 0 || cols == 0 {
        out_a.fill(0.0);
        out_b.fill(0.0);
        return Ok(());
    }
    let n_blocks = cols / ELEMS_PER_BLOCK;
    let row_bytes = n_blocks * BLOCK_BYTES;

    // Precompute sum_x once.
    let n_subblocks = n_blocks * 8;
    let mut sum_x: Vec<f32> = Vec::with_capacity(n_subblocks);
    for sub in 0..n_subblocks {
        let chunk = &x[sub * 32..(sub + 1) * 32];
        let mut s = 0.0f32;
        for &v in chunk {
            s += v;
        }
        sum_x.push(s);
    }

    // Row-parallel — same outer structure as `q4k_matvec_into` but
    // each worker computes both outputs for its assigned row index.
    // Zip `out_a` and `out_b` so rayon stays simple and the two
    // writes hit different cache lines per row.
    let sum_x_ref = &sum_x[..];
    let w_a_ref = w_a;
    let w_b_ref = w_b;
    let x_ref = x;
    // Fewer-but-larger work units (CHUNK_ROWS rows each) reduce
    // work-stealing overhead; same rationale as `q4k_matvec_into`.
    const CHUNK_ROWS: usize = 32;
    crate::cpu::spin_pool::par_chunks_mut2(
        out_a,
        out_b,
        CHUNK_ROWS,
        |chunk_idx, chunk_a, chunk_b| {
            let row_base_chunk = chunk_idx * CHUNK_ROWS;
            for (local_r, (out_a_slot, out_b_slot)) in
                chunk_a.iter_mut().zip(chunk_b.iter_mut()).enumerate()
            {
                let r = row_base_chunk + local_r;
                if r >= rows {
                    break;
                }
                let row_base = r * row_bytes;
                let mut acc_a = 0.0f32;
                let mut acc_b = 0.0f32;
                for sb in 0..n_blocks {
                    let blk_a =
                        &w_a_ref[row_base + sb * BLOCK_BYTES..row_base + (sb + 1) * BLOCK_BYTES];
                    let blk_b =
                        &w_b_ref[row_base + sb * BLOCK_BYTES..row_base + (sb + 1) * BLOCK_BYTES];
                    let d_a = f16_to_f32(u16::from_le_bytes([blk_a[0], blk_a[1]]));
                    let dmin_a = f16_to_f32(u16::from_le_bytes([blk_a[2], blk_a[3]]));
                    let d_b = f16_to_f32(u16::from_le_bytes([blk_b[0], blk_b[1]]));
                    let dmin_b = f16_to_f32(u16::from_le_bytes([blk_b[2], blk_b[3]]));
                    let pa = &blk_a[4..16];
                    let pb = &blk_b[4..16];
                    let mut scales_a = [0u8; 8];
                    let mut mins_a = [0u8; 8];
                    let mut scales_b = [0u8; 8];
                    let mut mins_b = [0u8; 8];
                    for j in 0..4 {
                        scales_a[j] = pa[j] & 0x3F;
                        mins_a[j] = pa[j + 4] & 0x3F;
                        scales_a[j + 4] = (pa[j + 8] & 0x0F) | ((pa[j] >> 6) << 4);
                        mins_a[j + 4] = (pa[j + 8] >> 4) | ((pa[j + 4] >> 6) << 4);
                        scales_b[j] = pb[j] & 0x3F;
                        mins_b[j] = pb[j + 4] & 0x3F;
                        scales_b[j + 4] = (pb[j + 8] & 0x0F) | ((pb[j] >> 6) << 4);
                        mins_b[j + 4] = (pb[j + 8] >> 4) | ((pb[j + 4] >> 6) << 4);
                    }
                    let qa = &blk_a[16..144];
                    let qb = &blk_b[16..144];
                    let x_sb_base = sb * ELEMS_PER_BLOCK;

                    for g in 0..4 {
                        let sb_lo = 2 * g;
                        let sb_hi = 2 * g + 1;
                        let sc_a_lo = d_a * scales_a[sb_lo] as f32;
                        let sc_a_hi = d_a * scales_a[sb_hi] as f32;
                        let mn_a_lo = dmin_a * mins_a[sb_lo] as f32;
                        let mn_a_hi = dmin_a * mins_a[sb_hi] as f32;
                        let sc_b_lo = d_b * scales_b[sb_lo] as f32;
                        let sc_b_hi = d_b * scales_b[sb_hi] as f32;
                        let mn_b_lo = dmin_b * mins_b[sb_lo] as f32;
                        let mn_b_hi = dmin_b * mins_b[sb_hi] as f32;
                        let chunk_a = &qa[g * 32..(g + 1) * 32];
                        let chunk_b = &qb[g * 32..(g + 1) * 32];
                        let x_lo_base = x_sb_base + sb_lo * 32;
                        let x_hi_base = x_sb_base + sb_hi * 32;
                        let x_lo = &x_ref[x_lo_base..x_lo_base + 32];
                        let x_hi = &x_ref[x_hi_base..x_hi_base + 32];
                        let sumy_lo = sum_x_ref[sb * 8 + sb_lo];
                        let sumy_hi = sum_x_ref[sb * 8 + sb_hi];

                        // Decode W_a's nibbles against x — x stays hot
                        // because the next call decodes W_b against the
                        // same x slice.
                        let (dot_a_lo, dot_a_hi) = q4_dual_dot_32(chunk_a, x_lo, x_hi);
                        let (dot_b_lo, dot_b_hi) = q4_dual_dot_32(chunk_b, x_lo, x_hi);

                        acc_a += sc_a_lo * dot_a_lo - mn_a_lo * sumy_lo;
                        acc_a += sc_a_hi * dot_a_hi - mn_a_hi * sumy_hi;
                        acc_b += sc_b_lo * dot_b_lo - mn_b_lo * sumy_lo;
                        acc_b += sc_b_hi * dot_b_hi - mn_b_hi * sumy_hi;
                    }
                }
                *out_a_slot = acc_a;
                *out_b_slot = acc_b;
            }
        },
    );
    Ok(())
}

/// 32-element dual nibble dot product: returns
/// `(sum(lo_nibbles[i] * x_lo[i]), sum(hi_nibbles[i] * x_hi[i]))` for
/// the 32 packed nibble pairs in `chunk`.
///
/// Dispatches to a NEON implementation on aarch64 (always available on
/// Apple Silicon) and falls back to scalar everywhere else. The hot
/// path runs ~3-4× the scalar version on M3 Max — 16 NEON FMAs vs 64
/// scalar FMAs per chunk, plus saved nibble-to-f32 widening cost.
#[inline]
pub(super) fn q4_dual_dot_32(chunk: &[u8], x_lo: &[f32], x_hi: &[f32]) -> (f32, f32) {
    #[cfg(target_arch = "aarch64")]
    {
        // SAFETY: NEON is part of the aarch64 base ISA. The slices are
        // guaranteed to be at least 32 elements (chunk) and 32 f32
        // (x_lo/x_hi) by the caller. We only read.
        unsafe { q4_dual_dot_32_neon(chunk, x_lo, x_hi) }
    }
    #[cfg(not(target_arch = "aarch64"))]
    {
        let mut dot_lo = 0.0f32;
        let mut dot_hi = 0.0f32;
        for l in 0..32 {
            let byte = chunk[l];
            let q_lo = (byte & 0x0F) as f32;
            let q_hi = ((byte >> 4) & 0x0F) as f32;
            dot_lo += q_lo * x_lo[l];
            dot_hi += q_hi * x_hi[l];
        }
        (dot_lo, dot_hi)
    }
}

#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
pub(super) unsafe fn q4_dual_dot_32_neon(chunk: &[u8], x_lo: &[f32], x_hi: &[f32]) -> (f32, f32) {
    use core::arch::aarch64::*;
    debug_assert!(chunk.len() >= 32);
    debug_assert!(x_lo.len() >= 32);
    debug_assert!(x_hi.len() >= 32);

    // Load 32 bytes of packed nibble pairs as two u8x16 registers.
    let bytes_0 = vld1q_u8(chunk.as_ptr()); // bytes[0..16]
    let bytes_1 = vld1q_u8(chunk.as_ptr().add(16)); // bytes[16..32]

    // Mask = 0x0F lane-broadcast; lo = byte & 0x0F, hi = byte >> 4.
    let mask = vdupq_n_u8(0x0F);
    let lo_nibs_0 = vandq_u8(bytes_0, mask);
    let lo_nibs_1 = vandq_u8(bytes_1, mask);
    let hi_nibs_0 = vshrq_n_u8::<4>(bytes_0);
    let hi_nibs_1 = vshrq_n_u8::<4>(bytes_1);

    // Eight independent f32x4 accumulators (4 lo + 4 hi). With one
    // accumulator per side the 4 FMAs per chunk would serialise on
    // the same destination register at M3's 4-cycle FMA latency
    // (= 25% of peak). Splitting into 4 lets the 4 FMAs pipeline at
    // 1/cycle, ~4× the inner-loop throughput.
    let mut acc_lo_a = vdupq_n_f32(0.0);
    let mut acc_lo_b = vdupq_n_f32(0.0);
    let mut acc_lo_c = vdupq_n_f32(0.0);
    let mut acc_lo_d = vdupq_n_f32(0.0);
    let mut acc_hi_a = vdupq_n_f32(0.0);
    let mut acc_hi_b = vdupq_n_f32(0.0);
    let mut acc_hi_c = vdupq_n_f32(0.0);
    let mut acc_hi_d = vdupq_n_f32(0.0);

    // Widen a u8x16 of nibbles into four f32x4 lanes, then FMA each
    // into a different accumulator so they pipeline.
    //
    // SAFETY of `xp.add(k)`: caller guarantees x_lo and x_hi each have
    // 32 contiguous f32, and we stop at offset 12 (last load reads
    // [12..16]).
    macro_rules! accumulate_16 {
        ($nibs:expr, $xp:expr, $acc_a:expr, $acc_b:expr, $acc_c:expr, $acc_d:expr) => {{
            let n: uint8x16_t = $nibs;
            let n_lo16 = vmovl_u8(vget_low_u8(n));
            let n_hi16 = vmovl_u8(vget_high_u8(n));
            let n_a = vcvtq_f32_u32(vmovl_u16(vget_low_u16(n_lo16)));
            let n_b = vcvtq_f32_u32(vmovl_u16(vget_high_u16(n_lo16)));
            let n_c = vcvtq_f32_u32(vmovl_u16(vget_low_u16(n_hi16)));
            let n_d = vcvtq_f32_u32(vmovl_u16(vget_high_u16(n_hi16)));
            let xp: *const f32 = $xp;
            let x_a = vld1q_f32(xp);
            let x_b = vld1q_f32(xp.add(4));
            let x_c = vld1q_f32(xp.add(8));
            let x_d = vld1q_f32(xp.add(12));
            $acc_a = vfmaq_f32($acc_a, n_a, x_a);
            $acc_b = vfmaq_f32($acc_b, n_b, x_b);
            $acc_c = vfmaq_f32($acc_c, n_c, x_c);
            $acc_d = vfmaq_f32($acc_d, n_d, x_d);
        }};
    }

    accumulate_16!(
        lo_nibs_0,
        x_lo.as_ptr(),
        acc_lo_a,
        acc_lo_b,
        acc_lo_c,
        acc_lo_d
    );
    accumulate_16!(
        lo_nibs_1,
        x_lo.as_ptr().add(16),
        acc_lo_a,
        acc_lo_b,
        acc_lo_c,
        acc_lo_d
    );
    accumulate_16!(
        hi_nibs_0,
        x_hi.as_ptr(),
        acc_hi_a,
        acc_hi_b,
        acc_hi_c,
        acc_hi_d
    );
    accumulate_16!(
        hi_nibs_1,
        x_hi.as_ptr().add(16),
        acc_hi_a,
        acc_hi_b,
        acc_hi_c,
        acc_hi_d
    );

    // Tree-reduce: (a+b) + (c+d) per side, then horizontal sum.
    let acc_lo = vaddq_f32(vaddq_f32(acc_lo_a, acc_lo_b), vaddq_f32(acc_lo_c, acc_lo_d));
    let acc_hi = vaddq_f32(vaddq_f32(acc_hi_a, acc_hi_b), vaddq_f32(acc_hi_c, acc_hi_d));
    (vaddvq_f32(acc_lo), vaddvq_f32(acc_hi))
}
