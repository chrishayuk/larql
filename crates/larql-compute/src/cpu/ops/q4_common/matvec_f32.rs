use super::dual::q4_dual_dot_32;
use super::f16_to_f32;
use crate::cpu::ops::KernelShapeError;

/// Direct Q4_K matrix-vector product: `out = W · x` where `W` is the raw
/// Q4_K byte stream (`rows × cols` weights, 144 bytes per 256 elements).
///
/// Decodes nibbles + per-sub-block scales/mins on the fly while
/// accumulating the dot product — avoids the f32 dequant cache that
/// quadruples the bandwidth bill.  At Gemma 4 26B-A4B sizes
/// (`hidden=2816`, `inter=704`, ~7.9 MB f32 per row otherwise) this drops
/// per-matmul bandwidth pressure from ~8 MB → ~2 MB and should land ~3–4×
/// faster than `dequantize_q4_k` + BLAS sgemv on a same-sized f32 view.
///
/// Math (matches `dequantize_q4_k`'s `out = sc * q - mn` per-element form):
///
/// ```text
/// for each super-block sb of 256 elements (8 sub-blocks of 32 each):
///   for each sub-block subblk in [0..8):
///     sc = d    * scales[subblk]
///     mn = dmin * mins[subblk]
///     dot = Σ  q_l · x[base + l]    (l in 0..32)
///     sumx = Σ x[base + l]          (precomputed once across all rows)
///     acc += sc * dot − mn * sumx
/// out[r] = acc
/// ```
///
/// `sumx` precomputation: x is shared across rows, so its per-sub-block
/// sum is row-invariant.  Computing it once outside the row loop saves
/// `rows × 8 · n_blocks` redundant sums.
///
/// Refuses a shape mismatch or a Q4_K layout error (input too short, or
/// `cols` not a multiple of 256) with a [`KernelShapeError`] naming the
/// operands, leaving `out` untouched.
///
/// Caller layout: `w.len() == rows * (cols / 256) * 144` bytes.
pub fn q4k_matvec_into(
    out: &mut [f32],
    x: &[f32],
    w: &[u8],
    rows: usize,
    cols: usize,
) -> Result<(), KernelShapeError> {
    const BLOCK_BYTES: usize = 144;
    const ELEMS_PER_BLOCK: usize = 256;
    KernelShapeError::check(
        "q4k_matvec_into",
        out.len(),
        rows,
        x.len(),
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

    // Precompute per-sub-block sum_x (one f32 per 32-element chunk of x).
    // 2-byte stride per (sb, subblk) pair lets us index by `sb * 8 + subblk`.
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

    // Row-parallel. Decode rows are independent and the typical matvec
    // shape this gets called with (Gemma-3-4B: 2560×2560 to 8192×2560
    // for Q4_K) is large enough to amortise rayon's join overhead by
    // 100×+. Empirically on M3 Max this drops a 2560-row decode from
    // ~70ms → ~10ms (≈ 7× across 11 perf cores).
    let sum_x_ref = &sum_x[..];
    let w_ref = w;
    let x_ref = x;
    // par_chunks_mut(CHUNK_ROWS) instead of per-row par_iter_mut: each
    // task processes a contiguous block of rows sequentially. Cuts the
    // number of work-stealing units from `rows` (10K+) down to
    // ~rows/CHUNK_ROWS, reducing scheduler overhead while keeping enough
    // granularity for the 11 perf cores on M3 Max to load-balance.
    const CHUNK_ROWS: usize = 32;
    crate::cpu::spin_pool::par_chunks_mut(out, CHUNK_ROWS, |chunk_idx, chunk_slots| {
        let row_base_chunk = chunk_idx * CHUNK_ROWS;
        for (local_r, out_slot) in chunk_slots.iter_mut().enumerate() {
            let r = row_base_chunk + local_r;
            if r >= rows {
                break;
            }
            let row_base = r * row_bytes;
            let mut acc = 0.0f32;
            for sb in 0..n_blocks {
                acc += process_q4k_superblock(w_ref, x_ref, sum_x_ref, row_base, sb);
            }
            *out_slot = acc;
        }
    });
    Ok(())
}

/// Per-super-block dot contribution for a Q4_K row. Returned scalar
/// is the super-block's contribution to the row's dot product.
/// Inlined into both `q4k_matvec_into`'s 2-super-block-unrolled outer
/// loop and `q4k_dual_matvec_into`'s outer loop (which keeps its
/// per-matrix accumulator separate so it doesn't get the 2-acc
/// scheduling boost, but trades that for the gate+up x-locality
/// already in place).
#[inline(always)]
fn process_q4k_superblock(w: &[u8], x: &[f32], sum_x: &[f32], row_base: usize, sb: usize) -> f32 {
    const BLOCK_BYTES: usize = 144;
    const ELEMS_PER_BLOCK: usize = 256;

    let block = &w[row_base + sb * BLOCK_BYTES..row_base + (sb + 1) * BLOCK_BYTES];
    let d = f16_to_f32(u16::from_le_bytes([block[0], block[1]]));
    let dmin = f16_to_f32(u16::from_le_bytes([block[2], block[3]]));
    let p = &block[4..16];
    let mut scales = [0u8; 8];
    let mut mins = [0u8; 8];
    for j in 0..4 {
        scales[j] = p[j] & 0x3F;
        mins[j] = p[j + 4] & 0x3F;
        scales[j + 4] = (p[j + 8] & 0x0F) | ((p[j] >> 6) << 4);
        mins[j + 4] = (p[j + 8] >> 4) | ((p[j + 4] >> 6) << 4);
    }
    let quants = &block[16..144];
    let x_sb_base = sb * ELEMS_PER_BLOCK;

    let mut acc = 0.0f32;
    for g in 0..4 {
        let sb_lo = 2 * g;
        let sb_hi = 2 * g + 1;
        let sc_lo = d * scales[sb_lo] as f32;
        let sc_hi = d * scales[sb_hi] as f32;
        let mn_lo = dmin * mins[sb_lo] as f32;
        let mn_hi = dmin * mins[sb_hi] as f32;
        let chunk = &quants[g * 32..(g + 1) * 32];
        let x_lo_base = x_sb_base + sb_lo * 32;
        let x_hi_base = x_sb_base + sb_hi * 32;
        let sumy_lo = sum_x[sb * 8 + sb_lo];
        let sumy_hi = sum_x[sb * 8 + sb_hi];
        let x_lo = &x[x_lo_base..x_lo_base + 32];
        let x_hi = &x[x_hi_base..x_hi_base + 32];
        let (dot_lo, dot_hi) = q4_dual_dot_32(chunk, x_lo, x_hi);
        acc += sc_lo * dot_lo - mn_lo * sumy_lo;
        acc += sc_hi * dot_hi - mn_hi * sumy_hi;
    }
    acc
}

/// Decode one Q4_K super-block (256 elements, 144 bytes) of row `row_base`
/// into `wf` as full f32 weight values. Per element the dequant is
/// `d * scale[sb] * q - dmin * min[sb]` — identical arithmetic to
/// [`process_q4k_superblock`], but materialised per element so the decoded
/// weights can be reused across many activation columns instead of folded
/// into a single dot. Nibble packing mirrors the matvec: each of the 4
/// 32-byte groups holds sub-block `2g` in the low nibble and `2g+1` in the
/// high nibble.
#[inline(always)]
pub(super) fn decode_q4k_superblock_into(
    w: &[u8],
    row_base: usize,
    sb: usize,
    wf: &mut [f32; 256],
) {
    const BLOCK_BYTES: usize = 144;
    let block = &w[row_base + sb * BLOCK_BYTES..row_base + (sb + 1) * BLOCK_BYTES];
    let d = f16_to_f32(u16::from_le_bytes([block[0], block[1]]));
    let dmin = f16_to_f32(u16::from_le_bytes([block[2], block[3]]));
    let p = &block[4..16];
    let mut scales = [0u8; 8];
    let mut mins = [0u8; 8];
    for j in 0..4 {
        scales[j] = p[j] & 0x3F;
        mins[j] = p[j + 4] & 0x3F;
        scales[j + 4] = (p[j + 8] & 0x0F) | ((p[j] >> 6) << 4);
        mins[j + 4] = (p[j + 8] >> 4) | ((p[j + 4] >> 6) << 4);
    }
    let quants = &block[16..144];
    for g in 0..4 {
        let sb_lo = 2 * g;
        let sb_hi = 2 * g + 1;
        let sc_lo = d * scales[sb_lo] as f32;
        let sc_hi = d * scales[sb_hi] as f32;
        let mn_lo = dmin * mins[sb_lo] as f32;
        let mn_hi = dmin * mins[sb_hi] as f32;
        let chunk = &quants[g * 32..(g + 1) * 32];
        for i in 0..32 {
            wf[sb_lo * 32 + i] = sc_lo * (chunk[i] & 0x0F) as f32 - mn_lo;
            wf[sb_hi * 32 + i] = sc_hi * (chunk[i] >> 4) as f32 - mn_hi;
        }
    }
}

/// Decode one Q6_K super-block (256 elements, 210 bytes) of row `row_base` into
/// `wf` as full f32 weight values — mirrors [`larql_models::quant::ggml`]'s
/// `dequantize_q6_k` per-block math: a 4-bit low nibble (`ql`) plus a 2-bit high
/// part (`qh`), biased by −32, times the per-16-element int8 scale and the f16
/// super-block scale.
#[inline(always)]
pub(super) fn decode_q6k_superblock_into(
    w: &[u8],
    row_base: usize,
    sb: usize,
    wf: &mut [f32; 256],
) {
    const BLOCK_BYTES: usize = 210;
    let block = &w[row_base + sb * BLOCK_BYTES..row_base + (sb + 1) * BLOCK_BYTES];
    let scales = &block[192..208];
    let d = f16_to_f32(u16::from_le_bytes([block[208], block[209]]));
    for (j, &sc_byte) in scales.iter().enumerate() {
        let sc = d * (sc_byte as i8) as f32;
        let vals = larql_models::quant::ggml::q6_k::q6k_subblock_vals(block, j);
        for (i, &v) in vals.iter().enumerate() {
            wf[j * 16 + i] = sc * v as f32;
        }
    }
}

/// Dot of a decoded 256-element f32 weight block against a 256-element f32
/// activation slice — the inner of the amortised k-quant matmul. Dispatches to
/// NEON on aarch64; portable multi-accumulator fallback elsewhere.
#[inline]
pub(super) fn dot_256_f32(wf: &[f32; 256], xs: &[f32]) -> f32 {
    debug_assert!(xs.len() >= 256);
    #[cfg(target_arch = "aarch64")]
    {
        // SAFETY: NEON is in the aarch64 base ISA; `wf` is 256 f32 and `xs` has
        // ≥256 contiguous f32 (the caller slices exactly one super-block).
        unsafe { dot_256_f32_neon(wf, xs) }
    }
    #[cfg(not(target_arch = "aarch64"))]
    {
        dot_256_f32_scalar(wf, xs)
    }
}

/// Portable reference: 8 independent accumulators so the reduction isn't a
/// scalar fp-add chain (Rust f32 add isn't associative). Also the parity oracle
/// for the NEON path. On aarch64 it's reached only from tests (the NEON path
/// serves the lib), so allow it to be otherwise-unused there.
#[cfg_attr(target_arch = "aarch64", allow(dead_code))]
#[inline]
pub(super) fn dot_256_f32_scalar(wf: &[f32; 256], xs: &[f32]) -> f32 {
    let mut acc = [0.0f32; 8];
    for c in 0..256 / 8 {
        for l in 0..8 {
            acc[l] += wf[c * 8 + l] * xs[c * 8 + l];
        }
    }
    acc.iter().sum::<f32>()
}

/// NEON: 8 `float32x4` accumulators (32 elems/iter × 8 iters = 256). Eight
/// independent accumulators keep the FMA units busy across M-series' ~4-cycle
/// FMA latency (one accumulator would serialise at ~25% of peak).
#[cfg(target_arch = "aarch64")]
#[inline]
pub(super) unsafe fn dot_256_f32_neon(wf: &[f32; 256], xs: &[f32]) -> f32 {
    use std::arch::aarch64::*;
    let wp = wf.as_ptr();
    let xp = xs.as_ptr();
    let mut a0 = vdupq_n_f32(0.0);
    let mut a1 = vdupq_n_f32(0.0);
    let mut a2 = vdupq_n_f32(0.0);
    let mut a3 = vdupq_n_f32(0.0);
    let mut a4 = vdupq_n_f32(0.0);
    let mut a5 = vdupq_n_f32(0.0);
    let mut a6 = vdupq_n_f32(0.0);
    let mut a7 = vdupq_n_f32(0.0);
    let mut i = 0usize;
    while i < 256 {
        a0 = vfmaq_f32(a0, vld1q_f32(wp.add(i)), vld1q_f32(xp.add(i)));
        a1 = vfmaq_f32(a1, vld1q_f32(wp.add(i + 4)), vld1q_f32(xp.add(i + 4)));
        a2 = vfmaq_f32(a2, vld1q_f32(wp.add(i + 8)), vld1q_f32(xp.add(i + 8)));
        a3 = vfmaq_f32(a3, vld1q_f32(wp.add(i + 12)), vld1q_f32(xp.add(i + 12)));
        a4 = vfmaq_f32(a4, vld1q_f32(wp.add(i + 16)), vld1q_f32(xp.add(i + 16)));
        a5 = vfmaq_f32(a5, vld1q_f32(wp.add(i + 20)), vld1q_f32(xp.add(i + 20)));
        a6 = vfmaq_f32(a6, vld1q_f32(wp.add(i + 24)), vld1q_f32(xp.add(i + 24)));
        a7 = vfmaq_f32(a7, vld1q_f32(wp.add(i + 28)), vld1q_f32(xp.add(i + 28)));
        i += 32;
    }
    let s = vaddq_f32(
        vaddq_f32(vaddq_f32(a0, a1), vaddq_f32(a2, a3)),
        vaddq_f32(vaddq_f32(a4, a5), vaddq_f32(a6, a7)),
    );
    vaddvq_f32(s)
}
