use super::common::{
    unpack_scales_mins, BLOCK_BYTES, ELEMS_PER_BLOCK, SUBBLOCKS_PER_BLOCK, SUBBLOCK_SIZE,
};
use super::q8k_activation::Q8KActivation;
use crate::cpu::ops::q4_common::f16_to_f32;
use crate::cpu::ops::KernelShapeError;

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
) -> Result<(), KernelShapeError> {
    KernelShapeError::check(
        "q4k_q8k_matvec_scalar",
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
    Ok(())
}
