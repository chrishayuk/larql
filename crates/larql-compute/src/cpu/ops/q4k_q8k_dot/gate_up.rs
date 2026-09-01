#[cfg(all(target_arch = "aarch64", target_feature = "neon"))]
use super::common::{
    unpack_scales_mins, BLOCK_BYTES, ELEMS_PER_BLOCK, SUBBLOCKS_PER_BLOCK, SUBBLOCK_SIZE,
};
#[cfg(all(target_arch = "aarch64", target_feature = "neon"))]
use super::q4k_asm::use_asm_kernel;
#[cfg(all(target_arch = "aarch64", target_feature = "neon"))]
use super::q4k_neon::sdot_acc;
use super::q4k_scalar::q4k_q8k_matvec_scalar;
use super::q8k_activation::Q8KActivation;
#[cfg(all(target_arch = "aarch64", target_feature = "neon"))]
use crate::cpu::ops::q4_common::f16_to_f32;
use crate::cpu::ops::KernelShapeError;

/// Fused gate+up matvec: produce two output vectors from two weight matrices
/// against the SAME pre-quantised Q8_K activation in one pass.  Each
/// super-block of `q8k_x` is loaded once and SDOT'd against both `gate_w`
/// and `up_w` per row — gate and up SDOTs interleave on the OoO engine,
/// hiding cross-instruction latency that the back-to-back independent
/// `q4k_q8k_matvec_into` calls couldn't.
///
/// Caller layouts: `gate_w.len() == up_w.len() == rows * (cols / 256) * 144`,
/// `gate_out.len() == up_out.len() == rows`.
///
/// **x86_64 has no AVX2 branch here** (unlike [`q4k_q8k_matvec_into`], which
/// does) — this always falls through to the scalar reference on x86_64,
/// ~12–16× slower than AVX2/NEON. This is the dense `/v1/walk-ffn-q8k`
/// serving kernel; see `kernel_class_summary` (logged once at server
/// startup) and docs/audits/dec-readiness-review-2026-07-22.md §1b.
pub fn q4k_q8k_gate_up_into(
    gate_out: &mut [f32],
    up_out: &mut [f32],
    q8k_x: &Q8KActivation,
    gate_w: &[u8],
    up_w: &[u8],
    rows: usize,
    cols: usize,
) -> Result<(), KernelShapeError> {
    #[cfg(all(target_arch = "aarch64", target_feature = "neon"))]
    {
        // C12: same opt-in as `q4k_q8k_matvec_into` — `LARQL_Q4K_ASM=1`
        // routes the fused kernel through the hand-asm form. Bit-exact
        // (`q8k_gate_up_asm_matches_scalar_bit_exact`); default off.
        return if use_asm_kernel() {
            q4k_q8k_gate_up_asm(gate_out, up_out, q8k_x, gate_w, up_w, rows, cols)
        } else {
            q4k_q8k_gate_up_neon(gate_out, up_out, q8k_x, gate_w, up_w, rows, cols)
        };
    }
    #[allow(unreachable_code)]
    {
        // Scalar fallback: just call the existing single-matvec path twice.
        q4k_q8k_matvec_scalar(gate_out, q8k_x, gate_w, rows, cols)?;
        q4k_q8k_matvec_scalar(up_out, q8k_x, up_w, rows, cols)
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
) -> Result<(), KernelShapeError> {
    use std::arch::aarch64::*;

    KernelShapeError::check(
        "q4k_q8k_gate_up_neon (gate)",
        gate_out.len(),
        rows,
        q8k_x.qs.len(),
        cols,
        gate_w.len(),
        ELEMS_PER_BLOCK,
        BLOCK_BYTES,
    )?;
    KernelShapeError::check(
        "q4k_q8k_gate_up_neon (up)",
        up_out.len(),
        rows,
        q8k_x.qs.len(),
        cols,
        up_w.len(),
        ELEMS_PER_BLOCK,
        BLOCK_BYTES,
    )?;
    if rows == 0 || cols == 0 {
        gate_out.fill(0.0);
        up_out.fill(0.0);
        return Ok(());
    }
    let n_blocks = cols / ELEMS_PER_BLOCK;
    let row_bytes = n_blocks * BLOCK_BYTES;

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
    Ok(())
}

/// Fused gate+up twin of [`q4k_sb_sum1_asm`] (C12): one super-block's integer
/// `sum1` for BOTH the gate and up matrices in a single `asm!` block, sharing
/// the four activation vector loads per group between them — the point of the
/// fusion (the separate-matvec form streams the same 64 activation bytes per
/// group twice, and the doubled SDOT stream gives the OoO core independent
/// work to fill the ~20 stalled cycles the single-matrix form measures).
/// Scale handling matches the single-matrix form: 8 scales per matrix arrive
/// as two i32x4 vectors, applied with `mul (by element)` — no scalar `ldrb`.
///
/// Register map: v16 nibble mask; v17/v26 gate/up accumulators; v18-v19
/// gate scales, v24-v25 up scales; per group v0-v1/v8-v9 raw nibbles,
/// v2-v5/v10-v13 unpacked, v6-v7/v14-v15 dot temps, v20-v23 shared
/// activation. i32 lane sums are order-independent (wrapping add is
/// associative), so the tree-sum is bit-exact with the scalar reference.
#[cfg(all(target_arch = "aarch64", target_feature = "neon"))]
#[inline]
unsafe fn q4k_gate_up_sb_sum1_asm(
    g_quants: *const u8,
    u_quants: *const u8,
    act: *const i8,
    g_scales: *const i32,
    u_scales: *const i32,
) -> (i32, i32) {
    let sum1_g: i32;
    let sum1_u: i32;
    // One group of the unrolled body: `$gsv`/`$usv` = gate/up scale vectors,
    // `$l0`/`$l1` = lane indices for sub-blocks 2g / 2g+1.
    macro_rules! grp2 {
        ($gsv:literal, $usv:literal, $l0:literal, $l1:literal) => {
            concat!(
                "ld1 {{v0.16b, v1.16b}}, [{g}], #32\n",
                "ld1 {{v8.16b, v9.16b}}, [{u}], #32\n",
                "ld1 {{v20.16b, v21.16b, v22.16b, v23.16b}}, [{a}], #64\n",
                "and  v2.16b, v0.16b, v16.16b\n", // gate lo (sub-block 2g)
                "and  v3.16b, v1.16b, v16.16b\n",
                "ushr v4.16b, v0.16b, #4\n", // gate hi (sub-block 2g+1)
                "ushr v5.16b, v1.16b, #4\n",
                "and  v10.16b, v8.16b, v16.16b\n", // up lo
                "and  v11.16b, v9.16b, v16.16b\n",
                "ushr v12.16b, v8.16b, #4\n", // up hi
                "ushr v13.16b, v9.16b, #4\n",
                "movi v6.4s, #0\n",
                "movi v7.4s, #0\n",
                "movi v14.4s, #0\n",
                "movi v15.4s, #0\n",
                // Gate / up SDOTs interleaved — independent chains on the
                // same shared activation registers.
                "sdot v6.4s,  v2.16b,  v20.16b\n",
                "sdot v14.4s, v10.16b, v20.16b\n",
                "sdot v6.4s,  v3.16b,  v21.16b\n",
                "sdot v14.4s, v11.16b, v21.16b\n",
                "sdot v7.4s,  v4.16b,  v22.16b\n",
                "sdot v15.4s, v12.16b, v22.16b\n",
                "sdot v7.4s,  v5.16b,  v23.16b\n",
                "sdot v15.4s, v13.16b, v23.16b\n",
                "mul  v6.4s,  v6.4s,  ",
                $gsv,
                ".s[",
                $l0,
                "]\n",
                "mul  v7.4s,  v7.4s,  ",
                $gsv,
                ".s[",
                $l1,
                "]\n",
                "mul  v14.4s, v14.4s, ",
                $usv,
                ".s[",
                $l0,
                "]\n",
                "mul  v15.4s, v15.4s, ",
                $usv,
                ".s[",
                $l1,
                "]\n",
                "add  v17.4s, v17.4s, v6.4s\n",
                "add  v17.4s, v17.4s, v7.4s\n",
                "add  v26.4s, v26.4s, v14.4s\n",
                "add  v26.4s, v26.4s, v15.4s\n",
            )
        };
    }
    unsafe {
        core::arch::asm!(
            "movi v16.16b, #0x0f",                // nibble mask
            "movi v17.4s, #0",                    // gate sum1 accumulator
            "movi v26.4s, #0",                    // up sum1 accumulator
            "ld1 {{v18.4s, v19.4s}}, [{gs}]",     // gate scales[0..4], [4..8]
            "ld1 {{v24.4s, v25.4s}}, [{us}]",     // up scales[0..4], [4..8]
            grp2!("v18", "v24", "0", "1"),        // group 0 → sub-blocks 0,1
            grp2!("v18", "v24", "2", "3"),        // group 1 → sub-blocks 2,3
            grp2!("v19", "v25", "0", "1"),        // group 2 → sub-blocks 4,5
            grp2!("v19", "v25", "2", "3"),        // group 3 → sub-blocks 6,7
            "addv s17, v17.4s",
            "addv s26, v26.4s",
            "fmov {sg:w}, s17",
            "fmov {su:w}, s26",
            g = inout(reg) g_quants => _,
            u = inout(reg) u_quants => _,
            a = inout(reg) act => _,
            gs = in(reg) g_scales,
            us = in(reg) u_scales,
            sg = out(reg) sum1_g,
            su = out(reg) sum1_u,
            out("v0") _, out("v1") _, out("v2") _, out("v3") _,
            out("v4") _, out("v5") _, out("v6") _, out("v7") _,
            out("v8") _, out("v9") _, out("v10") _, out("v11") _,
            out("v12") _, out("v13") _, out("v14") _, out("v15") _,
            out("v16") _, out("v17") _, out("v18") _, out("v19") _,
            out("v20") _, out("v21") _, out("v22") _, out("v23") _,
            out("v24") _, out("v25") _, out("v26") _,
            options(nostack, readonly),
        );
    }
    (sum1_g, sum1_u)
}

/// Hand-asm fused gate+up matvec (C12). Identical interface and output to
/// [`q4k_q8k_gate_up_neon`] — integer `sum1` pairs come from
/// [`q4k_gate_up_sb_sum1_asm`], the `sum2` terms and the f32 epilogue are
/// the same Rust code as the neon/scalar forms, so it is bit-exact with two
/// independent scalar matvecs (`q8k_gate_up_asm_matches_scalar_bit_exact`).
#[cfg(all(target_arch = "aarch64", target_feature = "neon"))]
#[allow(clippy::too_many_arguments)]
pub fn q4k_q8k_gate_up_asm(
    gate_out: &mut [f32],
    up_out: &mut [f32],
    q8k_x: &Q8KActivation,
    gate_w: &[u8],
    up_w: &[u8],
    rows: usize,
    cols: usize,
) -> Result<(), KernelShapeError> {
    KernelShapeError::check(
        "q4k_q8k_gate_up_asm (gate)",
        gate_out.len(),
        rows,
        q8k_x.qs.len(),
        cols,
        gate_w.len(),
        ELEMS_PER_BLOCK,
        BLOCK_BYTES,
    )?;
    KernelShapeError::check(
        "q4k_q8k_gate_up_asm (up)",
        up_out.len(),
        rows,
        q8k_x.qs.len(),
        cols,
        up_w.len(),
        ELEMS_PER_BLOCK,
        BLOCK_BYTES,
    )?;
    if rows == 0 || cols == 0 {
        gate_out.fill(0.0);
        up_out.fill(0.0);
        return Ok(());
    }
    let n_blocks = cols / ELEMS_PER_BLOCK;
    let row_bytes = n_blocks * BLOCK_BYTES;

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

            let sc_g_i32 = [
                sc_g[0] as i32,
                sc_g[1] as i32,
                sc_g[2] as i32,
                sc_g[3] as i32,
                sc_g[4] as i32,
                sc_g[5] as i32,
                sc_g[6] as i32,
                sc_g[7] as i32,
            ];
            let sc_u_i32 = [
                sc_u[0] as i32,
                sc_u[1] as i32,
                sc_u[2] as i32,
                sc_u[3] as i32,
                sc_u[4] as i32,
                sc_u[5] as i32,
                sc_u[6] as i32,
                sc_u[7] as i32,
            ];

            let q8_base = sb * ELEMS_PER_BLOCK;
            let q8_qs_ptr = q8k_x.qs[q8_base..q8_base + ELEMS_PER_BLOCK].as_ptr();
            let q8_sums = &q8k_x.sums[sb * SUBBLOCKS_PER_BLOCK..(sb + 1) * SUBBLOCKS_PER_BLOCK];
            let d_y = q8k_x.d[sb];

            // SAFETY: each Q4_K super-block is 144 bytes (16 header + 128
            // quants), `q8_qs_ptr` spans a full 256-i8 super-block, and both
            // scale arrays are 8 i32.
            let (s1_g, s1_u) = unsafe {
                q4k_gate_up_sb_sum1_asm(
                    g_block[16..].as_ptr(),
                    u_block[16..].as_ptr(),
                    q8_qs_ptr,
                    sc_g_i32.as_ptr(),
                    sc_u_i32.as_ptr(),
                )
            };

            // sum2 stays scalar (precomputed Q8_K sums; no SDOT) — identical
            // to the neon/scalar paths so the f32 epilogue is bit-for-bit the
            // same.
            let mut s2_g: i32 = 0;
            let mut s2_u: i32 = 0;
            for s in 0..SUBBLOCKS_PER_BLOCK {
                s2_g += mn_g[s] as i32 * q8_sums[s] as i32;
                s2_u += mn_u[s] as i32 * q8_sums[s] as i32;
            }
            acc_g += d_g * d_y * s1_g as f32 - dmin_g * d_y * s2_g as f32;
            acc_u += d_u * d_y * s1_u as f32 - dmin_u * d_y * s2_u as f32;
        }
        gate_out[r] = acc_g;
        up_out[r] = acc_u;
    }
    Ok(())
}
