use super::common::ELEMS_PER_BLOCK;
// TODO(q6k-planar): re-import `super::q4k_asm::use_asm_kernel` when the
// hand-asm Q6_K kernel is reworked onto ggml's planar layout, so the
// dispatcher can offer the `LARQL_Q4K_ASM=1` opt-in again. The NEON
// intrinsic form is already planar and is what the dispatcher uses.
#[cfg(all(target_arch = "aarch64", target_feature = "neon"))]
use super::q4k_neon::sdot_acc;
use super::q8k_activation::Q8KActivation;
use crate::cpu::ops::q4_common::f16_to_f32;
use crate::cpu::ops::KernelShapeError;

// ── Q6_K × Q8_K matvec ───────────────────────────────────────────────────────
//
// Q6_K super-block: 210 bytes per 256 values.
//   [0..128]   128 bytes: ql — lo4 bits packed 2 per byte (nibble-packed)
//   [128..192]  64 bytes: qh — hi2 bits packed 4 per byte (2 bits each)
//   [192..208]  16 bytes: scales — one int8 per 16 elements
//   [208..210]   2 bytes: d — f16 super-block scale
//
// Element placement follows ggml's planar layout (see
// `larql_models::quant::ggml::q6_k::q6k_subblock_vals`): within each
// 128-element half, ql low nibbles hold elements 0..63 and high nibbles
// 64..127; qh[l] packs the hi2 bits of elements l/l+32/l+64/l+96.
//            w[i] = d * scales[i/16] * (raw6 - 32)
//
// Dot product with Q8_K activation `q8k`:
//   out[r] = Σ_blocks d_w * d_y * Σ_{g=0..15} scales[g] * dot_g
//   where dot_g = Σ_{i in g*16..(g+1)*16} (raw6[i] - 32) * q8k_q[i]
//
// The -(raw6 - 32) sign matches llama.cpp's `ggml_vec_dot_q6_K_q8_K`.
// No `mins` term (Q6_K doesn't have per-group mins — it's symmetric around 32).

/// Q6_K super-block size in bytes (re-export of the wire-format constant).
pub(super) const Q6K_BLOCK_BYTES: usize = larql_models::quant::ggml::Q6_K_BLOCK_BYTES;

/// Scalar reference: Q6_K weights × Q8_K activation matvec.
/// Correctness oracle for the NEON implementation below.
pub fn q6k_q8k_matvec_scalar(
    out: &mut [f32],
    q8k_x: &Q8KActivation,
    w: &[u8],
    rows: usize,
    cols: usize,
) -> Result<(), KernelShapeError> {
    KernelShapeError::check(
        "q6k_q8k_matvec_scalar",
        out.len(),
        rows,
        q8k_x.qs.len(),
        cols,
        w.len(),
        ELEMS_PER_BLOCK,
        Q6K_BLOCK_BYTES,
    )?;
    let n_blocks = cols / ELEMS_PER_BLOCK;
    let row_bytes = n_blocks * Q6K_BLOCK_BYTES;
    out.fill(0.0);
    if rows == 0 || cols == 0 {
        return Ok(());
    }
    for (r, out_r) in out.iter_mut().enumerate().take(rows) {
        let row_base = r * row_bytes;
        let mut acc = 0.0f32;
        for sb in 0..n_blocks {
            let block = &w[row_base + sb * Q6K_BLOCK_BYTES..];
            let sc = &block[192..208]; // 16 × int8
            let d_w = f16_to_f32(u16::from_le_bytes([block[208], block[209]]));
            let d_y = q8k_x.d[sb];
            let q8_base = sb * ELEMS_PER_BLOCK;
            let q8_qs = &q8k_x.qs[q8_base..q8_base + ELEMS_PER_BLOCK];

            let mut sum1: i32 = 0;
            for (g, scale_byte) in sc.iter().enumerate().take(16usize) {
                // 16-element group g, using scale sc[g]. Weights decode
                // through the shared ggml planar-layout helper.
                let scale = *scale_byte as i8 as i32;
                let vals = larql_models::quant::ggml::q6_k::q6k_subblock_vals(
                    &block[..Q6K_BLOCK_BYTES],
                    g,
                );
                let mut dot_g: i32 = 0;
                for (k, &v) in vals.iter().enumerate() {
                    dot_g += (v as i32) * q8_qs[g * 16 + k] as i32;
                }
                sum1 += scale * dot_g;
            }
            acc += d_w * d_y * sum1 as f32;
        }
        *out_r = acc;
    }
    Ok(())
}

/// NEON-accelerated Q6_K × Q8_K matvec for `aarch64`.
///
/// Per 16-element scale group, under ggml's planar layout the group's
/// source bytes are contiguous in both planes, so the dequant is two
/// loads and no shuffle:
/// 1. 16 ql bytes → lo4[16] via one `vld1q_u8` plus a mask (planes 0/1)
///    or a constant `vshrq_n_u8(_, 4)` (planes 2/3).
///    16 qh bytes → hi2[16] via one `vld1q_u8`, a uniform right-shift of
///    `plane*2`, and a 2-bit mask.
///    raw6 = lo4 | (hi2 << 4); signed = raw6 - 32 → int8.
/// 2. One SDOT over the 16 int8 weight × int8 activation products —
///    the Q8_K activation is sequential, so it pairs directly.
/// 3. scale * dot_g accumulated into sum1.
///
/// Final: acc += d_w * d_y * sum1.
///
/// Decodes ggml's planar layout, bit-exact with `q6k_q8k_matvec_scalar`
/// (`q6k_matvec_neon_matches_scalar_bit_exact`).
#[cfg(all(target_arch = "aarch64", target_feature = "neon"))]
pub fn q6k_q8k_matvec_neon(
    out: &mut [f32],
    q8k_x: &Q8KActivation,
    w: &[u8],
    rows: usize,
    cols: usize,
) -> Result<(), KernelShapeError> {
    use std::arch::aarch64::*;

    KernelShapeError::check(
        "q6k_q8k_matvec_neon",
        out.len(),
        rows,
        q8k_x.qs.len(),
        cols,
        w.len(),
        ELEMS_PER_BLOCK,
        Q6K_BLOCK_BYTES,
    )?;
    let n_blocks = cols / ELEMS_PER_BLOCK;
    let row_bytes = n_blocks * Q6K_BLOCK_BYTES;
    out.fill(0.0);
    if rows == 0 || cols == 0 {
        return Ok(());
    }

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
                // ggml planar layout (mirrors
                // `larql_models::quant::ggml::q6_k::q6k_subblock_vals`).
                // Scale group g covers elements g*16..(g+1)*16, which always
                // lie in a single nibble plane — so both the 16 `ql` source
                // bytes and the 16 `qh` source bytes are *contiguous*, and the
                // interleaved form's vzip disappears: one load each, paired
                // directly against the sequential Q8 activation.
                let half = g / 8; // which 128-element half
                let grp = g % 8; // scale group within the half
                let plane = grp / 2; // q1/q2/q3/q4 in ggml's naming
                let lbase = (grp % 2) * 16; // byte offset within the 32-byte plane row
                let ql_g = unsafe { ql_base.add(half * 64 + (plane & 1) * 32 + lbase) };
                let qh_g = unsafe { qh_base.add(half * 32 + lbase) };
                let q8_g = unsafe { q8_ptr.add(q8_base + g * 16) };
                let scale = unsafe { *sc_base.add(g) as i32 };

                // ── Lo4: 16 contiguous ql bytes; planes 0/1 take the low
                // nibble, planes 2/3 the high nibble. Uniform across the group.
                let ql_v = unsafe { vld1q_u8(ql_g) };
                let lo4_v = if plane < 2 {
                    unsafe { vandq_u8(ql_v, mask_0f) }
                } else {
                    unsafe { vshrq_n_u8(ql_v, 4) }
                };

                // ── Hi2: 16 contiguous qh bytes, one uniform right-shift of
                // plane*2, then mask to 2 bits. `vshlq_u8` with a negative
                // amount shifts right.
                let qh_v = unsafe { vld1q_u8(qh_g) };
                let hi2_v =
                    unsafe { vandq_u8(vshlq_u8(qh_v, vdupq_n_s8(-((plane as i8) * 2))), mask_03) };

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
    Ok(())
}

/// TBL index table for the Q6_K hi2 replicate: group `j` (of 4 within one
/// 16-byte `qh` vector) selects bytes `4j..4j+3`, each repeated 4×, so a
/// single `tbl` builds the per-element hi2 source that the neon form
/// assembles with four scalar multiplies per group.
#[cfg(all(target_arch = "aarch64", target_feature = "neon"))]
#[rustfmt::skip]
static Q6K_TBL_IDX: [u8; 64] = [
    0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3,
    4, 4, 4, 4, 5, 5, 5, 5, 6, 6, 6, 6, 7, 7, 7, 7,
    8, 8, 8, 8, 9, 9, 9, 9, 10, 10, 10, 10, 11, 11, 11, 11,
    12, 12, 12, 12, 13, 13, 13, 13, 14, 14, 14, 14, 15, 15, 15, 15,
];

/// Right-shift pattern for the replicated hi2 bytes (negative = shift right
/// under `sshl`): element 4j+k needs `qh_byte >> 2k`.
#[cfg(all(target_arch = "aarch64", target_feature = "neon"))]
static Q6K_SHIFT_RIGHT: [i8; 16] = [0, -2, -4, -6, 0, -2, -4, -6, 0, -2, -4, -6, 0, -2, -4, -6];

/// One Q6_K super-block's integer `sum1 = Σ_g scale[g] · dot16_g` in a single
/// `asm!` block (C12). Differences from [`q6k_q8k_matvec_neon`]'s inner loop:
/// the hi2 replicate is one `tbl` (vs 4 scalar multiplies + vector rebuild),
/// and the per-group scale lands as a vector-lane `mul` on the 4-lane SDOT
/// partials with a single `addv` at the end (vs 16 horizontal `addv` + scalar
/// multiply-adds). i32 lane sums are order-independent (wrapping add), so the
/// result is bit-exact with the neon/scalar forms.
#[cfg(all(target_arch = "aarch64", target_feature = "neon"))]
#[inline]
unsafe fn q6k_sb_sum1_asm(ql: *const u8, qh: *const u8, act: *const i8, scales: *const i32) -> i32 {
    let sum1: i32;
    // One 16-element group: `$qh` = the loaded qh vector for this group's
    // quad (v8-v11), `$idx` = the TBL replicate index vector for the group's
    // position within that quad (v24-v27), `$sv`/`$lane` = widened scale
    // vector (v12-v15) and lane.
    macro_rules! q6grp {
        ($qh:literal, $idx:literal, $sv:literal, $lane:literal) => {
            concat!(
                "ld1 {{v0.8b}}, [{ql}], #8\n",
                "ld1 {{v5.16b}}, [{a}], #16\n",
                "and  v1.16b, v0.16b, v29.16b\n", // lo4 of even elements
                "ushr v2.16b, v0.16b, #4\n",      // lo4 of odd elements
                "zip1 v3.16b, v1.16b, v2.16b\n",  // restore element order
                "tbl  v4.16b, {{",
                $qh,
                ".16b}}, ",
                $idx,
                ".16b\n",
                "sshl v4.16b, v4.16b, v28.16b\n",
                "and  v4.16b, v4.16b, v30.16b\n",
                "shl  v4.16b, v4.16b, #4\n",
                "orr  v3.16b, v3.16b, v4.16b\n", // raw6 = lo4 | hi2<<4
                "sub  v3.16b, v3.16b, v31.16b\n", // signed: raw6 - 32
                "movi v6.4s, #0\n",
                "sdot v6.4s, v3.16b, v5.16b\n",
                "mul  v6.4s, v6.4s, ",
                $sv,
                ".s[",
                $lane,
                "]\n",
                "add  v16.4s, v16.4s, v6.4s\n",
            )
        };
    }
    unsafe {
        core::arch::asm!(
            "movi v16.4s, #0",                           // sum1 accumulator
            "movi v29.16b, #0x0f",                       // lo4 mask
            "movi v30.16b, #0x03",                       // hi2 mask
            "movi v31.16b, #32",                         // raw6 bias
            "ld1 {{v8.16b, v9.16b, v10.16b, v11.16b}}, [{qh}]",      // 64B qh
            "ld1 {{v12.4s, v13.4s, v14.4s, v15.4s}}, [{scales}]",    // 16 i32 scales
            "ld1 {{v24.16b, v25.16b, v26.16b, v27.16b}}, [{idx}]",   // TBL tables
            "ld1 {{v28.16b}}, [{shift}]",                            // shift pattern
            q6grp!("v8", "v24", "v12", "0"),
            q6grp!("v8", "v25", "v12", "1"),
            q6grp!("v8", "v26", "v12", "2"),
            q6grp!("v8", "v27", "v12", "3"),
            q6grp!("v9", "v24", "v13", "0"),
            q6grp!("v9", "v25", "v13", "1"),
            q6grp!("v9", "v26", "v13", "2"),
            q6grp!("v9", "v27", "v13", "3"),
            q6grp!("v10", "v24", "v14", "0"),
            q6grp!("v10", "v25", "v14", "1"),
            q6grp!("v10", "v26", "v14", "2"),
            q6grp!("v10", "v27", "v14", "3"),
            q6grp!("v11", "v24", "v15", "0"),
            q6grp!("v11", "v25", "v15", "1"),
            q6grp!("v11", "v26", "v15", "2"),
            q6grp!("v11", "v27", "v15", "3"),
            "addv s16, v16.4s",
            "fmov {sum1:w}, s16",
            ql = inout(reg) ql => _,
            a = inout(reg) act => _,
            qh = in(reg) qh,
            scales = in(reg) scales,
            idx = in(reg) Q6K_TBL_IDX.as_ptr(),
            shift = in(reg) Q6K_SHIFT_RIGHT.as_ptr(),
            sum1 = out(reg) sum1,
            out("v0") _, out("v1") _, out("v2") _, out("v3") _,
            out("v4") _, out("v5") _, out("v6") _,
            out("v8") _, out("v9") _, out("v10") _, out("v11") _,
            out("v12") _, out("v13") _, out("v14") _, out("v15") _,
            out("v16") _,
            out("v24") _, out("v25") _, out("v26") _, out("v27") _,
            out("v28") _, out("v29") _, out("v30") _, out("v31") _,
            options(nostack, readonly),
        );
    }
    sum1
}

/// Hand-asm Q6_K × Q8_K matvec (C12). Identical interface and output to
/// [`q6k_q8k_matvec_neon`] — `sum1` comes from [`q6k_sb_sum1_asm`], the f32
/// epilogue (`acc += d_w·d_y·sum1`, no mins term) is the same Rust code, so
/// it is bit-exact with the scalar reference
/// (`q6k_matvec_asm_matches_scalar_bit_exact`).
///
/// TODO(q6k-planar): still decodes the pre-fix interleaved layout —
/// unreachable from the dispatcher until reworked for ggml's planar
/// layout and re-verified on ARM.
#[allow(dead_code)]
#[cfg(all(target_arch = "aarch64", target_feature = "neon"))]
pub fn q6k_q8k_matvec_asm(
    out: &mut [f32],
    q8k_x: &Q8KActivation,
    w: &[u8],
    rows: usize,
    cols: usize,
) -> Result<(), KernelShapeError> {
    KernelShapeError::check(
        "q6k_q8k_matvec_asm",
        out.len(),
        rows,
        q8k_x.qs.len(),
        cols,
        w.len(),
        ELEMS_PER_BLOCK,
        Q6K_BLOCK_BYTES,
    )?;
    let n_blocks = cols / ELEMS_PER_BLOCK;
    let row_bytes = n_blocks * Q6K_BLOCK_BYTES;
    out.fill(0.0);
    if rows == 0 || cols == 0 {
        return Ok(());
    }

    for (r, out_r) in out.iter_mut().enumerate().take(rows) {
        let row_base = r * row_bytes;
        let mut acc = 0.0f32;
        for sb in 0..n_blocks {
            let block = &w[row_base + sb * Q6K_BLOCK_BYTES..];
            let d_w = f16_to_f32(u16::from_le_bytes([block[208], block[209]]));
            let d_y = q8k_x.d[sb];

            // 16 per-group i8 scales widened to i32 for the vector-lane muls.
            let mut sc = [0i32; 16];
            for (g, s) in sc.iter_mut().enumerate() {
                *s = block[192 + g] as i8 as i32;
            }

            let q8_base = sb * ELEMS_PER_BLOCK;
            let q8_ptr = q8k_x.qs[q8_base..q8_base + ELEMS_PER_BLOCK].as_ptr();

            // SAFETY: a Q6_K super-block is 210 bytes (128 ql + 64 qh + 16
            // scales + 2 d); `q8_ptr` spans a full 256-i8 super-block; `sc`
            // is 16 i32; the static TBL/shift tables are 64/16 bytes.
            let sum1 = unsafe {
                q6k_sb_sum1_asm(block.as_ptr(), block.as_ptr().add(128), q8_ptr, sc.as_ptr())
            };
            acc += d_w * d_y * sum1 as f32;
        }
        *out_r = acc;
    }
    Ok(())
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
) -> Result<(), KernelShapeError> {
    #[cfg(all(target_arch = "aarch64", target_feature = "neon"))]
    {
        // TODO(q6k-planar): the `LARQL_Q4K_ASM=1` hand-asm form still decodes
        // the pre-fix interleaved layout, so unlike the Q4_K kernels there is
        // no asm opt-in here yet. The NEON intrinsic form is ggml-planar and
        // bit-exact with the scalar reference.
        return q6k_q8k_matvec_neon(out, q8k_x, w, rows, cols);
    }
    #[allow(unreachable_code)]
    q6k_q8k_matvec_scalar(out, q8k_x, w, rows, cols)
}
