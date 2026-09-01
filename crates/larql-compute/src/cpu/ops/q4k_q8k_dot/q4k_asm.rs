use super::common::{unpack_scales_mins, BLOCK_BYTES, ELEMS_PER_BLOCK, SUBBLOCKS_PER_BLOCK};
use super::q8k_activation::Q8KActivation;
use crate::cpu::ops::q4_common::f16_to_f32;
use crate::cpu::ops::KernelShapeError;

/// Hand-asm inner loop (C12 Phase 1): the per-super-block scaled integer dot
/// `sum1 = Σ_sb scale[sb] · Σ_i nibble[sb][i]·y[sb][i]`, computed in one
/// `asm!` block so the schedule is ours, not LLVM's.
///
/// Returns the same i32 `sum1` as the intrinsic / scalar paths (integer math
/// is exact regardless of order), so the f32 epilogue and `sum2` stay in Rust
/// and bit-parity reduces to "does this produce the same `sum1`".
///
/// vs `q4k_q8k_matvec_neon`'s inner loop it kills the 8 scalar `ldrb` scale
/// loads + scalar→vector broadcast: the 8 6-bit scales arrive as two i32x4
/// vectors and the per-sub-block scale is applied with `mul (by element)`.
/// The roofline microbench (`benches/q4k_q8k_matvec.rs`) showed the kernel is
/// compute/issue-bound (~33 cyc/super-block), not DRAM-bound, so cutting
/// issue-port pressure is the lever — see `docs/q4k-decode-kernel.md`
/// §"2026-06-02 roofline measurement".
///
/// Layout (matches `q4k_q8k_matvec_neon` exactly): the 128 nibble bytes walk
/// in 4 groups of 32; group `g` low nibbles → sub-block `2g`, high nibbles →
/// `2g+1`. Activation walks in 4 groups of 64 i8 (two sub-blocks each). Both
/// pointers post-increment through the super-block.
///
/// SAFETY: `quants` must point to ≥128 readable bytes, `act` to ≥256, and
/// `scales` to an 8-element i32 array. Requires the `dotprod` extension (SDOT),
/// baseline on `aarch64-apple-darwin` — same assumption as `sdot_acc`.
#[cfg(all(target_arch = "aarch64", target_feature = "neon"))]
#[inline]
#[doc(hidden)] // pub for the C12 decomposition microbench (benches/q4k_q8k_matvec.rs)
pub unsafe fn q4k_sb_sum1_asm(quants: *const u8, act: *const i8, scales: *const i32) -> i32 {
    let sum1: i32;
    // One group of the unrolled body, parameterised by the two scale lanes
    // (`$sv` = scale vector, `$l0`/`$l1` = lane indices for sub-blocks 2g/2g+1).
    // Single running accumulator `v17`: a 4-private-accumulator variant was
    // tried 2026-06-02 to break the per-super-block RAW chain but showed no
    // reliable gain — the asm/neon ratio swings ±1.5% run-to-run (observed
    // +3.7%..+4.9% for THIS form), larger than any v1↔4-acc difference. The
    // row loop inlines this fn, so the OoO core already overlaps the next
    // super-block's compute with this accumulator chain; the chain isn't the
    // bottleneck. See `docs/q4k-decode-kernel.md` §"Finding — latency-hiding
    // has low headroom".
    macro_rules! grp {
        ($sv:literal, $l0:literal, $l1:literal) => {
            concat!(
                "ld1 {{v0.16b, v1.16b}}, [{q}], #32\n",
                "ld1 {{v20.16b, v21.16b, v22.16b, v23.16b}}, [{a}], #64\n",
                "and  v2.16b, v0.16b, v16.16b\n", // lo0 (sub-block 2g, lanes 0..16)
                "and  v3.16b, v1.16b, v16.16b\n", // lo1 (sub-block 2g, lanes 16..32)
                "ushr v4.16b, v0.16b, #4\n",      // hi0 (sub-block 2g+1)
                "ushr v5.16b, v1.16b, #4\n",      // hi1
                "movi v6.4s, #0\n",
                "movi v7.4s, #0\n",
                "sdot v6.4s, v2.16b, v20.16b\n", // dot[2g]   lanes += lo0·y
                "sdot v6.4s, v3.16b, v21.16b\n", //            += lo1·y
                "sdot v7.4s, v4.16b, v22.16b\n", // dot[2g+1] += hi0·y
                "sdot v7.4s, v5.16b, v23.16b\n", //            += hi1·y
                "mul  v6.4s, v6.4s, ",
                $sv,
                ".s[",
                $l0,
                "]\n", // × scale[2g]
                "mul  v7.4s, v7.4s, ",
                $sv,
                ".s[",
                $l1,
                "]\n", // × scale[2g+1]
                "add  v17.4s, v17.4s, v6.4s\n",
                "add  v17.4s, v17.4s, v7.4s\n",
            )
        };
    }
    unsafe {
        core::arch::asm!(
            "movi v16.16b, #0x0f",                  // nibble mask
            "movi v17.4s, #0",                      // sum1 accumulator (i32x4)
            "ld1 {{v18.4s, v19.4s}}, [{scales}]",   // scales[0..4], scales[4..8]
            grp!("v18", "0", "1"),                  // group 0 → sub-blocks 0,1
            grp!("v18", "2", "3"),                  // group 1 → sub-blocks 2,3
            grp!("v19", "0", "1"),                  // group 2 → sub-blocks 4,5
            grp!("v19", "2", "3"),                  // group 3 → sub-blocks 6,7
            "addv s17, v17.4s",                     // horizontal sum of the 4 lanes
            "fmov {sum1:w}, s17",
            q = inout(reg) quants => _,
            a = inout(reg) act => _,
            scales = in(reg) scales,
            sum1 = out(reg) sum1,
            out("v0") _, out("v1") _, out("v2") _, out("v3") _,
            out("v4") _, out("v5") _, out("v6") _, out("v7") _,
            out("v16") _, out("v17") _, out("v18") _, out("v19") _,
            out("v20") _, out("v21") _, out("v22") _, out("v23") _,
            options(nostack, readonly),
        );
    }
    sum1
}

/// Hand-asm Q4_K × Q8_K matvec (C12 Phase 1). Identical interface and output
/// to [`q4k_q8k_matvec_neon`] — `sum1` comes from [`q4k_sb_sum1_asm`], the
/// `sum2` term and f32 epilogue are the same Rust code, so it is bit-exact
/// with the scalar reference (`q8k_matvec_asm_matches_scalar_bit_exact`).
#[cfg(all(target_arch = "aarch64", target_feature = "neon"))]
pub fn q4k_q8k_matvec_asm(
    out: &mut [f32],
    q8k_x: &Q8KActivation,
    w: &[u8],
    rows: usize,
    cols: usize,
) -> Result<(), KernelShapeError> {
    KernelShapeError::check(
        "q4k_q8k_matvec_asm",
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
            let quants_ptr = block[16..].as_ptr();

            // Scales as i32 for the `ld1 {v18,v19}` load inside the asm.
            let sc = [
                scales[0] as i32,
                scales[1] as i32,
                scales[2] as i32,
                scales[3] as i32,
                scales[4] as i32,
                scales[5] as i32,
                scales[6] as i32,
                scales[7] as i32,
            ];

            let q8_base = sb * ELEMS_PER_BLOCK;
            let q8_qs_ptr = q8k_x.qs[q8_base..q8_base + ELEMS_PER_BLOCK].as_ptr();
            let q8_sums = &q8k_x.sums[sb * SUBBLOCKS_PER_BLOCK..(sb + 1) * SUBBLOCKS_PER_BLOCK];
            let d_y = q8k_x.d[sb];

            // SAFETY: a Q4_K super-block is 144 bytes (16 header + 128 quants),
            // `q8_qs_ptr` spans a full 256-i8 super-block, `sc` is 8 i32.
            let sum1 = unsafe { q4k_sb_sum1_asm(quants_ptr, q8_qs_ptr, sc.as_ptr()) };

            // sum2 stays scalar (precomputed Q8_K sums; no SDOT) — identical
            // to the neon path so the f32 epilogue is bit-for-bit the same.
            let mut sum2_acc: i32 = 0;
            for s in 0..SUBBLOCKS_PER_BLOCK {
                sum2_acc += mins[s] as i32 * q8_sums[s] as i32;
            }
            acc += d_w * d_y * sum1 as f32 - dmin_w * d_y * sum2_acc as f32;
        }
        *out_slot = acc;
    }
    Ok(())
}

/// TBL index tables for the v2 super-block kernel's vectorised scale/min
/// unpack. The 16-byte header vector holds `d`(0-1) `dmin`(2-3) and the 12
/// packed scale bytes at lanes 4..16, so the `unpack_scales_mins` byte
/// positions shift by +4: A=p[0..4]→lanes 4..8, B=p[4..8]→lanes 8..12,
/// C=p[8..12]→lanes 12..16. 0xFF lanes produce zeros (TBL out-of-range).
/// Order: SCLO, SCHI_HI2, HI_LO4 (shared by scales and mins), MNLO,
/// MNHI_HI2 → loaded into v24..v28.
#[cfg(all(target_arch = "aarch64", target_feature = "neon"))]
#[rustfmt::skip]
static Q4K_UNPACK_IDX: [u8; 80] = [
    // SCLO: scales[0..4] = lo6 of A
    4, 5, 6, 7, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff,
    // SCHI_HI2: scales[4..8] |= (A >> 6) << 4
    0xff, 0xff, 0xff, 0xff, 4, 5, 6, 7, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff,
    // HI_LO4: scales[4..8] = lo4 of C (and mins[4..8] = hi4 of C, same lanes)
    0xff, 0xff, 0xff, 0xff, 12, 13, 14, 15, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff,
    // MNLO: mins[0..4] = lo6 of B
    8, 9, 10, 11, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff,
    // MNHI_HI2: mins[4..8] |= (B >> 6) << 4
    0xff, 0xff, 0xff, 0xff, 8, 9, 10, 11, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff,
];

/// v2 super-block kernel (C12): the WHOLE per-super-block computation in one
/// `asm!` block — vectorised 6-bit scale/min unpack (TBL, replaces the
/// scalar `unpack_scales_mins` + i32 array round-trip), the 4-group SDOT
/// `sum1` body (identical to [`q4k_sb_sum1_asm`]), `sum2` as
/// `smull`/`smlal2` over the Q8_K sub-block sums, hardware `fcvt` for
/// `d`/`dmin` (replaces two software `f16_to_f32` calls), and the exact-order
/// f32 epilogue. Returns the super-block's contribution
/// `d_w·d_y·sum1 − dmin_w·d_y·sum2`.
///
/// Built from the 2026-06-11 decomposition measurement: the v1 asm block is
/// 16.3 cyc/SB but the Rust glue around it costs 19.2 cyc/SB with only ~3.6
/// hidden by OoO overlap — the glue, not the asm schedule, is the fat.
///
/// Bit-exactness: `fcvt` h→s is exact (every f16 is representable), `scvtf`
/// rounds i32→f32 identically to Rust's `as f32`, and the epilogue
/// multiplication tree matches the scalar reference's expression order.
#[cfg(all(target_arch = "aarch64", target_feature = "neon"))]
#[inline]
unsafe fn q4k_sb_contrib_asm(
    header: *const u8,
    quants: *const u8,
    act: *const i8,
    q8_sums: *const i16,
    d_y: f32,
) -> f32 {
    let contrib: f32;
    macro_rules! grp {
        ($sv:literal, $l0:literal, $l1:literal) => {
            concat!(
                "ld1 {{v0.16b, v1.16b}}, [{q}], #32\n",
                "ld1 {{v20.16b, v21.16b, v22.16b, v23.16b}}, [{a}], #64\n",
                "and  v2.16b, v0.16b, v16.16b\n",
                "and  v3.16b, v1.16b, v16.16b\n",
                "ushr v4.16b, v0.16b, #4\n",
                "ushr v5.16b, v1.16b, #4\n",
                "movi v6.4s, #0\n",
                "movi v7.4s, #0\n",
                "sdot v6.4s, v2.16b, v20.16b\n",
                "sdot v6.4s, v3.16b, v21.16b\n",
                "sdot v7.4s, v4.16b, v22.16b\n",
                "sdot v7.4s, v5.16b, v23.16b\n",
                "mul  v6.4s, v6.4s, ",
                $sv,
                ".s[",
                $l0,
                "]\n",
                "mul  v7.4s, v7.4s, ",
                $sv,
                ".s[",
                $l1,
                "]\n",
                "add  v17.4s, v17.4s, v6.4s\n",
                "add  v17.4s, v17.4s, v7.4s\n",
            )
        };
    }
    unsafe {
        core::arch::asm!(
            "movi v16.16b, #0x0f",
            "movi v30.16b, #0x3f",
            "ld1 {{v24.16b, v25.16b, v26.16b, v27.16b}}, [{idx}]",
            "ld1 {{v28.16b}}, [{idx2}]",
            "ld1 {{v0.16b}}, [{hdr}]",      // d | dmin | 12 packed scale bytes
            // ── vectorised unpack_scales_mins ──
            "and  v1.16b, v0.16b, v30.16b", // lo6 of every byte
            "ushr v2.16b, v0.16b, #6",
            "shl  v2.16b, v2.16b, #4",      // (byte >> 6) << 4
            "and  v3.16b, v0.16b, v16.16b", // lo4
            "ushr v4.16b, v0.16b, #4",      // hi4
            "tbl  v5.16b, {{v1.16b}}, v24.16b", // scales[0..4]
            "tbl  v6.16b, {{v3.16b}}, v26.16b", // scales[4..8] lo4 (from C)
            "tbl  v7.16b, {{v2.16b}}, v25.16b", // scales[4..8] hi2 (from A)
            "orr  v5.16b, v5.16b, v6.16b",
            "orr  v5.16b, v5.16b, v7.16b",      // sc8 in lanes 0..8
            "tbl  v6.16b, {{v1.16b}}, v27.16b", // mins[0..4]
            "tbl  v7.16b, {{v4.16b}}, v26.16b", // mins[4..8] lo4 (from C)
            "tbl  v1.16b, {{v2.16b}}, v28.16b", // mins[4..8] hi2 (from B)
            "orr  v6.16b, v6.16b, v7.16b",
            "orr  v31.16b, v6.16b, v1.16b",     // mn8 stashed in v31
            "ushll v5.8h, v5.8b, #0",
            "ushll  v18.4s, v5.4h, #0",
            "ushll2 v19.4s, v5.8h, #0",         // scales as i32x4 ×2
            // ── sum1: 4 groups, identical to the v1 asm ──
            "movi v17.4s, #0",
            grp!("v18", "0", "1"),
            grp!("v18", "2", "3"),
            grp!("v19", "0", "1"),
            grp!("v19", "2", "3"),
            "addv s17, v17.4s",                 // sum1 (i32 in s17)
            // ── sum2 = Σ mins[s] · q8_sums[s] (i16 sums, mins ≤ 63) ──
            "ushll v1.8h, v31.8b, #0",
            "ld1 {{v2.8h}}, [{sums}]",
            "smull  v3.4s, v2.4h, v1.4h",
            "smlal2 v3.4s, v2.8h, v1.8h",
            "addv s3, v3.4s",                   // sum2 (i32 in s3)
            // ── epilogue: d_w·d_y·sum1f − dmin_w·d_y·sum2f, exact order ──
            "ldr  s0, [{hdr}]",                 // d (h[0]) | dmin (h[1])
            "mov  h4, v0.h[0]",
            "fcvt s4, h4",                      // d_w (exact)
            "mov  h5, v0.h[1]",
            "fcvt s5, h5",                      // dmin_w (exact)
            "scvtf s6, s17",                    // sum1 as f32 (same rounding as Rust)
            "scvtf s7, s3",                     // sum2 as f32
            "fmul s4, s4, {dy:s}",
            "fmul s4, s4, s6",
            "fmul s5, s5, {dy:s}",
            "fmul s5, s5, s7",
            "fsub {contrib:s}, s4, s5",
            hdr = in(reg) header,
            q = inout(reg) quants => _,
            a = inout(reg) act => _,
            sums = in(reg) q8_sums,
            idx = in(reg) Q4K_UNPACK_IDX.as_ptr(),
            idx2 = in(reg) Q4K_UNPACK_IDX.as_ptr().wrapping_add(64),
            dy = in(vreg) d_y,
            contrib = out(vreg) contrib,
            out("v0") _, out("v1") _, out("v2") _, out("v3") _,
            out("v4") _, out("v5") _, out("v6") _, out("v7") _,
            out("v16") _, out("v17") _, out("v18") _, out("v19") _,
            out("v20") _, out("v21") _, out("v22") _, out("v23") _,
            out("v24") _, out("v25") _, out("v26") _, out("v27") _,
            out("v28") _, out("v30") _, out("v31") _,
            options(nostack, readonly),
        );
    }
    contrib
}

/// v2 hand-asm Q4_K × Q8_K matvec: per super-block one [`q4k_sb_contrib_asm`]
/// call — no per-block Rust glue at all (the v1 form's `unpack_scales_mins` +
/// scale-array + `sum2` + 2× software `f16_to_f32` measured 19.2 cyc/SB,
/// as much as the asm block itself). Bit-exact with the scalar reference
/// (`q8k_matvec_asm_v2_matches_scalar_bit_exact`).
#[cfg(all(target_arch = "aarch64", target_feature = "neon"))]
pub fn q4k_q8k_matvec_asm_v2(
    out: &mut [f32],
    q8k_x: &Q8KActivation,
    w: &[u8],
    rows: usize,
    cols: usize,
) -> Result<(), KernelShapeError> {
    KernelShapeError::check(
        "q4k_q8k_matvec_asm_v2",
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
            let q8_base = sb * ELEMS_PER_BLOCK;
            // SAFETY: 144-byte super-block (16 header + 128 quants); the act
            // slice spans 256 i8; q8_sums has 8 i16 per super-block; the
            // static index tables are 80 bytes.
            acc += unsafe {
                q4k_sb_contrib_asm(
                    block.as_ptr(),
                    block[16..].as_ptr(),
                    q8k_x.qs[q8_base..q8_base + ELEMS_PER_BLOCK].as_ptr(),
                    q8k_x.sums[sb * SUBBLOCKS_PER_BLOCK..].as_ptr(),
                    q8k_x.d[sb],
                )
            };
        }
        *out_slot = acc;
    }
    Ok(())
}

/// v3 (C12): one `asm!` block per ROW — the super-block loop lives inside
/// the asm, so the TBL tables / masks load once per row instead of once per
/// super-block (v2 paid ~4-5 cyc/SB reloading loop-invariant constants).
/// The walking pointer exploits the Q4_K layout: each 144-byte block is
/// [16B header][128B quants] contiguous, so the header `ld1 ..., #16`
/// followed by the four group loads (4×32B) lands the pointer exactly on
/// the next block's header — no pointer arithmetic at all.
///
/// Accumulation order matches the v1/v2/scalar forms exactly (sequential
/// `fadd` of per-block contributions), so it stays bit-exact.
#[cfg(all(target_arch = "aarch64", target_feature = "neon"))]
#[inline]
unsafe fn q4k_row_dot_asm(
    row: *const u8,
    act: *const i8,
    q8_sums: *const i16,
    d: *const f32,
    n_blocks: usize,
) -> f32 {
    let acc: f32;
    macro_rules! grp {
        ($sv:literal, $l0:literal, $l1:literal) => {
            concat!(
                "ld1 {{v0.16b, v1.16b}}, [{p}], #32\n",
                "ld1 {{v20.16b, v21.16b, v22.16b, v23.16b}}, [{a}], #64\n",
                "and  v2.16b, v0.16b, v16.16b\n",
                "and  v3.16b, v1.16b, v16.16b\n",
                "ushr v4.16b, v0.16b, #4\n",
                "ushr v5.16b, v1.16b, #4\n",
                "movi v6.4s, #0\n",
                "movi v7.4s, #0\n",
                "sdot v6.4s, v2.16b, v20.16b\n",
                "sdot v6.4s, v3.16b, v21.16b\n",
                "sdot v7.4s, v4.16b, v22.16b\n",
                "sdot v7.4s, v5.16b, v23.16b\n",
                "mul  v6.4s, v6.4s, ",
                $sv,
                ".s[",
                $l0,
                "]\n",
                "mul  v7.4s, v7.4s, ",
                $sv,
                ".s[",
                $l1,
                "]\n",
                "add  v17.4s, v17.4s, v6.4s\n",
                "add  v17.4s, v17.4s, v7.4s\n",
            )
        };
    }
    unsafe {
        core::arch::asm!(
            // ── loop-invariant constants (per row, not per super-block) ──
            "movi v16.16b, #0x0f",
            "movi v30.16b, #0x3f",
            "ld1 {{v24.16b, v25.16b, v26.16b, v27.16b}}, [{idx}]",
            "ld1 {{v28.16b}}, [{idx2}]",
            "fmov s29, wzr",                    // row accumulator
            "2:",
            "ld1 {{v0.16b}}, [{p}], #16",       // header; pointer → quants
            // ── vectorised unpack_scales_mins ──
            "and  v1.16b, v0.16b, v30.16b",
            "ushr v2.16b, v0.16b, #6",
            "shl  v2.16b, v2.16b, #4",
            "and  v3.16b, v0.16b, v16.16b",
            "ushr v4.16b, v0.16b, #4",
            "tbl  v5.16b, {{v1.16b}}, v24.16b",
            "tbl  v6.16b, {{v3.16b}}, v26.16b",
            "tbl  v7.16b, {{v2.16b}}, v25.16b",
            "orr  v5.16b, v5.16b, v6.16b",
            "orr  v5.16b, v5.16b, v7.16b",
            "tbl  v6.16b, {{v1.16b}}, v27.16b",
            "tbl  v7.16b, {{v4.16b}}, v26.16b",
            "tbl  v1.16b, {{v2.16b}}, v28.16b",
            "orr  v6.16b, v6.16b, v7.16b",
            "orr  v31.16b, v6.16b, v1.16b",     // mn8
            "mov  v15.16b, v0.16b",             // keep header (d|dmin) for epilogue
            "ushll v5.8h, v5.8b, #0",
            "ushll  v18.4s, v5.4h, #0",
            "ushll2 v19.4s, v5.8h, #0",
            // ── sum1 ──
            "movi v17.4s, #0",
            grp!("v18", "0", "1"),
            grp!("v18", "2", "3"),
            grp!("v19", "0", "1"),
            grp!("v19", "2", "3"),
            "addv s17, v17.4s",
            // ── sum2 ──
            "ushll v1.8h, v31.8b, #0",
            "ld1 {{v2.8h}}, [{sums}], #16",
            "smull  v3.4s, v2.4h, v1.4h",
            "smlal2 v3.4s, v2.8h, v1.8h",
            "addv s3, v3.4s",
            // ── epilogue ──
            "ldr  s8, [{d}], #4",               // d_y for this super-block
            "mov  h4, v15.h[0]",
            "fcvt s4, h4",
            "mov  h5, v15.h[1]",
            "fcvt s5, h5",
            "scvtf s6, s17",
            "scvtf s7, s3",
            "fmul s4, s4, s8",
            "fmul s4, s4, s6",
            "fmul s5, s5, s8",
            "fmul s5, s5, s7",
            "fsub s4, s4, s5",
            "fadd s29, s29, s4",
            "subs {n}, {n}, #1",
            "b.ne 2b",
            "fmov {acc:s}, s29",
            p = inout(reg) row => _,
            a = inout(reg) act => _,
            sums = inout(reg) q8_sums => _,
            d = inout(reg) d => _,
            n = inout(reg) n_blocks => _,
            idx = in(reg) Q4K_UNPACK_IDX.as_ptr(),
            idx2 = in(reg) Q4K_UNPACK_IDX.as_ptr().wrapping_add(64),
            acc = out(vreg) acc,
            out("v0") _, out("v1") _, out("v2") _, out("v3") _,
            out("v4") _, out("v5") _, out("v6") _, out("v7") _,
            out("v8") _, out("v15") _,
            out("v16") _, out("v17") _, out("v18") _, out("v19") _,
            out("v20") _, out("v21") _, out("v22") _, out("v23") _,
            out("v24") _, out("v25") _, out("v26") _, out("v27") _,
            out("v28") _, out("v29") _, out("v30") _, out("v31") _,
            options(nostack, readonly),
        );
    }
    acc
}

/// v3 hand-asm Q4_K × Q8_K matvec: one asm block per row (constants hoisted,
/// zero per-super-block Rust glue). Bit-exact with the scalar reference
/// (`q8k_matvec_asm_v3_matches_scalar_bit_exact`).
#[cfg(all(target_arch = "aarch64", target_feature = "neon"))]
pub fn q4k_q8k_matvec_asm_v3(
    out: &mut [f32],
    q8k_x: &Q8KActivation,
    w: &[u8],
    rows: usize,
    cols: usize,
) -> Result<(), KernelShapeError> {
    KernelShapeError::check(
        "q4k_q8k_matvec_asm_v3",
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
        // SAFETY: the row spans n_blocks × 144 bytes (checked above); the
        // activation arrays carry n_blocks super-blocks of qs/sums/d.
        *out_slot = unsafe {
            q4k_row_dot_asm(
                w[r * row_bytes..].as_ptr(),
                q8k_x.qs.as_ptr(),
                q8k_x.sums.as_ptr(),
                q8k_x.d.as_ptr(),
                n_blocks,
            )
        };
    }
    Ok(())
}

/// C12: route Q4_K × Q8_K matvecs through the hand-asm kernel
/// (`q4k_q8k_matvec_asm`) instead of the intrinsic path. **Default on**
/// (`LARQL_Q4K_ASM=0` opts out); both paths are bit-exact. The truth table +
/// caching live in [`crate::options::q4k_asm_enabled`].
#[cfg(all(target_arch = "aarch64", target_feature = "neon"))]
pub(super) fn use_asm_kernel() -> bool {
    crate::options::q4k_asm_enabled()
}
