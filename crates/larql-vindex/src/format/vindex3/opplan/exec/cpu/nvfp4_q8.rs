//! **NVFP4 × Q8** (NVFP4-Q8-1): the stored NVFP4 pack against a Q8
//! activation, multiplied in integers.
//!
//! The weight side is exact. E2M1's magnitudes doubled are the integers
//! `{0, 1, 2, 3, 4, 6, 8, 12}`, so a 16-entry table turns every 4-bit code
//! into an int8 with no rounding at all, and the halving is folded back
//! into the group's scale. What this kernel changes against
//! [`super::kernels::FusedNvfp4`] is ONLY the activation — quantised to
//! int8 once per slab, one scale per NVFP4 group — and the accumulation,
//! which becomes an exact int32 sum per group.

use larql_models::quant::nvfp4::{NVFP4_GROUP_BYTES, NVFP4_GROUP_ELEMS};

use super::super::backend::Nvfp4Activation;
#[cfg(target_arch = "aarch64")]
use super::integer::has_dotprod;
use super::integer::quantise_activation_blocked;
use super::kernels::{e4m3_steps, E2M1_DOUBLED};
use super::projector::{CpuParallelism, DenseProjector, WeightRows};

/// How many activation elements share one Q8 scale: exactly one NVFP4
/// group. A group already pays one f32 multiply for its E4M3 scale, so
/// folding the activation's scale into the same multiply costs nothing,
/// and no coarser block could be cheaper while being less exact.
pub const NVFP4_Q8_ACTIVATION_BLOCK: usize = NVFP4_GROUP_ELEMS;

// The row kernels read activation scale `g` for NVFP4 group `g`, so the
// block IS the group; a different value must fail to build, not index
// past the scale vector.
const _: () = assert!(NVFP4_Q8_ACTIVATION_BLOCK == NVFP4_GROUP_ELEMS);

/// The fused NVFP4 × Q8 kernel.
///
/// [`CpuParallelism::ExternalPool`] for [`super::kernels::FusedNvfp4`]'s
/// reason: it computes the rows it is handed and spawns nothing.
/// Lossy in the ACTIVATION by declaration; its fidelity contract is
/// NVFP4-Q8-1's F verdict, judged against `FusedNvfp4` on the same bytes.
pub struct FusedNvfp4Q8;

impl DenseProjector for FusedNvfp4Q8 {
    fn parallelism(&self) -> CpuParallelism {
        CpuParallelism::ExternalPool
    }

    fn project_rows(&self, weight_rows: WeightRows<'_>, x: &[f32], out: &mut [f32]) {
        let WeightRows::Nvfp4 {
            packed,
            scales,
            tensor_scale,
            activation: Nvfp4Activation::Q8,
        } = weight_rows
        else {
            panic!("the NVFP4 x Q8 kernel consumes NVFP4 packs bound for a Q8 activation only");
        };
        let k = x.len();
        let groups = k / NVFP4_GROUP_ELEMS;
        // Settled before dispatch, as for `FusedNvfp4`: a mismatch is a
        // bug in the slab cut, not a runtime condition to absorb.
        assert!(
            k.is_multiple_of(NVFP4_GROUP_ELEMS)
                && packed.len() == out.len() * groups * NVFP4_GROUP_BYTES
                && scales.len() == out.len() * groups,
            "nvfp4 slab geometry does not describe [{}, {k}]",
            out.len()
        );
        let (qx, xs) = quantise_activation_blocked(x, NVFP4_Q8_ACTIVATION_BLOCK);
        assert_eq!(xs.len(), groups, "one activation scale per NVFP4 group");
        let steps = e4m3_steps();
        for (row, slot) in out.iter_mut().enumerate() {
            *slot = nvfp4_q8_row_dot(
                &packed[row * groups * NVFP4_GROUP_BYTES..][..groups * NVFP4_GROUP_BYTES],
                &scales[row * groups..][..groups],
                tensor_scale,
                steps,
                &qx,
                &xs,
            );
        }
    }
}

/// One group's 16 codes as the int8 the integer dot consumes: DOUBLED
/// E2M1 values, in element order (byte `b` carries element `2b` in its
/// low nibble and `2b + 1` in its high one).
///
/// `code as f32 * (0.5 * tensor_scale * step)` is exactly the reference
/// decoder's value: doubling and halving are exact in binary floating
/// point, so the only rounding either side performs is the same one.
pub fn nvfp4_group_codes(group: &[u8]) -> [i8; NVFP4_GROUP_ELEMS] {
    let mut out = [0i8; NVFP4_GROUP_ELEMS];
    for (b, byte) in group[..NVFP4_GROUP_BYTES].iter().enumerate() {
        out[2 * b] = E2M1_DOUBLED[(byte & 0x0f) as usize];
        out[2 * b + 1] = E2M1_DOUBLED[(byte >> 4) as usize];
    }
    out
}

/// One row against the quantised activation.
#[inline]
fn nvfp4_q8_row_dot(
    packed: &[u8],
    scales: &[u8],
    tensor_scale: f32,
    steps: &[f32; 256],
    qx: &[i8],
    xs: &[f32],
) -> f32 {
    #[cfg(target_arch = "aarch64")]
    if has_dotprod() {
        // SAFETY: guarded by the runtime feature check; the caller cut
        // `packed`/`scales` to one row and `qx`/`xs` to the whole input
        // (8 bytes, one scale, 16 codes and one activation scale per group).
        return unsafe { nvfp4_q8_row_sdot(packed, scales, tensor_scale, steps, qx, xs) };
    }
    nvfp4_q8_row_portable(packed, scales, tensor_scale, steps, qx, xs)
}

/// The SDOT row: two groups per 16-byte load, decoded to int8 by one
/// table lookup each, dotted against 16 activation codes into int32
/// lanes, converted (exactly: `|sum| <= 16 * 12 * 127`) and scaled into
/// two f32 accumulators so consecutive groups' FMAs do not serialise.
///
/// `dotprod` intrinsics are stable since 1.98; see `integer::q8_row_sdot`
/// for why the MSRV lint is allowed here rather than raised.
#[cfg(target_arch = "aarch64")]
#[allow(clippy::incompatible_msrv)]
#[target_feature(enable = "dotprod")]
unsafe fn nvfp4_q8_row_sdot(
    packed: &[u8],
    scales: &[u8],
    tensor_scale: f32,
    steps: &[f32; 256],
    qx: &[i8],
    xs: &[f32],
) -> f32 {
    use std::arch::aarch64::*;
    let lut = vld1q_s8(E2M1_DOUBLED.as_ptr());
    let (pp, qp) = (packed.as_ptr(), qx.as_ptr());
    let zero = vdupq_n_s32(0);
    let scale = |g: usize| steps[*scales.get_unchecked(g) as usize] * *xs.get_unchecked(g);
    let (mut acc0, mut acc1) = (vdupq_n_f32(0.0), vdupq_n_f32(0.0));
    let groups = scales.len();
    let mut g = 0usize;
    // Eight groups — one 64-byte line of codes — per step. Each group's
    // int32 lanes reduce pairwise into ONE lane of a vector holding four
    // groups, so the conversion and the scale FMA run once per four
    // groups instead of once per group.
    let decode2 = |at: usize| {
        let raw = vld1q_u8(pp.add(at * NVFP4_GROUP_BYTES));
        let lo = vandq_u8(raw, vdupq_n_u8(0x0f));
        let hi = vshrq_n_u8::<4>(raw);
        (
            vqtbl1q_s8(lut, vzip1q_u8(lo, hi)),
            vqtbl1q_s8(lut, vzip2q_u8(lo, hi)),
        )
    };
    let dot =
        |c: int8x16_t, at: usize| vdotq_s32(zero, c, vld1q_s8(qp.add(at * NVFP4_GROUP_ELEMS)));
    let four = |at: usize| {
        let (c0, c1) = decode2(at);
        let (c2, c3) = decode2(at + 2);
        let sums = vpaddq_s32(
            vpaddq_s32(dot(c0, at), dot(c1, at + 1)),
            vpaddq_s32(dot(c2, at + 2), dot(c3, at + 3)),
        );
        // Straight into lanes from the table: an array built on the stack
        // and reloaded as a vector stalls on store-to-load forwarding.
        let step = |i: usize| steps.as_ptr().add(*scales.get_unchecked(at + i) as usize);
        let mut w = vld1q_dup_f32(step(0));
        w = vld1q_lane_f32::<1>(step(1), w);
        w = vld1q_lane_f32::<2>(step(2), w);
        w = vld1q_lane_f32::<3>(step(3), w);
        let scale = vmulq_f32(w, vld1q_f32(xs.as_ptr().add(at)));
        (vcvtq_f32_s32(sums), scale)
    };
    while g + 8 <= groups {
        let (s0, k0) = four(g);
        let (s1, k1) = four(g + 4);
        acc0 = vfmaq_f32(acc0, s0, k0);
        acc1 = vfmaq_f32(acc1, s1, k1);
        g += 8;
    }
    while g + 2 <= groups {
        let raw = vld1q_u8(pp.add(g * NVFP4_GROUP_BYTES));
        let lo = vandq_u8(raw, vdupq_n_u8(0x0f));
        let hi = vshrq_n_u8::<4>(raw);
        // Element 2b is byte b's low nibble, 2b+1 its high one: the first
        // zip is group g in element order, the second group g + 1.
        let c0 = vqtbl1q_s8(lut, vzip1q_u8(lo, hi));
        let c1 = vqtbl1q_s8(lut, vzip2q_u8(lo, hi));
        let d0 = vdotq_s32(zero, c0, vld1q_s8(qp.add(g * NVFP4_GROUP_ELEMS)));
        let d1 = vdotq_s32(zero, c1, vld1q_s8(qp.add((g + 1) * NVFP4_GROUP_ELEMS)));
        acc0 = vfmaq_n_f32(acc0, vcvtq_f32_s32(d0), scale(g));
        acc1 = vfmaq_n_f32(acc1, vcvtq_f32_s32(d1), scale(g + 1));
        g += 2;
    }
    if g < groups {
        let raw = vld1_u8(pp.add(g * NVFP4_GROUP_BYTES));
        let lo = vand_u8(raw, vdup_n_u8(0x0f));
        let hi = vshr_n_u8::<4>(raw);
        let c0 = vqtbl1q_s8(lut, vcombine_u8(vzip1_u8(lo, hi), vzip2_u8(lo, hi)));
        let d0 = vdotq_s32(zero, c0, vld1q_s8(qp.add(g * NVFP4_GROUP_ELEMS)));
        acc0 = vfmaq_n_f32(acc0, vcvtq_f32_s32(d0), scale(g));
    }
    0.5 * tensor_scale * vaddvq_f32(vaddq_f32(acc0, acc1))
}

/// **The DEFINITION** the vector path must agree with: per group, the
/// EXACT int32 `Σ code_i · qx_i`, scaled once by `e4m3(scale) · xs_g`;
/// the tensor scale, halved to undo the doubling, once per row.
pub fn nvfp4_q8_row_portable(
    packed: &[u8],
    scales: &[u8],
    tensor_scale: f32,
    steps: &[f32; 256],
    qx: &[i8],
    xs: &[f32],
) -> f32 {
    let mut acc = 0.0f32;
    for (g, &scale) in scales.iter().enumerate() {
        let codes = nvfp4_group_codes(&packed[g * NVFP4_GROUP_BYTES..]);
        let q = &qx[g * NVFP4_GROUP_ELEMS..][..NVFP4_GROUP_ELEMS];
        let sum: i32 = codes
            .iter()
            .zip(q)
            .map(|(&w, &a)| w as i32 * a as i32)
            .sum();
        acc += sum as f32 * (steps[scale as usize] * xs[g]);
    }
    0.5 * tensor_scale * acc
}
