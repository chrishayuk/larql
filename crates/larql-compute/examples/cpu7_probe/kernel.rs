//! The weight-stationary row kernel, and the reference that keeps it honest.
//!
//! [`row_stationary`] is a transcription of the frozen CPU-5 K5 row
//! (`q8_row_b16_register_sdot`) generalised over `N` activation vectors.
//! The single change is the loop nest:
//!
//! ```text
//! frozen:      for row { for block {           load w, one SDOT set } }
//! stationary:  for row { for block { load w, for n { one SDOT set } } }
//! ```
//!
//! The weight vectors `w0..w3` are loaded ONCE per block and applied to
//! every activation. That is the whole experiment: at `N = 1` the two are
//! the same loop, so the `N = 1` arm doubles as the control that the
//! transcription still reproduces the banked rate.
//!
//! **Two things are amortised here, not one**, and the report must not
//! conflate them:
//!
//!   1. the weight LOADS — the memory traffic the programme is about;
//!   2. the weight-code SUMS `sv` — the asymmetric path's `w . 1` term,
//!      which depends only on the weights, so a stationary kernel computes
//!      it once where `N` separate calls compute it `N` times.
//!
//! (2) is real and is part of what weight-stationary buys, but it is
//! ARITHMETIC, not traffic. If the amortisation curve is good while the
//! weight-GB/s is flat, (2) is what moved and the finding is smaller than
//! it looks. The ledger in `arms` reports both so the two cannot be told
//! apart by hope.
//!
//! Bit-identity to the frozen kernel is achievable and required (control
//! C3): the per-`n` operation ORDER is preserved exactly — the `dv` fma
//! first, then the `mid` fma — and `ws * ascale[b]` is the same `f32`
//! whether one vector or eight are in flight. Hoisting `sv` changes when
//! it is computed, never its value.

/// Elements one `SDOT` consumes. Sixteen `i8` lanes, four `i32` out.
pub const SDOT_LANES: usize = 16;

/// Activation blocks inside one weight block at `ablock` 16, `block` 64.
/// The value that makes the weight scale constant within a group, which
/// is what let K5 delete its fold buffers.
pub const PER_WEIGHT_B16: usize = 4;

/// Elements sharing one weight scale.
pub const WEIGHT_BLOCK: usize = SDOT_LANES * PER_WEIGHT_B16;

/// Elements sharing one activation scale and midpoint.
pub const ACT_BLOCK: usize = SDOT_LANES;

/// `N` activation vectors, already quantised, in the layout the kernel
/// consumes. Slices rather than owned data: the caller holds one set for
/// the whole sweep and every row reads it again.
///
/// Only the NEON row reads these fields, so off `aarch64` — where `main`
/// refuses to run at all — they are legitimately unused.
#[cfg_attr(not(target_arch = "aarch64"), allow(dead_code))]
pub struct ActVectors<'a, const N: usize> {
    pub codes: [&'a [i8]; N],
    pub scales: [&'a [f32]; N],
    pub mids: [&'a [f32]; N],
}

/// Whether this machine has the `SDOT` the whole programme is built on.
pub fn has_dotprod() -> bool {
    #[cfg(target_arch = "aarch64")]
    {
        std::arch::is_aarch64_feature_detected!("dotprod")
    }
    #[cfg(not(target_arch = "aarch64"))]
    {
        false
    }
}

/// One Q8 row against `N` quantised activations, weight-stationary.
///
/// `out[n]` receives this row's contribution for activation `n`.
///
/// # Safety
/// Requires `dotprod`, checked by [`has_dotprod`]. `in_dim` must be a
/// multiple of [`WEIGHT_BLOCK`] — the probe restricts itself to the
/// aligned production shape so that no tail path is in the measurement,
/// and asserts this at fixture build time rather than trusting the caller.
#[cfg(target_arch = "aarch64")]
#[allow(clippy::incompatible_msrv)]
#[target_feature(enable = "dotprod")]
pub unsafe fn row_stationary<const N: usize>(
    codes: &[i8],
    wscales: &[f32],
    act: &ActVectors<'_, N>,
    in_dim: usize,
    out: &mut [f32; N],
) {
    use std::arch::aarch64::*;
    debug_assert!(in_dim.is_multiple_of(WEIGHT_BLOCK));
    let ones = vdupq_n_s8(1);
    let z = vdupq_n_s32(0);
    let blocks = in_dim / SDOT_LANES;
    let mut acc = [vdupq_n_f32(0.0); N];
    let mut b = 0usize;
    while b + PER_WEIGHT_B16 <= blocks {
        let i0 = b * SDOT_LANES;
        // Loaded ONCE. Everything below reuses these four registers.
        let w0 = vld1q_s8(codes.as_ptr().add(i0));
        let w1 = vld1q_s8(codes.as_ptr().add(i0 + SDOT_LANES));
        let w2 = vld1q_s8(codes.as_ptr().add(i0 + 2 * SDOT_LANES));
        let w3 = vld1q_s8(codes.as_ptr().add(i0 + 3 * SDOT_LANES));
        // The `w . 1` term depends only on the weights, so it is hoisted
        // out of the activation loop. See the module note: this is the
        // ARITHMETIC half of what stationarity buys.
        let s0 = vdotq_s32(z, w0, ones);
        let s1 = vdotq_s32(z, w1, ones);
        let s2 = vdotq_s32(z, w2, ones);
        let s3 = vdotq_s32(z, w3, ones);
        let svf = vcvtq_f32_s32(vpaddq_s32(vpaddq_s32(s0, s1), vpaddq_s32(s2, s3)));
        let ws = *wscales.get_unchecked(b / PER_WEIGHT_B16);
        for n in 0..N {
            let qx = act.codes.get_unchecked(n).as_ptr();
            let d0 = vdotq_s32(z, w0, vld1q_s8(qx.add(i0)));
            let d1 = vdotq_s32(z, w1, vld1q_s8(qx.add(i0 + SDOT_LANES)));
            let d2 = vdotq_s32(z, w2, vld1q_s8(qx.add(i0 + 2 * SDOT_LANES)));
            let d3 = vdotq_s32(z, w3, vld1q_s8(qx.add(i0 + 3 * SDOT_LANES)));
            let dv = vpaddq_s32(vpaddq_s32(d0, d1), vpaddq_s32(d2, d3));
            // Same two fused multiply-adds, in the same order, as the
            // frozen row. Reordering them would be a different f32.
            let scale_v = vmulq_n_f32(vld1q_f32(act.scales.get_unchecked(n).as_ptr().add(b)), ws);
            let a = vfmaq_f32(*acc.get_unchecked(n), scale_v, vcvtq_f32_s32(dv));
            let mid_v = vmulq_n_f32(vld1q_f32(act.mids.get_unchecked(n).as_ptr().add(b)), ws);
            *acc.get_unchecked_mut(n) = vfmaq_f32(a, mid_v, svf);
        }
        b += PER_WEIGHT_B16;
    }
    for n in 0..N {
        *out.get_unchecked_mut(n) = vaddvq_f32(*acc.get_unchecked(n));
    }
}

/// The same row, in `f64` scalar arithmetic, decoded from the stated
/// representation rather than transcribed from the kernel.
///
/// Exists so that a transcription error in [`row_stationary`] cannot pass
/// as a result. Bit-identity across `N` only proves the arms agree with
/// EACH OTHER; if the transcription dropped a term they would agree on
/// the same wrong number. This reference is written from the format
/// definition — `x_i ~ ascale[b] * qx_i + amid[b]` — and never from the
/// kernel, so the two can only agree by both being right.
pub fn row_reference(
    codes: &[i8],
    wscales: &[f32],
    qx: &[i8],
    ascales: &[f32],
    amids: &[f32],
    in_dim: usize,
) -> f64 {
    let mut acc = 0.0f64;
    for b in 0..in_dim / ACT_BLOCK {
        let ws = wscales[b / PER_WEIGHT_B16] as f64;
        let (asc, amid) = (ascales[b] as f64, amids[b] as f64);
        let (mut dot, mut wsum) = (0.0f64, 0.0f64);
        for i in b * ACT_BLOCK..(b + 1) * ACT_BLOCK {
            let w = codes[i] as f64;
            dot += w * qx[i] as f64;
            wsum += w;
        }
        acc += ws * (asc * dot + amid * wsum);
    }
    acc
}

/// Non-aarch64 stub.
///
/// Present so this example COMPILES on every target the workspace builds
/// for — a `cfg`-gated path that only exists on one architecture has broken
/// CI here twice. It is never reached: `main` refuses to run without
/// [`has_dotprod`], which is `false` everywhere this stub is compiled.
#[cfg(not(target_arch = "aarch64"))]
pub unsafe fn row_stationary<const N: usize>(
    _codes: &[i8],
    _wscales: &[f32],
    _act: &ActVectors<'_, N>,
    _in_dim: usize,
    _out: &mut [f32; N],
) {
    unreachable!("cpu7_probe refuses to run without aarch64 dotprod")
}
