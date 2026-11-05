//! BF16 element arithmetic for weight patches.
//!
//! The contract is one rounding of the exact result: a patched element is
//! `RNE_bf16(old + delta)`, where `old` is the stored bfloat16 value and
//! `delta` is an `f32`, added **as real numbers** and rounded once to the
//! nearest bfloat16, ties to even. Adding in `f32` first and rounding that to
//! bfloat16 is two roundings and can differ from the exact sum on a tie, so
//! the sum is taken in `f64`, which holds it exactly whenever it can matter.
//!
//! Why `f64` is exact enough: a bfloat16 has 8 significant bits and an `f32`
//! 24, so the sum is exact in `f64` (53 bits) whenever the two exponents are
//! within 29 of each other. Past that gap the `f32` is far below half a
//! bfloat16 step of the old value, the sum rounds back to `old` whatever the
//! rounding of the sum, and no tie is possible.

/// Smallest normal exponent of a bfloat16 (same range as `f32`).
const MIN_NORMAL_EXPONENT: i32 = -126;
/// Explicit mantissa bits of a bfloat16.
const MANTISSA_BITS: i32 = 7;
/// First exponent that does not fit: values at or above `2^128` overflow.
const OVERFLOW_EXPONENT: i32 = 128;
const F64_EXPONENT_BIAS: i32 = 1023;
const F64_MANTISSA_BITS: u32 = 52;
const F64_EXPONENT_MASK: u64 = 0x7ff;
const BF16_SIGN_BIT: u16 = 0x8000;
/// Exponent field all ones: infinity or NaN.
const BF16_EXPONENT_MASK: u16 = 0x7F80;

/// `2^k` exactly, for `k` in the `f64` normal range.
fn pow2(k: i32) -> f64 {
    debug_assert!((-1022..=1023).contains(&k));
    f64::from_bits(((F64_EXPONENT_BIAS + k) as u64) << F64_MANTISSA_BITS)
}

/// A stored bfloat16 widened to `f32` (exact).
pub fn bf16_to_f32(bits: u16) -> f32 {
    f32::from_bits((bits as u32) << 16)
}

/// Whether a stored bfloat16 is finite (not an infinity or a NaN).
pub fn is_finite(bits: u16) -> bool {
    bits & BF16_EXPONENT_MASK != BF16_EXPONENT_MASK
}

/// The nearest bfloat16 to `x`, ties to even, or `None` when `x` is not
/// finite or rounds to a magnitude of `2^128` or more.
pub fn round_to_bf16(x: f64) -> Option<u16> {
    if !x.is_finite() {
        return None;
    }
    let sign = if x.is_sign_negative() {
        BF16_SIGN_BIT
    } else {
        0
    };
    let field = ((x.to_bits() >> F64_MANTISSA_BITS) & F64_EXPONENT_MASK) as i32;
    if x == 0.0 || field == 0 {
        // Zero, or an f64 subnormal: far below the smallest bfloat16 step.
        return Some(sign);
    }
    let exponent = (field - F64_EXPONENT_BIAS).max(MIN_NORMAL_EXPONENT);
    let step = pow2(exponent - MANTISSA_BITS);
    let rounded = (x / step).round_ties_even() * step;
    if rounded.abs() >= pow2(OVERFLOW_EXPONENT) {
        return None;
    }
    // `rounded` is a bfloat16 value, which an f32 holds exactly.
    Some(((rounded as f32).to_bits() >> 16) as u16)
}

/// `RNE_bf16(old + delta)`, or `None` when the result overflows.
pub fn patched_bits(old: u16, delta: f32) -> Option<u16> {
    round_to_bf16(bf16_to_f32(old) as f64 + delta as f64)
}
