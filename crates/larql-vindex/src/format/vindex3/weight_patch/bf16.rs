//! BF16 element arithmetic for weight patches.
//!
//! The contract is one rounding of the exact result: a patched element is
//! `RNE_bf16(old + delta)`, where `old` is the stored bfloat16 value and
//! `delta` is an `f32`, added **as real numbers** and rounded once to the
//! nearest bfloat16, ties to even. Adding in `f32` first and rounding that to
//! bfloat16 is two roundings and can differ from the exact sum on a tie, so
//! the sum is taken in `f64` together with its exact rounding error.
//!
//! A bfloat16 has 8 significant bits and an `f32` 24, so the sum is exact in
//! `f64` (53 bits) whenever the two exponents are within 29 of each other.
//! Past that gap the `f64` sum drops the smaller term. That matters in one
//! case only: the `f64` sum lands exactly halfway between two bfloat16
//! values while the exact sum does not, which happens when the larger term
//! is itself on a tie (a delta of `2^60·(1 + 2^-8)` added to `1.0`). The
//! dropped part then decides the direction, not the parity of the
//! neighbours. Anywhere else the `f64` sum and the exact sum round alike: a
//! halfway point is an `f64`, so none lies strictly between a sum and its
//! nearest `f64`.

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

/// The fractional part of a quotient that sits exactly halfway between two bfloat16 values.
const HALFWAY: f64 = 0.5;

/// The nearest bfloat16 to `x`, ties to even, or `None` when `x` is not
/// finite or rounds to a magnitude of `2^128` or more.
pub fn round_to_bf16(x: f64) -> Option<u16> {
    round_exact_to_bf16(x, 0.0)
}

/// The nearest bfloat16 to the real number `x + residue`, ties to even, where
/// `residue` is what the `f64` addition that produced `x` rounded away (zero
/// when it was exact). The residue only ever breaks a tie.
fn round_exact_to_bf16(x: f64, residue: f64) -> Option<u16> {
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
    let steps = x / step;
    let on_a_tie = (steps - steps.trunc()).abs() == HALFWAY;
    let whole = if on_a_tie && residue > 0.0 {
        steps.ceil()
    } else if on_a_tie && residue < 0.0 {
        steps.floor()
    } else {
        steps.round_ties_even()
    };
    let rounded = whole * step;
    if rounded.abs() >= pow2(OVERFLOW_EXPONENT) {
        return None;
    }
    // `rounded` is a bfloat16 value, which an f32 holds exactly.
    Some(((rounded as f32).to_bits() >> 16) as u16)
}

/// `RNE_bf16(old + delta)`, or `None` when the result overflows.
pub fn patched_bits(old: u16, delta: f32) -> Option<u16> {
    let (stored, added) = (bf16_to_f32(old) as f64, delta as f64);
    let sum = stored + added;
    if !sum.is_finite() {
        return None;
    }
    // Knuth's two-sum: `stored + added == sum + residue` exactly. Both terms
    // are f32 values, so nothing here overflows.
    let added_part = sum - stored;
    let residue = (stored - (sum - added_part)) + (added - added_part);
    round_exact_to_bf16(sum, residue)
}
