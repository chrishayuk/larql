//! One IEEE-754 operation at a time, in the profile's working format, and the conversion chain to BF16.
//!
//! Every function here is a single `*`, `+` or `-` in `f64` or `f32`, followed by a finiteness check. Rust
//! rounds each of these to nearest, ties to even, and never fuses them. The only fused operation is
//! `mul_add`, used by [`Mutation::FmaInReduction`] and by the exact-product helper of
//! [`Mutation::F32Accumulation`], neither of which is the contract's semantics.

use super::refusal::{Refusal, RefusalCode};
use super::Mutation;
use crate::format::vindex3::weight_patch::bf16::{bf16_to_f32, round_to_bf16};

/// BF16 bits of the sign.
const BF16_SIGN: u16 = 0x8000;
/// BF16 exponent field.
const BF16_EXPONENT: u16 = 0x7F80;

/// A working format: `f64` or `f32`.
pub(super) trait Work: Copy + PartialEq + std::fmt::Debug {
    const IS_F64: bool;
    fn zero() -> Self;
    fn widen_bf16(bits: u16) -> Self;
    fn mul(self, o: Self) -> Self;
    fn add(self, o: Self) -> Self;
    fn sub(self, o: Self) -> Self;
    fn fused(self, b: Self, c: Self) -> Self;
    fn finite(self) -> bool;
    fn subnormal(self) -> bool;
    /// A zero with the sign of `self`.
    fn signed_zero(self) -> Self;
    /// Exact.
    fn to_f64(self) -> f64;
    /// Exact for any value of `Self` carried as `f64`.
    fn from_f64_exact(x: f64) -> Self;
}

impl Work for f64 {
    const IS_F64: bool = true;
    fn zero() -> f64 {
        0.0
    }
    fn widen_bf16(bits: u16) -> f64 {
        bf16_to_f32(bits) as f64
    }
    fn mul(self, o: f64) -> f64 {
        self * o
    }
    fn add(self, o: f64) -> f64 {
        self + o
    }
    fn sub(self, o: f64) -> f64 {
        self - o
    }
    fn fused(self, b: f64, c: f64) -> f64 {
        self.mul_add(b, c)
    }
    fn finite(self) -> bool {
        self.is_finite()
    }
    fn subnormal(self) -> bool {
        self.is_subnormal()
    }
    fn signed_zero(self) -> f64 {
        0.0f64.copysign(self)
    }
    fn to_f64(self) -> f64 {
        self
    }
    fn from_f64_exact(x: f64) -> f64 {
        x
    }
}

impl Work for f32 {
    const IS_F64: bool = false;
    fn zero() -> f32 {
        0.0
    }
    fn widen_bf16(bits: u16) -> f32 {
        bf16_to_f32(bits)
    }
    fn mul(self, o: f32) -> f32 {
        self * o
    }
    fn add(self, o: f32) -> f32 {
        self + o
    }
    fn sub(self, o: f32) -> f32 {
        self - o
    }
    fn fused(self, b: f32, c: f32) -> f32 {
        self.mul_add(b, c)
    }
    fn finite(self) -> bool {
        self.is_finite()
    }
    fn subnormal(self) -> bool {
        self.is_subnormal()
    }
    fn signed_zero(self) -> f32 {
        0.0f32.copysign(self)
    }
    fn to_f64(self) -> f64 {
        self as f64
    }
    fn from_f64_exact(x: f64) -> f32 {
        x as f32
    }
}

/// The arithmetic of one op run: the contract's, or a named wrong one.
#[derive(Debug, Clone, Copy)]
pub(super) struct Arith {
    pub mutation: Mutation,
}

impl Arith {
    fn ftz(self) -> bool {
        self.mutation == Mutation::FtzDaz
    }

    /// An input as the arithmetic reads it (DAZ under [`Mutation::FtzDaz`]).
    fn read<W: Work>(self, x: W) -> W {
        if self.ftz() && x.subnormal() {
            x.signed_zero()
        } else {
            x
        }
    }

    /// A result: flushed under [`Mutation::FtzDaz`], refused when not finite.
    pub fn result<W: Work>(self, x: W) -> Result<W, Refusal> {
        if !x.finite() {
            return Err(Refusal::new(
                RefusalCode::NonFiniteIntermediate,
                format!("{x:?} in the working format"),
            ));
        }
        Ok(if self.ftz() && x.subnormal() {
            x.signed_zero()
        } else {
            x
        })
    }

    pub fn mul<W: Work>(self, a: W, b: W) -> Result<W, Refusal> {
        self.result(self.read(a).mul(self.read(b)))
    }

    pub fn add<W: Work>(self, a: W, b: W) -> Result<W, Refusal> {
        self.result(self.read(a).add(self.read(b)))
    }

    pub fn sub<W: Work>(self, a: W, b: W) -> Result<W, Refusal> {
        self.result(self.read(a).sub(self.read(b)))
    }

    /// `a·b + c`, rounded once. Only [`Mutation::FmaInReduction`] calls this.
    pub fn fused<W: Work>(self, a: W, b: W, c: W) -> Result<W, Refusal> {
        self.result(self.read(a).fused(self.read(b), self.read(c)))
    }

    /// A working-format value converted to BF16 bits by the profile's chain (`through_f32`), or `overflow`.
    pub fn to_bf16<W: Work>(self, x: W, through_f32: bool) -> Result<u16, Refusal> {
        let mut v = x.to_f64();
        if through_f32 {
            let y = v as f32; // round to nearest, ties to even; ±inf past the f32 range
            if !y.is_finite() {
                return Err(Refusal::new(
                    RefusalCode::Overflow,
                    format!("{v:e} overflows at the f32 step"),
                ));
            }
            v = if self.ftz() && y.is_subnormal() {
                0.0f64.copysign(v)
            } else {
                y as f64
            };
        }
        let bits = round_to_bf16(v)
            .ok_or_else(|| Refusal::new(RefusalCode::Overflow, format!("{v:e} overflows BF16")))?;
        let subnormal = bits & BF16_EXPONENT == 0 && bits & !BF16_SIGN != 0;
        Ok(if self.mutation == Mutation::Bf16OutputFlush && subnormal {
            bits & BF16_SIGN
        } else {
            bits
        })
    }
}

/// The exact product `u·w` rounded once to f32. The f64 product is rounded already, so it can sit exactly on
/// an f32 rounding midpoint that the exact product is not on; the exact error term, from a fused multiply-add,
/// decides that tie. Away from a midpoint the f64 rounding cannot move the f32 result: an f32 midpoint is an f64
/// value, so it cannot lie strictly between the exact product and its nearest f64.
pub(super) fn exact_product_to_f32(u: f64, w: f64) -> f32 {
    let p = u * w;
    let r = p as f32;
    let err = u.mul_add(w, -p);
    if err == 0.0 || !p.is_finite() {
        return r;
    }
    if r.is_infinite() {
        // p is at or past f32's overflow threshold, the tie above f32::MAX. Only an exact product just below
        // the tie itself rounds back to f32::MAX.
        let tie = f32::MAX as f64
            + (f32::MAX as f64 - f32::from_bits(f32::MAX.to_bits() - 1) as f64) / 2.0;
        let below = (p > 0.0 && err < 0.0) || (p < 0.0 && err > 0.0);
        return if p.abs() == tie && below {
            f32::MAX.copysign(r)
        } else {
            r
        };
    }
    let diff = p - r as f64; // exact: p and r are close and both f64 values
    let other = (r as f64 + 2.0 * diff) as f32;
    let tie = diff != 0.0 && other as f64 == r as f64 + 2.0 * diff;
    if tie && (err > 0.0) == (diff > 0.0) {
        other
    } else {
        r
    }
}
