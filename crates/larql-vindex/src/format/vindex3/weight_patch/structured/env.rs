//! The floating-point environment the contract needs: gradual underflow, in f64 and f32.
//!
//! A library loaded into the process can switch on flush-to-zero (FTZ) or denormals-are-zero (DAZ) for the
//! whole thread, silently. These canaries run on **runtime** values through [`std::hint::black_box`]: a constant
//! expression is folded at compile time, under the compiler's IEEE semantics, and would pass under any FPU mode.
//! Results are compared by **bits**, because under DAZ a comparison would read a subnormal as zero too.

use std::hint::black_box;

/// The platform cannot run the contract's arithmetic exactly. This is not an op refusal: the op may be valid,
/// but no byte it produced here could be trusted.
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
#[error("the floating-point environment is not IEEE-754 gradual underflow: {0}")]
pub struct FloatEnvironmentError(pub &'static str);

/// What each canary computed, as bits, and what IEEE-754 requires.
pub struct Canaries {
    pub f64_half_min: u64,
    pub f64_sub_times_one: u64,
    pub f32_half_min: u32,
    pub f32_sub_times_one: u32,
}

impl Canaries {
    /// Measure on this thread, now.
    pub fn measure() -> Canaries {
        Canaries {
            f64_half_min: (black_box(f64::MIN_POSITIVE) / black_box(2.0f64)).to_bits(),
            f64_sub_times_one: (f64::from_bits(black_box(1u64)) * black_box(1.0f64)).to_bits(),
            f32_half_min: (black_box(f32::MIN_POSITIVE) / black_box(2.0f32)).to_bits(),
            f32_sub_times_one: (f32::from_bits(black_box(1u32)) * black_box(1.0f32)).to_bits(),
        }
    }

    /// The first canary that differs from IEEE-754, if any.
    pub fn verdict(&self) -> Result<(), FloatEnvironmentError> {
        let want = Canaries {
            f64_half_min: (f64::MIN_POSITIVE / 2.0).to_bits(),
            f64_sub_times_one: 1,
            f32_half_min: (f32::MIN_POSITIVE / 2.0).to_bits(),
            f32_sub_times_one: 1,
        };
        let checks = [
            (
                self.f64_half_min == want.f64_half_min,
                "f64 MIN_POSITIVE / 2 is flushed (FTZ)",
            ),
            (
                self.f64_sub_times_one == want.f64_sub_times_one,
                "an f64 subnormal is read as zero (DAZ)",
            ),
            (
                self.f32_half_min == want.f32_half_min,
                "f32 MIN_POSITIVE / 2 is flushed (FTZ)",
            ),
            (
                self.f32_sub_times_one == want.f32_sub_times_one,
                "an f32 subnormal is read as zero (DAZ)",
            ),
        ];
        match checks.into_iter().find(|(ok, _)| !ok) {
            Some((_, what)) => Err(FloatEnvironmentError(what)),
            None => Ok(()),
        }
    }
}

/// Refuse unless subnormal results and inputs behave as IEEE-754 requires, in both working formats. Whatever
/// applies structured ops runs this first, on the thread that computes; [`super::run`] itself is pure.
pub fn check_float_environment() -> Result<(), FloatEnvironmentError> {
    Canaries::measure().verdict()
}
