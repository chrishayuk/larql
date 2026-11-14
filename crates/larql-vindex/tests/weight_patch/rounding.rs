//! The rounding contract: one rounding of the exact sum, ties to even.

use larql_vindex::format::vindex3::weight_patch::bf16::{
    bf16_to_f32, is_finite, patched_bits, round_to_bf16,
};

#[test]
fn an_exact_sum_that_needs_no_rounding_is_exact() {
    // 0.0317 + 2^-7 has bits that fit; powers of two keep the sum representable.
    let old = 0x3d02; // 0.03173828125
    assert_eq!(patched_bits(old, 0.0078125).unwrap(), 0x3d22);
}

#[test]
fn a_tie_goes_to_the_even_neighbour() {
    // 1.0 + 2^-8 is exactly halfway between 1.0 (even mantissa) and 1 + 2^-7.
    assert_eq!(patched_bits(0x3f80, 2f32.powi(-8)).unwrap(), 0x3f80);
    // 1 + 2^-7 (odd mantissa) + 2^-8 is halfway to 1 + 2^-6 (even): rounds up.
    assert_eq!(patched_bits(0x3f81, 2f32.powi(-8)).unwrap(), 0x3f82);
}

#[test]
fn the_sum_is_rounded_once_not_twice() {
    // 1.0 + (2^-8 + 2^-30) is just ABOVE the halfway point, so the exact sum
    // rounds up to 1 + 2^-7. Rounding the f32 sum first loses the 2^-30
    // (f32 keeps 23 fraction bits), lands exactly on the tie, and goes to the
    // even 1.0: two roundings give a different answer.
    let delta = 2f32.powi(-8) + 2f32.powi(-30);
    assert_eq!(
        delta as f64,
        2f64.powi(-8) + 2f64.powi(-30),
        "the delta itself is exact in f32"
    );
    let via_f32 = round_to_bf16((bf16_to_f32(0x3f80) + delta) as f64).unwrap();
    assert_eq!(via_f32, 0x3f80, "double rounding loses the bit");
    assert_eq!(
        patched_bits(0x3f80, delta).unwrap(),
        0x3f81,
        "the exact sum rounds up"
    );
}

#[test]
fn a_carry_moves_into_the_next_binade() {
    // The largest mantissa in a binade plus a step carries into the exponent.
    assert_eq!(patched_bits(0x3fff, 2f32.powi(-8) * 1.5).unwrap(), 0x4000);
}

#[test]
fn signs_and_cancellation_behave() {
    let neg = 0xbd82; // -0.0634765625
    assert_eq!(patched_bits(neg, 0.0078125).unwrap(), 0xbd64);
    // A sum that cancels exactly is +0.
    assert_eq!(patched_bits(0x3f80, -1.0).unwrap(), 0x0000);
    // Rounding to zero keeps the sign of the exact result.
    assert_eq!(round_to_bf16(-1e-300), Some(0x8000));
}

#[test]
fn bfloat16_subnormals_round_on_their_own_grid() {
    // The smallest bfloat16 step is 2^-133; half of it ties to even (zero).
    assert_eq!(round_to_bf16(2f64.powi(-134)), Some(0x0000));
    assert_eq!(round_to_bf16(2f64.powi(-133)), Some(0x0001));
    assert_eq!(
        round_to_bf16(1.5 * 2f64.powi(-133)),
        Some(0x0002),
        "ties to the even neighbour"
    );
}

#[test]
fn overflow_and_non_finite_are_refused_not_saturated() {
    assert_eq!(round_to_bf16(f64::INFINITY), None);
    assert_eq!(round_to_bf16(f64::NAN), None);
    assert_eq!(round_to_bf16(2f64.powi(128)), None);
    assert!(
        round_to_bf16(3.3895e38).is_some(),
        "the largest finite bfloat16 is representable"
    );
    assert!(is_finite(0x7f7f) && !is_finite(0x7f80) && !is_finite(0xffc0));
}

/// The nearest bfloat16 to `old + delta` by exact integer arithmetic: an oracle that shares no code with
/// `patched_bits`. Both values are scaled to a common power of two and summed in a `u128`, then the 8-bit
/// significand is rounded by comparing the remainder against half. Positive normals whose sum fits 128 bits.
fn exact_oracle(old: u16, delta: f32) -> u16 {
    const F32_FRACTION_BITS: u32 = 23;
    const F32_EXPONENT_MASK: u32 = 0xff;
    const F32_BIAS: i32 = 127;
    const BF16_SIGNIFICAND_BITS: u32 = 8;
    const BF16_FRACTION_MASK: u128 = 0x7f;
    const WIDEST_SUM_BITS: u32 = 127;
    // (integer significand, exponent of its unit) of a positive normal f32.
    let parts = |x: f32| -> (u128, i32) {
        let bits = x.to_bits();
        let field = ((bits >> F32_FRACTION_BITS) & F32_EXPONENT_MASK) as i32;
        assert!(x > 0.0 && field > 0, "the oracle takes positive normals");
        let significand = (bits & ((1 << F32_FRACTION_BITS) - 1)) | (1 << F32_FRACTION_BITS);
        (
            significand as u128,
            field - F32_BIAS - F32_FRACTION_BITS as i32,
        )
    };
    let (a, ea) = parts(bf16_to_f32(old));
    let (b, eb) = parts(delta);
    let unit = ea.min(eb);
    assert!(
        (ea.max(eb) - unit) as u32 + F32_FRACTION_BITS + 2 <= WIDEST_SUM_BITS,
        "the oracle's sum must fit a u128"
    );
    let sum = (a << (ea - unit)) + (b << (eb - unit));
    let drop = (u128::BITS - sum.leading_zeros()) - BF16_SIGNIFICAND_BITS;
    let mut kept = sum >> drop;
    let remainder = sum - (kept << drop);
    let half = 1u128 << (drop - 1);
    if remainder > half || (remainder == half && kept % 2 == 1) {
        kept += 1;
    }
    let mut exponent = unit + drop as i32 + (BF16_SIGNIFICAND_BITS as i32 - 1);
    if kept >> BF16_SIGNIFICAND_BITS != 0 {
        kept >>= 1;
        exponent += 1;
    }
    let fraction =
        ((kept & BF16_FRACTION_MASK) as u32) << (F32_FRACTION_BITS + 1 - BF16_SIGNIFICAND_BITS);
    ((((exponent + F32_BIAS) as u32) << F32_FRACTION_BITS | fraction) >> 16) as u16
}

#[test]
fn a_delta_on_a_tie_far_above_the_stored_value_rounds_by_the_exact_sum() {
    // 2^60·(1 + 2^-8) is halfway between two bfloat16 values. Adding 1.0 puts the exact sum just above
    // halfway, so it rounds up. An f64 sum drops the 1.0 and would land on the tie, going to even.
    let on_a_tie = (2f64.powi(60) * (1.0 + 2f64.powi(-8))) as f32;
    assert_eq!(patched_bits(0x3f80, on_a_tie), Some(0x5d81));
    assert_eq!(exact_oracle(0x3f80, on_a_tie), 0x5d81);
    // The stored value as the small term: 2^-53 under a delta of 1 + 2^-8.
    let one_on_a_tie = 1.0 + 2f32.powi(-8);
    assert_eq!(patched_bits(0x2500, one_on_a_tie), Some(0x3f81));
    assert_eq!(exact_oracle(0x2500, one_on_a_tie), 0x3f81);
    // A negative small term pulls the exact sum below halfway: down, to the even neighbour's side.
    assert_eq!(patched_bits(0xbf80, on_a_tie), Some(0x5d80));
    // An odd lower neighbour: ties-to-even alone would go up; the exact sum is below halfway and goes down.
    let odd_tie = (2f64.powi(60) * (1.0 + 3.0 * 2f64.powi(-8))) as f32;
    assert_eq!(patched_bits(0xbf80, odd_tie), Some(0x5d81));
    assert_eq!(patched_bits(0x3f80, odd_tie), Some(0x5d82));
    assert_eq!(exact_oracle(0x3f80, odd_tie), 0x5d82);
}

#[test]
fn patched_bits_matches_exact_integer_arithmetic_across_exponent_gaps() {
    // Every stored significand pattern near a tie, against deltas on and beside ties, at gaps on both sides
    // of what an f64 holds exactly.
    let mut checked = 0usize;
    for old in [0x3f80u16, 0x3f81, 0x3fff, 0x2500, 0x2501, 0x4a7f] {
        for exponent in [-40, -20, -9, -8, 0, 8, 30, 45] {
            for numerator in [256u32, 257, 258, 259, 511] {
                for nudge in [0.0f64, 2f64.powi(-23)] {
                    let delta = (2f64.powi(exponent) * (numerator as f64 / 256.0 + nudge)) as f32;
                    assert_eq!(
                        patched_bits(old, delta),
                        Some(exact_oracle(old, delta)),
                        "old {old:#06x}, delta {delta:e}"
                    );
                    checked += 1;
                }
            }
        }
    }
    assert_eq!(checked, 6 * 8 * 5 * 2);
}
