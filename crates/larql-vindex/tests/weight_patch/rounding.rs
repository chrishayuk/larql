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
