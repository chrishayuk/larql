//! Gates for the single byte-formatting authority.
//!
//! Everything this feature prints about transfer and storage goes through
//! [`artifact::size`], so a wrong divisor here misreports every number at
//! once — and the numbers are the claim. `1e9` against `1 << 30` is a 7%
//! difference, which is exactly the size of gap that reads as "one of
//! these two figures is a bug".

use crate::format::vindex3::artifact::size;

/// The real granite-4.2-3b payload total, from its staged headers.
const GRANITE_PAYLOAD: u64 = 7_319_475_200;

/// What granite's shard index declares instead — short by one tied
/// 513,802,240-byte member.
const GRANITE_DECLARED: u64 = 6_805_672_960;

#[test]
fn gb_scale_states_both_units() {
    // 7_319_475_200 / 1e9 = 7.3194…; / 2^30 = 6.8167…
    assert_eq!(size(GRANITE_PAYLOAD), "7.32 GB (6.82 GiB)");
    assert_eq!(size(GRANITE_DECLARED), "6.81 GB (6.34 GiB)");
}

#[test]
fn the_tied_weight_gap_is_reported_in_mb() {
    // The difference the CLI prints beside the two figures above.
    assert_eq!(size(GRANITE_PAYLOAD - GRANITE_DECLARED), "513.80 MB");
}

#[test]
fn below_gb_scale_one_unit_stays_readable() {
    // A staging line quotes four figures; two units each is noise, not
    // clarity, and at MB scale nothing is being compared across units.
    assert_eq!(size(10_680_000), "10.68 MB");
    assert_eq!(size(0), "0.00 MB");
}

#[test]
fn the_boundary_is_a_decimal_gigabyte() {
    // Exactly 1e9 crosses into dual units; one byte below does not.
    assert_eq!(size(1_000_000_000), "1.00 GB (0.93 GiB)");
    assert_eq!(size(999_999_999), "1000.00 MB");
}
