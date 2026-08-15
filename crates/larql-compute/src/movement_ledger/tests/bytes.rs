use super::*;

/// A fully-consumed read sets useful == physical, and physical may still
/// exceed semantic: that gap is layout amplification, not access waste.
#[test]
fn fully_consumed_separates_amplification_from_waste() {
    // Q6_K's real shape: 3072 padded columns carrying 2880 semantic ones.
    let m = OperandMovement::fully_consumed(2880, 3072, Tier::Dram);
    assert_eq!(m.useful, m.physical);
    assert!(m.physical > m.semantic);
}

/// `useful` is clamped to `physical` — a mis-instrumented site must not
/// be able to manufacture an efficiency above 1.0.
#[test]
fn partially_consumed_clamps_useful_to_physical() {
    let m = OperandMovement::partially_consumed(100, 200, 900, Tier::Nvme);
    assert_eq!(m.useful, 200);
    let ok = OperandMovement::partially_consumed(100, 200, 50, Tier::Nvme);
    assert_eq!(ok.useful, 50);
}

/// Recording moves the shared totals AND the tier column by the physical
/// count, and a scope delta isolates one window.
#[test]
fn record_moves_totals_and_tier_column() {
    let _g = COUNTER_LOCK.lock().unwrap_or_else(|e| e.into_inner());
    let before = snapshot();
    record(OperandMovement::fully_consumed(10, 16, Tier::Dram));
    record(OperandMovement::partially_consumed(20, 40, 30, Tier::Nvme));
    record(OperandMovement::fully_consumed(1, 2, Tier::Network));
    let d = before.delta(&snapshot());

    assert_eq!(d.semantic_requested, 31);
    assert_eq!(d.physical_touched, 58);
    assert_eq!(d.useful_physical, 16 + 30 + 2);
    assert_eq!(d.dram, 16);
    assert_eq!(d.nvme, 40);
    assert_eq!(d.network, 2);
    assert_eq!(d.tier_unattributed(), 0, "tiers must partition physical");
}

/// Amplification and efficiency are the two ratios that keep a logical
/// saving honest. LA-6 measured scattered selection at 29-89x here.
#[test]
fn ratios_report_amplification_and_efficiency() {
    let b = ByteMovement {
        semantic_requested: 1_000,
        physical_touched: 50_000,
        useful_physical: 1_000,
        dram: 50_000,
        ..Default::default()
    };
    assert_eq!(b.amplification(), Some(50.0));
    assert_eq!(b.useful_ratio(), Some(0.02));
    assert_eq!(b.external(), 0);
}

/// Zero denominators yield None, never NaN or a fabricated 1.0.
#[test]
fn empty_movement_yields_no_ratios() {
    let b = ByteMovement::default();
    assert_eq!(b.amplification(), None);
    assert_eq!(b.useful_ratio(), None);
    assert_eq!(b.prefetch_waste_ratio(), None);
    assert_eq!(b.tier_unattributed(), 0);
}

/// External bytes are the cold-estate divisor: NVMe plus network, never
/// DRAM.
#[test]
fn external_counts_only_tiers_outside_local_memory() {
    let b = ByteMovement {
        physical_touched: 600,
        dram: 100,
        nvme: 200,
        network: 300,
        ..Default::default()
    };
    assert_eq!(b.external(), 500);
    assert_eq!(b.tier_unattributed(), 0);
}

/// Physical bytes recorded without a tier show up as unattributed rather
/// than being absorbed into DRAM — a gap in the instrument must be
/// visible, not flattering.
#[test]
fn untiered_physical_bytes_surface_as_unattributed() {
    let b = ByteMovement {
        physical_touched: 1_000,
        dram: 400,
        ..Default::default()
    };
    assert_eq!(b.tier_unattributed(), 600);
}

/// An absent reporter must be distinguishable from a measured zero.
#[test]
fn absent_reporters_are_distinguishable_from_measured_zero() {
    let _g = COUNTER_LOCK.lock().unwrap_or_else(|e| e.into_inner());
    reset_for_test();
    let quiet = snapshot();
    assert!(!quiet.reuse_observed, "nobody reported reuse yet");
    assert!(!quiet.prefetch_observed);
    assert_eq!(quiet.prefetch_waste_ratio(), None);

    record_reuse(0);
    record_prefetch(0);
    let observed = snapshot();
    assert!(
        observed.reuse_observed,
        "a zero-byte report still registers"
    );
    assert!(observed.prefetch_observed);
    reset_for_test();
}

/// Prefetch waste is a ratio of speculative traffic, reported only when a
/// prefetcher actually ran.
#[test]
fn prefetch_waste_ratio_needs_an_observed_prefetcher() {
    let unobserved = ByteMovement {
        prefetched: 100,
        prefetched_unused: 25,
        ..Default::default()
    };
    assert_eq!(unobserved.prefetch_waste_ratio(), None);

    let observed = ByteMovement {
        prefetch_observed: true,
        ..unobserved
    };
    assert_eq!(observed.prefetch_waste_ratio(), Some(0.25));
}

/// Deltas saturate: a counter reset inside an open window yields zero,
/// never an underflowed multi-exabyte reading.
#[test]
fn delta_saturates_instead_of_underflowing() {
    let later = ByteMovement::default();
    let earlier = ByteMovement {
        semantic_requested: 500,
        physical_touched: 900,
        useful_physical: 800,
        dram: 900,
        nvme: 10,
        network: 5,
        reused: 3,
        prefetched: 2,
        prefetched_unused: 1,
        ..Default::default()
    };
    let d = earlier.delta(&later);
    assert_eq!(d.physical_touched, 0);
    assert_eq!(d.semantic_requested, 0);
    assert_eq!(d.useful_physical, 0);
    assert_eq!(d.dram, 0);
    assert_eq!(d.nvme, 0);
    assert_eq!(d.network, 0);
    assert_eq!(d.reused, 0);
    assert_eq!(d.prefetched, 0);
    assert_eq!(d.prefetched_unused, 0);
}

/// Reuse and prefetch counters accumulate and survive a delta.
#[test]
fn reuse_and_prefetch_counters_accumulate() {
    let _g = COUNTER_LOCK.lock().unwrap_or_else(|e| e.into_inner());
    let before = snapshot();
    record_reuse(64);
    record_prefetch(128);
    record_prefetch_unused(32);
    let d = before.delta(&snapshot());
    assert_eq!(d.reused, 64);
    assert_eq!(d.prefetched, 128);
    assert_eq!(d.prefetched_unused, 32);
    assert!(d.reuse_observed && d.prefetch_observed);
    reset_for_test();
}

/// `accumulate` folds every field, including the observation flags —
/// two independent totals built with it (decode, prefill) cannot drift
/// out of sync with each other's arithmetic.
#[test]
fn accumulate_folds_every_field_including_observation_flags() {
    let mut total = ByteMovement {
        semantic_requested: 1,
        physical_touched: 2,
        useful_physical: 2,
        dram: 2,
        ..Default::default()
    };
    let next = ByteMovement {
        semantic_requested: 10,
        physical_touched: 20,
        useful_physical: 15,
        nvme: 20,
        reused: 5,
        reuse_observed: true,
        ..Default::default()
    };
    total.accumulate(&next);

    assert_eq!(total.semantic_requested, 11);
    assert_eq!(total.physical_touched, 22);
    assert_eq!(total.useful_physical, 17);
    assert_eq!(total.dram, 2, "untouched field stays as-is");
    assert_eq!(total.nvme, 20);
    assert_eq!(total.reused, 5);
    assert!(total.reuse_observed, "observation flags OR together");
    assert!(
        !total.prefetch_observed,
        "no prefetcher reported on either side"
    );
}
