use super::*;

/// Every regime round-trips its label, and the aliases parse.
#[test]
fn parse_accepts_labels_and_aliases() {
    for r in [
        Regime::Resident,
        Regime::CapacityConstrained,
        Regime::ColdEstate,
    ] {
        assert_eq!(Regime::parse(r.label()), Some(r), "{r} must round-trip");
    }
    assert_eq!(Regime::parse("capacity"), Some(Regime::CapacityConstrained));
    assert_eq!(
        Regime::parse("capacity_constrained"),
        Some(Regime::CapacityConstrained)
    );
    assert_eq!(Regime::parse("cold"), Some(Regime::ColdEstate));
    assert_eq!(Regime::parse("cold_estate"), Some(Regime::ColdEstate));
    assert_eq!(Regime::parse("  RESIDENT  "), Some(Regime::Resident));
}

/// A typo must NOT silently select a regime. This is the whole guard: an
/// unrecognised value has to suppress the verdict, not pick the
/// permissive reading.
#[test]
fn parse_refuses_unknown_rather_than_defaulting() {
    for bad in ["", "residentt", "dram", "hot", "cold estate", "1"] {
        assert_eq!(Regime::parse(bad), None, "{bad:?} must not parse");
    }
}

/// Each regime states a distinct licence — they are not decoration, they
/// are what the reader is permitted to conclude.
#[test]
fn each_regime_carries_a_distinct_claim_licence() {
    let licences = [
        Regime::Resident.byte_claim_licence(),
        Regime::CapacityConstrained.byte_claim_licence(),
        Regime::ColdEstate.byte_claim_licence(),
    ];
    for (i, a) in licences.iter().enumerate() {
        assert!(!a.is_empty());
        for b in licences.iter().skip(i + 1) {
            assert_ne!(a, b, "regimes must not share a licence");
        }
    }
    assert!(Regime::ColdEstate
        .byte_claim_licence()
        .contains("existential"));
    assert!(Regime::Resident.byte_claim_licence().contains("diagnostic"));
}

/// Display matches the label used in banners.
#[test]
fn display_matches_label() {
    assert_eq!(Regime::ColdEstate.to_string(), "cold-estate");
    assert_eq!(Regime::Resident.to_string(), "resident");
    assert_eq!(
        Regime::CapacityConstrained.to_string(),
        "capacity-constrained"
    );
}

/// The banked M3 Max figure is the attainable PROBE result, not spec
/// peak. Pinned because a silent drift to 400 would inflate every
/// utilisation figure the ledger reports by 9%.
#[test]
fn m3_max_dram_is_the_attainable_probe_not_spec_peak() {
    let bw = TierBandwidth::m3_max_dram();
    assert_eq!(bw.gb_per_s(), 367.0);
    assert_eq!(bw.gb_per_s(), M3_MAX_ATTAINABLE_DRAM_GBPS);
    assert!(bw.source().contains("probe"));
}

/// Transfer floor uses DECIMAL GB, matching the probe's convention. One
/// decimal GB at 1 GB/s is exactly 1000 ms; a binary-unit slip would read
/// 1073.7 and put a 7.4% error into every share this ledger prints.
#[test]
fn transfer_floor_uses_decimal_gigabytes() {
    let bw = TierBandwidth::measured(1.0, "unit test");
    assert!((bw.transfer_floor_ms(1_000_000_000) - 1000.0).abs() < 1e-9);

    let m3 = TierBandwidth::m3_max_dram();
    // 2.97 GB/token at 367 GB/s — the banked Q6_K figure.
    let ms = m3.transfer_floor_ms(2_970_000_000);
    assert!((ms - 8.093).abs() < 0.01, "got {ms}");
}

/// A non-positive ceiling yields zero rather than an infinity that would
/// propagate into every derived ratio.
#[test]
fn non_positive_bandwidth_yields_zero_floor() {
    assert_eq!(
        TierBandwidth::measured(0.0, "x").transfer_floor_ms(1 << 30),
        0.0
    );
    assert_eq!(
        TierBandwidth::measured(-5.0, "x").transfer_floor_ms(1 << 30),
        0.0
    );
}
