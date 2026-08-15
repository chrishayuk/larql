use crate::movement_ledger::{
    ByteMovement, DecisionCounts, LedgerConfig, Phase, Regime, Rooflines, TierBandwidth,
    TimeAttribution, TokenRecord,
};

fn cfg() -> LedgerConfig {
    LedgerConfig::new(Rooflines::dram_only(TierBandwidth::m3_max_dram()))
}

fn q6k_token() -> TokenRecord {
    TokenRecord {
        bytes: ByteMovement {
            semantic_requested: 2_784_000_000,
            physical_touched: 2_970_000_000,
            useful_physical: 2_970_000_000,
            dram: 2_970_000_000,
            ..Default::default()
        },
        time: TimeAttribution {
            wall_ms: 14.56,
            gpu_busy_ms: 11.56,
            gpu_bubble_ms: 0.87,
            host_wait_ms: 3.1,
            io_wait_ms: 0.0,
            host_wait_reported: true,
            io_wait_reported: true,
        },
        decisions: DecisionCounts::default(),
        phase: Some(Phase::Decode),
    }
}

/// Without a declared regime the raw arms still print — a reader must
/// always be able to see the counters — but the verdict REFUSES.
#[test]
fn verdict_refuses_without_a_declared_regime() {
    let out = q6k_token().render(&cfg());
    assert!(out.contains("[bw10/bytes]"), "raw byte arms must print");
    assert!(out.contains("[bw10/time]"), "raw time arms must print");
    assert!(out.contains("[bw10/cost]"), "the cost join must print");
    assert!(out.contains("[bw10/verdict] REFUSED"));
    assert!(out.contains("LARQL_MOVEMENT_REGIME"));
    assert!(
        !out.contains("[bw10/regime]"),
        "no regime banner without a regime"
    );
}

/// With a regime declared, the banner, the licence and the bracketed
/// share all appear together.
#[test]
fn declared_regime_unlocks_the_verdict_with_its_licence() {
    let out = q6k_token().render(&cfg().with_regime(Regime::Resident));
    assert!(out.contains("[bw10/regime] resident"));
    assert!(out.contains(Regime::Resident.byte_claim_licence()));
    assert!(out.contains("[bw10/verdict] movement share of wall is bracketed"));
    assert!(!out.contains("REFUSED"));
}

/// The cold-estate banner carries a different licence from the resident
/// one — the same numbers, a different permitted conclusion.
#[test]
fn cold_estate_prints_a_different_licence_from_resident() {
    let rec = q6k_token();
    let resident = rec.render(&cfg().with_regime(Regime::Resident));
    let cold = rec.render(&cfg().with_regime(Regime::ColdEstate));
    assert_ne!(resident, cold);
    assert!(cold.contains("existential"));
    assert!(resident.contains("diagnostic"));
}

/// Absent reporters must render as `n/a` with the reason, never as a
/// measured zero that reads like a clean cache.
#[test]
fn absent_reporters_render_as_not_available() {
    let out = q6k_token().render(&cfg());
    assert!(out.contains("no reuse reporter registered"));
    assert!(out.contains("no prefetcher registered"));
    assert!(
        !out.contains("reused=0.000"),
        "an unmeasured zero must not print as a measurement"
    );
}

/// A registered prefetcher renders its waste ratio instead.
#[test]
fn observed_prefetcher_renders_its_waste() {
    let mut rec = q6k_token();
    rec.bytes.prefetch_observed = true;
    rec.bytes.prefetched = 400_000_000;
    rec.bytes.prefetched_unused = 100_000_000;
    rec.bytes.reuse_observed = true;
    rec.bytes.reused = 50_000_000;
    let out = rec.render(&cfg());
    assert!(out.contains("waste 0.250"));
    assert!(!out.contains("no prefetcher registered"));
    assert!(!out.contains("no reuse reporter registered"));
}

/// Bytes on a tier with no declared ceiling force a printed warning that
/// the share is a lower bound only.
#[test]
fn unroofed_tier_bytes_force_a_lower_bound_warning() {
    let mut rec = q6k_token();
    rec.bytes.nvme = 500_000_000;
    rec.bytes.physical_touched += 500_000_000;
    let out = rec.render(&cfg().with_regime(Regime::ColdEstate));
    assert!(out.contains("no declared bandwidth"));
    assert!(out.contains("LOWER bound only"));
}

/// The overlapping terms are labelled as such. `host_wait` contains GPU
/// execution, so a reader who adds it to `gpu_busy` double-counts — the
/// line has to say so.
#[test]
fn overlapping_time_terms_are_labelled_non_additive() {
    let out = q6k_token().render(&cfg());
    assert!(out.contains("of which (overlapping, not additive)"));
    assert!(out.contains("host_wait="));
    assert!(out.contains("io_wait="));
}

/// The wall decomposition prints as an explicit identity so the residual
/// is visibly derived rather than measured.
#[test]
fn wall_decomposition_prints_as_an_identity() {
    let out = q6k_token().render(&cfg());
    assert!(out.contains("wall=14.560ms = gpu_busy 11.560ms + bubble 0.870ms"));
    assert!(out.contains("host_outside_gpu 2.130ms"));
}

/// Decimal units throughout, matching the GB/s convention the bandwidth
/// probes report in.
#[test]
fn bytes_render_in_decimal_units() {
    let out = q6k_token().render(&cfg());
    assert!(out.contains("2.970 GB"), "2.97e9 B is 2.970 GB decimal");
    assert!(!out.contains("GiB"));
}

/// The compact line carries both currencies — bytes and time — plus the
/// bracket, so a per-token stream is still adjudicable.
#[test]
fn compact_line_carries_both_currencies() {
    let out = q6k_token().render_compact(&cfg());
    assert!(out.starts_with("[bw10:decode]"));
    assert!(out.contains("physical=2.970 GB"));
    assert!(out.contains("wall=14.560ms"));
    assert!(out.contains("bubble=0.870ms"));
    assert!(out.contains("eta="));
    assert!(out.contains("share=["));
    assert_eq!(out.lines().count(), 1);
}

/// The tag names the phase, so a per-token stream can tell a prefill
/// position apart from a real decode step — the exact distinction whose
/// absence let 129 prefill positions render as decode tokens.
#[test]
fn compact_line_tag_names_the_phase() {
    let mut rec = q6k_token();
    rec.phase = Some(Phase::Prefill);
    assert!(rec.render_compact(&cfg()).starts_with("[bw10:prefill]"));
    rec.phase = None;
    assert!(rec.render_compact(&cfg()).starts_with("[bw10:?]"));
}

/// Every byte magnitude has a unit, down to raw bytes.
#[test]
fn byte_formatting_covers_every_magnitude() {
    let mut rec = q6k_token();
    for (n, want) in [
        (5_000_000_000u64, "5.000 GB"),
        (5_000_000, "5.000 MB"),
        (5_000, "5.000 kB"),
        (5, "5.000 B"),
    ] {
        rec.bytes.physical_touched = n;
        rec.bytes.dram = n;
        assert!(
            rec.render_compact(&cfg()).contains(want),
            "{n} should render as {want}"
        );
    }
    rec.bytes.physical_touched = 0;
    rec.bytes.dram = 0;
    assert!(rec.render_compact(&cfg()).contains("0 B"));
}

/// A zero-length window must not emit NaN into a verdict.
#[test]
fn empty_window_verdict_says_not_computable() {
    let rec = TokenRecord {
        bytes: ByteMovement::default(),
        time: TimeAttribution::default(),
        decisions: DecisionCounts::default(),
        phase: Some(Phase::Decode),
    };
    let out = rec.render(&cfg().with_regime(Regime::Resident));
    assert!(out.contains("not computable"));
    assert!(!out.contains("NaN"));
}

/// A window in which no expert group reached the seam says so. "0% of
/// nothing" and "0% of 24 opportunities" are different facts, and only
/// the second is evidence about a policy.
#[test]
fn an_unmeasured_policy_window_says_not_measured() {
    let out = q6k_token().render(&cfg().with_regime(Regime::Resident));
    assert!(out.contains("[bw10/policy]"), "the policy arm must print");
    assert!(out.contains("NOT MEASURED"), "{out}");
    assert!(
        !out.contains("[bw10/avoided]"),
        "nothing was avoided, so nothing may be claimed: {out}"
    );
}

/// With the seam reached but nothing skipped, the denominator prints and
/// the skip rate is a measured 0.
#[test]
fn a_canonical_window_prints_the_denominator() {
    let mut rec = q6k_token();
    rec.decisions = DecisionCounts {
        requested: 24,
        executed: 24,
        ..Default::default()
    };
    let out = rec.render(&cfg().with_regime(Regime::Resident));
    assert!(out.contains("requested=24 executed=24 skipped=0"), "{out}");
    assert!(out.contains("skip_rate=0.000"), "{out}");
    assert!(!out.contains("[bw10/avoided]"), "{out}");
}

/// The whole point of the arm: avoided bytes are reported, and the time
/// they would have cost is labelled a PROJECTION. A latency saving is a
/// difference between two runs and this window contains one of them.
#[test]
fn a_skipping_window_reports_avoided_bytes_as_a_projection() {
    let mut rec = q6k_token();
    rec.decisions = DecisionCounts {
        requested: 24,
        executed: 21,
        skipped: 3,
        semantic_avoided: 300_000_000,
        physical_avoided: 330_000_000,
    };
    let out = rec.render(&cfg().with_regime(Regime::Resident));
    assert!(out.contains("skipped=3"), "{out}");
    assert!(out.contains("[bw10/avoided]"), "{out}");
    assert!(out.contains("330.000 MB"), "{out}");
    assert!(
        out.contains("PROJECTED") && out.contains("NOT a measured saving"),
        "the projection must never read as a measurement: {out}"
    );
}

/// A count that violates `requested == executed + skipped` means some
/// dispatch site recorded a decision outside the one authority. That is
/// an instrumentation bug and the report must say so loudly rather than
/// print a plausible rate.
#[test]
fn an_inconsistent_decision_count_renders_a_bug_line() {
    let mut rec = q6k_token();
    rec.decisions = DecisionCounts {
        requested: 24,
        executed: 20,
        skipped: 2,
        ..Default::default()
    };
    let out = rec.render(&cfg().with_regime(Regime::Resident));
    assert!(out.contains("BUG requested != executed + skipped"), "{out}");
}

/// The per-token line carries the skip term only when a policy actually
/// deleted something — but when it does, it sits beside the physical
/// total, so a reader watching the byte count drop cannot miss why.
#[test]
fn compact_line_carries_skips_only_when_they_happened() {
    let mut rec = q6k_token();
    assert!(!rec.render_compact(&cfg()).contains("skipped="));

    rec.decisions = DecisionCounts {
        requested: 24,
        executed: 23,
        skipped: 1,
        semantic_avoided: 100_000_000,
        physical_avoided: 110_000_000,
    };
    let line = rec.render_compact(&cfg());
    assert!(line.contains("skipped=1/24 groups"), "{line}");
    assert!(line.contains("110.000 MB"), "{line}");
}
