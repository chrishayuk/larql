//! Tests for [`super`] — the stage attribution arithmetic.
//!
//! Everything here runs without a device, which is the point:
//! [`super::profile_from_samples`] documents itself as "pure, so the
//! arithmetic is testable without a device", and the fold it performs is
//! what turns a raw timestamp array into the per-stage numbers a report
//! is read from. A mistake in that fold does not crash — it produces a
//! plausible attribution that sends someone optimising the wrong kernel.
//!
//! The device-touching half ([`super::StageProfiler`], `gpu_span_ms`,
//! `resolve_timestamps`) is deliberately not reached from here: those
//! need a real command buffer with resolved counters, and asserting on
//! them without one would pin a mock rather than the GPU.

use std::collections::BTreeMap;

use super::{profile_from_samples, Stage, StageProfile};

/// `ALL` is what every report iterates, so a stage missing from it is a
/// stage that silently never appears. Pinned by count *and* by content:
/// a duplicate would keep the length right while dropping a variant.
#[test]
fn all_lists_every_stage_exactly_once() {
    let mut seen = Stage::ALL.to_vec();
    seen.sort();
    seen.dedup();
    assert_eq!(
        seen.len(),
        Stage::ALL.len(),
        "Stage::ALL contains a duplicate: {:?}",
        Stage::ALL
    );
    assert_eq!(
        Stage::ALL.len(),
        16,
        "a stage was added without updating ALL"
    );
}

/// The doc comment promises encode order, and `Ord` is derived from
/// declaration order — so the two agree only while `ALL` is sorted. A
/// report that read top-to-bottom like the layer would quietly stop
/// doing so if someone reordered the array alone.
#[test]
fn all_is_in_encode_order() {
    let mut sorted = Stage::ALL.to_vec();
    sorted.sort();
    assert_eq!(
        sorted,
        Stage::ALL.to_vec(),
        "ALL must be in the enum's own (encode) order"
    );
}

/// Labels are the report's only identifier for a stage, so two stages
/// sharing one would merge two kernels' costs under a single line.
#[test]
fn every_stage_has_a_distinct_label() {
    let mut labels: Vec<&str> = Stage::ALL.iter().map(|s| s.label()).collect();
    let before = labels.len();
    labels.sort_unstable();
    labels.dedup();
    assert_eq!(before, labels.len(), "two stages share a label");
    assert!(
        Stage::ALL.iter().all(|s| !s.label().is_empty()),
        "an empty label would render as a blank report row"
    );
}

fn profile_of(stages: &[(Stage, u64)], span_ns: u64) -> StageProfile {
    let mut stage_ns = BTreeMap::new();
    let mut stage_runs = BTreeMap::new();
    for &(s, ns) in stages {
        *stage_ns.entry(s).or_insert(0) += ns;
        *stage_runs.entry(s).or_insert(0) += 1;
    }
    StageProfile {
        stage_ns,
        stage_runs,
        span_ns,
        overflowed: 0,
    }
}

#[test]
fn attributed_ns_sums_every_stage() {
    let p = profile_of(&[(Stage::AttnCore, 400), (Stage::Experts, 600)], 1_200);
    assert_eq!(p.attributed_ns(), 1_000);
}

/// The gap is what the report shows as unattributed — sampling drain
/// plus whatever ran outside any stage.
#[test]
fn gap_ns_is_the_span_the_stages_do_not_account_for() {
    let p = profile_of(&[(Stage::AttnCore, 400), (Stage::Experts, 600)], 1_200);
    assert_eq!(p.gap_ns(), 200);
}

/// `gap_ns` saturates rather than underflowing. Attribution can exceed
/// the span when sampled runs overlap, and a wrapped `u64` would render
/// as roughly 18 quintillion nanoseconds of "unattributed" GPU time.
#[test]
fn gap_ns_saturates_when_attribution_exceeds_the_span() {
    let p = profile_of(&[(Stage::AttnCore, 900)], 400);
    assert_eq!(p.gap_ns(), 0, "gap must floor at zero, never wrap");
}

/// Accumulating tokens is how a multi-token report is built, so every
/// field has to carry — including `overflowed`, which is the signal that
/// the report is incomplete.
#[test]
fn add_accumulates_every_field() {
    let mut a = profile_of(&[(Stage::AttnCore, 100)], 500);
    a.overflowed = 1;
    let mut b = profile_of(&[(Stage::AttnCore, 200), (Stage::Head, 50)], 700);
    b.overflowed = 2;

    a.add(&b);

    assert_eq!(a.stage_ns[&Stage::AttnCore], 300, "spans must sum");
    assert_eq!(a.stage_ns[&Stage::Head], 50, "a new stage must be inserted");
    assert_eq!(a.stage_runs[&Stage::AttnCore], 2, "runs must sum");
    assert_eq!(a.span_ns, 1_200);
    assert_eq!(a.overflowed, 3, "dropped requests must not be lost");
}

/// The ordinary fold: each run spans `ts[idx]..ts[idx + 1]`, repeats of
/// one stage sum, and the span runs from the first start to the last end.
#[test]
fn profile_from_samples_folds_runs_into_per_stage_totals() {
    // attn.core 100..250, experts 250..600, attn.core again 600..700
    let ts = [100u64, 250, 250, 600, 600, 700];
    let runs = [
        (Stage::AttnCore, 0),
        (Stage::Experts, 2),
        (Stage::AttnCore, 4),
    ];
    let p = profile_from_samples(&runs, &ts, 0);

    assert_eq!(p.stage_ns[&Stage::AttnCore], 150 + 100);
    assert_eq!(p.stage_ns[&Stage::Experts], 350);
    assert_eq!(p.stage_runs[&Stage::AttnCore], 2);
    assert_eq!(p.stage_runs[&Stage::Experts], 1);
    assert_eq!(p.span_ns, 600, "first start (100) to last end (700)");
    assert_eq!(
        p.attributed_ns(),
        p.span_ns,
        "back-to-back runs leave no gap"
    );
}

/// A run whose end index is past the resolved array is skipped rather
/// than counted as zero. Counting it would add a phantom run at 0 ns and
/// pull the stage's mean down without changing the total — the kind of
/// error that reads as "this kernel is cheap" instead of as missing data.
#[test]
fn profile_from_samples_skips_runs_the_timestamps_do_not_cover() {
    let ts = [10u64, 40];
    // Second run starts at index 2: neither its start nor its end exists.
    let runs = [(Stage::AttnCore, 0), (Stage::Experts, 2)];
    let p = profile_from_samples(&runs, &ts, 0);

    assert_eq!(p.stage_ns[&Stage::AttnCore], 30);
    assert!(
        !p.stage_ns.contains_key(&Stage::Experts),
        "an unresolvable run must not appear at all"
    );
    assert!(!p.stage_runs.contains_key(&Stage::Experts));
}

/// A run whose *end* alone is missing takes the same path — the `get`
/// pair fails as a unit, so a truncated final sample cannot produce a
/// run measured against a zero end.
#[test]
fn profile_from_samples_skips_a_run_with_a_missing_end() {
    let ts = [10u64, 40, 90];
    let runs = [(Stage::AttnCore, 0), (Stage::Head, 2)];
    let p = profile_from_samples(&runs, &ts, 0);

    assert_eq!(p.stage_ns[&Stage::AttnCore], 30);
    assert!(
        !p.stage_ns.contains_key(&Stage::Head),
        "ts[3] does not exist, so the head run is not attributable"
    );
}

/// Timestamps are read off the device and are not guaranteed monotonic
/// across encoders; an inverted pair must contribute zero, not wrap.
#[test]
fn profile_from_samples_saturates_an_inverted_pair() {
    let ts = [500u64, 200];
    let p = profile_from_samples(&[(Stage::AttnCore, 0)], &ts, 0);
    assert_eq!(p.stage_ns[&Stage::AttnCore], 0, "must floor at zero");
}

/// No runs at all resolves to an empty profile rather than a panic:
/// `first` is never set, and the span must not underflow off `last`.
#[test]
fn profile_from_samples_handles_no_runs() {
    let p = profile_from_samples(&[], &[], 0);
    assert!(p.stage_ns.is_empty());
    assert!(p.stage_runs.is_empty());
    assert_eq!(p.span_ns, 0);
    assert_eq!(p.attributed_ns(), 0);
    assert_eq!(p.gap_ns(), 0);
}

/// The overflow count is passed through untouched — it is the report's
/// own admission that stages went unattributed, and silently zeroing it
/// would present a partial profile as a complete one.
#[test]
fn profile_from_samples_carries_the_overflow_count() {
    let ts = [0u64, 10];
    let p = profile_from_samples(&[(Stage::AttnNorm, 0)], &ts, 7);
    assert_eq!(p.overflowed, 7);
}

/// Gaps between runs are real GPU time outside any stage, so the span
/// must exceed attribution rather than being derived from it.
#[test]
fn profile_from_samples_span_includes_time_between_runs() {
    // 0..100, then a 400 ns gap, then 500..600.
    let ts = [0u64, 100, 500, 600];
    let runs = [(Stage::AttnNorm, 0), (Stage::Head, 2)];
    let p = profile_from_samples(&runs, &ts, 0);

    assert_eq!(p.attributed_ns(), 200);
    assert_eq!(p.span_ns, 600);
    assert_eq!(p.gap_ns(), 400, "the unsampled 400 ns must survive as gap");
}
