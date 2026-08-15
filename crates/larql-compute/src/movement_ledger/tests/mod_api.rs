use super::*;
use crate::movement_ledger::bytes::{record, reset_for_test, OperandMovement, COUNTER_LOCK};

fn rec(physical: u64, wall_ms: f64) -> TokenRecord {
    rec_phase(physical, wall_ms, Some(Phase::Decode))
}

fn rec_phase(physical: u64, wall_ms: f64, phase: Option<Phase>) -> TokenRecord {
    TokenRecord {
        bytes: ByteMovement {
            semantic_requested: physical,
            physical_touched: physical,
            useful_physical: physical,
            dram: physical,
            ..Default::default()
        },
        time: TimeAttribution {
            wall_ms,
            gpu_busy_ms: wall_ms * 0.8,
            ..Default::default()
        },
        decisions: DecisionCounts::default(),
        phase,
    }
}

/// A scope isolates its own window from counters that moved before it.
#[test]
fn token_scope_measures_only_its_own_window() {
    let _g = COUNTER_LOCK.lock().unwrap_or_else(|e| e.into_inner());
    reset_for_test();

    record(OperandMovement::fully_consumed(100, 100, Tier::Dram));
    let scope = TokenScope::open();
    record(OperandMovement::fully_consumed(7, 9, Tier::Dram));
    let out = scope.close(TimeAttribution {
        wall_ms: 2.0,
        gpu_busy_ms: 1.0,
        ..Default::default()
    });

    assert_eq!(out.bytes.semantic_requested, 7, "pre-scope bytes excluded");
    assert_eq!(out.bytes.physical_touched, 9);
    assert_eq!(out.time.wall_ms, 2.0);
    reset_for_test();
}

/// Scopes nest: an outer window sees everything an inner one saw.
#[test]
fn scopes_compose_without_a_global_reset() {
    let _g = COUNTER_LOCK.lock().unwrap_or_else(|e| e.into_inner());
    reset_for_test();

    let outer = TokenScope::open();
    let inner = TokenScope::open();
    record(OperandMovement::fully_consumed(4, 4, Tier::Dram));
    let i = inner.close(TimeAttribution::default());
    record(OperandMovement::fully_consumed(6, 6, Tier::Dram));
    let o = outer.close(TimeAttribution::default());

    assert_eq!(i.bytes.physical_touched, 4);
    assert_eq!(o.bytes.physical_touched, 10);
    reset_for_test();
}

/// `Default` opens a scope, matching `TokenScope::open`.
#[test]
fn default_scope_opens_a_window() {
    let _g = COUNTER_LOCK.lock().unwrap_or_else(|e| e.into_inner());
    reset_for_test();
    let scope = TokenScope::default();
    record(OperandMovement::fully_consumed(1, 3, Tier::Dram));
    assert_eq!(
        scope
            .close(TimeAttribution::default())
            .bytes
            .physical_touched,
        3
    );
    reset_for_test();
}

/// Warmup tokens are DISCARDED, not averaged in. A run that folds cold
/// tokens into the mean reads materially slow and would mis-price every
/// byte the ledger accounts for.
#[test]
fn steady_state_discards_warmup_tokens() {
    let mut ss = SteadyState::new(2);
    ss.push(&rec(1_000, 100.0)); // warmup — a slow cold token
    ss.push(&rec(1_000, 100.0)); // warmup
    ss.push(&rec(1_000, 10.0));
    ss.push(&rec(1_000, 20.0));

    assert_eq!(ss.counted(), 2);
    assert_eq!(ss.discarded(), 2);
    let mean = ss.mean().expect("two tokens cleared warmup");
    assert!(
        (mean.time.wall_ms - 15.0).abs() < 1e-9,
        "cold tokens excluded"
    );
    assert_eq!(mean.bytes.physical_touched, 1_000);
}

/// Before any token clears warmup there is no mean to report — and the
/// discarded count still prints, so a run that measured nothing cannot
/// pass as a steady-state measurement.
#[test]
fn steady_state_reports_nothing_until_warmup_clears() {
    let mut ss = SteadyState::new(DEFAULT_WARMUP_TOKENS);
    assert!(ss.mean().is_none());
    for _ in 0..DEFAULT_WARMUP_TOKENS {
        ss.push(&rec(10, 1.0));
    }
    assert!(ss.mean().is_none(), "exactly warmup-many is still warmup");
    assert_eq!(ss.discarded(), DEFAULT_WARMUP_TOKENS);
    ss.push(&rec(10, 1.0));
    assert_eq!(ss.counted(), 1);
    assert!(ss.mean().is_some());
}

/// Observation flags are sticky across the window: one token that saw a
/// prefetcher marks the whole aggregate as measured.
#[test]
fn steady_state_keeps_observation_flags_sticky() {
    let mut ss = SteadyState::new(0);
    ss.push(&rec(10, 1.0));
    let mut with_reporters = rec(10, 1.0);
    with_reporters.bytes.reuse_observed = true;
    with_reporters.bytes.prefetch_observed = true;
    with_reporters.bytes.reused = 4;
    ss.push(&with_reporters);
    ss.push(&rec(10, 1.0));

    let mean = ss.mean().unwrap();
    assert!(mean.bytes.reuse_observed);
    assert!(mean.bytes.prefetch_observed);
}

/// Prefill tokens never enter the decode mean or its warmup count — the
/// BW-A live gate defect this type exists to prevent. They still
/// accumulate in full, in their own bucket, from the first one.
#[test]
fn prefill_tokens_are_excluded_from_the_decode_mean() {
    let mut ss = SteadyState::new(2);
    for _ in 0..129 {
        ss.push(&rec_phase(2_090_000_000, 11.0, Some(Phase::Prefill)));
    }
    ss.push(&rec(1_000, 10.0));
    ss.push(&rec(1_000, 10.0));
    ss.push(&rec(1_000, 20.0));

    assert_eq!(
        ss.counted(),
        1,
        "only the one decode token past warmup counts"
    );
    assert_eq!(ss.discarded(), 2, "warmup counts decode tokens only");
    assert_eq!(
        ss.mean().unwrap().bytes.physical_touched,
        1_000,
        "129 prefill positions must not appear in the decode mean"
    );

    assert_eq!(ss.prefill_counted(), 129, "no warmup discard on prefill");
    let pmean = ss.prefill_mean().unwrap();
    assert_eq!(pmean.bytes.physical_touched, 2_090_000_000);
    assert_eq!(pmean.phase, Some(Phase::Prefill));
    assert!((pmean.time.wall_ms - 11.0).abs() < 1e-9);
}

/// A token with no phase scope active is refused into neither bucket —
/// it is reported, not guessed, per the contract in [`crate::movement_ledger::phase`].
#[test]
fn unattributed_tokens_enter_neither_mean() {
    let mut ss = SteadyState::new(0);
    ss.push(&rec_phase(500, 5.0, None));
    ss.push(&rec(1_000, 10.0));

    assert_eq!(ss.unattributed(), 1);
    assert_eq!(
        ss.counted(),
        1,
        "the unattributed token did not count as decode"
    );
    assert_eq!(ss.prefill_counted(), 0);
    assert_eq!(ss.mean().unwrap().bytes.physical_touched, 1_000);
}

/// `prefill_mean` is `None` before any prefill token is recorded — a
/// batched-prefill path that never touches the per-position GPU entry
/// point has nothing to report, and the accessor must say so rather than
/// dividing by zero.
#[test]
fn prefill_mean_is_none_before_any_prefill_token() {
    let ss = SteadyState::new(0);
    assert!(ss.prefill_mean().is_none());
    assert_eq!(ss.prefill_counted(), 0);
}

/// The regime is undeclared until someone declares it. That is the whole
/// guard, so it is pinned at the config level too.
#[test]
fn config_starts_with_no_regime_declared() {
    let cfg = LedgerConfig::new(Rooflines::dram_only(TierBandwidth::m3_max_dram()));
    assert_eq!(cfg.regime, None);
    assert_eq!(
        cfg.with_regime(Regime::ColdEstate).regime,
        Some(Regime::ColdEstate)
    );
}

/// `TokenRecord::cost` routes through the declared rooflines.
#[test]
fn record_cost_uses_the_declared_rooflines() {
    let cfg = LedgerConfig::new(Rooflines::dram_only(TierBandwidth::measured(
        100.0,
        "unit test",
    )));
    let r = rec(1_000_000_000, 20.0);
    let c = r.cost(&cfg);
    // 1 GB at 100 GB/s = 10 ms floor against a 20 ms wall.
    assert!((c.floor_ms - 10.0).abs() < 1e-9);
    assert!((c.share_of_wall.unwrap() - 0.5).abs() < 1e-9);
}
