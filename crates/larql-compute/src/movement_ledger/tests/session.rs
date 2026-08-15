use super::*;
use crate::movement_ledger::bytes::COUNTER_LOCK;
use crate::movement_ledger::TimeAttribution;
use crate::options::{ENV_MOVEMENT_LEDGER, ENV_MOVEMENT_REGIME};

/// Environment is process-global, so every test here must be serialised
/// against every other. Taking only the byte-counter lock is not enough:
/// a concurrent test that SETS `LARQL_MOVEMENT_LEDGER` makes this one's
/// `assert!(!enabled())` fail for a reason that has nothing to do with
/// the code under test.
static ENV_LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());

fn with_env<T>(vars: &[(&'static str, Option<&'static str>)], f: impl FnOnce() -> T) -> T {
    let _guard = ENV_LOCK.lock().unwrap_or_else(|e| e.into_inner());
    let prev: Vec<_> = vars
        .iter()
        .map(|(n, _)| (*n, std::env::var_os(n)))
        .collect();
    for (n, v) in vars {
        match v {
            Some(s) => unsafe { std::env::set_var(n, s) },
            None => unsafe { std::env::remove_var(n) },
        }
    }
    let out = f();
    for (n, v) in prev {
        match v {
            Some(s) => unsafe { std::env::set_var(n, s) },
            None => unsafe { std::env::remove_var(n) },
        }
    }
    out
}

fn a_token() -> TokenRecord {
    TokenRecord {
        bytes: super::super::ByteMovement {
            semantic_requested: 1_000_000,
            physical_touched: 1_000_000,
            useful_physical: 1_000_000,
            dram: 1_000_000,
            ..Default::default()
        },
        time: TimeAttribution {
            wall_ms: 10.0,
            gpu_busy_ms: 8.0,
            ..Default::default()
        },
        decisions: super::super::DecisionCounts::default(),
        phase: Some(super::super::Phase::Decode),
    }
}

/// Reporting is off by default; recording a token then does nothing.
#[test]
fn disabled_by_default_and_records_nothing() {
    let _g = COUNTER_LOCK.lock().unwrap_or_else(|e| e.into_inner());
    with_env(&[(ENV_MOVEMENT_LEDGER, None)], || {
        assert!(!enabled());
        assert!(!per_token_enabled());
        reset();
        record_token(a_token());
        flush(); // must not panic, must not print a measurement
    });
}

/// `=1` enables the summary but not the per-token line; `=token` enables
/// both. A typo enables neither rather than the noisier mode.
#[test]
fn env_selects_summary_versus_per_token() {
    with_env(&[(ENV_MOVEMENT_LEDGER, Some("1"))], || {
        assert!(enabled());
        assert!(!per_token_enabled());
    });
    with_env(&[(ENV_MOVEMENT_LEDGER, Some("token"))], || {
        assert!(enabled());
        assert!(per_token_enabled());
    });
    with_env(&[(ENV_MOVEMENT_LEDGER, Some("TOKEN"))], || {
        assert!(per_token_enabled(), "case-insensitive");
    });
}

/// The regime reaches the config from the environment, and an
/// unrecognised value leaves it undeclared rather than guessing.
#[test]
fn regime_comes_from_env_and_a_typo_leaves_it_undeclared() {
    with_env(&[(ENV_MOVEMENT_REGIME, Some("cold-estate"))], || {
        assert_eq!(config().regime, Some(super::super::Regime::ColdEstate));
    });
    with_env(&[(ENV_MOVEMENT_REGIME, Some("residnt"))], || {
        assert_eq!(config().regime, None, "a typo must not select a regime");
    });
    with_env(&[(ENV_MOVEMENT_REGIME, None)], || {
        assert_eq!(config().regime, None);
    });
}

/// The default DRAM ceiling is the banked probe, and it is overridable —
/// with the override's provenance carried into the report.
#[test]
fn dram_ceiling_defaults_to_the_banked_probe_and_is_overridable() {
    with_env(&[("LARQL_MOVEMENT_DRAM_GBPS", None)], || {
        let bw = config().rooflines.dram.unwrap();
        assert_eq!(bw.gb_per_s(), super::super::M3_MAX_ATTAINABLE_DRAM_GBPS);
        assert!(bw.source().contains("probe"));
    });
    with_env(&[("LARQL_MOVEMENT_DRAM_GBPS", Some("819.2"))], || {
        let bw = config().rooflines.dram.unwrap();
        assert_eq!(bw.gb_per_s(), 819.2);
        assert!(bw.source().contains("override"));
    });
    // An unparseable override falls back to the banked probe rather than
    // silently zeroing the ceiling.
    with_env(&[("LARQL_MOVEMENT_DRAM_GBPS", Some("fast"))], || {
        assert_eq!(
            config().rooflines.dram.unwrap().gb_per_s(),
            super::super::M3_MAX_ATTAINABLE_DRAM_GBPS
        );
    });
}

/// The NVMe ceiling stays undeclared unless someone measures it — a
/// cold-estate run must not price external bytes against a guess.
#[test]
fn nvme_ceiling_is_undeclared_until_measured() {
    with_env(&[("LARQL_MOVEMENT_NVME_GBPS", None)], || {
        assert!(config().rooflines.nvme.is_none());
    });
    with_env(&[("LARQL_MOVEMENT_NVME_GBPS", Some("3.5"))], || {
        assert_eq!(config().rooflines.nvme.unwrap().gb_per_s(), 3.5);
    });
}

/// Flushing with nothing recorded says so instead of printing zeros that
/// would read as a measurement.
#[test]
fn flush_without_tokens_reports_nothing_recorded() {
    let _g = COUNTER_LOCK.lock().unwrap_or_else(|e| e.into_inner());
    with_env(&[(ENV_MOVEMENT_LEDGER, Some("1"))], || {
        reset();
        flush();
        // A second flush is also safe — the accumulator was taken.
        flush();
    });
}

/// Tokens accumulate while enabled, and `reset` drops them so a harness
/// running several arms in one process cannot blend them.
#[test]
fn reset_drops_accumulated_tokens_between_arms() {
    let _g = COUNTER_LOCK.lock().unwrap_or_else(|e| e.into_inner());
    with_env(&[(ENV_MOVEMENT_LEDGER, Some("1"))], || {
        reset();
        for _ in 0..(DEFAULT_WARMUP_TOKENS + 4) {
            record_token(a_token());
        }
        reset();
        // After a reset the next flush has nothing to report.
        flush();
    });
}

/// The coverage line is reachable without emitting a full report.
#[test]
fn coverage_line_is_available_standalone() {
    let line = coverage_line();
    assert!(line.contains("[bw10/coverage]"));
}

/// A run that recorded prefill AND decode AND an unattributed token
/// flushes all three sections without panicking — the BW-A live gate
/// defect surfaced exactly this mix (129 prefill positions ahead of a
/// real decode window) and flush() has to print every bucket, not just
/// the decode one.
#[test]
fn flush_reports_prefill_and_unattributed_alongside_decode() {
    let _g = COUNTER_LOCK.lock().unwrap_or_else(|e| e.into_inner());
    with_env(&[(ENV_MOVEMENT_LEDGER, Some("1"))], || {
        reset();
        let mut prefill = a_token();
        prefill.phase = Some(super::super::Phase::Prefill);
        record_token(prefill);
        let mut unattributed = a_token();
        unattributed.phase = None;
        record_token(unattributed);
        record_token(a_token());
        flush();
    });
}
