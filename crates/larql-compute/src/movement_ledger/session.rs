//! Process-lifetime plumbing: accumulate per-token records, print the
//! steady-state summary once at the end of a run.
//!
//! The counters underneath always run; only the REPORTING is env-gated.
//! That distinction matters — a counter that only exists under a
//! diagnostic flag measures a different program from the one that ships,
//! and this ledger exists to adjudicate claims about the shipping one.

use std::sync::Mutex;

use crate::options;

use super::{
    coverage, LedgerConfig, Regime, Rooflines, SteadyState, TierBandwidth, TokenRecord,
    DEFAULT_WARMUP_TOKENS,
};

/// Env value selecting per-token output in addition to the summary.
const PER_TOKEN_TOKEN: &str = "token";

/// Override for the DRAM ceiling on non-M3-Max hardware, in decimal GB/s.
const ENV_DRAM_GBPS: &str = "LARQL_MOVEMENT_DRAM_GBPS";
/// Override for the external-storage ceiling, in decimal GB/s. Required
/// before a cold-estate run can price its NVMe bytes.
const ENV_NVME_GBPS: &str = "LARQL_MOVEMENT_NVME_GBPS";

static SESSION: Mutex<Option<SteadyState>> = Mutex::new(None);

/// Whether the ledger should print at all.
pub fn enabled() -> bool {
    options::env_flag(options::ENV_MOVEMENT_LEDGER)
}

/// Whether to print the compact line for every token, not just the
/// steady-state summary.
pub fn per_token_enabled() -> bool {
    options::env_nonempty_value(options::ENV_MOVEMENT_LEDGER)
        .is_some_and(|v| v.eq_ignore_ascii_case(PER_TOKEN_TOKEN))
}

fn gbps_from_env(name: &'static str) -> Option<f64> {
    options::env_nonempty_value(name).and_then(|v| v.parse::<f64>().ok())
}

/// The declared measurement context for this process.
///
/// The DRAM default is this project's banked M3 Max attainable probe. On
/// any other host it is wrong, and the report prints the source so that a
/// utilisation figure can never be read without its ceiling's provenance.
pub fn config() -> LedgerConfig {
    let dram = match gbps_from_env(ENV_DRAM_GBPS) {
        Some(g) => TierBandwidth::measured(g, "LARQL_MOVEMENT_DRAM_GBPS override"),
        None => TierBandwidth::m3_max_dram(),
    };
    let rooflines = Rooflines {
        dram: Some(dram),
        nvme: gbps_from_env(ENV_NVME_GBPS)
            .map(|g| TierBandwidth::measured(g, "LARQL_MOVEMENT_NVME_GBPS")),
        network: None,
    };
    let mut cfg = LedgerConfig::new(rooflines);
    cfg.regime =
        options::env_nonempty_value(options::ENV_MOVEMENT_REGIME).and_then(|v| Regime::parse(&v));
    cfg
}

/// Record one completed token. Cheap and inert when reporting is off.
///
/// Takes the whole [`TokenRecord`] (not separate bytes/time) so its
/// `phase` travels with it — [`SteadyState::push`] is what routes a
/// prefill position away from the decode mean, and it can only do that
/// if the phase reaches this call.
pub fn record_token(rec: TokenRecord) {
    if !enabled() {
        return;
    }
    if per_token_enabled() {
        eprintln!("{}", rec.render_compact(&config()));
    }
    let mut guard = SESSION.lock().unwrap_or_else(|e| e.into_inner());
    guard
        .get_or_insert_with(|| SteadyState::new(DEFAULT_WARMUP_TOKENS))
        .push(&rec);
}

/// Print the steady-state summary and reset the accumulator. Safe to call
/// when nothing was recorded — it says so rather than printing an empty
/// table that reads like a measurement.
///
/// Order: prefill, then unattributed, then the decode window — a reader
/// sees what was EXCLUDED before they see what was KEPT, so the decode
/// mean's provenance is legible without cross-referencing anything else.
pub fn flush() {
    if !enabled() {
        return;
    }
    let taken = {
        let mut guard = SESSION.lock().unwrap_or_else(|e| e.into_inner());
        guard.take()
    };
    let cfg = config();
    let Some(ss) = taken else {
        eprintln!("[bw10] no tokens recorded — nothing to report");
        return;
    };

    if let Some(pmean) = ss.prefill_mean() {
        eprintln!(
            "[bw10/prefill] {} position(s) recorded during prefill — EXCLUDED from the decode \
             steady-state mean (prefill and decode are different operations; see \
             larql_compute::movement_ledger::phase)",
            ss.prefill_counted(),
        );
        eprintln!("{}", pmean.render_compact(&cfg));
    }
    let unattributed = ss.unattributed();
    if unattributed > 0 {
        eprintln!(
            "[bw10/phase]  {unattributed} token(s) recorded with NO phase scope active — \
             neither prefill nor decode; excluded from both means. A driver loop reached the \
             ledger boundary outside any PhaseScope — the gap is that driver loop, not these \
             tokens."
        );
    }

    let counted = ss.counted();
    let discarded = ss.discarded();
    let Some(mean) = ss.mean() else {
        eprintln!(
            "[bw10] {discarded} decode token(s) recorded, all inside warmup \
             ({DEFAULT_WARMUP_TOKENS}) — no steady-state mean. This is NOT a measurement."
        );
        return;
    };
    eprintln!(
        "[bw10/window]  steady-state mean over {counted} decode token(s); {discarded} warmup \
         token(s) discarded"
    );
    eprint!("{}", mean.render(&cfg));
}

/// Drop any accumulated state without printing — for a harness that runs
/// several arms in one process and must not blend them.
pub fn reset() {
    let mut guard = SESSION.lock().unwrap_or_else(|e| e.into_inner());
    *guard = None;
}

/// One-line coverage statement, for callers that want to log what the
/// ledger accounts for without emitting a full report.
pub fn coverage_line() -> String {
    coverage::render()
}

#[cfg(test)]
#[path = "tests/session.rs"]
mod tests;
