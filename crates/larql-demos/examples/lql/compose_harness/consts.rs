//! Thresholds and sizes, each traceable to `docs/preregistration/compose-correction-1-prereg.md`.

/// Candidates requested from every forward pass (the harness scores on ids, from this list).
pub const TOP_K: u32 = 20;

/// §6: the canonical prompt's top-1 must be the target's first token with at least this probability.
pub const TARGET_P_MIN: f64 = 0.30;
/// §6: share of scored same-template neighbours that must keep their top-1.
pub const NEIGHBOUR_KEEP: f64 = 0.90;
/// §6: largest allowed move of a kept neighbour's top-1 probability.
pub const NEIGHBOUR_DELTA_P_MAX: f64 = 0.05;
/// §5: a neighbour is scored only if its baseline top-1 probability is at least this.
pub const NEIGHBOUR_BASE_P_MIN: f64 = 0.20;

/// Neighbours and unrelated prompts measured at every balance step (a harness cost knob; the
/// final measurement uses the full per-fact sets).
pub const TRAJECTORY_NEIGHBOURS: usize = 3;
pub const TRAJECTORY_UNRELATED: usize = 3;

/// A control passes when its probability is within this many percentage points of the recorded
/// value (the recorded values are printed to two decimals).
pub const CONTROL_TOLERANCE_PCT: f64 = 0.02;

/// Files the registered set is guarded by, next to the bank.
pub const FROZEN_MARKER: &str = "FROZEN";
pub const REGISTERED_LEDGER: &str = "registered-attempts.jsonl";

/// §6 (reported, not gated): the balance controller's upper bound (`tuning::PROB_CEILING`).
pub const REPORT_CEILING: f64 = 0.95;
