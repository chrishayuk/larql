//! Replay a recorded oracle trace on the production dispatch path.
//!
//! BW-C5 established the ceiling offline: at ONE late layer, with a
//! 6-token lookahead deciding each skip from the CURRENT (already
//! modified) state, 227 of 256 opportunities were skippable and 6 of 8
//! prompts kept full 32-token exact-token parity. That result lives in a
//! CPU KV-fork harness, and its per-`(step, layer)` decisions are exactly
//! a set of `(layer, step)` pairs.
//!
//! This policy is the bridge: hand it that recorded set and the SAME
//! decisions execute on the Metal serve path, so the offline ceiling and
//! the production path can be compared in bytes and wall time rather than
//! only in token parity. It is a replay, not a predictor — the trace was
//! computed with lookahead into the future of the very run it is being
//! replayed against, which is why it bounds what a predictor could
//! achieve and licenses nothing about what one would achieve.
//!
//! # Determinism is a precondition, not an assumption
//!
//! A replayed `(layer, step)` address only means the same thing if the
//! run being replayed reaches the same layers in the same order at the
//! same step indices. That holds for a greedy decode from a fixed prompt
//! on a deterministic engine, and does NOT hold under sampling — where a
//! replayed trace addresses a trajectory that no longer exists. The
//! caller owns that precondition; this policy cannot check it.

use std::collections::HashSet;

use crate::exec_policy::{ExecutionPolicy, ExecutionStrategy, ExpertGroupSite};
use crate::movement_ledger::Phase;

/// Skip exactly the `(layer, step)` pairs a recorded trace names.
pub struct TraceReplay {
    name: String,
    phase: Phase,
    skips: HashSet<(usize, u64)>,
}

impl TraceReplay {
    /// Build from recorded `(layer, step)` pairs, to be replayed during
    /// `phase`.
    ///
    /// The phase is REQUIRED rather than optional: a trace recorded
    /// against decode steps replayed without a phase restriction would
    /// also fire at the same indices during prefill, where step 7 is a
    /// prompt position and not the token the trace is about.
    pub fn new(phase: Phase, pairs: impl IntoIterator<Item = (usize, u64)>) -> Self {
        let skips: HashSet<(usize, u64)> = pairs.into_iter().collect();
        let name = format!(
            "trace-replay{{phase={},pairs={}}}",
            phase.label(),
            skips.len()
        );
        Self { name, phase, skips }
    }

    /// How many `(layer, step)` skips the trace holds. A replay that
    /// reports fewer skips than this reached fewer sites than the trace
    /// recorded — worth checking, since it means the two runs diverged.
    pub fn len(&self) -> usize {
        self.skips.len()
    }

    /// Whether the trace names no skips at all.
    pub fn is_empty(&self) -> bool {
        self.skips.is_empty()
    }
}

impl ExecutionPolicy for TraceReplay {
    fn name(&self) -> &str {
        &self.name
    }

    fn expert_group(&self, site: &ExpertGroupSite) -> ExecutionStrategy {
        if site.phase != Some(self.phase) {
            return ExecutionStrategy::Canonical;
        }
        // An undeclared step cannot be addressed by a trace — refuse,
        // never fall back to "the next one".
        let Some(step) = site.step else {
            return ExecutionStrategy::Canonical;
        };
        if self.skips.contains(&(site.layer, step)) {
            ExecutionStrategy::Skip
        } else {
            ExecutionStrategy::Canonical
        }
    }
}

#[cfg(test)]
#[path = "../tests/trace_replay.rs"]
mod tests;
