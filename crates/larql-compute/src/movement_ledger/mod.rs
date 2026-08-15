//! BW10 — the per-token movement and causality ledger.
//!
//! Every other bandwidth experiment reports against this instrument, so
//! that results are comparable in one currency. It answers two questions
//! together, because either alone misleads:
//!
//! > **how many bytes moved** — and — **how much of the token's time is
//! > attributable to moving them**
//!
//! The second question is the point. This engine has already produced a
//! 1.39× win with zero byte reduction (S2, queue starvation) and a 1.09×
//! win from a 25% byte reduction (MXFP4) — the same ledger has to explain
//! both, and a byte-only ledger explains neither.
//!
//! # Using it
//!
//! ```no_run
//! use larql_compute::movement_ledger::{
//!     LedgerConfig, Regime, Rooflines, TierBandwidth, TokenScope,
//! };
//!
//! let cfg = LedgerConfig::new(Rooflines::dram_only(TierBandwidth::m3_max_dram()))
//!     .with_regime(Regime::Resident);
//! let scope = TokenScope::open();
//! // ... decode one token; bind sites bump the byte counters ...
//! let record = scope.close(Default::default());
//! println!("{}", record.render(&cfg));
//! ```
//!
//! # What it deliberately does not do
//!
//! It does not decide whether a change is good. It reports the raw arms,
//! then a verdict line that is only emitted once a [`Regime`] is
//! declared — because the same byte delta licenses opposite conclusions
//! in the resident and cold-estate regimes, and a verdict that does not
//! name its regime is the error this whole module exists to prevent.

pub mod bytes;
pub mod coverage;
pub mod decisions;
pub mod phase;
pub mod regime;
pub mod report;
pub mod session;
pub mod timing;

pub use bytes::{ByteMovement, OperandMovement, Tier};
pub use coverage::{Surface, SurfaceState};
pub use decisions::DecisionCounts;
pub use phase::{Phase, PhaseScope};
pub use regime::{Regime, TierBandwidth, M3_MAX_ATTAINABLE_DRAM_GBPS};
pub use timing::{MovementCost, Rooflines, TimeAttribution};

/// Declared measurement context: the rooflines the bytes are priced
/// against and, optionally, the regime the claim targets.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct LedgerConfig {
    pub rooflines: Rooflines,
    /// `None` until the caller declares one. The raw counters print
    /// regardless; the verdict line does not.
    pub regime: Option<Regime>,
}

impl LedgerConfig {
    pub fn new(rooflines: Rooflines) -> Self {
        Self {
            rooflines,
            regime: None,
        }
    }

    pub fn with_regime(mut self, regime: Regime) -> Self {
        self.regime = Some(regime);
        self
    }

    /// Read the regime from `LARQL_MOVEMENT_REGIME`. An unset variable
    /// leaves the regime undeclared; an unrecognised value ALSO leaves it
    /// undeclared rather than guessing, so a typo suppresses the verdict
    /// instead of silently selecting the permissive reading.
    pub fn with_regime_from_env(mut self) -> Self {
        self.regime = crate::options::env_nonempty_value(crate::options::ENV_MOVEMENT_REGIME)
            .and_then(|v| Regime::parse(&v));
        self
    }
}

/// One token's complete ledger entry: what moved, what an execution
/// policy chose not to move, and where the time went.
///
/// `phase` is `None` unless a [`PhaseScope`] was active for the whole
/// window — a hand-built test fixture or a caller that never installed
/// one is unattributed, not silently decode. See [`phase`] for why this
/// distinction exists.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct TokenRecord {
    pub bytes: ByteMovement,
    pub time: TimeAttribution,
    /// Execution-policy decisions in this record's window, as a TOTAL —
    /// never divided, even on a mean record. These are counts of
    /// discrete events, and dividing "3 groups skipped over 240 decode
    /// tokens" by 240 renders a real intervention as zero. Read them
    /// against `bytes` from the SAME record, which
    /// [`TokenScope`] snapshots at the same instants.
    pub decisions: DecisionCounts,
    pub phase: Option<Phase>,
}

impl TokenRecord {
    /// Join against the declared rooflines to get the causality terms.
    pub fn cost(&self, cfg: &LedgerConfig) -> MovementCost {
        MovementCost::derive(&self.bytes, &self.time, &cfg.rooflines)
    }
}

/// An open measurement window over one token.
///
/// Byte counters are process-wide and always running; the scope captures
/// the baseline at open and subtracts at close, so windows compose and no
/// global reset is ever needed. The phase is also captured at open —
/// whichever [`PhaseScope`] the caller installed before opening this
/// window, or `None` if none is active — and carried onto the resulting
/// [`TokenRecord`] unchanged.
#[derive(Clone, Copy, Debug)]
pub struct TokenScope {
    opened_at: ByteMovement,
    decisions_at: DecisionCounts,
    phase: Option<Phase>,
}

impl Default for TokenScope {
    fn default() -> Self {
        Self::open()
    }
}

impl TokenScope {
    pub fn open() -> Self {
        Self {
            opened_at: bytes::snapshot(),
            decisions_at: decisions::snapshot(),
            phase: phase::current_phase(),
        }
    }

    /// Close the window, pairing the byte delta with caller-measured time.
    ///
    /// The decision counters are snapshotted in the same two instants as
    /// the byte counters, so `bytes.physical_touched` and
    /// `decisions.physical_avoided` on the resulting record always
    /// describe the same window — the precondition
    /// [`DecisionCounts::avoided_share`] cannot check for itself.
    pub fn close(self, time: TimeAttribution) -> TokenRecord {
        TokenRecord {
            bytes: self.opened_at.delta(&bytes::snapshot()),
            decisions: self.decisions_at.delta(&decisions::snapshot()),
            time,
            phase: self.phase,
        }
    }
}

/// Steady-state accumulator over a decode run.
///
/// Discards the first `warmup` DECODE tokens. This is not tidiness: a run
/// that includes cold tokens reads materially slow (a 49-step run measured
/// 22% below steady state on this engine), and a ledger that averaged them
/// in would mis-price every byte it accounts for.
///
/// Prefill and unattributed tokens are kept in their own totals and never
/// enter the decode mean or its warmup count — see [`super::phase`] for
/// why blending them is exactly the defect this type exists to avoid. A
/// prefill token still contributes to [`Self::prefill_mean`], so nothing
/// measured is thrown away; it is just never averaged with decode.
///
/// **Invariant**: a record with `phase != Some(Phase::Decode)` can never
/// reach `self.bytes`, `self.time`, `self.counted`, or the warmup
/// comparison in [`Self::push`] — enforced by the `match` at the top of
/// that function, not by caller discipline, and pinned by
/// `prefill_tokens_are_excluded_from_the_decode_mean` and
/// `unattributed_tokens_enter_neither_mean` in `tests/mod_api.rs`.
#[derive(Clone, Debug)]
pub struct SteadyState {
    warmup: usize,
    seen: usize,
    bytes: ByteMovement,
    time: TimeAttribution,
    decisions: DecisionCounts,
    counted: usize,
    prefill_seen: usize,
    prefill_bytes: ByteMovement,
    prefill_time: TimeAttribution,
    prefill_decisions: DecisionCounts,
    unattributed_seen: usize,
}

/// Warmup tokens discarded by default, matching this project's banked
/// steady-state bench protocol (warmup 16, n 256, long prompt).
pub const DEFAULT_WARMUP_TOKENS: usize = 16;

impl SteadyState {
    pub fn new(warmup: usize) -> Self {
        Self {
            warmup,
            seen: 0,
            bytes: ByteMovement::default(),
            time: TimeAttribution::default(),
            decisions: DecisionCounts::default(),
            counted: 0,
            prefill_seen: 0,
            prefill_bytes: ByteMovement::default(),
            prefill_time: TimeAttribution::default(),
            prefill_decisions: DecisionCounts::default(),
            unattributed_seen: 0,
        }
    }

    /// Accumulate one token into the bucket its [`Phase`] names.
    ///
    /// `Decode` is the only bucket subject to warmup discard. `Prefill`
    /// tokens accumulate in full from the first one — a run's prefill is
    /// typically one window, not a steady-state series, so there is no
    /// "cold prefill" transient to discard. `None` (no [`PhaseScope`] was
    /// active) is counted and reported but enters neither mean, per the
    /// refusal contract in [`super::phase`].
    pub fn push(&mut self, rec: &TokenRecord) {
        match rec.phase {
            Some(Phase::Prefill) => {
                self.prefill_seen += 1;
                self.prefill_bytes.accumulate(&rec.bytes);
                self.prefill_time = self.prefill_time.add(&rec.time);
                self.prefill_decisions.accumulate(&rec.decisions);
                return;
            }
            None => {
                self.unattributed_seen += 1;
                return;
            }
            Some(Phase::Decode) => {}
        }
        self.seen += 1;
        if self.seen <= self.warmup {
            return;
        }
        self.bytes.accumulate(&rec.bytes);
        self.time = self.time.add(&rec.time);
        // Decisions accumulate only for tokens that survived warmup, for
        // the same reason their bytes do: the skip rate and the avoided
        // bytes must describe the SAME window as the physical bytes they
        // are read against, or `touched + avoided` stops reconstructing
        // anything.
        self.decisions.accumulate(&rec.decisions);
        self.counted += 1;
    }

    /// Decode tokens that survived warmup and entered the average.
    pub fn counted(&self) -> usize {
        self.counted
    }

    /// Decode tokens discarded as warmup — printed, never silent, so a
    /// run that counted almost nothing cannot pass as a steady-state
    /// measurement.
    pub fn discarded(&self) -> usize {
        self.seen.min(self.warmup)
    }

    /// Per-token means over the counted DECODE window. `None` before any
    /// decode token clears warmup.
    pub fn mean(&self) -> Option<TokenRecord> {
        Self::divide(
            &self.bytes,
            &self.time,
            &self.decisions,
            self.counted,
            Phase::Decode,
        )
    }

    /// Prefill positions recorded — every one, not subject to warmup.
    pub fn prefill_counted(&self) -> usize {
        self.prefill_seen
    }

    /// Per-position mean over every recorded prefill token. `None` if
    /// none were recorded (e.g. a batched-prefill path that never
    /// touches the per-position GPU entry point).
    pub fn prefill_mean(&self) -> Option<TokenRecord> {
        Self::divide(
            &self.prefill_bytes,
            &self.prefill_time,
            &self.prefill_decisions,
            self.prefill_seen,
            Phase::Prefill,
        )
    }

    /// Tokens recorded with no [`PhaseScope`] active — neither prefill
    /// nor decode. Non-zero means some driver loop reached the ledger
    /// boundary outside any scope; the gap is that driver loop, not the
    /// tokens it produced.
    pub fn unattributed(&self) -> usize {
        self.unattributed_seen
    }

    /// Shared division for the decode and prefill buckets, so the two
    /// cannot compute their means with different arithmetic.
    ///
    /// `decisions` passes through UNDIVIDED — see [`TokenRecord`]'s field
    /// doc. Everything else here is a rate that a per-token mean makes
    /// more legible; a count of discrete interventions is the one thing
    /// it makes less legible, because integer division rounds a real skip
    /// to zero.
    fn divide(
        bytes: &ByteMovement,
        time: &TimeAttribution,
        decisions: &DecisionCounts,
        n: usize,
        phase: Phase,
    ) -> Option<TokenRecord> {
        if n == 0 {
            return None;
        }
        let d = n as u64;
        Some(TokenRecord {
            decisions: *decisions,
            bytes: ByteMovement {
                semantic_requested: bytes.semantic_requested / d,
                physical_touched: bytes.physical_touched / d,
                useful_physical: bytes.useful_physical / d,
                dram: bytes.dram / d,
                nvme: bytes.nvme / d,
                network: bytes.network / d,
                reused: bytes.reused / d,
                prefetched: bytes.prefetched / d,
                prefetched_unused: bytes.prefetched_unused / d,
                reuse_observed: bytes.reuse_observed,
                prefetch_observed: bytes.prefetch_observed,
            },
            time: time.per(n),
            phase: Some(phase),
        })
    }
}

#[cfg(test)]
#[path = "tests/mod_api.rs"]
mod tests;
