//! What the execution policy DECIDED, in the same currency as the bytes.
//!
//! [`super::bytes`] answers "how many bytes moved". Once a runtime policy
//! can delete an operation ([`crate::exec_policy`]), that is no longer
//! the whole story: a byte the engine never touched is invisible to a
//! counter that only fires at bind sites, so a run under a skip policy
//! would report a smaller physical total with nothing to say WHY. The
//! difference between "this arm streams less because MXFP4 stores less"
//! and "this arm streams less because 40% of the expert groups were
//! deleted" is the entire result, and a ledger that cannot separate them
//! adjudicates neither.
//!
//! So skipped work is counted HERE, and never folded into the byte
//! counters:
//!
//! - a byte that moved bumps [`super::bytes`], as before;
//! - a byte that was avoided bumps [`physical_avoided`] and nothing else.
//!
//! `physical_touched + physical_avoided` is what the canonical arm would
//! have moved on the covered surfaces. Neither term is derivable from the
//! other, and adding them into one field would put a byte that never
//! crossed the memory bus into a bandwidth measurement.
//!
//! # Requested = executed + skipped, always
//!
//! Every visit to the seam bumps `requested`, then exactly one of
//! `executed`/`skipped` — including when no policy is installed, where
//! the identity degenerates to `requested == executed`. That is
//! deliberate: without the denominator running unconditionally a skip
//! rate has no base, and a run with the instrument switched off would be
//! indistinguishable from a run where the seam was never reached. It also
//! keeps this module honest about the same thing
//! [`super::coverage`] is honest about — a zero here means "measured
//! zero" only because the counter was always running.

use std::sync::atomic::{AtomicU64, Ordering};

use super::bytes::OperandMovement;

/// Policy decisions over a window, and the work they removed.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct DecisionCounts {
    /// Semantic operations the route asked for — the denominator.
    pub requested: u64,
    /// Of those, physically executed.
    pub executed: u64,
    /// Of those, omitted by an installed policy.
    pub skipped: u64,
    /// Semantic bytes the skipped operations would have asked for.
    pub semantic_avoided: u64,
    /// Physical bytes the skipped operations would have streamed. This is
    /// the term that pairs with [`super::ByteMovement::physical_touched`]
    /// to reconstruct the canonical arm's traffic.
    pub physical_avoided: u64,
}

impl DecisionCounts {
    /// Whether the seam was reached at all in this window. A window with
    /// `requested == 0` has no skip rate to report — not a 0% one.
    pub fn is_measured(&self) -> bool {
        self.requested > 0
    }

    /// Fraction of requested operations that were skipped. `None` when
    /// nothing was requested, never `0.0`, which would read as "the
    /// policy declined every opportunity".
    pub fn skip_rate(&self) -> Option<f64> {
        self.is_measured()
            .then(|| self.skipped as f64 / self.requested as f64)
    }

    /// Avoided physical bytes as a fraction of what the canonical arm
    /// would have moved on this surface (`touched + avoided`). `None`
    /// when neither term is non-zero.
    ///
    /// Takes the touched count rather than reading it, so a caller must
    /// pass the byte total from the SAME window — the two counters are
    /// snapshotted together by [`super::TokenScope`] and pairing them
    /// across windows would silently mix arms.
    pub fn avoided_share(&self, physical_touched: u64) -> Option<f64> {
        let canonical = physical_touched + self.physical_avoided;
        (canonical > 0).then(|| self.physical_avoided as f64 / canonical as f64)
    }

    /// Whether the internal identity holds. A false here is an
    /// instrumentation bug (a site that recorded a decision without
    /// routing it through this module), not a property of the run.
    pub fn is_consistent(&self) -> bool {
        self.requested == self.executed + self.skipped
    }

    /// Fold `other`'s counters into `self`, field by field.
    pub fn accumulate(&mut self, other: &DecisionCounts) {
        self.requested += other.requested;
        self.executed += other.executed;
        self.skipped += other.skipped;
        self.semantic_avoided += other.semantic_avoided;
        self.physical_avoided += other.physical_avoided;
    }

    /// Difference `later - self`, saturating — matching
    /// [`super::ByteMovement::delta`]'s contract so a counter reset
    /// between reads cannot underflow into a huge bogus delta.
    pub fn delta(&self, later: &DecisionCounts) -> DecisionCounts {
        DecisionCounts {
            requested: later.requested.saturating_sub(self.requested),
            executed: later.executed.saturating_sub(self.executed),
            skipped: later.skipped.saturating_sub(self.skipped),
            semantic_avoided: later.semantic_avoided.saturating_sub(self.semantic_avoided),
            physical_avoided: later.physical_avoided.saturating_sub(self.physical_avoided),
        }
    }
}

static REQUESTED: AtomicU64 = AtomicU64::new(0);
static EXECUTED: AtomicU64 = AtomicU64::new(0);
static SKIPPED: AtomicU64 = AtomicU64::new(0);
static SEMANTIC_AVOIDED: AtomicU64 = AtomicU64::new(0);
static PHYSICAL_AVOIDED: AtomicU64 = AtomicU64::new(0);

/// Record one operation that the policy let run. Its bytes are recorded
/// separately, at the bind site, exactly as they always were.
#[inline]
pub fn record_executed() {
    REQUESTED.fetch_add(1, Ordering::Relaxed);
    EXECUTED.fetch_add(1, Ordering::Relaxed);
}

/// Record one operation the policy deleted, and the movement it would
/// have generated had it run.
///
/// `avoided` is the movement the operation WOULD have produced — the
/// same [`OperandMovement`] the bind site would have recorded. Deriving
/// it from the same shape arithmetic the executed path uses is what makes
/// `touched + avoided` a like-for-like reconstruction rather than two
/// independently-estimated numbers.
#[inline]
pub fn record_skipped(avoided: &OperandMovement) {
    REQUESTED.fetch_add(1, Ordering::Relaxed);
    SKIPPED.fetch_add(1, Ordering::Relaxed);
    SEMANTIC_AVOIDED.fetch_add(avoided.semantic, Ordering::Relaxed);
    PHYSICAL_AVOIDED.fetch_add(avoided.physical, Ordering::Relaxed);
}

/// Point-in-time reading of every decision counter.
pub fn snapshot() -> DecisionCounts {
    DecisionCounts {
        requested: REQUESTED.load(Ordering::Relaxed),
        executed: EXECUTED.load(Ordering::Relaxed),
        skipped: SKIPPED.load(Ordering::Relaxed),
        semantic_avoided: SEMANTIC_AVOIDED.load(Ordering::Relaxed),
        physical_avoided: PHYSICAL_AVOIDED.load(Ordering::Relaxed),
    }
}

/// Zero every decision counter. Available outside tests for the same
/// reason [`super::session::reset`] is: a harness running several arms in
/// one process must be able to separate them.
pub fn reset() {
    for c in [
        &REQUESTED,
        &EXECUTED,
        &SKIPPED,
        &SEMANTIC_AVOIDED,
        &PHYSICAL_AVOIDED,
    ] {
        c.store(0, Ordering::Relaxed);
    }
}

#[cfg(test)]
#[path = "tests/decisions.rs"]
mod tests;
