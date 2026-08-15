//! The byte half of the BW10 ledger: what moved, across which tier, and
//! how much of it the computation actually consumed.
//!
//! Three byte classes, deliberately distinct, because collapsing them is
//! what makes "logical sparsity" read as a saving it never delivered:
//!
//! - **semantic** — what the plan logically asked for (the operand's
//!   meaningful extent: `inter × hidden` weights, unpadded).
//! - **physical** — what was actually bound and streamed (padded rows,
//!   whole quant blocks, whole pages).
//! - **useful** — of the physical bytes, those that reached the result.
//!
//! `physical / semantic` is representation-and-layout amplification;
//! `useful / physical` is access-pattern efficiency. LA-6 measured
//! scattered selection touching 29–89× its logical count — that ratio is
//! this field, and it is the reason a byte ledger that reports only one
//! number cannot tell a real saving from a bookkeeping one.

use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};

/// Which memory tier a byte crossed. Bytes are attributed to exactly one
/// tier at the point they are recorded, so the tier columns partition the
/// physical total and any shortfall is reported as unattributed rather
/// than silently absorbed.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Tier {
    /// Resident traffic: the compute engine reads it from local memory.
    Dram,
    /// Cold-estate traffic: pulled from local storage this token.
    Nvme,
    /// Distributed-estate traffic: crossed a network link this token.
    Network,
}

/// One operand's movement, as recorded at a bind/read site.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct OperandMovement {
    pub semantic: u64,
    pub physical: u64,
    pub useful: u64,
    pub tier: Tier,
}

impl OperandMovement {
    /// Every physical byte bound is consumed by the kernel — the dense
    /// streaming read. `physical` may still exceed `semantic` (padded
    /// rows, quant block granularity); that is layout amplification, not
    /// waste at the access level.
    pub fn fully_consumed(semantic: u64, physical: u64, tier: Tier) -> Self {
        Self {
            semantic,
            physical,
            useful: physical,
            tier,
        }
    }

    /// Only `useful` of the `physical` bytes fetched reach the result —
    /// the gather / page-amplification case. Callers must supply a useful
    /// count they can defend; `useful` is clamped to `physical` so a
    /// mis-instrumented site cannot manufacture efficiency above 1.0.
    pub fn partially_consumed(semantic: u64, physical: u64, useful: u64, tier: Tier) -> Self {
        Self {
            semantic,
            physical,
            useful: useful.min(physical),
            tier,
        }
    }
}

/// Per-token byte totals — the value form, read out of the counters.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct ByteMovement {
    pub semantic_requested: u64,
    pub physical_touched: u64,
    pub useful_physical: u64,
    pub dram: u64,
    pub nvme: u64,
    pub network: u64,
    /// Bytes a reuse reporter attributed to an already-resident copy.
    /// Meaningful only when [`Self::reuse_observed`] is true.
    pub reused: u64,
    /// Bytes fetched speculatively ahead of the consuming operation.
    pub prefetched: u64,
    /// Of `prefetched`, those never consumed before eviction — prediction
    /// waste. Meaningful only when [`Self::prefetch_observed`] is true.
    pub prefetched_unused: u64,
    /// A reuse reporter bumped at least one counter this window. Without
    /// it, `reused == 0` means "nobody measured", not "no reuse".
    pub reuse_observed: bool,
    /// A prefetcher reported at least one fetch this window. Without it,
    /// `prefetched == 0` means "no prefetcher registered".
    pub prefetch_observed: bool,
}

impl ByteMovement {
    /// Physical bytes not attributed to any tier. Non-zero means a bind
    /// site records movement without naming where the byte came from —
    /// a gap in the instrument, printed rather than absorbed.
    pub fn tier_unattributed(&self) -> u64 {
        self.physical_touched
            .saturating_sub(self.dram + self.nvme + self.network)
    }

    /// `physical / semantic` — layout and representation amplification.
    /// `None` when nothing semantic was requested (no division to make).
    pub fn amplification(&self) -> Option<f64> {
        (self.semantic_requested > 0)
            .then(|| self.physical_touched as f64 / self.semantic_requested as f64)
    }

    /// `useful / physical` — access-pattern efficiency. `None` when
    /// nothing physical moved.
    pub fn useful_ratio(&self) -> Option<f64> {
        (self.physical_touched > 0)
            .then(|| self.useful_physical as f64 / self.physical_touched as f64)
    }

    /// Bytes that crossed a tier external to the compute engine's own
    /// memory. This is the quantity the cold-estate floor divides.
    pub fn external(&self) -> u64 {
        self.nvme + self.network
    }

    /// Prediction waste as a fraction of speculative traffic. `None` when
    /// no prefetcher reported — never 0.0, which would read as perfect.
    pub fn prefetch_waste_ratio(&self) -> Option<f64> {
        (self.prefetch_observed && self.prefetched > 0)
            .then(|| self.prefetched_unused as f64 / self.prefetched as f64)
    }

    /// Fold `other`'s counters into `self`, field by field. Used to build
    /// a running total from a stream of token records — [`super::SteadyState`]
    /// keeps two independent totals (decode, prefill) built with this same
    /// method, so the two buckets cannot drift out of sync with each
    /// other's arithmetic.
    pub fn accumulate(&mut self, other: &ByteMovement) {
        self.semantic_requested += other.semantic_requested;
        self.physical_touched += other.physical_touched;
        self.useful_physical += other.useful_physical;
        self.dram += other.dram;
        self.nvme += other.nvme;
        self.network += other.network;
        self.reused += other.reused;
        self.prefetched += other.prefetched;
        self.prefetched_unused += other.prefetched_unused;
        self.reuse_observed |= other.reuse_observed;
        self.prefetch_observed |= other.prefetch_observed;
    }

    /// Difference `later - self`, field by field. Saturating so a counter
    /// reset between reads cannot underflow into a huge bogus delta.
    pub fn delta(&self, later: &ByteMovement) -> ByteMovement {
        ByteMovement {
            semantic_requested: later
                .semantic_requested
                .saturating_sub(self.semantic_requested),
            physical_touched: later.physical_touched.saturating_sub(self.physical_touched),
            useful_physical: later.useful_physical.saturating_sub(self.useful_physical),
            dram: later.dram.saturating_sub(self.dram),
            nvme: later.nvme.saturating_sub(self.nvme),
            network: later.network.saturating_sub(self.network),
            reused: later.reused.saturating_sub(self.reused),
            prefetched: later.prefetched.saturating_sub(self.prefetched),
            prefetched_unused: later
                .prefetched_unused
                .saturating_sub(self.prefetched_unused),
            reuse_observed: later.reuse_observed,
            prefetch_observed: later.prefetch_observed,
        }
    }
}

/// Process-wide byte counters, in the [`crate::movement_ledger`] contract
/// style: production bind sites bump them unconditionally (a relaxed add
/// is far below dispatch noise), and readers take snapshot deltas.
///
/// Unconditional bumping is deliberate. A counter that only runs under a
/// diagnostic env flag measures a different program from the one that
/// ships, and this ledger exists to adjudicate performance claims about
/// the shipping program.
static SEMANTIC: AtomicU64 = AtomicU64::new(0);
static PHYSICAL: AtomicU64 = AtomicU64::new(0);
static USEFUL: AtomicU64 = AtomicU64::new(0);
static DRAM: AtomicU64 = AtomicU64::new(0);
static NVME: AtomicU64 = AtomicU64::new(0);
static NETWORK: AtomicU64 = AtomicU64::new(0);
static REUSED: AtomicU64 = AtomicU64::new(0);
static PREFETCHED: AtomicU64 = AtomicU64::new(0);
static PREFETCHED_UNUSED: AtomicU64 = AtomicU64::new(0);
static REUSE_OBSERVED: AtomicBool = AtomicBool::new(false);
static PREFETCH_OBSERVED: AtomicBool = AtomicBool::new(false);

/// Record one operand read at a bind site.
#[inline]
pub fn record(m: OperandMovement) {
    SEMANTIC.fetch_add(m.semantic, Ordering::Relaxed);
    PHYSICAL.fetch_add(m.physical, Ordering::Relaxed);
    USEFUL.fetch_add(m.useful, Ordering::Relaxed);
    let tier = match m.tier {
        Tier::Dram => &DRAM,
        Tier::Nvme => &NVME,
        Tier::Network => &NETWORK,
    };
    tier.fetch_add(m.physical, Ordering::Relaxed);
}

/// Report `bytes` served from an already-resident copy — a cache hit that
/// avoided a tier crossing. Registers the reporter, so a later zero is
/// distinguishable from an unmeasured zero.
#[inline]
pub fn record_reuse(bytes: u64) {
    REUSE_OBSERVED.store(true, Ordering::Relaxed);
    REUSED.fetch_add(bytes, Ordering::Relaxed);
}

/// Report `bytes` fetched speculatively. Pair with
/// [`record_prefetch_unused`] when the speculation is resolved.
#[inline]
pub fn record_prefetch(bytes: u64) {
    PREFETCH_OBSERVED.store(true, Ordering::Relaxed);
    PREFETCHED.fetch_add(bytes, Ordering::Relaxed);
}

/// Report `bytes` that were prefetched and evicted without being read.
#[inline]
pub fn record_prefetch_unused(bytes: u64) {
    PREFETCH_OBSERVED.store(true, Ordering::Relaxed);
    PREFETCHED_UNUSED.fetch_add(bytes, Ordering::Relaxed);
}

/// Point-in-time reading of every byte counter.
pub fn snapshot() -> ByteMovement {
    ByteMovement {
        semantic_requested: SEMANTIC.load(Ordering::Relaxed),
        physical_touched: PHYSICAL.load(Ordering::Relaxed),
        useful_physical: USEFUL.load(Ordering::Relaxed),
        dram: DRAM.load(Ordering::Relaxed),
        nvme: NVME.load(Ordering::Relaxed),
        network: NETWORK.load(Ordering::Relaxed),
        reused: REUSED.load(Ordering::Relaxed),
        prefetched: PREFETCHED.load(Ordering::Relaxed),
        prefetched_unused: PREFETCHED_UNUSED.load(Ordering::Relaxed),
        reuse_observed: REUSE_OBSERVED.load(Ordering::Relaxed),
        prefetch_observed: PREFETCH_OBSERVED.load(Ordering::Relaxed),
    }
}

/// Serialises tests that move the process-wide counters. Lives beside the
/// counters it guards so every test module reaches for the same one.
#[cfg(test)]
pub(crate) static COUNTER_LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());

/// Zero every byte counter. Test-only: production readers take deltas so
/// they compose with concurrent work, and a global reset would corrupt an
/// outer window that is still open.
#[cfg(test)]
pub(crate) fn reset_for_test() {
    for c in [
        &SEMANTIC,
        &PHYSICAL,
        &USEFUL,
        &DRAM,
        &NVME,
        &NETWORK,
        &REUSED,
        &PREFETCHED,
        &PREFETCHED_UNUSED,
    ] {
        c.store(0, Ordering::Relaxed);
    }
    REUSE_OBSERVED.store(false, Ordering::Relaxed);
    PREFETCH_OBSERVED.store(false, Ordering::Relaxed);
}

#[cfg(test)]
#[path = "tests/bytes.rs"]
mod tests;
