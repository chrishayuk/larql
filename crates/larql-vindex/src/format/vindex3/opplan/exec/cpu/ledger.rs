//! What the executor ACTUALLY ran — the counterpart to what the loader
//! decided.
//!
//! The residency census reads the loader's own bookkeeping, so on its own
//! it cannot fail the way that matters: a census can report 51 GB compact
//! while every projection quietly widens a tile before computing. The two
//! instruments answer different questions and only agree if both are true.
//!
//! Global rather than per backend, for the reason the pool is:
//! `ProductionBackend` is a zero-sized value that call sites construct
//! freely, so per-instance counters would each see a fraction of a decode
//! and none of them the whole.
//!
//! Cost is two relaxed atomic adds per projection against roughly 400
//! projections and 51 GB of streaming per token — unmeasurable, so it is
//! always on rather than behind a feature that would be off exactly when
//! a number needed explaining.

use std::cell::Cell;
use std::sync::atomic::{AtomicU64, Ordering};

use super::physical::PhysicalProjectionPlan;

/// One plan's tally.
#[derive(Default)]
struct Tally {
    calls: AtomicU64,
    bytes: AtomicU64,
    /// Row slabs handed to workers. Equal to `calls` for an unpartitioned
    /// kernel, and `calls * workers` for a fully fanned-out one — which is
    /// what makes per-dispatch overhead visible in a decode rather than
    /// only in a bench.
    slabs: AtomicU64,
}

impl Tally {
    fn snapshot(&self) -> PlanTally {
        PlanTally {
            calls: self.calls.load(Ordering::Relaxed),
            bytes: self.bytes.load(Ordering::Relaxed),
            slabs: self.slabs.load(Ordering::Relaxed),
        }
    }

    fn reset(&self) {
        self.calls.store(0, Ordering::Relaxed);
        self.bytes.store(0, Ordering::Relaxed);
        self.slabs.store(0, Ordering::Relaxed);
    }
}

/// One plan's tally, read out.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct PlanTally {
    pub calls: u64,
    /// Weight bytes read in the representation they were resident as —
    /// directly comparable across plans, and the quantity the roofline is
    /// stated in.
    pub bytes: u64,
    pub slabs: u64,
}

/// Every projection the CPU executor has run, by plan.
#[derive(Default)]
pub struct ProjectionLedger {
    scalar: Tally,
    blas: Tally,
    fused: Tally,
    fused_q8: Tally,
    fused_q4: Tally,
    // One tally PER ARM, never a shared "integer" bucket. The ledger is
    // the consumption half of the residency instrument, and an arm that
    // folded into another's counter would make the byte census agree with
    // itself while describing a mixture.
    q8_x_q8: Tally,
    q4_x_q8: Tally,
    bf16_x_q8: Tally,
}

impl ProjectionLedger {
    fn tally(&self, plan: PhysicalProjectionPlan) -> &Tally {
        match plan {
            PhysicalProjectionPlan::ScalarF32 => &self.scalar,
            PhysicalProjectionPlan::BlasF32 => &self.blas,
            PhysicalProjectionPlan::FusedBf16 => &self.fused,
            PhysicalProjectionPlan::FusedQ8 => &self.fused_q8,
            PhysicalProjectionPlan::FusedQ4 => &self.fused_q4,
            PhysicalProjectionPlan::Q8xQ8 => &self.q8_x_q8,
            PhysicalProjectionPlan::Q4xQ8 => &self.q4_x_q8,
            PhysicalProjectionPlan::Bf16xQ8 => &self.bf16_x_q8,
        }
    }

    pub(super) fn record(&self, plan: PhysicalProjectionPlan, bytes: usize, slabs: usize) {
        THREAD_CALLS.with(|c| c.set(c.get() + 1));
        let t = self.tally(plan);
        t.calls.fetch_add(1, Ordering::Relaxed);
        t.bytes.fetch_add(bytes as u64, Ordering::Relaxed);
        t.slabs.fetch_add(slabs as u64, Ordering::Relaxed);
    }

    pub fn get(&self, plan: PhysicalProjectionPlan) -> PlanTally {
        self.tally(plan).snapshot()
    }

    /// Every plan, so a reader enumerates rather than remembers. A caller
    /// that listed the plans itself would stop covering a new one on the
    /// day it was added.
    pub fn all(&self) -> [(PhysicalProjectionPlan, PlanTally); 8] {
        [
            PhysicalProjectionPlan::ScalarF32,
            PhysicalProjectionPlan::BlasF32,
            PhysicalProjectionPlan::FusedBf16,
            PhysicalProjectionPlan::FusedQ8,
            PhysicalProjectionPlan::FusedQ4,
            PhysicalProjectionPlan::Q8xQ8,
            PhysicalProjectionPlan::Q4xQ8,
            PhysicalProjectionPlan::Bf16xQ8,
        ]
        .map(|p| (p, self.get(p)))
    }

    /// Weight bytes across every plan — what one decode step streamed.
    pub fn total_bytes(&self) -> u64 {
        self.all().iter().map(|(_, t)| t.bytes).sum()
    }

    /// Zero the counters, so a caller can price ONE step.
    ///
    /// Nothing here is per session, so a reader that forgot this would be
    /// measuring the weight load and every warm-up step as well.
    pub fn reset(&self) {
        self.scalar.reset();
        self.blas.reset();
        self.fused.reset();
        self.fused_q8.reset();
        self.fused_q4.reset();
        self.q8_x_q8.reset();
        self.q4_x_q8.reset();
        self.bf16_x_q8.reset();
    }
}

impl ProjectionLedger {
    /// An empty ledger. `const` so the process one is a static, and so a
    /// test can hold its own rather than race the shared counters.
    pub(crate) const fn new() -> Self {
        #[allow(clippy::declare_interior_mutable_const)]
        const ZERO: Tally = Tally {
            calls: AtomicU64::new(0),
            bytes: AtomicU64::new(0),
            slabs: AtomicU64::new(0),
        };
        Self {
            scalar: ZERO,
            blas: ZERO,
            fused: ZERO,
            fused_q8: ZERO,
            fused_q4: ZERO,
            q8_x_q8: ZERO,
            q4_x_q8: ZERO,
            bf16_x_q8: ZERO,
        }
    }
}

static LEDGER: ProjectionLedger = ProjectionLedger::new();

/// The process's projection ledger.
pub fn ledger() -> &'static ProjectionLedger {
    &LEDGER
}

thread_local! {
    /// Projections ISSUED BY THIS THREAD.
    static THREAD_CALLS: Cell<u64> = const { Cell::new(0) };
}

/// How many projections this thread has issued.
///
/// The process ledger prices a decode step, which runs on one thread, so
/// for that purpose the two agree. This exists for the case they do not:
/// a caller — a test, most often — that needs a count immune to whatever
/// else the process is doing concurrently. Comparing two arms against a
/// shared counter while the rest of a suite runs its own projections
/// measures the suite, not the arms.
///
/// Counts the CALL, not the worker slabs it fans out into, because it is
/// recorded on the issuing thread before the fan-out.
pub fn thread_projection_calls() -> u64 {
    THREAD_CALLS.with(|c| c.get())
}
