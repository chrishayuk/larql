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
    /// Calls that served more than one position from ONE weight
    /// traversal (CPU-7C).
    grouped: AtomicU64,
    /// Positions served, summed over calls. Equal to `calls` where every
    /// projection served one position, and `calls * K` where a batch of
    /// `K` grouped perfectly — so `positions / calls` is the realised
    /// group width, which is the quantity a CPU-7C timing has to be read
    /// against. A ratio near 1 with a good clock means arm B's cache
    /// reuse, not this.
    positions: AtomicU64,
    /// Wall nanoseconds inside this plan's projection calls.
    nanos: AtomicU64,
    /// Of those, nanoseconds inside calls that arrived through the
    /// MULTI-POSITION entry — the sites arm C actually groups.
    ///
    /// **Read this at K=1 and nowhere else.** These are wall intervals
    /// summed per call, so wherever the caller runs positions
    /// concurrently — `execute_layer` fans the FFN across positions with
    /// `par_iter_mut` — the intervals OVERLAP and the sum exceeds the
    /// elapsed time it is being divided by. At one position there is no
    /// cross-position concurrency, the intervals are disjoint, and
    /// `nanos_many / step_wall` is a true share of one ordinary token.
    ///
    /// At K=1 nothing is grouped (`supports` admits only 2 and 4), so
    /// this is an ELIGIBILITY tag rather than an achievement: the time in
    /// sites that arrive through the multi-position entry and would
    /// therefore amortise at K>1.
    ///
    /// This is what makes `g` a TIME share rather than a byte share.
    /// Projection shapes do not all run at the same rate, so grouped
    /// bytes over total bytes is a different number from grouped time
    /// over total time, and the prediction `1 - g/2` is about time. With
    /// the stationary sweep OFF a `K`-position run is `K` times a
    /// single-token run in every component, so this ratio is the same at
    /// `K` as at one — which is what lets `g` be read off the arm-B run
    /// rather than needing a separately instrumented arm.
    nanos_many: AtomicU64,
}

impl Tally {
    fn snapshot(&self) -> PlanTally {
        PlanTally {
            calls: self.calls.load(Ordering::Relaxed),
            bytes: self.bytes.load(Ordering::Relaxed),
            slabs: self.slabs.load(Ordering::Relaxed),
            grouped: self.grouped.load(Ordering::Relaxed),
            positions: self.positions.load(Ordering::Relaxed),
            nanos: self.nanos.load(Ordering::Relaxed),
            nanos_many: self.nanos_many.load(Ordering::Relaxed),
        }
    }

    fn reset(&self) {
        self.calls.store(0, Ordering::Relaxed);
        self.bytes.store(0, Ordering::Relaxed);
        self.slabs.store(0, Ordering::Relaxed);
        self.grouped.store(0, Ordering::Relaxed);
        self.positions.store(0, Ordering::Relaxed);
        self.nanos.store(0, Ordering::Relaxed);
        self.nanos_many.store(0, Ordering::Relaxed);
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
    /// Calls that served several positions from one weight traversal.
    pub grouped: u64,
    /// Positions served across all calls. See `Tally::positions`.
    pub positions: u64,
    /// Wall nanoseconds inside this plan's projections.
    pub nanos: u64,
    /// Of those, time in multi-position calls. See `Tally::nanos_many`.
    pub nanos_many: u64,
}

impl PlanTally {
    /// Positions served per call — the REALISED group width.
    ///
    /// The number that turns a disappointing CPU-7C clock into a
    /// diagnosis: at 1.0 the eligible projections never grouped, and the
    /// timing is measuring something else entirely.
    pub fn group_width(&self) -> f64 {
        if self.calls == 0 {
            0.0
        } else {
            self.positions as f64 / self.calls as f64
        }
    }
}

/// One projection call, as recorded.
///
/// A struct rather than six positional arguments: `bytes`, `slabs`,
/// `positions`, `nanos` and `nanos_many` are five integers whose
/// order a call site cannot check, and transposing two of them would
/// produce a plausible ledger that is quietly wrong.
pub(super) struct Call {
    pub bytes: usize,
    pub slabs: usize,
    pub positions: usize,
    /// The kernel's OWN answer to whether it shared the traversal —
    /// never inferred from `positions > 1`, because the looping
    /// default also serves several positions per call and counting it
    /// as grouped would make the ledger agree with the hope rather
    /// than with the machine.
    pub grouped: bool,
    pub nanos: u64,
    pub nanos_many: u64,
}

/// Which operator class a projection belongs to.
///
/// A SECOND axis on the ledger, orthogonal to the plan. "Which arithmetic
/// ran" and "which part of the model ran it" are different questions, and
/// CPU-7C2 needs the second: `g` has to be reported as `g_GD` and `g_FFN`
/// separately, in ONE binary, because taking the difference against a
/// previously banked figure would be a comparison across builds — and a
/// rebuild has already moved an untouched function 14% on this codebase.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Site {
    /// Outside any declared class — the head, the embedding, a test.
    Unclassified,
    /// GatedDelta's dense projections.
    Recurrent,
    /// FFN up / gate / down.
    Ffn,
    /// Softmax attention q / k / v / o.
    Attention,
}

impl Site {
    pub const ALL: [Site; 4] = [
        Site::Unclassified,
        Site::Recurrent,
        Site::Ffn,
        Site::Attention,
    ];

    pub fn name(self) -> &'static str {
        match self {
            Site::Unclassified => "other",
            Site::Recurrent => "recurrent",
            Site::Ffn => "ffn",
            Site::Attention => "attention",
        }
    }

    /// This class's bit in a class mask.
    pub fn bit(self) -> u8 {
        1 << self.index()
    }

    fn index(self) -> usize {
        match self {
            Site::Unclassified => 0,
            Site::Recurrent => 1,
            Site::Ffn => 2,
            Site::Attention => 3,
        }
    }
}

thread_local! {
    /// The class the calling thread is currently inside.
    ///
    /// Thread-local rather than a parameter on `DenseProjections`: the
    /// record happens on the ISSUING thread before any fan-out, so this
    /// is correct under the executor's own parallelism, and it keeps a
    /// diagnostic axis out of a domain trait's signature.
    static SITE: Cell<Site> = const { Cell::new(Site::Unclassified) };
}

/// Restores the enclosing class on drop, so nesting cannot leak.
pub struct SiteGuard(Site);

impl Drop for SiteGuard {
    fn drop(&mut self) {
        SITE.with(|s| s.set(self.0));
    }
}

/// Attribute every projection issued on this thread, until the guard is
/// dropped, to `site`.
pub fn in_site(site: Site) -> SiteGuard {
    SITE.with(|s| {
        let previous = s.get();
        s.set(site);
        SiteGuard(previous)
    })
}

/// The class the calling thread is currently inside.
pub fn current_site() -> Site {
    SITE.with(Cell::get)
}

/// Time in one operator class, and how much of it arrived through the
/// multi-position entry.
#[derive(Default)]
struct SiteTally {
    nanos: AtomicU64,
    nanos_many: AtomicU64,
}

/// One class, read out.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct SiteShare {
    pub nanos: u64,
    pub nanos_many: u64,
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
    fused_nvfp4: Tally,
    fused_kquant: Tally,
    fused_fp8_block: Tally,
    q4_x_q8: Tally,
    bf16_x_q8: Tally,
    /// The same time, cut by operator class instead of by arithmetic.
    sites: [SiteTally; 4],
}

impl ProjectionLedger {
    fn tally(&self, plan: PhysicalProjectionPlan) -> &Tally {
        match plan {
            PhysicalProjectionPlan::ScalarF32 => &self.scalar,
            PhysicalProjectionPlan::BlasF32 => &self.blas,
            PhysicalProjectionPlan::FusedBf16 => &self.fused,
            PhysicalProjectionPlan::FusedQ8 => &self.fused_q8,
            PhysicalProjectionPlan::FusedQ4 => &self.fused_q4,
            PhysicalProjectionPlan::FusedNvfp4 => &self.fused_nvfp4,
            PhysicalProjectionPlan::FusedKQuant => &self.fused_kquant,
            PhysicalProjectionPlan::FusedFp8Block => &self.fused_fp8_block,
            PhysicalProjectionPlan::Q8xQ8 => &self.q8_x_q8,
            PhysicalProjectionPlan::Q4xQ8 => &self.q4_x_q8,
            PhysicalProjectionPlan::Bf16xQ8 => &self.bf16_x_q8,
        }
    }

    pub(super) fn record(
        &self,
        plan: PhysicalProjectionPlan,
        bytes: usize,
        slabs: usize,
        nanos: u64,
    ) {
        self.record_many(
            plan,
            Call {
                bytes,
                slabs,
                positions: 1,
                grouped: false,
                nanos,
                nanos_many: 0,
            },
        );
    }

    pub(super) fn record_many(&self, plan: PhysicalProjectionPlan, call: Call) {
        THREAD_CALLS.with(|c| c.set(c.get() + 1));
        let t = self.tally(plan);
        t.calls.fetch_add(1, Ordering::Relaxed);
        t.bytes.fetch_add(call.bytes as u64, Ordering::Relaxed);
        t.slabs.fetch_add(call.slabs as u64, Ordering::Relaxed);
        t.positions
            .fetch_add(call.positions as u64, Ordering::Relaxed);
        t.nanos.fetch_add(call.nanos, Ordering::Relaxed);
        t.nanos_many.fetch_add(call.nanos_many, Ordering::Relaxed);
        let site = &self.sites[SITE.with(Cell::get).index()];
        site.nanos.fetch_add(call.nanos, Ordering::Relaxed);
        site.nanos_many
            .fetch_add(call.nanos_many, Ordering::Relaxed);
        if call.grouped {
            t.grouped.fetch_add(1, Ordering::Relaxed);
        }
    }

    pub fn get(&self, plan: PhysicalProjectionPlan) -> PlanTally {
        self.tally(plan).snapshot()
    }

    /// Every plan, so a reader enumerates rather than remembers. A caller
    /// that listed the plans itself would stop covering a new one on the
    /// day it was added.
    pub fn all(&self) -> [(PhysicalProjectionPlan, PlanTally); 11] {
        [
            PhysicalProjectionPlan::ScalarF32,
            PhysicalProjectionPlan::BlasF32,
            PhysicalProjectionPlan::FusedBf16,
            PhysicalProjectionPlan::FusedQ8,
            PhysicalProjectionPlan::FusedQ4,
            // The observation-only plans are enumerated too. NVFP4 had a
            // slot here and no row, so a decode that ran a compiled pack
            // reported its bytes nowhere — exactly the silent omission
            // this method's doc comment says it exists to prevent.
            PhysicalProjectionPlan::FusedNvfp4,
            PhysicalProjectionPlan::FusedKQuant,
            PhysicalProjectionPlan::FusedFp8Block,
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

    /// Nanoseconds in projections, and the share of them in the sites the
    /// multi-position entry serves.
    ///
    /// The numerator of `g`. The denominator is the STEP's wall time,
    /// which the ledger cannot see and the caller must supply — keeping
    /// that division outside here is deliberate: a ledger that divided by
    /// its own total would report projection-share-of-projection, which
    /// is 1 by construction and has been mistaken for a result before.
    pub fn projection_nanos(&self) -> (u64, u64) {
        self.all()
            .iter()
            .fold((0, 0), |(a, b), (_, t)| (a + t.nanos, b + t.nanos_many))
    }

    /// Projection time in one operator class.
    ///
    /// `nanos_many` is the class's contribution to `g` — and, like the
    /// whole-ledger figure, is a true share only at K=1, where no caller
    /// runs positions concurrently and the per-call intervals are disjoint.
    pub fn site(&self, site: Site) -> SiteShare {
        let t = &self.sites[site.index()];
        SiteShare {
            nanos: t.nanos.load(Ordering::Relaxed),
            nanos_many: t.nanos_many.load(Ordering::Relaxed),
        }
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
        // The observation-only plans were missing here too: NVFP4 bytes
        // survived every reset, so a priced step could carry the load.
        self.fused_nvfp4.reset();
        self.fused_kquant.reset();
        self.fused_fp8_block.reset();
        self.q8_x_q8.reset();
        self.q4_x_q8.reset();
        self.bf16_x_q8.reset();
        for s in &self.sites {
            s.nanos.store(0, Ordering::Relaxed);
            s.nanos_many.store(0, Ordering::Relaxed);
        }
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
            grouped: AtomicU64::new(0),
            positions: AtomicU64::new(0),
            nanos: AtomicU64::new(0),
            nanos_many: AtomicU64::new(0),
        };
        Self {
            scalar: ZERO,
            blas: ZERO,
            fused: ZERO,
            fused_q8: ZERO,
            fused_q4: ZERO,
            fused_nvfp4: ZERO,
            fused_kquant: ZERO,
            fused_fp8_block: ZERO,
            q8_x_q8: ZERO,
            q4_x_q8: ZERO,
            bf16_x_q8: ZERO,
            #[allow(clippy::declare_interior_mutable_const)]
            sites: [const {
                SiteTally {
                    nanos: AtomicU64::new(0),
                    nanos_many: AtomicU64::new(0),
                }
            }; 4],
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
