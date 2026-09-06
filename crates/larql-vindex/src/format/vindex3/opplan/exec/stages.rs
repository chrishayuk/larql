//! The stage ledger: where a token's wall time goes, at the level a
//! residency decision is priced at — attention, the router, the routed
//! experts, the shared expert — and nothing finer.
//!
//! This is a second instrument beside [`super::timing`], not a
//! replacement. The leaf ledger refuses to reconcile when timers nest,
//! because nested leaves double-count; a stage CONTAINS leaves by
//! definition, so a stage timer must never be a leaf timer. Stages are
//! disjoint at their own level: one stage starting inside another on the
//! same thread is counted as a violation and voids the stage sum, the
//! same discipline the leaf ledger keeps for itself.

use std::cell::Cell;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::Instant;

/// One stage of a token's execution.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Stage {
    /// The layer's attention operator, whatever its kind: KDA, MLA,
    /// softmax, gated delta, Mamba-2, conv-QKV.
    Attention,
    /// The router: its projection and the selection it makes.
    Router,
    /// The routed experts' projections, over the selected set.
    RoutedExperts,
    /// The always-active shared expert.
    SharedExpert,
    /// Bringing the selected experts' pages in ahead of the routed loop.
    Prefetch,
}

impl Stage {
    pub const ALL: [Stage; 5] = [
        Stage::Attention,
        Stage::Router,
        Stage::RoutedExperts,
        Stage::SharedExpert,
        Stage::Prefetch,
    ];

    pub fn name(self) -> &'static str {
        match self {
            Stage::Attention => "attention",
            Stage::Router => "router",
            Stage::RoutedExperts => "routed_experts",
            Stage::SharedExpert => "shared_expert",
            Stage::Prefetch => "prefetch",
        }
    }

    fn index(self) -> usize {
        match self {
            Stage::Attention => 0,
            Stage::Router => 1,
            Stage::RoutedExperts => 2,
            Stage::SharedExpert => 3,
            Stage::Prefetch => 4,
        }
    }
}

/// What one stage has accumulated since the last reset.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct StageTally {
    pub calls: u64,
    pub nanos: u64,
}

struct Slot {
    calls: AtomicU64,
    nanos: AtomicU64,
}

/// Every stage's tally, and the count of stages that started inside
/// another on the same thread.
pub struct StageLedger {
    slots: [Slot; 5],
    nested: AtomicU64,
}

thread_local! {
    static ACTIVE: Cell<bool> = const { Cell::new(false) };
}

impl StageLedger {
    pub(super) const fn new() -> Self {
        #[allow(clippy::declare_interior_mutable_const)]
        const ZERO: Slot = Slot {
            calls: AtomicU64::new(0),
            nanos: AtomicU64::new(0),
        };
        Self {
            slots: [ZERO; 5],
            nested: AtomicU64::new(0),
        }
    }

    fn record(&self, stage: Stage, nanos: u64) {
        let slot = &self.slots[stage.index()];
        slot.calls.fetch_add(1, Ordering::Relaxed);
        slot.nanos.fetch_add(nanos, Ordering::Relaxed);
    }

    pub fn get(&self, stage: Stage) -> StageTally {
        let slot = &self.slots[stage.index()];
        StageTally {
            calls: slot.calls.load(Ordering::Relaxed),
            nanos: slot.nanos.load(Ordering::Relaxed),
        }
    }

    pub fn all(&self) -> [(Stage, StageTally); 5] {
        Stage::ALL.map(|s| (s, self.get(s)))
    }

    /// Stages that started while another stage was running on the same
    /// thread. Any value above zero voids the stage sum.
    pub fn nested(&self) -> u64 {
        self.nested.load(Ordering::Relaxed)
    }

    /// The sum of every stage's nanoseconds — meaningful only when
    /// [`Self::nested`] is zero.
    pub fn total_nanos(&self) -> u64 {
        self.slots
            .iter()
            .map(|s| s.nanos.load(Ordering::Relaxed))
            .sum()
    }

    pub fn reset(&self) {
        for slot in &self.slots {
            slot.calls.store(0, Ordering::Relaxed);
            slot.nanos.store(0, Ordering::Relaxed);
        }
        self.nested.store(0, Ordering::Relaxed);
    }
}

/// A running stage timer. Records on drop.
pub struct Staged {
    stage: Stage,
    started: Instant,
    outermost: bool,
}

impl Drop for Staged {
    fn drop(&mut self) {
        let nanos = self.started.elapsed().as_nanos() as u64;
        ledger().record(self.stage, nanos);
        if self.outermost {
            ACTIVE.with(|a| a.set(false));
        }
    }
}

/// Start one stage. Bind the value to a name for the stage's extent.
pub fn stage(stage: Stage) -> Staged {
    let outermost = ACTIVE.with(|a| {
        let was = a.get();
        a.set(true);
        !was
    });
    if !outermost {
        ledger().nested.fetch_add(1, Ordering::Relaxed);
    }
    Staged {
        stage,
        started: Instant::now(),
        outermost,
    }
}

static LEDGER: StageLedger = StageLedger::new();

/// The process's stage ledger.
pub fn ledger() -> &'static StageLedger {
    &LEDGER
}
