//! Execution-path witness counters — part of rung E's CONTRACT, not a
//! temporary diagnostic.
//!
//! The poison proof shows layer output does not DEPEND on host-visible
//! route values; these counters prove the route-dependent host work is
//! genuinely ABSENT — because F's performance claim needs the work gone,
//! not merely output-irrelevant, and because a later refactor could
//! quietly reintroduce a host dependency while preserving numerical
//! parity. The legacy CPU-routed path bumps them at every
//! route-dependent host action; the descriptor path must leave them
//! untouched. Gates assert a zero delta across the candidate encode and
//! a non-zero delta across the control encode (the witness's own
//! positive control).
//!
//! Process-wide relaxed atomics: encode paths are already serialized per
//! backend, and the gates only compare deltas taken on one thread.

use std::sync::atomic::{AtomicU64, Ordering};

/// CPU resolutions of selected experts to buffer bindings
/// (`resolve_selected_experts`).
pub static HOST_RESOLVES: AtomicU64 = AtomicU64::new(0);
/// CPU staging copies of selected experts' bias rows (gate/up loop,
/// down-bias loop).
pub static BIAS_COPIES: AtomicU64 = AtomicU64::new(0);
/// Routing weights injected into a command via `set_bytes`.
pub static WEIGHT_BINDS: AtomicU64 = AtomicU64::new(0);
/// Per-slot expert offset tables injected via `set_bytes`.
pub static OFFSET_BINDS: AtomicU64 = AtomicU64::new(0);
/// Layers encoded via the GPU-dataflow route (serve rung S1) — the
/// positive half of the serve gate: "descriptor path fires on every
/// routed layer" is checked against this, not inferred from silence.
pub static GPU_ROUTE_LAYERS: AtomicU64 = AtomicU64::new(0);
/// Per-layer attention completions whose only purpose was letting the
/// CPU read the route input (`handle_moe_interleave`'s inherited wait).
/// Under a fully GPU-routed token this MUST stay zero — any residual
/// bubble then belongs to a DIFFERENT, named boundary.
pub static WAIT_MOE_ROUTE_LEGACY: AtomicU64 = AtomicU64::new(0);

/// Attention dispatches encoded by the VINDEX3 lowering on the serial
/// phase-3 kernel (`kv_attention` / `kv_attention_long`).
pub static LOWERED_ATTEND_SERIAL: AtomicU64 = AtomicU64::new(0);
/// Attention dispatches encoded by the VINDEX3 lowering on the KV-B1
/// sequence-parallel kernel. The lowering's seqpar port is judged by this
/// moving, not by a throughput number that might have another cause.
pub static LOWERED_ATTEND_SEQPAR: AtomicU64 = AtomicU64::new(0);
/// Attention ops encoded by the VINDEX3 lowering on SPLITK-1 (span split
/// across threadgroups, then merged) — one bump per op, both passes.
pub static LOWERED_ATTEND_SPLITK: AtomicU64 = AtomicU64::new(0);

#[inline]
pub(crate) fn bump(counter: &AtomicU64) {
    counter.fetch_add(1, Ordering::Relaxed);
}

/// Point-in-time reading of all four counters.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Snapshot {
    pub host_resolves: u64,
    pub bias_copies: u64,
    pub weight_binds: u64,
    pub offset_binds: u64,
}

pub fn snapshot() -> Snapshot {
    Snapshot {
        host_resolves: HOST_RESOLVES.load(Ordering::Relaxed),
        bias_copies: BIAS_COPIES.load(Ordering::Relaxed),
        weight_binds: WEIGHT_BINDS.load(Ordering::Relaxed),
        offset_binds: OFFSET_BINDS.load(Ordering::Relaxed),
    }
}

impl Snapshot {
    /// Counter movement since `self` (later minus earlier).
    pub fn delta(&self, later: &Snapshot) -> Snapshot {
        Snapshot {
            host_resolves: later.host_resolves - self.host_resolves,
            bias_copies: later.bias_copies - self.bias_copies,
            weight_binds: later.weight_binds - self.weight_binds,
            offset_binds: later.offset_binds - self.offset_binds,
        }
    }

    /// True when no route-dependent host action happened in the window.
    pub fn is_zero(&self) -> bool {
        self.host_resolves == 0
            && self.bias_copies == 0
            && self.weight_binds == 0
            && self.offset_binds == 0
    }
}
