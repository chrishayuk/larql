//! What the ledger actually accounts for.
//!
//! A partial byte ledger that renders like a complete one is worse than
//! no ledger, because every share it derives is silently understated and
//! the reader has no way to see it. So the instrumented surfaces are
//! enumerated here, each carries whether a bump site exists at all, and
//! each records FIRED EVIDENCE at runtime — a surface that is
//! instrumented but never fires is reported as silent, never assumed
//! covered.
//!
//! The report prints all three states. "Instrumented" is a property of
//! the code; "fired" is a property of the run; "not instrumented" is an
//! admission. Only the first two license a share.

use std::sync::atomic::{AtomicU64, Ordering};

/// A class of weight traffic the decode path can generate.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Surface {
    /// Routed MoE expert payloads (gate/up/down) plus their scale streams.
    MoeExperts,
    /// Q/K/V/O projection weights.
    AttentionProjections,
    /// Dense (non-routed) FFN weights on hybrid or dense layers.
    DenseFfn,
    /// Final norm → vocabulary projection.
    LmHead,
    /// KV cache reads and appends. Grows with context, so a ledger that
    /// omits it is increasingly wrong on long prompts.
    KvCache,
    /// Token embedding lookups.
    Embeddings,
    /// Norm weights — small, but named so the omission is explicit.
    Norms,
}

/// Every surface, in report order.
pub const ALL_SURFACES: [Surface; 7] = [
    Surface::MoeExperts,
    Surface::AttentionProjections,
    Surface::DenseFfn,
    Surface::LmHead,
    Surface::KvCache,
    Surface::Embeddings,
    Surface::Norms,
];

impl Surface {
    pub const fn label(self) -> &'static str {
        match self {
            Surface::MoeExperts => "moe-experts",
            Surface::AttentionProjections => "attention-qkvo",
            Surface::DenseFfn => "dense-ffn",
            Surface::LmHead => "lm-head",
            Surface::KvCache => "kv-cache",
            Surface::Embeddings => "embeddings",
            Surface::Norms => "norms",
        }
    }

    /// Whether a bump site exists for this surface anywhere in the tree.
    ///
    /// Flipping one of these to `true` without adding the corresponding
    /// [`record`] call is the one way to make this module lie, so the
    /// coverage test asserts every `true` surface can actually fire.
    pub const fn is_instrumented(self) -> bool {
        match self {
            Surface::MoeExperts => true,
            Surface::AttentionProjections => false,
            Surface::DenseFfn => false,
            Surface::LmHead => false,
            Surface::KvCache => false,
            Surface::Embeddings => false,
            Surface::Norms => false,
        }
    }

    fn counter(self) -> &'static AtomicU64 {
        match self {
            Surface::MoeExperts => &FIRED_MOE_EXPERTS,
            Surface::AttentionProjections => &FIRED_ATTENTION,
            Surface::DenseFfn => &FIRED_DENSE_FFN,
            Surface::LmHead => &FIRED_LM_HEAD,
            Surface::KvCache => &FIRED_KV_CACHE,
            Surface::Embeddings => &FIRED_EMBEDDINGS,
            Surface::Norms => &FIRED_NORMS,
        }
    }

    /// How many times this surface recorded movement, process-wide.
    pub fn fired(self) -> u64 {
        self.counter().load(Ordering::Relaxed)
    }
}

static FIRED_MOE_EXPERTS: AtomicU64 = AtomicU64::new(0);
static FIRED_ATTENTION: AtomicU64 = AtomicU64::new(0);
static FIRED_DENSE_FFN: AtomicU64 = AtomicU64::new(0);
static FIRED_LM_HEAD: AtomicU64 = AtomicU64::new(0);
static FIRED_KV_CACHE: AtomicU64 = AtomicU64::new(0);
static FIRED_EMBEDDINGS: AtomicU64 = AtomicU64::new(0);
static FIRED_NORMS: AtomicU64 = AtomicU64::new(0);

/// Record one operand read AND mark its surface as fired. This is the
/// entry point production sites should call: it makes the byte counters
/// and the coverage evidence impossible to update independently.
#[inline]
pub fn record(surface: Surface, m: super::bytes::OperandMovement) {
    surface.counter().fetch_add(1, Ordering::Relaxed);
    super::bytes::record(m);
}

/// A surface's reportable state in the current run.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SurfaceState {
    /// Instrumented and observed moving bytes this run.
    Covered(u64),
    /// Instrumented but never fired — either the path did not run, or
    /// the bump site is unreachable. Not the same as covered.
    Silent,
    /// No bump site exists. Its bytes are missing from every total.
    NotInstrumented,
}

/// Read every surface's state.
pub fn states() -> Vec<(Surface, SurfaceState)> {
    ALL_SURFACES
        .iter()
        .map(|&s| {
            let state = if !s.is_instrumented() {
                SurfaceState::NotInstrumented
            } else {
                match s.fired() {
                    0 => SurfaceState::Silent,
                    n => SurfaceState::Covered(n),
                }
            };
            (s, state)
        })
        .collect()
}

/// True when every surface is instrumented and fired — the only state in
/// which the ledger's physical total is the whole token's weight traffic.
pub fn is_complete() -> bool {
    states()
        .iter()
        .all(|(_, st)| matches!(st, SurfaceState::Covered(_)))
}

/// One line naming what is covered, what is silent, and what is missing.
pub fn render() -> String {
    let (mut covered, mut silent, mut missing) = (Vec::new(), Vec::new(), Vec::new());
    for (s, st) in states() {
        match st {
            SurfaceState::Covered(_) => covered.push(s.label()),
            SurfaceState::Silent => silent.push(s.label()),
            SurfaceState::NotInstrumented => missing.push(s.label()),
        }
    }
    let mut out = format!(
        "[bw10/coverage] covered: {}",
        if covered.is_empty() {
            "none".to_string()
        } else {
            covered.join(", ")
        }
    );
    if !silent.is_empty() {
        out.push_str(&format!(
            "  | instrumented but SILENT: {}",
            silent.join(", ")
        ));
    }
    if !missing.is_empty() {
        out.push_str(&format!("  | NOT instrumented: {}", missing.join(", ")));
    }
    if !is_complete() {
        out.push_str(
            "\n[bw10/coverage] PARTIAL — physical totals and every share derived from them are \
             UNDERSTATED. Byte DELTAS between two arms remain valid where the change is confined \
             to a covered surface.",
        );
    }
    out
}

#[cfg(test)]
pub(crate) fn reset_for_test() {
    for s in ALL_SURFACES {
        s.counter().store(0, Ordering::Relaxed);
    }
}

#[cfg(test)]
#[path = "tests/coverage.rs"]
mod tests;
