//! Which generation phase produced a token record — prefill or decode.
//!
//! Prefill and decode are physically different operations, and this
//! project's own call graph does not keep them apart: for a routed MoE
//! model, `larql-inference`'s GPU prefill walks the prompt
//! position-by-position through the SAME entry point
//! (`MetalBackend::decode_token_with_moe_split_fn`) that real
//! autoregressive decode steps use. Without an explicit tag the ledger
//! cannot tell a 129-position system-prompt prefill from 129 real decode
//! steps — and it didn't: the BW-A live gate against gpt-oss-20b
//! recorded 130 tokens from a window that requested ZERO decode-loop
//! iterations (`-n 1 --warmup 0`, i.e. `for step in 1..1`, which never
//! runs). Every one of those 130 was a prefill position of the model's
//! chat-template system prompt, silently entering what should have been
//! a pure decode steady-state mean.
//!
//! # Why a scoped phase rather than a parameter
//!
//! The instrumented function is reached from many callers (bench,
//! `larql run`, `larql walk`, tests) through an already-large parameter
//! list. Threading a phase argument through every intermediate signature
//! to serve one diagnostic would be the tail wagging the dog — the same
//! reasoning [`crate::moe_route_observe::LayerScope`] used for layer
//! attribution. Instead the two boundaries that genuinely know the
//! phase — the prefill walk and the decode loop, both inside
//! `larql-inference`'s shared `layer_graph::generate` — install a
//! [`PhaseScope`] for their duration, and the ledger reads it.
//!
//! # Refusal, not attribution by guess
//!
//! A token recorded with no scope active is neither prefill nor decode
//! by assumption — [`current_phase`] returns `None`, [`super::TokenScope`]
//! carries that through as `phase: None`, and [`super::SteadyState`]
//! reports it as unattributed rather than silently folding it into
//! either bucket. Defaulting an unscoped call to `Decode` would have
//! hidden exactly the defect this module fixes; defaulting it to
//! `Prefill` would hide the opposite one.

use std::cell::Cell;

thread_local! {
    static CURRENT_PHASE: Cell<Option<Phase>> = const { Cell::new(None) };
}

/// Which generation phase is executing on this thread.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Phase {
    /// Walking the prompt — one prompt position, for a per-position MoE
    /// walk, or the whole prompt for a batched path — before any token
    /// has been sampled.
    Prefill,
    /// One autoregressive step.
    Decode,
}

impl Phase {
    /// The short label the report prints inside the per-token tag.
    pub const fn label(self) -> &'static str {
        match self {
            Phase::Prefill => "prefill",
            Phase::Decode => "decode",
        }
    }
}

/// Marks the executing phase for as long as it is held.
///
/// Install one around the prefill walk and one around the decode loop —
/// the two places in `layer_graph::generate` that genuinely know which
/// is running. Nested scopes restore the outer value on drop, so a
/// nested call cannot leave a stale attribution behind for whatever runs
/// next on this thread.
pub struct PhaseScope {
    previous: Option<Phase>,
}

impl PhaseScope {
    pub fn new(phase: Phase) -> Self {
        let previous = CURRENT_PHASE.with(|c| c.replace(Some(phase)));
        Self { previous }
    }
}

impl Drop for PhaseScope {
    fn drop(&mut self) {
        CURRENT_PHASE.with(|c| c.set(self.previous));
    }
}

/// The phase this thread is executing, if any driver loop declared one.
pub fn current_phase() -> Option<Phase> {
    CURRENT_PHASE.with(|c| c.get())
}

#[cfg(test)]
pub(crate) fn reset_for_test() {
    CURRENT_PHASE.with(|c| c.set(None));
}

#[cfg(test)]
#[path = "tests/phase.rs"]
mod tests;
