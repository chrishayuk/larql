//! Token index within the executing phase — the other half of a policy's
//! address space, beside the layer index.
//!
//! # Why per-phase, and why not one global counter
//!
//! A single "tokens seen" counter would have exactly the defect
//! [`crate::movement_ledger::phase`] was built to fix: for a routed MoE
//! model this project's GPU prefill walks the prompt position-by-position
//! through the SAME entry point real decode steps use, so
//! `gpt-oss-20b`'s chat template alone contributes ~130 boundary
//! crossings before decode step 0. A policy written as "skip on token 7"
//! against a global counter would fire deep inside the system prompt and
//! the run would look like a null result.
//!
//! So the counter is bucketed by the phase declared at the boundary, and
//! a boundary crossed with NO phase scope active advances nothing:
//! [`current`] then keeps returning the previous phase's index rather
//! than inventing one. A step-selective policy sees `None` before the
//! first attributed boundary and must refuse — same contract as the
//! phase module's own.
//!
//! # Multi-arm harnesses
//!
//! [`reset`] exists for the same reason
//! [`crate::movement_ledger::session::reset`] does: a harness running two
//! arms in one process must be able to restart the address space, or arm
//! B's "step 0" lands wherever arm A stopped.

use std::sync::atomic::{AtomicU64, Ordering};

use crate::movement_ledger::phase::{current_phase, Phase};

/// Boundaries crossed with [`Phase::Decode`] active.
static DECODE_STEPS: AtomicU64 = AtomicU64::new(0);
/// Boundaries crossed with [`Phase::Prefill`] active.
static PREFILL_STEPS: AtomicU64 = AtomicU64::new(0);

fn counter(phase: Phase) -> &'static AtomicU64 {
    match phase {
        Phase::Decode => &DECODE_STEPS,
        Phase::Prefill => &PREFILL_STEPS,
    }
}

/// Advance the executing phase's token index. Call once per token, at the
/// same boundary that opens a [`crate::movement_ledger::TokenScope`], so
/// the ledger's window and the policy's address agree by construction.
///
/// A no-op when no phase scope is active — an unattributed boundary
/// advances nothing rather than corrupting either phase's index.
pub fn advance() {
    if let Some(p) = current_phase() {
        counter(p).fetch_add(1, Ordering::Relaxed);
    }
}

/// Zero-based token index within the executing phase.
///
/// `None` when no phase is declared, or when this phase has not yet
/// crossed a boundary — "no token declared", never a guessed `0`, since
/// `0` is a legitimate index a policy can select on.
pub fn current() -> Option<u64> {
    let n = counter(current_phase()?).load(Ordering::Relaxed);
    n.checked_sub(1)
}

/// Restart both phases' indices. For a harness running several arms in
/// one process; production never calls it.
pub fn reset() {
    DECODE_STEPS.store(0, Ordering::Relaxed);
    PREFILL_STEPS.store(0, Ordering::Relaxed);
}

#[cfg(test)]
#[path = "tests/step.rs"]
mod tests;
