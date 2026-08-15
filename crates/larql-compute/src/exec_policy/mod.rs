//! The execution-policy seam: a semantic operation exists, and the
//! runtime chooses how it is physically satisfied.
//!
//! ```text
//! router  →  "these experts conceptually participate"     (semantics)
//! seam    →  Canonical | Skip                             (physical strategy)
//! ```
//!
//! Those two questions have been entangled everywhere in this engine
//! until now: to not-run an expert you had to make the router not pick
//! it, which is a different perturbation with different numerics. This
//! module separates them, so that "which experts does this token need"
//! and "how do we actually satisfy that need this time" can be decided by
//! different code, measured separately, and changed independently.
//!
//! # Default is Canonical, and that is load-bearing
//!
//! With no policy installed [`decide_expert_group`] returns
//! [`ExecutionStrategy::Canonical`] after a single relaxed atomic load,
//! and production semantics are bit-identical to an engine built without
//! this module. Nothing here changes what the model computes unless a
//! caller explicitly installs a policy. Every dispatch site is written so
//! that the canonical branch is the one that would exist anyway.
//!
//! # Where it sits
//!
//! At the point immediately before the expensive expert kernels, on the
//! production Metal dispatch path — both of them:
//! `moe_gpu_route::encode`'s descriptor arm (the `LARQL_GPU_ROUTE=1`
//! serve path) and `moe_zero_copy`'s CPU-routed arm. Not in a research
//! harness: `cpu::ops::moe::expert_override` remains what it
//! always was — BW-C's per-`(layer, expert)` one-shot ablation
//! instrument, unchanged, so every BW-C1..C5 result stays reproducible.
//! The two are deliberately different tools with different units:
//!
//! | | `expert_override` | this module |
//! |---|---|---|
//! | unit | one `(layer, expert)` invocation | the whole routed group at a layer |
//! | lifetime | one-shot, disarms itself | standing policy |
//! | path | CPU `add_expert` | Metal expert dispatch |
//! | purpose | measure whether a deletion is safe | decide, in production, to delete |
//!
//! # Accounting
//!
//! [`resolve_expert_group`] is the single entry point, and it decides AND
//! records in one call for the same reason
//! [`crate::movement_ledger::coverage::record`] pairs the byte counters
//! with the coverage evidence: two counters a caller can update
//! independently will eventually disagree, and a disagreement between
//! "what the policy did" and "what the ledger saw" is indistinguishable
//! from a real byte delta. Executed groups record movement as before;
//! skipped groups record avoided bytes into
//! [`crate::movement_ledger::decisions`] and touch no byte counter.
//!
//! # The reserved third arm
//!
//! BW-B closed with a compiled compact-dense representation beating both
//! sparse-gather and dense across the whole tested range, which is a
//! third answer to the same question this seam asks: not "run it" or
//! "delete it" but "run it against a derived representation". That lands
//! here as a `CompactDense(..)` variant of [`ExecutionStrategy`] when it
//! has a real materialisation to point at. The enum is exhaustive
//! precisely so that adding it breaks every dispatch site until each one
//! decides what to do — see [`ExecutionStrategy`]'s own doc.

pub mod policies;
pub mod spec;
pub mod step;
pub mod strategy;
pub mod trace;

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, RwLock};

use crate::movement_ledger::{
    bytes::OperandMovement,
    coverage::{self, Surface},
    decisions,
    phase::current_phase,
};

pub use strategy::{ExecutionStrategy, ExpertGroupSite};

/// A runtime choice of physical execution strategy.
///
/// Implementors are consulted on the decode hot path, once per routed
/// layer per token, so a decision must be cheap and must not allocate.
/// They are shared across threads by [`install`] and must be `Sync`.
pub trait ExecutionPolicy: Send + Sync {
    /// A stable name for reports. Printed beside the skip rate so a
    /// result can never be read without knowing what produced it.
    fn name(&self) -> &str;

    /// How to satisfy the routed expert group described by `site`.
    ///
    /// The default is [`ExecutionStrategy::Canonical`], so a policy that
    /// only cares about some other operation class does not have to
    /// opt back into correctness for this one.
    fn expert_group(&self, site: &ExpertGroupSite) -> ExecutionStrategy {
        let _ = site;
        ExecutionStrategy::Canonical
    }
}

/// Fast gate so the disarmed hot path never takes the lock. Flipped
/// AFTER the policy is stored on install, and BEFORE it is cleared on
/// uninstall, so a reader that sees `true` always finds a policy and a
/// reader that sees `false` never needs one.
static INSTALLED: AtomicBool = AtomicBool::new(false);
static POLICY: RwLock<Option<Arc<dyn ExecutionPolicy>>> = RwLock::new(None);

/// Install `policy` process-wide, replacing any previous one, and return
/// a guard that uninstalls on drop.
///
/// The guard is the intended form: a policy left installed by a test or a
/// harness arm silently changes every later arm's numerics, which is the
/// failure mode this whole module exists to make visible rather than
/// commit.
#[must_use = "the policy is uninstalled when the guard drops"]
pub fn install(policy: Arc<dyn ExecutionPolicy>) -> PolicyGuard {
    *POLICY.write().unwrap_or_else(|e| e.into_inner()) = Some(policy);
    INSTALLED.store(true, Ordering::Relaxed);
    PolicyGuard { _private: () }
}

/// Remove any installed policy, restoring canonical execution.
pub fn uninstall() {
    INSTALLED.store(false, Ordering::Relaxed);
    *POLICY.write().unwrap_or_else(|e| e.into_inner()) = None;
}

/// Uninstalls the policy it was returned from when dropped.
pub struct PolicyGuard {
    _private: (),
}

impl Drop for PolicyGuard {
    fn drop(&mut self) {
        uninstall();
    }
}

/// The installed policy's name, for reports. `None` when execution is
/// canonical — which a report must state, since "0 skips" under a policy
/// and "0 skips" with no policy are different facts.
pub fn installed_name() -> Option<String> {
    if !INSTALLED.load(Ordering::Relaxed) {
        return None;
    }
    POLICY
        .read()
        .unwrap_or_else(|e| e.into_inner())
        .as_ref()
        .map(|p| p.name().to_string())
}

/// Ask the installed policy how to satisfy the routed expert group at
/// `layer`, which would dispatch `slots` expert slots.
///
/// Pure — it records nothing. Callers on a dispatch path want
/// [`resolve_expert_group`] instead, which is the same decision paired
/// with its accounting.
pub fn decide_expert_group(layer: usize, slots: usize) -> ExecutionStrategy {
    if !INSTALLED.load(Ordering::Relaxed) {
        return ExecutionStrategy::Canonical;
    }
    let site = ExpertGroupSite {
        layer,
        phase: current_phase(),
        step: step::current(),
        slots,
    };
    let guard = POLICY.read().unwrap_or_else(|e| e.into_inner());
    match guard.as_ref() {
        Some(p) => p.expert_group(&site),
        None => ExecutionStrategy::Canonical,
    }
}

/// Decide how to satisfy the routed expert group at `layer`, AND record
/// the decision against the BW10 ledger. The one entry point a dispatch
/// path should use.
///
/// `movement` is what this group WOULD move if executed, computed by the
/// backend's own shape arithmetic. On [`ExecutionStrategy::Canonical`] it
/// is recorded as real traffic exactly as before; on
/// [`ExecutionStrategy::Skip`] it is recorded as avoided, and no byte
/// counter moves. Both arms therefore price the same operation with the
/// same arithmetic, which is what makes `touched + avoided` a
/// reconstruction of the canonical arm rather than two estimates.
///
/// The caller must honour the returned strategy. That is the one part
/// this function cannot enforce — a backend that ignores a `Skip` would
/// report avoided bytes it actually moved. The gate test
/// (`test_exec_policy_expert_skip`) exists to hold each backend to it.
pub fn resolve_expert_group(
    layer: usize,
    slots: usize,
    movement: OperandMovement,
) -> ExecutionStrategy {
    let strategy = decide_expert_group(layer, slots);
    match strategy {
        ExecutionStrategy::Canonical => {
            decisions::record_executed();
            coverage::record(Surface::MoeExperts, movement);
        }
        ExecutionStrategy::Skip => decisions::record_skipped(&movement),
    }
    strategy
}

#[cfg(test)]
#[path = "tests/mod_api.rs"]
mod tests;
