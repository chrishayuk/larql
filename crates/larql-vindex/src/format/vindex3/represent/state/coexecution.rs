//! **Measuring several already-identified experiments together.**
//!
//! Eight representation candidates traversed a 45-layer MoE and expanded
//! the materialised expert set by only **1.08x** against a single
//! candidate, flat across all 42 sparse layers. Scientifically that was
//! eight experiments; physically it was one traversal. This module is
//! that separation and nothing else.
//!
//! ## Frozen scope
//!
//! > **It batches EXECUTION. It does not batch scientific identity, and
//! > it does not choose which experiments should exist.**
//!
//! Three things it deliberately does not do, each because doing it would
//! damage an abstraction that is already correct:
//!
//! - **No population measurement key.** [`MeasurementKey`] is one
//!   physical state + bank + scale + instrument, and its whole job is to
//!   say *this observation belongs to this state under these query
//!   conditions*. A key naming several states would make that sentence
//!   false. Members keep their own keys, unchanged, and the registry
//!   records N observations exactly as it would have.
//! - **No candidate selection.** Deciding that eight candidates should be
//!   measured *before* any of their results are incorporated is a change
//!   to SEARCH SEMANTICS, not an execution detail: adaptive best-first
//!   would have let evidence from A retire B unmeasured. Batch width and
//!   population policy belong in a registered search policy if evidence
//!   ever shows population search beats sequential. This module takes the
//!   set it is given.
//! - **No sharing assumption.** How much work a batch actually shared is
//!   recorded by [`SharingLedger`] as an observation. The 1.08x above is
//!   one model on one prompt; nothing here treats it as a constant.
//!
//! ## The falsifier
//!
//! > **If co-executing A and B changes either one's key, evidence, or
//! > registry semantics compared with executing them separately, the
//! > abstraction is wrong.**
//!
//! That is what `coexecution_tests` asserts, rather than asserting that
//! batching is faster.
//!
//! ## One boundary to hold when performance evidence arrives
//!
//! Batch context must not affect QUALITY identity — that is the property
//! above. It may well affect THROUGHPUT: an eight-member batch has
//! different dispatch and materialisation economics from an isolated
//! run, and a latency measured inside one is not a latency measured
//! alone.
//!
//! **That difference belongs to the physical/performance observation or
//! to realization qualification — never to [`MeasurementKey`].** Adding
//! batch width to the key would make two quality readings of one state
//! different experiments because of how they were scheduled, which is
//! exactly the collapse this module exists to prevent. Nothing here
//! records timing for that reason.

use std::collections::BTreeSet;

use super::key::MeasurementKey;

/// Why a set of requests cannot share one execution.
///
/// Co-execution is only sound when the members differ *in the state
/// being measured* and agree on everything describing the query. Two
/// requests over different banks are two different bodies of data and
/// share no traversal; batching them would either measure one against
/// the other's inputs or silently run twice under one name.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum BatchRefusal {
    /// Nothing to execute. Refused rather than returning an empty batch,
    /// because an empty batch that "succeeds" reads downstream as a
    /// completed measurement of nothing.
    Empty,
    MixedBank,
    MixedScale,
    MixedInstrument,
    /// The same state twice. Its observation would be recorded twice
    /// under one key, which the registry cannot distinguish from a
    /// repeated measurement.
    DuplicateState,
}

impl std::fmt::Display for BatchRefusal {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let s = match self {
            BatchRefusal::Empty => "no requests to co-execute",
            BatchRefusal::MixedBank => {
                "requests span different evidence banks; they share no traversal"
            }
            BatchRefusal::MixedScale => {
                "requests span diagnostic and authority scales; one run cannot be both"
            }
            BatchRefusal::MixedInstrument => {
                "requests span different instrument semantics; one run cannot answer both"
            }
            BatchRefusal::DuplicateState => {
                "the same representation state appears twice; its observation would be \
                 recorded twice under one key"
            }
        };
        f.write_str(s)
    }
}

/// A set of measurement requests that may share one physical execution.
///
/// Holds keys, never states: assembling a batch must be unable to alter
/// what any member is measuring.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ExecutionBatch {
    members: Vec<MeasurementKey>,
}

impl ExecutionBatch {
    /// Decide whether these already-authorised requests are physically
    /// co-executable. **Does not decide whether they should exist.**
    pub fn assemble(requests: Vec<MeasurementKey>) -> Result<Self, BatchRefusal> {
        let first = requests.first().ok_or(BatchRefusal::Empty)?;
        for r in &requests[1..] {
            if r.bank() != first.bank() {
                return Err(BatchRefusal::MixedBank);
            }
            if r.scale() != first.scale() {
                return Err(BatchRefusal::MixedScale);
            }
            if r.instrument() != first.instrument() {
                return Err(BatchRefusal::MixedInstrument);
            }
        }
        let distinct: BTreeSet<_> = requests.iter().map(|r| r.state()).collect();
        if distinct.len() != requests.len() {
            return Err(BatchRefusal::DuplicateState);
        }
        Ok(Self { members: requests })
    }

    /// The members, keys unchanged and in the order given.
    pub fn members(&self) -> &[MeasurementKey] {
        &self.members
    }

    pub fn len(&self) -> usize {
        self.members.len()
    }

    pub fn is_empty(&self) -> bool {
        self.members.is_empty()
    }
}

/// **What a batch actually shared** — measured, never assumed.
///
/// `units` is whatever the executor materialises and can count: routed
/// experts fetched, tensors bound, bytes read. The ledger does not know
/// which, on purpose, because the saving is the same shape whatever the
/// unit is.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SharingLedger {
    pub members: usize,
    /// Units each member would have materialised alone, summed.
    pub units_if_isolated: u64,
    /// Units the batch actually materialised.
    pub units_materialised: u64,
}

impl SharingLedger {
    /// Materialised units against what ONE member would have cost.
    ///
    /// This is the number worth reporting: 1.0 means the batch was free
    /// beyond a single member, `members` means it shared nothing. It is
    /// undefined for an empty batch and for a batch whose members
    /// materialise nothing, and returns `None` rather than a ratio in
    /// both cases.
    pub fn amplification_vs_one(&self) -> Option<f64> {
        if self.members == 0 || self.units_if_isolated == 0 {
            return None;
        }
        let per_member = self.units_if_isolated as f64 / self.members as f64;
        Some(self.units_materialised as f64 / per_member)
    }

    /// Fraction of isolated cost avoided.
    pub fn saving(&self) -> Option<f64> {
        if self.units_if_isolated == 0 {
            return None;
        }
        Some(1.0 - self.units_materialised as f64 / self.units_if_isolated as f64)
    }

    /// One line for a search trace.
    pub fn describe(&self) -> String {
        match (self.amplification_vs_one(), self.saving()) {
            (Some(a), Some(s)) => format!(
                "{} members: materialised {} of {} isolated units — {a:.2}x one member, \
                 {:.1}% saved",
                self.members,
                self.units_materialised,
                self.units_if_isolated,
                s * 100.0
            ),
            _ => format!("{} members: nothing materialised", self.members),
        }
    }
}

#[cfg(test)]
#[path = "coexecution_tests.rs"]
mod tests;
