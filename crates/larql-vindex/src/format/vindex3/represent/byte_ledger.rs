//! **What a representation map costs to read, per token.**
//!
//! The intrinsic half of the economics, and the only half that is a
//! property of the model rather than of a machine: how many bytes the
//! decoder reads for one token under a baseline representation, and how
//! many under a candidate. No hardware, no backend, no coefficient —
//! those belong to [`super::execution_cost`], which is where the
//! provenance lives.
//!
//! Keeping the two apart is the point. A byte count is a fact that
//! survives a new machine; a byte count multiplied by a measured
//! conversion factor is an observation that does not, and code that
//! cannot tell them apart will eventually quote the second as if it
//! were the first.
//!
//! ```text
//! routed experts (8 of 256 x 26 layers)   2.944 GB   49.2 %
//! KDA projections (20 layers)             1.510 GB   25.2 %
//! output head                             0.755 GB   12.6 %
//! MLA projections (7 layers)              0.408 GB    6.8 %
//! shared experts (26 layers)              0.368 GB    6.1 %
//! ```
//!
//! That is Kimi-Linear-48B-A3B at BF16 — the ledger which showed that a
//! checkpoint 97 % experts by size is only 49 % experts by TOKEN, and
//! so that a whole-decoder map beats an expert-only one.
//!
//! **Scopes are supplied, not derived.** A ledger is a list of the
//! scopes the decoder actually reads and what each costs under both
//! representations. It does not attempt to infer that from geometry,
//! because "which layers are MLA" and "how many experts are routed" are
//! model-shape questions whose wrong answer would be silent, and a
//! wrong byte count propagates into every prediction downstream.

use std::collections::BTreeSet;

use serde::{Deserialize, Serialize};

/// One scope's contribution to the per-token read, under both
/// representations.
///
/// `candidate_bytes == baseline_bytes` is the normal case: most of a
/// map is unchanged, and a ledger that listed only the changed scopes
/// could not compute what fraction of the whole was removed.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ScopeBytes {
    /// The scope itself, e.g. `"routed experts L20-26"`.
    pub scope: String,
    /// The byte family it belongs to, e.g. `"routed experts"`. Breadth
    /// is counted in families as well as scopes, because taking four
    /// layers of one family is a different kind of change from taking
    /// one layer of each of four.
    pub family: String,
    pub baseline_bytes: u64,
    pub candidate_bytes: u64,
}

impl ScopeBytes {
    /// Bytes this scope no longer reads. Saturating, so a candidate
    /// that grew reports zero removed rather than an underflow.
    pub fn removed(&self) -> u64 {
        self.baseline_bytes.saturating_sub(self.candidate_bytes)
    }

    /// Whether this scope's representation changed at all.
    pub fn changed(&self) -> bool {
        self.candidate_bytes != self.baseline_bytes
    }
}

/// The per-token read of a whole decoder, under a baseline
/// representation and a candidate one.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ByteLedger {
    /// Which model this is a ledger for — a prediction may not be
    /// carried from one model to another, and this is what lets that be
    /// checked rather than assumed.
    pub model: String,
    /// Names of the two representations, for the record.
    pub baseline_representation: String,
    pub candidate_representation: String,
    /// EVERY scope the decoder reads per token, changed or not.
    pub scopes: Vec<ScopeBytes>,
}

impl ByteLedger {
    pub fn baseline_bytes_per_token(&self) -> u64 {
        self.scopes.iter().map(|s| s.baseline_bytes).sum()
    }

    pub fn candidate_bytes_per_token(&self) -> u64 {
        self.scopes.iter().map(|s| s.candidate_bytes).sum()
    }

    pub fn bytes_removed(&self) -> u64 {
        self.baseline_bytes_per_token()
            .saturating_sub(self.candidate_bytes_per_token())
    }

    /// Removed bytes as a fraction of the baseline read — the quantity
    /// a throughput prediction is a function of.
    ///
    /// Zero for an empty ledger rather than `NaN`: nothing read is
    /// nothing saved, and a `NaN` here would propagate silently.
    pub fn fraction_removed(&self) -> f64 {
        let baseline = self.baseline_bytes_per_token();
        if baseline == 0 {
            return 0.0;
        }
        self.bytes_removed() as f64 / baseline as f64
    }

    /// **Breadth, in families.** Distinct families with at least one
    /// changed scope.
    pub fn families_changed(&self) -> BTreeSet<&str> {
        self.scopes
            .iter()
            .filter(|s| s.changed())
            .map(|s| s.family.as_str())
            .collect()
    }

    /// **Breadth, in scopes.** How many individual scopes moved.
    pub fn scopes_changed(&self) -> usize {
        self.scopes.iter().filter(|s| s.changed()).count()
    }

    /// What each family contributes to the baseline read, largest
    /// first — the table that says where the bytes actually are, as
    /// opposed to where the checkpoint's size is.
    pub fn baseline_by_family(&self) -> Vec<(&str, u64)> {
        let mut totals: std::collections::BTreeMap<&str, u64> = std::collections::BTreeMap::new();
        for s in &self.scopes {
            *totals.entry(s.family.as_str()).or_default() += s.baseline_bytes;
        }
        let mut rows: Vec<(&str, u64)> = totals.into_iter().collect();
        rows.sort_by(|a, b| b.1.cmp(&a.1).then_with(|| a.0.cmp(b.0)));
        rows
    }
}

#[cfg(test)]
#[path = "byte_ledger_tests.rs"]
// Visible to `execution_cost`'s tests: they share ONE measured Kimi
// ledger, so the two modules cannot drift apart on what the map costs.
pub(super) mod tests;
