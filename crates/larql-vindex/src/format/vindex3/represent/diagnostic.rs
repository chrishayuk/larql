//! **BS2-F1 — what a search may OBSERVE, stated independently of what a
//! contract ADMITS.**
//!
//! The real-bank smoke test found `route flip rate` — the one statistic
//! ROUTE-CAL-1 established as transferring across scale — absent from
//! the diagnostic vector entirely. The cause was structural, not a bug
//! in any one function:
//!
//! ```text
//! before   diagnostic dimensions := statistics the CONTRACT bounds
//!          so anything the contract does not judge is invisible
//!
//! after    diagnostic dimensions := an explicit DiagnosticPolicy
//! ```
//!
//! The tempting repair — give `balanced-v1` a route-flip limit so the
//! search can see it — is the one that must be refused. It would turn an
//! INSTRUMENTATION requirement into behavioural policy, and freeze into
//! a contract a number that was only ever earned as an ordering proxy.
//!
//! # The two invariants
//!
//! **A diagnostic statistic does not become an authority criterion
//! merely by being observed or predictive.** [`DiagnosticReading`]
//! therefore has no `limit`, no `utilisation` and no `headroom`: there
//! is no bound to divide by, so a price cannot be computed here even by
//! mistake. That is a type boundary, not a discipline.
//!
//! **An authority criterion need not be directly usable at diagnostic
//! scale.** `routed mixture moved at p99` is observed by both, and at
//! 256 positions its 46 events make it `Unusable` — the contract still
//! judges it at authority scale, unchanged.
//!
//! The relation between the two schemas is many-to-many, and neither
//! side owns the other:
//!
//! ```text
//! diagnostic kl p99             ──→ authority kl p99
//! diagnostic route flip rate    ──→ authority routed mixture mass
//! diagnostic routed mixture p99 ──→ authority routed mixture mass
//!                                   (but Unusable at this scale)
//! ```
//!
//! Which authority behaviour a proxy informs is recorded where it is
//! used, on [`super::promotion::ProxyObservation::for_criterion`], not
//! here — a policy says what to look at, not what to conclude.

use serde::{Deserialize, Serialize};

use super::measurement::{MeasurementStatus, TailSupport, TailSupportPolicy};
use super::quality::{QualityBank, Statistic};
use super::search_evidence::{SearchCalibrationRegistry, SearchEvidence};

/// Why a statistic is observed at diagnostic scale.
///
/// Records the RELATIONSHIP to the contract without granting one: a
/// `SearchEvidence` observation is not a criterion in waiting.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ObservationPurpose {
    /// The contract judges this same statistic; reading it early says
    /// where the candidate stands, subject to the registry's verdict on
    /// whether this scale supports it.
    ContractPreview,
    /// Observed ONLY to order candidates. The contract does not judge
    /// it and must not begin to because it proved useful.
    SearchEvidence,
}

/// One statistic a policy asks to be observed. **No bound** — that is
/// the whole point.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct DiagnosticObservation {
    pub statistic: Statistic,
    pub purpose: ObservationPurpose,
}

/// What a search observes, declared independently of any contract.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct DiagnosticPolicy {
    pub id: String,
    pub observations: Vec<DiagnosticObservation>,
}

impl DiagnosticPolicy {
    pub fn observes(&self, statistic: Statistic) -> bool {
        self.observations.iter().any(|o| o.statistic == statistic)
    }

    /// **`bs2-kimi-v1`** — what rung 4 needs to see on a Kimi bank.
    ///
    /// Seeded with the contract's own magnitudes as previews, plus the
    /// one observation that exists ONLY here: `route flip rate`, a count
    /// statistic whose flips-per-position transfer at 0.89-1.02 for any
    /// map with at least forty diagnostic events, where the same bank's
    /// mixture-mass p99 rests on 46 events and is wrong on two of five
    /// moves.
    ///
    /// Adding to this list is cheap and reversible BY DESIGN: it cannot
    /// change what `balanced-v1` admits, so a new observable never
    /// re-opens a frozen contract.
    pub fn bs2_kimi_v1() -> Self {
        use ObservationPurpose::{ContractPreview, SearchEvidence};
        Self {
            id: "bs2-kimi-v1".into(),
            observations: vec![
                DiagnosticObservation {
                    statistic: Statistic::KlP99,
                    purpose: ContractPreview,
                },
                DiagnosticObservation {
                    statistic: Statistic::Top1MassDisplaced,
                    purpose: ContractPreview,
                },
                DiagnosticObservation {
                    statistic: Statistic::Top10MassDisplacedP99,
                    purpose: ContractPreview,
                },
                DiagnosticObservation {
                    statistic: Statistic::RouteMixtureMassP99,
                    purpose: ContractPreview,
                },
                DiagnosticObservation {
                    statistic: Statistic::RouteFlipRate,
                    purpose: SearchEvidence,
                },
            ],
        }
    }
}

/// One statistic as a bank actually reports it.
///
/// **Deliberately without a bound.** A reading cannot be turned into a
/// fraction of a budget because there is no budget here to divide by —
/// see this module's first invariant.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct DiagnosticReading {
    pub statistic: Statistic,
    pub purpose: ObservationPurpose,
    /// `None` when the bank did not record it. Not zero: an unobserved
    /// quantity is unobserved.
    pub observed: Option<f64>,
    pub tail_support: Option<TailSupport>,
}

impl DiagnosticReading {
    /// Whether the evidence supports the statistic this reading reports.
    pub fn measurement_status(&self, policy: &TailSupportPolicy) -> MeasurementStatus {
        policy.status_of(self.observed.is_some(), self.tail_support)
    }

    /// How a search may USE this reading — which is never as a price
    /// unless the registry says the scale earns it.
    pub fn evidence(
        &self,
        registry: &SearchCalibrationRegistry,
        policy: &TailSupportPolicy,
    ) -> SearchEvidence {
        registry.evidence_for(
            self.statistic,
            super::measurement::EvidenceScale::Diagnostic,
            &self.measurement_status(policy),
        )
    }
}

/// Everything a policy asked to see, as one bank reports it.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct DiagnosticVector {
    pub policy_id: String,
    pub readings: Vec<DiagnosticReading>,
}

impl DiagnosticVector {
    /// Read `bank` through `policy`. **No gate is consulted**, which is
    /// the coupling BS2-F1 severed.
    pub fn of(policy: &DiagnosticPolicy, bank: &QualityBank) -> Self {
        Self {
            policy_id: policy.id.clone(),
            readings: policy
                .observations
                .iter()
                .map(|o| {
                    let (observed, tail_support) = o.statistic.observe(bank);
                    DiagnosticReading {
                        statistic: o.statistic,
                        purpose: o.purpose,
                        observed,
                        tail_support,
                    }
                })
                .collect(),
        }
    }

    pub fn reading(&self, statistic: Statistic) -> Option<&DiagnosticReading> {
        self.readings.iter().find(|r| r.statistic == statistic)
    }

    /// The readings a search may order on, with the evidence that says
    /// so. Excludes anything the registry judges `Unusable` at this
    /// scale.
    pub fn ordering(
        &self,
        registry: &SearchCalibrationRegistry,
        policy: &TailSupportPolicy,
    ) -> Vec<(&DiagnosticReading, SearchEvidence)> {
        self.readings
            .iter()
            .map(|r| (r, r.evidence(registry, policy)))
            .filter(|(_, e)| e.orders())
            .collect()
    }
}

#[cfg(test)]
#[path = "diagnostic_tests.rs"]
// `pub(crate)` so the promotion tests can share this module's 256-position
// bank rather than keeping a second copy of it that could drift.
pub(crate) mod tests;
