//! **Whether a number is a measurement of the thing it is named after.**
//!
//! A percentile is only measurable at a scale that supplies enough
//! observations in its tail. Nearest-rank p99 of *n* values is the
//! LARGEST value whenever *n* < 100 — so a "p99" over ninety
//! observations is a maximum wearing a percentile's name, and a search
//! that prices a candidate on it is pricing a single event.
//!
//! ```text
//! tail support = observations x (1 - quantile)
//!
//! p99 over    100 observations ->   1 expected tail observation
//! p99 over    500 observations ->   5
//! p99 over  1,000 observations ->  10
//! ```
//!
//! This was found the expensive way. ROUTE-CAL-1 (2026-08-31): a
//! 256-position diagnostic bank observes 1-105 route-change events, so
//! its `route_mass_p99` was the maximum in twenty-four of twenty-six
//! reports, and repeated values across unrelated maps turned out to be
//! the SAME worst event recurring rather than candidates agreeing. The
//! consequence was not noise, it was a sign error: the diagnostic
//! reported the most expensive route move in the programme as free.
//!
//! The abstraction is not about routing. Any p95, p99 or p999 criterion
//! screened at reduced scale has this problem, and the point of putting
//! it here is that a future criterion inherits the protection without
//! anyone remembering to ask.
//!
//! **A threshold is a policy, not a fact.** [`TailSupportPolicy`]
//! carries its own provenance so that a number refused for thin support
//! can be traced to the decision that refused it, and so that
//! tightening it later is a visible change rather than an edited
//! constant.

use serde::{Deserialize, Serialize};

/// Whether an assessment rests on a diagnostic screen or on
/// authority-scale evidence.
///
/// Lives here rather than beside the assessment because it is a
/// property of the EVIDENCE: a scale is the thing a statistic's support
/// is judged at, and the calibration registry has to key on it without
/// depending on anything that consumes it.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum EvidenceScale {
    /// A short bank, used to rank and to search.
    Diagnostic,
    /// A full bank at the contract's own position count.
    Authority,
}

/// Whether a reported statistic is supported by its evidence.
///
/// The three states are distinct in the way that matters to a search:
/// a number that was never observed, a number observed too thinly to
/// mean what it is called, and a number that stands.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum MeasurementStatus {
    /// The evidence supports the statistic.
    Measured,
    /// The statistic was computed, but too few observations fall in its
    /// tail for it to be the quantity it is named after.
    InsufficientTailSupport { observations: u64, required: u64 },
    /// Nothing was observed at all.
    NotObserved,
}

impl MeasurementStatus {
    /// Whether this number may be used to PRICE a candidate.
    ///
    /// Deliberately not called `is_ok`: a thinly supported percentile is
    /// not a small cost, it is an unknown one, and the difference is
    /// what stops a search preferring candidates whose expensive
    /// dimension happens to be unmeasured.
    pub fn is_priceable(&self) -> bool {
        matches!(self, Self::Measured)
    }
}

/// How many observations a percentile has behind its tail.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct TailSupport {
    /// e.g. `0.99`.
    pub quantile: f64,
    /// Observations the statistic was computed over — for a criterion
    /// measured only where something CHANGED, that is the number of
    /// changes, not the number of positions.
    pub observations: u64,
}

impl TailSupport {
    /// Observations expected to fall above the quantile.
    ///
    /// The quantity that decides whether a percentile is a percentile:
    /// below one, it is the maximum; below a handful, it is one or two
    /// events with a percentile's name on it.
    pub fn expected_tail_observations(&self) -> f64 {
        self.observations as f64 * (1.0 - self.quantile)
    }
}

/// **How much tail a percentile needs before it may price a candidate.**
///
/// A named, provenance-carrying policy rather than a constant, because
/// the number is a judgement about how much sparsity a search will
/// tolerate and not a law. Recording where it came from means a refusal
/// can be argued with.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct TailSupportPolicy {
    /// Expected tail observations required.
    pub min_tail_observations: f64,
    /// Why this number.
    pub provenance: String,
}

impl TailSupportPolicy {
    /// **The first policy: five expected tail observations.**
    ///
    /// Not a statistical law. Nearest-rank p99 stops literally being
    /// the maximum at a hundred observations, but one observation
    /// defining a tail is not a tail — it is technically a percentile
    /// and useless for ranking. Five makes the semantics honest: below
    /// this, the estimate is too sparsely supported to price a
    /// candidate, and the assessment says so rather than reporting a
    /// small number.
    ///
    /// For p99 that is roughly five hundred relevant observations.
    pub fn route_cal_1() -> Self {
        Self {
            min_tail_observations: 5.0,
            provenance: "ROUTE-CAL-1 (2026-08-31): a 256-position diagnostic observed 1-105 \
                         route-change events and reported the most expensive route move in the \
                         programme as free"
                .into(),
        }
    }

    /// Observations needed for `quantile` under this policy.
    pub fn required_observations(&self, quantile: f64) -> u64 {
        let tail = 1.0 - quantile;
        if tail <= 0.0 {
            return u64::MAX;
        }
        (self.min_tail_observations / tail).ceil() as u64
    }

    /// Judge one reported percentile.
    pub fn status(&self, support: Option<TailSupport>) -> MeasurementStatus {
        let Some(s) = support else {
            return MeasurementStatus::NotObserved;
        };
        if s.expected_tail_observations() >= self.min_tail_observations {
            MeasurementStatus::Measured
        } else {
            MeasurementStatus::InsufficientTailSupport {
                observations: s.observations,
                required: self.required_observations(s.quantile),
            }
        }
    }
}

#[cfg(test)]
#[path = "measurement_tests.rs"]
mod tests;
