//! **How a search may use a statistic, which is not the same question
//! as whether that statistic was measured.**
//!
//! [`super::measurement`] answers the first question and answers it
//! strictly: a p99 over forty-six observations is
//! `InsufficientTailSupport`, and no amount of correlation makes the
//! missing tail observations appear. This module answers the second.
//!
//! ```text
//! the STATISTIC decides whether it was measured
//! the EVIDENCE decides whether an imperfect statistic is still useful
//! ```
//!
//! Those are orthogonal, and collapsing them is a specific trap:
//!
//! ```text
//! thin tail + good rank correlation  ->  "Measured, provisionally"   WRONG
//! ```
//!
//! That conflates predictive utility with measurement adequacy.
//! Diagnostic `kl p99` correlates with authority ordering at rho
//! +0.857 over seven paired maps, which is good evidence for *"A or B
//! first?"*. It is NOT evidence that a diagnostic reading of 2.4e-3
//! means authority will be near 2.4e-3 — and that is precisely what
//! pricing a candidate against a 3.5e-3 budget would assume. The same
//! programme measured promotion drift of +84 % on one map and -3.9 %
//! on another.
//!
//! So the ladder has four rungs, and only the top two may produce a
//! number a contract is priced against:
//!
//! ```text
//! Direct              enough support to price against the contract
//! CalibratedEstimate  magnitude transfer demonstrated across breadths
//! OrderingProxy       useful for ordering; magnitude not trusted
//! Unusable            no demonstrated search value
//! ```
//!
//! **`Unscorable` therefore stops meaning "the search knows nothing".**
//! It means this authority constraint cannot be priced numerically at
//! this evidence scale — while the search may still hold calibrated
//! proxy evidence about which candidate to measure next.

use serde::{Deserialize, Serialize};

use super::measurement::{EvidenceScale, MeasurementStatus};
use super::quality::Statistic;

/// How a search may use one statistic at one evidence scale.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum SearchEvidence {
    /// Directly supported: may be priced against the contract.
    Direct,
    /// Thinly supported, but magnitude transfer has been demonstrated
    /// across breadths. May be priced, provisionally, naming the
    /// calibration.
    CalibratedEstimate { calibration: String },
    /// Useful for ORDERING candidates. The magnitude is not trusted, so
    /// it may not be turned into a fraction of a remaining budget.
    OrderingProxy { calibration: String },
    /// No demonstrated search value.
    Unusable,
}

impl SearchEvidence {
    /// Whether a number from this evidence may be priced against the
    /// contract — turned into a fraction of remaining headroom.
    pub fn is_priceable(&self) -> bool {
        matches!(self, Self::Direct | Self::CalibratedEstimate { .. })
    }

    /// Whether it may be used to order candidates, which is a weaker
    /// claim and true of everything above `Unusable`.
    pub fn orders(&self) -> bool {
        !matches!(self, Self::Unusable)
    }

    /// Position on the ladder, higher being stronger. Exists so that a
    /// search cannot accidentally prefer weaker evidence, and so the
    /// ladder's order is stated once rather than implied by variant
    /// declaration order.
    pub fn confidence_rank(&self) -> u8 {
        match self {
            Self::Direct => 3,
            Self::CalibratedEstimate { .. } => 2,
            Self::OrderingProxy { .. } => 1,
            Self::Unusable => 0,
        }
    }
}

/// One recorded finding about how a statistic behaves at a scale.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SearchCalibration {
    /// The calibration this came from, e.g. `"ROUTE-CAL-1"`.
    pub id: String,
    /// The statistic, in the gate's own wording where it is a contract
    /// criterion — `"kl p99"`, `"routed mixture moved at p99"` — or its
    /// own name where it is a proxy, `"route flip rate"`.
    pub statistic: Statistic,
    pub scale: EvidenceScale,
    pub verdict: SearchEvidence,
    /// Paired diagnostic/authority observations behind the verdict.
    pub pairs: u32,
    /// Rank correlation, where one was computed.
    pub rank_correlation: Option<f64>,
    /// What was found, in a sentence a reader can disagree with.
    pub finding: String,
}

/// **The registry: what this programme has learned about its own
/// instruments.**
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct SearchCalibrationRegistry {
    pub entries: Vec<SearchCalibration>,
}

impl SearchCalibrationRegistry {
    pub fn new(entries: Vec<SearchCalibration>) -> Self {
        Self { entries }
    }

    pub fn lookup(&self, statistic: Statistic, scale: EvidenceScale) -> Option<&SearchCalibration> {
        self.entries
            .iter()
            .find(|e| e.statistic == statistic && e.scale == scale)
    }

    /// **The bridge.** What a search may do with `statistic` at `scale`,
    /// given how well the evidence supports it.
    ///
    /// A registration WINS where one exists, in both directions: it can
    /// raise a thin measurement to an ordering proxy, and it can lower
    /// a perfectly well measured one — route flip rate is a sound count
    /// statistic that is still only a PROXY, because the contract judges
    /// mixture mass and not flips.
    ///
    /// Without a registration the measurement decides, and an
    /// unregistered thin percentile is `Unusable` rather than usable —
    /// the failure this whole layer exists to prevent is a search
    /// preferring a candidate because its expensive dimension happened
    /// to be unmeasured.
    ///
    /// **At [`EvidenceScale::Diagnostic`] the fall-through never reaches
    /// `Direct`.** Being well measured at a small scale does not make a
    /// number transferable, and the previous default said otherwise: a
    /// COUNT is always `Measured` — counts have no tail to be thin — so
    /// an unregistered bounded count came back directly priceable. It
    /// would then be spent against a budget written for a different
    /// sample size, and a count SCALES with positions: 46 route flips
    /// over 256 positions against a bound set for 8,192 is not 72 % of
    /// anything. Magnitude transfer at diagnostic scale is a claim that
    /// has to be EARNED by a calibration, which is what the registry
    /// records — so an unregistered statistic is `Unusable` here, and
    /// the way to change that is to measure the transfer.
    pub fn evidence_for(
        &self,
        statistic: Statistic,
        scale: EvidenceScale,
        status: &MeasurementStatus,
    ) -> SearchEvidence {
        if let Some(e) = self.lookup(statistic, scale) {
            return e.verdict.clone();
        }
        match scale {
            EvidenceScale::Authority => {
                if status.is_priceable() {
                    SearchEvidence::Direct
                } else {
                    SearchEvidence::Unusable
                }
            }
            EvidenceScale::Diagnostic => SearchEvidence::Unusable,
        }
    }

    /// **What ROUTE-CAL-1 established**, as the registry a search starts
    /// from. Every entry names its evidence; none is an assumption.
    pub fn route_cal_1() -> Self {
        Self::new(vec![
            SearchCalibration {
                id: "ROUTE-CAL-1".into(),
                statistic: Statistic::KlP99,
                scale: EvidenceScale::Diagnostic,
                verdict: SearchEvidence::OrderingProxy {
                    calibration: "ROUTE-CAL-1".into(),
                },
                pairs: 7,
                rank_correlation: Some(0.857),
                finding: "diagnostic kl orders candidates well, but magnitude transfer is NOT \
                          established — the same programme measured promotion drift of +84% on a \
                          narrow map and -3.9% on a broad one, so a diagnostic reading may not be \
                          priced against the contract's budget"
                    .into(),
            },
            SearchCalibration {
                id: "ROUTE-CAL-1".into(),
                statistic: Statistic::RouteMixtureMassP99,
                scale: EvidenceScale::Diagnostic,
                verdict: SearchEvidence::Unusable,
                pairs: 7,
                rank_correlation: Some(0.606),
                finding: "a 256-position bank observes 1-105 route-change events, so its p99 is \
                          the MAXIMUM; it reported the most expensive route move in the programme \
                          (strict->wide, 32.1% of remaining budget) as free, and got the sign \
                          wrong on two of five moves"
                    .into(),
            },
            SearchCalibration {
                id: "ROUTE-CAL-1".into(),
                statistic: Statistic::RouteFlipRate,
                scale: EvidenceScale::Diagnostic,
                verdict: SearchEvidence::OrderingProxy {
                    calibration: "ROUTE-CAL-1".into(),
                },
                pairs: 7,
                rank_correlation: Some(0.991),
                finding: "flips per position transfer at a ratio of 0.89-1.02 for every map with \
                          at least 40 diagnostic events — a COUNT statistic survives the sample \
                          size where a TAIL statistic does not. Usable to order and to screen; \
                          never to admit, because the contract judges mixture mass and not flips"
                    .into(),
            },
            SearchCalibration {
                id: "ROUTE-CAL-1".into(),
                statistic: Statistic::Top10MassDisplacedP99,
                scale: EvidenceScale::Diagnostic,
                verdict: SearchEvidence::Unusable,
                pairs: 0,
                rank_correlation: None,
                finding: "thin-tailed at diagnostic scale (74 observations on the four-family \
                          map) and NO paired calibration has been run — unusable by default \
                          rather than by measurement, and the way to change that is to measure it"
                    .into(),
            },
        ])
    }
}

#[cfg(test)]
#[path = "search_evidence_tests.rs"]
mod tests;
