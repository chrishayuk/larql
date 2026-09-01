//! The statistic vocabulary — the join key between a constraint vector,
//! the calibration registry, and a candidate's proxy observations.
//!
//! Lives apart from [`super::quality`] because that module owns the
//! CONTRACT (gates, criteria, verdicts) while this owns what a bank
//! REPORTS, and because BS2-F1 gave the diagnostic side statistics the
//! contract never bounds.

use serde::{Deserialize, Serialize};

use super::measurement::TailSupport;
use super::quality::QualityBank;

/// The exact quantity a [`super::constraint::Margin`] reports — the JOIN
/// KEY between a constraint vector, the calibration registry, and a
/// candidate's proxy observations.
///
/// **Finer than [`Criterion`] on purpose.** `RouteDisplacement` carries
/// both a p99 and a max limit and they bind independently: at 256
/// positions the p99 is a maximum wearing a percentile's name while the
/// max is exactly what it says, so keying evidence on the criterion
/// would hand the thin percentile the max's confidence.
///
/// **An enum and not a string, on purpose.** BS2-F2: the registry keyed
/// `"route flip rate"` while the vector emitted `"route flips"`. The
/// lookup missed, `SearchCalibrationRegistry::evidence_for` fell through
/// to its `is_priceable()` arm, and a COUNT — always `Measured` —
/// returned `Direct`. That silently PRICED the one statistic ROUTE-CAL-1
/// calibrated as ordering-only, which is the failure the whole evidence
/// ladder exists to prevent. Two of the three keys matched, so the
/// mechanism looked like it worked. A typed key cannot drift.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum Statistic {
    KlP99,
    Top1Flips,
    Top10Changes,
    RouteFlips,
    /// Route flips per POSITION — what ROUTE-CAL-1 actually measured
    /// transferring across scale (0.89-1.02 for every map with >= 40
    /// diagnostic events), and a different quantity from the raw
    /// [`Self::RouteFlips`] count a contract might one day bound.
    /// Diagnostic-only: no gate reads it, by design (BS2-F1).
    RouteFlipRate,
    Top1MassDisplaced,
    Top10MassDisplacedP99,
    RouteMixtureMassP99,
    RouteMixtureMassMax,
    Positions,
    CoveredMass,
}

impl Statistic {
    /// The human label, for traces. Not a key — nothing joins on this.
    pub fn label(self) -> &'static str {
        match self {
            Self::KlP99 => "kl p99",
            Self::Top1Flips => "top-1 flips",
            Self::Top10Changes => "top-10 changes",
            Self::RouteFlips => "route flips",
            Self::RouteFlipRate => "route flips per position",
            Self::Top1MassDisplaced => "top-1 probability given up",
            Self::Top10MassDisplacedP99 => "top-10 mass displaced at p99",
            Self::RouteMixtureMassP99 => "routed mixture moved at p99",
            Self::RouteMixtureMassMax => "routed mixture moved at max",
            Self::Positions => "positions",
            Self::CoveredMass => "covered mass at the worst position",
        }
    }
}

impl Statistic {
    /// Read this statistic off a bank: its value, and the observations
    /// behind its tail if it is a percentile.
    ///
    /// **The single mapping from a statistic to the bank.** Authority
    /// margins and diagnostic readings are built by different code for
    /// different purposes, but they must not DISAGREE about what a
    /// number is — BS2-F2 was two vocabularies drifting apart, and two
    /// bank extractions would drift the same way.
    pub fn observe(self, bank: &QualityBank) -> (Option<f64>, Option<TailSupport>) {
        let p99 = |observations: u64| {
            Some(TailSupport {
                quantile: 0.99,
                observations,
            })
        };
        match self {
            // Dense: every position contributes a value.
            Self::KlP99 => (Some(bank.logits.kl_p99), p99(bank.positions)),
            Self::Top1Flips => (Some(bank.logits.top1_flips as f64), None),
            Self::Top10Changes => (Some(bank.logits.top10_changes as f64), None),
            Self::RouteFlips => (Some(bank.routing.route_flips as f64), None),
            // A rate over ALL positions is dense, so it has no thin tail
            // — which is exactly why it survives a small bank.
            Self::RouteFlipRate => (
                (bank.positions > 0)
                    .then(|| bank.routing.route_flips as f64 / bank.positions as f64),
                None,
            ),
            Self::Top1MassDisplaced => (bank.top1_mass_displaced.map(|d| d.max), None),
            Self::Top10MassDisplacedP99 => (
                bank.top10_mass_displaced.map(|d| d.p99),
                bank.top10_mass_displaced.map(|d| d.count).and_then(p99),
            ),
            Self::RouteMixtureMassP99 => (
                bank.routing.route_weight_mass_moved.map(|d| d.p99),
                bank.routing
                    .route_weight_mass_moved
                    .map(|d| d.count)
                    .and_then(p99),
            ),
            Self::RouteMixtureMassMax => {
                (bank.routing.route_weight_mass_moved.map(|d| d.max), None)
            }
            Self::Positions => (Some(bank.positions as f64), None),
            Self::CoveredMass => (bank.min_covered_mass, None),
        }
    }
}

impl std::fmt::Display for Statistic {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.label())
    }
}

/// Which way is BETTER for a statistic.
///
/// **One canonical derivation.** An ordinal comparison needs to know
/// which end is good, and deriving that from a gate's [`LimitKind`]
/// would work only for statistics a contract happens to bound —
/// `RouteFlipRate` is bounded by nothing and still has an unambiguous
/// direction. Two places deciding this would drift, which is the
/// failure BS2-F2 and BS2-F1b both were.
///
/// [`LimitKind`]: super::constraint::LimitKind
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum Better {
    /// A cost: less movement is better.
    Lower,
    /// A sufficiency: more is better.
    Higher,
}

impl Statistic {
    /// Which direction is better for this statistic.
    pub fn better(self) -> Better {
        match self {
            // Every consequence magnitude and every change count is a
            // cost — the candidate moved the model, and less is less.
            Self::KlP99
            | Self::Top1Flips
            | Self::Top10Changes
            | Self::RouteFlips
            | Self::RouteFlipRate
            | Self::Top1MassDisplaced
            | Self::Top10MassDisplacedP99
            | Self::RouteMixtureMassP99
            | Self::RouteMixtureMassMax => Better::Lower,
            // Sufficiency conditions on the MEASUREMENT, not costs the
            // candidate pays: more positions and wider coverage are
            // strictly more evidence.
            Self::Positions | Self::CoveredMass => Better::Higher,
        }
    }

    /// Order `self`'s value against `other`'s, in BETTER-FIRST terms:
    /// `Less` means `a` is the better of the two.
    ///
    /// Ordinal only. It reports WHICH is better and never by how much —
    /// see [`super::search_evidence::SearchEvidence::OrderingProxy`].
    pub fn order(self, a: f64, b: f64) -> std::cmp::Ordering {
        match self.better() {
            Better::Lower => a.total_cmp(&b),
            Better::Higher => b.total_cmp(&a),
        }
    }
}

#[cfg(test)]
#[path = "statistic_tests.rs"]
mod tests;
