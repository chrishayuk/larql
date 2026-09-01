//! **What a candidate spends, criterion by criterion.**
//!
//! A behavioural contract is not one number, and the criteria in it do
//! not deplete together. Measured on the four-family Kimi map at 8,192
//! positions under `kimi-logit-balanced-v1`:
//!
//! ```text
//! kl p99                68 % of limit    plenty
//! top-1 mass displaced  46 % of limit    plenty
//! top-10 mass p99       54 % of limit    plenty
//! route mixture p99     83 % of limit    BINDING
//! route mixture max     80 % of limit    BINDING
//! ```
//!
//! Ranking candidates by "bytes saved per unit KL" implicitly treats
//! the remaining budget as one scalar pool. It is not. A candidate that
//! looks cheap on logits but moves routing spends the resource that is
//! actually scarce, and a candidate with slightly worse KL and no route
//! movement may be strictly better for the map. This module is the type
//! that makes that difference expressible.
//!
//! **Not every limit is a budget.** Two of them are conditions on the
//! MEASUREMENT rather than costs the candidate pays:
//!
//! ```text
//! kl, displacement, counts   CEILING   the candidate spends these
//! positions, covered mass    FLOOR     the instrument must clear these
//! ```
//!
//! Conflating the two is not pedantry — it is the difference between a
//! verdict and an accident. A blind instrument reports `kl_p99 = 0.0`,
//! a perfect score on every ceiling, and is caught only by the floors.
//! That happened twice while this map was being earned; see
//! `docs/kimi-precision-topology.md`. So `binding()` ranks ceilings
//! only, and floors are asked about separately through [`sound`].
//!
//! [`sound`]: ConstraintVector::sound

use serde::{Deserialize, Serialize};

use super::measurement::{MeasurementStatus, TailSupport, TailSupportPolicy};
use super::quality::{Criterion, QualityBank, QualityGate, Statistic};

/// Whether a limit is a ceiling the candidate spends against, or a
/// floor the measurement itself has to clear.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum LimitKind {
    /// Higher observed values are worse; the budget is `limit`.
    Ceiling,
    /// Lower observed values are worse; `limit` is a minimum.
    Floor,
}

/// One criterion's standing: what the gate asks, what the bank
/// measured, and how much of the budget that leaves.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Margin {
    pub criterion: Criterion,
    /// Which limit this is — `Criterion` is coarser than the gate:
    /// `RouteDisplacement` carries both a p99 and a max limit, and they
    /// can bind independently. TYPED, because this is the join key the
    /// calibration registry and a candidate's proxies look up on; see
    /// [`Statistic`] for what a free string cost.
    pub what: Statistic,
    pub kind: LimitKind,
    pub limit: f64,
    /// `None` when the bank did not record this quantity. That is NOT
    /// zero: an unmeasured consequence is unmeasured, and a candidate
    /// whose evidence is silent about a criterion cannot be ranked on
    /// it. See [`Margin::satisfied`] for the one case where silence is
    /// legitimately a pass.
    pub observed: Option<f64>,
    /// True when the bank recorded no change at all of the kind this
    /// criterion measures, so there was nothing to record a magnitude
    /// for. Only then does `observed: None` mean "cost zero".
    pub vacuous: bool,
    /// For a PERCENTILE, the observations behind its tail. `None` for a
    /// statistic that is not a percentile — a count, a maximum, a mean —
    /// where the question does not arise.
    ///
    /// Carried because a p99 over forty-six observations is a maximum
    /// wearing a percentile's name, and the margin is the last place
    /// that fact is still recoverable.
    pub tail_support: Option<TailSupport>,
}

impl Margin {
    /// Whether the evidence supports the statistic this margin reports.
    ///
    /// A non-percentile is `Measured` whenever it was observed at all:
    /// counts and maxima do not have tails to be thin.
    pub fn measurement_status(&self, policy: &TailSupportPolicy) -> MeasurementStatus {
        if self.observed.is_none() {
            return MeasurementStatus::NotObserved;
        }
        match self.tail_support {
            Some(s) => policy.status(Some(s)),
            None => MeasurementStatus::Measured,
        }
    }

    /// Fraction of this criterion's budget the candidate consumed, for
    /// ceilings — `1.0` is exactly at the limit.
    ///
    /// `None` for floors (a floor is not a budget) and for a criterion
    /// the bank did not measure. A vacuous ceiling consumed nothing, so
    /// it reports `0.0`.
    pub fn utilisation(&self) -> Option<f64> {
        if self.kind != LimitKind::Ceiling || self.limit <= 0.0 {
            return None;
        }
        match self.observed {
            Some(got) => Some(got / self.limit),
            None if self.vacuous => Some(0.0),
            None => None,
        }
    }

    /// How much of this criterion's budget is left, as a fraction.
    /// Negative when the limit is already exceeded.
    pub fn headroom(&self) -> Option<f64> {
        self.utilisation().map(|u| 1.0 - u)
    }

    /// Whether this criterion is currently met.
    ///
    /// An unmeasured criterion is NOT satisfied unless it is vacuous —
    /// the gate's own rule, and the reason a gate that asks for
    /// coverage and gets nothing must refuse rather than assume the
    /// truncation was wide enough.
    pub fn satisfied(&self) -> bool {
        match (self.observed, self.kind) {
            (Some(got), LimitKind::Ceiling) => got <= self.limit,
            (Some(got), LimitKind::Floor) => got >= self.limit,
            (None, _) => self.vacuous,
        }
    }
}

/// Every criterion a gate judges, with what the bank spent against it.
///
/// Built from the same gate and bank a [`QualityGate::evaluate`] call
/// would use, and agreeing with it criterion for criterion — pinned by
/// a congruence test rather than by shared code, so a drift between the
/// two tables fails a test instead of silently mis-ranking candidates.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ConstraintVector {
    pub gate_id: String,
    pub margins: Vec<Margin>,
}

impl ConstraintVector {
    /// The standing of `bank` against every criterion `gate` judges.
    pub fn of(gate: &QualityGate, bank: &QualityBank) -> Self {
        let mut margins = Vec::new();
        let mut ceiling =
            |criterion, what: Statistic, limit: Option<f64>, observed, vacuous, tail_support| {
                if let Some(limit) = limit {
                    margins.push(Margin {
                        criterion,
                        what,
                        kind: LimitKind::Ceiling,
                        limit,
                        observed,
                        vacuous,
                        tail_support,
                    });
                }
            };
        let p99 = |observations: u64| {
            Some(TailSupport {
                quantile: 0.99,
                observations,
            })
        };

        // kl is DENSE: every position contributes a value.
        ceiling(
            Criterion::KlP99,
            Statistic::KlP99,
            Some(gate.kl_p99_max),
            Some(bank.logits.kl_p99),
            false,
            p99(bank.positions),
        );
        for (criterion, what, limit, got) in [
            (
                Criterion::Top1Flips,
                Statistic::Top1Flips,
                gate.top1_flip_max,
                bank.logits.top1_flips,
            ),
            (
                Criterion::Top10Changes,
                Statistic::Top10Changes,
                gate.top10_change_max,
                bank.logits.top10_changes,
            ),
            (
                Criterion::RouteFlips,
                Statistic::RouteFlips,
                gate.route_flip_max,
                bank.routing.route_flips,
            ),
        ] {
            // Raw counts are not percentiles.
            ceiling(
                criterion,
                what,
                limit.map(|l| l as f64),
                Some(got as f64),
                false,
                None,
            );
        }
        // Each consequence magnitude sits beside the COUNT whose being
        // zero makes its absence vacuous rather than missing. Pairing
        // them here, in one table, is what lets a reader check the
        // correspondence — the alternative is a second function that
        // re-derives it from the criterion and needs a wildcard arm.
        // Each consequence magnitude sits beside the COUNT whose being
        // zero makes its absence vacuous rather than missing, and beside
        // the tail support of the statistic it reports. A MAX has no
        // tail to be thin; a p99 over a handful of events is a max.
        for (criterion, what, limit, observed, changes, support) in [
            (
                Criterion::Top1Displacement,
                Statistic::Top1MassDisplaced,
                gate.top1_mass_displaced_max,
                bank.top1_mass_displaced.map(|d| d.max),
                bank.logits.top1_flips,
                None,
            ),
            (
                Criterion::TopKDisplacement,
                Statistic::Top10MassDisplacedP99,
                gate.top10_mass_displaced_p99_max,
                bank.top10_mass_displaced.map(|d| d.p99),
                bank.logits.top10_changes,
                bank.top10_mass_displaced.map(|d| d.count).and_then(&p99),
            ),
            (
                Criterion::RouteDisplacement,
                Statistic::RouteMixtureMassP99,
                gate.route_mixture_mass_p99_max,
                bank.routing.route_weight_mass_moved.map(|d| d.p99),
                bank.routing.route_flips,
                bank.routing
                    .route_weight_mass_moved
                    .map(|d| d.count)
                    .and_then(&p99),
            ),
            (
                Criterion::RouteDisplacement,
                Statistic::RouteMixtureMassMax,
                gate.route_mixture_mass_max,
                bank.routing.route_weight_mass_moved.map(|d| d.max),
                bank.routing.route_flips,
                None,
            ),
        ] {
            let vacuous = observed.is_none() && changes == 0;
            ceiling(criterion, what, limit, observed, vacuous, support);
        }

        // The floors. Neither is a resource a candidate spends; both
        // are conditions without which the ceilings above mean nothing.
        margins.push(Margin {
            criterion: Criterion::Positions,
            what: Statistic::Positions,
            kind: LimitKind::Floor,
            limit: gate.positions_min as f64,
            observed: Some(bank.positions as f64),
            vacuous: false,
            tail_support: None,
        });
        if let Some(required) = gate.covered_mass_min {
            margins.push(Margin {
                criterion: Criterion::CoveredMass,
                what: Statistic::CoveredMass,
                kind: LimitKind::Floor,
                limit: required,
                observed: bank.min_covered_mass,
                vacuous: false,
                tail_support: None,
            });
        }

        Self {
            gate_id: gate.id.clone(),
            margins,
        }
    }

    /// Criteria the candidate SPENDS against — ceilings only.
    pub fn spendable(&self) -> impl Iterator<Item = &Margin> {
        self.margins.iter().filter(|m| m.kind == LimitKind::Ceiling)
    }

    /// **The scarce resource: the ceiling closest to its limit.**
    ///
    /// This is what a search should rank against. Bytes saved per unit
    /// of KL is the wrong objective when KL is at 68 % of budget and
    /// route movement is at 83 %: the next candidate is admitted or
    /// refused by the route limit, whatever it does to KL.
    ///
    /// `None` when no ceiling could be scored — every one of them
    /// unmeasured, which is a statement about the evidence, not about
    /// the candidate.
    pub fn binding(&self) -> Option<&Margin> {
        self.spendable()
            .filter(|m| m.utilisation().is_some())
            .max_by(|a, b| {
                let (x, y) = (
                    a.utilisation().unwrap_or(f64::NEG_INFINITY),
                    b.utilisation().unwrap_or(f64::NEG_INFINITY),
                );
                x.partial_cmp(&y).unwrap_or(std::cmp::Ordering::Equal)
            })
    }

    /// Worst utilisation recorded for `criterion` — worst, because
    /// `Criterion` is coarser than the gate and one criterion can carry
    /// several limits that bind independently.
    pub fn utilisation_of(&self, criterion: Criterion) -> Option<f64> {
        self.spendable()
            .filter(|m| m.criterion == criterion)
            .filter_map(|m| m.utilisation())
            .max_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
    }

    /// Whether the MEASUREMENT is sound — every floor cleared.
    ///
    /// Asked separately from the ceilings on purpose. A blind
    /// instrument passes every ceiling perfectly, and the floors are
    /// the only thing standing between that and a promoted map.
    pub fn sound(&self) -> bool {
        self.margins
            .iter()
            .filter(|m| m.kind == LimitKind::Floor)
            .all(Margin::satisfied)
    }

    /// Whether every criterion — ceiling and floor — is met.
    pub fn admissible(&self) -> bool {
        self.margins.iter().all(Margin::satisfied)
    }
}

#[cfg(test)]
#[path = "constraint_tests.rs"]
mod tests;
