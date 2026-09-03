//! The premises a `frontier` row is conditional on, made explicit and printable.
//!
//! Pure. This file exists because of a specific misquote: the frontier's
//! "8 tok/s needs ~3.8 payload bits" row was read beside the composed exact
//! ceiling of 5.77 tok/s as though the two answered the same question.
//!
//! They do not, and the difference is not a detail:
//!
//! | | `frontier` | `ceilings` |
//! |---|---|---|
//! | question | given a credited routed reduction, what must the **dense path** reach? | what does the **whole model** cost as configured? |
//! | routed bytes | credited a retention (default: the banked up-fold, 2/3) | priced in full at their format |
//! | efficiency | **one scalar**, defaulting to **1.00 — ideal** | **per-class measured** eta, 0.64-0.89 |
//! | output | a *requirement* on dense precision | a *ceiling* in tok/s |
//!
//! So a frontier row is a target generator: it says what would have to be true.
//! A ceilings row is a ledger: it says what is true given the formats. Quoting
//! the first as if it were the second credits a routed reduction that has not
//! been measured and an efficiency of 1.00 that no kernel reaches.
//!
//! Every warning below names a premise that, left unstated, would let a row
//! travel further than its evidence.

use serde::Serialize;

use super::frontier::ServingPremises;
use super::geometry::K3Geometry;

/// Efficiency of a kernel that loses nothing. No measured kernel reaches it;
/// the banked Metal band is 0.74-0.86.
pub const IDEAL_EFFICIENCY: f64 = 1.0;

/// Retention of 1.0 means no routed reduction is credited at all.
pub const NO_ROUTED_REDUCTION: f64 = 1.0;

/// The banner every frontier rendering must carry.
pub const SCENARIO_BANNER: &str = "SCENARIO, NOT A WHOLE-MODEL CEILING";

/// What the bits columns of a frontier row actually mean.
pub const BITS_COLUMN_MEANING: &str = "required_dense_bits_under_assumptions";

/// The premises one frontier table was computed against.
#[derive(Debug, Clone, Serialize)]
pub struct ScenarioPremises {
    pub routed_retention: f64,
    /// `1 / retention` — the reduction credited to the routed path.
    pub routed_reduction: f64,
    /// All-in bits the routed bytes are priced at. Derived from the checkpoint
    /// geometry rather than assumed, so it stays honest if the codec changes.
    pub routed_all_in_bits: f64,
    /// The single efficiency applied to the whole read. `frontier` has no
    /// per-class eta; that asymmetry is the main reason its rows are not
    /// comparable with `ceilings`.
    pub scalar_efficiency: f64,
    pub bandwidth_gb_s: f64,
    /// Used only for the R4 zero-out ceiling, not for the table rows.
    pub dense_all_in_bits: f64,
    /// K3-L1-F1: true when the dense side is priced at a width the
    /// checkpoint does not have. Such a row describes a FUTURE model.
    pub dense_is_hypothetical: bool,
    /// How the dense pricing must be described.
    pub dense_label: String,
}

impl ScenarioPremises {
    pub fn describe(geom: &K3Geometry, p: &ServingPremises) -> Self {
        let routed_params = geom.routed_activated_params();
        Self {
            routed_retention: p.routed_retention,
            routed_reduction: if p.routed_retention > 0.0 {
                1.0 / p.routed_retention
            } else {
                f64::INFINITY
            },
            routed_all_in_bits: geom.routed_bytes_per_position() * 8.0 / routed_params as f64,
            scalar_efficiency: p.dequant_efficiency,
            bandwidth_gb_s: p.bandwidth_gb_s,
            dense_all_in_bits: p.dense_all_in_bits(geom),
            dense_is_hypothetical: p.dense.is_hypothetical(),
            dense_label: p.dense.label(geom),
        }
    }

    /// True when the efficiency premise is the unreachable ideal.
    pub fn efficiency_is_ideal(&self) -> bool {
        self.scalar_efficiency >= IDEAL_EFFICIENCY
    }

    /// True when a routed reduction is credited that the ledger has not measured.
    pub fn credits_unmeasured_routed_reduction(&self) -> bool {
        self.routed_retention < NO_ROUTED_REDUCTION
    }

    /// Conditions that, unstated, would let a row be quoted beyond its evidence.
    pub fn warnings(&self) -> Vec<&'static str> {
        let mut w = Vec::new();
        if self.dense_is_hypothetical {
            w.push(
                "! DENSE PRICED AT A WIDTH THE CHECKPOINT DOES NOT HAVE — this row \
                 describes a FUTURE representation, not the source model (K3-L1-F1)",
            );
        }
        if self.efficiency_is_ideal() {
            w.push(
                "efficiency 1.00 is IDEAL — no measured kernel reaches it (banked band 0.74-0.86)",
            );
        }
        if self.credits_unmeasured_routed_reduction() {
            w.push("a routed reduction is CREDITED here — the dense requirement assumes it lands");
        }
        w.push("one scalar efficiency, not the per-class measured eta that `ceilings` prices");
        w.push("rows are a REQUIREMENT on dense precision, not an achieved rate");
        w
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::commands::primary::k3_ledger::geometry::k3_reference;

    fn approx(a: f64, b: f64, tol: f64) {
        assert!((a - b).abs() <= tol, "expected ~{b}, got {a}");
    }

    fn described(p: &ServingPremises) -> ScenarioPremises {
        ScenarioPremises::describe(&k3_reference(), p)
    }

    #[test]
    fn routed_bits_are_derived_from_geometry_not_assumed() {
        // The checkpoint is MXFP4, so this must come back 4.25 without anyone
        // typing 4.25 — that is the point of deriving it.
        approx(
            described(&ServingPremises::default()).routed_all_in_bits,
            4.25,
            1e-6,
        );
    }

    #[test]
    fn the_default_frontier_carries_both_dangerous_premises() {
        // Regression guard on the exact pair that produced the misquote.
        let s = described(&ServingPremises::default());
        assert!(s.efficiency_is_ideal());
        assert!(s.credits_unmeasured_routed_reduction());
        assert_eq!(s.warnings().len(), 4);
    }

    #[test]
    fn a_measured_efficiency_drops_the_ideal_warning() {
        let s = described(&ServingPremises {
            dequant_efficiency: 0.80,
            ..Default::default()
        });
        assert!(!s.efficiency_is_ideal());
        assert!(!s.warnings().iter().any(|w| w.contains("IDEAL")));
    }

    #[test]
    fn full_retention_drops_the_credited_reduction_warning() {
        let s = described(&ServingPremises {
            routed_retention: NO_ROUTED_REDUCTION,
            ..Default::default()
        });
        assert!(!s.credits_unmeasured_routed_reduction());
        approx(s.routed_reduction, 1.0, 1e-12);
        assert!(!s.warnings().iter().any(|w| w.contains("CREDITED")));
    }

    #[test]
    fn the_two_unconditional_warnings_always_survive() {
        // Even a fully-honest scenario is still not a whole-model ceiling.
        let s = described(&ServingPremises {
            dequant_efficiency: 0.80,
            routed_retention: NO_ROUTED_REDUCTION,
            ..Default::default()
        });
        assert_eq!(s.warnings().len(), 2);
        assert!(s.warnings().iter().any(|w| w.contains("per-class")));
        assert!(s.warnings().iter().any(|w| w.contains("REQUIREMENT")));
    }

    #[test]
    fn zero_retention_does_not_divide_by_zero() {
        let s = described(&ServingPremises {
            routed_retention: 0.0,
            ..Default::default()
        });
        assert!(s.routed_reduction.is_infinite());
    }

    #[test]
    fn the_banked_up_fold_is_what_the_default_credits() {
        let s = described(&ServingPremises::default());
        approx(
            s.routed_retention,
            k3_reference().up_fold_retention(),
            1e-12,
        );
        approx(s.routed_reduction, 1.5, 1e-9);
    }
}
