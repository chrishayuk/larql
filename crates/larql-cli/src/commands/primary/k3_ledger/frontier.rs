//! DEC-8.7a — the dense-path throughput frontier, and the R4 zero-out ceiling.
//!
//! Pure. Inverts the bandwidth-bound decode relation `tok/s = BW / bytes-per-token`
//! to answer, for each target rate, what average precision the always-on dense
//! weights must reach once routed experts are credited with whatever reduction
//! they earn.
//!
//! The frontier exists so that no single assumed compression result gets baked
//! into a headline. Every downstream fidelity bar cites a row of this table.

use serde::{Deserialize, Serialize};

use super::geometry::{BitConvention, DensePricing, K3Geometry, MXFP4_PAYLOAD_BITS};

/// Measured attainable bandwidth, not spec sheet (`project_memory_bandwidth_roofline`).
pub const BW_GPU_GB_S: f64 = 367.0;
pub const BW_CPU_GB_S: f64 = 127.0;

/// Serving premises the frontier is computed against.
///
/// `routed_retention` is the fraction of routed bytes still read. It is NOT
/// 1.87 — that figure was DEC-8.6's R4 zero-out ratio (`total/dense`) and was
/// briefly, wrongly, used as a routed-side divisor. The banked floor is the
/// up-fold's two thirds; anything below is row sparsity, owner DEC-8.1, unset.
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct ServingPremises {
    pub bandwidth_gb_s: f64,
    /// Fraction of attainable bandwidth the kernel actually realises. Owner:
    /// the MXFP4 kernel bench (DEC-8.8). The banked Q4K band is 0.74-0.86.
    pub dequant_efficiency: f64,
    pub routed_retention: f64,
    /// How the dense side is priced. Defaults to the checkpoint's own
    /// representation; a hypothetical must be asked for and is labelled.
    pub dense: DensePricing,
}

impl Default for ServingPremises {
    fn default() -> Self {
        Self {
            bandwidth_gb_s: BW_GPU_GB_S,
            dequant_efficiency: 1.0,
            routed_retention: 2.0 / 3.0,
            dense: DensePricing::AsStored,
        }
    }
}

impl ServingPremises {
    /// Bytes per second actually available.
    pub fn effective_bw(&self) -> f64 {
        self.bandwidth_gb_s * 1e9 * self.dequant_efficiency
    }

    /// All-in bits the dense side is priced at, resolved against the
    /// checkpoint.
    pub fn dense_all_in_bits(&self, geom: &K3Geometry) -> f64 {
        self.dense.all_in_bits(geom)
    }

    /// Bytes a token may read at `target` tokens/second.
    pub fn budget_per_token(&self, target: f64) -> f64 {
        self.effective_bw() / target
    }
}

/// One row of the frontier: what a target rate demands of the dense path.
#[derive(Debug, Clone, Serialize)]
pub struct FrontierRow {
    pub target_tok_s: f64,
    pub budget_bytes: f64,
    pub dense_budget_bytes: f64,
    pub payload_bits: f64,
    pub all_in_bits: f64,
    pub verdict: DenseVerdict,
}

/// How hard the dense-precision requirement is, in difficulty classes.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
pub enum DenseVerdict {
    /// Routed traffic alone already exceeds the budget.
    Impossible,
    ExistingFourBitFits,
    /// ~3-bit dense — a credible research target.
    CredibleResearch,
    /// ~2-bit dense — a serious stretch.
    SeriousStretch,
    /// Sub-2-bit — retrain or change the architecture.
    RetrainOrArchitecture,
    /// Not a quantisation problem at all.
    NewModelProgramme,
}

impl DenseVerdict {
    fn classify(dense_budget: f64, payload_bits: f64) -> Self {
        if dense_budget <= 0.0 {
            Self::Impossible
        } else if payload_bits >= MXFP4_PAYLOAD_BITS {
            Self::ExistingFourBitFits
        } else if payload_bits >= 3.0 {
            Self::CredibleResearch
        } else if payload_bits >= 2.0 {
            Self::SeriousStretch
        } else if payload_bits >= 1.0 {
            Self::RetrainOrArchitecture
        } else {
            Self::NewModelProgramme
        }
    }
}

/// Routed bytes per token after the credited retention.
pub fn routed_floor(geom: &K3Geometry, p: &ServingPremises) -> f64 {
    geom.routed_bytes_per_position() * p.routed_retention
}

/// Price one target rate against the dense path.
pub fn frontier_row(geom: &K3Geometry, p: &ServingPremises, target: f64) -> FrontierRow {
    let budget = p.budget_per_token(target);
    let dense_budget = budget - routed_floor(geom, p);
    let params = geom.dense_params();
    let payload_bits = BitConvention::Payload.bits(dense_budget.max(0.0), params);
    let all_in_bits = BitConvention::AllIn.bits(dense_budget.max(0.0), params);
    FrontierRow {
        target_tok_s: target,
        budget_bytes: budget,
        dense_budget_bytes: dense_budget,
        payload_bits,
        all_in_bits,
        verdict: DenseVerdict::classify(dense_budget, payload_bits),
    }
}

/// **Standing rule R4, executable.** Zero the routed term entirely and see what
/// the untouched dense half still permits.
///
/// If a lever targets routed bytes and this ceiling already sits below the
/// target, that lever cannot decide the target — however good its result. One
/// division, run before an experiment is designed.
pub fn expert_side_ceiling_tok_s(geom: &K3Geometry, p: &ServingPremises) -> f64 {
    p.effective_bw() / geom.dense_bytes(p.dense_all_in_bits(geom))
}

/// Whether an expert-side experiment could decide `target` at all.
pub fn expert_side_can_decide(geom: &K3Geometry, p: &ServingPremises, target: f64) -> bool {
    expert_side_ceiling_tok_s(geom, p) >= target
}

pub fn frontier(geom: &K3Geometry, p: &ServingPremises, targets: &[f64]) -> Vec<FrontierRow> {
    targets.iter().map(|&t| frontier_row(geom, p, t)).collect()
}

#[cfg(test)]
mod tests {
    use super::super::geometry::DENSE_4_25;
    use super::*;
    use crate::commands::primary::k3_ledger::geometry::k3_reference;

    fn approx(a: f64, b: f64, tol: f64) {
        assert!((a - b).abs() <= tol, "expected ~{b}, got {a}");
    }

    #[test]
    fn expert_side_ceiling_is_the_governing_constant() {
        // 12.3 tok/s with dense at 4-bit: no expert-side result can exceed it.
        let ceiling = expert_side_ceiling_tok_s(
            &k3_reference(),
            &ServingPremises {
                dense: DENSE_4_25,
                ..Default::default()
            },
        );
        approx(ceiling, 12.3, 0.2);
    }

    #[test]
    fn r4_refuses_twenty_tok_s_for_every_expert_side_lever() {
        let (g, p) = (
            k3_reference(),
            ServingPremises {
                dense: DENSE_4_25,
                ..Default::default()
            },
        );
        assert!(!expert_side_can_decide(&g, &p, 20.0));
        assert!(!expert_side_can_decide(&g, &p, 14.0));
        assert!(expert_side_can_decide(&g, &p, 8.0));
    }

    #[test]
    fn r4_holds_even_with_routed_traffic_deleted() {
        // Retention zero is the strongest possible expert-side result.
        let g = k3_reference();
        let p = ServingPremises {
            routed_retention: 0.0,
            ..Default::default()
        };
        let row = frontier_row(&g, &p, 20.0);
        assert!(row.dense_budget_bytes > 0.0);
        assert!(
            row.payload_bits < 3.0,
            "20 tok/s with zero routed traffic still demands {} payload bits",
            row.payload_bits
        );
    }

    #[test]
    fn twenty_tok_s_is_not_a_quantisation_problem() {
        let row = frontier_row(
            &k3_reference(),
            &ServingPremises {
                dense: DENSE_4_25,
                ..Default::default()
            },
            20.0,
        );
        assert_eq!(row.verdict, DenseVerdict::NewModelProgramme);
    }

    #[test]
    fn the_ladder_separates_into_difficulty_classes() {
        let (g, p) = (
            k3_reference(),
            ServingPremises {
                dense: DENSE_4_25,
                ..Default::default()
            },
        );
        let rows = frontier(&g, &p, &[8.0, 10.0, 12.0, 14.0, 20.0]);
        let verdicts: Vec<_> = rows.iter().map(|r| r.verdict).collect();
        // Under the BANKED up-fold retention (2/3). These are harsher than the
        // pre-correction table, which credited an illegitimate 1.87x routed
        // divisor and so left ~3.4 GB/token of budget that was never available.
        assert_eq!(verdicts[0], DenseVerdict::CredibleResearch);
        assert_eq!(verdicts[1], DenseVerdict::SeriousStretch);
        assert_eq!(verdicts[2], DenseVerdict::RetrainOrArchitecture);
        assert_eq!(verdicts[3], DenseVerdict::RetrainOrArchitecture);
        assert_eq!(verdicts[4], DenseVerdict::NewModelProgramme);
    }

    #[test]
    fn requirement_tightens_monotonically_with_the_target() {
        let (g, p) = (
            k3_reference(),
            ServingPremises {
                dense: DENSE_4_25,
                ..Default::default()
            },
        );
        let rows = frontier(&g, &p, &[6.0, 8.0, 10.0, 12.0, 14.0, 20.0]);
        for w in rows.windows(2) {
            assert!(w[1].payload_bits < w[0].payload_bits);
        }
    }

    #[test]
    fn dequant_efficiency_moves_the_ceiling_a_whole_class() {
        // The banked Metal band is 0.74-0.86, so this is not hypothetical: the
        // 10 tok/s row moves from credible-3-bit to serious-stretch-2-bit.
        let g = k3_reference();
        let ideal = ServingPremises {
            dense: DENSE_4_25,
            ..Default::default()
        };
        let real = ServingPremises {
            dequant_efficiency: 0.80,
            ..Default::default()
        };
        assert!(expert_side_ceiling_tok_s(&g, &real) < expert_side_ceiling_tok_s(&g, &ideal));
        assert_eq!(
            frontier_row(&g, &ideal, 10.0).verdict,
            DenseVerdict::SeriousStretch
        );
        assert_eq!(
            frontier_row(&g, &real, 10.0).verdict,
            DenseVerdict::RetrainOrArchitecture
        );
    }

    #[test]
    fn retention_defaults_to_the_banked_up_fold_not_to_one_point_eight_seven() {
        // Regression guard: 1.87 was the R4 zero-out ratio wearing the wrong hat.
        let p = ServingPremises {
            dense: DENSE_4_25,
            ..Default::default()
        };
        approx(
            p.routed_retention,
            k3_reference().up_fold_retention(),
            1e-12,
        );
        assert!((1.0 / p.routed_retention - 1.87).abs() > 0.3);
    }

    #[test]
    fn routed_floor_tracks_retention() {
        let g = k3_reference();
        let full = ServingPremises {
            routed_retention: 1.0,
            ..Default::default()
        };
        approx(routed_floor(&g, &full), g.routed_bytes_per_position(), 1.0);
        let none = ServingPremises {
            routed_retention: 0.0,
            ..Default::default()
        };
        approx(routed_floor(&g, &none), 0.0, 1e-9);
    }
}
