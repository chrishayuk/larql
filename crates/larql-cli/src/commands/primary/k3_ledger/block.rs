//! DEC-9.2 — speculative block economics, with work-width separated from
//! commit-width (R5) and logical reuse separated from physical (R6).
//!
//! Pure. This module's pre-registration was withdrawn twice, both times before
//! any compute was spent, and both times for an arithmetic error over correct
//! measurements. The types here exist so neither can recur silently:
//!
//!   v1 scaled DEC-8.7a's routed-*zeroed* ceiling by block length, deleting the
//!      routed term from the one setting where it returns and grows.
//!   v2 used one variable as both union index and amortisation divisor, which
//!      silently assumes perfect acceptance.
//!
//! The honest condition is
//!
//! ```text
//!   D*d(T) + rho_r*R*r(T) + S(T)  <=  eta * B_target * A(T)
//! ```
//!
//! with `T` (positions evaluated), `A(T)` (positions committed), `u(T)` (logical
//! union) and `d(T)`/`r(T)` (physical bytes actually read) all independent.

use serde::{Deserialize, Serialize};

use super::frontier::ServingPremises;
use super::geometry::K3Geometry;

/// A drafter proposing `width` positions, of which `mean_accepted` commit.
///
/// **R5.** Costs key on `width`; throughput keys on `mean_accepted`. Constructing
/// one where they are equal asserts perfect acceptance, which `assumes_perfect`
/// reports so a caller can refuse it.
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct DraftProfile {
    pub width: u32,
    pub mean_accepted: f64,
}

impl DraftProfile {
    pub fn new(width: u32, mean_accepted: f64) -> Self {
        Self {
            width,
            mean_accepted,
        }
    }

    /// True when the profile implies every drafted position commits.
    pub fn assumes_perfect(&self) -> bool {
        (self.mean_accepted - self.width as f64).abs() < 1e-9
    }

    /// Per-token acceptance probability under the standard geometric model,
    /// fitted from this one observed `(width, mean_accepted)` pair.
    ///
    /// **Provisional by construction.** One point does not determine the real
    /// `A(T)` curve — early-token acceptance can be much stronger or weaker than
    /// geometric. Any optimum derived from this is a design point, not a
    /// performance bar, until a swept acceptance distribution replaces it.
    pub fn fitted_alpha(&self) -> f64 {
        let (mut lo, mut hi) = (1e-6_f64, 1.0 - 1e-9);
        for _ in 0..200 {
            let mid = 0.5 * (lo + hi);
            if accepted_at(mid, self.width) < self.mean_accepted {
                lo = mid;
            } else {
                hi = mid;
            }
        }
        0.5 * (lo + hi)
    }

    /// Expected committed tokens at a different proposal width, same drafter.
    pub fn accepted_at_width(&self, width: u32) -> f64 {
        accepted_at(self.fitted_alpha(), width)
    }
}

/// Expected committed tokens over `width` drafted positions, bonus token included.
fn accepted_at(alpha: f64, width: u32) -> f64 {
    if (alpha - 1.0).abs() < 1e-12 {
        return width as f64 + 1.0;
    }
    (1.0 - alpha.powi(width as i32 + 1)) / (1.0 - alpha)
}

/// How many bytes execution *actually* reads, against what the logical model hopes.
///
/// **R6.** A union is only potential reuse. Ideal execution gives `dense = 1.0`
/// and `routed = u(T)`; a naive token loop gives both `= T`. Nothing establishes
/// the ideal without measuring it — owner DEC-9.1a.
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct PhysicalReuse {
    /// `d(T)`: dense bytes read / D.
    pub dense: f64,
    /// `r(T)`: routed bytes read / (rho_r * R), in single-position equivalents.
    pub routed: f64,
    /// False until DEC-9.1a reports; keeps optimism auditable.
    pub measured: bool,
}

impl PhysicalReuse {
    /// The assumption the arithmetic banks when nobody has measured: perfect
    /// grouping and one dense read per block.
    pub fn assumed_ideal(union_equivalents: f64) -> Self {
        Self {
            dense: 1.0,
            routed: union_equivalents,
            measured: false,
        }
    }

    /// The pessimal case: no grouping, no block GEMM.
    pub fn token_loop(width: u32) -> Self {
        Self {
            dense: width as f64,
            routed: width as f64,
            measured: false,
        }
    }
}

/// State and transaction traffic — KDA checkpoint/restore, commit, rollback.
///
/// Owner: M1-B. Defaults to zero and says so, rather than being silently absent
/// from the budget the way it was in the first port of this model.
///
/// The split matters. A flat per-pass cost amortises over a longer block and so
/// makes wider proposals look *better*; but M1-B's prefix-state ladder retains a
/// recurrent state after every *proposed* position, which is a per-position cost
/// that grows with width while acceptance saturates. Modelling only the flat
/// term inverts the sign of state traffic's effect on the optimum.
#[derive(Debug, Clone, Copy, Default, Serialize, Deserialize)]
pub struct StateTraffic {
    /// Fixed cost of one pass: base checkpoint, commit.
    pub bytes_per_pass: f64,
    /// Cost per *proposed* position: prefix-state retention, rollback surface.
    pub bytes_per_position: f64,
    /// False until M1-B reports; keeps optimism auditable.
    pub measured: bool,
}

impl StateTraffic {
    /// Total state traffic for a pass evaluating `width` positions.
    pub fn total(&self, width: u32) -> f64 {
        self.bytes_per_pass + self.bytes_per_position * width as f64
    }

    /// K3's prefix-state ladder: one `[96,128,128]` KDA recurrent state per
    /// proposed position, over the linear-attention layers.
    pub fn prefix_state_ladder(n_kda_layers: usize, state_dtype_bytes: u64) -> Self {
        let per_layer = 96.0 * 128.0 * 128.0 * state_dtype_bytes as f64;
        Self {
            bytes_per_pass: 0.0,
            bytes_per_position: per_layer * n_kda_layers as f64,
            measured: false,
        }
    }
}

/// The composed per-pass byte budget and what it implies.
#[derive(Debug, Clone, Serialize)]
pub struct BlockEconomics {
    pub width: u32,
    pub accepted: f64,
    pub union_equivalents: f64,
    pub bytes_per_pass: f64,
    pub bytes_per_committed_token: f64,
    pub tok_s: f64,
    /// Maximum routed retention still compatible with the target, or `None`
    /// when the dense term alone already refuses it.
    pub rho_max: Option<f64>,
    /// `1 / rho_max` — the routed-side reduction the drafter demands.
    pub routed_reduction_needed: Option<f64>,
    /// True when the arithmetic rests on unmeasured physical reuse or state traffic.
    pub rests_on_assumptions: bool,
}

/// Evaluate one proposal width against the premises.
pub fn evaluate(
    geom: &K3Geometry,
    p: &ServingPremises,
    draft: &DraftProfile,
    width: u32,
    reuse: PhysicalReuse,
    state: StateTraffic,
    target_tok_s: f64,
) -> BlockEconomics {
    let accepted = draft.accepted_at_width(width);
    let union_equivalents = geom.expert_union(width) / geom.top_k as f64;

    let dense = geom.dense_bytes(p.dense_all_in_bits(geom)) * reuse.dense;
    let routed = geom.routed_bytes_per_position() * p.routed_retention * reuse.routed;
    let bytes_per_pass = dense + routed + state.total(width);

    let per_token = bytes_per_pass / accepted;
    let tok_s = p.effective_bw() / per_token;

    // rho_max(T) = (eta*B_target*A(T) - D*d(T) - S(T)) / (R * r(T))
    let headroom = p.budget_per_token(target_tok_s) * accepted - dense - state.total(width);
    let denom = geom.routed_bytes_per_position() * reuse.routed;
    let rho_max = if headroom > 0.0 && denom > 0.0 {
        Some(headroom / denom)
    } else {
        None
    };

    BlockEconomics {
        width,
        accepted,
        union_equivalents,
        bytes_per_pass,
        bytes_per_committed_token: per_token,
        tok_s,
        rho_max,
        routed_reduction_needed: rho_max.map(|r| 1.0 / r),
        rests_on_assumptions: !reuse.measured || !state.measured,
    }
}

/// **R4 in speculative clothing.** Zero the routed term: does the dense half
/// already refuse this arm, whatever the routed side achieves?
pub fn dense_alone_refuses(
    geom: &K3Geometry,
    p: &ServingPremises,
    accepted: f64,
    width: u32,
    reuse: PhysicalReuse,
    state: StateTraffic,
    target_tok_s: f64,
) -> bool {
    let dense = geom.dense_bytes(p.dense_all_in_bits(geom)) * reuse.dense + state.total(width);
    p.budget_per_token(target_tok_s) * accepted <= dense
}

/// Sweep proposal width and return the interior optimum by `rho_max`.
///
/// `u(T)` grows near-linearly at K3's sparsity while `A(T)` saturates, so the
/// tolerable retention peaks in the interior. A drafter's shipped width is not
/// its best width.
pub fn sweep<'a>(
    geom: &K3Geometry,
    p: &ServingPremises,
    draft: &DraftProfile,
    widths: impl IntoIterator<Item = &'a u32>,
    state: StateTraffic,
    target_tok_s: f64,
) -> Vec<BlockEconomics> {
    widths
        .into_iter()
        .map(|&t| {
            let reuse = PhysicalReuse::assumed_ideal(geom.expert_union(t) / geom.top_k as f64);
            evaluate(geom, p, draft, t, reuse, state, target_tok_s)
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::super::geometry::DENSE_4_25;
    use super::*;
    use crate::commands::primary::k3_ledger::geometry::k3_reference;

    fn dspark() -> DraftProfile {
        DraftProfile::new(7, 3.85)
    }

    fn approx(a: f64, b: f64, tol: f64) {
        assert!((a - b).abs() <= tol, "expected ~{b}, got {a}");
    }

    #[test]
    fn perfect_acceptance_is_detectable() {
        // R5 guard: T == A is the error, and callers must be able to see it.
        assert!(DraftProfile::new(4, 4.0).assumes_perfect());
        assert!(!dspark().assumes_perfect());
    }

    #[test]
    fn fitted_alpha_reproduces_its_own_observation() {
        let d = dspark();
        approx(d.accepted_at_width(d.width), d.mean_accepted, 1e-6);
    }

    #[test]
    fn acceptance_saturates_while_union_grows() {
        // The reason rho_max has an interior optimum at all.
        let (g, d) = (k3_reference(), dspark());
        let a_gain = d.accepted_at_width(12) / d.accepted_at_width(6);
        let u_gain = g.expert_union(12) / g.expert_union(6);
        assert!(a_gain < u_gain, "A(T) must saturate faster than u(T) grows");
    }

    #[test]
    fn routed_term_is_carried_not_dropped() {
        // v1's error: scaling the routed-zeroed ceiling by T. Adding routed
        // bytes must always reduce the achievable rate.
        let (g, p, d) = (
            k3_reference(),
            ServingPremises {
                dense: DENSE_4_25,
                ..Default::default()
            },
            dspark(),
        );
        let st = StateTraffic::default();
        let with_routed = evaluate(&g, &p, &d, 4, PhysicalReuse::assumed_ideal(3.9), st, 20.0);
        let without = evaluate(&g, &p, &d, 4, PhysicalReuse::assumed_ideal(0.0), st, 20.0);
        assert!(with_routed.tok_s < without.tok_s);
    }

    #[test]
    fn twenty_tok_s_needs_far_more_than_the_withdrawn_block_length() {
        // The withdrawn claim was L ~= 1.6. Carrying the routed term, no width
        // near that reaches the target under banked retention.
        let (g, p, d) = (
            k3_reference(),
            ServingPremises {
                dense: DENSE_4_25,
                ..Default::default()
            },
            dspark(),
        );
        let rows = sweep(&g, &p, &d, &[2, 3, 4], StateTraffic::default(), 20.0);
        for r in &rows {
            assert!(
                r.tok_s < 20.0,
                "width {} reached {} tok/s",
                r.width,
                r.tok_s
            );
        }
    }

    #[test]
    fn rho_max_has_an_interior_optimum() {
        let (g, p, d) = (
            k3_reference(),
            ServingPremises {
                dense: DENSE_4_25,
                ..Default::default()
            },
            dspark(),
        );
        let widths: Vec<u32> = (1..=12).collect();
        let rows = sweep(&g, &p, &d, &widths, StateTraffic::default(), 20.0);
        let best = rows
            .iter()
            .filter(|r| r.rho_max.is_some())
            .max_by(|a, b| a.rho_max.partial_cmp(&b.rho_max).unwrap())
            .expect("some width must be feasible");
        assert!(
            best.width > 1 && best.width < 12,
            "optimum should be interior, got width {}",
            best.width
        );
    }

    #[test]
    fn physical_reuse_is_unmeasured_until_someone_measures_it() {
        // R6: the ideal is an assumption and the record must say so.
        let ideal = PhysicalReuse::assumed_ideal(3.0);
        assert!(!ideal.measured);
        let (g, p, d) = (
            k3_reference(),
            ServingPremises {
                dense: DENSE_4_25,
                ..Default::default()
            },
            dspark(),
        );
        let row = evaluate(&g, &p, &d, 4, ideal, StateTraffic::default(), 20.0);
        assert!(row.rests_on_assumptions);
    }

    #[test]
    fn a_token_loop_destroys_the_amortisation() {
        // Without grouping and block GEMM, speculation improves orchestration
        // and leaves weight traffic nearly unchanged.
        let (g, p, d) = (
            k3_reference(),
            ServingPremises {
                dense: DENSE_4_25,
                ..Default::default()
            },
            dspark(),
        );
        let st = StateTraffic::default();
        let grouped = evaluate(&g, &p, &d, 6, PhysicalReuse::assumed_ideal(5.74), st, 20.0);
        let looped = evaluate(&g, &p, &d, 6, PhysicalReuse::token_loop(6), st, 20.0);
        assert!(looped.tok_s < grouped.tok_s);
    }

    #[test]
    fn state_traffic_lowers_the_optimum_when_it_is_priced() {
        // S(T) scales with width while A(T) saturates, so pricing it must not
        // move the optimum upward.
        let (g, p, d) = (
            k3_reference(),
            ServingPremises {
                dense: DENSE_4_25,
                ..Default::default()
            },
            dspark(),
        );
        let widths: Vec<u32> = (1..=12).collect();
        let best = |st: StateTraffic| {
            sweep(&g, &p, &d, &widths, st, 20.0)
                .into_iter()
                .filter(|r| r.rho_max.is_some())
                .max_by(|a, b| a.rho_max.partial_cmp(&b.rho_max).unwrap())
                .unwrap()
                .width
        };
        let free = best(StateTraffic::default());
        let priced = best(StateTraffic::prefix_state_ladder(69, 4));
        assert!(
            priced <= free,
            "pricing state traffic must not raise the optimum"
        );
    }

    #[test]
    fn dense_alone_refuses_when_acceptance_is_too_low() {
        let (g, p) = (
            k3_reference(),
            ServingPremises {
                dense: DENSE_4_25,
                ..Default::default()
            },
        );
        let (r, st) = (PhysicalReuse::assumed_ideal(1.0), StateTraffic::default());
        assert!(dense_alone_refuses(&g, &p, 1.0, 7, r, st, 20.0));
        assert!(!dense_alone_refuses(&g, &p, 3.85, 7, r, st, 20.0));
    }
}
