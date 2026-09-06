//! How an expert's gate and up projections combine into the down
//! projection's input.
//!
//! Two policies, selected by [`ExpertGatePolicy`] on the architecture. The
//! interesting one is GPT-OSS's, which is *not* SwiGLU and whose deviations
//! all push in the same direction — larger outputs — so approximating it with
//! SiLU produces a forward pass that looks healthy and predicts badly.

use larql_models::{Activation, ExpertGatePolicy};
use ndarray::Array2;

use crate::ffn::{gelu_tanh_gate_up, silu_gate_up};

/// Combine `gate` and `up` into the activation fed to the down projection.
///
/// Both inputs are `[tokens, intermediate]` and already carry their biases.
pub fn apply(
    gate: &Array2<f32>,
    up: &Array2<f32>,
    policy: ExpertGatePolicy,
    activation: Activation,
) -> Array2<f32> {
    match policy {
        ExpertGatePolicy::Gated => {
            if activation.uses_gelu_tanh_gate_up() {
                gelu_tanh_gate_up(gate, up)
            } else {
                silu_gate_up(gate, up)
            }
        }
        ExpertGatePolicy::ClampedGlu { limit, alpha } => clamped_glu(gate, up, limit, alpha),
        ExpertGatePolicy::SituGlu { beta, linear_beta } => {
            elementwise(gate, up, crate::MoeGateRule::SituGlu { beta, linear_beta })
        }
        // GLM-5.3-Flash: GPT-OSS's clamp, then ORDINARY SwiGLU.
        // Through the scalar rule, like SiTU-GLU above, so the combine
        // math has one authority.
        ExpertGatePolicy::ClampedGated { limit } => elementwise(
            gate,
            up,
            crate::MoeGateRule::ClampedGated {
                limit,
                activation: activation.into(),
            },
        ),
    }
}

/// GPT-OSS's expert gating, transcribed from `GptOssExperts._apply_gate`:
///
/// ```text
/// g   = gate.clamp(min=None, max=limit)    // upper bound only
/// u   = up.clamp(-limit, limit)            // symmetric
/// glu = g * sigmoid(g * alpha)
/// out = (u + 1) * glu
/// ```
///
/// Three details that each change the numbers and none of which a plain
/// `silu(gate) * up` reproduces: the gate's clamp is **one-sided** (there is
/// no lower bound), `alpha` scales the sigmoid's argument rather than the
/// value, and the up branch is offset by one — so an `up` of exactly zero
/// still passes `glu` through instead of annihilating it.
///
/// The scalar math is owned by [`crate::MoeGateRule::combine`], which the
/// quantised slice paths use directly — this wrapper only shapes it over
/// `Array2`, so the PyTorch-pinned tests below cover both tiers.
fn clamped_glu(gate: &Array2<f32>, up: &Array2<f32>, limit: f32, alpha: f32) -> Array2<f32> {
    elementwise(gate, up, crate::MoeGateRule::ClampedGlu { limit, alpha })
}

/// Shape one scalar combine rule over `[tokens, intermediate]`.
///
/// Every non-plain policy routes through here rather than reimplementing
/// its formula on `Array2`: the arithmetic authority is
/// [`crate::MoeGateRule::combine`] and this only iterates, so the tests
/// that pin the scalar rule against its reference cover this tier too.
fn elementwise(gate: &Array2<f32>, up: &Array2<f32>, rule: crate::MoeGateRule) -> Array2<f32> {
    let mut out = Array2::<f32>::zeros(gate.raw_dim());
    for ((o, &g), &u) in out.iter_mut().zip(gate.iter()).zip(up.iter()) {
        *o = rule.combine(g, u);
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ffn::sigmoid;
    use ndarray::arr2;

    const LIMIT: f32 = 7.0;
    const ALPHA: f32 = 1.702;

    fn clamped() -> ExpertGatePolicy {
        ExpertGatePolicy::ClampedGlu {
            limit: LIMIT,
            alpha: ALPHA,
        }
    }

    /// Values produced by running `GptOssExperts._apply_gate` itself in
    /// PyTorch on these inputs, hardcoded here so the assertion does not
    /// restate our own formula. See `docs/k3-funnel.md` §4.7.
    ///
    /// The inputs are chosen to exercise every branch: a negative gate (no
    /// lower clamp), a zero gate (GLU vanishes whatever `up` is), a normal
    /// pair, and a pair that trips both clamps at once.
    #[test]
    fn clamped_glu_matches_reference_implementation() {
        let gate = arr2(&[[-2.0f32, 0.0, 1.5, 9.0]]);
        let up = arr2(&[[0.5f32, -1.0, 2.0, -9.0]]);
        let expected = [-0.0965121_f32, 0.0, 4.174_987, -41.999_718];

        let got = apply(&gate, &up, clamped(), Activation::Silu);
        for (&o, &want) in got.iter().zip(expected.iter()) {
            assert!((o - want).abs() < 1e-5, "got {o}, reference says {want}");
        }
    }

    /// The same inputs under plain SwiGLU, to show the policies are not
    /// interchangeable: three of the four elements move, one by 40×.
    #[test]
    fn plain_swiglu_would_give_materially_different_values() {
        let gate = arr2(&[[-2.0f32, 0.0, 1.5, 9.0]]);
        let up = arr2(&[[0.5f32, -1.0, 2.0, -9.0]]);
        let clamped_out = apply(&gate, &up, clamped(), Activation::Silu);
        let swiglu_out = apply(&gate, &up, ExpertGatePolicy::Gated, Activation::Silu);
        let max_diff = clamped_out
            .iter()
            .zip(swiglu_out.iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f32, f32::max);
        assert!(max_diff > 30.0, "max divergence was only {max_diff}");
    }

    /// The gate clamp is one-sided, and the observable consequence is not
    /// magnitude but *monotonicity*: past `-limit` the true GLU keeps decaying
    /// toward zero, whereas a symmetric clamp would pin every value below
    /// `-limit` to the same constant.
    ///
    /// Worth stating explicitly because the absolute difference is tiny — at
    /// `g = -7` the GLU has already saturated to about -4.7e-5 — so an
    /// absolute-tolerance assertion here would pass under either clamp and
    /// prove nothing. The asymmetry is faithful to the reference but
    /// numerically almost inert; this test pins the shape, not a magnitude.
    #[test]
    fn gate_clamp_has_no_lower_bound() {
        let up = arr2(&[[0.0f32, 0.0, 0.0]]);
        let gate = arr2(&[[-LIMIT, -2.0 * LIMIT, -4.0 * LIMIT]]);
        let got = apply(&gate, &up, clamped(), Activation::Silu);

        // Strictly decreasing in magnitude — a floored gate would be constant.
        assert!(
            got[[0, 0]].abs() > got[[0, 1]].abs(),
            "expected decay past -limit, got {} then {}",
            got[[0, 0]],
            got[[0, 1]]
        );
        assert!(got[[0, 1]].abs() > got[[0, 2]].abs());

        // And the value at exactly -limit is what a symmetric clamp would
        // wrongly return for all three.
        let floored = -LIMIT * sigmoid(-LIMIT * ALPHA);
        assert!((got[[0, 0]] - floored).abs() < 1e-9);
        assert!((got[[0, 2]] - floored).abs() > 1e-9);
    }

    /// The up branch is offset by one, so up = 0 still passes the GLU through.
    /// Plain SwiGLU would zero the whole element.
    #[test]
    fn up_offset_of_one_keeps_zero_up_alive() {
        let gate = arr2(&[[2.0f32]]);
        let up = arr2(&[[0.0f32]]);
        let got = apply(&gate, &up, clamped(), Activation::Silu);
        let glu = 2.0 * sigmoid(2.0 * ALPHA);
        assert!((got[[0, 0]] - glu).abs() < 1e-6);
        assert!(got[[0, 0]] > 1.0, "SwiGLU would give exactly 0 here");
    }

    /// alpha scales the sigmoid's argument. With alpha = 1 this would be SiLU;
    /// at 1.702 it must differ measurably.
    #[test]
    fn alpha_scales_the_sigmoid_argument() {
        let gate = arr2(&[[1.0f32]]);
        let up = arr2(&[[0.0f32]]);
        let got = apply(&gate, &up, clamped(), Activation::Silu)[[0, 0]];
        let silu = 1.0 * sigmoid(1.0);
        assert!(
            (got - silu).abs() > 0.05,
            "alpha=1.702 must not collapse to SiLU ({got} vs {silu})"
        );
        assert!((got - sigmoid(ALPHA)).abs() < 1e-6);
    }

    #[test]
    fn up_clamp_is_symmetric() {
        let gate = arr2(&[[1.0f32, 1.0]]);
        let up = arr2(&[[100.0f32, -100.0]]);
        let got = apply(&gate, &up, clamped(), Activation::Silu);
        let glu = 1.0 * sigmoid(ALPHA);
        assert!((got[[0, 0]] - (LIMIT + 1.0) * glu).abs() < 1e-6);
        assert!((got[[0, 1]] - (-LIMIT + 1.0) * glu).abs() < 1e-6);
    }

    #[test]
    fn gated_policy_is_plain_silu_gate_up() {
        let gate = arr2(&[[1.0f32, -2.0]]);
        let up = arr2(&[[3.0f32, 4.0]]);
        let got = apply(&gate, &up, ExpertGatePolicy::Gated, Activation::Silu);
        let want = silu_gate_up(&gate, &up);
        assert_eq!(got, want);
    }

    #[test]
    fn gated_policy_honours_gelu_tanh() {
        let gate = arr2(&[[1.0f32, -2.0]]);
        let up = arr2(&[[3.0f32, 4.0]]);
        let got = apply(&gate, &up, ExpertGatePolicy::Gated, Activation::GeluTanh);
        assert_eq!(got, gelu_tanh_gate_up(&gate, &up));
    }

    #[test]
    fn clamped_glu_preserves_shape() {
        let gate = Array2::<f32>::zeros((3, 5));
        let up = Array2::<f32>::zeros((3, 5));
        assert_eq!(
            apply(&gate, &up, clamped(), Activation::Silu).shape(),
            &[3, 5]
        );
    }
}

#[cfg(test)]
mod clamped_gated_tests {
    use super::*;
    use crate::MoeGateRule;
    use ndarray::arr2;

    const LIMIT: f32 = 10.0;

    fn silu(x: f32) -> f32 {
        x / (1.0 + (-x).exp())
    }

    /// GLM-5.3-Flash's combine: GPT-OSS's clamp, then ORDINARY SwiGLU.
    ///
    /// Pinned against a scalar definition written here, so the rule is
    /// checked against the formula rather than against itself.
    #[test]
    fn the_rule_computes_clamped_plain_swiglu() {
        let rule = MoeGateRule::ClampedGated {
            limit: LIMIT,
            activation: crate::Activation::Silu,
        };
        for (g, u) in [
            (0.5f32, 0.25f32),
            (-0.5, 0.25),
            (25.0, 0.5),  // gate above the cap
            (-25.0, 0.5), // gate BELOW -cap: not clamped, one-sided
            (0.5, 25.0),  // up above the cap
            (0.5, -25.0), // up below -cap: clamped, symmetric
        ] {
            let want = silu(g.min(LIMIT)) * u.clamp(-LIMIT, LIMIT);
            let got = rule.combine(g, u);
            assert!((got - want).abs() <= 1e-6, "g={g} u={u}: {got} vs {want}");
        }
    }

    /// **The clamp is ASYMMETRIC**, and the two halves are separated:
    /// the gate is bounded above only, the up branch on both sides. A
    /// symmetric gate clamp would agree on every input whose gate stays
    /// above `-limit`, which is why the control uses one that does not.
    #[test]
    fn the_gate_is_capped_above_only_and_the_up_branch_on_both_sides() {
        let rule = MoeGateRule::ClampedGated {
            limit: LIMIT,
            activation: crate::Activation::Silu,
        };
        // A gate far below -limit is NOT clamped.
        let deep = rule.combine(-40.0, 1.0);
        assert!(
            (deep - silu(-40.0)).abs() <= 1e-6,
            "the gate must not be clamped from below: {deep}"
        );
        assert_ne!(
            deep,
            silu(-LIMIT),
            "clamping the gate below would collapse this to silu(-limit)"
        );
        // The up branch IS clamped from below.
        assert_eq!(rule.combine(1.0, -40.0), rule.combine(1.0, -LIMIT));
    }

    /// It is NOT `ClampedGlu`. Same clamp, different arithmetic — the
    /// defect this variant exists to prevent, and the one that measured
    /// relative 31.7 on GLM's real 288-expert bank.
    #[test]
    fn it_differs_from_gpt_oss_clamped_glu_at_a_residual_scale_activation() {
        let gated = MoeGateRule::ClampedGated {
            limit: LIMIT,
            activation: crate::Activation::Silu,
        };
        let glu = MoeGateRule::ClampedGlu {
            limit: LIMIT,
            alpha: 1.0,
        };
        // `(u + 1) ~ 1` while `u ~ 0.03`, so the GPT-OSS form is larger
        // by roughly `1/|u|` — the ratio the real-bank measurement saw.
        let (g, u) = (0.2f32, 0.03f32);
        let a = gated.combine(g, u);
        let b = glu.combine(g, u);
        assert!(
            (b / a).abs() > 20.0,
            "the two forms must diverge by ~1/|u|: {a} vs {b}"
        );
    }

    /// The `Array2` tier delegates to the scalar rule, so the two agree
    /// by construction — asserted rather than assumed, because that
    /// delegation is what keeps one authority for the math.
    #[test]
    fn the_matrix_tier_agrees_with_the_scalar_rule() {
        let gate = arr2(&[[0.5f32, 25.0], [-25.0, 0.1]]);
        let up = arr2(&[[0.25f32, -25.0], [0.5, 2.0]]);
        let out = apply(
            &gate,
            &up,
            ExpertGatePolicy::ClampedGated { limit: LIMIT },
            Activation::Silu,
        );
        let rule = MoeGateRule::ClampedGated {
            limit: LIMIT,
            activation: crate::Activation::Silu,
        };
        for (o, (&g, &u)) in out.iter().zip(gate.iter().zip(up.iter())) {
            assert!((o - rule.combine(g, u)).abs() <= 1e-6, "g={g} u={u}");
        }
    }

    /// A GeLU-tanh family takes its own branch — the activation is not
    /// hard-coded to SiLU inside the rule.
    #[test]
    fn the_activation_is_read_rather_than_assumed() {
        let silu_rule = MoeGateRule::ClampedGated {
            limit: LIMIT,
            activation: crate::Activation::Silu,
        };
        let gelu_rule = MoeGateRule::ClampedGated {
            limit: LIMIT,
            activation: crate::Activation::GeluTanh,
        };
        assert_ne!(silu_rule.combine(0.7, 1.3), gelu_rule.combine(0.7, 1.3));
    }
}
