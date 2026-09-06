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
