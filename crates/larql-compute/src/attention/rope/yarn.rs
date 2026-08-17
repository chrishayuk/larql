//! YaRN RoPE scaling — frequency ramp plus attention amplitude.
//!
//! Mirrors `_compute_yarn_parameters` in `transformers`'
//! `modeling_rope_utils.py`. YaRN is the only scaling in this crate that
//! returns **two** things, and the second one is the one that bites:
//!
//! * `inv_freq` — a per-dimension blend of the unscaled *extrapolation*
//!   frequency and the `÷ factor` *interpolation* frequency, ramped linearly
//!   between the `beta_fast` and `beta_slow` correction bounds.
//! * `amplitude` — a scalar multiplying **both** `cos` and `sin`. That is not
//!   a rotation: it rescales Q and K, so it changes attention logits at every
//!   position including position 0. A model served without it runs attention
//!   at the wrong temperature everywhere, not merely at long range.
//!
//! For `openai/gpt-oss-20b` (`factor: 32`, `truncate: false`) the amplitude is
//! 1.3466 and 23 of 32 rotary dimensions carry a rescaled frequency, 14 of
//! them by the full 32×.

use larql_models::YarnRopeScaling;

/// Amplitude applied to `cos`/`sin` when no scaling asks for one.
pub const UNIT_AMPLITUDE: f64 = 1.0;

/// The attention amplitude for a YaRN block — delegates to the config
/// crate's [`YarnRopeScaling::attention_amplitude`], the single authority
/// (the VINDEX3 container carries the block and is judged against the
/// same number).
pub fn attention_amplitude(s: &YarnRopeScaling) -> f64 {
    s.attention_amplitude()
}

/// HF's `find_correction_dim` — the dimension index at which the rotary
/// frequency completes `num_rotations` full turns over `max_position`.
fn correction_dim(num_rotations: f64, dim: f64, base: f64, max_position: f64) -> f64 {
    (dim * (max_position / (num_rotations * std::f64::consts::TAU)).ln()) / (2.0 * base.ln())
}

/// HF's `find_correction_range`, returning `(low, high)` dimension bounds.
///
/// Called with `low_rot = beta_fast` and `high_rot = beta_slow`; since
/// `correction_dim` decreases in `num_rotations` and `beta_fast > beta_slow`,
/// the result is ordered `low <= high`.
fn correction_range(s: &YarnRopeScaling, dim: f64, base: f64) -> (f64, f64) {
    let max_position = s.original_max_position_embeddings;
    let mut low = correction_dim(s.beta_fast, dim, base, max_position);
    let mut high = correction_dim(s.beta_slow, dim, base, max_position);
    if s.truncate {
        low = low.floor();
        high = high.ceil();
    }
    (low.max(0.0), high.min(dim - 1.0))
}

/// HF's `linear_ramp_factor`: a clamped 0→1 ramp across `n` entries.
///
/// The `min == max` singularity is nudged exactly as HF nudges it, so a
/// degenerate correction range produces the same near-step ramp on both
/// sides rather than a NaN here and a step there.
fn linear_ramp(min: f64, max: f64, n: usize) -> Vec<f64> {
    let max = if min == max { max + 0.001 } else { max };
    (0..n)
        .map(|i| ((i as f64 - min) / (max - min)).clamp(0.0, 1.0))
        .collect()
}

/// YaRN-scaled inverse frequencies and the amplitude that accompanies them.
///
/// `rotary_dim` is the rotated width of one head (`head_dim ×
/// partial_rotary_factor`), matching HF's `dim`. `base_inv_freq` must be the
/// standard `1 / base^(2i/rotary_dim)` series of length `rotary_dim / 2` —
/// it is passed in rather than recomputed so the caller's `rotary_dim`
/// rounding is the single source of truth for both.
pub fn yarn_inv_freq(
    s: &YarnRopeScaling,
    base_inv_freq: &[f64],
    rotary_dim: usize,
    base: f64,
) -> (Vec<f64>, f64) {
    let dim = rotary_dim as f64;
    let (low, high) = correction_range(s, dim, base);
    let ramp = linear_ramp(low, high, base_inv_freq.len());

    let inv_freq = base_inv_freq
        .iter()
        .zip(ramp.iter())
        .map(|(&extrapolation, &r)| {
            // `1 - ramp` is HF's `inv_freq_extrapolation_factor`: 1 means
            // "keep the unscaled frequency", 0 means "fully interpolate".
            let extrapolation_factor = 1.0 - r;
            let interpolation = extrapolation / s.factor;
            interpolation * (1.0 - extrapolation_factor) + extrapolation * extrapolation_factor
        })
        .collect();

    (inv_freq, attention_amplitude(s))
}

#[cfg(test)]
mod tests {
    use super::*;

    /// `openai/gpt-oss-20b`'s block, verbatim from its `config.json`.
    fn gpt_oss() -> YarnRopeScaling {
        YarnRopeScaling {
            factor: 32.0,
            beta_fast: 32.0,
            beta_slow: 1.0,
            original_max_position_embeddings: 4096.0,
            truncate: false,
            mscale: None,
            mscale_all_dim: None,
        }
    }

    fn base_inv_freq(rotary_dim: usize, base: f64) -> Vec<f64> {
        (0..rotary_dim / 2)
            .map(|i| 1.0 / base.powf(2.0 * i as f64 / rotary_dim as f64))
            .collect()
    }

    /// The number larql was missing entirely. Pinned against
    /// `transformers.modeling_rope_utils.ROPE_INIT_FUNCTIONS['yarn']` run on
    /// the real checkpoint config, which returned 1.3465735902799727.
    #[test]
    fn gpt_oss_amplitude_matches_reference() {
        let amp = attention_amplitude(&gpt_oss());
        assert!(
            (amp - 1.346_573_590_279_972_7).abs() < 1e-12,
            "amplitude {amp} != reference 1.3465735902799727"
        );
    }

    /// An amplitude of 1.0 would be indistinguishable from "YaRN not applied",
    /// which is precisely the bug this module fixes. Guard the distinction.
    #[test]
    fn gpt_oss_amplitude_is_not_unity() {
        assert!((attention_amplitude(&gpt_oss()) - UNIT_AMPLITUDE).abs() > 0.3);
    }

    /// DeepSeek's paired form collapses to exactly 1.0 when the two mscales
    /// agree. Applying the single-argument form there would give 1.35 — a
    /// 35 % error introduced *by* honouring yarn, which is why the pair is
    /// parsed rather than ignored.
    #[test]
    fn paired_mscale_is_a_ratio_not_the_single_argument_form() {
        let s = YarnRopeScaling {
            mscale: Some(1.0),
            mscale_all_dim: Some(1.0),
            ..gpt_oss()
        };
        assert!((attention_amplitude(&s) - 1.0).abs() < 1e-12);
        assert!((attention_amplitude(&gpt_oss()) - 1.0).abs() > 0.3);
    }

    /// A factor that does not extend context must not rescale attention.
    #[test]
    fn factor_of_one_leaves_amplitude_unscaled() {
        let s = YarnRopeScaling {
            factor: 1.0,
            ..gpt_oss()
        };
        assert!((attention_amplitude(&s) - UNIT_AMPLITUDE).abs() < 1e-15);
    }

    /// Per-dimension ratios against the plain series, pinned to the values
    /// `ROPE_INIT_FUNCTIONS['yarn']` produced for gpt-oss-20b: the first 9
    /// dims untouched, the last 14 fully interpolated at 1/32, and a
    /// monotonic ramp between.
    #[test]
    fn gpt_oss_inv_freq_ramp_matches_reference() {
        let (rotary_dim, base) = (64usize, 150_000.0);
        let plain = base_inv_freq(rotary_dim, base);
        let (scaled, _) = yarn_inv_freq(&gpt_oss(), &plain, rotary_dim, base);

        let ratio: Vec<f64> = scaled.iter().zip(&plain).map(|(s, p)| s / p).collect();
        assert_eq!(ratio.len(), 32);

        let untouched = ratio.iter().filter(|r| **r > 0.999).count();
        assert_eq!(
            untouched, 9,
            "expected 9 extrapolated dims, got {untouched}"
        );

        let interpolated = ratio.iter().filter(|r| **r < 0.0313).count();
        assert_eq!(
            interpolated, 14,
            "expected 14 fully-interpolated dims, got {interpolated}"
        );

        assert!(
            (ratio[0] - 1.0).abs() < 1e-12,
            "dim 0 must be pure extrapolation, got {}",
            ratio[0]
        );
        assert!(
            (ratio[31] - 1.0 / 32.0).abs() < 1e-6,
            "dim 31 must be pure interpolation, got {}",
            ratio[31]
        );
        assert!(
            ratio.windows(2).all(|w| w[0] >= w[1] - 1e-12),
            "ramp must be monotonically non-increasing"
        );
    }

    /// The whole point: a plain series and a YaRN series must differ. Without
    /// this, a scaling that silently no-ops looks identical to a correct one.
    #[test]
    fn yarn_actually_changes_the_frequencies() {
        let (rotary_dim, base) = (64usize, 150_000.0);
        let plain = base_inv_freq(rotary_dim, base);
        let (scaled, _) = yarn_inv_freq(&gpt_oss(), &plain, rotary_dim, base);
        let changed = plain
            .iter()
            .zip(&scaled)
            .filter(|(p, s)| (*p - *s).abs() > 1e-12)
            .count();
        assert_eq!(
            changed, 23,
            "expected 23 of 32 dims rescaled, got {changed}"
        );
    }

    /// `truncate` rounds the correction bounds outward, so it moves where the
    /// ramp starts and ends. GPT-OSS ships `false`; inheriting HF's `true`
    /// default would be a quiet, small, everywhere-wrong difference.
    #[test]
    fn truncate_changes_the_ramp() {
        let (rotary_dim, base) = (64usize, 150_000.0);
        let plain = base_inv_freq(rotary_dim, base);
        let (a, _) = yarn_inv_freq(&gpt_oss(), &plain, rotary_dim, base);
        let (b, _) = yarn_inv_freq(
            &YarnRopeScaling {
                truncate: true,
                ..gpt_oss()
            },
            &plain,
            rotary_dim,
            base,
        );
        assert!(
            a.iter().zip(&b).any(|(x, y)| (x - y).abs() > 1e-12),
            "truncate must change the ramp"
        );
    }

    /// A degenerate correction range must produce a finite ramp, not NaN.
    #[test]
    fn degenerate_correction_range_stays_finite() {
        let s = YarnRopeScaling {
            beta_fast: 4.0,
            beta_slow: 4.0,
            ..gpt_oss()
        };
        let (rotary_dim, base) = (64usize, 150_000.0);
        let plain = base_inv_freq(rotary_dim, base);
        let (scaled, amp) = yarn_inv_freq(&s, &plain, rotary_dim, base);
        assert!(scaled.iter().all(|v| v.is_finite()), "{scaled:?}");
        assert!(amp.is_finite());
    }

    /// Scaled frequencies stay ordered — a ramp that inverted the series
    /// would reorder the rotary bands without changing any summary statistic.
    #[test]
    fn scaled_inv_freq_stays_monotonic() {
        let (rotary_dim, base) = (64usize, 150_000.0);
        let plain = base_inv_freq(rotary_dim, base);
        let (scaled, _) = yarn_inv_freq(&gpt_oss(), &plain, rotary_dim, base);
        assert!(scaled.windows(2).all(|w| w[0] >= w[1]));
    }
}
