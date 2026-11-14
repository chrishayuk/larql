//! Phase 3 — confidence gate.
//!
//! [`apply`] reads per-boundary metadata and returns a [`BoundaryDecision`]:
//!
//! ```text
//! codec_fragile  = true  →  hard reject (codec moved the argmax)
//! boundary_fragile = true  →  soft reject (model uncertain at this position)
//! both false             →  CompressedOk { contract: D-@high }
//! ```
//!
//! # Fragility taxonomy
//!
//! Two independent properties — do not conflate them:
//!
//! **Codec fragility** (`codec_fragile = true`): `compressed_agrees == Disagrees`.
//! The codec changed the argmax. This is a hard reject regardless of margin.
//! Switching to bf16 eliminates codec fragility entirely.
//!
//! **Boundary fragility** (`boundary_fragile = true`): `raw_margin < threshold`.
//! The *uncompressed* model is uncertain at this position. Even a bf16 boundary
//! here is fragile in the sense that any small perturbation can flip the cascade.
//! Changing codec does not fix boundary fragility — it is a property of the
//! model state at this position.
//!
//! # Calibration mode
//!
//! When `calibration_mode = true` (the default), the gate ignores all
//! thresholds and always returns `UseBf16`. It still computes and sets
//! `boundary_fragile` so callers can collect telemetry for Track A.
//! Set `calibration_mode = false` only after Exp 44 ships fitted thresholds.

use crate::frame::{BoundaryAgreement, BoundaryContract, FallbackPolicy};
use crate::metadata::BoundaryMetadata;

/// Configuration for the per-boundary gate.
///
/// All threshold fields are marked UNCALIBRATED until Exp 44 Track A
/// ships fitted values. Do not treat the defaults as authoritative.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct BoundaryGateConfig {
    /// Primary gate threshold in **log-probability margin units**.
    ///
    /// Compare against `BoundaryMetadata::raw_log_prob_margin`, which is
    /// `log_softmax[top1] - log_softmax[top2]`. Do NOT use a raw-logit value
    /// here — raw logit scales vary across positions and are not comparable.
    ///
    /// Calibrated by Exp 44 Track A: **2.16** for Gemma 3 4B (Frankenstein corpus,
    /// 396 boundary positions, 90 continuation tests, 20 tokens each).
    /// At 2.16: accept=68.9%, early-div=4.8%, total-div=19.8%, system=1.69×.
    ///
    /// **UNCALIBRATED default: 1.0** — safe but conservative. Set to 2.16 once
    /// `calibration_mode = false` to use the fitted value.
    pub min_log_prob_margin: f32,

    /// Secondary floor on the raw top-1 softmax probability.
    ///
    /// Used alongside `min_log_prob_margin`: a boundary with high margin but low
    /// top-1 probability (flat distribution with a slight leader) is more fragile
    /// than the margin alone suggests.
    ///
    /// **UNCALIBRATED default: 0.5.**
    pub min_top1_prob: f32,

    /// Reject compressed frames where `boundary_agreement != Agrees`.
    pub require_compressed_agreement: bool,

    /// Action when the gate rejects.
    pub fallback_policy: FallbackPolicy,

    /// When `true` (the default): override all thresholds, always return
    /// `UseBf16`, emit `BoundaryContract::Calibrating` on all frames.
    ///
    /// Flip to `false` only after Exp 44 calibration is complete.
    pub calibration_mode: bool,
}

impl Default for BoundaryGateConfig {
    fn default() -> Self {
        Self {
            min_log_prob_margin: 1.0, // conservative pre-calibration placeholder
            min_top1_prob: 0.5,       // UNCALIBRATED
            require_compressed_agreement: true,
            fallback_policy: FallbackPolicy::Bf16Boundary,
            calibration_mode: true, // conservative default
        }
    }
}

/// Result of applying the gate to one boundary.
#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
pub enum BoundaryDecision {
    /// Emit compressed frame with the specified contract.
    CompressedOk { contract: BoundaryContract },
    /// Use bf16 (uncompressed) frame.
    UseBf16,
    /// Use cold-replay reference.
    UseColdReplay,
    /// Reject this boundary entirely.
    Reject,
}

/// Apply the gate to one boundary's metadata.
///
/// Mutates `metadata.boundary_fragile` in place so the caller can log it
/// regardless of the decision.
pub fn apply(metadata: &mut BoundaryMetadata, config: &BoundaryGateConfig) -> BoundaryDecision {
    // Calibration mode: always bf16, but still populate fragile flag for telemetry.
    if config.calibration_mode {
        metadata.boundary_fragile = is_fragile(metadata, config);
        return BoundaryDecision::UseBf16;
    }

    // Hard reject: codec moved the argmax, or sender didn't check.
    // Agree is the only non-reject state; Disagrees and NotChecked both fail.
    // Fragility is still evaluated: it is a property of the raw model state,
    // independent of the codec, and Track A threshold fitting counts exactly
    // this rejected population — stamping `false` here would undercount it.
    let agreement_fails = !matches!(metadata.boundary_agreement, BoundaryAgreement::Agrees);
    if config.require_compressed_agreement && agreement_fails {
        metadata.boundary_fragile = is_fragile(metadata, config);
        return to_fallback(config);
    }

    // Soft reject: model uncertain at this boundary position.
    let fragile = is_fragile(metadata, config);
    metadata.boundary_fragile = fragile;
    if fragile {
        return to_fallback(config);
    }

    // Safe: assign contract band (tied to gate threshold — see spec §6).
    BoundaryDecision::CompressedOk {
        contract: BoundaryContract::ArgmaxNearEquivalentHighMargin,
    }
}

fn is_fragile(meta: &BoundaryMetadata, config: &BoundaryGateConfig) -> bool {
    // Compare log-prob margin against the log-prob threshold.
    // Using raw_logit_margin here would be wrong — logit scales are not
    // comparable across positions.
    //
    // Written as "not known to clear both thresholds" so that a NaN is
    // fragile. `margin < threshold` is false for NaN, and a boundary whose
    // margin or probability could not be computed was passed as safe.
    let clears_both = meta.raw_log_prob_margin >= config.min_log_prob_margin
        && meta.raw_top1_prob >= config.min_top1_prob;
    !clears_both
}

fn to_fallback(config: &BoundaryGateConfig) -> BoundaryDecision {
    match config.fallback_policy {
        FallbackPolicy::None | FallbackPolicy::RejectIfUnsafe => BoundaryDecision::Reject,
        FallbackPolicy::Bf16Boundary => BoundaryDecision::UseBf16,
        FallbackPolicy::ColdReplay => BoundaryDecision::UseColdReplay,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::frame::BoundaryAgreement;
    use crate::metadata::BoundaryMetadata;

    fn meta(logit_margin: f32, top1_prob: f32, agreement: BoundaryAgreement) -> BoundaryMetadata {
        BoundaryMetadata {
            raw_top1_token: 42,
            compressed_top1_token: Some(42),
            boundary_agreement: agreement.clone(),
            raw_logit_margin: logit_margin,
            raw_log_prob_margin: logit_margin * 0.9, // approximate; exact value from compute()
            raw_top1_prob: top1_prob,
            codec_fragile: matches!(agreement, BoundaryAgreement::Disagrees),
            boundary_fragile: false,
        }
    }

    fn live() -> BoundaryGateConfig {
        BoundaryGateConfig {
            calibration_mode: false,
            ..Default::default()
        }
    }

    #[test]
    fn calibration_mode_always_bf16() {
        // Default config has calibration_mode = true.
        let config = BoundaryGateConfig::default();
        let mut m = meta(10.0, 0.99, BoundaryAgreement::Agrees);
        assert_eq!(apply(&mut m, &config), BoundaryDecision::UseBf16);
    }

    #[test]
    fn disagrees_hard_rejects() {
        let config = live();
        let mut m = meta(5.0, 0.9, BoundaryAgreement::Disagrees);
        assert_eq!(apply(&mut m, &config), BoundaryDecision::UseBf16);
    }

    #[test]
    fn not_checked_hard_rejects() {
        let config = live();
        let mut m = meta(5.0, 0.9, BoundaryAgreement::NotChecked);
        assert_eq!(apply(&mut m, &config), BoundaryDecision::UseBf16);
    }

    #[test]
    fn low_margin_is_boundary_fragile() {
        let mut config = live();
        config.min_log_prob_margin = 2.0;
        let mut m = meta(0.5, 0.9, BoundaryAgreement::Agrees);
        let decision = apply(&mut m, &config);
        assert!(m.boundary_fragile, "expected boundary_fragile = true");
        assert_eq!(decision, BoundaryDecision::UseBf16);
    }

    #[test]
    fn hard_reject_populates_boundary_fragile_for_telemetry() {
        // Track A threshold fitting counts fragile boundaries; the hard-reject
        // path must evaluate fragility like every other path, not stamp false.
        let mut config = live();
        config.min_log_prob_margin = 2.0;
        // Disagrees → hard reject; margin below threshold → position IS fragile.
        let mut m = meta(0.5, 0.9, BoundaryAgreement::Disagrees);
        let decision = apply(&mut m, &config);
        assert_eq!(decision, BoundaryDecision::UseBf16);
        assert!(
            m.boundary_fragile,
            "hard-reject path must still evaluate fragility for telemetry"
        );
    }

    #[test]
    fn hard_reject_confident_boundary_is_not_fragile() {
        // Companion: hard reject with a confident margin stays non-fragile.
        let config = live();
        let mut m = meta(5.0, 0.9, BoundaryAgreement::Disagrees);
        assert_eq!(apply(&mut m, &config), BoundaryDecision::UseBf16);
        assert!(!m.boundary_fragile);
    }

    #[test]
    fn confident_boundary_compresses() {
        let config = live();
        let mut m = meta(3.0, 0.8, BoundaryAgreement::Agrees);
        let decision = apply(&mut m, &config);
        assert!(!m.boundary_fragile);
        assert!(matches!(decision, BoundaryDecision::CompressedOk { .. }));
    }

    #[test]
    fn cold_replay_fallback() {
        let mut config = live();
        config.fallback_policy = FallbackPolicy::ColdReplay;
        let mut m = meta(0.1, 0.3, BoundaryAgreement::Agrees);
        assert_eq!(apply(&mut m, &config), BoundaryDecision::UseColdReplay);
    }

    #[test]
    fn reject_if_unsafe_fallback() {
        let mut config = live();
        config.fallback_policy = FallbackPolicy::RejectIfUnsafe;
        let mut m = meta(0.1, 0.3, BoundaryAgreement::Agrees);
        assert_eq!(apply(&mut m, &config), BoundaryDecision::Reject);
    }
}
