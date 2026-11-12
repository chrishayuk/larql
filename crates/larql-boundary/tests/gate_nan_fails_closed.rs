//! A boundary whose confidence could not be computed is not a safe one.
//!
//! The gate called a boundary fragile when its margin or its top-1
//! probability was *below* a threshold. A NaN is not below anything, so a
//! boundary with a NaN margin or probability (a non-finite logit upstream)
//! was not fragile, and the live gate emitted it as a compressed frame
//! under the high-margin contract.

use larql_boundary::frame::{BoundaryAgreement, FallbackPolicy};
use larql_boundary::gate::{apply, BoundaryDecision, BoundaryGateConfig};
use larql_boundary::metadata::BoundaryMetadata;

/// Values comfortably above the default thresholds.
const SAFE_MARGIN: f32 = 5.0;
const SAFE_PROB: f32 = 0.9;

fn metadata(log_prob_margin: f32, top1_prob: f32) -> BoundaryMetadata {
    BoundaryMetadata {
        raw_top1_token: 42,
        compressed_top1_token: Some(42),
        boundary_agreement: BoundaryAgreement::Agrees,
        raw_logit_margin: SAFE_MARGIN,
        raw_log_prob_margin: log_prob_margin,
        raw_top1_prob: top1_prob,
        codec_fragile: false,
        boundary_fragile: false,
    }
}

/// The gate as it runs once calibration mode is switched off.
fn live(fallback_policy: FallbackPolicy) -> BoundaryGateConfig {
    BoundaryGateConfig {
        calibration_mode: false,
        fallback_policy,
        ..Default::default()
    }
}

#[test]
fn a_confident_boundary_is_emitted_compressed() {
    // The control: the gate does not refuse everything.
    let mut meta = metadata(SAFE_MARGIN, SAFE_PROB);

    let decision = apply(&mut meta, &live(FallbackPolicy::Bf16Boundary));

    assert!(matches!(decision, BoundaryDecision::CompressedOk { .. }));
    assert!(!meta.boundary_fragile);
}

#[test]
fn a_nan_margin_is_fragile_and_falls_back() {
    let mut meta = metadata(f32::NAN, SAFE_PROB);

    let decision = apply(&mut meta, &live(FallbackPolicy::Bf16Boundary));

    assert_eq!(decision, BoundaryDecision::UseBf16);
    assert!(meta.boundary_fragile);
}

#[test]
fn a_nan_probability_is_fragile_and_falls_back() {
    let mut meta = metadata(SAFE_MARGIN, f32::NAN);

    let decision = apply(&mut meta, &live(FallbackPolicy::RejectIfUnsafe));

    assert_eq!(decision, BoundaryDecision::Reject);
    assert!(meta.boundary_fragile);
}

#[test]
fn a_nan_is_recorded_as_fragile_in_calibration_mode_too() {
    // Calibration mode always answers bf16, and still records fragility
    // for the telemetry the thresholds are fitted from.
    let mut meta = metadata(f32::NAN, f32::NAN);

    let decision = apply(&mut meta, &BoundaryGateConfig::default());

    assert_eq!(decision, BoundaryDecision::UseBf16);
    assert!(meta.boundary_fragile);
}

#[test]
fn a_nan_threshold_passes_nothing() {
    // A threshold that is not a number cannot be cleared.
    let mut meta = metadata(SAFE_MARGIN, SAFE_PROB);
    let config = BoundaryGateConfig {
        min_log_prob_margin: f32::NAN,
        ..live(FallbackPolicy::Bf16Boundary)
    };

    assert_eq!(apply(&mut meta, &config), BoundaryDecision::UseBf16);
}

#[test]
fn an_infinite_margin_is_judged_by_its_sign() {
    let policy = || live(FallbackPolicy::Bf16Boundary);

    let mut positive = metadata(f32::INFINITY, SAFE_PROB);
    let mut negative = metadata(f32::NEG_INFINITY, SAFE_PROB);

    assert!(matches!(
        apply(&mut positive, &policy()),
        BoundaryDecision::CompressedOk { .. }
    ));
    assert_eq!(apply(&mut negative, &policy()), BoundaryDecision::UseBf16);
}
