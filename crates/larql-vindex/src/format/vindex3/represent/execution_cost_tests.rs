//! `tests` for [`super`].

use super::super::byte_ledger::ByteLedger;
use super::*;

/// The measured Kimi ledger, shared with the byte-ledger tests so the
/// two modules cannot drift apart on what the map costs.
fn kimi_ledger() -> ByteLedger {
    super::super::byte_ledger::tests::kimi_four_family()
}

/// A second observation, at a DIFFERENT breadth. Not measured — it
/// exists only to exercise the calibration rule, and says so.
fn hypothetical_at(id: &str, fraction: f64, beta: f64) -> ExecutionCostObservation {
    let baseline_bytes = 1_000_000_000u64;
    let baseline_gpu = 20.0;
    ExecutionCostObservation {
        id: id.into(),
        candidate_bytes_per_token: baseline_bytes - (baseline_bytes as f64 * fraction) as u64,
        baseline_bytes_per_token: baseline_bytes,
        baseline_gpu_ms_per_token: baseline_gpu,
        candidate_gpu_ms_per_token: baseline_gpu * (1.0 - beta * fraction),
        ..m3max_metal_001()
    }
}

// ── The observation itself. ──

#[test]
fn beta_is_derived_from_the_measurement_and_matches_the_bench() {
    let o = m3max_metal_001();
    assert!((o.byte_fraction_removed() - 0.15990).abs() < 1e-5);
    assert!((o.gpu_fraction_removed() - 0.12802).abs() < 1e-5);
    // 0.8006 in session 1; session 2 measured 0.7907 independently.
    let beta = o.bytes_to_gpu_factor().expect("bytes were removed");
    assert!((beta - 0.8006).abs() < 1e-3, "beta was {beta}");
    assert!(
        beta < 1.0,
        "a pure-bandwidth-bound decode would give 1.0; the shortfall IS the finding"
    );
}

#[test]
fn a_candidate_that_removes_nothing_has_no_beta() {
    // 0/0, and a NaN beta would propagate into every prediction.
    let o = ExecutionCostObservation {
        candidate_bytes_per_token: m3max_metal_001().baseline_bytes_per_token,
        ..m3max_metal_001()
    };
    assert_eq!(o.byte_fraction_removed(), 0.0);
    assert_eq!(o.bytes_to_gpu_factor(), None);
}

#[test]
fn the_observation_records_its_breadth_and_its_provenance() {
    let o = m3max_metal_001();
    // Breadth, so beta(breadth) is answerable later rather than assumed
    // flat now.
    assert_eq!(o.families_changed.len(), 4);
    assert_eq!(o.scopes_changed, 15);
    // Provenance a later reader can chase or invalidate.
    assert!(!o.compiler_commit.is_empty(), "a kernel change moves beta");
    assert_eq!(o.evidence.len(), 2, "both bench sessions");
    assert!(o.benchmark_protocol.contains("interleaved"));
}

// ── Calibration status is about BREADTH, not about count. ──

#[test]
fn one_observation_is_provisional() {
    let m = ExecutionCostModel::new(vec![m3max_metal_001()]);
    assert_eq!(m.status(), CalibrationStatus::Provisional);
}

#[test]
fn many_observations_at_one_breadth_are_still_provisional() {
    // The trap this rule exists to refuse: ten measurements at the same
    // byte fraction agree with each other and say nothing whatever
    // about whether beta varies with breadth.
    let m = ExecutionCostModel::new(
        (0..10)
            .map(|i| hypothetical_at(&format!("same-{i}"), 0.16, 0.80))
            .collect(),
    );
    assert_eq!(m.status(), CalibrationStatus::Provisional);
}

#[test]
fn two_separated_breadths_are_calibrated() {
    let m = ExecutionCostModel::new(vec![
        hypothetical_at("low", 0.08, 0.80),
        hypothetical_at("high", 0.32, 0.72),
    ]);
    assert_eq!(m.status(), CalibrationStatus::Calibrated);
}

#[test]
fn an_empty_model_is_provisional_and_refuses_to_predict() {
    let m = ExecutionCostModel::default();
    assert_eq!(m.status(), CalibrationStatus::Provisional);
    assert_eq!(m.predict(&kimi_ledger()), Err(CostRefusal::NoObservations));
}

// ── Prediction. ──

#[test]
fn predicting_the_ledger_it_was_measured_on_returns_the_measurement() {
    // The round trip. If this drifts, the arithmetic and the record
    // have come apart.
    let m = ExecutionCostModel::new(vec![m3max_metal_001()]);
    let p = m.predict(&kimi_ledger()).expect("same model");
    let o = m3max_metal_001();
    assert!(
        (p.gpu_ms_per_token - o.candidate_gpu_ms_per_token).abs() < 1e-6,
        "predicted {} vs measured {}",
        p.gpu_ms_per_token,
        o.candidate_gpu_ms_per_token
    );
    // Wall, and therefore tok/s, needs the fixed overhead — a GPU-time
    // prediction alone cannot produce a throughput figure.
    assert!((p.wall_ms_per_token - (23.43 + 1.05)).abs() < 1e-6);
    assert!(
        (p.tokens_per_second - 40.85).abs() < 0.05,
        "predicted {} tok/s",
        p.tokens_per_second
    );
    // The round trip is EXACT in GPU time, by construction, and only
    // approximate in wall time. The model carries ONE fixed overhead
    // and so assumes it is arm-independent; the bench actually measured
    // 1.05 ms on the baseline arm and 1.09 ms on the candidate, which
    // is why the predicted wall speedup is 1.140 against a measured
    // 1.139. Pinned rather than hidden: 0.1% is the size of that
    // assumption, and if a future backend makes overhead depend on the
    // map, this is the test that will say so.
    assert!(
        (p.speedup - 1.1405).abs() < 1e-3,
        "speedup was {}",
        p.speedup
    );
    assert!(
        (p.speedup - 1.139).abs() < 3e-3,
        "and it must stay close to the measured wall speedup"
    );
    assert_eq!(p.breadth, Breadth::Measured);
}

#[test]
fn a_prediction_carries_its_evidence_and_its_status() {
    let m = ExecutionCostModel::new(vec![m3max_metal_001()]);
    let p = m.predict(&kimi_ledger()).expect("same model");
    assert_eq!(p.calibration_id, "m3max-metal-001");
    assert_eq!(p.status, CalibrationStatus::Provisional);
    let line = p.describe();
    assert!(line.contains("provisional"), "{line}");
    assert!(line.contains("m3max-metal-001"), "{line}");
    assert!(line.contains("tok/s"), "{line}");
}

#[test]
fn a_breadth_far_from_any_measurement_is_flagged_as_extrapolated() {
    // Halving the whole decoder is nowhere near the 16% that was
    // measured, and beta may well not be flat out there.
    let mut l = kimi_ledger();
    for s in &mut l.scopes {
        s.candidate_bytes = s.baseline_bytes / 2;
    }
    let m = ExecutionCostModel::new(vec![m3max_metal_001()]);
    let p = m.predict(&l).expect("same model");
    match p.breadth {
        Breadth::Extrapolated {
            nearest_measured_fraction,
        } => assert!((nearest_measured_fraction - 0.15990).abs() < 1e-5),
        Breadth::Measured => panic!("50% removed is not the 16% that was measured"),
    }
    assert!(p.describe().contains("extrapolated"), "{}", p.describe());
}

#[test]
fn byte_economics_do_not_transfer_across_models() {
    let l = ByteLedger {
        model: "some-other-model".into(),
        ..kimi_ledger()
    };
    let m = ExecutionCostModel::new(vec![m3max_metal_001()]);
    match m.predict(&l) {
        Err(CostRefusal::DifferentModel { ledger_model, .. }) => {
            assert_eq!(ledger_model, "some-other-model");
        }
        other => panic!("expected a refusal naming the model, got {other:?}"),
    }
}

#[test]
fn prediction_picks_the_observation_nearest_the_breadth_asked_for() {
    // Nearest, not fitted: with the points this programme has, a fit is
    // a straight line through one of them wearing a lab coat.
    let m = ExecutionCostModel::new(vec![
        ExecutionCostObservation {
            id: "far".into(),
            ..hypothetical_at("far", 0.45, 0.60)
        },
        ExecutionCostObservation {
            id: "near".into(),
            ..hypothetical_at("near", 0.17, 0.80)
        },
    ]);
    // Both are recorded against the Kimi identity by `hypothetical_at`.
    let p = m.predict(&kimi_ledger()).expect("same model");
    assert_eq!(p.calibration_id, "near");
}

#[test]
fn a_map_that_removes_nothing_predicts_the_baseline() {
    let mut l = kimi_ledger();
    for s in &mut l.scopes {
        s.candidate_bytes = s.baseline_bytes;
    }
    let m = ExecutionCostModel::new(vec![m3max_metal_001()]);
    let p = m.predict(&l).expect("same model");
    assert!((p.gpu_ms_per_token - 26.87).abs() < 1e-6);
    assert!((p.speedup - 1.0).abs() < 1e-9);
}

#[test]
fn a_refusal_says_which_measurement_to_go_and_take() {
    assert!(CostRefusal::NoObservations
        .to_string()
        .contains("run the decode benchmark"));
    let r = CostRefusal::DifferentModel {
        ledger_model: "gpt-oss-20b".into(),
        observed_models: vec!["Kimi-Linear-48B-A3B-Instruct".into()],
    };
    assert!(r.to_string().contains("gpt-oss-20b"));
    assert!(r.to_string().contains("Kimi-Linear-48B-A3B-Instruct"));
}

#[test]
fn a_degenerate_observation_reports_zero_rather_than_a_nan() {
    // Nothing in the crate builds these, and that is exactly why the
    // guards are here: a NaN beta or a NaN speedup does not fail, it
    // propagates and then sorts arbitrarily wherever it lands.
    let empty = ExecutionCostObservation {
        baseline_bytes_per_token: 0,
        candidate_bytes_per_token: 0,
        baseline_gpu_ms_per_token: 0.0,
        candidate_gpu_ms_per_token: 0.0,
        fixed_overhead_ms: 0.0,
        ..m3max_metal_001()
    };
    assert_eq!(empty.byte_fraction_removed(), 0.0);
    assert_eq!(empty.gpu_fraction_removed(), 0.0);
    assert_eq!(empty.bytes_to_gpu_factor(), None);
    // And it cannot be used to predict: with no beta there is nothing
    // to predict WITH, so the model looks past it.
    let m = ExecutionCostModel::new(vec![empty]);
    assert!(matches!(
        m.predict(&kimi_ledger()),
        Err(CostRefusal::DifferentModel { .. })
    ));
}

#[test]
fn a_calibrated_prediction_says_so() {
    let m = ExecutionCostModel::new(vec![
        hypothetical_at("low", 0.08, 0.80),
        hypothetical_at("high", 0.32, 0.72),
    ]);
    let p = m.predict(&kimi_ledger()).expect("same model");
    assert_eq!(p.status, CalibrationStatus::Calibrated);
    let line = p.describe();
    assert!(line.contains("calibrated"), "{line}");
    assert!(!line.contains("provisional"), "{line}");
}
