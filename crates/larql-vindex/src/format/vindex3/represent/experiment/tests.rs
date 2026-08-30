//! `tests` for [`super`].

use super::*;

fn record() -> RepresentationExperiment {
    RepresentationExperiment {
        model: "Kimi-Linear-48B-A3B".into(),
        scope: RoleScope::role(crate::format::vindex3::represent::policy::Role::ExpertWeight),
        component: "RoutedExpertBank(layer=1, slots=9)".into(),
        source: "BF16".into(),
        target: "Q6_K".into(),
        hardware: "Apple M3 Max".into(),
        bits_per_weight: 6.5625,
        source_bytes: 127_401_984,
        target_bytes: 52_254_720,
        baseline_tokens_per_second: Some(37.33),
        result_tokens_per_second: None,
        baseline_gpu_ms: Some(1.4),
        target_gpu_ms: Some(0.65),
        target_achieved_gb_per_s: Some(321.0),
        bandwidth_bound_fraction: Some(0.88),
        component_rel_rms: Some(3.291e-2),
        component_max_over_scale: Some(3.459e-2),
        quality: None,
        status: RepresentationStatus {
            represented: true,
            available: true,
            backend_supported: true,
            runnable: true,
            measured: true,
            selected: false,
        },
        provenance: Provenance {
            gate: "report_expert_bank_representation_screen".into(),
            fixture: "/tmp/kimi_moe_fixture".into(),
            native_kernel: true,
            caveats: vec![],
        },
    }
}

#[test]
fn byte_ratio_comes_from_the_measured_sizes() {
    let r = record();
    assert!((r.byte_ratio() - 2.4381).abs() < 1e-3);
}

/// A zero target size is a broken record, not an infinite win.
#[test]
fn byte_ratio_of_an_empty_target_is_not_a_number() {
    let mut r = record();
    r.target_bytes = 0;
    assert!(r.byte_ratio().is_nan());
}

/// **The property the type exists for.** Component-level error is not
/// evidence about the model's output distribution, so a record carrying
/// only `rel_rms` must refuse to back a quality claim — otherwise a
/// reader infers from a small number that the tokens are fine.
#[test]
fn component_error_alone_does_not_support_a_quality_claim() {
    use crate::format::vindex3::represent::quality::*;
    let mut r = record();
    assert!(r.component_rel_rms.is_some());
    assert!(!r.supports_quality_claim());
    assert_eq!(r.quality_proven_by(), None);

    let gate = QualityGate {
        id: "kimi-logit-v1".into(),
        positions_min: 512,
        kl_p99_max: 1e-3,
        top1_flip_max: 0,
        top10_change_max: 8,
        route_flip_max: 0,
        covered_mass_min: None,
    };
    let bank = QualityBank {
        positions: 19,
        logits: LogitEvidence {
            kl_p50: 1e-5,
            kl_p95: 1e-4,
            kl_p99: 5e-4,
            max_logit_delta: 1e-2,
            top1_flips: 0,
            top10_changes: 0,
        },
        routing: RoutingEvidence {
            route_flips: 0,
            positions_with_route_change: 0,
            layers_with_route_change: 0,
            first_layer_with_route_change: None,
            route_margin: None,
            route_weight_mass_moved: None,
        },
        min_covered_mass: None,
        top10_margin: None,
        top10_candidate_margin: None,
        top10_mass_displaced: None,
        top10_rank_displacement: None,
        top1_margin: None,
        top1_candidate_margin: None,
        top1_mass_displaced: None,
    };
    // A bank too short for its own tail statistic proves nothing, and
    // the record must say so rather than accept a small-looking p99.
    r.quality = Some(QualityEvidence {
        gate: gate.clone(),
        bank: bank.clone(),
    });
    assert!(!r.supports_quality_claim(), "19 positions is not a p99");

    r.quality = Some(QualityEvidence {
        gate,
        bank: QualityBank {
            positions: 512,
            ..bank
        },
    });
    assert_eq!(r.quality_proven_by(), Some("kimi-logit-v1"));
    assert!(r.supports_quality_claim());
}

/// A simulated point may carry quality numbers but never throughput.
#[test]
fn a_simulated_point_does_not_support_a_throughput_claim() {
    let mut r = record();
    r.provenance.native_kernel = false;
    r.target_gpu_ms = Some(1.39);
    assert!(!r.supports_throughput_claim());
    r.provenance.native_kernel = true;
    assert!(r.supports_throughput_claim());
    r.target_gpu_ms = None;
    assert!(!r.supports_throughput_claim());
}

/// Absent means NOT MEASURED, and must survive a round trip as absent
/// rather than defaulting to zero.
#[test]
fn unmeasured_fields_survive_serialisation_as_absent() {
    let r = record();
    let json = serde_json::to_string(&r).expect("serialises");
    let back: RepresentationExperiment = serde_json::from_str(&json).expect("round trips");
    assert_eq!(back, r);
    assert!(back.quality.is_none());
    assert!(
        json.contains("\"result_tokens_per_second\":null"),
        "absence must be explicit for a field that could be mistaken for zero"
    );
    assert!(
        !json.contains("\"quality\""),
        "no quality bank means the field is absent, not an empty one"
    );
}

/// The scope is the precision map's own selector, so evidence can be
/// turned into policy without a translation step that could lose the
/// region it was measured on.
#[test]
fn a_scope_becomes_the_exception_it_justifies() {
    use crate::format::vindex3::represent::policy::Role;
    let scope = RoleScope::role(Role::ExpertWeight)
        .projection("down_proj")
        .layers(20, 26);
    let e = scope.as_exception("Q6_K");
    assert_eq!(e.projection.as_deref(), Some("down_proj"));
    assert_eq!(e.layers, Some((20, 26)));
    assert_eq!(e.encoding.as_deref(), Some("Q6_K"));
}

/// Capability is not on the promotable list, deliberately.
#[test]
fn a_backend_kernel_existing_does_not_make_a_representation_promotable() {
    let full = RepresentationStatus {
        represented: true,
        available: true,
        backend_supported: false,
        runnable: true,
        measured: true,
        selected: false,
    };
    assert!(
        full.ladder_complete(),
        "the ladder must not depend on capability"
    );
    let capability_only = RepresentationStatus {
        backend_supported: true,
        ..RepresentationStatus::default()
    };
    assert!(!capability_only.ladder_complete());
}

/// **Promotion needs all three independent facts**, and each one alone
/// withholds it: the ladder complete, a quality claim resting on a
/// named gate, and a throughput claim from the target's own kernel.
///
/// Checked by removing one at a time from a record that otherwise
/// promotes, so a future change that quietly drops a conjunct fails
/// here rather than promoting on weaker evidence.
#[test]
fn promotable_requires_the_ladder_quality_and_throughput_together() {
    use crate::format::vindex3::represent::quality::{
        kimi_logit_v1, LogitEvidence, QualityBank, QualityEvidence, RoutingEvidence,
    };

    let passing = QualityEvidence {
        gate: kimi_logit_v1(),
        bank: QualityBank {
            positions: 8192,
            logits: LogitEvidence {
                kl_p50: 1e-6,
                kl_p95: 1e-5,
                kl_p99: 1e-4,
                max_logit_delta: 1e-3,
                top1_flips: 0,
                top10_changes: 0,
            },
            routing: RoutingEvidence {
                route_flips: 0,
                positions_with_route_change: 0,
                layers_with_route_change: 0,
                first_layer_with_route_change: None,
                route_margin: None,
                route_weight_mass_moved: None,
            },
            min_covered_mass: None,
            top10_margin: None,
            top10_candidate_margin: None,
            top10_mass_displaced: None,
            top10_rank_displacement: None,
            top1_margin: None,
            top1_candidate_margin: None,
            top1_mass_displaced: None,
        },
    };

    let mut r = record();
    r.quality = Some(passing.clone());
    assert!(r.promotable(), "everything present must promote");
    assert_eq!(r.quality_proven_by(), Some("kimi-logit-v1"));
    assert!(r.supports_quality_claim() && r.supports_throughput_claim());

    // No quality evidence at all.
    let mut no_quality = r.clone();
    no_quality.quality = None;
    assert!(!no_quality.promotable());

    // Evidence that FAILS its gate proves nothing — the verdict is
    // derived, so a failing bank cannot carry a passing claim.
    let mut failing = passing.clone();
    failing.bank.positions = 1024;
    let mut unproven = r.clone();
    unproven.quality = Some(failing);
    assert!(!unproven.promotable());
    assert_eq!(unproven.quality_proven_by(), None);

    // A simulated carrier may hold quality numbers but never throughput.
    let mut simulated = r.clone();
    simulated.provenance.native_kernel = false;
    assert!(!simulated.supports_throughput_claim());
    assert!(!simulated.promotable());

    // Untimed on its own kernel.
    let mut untimed = r.clone();
    untimed.target_gpu_ms = None;
    assert!(!untimed.promotable());

    // Ladder incomplete.
    let mut unmeasured = r.clone();
    unmeasured.status.measured = false;
    assert!(!unmeasured.promotable());
}
