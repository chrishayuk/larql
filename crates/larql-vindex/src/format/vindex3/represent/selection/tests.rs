//! `tests` for [`super`].

use super::super::experiment::{Provenance, RepresentationExperiment, RepresentationStatus};
use super::*;

/// A candidate exactly as Q1 emits one: measured, natively dispatched,
/// and with NO logit bank behind it.
fn q1_candidate(target: &str, bpw: f64, rel_rms: f64, native: bool) -> RepresentationExperiment {
    RepresentationExperiment {
        model: "Kimi-Linear-48B-A3B-Instruct".into(),
        scope: RoleScope::role(Role::ExpertWeight),
        component: "RoutedExpertBank(layer=1, slots=9)".into(),
        source: "BF16".into(),
        target: target.into(),
        hardware: "Apple M3 Max".into(),
        bits_per_weight: bpw,
        source_bytes: 127_401_984,
        target_bytes: 52_254_720,
        baseline_tokens_per_second: Some(37.33),
        result_tokens_per_second: None,
        baseline_gpu_ms: Some(1.4025),
        target_gpu_ms: native.then_some(0.6516),
        target_achieved_gb_per_s: native.then_some(321.0),
        bandwidth_bound_fraction: native.then_some(0.88),
        component_rel_rms: Some(rel_rms),
        component_max_over_scale: Some(rel_rms),
        quality: None,
        status: RepresentationStatus {
            represented: true,
            available: true,
            backend_supported: native,
            runnable: true,
            measured: native,
            selected: false,
        },
        provenance: Provenance {
            gate: "emit_representation_experiment_records".into(),
            fixture: "LARQL_KIMI_MOE_FIXTURE".into(),
            native_kernel: native,
            caveats: vec![],
        },
    }
}

/// A record that has PASSED `kimi-logit-v1`, which is the only way a
/// quality claim can exist — there is no boolean to set.
fn with_quality_bank(mut c: RepresentationExperiment) -> RepresentationExperiment {
    use crate::format::vindex3::represent::quality::*;
    c.quality = Some(QualityEvidence {
        gate: QualityGate {
            id: "kimi-logit-v1".into(),
            positions_min: 512,
            kl_p99_max: 1e-3,
            top1_flip_max: 0,
            top10_change_max: 8,
            route_flip_max: 0,
            covered_mass_min: None,
        },
        bank: QualityBank {
            positions: 512,
            logits: LogitEvidence {
                kl_p50: 1.2e-4,
                kl_p95: 5.0e-4,
                kl_p99: 9.1e-4,
                max_logit_delta: 2.2e-2,
                top1_flips: 0,
                top10_changes: 2,
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
        },
    });
    c
}

/// **Today's actual state, and the point of the gate.**
///
/// Q1 measured Q6_K thoroughly: it encodes, it dispatches through its
/// own kernel, it is 2.15x faster at 88 % of the bandwidth-bound ideal.
/// None of that is evidence about the model's output distribution, so
/// promotion must REFUSE — and say which gate is missing.
#[test]
fn a_measured_but_quality_unproven_candidate_is_refused() {
    let p = promote(
        "kimi-q1",
        "Q6_K",
        &[q1_candidate("Q6_K", 6.5625, 0.0329, true)],
    );
    assert_eq!(p.promoted(), 0);
    assert_eq!(p.verdicts[0].outcome, Err(Refusal::QualityUnproven));
    assert!(
        p.map.roles.is_empty(),
        "a map may not name a role no evidence promoted"
    );
    // The fail-safe direction: an unpromoted role resolves to SOURCE
    // precision, so a refusal costs bytes, never correctness.
    assert!(matches!(
        p.map
            .resolve(Role::ExpertWeight, "3.mlp.experts.7.down_proj.weight"),
        super::super::map::Precision::Source
    ));
    assert!(p.describe().contains("no logit-level quality bank"));
}

/// With a bank behind it, the same candidate promotes, and the map now
/// governs the role.
#[test]
fn a_quality_proven_candidate_promotes_and_governs_its_role() {
    let c = with_quality_bank(q1_candidate("Q6_K", 6.5625, 0.0329, true));
    let p = promote("kimi-q2", "Q6_K", &[c]);
    assert_eq!(p.promoted(), 1);
    assert_eq!(p.verdicts[0].outcome, Ok(()));
    assert_eq!(p.map.roles, vec!["expert-weight"]);
    assert!(matches!(
        p.map
            .resolve(Role::ExpertWeight, "3.mlp.experts.7.down_proj.weight"),
        super::super::map::Precision::Compiled("Q6_K")
    ));
    // Roles nobody measured stay at source precision — the authority is
    // the map, not the backend's capability list.
    for role in [Role::Router, Role::Norm, Role::OutputHead] {
        assert!(matches!(
            p.map.resolve(role, "whatever"),
            super::super::map::Precision::Source
        ));
    }
    assert_eq!(
        unselected(&p.map, &[Role::ExpertWeight, Role::Router, Role::Norm]),
        vec![Role::Router, Role::Norm]
    );
}

/// **Capability is not authority.** A candidate whose backend support is
/// the ONLY thing it has going for it must not promote — this is the
/// failure the whole module exists to prevent.
#[test]
fn backend_support_alone_never_promotes() {
    let mut c = q1_candidate("Q4_K", 4.5, 0.132, true);
    c.status = RepresentationStatus {
        backend_supported: true,
        ..RepresentationStatus::default()
    };
    let p = promote("capability-only", "Q4_K", &[c]);
    assert_eq!(p.promoted(), 0);
    assert!(!RepresentationStatus {
        backend_supported: true,
        ..RepresentationStatus::default()
    }
    .ladder_complete());
}

/// A simulated point may carry quality numbers and still not promote:
/// its speed was never measured through a kernel that exists.
#[test]
fn a_simulated_candidate_is_refused_even_with_a_quality_bank() {
    let c = with_quality_bank(q1_candidate("MXFP4", 4.25, 0.1947, false));
    let p = promote("kimi-mxfp4", "MXFP4", &[c]);
    assert_eq!(p.verdicts[0].outcome, Err(Refusal::NotNativelyMeasured));
}

/// Two candidates for one scope refuse BOTH, rather than letting input
/// order decide which representation a region gets.
#[test]
fn two_candidates_for_one_scope_refuse_each_other() {
    let a = with_quality_bank(q1_candidate("Q6_K", 6.5625, 0.0329, true));
    let b = with_quality_bank(q1_candidate("Q4_K", 4.5, 0.132, true));
    let p = promote("tie", "Q6_K", &[a, b]);
    assert_eq!(p.promoted(), 0);
    for v in &p.verdicts {
        assert!(matches!(
            v.outcome,
            Err(Refusal::ConflictingCandidates { .. })
        ));
    }
}

/// A narrower scope becomes a precision-map EXCEPTION, which is how
/// "gate/up at Q6, down at BF16" or "late layers protected" are
/// expressed — the thing a model-wide format label cannot say.
#[test]
fn a_narrow_scope_promotes_to_an_exception() {
    let mut c = with_quality_bank(q1_candidate("Q6_K", 6.5625, 0.0329, true));
    c.scope = RoleScope::role(Role::ExpertWeight)
        .projection("down_proj")
        .layers(20, 26);
    let p = promote("late-down-protected", "Q4_K", &[c]);
    assert_eq!(p.promoted(), 1);
    assert_eq!(p.map.exceptions.len(), 1);
    assert_eq!(p.map.exceptions[0].projection.as_deref(), Some("down_proj"));
    assert_eq!(p.map.exceptions[0].layers, Some((20, 26)));
    // The exception governs its own region and nothing else. Names are
    // OBJECT-RELATIVE — they begin at the depth index and end at
    // `.weight` — because that is what `layer_of`/`projection_of` parse.
    assert!(matches!(
        p.map
            .resolve(Role::ExpertWeight, "22.mlp.experts.3.down_proj.weight"),
        super::super::map::Precision::Compiled("Q6_K")
    ));
    assert!(matches!(
        p.map
            .resolve(Role::ExpertWeight, "22.mlp.experts.3.gate_proj.weight"),
        super::super::map::Precision::Compiled("Q4_K")
    ));
}

/// Several candidates for one scope is the NORMAL state, not a tie:
/// `CAN_REPRESENT_AS` is a many edge. When none of them is eligible, the
/// report must say why none was eligible — not that they conflict.
#[test]
fn unproven_rivals_report_their_own_missing_evidence_not_a_conflict() {
    let p = promote(
        "kimi-q1-candidates",
        "BF16",
        &[
            q1_candidate("Q6_K", 6.5625, 0.0329, true),
            q1_candidate("Q4_K", 4.5, 0.132, true),
            q1_candidate("MXFP4", 4.25, 0.1947, false),
        ],
    );
    assert_eq!(p.promoted(), 0);
    for v in &p.verdicts {
        assert_eq!(
            v.outcome,
            Err(Refusal::QualityUnproven),
            "{} should report its own missing bank, not a tie",
            v.target
        );
    }
}

/// **"Not measured" and "measured and refused" are different
/// instructions.**
///
/// `QualityUnproven` says go and run the bank. `QualityGateFailed` says
/// this representation is not good enough for THIS region — which is the
/// signal to narrow the scope, not to gather more positions. Collapsing
/// them would make the precision-allocation search unguidable.
#[test]
fn a_failed_gate_is_reported_differently_from_an_absent_one() {
    use crate::format::vindex3::represent::quality::*;
    let unmeasured = q1_candidate("Q6_K", 6.5625, 0.0329, true);
    let p = promote("m", "BF16", &[unmeasured]);
    assert_eq!(p.verdicts[0].outcome, Err(Refusal::QualityUnproven));

    let mut refused = with_quality_bank(q1_candidate("Q6_K", 6.5625, 0.0329, true));
    // The bank ran, over enough positions, and the ROUTING moved — a
    // decision change, not an arithmetic one.
    if let Some(q) = refused.quality.as_mut() {
        q.bank.routing = RoutingEvidence {
            route_flips: 37,
            positions_with_route_change: 21,
            layers_with_route_change: 4,
            first_layer_with_route_change: None,
            route_margin: None,
            route_weight_mass_moved: None,
        };
    }
    let p = promote("m", "BF16", &[refused.clone()]);
    match &p.verdicts[0].outcome {
        Err(Refusal::QualityGateFailed { verdict }) => {
            assert!(
                verdict.contains("kimi-logit-v1"),
                "names the gate: {verdict}"
            );
            assert!(verdict.contains("route_flips"), "names the criterion");
        }
        other => panic!("expected a gate failure, got {other:?}"),
    }
    // And the mechanism is legible: logits were fine, routing was not.
    let q = refused.quality.as_ref().expect("bank");
    assert!(!q.is_arithmetic_only());
    assert!(q.bank.logits.kl_p99 < q.gate.kl_p99_max);
}
