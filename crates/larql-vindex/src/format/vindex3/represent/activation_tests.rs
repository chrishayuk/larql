//! The one property that matters: a synthetic substrate cannot promote.

use super::*;

fn authority(role: BankRole) -> ActivationAuthority {
    ActivationAuthority {
        checkpoint: "GLM-5.3-Flash".into(),
        operator_boundary: "post_input_layernorm_post_mhc".into(),
        layer: Some(0),
        bank: "calibration-v1".into(),
        bank_digest: "d639576a334895f8".into(),
        role,
    }
}

#[test]
fn synthetic_activations_cannot_support_a_magnitude_claim() {
    let s = ActivationSource::Synthetic {
        note: "unit-RMS gaussian".into(),
    };
    assert!(!s.supports_magnitude_claim());
    assert!(s.authority().is_none());
}

#[test]
fn model_execution_supports_a_magnitude_claim() {
    let s = ActivationSource::ModelExecution(authority(BankRole::Calibration));
    assert!(s.supports_magnitude_claim());
    assert_eq!(s.authority().expect("authority").layer, Some(0));
}

/// **Absence is not permission.** A record that never stated its
/// substrate is refused exactly like a synthetic one, and the two
/// refusals are distinguishable because they call for different fixes.
#[test]
fn an_unstated_substrate_is_refused_and_named_apart_from_a_synthetic_one() {
    assert_eq!(
        refuse_unless_model_execution(None),
        Some(ActivationRefusal::Unstated)
    );
    let syn = ActivationSource::Synthetic {
        note: "rtn probe".into(),
    };
    assert_eq!(
        refuse_unless_model_execution(Some(&syn)),
        Some(ActivationRefusal::Synthetic("rtn probe".into()))
    );
    assert!(
        refuse_unless_model_execution(Some(&ActivationSource::ModelExecution(authority(
            BankRole::Calibration
        ))))
        .is_none()
    );
    // Both refusals must say WHY, not merely refuse.
    assert!(ActivationRefusal::Unstated
        .to_string()
        .contains("indistinguishable"));
    assert!(ActivationRefusal::Synthetic("x".into())
        .to_string()
        .contains("never a magnitude"));
}

#[test]
fn the_holdout_role_is_visible_on_the_evidence() {
    let cal = ActivationSource::ModelExecution(authority(BankRole::Calibration));
    let hold = ActivationSource::ModelExecution(authority(BankRole::Holdout));
    assert!(!cal.is_holdout());
    assert!(hold.is_holdout());
}

/// The operator boundary is carried verbatim, because reading the wrong
/// boundary is a silent order-of-magnitude error, not a rounding one.
#[test]
fn the_describe_line_names_the_boundary_and_the_bank_digest() {
    let s = ActivationSource::ModelExecution(authority(BankRole::Holdout));
    let d = s.describe();
    assert!(d.contains("post_input_layernorm_post_mhc"), "{d}");
    assert!(d.contains("d639576a334895f8"), "{d}");
    assert!(d.contains("Holdout"), "{d}");
    assert!(ActivationSource::Synthetic { note: "n".into() }
        .describe()
        .contains("may not support a magnitude claim"));
}

/// Round-trips, because this travels in serialized experiment records
/// and a substrate that silently vanishes on reload is worse than none.
#[test]
fn an_activation_source_round_trips_through_serde() {
    for s in [
        ActivationSource::Synthetic {
            note: "unit-RMS".into(),
        },
        ActivationSource::ModelExecution(authority(BankRole::Holdout)),
    ] {
        let json = serde_json::to_string(&s).expect("ser");
        let back: ActivationSource = serde_json::from_str(&json).expect("de");
        assert_eq!(back, s);
        assert_eq!(
            back.supports_magnitude_claim(),
            s.supports_magnitude_claim()
        );
    }
}

// ---- the gate integration: the property that makes this structural ----

use super::super::bank::BankBuilder;
use super::super::quality::{Criterion, QualityGate};

fn magnitude_gate() -> QualityGate {
    let mut g = super::super::quality::kimi_logit_v1();
    g.id = "test-magnitude-v1".into();
    g.positions_min = 0;
    g.kl_p99_max = 1.0;
    g.top1_flip_max = None;
    g.top10_change_max = None;
    g.route_flip_max = None;
    g.require_model_activations = Some(true);
    g
}

/// A gate that carries a magnitude REFUSES a synthetic bank, refuses an
/// unstated one, and accepts a model-execution one — with everything
/// else about the three banks identical.
#[test]
fn a_magnitude_gate_refuses_synthetic_and_unstated_substrates() {
    let gate = magnitude_gate();
    let build = |src: Option<ActivationSource>| {
        let mut b = BankBuilder::default();
        if let Some(s) = src {
            b = b.activations(s);
        }
        b.finish()
    };
    let named = |g: &QualityGate, bank: &super::super::quality::QualityBank| {
        g.evaluate(bank)
            .failures
            .iter()
            .any(|(c, _)| *c == Criterion::ActivationAuthority)
    };

    let unstated = build(None);
    let synthetic = build(Some(ActivationSource::Synthetic { note: "rtn".into() }));
    let real = build(Some(ActivationSource::ModelExecution(authority(
        BankRole::Calibration,
    ))));
    assert!(named(&gate, &unstated), "unstated must fail");
    assert!(named(&gate, &synthetic), "synthetic must fail");
    assert!(!named(&gate, &real), "model execution must pass");

    // And the gate can be silent: a MECHANISM gate does not judge the
    // substrate, so the same synthetic bank passes it. Without this the
    // criterion would be unavoidable rather than declared.
    let mut mech = gate.clone();
    mech.id = "test-mechanism-v1".into();
    mech.require_model_activations = None;
    assert!(!named(&mech, &synthetic));
}

/// The refusal has to SAY which of the two problems it is, because they
/// call for different fixes.
#[test]
fn the_gate_failure_text_distinguishes_unstated_from_synthetic() {
    let gate = magnitude_gate();
    let text = |src: Option<ActivationSource>| {
        let mut b = BankBuilder::default();
        if let Some(s) = src {
            b = b.activations(s);
        }
        let bank = b.finish();
        gate.evaluate(&bank)
            .failures
            .iter()
            .find(|(c, _)| *c == Criterion::ActivationAuthority)
            .map(|(_, m)| m.clone())
            .expect("refused")
    };
    assert!(text(None).contains("indistinguishable"));
    assert!(text(Some(ActivationSource::Synthetic {
        note: "unit-RMS".into()
    }))
    .contains("unit-RMS"));
}
