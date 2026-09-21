//! The run-provenance record (V3-OBS-1, after the cross-backend
//! finding): it names the forms an image pinned and the arm the process
//! resolved, two images of one backend fingerprint identically, two
//! backends do not, the basis rides along, and every field moves the
//! fingerprint.

use super::decode::fixture as golden_fixture;
use crate::format::vindex3::fixtures::G_HIDDEN;
use crate::format::vindex3::opplan::exec::cpu::physical::{arithmetic_arm, ArithmeticArm};
use crate::format::vindex3::opplan::exec::observe_stats::{
    FixedBasis, HeadProbe, StatsObserver, NORM_METHOD, PROBE_METHOD,
};
use crate::format::vindex3::opplan::exec::prepared::{ExecutionSlice, PreparedOperands};
use crate::format::vindex3::opplan::exec::production::ProductionBackend;
use crate::format::vindex3::opplan::exec::provenance::{ExecutionProvenance, RunProvenance};
use crate::format::vindex3::opplan::exec::reference::ReferenceBackend;

const DIMS: usize = 2;
const SEED: u64 = 7;

#[test]
fn the_record_names_what_the_image_pinned_and_what_the_process_resolved() {
    let (_c, plan, store) = golden_fixture();
    let backend = ProductionBackend::new();
    let ops = PreparedOperands::load(&plan, &store, &backend, ExecutionSlice::Full).unwrap();
    let provenance = ExecutionProvenance::of(&ops);
    assert_eq!(provenance.lowering, vec!["cpu-production@1".to_string()]);
    assert!(!provenance.realizations.is_empty());
    assert_eq!(provenance.operands(), ops.realizations().len());
    for class in &provenance.realizations {
        assert!(!class.representation.is_empty());
        assert!(class.form.contains("Cpu"), "{}", class.form);
        assert!(class.operands > 0);
    }
    // Sorted, so the serialisation is canonical.
    let keys: Vec<_> = provenance
        .realizations
        .iter()
        .map(|c| (c.representation.clone(), c.codec.clone(), c.form.clone()))
        .collect();
    let mut sorted = keys.clone();
    sorted.sort();
    assert_eq!(keys, sorted);
    assert_eq!(provenance.arithmetic_arm, arithmetic_arm());
    assert_eq!(provenance.fingerprint().len(), 64);
}

#[test]
fn one_backend_fingerprints_identically_and_two_backends_do_not() {
    let (_c, plan, store) = golden_fixture();
    let production = ProductionBackend::new();
    let reference = ReferenceBackend::new();
    let a = ExecutionProvenance::of(
        &PreparedOperands::load(&plan, &store, &production, ExecutionSlice::Full).unwrap(),
    );
    let b = ExecutionProvenance::of(
        &PreparedOperands::load(&plan, &store, &production, ExecutionSlice::Full).unwrap(),
    );
    let r = ExecutionProvenance::of(
        &PreparedOperands::load(&plan, &store, &reference, ExecutionSlice::Full).unwrap(),
    );
    assert_eq!(a, b);
    assert_eq!(a.fingerprint(), b.fingerprint());
    assert_ne!(a.lowering, r.lowering);
    assert_ne!(a.fingerprint(), r.fingerprint());
    assert_eq!(r.lowering, vec!["reference@1".to_string()]);
}

#[test]
fn the_run_record_carries_the_basis_and_the_probe_and_serialises() {
    let (_c, plan, store) = golden_fixture();
    let backend = ProductionBackend::new();
    let ops = PreparedOperands::load(&plan, &store, &backend, ExecutionSlice::Full).unwrap();
    let execution = ExecutionProvenance::of(&ops);

    let bare = RunProvenance::new(execution.clone(), None);
    assert_eq!(bare.basis, None);
    assert_eq!(bare.probe_tokens, None);
    assert_eq!(
        (bare.norm_method, bare.probe_method),
        (NORM_METHOD, PROBE_METHOD)
    );
    let bare_json = bare.to_json();
    assert!(bare_json.contains("\"basis\":null"), "{bare_json}");
    assert!(bare_json.contains("\"arithmetic_arm\":"), "{bare_json}");

    let basis = FixedBasis::seeded(G_HIDDEN, DIMS, SEED).unwrap();
    let probe = HeadProbe::new(vec![3, 5], vec![vec![0.0; G_HIDDEN]; 2], G_HIDDEN).unwrap();
    let observer = StatsObserver::new(basis.clone(), Some(probe));
    let full = RunProvenance::new(execution, Some(&observer));
    assert_eq!(full.basis.as_ref(), Some(&basis.identity()));
    assert_eq!(full.probe_tokens, Some(vec![3, 5]));
    let json = full.to_json();
    assert!(json.contains(&basis.identity().hash_hex), "{json}");
    assert_ne!(full.fingerprint(), bare.fingerprint());
    assert_eq!(full.fingerprint(), full.clone().fingerprint());
}

#[test]
fn every_field_moves_the_fingerprint() {
    let (_c, plan, store) = golden_fixture();
    let backend = ProductionBackend::new();
    let ops = PreparedOperands::load(&plan, &store, &backend, ExecutionSlice::Full).unwrap();
    let base = ExecutionProvenance::of(&ops);
    let original = base.fingerprint();

    let mut count = base.clone();
    count.realizations[0].operands += 1;
    assert_ne!(count.fingerprint(), original);

    let mut form = base.clone();
    form.realizations[0].form.push('!');
    assert_ne!(form.fingerprint(), original);

    let mut arm = base.clone();
    arm.arithmetic_arm = match base.arithmetic_arm {
        ArithmeticArm::FloatActivation => ArithmeticArm::Bf16TimesQ8,
        _ => ArithmeticArm::FloatActivation,
    };
    assert_ne!(arm.fingerprint(), original);

    let mut lowering = base.clone();
    lowering.lowering.push("other@9".to_string());
    assert_ne!(lowering.fingerprint(), original);
}
