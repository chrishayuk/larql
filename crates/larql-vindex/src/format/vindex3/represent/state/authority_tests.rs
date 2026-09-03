//! Ruling 2 and R5-F10, pinned to the measurements that produced them.
//!
//! Ported onto the canonical `state/` substrate when the duplicate
//! map-search modules were retired; the implementation moved, the
//! evidence did not.
//!
//! The rung-5 numbers are used as fixtures on purpose: a classifier that
//! stops reproducing K25's and T1's recorded classifications has changed
//! meaning, whatever its tests say in the abstract.

use super::*;

/// The guard rung 5 measured: threshold `kl_p99_max`, band from this
/// programme's own selection-versus-held-out disagreement (N=3).
fn guard() -> DisagreementGuard {
    DisagreementGuard::new(3.5e-3, 1.7494e-4).expect("a positive finite band")
}

// ---- the guard --------------------------------------------------------

#[test]
fn a_band_must_be_positive_and_finite() {
    assert!(DisagreementGuard::new(3.5e-3, 0.0).is_none());
    assert!(DisagreementGuard::new(3.5e-3, -1.0).is_none());
    assert!(DisagreementGuard::new(3.5e-3, f64::INFINITY).is_none());
    assert!(DisagreementGuard::new(f64::NAN, 1.0).is_none());
    assert!(DisagreementGuard::new(3.5e-3, 1.7494e-4).is_some());
}

#[test]
fn band_positions_reproduce_the_recorded_classifications() {
    let g = guard();
    // K25 worst bank: admitted, but only +0.08 bands inside.
    assert_eq!(g.position(3.485444e-3), BandPosition::Indeterminate);
    // T1 selection: -0.85 bands, one bank cannot classify itself.
    assert_eq!(g.position(3.6480e-3), BandPosition::Indeterminate);
    // T1 held-out: -1.61 bands, determinate.
    assert_eq!(g.position(3.7821e-3), BandPosition::BeyondThreshold);
    // E26: +2.01 bands.
    assert_eq!(g.position(3.148652e-3), BandPosition::ClearOfThreshold);
    // U1: -4.69 bands, and S2 at -3.18.
    assert_eq!(g.position(4.3197e-3), BandPosition::BeyondThreshold);
    assert_eq!(g.position(4.0563e-3), BandPosition::BeyondThreshold);
}

#[test]
fn bands_are_signed_with_positive_meaning_inside() {
    let g = guard();
    assert!((g.bands(3.485444e-3) - 0.0832).abs() < 5e-4, "K25 +0.08");
    assert!(
        (g.bands(3.6480e-3) + 0.8458).abs() < 5e-4,
        "T1 selection -0.85"
    );
    assert!(
        (g.bands(3.7821e-3) + 1.6124).abs() < 5e-4,
        "T1 held-out -1.61"
    );
    assert!((g.bands(4.0563e-3) + 3.1801).abs() < 5e-4, "S2 -3.18");
}

// ---- admission is the contract, and a FAIL is final -------------------

#[test]
fn one_failing_bank_refuses_permanently() {
    let sel = BankOutcome::new("selection", false, 3.6480e-3);
    assert_eq!(
        AuthorityState::of(std::slice::from_ref(&sel), 2),
        AuthorityState::Refused
    );

    // A later PASSING bank cannot resurrect it. This is the asymmetry.
    let held = BankOutcome::new("held-out", true, 3.1e-3);
    assert_eq!(AuthorityState::of(&[sel, held], 2), AuthorityState::Refused);
}

#[test]
fn passing_fewer_banks_than_required_is_pending_not_admitted() {
    let sel = BankOutcome::new("selection", true, 3.3532e-3);
    assert_eq!(
        AuthorityState::of(std::slice::from_ref(&sel), 2),
        AuthorityState::Pending
    );
    assert!(!AuthorityState::Pending.is_final());

    let held = BankOutcome::new("held-out", true, 3.485444e-3);
    assert_eq!(
        AuthorityState::of(&[sel, held], 2),
        AuthorityState::Admitted
    );
}

#[test]
fn no_evidence_is_pending_never_admitted() {
    assert_eq!(AuthorityState::of(&[], 2), AuthorityState::Pending);
    assert_eq!(AuthorityState::of(&[], 0), AuthorityState::Pending);
}

#[test]
fn admitted_and_refused_are_final() {
    assert!(AuthorityState::Admitted.is_final());
    assert!(AuthorityState::Refused.is_final());
    assert_eq!(AuthorityState::Admitted.to_string(), "Admitted");
    assert_eq!(AuthorityState::Refused.to_string(), "Refused");
    assert_eq!(AuthorityState::Pending.to_string(), "Pending");
}

// ---- robustness is decisiveness, and only that ------------------------

#[test]
fn one_bank_cannot_classify_decisiveness() {
    let one = [BankOutcome::new("selection", false, 3.6480e-3)];
    assert_eq!(Robustness::of(&one, guard()), Robustness::Unreplicated);
    assert!(Robustness::Unreplicated.is_low_confidence());
}

#[test]
fn k25_is_admitted_and_boundary_adjacent() {
    let banks = [
        BankOutcome::new("selection", true, 3.3532e-3),
        BankOutcome::new("held-out", true, 3.485444e-3),
    ];
    assert_eq!(AuthorityState::of(&banks, 2), AuthorityState::Admitted);
    assert_eq!(
        Robustness::of(&banks, guard()),
        Robustness::BoundaryAdjacent
    );
    assert!(Robustness::BoundaryAdjacent.is_low_confidence());
}

#[test]
fn t1_is_refused_and_its_refusal_is_replicated() {
    let banks = [
        BankOutcome::new("selection", false, 3.6480e-3),
        BankOutcome::new("held-out", false, 3.7821e-3),
    ];
    assert_eq!(AuthorityState::of(&banks, 2), AuthorityState::Refused);
    assert_eq!(
        Robustness::of(&banks, guard()),
        Robustness::ReplicatedRefusal
    );
    assert!(!Robustness::ReplicatedRefusal.is_low_confidence());
}

#[test]
fn r5_f10_the_same_starting_classification_resolves_both_ways() {
    // K25 and T1 BOTH began INDETERMINATE on their first bank — K25 at
    // +0.84 bands inside, T1 at -0.85 outside, and neither clears the
    // guard. On bank one they are the same classification.
    let g = guard();
    assert_eq!(g.position(3.3532e-3), BandPosition::Indeterminate);
    assert_eq!(g.position(3.6480e-3), BandPosition::Indeterminate);

    // K25 resolved UP to a replicated pass; T1 resolved DOWN to a
    // determinate refusal. The sign of the resolution is not predictable
    // from the first bank, which is why this may never be a probability.
    let k25 = [
        BankOutcome::new("selection", true, 3.3532e-3),
        BankOutcome::new("held-out", true, 3.485444e-3),
    ];
    let t1 = [
        BankOutcome::new("selection", false, 3.6480e-3),
        BankOutcome::new("held-out", false, 3.7821e-3),
    ];
    assert_eq!(Robustness::of(&k25, g), Robustness::BoundaryAdjacent);
    assert_eq!(Robustness::of(&t1, g), Robustness::ReplicatedRefusal);
}

#[test]
fn banks_disagreeing_on_the_verdict_is_its_own_state() {
    let banks = [
        BankOutcome::new("selection", true, 3.40e-3),
        BankOutcome::new("held-out", false, 3.60e-3),
    ];
    assert_eq!(
        Robustness::of(&banks, guard()),
        Robustness::CrossBankDisagreement
    );
    assert!(Robustness::CrossBankDisagreement.is_low_confidence());
    // And admission is still Refused: one bank failed the contract.
    assert_eq!(AuthorityState::of(&banks, 2), AuthorityState::Refused);
}

#[test]
fn disagreement_is_judged_on_verdicts_not_on_the_binding_value() {
    // Both readings sit clear of the threshold on the binding statistic,
    // but one bank FAILED the contract — on some other statistic. That is
    // disagreement, and calling it agreement would report a stability the
    // evidence does not have.
    let banks = [
        BankOutcome::new("selection", true, 3.0e-3),
        BankOutcome::new("held-out", false, 3.0e-3),
    ];
    assert_eq!(
        Robustness::of(&banks, guard()),
        Robustness::CrossBankDisagreement
    );
}

#[test]
fn a_replicated_pass_clear_of_the_band_is_interior() {
    let banks = [
        BankOutcome::new("selection", true, 2.9737e-3),
        BankOutcome::new("held-out", true, 3.148652e-3),
    ];
    assert_eq!(
        Robustness::of(&banks, guard()),
        Robustness::ReplicatedInterior
    );
    assert!(!Robustness::ReplicatedInterior.is_low_confidence());
    assert_eq!(
        Robustness::ReplicatedInterior.to_string(),
        "ReplicatedInterior"
    );
}

#[test]
fn a_replicated_refusal_inside_the_band_is_boundary_adjacent_not_decisive() {
    let banks = [
        BankOutcome::new("selection", false, 3.55e-3),
        BankOutcome::new("held-out", false, 3.56e-3),
    ];
    assert_eq!(
        Robustness::of(&banks, guard()),
        Robustness::BoundaryAdjacent
    );
}

#[test]
fn the_worst_bank_decides_not_the_first_or_the_mean() {
    // Order must not matter, and a favourable first bank must not carry.
    let a = [
        BankOutcome::new("selection", false, 3.6480e-3),
        BankOutcome::new("held-out", false, 3.7821e-3),
    ];
    let b = [
        BankOutcome::new("held-out", false, 3.7821e-3),
        BankOutcome::new("selection", false, 3.6480e-3),
    ];
    assert_eq!(Robustness::of(&a, guard()), Robustness::of(&b, guard()));
}

// ---- the two machines, side by side and never merged ------------------

#[test]
fn map_outcome_derives_both_independently() {
    let t1 = MapOutcome::of(
        vec![
            BankOutcome::new("selection", false, 3.6480e-3),
            BankOutcome::new("held-out", false, 3.7821e-3),
        ],
        guard(),
        2,
    );
    assert_eq!(t1.authority, AuthorityState::Refused);
    assert_eq!(t1.robustness, Robustness::ReplicatedRefusal);
    assert_eq!(t1.worst_binding(), Some(3.7821e-3));
    assert_eq!(
        t1.outcomes.len(),
        2,
        "the evidence is kept, not just its verdict"
    );
}

#[test]
fn an_outcome_with_no_banks_has_no_worst_binding() {
    let none = MapOutcome::of(vec![], guard(), 2);
    assert_eq!(none.worst_binding(), None);
    assert_eq!(none.authority, AuthorityState::Pending);
    assert_eq!(none.robustness, Robustness::Unreplicated);
}

#[test]
fn a_map_outcome_round_trips_through_serde() {
    let m = MapOutcome::of(
        vec![BankOutcome::new("selection", true, 3.3532e-3)],
        guard(),
        2,
    );
    let back: MapOutcome = serde_json::from_str(&serde_json::to_string(&m).unwrap()).unwrap();
    assert_eq!(m, back);
    let g: DisagreementGuard =
        serde_json::from_str(&serde_json::to_string(&guard()).unwrap()).unwrap();
    assert_eq!(g, guard());
}
