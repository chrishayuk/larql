//! Under `--representation-source stored` the selector never offers or
//! picks a requantised image: the size policy's Q8/Q4 choice for a large
//! bf16 operand falls back to the stored bf16 bytes (#665).

use super::*;

fn large_head() -> PlannedOperand {
    synthetic(Operation::OutputHead, LARGE)
}

fn is_requantise(form: &RealizationId) -> bool {
    matches!(form.form, RealizationForm::Requantise(_))
}

#[test]
fn stored_offers_no_requantised_candidate() {
    let facts = RepresentationFacts::resolve("BF16").with_requantise_forbidden(true);
    let selected = select_cpu(&large_head(), &facts, KQuantExecution::Direct).unwrap();
    assert!(
        !selected.candidates.iter().any(is_requantise),
        "a requantised image is manufactured at load and is no candidate under stored: {:?}",
        selected.candidates
    );
    assert!(!is_requantise(&selected.realization));
}

#[test]
fn stored_holds_the_bf16_bytes_where_the_size_policy_would_requantise() {
    let auto = RepresentationFacts::resolve("BF16");
    let unforbidden = select_cpu(&large_head(), &auto, KQuantExecution::Direct).unwrap();
    let stored = select_cpu(
        &large_head(),
        &auto.clone().with_requantise_forbidden(true),
        KQuantExecution::Direct,
    )
    .unwrap();
    if is_requantise(&unforbidden.realization) {
        assert_eq!(
            stored.realization,
            RealizationId::cpu(RealizationForm::Direct(PhysicalProjectionPlan::FusedBf16))
        );
        assert_eq!(stored.reason, SelectionReason::SourcePrecisionHeld);
    } else {
        // The host's L2 or the format cap kept the head out of the Q8
        // population; stored then changes nothing.
        assert_eq!(stored.realization, unforbidden.realization);
    }
}

#[test]
fn the_flag_changes_nothing_for_a_small_operand() {
    let facts = RepresentationFacts::resolve("BF16");
    let small = synthetic(Operation::OutputHead, SMALL);
    let auto = select_cpu(&small, &facts, KQuantExecution::Direct).unwrap();
    let stored = select_cpu(
        &small,
        &facts.clone().with_requantise_forbidden(true),
        KQuantExecution::Direct,
    )
    .unwrap();
    assert_eq!(auto.realization, stored.realization);
}
