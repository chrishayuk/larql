//! **The tool that refuses**, and that its refusal is specific.

use super::super::super::measurement::EvidenceScale;
use super::super::NextExperiment;
use super::{reloaded, view};

#[test]
fn the_record_declares_an_accounting_rule_it_cannot_evaluate() {
    let snap = reloaded();
    let NextExperiment::NoFootprintOracle(refusal) = view(&snap).next_experiment();

    assert_eq!(
        refusal.declared_accounting,
        snap.semantics().physical_accounting
    );
    assert_eq!(
        refusal.declared_accounting, "logical-bytes/v1",
        "a name for a procedure, and not the procedure"
    );
}

#[test]
fn the_refusal_names_both_missing_facts() {
    let snap = reloaded();
    let NextExperiment::NoFootprintOracle(refusal) = view(&snap).next_experiment();

    assert_eq!(refusal.missing.len(), 2);
    let facts: Vec<&str> = refusal.missing.iter().map(|m| m.fact.as_str()).collect();
    assert!(facts.iter().any(|f| f.contains("Footprint")));
    assert!(facts.iter().any(|f| f.contains("source dtype")));
    for missing in &refusal.missing {
        assert!(
            !missing.because.is_empty(),
            "a refusal that does not say why can only be argued with by guessing"
        );
    }
}

#[test]
fn the_facts_that_need_no_footprint_are_still_served() {
    let snap = reloaded();
    let NextExperiment::NoFootprintOracle(refusal) = view(&snap).next_experiment();

    // The vocabulary is an input. R5-F6 was a vocabulary failure and
    // cost two ~430 MB moves, so the move set is worth showing even
    // when none of it can be priced.
    assert_eq!(&refusal.vocabulary, &snap.space().vocabulary);

    // These states are already in the graph and already priced, so
    // reporting them needs no oracle at all.
    for gap in &refusal.unmeasured {
        assert_eq!(gap.states, snap.unmeasured_at(gap.scale));
    }
    assert_eq!(
        refusal
            .unmeasured
            .iter()
            .map(|g| g.scale)
            .collect::<Vec<_>>(),
        EvidenceScale::ALL.to_vec()
    );
}

#[test]
fn no_candidate_ranking_is_served_under_any_name() {
    let snap = reloaded();
    let rendered = serde_json::to_value(view(&snap).next_experiment()).expect("serializes");
    let text = rendered.to_string();

    // The refusal must not quietly become a recommendation. These are
    // the words a caller would look for, and none of them may appear
    // as a KEY carrying a candidate.
    for forbidden in [
        "\"recommendation\"",
        "\"ranked\"",
        "\"score\"",
        "\"candidate\"",
        "\"opportunity\"",
        "\"best\"",
    ] {
        assert!(
            !text.contains(forbidden),
            "the refusal grew a {forbidden} field"
        );
    }
}
