//! **The tool that refused, and the record that makes it answer.**
//!
//! Two records through ONE view method and one transport:
//!
//! ```text
//! no accounting authority   → Unavailable(no-accounting-authority)
//! sealed container facts    → Available(a real experiment)
//! ```
//!
//! Nothing under `optimizer_mcp/` differs between them. That is the
//! claim worth proving: MCP was complete when it refused, and supplying
//! the missing substrate truth made the existing transport answer.

use super::super::super::measurement::EvidenceScale;
use super::super::super::nvfp4_pack::DTYPE_NVFP4;
use super::super::super::state::resolved::NO_LAYOUT_CONSTRAINT;
use super::super::super::state::resolved::PACK_LAYOUT_ADMISSION;
use super::super::super::state::snapshot::SearchSnapshot;
use super::super::super::state::tests::container;
use super::super::NextExperiment;
use super::{
    priced_record, priced_record_ranked, priced_record_shaped, priced_record_under,
    priced_record_with_foreign_accounting, reloaded, view,
};

// ------------------------------------------------------ the refusal

#[test]
fn a_record_with_no_accounting_authority_refuses_and_says_which() {
    let snap = reloaded();
    let NextExperiment::Unavailable(refusal) = view(&snap).next_experiment() else {
        panic!("the rung-5 record carries no accounting facts");
    };

    assert_eq!(refusal.reason, "no-accounting-authority");
    assert_eq!(
        refusal.accounting.procedure,
        snap.semantics().physical_accounting
    );
    assert_eq!(refusal.accounting.procedure, "logical-bytes/v1");
    assert!(
        refusal.accounting.semantics.is_none(),
        "a procedure that did not run has no meaning to report"
    );
    assert!(refusal.accounting.source.is_none());
    assert!(refusal.accounting.priced_tensors.is_none());

    assert_eq!(refusal.missing.len(), 1);
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
    let NextExperiment::Unavailable(refusal) = view(&snap).next_experiment() else {
        panic!("expected a refusal");
    };

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

// ------------------------------------------------- the record that answers

#[test]
fn a_record_carrying_sealed_container_facts_answers_with_a_real_experiment() {
    let dir = container::glimmer();
    let snap = priced_record(dir.path());

    // Through the SAME view method the refusing record goes through.
    let NextExperiment::Available(available) = view(&snap).next_experiment() else {
        panic!("a record with accounting authority must be able to answer");
    };

    assert_eq!(
        available.routes, 2,
        "two edits resolve to one physical state, so one experiment has two routes"
    );
    assert_eq!(
        available.considered, 1,
        "and there is one opportunity, so there was nothing to rank"
    );
    assert!(
        available.physical_delta < 0,
        "the objective minimises bytes, so the selected move removes some: {}",
        available.physical_delta
    );
    assert_eq!(available.accounting.procedure, "logical-bytes/v1");
    assert_eq!(available.accounting.layout_admission, PACK_LAYOUT_ADMISSION);
    assert_eq!(
        available.accounting.priced_tensors,
        Some(snap.space().surface.len()),
        "every surface tensor was priced"
    );
    assert_eq!(
        available.accounting.selectable_encodings,
        vec![DTYPE_NVFP4.to_string()]
    );
    assert!(available.accounting.semantics.is_some());
    assert!(available.accounting.source.is_some());
}

#[test]
fn the_answer_survives_a_round_trip_through_stored_json() {
    // **The whole claim.** Serialise the record, delete everything in
    // memory, reload it, and ask again: the same experiment comes back.
    // Nothing derived was stored, so this is derivation and not recall.
    let dir = container::glimmer();
    let written = serde_json::to_string(&priced_record(dir.path())).expect("serialize");
    let reloaded: SearchSnapshot = serde_json::from_str(&written).expect("reload");
    reloaded.check_schema().expect("same schema");

    let NextExperiment::Available(from_memory) = view(&priced_record(dir.path())).next_experiment()
    else {
        panic!("expected an answer");
    };
    let NextExperiment::Available(from_disk) = view(&reloaded).next_experiment() else {
        panic!("a reloaded record must answer the same question the same way");
    };
    assert_eq!(from_memory, from_disk);
}

#[test]
fn the_stored_record_carries_the_facts_and_not_the_answer() {
    // 1d's anti-cheat, pointed at the new factual input. The accounting
    // facts are sealed source lengths and dtypes; the price table, the
    // footprint, the ranking and the selected experiment are derived
    // and must appear nowhere.
    let dir = container::glimmer();
    let json = serde_json::to_value(priced_record(dir.path())).expect("serialize");
    let text = json.to_string();

    // `logical_bytes` is deliberately NOT forbidden: on a
    // `SourceStorageFact` it is the sealed source length, which is a
    // FACT. What must be absent is anything DERIVED from it.
    for forbidden in [
        "\"compiled\"",
        "\"physical_delta\"",
        "\"experiment\"",
        "\"considered\"",
        "\"routes\"",
        "\"selection\"",
        "\"next_experiment\"",
    ] {
        assert!(
            !text.contains(forbidden),
            "the record stores {forbidden}, which is derived"
        );
    }

    // And the FACTS are there, so the emptiness is not vacuous.
    let accounting = &json["facts"]["accounting"];
    assert!(
        accounting["source"].is_string(),
        "the source it was read from"
    );
    assert!(
        accounting["semantics"].is_string(),
        "the procedure's meaning"
    );
    let stored = accounting["source_storage"]
        .as_array()
        .expect("one entry per stored tensor");
    assert!(!stored.is_empty());
    for held in stored {
        assert!(held["tensor"]["object"].is_string());
        assert!(held["tensor"]["tensor"].is_string());
        assert!(held["fact"]["dtype"].is_string());
        assert!(
            held["fact"]["logical_bytes"].is_number(),
            "the sealed source length is a FACT and is stored"
        );
    }
    assert_eq!(
        json["config"]["semantics"]["layout_admission"],
        PACK_LAYOUT_ADMISSION
    );
}

#[test]
fn a_record_with_no_accounting_authority_builds_no_price_table() {
    // The substrate refusal the view short-circuits, asserted at its
    // own level so it cannot rot behind the view's earlier check.
    let snap = reloaded();
    let err = snap.footprint().expect_err("nothing to price from");
    assert!(
        format!("{err}").contains("no physical accounting authority"),
        "{err}"
    );

    // And the encodings the search may select are read from the record,
    // not guessed: the base map's own plus every edit that names one.
    let dir = container::dense();
    let priced = priced_record(dir.path());
    assert_eq!(
        priced.selectable_encodings(),
        vec![DTYPE_NVFP4.to_string()],
        "one encoding, declared twice, listed once"
    );
    assert!(priced.footprint().is_ok());
}

// ----------------------------- the record's policies, and no others

#[test]
fn a_layout_policy_this_build_does_not_implement_is_refused_not_defaulted() {
    // **The second-layout-truth guard.** A layout refusal removes a
    // tensor from the action space and collapses its state onto the
    // protected one. Resolving an unknown policy as this build's
    // favourite would silently re-answer every refusal the stored
    // states were resolved under.
    let dir = container::dense();
    let snap = priced_record_under(
        dir.path(),
        Default::default(),
        "some-other-layout/v9",
        "logical-bytes/v1",
    );
    let NextExperiment::Unavailable(refusal) = view(&snap).next_experiment() else {
        panic!("expected a refusal");
    };
    assert_eq!(refusal.reason, "accounting-unusable");
    assert!(
        refusal.detail.contains("some-other-layout/v9"),
        "{}",
        refusal.detail
    );
    assert!(
        refusal.detail.contains("re-answer every layout refusal"),
        "{}",
        refusal.detail
    );
    assert_eq!(refusal.accounting.layout_admission, "some-other-layout/v9");
}

#[test]
fn an_accounting_procedure_this_build_does_not_implement_is_refused() {
    // The same discipline on the compiled side. Pricing a record's
    // states under a procedure it did not declare would compare figures
    // nothing produced together.
    let dir = container::dense();
    let snap = priced_record_under(
        dir.path(),
        Default::default(),
        PACK_LAYOUT_ADMISSION,
        "bytes-per-token/v1",
    );
    let NextExperiment::Unavailable(refusal) = view(&snap).next_experiment() else {
        panic!("expected a refusal");
    };
    assert_eq!(refusal.reason, "accounting-unusable");
    assert!(
        refusal.detail.contains("bytes-per-token/v1"),
        "{}",
        refusal.detail
    );
    assert_eq!(refusal.accounting.procedure, "bytes-per-token/v1");
}

#[test]
fn the_footprint_is_built_under_the_records_layout_policy_and_not_this_builds() {
    // **The cross-check that keeps ONE layout truth.** Under
    // `pack-layout-admission/v1` a `k = 24` tensor is layout-refused,
    // so it resolves to source and needs no compiled price. Under
    // `no-layout-constraint/v1` the same tensor is ADMITTED, so a
    // compiled price becomes required — and nothing prices an NVFP4
    // pack whose k is not a whole number of groups.
    //
    // If the price table were built under a policy of this module's
    // choosing rather than the record's, one of these two would answer
    // like the other.
    let dir = container::dense();
    let admitted = priced_record_under(
        dir.path(),
        Default::default(),
        NO_LAYOUT_CONSTRAINT,
        "logical-bytes/v1",
    );
    let refused = priced_record_under(
        dir.path(),
        Default::default(),
        PACK_LAYOUT_ADMISSION,
        "logical-bytes/v1",
    );

    // The surface is [64, 64] throughout, which BOTH policies admit and
    // NVFP4 prices, so both answer — the control that makes the shape
    // test below mean something rather than passing on any difference.
    assert!(matches!(
        view(&admitted).next_experiment(),
        NextExperiment::Available(_)
    ));
    assert!(matches!(
        view(&refused).next_experiment(),
        NextExperiment::Available(_)
    ));

    // Now a k the pack cannot hold. The declared policy is the only
    // thing that differs between these two records.
    let admitted = priced_record_shaped(dir.path(), vec![64, 24], NO_LAYOUT_CONSTRAINT);
    let refused = priced_record_shaped(dir.path(), vec![64, 24], PACK_LAYOUT_ADMISSION);

    let NextExperiment::Unavailable(under_admission) = view(&admitted).next_experiment() else {
        panic!("an admitted encoding nothing prices must refuse");
    };
    assert!(
        under_admission.detail.contains("no oracle prices"),
        "{}",
        under_admission.detail
    );
    assert!(
        !matches!(
            view(&refused).next_experiment(),
            NextExperiment::Unavailable(_)
        ),
        "the same tensor under a policy that REFUSES it needs no compiled price"
    );
}

#[test]
fn accounting_facts_from_another_container_are_refused() {
    // The facts are an input, so they can be the wrong ones. 4b-d's
    // `ForeignSource` reaches the agent as a distinct reason.
    let a = container::dense();
    let b = container::glimmer();
    let snap = priced_record_with_foreign_accounting(a.path(), b.path());
    let NextExperiment::Unavailable(refusal) = view(&snap).next_experiment() else {
        panic!("expected a refusal");
    };
    assert_eq!(refusal.reason, "accounting-unusable");
    assert!(
        refusal
            .detail
            .contains("pricing one model's surface from another's storage"),
        "{}",
        refusal.detail
    );
}

#[test]
fn several_opportunities_are_ranked_and_the_count_is_reported() {
    // `considered` is `Selection::Ranked`'s own figure, so an agent can
    // tell "one thing to do" from "one of several, chosen by the rule".
    let dir = container::glimmer();
    let snap = priced_record_ranked(dir.path());
    let NextExperiment::Available(available) = view(&snap).next_experiment() else {
        panic!("expected an answer");
    };
    assert!(
        available.considered > 1,
        "two edits resolving differently are two opportunities: {available:?}"
    );
}

#[test]
fn no_candidate_ranking_is_served_under_any_name() {
    let snap = reloaded();
    let rendered = serde_json::to_value(view(&snap).next_experiment()).expect("serializes");
    let text = rendered.to_string();

    // A refusal must not quietly become a recommendation. These are the
    // words a caller would look for, and none of them may appear as a
    // KEY carrying a candidate.
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
