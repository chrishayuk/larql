//! `DISCARD WEIGHT EDITS`: a mis-staged stored-weight edit can be withdrawn without re-binding the container.

use larql_lql::{parse, CapabilityProfile, LqlError, Statement};

use crate::support::*;

const DISCARD: &str = "DISCARD WEIGHT EDITS;";

fn pending_line(s: &mut larql_lql::Session) -> Option<String> {
    run(s, "SHOW PATCHES;")
        .into_iter()
        .find(|l| l.contains("Stored-weight edits"))
}

#[test]
fn it_parses_and_every_word_is_required() {
    assert!(matches!(
        parse(DISCARD).unwrap(),
        Statement::DiscardWeightEdits
    ));
    assert!(matches!(
        parse("discard weight edits").unwrap(),
        Statement::DiscardWeightEdits
    ));
    for text in [
        "DISCARD;",
        "DISCARD WEIGHT;",
        "DISCARD EDITS;",
        "DISCARD WEIGHTS EDITS;",
    ] {
        assert!(parse(text).is_err(), "{text}");
    }
}

#[test]
fn it_drops_what_the_session_staged_so_the_elements_can_be_staged_again() {
    let fx = Fixture::new();
    let mut s = session_on(&fx.container);
    run(&mut s, &update(&COLS, DELTA));
    assert!(pending_line(&mut s).is_some());

    let out = run(&mut s, DISCARD);
    assert!(
        out[0].contains("Discarded: 1 stored-weight op(s)"),
        "{out:?}"
    );
    assert_eq!(pending_line(&mut s), None);

    // The same elements again: nothing is left to conflict with, and the compile holds one edit, not two.
    run(&mut s, &update(&COLS, DELTA));
    let compiled = fx.path("out.vindex3");
    run(
        &mut s,
        &format!("COMPILE CURRENT INTO VINDEX \"{}\";", q(&compiled)),
    );
    assert_eq!(
        ledger(&compiled)["targets"][0]["edits"]
            .as_array()
            .unwrap()
            .len(),
        COLS.len()
    );
}

#[test]
fn with_nothing_staged_it_says_so_and_is_not_an_error() {
    let fx = Fixture::new();
    let mut s = session_on(&fx.container);
    let out = run(&mut s, DISCARD);
    assert!(out[0].contains("Nothing to discard"), "{out:?}");
}

#[test]
fn an_applied_patchs_edits_stay_with_the_patch() {
    let fx = Fixture::new();
    let vlp = save_patch(&fx, "edit.vlp", &COLS[..2], DELTA);
    let mut s = session_on(&fx.container);
    run(&mut s, &format!("APPLY PATCH \"{}\";", q(&vlp)));
    run(&mut s, &update(&COLS[2..], DELTA));

    let out = run(&mut s, DISCARD);
    assert!(
        out[0].contains("Discarded: 1 stored-weight op(s)"),
        "{out:?}"
    );
    assert!(
        out.iter()
            .any(|l| l.contains("1 stored-weight op(s) from applied patches remain pending")),
        "{out:?}"
    );
    let line = pending_line(&mut s).unwrap();
    assert!(line.contains("2 element(s) in 1 op(s)"), "{line}");
}

#[test]
fn a_discarded_edit_is_not_saved_by_an_open_recording() {
    let fx = Fixture::new();
    let vlp = fx.path("recorded.vlp");
    let mut s = session_on(&fx.container);
    run(&mut s, &format!("BEGIN PATCH \"{}\";", q(&vlp)));
    run(&mut s, &update(&COLS[..2], DELTA));
    let out = run(&mut s, DISCARD);
    assert!(
        out.iter()
            .any(|l| l.contains("1 op(s) removed from the open patch recording")),
        "{out:?}"
    );
    run(&mut s, &update(&COLS[2..], DELTA));
    run(&mut s, "SAVE PATCH;");

    let saved: serde_json::Value =
        serde_json::from_str(&std::fs::read_to_string(&vlp).unwrap()).unwrap();
    let ops = saved["operations"].as_array().unwrap();
    assert_eq!(
        ops.len(),
        1,
        "only the edit staged after the discard: {ops:?}"
    );
    assert_eq!(ops[0]["edits"].as_array().unwrap().len(), COLS.len() - 2);
}

#[test]
fn it_needs_a_vindex3_binding_and_is_refused_to_a_public_explorer() {
    let mut unbound = larql_lql::Session::new();
    assert!(try_run(&mut unbound, DISCARD).is_err());

    let fx = Fixture::new();
    let mut s = session_on(&fx.container);
    run(&mut s, &update(&COLS, DELTA));
    s.set_profile(CapabilityProfile::PublicExplorer);
    let err = s.execute(&parse(DISCARD).unwrap()).unwrap_err();
    assert!(matches!(err, LqlError::Refused { .. }), "{err}");
}
