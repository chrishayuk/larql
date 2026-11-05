//! Editing directly: stage with `UPDATE WEIGHTS`, compile into a new container.

use crate::support::*;

#[test]
fn staging_writes_nothing_and_compile_writes_a_clean_container() {
    let fx = Fixture::new();
    let before = fingerprint(&fx.container);
    let mut s = session_on(&fx.container);
    let staged = run(&mut s, &update(&COLS, DELTA));
    assert!(staged[0].starts_with("Staged: 4 element(s)"), "{staged:?}");
    assert!(staged
        .iter()
        .any(|l| l.contains("INFER in this session does not see it")));
    assert_eq!(
        fingerprint(&fx.container),
        before,
        "staging never touches the container"
    );

    let out = fx.path("edited.vindex3");
    let compiled = run(
        &mut s,
        &format!("COMPILE CURRENT INTO VINDEX \"{}\";", q(&out)),
    );
    assert!(compiled[0].starts_with("Compiled:"), "{compiled:?}");
    assert_eq!(fingerprint(&fx.container), before, "nor does compiling");
    assert!(is_coherent(&out));
    let (a, b) = (digests(&fx.container), digests(&out));
    for id in [CANONICAL, DERIVED] {
        assert_ne!(a[id], b[id], "{id} changed");
    }
    let ledger = ledger(&out);
    assert_eq!(ledger["target"]["tensor"], TENSOR);
    assert_eq!(ledger["edits"].as_array().unwrap().len(), COLS.len());
    assert_eq!(
        ledger["origins"].as_array().unwrap().len(),
        1,
        "the staged op is the ledger's origin"
    );
}

#[test]
fn the_compiled_container_needs_no_overlay() {
    let fx = Fixture::new();
    let mut s = session_on(&fx.container);
    run(&mut s, &update(&COLS, DELTA));
    let out = fx.path("edited.vindex3");
    run(
        &mut s,
        &format!("COMPILE CURRENT INTO VINDEX \"{}\";", q(&out)),
    );

    let mut fresh = session_on(&out);
    let shown = run(&mut fresh, "SHOW PATCHES;");
    assert!(
        shown.iter().any(|l| l.contains("no patches applied")),
        "{shown:?}"
    );
    assert!(!shown.iter().any(|l| l.contains("Stored-weight edits")));
}

#[test]
fn a_negative_delta_and_several_statements_accumulate_without_overlap() {
    let fx = Fixture::new();
    let mut s = session_on(&fx.container);
    run(&mut s, &update(&[32, 33], "-0.125"));
    let second = run(&mut s, &update(&[40], DELTA));
    assert!(
        second
            .iter()
            .any(|l| l.contains("3 weight element(s) pending")),
        "{second:?}"
    );
    let shown = run(&mut s, "SHOW PATCHES;");
    assert!(
        shown
            .iter()
            .any(|l| l.contains("Stored-weight edits: 3 element(s) in 2 op(s)")),
        "{shown:?}"
    );
}

#[test]
fn a_direct_edit_is_not_recorded_without_a_patch_session() {
    let fx = Fixture::new();
    let mut s = session_on(&fx.container);
    run(&mut s, &update(&COLS, DELTA));
    let err = try_run(&mut s, "SAVE PATCH;").unwrap_err();
    assert!(err.contains("no active patch session"), "{err}");
}

#[test]
fn compiling_with_no_pending_weight_edit_is_the_unchanged_bake() {
    let fx = Fixture::new();
    let mut s = session_on(&fx.container);
    let out = fx.path("plain.vindex3");
    run(
        &mut s,
        &format!("COMPILE CURRENT INTO VINDEX \"{}\";", q(&out)),
    );
    assert_eq!(digests(&fx.container)[CANONICAL], digests(&out)[CANONICAL]);
    assert!(!out.join("weight_patches.json").exists());
}
