//! Bad edits are refused when they are written down, and leave nothing staged.

use crate::support::*;

fn nothing_staged(s: &mut larql_lql::Session) {
    let shown = run(s, "SHOW PATCHES;");
    assert!(
        !shown.iter().any(|l| l.contains("Stored-weight edits")),
        "{shown:?}"
    );
}

#[test]
fn each_bad_edit_is_refused_by_name_and_stages_nothing() {
    let fx = Fixture::new();
    let mut s = session_on(&fx.container);
    let cases = [
        ("UPDATE WEIGHTS SET DELTA = 0.5 WHERE TENSOR = \"99.mlp.down_proj.weight\" AND ROW = 0 AND COL = 0;".to_string(), "not held"),
        (format!("UPDATE WEIGHTS SET DELTA = 0.5 WHERE TENSOR = \"{TENSOR}\" AND ROW = 64 AND COL = 0;"), "outside"),
        (format!("UPDATE WEIGHTS SET DELTA = 0.5 WHERE TENSOR = \"{TENSOR}\" AND ROW = 0 AND COL = 256;"), "outside"),
        (format!("UPDATE WEIGHTS SET DELTA = 0.5 WHERE TENSOR = \"{TENSOR}\" AND ROW = 0 AND COL IN (3, 3);"), "more than once"),
        ("UPDATE WEIGHTS SET DELTA = 0.5 WHERE TENSOR = \"0.input_layernorm.weight\" AND ROW = 0 AND COL = 0;".to_string(), "matrix"),
        (format!("UPDATE WEIGHTS SET DELTA = 340000000000000000000000000000000000000.0 WHERE TENSOR = \"{TENSOR}\" AND ROW = 0 AND COL = 0;"), "overflows"),
    ];
    for (stmt, why) in cases {
        let err = try_run(&mut s, &stmt).unwrap_err();
        assert!(
            err.contains("UPDATE WEIGHTS refused") && err.contains(why),
            "{stmt}: {err}"
        );
        nothing_staged(&mut s);
    }
}

#[test]
fn staging_an_element_twice_is_refused_and_never_doubles_the_edit() {
    let fx = Fixture::new();
    let mut s = session_on(&fx.container);
    run(&mut s, &update(&[33], DELTA));
    let err = try_run(&mut s, &update(&[33, 34], DELTA)).unwrap_err();
    assert!(err.contains("already edited"), "{err}");
    let shown = run(&mut s, "SHOW PATCHES;");
    assert!(
        shown
            .iter()
            .any(|l| l.contains("Stored-weight edits: 1 element(s) in 1 op(s)")),
        "{shown:?}"
    );
}

#[test]
fn compile_into_an_existing_path_is_refused_and_leaves_it_alone() {
    let fx = Fixture::new();
    let mut s = session_on(&fx.container);
    run(&mut s, &update(&[33], DELTA));
    let out = fx.path("exists");
    std::fs::create_dir(&out).unwrap();
    std::fs::write(out.join("keep"), "x").unwrap();
    let err = try_run(
        &mut s,
        &format!("COMPILE CURRENT INTO VINDEX \"{}\";", q(&out)),
    )
    .unwrap_err();
    assert!(err.contains("already exists"), "{err}");
    assert_eq!(std::fs::read_to_string(out.join("keep")).unwrap(), "x");
}

/// The weight-edit compile shares the destination boundary: an output
/// inside the source — reached through a parent that does not exist yet,
/// which used to be created in the source before the compile failed — is
/// refused before anything is written.
#[test]
fn compiling_weight_edits_into_the_source_is_refused_and_writes_nothing() {
    let fx = Fixture::new();
    let before = fingerprint(&fx.container);
    let mut s = session_on(&fx.container);
    run(&mut s, &update(&COLS, DELTA));

    for inside in [
        fx.container.join("edited.vindex3"),
        fx.container.join("new-parent").join("edited.vindex3"),
    ] {
        let err = try_run(
            &mut s,
            &format!("COMPILE CURRENT INTO VINDEX \"{}\";", q(&inside)),
        )
        .expect_err("an output inside the source");
        assert!(err.contains("inside the source vindex"), "{err}");
        assert_eq!(
            fingerprint(&fx.container),
            before,
            "the source gained nothing"
        );
    }
}

#[cfg(unix)]
#[test]
fn compiling_weight_edits_over_a_dangling_symlink_is_refused() {
    let fx = Fixture::new();
    let mut s = session_on(&fx.container);
    run(&mut s, &update(&COLS, DELTA));
    let link = fx.path("dangling.vindex3");
    std::os::unix::fs::symlink(fx.path("nowhere"), &link).unwrap();

    let err = try_run(
        &mut s,
        &format!("COMPILE CURRENT INTO VINDEX \"{}\";", q(&link)),
    )
    .expect_err("a symlink destination");

    assert!(err.contains("symbolic link"), "{err}");
    assert!(link.symlink_metadata().unwrap().file_type().is_symlink());
}
