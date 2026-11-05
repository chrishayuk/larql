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
fn edits_to_two_tensors_cannot_be_compiled_together_yet() {
    let fx = Fixture::new();
    let mut s = session_on(&fx.container);
    run(&mut s, &update(&[33], DELTA));
    run(&mut s, "UPDATE WEIGHTS SET DELTA = 0.0625 WHERE TENSOR = \"0.mlp.down_proj.weight\" AND ROW = 1 AND COL = 1;");
    let out = fx.path("two.vindex3");
    let err = try_run(
        &mut s,
        &format!("COMPILE CURRENT INTO VINDEX \"{}\";", q(&out)),
    )
    .unwrap_err();
    assert!(err.contains("one tensor"), "{err}");
    assert!(!out.exists(), "nothing written");
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
