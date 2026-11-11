//! The VINDEX2 backend.

use larql_inference::test_utils::write_synthetic_model_dir;
use larql_lql::Session;
use larql_vindex::VindexPatch;

use crate::support::*;

const INSERT: &str =
    r#"INSERT INTO EDGES (entity, relation, target) VALUES ("[1]", "rel", "[2]");"#;
/// What a session shows of its overlay: the applied patches, every edge, and
/// the inserted entity. DESCRIBE is the one that reads the KNN store, which
/// is where this fixture's INSERT lands.
const SHOWN: &[&str] = &[
    "SHOW PATCHES;",
    "SELECT * FROM EDGES LIMIT 100000;",
    r#"DESCRIBE "[1]";"#,
];

struct Fixture {
    dir: tempfile::TempDir,
    /// A patch this vindex's own session recorded and saved.
    valid: VindexPatch,
    hidden_size: usize,
}

fn fixture() -> Fixture {
    let dir = tempfile::tempdir().expect("tempdir");
    write_synthetic_model_dir(dir.path()).expect("fixture write");
    let mut session = open(dir.path());
    let vlp = dir.path().join("valid.vlp");
    run(
        &mut session,
        &format!(r#"BEGIN PATCH "{}";"#, lql_path(&vlp)),
    );
    run(&mut session, INSERT);
    run(&mut session, "SAVE PATCH;");
    let valid = VindexPatch::load(&vlp).expect("the saved patch loads");
    assert!(!valid.is_empty(), "the fixture's INSERT recorded nothing");
    let hidden_size = larql_vindex::load_vindex_config(dir.path())
        .expect("config")
        .hidden_size;
    Fixture {
        dir,
        valid,
        hidden_size,
    }
}

/// A session with nothing applied.
fn open(dir: &std::path::Path) -> Session {
    let mut session = Session::new();
    run(&mut session, &format!(r#"USE "{}";"#, lql_path(dir)));
    session
}

/// Apply `patch` to a fresh session and require a refusal that changes nothing.
fn assert_refused(fixture: &Fixture, name: &str, patch: &VindexPatch, expect: &str) {
    let mut session = open(fixture.dir.path());
    let before = snapshot(&mut session, SHOWN);
    let stmt = apply_statement(fixture.dir.path(), name, patch);

    let err = try_run(&mut session, &stmt).expect_err("an invalid patch is refused");
    assert!(err.contains(expect), "expected `{expect}` in: {err}");
    assert_eq!(snapshot(&mut session, SHOWN), before, "the overlay changed");
}

#[test]
fn the_patch_a_session_saved_applies_to_that_vindex() {
    // The control: without it, every refusal below could be a fixture that
    // refuses everything.
    let fixture = fixture();
    let mut session = open(fixture.dir.path());
    let before = snapshot(&mut session, SHOWN);
    let stmt = apply_statement(fixture.dir.path(), "ok.vlp", &fixture.valid);

    let out = run(&mut session, &stmt).join("\n");
    assert!(out.contains("Applied"), "{out}");
    assert_ne!(snapshot(&mut session, SHOWN), before, "nothing was applied");
}

#[test]
fn a_corrupt_vector_is_refused() {
    let f = fixture();
    let patch = with_corrupt_vector(f.valid.clone(), f.hidden_size);
    assert_refused(&f, "corrupt.vlp", &patch, "corrupt");
}

#[test]
fn a_vector_of_the_wrong_width_is_refused() {
    let f = fixture();
    let patch = with_wide_vector(f.valid.clone(), f.hidden_size);
    assert_refused(&f, "wide.vlp", &patch, "hidden size");
}

#[test]
fn a_patch_written_against_another_model_is_refused() {
    let f = fixture();
    let patch = with_base_model(f.valid.clone(), OTHER_MODEL);
    assert_refused(&f, "other-model.vlp", &patch, OTHER_MODEL);
}

#[test]
fn a_patch_that_declares_no_model_is_refused() {
    let f = fixture();
    let patch = with_base_model(f.valid.clone(), "");
    assert_refused(&f, "no-model.vlp", &patch, "declares no base_model");
}

#[test]
fn a_layer_out_of_bounds_is_refused() {
    let f = fixture();
    let patch = with_layer_out_of_bounds(f.valid.clone());
    assert_refused(&f, "layer.vlp", &patch, "out of bounds");
}

#[test]
fn a_feature_out_of_bounds_is_refused() {
    let f = fixture();
    let patch = with_feature_out_of_bounds(f.valid.clone());
    assert_refused(&f, "feature.vlp", &patch, "out of bounds");
}

#[test]
fn one_invalid_operation_stops_the_valid_ones_before_it() {
    let f = fixture();
    let patch = with_trailing_invalid_op(f.valid.clone());
    assert_refused(&f, "partial.vlp", &patch, "out of bounds");
}

#[test]
fn a_refused_patch_leaves_the_patches_under_it_and_the_stack_usable() {
    let f = fixture();
    let mut session = open(f.dir.path());
    run(
        &mut session,
        &apply_statement(f.dir.path(), "first.vlp", &f.valid),
    );
    let stacked = snapshot(&mut session, SHOWN);

    for (name, patch) in [
        ("s-model.vlp", with_base_model(f.valid.clone(), OTHER_MODEL)),
        (
            "s-wide.vlp",
            with_wide_vector(f.valid.clone(), f.hidden_size),
        ),
        ("s-partial.vlp", with_trailing_invalid_op(f.valid.clone())),
    ] {
        let stmt = apply_statement(f.dir.path(), name, &patch);
        try_run(&mut session, &stmt).expect_err("an invalid patch is refused");
        assert_eq!(
            snapshot(&mut session, SHOWN),
            stacked,
            "{name} changed the stack"
        );
    }

    // A valid patch still stacks on top afterwards.
    let out = run(
        &mut session,
        &apply_statement(f.dir.path(), "second.vlp", &f.valid),
    )
    .join("\n");
    assert!(out.contains("Applied"), "{out}");
    let shown = run(&mut session, "SHOW PATCHES;").join("\n");
    assert!(
        shown.contains("first.vlp") && shown.contains("second.vlp"),
        "{shown}"
    );
}
