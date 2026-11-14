//! The VINDEX3 backend. Its knowledge overlay holds no base, so the
//! container's shape and model name are supplied by the session.

use larql_lql::Session;
use larql_vindex::format::vindex3::fixtures::{
    encode_fixture_container, miniature_glimmer, G_VOCAB,
};
use larql_vindex::VindexPatch;

use crate::support::*;

const INSERT: &str = r#"INSERT INTO EDGES (entity, relation, target) VALUES ("a", "b", "[5]");"#;
/// What a session shows of its overlay: the applied patches and the entity
/// the fixture's patch is about.
const SHOWN: &[&str] = &["SHOW PATCHES;", r#"DESCRIBE "a";"#];

/// The synthetic `[N]` ↔ id N tokenizer with spaced forms, as the V3
/// mutation gates use: a target is encoded as `" {target}"`.
fn tokenizer_json(vocab: usize) -> String {
    let mut entries: Vec<String> = (0..vocab).map(|i| format!("\"[{i}]\":{i}")).collect();
    entries.extend((0..vocab).map(|i| format!("\" [{i}]\":{i}")));
    format!(
        "{{\"version\":\"1.0\",\"truncation\":null,\"padding\":null,\"added_tokens\":[],\
         \"normalizer\":null,\"pre_tokenizer\":null,\"post_processor\":null,\"decoder\":null,\
         \"model\":{{\"type\":\"WordLevel\",\"vocab\":{{{}}},\"unk_token\":\"[0]\"}}}}",
        entries.join(",")
    )
}

struct Fixture {
    container: tempfile::TempDir,
    /// A patch this container's own session recorded and saved.
    valid: VindexPatch,
}

fn fixture() -> Fixture {
    let checkpoint = tempfile::tempdir().expect("tempdir");
    let container = tempfile::tempdir().expect("tempdir");
    encode_fixture_container(
        miniature_glimmer,
        checkpoint.path(),
        container.path(),
        "apply-patch-fixture",
    );
    std::fs::write(
        container.path().join("tokenizer.json"),
        tokenizer_json(G_VOCAB),
    )
    .expect("tokenizer");

    let mut session = open(container.path());
    let vlp = container.path().join("valid.vlp");
    run(
        &mut session,
        &format!(r#"BEGIN PATCH "{}";"#, lql_path(&vlp)),
    );
    run(&mut session, INSERT);
    run(&mut session, "SAVE PATCH;");
    let valid = VindexPatch::load(&vlp).expect("the saved patch loads");
    assert!(!valid.is_empty(), "the fixture's INSERT recorded nothing");
    Fixture { container, valid }
}

fn open(container: &std::path::Path) -> Session {
    let mut session = Session::new();
    run(&mut session, &format!(r#"USE "{}";"#, lql_path(container)));
    session
}

fn assert_refused(fixture: &Fixture, name: &str, patch: &VindexPatch, expect: &str) {
    let mut session = open(fixture.container.path());
    let before = snapshot(&mut session, SHOWN);
    let stmt = apply_statement(fixture.container.path(), name, patch);

    let err = try_run(&mut session, &stmt).expect_err("an invalid patch is refused");
    assert!(err.contains(expect), "expected `{expect}` in: {err}");
    assert_eq!(snapshot(&mut session, SHOWN), before, "the overlay changed");
}

#[test]
fn the_patch_a_session_saved_applies_to_that_container() {
    let f = fixture();
    let mut session = open(f.container.path());
    let before = snapshot(&mut session, SHOWN);
    let stmt = apply_statement(f.container.path(), "ok.vlp", &f.valid);

    let out = run(&mut session, &stmt).join("\n");
    assert!(out.contains("Applied"), "{out}");
    assert_ne!(snapshot(&mut session, SHOWN), before, "nothing was applied");
}

#[test]
fn a_corrupt_vector_is_refused() {
    let f = fixture();
    // The width is never needed: a V3 KNN op always carries its key.
    let patch = with_corrupt_vector(f.valid.clone(), 0);
    assert_refused(&f, "corrupt.vlp", &patch, "corrupt");
}

#[test]
fn a_vector_of_the_wrong_width_is_refused() {
    let f = fixture();
    let patch = with_wide_vector(f.valid.clone(), 0);
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
    assert_refused(&f, "feature.vlp", &patch, "feature");
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
    let mut session = open(f.container.path());
    run(
        &mut session,
        &apply_statement(f.container.path(), "first.vlp", &f.valid),
    );
    let stacked = snapshot(&mut session, SHOWN);

    for (name, patch) in [
        ("s-model.vlp", with_base_model(f.valid.clone(), OTHER_MODEL)),
        ("s-wide.vlp", with_wide_vector(f.valid.clone(), 0)),
        ("s-partial.vlp", with_trailing_invalid_op(f.valid.clone())),
    ] {
        let stmt = apply_statement(f.container.path(), name, &patch);
        try_run(&mut session, &stmt).expect_err("an invalid patch is refused");
        assert_eq!(
            snapshot(&mut session, SHOWN),
            stacked,
            "{name} changed the stack"
        );
    }

    let out = run(
        &mut session,
        &apply_statement(f.container.path(), "second.vlp", &f.valid),
    )
    .join("\n");
    assert!(out.contains("Applied"), "{out}");
}
