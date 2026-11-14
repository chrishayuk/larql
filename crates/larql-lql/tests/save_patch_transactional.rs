//! `SAVE PATCH` either writes the patch file or leaves the session as it was.
//!
//! The recording is the only copy of a session's edits in a form that can be
//! saved. Before this gate, `SAVE PATCH` took it out of the session first and
//! checked second, so a refusal (an anonymous patch) or a failed write threw
//! the recorded operations away while the overlay still held the edits: they
//! stayed live and could no longer be saved.

use larql_inference::test_utils::write_synthetic_model_dir;
use larql_lql::executor::Session;
use larql_lql::parser;
use larql_vindex::VindexPatch;

const INSERT: &str =
    r#"INSERT INTO EDGES (entity, relation, target) VALUES ("[1]", "rel", "[2]");"#;

fn fresh_session() -> (Session, tempfile::TempDir) {
    let dir = tempfile::tempdir().expect("tempdir");
    write_synthetic_model_dir(dir.path()).expect("fixture write");
    let mut session = Session::new();
    run(&mut session, &format!(r#"USE "{}";"#, lql_path(dir.path())));
    (session, dir)
}

/// A path as it is written inside an LQL string literal.
fn lql_path(path: &std::path::Path) -> String {
    path.display().to_string().replace('\\', "\\\\")
}

fn try_run(session: &mut Session, sql: &str) -> Result<Vec<String>, String> {
    let parsed = parser::parse(sql).map_err(|e| format!("parse {sql:?}: {e}"))?;
    session.execute(&parsed).map_err(|e| e.to_string())
}

fn run(session: &mut Session, sql: &str) -> Vec<String> {
    try_run(session, sql).unwrap_or_else(|e| panic!("{sql}: {e}"))
}

/// The `Recording:` line of `SHOW PATCHES`, which is how a session reports
/// what `SAVE PATCH` would write. `None` when nothing is being recorded.
fn recording_line(session: &mut Session) -> Option<String> {
    run(session, "SHOW PATCHES;")
        .into_iter()
        .find(|line| line.contains("Recording:"))
}

/// One recorded operation, checked so that a fixture on which INSERT stops
/// recording cannot make every test below pass with nothing to lose.
fn insert_one(session: &mut Session) {
    run(session, INSERT);
    let line = recording_line(session).expect("INSERT starts a recording");
    assert!(line.contains("1 ops pending"), "{line}");
}

#[test]
fn a_refused_anonymous_save_keeps_the_recorded_operations() {
    let (mut session, _dir) = fresh_session();
    insert_one(&mut session);

    let err = try_run(&mut session, "SAVE PATCH;").expect_err("an anonymous patch has no file");
    assert!(err.contains("anonymous"), "{err}");

    let line = recording_line(&mut session).expect("the recording survives the refusal");
    assert!(
        line.contains("(anonymous)") && line.contains("1 ops pending"),
        "{line}"
    );
}

#[test]
fn the_recovery_the_refusal_names_saves_the_operations() {
    let (mut session, dir) = fresh_session();
    insert_one(&mut session);
    let err = try_run(&mut session, "SAVE PATCH;").expect_err("an anonymous patch has no file");

    // Every statement the message offers must parse: it used to offer
    // `SAVE PATCH "file"`, which the grammar has never accepted.
    assert!(err.contains("BEGIN PATCH"), "{err}");
    assert!(!err.contains(r#"SAVE PATCH ""#), "{err}");

    let vlp = dir.path().join("recovered.vlp");
    run(
        &mut session,
        &format!(r#"BEGIN PATCH "{}";"#, lql_path(&vlp)),
    );
    let saved = run(&mut session, "SAVE PATCH;").join("\n");
    assert!(saved.contains("1 inserts"), "{saved}");
    assert_eq!(VindexPatch::load(&vlp).expect("the patch loads").len(), 1);
}

#[test]
fn the_auto_patch_notice_names_only_statements_that_parse() {
    let (mut session, _dir) = fresh_session();
    let notice = run(&mut session, INSERT).join("\n");
    assert!(notice.contains("Auto-patch started"), "{notice}");
    assert!(!notice.contains(r#"SAVE PATCH ""#), "{notice}");
    assert!(notice.contains("BEGIN PATCH"), "{notice}");
}

#[test]
fn a_failed_write_keeps_the_recording_and_a_retry_saves_it() {
    let (mut session, dir) = fresh_session();
    let missing = dir.path().join("not-created-yet");
    let vlp = missing.join("edits.vlp");
    run(
        &mut session,
        &format!(r#"BEGIN PATCH "{}";"#, lql_path(&vlp)),
    );
    insert_one(&mut session);

    let err = try_run(&mut session, "SAVE PATCH;").expect_err("the directory does not exist");
    assert!(err.contains("1 pending operation(s) are kept"), "{err}");
    assert!(!vlp.exists());

    // Still the named recording, not an anonymous one and not an empty one.
    let line = recording_line(&mut session).expect("the recording survives the failed write");
    assert!(
        line.contains("edits.vlp") && line.contains("1 ops pending"),
        "{line}"
    );

    std::fs::create_dir(&missing).expect("create the directory");
    let saved = run(&mut session, "SAVE PATCH;").join("\n");
    assert!(saved.contains("1 inserts"), "{saved}");
    assert_eq!(VindexPatch::load(&vlp).expect("the patch loads").len(), 1);
}

#[test]
fn a_failed_write_does_not_end_the_named_session() {
    let (mut session, dir) = fresh_session();
    let vlp = dir.path().join("not-created-yet").join("edits.vlp");
    run(
        &mut session,
        &format!(r#"BEGIN PATCH "{}";"#, lql_path(&vlp)),
    );
    insert_one(&mut session);
    try_run(&mut session, "SAVE PATCH;").expect_err("the directory does not exist");

    // A second BEGIN PATCH is refused only while a named session is open. If
    // the failed save had closed it, this would start a new, empty recording.
    let other = dir.path().join("other.vlp");
    let err = try_run(
        &mut session,
        &format!(r#"BEGIN PATCH "{}";"#, lql_path(&other)),
    )
    .expect_err("the named session is still open");
    assert!(err.contains("already active"), "{err}");
}

#[test]
fn a_successful_save_closes_the_recording() {
    let (mut session, dir) = fresh_session();
    let vlp = dir.path().join("edits.vlp");
    run(
        &mut session,
        &format!(r#"BEGIN PATCH "{}";"#, lql_path(&vlp)),
    );
    insert_one(&mut session);
    run(&mut session, "SAVE PATCH;");

    assert_eq!(recording_line(&mut session), None);
    let err = try_run(&mut session, "SAVE PATCH;").expect_err("nothing is being recorded");
    assert!(err.contains("no active patch session"), "{err}");
}

#[test]
fn an_empty_patch_name_is_refused_and_opens_nothing() {
    // An empty name marks an anonymous recording. A named session opened with
    // one could not be saved (it reads as anonymous) and could not be adopted
    // by a later BEGIN PATCH (it is not an auto-patch), so its operations
    // would be kept and never saveable.
    let (mut session, dir) = fresh_session();
    let err = try_run(&mut session, r#"BEGIN PATCH "";"#).expect_err("a patch needs a name");
    assert!(err.contains("file name"), "{err}");
    assert_eq!(recording_line(&mut session), None);

    // The refusal left the session free to open a real one and save into it.
    let vlp = dir.path().join("edits.vlp");
    run(
        &mut session,
        &format!(r#"BEGIN PATCH "{}";"#, lql_path(&vlp)),
    );
    insert_one(&mut session);
    let saved = run(&mut session, "SAVE PATCH;").join("\n");
    assert!(saved.contains("1 inserts"), "{saved}");
}
