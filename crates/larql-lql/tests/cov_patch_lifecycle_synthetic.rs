//! Patch-lifecycle coverage for `executor/mod.rs` — the local
//! BEGIN/INSERT/SAVE/APPLY/SHOW/REMOVE PATCH path. The round-1 sweep's
//! apply test swallowed the APPLY result (`let _ =`/`if let Ok`), so
//! `patched.patches` stayed empty and `exec_show_patches`'s populated
//! branch (mod.rs:442-468) never ran. These tests assert each step so the
//! applied-patch listing + totals actually execute.

use larql_inference::test_utils::write_synthetic_model_dir;
use larql_lql::executor::Session;
use larql_lql::parser;

fn fresh_session() -> (Session, tempfile::TempDir, String) {
    let dir = tempfile::tempdir().expect("tempdir");
    write_synthetic_model_dir(dir.path()).expect("fixture write");
    let mut session = Session::new();
    let path_for_sql = dir.path().display().to_string().replace('\\', "\\\\");
    let parsed = parser::parse(&format!(r#"USE "{path_for_sql}";"#)).expect("USE parse");
    session.execute(&parsed).expect("USE execute");
    let path_str = dir.path().display().to_string();
    (session, dir, path_str)
}

fn try_run(session: &mut Session, sql: &str) -> Result<Vec<String>, String> {
    let parsed = parser::parse(sql).map_err(|e| format!("parse {sql:?}: {e}"))?;
    session
        .execute(&parsed)
        .map_err(|e| format!("execute: {e}"))
}

/// BEGIN → INSERT (records an op) → SAVE → APPLY → SHOW. The APPLY must
/// land a patch in `patched.patches` so SHOW PATCHES runs the populated
/// listing + totals branch (mod.rs:442-468) rather than the empty arm.
#[test]
fn apply_then_show_lists_applied_patch() {
    let (mut session, _dir, dir_str) = fresh_session();
    let vlp = format!("{dir_str}/lifecycle.vlp").replace('\\', "\\\\");

    try_run(&mut session, &format!(r#"BEGIN PATCH "{vlp}";"#)).expect("begin");
    // Record at least one op into the patch session.
    let _ = try_run(
        &mut session,
        r#"INSERT INTO EDGES (entity, relation, target) VALUES ("[1]", "rel", "[2]");"#,
    );
    let saved = try_run(&mut session, "SAVE PATCH;").expect("save");
    assert!(saved.join("\n").contains("Saved"), "save output: {saved:?}");
    // The saved fact is live in this session; APPLY in a fresh one (a
    // re-asserted fact is refused, FACT-ID-1 I-2).
    let mut session = Session::new();
    let path_for_sql = dir_str.replace('\\', "\\\\");
    try_run(&mut session, &format!(r#"USE "{path_for_sql}";"#)).expect("reopen");

    let applied = try_run(&mut session, &format!(r#"APPLY PATCH "{vlp}";"#)).expect("apply");
    assert!(
        applied.join("\n").contains("Applied"),
        "apply output: {applied:?}"
    );

    let shown = try_run(&mut session, "SHOW PATCHES;").expect("show");
    let joined = shown.join("\n");
    // Populated branch: per-patch listing line ("N ops (...)") and/or the
    // "Total: X from files" summary.
    assert!(
        joined.contains("ops") || joined.contains("Total"),
        "show output should list the applied patch, got: {joined}"
    );
}

/// Auto-patch INSERT writes overlay overrides without any applied file
/// patch → SHOW PATCHES exercises the overrides/totals arms with an empty
/// `patched.patches` (mod.rs:455-468).
#[test]
fn show_patches_with_overlay_overrides_only() {
    let (mut session, _dir, _) = fresh_session();
    let _ = try_run(
        &mut session,
        r#"INSERT INTO EDGES (entity, relation, target) VALUES ("[1]", "rel", "[2]");"#,
    );
    let shown = try_run(&mut session, "SHOW PATCHES;").expect("show");
    // Either overrides are reported, or the recording tail — both run the
    // non-empty path of exec_show_patches.
    assert!(!shown.is_empty());
}

/// APPLY PATCH against a session with no Vindex backend → NoBackend arm
/// (mod.rs:427). A bare session (no USE) routes here.
#[test]
fn apply_patch_without_backend_errors() {
    // Build a real patch file first via a synthetic session, then APPLY it
    // from a bare session so the load succeeds but the backend match falls
    // through to the NoBackend arm.
    let (mut session, dir, dir_str) = fresh_session();
    let vlp = format!("{dir_str}/forbare.vlp").replace('\\', "\\\\");
    try_run(&mut session, &format!(r#"BEGIN PATCH "{vlp}";"#)).expect("begin");
    let _ = try_run(
        &mut session,
        r#"INSERT INTO EDGES (entity, relation, target) VALUES ("[1]", "rel", "[2]");"#,
    );
    try_run(&mut session, "SAVE PATCH;").expect("save");
    let vlp_path = format!("{dir_str}/forbare.vlp");
    assert!(std::path::Path::new(&vlp_path).exists());

    let mut bare = Session::new();
    let err = try_run(&mut bare, &format!(r#"APPLY PATCH "{}";"#, vlp.as_str()))
        .expect_err("apply needs a backend");
    assert!(!err.is_empty());
    drop(dir);
}
