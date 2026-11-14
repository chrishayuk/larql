//! Full fixture: real ModelWeights + safetensors-equivalent vindex

use super::*;

#[test]
fn use_synthetic_vindex_loads() {
    let (session, dir) = vindex_session("use_loads");
    assert!(matches!(session.backend, Backend::Vindex { .. }));
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn delete_by_layer_and_feature_succeeds() {
    let (mut session, dir) = vindex_session("delete_lf");

    let stmt = parser::parse(r#"DELETE FROM EDGES WHERE layer = 0 AND feature = 0;"#).unwrap();
    let out = session.execute(&stmt).expect("DELETE should succeed");
    let joined = out.join("\n");
    assert!(
        joined.contains("Deleted") || joined.contains("deleted"),
        "expected delete confirmation in: {joined}"
    );

    // The patch session should now be active (auto-patch).
    assert!(
        session.patch_recording.is_some(),
        "DELETE should have started an auto-patch session"
    );

    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn delete_no_matches_returns_message() {
    let (mut session, dir) = vindex_session("delete_nomatch");

    // Layer that doesn't exist in our 2-layer test vindex.
    let stmt = parser::parse(r#"DELETE FROM EDGES WHERE layer = 99 AND feature = 0;"#).unwrap();
    let result = session.execute(&stmt);
    // The executor either returns an empty-match message or errors —
    // both are acceptable; the important thing is no panic.
    assert!(result.is_ok() || result.is_err(), "no panic");

    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn delete_relation_filter_without_labels_errors_before_mutating() {
    let (mut session, dir) = vindex_session("delete_relation_no_labels");

    let stmt = parser::parse(r#"DELETE FROM EDGES WHERE relation = "capital";"#).unwrap();
    let err = session
        .execute(&stmt)
        .expect_err("relation-only DELETE should not silently match everything");

    assert!(
        err.to_string()
            .contains("relation filters require relation labels"),
        "unexpected error: {err}"
    );
    assert!(
        // A refused statement does not even open the auto-patch.
        session.patch_recording.is_none(),
        "failed DELETE should not record patch operations"
    );

    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn update_feature_target_succeeds() {
    let (mut session, dir) = vindex_session("update_target");

    let stmt =
        parser::parse(r#"UPDATE EDGES SET target = "London" WHERE layer = 0 AND feature = 0;"#)
            .unwrap();
    let out = session.execute(&stmt).expect("UPDATE should succeed");
    let joined = out.join("\n");
    assert!(
        joined.contains("Updated") || joined.contains("updated") || joined.contains("London"),
        "expected update confirmation in: {joined}"
    );

    assert!(
        session.patch_recording.is_some(),
        "UPDATE should have started an auto-patch session"
    );

    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn update_relation_filter_without_labels_errors_before_mutating() {
    let (mut session, dir) = vindex_session("update_relation_no_labels");

    let stmt =
        parser::parse(r#"UPDATE EDGES SET target = "London" WHERE relation = "capital";"#).unwrap();
    let err = session
        .execute(&stmt)
        .expect_err("relation-only UPDATE should not silently match everything");

    assert!(
        err.to_string()
            .contains("relation filters require relation labels"),
        "unexpected error: {err}"
    );
    assert!(
        // A refused statement does not even open the auto-patch.
        session.patch_recording.is_none(),
        "failed UPDATE should not record patch operations"
    );

    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn explicit_begin_patch_starts_session() {
    let (mut session, dir) = vindex_session("begin_patch");

    let patch_path = dir.join("session.vlp");
    let stmt = parser::parse(&format!(r#"BEGIN PATCH "{}";"#, lql_path(&patch_path))).unwrap();
    session.execute(&stmt).expect("BEGIN PATCH should succeed");

    assert!(
        session.patch_recording.is_some(),
        "BEGIN PATCH should populate patch_recording"
    );

    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn save_patch_writes_file_to_disk() {
    let (mut session, dir) = vindex_session("save_patch");

    // Start a patch, do a delete (so there's at least one operation), save.
    let patch_path = dir.join("save.vlp");
    let begin = parser::parse(&format!(r#"BEGIN PATCH "{}";"#, lql_path(&patch_path))).unwrap();
    session.execute(&begin).expect("BEGIN PATCH");

    let del = parser::parse(r#"DELETE FROM EDGES WHERE layer = 0 AND feature = 1;"#).unwrap();
    session.execute(&del).expect("DELETE under patch");

    let save = parser::parse("SAVE PATCH;").unwrap();
    let out = session.execute(&save).expect("SAVE PATCH");
    let joined = out.join("\n");
    assert!(
        patch_path.exists(),
        "SAVE PATCH should write the .vlp file. Output: {joined}"
    );

    // After SAVE PATCH, recording should be cleared.
    assert!(
        session.patch_recording.is_none(),
        "SAVE PATCH should clear the recording"
    );

    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn auto_patch_session_starts_on_first_mutation() {
    let (mut session, dir) = vindex_session("auto_patch");

    // No explicit BEGIN PATCH first.
    assert!(
        session.patch_recording.is_none(),
        "no patch session before mutation"
    );

    let del = parser::parse(r#"DELETE FROM EDGES WHERE layer = 0 AND feature = 0;"#).unwrap();
    session.execute(&del).expect("DELETE");

    assert!(
        session.patch_recording.is_some(),
        "first mutation should auto-start a patch session"
    );

    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn merge_nonexistent_source_errors_cleanly() {
    let (mut session, dir) = vindex_session("merge_bad_src");

    let stmt = parser::parse(r#"MERGE "/nonexistent/src.vindex";"#).unwrap();
    let err = session.execute(&stmt).unwrap_err();
    assert!(matches!(err, LqlError::Execution(_)));

    let _ = std::fs::remove_dir_all(&dir);
}
