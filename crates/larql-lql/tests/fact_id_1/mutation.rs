//! ID-3, ID-4, ID-5, ID-11, ID-12: what UPDATE and DELETE by identity
//! reach, that a KNN UPDATE changes INFER, the single-token rule, atomic
//! refusal, and duplicate keys.

use larql_lql::FactStatus;

use super::*;

const UPDATE_AB: &str = r#"UPDATE EDGES SET target = "[9]" WHERE entity = "a" AND relation = "b";"#;
const DELETE_AB: &str = r#"DELETE FROM EDGES WHERE entity = "a" AND relation = "b";"#;

/// ID-3: a semantic mutation never touches a discovered row. The discovered
/// rows of every layer are compared before and after, on both the success
/// and the refusal path, and a key with only discovered rows is refused.
#[test]
fn semantic_mutations_never_touch_discovered_rows() {
    let container = v3_container();
    let mut session = bound(container.path());
    let edges = |s: &mut Session| run(s, "SELECT * FROM EDGES LIMIT 100000;");
    let before = edges(&mut session);
    assert!(!before.trim().is_empty(), "the fixture has discovered rows");

    run(&mut session, INSERT_AB);
    run(&mut session, UPDATE_AB);
    run(&mut session, DELETE_AB);
    assert_eq!(edges(&mut session), before);

    // `[3]` is a token the fixture's features carry, so discovered rows
    // exist for it; it is not an asserted fact, so both verbs refuse.
    for stmt in [
        r#"UPDATE EDGES SET target = "[9]" WHERE entity = "[3]" AND relation = "b";"#,
        r#"DELETE FROM EDGES WHERE entity = "[3]" AND relation = "b";"#,
    ] {
        let why = refused(&mut session, stmt);
        assert!(why.contains("no asserted fact"), "{why}");
        assert!(why.contains("never discovered rows"), "{why}");
    }
    assert_eq!(edges(&mut session), before);
}

/// ID-4: a KNN UPDATE changes what INFER answers, by token id and text.
#[test]
fn a_knn_update_changes_the_infer_answer() {
    let container = v3_container();
    let mut session = bound(container.path());
    run(&mut session, INSERT_AB);
    assert!(infer_top1(&mut session, CANONICAL_PROMPT).contains("[7]"));

    run(&mut session, UPDATE_AB);

    let top1 = infer_top1(&mut session, CANONICAL_PROMPT);
    assert!(top1.contains("[9]") && !top1.contains("[7]"), "{top1}");
    let fact = only_fact(&session);
    assert_eq!(fact.target_id, Some(9), "the stored id moved with the text");
    assert_eq!(fact.target, "[9]");
}

/// ID-5: KNN targets must be one token; legacy multi-token records are
/// readable and refuse UPDATE; compose is not restricted by the rule.
#[test]
fn the_single_token_rule_binds_knn_inserts_and_updates() {
    let container = v3_container();
    let dir = tempfile::tempdir().unwrap();
    let mut session = bound(container.path());

    let why = refused(
        &mut session,
        r#"INSERT INTO EDGES (entity, relation, target) VALUES ("a", "b", "[7] [8]");"#,
    );
    assert!(why.contains("not a single token"), "{why}");
    run(&mut session, INSERT_AB);
    let why = refused(
        &mut session,
        r#"UPDATE EDGES SET target = "Paris" WHERE entity = "a" AND relation = "b";"#,
    );
    assert!(why.contains("not a single token"), "{why}");

    // A legacy record: the saved insert, its target text rewritten to two
    // tokens while the id stays 7 (what a pre-FACT-ID-1 insert stored).
    let mut fresh = bound(container.path());
    let mut patch = saved_patch(&mut fresh, dir.path(), "one.vlp", &[INSERT_AB]);
    patch["operations"][0]["target"] = "[7] [8]".into();
    let legacy_path = write_patch(dir.path(), "legacy.vlp", &patch);
    let mut session = bound(container.path());
    run(&mut session, &format!("APPLY PATCH \"{legacy_path}\";"));
    assert_eq!(only_fact(&session).status, FactStatus::MultiToken);
    let why = refused(&mut session, UPDATE_AB);
    assert!(
        why.contains("multi_token") || why.contains("multi-token"),
        "{why}"
    );
    run(&mut session, DELETE_AB);
    assert!(
        facts(&session).is_empty(),
        "a multi-token fact can be deleted"
    );
}

/// ID-11: every refusal leaves the session exactly as it was.
#[test]
fn every_refusal_leaves_the_session_unchanged() {
    let container = v3_container();
    let dir = tempfile::tempdir().unwrap();
    let mut session = bound(container.path());
    run(&mut session, INSERT_AB);
    let conflicting = {
        let mut other = bound(container.path());
        let patch = saved_patch(&mut other, dir.path(), "conflict.vlp", &[INSERT_AB]);
        write_patch(dir.path(), "conflict2.vlp", &patch)
    };
    let before = observe(&mut session);

    let refusals = [
        INSERT_AB.to_string(),
        r#"INSERT INTO EDGES (entity, relation, target) VALUES ("c", "d", "[7] [8]");"#.into(),
        r#"UPDATE EDGES SET target = "Paris" WHERE entity = "a" AND relation = "b";"#.into(),
        r#"UPDATE EDGES SET target = "[9]" WHERE entity = "zz" AND relation = "b";"#.into(),
        r#"DELETE FROM EDGES WHERE entity = "zz" AND relation = "b";"#.into(),
        r#"UPDATE EDGES SET nonsense = 1 WHERE entity = "a" AND relation = "b";"#.into(),
        format!("APPLY PATCH \"{conflicting}\";"),
    ];
    for stmt in &refusals {
        refused(&mut session, stmt);
        assert_eq!(observe(&mut session), before, "{stmt} changed the session");
    }
}

/// ID-12: a second INSERT of a live key is refused; a legacy patch holding
/// two entries for one key loads as `ambiguous`, refuses UPDATE, and DELETE
/// removes both.
#[test]
fn duplicate_keys_are_refused_and_legacy_duplicates_are_ambiguous() {
    let container = v3_container();
    let dir = tempfile::tempdir().unwrap();
    let mut session = bound(container.path());
    run(&mut session, INSERT_AB);
    let why = refused(
        &mut session,
        r#"INSERT INTO EDGES (entity, relation, target) VALUES ("A", "b", "[8]");"#,
    );
    assert!(why.contains("already exists"), "{why}");
    assert_eq!(only_fact(&session).target, "[7]");

    let mut fresh = bound(container.path());
    let mut patch = saved_patch(&mut fresh, dir.path(), "dup.vlp", &[INSERT_AB]);
    let mut second = patch["operations"][0].clone();
    second["target"] = "[8]".into();
    second["target_id"] = 8.into();
    patch["operations"].as_array_mut().unwrap().push(second);
    let legacy = write_patch(dir.path(), "legacy-dup.vlp", &patch);

    let mut session = bound(container.path());
    run(&mut session, &format!("APPLY PATCH \"{legacy}\";"));
    let fact = only_fact(&session);
    assert_eq!(fact.status, FactStatus::Ambiguous);
    let why = refused(&mut session, UPDATE_AB);
    assert!(why.contains("more than one realization"), "{why}");
    run(&mut session, DELETE_AB);
    assert!(facts(&session).is_empty());
    assert!(!infer(&mut session, CANONICAL_PROMPT).contains("knn_override"));
}

/// A KNN fact the binding has no tokenizer to check is `unverified`, never
/// `active`: its text and token id cannot be shown to agree, so no INFER
/// behaviour is claimed. UPDATE is refused with the session unchanged, and
/// DELETE (which does not depend on the target) removes it.
#[test]
fn a_fact_with_no_tokenizer_to_check_it_is_unverified() {
    let container = v3_container();
    let dir = tempfile::tempdir().unwrap();
    let mut session = bound(container.path());
    run(&mut session, INSERT_AB);
    let compiled = dir.path().join("compiled");
    run(
        &mut session,
        &format!("COMPILE CURRENT INTO VINDEX \"{}\";", lql_path(&compiled)),
    );
    std::fs::remove_file(compiled.join("tokenizer.json")).unwrap();

    let mut reopened = bound(&compiled);
    let fact = only_fact(&reopened);
    assert_eq!(fact.status, FactStatus::Unverified);
    assert_eq!(fact.target_id, Some(7));
    // SELECT is not reachable here: VINDEX3 browse needs the tokenizer
    // capability. The typed catalogue is the structured result.

    let before = facts(&reopened);
    let why = refused(&mut reopened, UPDATE_AB);
    assert!(
        why.contains("unverified") && why.contains("no tokenizer"),
        "{why}"
    );
    assert_eq!(facts(&reopened), before);

    run(&mut reopened, DELETE_AB);
    assert!(facts(&reopened).is_empty());
}
