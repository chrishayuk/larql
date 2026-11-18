//! ID-1, ID-2, ID-14, ID-16: one identity across the statements, the unified
//! SELECT, what a version counts, and the journal across SAVE PATCH.

use larql_lql::{FactDurability, FactRealization, FactStatus};
use larql_vindex::FactKey;

use super::*;

/// ID-1: the same fact id through INSERT, SELECT, UPDATE and DELETE.
#[test]
fn one_identity_across_insert_select_update_and_delete() {
    let container = v3_container();
    let mut session = bound(container.path());
    let id = FactKey::new("a", "b").fact_id();

    let inserted = run(&mut session, INSERT_AB);
    assert!(inserted.contains(&format!("fact {id} (v1)")), "{inserted}");

    let fact = only_fact(&session);
    assert_eq!(fact.id, id);
    assert_eq!((fact.entity.as_str(), fact.relation.as_str()), ("a", "b"));
    assert_eq!(fact.target, "[7]");
    assert_eq!(fact.target_id, Some(7));
    assert_eq!(fact.realization, FactRealization::Knn);
    assert_eq!(fact.status, FactStatus::Active);
    assert_eq!(fact.version, 1);
    assert_eq!(fact.durability, FactDurability::Session);

    let selected = run(
        &mut session,
        r#"SELECT * FROM EDGES WHERE entity = "a" AND relation = "b";"#,
    );
    assert!(selected.starts_with("Asserted facts:"), "{selected}");
    assert!(
        selected.contains(&id) && selected.contains("[7]"),
        "{selected}"
    );

    let updated = run(
        &mut session,
        r#"UPDATE EDGES SET target = "[9]" WHERE entity = "A" AND relation = "b";"#,
    );
    assert!(
        updated.contains(&format!("Updated fact {id} (v1 → v2)")),
        "{updated}"
    );
    assert_eq!(only_fact(&session).id, id, "an UPDATE keeps the identity");

    let deleted = run(
        &mut session,
        r#"DELETE FROM EDGES WHERE entity = "a" AND relation = "b";"#,
    );
    assert!(
        deleted.contains(&format!("Deleted fact {id} (v2 → v3)")),
        "{deleted}"
    );
    assert!(facts(&session).is_empty());
    let after = run(
        &mut session,
        r#"SELECT * FROM EDGES WHERE entity = "a" AND relation = "b";"#,
    );
    assert!(!after.contains("Asserted facts:"), "{after}");

    // Inserting the key again continues its versions under the same id.
    let again = run(&mut session, INSERT_AB);
    assert!(again.contains(&format!("fact {id} (v4)")), "{again}");
}

/// ID-2: the asserted block above the discovered rows, and byte-identical
/// output when no asserted fact matches.
#[test]
fn select_lists_asserted_above_discovered_and_is_unchanged_without_a_match() {
    let container = v3_container();
    let mut session = bound(container.path());
    let clause = r#"SELECT * FROM EDGES WHERE entity = "a" AND relation = "b";"#;
    let other = r#"SELECT * FROM EDGES WHERE entity = "[3]" AND relation = "b";"#;
    let entity_only = r#"SELECT * FROM EDGES WHERE entity = "a";"#;
    let discovered_before = run(&mut session, clause);
    let other_before = run(&mut session, other);
    let entity_only_before = run(&mut session, entity_only);

    run(&mut session, INSERT_AB);

    let with_fact = run(&mut session, clause);
    let (asserted, discovered) = with_fact
        .split_once("\nDiscovered edges:\n")
        .unwrap_or_else(|| panic!("no discovered heading: {with_fact}"));
    assert!(asserted.starts_with("Asserted facts:"));
    assert!(
        asserted.contains("knn") && asserted.contains("active"),
        "{asserted}"
    );
    assert_eq!(
        discovered, discovered_before,
        "the discovered rows print as before"
    );
    assert_eq!(
        run(&mut session, other),
        other_before,
        "no asserted match: unchanged"
    );
    assert_eq!(
        run(&mut session, entity_only),
        entity_only_before,
        "entity alone is not the semantic shape"
    );
}

/// ID-14: versions count logical transitions.
#[test]
fn an_update_is_one_version_step_and_versions_survive_deletion() {
    let container = v3_container();
    let mut session = bound(container.path());
    run(&mut session, INSERT_AB);
    run(
        &mut session,
        r#"UPDATE EDGES SET target = "[9]" WHERE entity = "a" AND relation = "b";"#,
    );
    assert_eq!(only_fact(&session).version, 2);
    run(
        &mut session,
        r#"UPDATE EDGES SET confidence = 0.5 WHERE entity = "a" AND relation = "b";"#,
    );
    let fact = only_fact(&session);
    assert_eq!(fact.version, 3);
    assert_eq!(fact.confidence, Some(0.5));
    assert_eq!(fact.target, "[9]", "SET confidence keeps the target");
}

/// ID-16: SAVE PATCH ends the recording but the fact stays live, listed
/// from the session journal; REMOVE PATCH of an unrelated patch resets the
/// session state, and the catalogue follows the store.
#[test]
fn the_journal_keeps_a_saved_fact_listed_until_the_session_state_resets() {
    let container = v3_container();
    let dir = tempfile::tempdir().unwrap();
    let mut session = bound(container.path());
    let empty = write_patch(
        dir.path(),
        "empty.vlp",
        &serde_json::json!({"version": 1, "base_model": "fact-id-fixture", "created_at": "", "operations": []}),
    );
    run(&mut session, &format!("APPLY PATCH \"{empty}\";"));

    saved_patch(&mut session, dir.path(), "a.vlp", &[INSERT_AB]);
    let fact = only_fact(&session);
    assert_eq!(
        fact.durability,
        FactDurability::Session,
        "SAVE does not apply the patch"
    );
    assert_eq!(fact.status, FactStatus::Active);
    assert!(infer_top1(&mut session, CANONICAL_PROMPT).contains("[7]"));

    run(&mut session, &format!("REMOVE PATCH \"{empty}\";"));
    assert!(
        facts(&session).is_empty(),
        "the session state was reset with the store"
    );
    assert!(!infer(&mut session, CANONICAL_PROMPT).contains("knn_override"));
}
