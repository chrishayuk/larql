//! ID-6, ID-7, ID-13, ID-15: rollback, replay and reload, cross-patch
//! conflicts, and old readers refusing the new ops.

use larql_lql::{FactDurability, FactStatus};
use larql_vindex::FactKey;

use super::*;

const UPDATE_AB: &str = r#"UPDATE EDGES SET target = "[9]" WHERE entity = "a" AND relation = "b";"#;
const DELETE_AB: &str = r#"DELETE FROM EDGES WHERE entity = "a" AND relation = "b";"#;

/// Patch A inserts (a, b) → [7]; patch B, written over A, updates it to
/// [9]; patch C, written over A, deletes it. Returns their LQL paths.
fn three_patches(container: &Path, dir: &Path) -> (String, String, String) {
    let a = lql_path(dir.join("a.vlp"));
    saved_patch(&mut bound(container), dir, "a.vlp", &[INSERT_AB]);
    for (name, stmt) in [("b.vlp", UPDATE_AB), ("c.vlp", DELETE_AB)] {
        let mut session = bound(container);
        run(&mut session, &format!("APPLY PATCH \"{a}\";"));
        saved_patch(&mut session, dir, name, &[stmt]);
    }
    (a, lql_path(dir.join("b.vlp")), lql_path(dir.join("c.vlp")))
}

/// ID-6: removing the patch that changed a fact restores the fact.
#[test]
fn removing_an_update_or_a_delete_patch_restores_the_fact() {
    let container = v3_container();
    let dir = tempfile::tempdir().unwrap();
    let (a, b, c) = three_patches(container.path(), dir.path());

    let mut session = bound(container.path());
    run(&mut session, &format!("APPLY PATCH \"{a}\";"));
    run(&mut session, &format!("APPLY PATCH \"{b}\";"));
    assert_eq!(only_fact(&session).target, "[9]");
    assert!(infer_top1(&mut session, CANONICAL_PROMPT).contains("[9]"));
    run(&mut session, &format!("REMOVE PATCH \"{b}\";"));
    let fact = only_fact(&session);
    assert_eq!(
        (fact.target.as_str(), fact.target_id, fact.version),
        ("[7]", Some(7), 1)
    );
    assert!(infer_top1(&mut session, CANONICAL_PROMPT).contains("[7]"));

    run(&mut session, &format!("APPLY PATCH \"{c}\";"));
    assert!(facts(&session).is_empty());
    run(&mut session, &format!("REMOVE PATCH \"{c}\";"));
    assert_eq!(only_fact(&session).target, "[7]");
    assert!(infer_top1(&mut session, CANONICAL_PROMPT).contains("[7]"));
}

/// ID-7: a fresh session applying the saved patch reproduces the catalogue
/// and INFER; COMPILE then reopen keeps KNN facts, as container facts with
/// the same ids; and (ID-6, VINDEX3) a container fact deleted by a patch is
/// restored by removing it.
#[test]
fn replay_and_compile_reload_reproduce_the_facts() {
    let container = v3_container();
    let dir = tempfile::tempdir().unwrap();
    let mut session = bound(container.path());
    saved_patch(
        &mut session,
        dir.path(),
        "script.vlp",
        &[
            INSERT_AB,
            r#"INSERT INTO EDGES (entity, relation, target) VALUES ("c", "d", "[4]");"#,
            UPDATE_AB,
        ],
    );
    let live_facts = facts(&session);
    let live_infer = infer(&mut session, CANONICAL_PROMPT);

    let mut replay = bound(container.path());
    let script = lql_path(dir.path().join("script.vlp"));
    run(&mut replay, &format!("APPLY PATCH \"{script}\";"));
    let replayed = facts(&replay);
    assert_eq!(replayed.len(), live_facts.len());
    for (live, again) in live_facts.iter().zip(&replayed) {
        assert_eq!(
            (
                &live.id,
                &live.target,
                live.target_id,
                live.status,
                live.version
            ),
            (
                &again.id,
                &again.target,
                again.target_id,
                again.status,
                again.version
            )
        );
        assert_eq!(again.durability, FactDurability::Patch);
    }
    assert_eq!(infer(&mut replay, CANONICAL_PROMPT), live_infer);

    let compiled = dir.path().join("compiled");
    run(
        &mut replay,
        &format!("COMPILE CURRENT INTO VINDEX \"{}\";", lql_path(&compiled)),
    );
    let mut reopened = bound(&compiled);
    let reloaded = facts(&reopened);
    assert_eq!(
        reloaded
            .iter()
            .map(|f| (&f.id, &f.target))
            .collect::<Vec<_>>(),
        live_facts
            .iter()
            .map(|f| (&f.id, &f.target))
            .collect::<Vec<_>>()
    );
    for fact in &reloaded {
        assert_eq!(fact.durability, FactDurability::Container);
        assert_eq!(
            fact.version, 1,
            "a compiled fact restarts at its container entry"
        );
        assert_eq!(fact.status, FactStatus::Active);
    }
    assert_eq!(infer(&mut reopened, CANONICAL_PROMPT), live_infer);

    // A container fact deleted by an applied patch, restored by removing it.
    let del = lql_path(dir.path().join("del.vlp"));
    saved_patch(&mut bound(&compiled), dir.path(), "del.vlp", &[DELETE_AB]);
    run(&mut reopened, &format!("APPLY PATCH \"{del}\";"));
    assert!(!facts(&reopened)
        .iter()
        .any(|f| f.id == FactKey::new("a", "b").fact_id()));
    run(&mut reopened, &format!("REMOVE PATCH \"{del}\";"));
    assert_eq!(facts(&reopened), reloaded);
}

/// ID-13: APPLY PATCH refuses a patch that asserts a live fact or changes an
/// absent one, and accepts one that deletes a fact and then inserts it.
#[test]
fn apply_patch_refuses_conflicting_assertions_whole() {
    let container = v3_container();
    let dir = tempfile::tempdir().unwrap();
    let (a, b, _) = three_patches(container.path(), dir.path());

    // B updates a fact this session does not hold.
    let mut session = bound(container.path());
    let why = refused(&mut session, &format!("APPLY PATCH \"{b}\";"));
    assert!(why.contains("not a live KNN fact"), "{why}");
    assert!(run(&mut session, "SHOW PATCHES;").contains("no patches applied"));

    // A asserts a fact the session already holds.
    run(&mut session, INSERT_AB);
    let why = refused(&mut session, &format!("APPLY PATCH \"{a}\";"));
    assert!(why.contains("already the live fact"), "{why}");

    // Delete-then-insert of the same key is a replacement, and is accepted.
    let mut patch: serde_json::Value =
        serde_json::from_slice(&std::fs::read(dir.path().join("a.vlp")).unwrap()).unwrap();
    let mut insert = patch["operations"][0].clone();
    insert["target"] = "[4]".into();
    insert["target_id"] = 4.into();
    patch["operations"] = serde_json::json!([
        {"op": "delete_knn_fact", "entity": "a", "relation": "b"},
        insert,
    ]);
    let replace = write_patch(dir.path(), "replace.vlp", &patch);
    run(&mut session, &format!("APPLY PATCH \"{replace}\";"));
    assert_eq!(only_fact(&session).target, "[4]");
}

/// ID-15: a patch carrying an op tag this reader does not know is refused,
/// and nothing is applied.
#[test]
fn a_patch_with_an_unknown_op_is_refused_whole() {
    let container = v3_container();
    let dir = tempfile::tempdir().unwrap();
    let mut session = bound(container.path());
    let unknown = write_patch(
        dir.path(),
        "unknown.vlp",
        &serde_json::json!({
            "version": 1, "base_model": "fact-id-fixture", "created_at": "",
            "operations": [{"op": "delete_knn_fact_v9", "entity": "a", "relation": "b"}]
        }),
    );
    let before = observe(&mut session);
    refused(&mut session, &format!("APPLY PATCH \"{unknown}\";"));
    assert_eq!(observe(&mut session), before);
}
