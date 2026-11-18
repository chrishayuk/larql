//! ID-9: compose facts are read-only by identity, and COMPILE says that
//! their identity does not survive it.

use larql_lql::{FactRealization, FactStatus};

use super::*;

/// Entity and relation are real fixture tokens, so the capture prompt
/// carries embedding signal (see `tests/vindex3_mutation.rs`).
const COMPOSE_INSERT: &str = r#"INSERT INTO EDGES (entity, relation, target) VALUES ("[3]", "[7]", "[5]") ALPHA 5000.0 MODE COMPOSE;"#;

#[test]
fn compose_facts_are_listed_read_only_and_lose_identity_on_compile() {
    let container = v3_container();
    let dir = tempfile::tempdir().unwrap();
    let mut session = bound(container.path());
    let out = run(&mut session, COMPOSE_INSERT);
    let fact = only_fact(&session);
    assert_eq!(fact.realization, FactRealization::Compose);
    assert_eq!(fact.status, FactStatus::Compose);
    assert_eq!(fact.version, 1, "one compose INSERT is one transition");
    assert!(!fact.slots.is_empty());
    assert!(out.contains(&format!("fact {} (v1)", fact.id)), "{out}");

    let before = observe(&mut session);
    for stmt in [
        r#"UPDATE EDGES SET target = "[6]" WHERE entity = "[3]" AND relation = "[7]";"#,
        r#"DELETE FROM EDGES WHERE entity = "[3]" AND relation = "[7]";"#,
    ] {
        let why = refused(&mut session, stmt);
        assert!(why.contains("compose install"), "{why}");
        assert!(why.contains("REMOVE PATCH"), "{why}");
        assert_eq!(observe(&mut session), before, "{stmt} changed the session");
    }
    // A compose target is not held to the KNN single-token rule (D-5).
    run(
        &mut session,
        r#"INSERT INTO EDGES (entity, relation, target) VALUES ("[4]", "[7]", "[5] [6]") ALPHA 5000.0 MODE COMPOSE;"#,
    );

    let compiled = dir.path().join("compiled");
    let compile = run(
        &mut session,
        &format!("COMPILE CURRENT INTO VINDEX \"{}\";", lql_path(&compiled)),
    );
    assert!(
        compile.contains("2 asserted compose fact(s) lose their identity"),
        "{compile}"
    );
    let reopened = bound(&compiled);
    assert!(
        facts(&reopened).is_empty(),
        "compiled compose installs read back as discovered rows"
    );
}

#[test]
fn a_physical_delete_of_a_compose_slot_degrades_the_fact() {
    let container = v3_container();
    let mut session = bound(container.path());
    run(&mut session, COMPOSE_INSERT);
    let (layer, feature) = only_fact(&session).slots[0];
    run(
        &mut session,
        &format!("DELETE FROM EDGES WHERE layer = {layer} AND feature = {feature};"),
    );
    assert_eq!(only_fact(&session).status, FactStatus::Degraded);
    let why = refused(
        &mut session,
        r#"DELETE FROM EDGES WHERE entity = "[3]" AND relation = "[7]";"#,
    );
    assert!(why.contains("degraded"), "{why}");
}
