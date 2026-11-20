//! Gate B, B6 §3.4, R5: what a live session compiles is byte for byte what
//! an independent replay compiles: a fresh session that binds the same
//! source and applies the saved patch.
//!
//! The edits include two compose installs at one layer, so the second
//! re-fits the first (B3), a KNN fact, and a tombstone.

use std::path::Path;

use crate::support::*;

const V2_SECOND: &str = r#"INSERT INTO EDGES (entity, relation, target) VALUES ("[4]", "[7]", "[6]") AT LAYER 0 MODE COMPOSE;"#;
const V3_SECOND: &str = r#"INSERT INTO EDGES (entity, relation, target) VALUES ("c", "b", "[6]") AT LAYER 1 MODE COMPOSE;"#;

fn assert_live_and_replayed_compiles_agree(source: &Path, edits: &[&str], dir: &Path) {
    let patch = dir.join("edits.vlp");
    let live = dir.join("live.out");
    let replayed = dir.join("replayed.out");

    let mut session = bound(source);
    run(
        &mut session,
        &format!("BEGIN PATCH \"{}\";", lql_path(&patch)),
    );
    for edit in edits {
        run(&mut session, edit);
    }
    run(&mut session, "SAVE PATCH;");
    run(
        &mut session,
        &format!("COMPILE CURRENT INTO VINDEX \"{}\";", lql_path(&live)),
    );

    let mut replay = bound(source);
    run(
        &mut replay,
        &format!("APPLY PATCH \"{}\";", lql_path(&patch)),
    );
    run(
        &mut replay,
        &format!("COMPILE CURRENT INTO VINDEX \"{}\";", lql_path(&replayed)),
    );

    assert_same_tree(&live, &replayed);
}

#[test]
fn vindex2_a_live_compile_equals_the_replayed_compile() {
    let dir = tempfile::tempdir().unwrap();
    let source = v2_vindex(dir.path());
    assert_live_and_replayed_compiles_agree(
        &source,
        &[
            V2_INSERT,
            V2_SECOND,
            V2_KNN[2],
            "DELETE FROM EDGES WHERE layer = 0 AND feature = 1;",
        ],
        dir.path(),
    );
}

#[test]
fn vindex3_a_live_compile_equals_the_replayed_compile() {
    let dir = tempfile::tempdir().unwrap();
    let source = v3_container(dir.path());
    assert_live_and_replayed_compiles_agree(
        &source,
        &[V3_INSERT, V3_SECOND, V3_KNN[2]],
        dir.path(),
    );
}
