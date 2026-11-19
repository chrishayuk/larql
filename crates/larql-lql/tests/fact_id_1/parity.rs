//! ID-8: VINDEX2 parity. One checkpoint is realised as a VINDEX2 vindex
//! (real extraction) and a VINDEX3 container; the same fact script gives the
//! same ids, targets, statuses and versions on both. VINDEX2's REMOVE PATCH
//! keeps the vindex's own KNN facts (Gate B, B2, which fixes
//! docs/fact-id-1.md C-11). A change to the store that no op records is
//! still listed `inconsistent`, not passed off as the ledger's view.

use larql_lql::{FactDurability, FactStatus};
use larql_vindex::format::vindex3::fixtures::{dense_f32_model, DENSE_VOCAB};

use super::*;

const MODEL: &str = "fact-id-dense";

fn v2_vindex() -> tempfile::TempDir {
    let checkpoint = tempfile::tempdir().unwrap();
    dense_f32_model(checkpoint.path());
    let weights = larql_inference::load_model_dir(checkpoint.path()).expect("load checkpoint");
    let out = tempfile::tempdir().unwrap();
    let tok_json = knn_tokenizer_json(DENSE_VOCAB);
    let tokenizer = larql_vindex::tokenizers::Tokenizer::from_bytes(tok_json.as_bytes()).unwrap();
    let mut cb = larql_vindex::SilentBuildCallbacks;
    larql_vindex::build_vindex(
        &weights,
        &tokenizer,
        MODEL,
        out.path(),
        8,
        larql_vindex::ExtractLevel::All,
        larql_vindex::StorageDtype::F32,
        &mut cb,
    )
    .expect("build VINDEX2 vindex");
    std::fs::write(out.path().join("tokenizer.json"), &tok_json).unwrap();
    out
}

fn v3_dense() -> tempfile::TempDir {
    let checkpoint = tempfile::tempdir().unwrap();
    let container = tempfile::tempdir().unwrap();
    encode_fixture_container(dense_f32_model, checkpoint.path(), container.path(), MODEL);
    std::fs::write(
        container.path().join("tokenizer.json"),
        knn_tokenizer_json(DENSE_VOCAB),
    )
    .unwrap();
    container
}

const SCRIPT: &[&str] = &[
    r#"INSERT INTO EDGES (entity, relation, target) VALUES ("[2]", "b", "[5]");"#,
    r#"INSERT INTO EDGES (entity, relation, target) VALUES ("[3]", "c", "[6]");"#,
    r#"UPDATE EDGES SET target = "[7]" WHERE entity = "[2]" AND relation = "b";"#,
    r#"DELETE FROM EDGES WHERE entity = "[3]" AND relation = "c";"#,
    r#"INSERT INTO EDGES (entity, relation, target) VALUES ("[3]", "c", "[4]");"#,
];

fn logical(session: &Session) -> Vec<(String, String, Option<u32>, FactStatus, u32)> {
    facts(session)
        .into_iter()
        .map(|f| (f.id, f.target, f.target_id, f.status, f.version))
        .collect()
}

#[test]
fn the_same_script_gives_the_same_facts_on_vindex2_and_vindex3() {
    let (v2, v3) = (v2_vindex(), v3_dense());
    let mut on_v2 = bound(v2.path());
    let mut on_v3 = bound(v3.path());
    for stmt in SCRIPT {
        run(&mut on_v2, stmt);
        run(&mut on_v3, stmt);
    }
    let v2_facts = logical(&on_v2);
    assert_eq!(v2_facts.len(), 2);
    assert_eq!(v2_facts, logical(&on_v3));
    assert_eq!(v2_facts[0].4, 2, "([2], b): insert + update");
    assert_eq!(v2_facts[1].4, 3, "([3], c): insert + delete + insert");
}

/// A VINDEX2 vindex compiled with one KNN fact, bound in a fresh session.
fn compiled_with_one_fact(dir: &Path) -> Session {
    let v2 = v2_vindex();
    let mut session = bound(v2.path());
    run(&mut session, SCRIPT[0]);
    let compiled = dir.join("compiled");
    run(
        &mut session,
        &format!("COMPILE CURRENT INTO VINDEX \"{}\";", lql_path(&compiled)),
    );
    let reopened = bound(&compiled);
    assert_eq!(only_fact(&reopened).status, FactStatus::Active);
    reopened
}

/// The prompt the vindex's own fact `([2], b) -> [5]` answers.
const OWN_FACT_PROMPT: &str = "The b of [2] is";

/// ID-8 after Gate B, B2 (docs/fact-id-1.md C-11, fixed): REMOVE PATCH on
/// VINDEX2 keeps the vindex's own KNN facts as they were. A stronger
/// guarantee than before, which only detected the loss: the same fact ids,
/// still `active`, still held by the container, and INFER answers as it did
/// before the patch was applied.
#[test]
fn vindex2_remove_patch_keeps_the_vindexs_own_facts() {
    let dir = tempfile::tempdir().unwrap();
    let mut reopened = compiled_with_one_fact(dir.path());
    let own = only_fact(&reopened);
    assert_eq!(own.id, larql_vindex::FactKey::new("[2]", "b").fact_id());
    assert_eq!(own.status, FactStatus::Active);
    assert_eq!(own.durability, FactDurability::Container);
    let answer = infer_top1(&mut reopened, OWN_FACT_PROMPT);
    assert!(
        answer.contains("[5]"),
        "the own fact must drive INFER: {answer}"
    );
    let before = observe(&mut reopened);

    let empty = write_patch(
        dir.path(),
        "empty.vlp",
        &serde_json::json!({"version": 1, "base_model": MODEL, "created_at": "", "operations": []}),
    );
    run(&mut reopened, &format!("APPLY PATCH \"{empty}\";"));
    run(&mut reopened, &format!("REMOVE PATCH \"{empty}\";"));

    assert_eq!(only_fact(&reopened), own);
    assert_eq!(infer_top1(&mut reopened, OWN_FACT_PROMPT), answer);
    assert_eq!(observe(&mut reopened), before);
}

#[test]
fn a_store_change_no_op_records_reads_inconsistent() {
    let dir = tempfile::tempdir().unwrap();
    let mut reopened = compiled_with_one_fact(dir.path());
    // What an unrecorded removal (COMPACT MINOR, docs/fact-id-1.md C-12)
    // leaves: the entry is gone from the store, and no op says so.
    reopened
        .patched_overlay_mut()
        .expect("a VINDEX2 binding")
        .knn_store
        .remove_by_entity("[2]");
    let fact = only_fact(&reopened);
    assert_eq!(fact.status, FactStatus::Inconsistent);
    let why = refused(
        &mut reopened,
        r#"DELETE FROM EDGES WHERE entity = "[2]" AND relation = "b";"#,
    );
    assert!(why.contains("inconsistent"), "{why}");
}
