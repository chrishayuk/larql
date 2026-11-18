//! ID-8: VINDEX2 parity. One checkpoint is realised as a VINDEX2 vindex
//! (real extraction) and a VINDEX3 container; the same fact script gives the
//! same ids, targets, statuses and versions on both. VINDEX2's REMOVE PATCH
//! drops the vindex's own KNN facts (docs/fact-id-1.md C-11), and the
//! catalogue reports them `inconsistent` rather than pretending otherwise.

use larql_lql::FactStatus;
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

#[test]
fn vindex2_remove_patch_dropping_container_facts_reads_inconsistent() {
    let v2 = v2_vindex();
    let dir = tempfile::tempdir().unwrap();
    let mut session = bound(v2.path());
    run(&mut session, SCRIPT[0]);
    let compiled = dir.path().join("compiled");
    run(
        &mut session,
        &format!("COMPILE CURRENT INTO VINDEX \"{}\";", lql_path(&compiled)),
    );

    let mut reopened = bound(&compiled);
    assert_eq!(only_fact(&reopened).status, FactStatus::Active);
    let empty = write_patch(
        dir.path(),
        "empty.vlp",
        &serde_json::json!({"version": 1, "base_model": MODEL, "created_at": "", "operations": []}),
    );
    run(&mut reopened, &format!("APPLY PATCH \"{empty}\";"));
    run(&mut reopened, &format!("REMOVE PATCH \"{empty}\";"));
    // VINDEX2 rebuilt its KNN store from empty: the container fact no longer
    // executes, and the catalogue lists it as inconsistent, not active.
    let fact = only_fact(&reopened);
    assert_eq!(fact.status, FactStatus::Inconsistent);
    let why = refused(
        &mut reopened,
        r#"DELETE FROM EDGES WHERE entity = "[2]" AND relation = "b";"#,
    );
    assert!(why.contains("inconsistent"), "{why}");
}
