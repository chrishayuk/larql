//! FACT-ID-1 storage layer (`docs/fact-id-1.md` §3.2, §6.1): the fact key,
//! the two fact ops, how they serialise, and how both overlays replay them.

use larql_vindex::format::vindex3::knowledge::KnowledgeOverlay;
use larql_vindex::patch::format::encode_gate_vector;
use larql_vindex::{
    normalize_fact_part, FactKey, KnnStore, PatchOp, PatchedVindex, VectorIndex, VindexPatch,
};

const HIDDEN: usize = 4;
const LAYERS: usize = 3;
const FEATURES: usize = 2;
const LAYER: usize = 1;

fn patch(operations: Vec<PatchOp>) -> VindexPatch {
    VindexPatch {
        version: 1,
        base_model: "fact/model".into(),
        base_checksum: None,
        created_at: String::new(),
        description: None,
        author: None,
        tags: vec![],
        operations,
    }
}

fn insert_knn(entity: &str, relation: &str, target: &str, target_id: u32) -> PatchOp {
    PatchOp::InsertKnn {
        layer: LAYER,
        entity: entity.into(),
        relation: relation.into(),
        target: target.into(),
        target_id,
        confidence: Some(0.5),
        key_vector_b64: encode_gate_vector(&[1.0, 0.0, 0.0, 0.0]),
    }
}

fn update_fact(entity: &str, relation: &str, target: &str, target_id: u32) -> PatchOp {
    PatchOp::UpdateKnnFact {
        entity: entity.into(),
        relation: relation.into(),
        target: target.into(),
        target_id,
        confidence: None,
    }
}

fn delete_fact(entity: &str, relation: &str) -> PatchOp {
    PatchOp::DeleteKnnFact {
        entity: entity.into(),
        relation: relation.into(),
    }
}

fn v2() -> PatchedVindex {
    let gates = vec![Some(ndarray::Array2::<f32>::zeros((FEATURES, HIDDEN))); LAYERS];
    PatchedVindex::new(VectorIndex::new(gates, vec![None; LAYERS], LAYERS, HIDDEN))
}

/// (layer, entity, relation, target, target_id, confidence) of every entry.
fn rows(store: &KnnStore) -> Vec<(usize, String, String, String, u32, f32)> {
    store
        .ordered_entries()
        .into_iter()
        .map(|(l, e)| {
            (
                l,
                e.entity.clone(),
                e.relation.clone(),
                e.target_token.clone(),
                e.target_id,
                e.confidence,
            )
        })
        .collect()
}

// ── the key ─────────────────────────────────────────────────────────────

#[test]
fn the_normal_form_folds_case_separators_and_whitespace() {
    assert_eq!(
        normalize_fact_part("  Capital_of-the  CITY "),
        "capital of the city"
    );
    assert_eq!(
        FactKey::new("France", "capital-of"),
        FactKey::new("FRANCE", "Capital Of")
    );
    assert_ne!(
        FactKey::new("France", "capital"),
        FactKey::new("France", "currency")
    );
}

#[test]
fn the_fact_id_depends_on_the_normal_form_only() {
    let id = FactKey::new("France", "capital_of").fact_id();
    assert_eq!(id, FactKey::new(" france ", "Capital-Of").fact_id());
    assert!(id.starts_with("f_"));
    assert_eq!(id.len(), 2 + 16);
    assert!(id[2..].chars().all(|c| c.is_ascii_hexdigit()));
    assert_ne!(id, FactKey::new("France", "capital").fact_id());
    // The separator keeps (ab, c) and (a, bc) apart.
    assert_ne!(
        FactKey::new("ab", "c").fact_id(),
        FactKey::new("a", "bc").fact_id()
    );
}

#[test]
fn a_key_names_spellings_of_itself_and_exposes_its_parts() {
    let key = FactKey::new("New_York", "Located In");
    assert!(key.names("new york", "located-in"));
    assert!(!key.names("new york", "located"));
    assert_eq!(key.entity(), "new york");
    assert_eq!(key.relation(), "located in");
}

// ── the ops on the wire ─────────────────────────────────────────────────

#[test]
fn the_fact_ops_serialise_under_their_own_tags() {
    let delete = serde_json::to_value(delete_fact("a", "b")).unwrap();
    assert_eq!(delete["op"], "delete_knn_fact");
    assert_eq!(delete["relation"], "b");
    let update = serde_json::to_value(update_fact("a", "b", "t", 7)).unwrap();
    assert_eq!(update["op"], "update_knn_fact");
    assert_eq!(update["target_id"], 7);
    for op in [delete_fact("a", "b"), update_fact("a", "b", "t", 7)] {
        let back: PatchOp = serde_json::from_value(serde_json::to_value(&op).unwrap()).unwrap();
        assert_eq!(
            serde_json::to_value(&back).unwrap(),
            serde_json::to_value(&op).unwrap()
        );
        assert_eq!(back.key(), None, "a fact op names no slot");
    }
}

#[test]
fn an_unknown_op_tag_fails_to_parse_so_an_older_reader_refuses_the_patch() {
    let patch_with = |op: &str| {
        format!(
            r#"{{"version":1,"base_model":"m","created_at":"","operations":[{{"op":"{op}","entity":"a","relation":"b"}}]}}"#
        )
    };
    // The control parses, so the refusal below is the tag's alone.
    assert!(serde_json::from_str::<VindexPatch>(&patch_with("delete_knn_fact")).is_ok());
    assert!(serde_json::from_str::<VindexPatch>(&patch_with("delete_knn_fact_v9")).is_err());
}

#[test]
fn patch_counts_read_an_update_as_an_update_and_a_delete_as_a_delete() {
    let p = patch(vec![
        insert_knn("a", "b", "t", 1),
        update_fact("a", "b", "u", 2),
        delete_fact("a", "b"),
    ]);
    assert_eq!(p.counts(), (1, 1, 1));
}

// ── the store ───────────────────────────────────────────────────────────

#[test]
fn update_fact_replaces_target_id_and_text_and_keeps_key_and_layer() {
    let mut store = KnnStore::default();
    store.add(
        LAYER,
        vec![1.0, 0.0, 0.0, 0.0],
        3,
        "t3".into(),
        "A".into(),
        "rel".into(),
        0.5,
    );
    store.add(
        LAYER,
        vec![0.0, 1.0, 0.0, 0.0],
        4,
        "t4".into(),
        "A".into(),
        "other".into(),
        0.5,
    );
    let key_before = store.ordered_entries()[0].1.key.clone();

    assert_eq!(
        store.update_fact(&FactKey::new("a", "REL"), 9, "t9", Some(0.75)),
        1
    );

    assert_eq!(
        rows(&store),
        vec![
            (LAYER, "A".into(), "rel".into(), "t9".into(), 9, 0.75),
            (LAYER, "A".into(), "other".into(), "t4".into(), 4, 0.5),
        ]
    );
    assert_eq!(store.ordered_entries()[0].1.key, key_before);
    // The retrieval key still finds the updated entry.
    let (hit, _) = store.query_top1(LAYER, &[1.0, 0.0, 0.0, 0.0]).unwrap();
    assert_eq!(hit.target_id, 9);
    // `None` keeps the stored confidence; an absent key changes nothing.
    assert_eq!(
        store.update_fact(&FactKey::new("a", "rel"), 8, "t8", None),
        1
    );
    assert_eq!(store.ordered_entries()[0].1.confidence, 0.75);
    assert_eq!(
        store.update_fact(&FactKey::new("z", "rel"), 1, "x", None),
        0
    );
}

#[test]
fn remove_fact_removes_one_relation_of_an_entity_and_keeps_the_others() {
    let mut store = KnnStore::default();
    store.add(
        0,
        vec![1.0, 0.0, 0.0, 0.0],
        1,
        "x".into(),
        "A".into(),
        "capital_of".into(),
        1.0,
    );
    store.add(
        2,
        vec![0.0, 1.0, 0.0, 0.0],
        2,
        "y".into(),
        "a".into(),
        "Capital Of".into(),
        1.0,
    );
    store.add(
        2,
        vec![0.0, 0.0, 1.0, 0.0],
        3,
        "z".into(),
        "A".into(),
        "currency".into(),
        1.0,
    );

    store.remove_fact(&FactKey::new("a", "capital-of"));

    assert_eq!(
        rows(&store),
        vec![(2, "A".into(), "currency".into(), "z".into(), 3, 1.0)]
    );
    let (hit, _) = store.query_top1(2, &[0.0, 0.0, 1.0, 0.0]).unwrap();
    assert_eq!(hit.target_id, 3, "key rows renumber with the entries");
}

#[test]
fn apply_fact_op_handles_the_fact_ops_and_declines_every_other() {
    let mut store = KnnStore::default();
    store.add(
        0,
        vec![1.0, 0.0, 0.0, 0.0],
        1,
        "x".into(),
        "A".into(),
        "r".into(),
        1.0,
    );
    assert!(store.apply_fact_op(&update_fact("a", "r", "y", 2)));
    assert_eq!(store.ordered_entries()[0].1.target_id, 2);
    assert!(!store.apply_fact_op(&PatchOp::DeleteKnn { entity: "a".into() }));
    assert_eq!(store.len(), 1);
    assert!(store.apply_fact_op(&delete_fact("A", "R")));
    assert!(store.is_empty());
}

// ── replay on both overlays ─────────────────────────────────────────────

#[test]
fn the_vindex2_overlay_replays_fact_ops_and_rebuilds_on_removal() {
    let mut pv = v2();
    pv.try_apply_patch(patch(vec![
        insert_knn("A", "r", "t1", 1),
        insert_knn("A", "s", "t5", 5),
    ]))
    .unwrap();
    pv.try_apply_patch(patch(vec![
        update_fact("a", "r", "t2", 2),
        delete_fact("a", "s"),
    ]))
    .unwrap();
    assert_eq!(
        rows(&pv.knn_store),
        vec![(LAYER, "A".into(), "r".into(), "t2".into(), 2, 0.5)]
    );

    pv.remove_patch(1);
    assert_eq!(
        rows(&pv.knn_store).len(),
        2,
        "removing the patch restores both facts"
    );
    assert_eq!(rows(&pv.knn_store)[0].4, 1);
}

#[test]
fn the_vindex3_overlay_replays_fact_ops_over_the_container_entries() {
    let mut base = KnnStore::default();
    base.add(
        LAYER,
        vec![1.0, 0.0, 0.0, 0.0],
        1,
        "t1".into(),
        "A".into(),
        "r".into(),
        0.5,
    );
    let mut overlay = KnowledgeOverlay::default();
    overlay.bind_base_knn(base);

    overlay
        .try_apply_patch(patch(vec![update_fact("a", "r", "t2", 2)]))
        .unwrap();
    assert_eq!(rows(&overlay.knn_store)[0].4, 2);
    overlay
        .try_apply_patch(patch(vec![delete_fact("a", "r")]))
        .unwrap();
    assert!(overlay.knn_store.is_empty());

    overlay.remove_patch(1);
    assert_eq!(
        rows(&overlay.knn_store)[0].4,
        2,
        "the update patch still applies"
    );
    overlay.remove_patch(0);
    assert_eq!(
        rows(&overlay.knn_store)[0].4,
        1,
        "the container's own entry is back"
    );
    assert_eq!(rows(overlay.base_knn()), rows(&overlay.knn_store));
}

#[test]
fn the_session_journal_records_in_order_and_clears_when_the_overlay_resets() {
    let mut overlay = KnowledgeOverlay::default();
    overlay.record_session_op(insert_knn("a", "r", "t", 1));
    overlay.try_apply_patch(patch(vec![])).unwrap();
    overlay.record_session_op(delete_fact("a", "r"));
    let stamps: Vec<usize> = overlay.session_ops().iter().map(|(n, _)| *n).collect();
    assert_eq!(
        stamps,
        vec![0, 1],
        "each op carries the patches applied before it"
    );
    overlay.remove_patch(0);
    assert!(overlay.session_ops().is_empty());

    let mut pv = v2();
    let mut base = KnnStore::default();
    base.add(
        0,
        vec![1.0, 0.0, 0.0, 0.0],
        1,
        "x".into(),
        "c".into(),
        "r".into(),
        1.0,
    );
    pv.bind_base_knn(base);
    assert_eq!(pv.base_knn().len(), 1);
    assert_eq!(pv.knn_store.len(), 1);
    pv.record_session_op(delete_fact("c", "r"));
    assert_eq!(pv.session_ops().len(), 1);
    pv.try_apply_patch(patch(vec![])).unwrap();
    pv.remove_patch(0);
    assert!(pv.session_ops().is_empty());
    // VINDEX2 rebuilds the store from the vindex's own entries (Gate B, B2;
    // this was docs/fact-id-1.md C-11).
    assert_eq!(pv.knn_store.len(), 1);
    assert_eq!(pv.base_knn().len(), 1);
}
