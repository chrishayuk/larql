use larql_vindex::ndarray::Array2;
use larql_vindex::patch::format::encode_gate_vector;
use larql_vindex::{KnnStore, PatchOp, PatchedVindex, VectorIndex, VindexPatch};

const HIDDEN: usize = 4;
const LAYERS: usize = 3;
const FEATURES: usize = 6;

fn pv() -> PatchedVindex {
    let gates = vec![Some(Array2::<f32>::zeros((FEATURES, HIDDEN))); LAYERS];
    PatchedVindex::new(VectorIndex::new(gates, vec![None; LAYERS], LAYERS, HIDDEN))
}
fn patch(name: &str, operations: Vec<PatchOp>) -> VindexPatch {
    VindexPatch { version: 1, base_model: "m".into(), base_checksum: None, created_at: String::new(),
        description: Some(name.into()), author: None, tags: vec![], operations }
}
fn v(x: f32) -> Option<String> { Some(encode_gate_vector(&vec![x; HIDDEN])) }

#[test]
fn edit9_remove_patch_restores_up_down() {
    let mut p = pv();
    p.try_apply_patch(patch("p", vec![PatchOp::Insert { layer: 1, feature: 2, relation: Some("r".into()),
        entity: "e".into(), target: "t".into(), confidence: Some(0.9),
        gate_vector_b64: v(1.0), up_vector_b64: v(2.0), down_vector_b64: v(3.0), down_meta: None }])).unwrap();
    assert!(p.up_override_at(1, 2).is_some() && p.down_override_at(1, 2).is_some());
    p.remove_patch(0);
    assert!(p.overrides_gate_at(1, 2).is_none(), "gate removed");
    assert!(p.feature_meta(1, 2).is_none(), "meta removed");
    eprintln!("after REMOVE: up={:?} down={:?}", p.up_override_at(1, 2), p.down_override_at(1, 2));
    assert!(p.up_override_at(1, 2).is_none(), "EDIT-9: up override survives remove_patch");
    assert!(p.down_override_at(1, 2).is_none(), "EDIT-9: down override survives remove_patch");
}

#[test]
fn v2_remove_patch_keeps_base_knn() {
    let mut p = pv();
    // as USE does: knn_store.bin loaded straight into patched.knn_store
    p.knn_store.add(1, vec![1.0; HIDDEN], 7, "Paris".into(), "France".into(), "capital".into(), 1.0);
    p.try_apply_patch(patch("q", vec![PatchOp::Delete { layer: 0, feature: 0, reason: None }])).unwrap();
    p.remove_patch(0);
    assert_eq!(p.knn_store.len(), 1, "V2 REMOVE PATCH dropped the vindex's own knn_store.bin entries");
}

#[test]
fn knn_duplicate_key_first_insert_wins() {
    let mut s = KnnStore::default();
    s.add(1, vec![1.0; HIDDEN], 7, "Paris".into(), "France".into(), "capital".into(), 1.0);
    s.add(1, vec![1.0; HIDDEN], 8, "Lyon".into(), "France".into(), "capital".into(), 1.0);
    assert_eq!(s.len(), 2);
    let (e, _) = s.query_top1(1, &vec![1.0; HIDDEN]).unwrap();
    assert_eq!(e.target_token, "Lyon", "re-assertion did not take effect; got {}", e.target_token);
}
