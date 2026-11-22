//! Gate S reproducers at the overlay level: the exact `PatchedVindex` calls
//! larql-server makes for `/v1/insert`, remote UPDATE and REMOVE PATCH
//! (docs/gate-b-audit-2026-11-18.md, S-INSERT and S-UPDATE).

use larql_vindex::{ndarray::Array2, FeatureMeta, PatchOp, PatchedVindex, VectorIndex, VindexPatch};

fn meta(t: &str, s: f32) -> FeatureMeta {
    FeatureMeta { top_token: t.into(), top_token_id: 1, c_score: s, top_k: vec![] }
}

/// One layer, three slots that all hold model features.
fn base() -> PatchedVindex {
    let hidden = 4;
    let mut gate = Array2::<f32>::zeros((3, hidden));
    gate[[0, 0]] = 1.0;
    gate[[1, 1]] = 1.0;
    gate[[2, 2]] = 1.0;
    let m = vec![Some(meta("Paris", 0.95)), Some(meta("French", 0.88)), Some(meta("Europe", 0.75))];
    PatchedVindex::new(VectorIndex::new(vec![Some(gate)], vec![Some(m)], 1, hidden))
}

fn patch(name: &str, operations: Vec<PatchOp>) -> VindexPatch {
    VindexPatch {
        version: 1,
        base_model: "m".into(),
        base_checksum: None,
        created_at: String::new(),
        description: Some(name.into()),
        author: None,
        tags: vec![],
        operations,
    }
}

/// The op remote `UPDATE EDGES SET confidence = …` builds when SET has no
/// target (`larql-lql/src/executor/remote/mutation.rs`): no metadata at all.
fn bare_update(layer: usize, feature: usize) -> PatchOp {
    PatchOp::Update {
        layer,
        feature,
        gate_vector_b64: None,
        up_vector_b64: None,
        down_vector_b64: None,
        down_meta: None,
    }
}

/// Characterisation, not a failure: the server's "free slot" can be the
/// weakest real model feature (documented in server-spec §8.6).
#[test]
fn s_slot_the_chooser_can_pick_a_model_feature() {
    let pv = base();
    let slot = pv.find_free_feature_excluding(0, &Default::default());
    let held = slot.and_then(|s| pv.base().feature_meta(0, s)).map(|m| m.top_token);
    eprintln!("chosen slot {slot:?} holds base feature {held:?}");
}

/// S-INSERT: `/v1/insert` writes the overlay with no patch. Removing any
/// unrelated patch then drops the insert's metadata and gate but leaves its
/// down vector on the base slot.
#[test]
fn s_insert_survives_or_vanishes_whole_after_an_unrelated_remove() {
    let mut pv = base();
    pv.insert_feature(0, 2, vec![0.0, 0.0, 0.0, 1.0], meta("Rome", 0.9));
    pv.set_down_vector(0, 2, vec![9.0; 4]);
    pv.try_apply_patch(patch("unrelated", vec![PatchOp::Delete { layer: 0, feature: 0, reason: None }]))
        .expect("apply");
    pv.remove_patch(0);
    let meta_is_rome = pv.feature_meta(0, 2).map(|m| m.top_token).as_deref() == Some("Rome");
    let down_kept = pv.down_override_at(0, 2).is_some();
    assert_eq!(
        meta_is_rome, down_kept,
        "half an insert survived: meta Rome = {meta_is_rome}, down override kept = {down_kept}"
    );
}

/// S-UPDATE (1): remote `SET confidence = 0.5` alone changes nothing, yet
/// the client reports "Updated".
#[test]
fn s_update_set_confidence_takes_effect() {
    let mut pv = base();
    pv.try_apply_patch(patch("UPDATE L0 F1", vec![bare_update(0, 1)])).expect("apply");
    let c = pv.feature_meta(0, 1).map(|m| m.c_score);
    assert_eq!(c, Some(0.5), "remote SET confidence = 0.5 left c_score at {c:?}");
}

/// S-UPDATE (2): remote UPDATE on a deleted slot brings it back; local
/// UPDATE leaves it deleted.
#[test]
fn s_update_does_not_resurrect_a_deleted_slot() {
    let mut pv = base();
    pv.try_apply_patch(patch("DELETE L0 F1", vec![PatchOp::Delete { layer: 0, feature: 1, reason: None }]))
        .unwrap();
    pv.try_apply_patch(patch("UPDATE L0 F1", vec![bare_update(0, 1)])).unwrap();
    let m = pv.feature_meta(0, 1).map(|m| m.top_token);
    assert!(m.is_none(), "UPDATE resurrected deleted slot as {m:?}");
}
