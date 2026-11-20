//! Gate B, B5: a compose install snapshot, restored, puts back every part of
//! an overlay an install writes (feature metadata, tombstones, gate, up and
//! down overrides, the session journal), on both overlays. The writes cover
//! each kind an install makes: a new slot, a rewrite of an existing slot (the
//! refine pass), a down column rescale (balance), a tombstone cleared by
//! resurrection, and a journal entry.

use larql_vindex::format::vindex3::knowledge::KnowledgeOverlay;
use larql_vindex::{FeatureMeta, PatchOp, PatchedVindex, VectorIndex};

const HIDDEN: usize = 4;
const LAYERS: usize = 2;
const FEATURES: usize = 4;
const LAYER: usize = 1;

fn meta(token: &str) -> FeatureMeta {
    FeatureMeta {
        top_token: token.into(),
        top_token_id: 1,
        c_score: 0.5,
        top_k: vec![],
    }
}

fn delete(feature: usize) -> PatchOp {
    PatchOp::Delete {
        layer: LAYER,
        feature,
        reason: None,
    }
}

fn v2() -> PatchedVindex {
    let gates = vec![Some(ndarray::Array2::<f32>::zeros((FEATURES, HIDDEN))); LAYERS];
    PatchedVindex::new(VectorIndex::new(gates, vec![None; LAYERS], LAYERS, HIDDEN))
}

fn v2_state(pv: &PatchedVindex) -> String {
    let mut gates: Vec<_> = pv
        .overrides_gate_iter()
        .map(|(l, f, g)| ((l, f), g.to_vec()))
        .collect();
    gates.sort_by_key(|(k, _)| *k);
    let mut ups: Vec<_> = pv.up_overrides().iter().collect();
    ups.sort_by(|a, b| a.0.cmp(b.0));
    let mut downs: Vec<_> = pv.down_overrides().iter().collect();
    downs.sort_by(|a, b| a.0.cmp(b.0));
    let metas: Vec<_> = (0..FEATURES)
        .map(|f| {
            (
                pv.feature_meta(LAYER, f).map(|m| m.top_token),
                pv.is_deleted(LAYER, f),
            )
        })
        .collect();
    format!(
        "{gates:?}\n{ups:?}\n{downs:?}\n{metas:?}\n{:?}",
        pv.session_ops()
    )
}

fn v3_state(ov: &KnowledgeOverlay) -> String {
    let mut gates: Vec<_> = ov
        .gate_overrides_at(LAYER)
        .into_iter()
        .map(|(f, g)| (f, g.to_vec()))
        .collect();
    gates.sort_by_key(|(f, _)| *f);
    let slots: Vec<_> = (0..FEATURES)
        .map(|f| {
            (
                ov.up_override_at(LAYER, f).map(<[f32]>::to_vec),
                ov.down_override_at(LAYER, f).map(<[f32]>::to_vec),
                ov.resolve_feature_meta(LAYER, f, None).map(|m| m.top_token),
                ov.is_tombstoned(LAYER, f),
            )
        })
        .collect();
    format!(
        "{gates:?}\n{slots:?}\n{}\n{:?}",
        ov.num_overrides(),
        ov.session_ops()
    )
}

#[test]
fn vindex2_restoring_the_snapshot_undoes_every_install_write() {
    let mut pv = v2();
    // Before: one installed slot, one tombstone, one journal entry.
    pv.insert_feature(LAYER, 0, vec![1.0; HIDDEN], meta("old"));
    pv.set_up_vector(LAYER, 0, vec![2.0; HIDDEN]);
    pv.set_down_vector(LAYER, 0, vec![3.0; HIDDEN]);
    pv.delete_feature(LAYER, 2);
    pv.record_session_op(delete(2));
    let before = v2_state(&pv);

    let snapshot = pv.compose_install_snapshot();
    pv.insert_feature(LAYER, 1, vec![4.0; HIDDEN], meta("new"));
    pv.set_up_vector(LAYER, 1, vec![5.0; HIDDEN]);
    pv.set_down_vector(LAYER, 1, vec![6.0; HIDDEN]);
    pv.set_gate_override(LAYER, 0, vec![7.0; HIDDEN]);
    pv.set_up_vector(LAYER, 0, vec![8.0; HIDDEN]);
    pv.set_down_vector(LAYER, 0, vec![9.0; HIDDEN]);
    pv.insert_feature(LAYER, 2, vec![1.5; HIDDEN], meta("resurrected"));
    pv.record_session_op(delete(1));
    assert_ne!(v2_state(&pv), before);

    pv.restore_compose_install(snapshot);
    assert_eq!(v2_state(&pv), before);
}

#[test]
fn vindex3_restoring_the_snapshot_undoes_every_install_write() {
    let mut ov = KnowledgeOverlay::new();
    ov.insert_feature(LAYER, 0, vec![1.0; HIDDEN], meta("old"));
    ov.set_up_vector(LAYER, 0, vec![2.0; HIDDEN]);
    ov.set_down_vector(LAYER, 0, vec![3.0; HIDDEN]);
    ov.delete_feature(LAYER, 2);
    ov.record_session_op(delete(2));
    let before = v3_state(&ov);

    let snapshot = ov.compose_install_snapshot();
    ov.insert_feature(LAYER, 1, vec![4.0; HIDDEN], meta("new"));
    ov.set_up_vector(LAYER, 1, vec![5.0; HIDDEN]);
    ov.set_down_vector(LAYER, 1, vec![6.0; HIDDEN]);
    ov.set_gate_vector(LAYER, 0, vec![7.0; HIDDEN]);
    ov.set_up_vector(LAYER, 0, vec![8.0; HIDDEN]);
    ov.set_down_vector(LAYER, 0, vec![9.0; HIDDEN]);
    ov.insert_feature(LAYER, 2, vec![1.5; HIDDEN], meta("resurrected"));
    ov.record_session_op(delete(1));
    assert_ne!(v3_state(&ov), before);

    ov.restore_compose_install(snapshot);
    assert_eq!(v3_state(&ov), before);
}

/// Restoring a snapshot taken with nothing written since changes nothing.
#[test]
fn an_unused_snapshot_restores_to_the_same_state() {
    let mut pv = v2();
    pv.insert_feature(LAYER, 0, vec![1.0; HIDDEN], meta("old"));
    let before = v2_state(&pv);
    let snapshot = pv.compose_install_snapshot();
    pv.restore_compose_install(snapshot);
    assert_eq!(v2_state(&pv), before);

    let mut ov = KnowledgeOverlay::new();
    ov.insert_feature(LAYER, 0, vec![1.0; HIDDEN], meta("old"));
    let before = v3_state(&ov);
    let snapshot = ov.compose_install_snapshot();
    ov.restore_compose_install(snapshot);
    assert_eq!(v3_state(&ov), before);
}
