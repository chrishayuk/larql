//! Gate B, B2 (`docs/gate-b.md`): removing a patch from a VINDEX2 overlay
//! restores exactly the state of the base as bound plus the remaining
//! patches, for every kind of state a patch writes.
//!
//! Before the fix, `rebuild_overrides` left the up/down vectors a removed
//! patch had written on the base (EDIT-9), and reset the KNN store to empty,
//! dropping the entries the vindex itself carries.
//!
//! The comparison is byte for byte: every f32 is compared by its bits.

use larql_vindex::ndarray::Array2;
use larql_vindex::patch::format::{encode_gate_vector, PatchDownMeta};
use larql_vindex::{KnnStore, PatchOp, PatchedVindex, VectorIndex, VindexPatch};

const HIDDEN: usize = 4;
const LAYERS: usize = 3;
const FEATURES: usize = 6;
/// The slot that carries the base's own up/down override.
const BASE_SLOT: (usize, usize) = (2, 3);
const BASE_UP: f32 = 7.0;
const BASE_DOWN: f32 = 8.0;
const KNN_LAYER: usize = 1;

fn vector(x: f32) -> Vec<f32> {
    vec![x; HIDDEN]
}

fn b64(x: f32) -> Option<String> {
    Some(encode_gate_vector(&vector(x)))
}

/// A base with its own up/down override and its own KNN entry.
fn bound() -> PatchedVindex {
    let gates = vec![Some(Array2::<f32>::zeros((FEATURES, HIDDEN))); LAYERS];
    let mut base = VectorIndex::new(gates, vec![None; LAYERS], LAYERS, HIDDEN);
    base.set_up_vector(BASE_SLOT.0, BASE_SLOT.1, vector(BASE_UP));
    base.set_down_vector(BASE_SLOT.0, BASE_SLOT.1, vector(BASE_DOWN));
    let mut pv = PatchedVindex::new(base);
    let mut knn = KnnStore::default();
    knn.add(
        KNN_LAYER,
        vector(1.0),
        7,
        "Paris".into(),
        "France".into(),
        "capital".into(),
        1.0,
    );
    pv.bind_base_knn(knn);
    pv
}

fn patch(name: &str, operations: Vec<PatchOp>) -> VindexPatch {
    VindexPatch {
        version: 1,
        base_model: "gate-b/remove".into(),
        base_checksum: None,
        created_at: String::new(),
        description: Some(name.into()),
        author: None,
        tags: vec![],
        operations,
    }
}

fn insert(layer: usize, feature: usize, x: f32) -> PatchOp {
    PatchOp::Insert {
        layer,
        feature,
        relation: Some("r".into()),
        entity: format!("e{layer}{feature}"),
        target: format!("t{layer}{feature}"),
        confidence: Some(0.9),
        gate_vector_b64: b64(x),
        up_vector_b64: b64(x + 1.0),
        down_vector_b64: b64(x + 2.0),
        down_meta: None,
    }
}

/// An update that rewrites the vectors (and metadata) of a slot.
fn update(layer: usize, feature: usize, x: f32) -> PatchOp {
    PatchOp::Update {
        layer,
        feature,
        gate_vector_b64: b64(x),
        up_vector_b64: b64(x + 1.0),
        down_vector_b64: b64(x + 2.0),
        down_meta: Some(PatchDownMeta {
            top_token: format!("u{layer}{feature}"),
            top_token_id: 3,
            c_score: 0.25,
        }),
    }
}

/// Every kind of op a patch can carry against the overlay, including ones
/// that overwrite the base's own override and delete its own KNN entry.
fn every_kind(name: &str, x: f32) -> VindexPatch {
    patch(
        name,
        vec![
            insert(0, 1, x),
            update(BASE_SLOT.0, BASE_SLOT.1, x),
            PatchOp::Delete {
                layer: 1,
                feature: 0,
                reason: None,
            },
            PatchOp::InsertKnn {
                layer: KNN_LAYER,
                entity: "Rome".into(),
                relation: "capital".into(),
                target: "Italy".into(),
                target_id: 9,
                confidence: Some(0.5),
                key_vector_b64: encode_gate_vector(&vector(x)),
            },
            PatchOp::DeleteKnn {
                entity: "Paris".into(),
            },
        ],
    )
}

fn bits(v: &[f32]) -> Vec<u32> {
    v.iter().map(|x| x.to_bits()).collect()
}

/// The effective state of every slot, the override maps, the tombstones as
/// `feature_meta` sees them and the KNN store, with every float as its bits.
fn state(pv: &PatchedVindex) -> String {
    let mut out = Vec::new();
    for layer in 0..LAYERS {
        for feature in 0..FEATURES {
            let meta = pv
                .feature_meta(layer, feature)
                .map(|m| (m.top_token, m.top_token_id, m.c_score.to_bits()));
            let gate = pv.overrides_gate_at(layer, feature).map(bits);
            let up = pv.up_override_at(layer, feature).map(bits);
            let down = pv.down_override_at(layer, feature).map(bits);
            out.push(format!(
                "L{layer}F{feature} {meta:?} {gate:?} {up:?} {down:?}"
            ));
        }
    }
    let mut up_keys: Vec<_> = pv.up_overrides().keys().copied().collect();
    let mut down_keys: Vec<_> = pv.down_overrides().keys().copied().collect();
    up_keys.sort_unstable();
    down_keys.sort_unstable();
    out.push(format!("up keys {up_keys:?} down keys {down_keys:?}"));
    for (layer, e) in pv.knn_store.ordered_entries() {
        out.push(format!(
            "knn L{layer} {} {} {} {} {} {:?}",
            e.entity,
            e.relation,
            e.target_token,
            e.target_id,
            e.confidence.to_bits(),
            bits(&e.key)
        ));
    }
    out.join("\n")
}

#[test]
fn removing_the_only_patch_restores_the_bound_state_byte_for_byte() {
    let mut pv = bound();
    let before = state(&pv);
    pv.try_apply_patch(every_kind("p", 1.0)).unwrap();
    assert_ne!(state(&pv), before, "the patch must change the state");
    pv.remove_patch(0);
    assert_eq!(state(&pv), before);
}

/// EDIT-9: a removed patch's up and down vectors do not stay live.
#[test]
fn a_removed_inserts_up_and_down_vectors_do_not_survive() {
    let mut pv = bound();
    pv.try_apply_patch(patch("p", vec![insert(1, 2, 1.0)]))
        .unwrap();
    assert!(pv.up_override_at(1, 2).is_some() && pv.down_override_at(1, 2).is_some());
    pv.remove_patch(0);
    assert!(pv.up_override_at(1, 2).is_none(), "up override survived");
    assert!(
        pv.down_override_at(1, 2).is_none(),
        "down override survived"
    );
}

/// A removed update gives the base slot back its own override.
#[test]
fn a_removed_update_restores_the_bases_own_override() {
    let mut pv = bound();
    pv.try_apply_patch(patch("p", vec![update(BASE_SLOT.0, BASE_SLOT.1, 1.0)]))
        .unwrap();
    pv.remove_patch(0);
    assert_eq!(
        pv.up_override_at(BASE_SLOT.0, BASE_SLOT.1),
        Some(&vector(BASE_UP)[..])
    );
    assert_eq!(
        pv.down_override_at(BASE_SLOT.0, BASE_SLOT.1),
        Some(&vector(BASE_DOWN)[..])
    );
}

/// The vindex's own KNN entries survive the removal of an unrelated patch.
#[test]
fn removing_an_unrelated_patch_keeps_the_vindexs_own_knn_entries() {
    let mut pv = bound();
    pv.try_apply_patch(patch(
        "q",
        vec![PatchOp::Delete {
            layer: 0,
            feature: 0,
            reason: None,
        }],
    ))
    .unwrap();
    pv.remove_patch(0);
    assert_eq!(pv.knn_store.len(), 1);
    assert_eq!(pv.knn_store.ordered_entries()[0].1.target_token, "Paris");
}

/// Removing a patch from the middle of a stack equals applying the rest to
/// a fresh overlay, in order, byte for byte.
#[test]
fn removing_a_middle_patch_equals_a_fresh_stack_without_it() {
    let (a, b, c) = (
        every_kind("a", 1.0),
        every_kind("b", 2.0),
        every_kind("c", 3.0),
    );
    let mut live = bound();
    for p in [a.clone(), b, c.clone()] {
        live.try_apply_patch(p).unwrap();
    }
    live.remove_patch(1);

    let mut fresh = bound();
    fresh.try_apply_patch(a).unwrap();
    fresh.try_apply_patch(c).unwrap();
    assert_eq!(state(&live), state(&fresh));
}
