//! A patch is checked whole against the index before any of it is applied.
//!
//! Every refusal here is followed by the same question: is the overlay what
//! it was? The overlay under test already holds a patch that set a gate, an
//! up and a down vector, a meta override and a KNN entry, so a refused patch
//! that had half-landed anywhere would show.

use larql_vindex::patch::format::{encode_gate_vector, PatchDownMeta};
use larql_vindex::{PatchGeometry, PatchOp, PatchedVindex, VectorIndex, VindexPatch};

const HIDDEN: usize = 4;
const LAYERS: usize = 3;
const FEATURES: usize = 6;
const MODEL: &str = "validate/model";
/// Not valid base64, so the vector cannot be decoded at all.
const CORRUPT: &str = "!!!not-base64!!!";

fn vector(width: usize) -> String {
    encode_gate_vector(&vec![0.5; width])
}

fn patch(operations: Vec<PatchOp>) -> VindexPatch {
    VindexPatch {
        version: 1,
        base_model: MODEL.into(),
        base_checksum: None,
        created_at: String::new(),
        description: None,
        author: None,
        tags: vec![],
        operations,
    }
}

fn insert(layer: usize, feature: usize, vectors: [Option<String>; 3]) -> PatchOp {
    let [gate_vector_b64, up_vector_b64, down_vector_b64] = vectors;
    PatchOp::Insert {
        layer,
        feature,
        relation: None,
        entity: "e".into(),
        target: "t".into(),
        confidence: Some(0.9),
        gate_vector_b64,
        up_vector_b64,
        down_vector_b64,
        down_meta: None,
    }
}

fn update(layer: usize, feature: usize, vectors: [Option<String>; 3]) -> PatchOp {
    let [gate_vector_b64, up_vector_b64, down_vector_b64] = vectors;
    PatchOp::Update {
        layer,
        feature,
        gate_vector_b64,
        up_vector_b64,
        down_vector_b64,
        down_meta: Some(PatchDownMeta {
            top_token: "u".into(),
            top_token_id: 1,
            c_score: 0.5,
        }),
    }
}

fn knn(layer: usize, key: String) -> PatchOp {
    PatchOp::InsertKnn {
        layer,
        entity: "k".into(),
        relation: "r".into(),
        target: "t".into(),
        target_id: 1,
        confidence: Some(1.0),
        key_vector_b64: key,
    }
}

fn delete(layer: usize, feature: usize) -> PatchOp {
    PatchOp::Delete {
        layer,
        feature,
        reason: None,
    }
}

/// An overlay that already holds one valid patch touching every kind of
/// state a patch can write.
fn stacked() -> PatchedVindex {
    let gates = vec![Some(ndarray::Array2::<f32>::zeros((FEATURES, HIDDEN))); LAYERS];
    let mut patched =
        PatchedVindex::new(VectorIndex::new(gates, vec![None; LAYERS], LAYERS, HIDDEN));
    let full = || Some(vector(HIDDEN));
    patched
        .try_apply_patch(patch(vec![
            insert(0, 1, [full(), full(), full()]),
            knn(1, vector(HIDDEN)),
            delete(2, 3),
        ]))
        .expect("the base patch applies");
    patched
}

/// Everything a patch can change, read back.
#[derive(Debug, PartialEq)]
struct Observed {
    patches: usize,
    knn_entries: usize,
    metas: Vec<Option<String>>,
    gates: Vec<Option<Vec<f32>>>,
    ups: Vec<Option<Vec<f32>>>,
    downs: Vec<Option<Vec<f32>>>,
}

fn observe(patched: &PatchedVindex) -> Observed {
    let slots = || (0..LAYERS).flat_map(|l| (0..FEATURES).map(move |f| (l, f)));
    Observed {
        patches: patched.num_patches(),
        knn_entries: patched.knn_store.len(),
        metas: slots()
            .map(|(l, f)| patched.feature_meta(l, f).map(|m| m.top_token))
            .collect(),
        gates: slots()
            .map(|(l, f)| patched.overrides_gate_at(l, f).map(<[f32]>::to_vec))
            .collect(),
        ups: slots()
            .map(|(l, f)| patched.base.up_override_at(l, f).map(<[f32]>::to_vec))
            .collect(),
        downs: slots()
            .map(|(l, f)| patched.base.down_override_at(l, f).map(<[f32]>::to_vec))
            .collect(),
    }
}

/// Apply `operations` after a valid op, require a refusal naming `expect`,
/// and require the overlay to be exactly what it was.
fn assert_refused(operations: Vec<PatchOp>, expect: &str) {
    let mut patched = stacked();
    let before = observe(&patched);
    // A valid op first: if validation ran op by op, this one would land.
    let mut all = vec![insert(1, 2, [Some(vector(HIDDEN)), None, None])];
    all.extend(operations);

    let err = patched
        .try_apply_patch(patch(all))
        .expect_err("an invalid patch is refused")
        .to_string();
    assert!(err.contains(expect), "expected `{expect}` in: {err}");
    assert_eq!(observe(&patched), before, "the overlay changed");
}

#[test]
fn a_valid_patch_of_every_operation_kind_applies() {
    // The control: the refusals below are not a base that refuses everything.
    let mut patched = stacked();
    let before = observe(&patched);
    let full = || Some(vector(HIDDEN));
    patched
        .try_apply_patch(patch(vec![
            insert(1, 2, [full(), full(), full()]),
            update(0, 1, [full(), full(), full()]),
            knn(2, vector(HIDDEN)),
            delete(0, 4),
            PatchOp::DeleteKnn { entity: "k".into() },
        ]))
        .expect("a valid patch applies");
    assert_ne!(observe(&patched), before);
    assert_eq!(patched.num_patches(), 2);
}

#[test]
fn each_vector_field_is_refused_when_it_does_not_decode() {
    let corrupt = || Some(CORRUPT.to_string());
    for (field, vectors) in [
        ("gate_vector_b64", [corrupt(), None, None]),
        ("up_vector_b64", [None, corrupt(), None]),
        ("down_vector_b64", [None, None, corrupt()]),
    ] {
        assert_refused(vec![insert(0, 2, vectors.clone())], field);
        assert_refused(vec![update(0, 2, vectors)], field);
    }
    assert_refused(vec![knn(0, CORRUPT.into())], "key_vector_b64");
}

#[test]
fn each_vector_field_is_refused_when_it_is_not_hidden_size_wide() {
    for width in [HIDDEN - 1, HIDDEN + 1] {
        let wrong = || Some(vector(width));
        for (field, vectors) in [
            ("gate_vector_b64", [wrong(), None, None]),
            ("up_vector_b64", [None, wrong(), None]),
            ("down_vector_b64", [None, None, wrong()]),
        ] {
            let expect = format!("{field} has {width} elements");
            assert_refused(vec![insert(0, 2, vectors.clone())], &expect);
            assert_refused(vec![update(0, 2, vectors)], &expect);
        }
        assert_refused(
            vec![knn(0, vector(width))],
            &format!("key_vector_b64 has {width} elements"),
        );
    }
}

#[test]
fn a_layer_the_index_does_not_have_is_refused_for_every_operation_that_names_one() {
    let none = || [None, None, None];
    for op in [
        insert(LAYERS, 0, none()),
        update(LAYERS, 0, none()),
        delete(LAYERS, 0),
        knn(LAYERS, vector(HIDDEN)),
    ] {
        assert_refused(vec![op], &format!("layer {LAYERS} is out of bounds"));
    }
}

#[test]
fn a_feature_slot_the_layer_does_not_have_is_refused() {
    let none = || [None, None, None];
    for op in [
        insert(0, FEATURES, none()),
        update(0, FEATURES, none()),
        delete(0, FEATURES),
    ] {
        assert_refused(vec![op], &format!("feature {FEATURES} is out of bounds"));
    }
}

#[test]
fn the_last_slot_and_the_last_layer_are_in_bounds() {
    // The boundary the two tests above sit one past.
    let mut patched = stacked();
    patched
        .try_apply_patch(patch(vec![
            delete(LAYERS - 1, FEATURES - 1),
            knn(LAYERS - 1, vector(HIDDEN)),
        ]))
        .expect("the last layer and slot exist");
}

#[test]
fn a_patch_naming_this_model_passes_the_identity_check() {
    patch(vec![]).check_base_model(MODEL).expect("same model");
}

#[test]
fn a_patch_naming_another_model_is_refused_by_name() {
    let err = patch(vec![])
        .check_base_model("another/model")
        .expect_err("another model")
        .to_string();
    assert!(
        err.contains(MODEL) && err.contains("another/model"),
        "{err}"
    );
}

#[test]
fn a_patch_naming_no_model_is_refused() {
    let mut unnamed = patch(vec![]);
    unnamed.base_model.clear();
    let err = unnamed
        .check_base_model(MODEL)
        .expect_err("no model")
        .to_string();
    assert!(err.contains("declares no base_model"), "{err}");
}

#[test]
fn an_index_that_declares_no_width_refuses_a_vector_and_admits_a_patch_without_one() {
    let geometry = PatchGeometry {
        hidden_size: None,
        num_layers: LAYERS,
        features_at: &|_| FEATURES,
    };
    let err = patch(vec![knn(0, vector(HIDDEN))])
        .check_geometry(&geometry)
        .expect_err("the width cannot be checked")
        .to_string();
    assert!(err.contains("declares no hidden size"), "{err}");

    patch(vec![
        delete(0, 0),
        PatchOp::DeleteKnn { entity: "k".into() },
    ])
    .check_geometry(&geometry)
    .expect("no vector, nothing to check a width against");
}

#[test]
fn a_layer_with_no_feature_space_refuses_a_slot_and_admits_a_knn_entry() {
    let geometry = PatchGeometry {
        hidden_size: Some(HIDDEN),
        num_layers: LAYERS,
        features_at: &|_| 0,
    };
    patch(vec![delete(0, 0)])
        .check_geometry(&geometry)
        .expect_err("no slot exists");
    patch(vec![knn(0, vector(HIDDEN))])
        .check_geometry(&geometry)
        .expect("a KNN entry names a layer, not a slot");
}
