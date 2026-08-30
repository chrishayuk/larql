//! Gates for the logical diff — the COMPILE oracle in reverse:
//!
//! ```text
//! A = pristine     B = A + overlay     C = COMPILE(B)
//!
//! semantic_diff(A, B) == semantic_diff(A, C)   (meaning, not storage)
//! semantic_diff(B, C) == ∅                      (equivalent models)
//! physical_diff(A, C) != ∅                      (storage DID change)
//! ```

use crate::format::filenames::KNN_STORE_BIN;
use crate::format::vindex3::compile::bake_container;
use crate::format::vindex3::fixtures::{encode_fixture_container, miniature_glimmer, G_HIDDEN};
use crate::format::vindex3::opplan::exec::operands::{OperandEdit, OperandOverrides};
use crate::format::vindex3::opplan::LayerFfn;
use crate::patch::knn_store::KnnStore;

use super::{physical_diff, semantic_diff, DiffSide};

fn pristine() -> tempfile::TempDir {
    let checkpoint = tempfile::tempdir().unwrap();
    let container = tempfile::tempdir().unwrap();
    encode_fixture_container(
        miniature_glimmer,
        checkpoint.path(),
        container.path(),
        "diff-fixture",
    );
    container
}

/// A compose-shaped overlay: gate row + up row + down column at layer
/// 1 feature 3, plus one knowledge edge.
fn overlay_state(side: &DiffSide) -> (OperandOverrides, KnnStore) {
    let Some(LayerFfn::Dense(ffn)) = &side.plan.layers[1].ffn else {
        panic!("miniature layer 1 is dense");
    };
    let mut overrides = OperandOverrides::new();
    let gate = ffn.gate.as_ref().unwrap();
    overrides.push(
        gate,
        OperandEdit::Row {
            index: 3,
            values: vec![7.0; gate.shape[1]],
        },
    );
    overrides.push(
        &ffn.up,
        OperandEdit::Row {
            index: 3,
            values: vec![0.5; ffn.up.shape[1]],
        },
    );
    overrides.push(
        &ffn.down,
        OperandEdit::Column {
            index: 3,
            values: vec![0.25; ffn.down.shape[0]],
        },
    );
    let mut knn = KnnStore::default();
    knn.add(
        1,
        vec![1.0; G_HIDDEN],
        5,
        "[5]".into(),
        "atlantis".into(),
        "capital".into(),
        1.0,
    );
    (overrides, knn)
}

#[test]
fn a_container_has_no_diff_against_itself() {
    let a = pristine();
    let side_a = DiffSide::open(a.path(), "target").unwrap();
    let side_a2 = DiffSide::open(a.path(), "target").unwrap();
    let diff = semantic_diff(&side_a, &side_a2).unwrap();
    assert!(diff.is_empty(), "{diff:?}");
    let phys = physical_diff(side_a.index(), side_a2.index());
    assert!(phys.changed_segments.is_empty());
}

/// THE gate: the diff sees model meaning, not how the meaning is
/// stored — and physical storage is allowed to disagree.
#[test]
fn overlay_and_its_bake_diff_identically_and_diff_empty_against_each_other() {
    let a = pristine();
    let side_for_state = DiffSide::open(a.path(), "target").unwrap();
    let (overrides, knn) = overlay_state(&side_for_state);

    // B: the same container with the overlay layered on.
    let b = DiffSide::open(a.path(), "target")
        .unwrap()
        .with_overlay(overrides.clone(), knn.clone());

    // C: the clean bake of B.
    let out = tempfile::tempdir().unwrap();
    bake_container(a.path(), &overrides, out.path()).unwrap();
    knn.save(&out.path().join(KNN_STORE_BIN)).unwrap();
    let c = DiffSide::open(out.path(), "target").unwrap();

    let side_a = DiffSide::open(a.path(), "target").unwrap();
    let ab = semantic_diff(&side_a, &b).unwrap();
    let ac = semantic_diff(&side_a, &c).unwrap();

    // Identical logical reports, regardless of representation.
    assert_eq!(ab.features, ac.features);
    assert_eq!(ab.knowledge_added, ac.knowledge_added);
    assert_eq!(ab.knowledge_removed, ac.knowledge_removed);
    assert_eq!(ab.changed_tensors, ac.changed_tensors);
    assert_eq!(ab.metadata, ac.metadata);

    // The report is slot-granular and affirmative.
    assert_eq!(ab.features.len(), 1, "{:?}", ab.features);
    let slot = &ab.features[0];
    assert_eq!((slot.layer, slot.feature), (1, 3));
    assert!(slot.gate_changed && slot.up_changed && slot.down_changed);
    assert_eq!(
        ab.knowledge_added,
        vec![("atlantis".into(), "capital".into(), "[5]".into())]
    );
    assert!(ab.knowledge_removed.is_empty());
    assert!(ab.changed_tensors.is_empty(), "{:?}", ab.changed_tensors);

    // B vs C: semantically EMPTY — same model, different storage.
    let bc = semantic_diff(&b, &c).unwrap();
    assert!(bc.is_empty(), "{bc:?}");

    // …while the physical layer reports the rewrite.
    let phys = physical_diff(side_a.index(), c.index());
    assert_eq!(phys.changed_segments.len(), 1, "{phys:?}");
    assert!(phys.only_in_a.is_empty() && phys.only_in_b.is_empty());
}

/// A non-FFN edit lands in the representation layer (`changed_tensors`),
/// and a representation present on one side only surfaces physically.
#[test]
fn non_ffn_edits_and_missing_representations_are_reported() {
    let a = pristine();
    let side_a = DiffSide::open(a.path(), "target").unwrap();

    // Edit the embedding table — no FFN role, so the change is a
    // representation-level fact.
    let embed = side_a.plan.embedding.as_ref().unwrap().table.clone();
    let mut overrides = OperandOverrides::new();
    overrides.push(
        &embed,
        OperandEdit::Row {
            index: 1,
            values: vec![0.5; embed.shape[1]],
        },
    );
    let b = DiffSide::open(a.path(), "target")
        .unwrap()
        .with_overlay(overrides, KnnStore::default());
    let diff = semantic_diff(&side_a, &b).unwrap();
    assert!(diff.features.is_empty(), "{:?}", diff.features);
    assert_eq!(diff.changed_tensors.len(), 1, "{:?}", diff.changed_tensors);
    assert!(
        diff.changed_tensors[0].contains(&embed.tensor),
        "{:?}",
        diff.changed_tensors
    );

    // A representation present in only one index surfaces physically —
    // the physical layer reads indexes, so a broken/partial container
    // is still describable.
    let mut reduced = side_a.index().clone();
    let dropped = reduced.representations.keys().next().cloned().unwrap();
    reduced.representations.remove(&dropped);
    let phys = physical_diff(side_a.index(), &reduced);
    assert_eq!(phys.only_in_a, vec![dropped]);
    assert!(phys.only_in_b.is_empty());
}
