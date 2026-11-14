//! A weight-patched container through the rest of the lifecycle: compaction and the knowledge compile carry its
//! ledger or refuse, and the knowledge a container itself carries is not a session edit.

use larql_vindex::format::vindex3::compact::compact_container;
use larql_vindex::format::vindex3::compile::bake_container;
use larql_vindex::format::vindex3::inspect::inspect_container;
use larql_vindex::format::vindex3::knowledge::KnowledgeOverlay;
use larql_vindex::format::vindex3::opplan::exec::operands::{OperandEdit, OperandOverrides};
use larql_vindex::format::vindex3::opplan::OperandRef;
use larql_vindex::format::vindex3::weight_patch::{
    apply_weight_patch, WeightEdit, WeightPatch, WeightPatchError,
};
use larql_vindex::patch::core::encode_gate_vector;
use larql_vindex::{KnnStore, PatchOp, VindexPatch};

use crate::fixture::*;

const LEDGER: &str = "weight_patches.json";
const EDITED_ROW: usize = 5;
const EDITED_COL: usize = 33;
const DELTA: f32 = 0.0625;
/// Where the test KNN entries sit, and how wide their keys are. Nothing here queries them.
const KNN_LAYER: usize = 1;
const KNN_KEY_WIDTH: usize = 4;

/// The fixture with one element edited: a container that names a ledger.
fn patched(src: &Source) -> std::path::PathBuf {
    let out = src.out("patched.vindex3");
    apply_weight_patch(
        &src.container,
        &WeightPatch {
            tensor: TENSOR.into(),
            edits: vec![WeightEdit {
                row: EDITED_ROW,
                col: EDITED_COL,
                delta: DELTA,
            }],
        },
        &out,
    )
    .unwrap();
    out
}

fn coherent(container: &std::path::Path) -> bool {
    inspect_container(container, true).unwrap().is_coherent()
}

fn knn_with(entity: &str) -> KnnStore {
    let mut store = KnnStore::default();
    store.add(
        KNN_LAYER,
        vec![1.0; KNN_KEY_WIDTH],
        7,
        "t".into(),
        entity.into(),
        "r".into(),
        1.0,
    );
    store
}

fn knn_patch(entity: &str) -> VindexPatch {
    VindexPatch {
        version: 1,
        base_model: "wsp".into(),
        base_checksum: None,
        created_at: String::new(),
        description: Some("knn.vlp".into()),
        author: None,
        tags: vec![],
        operations: vec![PatchOp::InsertKnn {
            layer: KNN_LAYER,
            entity: entity.into(),
            relation: "r".into(),
            target: "t".into(),
            target_id: 7,
            confidence: Some(1.0),
            key_vector_b64: encode_gate_vector(&[1.0; KNN_KEY_WIDTH]),
        }],
    }
}

#[test]
fn compacting_a_weight_patched_container_carries_its_ledger() {
    let src = source_with_pack();
    let container = patched(&src);
    let out = src.out("compacted.vindex3");
    let report = compact_container(&container, &out).unwrap();
    assert!(
        !report.dropped.iter().any(|d| d == LEDGER),
        "the ledger is referenced, not junk: {:?}",
        report.dropped
    );
    assert_eq!(
        std::fs::read(container.join(LEDGER)).unwrap(),
        std::fs::read(out.join(LEDGER)).unwrap()
    );
    assert!(coherent(&out), "the index names a ledger the output holds");
}

#[test]
fn compacting_a_container_whose_named_ledger_is_missing_is_refused() {
    let src = source_with_pack();
    let container = patched(&src);
    std::fs::remove_file(container.join(LEDGER)).unwrap();
    let out = src.out("compacted.vindex3");
    let err = compact_container(&container, &out).unwrap_err();
    assert!(err.to_string().contains("weight-patch ledger"), "{err}");
    assert!(!out.exists());
}

#[test]
fn a_bake_that_leaves_the_patched_weights_alone_carries_the_ledger() {
    let src = source_with_pack();
    let container = patched(&src);
    let out = src.out("baked.vindex3");
    bake_container(&container, &OperandOverrides::new(), &out).unwrap();
    assert_eq!(
        std::fs::read(container.join(LEDGER)).unwrap(),
        std::fs::read(out.join(LEDGER)).unwrap()
    );
    assert!(coherent(&out));
}

#[test]
fn a_bake_that_would_rewrite_what_the_ledger_describes_is_refused() {
    let src = source_with_pack();
    let container = patched(&src);
    let index = index_of(&container);
    let canonical = &index.representations[CANONICAL];
    let mut overrides = OperandOverrides::new();
    overrides.push(
        &OperandRef {
            object: canonical.object.clone(),
            tensor: TENSOR.into(),
            dtype: "BF16".into(),
            shape: vec![TENSOR_ROWS, TENSOR_COLS],
        },
        OperandEdit::Row {
            index: 0,
            values: vec![1.0; TENSOR_COLS],
        },
    );
    let out = src.out("baked.vindex3");
    let err = bake_container(&container, &overrides, &out).unwrap_err();
    assert!(
        err.to_string().contains("weight-patched container") && err.to_string().contains(CANONICAL),
        "{err}"
    );
    assert!(!out.exists(), "a refused bake leaves nothing behind");
}

#[test]
fn knowledge_the_container_carries_is_not_a_session_edit() {
    let mut overlay = KnowledgeOverlay::new();
    overlay.bind_base_knn(knn_with("carried"));
    assert!(!overlay.knn_store.is_empty());
    assert!(!overlay.knn_edited());
    assert!(!overlay.has_knowledge_state());
    assert!(!overlay.has_pending_edits());

    overlay.try_apply_patch(knn_patch("added")).unwrap();
    assert!(overlay.knn_edited());
    assert!(overlay.has_knowledge_state());
    assert!(overlay.has_pending_edits());
}

#[test]
fn removing_a_patch_replays_over_the_containers_own_knowledge() {
    let mut overlay = KnowledgeOverlay::new();
    overlay.bind_base_knn(knn_with("carried"));
    overlay.try_apply_patch(knn_patch("added")).unwrap();
    assert_eq!(overlay.knn_store.len(), 2);

    overlay.remove_patch(0);
    assert_eq!(overlay.knn_store.len(), 1, "the patch's entry is gone");
    assert_eq!(
        overlay.knn_store.entries_for_entity("carried").len(),
        1,
        "the container's own entry is not"
    );
    assert!(!overlay.knn_edited());
}

#[test]
fn a_pending_weight_op_is_a_pending_edit() {
    let src = source_with_pack();
    let mut overlay = KnowledgeOverlay::new();
    assert!(!overlay.has_pending_edits());
    let op = larql_vindex::format::vindex3::weight_patch::record_op(
        &src.container,
        TENSOR,
        &[WeightEdit {
            row: EDITED_ROW,
            col: EDITED_COL,
            delta: DELTA,
        }],
    )
    .unwrap();
    overlay.stage_weight_op(op).unwrap();
    assert!(overlay.has_pending_edits());
    assert!(!overlay.has_knowledge_state());
}

#[test]
fn a_patched_container_still_refuses_a_second_patch() {
    // The reason a lost ledger matters downstream: the pointer in the index is what marks a container as
    // already patched.
    let src = source_with_pack();
    let container = patched(&src);
    let compacted = src.out("compacted.vindex3");
    compact_container(&container, &compacted).unwrap();
    let err = apply_weight_patch(
        &compacted,
        &WeightPatch {
            tensor: TENSOR.into(),
            edits: vec![WeightEdit {
                row: 0,
                col: 0,
                delta: DELTA,
            }],
        },
        &src.out("twice.vindex3"),
    )
    .unwrap_err();
    assert!(matches!(err, WeightPatchError::AlreadyPatched), "{err}");
}

#[test]
fn a_bake_of_a_container_whose_ledger_cannot_be_read_is_refused() {
    let src = source_with_pack();
    let container = patched(&src);
    let out = src.out("baked.vindex3");

    std::fs::write(container.join(LEDGER), "not a ledger").unwrap();
    let err = bake_container(&container, &OperandOverrides::new(), &out).unwrap_err();
    assert!(err.to_string().contains("weight-patch ledger"), "{err}");
    assert!(!out.exists());

    std::fs::remove_file(container.join(LEDGER)).unwrap();
    let err = bake_container(&container, &OperandOverrides::new(), &out).unwrap_err();
    assert!(err.to_string().contains("weight-patch ledger"), "{err}");
    assert!(!out.exists());
}
