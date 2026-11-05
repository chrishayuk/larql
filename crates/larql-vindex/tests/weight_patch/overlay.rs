//! The session overlay's stored-weight edits: staged directly or carried by patches.

use larql_vindex::format::vindex3::knowledge::KnowledgeOverlay;
use larql_vindex::format::vindex3::weight_patch::{
    record_op, WeightDeltaOp, WeightEdit, WeightPatchError,
};
use larql_vindex::{PatchOp, VindexPatch};

use crate::fixture::*;

fn op(src: &Source, cols: &[usize]) -> WeightDeltaOp {
    let edits: Vec<WeightEdit> = cols
        .iter()
        .map(|&col| WeightEdit {
            row: 5,
            col,
            delta: 0.0625,
        })
        .collect();
    record_op(&src.container, TENSOR, &edits).unwrap()
}

fn patch(name: &str, ops: Vec<WeightDeltaOp>) -> VindexPatch {
    VindexPatch {
        version: 2,
        base_model: "wsp".into(),
        base_checksum: None,
        created_at: String::new(),
        description: Some(name.into()),
        author: None,
        tags: vec![],
        operations: ops.into_iter().map(PatchOp::WeightDelta).collect(),
    }
}

#[test]
fn staged_edits_are_counted_and_listed_with_where_they_came_from() {
    let src = source_with_pack();
    let mut overlay = KnowledgeOverlay::new();
    assert_eq!(overlay.weight_edit_count(), 0);
    overlay.stage_weight_op(op(&src, &[1, 2])).unwrap();
    overlay
        .try_apply_patch(patch("a.vlp", vec![op(&src, &[10])]))
        .unwrap();
    assert_eq!(overlay.weight_edit_count(), 3);
    let origins: Vec<String> = overlay.weight_ops().into_iter().map(|(by, _)| by).collect();
    assert_eq!(
        origins,
        vec!["a.vlp".to_string(), "this session".to_string()],
        "patches first, then the session"
    );
}

#[test]
fn an_element_cannot_be_edited_twice_wherever_the_edits_come_from() {
    let src = source_with_pack();
    let mut overlay = KnowledgeOverlay::new();
    overlay.stage_weight_op(op(&src, &[1, 2])).unwrap();
    let again = overlay.stage_weight_op(op(&src, &[2, 3])).unwrap_err();
    assert!(
        matches!(again, WeightPatchError::Conflict { row: 5, col: 2, ref by, .. } if by == "this session"),
        "{again}"
    );
    // A patch overlapping the session is refused whole.
    let err = overlay
        .try_apply_patch(patch("p.vlp", vec![op(&src, &[7]), op(&src, &[2])]))
        .unwrap_err();
    assert!(err.to_string().contains("already edited"), "{err}");
    assert_eq!(
        overlay.weight_edit_count(),
        2,
        "nothing from the refused patch was kept"
    );
    assert!(overlay.patches.is_empty());
}

#[test]
fn two_ops_inside_one_patch_cannot_overlap_either() {
    let src = source_with_pack();
    let mut overlay = KnowledgeOverlay::new();
    let err = overlay
        .try_apply_patch(patch("p.vlp", vec![op(&src, &[1, 2]), op(&src, &[2, 3])]))
        .unwrap_err();
    assert!(err.to_string().contains("already edited"), "{err}");
    assert_eq!(overlay.weight_edit_count(), 0);
}

#[test]
fn the_same_elements_of_a_different_tensor_are_a_different_edit() {
    let src = source_with_pack();
    let mut overlay = KnowledgeOverlay::new();
    overlay.stage_weight_op(op(&src, &[1])).unwrap();
    let other = record_op(
        &src.container,
        "0.mlp.down_proj.weight",
        &[WeightEdit {
            row: 5,
            col: 1,
            delta: 0.0625,
        }],
    )
    .unwrap();
    overlay.stage_weight_op(other).unwrap();
    assert_eq!(overlay.weight_edit_count(), 2);
}

#[test]
fn an_altered_op_is_refused_when_staged_or_applied() {
    let src = source_with_pack();
    let mut bad = op(&src, &[1]);
    bad.edits[0].delta_bits ^= 1;
    let mut overlay = KnowledgeOverlay::new();
    assert!(matches!(
        overlay.stage_weight_op(bad.clone()),
        Err(WeightPatchError::PatchOp(_))
    ));
    assert!(overlay
        .try_apply_patch(patch("bad.vlp", vec![bad]))
        .is_err());
    assert_eq!(overlay.weight_edit_count(), 0);
}

#[test]
fn removing_a_patch_removes_its_edits_and_leaves_the_sessions() {
    let src = source_with_pack();
    let mut overlay = KnowledgeOverlay::new();
    overlay.stage_weight_op(op(&src, &[1])).unwrap();
    overlay
        .try_apply_patch(patch("a.vlp", vec![op(&src, &[2])]))
        .unwrap();
    overlay
        .try_apply_patch(patch("b.vlp", vec![op(&src, &[3])]))
        .unwrap();
    assert_eq!(overlay.weight_edit_count(), 3);
    overlay.remove_patch(0);
    assert_eq!(overlay.weight_edit_count(), 2);
    let names: Vec<String> = overlay.weight_ops().into_iter().map(|(by, _)| by).collect();
    assert_eq!(names, vec!["b.vlp".to_string(), "this session".to_string()]);
    overlay.remove_patch(9); // out of range: nothing happens
    assert_eq!(overlay.weight_edit_count(), 2);
}

#[test]
fn weight_edits_are_not_knowledge_state() {
    let src = source_with_pack();
    let mut overlay = KnowledgeOverlay::new();
    overlay.stage_weight_op(op(&src, &[1])).unwrap();
    assert!(
        !overlay.has_knowledge_state(),
        "only stored-weight edits are pending"
    );
    overlay.delete_feature(3, 4);
    assert!(
        overlay.has_knowledge_state(),
        "a tombstone is knowledge state"
    );
}
