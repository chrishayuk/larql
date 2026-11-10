//! The `.vlp` form of an edit: sealed, bound to its source, refused when altered.

use larql_vindex::format::vindex3::weight_patch::{
    apply_weight_patch, bind_source, record_op, WeightDeltaOp, WeightEdit, WeightPatch,
    WeightPatchError,
};
use larql_vindex::{PatchOp, VindexPatch};

use crate::fixture::*;

/// A change applied to an op, to show the digest covers the field it touches.
type Change = Box<dyn Fn(&mut WeightDeltaOp)>;

fn edits() -> Vec<WeightEdit> {
    [(5, 33, 0.0625), (5, 34, -0.125), (9, 7, 0.5)]
        .iter()
        .map(|&(row, col, delta)| WeightEdit { row, col, delta })
        .collect()
}

fn op(src: &Source) -> WeightDeltaOp {
    record_op(&src.container, TENSOR, &edits()).unwrap()
}

#[test]
fn a_recorded_op_is_intact_and_bound_to_the_exact_source() {
    let src = source_with_pack();
    let op = op(&src);
    op.check_intact().unwrap();
    op.check_binding(&src.container).unwrap();
    let index = index_of(&src.container);
    assert_eq!(op.source.model, index.model);
    assert_eq!(op.source.representation, CANONICAL);
    assert_eq!(
        op.source.payload_sha256,
        index.representations[CANONICAL].payload_sha256
    );
    assert_eq!(op.source, bind_source(&src.container, TENSOR).unwrap());
    let back = op.weight_edits();
    assert_eq!(back, edits(), "the exact f32 bits round-trip");
}

#[test]
fn the_digest_covers_every_field_and_ignores_the_order_of_edits() {
    let src = source_with_pack();
    let base = op(&src);
    let changed: Vec<(&str, Change)> = vec![
        ("tensor", Box::new(|o| o.tensor = "other".into())),
        ("rounding", Box::new(|o| o.rounding = "truncate".into())),
        ("model", Box::new(|o| o.source.model = "other".into())),
        (
            "index digest",
            Box::new(|o| o.source.index_sha256 = "0".repeat(64)),
        ),
        (
            "representation",
            Box::new(|o| o.source.representation = "other@BF16".into()),
        ),
        (
            "weights digest",
            Box::new(|o| o.source.payload_sha256 = "1".repeat(64)),
        ),
        ("a row", Box::new(|o| o.edits[0].row += 1)),
        ("a column", Box::new(|o| o.edits[0].col += 1)),
        ("a delta's bits", Box::new(|o| o.edits[0].delta_bits ^= 1)),
        (
            "an element dropped",
            Box::new(|o| {
                o.edits.pop();
            }),
        ),
    ];
    for (what, change) in changed {
        let mut o = base.clone();
        change(&mut o);
        assert_ne!(o.compute_digest(), base.digest, "the digest ignores {what}");
    }
    let mut reordered = base.clone();
    reordered.edits.reverse();
    assert_eq!(
        reordered.compute_digest(),
        base.digest,
        "order is not part of the meaning"
    );
}

#[test]
fn an_altered_op_is_refused_for_the_right_reason() {
    let src = source_with_pack();
    let base = op(&src);
    let reason = |o: &WeightDeltaOp| match o.check_intact() {
        Err(WeightPatchError::PatchOp(why)) => why,
        other => panic!("expected a patch-op refusal, got {other:?}"),
    };
    let mut o = base.clone();
    o.edits[0].delta_bits ^= 1;
    assert!(reason(&o).contains("digest does not match"));
    let mut o = base.clone();
    o.rounding = "truncate".into();
    assert!(reason(&o).contains("rounding"));
    // A delta that disagrees with its bits, with a digest that still matches (it covers the bits).
    let mut o = base.clone();
    o.edits[0].delta = 9.0;
    assert!(reason(&o).contains("disagrees"));
}

#[test]
fn an_op_is_bound_to_one_container_and_refused_by_another() {
    let src = source_with_pack();
    let op = op(&src);
    let other = src.out("other.vindex3");
    apply_weight_patch(
        &src.container,
        &WeightPatch {
            tensor: TENSOR.to_string(),
            edits: vec![WeightEdit {
                row: 20,
                col: 20,
                delta: 0.25,
            }],
        },
        &other,
    )
    .unwrap();
    assert!(matches!(
        op.check_binding(&other),
        Err(WeightPatchError::WrongSource { .. })
    ));
    assert!(op.check_binding(&src.container).is_ok());
}

#[test]
fn recording_an_op_for_a_tensor_the_container_does_not_hold_is_refused() {
    let src = source_with_pack();
    assert!(matches!(
        record_op(&src.container, "99.mlp.down_proj.weight", &edits()),
        Err(WeightPatchError::UnknownTensor(_))
    ));
}

#[test]
fn an_op_travels_inside_a_vlp_file() {
    let src = source_with_pack();
    let op = op(&src);
    let patch = VindexPatch {
        version: 2,
        base_model: index_of(&src.container).model,
        base_checksum: None,
        created_at: String::new(),
        description: Some("wsp".into()),
        author: None,
        tags: vec![],
        operations: vec![PatchOp::WeightDelta(op.clone())],
    };
    let path = src.out("edit.vlp");
    patch.save(&path).unwrap();
    let text = std::fs::read_to_string(&path).unwrap();
    assert!(text.contains("\"op\": \"weight_delta\""));
    let loaded = VindexPatch::load(&path).unwrap();
    assert_eq!(loaded.operations.len(), 1);
    let PatchOp::WeightDelta(back) = &loaded.operations[0] else {
        panic!("not a weight_delta")
    };
    assert_eq!(back, &op);
    assert_eq!(loaded.operations[0].key(), None, "it is not a feature slot");
    assert_eq!(loaded.counts(), (0, 1, 0), "counted with the updates");
}
