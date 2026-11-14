//! Structured ops in the session overlay: held by declared footprint, never composed, one form per tensor.

use larql_vindex::format::vindex3::knowledge::KnowledgeOverlay;
use larql_vindex::format::vindex3::weight_patch::structured::{OperandBits, ScalarBits};
use larql_vindex::format::vindex3::weight_patch::{
    read_bf16_tensor, record_op, StructuredWeightOp, WeightEdit, WeightOpRef, WeightPatchError,
};
use larql_vindex::{PatchOp, VindexPatch};

use crate::fixture::*;

const PROFILE: &str = "f64-seq-rne1/v1";
const F64_ONE: u64 = 0x3ff0_0000_0000_0000;

fn structured(src: &Source, columns: &[usize]) -> StructuredWeightOp {
    let ([rows, _], _) = read_bf16_tensor(&src.container, TENSOR).unwrap();
    let mut u = vec![0u64; rows];
    u[0] = F64_ONE;
    StructuredWeightOp::bind(
        &src.container,
        TENSOR,
        PROFILE,
        &OperandBits::F64(u),
        ScalarBits::F64(F64_ONE),
        columns.to_vec(),
    )
    .unwrap()
}

fn delta(src: &Source, col: usize) -> larql_vindex::format::vindex3::weight_patch::WeightDeltaOp {
    record_op(
        &src.container,
        TENSOR,
        &[WeightEdit {
            row: 5,
            col,
            delta: 0.0625,
        }],
    )
    .unwrap()
}

fn patch(name: &str, ops: Vec<PatchOp>) -> VindexPatch {
    VindexPatch {
        version: 2,
        base_model: "wsp".into(),
        base_checksum: None,
        created_at: String::new(),
        description: Some(name.into()),
        author: None,
        tags: vec![],
        operations: ops,
    }
}

#[test]
fn a_structured_op_is_staged_counted_and_listed_beside_delta_ops() {
    let src = source_with_pack();
    let mut overlay = KnowledgeOverlay::new();
    assert!(!overlay.has_weight_ops());
    overlay
        .stage_structured_op(structured(&src, &[1, 2]))
        .unwrap();
    assert!(overlay.has_weight_ops());
    assert_eq!(overlay.structured_op_count(), 1);
    assert_eq!(
        overlay.weight_edit_count(),
        0,
        "a structured op names columns, not elements"
    );
    let held = overlay.weight_ops();
    assert!(matches!(held[0].1, WeightOpRef::Structured(_)));
    assert_eq!(held[0].0, "this session");
    assert!(
        !overlay.has_knowledge_state(),
        "a weight op is not knowledge state"
    );
}

#[test]
fn two_structured_ops_cannot_write_the_same_column() {
    let src = source_with_pack();
    let mut overlay = KnowledgeOverlay::new();
    overlay
        .stage_structured_op(structured(&src, &[1, 2]))
        .unwrap();
    let error = overlay
        .stage_structured_op(structured(&src, &[2, 3]))
        .unwrap_err();
    assert!(
        matches!(error, WeightPatchError::ColumnConflict { col: 2, .. }),
        "{error}"
    );
    assert_eq!(
        overlay.structured_op_count(),
        1,
        "the refused op was not staged"
    );
    overlay
        .stage_structured_op(structured(&src, &[3, 4]))
        .unwrap();
}

#[test]
fn the_declared_footprint_counts_even_when_the_op_changes_no_element() {
    // alpha 0 rounds every element back to itself, so it changes nothing, and still declares its columns.
    let src = source_with_pack();
    let ([rows, _], _) = read_bf16_tensor(&src.container, TENSOR).unwrap();
    let mut u = vec![0u64; rows];
    u[0] = F64_ONE;
    let harmless = StructuredWeightOp::bind(
        &src.container,
        TENSOR,
        PROFILE,
        &OperandBits::F64(u),
        ScalarBits::F64(0),
        vec![7],
    )
    .unwrap();
    assert!(
        harmless.resolve(&src.container).is_err(),
        "it edits nothing"
    );
    let mut overlay = KnowledgeOverlay::new();
    overlay.stage_structured_op(harmless).unwrap();
    let error = overlay
        .stage_structured_op(structured(&src, &[7]))
        .unwrap_err();
    assert!(
        matches!(error, WeightPatchError::ColumnConflict { col: 7, .. }),
        "{error}"
    );
}

#[test]
fn a_delta_op_and_a_structured_op_on_one_tensor_are_refused_either_way_round() {
    let src = source_with_pack();
    let mut a = KnowledgeOverlay::new();
    a.stage_weight_op(delta(&src, 40)).unwrap();
    let error = a.stage_structured_op(structured(&src, &[1])).unwrap_err();
    assert!(
        matches!(error, WeightPatchError::MixedForms { .. }),
        "{error}"
    );

    let mut b = KnowledgeOverlay::new();
    b.stage_structured_op(structured(&src, &[1])).unwrap();
    let error = b.stage_weight_op(delta(&src, 40)).unwrap_err();
    assert!(
        matches!(error, WeightPatchError::MixedForms { .. }),
        "{error}"
    );
}

#[test]
fn a_patch_is_checked_against_itself_and_against_the_session() {
    let src = source_with_pack();
    let within = patch(
        "within.vlp",
        vec![
            PatchOp::StructuredWeight(structured(&src, &[1, 2])),
            PatchOp::StructuredWeight(structured(&src, &[2])),
        ],
    );
    let mut overlay = KnowledgeOverlay::new();
    let error = overlay.try_apply_patch(within).unwrap_err();
    assert!(error.to_string().contains("column 2"), "{error}");
    assert!(
        !overlay.has_weight_ops(),
        "a refused patch leaves the overlay as it was"
    );

    overlay.stage_structured_op(structured(&src, &[5])).unwrap();
    let error = overlay
        .try_apply_patch(patch(
            "clash.vlp",
            vec![PatchOp::StructuredWeight(structured(&src, &[5, 6]))],
        ))
        .unwrap_err();
    assert!(error.to_string().contains("this session"), "{error}");
    overlay
        .try_apply_patch(patch(
            "ok.vlp",
            vec![PatchOp::StructuredWeight(structured(&src, &[6]))],
        ))
        .unwrap();
    assert_eq!(overlay.structured_op_count(), 2);
}

#[test]
fn an_altered_structured_op_is_refused_when_staged_or_applied() {
    let src = source_with_pack();
    let mut damaged = structured(&src, &[1]);
    damaged.columns = vec![2];
    let mut overlay = KnowledgeOverlay::new();
    assert!(overlay.stage_structured_op(damaged.clone()).is_err());
    let error = overlay
        .try_apply_patch(patch("bad.vlp", vec![PatchOp::StructuredWeight(damaged)]))
        .unwrap_err();
    assert!(error.to_string().contains("digest"), "{error}");
}

#[test]
fn removing_a_patch_removes_its_structured_ops_and_leaves_the_sessions() {
    let src = source_with_pack();
    let mut overlay = KnowledgeOverlay::new();
    overlay.stage_structured_op(structured(&src, &[1])).unwrap();
    overlay
        .try_apply_patch(patch(
            "a.vlp",
            vec![PatchOp::StructuredWeight(structured(&src, &[2]))],
        ))
        .unwrap();
    assert_eq!(overlay.structured_op_count(), 2);
    overlay.remove_patch(0);
    assert_eq!(overlay.structured_op_count(), 1);
    assert_eq!(overlay.weight_ops()[0].0, "this session");
}

#[test]
fn a_structured_patch_round_trips_through_its_file() {
    let src = source_with_pack();
    let op = structured(&src, &[1, 9]);
    let file = src.out("s.vlp");
    patch("s.vlp", vec![PatchOp::StructuredWeight(op.clone())])
        .save(&file)
        .unwrap();
    let text = std::fs::read_to_string(&file).unwrap();
    assert!(text.contains("\"op\": \"structured_weight\""), "{text}");
    let loaded = VindexPatch::load(&file).unwrap();
    match &loaded.operations[0] {
        PatchOp::StructuredWeight(back) => {
            assert_eq!(back, &op);
            back.check_intact().unwrap();
        }
        other => panic!("{other:?}"),
    }
    assert_eq!(loaded.counts(), (0, 1, 0), "counted with the updates");
}

mod compile {
    use super::*;
    use larql_vindex::format::vindex3::weight_patch::{
        apply_declared_patches, check_ledger, declare_patches, DeclaredPatch, EditEncoding,
        WEIGHT_PATCHES_SCHEMA_V3,
    };

    fn refs<'a>(ops: &'a [StructuredWeightOp]) -> Vec<WeightOpRef<'a>> {
        ops.iter().map(WeightOpRef::Structured).collect()
    }

    #[test]
    fn two_structured_ops_on_a_tensor_compile_into_one_result_patch_and_one_container() {
        let src = source_with_pack();
        let ops = vec![structured(&src, &[1, 2]), structured(&src, &[9])];
        let (declared, origins) = declare_patches(&src.container, &refs(&ops)).unwrap();
        assert_eq!(declared.len(), 1);
        assert_eq!(
            origins,
            ops.iter().map(|o| o.digest.clone()).collect::<Vec<_>>()
        );
        let DeclaredPatch::Results(joined) = &declared[0] else {
            panic!("a tensor of structured ops compiles as results");
        };
        for op in &ops {
            let own = op.resolve(&src.container).unwrap();
            assert!(own.edits.iter().all(|e| joined.edits.contains(e)));
        }
        let out = src.out("two.vindex3");
        let report = apply_declared_patches(&src.container, &declared, &origins, &out).unwrap();
        assert_eq!(report.ledger.schema, WEIGHT_PATCHES_SCHEMA_V3);
        assert!(report.ledger.targets[0]
            .edits
            .iter()
            .all(|e| e.encoding == EditEncoding::ResultBits));
        assert!(check_ledger(&out, &index_of(&out)).is_empty());
    }

    #[test]
    fn the_order_of_the_ops_does_not_change_the_bytes() {
        let src = source_with_pack();
        let forward = vec![structured(&src, &[1]), structured(&src, &[9])];
        let reverse = vec![structured(&src, &[9]), structured(&src, &[1])];
        let mut digests = Vec::new();
        for (name, ops) in [("f", forward), ("r", reverse)] {
            let (declared, origins) = declare_patches(&src.container, &refs(&ops)).unwrap();
            let out = src.out(name);
            apply_declared_patches(&src.container, &declared, &origins, &out).unwrap();
            digests.push(tensor_bytes(&out, CANONICAL, TENSOR));
        }
        assert_eq!(digests[0], digests[1]);
    }

    #[test]
    fn a_mixed_tensor_overlapping_columns_and_a_stale_binding_all_refuse_before_any_write() {
        let src = source_with_pack();
        let out = src.out("never.vindex3");

        let d = delta(&src, 40);
        let s = structured(&src, &[1]);
        let mixed = [WeightOpRef::Delta(&d), WeightOpRef::Structured(&s)];
        assert!(matches!(
            declare_patches(&src.container, &mixed),
            Err(WeightPatchError::MixedForms { .. })
        ));

        let ops = vec![structured(&src, &[1, 2]), structured(&src, &[2])];
        assert!(matches!(
            declare_patches(&src.container, &refs(&ops)),
            Err(WeightPatchError::ColumnConflict { .. })
        ));

        let mut stale = structured(&src, &[1]);
        stale.source.payload_sha256 = "0".repeat(64);
        stale.digest = stale.compute_digest();
        let ops = vec![stale];
        assert!(matches!(
            declare_patches(&src.container, &refs(&ops)),
            Err(WeightPatchError::WrongSource { .. })
        ));
        assert!(!out.exists());
    }

    #[test]
    fn a_zero_effect_op_is_refused_by_the_compile_and_nothing_is_written() {
        let src = source_with_pack();
        let ([rows, _], _) = read_bf16_tensor(&src.container, TENSOR).unwrap();
        let mut u = vec![0u64; rows];
        u[0] = F64_ONE;
        let nothing = StructuredWeightOp::bind(
            &src.container,
            TENSOR,
            PROFILE,
            &OperandBits::F64(u),
            ScalarBits::F64(0),
            vec![3],
        )
        .unwrap();
        let ops = vec![nothing];
        assert!(matches!(
            declare_patches(&src.container, &refs(&ops)),
            Err(WeightPatchError::Empty)
        ));
    }

    #[test]
    fn delta_only_ops_still_declare_delta_patches_and_the_old_schema() {
        let src = source_with_pack();
        let d = delta(&src, 40);
        let (declared, _) = declare_patches(&src.container, &[WeightOpRef::Delta(&d)]).unwrap();
        assert!(matches!(declared[0], DeclaredPatch::Deltas(_)));
        let out = src.out("d");
        let report = apply_declared_patches(&src.container, &declared, &[], &out).unwrap();
        assert_eq!(
            report.ledger.schema,
            larql_vindex::format::vindex3::weight_patch::WEIGHT_PATCHES_SCHEMA
        );
    }
}
