//! A tensor whose bare name is held by more than one canonical representation is named `<representation>::<tensor>`.
//! The fixture names both the token embedding and the final norm `weight`, as a real Gemma container does.

use larql_vindex::format::vindex3::encode::segment::read_segment_header;
use larql_vindex::format::vindex3::weight_patch::{
    apply_weight_patch, bind_source, declare_patches, parse_ledger, qualified, read_bf16_tensor,
    record_op, WeightEdit, WeightOp, WeightPatch, WeightPatchError,
};

use crate::fixture::*;

const AMBIGUOUS: &str = "weight";
const EMBEDDING: &str = "target.embedding@BF16";
/// The BF16-only container the fixture builder leaves beside the packed one.
const BF16_ONLY: &str = "canonical.vindex3";

fn embedding_name() -> String {
    qualified(EMBEDDING, AMBIGUOUS)
}

fn embedding_shape(container: &std::path::Path) -> Vec<usize> {
    let entry = index_of(container).representations[EMBEDDING].clone();
    let (header, _) = read_segment_header(&container.join(&entry.segment)).unwrap();
    header
        .tensors
        .iter()
        .find(|t| t.name == AMBIGUOUS)
        .unwrap()
        .shape
        .clone()
}

#[test]
fn a_bare_ambiguous_name_is_refused_and_the_refusal_says_how_to_qualify_it() {
    let src = source_with_pack();
    let err = bind_source(&src.container, AMBIGUOUS).unwrap_err();
    assert!(
        matches!(&err, WeightPatchError::AmbiguousTensor { holders, .. } if holders.contains(&EMBEDDING.to_string())),
        "{err}"
    );
    let text = err.to_string();
    assert!(text.contains("<representation>::weight"), "{text}");
    assert!(text.contains("::weight`"), "{text}");
}

#[test]
fn a_qualified_name_binds_the_representation_it_names() {
    let src = source_with_pack();
    let binding = bind_source(&src.container, &embedding_name()).unwrap();
    assert_eq!(binding.representation, EMBEDDING);
}

#[test]
fn a_qualified_bare_name_binds_exactly_as_the_bare_name() {
    let src = source_with_pack();
    let bare = bind_source(&src.container, TENSOR).unwrap();
    let named = bind_source(&src.container, &qualified(CANONICAL, TENSOR)).unwrap();
    assert_eq!(bare, named);
}

#[test]
fn a_name_qualified_by_a_representation_that_does_not_hold_it_is_unknown() {
    let src = source_with_pack();
    let err = bind_source(&src.container, &qualified(CANONICAL, AMBIGUOUS)).unwrap_err();
    assert!(matches!(err, WeightPatchError::UnknownTensor(_)), "{err}");
    let err = bind_source(&src.container, &qualified(DERIVED, TENSOR)).unwrap_err();
    assert!(
        matches!(err, WeightPatchError::UnknownTensor(_)),
        "a derived representation is not canonical: {err}"
    );
}

#[test]
fn a_recorded_op_holds_the_bare_name_and_its_binding_names_the_representation() {
    let src = source_with_pack();
    let op = record_op(
        &src.container,
        &embedding_name(),
        &[WeightEdit {
            row: 0,
            col: 0,
            delta: 0.5,
        }],
    )
    .unwrap();
    assert_eq!(op.tensor, AMBIGUOUS);
    assert_eq!(op.source.representation, EMBEDDING);
    op.check_intact().unwrap();
    // The binding check finds the tensor again through the representation, not by the ambiguous bare name.
    op.check_binding(&src.container).unwrap();
}

#[test]
fn reading_a_qualified_tensor_reads_that_representation() {
    // `read_bf16_tensor` refuses what a patch would; in the packed fixture a derived pack depends on the embedding,
    // so the BF16-only container is the one a patch may read.
    let src = source_with_pack();
    let container = src.dir.path().join(BF16_ONLY);
    let ([rows, cols], bits) = read_bf16_tensor(&container, &embedding_name()).unwrap();
    assert_eq!(vec![rows, cols], embedding_shape(&container));
    assert_eq!(bits.len(), rows * cols);
}

#[test]
fn an_op_on_an_ambiguous_tensor_declares_a_patch_that_locates_it() {
    let src = source_with_pack();
    let op = WeightOp::Delta(
        record_op(
            &src.container,
            &embedding_name(),
            &[WeightEdit {
                row: 1,
                col: 2,
                delta: 0.25,
            }],
        )
        .unwrap(),
    );
    let (patches, origins) = declare_patches(&src.container, &[op.as_ref()]).unwrap();
    assert_eq!(patches.len(), 1);
    assert_eq!(patches[0].tensor(), embedding_name());
    assert_eq!(origins.len(), 1);
}

#[test]
fn a_compile_of_a_qualified_embedding_edit_records_the_bare_name_in_its_ledger() {
    let src = source_with_pack();
    let container = src.dir.path().join(BF16_ONLY);
    let out = src.out("embedding.vindex3");
    let patch = WeightPatch {
        tensor: embedding_name(),
        edits: vec![WeightEdit {
            row: 3,
            col: 4,
            delta: 0.5,
        }],
    };
    apply_weight_patch(&container, &patch, &out).unwrap();
    let ledger =
        parse_ledger(&std::fs::read_to_string(out.join("weight_patches.json")).unwrap()).unwrap();
    assert_eq!(ledger.targets.len(), 1);
    assert_eq!(ledger.targets[0].tensor, AMBIGUOUS);
    assert_eq!(ledger.targets[0].representation, EMBEDDING);
}
