//! A patch that cannot be applied exactly as declared is refused by name,
//! and a refusal leaves nothing behind.

use larql_vindex::format::vindex3::fixtures::{dense_f32_model, encode_fixture_container};
use larql_vindex::format::vindex3::weight_patch::{
    apply_weight_patch, WeightEdit, WeightPatch, WeightPatchError, MAX_PATCH_ELEMENTS,
};

use crate::fixture::*;

fn edit(row: usize, col: usize, delta: f32) -> WeightEdit {
    WeightEdit { row, col, delta }
}

fn refusal(patch: WeightPatch) -> WeightPatchError {
    let src = source_with_pack();
    apply_weight_patch(&src.container, &patch, &src.out("out.vindex3"))
        .expect_err("must be refused")
}

fn on(tensor: &str, edits: Vec<WeightEdit>) -> WeightPatch {
    WeightPatch {
        tensor: tensor.to_string(),
        edits,
    }
}

#[test]
fn an_empty_patch_is_refused() {
    assert!(matches!(
        refusal(on(TENSOR, vec![])),
        WeightPatchError::Empty
    ));
}

#[test]
fn the_element_limit_is_enforced() {
    let edits: Vec<WeightEdit> = (0..MAX_PATCH_ELEMENTS + 1)
        .map(|i| edit(i / TENSOR_COLS, i % TENSOR_COLS, 0.5))
        .collect();
    assert!(matches!(
        refusal(on(TENSOR, edits)),
        WeightPatchError::TooManyElements { count, limit } if count == MAX_PATCH_ELEMENTS + 1 && limit == MAX_PATCH_ELEMENTS
    ));
}

#[test]
fn conflicting_edits_to_one_element_are_refused() {
    let err = refusal(on(TENSOR, vec![edit(3, 4, 0.5), edit(3, 4, 0.25)]));
    assert!(matches!(
        err,
        WeightPatchError::DuplicateElement { row: 3, col: 4 }
    ));
}

#[test]
fn a_non_finite_delta_is_refused() {
    for bad in [f32::NAN, f32::INFINITY, f32::NEG_INFINITY] {
        assert!(matches!(
            refusal(on(TENSOR, vec![edit(0, 0, bad)])),
            WeightPatchError::NonFiniteDelta { .. }
        ));
    }
}

#[test]
fn an_unknown_tensor_is_refused() {
    assert!(matches!(
        refusal(on("99.mlp.down_proj.weight", vec![edit(0, 0, 0.5)])),
        WeightPatchError::UnknownTensor(_)
    ));
}

#[test]
fn elements_outside_the_tensor_are_refused() {
    assert!(matches!(
        refusal(on(TENSOR, vec![edit(TENSOR_ROWS, 0, 0.5)])),
        WeightPatchError::OutOfBounds { .. }
    ));
    assert!(matches!(
        refusal(on(TENSOR, vec![edit(0, TENSOR_COLS, 0.5)])),
        WeightPatchError::OutOfBounds { .. }
    ));
}

#[test]
fn a_vector_is_not_a_matrix() {
    assert!(matches!(
        refusal(on("0.input_layernorm.weight", vec![edit(0, 0, 0.5)])),
        WeightPatchError::NotAMatrix { .. }
    ));
}

#[test]
fn a_result_that_overflows_bfloat16_is_refused_not_saturated() {
    // 3.4e38 is nearer 2^128 than the largest finite bfloat16.
    assert!(matches!(
        refusal(on(TENSOR, vec![edit(0, 0, 3.4e38)])),
        WeightPatchError::Overflow { row: 0, col: 0 }
    ));
}

#[test]
fn only_bf16_weights_can_be_patched() {
    let dir = tempfile::tempdir().unwrap();
    let container = dir.path().join("f32.vindex3");
    let checkpoint = dir.path().join("checkpoint");
    std::fs::create_dir(&checkpoint).unwrap();
    encode_fixture_container(dense_f32_model, &checkpoint, &container, "wsp-f32");
    let err = apply_weight_patch(
        &container,
        &on(TENSOR, vec![edit(0, 0, 0.5)]),
        &dir.path().join("out.vindex3"),
    )
    .expect_err("F32 is not patchable");
    assert!(
        matches!(err, WeightPatchError::UnsupportedDtype { ref dtype, .. } if dtype == "F32"),
        "{err}"
    );
}

#[test]
fn an_existing_output_is_never_overwritten() {
    let src = source_with_pack();
    let out = src.out("exists.vindex3");
    std::fs::create_dir(&out).unwrap();
    let err =
        apply_weight_patch(&src.container, &on(TENSOR, vec![edit(0, 0, 0.5)]), &out).unwrap_err();
    assert!(matches!(err, WeightPatchError::OutputExists(_)));
}

#[test]
fn a_patched_container_cannot_be_patched_again() {
    let src = source_with_pack();
    let first = src.out("first.vindex3");
    apply_weight_patch(&src.container, &on(TENSOR, vec![edit(0, 0, 0.5)]), &first).unwrap();
    let err = apply_weight_patch(
        &first,
        &on(TENSOR, vec![edit(1, 1, 0.5)]),
        &src.out("second.vindex3"),
    )
    .unwrap_err();
    assert!(matches!(err, WeightPatchError::AlreadyPatched));
}

#[test]
fn a_failure_after_staging_leaves_nothing_behind() {
    let src = source_with_pack();
    // The derived pack was compiled under a precision map the default policy
    // does not reproduce: the patch gets as far as staging, then refuses.
    let index_path = src.container.join("index.json");
    let mut index: serde_json::Value =
        serde_json::from_str(&std::fs::read_to_string(&index_path).unwrap()).unwrap();
    index.as_object_mut().unwrap().remove("precision_map");
    std::fs::write(&index_path, serde_json::to_string_pretty(&index).unwrap()).unwrap();

    let out = src.out("edited.vindex3");
    let err =
        apply_weight_patch(&src.container, &on(TENSOR, vec![edit(0, 0, 0.5)]), &out).unwrap_err();
    assert!(matches!(err, WeightPatchError::NotReproducible(_)), "{err}");
    assert!(!out.exists(), "no output");
    let leftovers: Vec<String> = std::fs::read_dir(src.dir.path())
        .unwrap()
        .map(|e| e.unwrap().file_name().to_string_lossy().to_string())
        .filter(|n| n.contains(".wsp-"))
        .collect();
    assert!(
        leftovers.is_empty(),
        "scratch directories remain: {leftovers:?}"
    );
}
