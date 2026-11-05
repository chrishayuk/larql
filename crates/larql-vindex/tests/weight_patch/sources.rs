//! What a source container must look like, and what happens when it does not.

use larql_vindex::format::vindex3::inspect::inspect_container;
use larql_vindex::format::vindex3::weight_patch::{
    apply_weight_patch, apply_weight_patch_with_origins, plan_weight_patch, WeightEdit,
    WeightPatch, WeightPatchError,
};

use crate::fixture::*;

fn patch() -> WeightPatch {
    WeightPatch {
        tensor: TENSOR.to_string(),
        edits: vec![WeightEdit {
            row: 5,
            col: 33,
            delta: 0.0625,
        }],
    }
}

fn edit_index(container: &std::path::Path, change: impl FnOnce(&mut serde_json::Value)) {
    let path = container.join("index.json");
    let mut v: serde_json::Value =
        serde_json::from_str(&std::fs::read_to_string(&path).unwrap()).unwrap();
    change(&mut v);
    std::fs::write(path, serde_json::to_string_pretty(&v).unwrap()).unwrap();
}

#[test]
fn planning_validates_without_writing_and_reports_the_exact_bits() {
    let src = source_with_pack();
    let planned = plan_weight_patch(&src.container, &patch()).unwrap();
    assert_eq!(planned.len(), 1);
    assert_ne!(planned[0].old_bits, planned[0].new_bits);
    let bad = WeightPatch {
        tensor: TENSOR.to_string(),
        edits: vec![],
    };
    assert!(matches!(
        plan_weight_patch(&src.container, &bad),
        Err(WeightPatchError::Empty)
    ));
    assert!(std::fs::read_dir(src.dir.path()).unwrap().all(|e| !e
        .unwrap()
        .file_name()
        .to_string_lossy()
        .contains(".wsp-")));
}

#[test]
fn origins_are_recorded_in_the_ledger() {
    let src = source_with_pack();
    let out = src.out("edited.vindex3");
    let origins = vec!["a".repeat(64), "b".repeat(64)];
    let report = apply_weight_patch_with_origins(&src.container, &patch(), &origins, &out).unwrap();
    assert_eq!(report.ledger.origins, origins);
    let text = std::fs::read_to_string(out.join("weight_patches.json")).unwrap();
    assert!(text.contains(&origins[0]));
}

#[test]
fn a_container_with_no_derived_representation_is_patched_without_re_deriving() {
    let src = source_with_pack();
    let canonical_only = src.dir.path().join("canonical.vindex3");
    let out = src.out("edited.vindex3");
    let report = apply_weight_patch(&canonical_only, &patch(), &out).unwrap();
    assert!(report.rederived.is_empty());
    assert!(report.ledger.result.derived.is_empty());
    assert!(inspect_container(&out, true).unwrap().is_coherent());
}

#[test]
fn a_deployment_image_cannot_be_patched() {
    let src = source_with_pack();
    edit_index(&src.container, |v| v["authority"] = "derived".into());
    let err = apply_weight_patch(&src.container, &patch(), &src.out("o.vindex3")).unwrap_err();
    assert!(matches!(err, WeightPatchError::NotCanonical(_)), "{err}");
}

#[test]
fn an_unknown_entry_in_the_container_is_refused_not_dropped() {
    let src = source_with_pack();
    std::fs::create_dir(src.container.join("extra")).unwrap();
    let err = apply_weight_patch(&src.container, &patch(), &src.out("o.vindex3")).unwrap_err();
    assert!(
        matches!(err, WeightPatchError::UnexpectedEntry(ref n) if n == "extra"),
        "{err}"
    );
}

#[test]
fn a_derived_representation_of_another_object_is_refused() {
    let src = source_with_pack();
    edit_index(&src.container, |v| {
        v["representations"][DERIVED]["compiled_from"] = "target.embedding@BF16".into();
    });
    let err = apply_weight_patch(&src.container, &patch(), &src.out("o.vindex3")).unwrap_err();
    assert!(
        matches!(err, WeightPatchError::DerivedFromOther { .. }),
        "{err}"
    );
}

#[test]
fn an_incoherent_source_is_refused_before_anything_is_written() {
    let src = source_with_pack();
    let segment = src
        .container
        .join(&index_of(&src.container).representations[CANONICAL].segment);
    let file = std::fs::OpenOptions::new()
        .write(true)
        .open(&segment)
        .unwrap();
    file.set_len(file.metadata().unwrap().len() - 1).unwrap();
    let out = src.out("o.vindex3");
    let err = apply_weight_patch(&src.container, &patch(), &out).unwrap_err();
    assert!(
        matches!(err, WeightPatchError::SourceIncoherent(_)),
        "{err}"
    );
    assert!(!out.exists());
}

#[test]
fn a_stored_value_that_is_not_finite_cannot_be_patched() {
    let src = source_with_pack();
    let entry = index_of(&src.container).representations[CANONICAL].clone();
    let segment = src.container.join(&entry.segment);
    let (header, start) =
        larql_vindex::format::vindex3::encode::segment::read_segment_header(&segment).unwrap();
    let t = header.tensors.iter().find(|t| t.name == TENSOR).unwrap();
    let at = (start + t.offset) as usize + (5 * TENSOR_COLS + 33) * 2;
    let mut bytes = std::fs::read(&segment).unwrap();
    bytes[at..at + 2].copy_from_slice(&0x7fc0u16.to_le_bytes()); // a NaN
    std::fs::write(&segment, bytes).unwrap();
    let err = apply_weight_patch(&src.container, &patch(), &src.out("o.vindex3")).unwrap_err();
    assert!(
        matches!(
            err,
            WeightPatchError::NonFiniteSource {
                row: 5,
                col: 33,
                ..
            }
        ),
        "{err}"
    );
}
