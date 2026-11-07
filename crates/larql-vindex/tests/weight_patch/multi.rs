//! One patch, several tensors: one container, one ledger, one re-derivation.

use larql_vindex::format::vindex3::encode::segment::read_segment_header;
use larql_vindex::format::vindex3::inspect::inspect_container;
use larql_vindex::format::vindex3::weight_patch::{
    apply_weight_patch, apply_weight_patches, parse_ledger, WeightEdit, WeightPatch,
    WeightPatchError, MAX_PATCH_ELEMENTS, WEIGHT_PATCHES_SCHEMA, WEIGHT_PATCHES_SCHEMA_V1,
};

use crate::fixture::*;

/// Layer 0's down projection: a second tensor of the same canonical segment.
const OTHER: &str = "0.mlp.down_proj.weight";
const EMBEDDING: &str = "target.embedding@BF16";

fn edit(row: usize, col: usize, delta: f32) -> WeightEdit {
    WeightEdit { row, col, delta }
}

fn on(tensor: &str, edits: Vec<WeightEdit>) -> WeightPatch {
    WeightPatch {
        tensor: tensor.to_string(),
        edits,
    }
}

fn two() -> Vec<WeightPatch> {
    vec![
        on(TENSOR, vec![edit(5, 33, 0.0625), edit(5, 34, -0.125)]),
        on(OTHER, vec![edit(1, 1, 0.25)]),
    ]
}

/// Every tensor name a representation's segment holds.
fn tensors_of(container: &std::path::Path, representation: &str) -> Vec<String> {
    let entry = index_of(container).representations[representation].clone();
    let (header, _) = read_segment_header(&container.join(&entry.segment)).unwrap();
    header.tensors.into_iter().map(|t| t.name).collect()
}

fn elements_differing(a: &[u8], b: &[u8]) -> usize {
    (0..a.len() / 2)
        .filter(|&i| a[2 * i..2 * i + 2] != b[2 * i..2 * i + 2])
        .count()
}

#[test]
fn two_tensors_are_edited_in_one_container_and_nothing_else_moves() {
    let src = source_with_pack();
    let out = src.out("two.vindex3");
    let report = apply_weight_patches(&src.container, &two(), &[], &out).unwrap();
    assert_eq!(report.elements, 3);
    assert_eq!(report.tensors, vec![TENSOR.to_string(), OTHER.to_string()]);
    assert_eq!(report.rederived, vec![DERIVED.to_string()]);
    assert!(inspect_container(&out, true).unwrap().is_coherent());
    for name in tensors_of(&src.container, CANONICAL) {
        let changed = elements_differing(
            &tensor_bytes(&src.container, CANONICAL, &name),
            &tensor_bytes(&out, CANONICAL, &name),
        );
        let want = match name.as_str() {
            TENSOR => 2,
            OTHER => 1,
            _ => 0,
        };
        assert_eq!(changed, want, "{name}");
    }
    let ledger =
        parse_ledger(&std::fs::read_to_string(out.join("weight_patches.json")).unwrap()).unwrap();
    assert_eq!(ledger.schema, WEIGHT_PATCHES_SCHEMA);
    assert_eq!(ledger.edit_count(), 3);
    assert_eq!(ledger.targets.len(), 2);
    assert!(ledger.targets.iter().all(|t| t.representation == CANONICAL));
}

#[test]
fn one_compile_of_two_tensors_equals_the_same_edits_as_one_tensor_each_where_they_overlap() {
    // The edited bytes of each tensor do not depend on what else the patch edits.
    let src = source_with_pack();
    let (both, alone) = (src.out("both.vindex3"), src.out("alone.vindex3"));
    apply_weight_patches(&src.container, &two(), &[], &both).unwrap();
    apply_weight_patch(&src.container, &two()[0], &alone).unwrap();
    assert_eq!(
        tensor_bytes(&both, CANONICAL, TENSOR),
        tensor_bytes(&alone, CANONICAL, TENSOR)
    );
}

#[test]
fn a_tensor_given_twice_is_refused() {
    let src = source_with_pack();
    let patches = vec![
        on(TENSOR, vec![edit(0, 0, 0.5)]),
        on(TENSOR, vec![edit(1, 1, 0.5)]),
    ];
    let out = src.out("dup.vindex3");
    let err = apply_weight_patches(&src.container, &patches, &[], &out).unwrap_err();
    assert!(matches!(err, WeightPatchError::DuplicateTensor(t) if t == TENSOR));
    assert!(!out.exists());
}

#[test]
fn no_patches_is_refused() {
    let src = source_with_pack();
    let err = apply_weight_patches(&src.container, &[], &[], &src.out("none.vindex3")).unwrap_err();
    assert!(matches!(err, WeightPatchError::Empty));
}

#[test]
fn the_limit_counts_elements_across_every_tensor() {
    let src = source_with_pack();
    let half = MAX_PATCH_ELEMENTS / 2 + 1;
    let patches = vec![
        on(TENSOR, vec![edit(0, 0, 0.5); half]),
        on(OTHER, vec![edit(0, 0, 0.5); half]),
    ];
    let err =
        apply_weight_patches(&src.container, &patches, &[], &src.out("big.vindex3")).unwrap_err();
    assert!(
        matches!(err, WeightPatchError::TooManyElements { count, limit } if count == 2 * half && limit == MAX_PATCH_ELEMENTS),
        "{err}"
    );
}

#[test]
fn a_second_tensor_outside_the_decoder_segment_is_refused_and_nothing_is_written() {
    // In this fixture every non-decoder canonical representation names its one
    // tensor `weight`, so the name is ambiguous and is refused as such before
    // the one-representation rule is reached. Either way nothing is written.
    let src = source_with_pack();
    let embedding = tensors_of(&src.container, EMBEDDING).remove(0);
    let patches = vec![
        on(TENSOR, vec![edit(0, 0, 0.5)]),
        on(&embedding, vec![edit(0, 0, 0.5)]),
    ];
    let out = src.out("span.vindex3");
    let err = apply_weight_patches(&src.container, &patches, &[], &out).unwrap_err();
    assert!(
        matches!(&err, WeightPatchError::AmbiguousTensor { holders, .. } if holders.contains(&EMBEDDING.to_string())),
        "{err}"
    );
    assert!(!out.exists());
}

#[test]
fn a_v1_ledger_is_still_read_as_its_one_target() {
    let src = source_with_pack();
    let out = src.out("v1.vindex3");
    apply_weight_patch(&src.container, &two()[0], &out).unwrap();
    // Rewrite the ledger in the v1 shape: one `target`, `edits` beside it.
    let path = out.join("weight_patches.json");
    let mut v: serde_json::Value =
        serde_json::from_str(&std::fs::read_to_string(&path).unwrap()).unwrap();
    let mut target = v["targets"][0].take();
    let edits = target["edits"].take();
    target.as_object_mut().unwrap().remove("edits");
    let obj = v.as_object_mut().unwrap();
    obj.remove("targets");
    obj.insert("target".into(), target);
    obj.insert("edits".into(), edits);
    obj.insert("schema".into(), WEIGHT_PATCHES_SCHEMA_V1.into());
    let text = serde_json::to_string_pretty(&v).unwrap();
    std::fs::write(&path, &text).unwrap();

    let ledger = parse_ledger(&text).unwrap();
    assert_eq!(ledger.schema, WEIGHT_PATCHES_SCHEMA_V1);
    assert_eq!(ledger.targets.len(), 1);
    assert_eq!(ledger.targets[0].tensor, TENSOR);
    assert_eq!(ledger.edit_count(), 2);
    assert!(inspect_container(&out, true).unwrap().is_coherent());

    // A v1 ledger that lies is still caught.
    v["edits"][0]["new_bits"] = (v["edits"][0]["new_bits"].as_u64().unwrap() ^ 1).into();
    std::fs::write(&path, serde_json::to_string_pretty(&v).unwrap()).unwrap();
    assert!(!inspect_container(&out, true).unwrap().is_coherent());
}

#[test]
fn an_unknown_schema_is_a_defect() {
    let src = source_with_pack();
    let out = src.out("v9.vindex3");
    apply_weight_patch(&src.container, &two()[0], &out).unwrap();
    let path = out.join("weight_patches.json");
    let text = std::fs::read_to_string(&path)
        .unwrap()
        .replace(WEIGHT_PATCHES_SCHEMA, "weight-patches/v9");
    std::fs::write(&path, text).unwrap();
    let report = inspect_container(&out, true).unwrap();
    assert!(
        report
            .defects
            .iter()
            .any(|d| format!("{d:?}").contains("weight-patches/v9")),
        "{:?}",
        report.defects
    );
}
