//! Inspection checks that the ledger describes the container. It checks
//! consistency; it does not (and says it does not) judge the patch.

use larql_vindex::format::vindex3::inspect::inspect_container;
use larql_vindex::format::vindex3::weight_patch::{
    apply_weight_patch, WeightEdit, WeightPatch, WeightPatchLedger,
};

use crate::fixture::*;

/// A change applied to a ledger, to build an inconsistent one.
type Change = Box<dyn Fn(&mut WeightPatchLedger)>;

fn patched() -> (Source, std::path::PathBuf) {
    let src = source_with_pack();
    let out = src.out("edited.vindex3");
    let patch = WeightPatch {
        tensor: TENSOR.to_string(),
        edits: vec![
            WeightEdit {
                row: 5,
                col: 33,
                delta: 0.0625,
            },
            WeightEdit {
                row: 5,
                col: 34,
                delta: -0.125,
            },
        ],
    };
    apply_weight_patch(&src.container, &patch, &out).unwrap();
    (src, out)
}

fn ledger_defects(out: &std::path::Path) -> Vec<String> {
    inspect_container(out, false)
        .unwrap()
        .defects
        .iter()
        .map(|d| format!("{d:?}"))
        .filter(|d| d.contains("weight patch"))
        .collect()
}

fn edit_ledger(out: &std::path::Path, change: impl FnOnce(&mut WeightPatchLedger)) {
    let path = out.join("weight_patches.json");
    let mut ledger: WeightPatchLedger =
        serde_json::from_str(&std::fs::read_to_string(&path).unwrap()).unwrap();
    change(&mut ledger);
    std::fs::write(path, serde_json::to_string_pretty(&ledger).unwrap()).unwrap();
}

#[test]
fn a_faithful_ledger_passes() {
    let (_src, out) = patched();
    assert!(ledger_defects(&out).is_empty());
    assert!(inspect_container(&out, true).unwrap().is_coherent());
}

#[test]
fn a_container_without_a_ledger_is_unaffected() {
    let src = source_with_pack();
    assert!(inspect_container(&src.container, true)
        .unwrap()
        .is_coherent());
}

#[test]
fn a_ledger_that_is_named_but_missing_is_a_defect() {
    let (_src, out) = patched();
    std::fs::remove_file(out.join("weight_patches.json")).unwrap();
    assert!(!ledger_defects(&out).is_empty());
}

#[test]
fn a_malformed_ledger_is_a_defect() {
    let (_src, out) = patched();
    std::fs::write(out.join("weight_patches.json"), "{ not json").unwrap();
    assert!(!ledger_defects(&out).is_empty());
}

#[test]
fn each_inconsistency_between_ledger_and_container_is_reported() {
    let cases: Vec<(&str, Change)> = vec![
        (
            "a different schema",
            Box::new(|l| l.schema = "weight-patches/v0".into()),
        ),
        (
            "a different rounding rule",
            Box::new(|l| l.rounding = "truncate".into()),
        ),
        (
            "a different source model",
            Box::new(|l| l.source.model = "someone/else".into()),
        ),
        (
            "result digests that are not the catalogue's",
            Box::new(|l| l.result.canonical.payload_sha256 = "0".repeat(64)),
        ),
        (
            "a derived pack that is not listed",
            Box::new(|l| l.result.derived.clear()),
        ),
        (
            "a derived pack from another source",
            Box::new(|l| l.result.derived[0].source_representation_digest = "1".repeat(64)),
        ),
        (
            "an edit whose new value is not what is stored",
            Box::new(|l| l.targets[0].edits[0].new_bits ^= 1),
        ),
        (
            "an edit outside the tensor",
            Box::new(|l| l.targets[0].edits[0].row = 10_000),
        ),
        (
            "an element listed twice",
            Box::new(|l| {
                let e = l.targets[0].edits[0].clone();
                l.targets[0].edits.push(e);
            }),
        ),
        (
            "a delta that disagrees with its bits",
            Box::new(|l| l.targets[0].edits[0].delta = 3.0),
        ),
        (
            "a tensor the segment does not hold",
            Box::new(|l| l.targets[0].tensor = "no.such.weight".into()),
        ),
        (
            "a representation the container does not hold",
            Box::new(|l| l.targets[0].representation = "ghost@BF16".into()),
        ),
        (
            "a derived representation as the edited one",
            Box::new(|l| l.targets[0].representation = DERIVED.into()),
        ),
        (
            "a dtype the segment does not carry",
            Box::new(|l| l.targets[0].dtype = "F32".into()),
        ),
        (
            "a target that is not a matrix",
            Box::new(|l| l.targets[0].shape = vec![64]),
        ),
        ("no edited tensor at all", Box::new(|l| l.targets.clear())),
        (
            "two canonical representations",
            Box::new(|l| {
                let mut t = l.targets[0].clone();
                t.representation = "target.embedding@BF16".into();
                t.tensor = "elsewhere.weight".into();
                l.targets.push(t);
            }),
        ),
        (
            "a tensor listed with no edited element",
            Box::new(|l| l.targets[0].edits.clear()),
        ),
        (
            "one tensor listed twice",
            Box::new(|l| {
                let t = l.targets[0].clone();
                l.targets.push(t);
            }),
        ),
        (
            "derived digests that are not the catalogue's",
            Box::new(|l| l.result.derived[0].digests.payload_sha256 = "2".repeat(64)),
        ),
        (
            "a derived pack the container does not hold",
            Box::new(|l| {
                let mut ghost = l.result.derived[0].clone();
                ghost.representation = "ghost@Q8_0".into();
                l.result.derived.push(ghost);
            }),
        ),
    ];
    for (what, change) in cases {
        let (_src, out) = patched();
        edit_ledger(&out, |l| change(l));
        assert!(!ledger_defects(&out).is_empty(), "{what} was not reported");
    }
}

#[test]
fn the_stored_bytes_are_checked_not_just_the_ledger() {
    let (_src, out) = patched();
    // Change the stored element but leave the ledger alone: inspection must see it.
    let index = index_of(&out);
    let segment = out.join(&index.representations[CANONICAL].segment);
    let (header, start) =
        larql_vindex::format::vindex3::encode::segment::read_segment_header(&segment).unwrap();
    let t = header.tensors.iter().find(|t| t.name == TENSOR).unwrap();
    let at = (start + t.offset) as usize + (5 * TENSOR_COLS + 33) * 2;
    let mut bytes = std::fs::read(&segment).unwrap();
    bytes[at] ^= 0x01;
    std::fs::write(&segment, bytes).unwrap();
    assert!(!ledger_defects(&out).is_empty());
}

#[test]
fn a_container_whose_graph_is_an_older_schema_is_refused_by_name() {
    // `inspect_container` is the opening step for every consumer of a container, and
    // reads the graph's schema before the graph. An older one is not parsed as if it
    // were current and reported as a defect in something else.
    let src = source_with_pack();
    let path = src.container.join("system_graph.json");
    let mut graph: serde_json::Value =
        serde_json::from_str(&std::fs::read_to_string(&path).unwrap()).unwrap();
    graph["schema"] = 1.into();
    std::fs::write(&path, serde_json::to_string_pretty(&graph).unwrap()).unwrap();
    let err = inspect_container(&src.container, false)
        .unwrap_err()
        .to_string();
    assert!(
        err.contains("schema v1") && err.contains("re-encode the container"),
        "{err}"
    );
}
