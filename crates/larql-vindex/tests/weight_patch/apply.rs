//! A patch produces a new, complete, self-describing container.

use larql_vindex::format::vindex3::inspect::inspect_container;
use larql_vindex::format::vindex3::weight_patch::{
    apply_weight_patch, WeightEdit, WeightPatch, WeightPatchLedger,
};

use crate::fixture::*;

const ROW: usize = 5;
/// One Q8_0 block's worth of columns: 32..64.
const COLS: [usize; 4] = [32, 33, 34, 40];
const DELTA: f32 = 0.0625;

fn patch() -> WeightPatch {
    WeightPatch {
        tensor: TENSOR.to_string(),
        edits: COLS
            .iter()
            .map(|&col| WeightEdit {
                row: ROW,
                col,
                delta: DELTA,
            })
            .collect(),
    }
}

fn changed_elements(a: &[u8], b: &[u8]) -> Vec<usize> {
    (0..a.len() / 2)
        .filter(|&i| a[2 * i..2 * i + 2] != b[2 * i..2 * i + 2])
        .collect()
}

#[test]
fn the_result_is_complete_and_the_source_is_untouched() {
    let src = source_with_pack();
    let before = tree_digest(&src.container);
    let out = src.out("edited.vindex3");
    let report = apply_weight_patch(&src.container, &patch(), &out).unwrap();
    assert_eq!(
        tree_digest(&src.container),
        before,
        "the source container is never modified"
    );
    assert!(
        inspect_container(&out, true).unwrap().is_coherent(),
        "payload hashes included"
    );
    assert_eq!(report.elements, COLS.len());
    assert_eq!(report.rederived, vec![DERIVED.to_string()]);
    assert!(out.join("weight_patches.json").exists());
    assert_eq!(
        index_of(&out).weight_patches.as_deref(),
        Some("weight_patches.json")
    );
}

#[test]
fn exactly_the_declared_elements_change_in_the_canonical_weights() {
    let src = source_with_pack();
    let out = src.out("edited.vindex3");
    apply_weight_patch(&src.container, &patch(), &out).unwrap();
    let (a, b) = (
        tensor_bytes(&src.container, CANONICAL, TENSOR),
        tensor_bytes(&out, CANONICAL, TENSOR),
    );
    let expected: Vec<usize> = COLS.iter().map(|c| ROW * TENSOR_COLS + c).collect();
    assert_eq!(changed_elements(&a, &b), expected);
    for other in [
        "0.mlp.down_proj.weight",
        "1.mlp.up_proj.weight",
        "0.self_attn.q_proj.weight",
    ] {
        assert_eq!(
            tensor_bytes(&src.container, CANONICAL, other),
            tensor_bytes(&out, CANONICAL, other),
            "{other} must be untouched"
        );
    }
}

#[test]
fn only_the_blocks_holding_edited_elements_change_in_the_derived_pack() {
    let src = source_with_pack();
    let out = src.out("edited.vindex3");
    apply_weight_patch(&src.container, &patch(), &out).unwrap();
    let (a, b) = (
        tensor_bytes(&src.container, DERIVED, TENSOR),
        tensor_bytes(&out, DERIVED, TENSOR),
    );
    assert_eq!(a.len(), b.len());
    let blocks_per_row = TENSOR_COLS / Q8_BLOCK_ELEMS;
    let changed: Vec<usize> = (0..a.len() / Q8_BLOCK_BYTES)
        .filter(|&k| {
            a[k * Q8_BLOCK_BYTES..(k + 1) * Q8_BLOCK_BYTES]
                != b[k * Q8_BLOCK_BYTES..(k + 1) * Q8_BLOCK_BYTES]
        })
        .collect();
    assert_eq!(
        changed,
        vec![ROW * blocks_per_row + 1],
        "row 5, block 1 (columns 32..64) and nothing else"
    );
    for other in ["0.mlp.down_proj.weight", "1.mlp.gate_proj.weight"] {
        assert_eq!(
            tensor_bytes(&src.container, DERIVED, other),
            tensor_bytes(&out, DERIVED, other)
        );
    }
}

#[test]
fn a_zero_delta_patch_preserves_every_representation_byte_for_byte() {
    let src = source_with_pack();
    let out = src.out("noop.vindex3");
    let noop = WeightPatch {
        tensor: TENSOR.to_string(),
        edits: vec![WeightEdit {
            row: ROW,
            col: 33,
            delta: 0.0,
        }],
    };
    apply_weight_patch(&src.container, &noop, &out).unwrap();
    let (a, b) = (index_of(&src.container), index_of(&out));
    for (id, entry) in &a.representations {
        let new = &b.representations[id];
        assert_eq!(entry.payload_sha256, new.payload_sha256, "{id} payload");
        assert_eq!(entry.segment_sha256, new.segment_sha256, "{id} segment");
    }
    assert_eq!(a.model, b.model);
}

#[test]
fn the_ledger_binds_source_edits_and_result() {
    let src = source_with_pack();
    let out = src.out("edited.vindex3");
    let report = apply_weight_patch(&src.container, &patch(), &out).unwrap();
    let on_disk: WeightPatchLedger =
        serde_json::from_str(&std::fs::read_to_string(out.join("weight_patches.json")).unwrap())
            .unwrap();
    assert_eq!(on_disk, report.ledger);
    let (a, b) = (index_of(&src.container), index_of(&out));
    assert_eq!(b.model, a.model, "the upstream model identity is unchanged");
    assert_eq!(on_disk.source.model, a.model);
    assert_eq!(
        on_disk.source.canonical.payload_sha256,
        a.representations[CANONICAL].payload_sha256
    );
    assert_eq!(
        on_disk.result.canonical.payload_sha256,
        b.representations[CANONICAL].payload_sha256
    );
    assert_ne!(
        on_disk.source.canonical.payload_sha256,
        on_disk.result.canonical.payload_sha256
    );
    let derived = &on_disk.result.derived[0];
    assert_eq!(
        derived.source_representation_digest, on_disk.result.canonical.payload_sha256,
        "derives from the EDITED source"
    );
    assert_eq!(
        b.representations[DERIVED]
            .source_representation_digest
            .as_deref(),
        Some(derived.source_representation_digest.as_str())
    );
    assert_eq!(on_disk.targets[0].edits.len(), COLS.len());
    for e in &on_disk.targets[0].edits {
        assert_eq!(e.delta.to_bits(), e.delta_bits);
        assert_ne!(e.old_bits, e.new_bits);
    }
}

#[test]
fn a_patch_is_deterministic() {
    let src = source_with_pack();
    let (one, two) = (src.out("one.vindex3"), src.out("two.vindex3"));
    apply_weight_patch(&src.container, &patch(), &one).unwrap();
    apply_weight_patch(&src.container, &patch(), &two).unwrap();
    for id in [CANONICAL, DERIVED] {
        let (a, b) = (index_of(&one), index_of(&two));
        assert_eq!(
            a.representations[id].segment_sha256, b.representations[id].segment_sha256,
            "{id}"
        );
    }
}
