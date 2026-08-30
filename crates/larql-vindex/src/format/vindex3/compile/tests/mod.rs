//! Gates for the container bake at its source: effective bytes land in
//! the rewritten segment, untouched segments are linked byte-identical,
//! hashes stay truthful (verified inspection reopens the result), and
//! the refusals fail closed.

use crate::format::vindex3::fixtures::{encode_fixture_container, miniature_glimmer};
use crate::format::vindex3::inspect::inspect_container;
use crate::format::vindex3::opplan::exec::operands::{
    OperandEdit, OperandOverrides, OperandSource, OperandStore,
};
use crate::format::vindex3::opplan::{plan_component_ops, ComponentOpPlan, LayerFfn};

use super::bake_container;

fn fixture() -> (tempfile::TempDir, ComponentOpPlan, OperandStore) {
    let checkpoint = tempfile::tempdir().unwrap();
    let container = tempfile::tempdir().unwrap();
    encode_fixture_container(
        miniature_glimmer,
        checkpoint.path(),
        container.path(),
        "bake-fixture",
    );
    let inspection = inspect_container(container.path(), false).unwrap();
    let outcome = plan_component_ops(&inspection, container.path(), "target").unwrap();
    let plan = outcome.plan.unwrap();
    let store = OperandStore::open(container.path(), &inspection).unwrap();
    (container, plan, store)
}

fn gate_edit(
    plan: &ComponentOpPlan,
) -> (OperandOverrides, crate::format::vindex3::opplan::OperandRef) {
    let Some(LayerFfn::Dense(ffn)) = &plan.layers[0].ffn else {
        panic!("miniature layer 0 is dense");
    };
    let gate = ffn.gate.as_ref().unwrap().clone();
    let mut overrides = OperandOverrides::new();
    overrides.push(
        &gate,
        OperandEdit::Row {
            index: 2,
            values: vec![4.0; gate.shape[1]],
        },
    );
    (overrides, gate)
}

/// The bake's whole contract in one pass: rewritten segment serves the
/// effective bytes, untouched segments are byte-identical, and the new
/// index's hashes are TRUE — pinned by reopening with payload
/// verification on.
#[test]
fn bake_writes_effective_bytes_with_truthful_hashes() {
    let (container, plan, store) = fixture();
    let (overrides, gate) = gate_edit(&plan);
    let out = tempfile::tempdir().unwrap();

    let report = bake_container(container.path(), &overrides, out.path()).unwrap();
    assert_eq!(report.rewritten_tensors, 1);
    assert!(report.rewritten_segments >= 1);
    assert!(report.linked_segments >= 1);

    // Verified reopen: every hash in the new index must match the new
    // bytes, or the bake lied about what it wrote.
    let inspection = inspect_container(out.path(), true).unwrap();
    let baked_store = OperandStore::open(out.path(), &inspection).unwrap();

    // The baked container's STORED bytes equal the overlaid source's
    // EFFECTIVE bytes.
    let effective = OperandSource::overlaid(&store, &overrides)
        .load(&gate)
        .unwrap();
    assert_eq!(baked_store.load(&gate).unwrap(), effective);
    let cols = gate.shape[1];
    assert_eq!(&effective[2 * cols..3 * cols], &vec![4.0; cols][..]);

    // An untouched operand is byte-identical to the source.
    let embed = &plan.embedding.as_ref().unwrap().table;
    assert_eq!(baked_store.load(embed).unwrap(), store.load(embed).unwrap());
}

/// Fail-closed refusals: an edit into a non-f32 stored tensor refuses
/// (representation policy), and a misfit edit surfaces the resolver's
/// error rather than a half-written container.
#[test]
fn bake_refuses_misfit_edits() {
    let (container, plan, _store) = fixture();
    let Some(LayerFfn::Dense(ffn)) = &plan.layers[0].ffn else {
        panic!("miniature layer 0 is dense");
    };
    let gate = ffn.gate.as_ref().unwrap().clone();
    let mut overrides = OperandOverrides::new();
    overrides.push(
        &gate,
        OperandEdit::Row {
            index: 0,
            values: vec![1.0; 3],
        },
    );
    let out = tempfile::tempdir().unwrap();
    let err = bake_container(container.path(), &overrides, out.path())
        .expect_err("misfit edits must refuse");
    assert!(err.to_string().contains(&gate.tensor), "{err}");
}

/// An edit into a tensor stored in a non-f32 representation refuses —
/// rewriting through a lossy encoding is a representation-policy
/// decision the bake does not take. The fixture's segment header is
/// relabelled BF16 in place (the refusal fires before any payload
/// read, so the label is all that matters).
#[test]
fn bake_refuses_non_f32_edited_tensors() {
    let (container, plan, _store) = fixture();
    let (overrides, gate) = gate_edit(&plan);

    // Find the segment holding the gate tensor and relabel its dtype.
    let raw_index = std::fs::read_to_string(container.path().join("index.json")).unwrap();
    let index: crate::format::vindex3::index::Vindex3Index =
        serde_json::from_str(&raw_index).unwrap();
    let entry = index
        .representations
        .values()
        .find(|e| e.object == gate.object)
        .expect("gate object has a representation");
    let seg_path = container.path().join(&entry.segment);
    let bytes = std::fs::read(&seg_path).unwrap();
    let header_len = u64::from_le_bytes(bytes[..8].try_into().unwrap()) as usize;
    let header = String::from_utf8(bytes[8..8 + header_len].to_vec()).unwrap();
    let needle = format!("\"name\":\"{}\",\"dtype\":\"F32\"", gate.tensor);
    assert!(header.contains(&needle), "header shape changed: {header}");
    let relabelled = header.replace(&needle, &needle.replace("F32", "BF16"));
    let mut patched = Vec::new();
    patched.extend_from_slice(&(relabelled.len() as u64).to_le_bytes());
    patched.extend_from_slice(relabelled.as_bytes());
    patched.extend_from_slice(&bytes[8 + header_len..]);
    std::fs::write(&seg_path, patched).unwrap();

    let out = tempfile::tempdir().unwrap();
    let err = bake_container(container.path(), &overrides, out.path())
        .expect_err("non-f32 edited tensors must refuse");
    assert!(err.to_string().contains("BF16"), "{err}");
    assert!(err.to_string().contains("representation-policy"), "{err}");
}

/// Drill finding F13: a field this build does not understand must
/// survive the bake (candidate spec §5.7 — a rewriter MUST NOT drop
/// what it cannot interpret). COMPACT preserves by carrying the file
/// byte-identically; the bake rewrites the index, so preservation must
/// be structural.
#[test]
fn bake_preserves_unknown_index_fields() {
    use crate::format::filenames::INDEX_JSON;

    let (container, plan, _store) = fixture();
    let (overrides, _gate) = gate_edit(&plan);

    // A future 3.x writer added vocabulary this build has never seen.
    let index_path = container.path().join(INDEX_JSON);
    let mut raw: serde_json::Value =
        serde_json::from_str(&std::fs::read_to_string(&index_path).unwrap()).unwrap();
    raw["residency_contract"] = serde_json::json!({ "access_class": "demand_driven" });
    std::fs::write(&index_path, serde_json::to_string_pretty(&raw).unwrap()).unwrap();

    let out = tempfile::tempdir().unwrap();
    bake_container(container.path(), &overrides, out.path()).unwrap();

    let baked: serde_json::Value =
        serde_json::from_str(&std::fs::read_to_string(out.path().join(INDEX_JSON)).unwrap())
            .unwrap();
    assert_eq!(
        baked["residency_contract"]["access_class"],
        serde_json::json!("demand_driven"),
        "an unknown index field must survive the bake, not be dropped by the struct round-trip"
    );
}
