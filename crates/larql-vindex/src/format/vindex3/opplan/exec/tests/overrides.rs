//! The operand-source seam (V3-LQL-3B compose): execution asks for
//! operands through a resolver — base representation + overlay
//! override → effective operand. The gates here pin the seam's two
//! obligations:
//!
//! 1. a source with **no** overrides is the bare store, bit for bit —
//!    the seam may never become a semantic fork;
//! 2. an override **is observed by execution**: editing an FFN row
//!    changes the traversal's output, and dropping the override
//!    returns it bit-identically to baseline.

use super::golden::{miniature_glimmer, G_TOKENS};
use crate::format::vindex3::encode::encode_system;
use crate::format::vindex3::inspect::inspect_container;
use crate::format::vindex3::opplan::exec::execute_plan;
use crate::format::vindex3::opplan::exec::operands::{
    OperandEdit, OperandOverrides, OperandSource, OperandStore,
};
use crate::format::vindex3::opplan::exec::reference::ReferenceBackend;
use crate::format::vindex3::opplan::{plan_component_ops, ComponentOpPlan, LayerFfn};

fn fixture() -> (tempfile::TempDir, ComponentOpPlan, OperandStore) {
    let dir = tempfile::tempdir().unwrap();
    miniature_glimmer(dir.path());
    let inventory = larql_models::inventory::build_inventory(dir.path()).unwrap();
    let container = tempfile::tempdir().unwrap();
    encode_system(&[("mini-glimmer".to_string(), inventory)], container.path()).unwrap();
    let inspection = inspect_container(container.path(), false).unwrap();
    let outcome = plan_component_ops(&inspection, container.path(), "target").unwrap();
    let plan = outcome.plan.unwrap();
    let store = OperandStore::open(container.path(), &inspection).unwrap();
    (container, plan, store)
}

/// Layer 0's dense FFN gate operand — the compose install's target.
fn gate_ref(plan: &ComponentOpPlan) -> crate::format::vindex3::opplan::OperandRef {
    match &plan.layers[0].ffn {
        Some(LayerFfn::Dense(op)) => op.gate.clone().expect("miniature FFN is gated"),
        other => panic!("miniature layer 0 is dense, got {other:?}"),
    }
}

#[test]
fn an_empty_source_is_the_bare_store_bit_for_bit() {
    let (_c, plan, store) = fixture();
    let backend = ReferenceBackend::new();
    let base = execute_plan(&plan, &store, &G_TOKENS, &backend).unwrap();

    let empty = OperandOverrides::new();
    let overlaid = execute_plan(
        &plan,
        OperandSource::overlaid(&store, &empty),
        &G_TOKENS,
        &backend,
    )
    .unwrap();

    assert_eq!(base.logits, overlaid.logits, "logits diverge");
    for (a, b) in base.layers.iter().zip(&overlaid.layers) {
        assert_eq!(
            a.post_layer.rows(),
            b.post_layer.rows(),
            "a residual diverges"
        );
    }
}

#[test]
fn a_row_override_is_observed_by_execution_and_reverts_cleanly() {
    let (_c, plan, store) = fixture();
    let backend = ReferenceBackend::new();
    let base = execute_plan(&plan, &store, &G_TOKENS, &backend).unwrap();

    let gate = gate_ref(&plan);
    let cols = gate.shape[1];
    let mut overrides = OperandOverrides::new();
    overrides.push(
        &gate,
        OperandEdit::Row {
            index: 0,
            values: vec![3.0; cols],
        },
    );

    let edited = execute_plan(
        &plan,
        OperandSource::overlaid(&store, &overrides),
        &G_TOKENS,
        &backend,
    )
    .unwrap();
    assert_ne!(
        base.logits, edited.logits,
        "a gate-row edit must change what execution computes"
    );

    // Loading through the source shows exactly the edit, nothing else.
    let effective = OperandSource::overlaid(&store, &overrides)
        .load(&gate)
        .unwrap();
    let stored = store.load(&gate).unwrap();
    assert_eq!(&effective[..cols], &vec![3.0; cols][..]);
    assert_eq!(&effective[cols..], &stored[cols..], "other rows untouched");

    // Dropping the override returns execution bit-for-bit to baseline.
    let reverted = execute_plan(&plan, &store, &G_TOKENS, &backend).unwrap();
    assert_eq!(base.logits, reverted.logits);
}

#[test]
fn a_column_override_lands_on_the_declared_column() {
    let (_c, plan, store) = fixture();
    let down = match &plan.layers[0].ffn {
        Some(LayerFfn::Dense(op)) => op.down.clone(),
        other => panic!("miniature layer 0 is dense, got {other:?}"),
    };
    let (rows, cols) = (down.shape[0], down.shape[1]);
    let mut overrides = OperandOverrides::new();
    overrides.push(
        &down,
        OperandEdit::Column {
            index: 1,
            values: vec![7.0; rows],
        },
    );
    let effective = OperandSource::overlaid(&store, &overrides)
        .load(&down)
        .unwrap();
    let stored = store.load(&down).unwrap();
    for r in 0..rows {
        for c in 0..cols {
            let expect = if c == 1 { 7.0 } else { stored[r * cols + c] };
            assert_eq!(effective[r * cols + c], expect, "at ({r},{c})");
        }
    }
}

#[test]
fn misfit_edits_and_raw_access_refuse_loudly() {
    let (_c, plan, store) = fixture();
    let gate = gate_ref(&plan);

    // A row edit of the wrong width errors naming the operand.
    let mut bad = OperandOverrides::new();
    bad.push(
        &gate,
        OperandEdit::Row {
            index: 0,
            values: vec![1.0; 3],
        },
    );
    let err = OperandSource::overlaid(&store, &bad)
        .load(&gate)
        .expect_err("misfit edits must refuse");
    assert!(err.to_string().contains(&gate.tensor), "{err}");

    // Raw (unwidened) access to an overridden operand refuses — the
    // stored bytes would bypass the overlay.
    let cols = gate.shape[1];
    let mut overrides = OperandOverrides::new();
    overrides.push(
        &gate,
        OperandEdit::Row {
            index: 0,
            values: vec![1.0; cols],
        },
    );
    let source = OperandSource::overlaid(&store, &overrides);
    let err = match source.load_raw(&gate) {
        Err(e) => e,
        Ok(_) => panic!("raw access must refuse"),
    };
    assert!(err.to_string().contains("overlay"), "{err}");
    // …while a non-overridden operand still serves raw bytes.
    let embed = &plan.embedding.as_ref().unwrap().table;
    assert!(source.load_raw(embed).is_ok());
}
