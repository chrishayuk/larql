//! **A headless primary text model is not closed, and not complete** (#743).
//!
//! The issue's reproduction, on the dense fixture: a tied-embedding
//! container (no head tensor) whose graph is edited to
//! `head_reuses_embedding: false` declares a vocabulary and has no way to
//! produce a logit. `ops` used to report the plan closed and `inspect` the
//! container coherent; only scoring refused. Both verdicts now name it.

use crate::format::vindex3::encode::SYSTEM_GRAPH_JSON;
use crate::format::vindex3::fixtures::{dense_f32_model_with, HeadStorage};
use crate::format::vindex3::graph::CompletenessDefect;
use crate::format::vindex3::inspect::{inspect_container, SystemInspection};
use crate::format::vindex3::opplan::{plan_component_ops, ClosureDefect};

const COMPONENT: &str = "target";

/// Encode the dense fixture with `head` storage; when `unreuse` is set,
/// declare in the graph that the head does NOT reuse the embedding.
fn encoded(head: HeadStorage, unreuse: bool) -> (tempfile::TempDir, SystemInspection) {
    let checkpoint = tempfile::tempdir().unwrap();
    let container = tempfile::tempdir().unwrap();
    dense_f32_model_with(checkpoint.path(), head);
    let inventory = larql_models::inventory::build_inventory(checkpoint.path()).unwrap();
    crate::format::vindex3::encode::encode_system(
        &[(COMPONENT.to_string(), inventory)],
        container.path(),
    )
    .unwrap();
    if unreuse {
        let path = container.path().join(SYSTEM_GRAPH_JSON);
        let mut graph: serde_json::Value =
            serde_json::from_slice(&std::fs::read(&path).unwrap()).unwrap();
        let head = graph["components"]
            .as_array_mut()
            .unwrap()
            .iter_mut()
            .find(|c| c["id"] == COMPONENT)
            .unwrap()["execution"]["head"]
            .as_object_mut()
            .expect("the text component declares a head surface");
        head.insert("head_reuses_embedding".into(), false.into());
        std::fs::write(&path, serde_json::to_vec(&graph).unwrap()).unwrap();
    }
    let inspection = inspect_container(container.path(), false).unwrap();
    (container, inspection)
}

fn closure_names_no_head(inspection: &SystemInspection, root: &std::path::Path) -> bool {
    let outcome = plan_component_ops(inspection, root, COMPONENT).unwrap();
    outcome.defects.contains(&ClosureDefect::NoOutputHead {
        component: COMPONENT.into(),
    })
}

fn completeness_names_no_head(inspection: &SystemInspection) -> bool {
    inspection
        .execution_completeness()
        .contains(&CompletenessDefect::NoOutputHead {
            component: COMPONENT.into(),
        })
}

#[test]
fn a_headless_text_model_is_a_closure_and_completeness_defect() {
    let (dir, inspection) = encoded(HeadStorage::Tied, true);
    let outcome = plan_component_ops(&inspection, dir.path(), COMPONENT).unwrap();
    assert!(!outcome.closed(), "a headless text model must not close");
    assert!(outcome.plan.is_none());
    assert!(closure_names_no_head(&inspection, dir.path()));
    assert!(completeness_names_no_head(&inspection));
    let message = ClosureDefect::NoOutputHead {
        component: COMPONENT.into(),
    }
    .to_string();
    assert!(message.contains("cannot produce a logit"), "{message}");
    let message = CompletenessDefect::NoOutputHead {
        component: COMPONENT.into(),
    }
    .to_string();
    assert!(message.contains("cannot produce a logit"), "{message}");
}

#[test]
fn a_tied_head_and_a_separate_head_are_both_complete() {
    for head in [HeadStorage::Tied, HeadStorage::Separate] {
        let (dir, inspection) = encoded(head, false);
        assert!(
            plan_component_ops(&inspection, dir.path(), COMPONENT)
                .unwrap()
                .closed(),
            "{head:?}"
        );
        assert!(!completeness_names_no_head(&inspection), "{head:?}");
    }
}

#[test]
fn a_separate_head_needs_no_embedding_reuse() {
    // `head_reuses_embedding: false` is the normal declaration for a model
    // that ships its own head tensor.
    let (dir, inspection) = encoded(HeadStorage::Separate, true);
    assert!(!closure_names_no_head(&inspection, dir.path()));
    assert!(!completeness_names_no_head(&inspection));
}
