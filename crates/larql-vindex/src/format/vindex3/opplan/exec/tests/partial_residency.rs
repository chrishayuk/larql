//! **A container can be fully described and only partly resident.**
//!
//! This is the separation the whole remote line rests on, expressed at
//! the operand store: `index.json` and `system_graph.json` say what the
//! model *is*, and that description is complete whether or not the bytes
//! are here yet. Residency is a different question, asked per object.
//!
//! Before this, [`OperandStore::open`] read every object's segment header
//! eagerly and propagated the error, so a container missing one segment
//! could not be opened at all — which makes hydrating a *subset*
//! impossible, and a subset is the entire point of a hydration set.
//!
//! An absent segment is therefore recorded as absent rather than refused.
//! The refusal moves to the load path, where it can name the object and
//! say the true thing: this object is described, and its bytes are not
//! here. A container that is genuinely corrupt still fails — a segment
//! that exists but cannot be read is an error at open, as it always was.

use crate::format::vindex3::fixtures::{
    dense_f32_model_with, encode_fixture_container, HeadStorage,
};
use crate::format::vindex3::inspect::inspect_container;
use crate::format::vindex3::opplan::exec::operands::OperandStore;
use crate::format::vindex3::opplan::exec::prepared::{ExecutionSlice, PreparedOperands};
use crate::format::vindex3::opplan::exec::reference::ReferenceBackend;
use crate::format::vindex3::opplan::exec::requirements::required_objects;
use crate::format::vindex3::opplan::plan_component_ops;

const COMPONENT: &str = "target";
const OUTPUT_HEAD: &str = "target.output_head";

/// A separate-head container: four objects, of which the head is the one
/// an execution can be made not to need.
fn container() -> tempfile::TempDir {
    let checkpoint = tempfile::tempdir().unwrap();
    let container = tempfile::tempdir().unwrap();
    encode_fixture_container(
        |dir| dense_f32_model_with(dir, HeadStorage::Separate),
        checkpoint.path(),
        container.path(),
        COMPONENT,
    );
    container
}

/// Remove one object's segment, leaving the description intact — what a
/// hydration that left an object behind produces.
fn evict(root: &std::path::Path, object: &str) {
    let segment = root.join("segments").join(format!("{object}.bin"));
    assert!(segment.exists(), "fixture should have written {object}");
    std::fs::remove_file(&segment).unwrap();
}

#[test]
fn an_absent_segment_does_not_prevent_opening() {
    let container = container();
    evict(container.path(), OUTPUT_HEAD);
    let inspection = inspect_container(container.path(), false).unwrap();

    let store = OperandStore::open(container.path(), &inspection)
        .expect("a partly resident container must still open");

    // The description is unchanged: the graph still carries the object.
    assert!(
        inspection.graph.objects.iter().any(|o| o.id == OUTPUT_HEAD),
        "evicting bytes must not change what the container describes"
    );
    assert!(
        !store.is_resident(OUTPUT_HEAD),
        "the evicted object should be reported as not resident"
    );
    assert!(
        store.is_resident("target.embedding"),
        "an untouched object should still be resident"
    );
}

#[test]
fn loading_from_an_absent_segment_fails_naming_residency() {
    let container = container();
    let inspection = inspect_container(container.path(), false).unwrap();
    let outcome = plan_component_ops(&inspection, container.path(), COMPONENT).unwrap();
    let plan = outcome.plan.unwrap();
    let head = plan.output.as_ref().unwrap().projection.clone();
    assert_eq!(head.object, OUTPUT_HEAD, "fixture should have its own head");

    evict(container.path(), OUTPUT_HEAD);
    let inspection = inspect_container(container.path(), false).unwrap();
    let store = OperandStore::open(container.path(), &inspection).unwrap();

    let err = store
        .load(&head)
        .expect_err("an operand whose object is not resident must refuse");
    let message = err.to_string();
    assert!(
        message.contains(OUTPUT_HEAD),
        "the refusal must name the object, got: {message}"
    );
    assert!(
        message.contains("not resident"),
        "the refusal must say the bytes are absent rather than imply the \
         container is malformed, got: {message}"
    );
}

#[test]
fn a_slice_that_does_not_need_an_object_prepares_without_it() {
    // The property hydration depends on: an execution whose requirement
    // set excludes an object must succeed with that object's bytes
    // genuinely absent from disk — not merely unread.
    let container = container();
    let inspection = inspect_container(container.path(), false).unwrap();
    let outcome = plan_component_ops(&inspection, container.path(), COMPONENT).unwrap();
    let plan = outcome.plan.unwrap();

    let slice = ExecutionSlice::LayerRange { start: 0, end: 1 };
    let required = required_objects(&plan, &slice).unwrap();
    assert!(
        !required.contains(OUTPUT_HEAD),
        "a layer range should not require the output head; required {required:?}"
    );

    evict(container.path(), OUTPUT_HEAD);
    let inspection = inspect_container(container.path(), false).unwrap();
    let store = OperandStore::open(container.path(), &inspection).unwrap();
    let backend = ReferenceBackend;
    PreparedOperands::load(&plan, &store, &backend, slice)
        .expect("preparing a slice must not need an object it does not require");

    assert_eq!(
        store.touched_objects(),
        required,
        "preparation touched something other than its requirement set"
    );
}
