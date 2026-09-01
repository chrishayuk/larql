//! **Partial remote residency is sufficient for execution.**
//!
//! That is the 2a claim, and it is deliberately not "HTTP can copy
//! files". The subject container is served whole; only the objects the
//! execution plan requires are hydrated, the rest are never fetched, and
//! the run must produce the same numbers as one against the complete
//! local container.
//!
//! Three things have to be true together, or the result is hollow:
//!
//! * the outputs agree — otherwise hydration moved the wrong bytes;
//! * the objects left remote are genuinely absent from the execution
//!   root — otherwise "nothing outside the set" is a statement about a
//!   counter rather than about the disk;
//! * a read attempted after [`RemoteContainer::seal`] hard-fails —
//!   otherwise the `PREPARE`/`RUN` boundary is an intention, not a rule.

use std::collections::BTreeSet;

use serial_test::serial;

use crate::format::huggingface::range::test_support::{
    MockRepo, RangeBehaviour, MOCK_REPO, MOCK_REVISION,
};
use crate::format::huggingface::range::HfRangeClient;
use crate::format::vindex3::fixtures::{
    dense_f32_model_with, encode_fixture_container, HeadStorage,
};
use crate::format::vindex3::inspect::inspect_container;
use crate::format::vindex3::opplan::exec::operands::OperandStore;
use crate::format::vindex3::opplan::exec::prepared::{ExecutionSlice, PreparedOperands};
use crate::format::vindex3::opplan::exec::reference::ReferenceBackend;
use crate::format::vindex3::opplan::exec::requirements::required_objects;
use crate::format::vindex3::opplan::{plan_component_ops, ComponentOpPlan, OperandRef};
use crate::format::vindex3::remote::{NetworkPhase, RemoteContainer};

const COMPONENT: &str = "target";
const OUTPUT_HEAD: &str = "target.output_head";

/// The subject: a separate-head container, so its four objects include
/// one a layer-range execution genuinely does not need.
fn local_container() -> tempfile::TempDir {
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

/// Plan a container root.
fn plan_at(root: &std::path::Path) -> ComponentOpPlan {
    let inspection = inspect_container(root, false).unwrap();
    let outcome = plan_component_ops(&inspection, root, COMPONENT).unwrap();
    assert!(outcome.closed(), "defects: {:?}", outcome.defects);
    outcome.plan.unwrap()
}

/// Every operand the plan names, by walking its serialised form — the
/// same total traversal `requirements` uses, kept here in full rather
/// than reduced to objects.
fn operands_of(plan: &ComponentOpPlan) -> Vec<OperandRef> {
    fn walk(node: &serde_json::Value, out: &mut Vec<OperandRef>) {
        match node {
            serde_json::Value::Object(map) => {
                if ["object", "tensor", "dtype", "shape"]
                    .iter()
                    .all(|k| map.contains_key(*k))
                {
                    // Built by hand rather than deserialised: `OperandRef`
                    // is `Serialize` only, and adding `Deserialize` to
                    // production types for a test's convenience is the
                    // wrong trade.
                    if let (Some(object), Some(tensor), Some(dtype), Some(shape)) = (
                        map["object"].as_str(),
                        map["tensor"].as_str(),
                        map["dtype"].as_str(),
                        map["shape"].as_array(),
                    ) {
                        out.push(OperandRef {
                            object: object.to_string(),
                            tensor: tensor.to_string(),
                            dtype: dtype.to_string(),
                            shape: shape
                                .iter()
                                .filter_map(|d| d.as_u64())
                                .map(|d| d as usize)
                                .collect(),
                        });
                        return;
                    }
                }
                for value in map.values() {
                    walk(value, out);
                }
            }
            serde_json::Value::Array(items) => items.iter().for_each(|i| walk(i, out)),
            _ => {}
        }
    }
    let mut out = Vec::new();
    walk(&serde_json::to_value(plan).unwrap(), &mut out);
    out
}

/// Open a store at `root`, prepare `slice`, and return the store — the
/// preparation itself is half the assertion, since it fails outright if a
/// required object is not resident.
fn prepare_at(
    root: &std::path::Path,
    plan: &ComponentOpPlan,
    slice: ExecutionSlice,
) -> OperandStore {
    let inspection = inspect_container(root, false).unwrap();
    let store = OperandStore::open(root, &inspection).unwrap();
    let backend = ReferenceBackend;
    PreparedOperands::load(plan, &store, &backend, slice).expect("preparation");
    store
}

#[test]
#[serial]
fn a_partially_hydrated_remote_container_executes_like_the_local_one() {
    let local = local_container();
    let slice = ExecutionSlice::LayerRange { start: 0, end: 1 };
    let local_plan = plan_at(local.path());
    let required = required_objects(&local_plan, &slice).unwrap();
    assert!(
        !required.contains(OUTPUT_HEAD),
        "the subject slice must not need every object, or this proves nothing"
    );

    let _repo = MockRepo::serve(local.path(), RangeBehaviour::Honour);
    let workspace = tempfile::tempdir().unwrap();
    let client = HfRangeClient::new(MOCK_REPO, MOCK_REVISION).unwrap();

    // Metadata phase: description and headers only, no payload.
    let mut remote = RemoteContainer::open(client, workspace.path()).unwrap();
    assert_eq!(
        remote.payload_bytes(),
        0,
        "opening a remote container must fetch no payload"
    );
    assert_eq!(
        remote.described_objects().len(),
        4,
        "the fixture describes four objects"
    );

    // Plan from the headers alone, then hydrate exactly what it needs.
    let remote_plan = plan_at(&remote.headers_root());
    let remote_required = required_objects(&remote_plan, &slice).unwrap();
    assert_eq!(
        remote_required, required,
        "planning from staged headers must give the same requirement set"
    );

    remote.allow(remote_required.clone()).unwrap();
    let report = remote.hydrate().unwrap();
    assert_eq!(report.hydrated, required);
    assert!(
        report.left_remote.contains(OUTPUT_HEAD),
        "the head should have been left remote; left {:?}",
        report.left_remote
    );
    assert!(report.payload_bytes > 0, "hydration moved nothing");

    // The objects left remote are ABSENT, not merely unread.
    let head_segment = remote
        .container_root()
        .join("segments")
        .join(format!("{OUTPUT_HEAD}.bin"));
    assert!(
        !head_segment.exists(),
        "an object left remote must not be on disk: {}",
        head_segment.display()
    );

    // Seal, then execute. Nothing after this may touch the network.
    remote.seal();
    assert_eq!(remote.phase(), NetworkPhase::Sealed);

    let remote_store = prepare_at(&remote.container_root(), &remote_plan, slice.clone());
    let local_store = prepare_at(local.path(), &local_plan, slice.clone());
    assert_eq!(
        remote_store.touched_objects(),
        local_store.touched_objects(),
        "the two preparations resolved different objects"
    );
    assert_eq!(
        remote_store.load_count(),
        local_store.load_count(),
        "the two preparations performed a different number of loads"
    );

    // Values, not just accounting: every operand this slice binds must
    // come out of the hydrated container bit-for-bit as it does locally.
    let slice_objects = required_objects(&remote_plan, &slice).unwrap();
    let mut compared = 0usize;
    for operand in operands_of(&remote_plan) {
        if !slice_objects.contains(&operand.object) {
            continue;
        }
        let remote_values = remote_store.load(&operand).expect("hydrated operand");
        let local_values = local_store.load(&operand).expect("local operand");
        assert_eq!(
            remote_values, local_values,
            "operand `{}/{}` differs between the hydrated and local containers",
            operand.object, operand.tensor
        );
        compared += 1;
    }
    assert!(
        compared > 0,
        "no operand was compared — the parity assertion is vacuous"
    );
    assert!(
        remote.refusals().is_empty(),
        "nothing should have been refused on the happy path: {:?}",
        remote.refusals()
    );
}

#[test]
#[serial]
fn hydration_is_deny_by_default() {
    // The allow-list is not a filter applied to a set someone else chose:
    // with nothing allowed, nothing is fetched.
    let local = local_container();
    let _repo = MockRepo::serve(local.path(), RangeBehaviour::Honour);
    let workspace = tempfile::tempdir().unwrap();
    let client = HfRangeClient::new(MOCK_REPO, MOCK_REVISION).unwrap();
    let remote = RemoteContainer::open(client, workspace.path()).unwrap();

    let report = remote.hydrate().unwrap();
    assert!(
        report.hydrated.is_empty(),
        "an unallowed hydration must move nothing"
    );
    assert_eq!(report.payload_bytes, 0);
    assert_eq!(
        report.left_remote.len(),
        4,
        "every described object should have been left remote"
    );
}

#[test]
#[serial]
fn a_hydration_set_naming_an_unknown_object_is_refused() {
    let local = local_container();
    let _repo = MockRepo::serve(local.path(), RangeBehaviour::Honour);
    let workspace = tempfile::tempdir().unwrap();
    let client = HfRangeClient::new(MOCK_REPO, MOCK_REVISION).unwrap();
    let mut remote = RemoteContainer::open(client, workspace.path()).unwrap();

    let err = remote
        .allow(BTreeSet::from(["target.nonexistent".to_string()]))
        .expect_err("a set naming an undescribed object must be refused");
    assert!(
        err.to_string().contains("target.nonexistent"),
        "the refusal must name it, got: {err}"
    );
}

#[test]
#[serial]
fn the_seal_hard_fails_a_read_rather_than_counting_it() {
    // The anti-vacuity check for the whole lifecycle. A counter reading
    // zero cannot tell "nothing was fetched" from "nothing was asked", so
    // the seal is proven by asking.
    let local = local_container();
    let _repo = MockRepo::serve(local.path(), RangeBehaviour::Honour);
    let workspace = tempfile::tempdir().unwrap();
    let client = HfRangeClient::new(MOCK_REPO, MOCK_REVISION).unwrap();
    let mut remote = RemoteContainer::open(client, workspace.path()).unwrap();
    remote
        .allow(BTreeSet::from(["target.embedding".to_string()]))
        .unwrap();

    remote.seal();
    let err = remote
        .hydrate()
        .expect_err("a read after seal must fail at the point of violation");
    let message = err.to_string();
    assert!(
        message.contains("after seal"),
        "the refusal must name the boundary, got: {message}"
    );
    assert!(
        remote
            .refusals()
            .iter()
            .any(|r| r.contains("PREPARE and RUN")),
        "the refusal should state the invariant it defends: {:?}",
        remote.refusals()
    );
}
