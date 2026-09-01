//! Step-0 gates on the opener: one opening authority with a policy
//! axis. Default preservation (the canonical program, exactly as the
//! store opens it), policy parity (the stored NVFP4 pack, exactly as
//! `larql vindex3 exec --representation-source stored` binds it), the
//! stored-only refusal, and the container's declared identity.

use std::path::PathBuf;

use larql_vindex::format::filenames::INDEX_JSON;
use larql_vindex::format::vindex3::fixtures::{dense_f32_model, encode_fixture_container};
use larql_vindex::format::vindex3::inspect::inspect_container;
use larql_vindex::format::vindex3::opplan::exec::operands::{OperandStore, RepresentationSource};
use larql_vindex::format::vindex3::opplan::exec::prepared::{ExecutionSlice, PreparedOperands};
use larql_vindex::format::vindex3::opplan::exec::production::ProductionBackend;
use larql_vindex::format::vindex3::opplan::exec::reference::ReferenceBackend;
use larql_vindex::format::vindex3::represent::nvfp4_pack::DTYPE_NVFP4;
use larql_vindex::format::vindex3::represent::{compile_representation, RepresentSpec};

use super::super::{open_component, OpenPolicy, Vindex3Runtime};
use super::{container_with, COMPONENT};

/// Artifact name the compiled pair encodes under — deliberately unlike
/// any directory name, so identity tests cannot pass by coincidence.
const ARTIFACT: &str = "opener-fixture";

/// The dense fixture encoded, then compiled into a second container that
/// carries an NVFP4 pack — the exec verb's `--representation-source
/// stored` subject.
fn compiled_pair(tmp: &tempfile::TempDir) -> (PathBuf, PathBuf) {
    let checkpoint = tmp.path().join("ckpt");
    std::fs::create_dir_all(&checkpoint).unwrap();
    let src = tmp.path().join("src.vindex3");
    let out = tmp.path().join("nvfp4.vindex3");
    encode_fixture_container(dense_f32_model, &checkpoint, &src, ARTIFACT);
    compile_representation(&src, &out, &RepresentSpec::nvfp4())
        .expect("the dense fixture is 16-aligned throughout");
    (src, out)
}

/// What the exec verb asks for under `--representation-source stored`.
fn stored_nvfp4() -> OpenPolicy {
    OpenPolicy {
        want: Some(DTYPE_NVFP4.to_string()),
        source: RepresentationSource::Stored,
    }
}

#[test]
fn the_default_policy_binds_the_canonical_program_the_store_opens() {
    let container = container_with(dense_f32_model);
    let opened = open_component(container.path(), COMPONENT, OpenPolicy::default()).unwrap();
    let inspection = inspect_container(container.path(), false).unwrap();
    let canonical = OperandStore::open(container.path(), &inspection).unwrap();
    assert_eq!(opened.store.selection(), canonical.selection());
    assert!(opened.want.is_none());
    assert!(
        opened.store.selection().values().all(|s| !s.stored),
        "the canonical program never binds a pack"
    );
}

#[test]
fn open_is_open_with_under_the_default_policy() {
    let container = container_with(dense_f32_model);
    let plain = Vindex3Runtime::open(container.path(), COMPONENT, ReferenceBackend::new()).unwrap();
    let explicit = Vindex3Runtime::open_with(
        container.path(),
        COMPONENT,
        ReferenceBackend::new(),
        OpenPolicy::default(),
    )
    .unwrap();
    assert_eq!(plain.plan(), explicit.plan());
    assert_eq!(plain.model_name(), explicit.model_name());
    assert_eq!(plain.family(), explicit.family());
}

#[test]
fn the_stored_policy_binds_the_compiled_pack_the_exec_verb_binds() {
    let tmp = tempfile::tempdir().unwrap();
    let (_src, out) = compiled_pair(&tmp);
    let opened = open_component(&out, COMPONENT, stored_nvfp4()).unwrap();
    // The exec verb's own recipe on the same container.
    let inspection = inspect_container(&out, false).unwrap();
    let exec_store = OperandStore::open_for(
        &out,
        &inspection,
        Some(DTYPE_NVFP4),
        RepresentationSource::Stored,
    )
    .unwrap();
    assert_eq!(opened.store.selection(), exec_store.selection());
    assert_eq!(opened.want.as_deref(), Some(DTYPE_NVFP4));
    let from_pack = opened
        .store
        .selection()
        .values()
        .filter(|s| s.stored)
        .count();
    assert!(from_pack > 0, "{:?}", opened.store.selection());
    // Served entirely from stored bytes: lowering the operands quantises
    // nothing — the line the exec verb reports as `runtime compile: 0`.
    PreparedOperands::load(
        &opened.plan,
        &opened.store,
        &ProductionBackend::new(),
        ExecutionSlice::Full,
    )
    .unwrap();
    assert_eq!(opened.store.runtime_quantised(), 0);
}

#[test]
fn the_stored_policy_manufactures_nothing_when_the_pack_is_missing() {
    let tmp = tempfile::tempdir().unwrap();
    let (src, _out) = compiled_pair(&tmp);
    // The SOURCE container has no pack. The invariant is about work, not
    // coverage: the ask is recorded, nothing is bound from a pack, and
    // lowering the operands quantises nothing — a tensor the policy
    // cannot serve compiled runs at its stored precision instead.
    let opened = open_component(&src, COMPONENT, stored_nvfp4()).unwrap();
    assert_eq!(opened.want.as_deref(), Some(DTYPE_NVFP4));
    assert!(opened.store.selection().values().all(|s| !s.stored));
    PreparedOperands::load(
        &opened.plan,
        &opened.store,
        &ProductionBackend::new(),
        ExecutionSlice::Full,
    )
    .expect("a stored-only policy binds what is there rather than refusing the container");
    assert_eq!(
        opened.store.runtime_quantised(),
        0,
        "stored-only must never quantise at load"
    );
}

#[test]
fn the_opened_component_carries_the_containers_declared_identity() {
    let tmp = tempfile::tempdir().unwrap();
    let (src, _out) = compiled_pair(&tmp);
    let opened = open_component(&src, COMPONENT, OpenPolicy::default()).unwrap();
    let index: serde_json::Value =
        serde_json::from_str(&std::fs::read_to_string(src.join(INDEX_JSON)).unwrap()).unwrap();
    assert_eq!(opened.model_name, index["model"].as_str().unwrap());
    assert_eq!(opened.family, index["family"].as_str().unwrap());
    assert!(!opened.model_name.is_empty());
    // Identity travels with the artifact, never with the directory.
    let directory = src.file_name().unwrap().to_str().unwrap();
    assert_ne!(opened.model_name, directory);
}

#[test]
fn a_container_missing_a_segment_refuses_to_open() {
    let container = container_with(dense_f32_model);
    // Inspection reads the index and the graph; only binding the operand
    // store touches the payload. Remove one segment so the plan closes
    // and the store cannot open — the refusal must come from the opener,
    // not from a later step's surprise.
    let segments = container.path().join("segments");
    let victim = std::fs::read_dir(&segments)
        .unwrap()
        .map(|e| e.unwrap().path())
        .find(|p| p.extension().is_some_and(|ext| ext == "bin"))
        .expect("the encoded fixture has at least one segment");
    std::fs::remove_file(&victim).unwrap();
    let err = open_component(container.path(), COMPONENT, OpenPolicy::default())
        .err()
        .map(|e| e.to_string())
        .expect("a container that cannot bind its operands must refuse to open");
    assert!(!err.is_empty());
}
