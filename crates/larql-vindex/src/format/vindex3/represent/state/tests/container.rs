//! **Real encoded containers, and the harness that restates them.**
//!
//! One fixture rather than one per test file: the seal tests, the
//! semantic-identity tests and the identity-construction tests read the
//! same `index.json` and would otherwise keep three ideas of what a
//! container looks like — and three harnesses that could disagree about
//! what "the same container, restated" means.

use std::io::Read;
use std::path::Path;

use sha2::{Digest, Sha256};

use super::super::super::compiler::{read_source_identity, SourceIdentity};
use super::super::super::map::PrecisionMap;
use super::super::super::policy::Role;
use super::super::identity::{RepresentationState, RepresentationStateId};
use super::super::resolved::NoLayoutConstraint;
use super::super::surface::{SurfaceTensor, TensorSurface};
use crate::format::filenames::INDEX_JSON;
use crate::format::vindex3::encode::encode_system_unenforced;
use crate::format::vindex3::encode::segment::{SegmentHeader, SEGMENT_PAYLOAD_ALIGN};
use crate::format::vindex3::plan::tests_support::{
    drafter_shaped, glimmer_shaped_target, known_dense,
};

/// A container with ONE representation — the narrowest fixture, for
/// tests that move a single field and want nothing else in the way.
///
/// The source tempdir is dropped on return: the encode has already
/// copied every byte it needs, and what the tests read afterwards is
/// the container alone.
pub(crate) fn dense() -> tempfile::TempDir {
    let source = tempfile::tempdir().expect("source dir");
    let named = vec![("only-artifact".to_string(), known_dense(source.path()))];
    encode(named)
}

/// A container with EIGHT representations across two artifacts.
///
/// A one-representation container cannot tell "refused" from "returned
/// an identity over what was left", because nothing is left. This can.
pub(crate) fn glimmer() -> tempfile::TempDir {
    let target = tempfile::tempdir().expect("target dir");
    let drafter = tempfile::tempdir().expect("drafter dir");
    let named = vec![
        (
            "target-artifact".to_string(),
            glimmer_shaped_target(target.path()),
        ),
        (
            "drafter-artifact".to_string(),
            drafter_shaped(drafter.path()),
        ),
    ];
    encode(named)
}

fn encode(
    named: Vec<(String, larql_models::inventory::ArchitectureInventory)>,
) -> tempfile::TempDir {
    let out = tempfile::tempdir().expect("container dir");
    encode_system_unenforced(&named, out.path()).expect("encode");
    out
}

/// The container's index, as a document.
pub(crate) fn read_index(container: &Path) -> serde_json::Value {
    serde_json::from_str(&std::fs::read_to_string(container.join(INDEX_JSON)).expect("index"))
        .expect("index is JSON")
}

pub(crate) fn write_index(container: &Path, index: &serde_json::Value) {
    std::fs::write(
        container.join(INDEX_JSON),
        serde_json::to_string_pretty(index).expect("index"),
    )
    .expect("rewrite index");
}

/// Rewrite the index through `edit` and return the container's path.
pub(crate) fn with_index(container: &Path, edit: impl FnOnce(&mut serde_json::Value)) {
    let mut index = read_index(container);
    edit(&mut index);
    write_index(container, &index);
}

/// Rewrite the index through this harness's serialiser, changing no
/// value.
///
/// Two tests need this and want opposite things from it.
/// `source_semantics::three_siblings…` uses it to MAKE a sibling that
/// differs only in serialisation; `restate_table` uses it to remove
/// that difference before taking a baseline, so a comparison of the
/// exported bytes measures the edit and not this harness's
/// pretty-printer.
pub(crate) fn reserialise(container: &Path) {
    let index = read_index(container);
    write_index(container, &index);
}

/// The id of some representation entry, for tests that need to name one.
pub(crate) fn a_representation(index: &serde_json::Value) -> String {
    index["representations"]
        .as_object()
        .expect("representations")
        .keys()
        .next()
        .expect("the fixture writes at least one")
        .clone()
}

/// A container's identity before and after a restatement of its table.
pub(crate) struct Restated {
    pub(crate) before: SourceIdentity,
    pub(crate) after: SourceIdentity,
    /// Every `index.json` field whose value changed, as a dotted path.
    pub(crate) index_changes: Vec<String>,
}

/// **Rewrite one segment's tensor table, copying the payload verbatim.**
///
/// Models an honest writer: the table changes, the file digest is
/// recomputed, and `index.json` records the new one. What it must NOT
/// do is disturb the payload — asserted here rather than assumed, since
/// a harness that quietly moved a payload byte would make every test
/// below pass for the wrong reason.
pub(crate) fn restate_table(root: &Path, edit: impl FnOnce(&mut SegmentHeader)) -> Restated {
    // Put the index into this harness's own serialisation BEFORE the
    // baseline is taken. The SEMANTIC identity is blind to
    // serialisation, but `index_changes` and the artifact digest are
    // not, and without this the control below would report the
    // pretty-printer rather than the edit.
    reserialise(root);
    let before = read_source_identity(root).expect("a container identity");
    let index_before = read_index(root);

    let (id, entry) = index_before["representations"]
        .as_object()
        .expect("representations")
        .iter()
        .next()
        .expect("the fixture writes one representation");
    let segment_path = root.join(entry["segment"].as_str().expect("a segment path"));

    // Split the file into its three parts.
    let mut file = std::fs::File::open(&segment_path).expect("open segment");
    let mut length = [0u8; 8];
    file.read_exact(&mut length).expect("length prefix");
    let mut header_bytes = vec![0u8; u64::from_le_bytes(length) as usize];
    file.read_exact(&mut header_bytes).expect("header");
    let mut payload = Vec::new();
    file.read_to_end(&mut payload).expect("payload");
    let payload_before = format!("{:x}", Sha256::digest(&payload));

    let mut header: SegmentHeader = serde_json::from_slice(&header_bytes).expect("header is JSON");
    edit(&mut header);

    // Re-serialise under the writer's own padding rule, so a no-op edit
    // reproduces the file byte for byte and the control test means
    // something.
    let mut restated = serde_json::to_vec(&header).expect("serialise header");
    let pad = (SEGMENT_PAYLOAD_ALIGN - (8 + restated.len()) % SEGMENT_PAYLOAD_ALIGN)
        % SEGMENT_PAYLOAD_ALIGN;
    restated.extend(std::iter::repeat_n(b' ', pad));

    let mut written = Vec::new();
    written.extend_from_slice(&(restated.len() as u64).to_le_bytes());
    written.extend_from_slice(&restated);
    written.extend_from_slice(&payload);
    std::fs::write(&segment_path, &written).expect("rewrite segment");

    let payload_after = {
        let mut file = std::fs::File::open(&segment_path).expect("reopen");
        let mut length = [0u8; 8];
        file.read_exact(&mut length).expect("length");
        let mut skip = vec![0u8; u64::from_le_bytes(length) as usize];
        file.read_exact(&mut skip).expect("header");
        let mut payload = Vec::new();
        file.read_to_end(&mut payload).expect("payload");
        format!("{:x}", Sha256::digest(&payload))
    };
    assert_eq!(
        payload_before, payload_after,
        "the harness moved a payload byte; every assertion below would pass for the wrong reason"
    );

    let mut index_after = index_before.clone();
    index_after["representations"][id]["segment_sha256"] =
        serde_json::json!(format!("{:x}", Sha256::digest(&written)));
    write_index(root, &index_after);

    Restated {
        after: read_source_identity(root).expect("a container identity"),
        before,
        index_changes: changed_paths(&index_before, &index_after),
    }
}

/// Every leaf whose value differs between two JSON documents.
fn changed_paths(before: &serde_json::Value, after: &serde_json::Value) -> Vec<String> {
    let mut changes = Vec::new();
    walk(before, after, String::new(), &mut changes);
    changes.sort();
    changes
}

fn walk(a: &serde_json::Value, b: &serde_json::Value, path: String, out: &mut Vec<String>) {
    match (a, b) {
        (serde_json::Value::Object(a), serde_json::Value::Object(b)) => {
            for key in a
                .keys()
                .chain(b.keys())
                .collect::<std::collections::BTreeSet<_>>()
            {
                let child = match path.is_empty() {
                    true => key.clone(),
                    false => format!("{path}.{key}"),
                };
                match (a.get(key), b.get(key)) {
                    (Some(a), Some(b)) => walk(a, b, child, out),
                    _ => out.push(child),
                }
            }
        }
        _ if a != b => out.push(path),
        _ => {}
    }
}

/// The same map, surface and layout under two container identities.
pub(crate) fn state_id(model: &SourceIdentity) -> (RepresentationStateId, String) {
    let surface = TensorSurface::new([SurfaceTensor::new(
        "target.embedding",
        "weight",
        Role::Embedding,
        vec![128, 64],
    )])
    .expect("one tensor");
    let map = PrecisionMap {
        name: "m".into(),
        encoding: "BF16".into(),
        roles: vec!["embedding".into()],
        exceptions: Vec::new(),
    };
    let resolved = RepresentationState::resolve(model, &surface, &map, &NoLayoutConstraint);
    let decisions = resolved.decisions().canonical_full();
    (resolved.id().clone(), decisions)
}

/// A byte-for-byte copy of a container, so siblings diverge from one
/// export rather than from two encodes that might differ for reasons
/// nobody chose.
pub(crate) fn sibling(container: &Path) -> tempfile::TempDir {
    let out = tempfile::tempdir().expect("sibling dir");
    copy_into(container, out.path());
    out
}

fn copy_into(from: &Path, to: &Path) {
    for entry in std::fs::read_dir(from).expect("read container") {
        let entry = entry.expect("entry");
        let target = to.join(entry.file_name());
        match entry.file_type().expect("file type").is_dir() {
            true => {
                std::fs::create_dir_all(&target).expect("dir");
                copy_into(&entry.path(), &target);
            }
            false => {
                std::fs::copy(entry.path(), &target).expect("copy");
            }
        }
    }
}
