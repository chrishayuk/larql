//! **Real encoded containers, for tests that read a container's own
//! authority.**
//!
//! One fixture rather than one per test file: the seal tests and the
//! identity-construction tests read the same `index.json` and would
//! otherwise keep two ideas of what a container looks like.

use std::path::Path;

use crate::format::filenames::INDEX_JSON;
use crate::format::vindex3::encode::encode_system_unenforced;
use crate::format::vindex3::plan::tests_support::{
    drafter_shaped, glimmer_shaped_target, known_dense,
};

/// A container with ONE representation — the narrowest fixture, for
/// tests that move a single field and want nothing else in the way.
///
/// The source tempdir is dropped on return: the encode has already
/// copied every byte it needs, and what the tests read afterwards is
/// the container alone.
pub(super) fn dense() -> tempfile::TempDir {
    let source = tempfile::tempdir().expect("source dir");
    let named = vec![("only-artifact".to_string(), known_dense(source.path()))];
    encode(named)
}

/// A container with EIGHT representations across two artifacts.
///
/// A one-representation container cannot tell "refused" from "returned
/// an identity over what was left", because nothing is left. This can.
pub(super) fn glimmer() -> tempfile::TempDir {
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
pub(super) fn read_index(container: &Path) -> serde_json::Value {
    serde_json::from_str(&std::fs::read_to_string(container.join(INDEX_JSON)).expect("index"))
        .expect("index is JSON")
}

pub(super) fn write_index(container: &Path, index: &serde_json::Value) {
    std::fs::write(
        container.join(INDEX_JSON),
        serde_json::to_string_pretty(index).expect("index"),
    )
    .expect("rewrite index");
}

/// Rewrite the index through `edit` and return the container's path.
pub(super) fn with_index(container: &Path, edit: impl FnOnce(&mut serde_json::Value)) {
    let mut index = read_index(container);
    edit(&mut index);
    write_index(container, &index);
}

/// Rewrite the index through this harness's serialiser, changing no
/// value.
///
/// `manifest_hash` digests the file's BYTES, so a test that edits the
/// index must put the baseline into the same serialisation first or it
/// measures pretty-printing rather than the property under test. See
/// `source_seal::reserialising_the_index_alone_moves_the_state_id`.
pub(super) fn reserialise(container: &Path) {
    let index = read_index(container);
    write_index(container, &index);
}

/// The id of some representation entry, for tests that need to name one.
pub(super) fn a_representation(index: &serde_json::Value) -> String {
    index["representations"]
        .as_object()
        .expect("representations")
        .keys()
        .next()
        .expect("the fixture writes at least one")
        .clone()
}
