//! The generation gate on the inspect path.
//!
//! `inspect_container` is the opening step for every graph-path consumer,
//! so it must refuse by name before reading any other byte: a VINDEX2
//! directory, a schema this build does not read, and an index with no
//! version at all are three different refusals, and none of them may be
//! reported as a parse error in something else.

use super::*;
use crate::format::generation::{V2_CURRENT_SCHEMA, V3_CURRENT_SCHEMA};
use crate::format::vindex3::fixtures::{encode_fixture_container, miniature_glimmer};

fn fixture() -> tempfile::TempDir {
    let checkpoint = tempfile::tempdir().unwrap();
    let container = tempfile::tempdir().unwrap();
    encode_fixture_container(
        miniature_glimmer,
        checkpoint.path(),
        container.path(),
        "inspect-fixture",
    );
    container
}

fn rewrite_version(root: &Path, version: Option<u64>) {
    let path = root.join(INDEX_JSON);
    let mut index: serde_json::Value =
        serde_json::from_str(&std::fs::read_to_string(&path).unwrap()).unwrap();
    match version {
        Some(v) => {
            index["version"] = serde_json::json!(v);
        }
        None => {
            index.as_object_mut().unwrap().remove("version");
        }
    }
    std::fs::write(&path, serde_json::to_string_pretty(&index).unwrap()).unwrap();
}

#[test]
fn the_current_schema_inspects() {
    let container = fixture();
    let inspection = inspect_container(container.path(), false).expect("current schema inspects");
    assert!(inspection.defects.is_empty(), "{:?}", inspection.defects);
}

#[test]
fn a_future_schema_is_refused_by_version_before_anything_else_is_read() {
    let container = fixture();
    let future = u64::from(V3_CURRENT_SCHEMA) + 1;
    rewrite_version(container.path(), Some(future));
    let err = inspect_container(container.path(), false).expect_err("a future schema must refuse");
    assert!(
        matches!(&err, VindexError::UnknownContainerGeneration { found, .. } if u64::from(*found) == future),
        "refusal must name the version it found: {err}"
    );
    assert!(err.to_string().contains(&future.to_string()), "{err}");
}

#[test]
fn a_vindex2_index_is_refused_as_the_wrong_generation() {
    let container = fixture();
    rewrite_version(container.path(), Some(u64::from(V2_CURRENT_SCHEMA)));
    let err = inspect_container(container.path(), false).expect_err("a VINDEX2 index must refuse");
    assert!(
        matches!(
            &err,
            VindexError::WrongContainerGeneration {
                found: "VINDEX2",
                required: "VINDEX3"
            }
        ),
        "{err}"
    );
}

#[test]
fn an_index_with_no_version_is_refused_not_assumed_current() {
    let container = fixture();
    rewrite_version(container.path(), None);
    let err = inspect_container(container.path(), false).expect_err("no version, no inspection");
    assert!(
        matches!(
            &err,
            VindexError::UnknownContainerGeneration { found: 0, .. }
        ),
        "{err}"
    );
}
