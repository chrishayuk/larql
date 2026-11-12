//! A `.vlp` is published whole or not at all.
//!
//! `VindexPatch::save` used to write in place: it truncated the file at
//! the destination and then wrote into it. A write that failed part-way
//! left a truncated patch where a complete one had been, and anything
//! reading the previous patch through the same file saw it change under
//! it. The save now writes a temporary file and renames it into place.
//!
//! A hard link to the previous file is the observable difference: an
//! in-place write changes the bytes the link reads, and a rename does not.

use std::path::Path;

use larql_vindex::{PatchOp, VindexPatch};

const PATCH_FILE: &str = "edits.vlp";
const OLD_LINK: &str = "previous.vlp";
const OLD_DESCRIPTION: &str = "the previous patch";
const NEW_DESCRIPTION: &str = "the new patch";

fn patch(description: &str, operations: usize) -> VindexPatch {
    VindexPatch {
        version: 1,
        base_model: "test/patch-save".into(),
        base_checksum: None,
        created_at: "2026-11-11".into(),
        description: Some(description.into()),
        author: None,
        tags: Vec::new(),
        operations: (0..operations)
            .map(|feature| PatchOp::Delete {
                layer: 0,
                feature,
                reason: None,
            })
            .collect(),
    }
}

fn files_in(dir: &Path) -> Vec<String> {
    let mut names: Vec<String> = std::fs::read_dir(dir)
        .expect("read dir")
        .map(|entry| {
            entry
                .expect("entry")
                .file_name()
                .to_string_lossy()
                .into_owned()
        })
        .collect();
    names.sort();
    names
}

#[test]
fn a_save_over_an_existing_patch_does_not_write_into_the_old_file() {
    let dir = tempfile::tempdir().expect("tempdir");
    let path = dir.path().join(PATCH_FILE);
    patch(OLD_DESCRIPTION, 1).save(&path).expect("first save");
    let old_bytes = std::fs::read(&path).expect("old bytes");
    let link = dir.path().join(OLD_LINK);
    std::fs::hard_link(&path, &link).expect("link the previous file");

    patch(NEW_DESCRIPTION, 3).save(&path).expect("second save");

    assert_eq!(
        std::fs::read(&link).expect("previous file"),
        old_bytes,
        "the previous file's bytes changed: the save wrote in place"
    );
    let saved = VindexPatch::load(&path).expect("load the new patch");
    assert_eq!(saved.description.as_deref(), Some(NEW_DESCRIPTION));
    assert_eq!(saved.operations.len(), 3);
}

#[test]
fn a_save_leaves_only_the_patch_behind() {
    let dir = tempfile::tempdir().expect("tempdir");
    let path = dir.path().join(PATCH_FILE);

    patch(NEW_DESCRIPTION, 2).save(&path).expect("save");
    patch(NEW_DESCRIPTION, 2).save(&path).expect("save again");

    assert_eq!(files_in(dir.path()), [PATCH_FILE]);
}

#[test]
fn a_save_that_cannot_publish_leaves_the_destination_and_no_temporary_file() {
    // The destination is a non-empty directory, so the rename fails after
    // the temporary file has been written in full.
    const KEPT: &str = "kept.txt";
    let dir = tempfile::tempdir().expect("tempdir");
    let path = dir.path().join(PATCH_FILE);
    std::fs::create_dir(&path).expect("a directory at the destination");
    std::fs::write(path.join(KEPT), b"kept").expect("a file inside it");

    patch(NEW_DESCRIPTION, 1)
        .save(&path)
        .expect_err("a directory cannot be replaced by a patch");

    assert_eq!(files_in(dir.path()), [PATCH_FILE]);
    assert_eq!(std::fs::read(path.join(KEPT)).expect("kept"), b"kept");
}

#[test]
fn a_save_into_a_missing_directory_fails_and_creates_nothing() {
    let dir = tempfile::tempdir().expect("tempdir");
    let path = dir.path().join("missing").join(PATCH_FILE);

    patch(NEW_DESCRIPTION, 1)
        .save(&path)
        .expect_err("the parent directory does not exist");

    assert!(files_in(dir.path()).is_empty());
}

#[test]
fn a_path_that_names_no_file_is_refused() {
    let err = patch(NEW_DESCRIPTION, 1)
        .save(Path::new("/"))
        .expect_err("a root path names no file");

    assert!(err.to_string().contains("names no file"), "{err}");
}
