//! Gate B, B6 §3.4: COMPILE and COMPACT INTO hard-link the files they do
//! not change, so the output and the source share inodes. Nothing a
//! session does with the output afterwards may write a shared inode in
//! place, or the source changes with it.
//!
//! Each test compiles (or compacts) a source, checks that the output really
//! does share at least one inode with it (otherwise the test would show
//! nothing), then binds the output and runs every LQL statement that
//! writes: edits, patch recording and saving, APPLY and REMOVE PATCH, and
//! COMPILE and COMPACT INTO from the output. Every source file's bytes
//! must be unchanged, and so must every output file that shares an inode
//! with the source.

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

use crate::support::*;

/// The relative paths of the regular files under `out` that are the same
/// file (same device and inode) as the file at the same path under
/// `source`.
#[cfg(unix)]
fn shared_files(source: &Path, out: &Path) -> Vec<String> {
    use std::os::unix::fs::MetadataExt;
    tree(out)
        .into_iter()
        .filter(|(_, entry)| matches!(entry, Entry::File(_)))
        .filter_map(|(rel, _)| {
            let a = std::fs::metadata(source.join(&rel)).ok()?;
            let b = std::fs::metadata(out.join(&rel)).ok()?;
            (a.dev() == b.dev() && a.ino() == b.ino()).then_some(rel)
        })
        .collect()
}

/// The bytes of `files` under `dir`.
fn bytes_of(dir: &Path, files: &[String]) -> BTreeMap<String, Vec<u8>> {
    files
        .iter()
        .map(|rel| (rel.clone(), std::fs::read(dir.join(rel)).unwrap()))
        .collect()
}

/// Bind `output` and run every writing statement against it, writing new
/// files only under `scratch`.
fn exercise_writers(output: &Path, scratch: &Path, edits: &[&str], compact: bool) {
    std::fs::create_dir_all(scratch).unwrap();
    let saved = scratch.join("edits.vlp");
    let mut session = bound(output);
    run(
        &mut session,
        &format!("BEGIN PATCH \"{}\";", lql_path(&saved)),
    );
    for edit in edits {
        run(&mut session, edit);
    }
    run(&mut session, "SAVE PATCH;");
    run(
        &mut session,
        &format!(
            "COMPILE CURRENT INTO VINDEX \"{}\";",
            lql_path(scratch.join("again.out"))
        ),
    );

    // A second session applies and removes the saved patch, then compiles.
    let mut other = bound(output);
    run(
        &mut other,
        &format!("APPLY PATCH \"{}\";", lql_path(&saved)),
    );
    run(
        &mut other,
        &format!("REMOVE PATCH \"{}\";", lql_path(&saved)),
    );
    run(
        &mut other,
        &format!(
            "COMPILE CURRENT INTO VINDEX \"{}\";",
            lql_path(scratch.join("removed.out"))
        ),
    );

    if compact {
        let mut fresh = bound(output);
        run(
            &mut fresh,
            &format!(
                "COMPACT INTO VINDEX \"{}\";",
                lql_path(scratch.join("compacted.out"))
            ),
        );
    }
}

/// Produce `output` from `source` with `make`, then run the writers on it.
fn assert_writers_leave_shared_inodes_alone(
    source: &Path,
    output: &Path,
    scratch: PathBuf,
    edits: &[&str],
    compact: bool,
) {
    let source_before = tree(source);
    #[cfg(unix)]
    let shared = {
        let shared = shared_files(source, output);
        assert!(
            !shared.is_empty(),
            "the output shares no inode with the source, so this test would show nothing"
        );
        shared
    };
    #[cfg(not(unix))]
    let shared: Vec<String> = Vec::new();
    let shared_before = bytes_of(output, &shared);

    exercise_writers(output, &scratch, edits, compact);

    assert!(tree(source) == source_before, "the source changed");
    assert_eq!(
        bytes_of(output, &shared),
        shared_before,
        "a file the output shares with the source changed"
    );
}

#[test]
fn vindex2_writers_on_a_compiled_output_leave_the_source_alone() {
    let dir = tempfile::tempdir().unwrap();
    let source = v2_vindex(dir.path());
    let output = dir.path().join("compiled.out");
    run(
        &mut bound(&source),
        &format!("COMPILE CURRENT INTO VINDEX \"{}\";", lql_path(&output)),
    );
    assert_writers_leave_shared_inodes_alone(
        &source,
        &output,
        dir.path().join("scratch"),
        &[
            V2_INSERT,
            V2_KNN[1],
            "DELETE FROM EDGES WHERE layer = 0 AND feature = 1;",
        ],
        false,
    );
}

#[test]
fn vindex3_writers_on_a_compiled_output_leave_the_source_alone() {
    let dir = tempfile::tempdir().unwrap();
    let source = v3_container(dir.path());
    let output = dir.path().join("compiled.out");
    let mut session = bound(&source);
    run(&mut session, V3_KNN[2]);
    run(
        &mut session,
        &format!("COMPILE CURRENT INTO VINDEX \"{}\";", lql_path(&output)),
    );
    assert_writers_leave_shared_inodes_alone(
        &source,
        &output,
        dir.path().join("scratch"),
        &[V3_INSERT, V3_KNN[1]],
        true,
    );
}

#[test]
fn vindex3_writers_on_a_compacted_output_leave_the_source_alone() {
    let dir = tempfile::tempdir().unwrap();
    let source = v3_container(dir.path());
    let output = dir.path().join("compacted.out");
    run(
        &mut bound(&source),
        &format!("COMPACT INTO VINDEX \"{}\";", lql_path(&output)),
    );
    assert_writers_leave_shared_inodes_alone(
        &source,
        &output,
        dir.path().join("scratch"),
        &[V3_INSERT, V3_KNN[1]],
        true,
    );
}
