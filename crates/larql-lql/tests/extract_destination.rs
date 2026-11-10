//! Destination safety for `EXTRACT MODEL … INTO …`.
//!
//! EXTRACT writes a vindex directory like COMPILE does, and used to
//! write it in place wherever it was pointed: over an existing container,
//! beside unrelated files, or into the checkpoint it was reading. The V3
//! encoder also removes its output directory when the result fails
//! closure — which, on a directory that already existed, deletes what
//! was there before.
//!
//! The contract is COMPILE's (`compile_destination.rs`): the output must
//! be absent or an empty directory, and neither the model source nor
//! nested with it; a refusal happens before any model bytes move.

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

use larql_inference::test_utils::write_synthetic_model_dir;
use larql_lql::{parse, Session};
use larql_vindex::format::vindex3::fixtures::miniature_glimmer;

/// The phrase every destination refusal carries.
const REFUSAL: &str = "refusing to write";
const BYSTANDER_FILE: &str = "notes.txt";
const BYSTANDER_BYTES: &[u8] = b"not a vindex; must survive";

fn lql_path(path: impl AsRef<Path>) -> String {
    path.as_ref().display().to_string().replace('\\', "\\\\")
}

fn try_run(session: &mut Session, stmt: &str) -> Result<Vec<String>, String> {
    let parsed = parse(stmt).map_err(|e| format!("parse {stmt}: {e}"))?;
    session.execute(&parsed).map_err(|e| e.to_string())
}

fn extract(checkpoint: &Path, out: &Path, format: &str) -> Result<Vec<String>, String> {
    try_run(
        &mut Session::new(),
        &format!(
            "EXTRACT MODEL \"{}\" INTO \"{}\" FORMAT {format};",
            lql_path(checkpoint),
            lql_path(out)
        ),
    )
}

fn snapshot(root: &Path) -> BTreeMap<PathBuf, Vec<u8>> {
    let mut files = BTreeMap::new();
    let mut walk = vec![root.to_path_buf()];
    while let Some(dir) = walk.pop() {
        for entry in std::fs::read_dir(&dir).expect("read_dir") {
            let path = entry.expect("dir entry").path();
            if path.is_dir() {
                walk.push(path);
            } else {
                let rel = path.strip_prefix(root).expect("under root").to_path_buf();
                files.insert(rel, std::fs::read(&path).expect("read file"));
            }
        }
    }
    files
}

/// An HF-checkpoint-shaped directory at `<tmp>/outer/checkpoint`.
fn checkpoint() -> (tempfile::TempDir, PathBuf) {
    let tmp = tempfile::tempdir().expect("tempdir");
    let checkpoint = tmp.path().join("outer").join("checkpoint");
    std::fs::create_dir_all(&checkpoint).expect("checkpoint dir");
    miniature_glimmer(&checkpoint);
    (tmp, checkpoint)
}

fn bystander_dir(parent: &Path) -> PathBuf {
    let dir = parent.join("bystander");
    std::fs::create_dir_all(&dir).expect("bystander dir");
    std::fs::write(dir.join(BYSTANDER_FILE), BYSTANDER_BYTES).expect("bystander file");
    dir
}

fn assert_refused(result: Result<Vec<String>, String>, what: &str) {
    let err = match result {
        Ok(out) => panic!("{what}: expected a refusal, EXTRACT ran: {out:?}"),
        Err(e) => e,
    };
    assert!(err.contains(REFUSAL), "{what}: unexpected error: {err}");
}

// ── VINDEX3 ──────────────────────────────────────────────────────────────

#[test]
fn v3_extract_refuses_a_non_empty_directory_and_adds_nothing_to_it() {
    let (tmp, checkpoint) = checkpoint();
    let bystander = bystander_dir(tmp.path());

    assert_refused(
        extract(&checkpoint, &bystander, "VINDEX3"),
        "a non-empty output directory",
    );
    assert_eq!(snapshot(&bystander).len(), 1, "only the bystander file");
}

#[test]
fn v3_extract_refuses_to_write_over_an_existing_container() {
    let (tmp, checkpoint) = checkpoint();
    let out = tmp.path().join("model.vindex3");
    extract(&checkpoint, &out, "VINDEX3").expect("first extract into a fresh path");
    let first = snapshot(&out);

    assert_refused(
        extract(&checkpoint, &out, "VINDEX3"),
        "an existing container",
    );
    assert_eq!(snapshot(&out), first, "the existing container is untouched");
}

#[test]
fn v3_extract_refuses_the_checkpoint_itself_and_paths_nested_with_it() {
    let (_tmp, checkpoint) = checkpoint();
    let before = snapshot(&checkpoint);

    for out in [
        checkpoint.clone(),
        checkpoint.join("extracted.vindex3"),
        checkpoint.parent().expect("outer").to_path_buf(),
    ] {
        assert_refused(
            extract(&checkpoint, &out, "VINDEX3"),
            "an output nested with the checkpoint",
        );
        assert_eq!(snapshot(&checkpoint), before, "the checkpoint is untouched");
    }
}

#[test]
fn v3_extract_accepts_an_existing_empty_directory() {
    let (tmp, checkpoint) = checkpoint();
    let out = tmp.path().join("empty-out");
    std::fs::create_dir_all(&out).expect("empty out");

    extract(&checkpoint, &out, "VINDEX3").expect("an empty directory is a destination");
    assert!(out.join("index.json").is_file());
}

// ── VINDEX2 ──────────────────────────────────────────────────────────────

#[test]
fn v2_extract_refuses_a_non_empty_directory_before_loading_the_model() {
    // The source here is not a loadable model; the refusal must come
    // first, so nothing about the model decides whether the destination
    // is touched.
    let tmp = tempfile::tempdir().expect("tempdir");
    let source = tmp.path().join("source");
    std::fs::create_dir_all(&source).expect("source dir");
    write_synthetic_model_dir(&source).expect("fixture write");
    let bystander = bystander_dir(tmp.path());

    assert_refused(
        extract(&source, &bystander, "VINDEX2"),
        "a non-empty output directory",
    );
    assert_eq!(snapshot(&bystander).len(), 1, "only the bystander file");
}

#[test]
fn v2_extract_refuses_the_model_directory_as_its_own_output() {
    let tmp = tempfile::tempdir().expect("tempdir");
    let source = tmp.path().join("source");
    std::fs::create_dir_all(&source).expect("source dir");
    write_synthetic_model_dir(&source).expect("fixture write");
    let before = snapshot(&source);

    assert_refused(extract(&source, &source, "VINDEX2"), "output == model");
    assert_eq!(
        snapshot(&source),
        before,
        "the model directory is untouched"
    );
}

#[test]
fn v2_extract_from_a_source_that_is_not_a_local_path_judges_only_the_destination() {
    // A hub id names nothing on this filesystem, so there is no local
    // source to alias — but the destination is still held to "absent or
    // empty", and refused before the model is fetched.
    let tmp = tempfile::tempdir().expect("tempdir");
    let bystander = bystander_dir(tmp.path());
    let hub_id = Path::new("larql-test/no-such-model");

    assert_refused(
        extract(hub_id, &bystander, "VINDEX2"),
        "a non-empty output directory",
    );
}
