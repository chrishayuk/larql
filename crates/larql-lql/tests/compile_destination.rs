//! Destination safety for every LQL verb that writes a new vindex
//! directory: `COMPILE … INTO VINDEX` on both generations and
//! `COMPACT INTO VINDEX`.
//!
//! The contract these gates pin:
//!
//! - an output that is, contains, or sits inside the source is refused
//!   before any write, and the source is byte-identical afterwards;
//! - an output directory that already holds anything is refused and
//!   keeps what it held, a previously compiled vindex included —
//!   replacing an existing artifact is not something COMPILE decides;
//! - an existing *empty* directory is a legitimate destination.
//!
//! Each refusal case is a reproducer: before the guard, the legacy path
//! recursively deleted whatever was at the output, and the VINDEX3 paths
//! wrote into the source in place.

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

use larql_inference::test_utils::write_synthetic_model_dir;
use larql_lql::{parse, Session};
use larql_vindex::format::vindex3::fixtures::{encode_fixture_container, miniature_glimmer};

/// The phrase every destination refusal carries.
const REFUSAL: &str = "refusing to write";
/// A file a bystander directory holds; its survival is the assertion.
const BYSTANDER_FILE: &str = "notes.txt";
const BYSTANDER_BYTES: &[u8] = b"not a vindex; must survive";

fn lql_path(path: impl AsRef<Path>) -> String {
    path.as_ref().display().to_string().replace('\\', "\\\\")
}

fn try_run(session: &mut Session, stmt: &str) -> Result<Vec<String>, String> {
    let parsed = parse(stmt).map_err(|e| format!("parse {stmt}: {e}"))?;
    session.execute(&parsed).map_err(|e| e.to_string())
}

fn bound_session(source: &Path) -> Session {
    let mut session = Session::new();
    try_run(&mut session, &format!("USE \"{}\";", lql_path(source))).expect("USE");
    session
}

/// Every file under `root`, by relative path, with its bytes — the
/// "byte-identical afterwards" instrument.
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

/// A legacy vindex at `<tmp>/outer/source.vindex`, so the tests can name
/// an ancestor of the source that is still inside the tempdir.
fn legacy_source() -> (tempfile::TempDir, PathBuf) {
    let tmp = tempfile::tempdir().expect("tempdir");
    let source = tmp.path().join("outer").join("source.vindex");
    std::fs::create_dir_all(&source).expect("source dir");
    write_synthetic_model_dir(&source).expect("fixture write");
    (tmp, source)
}

/// A VINDEX3 container at `<tmp>/outer/source.vindex3`.
fn v3_source() -> (tempfile::TempDir, PathBuf) {
    let tmp = tempfile::tempdir().expect("tempdir");
    let checkpoint = tmp.path().join("checkpoint");
    let source = tmp.path().join("outer").join("source.vindex3");
    std::fs::create_dir_all(&checkpoint).expect("checkpoint dir");
    std::fs::create_dir_all(&source).expect("source dir");
    encode_fixture_container(
        miniature_glimmer,
        &checkpoint,
        &source,
        "destination-fixture",
    );
    (tmp, source)
}

fn bystander_dir(parent: &Path) -> PathBuf {
    let dir = parent.join("bystander");
    std::fs::create_dir_all(&dir).expect("bystander dir");
    std::fs::write(dir.join(BYSTANDER_FILE), BYSTANDER_BYTES).expect("bystander file");
    dir
}

fn assert_refused(result: Result<Vec<String>, String>, what: &str) {
    let err = result.expect_err(what);
    assert!(err.contains(REFUSAL), "{what}: unexpected error: {err}");
}

// ── Legacy COMPILE INTO VINDEX ───────────────────────────────────────────

#[test]
fn legacy_compile_refuses_a_non_empty_directory_and_keeps_its_files() {
    let (tmp, source) = legacy_source();
    let bystander = bystander_dir(tmp.path());
    let mut session = bound_session(&source);

    assert_refused(
        try_run(
            &mut session,
            &format!("COMPILE CURRENT INTO VINDEX \"{}\";", lql_path(&bystander)),
        ),
        "a non-empty output directory",
    );
    assert_eq!(
        std::fs::read(bystander.join(BYSTANDER_FILE)).expect("bystander survives"),
        BYSTANDER_BYTES
    );
}

#[test]
fn legacy_compile_refuses_to_replace_a_previously_compiled_vindex() {
    let (tmp, source) = legacy_source();
    let out = tmp.path().join("compiled.vindex");
    let mut session = bound_session(&source);
    let stmt = format!("COMPILE CURRENT INTO VINDEX \"{}\";", lql_path(&out));

    try_run(&mut session, &stmt).expect("first compile into a fresh path");
    let first = snapshot(&out);

    assert_refused(try_run(&mut session, &stmt), "an existing compiled vindex");
    assert_eq!(snapshot(&out), first, "the existing vindex is untouched");
}

#[test]
fn legacy_compile_refuses_an_ancestor_of_the_source() {
    let (_tmp, source) = legacy_source();
    let ancestor = source.parent().expect("outer").to_path_buf();
    let before = snapshot(&source);
    let mut session = bound_session(&source);

    assert_refused(
        try_run(
            &mut session,
            &format!("COMPILE CURRENT INTO VINDEX \"{}\";", lql_path(&ancestor)),
        ),
        "an output that contains the source",
    );
    assert_eq!(snapshot(&source), before, "the source is byte-identical");
}

#[test]
fn legacy_compile_refuses_an_output_inside_the_source() {
    let (_tmp, source) = legacy_source();
    let inside = source.join("compiled");
    let before = snapshot(&source);
    let mut session = bound_session(&source);

    assert_refused(
        try_run(
            &mut session,
            &format!("COMPILE CURRENT INTO VINDEX \"{}\";", lql_path(&inside)),
        ),
        "an output inside the source",
    );
    assert_eq!(snapshot(&source), before, "the source is byte-identical");
}

#[cfg(unix)]
#[test]
fn legacy_compile_refuses_a_symlink_that_names_the_source() {
    let (tmp, source) = legacy_source();
    let alias = tmp.path().join("alias.vindex");
    std::os::unix::fs::symlink(&source, &alias).expect("symlink");
    let before = snapshot(&source);
    let mut session = bound_session(&source);

    assert_refused(
        try_run(
            &mut session,
            &format!("COMPILE CURRENT INTO VINDEX \"{}\";", lql_path(&alias)),
        ),
        "a symlink to the source",
    );
    assert_eq!(snapshot(&source), before, "the source is byte-identical");
}

#[cfg(unix)]
#[test]
fn legacy_compile_refuses_a_path_that_reaches_the_source_through_a_symlinked_parent() {
    let (tmp, source) = legacy_source();
    let outer_alias = tmp.path().join("outer-alias");
    std::os::unix::fs::symlink(source.parent().expect("outer"), &outer_alias).expect("symlink");
    let aliased_source = outer_alias.join(source.file_name().expect("name"));
    let before = snapshot(&source);
    let mut session = bound_session(&source);

    assert_refused(
        try_run(
            &mut session,
            &format!(
                "COMPILE CURRENT INTO VINDEX \"{}\";",
                lql_path(&aliased_source)
            ),
        ),
        "the source reached through a symlinked parent",
    );
    assert_eq!(snapshot(&source), before, "the source is byte-identical");
}

#[test]
fn legacy_compile_accepts_an_existing_empty_directory() {
    let (tmp, source) = legacy_source();
    let out = tmp.path().join("empty-out");
    std::fs::create_dir_all(&out).expect("empty out");
    let mut session = bound_session(&source);

    try_run(
        &mut session,
        &format!("COMPILE CURRENT INTO VINDEX \"{}\";", lql_path(&out)),
    )
    .expect("an empty directory is a legitimate destination");
    assert!(!snapshot(&out).is_empty(), "the compiled vindex landed");
}

// ── VINDEX3 COMPILE INTO VINDEX ──────────────────────────────────────────

#[test]
fn v3_compile_refuses_the_source_as_its_own_output() {
    let (_tmp, source) = v3_source();
    let before = snapshot(&source);
    let mut session = bound_session(&source);

    assert_refused(
        try_run(
            &mut session,
            &format!("COMPILE CURRENT INTO VINDEX \"{}\";", lql_path(&source)),
        ),
        "output == source",
    );
    assert_eq!(snapshot(&source), before, "the source is byte-identical");
}

#[test]
fn v3_compile_refuses_a_non_empty_directory_and_keeps_its_files() {
    let (tmp, source) = v3_source();
    let bystander = bystander_dir(tmp.path());
    let mut session = bound_session(&source);

    assert_refused(
        try_run(
            &mut session,
            &format!("COMPILE CURRENT INTO VINDEX \"{}\";", lql_path(&bystander)),
        ),
        "a non-empty output directory",
    );
    assert_eq!(
        snapshot(&bystander).len(),
        1,
        "nothing was written beside the bystander file"
    );
}

#[test]
fn v3_compile_refuses_an_ancestor_of_the_source() {
    let (_tmp, source) = v3_source();
    let ancestor = source.parent().expect("outer").to_path_buf();
    let before = snapshot(&source);
    let mut session = bound_session(&source);

    assert_refused(
        try_run(
            &mut session,
            &format!("COMPILE CURRENT INTO VINDEX \"{}\";", lql_path(&ancestor)),
        ),
        "an output that contains the source",
    );
    assert_eq!(snapshot(&source), before, "the source is byte-identical");
}

#[test]
fn v3_compile_refuses_an_output_inside_the_source() {
    let (_tmp, source) = v3_source();
    let inside = source.join("compiled");
    let before = snapshot(&source);
    let mut session = bound_session(&source);

    assert_refused(
        try_run(
            &mut session,
            &format!("COMPILE CURRENT INTO VINDEX \"{}\";", lql_path(&inside)),
        ),
        "an output inside the source",
    );
    assert_eq!(snapshot(&source), before, "the source is byte-identical");
}

#[cfg(unix)]
#[test]
fn v3_compile_refuses_a_symlink_even_when_spelled_with_a_trailing_slash() {
    let (tmp, source) = v3_source();
    let target = tmp.path().join("real-empty");
    std::fs::create_dir(&target).expect("target");
    let link = tmp.path().join("link");
    std::os::unix::fs::symlink(&target, &link).expect("symlink");
    let mut session = bound_session(&source);

    for spelling in [lql_path(&link), format!("{}/", lql_path(&link))] {
        assert_refused(
            try_run(
                &mut session,
                &format!("COMPILE CURRENT INTO VINDEX \"{spelling}\";"),
            ),
            "a symlink destination",
        );
    }
    assert!(
        snapshot(&target).is_empty(),
        "nothing written through the link"
    );
}

#[test]
fn v3_compile_into_a_fresh_path_leaves_no_staging_directory_behind() {
    let (tmp, source) = v3_source();
    let out = tmp.path().join("compiled.vindex3");
    let mut session = bound_session(&source);

    try_run(
        &mut session,
        &format!("COMPILE CURRENT INTO VINDEX \"{}\";", lql_path(&out)),
    )
    .expect("compile into a fresh path");

    let siblings: Vec<String> = std::fs::read_dir(tmp.path())
        .expect("read tmp")
        .map(|e| e.expect("entry").file_name().to_string_lossy().into_owned())
        .collect();
    assert!(
        siblings.iter().all(|name| !name.contains(".tmp.")),
        "staging directory left behind: {siblings:?}"
    );
}

// ── VINDEX3 COMPACT INTO VINDEX ──────────────────────────────────────────

#[test]
fn v3_compact_refuses_the_source_as_its_own_output() {
    let (_tmp, source) = v3_source();
    let before = snapshot(&source);
    let mut session = bound_session(&source);

    assert_refused(
        try_run(
            &mut session,
            &format!("COMPACT INTO VINDEX \"{}\";", lql_path(&source)),
        ),
        "output == source",
    );
    assert_eq!(snapshot(&source), before, "the source is byte-identical");
}

#[test]
fn v3_compact_refuses_an_ancestor_of_and_a_path_inside_the_source() {
    let (_tmp, source) = v3_source();
    let before = snapshot(&source);
    let mut session = bound_session(&source);

    for out in [
        source.parent().expect("outer").to_path_buf(),
        source.join("compacted"),
    ] {
        assert_refused(
            try_run(
                &mut session,
                &format!("COMPACT INTO VINDEX \"{}\";", lql_path(&out)),
            ),
            "an output nested with the source",
        );
        assert_eq!(snapshot(&source), before, "the source is byte-identical");
    }
}

#[test]
fn v3_compact_refuses_a_non_empty_directory_and_keeps_its_files() {
    let (tmp, source) = v3_source();
    let bystander = bystander_dir(tmp.path());
    let mut session = bound_session(&source);

    assert_refused(
        try_run(
            &mut session,
            &format!("COMPACT INTO VINDEX \"{}\";", lql_path(&bystander)),
        ),
        "a non-empty output directory",
    );
    assert_eq!(
        snapshot(&bystander).len(),
        1,
        "nothing was written beside the bystander file"
    );
}
