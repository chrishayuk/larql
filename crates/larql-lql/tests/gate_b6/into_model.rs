//! Gate B, B6: `COMPILE INTO MODEL` is outside B6's scope for a successful
//! export (on `741ecdab` the export fails for every input, a defect tracked
//! separately), but its failure is in scope: it must change nothing.
//!
//! For each backend the test runs the statement on a session with edits and
//! checks the four parts of the observable state (`docs/gate-b-b6.md` §2):
//! the session, every source file's bytes, the destination (absent), and the
//! destination's parent (no entry gained, so no staging directory left
//! behind). It also requires the statement to report the failure rather
//! than success.
//!
//! When the export is fixed this test fails on its first assertion, and the
//! determinism and replay tests must then be extended to cover the output.

use std::path::Path;

use crate::support::*;

fn assert_into_model_fails_without_change(source: &Path, edits: &[&str], dir: &Path) {
    let out = dir.join("model.out");
    let mut session = bound(source);
    for edit in edits {
        run(&mut session, edit);
    }
    let session_before = state(&mut session);
    let source_before = tree(source);
    let parent_before = listing(dir);

    let result = try_run(
        &mut session,
        &format!("COMPILE CURRENT INTO MODEL \"{}\";", lql_path(&out)),
    );

    let err = result.expect_err("COMPILE INTO MODEL is not expected to succeed under B6");
    assert!(!err.is_empty(), "the failure is reported with a reason");
    assert!(!out.exists(), "the destination was created: {err}");
    assert_eq!(listing(dir), parent_before, "the parent changed: {err}");
    assert!(tree(source) == source_before, "the source changed: {err}");
    assert_eq!(
        state(&mut session),
        session_before,
        "the session changed: {err}"
    );
}

#[test]
fn vindex2_compile_into_model_fails_without_change() {
    let dir = tempfile::tempdir().unwrap();
    let source = v2_vindex(dir.path());
    assert_into_model_fails_without_change(&source, &[V2_INSERT, V2_KNN[1]], dir.path());
}

#[test]
fn vindex3_compile_into_model_fails_without_change() {
    let dir = tempfile::tempdir().unwrap();
    let source = v3_container(dir.path());
    assert_into_model_fails_without_change(&source, &[V3_INSERT, V3_KNN[1]], dir.path());
}
