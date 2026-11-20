//! Gate B, B6 §3.6 and §4: publication fails at every step without changing
//! anything, except a failure of the final sync, which is reported as
//! published but not confirmed durable.
//!
//! Each case is a session with edits (or none, for COMPACT INTO, which
//! refuses pending edits) and one verb. At each fault point the test
//! checks the four parts of the observable state (§2):
//!
//! 1. the session, through [`state`];
//! 2. the source tree, every file's bytes;
//! 3. the destination: absent;
//! 4. the destination's parent: no staging directory left behind.
//!
//! Then it retries with no fault and requires the output a session that
//! never failed writes.

use std::path::{Path, PathBuf};

use larql_vindex::format::fresh_output::{arm_publish_fault, PublishFault, STAGING_INFIX};

use crate::support::*;

/// Every point before the rename. A failure at any of them leaves the
/// observable state exactly as it was.
const BEFORE_PUBLICATION: [PublishFault; 6] = [
    PublishFault::AfterStagingCreated,
    PublishFault::AfterWork,
    PublishFault::SyncFile,
    PublishFault::SyncStagingDir,
    PublishFault::Validate,
    PublishFault::Rename,
];

/// One verb on one backend: the session's edits, the statement, and the
/// directory holding the source and every output.
struct Case {
    source: PathBuf,
    edits: &'static [&'static str],
    verb: &'static str,
    dir: tempfile::TempDir,
}

const V2_EDITS: &[&str] = &[
    V2_INSERT,
    r#"INSERT INTO EDGES (entity, relation, target) VALUES ("[4]", "[7]", "[6]") AT LAYER 1 MODE KNN;"#,
];
const V3_EDITS: &[&str] = &[
    V3_INSERT,
    r#"INSERT INTO EDGES (entity, relation, target) VALUES ("c", "b", "a") AT LAYER 0 MODE KNN;"#,
];
const COMPILE: &str = "COMPILE CURRENT INTO VINDEX";
const COMPACT: &str = "COMPACT INTO VINDEX";

fn vindex2_compile() -> Case {
    let dir = tempfile::tempdir().unwrap();
    let source = v2_vindex(dir.path());
    Case {
        source,
        edits: V2_EDITS,
        verb: COMPILE,
        dir,
    }
}

fn vindex3_compile() -> Case {
    let dir = tempfile::tempdir().unwrap();
    let source = v3_container(dir.path());
    Case {
        source,
        edits: V3_EDITS,
        verb: COMPILE,
        dir,
    }
}

fn vindex3_compact() -> Case {
    let dir = tempfile::tempdir().unwrap();
    let source = v3_container(dir.path());
    Case {
        source,
        edits: &[],
        verb: COMPACT,
        dir,
    }
}

impl Case {
    fn session(&self) -> larql_lql::Session {
        let mut session = bound(&self.source);
        for edit in self.edits {
            run(&mut session, edit);
        }
        session
    }

    fn statement(&self, out: &Path) -> String {
        format!("{} \"{}\";", self.verb, lql_path(out))
    }

    /// What a session that never failed writes.
    fn reference(&self) -> PathBuf {
        let out = self.dir.path().join("reference.out");
        run(&mut self.session(), &self.statement(&out));
        out
    }
}

/// The parts of §2's observable state a failed publication must leave alone.
struct Observed {
    session: String,
    source: std::collections::BTreeMap<String, Entry>,
    parent: Vec<String>,
}

fn observe(case: &Case, session: &mut larql_lql::Session) -> Observed {
    Observed {
        session: state(session),
        source: tree(&case.source),
        parent: listing(case.dir.path()),
    }
}

fn assert_unchanged(case: &Case, session: &mut larql_lql::Session, before: &Observed, at: &str) {
    let after = observe(case, session);
    assert_eq!(after.session, before.session, "{at}: the session changed");
    assert!(after.source == before.source, "{at}: the source changed");
    assert_eq!(
        after.parent, before.parent,
        "{at}: the output's parent gained or lost an entry"
    );
}

fn assert_fails_without_change(case: Case) {
    let reference = case.reference();
    for point in BEFORE_PUBLICATION {
        let at = format!("{point:?}");
        let out = case.dir.path().join(format!("{at}.out"));
        let mut session = case.session();
        let before = observe(&case, &mut session);

        arm_publish_fault(&[point]);
        let err = try_run(&mut session, &case.statement(&out));
        arm_publish_fault(&[]);
        let err = err.expect_err(&format!("{at}: the publication must fail"));
        assert!(err.contains("injected publish fault"), "{at}: {err}");
        assert!(!out.exists(), "{at}: the destination was created");
        assert_unchanged(&case, &mut session, &before, &at);

        // The B5 retry property: with the fault gone, the same session
        // writes what a session that never failed writes.
        run(&mut session, &case.statement(&out));
        assert_same_tree(&reference, &out);
    }
}

#[test]
fn vindex2_compile_fails_at_every_step_without_change() {
    assert_fails_without_change(vindex2_compile());
}

#[test]
fn vindex3_compile_fails_at_every_step_without_change() {
    assert_fails_without_change(vindex3_compile());
}

#[test]
fn vindex3_compact_fails_at_every_step_without_change() {
    assert_fails_without_change(vindex3_compact());
}

/// A failure whose cleanup also fails names the staging directory it left,
/// and still touches neither the destination, the source nor the session.
fn assert_a_failed_cleanup_names_its_leftover(case: Case) {
    let out = case.dir.path().join("cleanup.out");
    let mut session = case.session();
    let before = observe(&case, &mut session);

    arm_publish_fault(&[PublishFault::Validate, PublishFault::CleanUp]);
    let err = try_run(&mut session, &case.statement(&out));
    arm_publish_fault(&[]);
    let err = err.expect_err("the publication must fail");

    let leftovers: Vec<String> = listing(case.dir.path())
        .into_iter()
        .filter(|name| !before.parent.contains(name))
        .collect();
    assert_eq!(
        leftovers.len(),
        1,
        "one staging directory is left: {leftovers:?}"
    );
    assert!(leftovers[0].contains(STAGING_INFIX), "{leftovers:?}");
    assert!(err.contains("injected publish fault: Validate"), "{err}");
    assert!(err.contains("could not be removed"), "{err}");
    assert!(
        err.contains(&leftovers[0]),
        "the error names the leftover: {err}"
    );
    assert!(!out.exists(), "the destination was created");
    assert_eq!(state(&mut session), before.session, "the session changed");
    assert!(tree(&case.source) == before.source, "the source changed");

    std::fs::remove_dir_all(case.dir.path().join(&leftovers[0])).unwrap();
    assert_eq!(listing(case.dir.path()), before.parent);
}

#[test]
fn vindex2_compile_names_a_staging_directory_it_could_not_remove() {
    assert_a_failed_cleanup_names_its_leftover(vindex2_compile());
}

#[test]
fn vindex3_compile_names_a_staging_directory_it_could_not_remove() {
    assert_a_failed_cleanup_names_its_leftover(vindex3_compile());
}

#[test]
fn vindex3_compact_names_a_staging_directory_it_could_not_remove() {
    assert_a_failed_cleanup_names_its_leftover(vindex3_compact());
}

/// A failure of the sync after the rename is not rolled back: the output is
/// in place and complete, and the error says it was published but its
/// durability is not confirmed.
fn assert_a_failed_final_sync_reports_publication(case: Case) {
    let reference = case.reference();
    let out = case.dir.path().join("published.out");
    let mut session = case.session();
    let before = observe(&case, &mut session);

    arm_publish_fault(&[PublishFault::SyncParent]);
    let err = try_run(&mut session, &case.statement(&out));
    arm_publish_fault(&[]);
    let err = err.expect_err("the publication must report the failed sync");

    assert!(err.contains("published"), "{err}");
    assert!(err.contains("durability is not confirmed"), "{err}");
    assert!(err.contains("injected publish fault: SyncParent"), "{err}");
    assert_same_tree(&reference, &out);
    assert_eq!(state(&mut session), before.session, "the session changed");
    assert!(tree(&case.source) == before.source, "the source changed");
    let mut want = before.parent.clone();
    want.push("published.out".into());
    want.sort();
    assert_eq!(listing(case.dir.path()), want, "only the output was added");
}

#[test]
fn vindex2_compile_reports_a_failed_final_sync_as_published() {
    assert_a_failed_final_sync_reports_publication(vindex2_compile());
}

#[test]
fn vindex3_compile_reports_a_failed_final_sync_as_published() {
    assert_a_failed_final_sync_reports_publication(vindex3_compile());
}

#[test]
fn vindex3_compact_reports_a_failed_final_sync_as_published() {
    assert_a_failed_final_sync_reports_publication(vindex3_compact());
}
