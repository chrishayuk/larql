//! The weight-patch containers the independent verifier is tested on are engine output, committed:
//! `verifier/conformance/weight_patches_v4/` and `verifier/conformance/weight_patches_v2v3/`. This pins them to the
//! engine as it is now: each case's recorded statements, run again from the committed base, must write the
//! committed container byte for byte, and a case that carries the `.vlp` its ops were saved as must save the same
//! bytes again.
//!
//! Without this the two halves can drift apart unseen. A change to what the engine writes into a ledger (a renamed
//! field, a new one) would leave the engine's tests green and the verifier's tests green on fixtures no engine
//! still produces. With it, such a change fails here until the fixtures are regenerated (each suite's
//! `make_fixtures.py`), and regenerating them runs the verifier's tests on what the engine really writes.
//!
//! **A case is pinned because it is on disk** (Gate B, B7 r2 §5.3), not because it is named here: every directory
//! of a suite is a case and must carry its recorded statements and its compiled container. So a fixture added on
//! the verifier side is pinned by this test without an engine-side change, and the two sides keep to separate pull
//! requests (`docs/verification/independence.md`).

use std::path::{Path, PathBuf};

use crate::support::*;

/// The fixture suites, relative to this crate. Each subdirectory of a suite is one case.
const SUITES: [&str; 2] = [
    "../../verifier/conformance/weight_patches_v4",
    "../../verifier/conformance/weight_patches_v2v3",
];
const BASE: &str = "../../verifier/conformance/structured_patch/base";
const STATEMENTS_FILE: &str = "statements.lql";
const COMPILED_DIR: &str = "compiled";
const PATCH_FILE: &str = "patch.vlp";
const LEDGER_FILE: &str = "weight_patches.json";
/// How the recorded statements name the base and the output, relative to the fixture directory.
const RECORDED_BASE: &str = "\"../structured_patch/base\"";
const RECORDED_OUTPUT: &str = "\"compiled\"";
/// The one file that is not compared: it records where the container was built (`source.locator_hint`), a
/// different temporary path on every run, and holds no weight and nothing the verifier judges.
const BUILD_RECORD: &str = "candidate.json";

/// Not canonicalised: on Windows that yields a verbatim path, which does not take the `/` the recorded
/// statements join operand files with.
fn repo_path(relative: &str) -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join(relative)
}

/// Every case of every suite, as (suite directory, case name), in suite then name order.
fn cases() -> Vec<(PathBuf, String)> {
    let mut out = Vec::new();
    for suite in SUITES {
        let root = repo_path(suite);
        let mut names: Vec<String> = std::fs::read_dir(&root)
            .unwrap()
            .map(|e| e.unwrap().path())
            .filter(|p| p.is_dir())
            .map(|p| p.file_name().unwrap().to_string_lossy().to_string())
            .collect();
        names.sort();
        assert!(!names.is_empty(), "{suite}: no cases on disk");
        out.extend(names.into_iter().map(|n| (root.clone(), n)));
    }
    out
}

/// Every file under `root` but the build record, relative path -> bytes, in path order.
fn files(root: &Path) -> Vec<(String, Vec<u8>)> {
    fn walk(root: &Path, dir: &Path, out: &mut Vec<(String, Vec<u8>)>) {
        for entry in std::fs::read_dir(dir).unwrap() {
            let path = entry.unwrap().path();
            if path.is_dir() {
                walk(root, &path, out);
                continue;
            }
            let relative = path
                .strip_prefix(root)
                .unwrap()
                .components()
                .map(|c| c.as_os_str().to_string_lossy())
                .collect::<Vec<_>>()
                .join("/");
            if relative != BUILD_RECORD {
                out.push((relative, std::fs::read(&path).unwrap()));
            }
        }
    }
    let mut out = Vec::new();
    walk(root, root, &mut out);
    out.sort();
    out
}

/// A case's recorded statements, with the base, the output and its operand files resolved: operand files are
/// recorded relative to the suite directory, as `<case>/operands/<file>`.
fn resolved_statements(suite: &Path, case: &str, out: &Path) -> Vec<String> {
    let recorded = std::fs::read_to_string(suite.join(case).join(STATEMENTS_FILE)).unwrap();
    recorded
        .lines()
        .filter(|l| !l.trim().is_empty())
        .map(|line| {
            line.replace(RECORDED_BASE, &format!("\"{}\"", q(repo_path(BASE))))
                .replace(RECORDED_OUTPUT, &format!("\"{}\"", q(out)))
                .replace(
                    &format!("\"{case}/"),
                    &format!("\"{}/", q(suite.join(case))),
                )
        })
        .collect()
}

/// Run a case's recorded statements against the committed base, compiling into `out`.
fn recompile(suite: &Path, case: &str, out: &Path) {
    let mut session = larql_lql::Session::new();
    let mut compiled = false;
    for statement in resolved_statements(suite, case, out) {
        compiled |= statement.starts_with("COMPILE");
        run(&mut session, &statement);
    }
    assert!(compiled, "{case}: the recorded statements compile");
}

/// Record a case's edit statements as a `.vlp` in a fresh session, as `make_fixtures.py` saves `patch.vlp`.
fn resave_patch(suite: &Path, case: &str, out: &Path) -> Vec<u8> {
    let patch = out.join(PATCH_FILE);
    let mut session = larql_lql::Session::new();
    run(&mut session, &format!("USE \"{}\";", q(repo_path(BASE))));
    run(&mut session, &format!("BEGIN PATCH \"{}\";", q(&patch)));
    for statement in resolved_statements(suite, case, &out.join("unused")) {
        if !statement.starts_with("USE") && !statement.starts_with("COMPILE") {
            run(&mut session, &statement);
        }
    }
    run(&mut session, "SAVE PATCH;");
    std::fs::read(&patch).unwrap()
}

#[test]
fn every_case_on_disk_is_a_complete_case() {
    // A directory without its statements or its container would otherwise be silently unpinned.
    for (suite, case) in cases() {
        let dir = suite.join(&case);
        assert!(
            dir.join(STATEMENTS_FILE).is_file(),
            "{case}: no {STATEMENTS_FILE}"
        );
        assert!(
            dir.join(COMPILED_DIR).is_dir(),
            "{case}: no {COMPILED_DIR}/"
        );
    }
}

#[test]
fn the_engine_still_writes_the_containers_the_verifier_is_tested_on() {
    for (suite, case) in cases() {
        let dir = tempfile::tempdir().unwrap();
        let out = dir.path().join("compiled.vindex3");
        recompile(&suite, &case, &out);

        let committed = files(&suite.join(&case).join(COMPILED_DIR));
        let written = files(&out);
        let names =
            |fs: &[(String, Vec<u8>)]| fs.iter().map(|(n, _)| n.clone()).collect::<Vec<_>>();
        assert_eq!(names(&written), names(&committed), "{case}: the same files");
        for ((name, now), (_, then)) in written.iter().zip(&committed) {
            assert!(
                now == then,
                "{case}: `{name}` differs from the committed fixture. If the engine's output is meant to \
                 change, regenerate with the suite's make_fixtures.py and run the verifier's tests on the result"
            );
        }
        assert!(
            committed.iter().any(|(n, _)| n == LEDGER_FILE),
            "{case}: the comparison covered the ledger"
        );
    }
}

#[test]
fn the_engine_still_saves_the_patches_the_verifier_is_tested_on() {
    let mut pinned = 0;
    for (suite, case) in cases() {
        let committed = suite.join(&case).join(PATCH_FILE);
        if !committed.is_file() {
            continue;
        }
        let dir = tempfile::tempdir().unwrap();
        let saved = resave_patch(&suite, &case, dir.path());
        assert!(
            saved == std::fs::read(&committed).unwrap(),
            "{case}: {PATCH_FILE} differs from what the engine saves for the recorded statements"
        );
        pinned += 1;
    }
    assert!(pinned > 0, "the comparison covered at least one patch");
}
