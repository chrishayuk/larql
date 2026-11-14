//! The `weight-patches/v4` containers the independent verifier is tested on
//! (`verifier/conformance/weight_patches_v4/`) are engine output, committed. This pins them to the engine as it is
//! now: each case's recorded statements, run again from the committed base, must write the committed container
//! byte for byte.
//!
//! Without this the two halves can drift apart unseen. A change to what the engine writes into a ledger (a renamed
//! field, a new one) would leave the engine's tests green and the verifier's tests green on fixtures no engine
//! still produces. With it, such a change fails here until the fixtures are regenerated
//! (`verifier/conformance/weight_patches_v4/make_fixtures.py`), and regenerating them runs the verifier's tests on
//! what the engine really writes.

use std::path::{Path, PathBuf};

use crate::support::*;

const FIXTURES: &str = "../../verifier/conformance/weight_patches_v4";
const BASE: &str = "../../verifier/conformance/structured_patch/base";
const STATEMENTS_FILE: &str = "statements.lql";
const COMPILED_DIR: &str = "compiled";
/// How the recorded statements name the base and the output, relative to the fixture directory.
const RECORDED_BASE: &str = "\"../structured_patch/base\"";
const RECORDED_OUTPUT: &str = "\"compiled\"";
/// The one file that is not compared: it records where the container was built (`source.locator_hint`), a
/// different temporary path on every run, and holds no weight and nothing the verifier judges.
const BUILD_RECORD: &str = "candidate.json";
const CASES: [&str; 4] = ["low_rank_k2", "mixed", "sham", "f32_profile"];

/// Not canonicalised: on Windows that yields a verbatim path, which does not take the `/` the recorded
/// statements join operand files with.
fn repo_path(relative: &str) -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join(relative)
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

/// Run a case's recorded statements against the committed base, compiling into `out`.
fn recompile(case: &str, out: &Path) {
    let fixtures = repo_path(FIXTURES);
    let recorded = std::fs::read_to_string(fixtures.join(case).join(STATEMENTS_FILE)).unwrap();
    let mut session = larql_lql::Session::new();
    let mut compiled = false;
    for line in recorded.lines().filter(|l| !l.trim().is_empty()) {
        // Operand files are recorded relative to the fixture directory, as `<case>/operands/<file>`.
        let statement = line
            .replace(RECORDED_BASE, &format!("\"{}\"", q(repo_path(BASE))))
            .replace(RECORDED_OUTPUT, &format!("\"{}\"", q(out)))
            .replace(
                &format!("\"{case}/"),
                &format!("\"{}/", q(fixtures.join(case))),
            );
        compiled |= statement.starts_with("COMPILE");
        run(&mut session, &statement);
    }
    assert!(compiled, "{case}: the recorded statements compile");
}

#[test]
fn the_engine_still_writes_the_containers_the_verifier_is_tested_on() {
    for case in CASES {
        let dir = tempfile::tempdir().unwrap();
        let out = dir.path().join("compiled.vindex3");
        recompile(case, &out);

        let committed = files(&repo_path(FIXTURES).join(case).join(COMPILED_DIR));
        let written = files(&out);
        let names =
            |fs: &[(String, Vec<u8>)]| fs.iter().map(|(n, _)| n.clone()).collect::<Vec<_>>();
        assert_eq!(names(&written), names(&committed), "{case}: the same files");
        for ((name, now), (_, then)) in written.iter().zip(&committed) {
            assert!(
                now == then,
                "{case}: `{name}` differs from the committed fixture. If the engine's output is meant to \
                 change, regenerate with verifier/conformance/weight_patches_v4/make_fixtures.py and run the \
                 verifier's tests on the result"
            );
        }
        assert!(
            committed.iter().any(|(n, _)| n == "weight_patches.json"),
            "{case}: the comparison covered the ledger"
        );
    }
}

#[test]
fn every_committed_case_is_pinned() {
    // A case added to the verifier's fixtures without being added here would be unpinned.
    let mut on_disk: Vec<String> = std::fs::read_dir(repo_path(FIXTURES))
        .unwrap()
        .map(|e| e.unwrap().path())
        .filter(|p| p.join(STATEMENTS_FILE).is_file())
        .map(|p| p.file_name().unwrap().to_string_lossy().to_string())
        .collect();
    on_disk.sort();
    let mut pinned: Vec<String> = CASES.iter().map(|c| c.to_string()).collect();
    pinned.sort();
    assert_eq!(on_disk, pinned);
}
