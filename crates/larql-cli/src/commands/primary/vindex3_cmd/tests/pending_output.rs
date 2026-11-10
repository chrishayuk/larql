//! #744: a dump appears at its path only once complete; a refused or
//! failed run leaves nothing there — not an empty file, not a partial
//! one, and not a previous run's output.

use std::io::Write;
use std::path::Path;

use larql_vindex::format::vindex3::fixtures::{dense_f32_model, encode_fixture_container};

use super::super::pending_output::{parent_dir, remove_stale, PendingOutput, PARTIAL_PREFIX};
use super::super::teacher_force::META_EXTENSION;
use super::super::{run, ExecArgs, Vindex3Command};

/// Every file left in `dir`.
fn names(dir: &Path) -> Vec<String> {
    let mut names: Vec<String> = std::fs::read_dir(dir)
        .unwrap()
        .map(|e| e.unwrap().file_name().to_string_lossy().into_owned())
        .collect();
    names.sort();
    names
}

#[test]
fn a_committed_output_holds_its_bytes_and_leaves_no_partial() {
    let dir = tempfile::tempdir().unwrap();
    let dest = dir.path().join("out.f32");
    let mut out = PendingOutput::create(&dest).unwrap();
    out.write_all(b"abcd").unwrap();
    assert!(!dest.exists(), "nothing at the path before commit");
    out.commit().unwrap();
    assert_eq!(std::fs::read(&dest).unwrap(), b"abcd");
    assert_eq!(names(dir.path()), vec!["out.f32"]);
}

#[test]
fn an_uncommitted_output_leaves_nothing() {
    let dir = tempfile::tempdir().unwrap();
    let dest = dir.path().join("out.f32");
    {
        let mut out = PendingOutput::create(&dest).unwrap();
        out.write_all(b"partial").unwrap();
        out.flush().unwrap();
        let partials: Vec<String> = names(dir.path());
        assert_eq!(partials.len(), 1);
        assert!(partials[0].starts_with(PARTIAL_PREFIX), "{partials:?}");
    }
    assert!(names(dir.path()).is_empty(), "{:?}", names(dir.path()));
}

#[test]
fn creating_an_output_removes_a_previous_one() {
    let dir = tempfile::tempdir().unwrap();
    let dest = dir.path().join("out.f32");
    std::fs::write(&dest, b"an earlier run").unwrap();
    drop(PendingOutput::create(&dest).unwrap());
    assert!(!dest.exists(), "a failed run must not leave the old output");
}

#[test]
fn the_partial_lives_beside_its_destination() {
    assert_eq!(parent_dir(Path::new("out.f32")), Path::new("."));
    assert_eq!(parent_dir(Path::new("a/b.f32")), Path::new("a"));
    assert_eq!(parent_dir(Path::new("/x/y.f32")), Path::new("/x"));
}

#[test]
fn removing_a_stale_file_tolerates_absence_and_reports_other_failures() {
    let dir = tempfile::tempdir().unwrap();
    remove_stale(&dir.path().join("absent")).unwrap();
    // A directory cannot be removed as a file: a real failure, surfaced.
    assert!(remove_stale(dir.path()).is_err());
}

#[test]
fn a_logit_dump_that_fails_part_way_leaves_nothing_at_its_paths() {
    let root = tempfile::tempdir().unwrap();
    let checkpoint = root.path().join("checkpoint");
    let container = root.path().join("container");
    std::fs::create_dir_all(&checkpoint).unwrap();
    std::fs::create_dir_all(&container).unwrap();
    encode_fixture_container(dense_f32_model, &checkpoint, &container, "dense");
    let out_dir = root.path().join("out");
    std::fs::create_dir_all(&out_dir).unwrap();
    let dest = out_dir.join("logits.f32");
    let meta = dest.with_extension(META_EXTENSION);
    // A previous run's dump and sidecar, which must not survive as this
    // run's result.
    std::fs::write(&dest, b"stale").unwrap();
    std::fs::write(&meta, b"{}").unwrap();

    #[derive(clap::Parser)]
    struct ExecCli {
        #[command(flatten)]
        exec: ExecArgs,
    }
    let args = <ExecCli as clap::Parser>::try_parse_from([
        "exec",
        container.to_str().unwrap(),
        // The last id is outside the fixture's vocabulary: the run fails
        // after earlier positions were already written.
        "--tokens",
        "1,2,4000000000",
        "--logit-dump",
        dest.to_str().unwrap(),
    ])
    .unwrap()
    .exec;
    let err = run(Vindex3Command::Exec(args));
    assert!(err.is_err(), "an out-of-vocabulary id must fail the run");
    assert!(names(&out_dir).is_empty(), "{:?}", names(&out_dir));
}
