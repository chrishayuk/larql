//! Gates for the fresh-output boundary: which destinations are refused,
//! that a refusal or a failure changes nothing on disk, and that no
//! directory this module did not create is ever removed.

use std::path::{Path, PathBuf};

use super::{
    check_empty_destination, check_fresh_destination, write_fresh_directory, STAGING_INFIX,
};
use crate::error::VindexError;

const KEPT_FILE: &str = "kept.bin";
const KEPT_BYTES: &[u8] = b"pre-existing";
const WRITTEN_FILE: &str = "written.bin";
const WRITTEN_BYTES: &[u8] = b"fresh";

/// `<tmp>/outer/source` holding one file — a stand-in vindex.
fn source_in(tmp: &Path) -> PathBuf {
    let source = tmp.join("outer").join("source");
    std::fs::create_dir_all(&source).unwrap();
    std::fs::write(source.join(KEPT_FILE), KEPT_BYTES).unwrap();
    source
}

fn write_one(staging: &Path) -> Result<(), VindexError> {
    std::fs::write(staging.join(WRITTEN_FILE), WRITTEN_BYTES)?;
    Ok(())
}

fn unreachable_work(_: &Path) -> Result<(), VindexError> {
    panic!("work must not run for a refused destination");
}

fn refusal_reason(err: VindexError) -> String {
    match err {
        VindexError::UnsafeOutputDestination { reason, .. } => reason,
        other => panic!("expected a destination refusal, got: {other}"),
    }
}

fn staging_leftovers(parent: &Path) -> Vec<String> {
    std::fs::read_dir(parent)
        .unwrap()
        .map(|e| e.unwrap().file_name().to_string_lossy().into_owned())
        .filter(|name| name.contains(STAGING_INFIX))
        .collect()
}

// ── Accepted destinations ────────────────────────────────────────────────

#[test]
fn a_missing_destination_receives_the_staged_work() {
    let tmp = tempfile::tempdir().unwrap();
    let source = source_in(tmp.path());
    let out = tmp.path().join("out");

    write_fresh_directory(&source, &out, write_one).unwrap();

    assert_eq!(
        std::fs::read(out.join(WRITTEN_FILE)).unwrap(),
        WRITTEN_BYTES
    );
    assert!(staging_leftovers(tmp.path()).is_empty());
}

#[test]
fn an_existing_empty_directory_is_a_legitimate_destination() {
    let tmp = tempfile::tempdir().unwrap();
    let source = source_in(tmp.path());
    let out = tmp.path().join("out");
    std::fs::create_dir(&out).unwrap();

    write_fresh_directory(&source, &out, write_one).unwrap();

    assert_eq!(
        std::fs::read(out.join(WRITTEN_FILE)).unwrap(),
        WRITTEN_BYTES
    );
}

#[test]
fn missing_parent_directories_are_created() {
    let tmp = tempfile::tempdir().unwrap();
    let source = source_in(tmp.path());
    let out = tmp.path().join("not").join("yet").join("out");

    write_fresh_directory(&source, &out, write_one).unwrap();

    assert!(out.join(WRITTEN_FILE).exists());
}

#[test]
fn the_work_closure_writes_into_staging_not_the_destination() {
    let tmp = tempfile::tempdir().unwrap();
    let source = source_in(tmp.path());
    let out = tmp.path().join("out");

    write_fresh_directory(&source, &out, |staging| {
        assert_ne!(staging, out.as_path());
        assert_eq!(staging.parent(), out.parent(), "same filesystem as out");
        assert!(!out.exists(), "nothing at the destination until promotion");
        write_one(staging)
    })
    .unwrap();
}

// ── Refused destinations ─────────────────────────────────────────────────

#[test]
fn a_non_empty_directory_is_refused_and_keeps_its_contents() {
    let tmp = tempfile::tempdir().unwrap();
    let source = source_in(tmp.path());
    let out = tmp.path().join("out");
    std::fs::create_dir(&out).unwrap();
    std::fs::write(out.join(KEPT_FILE), KEPT_BYTES).unwrap();

    let err = write_fresh_directory(&source, &out, unreachable_work).unwrap_err();

    assert!(refusal_reason(err).contains("not empty"));
    assert_eq!(std::fs::read(out.join(KEPT_FILE)).unwrap(), KEPT_BYTES);
    assert!(staging_leftovers(tmp.path()).is_empty());
}

#[test]
fn a_file_at_the_destination_is_refused_and_kept() {
    let tmp = tempfile::tempdir().unwrap();
    let source = source_in(tmp.path());
    let out = tmp.path().join("out");
    std::fs::write(&out, KEPT_BYTES).unwrap();

    let err = write_fresh_directory(&source, &out, unreachable_work).unwrap_err();

    assert!(refusal_reason(err).contains("a file already exists"));
    assert_eq!(std::fs::read(&out).unwrap(), KEPT_BYTES);
}

#[test]
fn the_source_itself_is_refused() {
    let tmp = tempfile::tempdir().unwrap();
    let source = source_in(tmp.path());

    let err = write_fresh_directory(&source, &source, unreachable_work).unwrap_err();

    assert!(refusal_reason(err).contains("resolves to the source vindex"));
    assert_eq!(std::fs::read(source.join(KEPT_FILE)).unwrap(), KEPT_BYTES);
}

#[test]
fn the_source_spelled_with_dot_segments_is_refused() {
    let tmp = tempfile::tempdir().unwrap();
    let source = source_in(tmp.path());
    let spelled = source.join("..").join("source").join(".");

    let err = check_fresh_destination(&source, &spelled).unwrap_err();

    assert!(refusal_reason(err).contains("resolves to the source vindex"));
}

#[test]
fn an_ancestor_of_the_source_is_refused() {
    let tmp = tempfile::tempdir().unwrap();
    let source = source_in(tmp.path());
    let ancestor = source.parent().unwrap();

    let err = write_fresh_directory(&source, ancestor, unreachable_work).unwrap_err();

    assert!(refusal_reason(err).contains("contains the source vindex"));
    assert_eq!(std::fs::read(source.join(KEPT_FILE)).unwrap(), KEPT_BYTES);
}

#[test]
fn a_destination_inside_the_source_is_refused() {
    let tmp = tempfile::tempdir().unwrap();
    let source = source_in(tmp.path());

    let err = write_fresh_directory(&source, &source.join("nested"), unreachable_work).unwrap_err();

    assert!(refusal_reason(err).contains("inside the source vindex"));
    assert!(!source.join("nested").exists());
}

// `std::path::absolute` collapses `..` lexically on Windows, so there
// the path resolves without ambiguity; the refusal is the Unix reading.
#[cfg(unix)]
#[test]
fn a_parent_dir_segment_below_a_missing_directory_is_refused() {
    let tmp = tempfile::tempdir().unwrap();
    let source = source_in(tmp.path());
    let ambiguous = tmp.path().join("missing").join("..").join("out");

    let err = check_fresh_destination(&source, &ambiguous).unwrap_err();

    assert!(refusal_reason(err).contains("`..`"));
}

#[cfg(unix)]
mod symlinks {
    use super::*;
    use std::os::unix::fs::symlink;

    #[test]
    fn a_symlink_destination_is_refused_even_when_its_target_is_empty() {
        let tmp = tempfile::tempdir().unwrap();
        let source = source_in(tmp.path());
        let target = tmp.path().join("real-empty");
        std::fs::create_dir(&target).unwrap();
        let link = tmp.path().join("link");
        symlink(&target, &link).unwrap();

        let err = write_fresh_directory(&source, &link, unreachable_work).unwrap_err();

        assert!(refusal_reason(err).contains("symbolic link"));
        assert!(link.symlink_metadata().unwrap().file_type().is_symlink());
    }

    #[test]
    fn a_symlink_destination_spelled_with_a_trailing_slash_is_still_refused() {
        // `lstat("link/")` follows the link; unnormalised, the check
        // would pass and promotion would write through to the target.
        let tmp = tempfile::tempdir().unwrap();
        let source = source_in(tmp.path());
        let target = tmp.path().join("real-empty");
        std::fs::create_dir(&target).unwrap();
        let link = tmp.path().join("link");
        symlink(&target, &link).unwrap();
        let spelled = PathBuf::from(format!("{}/", link.display()));

        let err = write_fresh_directory(&source, &spelled, unreachable_work).unwrap_err();

        assert!(refusal_reason(err).contains("symbolic link"));
        assert!(target.is_dir(), "the link's target is untouched");
        assert_eq!(std::fs::read_dir(&target).unwrap().count(), 0);
    }

    #[test]
    fn a_dangling_symlink_destination_is_refused() {
        let tmp = tempfile::tempdir().unwrap();
        let source = source_in(tmp.path());
        let link = tmp.path().join("dangling");
        symlink(tmp.path().join("nowhere"), &link).unwrap();

        let err = check_fresh_destination(&source, &link).unwrap_err();

        assert!(refusal_reason(err).contains("symbolic link"));
    }

    #[test]
    fn the_source_reached_through_a_symlinked_parent_is_refused() {
        let tmp = tempfile::tempdir().unwrap();
        let source = source_in(tmp.path());
        let outer_alias = tmp.path().join("outer-alias");
        symlink(source.parent().unwrap(), &outer_alias).unwrap();

        let err = check_fresh_destination(&source, &outer_alias.join("source")).unwrap_err();

        assert!(refusal_reason(err).contains("resolves to the source vindex"));
    }

    #[test]
    fn a_missing_destination_inside_the_source_via_a_symlinked_parent_is_refused() {
        let tmp = tempfile::tempdir().unwrap();
        let source = source_in(tmp.path());
        let source_alias = tmp.path().join("source-alias");
        symlink(&source, &source_alias).unwrap();

        let err = check_fresh_destination(&source, &source_alias.join("nested")).unwrap_err();

        assert!(refusal_reason(err).contains("inside the source vindex"));
    }

    #[test]
    fn a_source_named_through_a_symlink_still_protects_the_real_directory() {
        let tmp = tempfile::tempdir().unwrap();
        let source = source_in(tmp.path());
        let source_alias = tmp.path().join("source-alias");
        symlink(&source, &source_alias).unwrap();

        let err = check_fresh_destination(&source_alias, &source).unwrap_err();

        assert!(refusal_reason(err).contains("resolves to the source vindex"));
    }
}

// ── Failure leaves nothing behind ────────────────────────────────────────

#[test]
fn a_failing_work_closure_leaves_no_destination_and_no_staging() {
    let tmp = tempfile::tempdir().unwrap();
    let source = source_in(tmp.path());
    let out = tmp.path().join("out");

    let err = write_fresh_directory(&source, &out, |staging| {
        write_one(staging)?;
        Err::<(), _>(VindexError::Parse("simulated bake failure".into()))
    })
    .unwrap_err();

    assert!(err.to_string().contains("simulated bake failure"));
    assert!(!out.exists(), "no partial output");
    assert!(staging_leftovers(tmp.path()).is_empty());
}

#[test]
fn a_failing_work_closure_keeps_an_existing_empty_destination_empty() {
    let tmp = tempfile::tempdir().unwrap();
    let source = source_in(tmp.path());
    let out = tmp.path().join("out");
    std::fs::create_dir(&out).unwrap();

    write_fresh_directory(&source, &out, |staging| {
        write_one(staging)?;
        Err::<(), _>(VindexError::Parse("simulated".into()))
    })
    .unwrap_err();

    assert!(out.is_dir());
    assert_eq!(std::fs::read_dir(&out).unwrap().count(), 0);
}

#[test]
fn a_destination_filled_during_the_work_is_not_deleted() {
    // The check passed on an empty directory; something else then put a
    // file there. Promotion must fail rather than remove it.
    let tmp = tempfile::tempdir().unwrap();
    let source = source_in(tmp.path());
    let out = tmp.path().join("out");
    std::fs::create_dir(&out).unwrap();

    let err = write_fresh_directory(&source, &out, |staging| {
        std::fs::write(out.join(KEPT_FILE), KEPT_BYTES)?;
        write_one(staging)
    })
    .unwrap_err();

    assert!(matches!(err, VindexError::OutputStaging { .. }), "{err}");
    assert_eq!(std::fs::read(out.join(KEPT_FILE)).unwrap(), KEPT_BYTES);
    assert!(staging_leftovers(tmp.path()).is_empty());
}

#[test]
fn a_leftover_staging_directory_is_stepped_over_not_removed() {
    // A crashed earlier run with a recycled process id can leave a
    // directory under the first staging name. It is not this call's to
    // delete.
    let tmp = tempfile::tempdir().unwrap();
    let source = source_in(tmp.path());
    let out = tmp.path().join("out");
    let leftover = tmp
        .path()
        .join(format!(".out{STAGING_INFIX}{}.0", std::process::id()));
    std::fs::create_dir(&leftover).unwrap();
    std::fs::write(leftover.join(KEPT_FILE), KEPT_BYTES).unwrap();

    write_fresh_directory(&source, &out, write_one).unwrap();

    assert_eq!(std::fs::read(leftover.join(KEPT_FILE)).unwrap(), KEPT_BYTES);
    assert!(out.join(WRITTEN_FILE).exists());
}

#[test]
fn a_destination_spelled_with_a_trailing_slash_is_written_normally() {
    let tmp = tempfile::tempdir().unwrap();
    let source = source_in(tmp.path());
    let out = tmp.path().join("out");
    let spelled = PathBuf::from(format!("{}/", out.display()));

    write_fresh_directory(&source, &spelled, write_one).unwrap();

    assert!(out.join(WRITTEN_FILE).exists());
}

#[test]
fn parents_created_for_a_failed_write_are_removed_again() {
    let tmp = tempfile::tempdir().unwrap();
    let source = source_in(tmp.path());
    let out = tmp.path().join("not").join("yet").join("out");

    write_fresh_directory(&source, &out, |staging| {
        write_one(staging)?;
        Err::<(), _>(VindexError::Parse("simulated".into()))
    })
    .unwrap_err();

    assert!(!tmp.path().join("not").exists(), "created parents removed");
}

#[test]
fn a_created_parent_that_gained_other_contents_is_kept() {
    let tmp = tempfile::tempdir().unwrap();
    let source = source_in(tmp.path());
    let parent = tmp.path().join("new-parent");
    let out = parent.join("out");

    write_fresh_directory(&source, &out, |_staging| {
        std::fs::write(parent.join(KEPT_FILE), KEPT_BYTES)?;
        Err::<(), _>(VindexError::Parse("simulated".into()))
    })
    .unwrap_err();

    assert_eq!(std::fs::read(parent.join(KEPT_FILE)).unwrap(), KEPT_BYTES);
    assert!(staging_leftovers(&parent).is_empty());
}

#[test]
fn staging_names_all_taken_is_an_error_that_removes_nothing() {
    let tmp = tempfile::tempdir().unwrap();
    let source = source_in(tmp.path());
    let out = tmp.path().join("out");
    let taken = |n: u32| {
        tmp.path()
            .join(format!(".out{STAGING_INFIX}{}.{n}", std::process::id()))
    };
    let mut attempts = 0;
    // Occupy names until the writer gives up; the bound is the module's.
    let err = loop {
        std::fs::create_dir(taken(attempts)).unwrap();
        attempts += 1;
        match write_fresh_directory(&source, &out, |_| Ok::<(), VindexError>(())) {
            Ok(()) => std::fs::remove_dir(&out).unwrap(),
            Err(e) => break e,
        }
    };

    assert!(matches!(err, VindexError::OutputStaging { .. }), "{err}");
    assert!(err.to_string().contains("earlier runs"), "{err}");
    assert!((0..attempts).all(|n| taken(n).is_dir()), "leftovers kept");
    assert!(!out.exists());
}

#[test]
fn a_destination_below_a_plain_file_is_an_error_and_the_file_is_kept() {
    let tmp = tempfile::tempdir().unwrap();
    let source = source_in(tmp.path());
    let parent_is_a_file = tmp.path().join("plain-file");
    std::fs::write(&parent_is_a_file, KEPT_BYTES).unwrap();

    let err = write_fresh_directory(&source, &parent_is_a_file.join("out"), unreachable_work)
        .unwrap_err();

    assert!(matches!(err, VindexError::OutputStaging { .. }), "{err}");
    assert_eq!(std::fs::read(&parent_is_a_file).unwrap(), KEPT_BYTES);
}

// ── A destination with no local source ───────────────────────────────────

#[test]
fn a_sourceless_check_judges_only_the_destination() {
    let tmp = tempfile::tempdir().unwrap();
    let empty = tmp.path().join("empty");
    std::fs::create_dir(&empty).unwrap();
    let full = tmp.path().join("full");
    std::fs::create_dir(&full).unwrap();
    std::fs::write(full.join(KEPT_FILE), KEPT_BYTES).unwrap();

    check_empty_destination(&tmp.path().join("absent")).unwrap();
    check_empty_destination(&empty).unwrap();
    let err = check_empty_destination(&full).unwrap_err();
    assert!(refusal_reason(err).contains("not empty"));
}

#[cfg(unix)]
#[test]
fn a_sourceless_check_still_refuses_a_symlink() {
    let tmp = tempfile::tempdir().unwrap();
    let link = tmp.path().join("link");
    std::os::unix::fs::symlink(tmp.path().join("nowhere"), &link).unwrap();

    let err = check_empty_destination(&link).unwrap_err();
    assert!(refusal_reason(err).contains("symbolic link"));
}
