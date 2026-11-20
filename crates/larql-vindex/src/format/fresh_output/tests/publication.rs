//! Gate B, B6 §3.6: the publication sequence (sync, validate, rename, sync
//! the parent), its fault points, and its two failure classes.

use std::path::{Path, PathBuf};

use super::super::{
    arm_publish_fault, write_fresh_directory, write_fresh_directory_validated, PublishFault,
    STAGING_INFIX,
};
use super::{source_in, staging_leftovers, write_one, WRITTEN_BYTES, WRITTEN_FILE};
use crate::error::VindexError;

const NESTED_DIR: &str = "segments/inner";
const NESTED_FILE: &str = "segments/inner/payload.bin";
const NESTED_BYTES: &[u8] = b"nested";

/// Every point before the rename.
const BEFORE_PUBLICATION: [PublishFault; 6] = [
    PublishFault::AfterStagingCreated,
    PublishFault::AfterWork,
    PublishFault::SyncFile,
    PublishFault::SyncStagingDir,
    PublishFault::Validate,
    PublishFault::Rename,
];

fn setup() -> (tempfile::TempDir, PathBuf, PathBuf) {
    let tmp = tempfile::tempdir().unwrap();
    let source = source_in(tmp.path());
    let out = tmp.path().join("outer").join("out");
    (tmp, source, out)
}

/// A file at the top, one in a nested directory, and (on Unix) a symlink,
/// which the sync must step over rather than follow.
fn write_tree(staging: &Path) -> Result<usize, VindexError> {
    std::fs::write(staging.join(WRITTEN_FILE), WRITTEN_BYTES)?;
    std::fs::create_dir_all(staging.join(NESTED_DIR))?;
    std::fs::write(staging.join(NESTED_FILE), NESTED_BYTES)?;
    #[cfg(unix)]
    std::os::unix::fs::symlink(WRITTEN_FILE, staging.join("link"))?;
    Ok(2)
}

fn parent(out: &Path) -> &Path {
    out.parent().unwrap()
}

#[test]
fn a_nested_tree_is_synced_validated_and_published_whole() {
    let (_tmp, source, out) = setup();
    let mut validated = None;

    let files = write_fresh_directory_validated(&source, &out, write_tree, |staging, value| {
        validated = Some((*value, std::fs::read(staging.join(NESTED_FILE))?));
        Ok::<(), VindexError>(())
    })
    .unwrap();

    assert_eq!(files, 2);
    assert_eq!(
        validated,
        Some((2, NESTED_BYTES.to_vec())),
        "validate saw the staged tree"
    );
    assert_eq!(std::fs::read(out.join(NESTED_FILE)).unwrap(), NESTED_BYTES);
    assert_eq!(
        std::fs::read(out.join(WRITTEN_FILE)).unwrap(),
        WRITTEN_BYTES
    );
    assert!(staging_leftovers(parent(&out)).is_empty());
}

#[test]
fn a_failure_before_the_rename_changes_nothing_at_every_point() {
    for point in BEFORE_PUBLICATION {
        let (_tmp, source, out) = setup();
        arm_publish_fault(&[point]);
        let err = write_fresh_directory(&source, &out, write_tree);
        arm_publish_fault(&[]);

        let err = err.expect_err(&format!("{point:?} must fail"));
        assert!(
            err.to_string()
                .contains(&format!("injected publish fault: {point:?}")),
            "{point:?}: {err}"
        );
        assert!(!out.exists(), "{point:?}: the destination was created");
        assert!(
            staging_leftovers(parent(&out)).is_empty(),
            "{point:?}: staging left"
        );
        assert_eq!(
            std::fs::read(source.join(super::KEPT_FILE)).unwrap(),
            super::KEPT_BYTES
        );
    }
}

#[test]
fn a_validation_that_fails_publishes_nothing() {
    let (_tmp, source, out) = setup();
    let err = write_fresh_directory_validated(&source, &out, write_one, |_, _| {
        Err(VindexError::Parse("the staged output is wrong".into()))
    })
    .unwrap_err();

    assert!(
        err.to_string().contains("the staged output is wrong"),
        "{err}"
    );
    assert!(!out.exists());
    assert!(staging_leftovers(parent(&out)).is_empty());
}

#[test]
fn a_failed_cleanup_names_the_staging_directory_it_left() {
    let (_tmp, source, out) = setup();
    arm_publish_fault(&[PublishFault::AfterWork, PublishFault::CleanUp]);
    let err = write_fresh_directory(&source, &out, write_one);
    arm_publish_fault(&[]);

    let leftovers = staging_leftovers(parent(&out));
    assert_eq!(leftovers.len(), 1, "{leftovers:?}");
    match err.unwrap_err() {
        VindexError::StagingLeftBehind { cause, staging, .. } => {
            assert!(cause.contains("AfterWork"), "{cause}");
            assert_eq!(staging, parent(&out).join(&leftovers[0]));
            assert!(staging.to_string_lossy().contains(STAGING_INFIX));
        }
        other => panic!("expected StagingLeftBehind, got {other}"),
    }
    assert!(!out.exists());
}

#[test]
fn a_failed_final_sync_is_reported_as_published_and_kept() {
    let (_tmp, source, out) = setup();
    arm_publish_fault(&[PublishFault::SyncParent]);
    let err = write_fresh_directory(&source, &out, write_one);
    arm_publish_fault(&[]);

    match err.unwrap_err() {
        VindexError::PublishedNotDurable { out: published, .. } => {
            assert_eq!(published, out);
        }
        other => panic!("expected PublishedNotDurable, got {other}"),
    }
    assert_eq!(
        std::fs::read(out.join(WRITTEN_FILE)).unwrap(),
        WRITTEN_BYTES
    );
    assert!(staging_leftovers(parent(&out)).is_empty());
}

#[test]
fn a_fault_fires_once_and_disarming_clears_every_fault() {
    let (tmp, source, out) = setup();
    arm_publish_fault(&[PublishFault::AfterWork]);
    write_fresh_directory(&source, &out, write_one).unwrap_err();
    // Fired once: the next publication runs normally.
    write_fresh_directory(&source, &out, write_one).unwrap();

    let second = tmp.path().join("outer").join("second");
    arm_publish_fault(&[PublishFault::Rename, PublishFault::SyncFile]);
    arm_publish_fault(&[]);
    write_fresh_directory(&source, &second, write_one).unwrap();
}

#[test]
fn the_two_publication_errors_say_what_happened() {
    let durable = VindexError::PublishedNotDurable {
        out: PathBuf::from("out.vindex"),
        source: std::io::Error::other("disk"),
    }
    .to_string();
    assert!(durable.contains("published out.vindex"), "{durable}");
    assert!(durable.contains("durability is not confirmed"), "{durable}");
    assert!(durable.contains("was not removed"), "{durable}");

    let left = VindexError::StagingLeftBehind {
        cause: "work failed".into(),
        staging: PathBuf::from(".out.tmp.1.0"),
        cleanup: std::io::Error::other("busy"),
    }
    .to_string();
    assert!(left.starts_with("work failed;"), "{left}");
    assert!(left.contains(".out.tmp.1.0"), "{left}");
    assert!(left.contains("remove it by hand"), "{left}");
}
