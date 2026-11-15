//! Paths: everything a request names stays inside the workdir. `..`, absolute paths elsewhere, symlinks (to
//! files or directories, pointing out or in), hard links and non-regular files are refused, and nothing is
//! written outside.

use std::os::unix::fs::symlink;

use serde_json::json;

use super::super::paths::Workdir;
use super::*;

fn refused_path(svc: &mut Service, op: &str, path: &str, needle: &str) {
    let (kind, reason) = err(svc, op, json!({"path": path}));
    assert_eq!(kind, "refused", "{op} {path}: {reason}");
    assert!(reason.contains(needle), "{op} {path}: {reason}");
}

#[test]
fn names_outside_the_workdir_are_refused() {
    let fx = Fixture::new();
    let outside = fx.logs.path().join("outside.vlp");
    std::fs::copy(fx.patch("edit.vlp"), &outside).unwrap();
    let mut svc = fx.start(5);
    for op in ["apply_patch", "begin_patch"] {
        refused_path(&mut svc, op, "../outside.vlp", "`..`");
        refused_path(&mut svc, op, "sub/../../outside.vlp", "`..`");
        refused_path(&mut svc, op, "./edit.vlp", "`..`");
        refused_path(
            &mut svc,
            op,
            outside.to_str().unwrap(),
            "outside the workdir",
        );
        refused_path(&mut svc, op, "/etc/hosts", "outside the workdir");
        refused_path(&mut svc, op, "", "empty");
        refused_path(&mut svc, op, "a\\b.vlp", "`\\`");
        refused_path(
            &mut svc,
            op,
            fx.workdir().to_str().unwrap(),
            "the workdir itself",
        );
        let traversal = format!("{}/../outside.vlp", fx.workdir().display());
        refused_path(&mut svc, op, &traversal, "`..`");
    }
    // An absolute name under the workdir is its relative name.
    let inside = fx.workdir().join("edit.vlp");
    assert_eq!(
        ok(&mut svc, "apply_patch", json!({"path": inside}))["path"],
        "edit.vlp"
    );
}

#[test]
fn symlinks_and_hard_links_are_refused() {
    let fx = Fixture::new();
    let outside = fx.logs.path().join("outside.vlp");
    std::fs::copy(fx.patch("edit.vlp"), &outside).unwrap();
    symlink(&outside, fx.workdir().join("out-link.vlp")).unwrap();
    symlink(
        fx.workdir().join("edit.vlp"),
        fx.workdir().join("in-link.vlp"),
    )
    .unwrap();
    symlink(fx.logs.path(), fx.workdir().join("out-dir")).unwrap();
    std::fs::hard_link(&outside, fx.workdir().join("hard.vlp")).unwrap();
    std::fs::create_dir(fx.workdir().join("adir")).unwrap();
    let mut svc = fx.start(5);

    refused_path(&mut svc, "apply_patch", "out-link.vlp", "symlink");
    refused_path(&mut svc, "apply_patch", "in-link.vlp", "symlink");
    refused_path(&mut svc, "apply_patch", "out-dir/outside.vlp", "symlink");
    refused_path(&mut svc, "apply_patch", "hard.vlp", "hard links");
    refused_path(&mut svc, "apply_patch", "adir", "not a regular file");
    refused_path(&mut svc, "apply_patch", "edit.vlp/x", "not a directory");
    refused_path(&mut svc, "apply_patch", "missing.vlp", "does not exist");

    // Writes: a recording is never begun at a link, through a linked directory, or at a hard link.
    for name in [
        "out-link.vlp",
        "in-link.vlp",
        "out-dir/new.vlp",
        "hard.vlp",
        "adir",
    ] {
        let (kind, reason) = err(&mut svc, "begin_patch", json!({"path": name}));
        assert_eq!(kind, "refused", "{name}: {reason}");
    }
    assert_eq!(ok(&mut svc, "patches", json!({}))["recording"], Value::Null);
    assert!(!fx.logs.path().join("new.vlp").exists());
    assert_eq!(
        std::fs::read(&outside).unwrap(),
        std::fs::read(fx.workdir().join("edit.vlp")).unwrap()
    );
}

#[test]
fn a_fifo_is_refused_without_hanging() {
    let fx = Fixture::new();
    let fifo = fx.workdir().join("pipe.vlp");
    let c = std::ffi::CString::new(fifo.to_str().unwrap()).unwrap();
    // SAFETY: a valid NUL-terminated path.
    assert_eq!(unsafe { libc::mkfifo(c.as_ptr(), 0o644) }, 0);
    let mut svc = fx.start(5);
    refused_path(&mut svc, "apply_patch", "pipe.vlp", "not a regular file");
    refused_path(
        &mut svc,
        "begin_patch",
        "pipe.vlp",
        "`pipe.vlp` in the workdir",
    );
}

#[test]
fn a_save_creates_its_directories_and_a_failed_save_leaves_nothing() {
    let fx = Fixture::new();
    let mut svc = fx.start(5);
    ok(&mut svc, "begin_patch", json!({"path": "a/b/c.vlp"}));
    ok(
        &mut svc,
        "update_weights",
        json!({"tensor": TENSOR, "row": ROW, "cols": [0], "delta": 2.0}),
    );
    ok(&mut svc, "save_patch", json!({}));
    assert!(fx.workdir().join("a/b/c.vlp").is_file());

    // A write target that is discarded removes the file it created, and only that.
    let workdir = Workdir::open(fx.workdir()).unwrap();
    let created = workdir.open_for_write("fresh.vlp").unwrap();
    assert!(created.created);
    created.discard();
    assert!(!fx.workdir().join("fresh.vlp").exists());
    let existing = workdir.open_for_write("a/b/c.vlp").unwrap();
    assert!(!existing.created);
    existing.discard();
    assert!(fx.workdir().join("a/b/c.vlp").is_file());
}

#[test]
fn a_workdir_that_is_not_a_directory_is_refused() {
    let fx = Fixture::new();
    let file = fx.workdir().join("f");
    std::fs::write(&file, "x").unwrap();
    assert!(Workdir::open(&file)
        .err()
        .unwrap()
        .contains("not a directory"));
    assert!(Workdir::open(&fx.workdir().join("missing")).is_err());
}

#[test]
fn reads_are_limited_and_a_read_through_a_file_component_is_refused() {
    let fx = Fixture::new();
    std::fs::write(fx.workdir().join("big"), vec![b'x'; 100]).unwrap();
    let workdir = Workdir::open(fx.workdir()).unwrap();
    assert!(workdir
        .read("big", 10)
        .unwrap_err()
        .reason
        .contains("max_patch_bytes"));
    assert_eq!(workdir.read("big", 100).unwrap().len(), 100);
    let f = workdir.open_for_write("big/inner").err().unwrap();
    assert!(f.reason.contains("not a directory"), "{}", f.reason);
}

#[test]
fn a_destination_swapped_for_a_link_after_begin_is_refused_at_save_and_the_recording_kept() {
    let fx = Fixture::new();
    let outside = fx.logs.path().join("victim");
    std::fs::write(&outside, "keep me").unwrap();
    let mut svc = fx.start(5);
    ok(&mut svc, "begin_patch", json!({"path": "late.vlp"}));
    ok(
        &mut svc,
        "update_weights",
        json!({"tensor": TENSOR, "row": ROW, "cols": [0], "delta": 2.0}),
    );
    symlink(&outside, fx.workdir().join("late.vlp")).unwrap();
    let (kind, reason) = err(&mut svc, "save_patch", json!({}));
    assert_eq!(kind, "refused", "{reason}");
    assert_eq!(std::fs::read_to_string(&outside).unwrap(), "keep me");
    assert_eq!(
        ok(&mut svc, "patches", json!({}))["recording"]["operations"],
        1
    );
    std::fs::remove_file(fx.workdir().join("late.vlp")).unwrap();
    assert_eq!(ok(&mut svc, "save_patch", json!({}))["path"], "late.vlp");
}

#[test]
fn a_link_planted_in_staging_is_removed_never_followed() {
    let fx = Fixture::new();
    fx.patch("edit.vlp");
    let victim = fx.logs.path().join("victim");
    std::fs::write(&victim, "keep me").unwrap();
    let mut svc = fx.start(5);
    // At the staged copy's place, before an apply.
    symlink(&victim, svc.staged_path("edit.vlp")).unwrap();
    ok(&mut svc, "apply_patch", json!({"path": "edit.vlp"}));
    assert_eq!(std::fs::read_to_string(&victim).unwrap(), "keep me");
    // The staged copy is deleted when the patch is removed.
    ok(&mut svc, "remove_patch", json!({"path": "edit.vlp"}));
    assert!(svc.staged_path("edit.vlp").symlink_metadata().is_err());

    // At a recording's staged place, between begin and save.
    ok(&mut svc, "begin_patch", json!({"path": "rec.vlp"}));
    ok(
        &mut svc,
        "update_weights",
        json!({"tensor": TENSOR, "row": ROW, "cols": [0], "delta": 2.0}),
    );
    symlink(&victim, svc.staged_path("rec.vlp")).unwrap();
    ok(&mut svc, "save_patch", json!({}));
    assert_eq!(std::fs::read_to_string(&victim).unwrap(), "keep me");
    assert!(fx.workdir().join("rec.vlp").is_file());
    assert!(svc.staged_path("rec.vlp").symlink_metadata().is_err());
}

#[test]
fn a_staged_file_that_is_not_the_one_written_is_refused_at_save() {
    // A hard link where the staged file goes: LARQL's write replaces it, the read-back is still confined.
    let fx = Fixture::new();
    let workdir = Workdir::open(fx.logs.path()).unwrap();
    std::fs::write(fx.logs.path().join("a"), "x").unwrap();
    std::fs::hard_link(fx.logs.path().join("a"), fx.logs.path().join("b")).unwrap();
    assert!(workdir
        .read("b", 10)
        .unwrap_err()
        .reason
        .contains("hard links"));
    workdir.clear("b").unwrap();
    workdir.clear("b").unwrap();
    assert!(workdir.create_new("a").is_err());
    std::fs::write(fx.logs.path().join("f"), "x").unwrap();
    assert!(workdir.clear("f/g").is_err());
}
