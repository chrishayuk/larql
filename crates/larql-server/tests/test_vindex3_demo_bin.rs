//! The `vindex3-demo` binary — the public explorer's boot-time
//! container generator. The encoding logic is the same fixture every
//! LQL/serve gate runs; what THIS binary owns is the argv contract and
//! the idempotence check, so those are what these tests run — as the
//! real subprocess, not a re-implementation.

use std::process::Command;

fn demo_bin() -> Command {
    Command::new(env!("CARGO_BIN_EXE_vindex3-demo"))
}

#[test]
fn no_output_dir_is_usage_error_2() {
    let out = demo_bin().output().expect("binary runs");
    assert_eq!(out.status.code(), Some(2));
    assert!(String::from_utf8_lossy(&out.stderr).contains("usage"));
}

#[test]
fn writes_a_container_once_and_is_idempotent_after() {
    let dest = tempfile::tempdir().unwrap();
    let out = demo_bin().arg(dest.path()).output().expect("binary runs");
    assert!(
        out.status.success(),
        "{}",
        String::from_utf8_lossy(&out.stderr)
    );
    assert!(dest.path().join("index.json").exists());
    assert!(dest.path().join("tokenizer.json").exists());
    assert!(String::from_utf8_lossy(&out.stdout).contains("written to"));

    // Boot-time regeneration: a present container is left alone.
    let again = demo_bin().arg(dest.path()).output().expect("binary runs");
    assert!(again.status.success());
    assert!(String::from_utf8_lossy(&again.stdout).contains("already present"));
}
