//! What `vindex3 exec --gemv-witness` refuses before starting a run.

use std::process::{Command, Output};

fn exec(args: &[&str]) -> Output {
    Command::new(env!("CARGO_BIN_EXE_larql"))
        .args(["vindex3", "exec", "/no/such/container", "--tokens", "1"])
        .args(args)
        .output()
        .unwrap()
}

fn refused(args: &[&str], expected: &str) {
    let out = exec(args);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert!(!out.status.success(), "{args:?} was accepted");
    assert!(stderr.contains(expected), "{args:?}: {stderr}");
}

#[test]
fn it_needs_a_logit_dump() {
    refused(&["--gemv-witness", "w"], "--logit-dump");
}

#[test]
fn it_refuses_a_bank_which_would_ignore_the_logit_dump() {
    refused(
        &[
            "--gemv-witness",
            "w",
            "--logit-dump",
            "l",
            "--bank",
            "b",
            "--dump-dir",
            "d",
        ],
        "cannot be used with",
    );
}

#[test]
fn it_refuses_a_limit_of_nothing() {
    refused(
        &[
            "--gemv-witness",
            "w",
            "--logit-dump",
            "l",
            "--gemv-witness-max-operands",
            "0",
        ],
        "records nothing",
    );
}

#[test]
fn it_refuses_an_existing_directory_before_opening_anything() {
    let tmp = tempfile::tempdir().unwrap();
    let logits = tmp.path().join("l");
    refused(
        &[
            "--gemv-witness",
            tmp.path().to_str().unwrap(),
            "--logit-dump",
            logits.to_str().unwrap(),
        ],
        "never overwritten",
    );
}
