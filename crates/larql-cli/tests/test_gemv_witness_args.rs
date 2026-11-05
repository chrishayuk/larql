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

#[test]
fn it_refuses_what_could_change_the_run_without_the_backend_saying_so() {
    let tmp = tempfile::tempdir().unwrap();
    let (w, l) = (tmp.path().join("w"), tmp.path().join("l"));
    let (w, l) = (w.to_str().unwrap(), l.to_str().unwrap());
    let base = ["--gemv-witness", w, "--logit-dump", l];
    for extra in [
        ["--plugin", "p"],
        ["--lowering", "cpu-production/v1"],
        ["--representation", "Q8_0"],
    ] {
        let args: Vec<&str> = base.iter().chain(extra.iter()).copied().collect();
        refused(&args, "can't describe a run");
    }
}
