//! **C6 of CONTINUATION-PLUGIN-1, through the `larql` binary.**
//!
//! The in-process gate (`vindex3_cmd/tests/continuation_plugin.rs`) holds
//! the loaded provider to the C5 journey bit for bit. This one runs the
//! built binary as a user would: `larql run --plugin <dylib>
//! --continuation hostile-test-provider/v77`. The run must report the
//! identity it resolved and generate exactly what `canonical/v1` does,
//! and the flag's refusals must reach the command line. The CLI has no
//! verb that saves a handoff and resumes it, which is why the resume half
//! lives in the in-process gate (recorded as C6's honest limit).
#![cfg(unix)]

#[path = "support/continuation_fixture.rs"]
mod fixture;

use std::path::Path;
use std::process::{Command, Output};

use fixture::{fixture_dylib, glimmer_container, HOSTILE};

/// A prompt that is its own id list under the fixture tokenizer.
const PROMPT: &str = "[1] [2] [3]";
/// Tokens to generate: enough to decode past the prompt.
const NEW_TOKENS: &str = "6";

fn larql(container: &Path, extra: &[&str]) -> Output {
    Command::new(env!("CARGO_BIN_EXE_larql"))
        .arg("run")
        .arg(container)
        .args([PROMPT, "-n", NEW_TOKENS, "--verbose"])
        .args(extra)
        .output()
        .expect("run larql")
}

fn text(bytes: &[u8]) -> String {
    String::from_utf8_lossy(bytes).into_owned()
}

#[test]
fn larql_run_loads_the_provider_and_generates_what_canonical_does() {
    let root = tempfile::tempdir().unwrap();
    let container = glimmer_container(root.path());
    let dylib = fixture_dylib();
    let dylib = dylib.to_str().unwrap();

    let loaded = larql(&container, &["--plugin", dylib, "--continuation", HOSTILE]);
    let (out, err) = (text(&loaded.stdout), text(&loaded.stderr));
    assert!(loaded.status.success(), "stderr:\n{err}");
    assert!(err.contains(&format!("continuations [{HOSTILE}]")), "{err}");
    assert!(err.contains(&format!("continuation={HOSTILE}")), "{err}");

    let canonical = larql(&container, &["--engine", "standard"]);
    let canonical_err = text(&canonical.stderr);
    assert!(canonical.status.success(), "stderr:\n{canonical_err}");
    assert!(
        canonical_err.contains("continuation=canonical/v1"),
        "{canonical_err}"
    );
    assert!(!out.trim().is_empty(), "the run generated nothing");
    assert_eq!(
        out,
        text(&canonical.stdout),
        "generation differs from canonical/v1"
    );
}

#[test]
fn an_unknown_continuation_is_refused_naming_the_loaded_ones_too() {
    let root = tempfile::tempdir().unwrap();
    let container = glimmer_container(root.path());
    let dylib = fixture_dylib();
    let refused = larql(
        &container,
        &[
            "--plugin",
            dylib.to_str().unwrap(),
            "--continuation",
            "nobody/v1",
        ],
    );
    let err = text(&refused.stderr);
    assert!(!refused.status.success(), "an unknown provider ran:\n{err}");
    assert!(err.contains("nobody/v1"), "{err}");
    assert!(
        err.contains(HOSTILE) && err.contains("canonical/v1"),
        "{err}"
    );
}

#[test]
fn continuation_with_engine_is_refused_at_the_command_line() {
    let root = tempfile::tempdir().unwrap();
    let container = glimmer_container(root.path());
    let refused = larql(&container, &["--continuation", "row/v1", "--engine", "row"]);
    let err = text(&refused.stderr);
    assert!(!refused.status.success(), "{err}");
    assert!(
        err.contains("--continuation") && err.contains("--engine"),
        "{err}"
    );
}

#[test]
fn a_continuation_option_without_a_continuation_is_refused() {
    let root = tempfile::tempdir().unwrap();
    let container = glimmer_container(root.path());
    let refused = larql(&container, &["--continuation-option", "bits=4"]);
    let err = text(&refused.stderr);
    assert!(!refused.status.success(), "{err}");
    assert!(
        err.contains("--continuation-option needs --continuation"),
        "{err}"
    );
}
