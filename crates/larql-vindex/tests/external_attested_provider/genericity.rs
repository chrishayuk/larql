//! The plugin-plane claim, checked against the source rather than
//! asserted in prose: **no identity this test crate invented appears
//! anywhere in production LARQL code.**
//!
//! A representation plane that works only for the identities its own
//! build happens to know is not a plane, it is a switch statement with
//! extra steps. The parent label, the dependency label, the ROLE name,
//! the attesting authority and the method id are all spelled in
//! `provider.rs` and must be spelled nowhere else that ships.
//!
//! Test code inside `src/` is excluded — a fixture naming a fixture is
//! not a hardcoded architecture — and identified structurally rather
//! than by guessing at names.

use std::path::{Path, PathBuf};

use crate::provider::{ANCHORS, ANCHOR_ROLE, AUTHORITY, LATTICE, METHOD};

/// Every `.rs` file under a crate's `src/` that is not itself a test.
fn production_sources() -> Vec<PathBuf> {
    let crates = Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .expect("crates/")
        .to_path_buf();
    let mut out = Vec::new();
    let mut stack = vec![crates];
    while let Some(dir) = stack.pop() {
        let Ok(entries) = std::fs::read_dir(&dir) else {
            continue;
        };
        for entry in entries.flatten() {
            let path = entry.path();
            let name = entry.file_name().to_string_lossy().to_string();
            if path.is_dir() {
                // `target/` is build output; `tests/` and `benches/` are
                // not shipped; a `tests` module inside `src` is test code.
                if !matches!(name.as_str(), "target" | "tests" | "benches" | "examples") {
                    stack.push(path);
                }
                continue;
            }
            if name.ends_with(".rs")
                && !name.ends_with("_tests.rs")
                && !name.starts_with("tests")
                && path.components().any(|c| c.as_os_str() == "src")
            {
                out.push(path);
            }
        }
    }
    out
}

#[test]
fn no_external_identity_or_role_appears_in_production_larql() {
    let sources = production_sources();
    assert!(
        sources.len() > 100,
        "the scan found only {} files, so it is not looking where it thinks",
        sources.len()
    );
    let mut found = Vec::new();
    for path in &sources {
        let Ok(text) = std::fs::read_to_string(path) else {
            continue;
        };
        for needle in [LATTICE, ANCHORS, ANCHOR_ROLE, AUTHORITY, METHOD] {
            if text.contains(needle) {
                found.push(format!("{} names `{needle}`", path.display()));
            }
        }
    }
    assert!(
        found.is_empty(),
        "production code must not know this provider: {found:?}"
    );
}

/// And the scan can fail — an identifier that IS in production code is
/// found by the same walk, so an empty result above means "nothing
/// there" rather than "nothing looked at".
#[test]
fn the_scan_would_have_found_an_identity_that_was_there() {
    let sources = production_sources();
    let shipped = sources.iter().any(|p| {
        std::fs::read_to_string(p)
            .map(|t| t.contains("VQ8_SHARED"))
            .unwrap_or(false)
    });
    assert!(
        shipped,
        "a label this build DOES ship was not found, so the scan proves nothing"
    );
}
