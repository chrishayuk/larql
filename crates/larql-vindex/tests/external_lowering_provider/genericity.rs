//! The plugin-plane claim at the LOWERING seam, checked against the
//! source rather than asserted in prose: **no identity this test crate
//! invented appears anywhere in production LARQL code.**
//!
//! The codec plane holds the same line for representations
//! (`external_attested_provider/genericity.rs`); this is its other half.
//! A lowering plane that works only for the providers its own build
//! happens to know is not a plane, it is a match arm with extra steps.
//! Both families and both diagnostic names are spelled in `provider.rs`
//! and must be spelled nowhere that ships.
//!
//! Test code inside `src/` is excluded — a fixture naming a fixture is
//! not a hardcoded architecture — and identified structurally rather than
//! by guessing at names.

use std::path::{Path, PathBuf};

use crate::provider::{FAMILY, NAME, SIBLING_FAMILY, SIBLING_NAME};

/// A family this build DOES ship, for the control below: the production
/// CPU provider's, spelled in `exec/production.rs`.
const SHIPPED_FAMILY: &str = "cpu-production";

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
                // `target/` is build output; `tests/`, `benches/` and
                // `examples/` are not shipped; a `tests` module inside
                // `src` is test code.
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
fn no_external_provider_identity_appears_in_production_larql() {
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
        for needle in [
            FAMILY,
            NAME,
            SIBLING_FAMILY,
            SIBLING_NAME,
            "OutsideProvider",
        ] {
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

/// And the scan can fail — a lowering family this build DOES ship is
/// found by the same walk, so an empty result above means "nothing
/// there" rather than "nothing looked at".
#[test]
fn the_scan_would_have_found_a_provider_that_was_there() {
    let sources = production_sources();
    let shipped = sources.iter().any(|p| {
        std::fs::read_to_string(p)
            .map(|t| t.contains(SHIPPED_FAMILY))
            .unwrap_or(false)
    });
    assert!(
        shipped,
        "the family `{SHIPPED_FAMILY}` this build DOES ship was not found, so the scan proves \
         nothing"
    );
}
