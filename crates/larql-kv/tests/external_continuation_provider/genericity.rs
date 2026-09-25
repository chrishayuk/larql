//! F4 for the external provider: nothing that ships knows it exists.
//!
//! A source scan over every non-test `.rs` under any crate's `src/` for
//! the external family, its type names and its constants. The positive
//! control runs the SAME scanner over the SAME files for the family this
//! build does ship (`canonical`) and must find it; otherwise an empty
//! result would say only that the scanner read nothing.

use std::path::{Path, PathBuf};

use super::provider::HOSTILE_FAMILY;

/// The external provider's names, as its own source spells them.
const EXTERNAL_NAMES: [&str; 3] = [HOSTILE_FAMILY, "HostileRows", "HostileFactory"];
/// A family this build ships, spelled as a string literal in production.
const SHIPPED_FAMILY_LITERAL: &str = "\"canonical\"";
/// A scan that reads fewer files than this read too little to mean
/// anything (the workspace has several hundred).
const MIN_FILES_SCANNED: usize = 100;

fn workspace_crates() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .expect("larql-kv sits under crates/")
        .to_path_buf()
}

/// Every `.rs` file under a crate's `src/` that is not itself a test:
/// no `tests`/`benches`/`examples` directory, no `tests*.rs` or
/// `*_tests.rs` file.
fn production_sources(crates: &Path) -> Vec<PathBuf> {
    let mut out = Vec::new();
    let mut stack = vec![crates.to_path_buf()];
    while let Some(dir) = stack.pop() {
        let Ok(entries) = std::fs::read_dir(&dir) else {
            continue;
        };
        for entry in entries.flatten() {
            let path = entry.path();
            let name = entry.file_name().to_string_lossy().to_string();
            if path.is_dir() {
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

/// The files among `sources` containing `needle`.
fn containing(sources: &[PathBuf], needle: &str) -> Vec<PathBuf> {
    sources
        .iter()
        .filter(|path| {
            std::fs::read_to_string(path)
                .map(|text| text.contains(needle))
                .unwrap_or(false)
        })
        .cloned()
        .collect()
}

#[test]
fn no_production_source_names_the_external_provider() {
    let sources = production_sources(&workspace_crates());
    assert!(
        sources.len() >= MIN_FILES_SCANNED,
        "scanned only {} files",
        sources.len()
    );
    for name in EXTERNAL_NAMES {
        let hits = containing(&sources, name);
        assert!(
            hits.is_empty(),
            "production source names `{name}`: {hits:?}"
        );
    }
}

#[test]
fn control_the_same_scan_finds_the_family_this_build_ships() {
    let sources = production_sources(&workspace_crates());
    let hits = containing(&sources, SHIPPED_FAMILY_LITERAL);
    assert!(
        !hits.is_empty(),
        "the scanner found no {SHIPPED_FAMILY_LITERAL} in {} files: it read nothing",
        sources.len()
    );
}
