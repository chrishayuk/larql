//! **The closure over continuation providers** — CONTINUATION-PLUGIN-1, C3.
//!
//! The forecast (`docs/represent/forecasts/continuation-plugin-1.json`)
//! makes the closure executable policy rather than a migration of the 16
//! construction sites known at baseline: production code may not NAME a
//! built-in continuation provider outside the files below. Banning the
//! type name rather than listed constructor spellings also catches struct
//! literals and an inferred `Default::default()`, so a seventeenth site
//! cannot appear unnoticed — every path that holds conversation state must
//! reach its provider through a selection.
//!
//! Tests and examples are excluded, as in the lowering closure: they are
//! not production paths, and they may name the provider they exercise.

use std::path::{Path, PathBuf};

/// The built-in providers' type names.
const BUILT_INS: [&str; 3] = ["RowKvState", "CanonicalKvState", "WindowKvState"];

/// Where a built-in may be named, and why.
const PERMITTED: [(&str, &str); 5] = [
    (
        "larql-vindex/src/format/vindex3/opplan/exec/kv.rs",
        "RowKvState's own module: its definition, identity and RowFactory",
    ),
    (
        "larql-kv/src/vindex3/mod.rs",
        "CanonicalKvState's own module: its definition and identity",
    ),
    (
        "larql-kv/src/vindex3/window.rs",
        "WindowKvState's own module (CONTINUATION-WINDOW-1): its definition, identity and WindowFactory",
    ),
    (
        "larql-kv/src/vindex3/registry.rs",
        "CanonicalFactory and shipped_continuations — the one place the shipped providers are registered, as a value",
    ),
    (
        "larql-kv/src/lib.rs",
        "the provider crate's public surface re-exports its own type (the from_cache/into_cache bridge to KvEngine machinery)",
    ),
];

/// Every `.rs` file under a crate's `src/` below `crates` that is not
/// itself a test.
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

fn workspace_crates() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .expect("crates/")
        .to_path_buf()
}

/// The built-in names `text` mentions as identifiers in CODE — line
/// comments stripped, so documentation may still explain history.
fn named(text: &str) -> Vec<&'static str> {
    let ident = |c: char| c.is_ascii_alphanumeric() || c == '_';
    let code: String = text
        .lines()
        .map(|line| line.split("//").next().unwrap_or(""))
        .collect::<Vec<_>>()
        .join("\n");
    BUILT_INS
        .into_iter()
        .filter(|name| {
            code.match_indices(name).any(|(at, _)| {
                let before = code[..at].chars().next_back();
                let after = code[at + name.len()..].chars().next();
                !before.is_some_and(ident) && !after.is_some_and(ident)
            })
        })
        .collect()
}

fn permitted(path: &Path) -> bool {
    let shown = path.to_string_lossy().replace('\\', "/");
    PERMITTED.iter().any(|(suffix, _)| shown.ends_with(suffix))
}

fn strays(crates: &Path) -> Vec<String> {
    production_sources(crates)
        .iter()
        .filter(|p| !permitted(p))
        .filter_map(|p| {
            let text = std::fs::read_to_string(p).ok()?;
            let found = named(&text);
            (!found.is_empty()).then(|| format!("{} names {found:?}", p.display()))
        })
        .collect()
}

#[test]
fn production_code_names_no_built_in_continuation_provider() {
    let crates = workspace_crates();
    let sources = production_sources(&crates);
    assert!(
        sources.len() > 100,
        "the scan found only {} files, so it is not looking where it thinks",
        sources.len()
    );
    let strays = strays(&crates);
    assert!(
        strays.is_empty(),
        "a built-in continuation provider is named outside its permitted sites — production \
         code must reach its provider through a selection: {strays:?}"
    );
}

/// The control in one direction: every permitted file still names a
/// built-in, so an entry cannot outlive its reason unnoticed.
#[test]
fn the_scan_finds_each_permitted_site_naming_a_built_in() {
    let sources = production_sources(&workspace_crates());
    for (suffix, why) in PERMITTED {
        let found = sources
            .iter()
            .find(|p| p.to_string_lossy().replace('\\', "/").ends_with(suffix))
            .unwrap_or_else(|| panic!("{suffix} is not in the scan"));
        let text = std::fs::read_to_string(found).unwrap();
        assert!(
            !named(&text).is_empty(),
            "{suffix} no longer names a built-in; its permission is stale ({why})"
        );
    }
}

/// The control in the other: the whole scan goes RED on a seeded
/// violation, in each spelling the name ban exists to catch — so an empty
/// stray list means "nothing there", not "nothing that could be seen".
#[test]
fn a_seeded_violation_turns_the_scan_red() {
    let seeds = [
        "fn f() { let kv = CanonicalKvState::new(); }",
        "fn f() { let kv = RowKvState { layers: vec![] }; }",
        "fn f() { let kv: CanonicalKvState = Default::default(); }",
        "use larql_vindex::format::vindex3::opplan::exec::kv::RowKvState;",
    ];
    for seed in seeds {
        let dir = tempfile::tempdir().unwrap();
        let src = dir.path().join("larql-seeded").join("src");
        std::fs::create_dir_all(&src).unwrap();
        std::fs::write(src.join("serve.rs"), seed).unwrap();
        let found = strays(dir.path());
        assert_eq!(found.len(), 1, "{seed}: {found:?}");
    }
    // What the ban deliberately lets through: prose, and names that merely
    // contain a built-in's name.
    assert!(named("// the CanonicalKvState used to live here").is_empty());
    assert!(named("struct RecordingRowKvStateProbe;").is_empty());
}
