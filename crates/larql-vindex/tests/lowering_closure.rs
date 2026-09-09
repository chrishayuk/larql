//! **The closure over provider construction** — LOWERING-PLUGIN-1, L3.
//!
//! The fate of every provider construction site was frozen in
//! `docs/represent/forecasts/lowering-plugin-1-notes.json` before any was
//! touched: route through the registry, move into the one composition
//! site, or stay as an implementation detail inside a provider. This is
//! the assertion that the table holds — every non-test source file in the
//! workspace that still spells a provider constructor is one of the
//! files the table says may. A forgotten `ProductionBackend::new()`
//! anywhere else is the hidden default the programme exists to prevent,
//! and it goes red here rather than surviving as an unregistered path.
//!
//! Examples are excluded as tests are: not shipped, not production paths.

use std::path::{Path, PathBuf};

/// What a construction looks like in source.
const CONSTRUCTORS: [&str; 4] = [
    "ProductionBackend::new()",
    "ReferenceBackend::new()",
    "DevicePlanBackend::new(",
    "DevicePlanBackend::with_formats(",
];

/// The files the fate table permits to construct a provider, and why.
const PERMITTED: [(&str, &str); 3] = [
    (
        "larql-vindex/src/format/vindex3/opplan/exec/lowering.rs",
        "the registry's own `shipped()` — the one place the shipped providers are built, as a value",
    ),
    (
        "larql-vindex/src/format/vindex3/opplan/exec/device.rs",
        "the device provider's CPU glue — an implementation detail inside a provider, not authority over which provider runs",
    ),
    (
        "larql-cli/src/commands/primary/vindex3_cmd/prepare.rs",
        "the CLI's composition site `lowerings_for` — the device arms register the provider they configure",
    ),
];

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

fn constructs(path: &Path) -> Vec<&'static str> {
    let Ok(text) = std::fs::read_to_string(path) else {
        return Vec::new();
    };
    CONSTRUCTORS
        .into_iter()
        .filter(|needle| text.contains(needle))
        .collect()
}

fn permitted(path: &Path) -> bool {
    let shown = path.to_string_lossy().replace('\\', "/");
    PERMITTED.iter().any(|(suffix, _)| shown.ends_with(suffix))
}

#[test]
fn only_the_fate_table_s_permitted_files_construct_a_provider() {
    let sources = production_sources();
    assert!(
        sources.len() > 100,
        "the scan found only {} files, so it is not looking where it thinks",
        sources.len()
    );
    let strays: Vec<String> = sources
        .iter()
        .filter(|p| !permitted(p))
        .filter_map(|p| {
            let found = constructs(p);
            (!found.is_empty()).then(|| format!("{} constructs {found:?}", p.display()))
        })
        .collect();
    assert!(
        strays.is_empty(),
        "a provider is constructed outside the fate table's permitted sites — the hidden \
         default the programme exists to prevent: {strays:?}"
    );
}

/// The control: the scan finds every permitted site, so an empty stray
/// list above means "nothing there" rather than "nothing looked at".
#[test]
fn the_scan_finds_each_permitted_site_constructing_a_provider() {
    let sources = production_sources();
    for (suffix, why) in PERMITTED {
        let found = sources
            .iter()
            .find(|p| p.to_string_lossy().replace('\\', "/").ends_with(suffix))
            .unwrap_or_else(|| panic!("{suffix} is not in the scan"));
        assert!(
            !constructs(found).is_empty(),
            "{suffix} no longer constructs a provider; the fate table says it does ({why})"
        );
    }
}
