//! Gates for the physical compact: referenced files carried
//! byte-identically (the index and its hashes unchanged), junk
//! dropped and NAMED, and a container referencing a missing segment
//! refuses rather than emitting a smaller broken copy.

use crate::format::vindex3::fixtures::{encode_fixture_container, miniature_glimmer};

use super::compact_container;

fn fixture() -> tempfile::TempDir {
    let checkpoint = tempfile::tempdir().unwrap();
    let container = tempfile::tempdir().unwrap();
    encode_fixture_container(
        miniature_glimmer,
        checkpoint.path(),
        container.path(),
        "compact-fixture",
    );
    container
}

#[test]
fn compact_carries_referenced_bytes_and_drops_named_junk() {
    let container = fixture();
    // Junk: a crash-leftover partial segment and a stray artifact.
    std::fs::write(
        container.path().join("segments").join("orphan.bin"),
        b"junk",
    )
    .unwrap();
    std::fs::write(container.path().join("notes.txt"), b"scratch").unwrap();

    let out = tempfile::tempdir().unwrap();
    let report = compact_container(container.path(), out.path()).unwrap();

    assert_eq!(
        report.dropped,
        vec!["notes.txt".to_string(), "segments/orphan.bin".to_string()]
    );
    assert!(report.carried_segments > 0);
    assert!(!out.path().join("segments/orphan.bin").exists());
    assert!(!out.path().join("notes.txt").exists());

    // Byte identity of everything referenced — index included, so the
    // recorded hashes stay TRUE of the carried bytes.
    for name in ["index.json", "system_graph.json"] {
        assert_eq!(
            std::fs::read(container.path().join(name)).unwrap(),
            std::fs::read(out.path().join(name)).unwrap(),
            "{name} must be byte-identical"
        );
    }
    for entry in std::fs::read_dir(container.path().join("segments")).unwrap() {
        let entry = entry.unwrap();
        if entry.file_name() == "orphan.bin" {
            continue;
        }
        assert_eq!(
            std::fs::read(entry.path()).unwrap(),
            std::fs::read(out.path().join("segments").join(entry.file_name())).unwrap()
        );
    }
}

#[test]
fn compact_refuses_a_container_missing_a_referenced_segment() {
    let container = fixture();
    let victim = std::fs::read_dir(container.path().join("segments"))
        .unwrap()
        .next()
        .unwrap()
        .unwrap()
        .path();
    std::fs::remove_file(&victim).unwrap();
    let out = tempfile::tempdir().unwrap();
    let err = compact_container(container.path(), out.path())
        .expect_err("a broken container must refuse");
    assert!(err.to_string().contains("missing"), "{err}");
}

/// The name an index gives its weight-patch ledger. Compaction does not read a ledger, so these tests name one
/// in the index and stand a file in for it: what is under test is that the named file travels, or its absence
/// is refused.
const LEDGER: &str = "weight_patches.json";
const LEDGER_BYTES: &[u8] = b"{\"schema\":\"stand-in\"}";

/// Make the fixture's index name a ledger.
fn name_a_ledger(container: &std::path::Path) {
    let path = container.join("index.json");
    let mut index: serde_json::Value =
        serde_json::from_str(&std::fs::read_to_string(&path).unwrap()).unwrap();
    index["weight_patches"] = serde_json::json!(LEDGER);
    std::fs::write(&path, serde_json::to_string_pretty(&index).unwrap()).unwrap();
}

#[test]
fn compact_carries_the_ledger_the_index_names() {
    let container = fixture();
    name_a_ledger(container.path());
    std::fs::write(container.path().join(LEDGER), LEDGER_BYTES).unwrap();

    let out = tempfile::tempdir().unwrap();
    let report = compact_container(container.path(), out.path()).unwrap();

    assert!(
        !report.dropped.iter().any(|d| d == LEDGER),
        "a ledger the index names is referenced, not junk: {:?}",
        report.dropped
    );
    assert_eq!(
        std::fs::read(out.path().join(LEDGER)).unwrap(),
        LEDGER_BYTES
    );
}

#[test]
fn compact_refuses_a_container_missing_the_ledger_its_index_names() {
    let container = fixture();
    name_a_ledger(container.path());

    let out = tempfile::tempdir().unwrap();
    let err = compact_container(container.path(), out.path())
        .expect_err("an index naming a ledger the container lacks must refuse");
    assert!(err.to_string().contains("weight-patch ledger"), "{err}");
}
