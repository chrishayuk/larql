//! Tests for `larql list`'s per-entry identity, readiness and rendering.
//!
//! Every fixture lives in a temp dir handed to [`describe`] directly — no
//! test reads or writes the real `~/.cache`.

use std::path::{Path, PathBuf};

use larql_vindex::format::filenames::INDEX_JSON;
use larql_vindex::format::generation::ContainerGeneration;
use larql_vindex::format::vindex3::fixtures::{encode_fixture_container, miniature_glimmer};
use larql_vindex::format::vindex3::graph::GRAPH_SCHEMA;
use larql_vindex::format::vindex3::Vindex3Index;

use super::{describe, reason_summary, render, EntryStatus, ListRow};
use crate::commands::primary::cache::{CacheSource, CachedVindex};

/// The smallest `index.json` the VINDEX2 loader accepts.
const V2_INDEX: &str = r#"{"version":2,"model":"tiny-v2","family":"test","num_layers":3,
"hidden_size":16,"intermediate_size":32,"vocab_size":10,"embed_scale":1.0,"layers":[],
"down_top_k":1}"#;

fn entry(repo: &str, dir: &Path) -> CachedVindex {
    CachedVindex {
        repo: repo.to_string(),
        snapshot: dir.to_path_buf(),
        size_bytes: 2_500_000,
        source: CacheSource::Local,
    }
}

/// A real, self-encoded VINDEX3 container.
fn v3_container(work: &Path) -> PathBuf {
    let checkpoint = work.join("checkpoint");
    let container = work.join("model.vindex3");
    std::fs::create_dir_all(&checkpoint).unwrap();
    std::fs::create_dir_all(&container).unwrap();
    encode_fixture_container(miniature_glimmer, &checkpoint, &container, "list-fixture");
    container
}

fn index_of(container: &Path) -> Vindex3Index {
    serde_json::from_str(&std::fs::read_to_string(container.join(INDEX_JSON)).unwrap()).unwrap()
}

#[test]
fn a_v2_entry_reports_its_config_and_is_ready() {
    let tmp = tempfile::tempdir().unwrap();
    std::fs::write(tmp.path().join(INDEX_JSON), V2_INDEX).unwrap();
    let row = describe(&entry("tiny-v2", tmp.path()));
    assert_eq!(row.generation, Some(ContainerGeneration::V2));
    assert_eq!(row.layers, Some(3));
    assert_eq!(row.hidden, Some(16));
    assert_eq!(row.status, EntryStatus::Ready);
}

#[test]
fn a_v3_entry_reports_real_layers_and_hidden_and_is_ready() {
    let tmp = tempfile::tempdir().unwrap();
    let container = v3_container(tmp.path());
    let index = index_of(&container);
    let row = describe(&entry("model", &container));
    assert_eq!(row.generation, Some(ContainerGeneration::V3));
    assert!(index.num_layers > 0 && index.hidden_size > 0);
    assert_eq!(row.layers, Some(index.num_layers));
    assert_eq!(row.hidden, Some(index.hidden_size));
    assert_eq!(row.status, EntryStatus::Ready);
}

#[test]
fn a_v3_entry_with_an_old_graph_schema_is_unusable_and_says_why() {
    let tmp = tempfile::tempdir().unwrap();
    let container = v3_container(tmp.path());
    let index = index_of(&container);
    let graph_path = container.join(index.system_graph.as_deref().unwrap());
    let mut graph: serde_json::Value =
        serde_json::from_str(&std::fs::read_to_string(&graph_path).unwrap()).unwrap();
    let stale = GRAPH_SCHEMA - 1;
    graph["schema"] = serde_json::json!(stale);
    std::fs::write(&graph_path, serde_json::to_string(&graph).unwrap()).unwrap();

    let row = describe(&entry("old-model", &container));
    // Identity still reads: the index is fine, only the graph is stale.
    assert_eq!(row.layers, Some(index.num_layers));
    let EntryStatus::Unusable(why) = &row.status else {
        panic!("a stale graph must not list as ready");
    };
    let note = reason_summary(why).to_string();
    assert!(note.contains(&format!("schema v{stale}")), "{note}");
    assert!(note.contains(&format!("reads v{GRAPH_SCHEMA}")), "{note}");
    assert!(note.contains("re-encode"), "{note}");

    let table = render(std::slice::from_ref(&row));
    assert!(table.contains("unusable"), "{table}");
    assert!(table.contains("Unusable with this build:"), "{table}");
    assert!(table.contains(&format!("  old-model: {note}")), "{table}");
}

#[test]
fn a_v2_index_the_loader_refuses_is_unusable_with_identity_kept() {
    let tmp = tempfile::tempdir().unwrap();
    std::fs::write(
        tmp.path().join(INDEX_JSON),
        r#"{"version":2,"num_layers":34,"hidden_size":2560}"#,
    )
    .unwrap();
    let row = describe(&entry("partial", tmp.path()));
    assert_eq!(row.layers, Some(34));
    assert_eq!(row.hidden, Some(2560));
    assert!(matches!(row.status, EntryStatus::Unusable(_)));
}

#[test]
fn an_unidentifiable_entry_is_listed_with_unknown_fields() {
    let tmp = tempfile::tempdir().unwrap();
    std::fs::write(tmp.path().join(INDEX_JSON), "{}").unwrap();
    let row = describe(&entry("mystery", tmp.path()));
    assert_eq!(row.generation, None);
    assert_eq!(row.layers, None);
    assert!(matches!(row.status, EntryStatus::Unusable(_)));
    let table = render(&[row]);
    let line = table.lines().nth(1).unwrap();
    assert!(line.contains("mystery"), "{line}");
    assert!(line.contains(" - "), "{line}");
    assert!(line.ends_with("unusable"), "{line}");
}

#[test]
fn a_ready_table_has_no_notes_section() {
    let row = ListRow {
        source: "hf",
        repo: "owner/name".into(),
        size_bytes: 1_514_700_000,
        generation: Some(ContainerGeneration::V3),
        layers: Some(28),
        hidden: Some(1024),
        status: EntryStatus::Ready,
    };
    let table = render(&[row]);
    let mut lines = table.lines();
    let header = lines.next().unwrap();
    assert!(header.starts_with("SOURCE"), "{header}");
    assert!(
        header.contains("GEN") && header.ends_with("STATUS"),
        "{header}"
    );
    let line = lines.next().unwrap();
    for want in ["owner/name", "v3", "1514.7", "28", "1024", "ready"] {
        assert!(line.contains(want), "{want} missing from {line}");
    }
    assert!(lines.next().is_none(), "{table}");
}

#[test]
fn reason_summary_keeps_the_first_sentence_only() {
    assert_eq!(reason_summary("first part. remedy prose"), "first part");
    assert_eq!(reason_summary("no sentence break"), "no sentence break");
}

/// An `hf://` checkpoint encoded to VINDEX3 lists from the encoded cache
/// with its real identity, not as a row of zeros.
#[test]
fn an_encoded_entry_lists_as_v3_with_real_identity() {
    use crate::commands::primary::cache::scan_cached_vindexes_at_all;
    use crate::commands::primary::hf_auto::store::EncodedStore;

    let tmp = tempfile::tempdir().unwrap();
    let store = EncodedStore::new(tmp.path().join("encoded"));
    let repo = "acme/tiny-model";
    let commit = "0123456789abcdef0123456789abcdef01234567";
    let partial = store.begin(repo).unwrap();
    let checkpoint = tmp.path().join("checkpoint");
    std::fs::create_dir_all(&checkpoint).unwrap();
    std::fs::create_dir_all(partial.path()).unwrap();
    encode_fixture_container(
        miniature_glimmer,
        &checkpoint,
        partial.path(),
        "list-fixture",
    );
    let snapshot = partial
        .commit(&store.snapshot_dir(repo, commit).unwrap())
        .unwrap();
    let index = index_of(&snapshot);

    let entries = scan_cached_vindexes_at_all(
        &tmp.path().join("hub"),
        &tmp.path().join("local"),
        store.root(),
    )
    .unwrap();
    let [only] = entries.as_slice() else {
        panic!("one encoded entry expected: {entries:?}");
    };
    let row = describe(only);
    assert_eq!(row.source, CacheSource::Encoded.label());
    assert_eq!(row.repo, repo);
    assert_eq!(row.generation, Some(ContainerGeneration::V3));
    assert_eq!(row.layers, Some(index.num_layers));
    assert_eq!(row.hidden, Some(index.hidden_size));
    assert_eq!(row.status, EntryStatus::Ready);
    assert!(render(&[row]).contains("encoded"));
}

/// A V2 index at `level` declaring `files` by checksum, with one gate
/// layer of `GATE_LAYER_BYTES`.
fn v2_index_declaring(level: &str, files: &[&str]) -> String {
    let sums: serde_json::Map<String, serde_json::Value> = files
        .iter()
        .map(|f| (f.to_string(), serde_json::json!("digest")))
        .collect();
    let mut index: serde_json::Value = serde_json::from_str(V2_INDEX).unwrap();
    index["extract_level"] = serde_json::json!(level);
    index["checksums"] = serde_json::Value::Object(sums);
    index["layers"] = serde_json::json!([
        {"layer": 0, "num_features": 1, "offset": 0, "length": GATE_LAYER_BYTES}
    ]);
    index.to_string()
}

/// Gate bytes the fixture index declares for its one layer.
const GATE_LAYER_BYTES: usize = 32;

/// What every index in the published gemma-3-4b-it-vindex family declares
/// by checksum — the full extract and each slice alike.
const PUBLISHED_CHECKSUMS: [&str; 4] = [
    larql_vindex::format::filenames::DOWN_META_BIN,
    larql_vindex::format::filenames::EMBEDDINGS_BIN,
    larql_vindex::format::filenames::GATE_VECTORS_BIN,
    larql_vindex::format::filenames::NORMS_BIN,
];

/// A hub-cache entry for `dir`.
fn hub_entry(repo: &str, dir: &Path) -> CachedVindex {
    CachedVindex {
        source: CacheSource::HuggingFace,
        ..entry(repo, dir)
    }
}

#[test]
fn a_metadata_only_v2_download_is_unusable_and_says_how_to_finish_it() {
    use larql_vindex::format::filenames::DOWN_META_BIN;
    let tmp = tempfile::tempdir().unwrap();
    std::fs::write(
        tmp.path().join(INDEX_JSON),
        v2_index_declaring("inference", &PUBLISHED_CHECKSUMS),
    )
    .unwrap();
    std::fs::write(tmp.path().join(DOWN_META_BIN), b"meta").unwrap();

    let row = describe(&hub_entry("acme/tiny-vindex", tmp.path()));
    // Identity still reads from the index: only readiness changed.
    assert_eq!(row.layers, Some(3));
    let EntryStatus::Unusable(why) = &row.status else {
        panic!("a metadata-only download must not list as ready");
    };
    assert_eq!(
        why,
        "incomplete download: 2 of 2 file(s) required at extract level inference missing \
         (first: embeddings.bin) — re-run `larql pull acme/tiny-vindex`"
    );
    // The listing note keeps the whole reason: it has no sentence break.
    let table = render(std::slice::from_ref(&row));
    assert!(
        table.contains(&format!("  acme/tiny-vindex: {why}")),
        "{table}"
    );
}

/// Shaped like `chrishayuk/gemma-3-4b-it-vindex-client`: level
/// `attention`, the source's whole checksum map, and only the client
/// files on disk (no gate vectors, no down_meta).
#[test]
fn a_published_client_slice_with_stale_checksums_is_ready() {
    use larql_vindex::format::filenames::{EMBEDDINGS_BIN, NORMS_BIN};
    let tmp = tempfile::tempdir().unwrap();
    std::fs::write(
        tmp.path().join(INDEX_JSON),
        v2_index_declaring("attention", &PUBLISHED_CHECKSUMS),
    )
    .unwrap();
    for file in [EMBEDDINGS_BIN, NORMS_BIN] {
        std::fs::write(tmp.path().join(file), b"x").unwrap();
    }
    assert_eq!(
        describe(&hub_entry("acme/tiny-vindex-client", tmp.path())).status,
        EntryStatus::Ready
    );
}

#[test]
fn an_incomplete_local_vindex_is_not_told_to_re_pull() {
    use larql_vindex::format::filenames::GATE_VECTORS_BIN;
    let tmp = tempfile::tempdir().unwrap();
    std::fs::write(
        tmp.path().join(INDEX_JSON),
        v2_index_declaring("browse", &[GATE_VECTORS_BIN]),
    )
    .unwrap();
    std::fs::write(tmp.path().join(GATE_VECTORS_BIN), b"short").unwrap();
    let EntryStatus::Unusable(why) = describe(&entry("mine", tmp.path())).status else {
        panic!("a truncated gate file is not ready");
    };
    assert!(
        why.starts_with("incomplete vindex: gate_vectors.bin is 5 bytes"),
        "{why}"
    );
    assert!(!why.contains("larql pull"), "{why}");
}
