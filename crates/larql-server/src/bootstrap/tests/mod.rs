//! Unit tests for the bootstrap module — argv/manifest parsers,
//! vindex discovery, and load-option semantics.

use std::path::{Path, PathBuf};

use larql_vindex::format::filenames::INDEX_JSON;

use super::*;

#[test]
fn parse_ram_bytes_gb() {
    assert_eq!(parse_ram_bytes("24GB").unwrap(), 24 * 1024 * 1024 * 1024);
    assert_eq!(parse_ram_bytes("16gb").unwrap(), 16 * 1024 * 1024 * 1024);
}

#[test]
fn parse_ram_bytes_mb() {
    assert_eq!(parse_ram_bytes("4096MB").unwrap(), 4096 * 1024 * 1024);
}

#[test]
fn parse_ram_bytes_raw() {
    assert_eq!(parse_ram_bytes("1073741824").unwrap(), 1024 * 1024 * 1024);
}

#[test]
fn parse_ram_bytes_invalid() {
    assert!(parse_ram_bytes("notanumber").is_err());
}

fn unique_temp_dir(name: &str) -> PathBuf {
    let mut dir = std::env::temp_dir();
    dir.push(format!(
        "larql-server-bootstrap-{name}-{}-{}",
        std::process::id(),
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos()
    ));
    std::fs::create_dir_all(&dir).unwrap();
    dir
}

// ── Unit-manifest parser ─────────────────────────────────────────────
//
// The JSON shape the operator hands the server must round-trip through
// `parse_unit_manifest` into a deterministic ownership set.  Tests
// cover: well-formed multi-range manifest, bad layer key, reversed
// range, missing file.  The data shape is exercised end-to-end here so
// ownership-check and warmup loops can rely on it without having to
// re-validate.

fn write_units_file(dir: &Path, body: &str) -> PathBuf {
    let path = dir.join("units.json");
    std::fs::write(&path, body).unwrap();
    path
}

#[test]
fn parse_unit_manifest_round_trips_per_layer_ranges() {
    let dir = unique_temp_dir("units-ok");
    let path = write_units_file(
        &dir,
        r#"{"layer_experts": {"0": [[0,2]], "3": [[5,7],[10,10]]}}"#,
    );
    let units = parse_unit_manifest(&path).unwrap();
    // Layer 0: experts 0..=2 → (0,0), (0,1), (0,2)
    // Layer 3: experts 5..=7 + 10 → (3,5), (3,6), (3,7), (3,10)
    let expected: std::collections::HashSet<(usize, usize)> =
        [(0, 0), (0, 1), (0, 2), (3, 5), (3, 6), (3, 7), (3, 10)]
            .into_iter()
            .collect();
    assert_eq!(units, expected);
}

#[test]
fn parse_unit_manifest_rejects_non_numeric_layer_key() {
    let dir = unique_temp_dir("units-bad-layer");
    let path = write_units_file(&dir, r#"{"layer_experts": {"oops": [[0,2]]}}"#);
    let err = parse_unit_manifest(&path).unwrap_err();
    let msg = err.to_string();
    assert!(msg.contains("layer key 'oops'"), "got: {msg}");
}

#[test]
fn parse_unit_manifest_rejects_reversed_range() {
    let dir = unique_temp_dir("units-bad-range");
    let path = write_units_file(&dir, r#"{"layer_experts": {"0": [[5,2]]}}"#);
    let err = parse_unit_manifest(&path).unwrap_err();
    let msg = err.to_string();
    assert!(msg.contains("end (2) must be >= start (5)"), "got: {msg}");
}

#[test]
fn parse_unit_manifest_missing_file_reports_path() {
    let bogus = PathBuf::from("/nonexistent/larql-units-not-here.json");
    let err = parse_unit_manifest(&bogus).unwrap_err();
    let msg = err.to_string();
    assert!(
        msg.contains("read"),
        "msg should mention read failure: {msg}"
    );
    assert!(
        msg.contains(bogus.to_str().unwrap()),
        "msg should name path: {msg}"
    );
}

#[test]
fn parse_unit_manifest_accepts_empty_object() {
    // Operator may want to test the wiring without owning any units —
    // empty manifest should yield an empty set, not error.
    let dir = unique_temp_dir("units-empty");
    let path = write_units_file(&dir, r#"{"layer_experts": {}}"#);
    let units = parse_unit_manifest(&path).unwrap();
    assert!(units.is_empty());
}

#[test]
fn parse_layer_range_accepts_inclusive_cli_range() {
    assert_eq!(parse_layer_range("0-19").unwrap(), (0, 20));
    assert_eq!(parse_layer_range(" 2 - 2 ").unwrap(), (2, 3));
}

#[test]
fn parse_layer_range_rejects_bad_shapes() {
    assert!(parse_layer_range("0").is_err());
    assert!(parse_layer_range("x-2").is_err());
    assert!(parse_layer_range("2-x").is_err());
    assert!(parse_layer_range("3-2").is_err());
    assert!(parse_layer_range(&format!("0-{}", usize::MAX)).is_err());
}

#[test]
fn normalize_serve_alias_removes_subcommand() {
    let filtered = normalize_serve_alias(vec![
        "larql-server".into(),
        "serve".into(),
        "model.vindex".into(),
    ]);
    assert_eq!(filtered, vec!["larql-server", "model.vindex"]);
}

#[test]
fn normalize_serve_alias_leaves_non_alias_args_unchanged() {
    let args = vec!["larql-server".into(), "model.vindex".into()];
    assert_eq!(normalize_serve_alias(args.clone()), args);
}

#[test]
fn discover_vindexes_returns_sorted_dirs_with_index_json() {
    let dir = unique_temp_dir("discover");
    let b = dir.join("b.vindex");
    let a = dir.join("a.vindex");
    let ignored = dir.join("ignored.vindex");
    std::fs::create_dir_all(&b).unwrap();
    std::fs::create_dir_all(&a).unwrap();
    std::fs::create_dir_all(&ignored).unwrap();
    std::fs::write(b.join(INDEX_JSON), "{}").unwrap();
    std::fs::write(a.join(INDEX_JSON), "{}").unwrap();

    let paths = discover_vindexes(&dir);
    assert_eq!(paths, vec![a, b]);
    let _ = std::fs::remove_dir_all(dir);
}

#[test]
fn load_options_are_copyable() {
    let opts = LoadVindexOptions {
        v3_backend: Default::default(),
        no_infer: true,
        ffn_only: false,
        embed_only: false,
        layer_range: Some((0, 2)),
        max_gate_cache_layers: 1,
        max_q4k_cache_layers: 2,
        hnsw: Some(200),
        warmup_hnsw: true,
        release_mmap_after_request: true,
        expert_filter: Some((3, 4)),
        unit_filter: None,
        moe_remote: None,
    };
    let copied = opts.clone();
    assert!(copied.no_infer);
    assert_eq!(copied.layer_range, Some((0, 2)));
    assert_eq!(copied.expert_filter, Some((3, 4)));
}
