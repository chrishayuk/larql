//! Completeness against a VINDEX2 index's own declarations — fixtures
//! only, never a real cache.

use std::collections::HashMap;

use super::*;
use crate::config::VindexLayerInfo;
use crate::format::filenames::DOWN_META_BIN;

/// Bytes per gate layer slice in the fixtures.
const SLICE: u64 = 64;

/// Every file the published gemma-3-4b-it-vindex family declares by
/// checksum — the full extract and each of its slices alike.
const PUBLISHED_CHECKSUMS: [&str; 4] = [DOWN_META_BIN, EMBEDDINGS_BIN, GATE_VECTORS_BIN, NORMS_BIN];

fn layer(layer: usize) -> VindexLayerInfo {
    VindexLayerInfo {
        layer,
        num_features: 1,
        offset: layer as u64 * SLICE,
        length: SLICE,
        num_experts: None,
        num_features_per_expert: None,
    }
}

/// An index at `level` declaring `files` and `layers` gate slices.
fn config(level: ExtractLevel, files: &[&str], layers: usize) -> VindexConfig {
    let sums: HashMap<String, String> = files
        .iter()
        .map(|f| (f.to_string(), "digest".to_string()))
        .collect();
    VindexConfig {
        extract_level: level,
        checksums: Some(sums),
        layers: (0..layers).map(layer).collect(),
        ..VindexConfig::default()
    }
}

fn write(dir: &Path, name: &str, len: u64) {
    std::fs::write(dir.join(name), vec![0u8; len as usize]).unwrap();
}

#[test]
fn each_level_requires_what_every_shape_claiming_it_carries() {
    assert!(files_required_at(ExtractLevel::Browse).is_empty());
    assert_eq!(files_required_at(ExtractLevel::Attention), [NORMS_BIN]);
    for level in [ExtractLevel::Inference, ExtractLevel::All] {
        assert_eq!(files_required_at(level), [EMBEDDINGS_BIN, NORMS_BIN]);
    }
}

#[test]
fn a_full_extract_holding_its_required_files_is_complete() {
    let dir = tempfile::tempdir().unwrap();
    let cfg = config(ExtractLevel::Inference, &PUBLISHED_CHECKSUMS, 2);
    write(dir.path(), EMBEDDINGS_BIN, 1);
    write(dir.path(), NORMS_BIN, 1);
    write(dir.path(), GATE_VECTORS_BIN, 2 * SLICE);
    assert_eq!(check_declared_files(dir.path(), &cfg), Ok(()));
}

#[test]
fn a_metadata_only_download_names_what_is_missing() {
    // The shape a metadata-only fetch of the full extract leaves:
    // down_meta arrived, the tensors did not.
    let dir = tempfile::tempdir().unwrap();
    let cfg = config(ExtractLevel::Inference, &PUBLISHED_CHECKSUMS, 2);
    write(dir.path(), DOWN_META_BIN, 1);
    let report = check_declared_files(dir.path(), &cfg).unwrap_err();
    assert_eq!(report.required, 2);
    assert_eq!(report.missing, [EMBEDDINGS_BIN, NORMS_BIN]);
    assert_eq!(
        report.to_string(),
        "2 of 2 file(s) required at extract level inference missing (first: embeddings.bin)"
    );
}

/// The published slices, as their index.json and files really are: each
/// keeps its source's whole checksum map and ships a subset.
#[test]
fn published_slices_are_judged_by_their_level_not_their_stale_checksums() {
    let slices: [(&str, ExtractLevel, &[&str]); 5] = [
        (
            "client",
            ExtractLevel::Attention,
            &[EMBEDDINGS_BIN, NORMS_BIN],
        ),
        ("attn", ExtractLevel::Attention, &[NORMS_BIN]),
        (
            "server",
            ExtractLevel::Browse,
            &[EMBEDDINGS_BIN, NORMS_BIN, GATE_VECTORS_BIN, DOWN_META_BIN],
        ),
        (
            "browse",
            ExtractLevel::Browse,
            &[EMBEDDINGS_BIN, GATE_VECTORS_BIN, DOWN_META_BIN],
        ),
        ("embed", ExtractLevel::Browse, &[EMBEDDINGS_BIN]),
    ];
    for (preset, level, ships) in slices {
        let dir = tempfile::tempdir().unwrap();
        let cfg = config(level, &PUBLISHED_CHECKSUMS, 2);
        for file in ships {
            let len = if *file == GATE_VECTORS_BIN {
                2 * SLICE
            } else {
                1
            };
            write(dir.path(), file, len);
        }
        assert_eq!(check_declared_files(dir.path(), &cfg), Ok(()), "{preset}");
    }
}

#[test]
fn an_attention_slice_without_its_norms_is_incomplete() {
    let dir = tempfile::tempdir().unwrap();
    let cfg = config(ExtractLevel::Attention, &PUBLISHED_CHECKSUMS, 0);
    let report = check_declared_files(dir.path(), &cfg).unwrap_err();
    assert_eq!(report.missing, [NORMS_BIN]);
}

#[test]
fn a_required_file_the_index_does_not_declare_is_not_required() {
    // No checksum, no evidence the writer produced it.
    let dir = tempfile::tempdir().unwrap();
    let cfg = config(ExtractLevel::Inference, &[GATE_VECTORS_BIN], 0);
    assert!(required_files(&cfg).is_empty());
    assert_eq!(check_declared_files(dir.path(), &cfg), Ok(()));
}

#[test]
fn a_gate_file_shorter_than_its_declared_layers_is_truncated_at_any_level() {
    let dir = tempfile::tempdir().unwrap();
    let cfg = config(ExtractLevel::Browse, &[GATE_VECTORS_BIN], 3);
    write(dir.path(), GATE_VECTORS_BIN, 3 * SLICE - 1);
    let report = check_declared_files(dir.path(), &cfg).unwrap_err();
    assert!(report.missing.is_empty());
    assert_eq!(
        report.truncated,
        [(GATE_VECTORS_BIN.to_string(), 3 * SLICE - 1, 3 * SLICE)]
    );
    assert!(
        report.to_string().contains("declares at least 192"),
        "{report}"
    );
}

#[test]
fn missing_and_truncated_are_both_reported() {
    let dir = tempfile::tempdir().unwrap();
    let cfg = config(ExtractLevel::Attention, &[NORMS_BIN], 1);
    write(dir.path(), GATE_VECTORS_BIN, 1);
    let text = check_declared_files(dir.path(), &cfg)
        .unwrap_err()
        .to_string();
    assert!(
        text.contains("1 of 1") && text.contains("; gate_vectors.bin is 1 bytes"),
        "{text}"
    );
}

#[test]
fn an_index_without_checksums_requires_nothing() {
    let dir = tempfile::tempdir().unwrap();
    let cfg = VindexConfig {
        extract_level: ExtractLevel::All,
        layers: vec![layer(0)],
        ..VindexConfig::default()
    };
    assert!(declared_files(&cfg).is_empty());
    assert_eq!(check_declared_files(dir.path(), &cfg), Ok(()));
}

#[test]
fn a_dangling_snapshot_link_reads_as_missing() {
    // A hub snapshot is symlinks into blobs/; a link whose blob never
    // arrived must not count as the file.
    let dir = tempfile::tempdir().unwrap();
    #[cfg(unix)]
    std::os::unix::fs::symlink(dir.path().join("blobs/absent"), dir.path().join(NORMS_BIN))
        .unwrap();
    let cfg = config(ExtractLevel::Attention, &[NORMS_BIN], 0);
    let report = check_declared_files(dir.path(), &cfg).unwrap_err();
    assert_eq!(report.missing, [NORMS_BIN]);
}

#[test]
fn only_the_gate_file_has_a_declared_extent() {
    let cfg = config(ExtractLevel::Browse, &[], 2);
    assert_eq!(declared_min_size(&cfg, GATE_VECTORS_BIN), Some(2 * SLICE));
    assert_eq!(declared_min_size(&cfg, EMBEDDINGS_BIN), None);
    assert_eq!(
        declared_min_size(&config(ExtractLevel::Browse, &[], 0), GATE_VECTORS_BIN),
        None
    );
    let mut overflowing = config(ExtractLevel::Browse, &[], 1);
    overflowing.layers[0].offset = u64::MAX;
    assert_eq!(declared_min_size(&overflowing, GATE_VECTORS_BIN), None);
}
