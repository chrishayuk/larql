use super::*;
use std::io::Write;
use tempfile::TempDir;
use zip::write::SimpleFileOptions;
use zip::ZipWriter;

// ── Synthetic .npy / .npz builders ────────────────────────────────────

/// Build a minimal v1.0 .npy header for the given dtype and shape.
fn npy_header(dtype: &str, shape: &[usize]) -> Vec<u8> {
    let shape_str = if shape.len() == 1 {
        format!("({},)", shape[0])
    } else {
        let parts: Vec<String> = shape.iter().map(|d| d.to_string()).collect();
        format!("({})", parts.join(", "))
    };
    let header = format!("{{'descr': '{dtype}', 'fortran_order': False, 'shape': {shape_str}, }}");
    let mut padded = header.into_bytes();
    let total = 10 + padded.len();
    let pad_to = (total + 63) & !63;
    while 10 + padded.len() + 1 < pad_to {
        padded.push(b' ');
    }
    padded.push(b'\n');
    padded
}

fn synth_f32_npy(values: &[f32]) -> Vec<u8> {
    let header = npy_header("<f4", &[values.len()]);
    let mut out = Vec::new();
    out.extend_from_slice(b"\x93NUMPY");
    out.push(1);
    out.push(0);
    out.extend_from_slice(&(header.len() as u16).to_le_bytes());
    out.extend_from_slice(&header);
    for v in values {
        out.extend_from_slice(&v.to_le_bytes());
    }
    out
}

fn synth_f32_3d_npy(values: &[f32], shape: &[usize]) -> Vec<u8> {
    let header = npy_header("<f4", shape);
    let mut out = Vec::new();
    out.extend_from_slice(b"\x93NUMPY");
    out.push(1);
    out.push(0);
    out.extend_from_slice(&(header.len() as u16).to_le_bytes());
    out.extend_from_slice(&header);
    for v in values {
        out.extend_from_slice(&v.to_le_bytes());
    }
    out
}

fn synth_u32_npy(values: &[u32]) -> Vec<u8> {
    let header = npy_header("<u4", &[values.len()]);
    let mut out = Vec::new();
    out.extend_from_slice(b"\x93NUMPY");
    out.push(1);
    out.push(0);
    out.extend_from_slice(&(header.len() as u16).to_le_bytes());
    out.extend_from_slice(&header);
    for v in values {
        out.extend_from_slice(&v.to_le_bytes());
    }
    out
}

/// Build a structured-dtype `.npy` blob with N VecInjectEntry rows.
fn synth_structured_entries_npy(rows: &[VecInjectEntry]) -> Vec<u8> {
    let descr = "[('token_id', '<u4'), ('coefficient', '<f4'), \
                 ('window_id', '<u2'), ('position_in_window', '<u2'), \
                 ('fact_id', '<u2')]";
    let header = format!(
        "{{'descr': {descr}, 'fortran_order': False, 'shape': ({},), }}",
        rows.len()
    );
    let mut padded = header.into_bytes();
    let total = 10 + padded.len();
    let pad_to = (total + 63) & !63;
    while 10 + padded.len() + 1 < pad_to {
        padded.push(b' ');
    }
    padded.push(b'\n');
    let mut out = Vec::new();
    out.extend_from_slice(b"\x93NUMPY");
    out.push(1);
    out.push(0);
    out.extend_from_slice(&(padded.len() as u16).to_le_bytes());
    out.extend_from_slice(&padded);
    for r in rows {
        out.extend_from_slice(&r.token_id.to_le_bytes());
        out.extend_from_slice(&r.coefficient.to_le_bytes());
        out.extend_from_slice(&r.window_id.to_le_bytes());
        out.extend_from_slice(&r.position_in_window.to_le_bytes());
        out.extend_from_slice(&r.fact_id.to_le_bytes());
    }
    out
}

fn synth_npz<I: IntoIterator<Item = (String, Vec<u8>)>>(members: I) -> Vec<u8> {
    let mut buf = Vec::new();
    {
        let cursor = std::io::Cursor::new(&mut buf);
        let mut zw = ZipWriter::new(cursor);
        let opts: SimpleFileOptions =
            SimpleFileOptions::default().compression_method(zip::CompressionMethod::Stored);
        for (name, data) in members {
            zw.start_file(&name, opts).unwrap();
            zw.write_all(&data).unwrap();
        }
        zw.finish().unwrap();
    }
    buf
}

fn write_minimal_store(
    dir: &Path,
    num_windows: usize,
    hidden: usize,
    window_size: usize,
    entries: &[VecInjectEntry],
) {
    // manifest.json
    let manifest = StoreManifest {
        version: 1,
        num_entries: entries.len(),
        num_windows,
        num_tokens: num_windows * window_size,
        entries_per_window: entries.len().checked_div(num_windows).unwrap_or(0),
        crystal_layer: 29,
        window_size,
        arch_config: ArchConfig::default(),
        has_residuals: false,
    };
    std::fs::write(
        dir.join("manifest.json"),
        serde_json::to_vec(&manifest).unwrap(),
    )
    .unwrap();

    // boundaries/window_NNN.npy — single (hidden,) per file
    let bdir = dir.join("boundaries");
    std::fs::create_dir_all(&bdir).unwrap();
    for w in 0..num_windows {
        let vals: Vec<f32> = (0..hidden).map(|i| (w * 100 + i) as f32).collect();
        std::fs::write(
            bdir.join(format!("window_{:03}.npy", w)),
            synth_f32_npy(&vals),
        )
        .unwrap();
    }

    // window_token_lists.npz: members "0.npy", "1.npy", ...
    let token_members: Vec<(String, Vec<u8>)> = (0..num_windows)
        .map(|w| {
            let toks: Vec<u32> = (0..window_size as u32)
                .map(|i| w as u32 * 1000 + i)
                .collect();
            (format!("{}.npy", w), synth_u32_npy(&toks))
        })
        .collect();
    std::fs::write(dir.join("window_token_lists.npz"), synth_npz(token_members)).unwrap();

    // entries.npz: single member "entries.npy"
    let entry_blob = synth_structured_entries_npy(entries);
    std::fs::write(
        dir.join("entries.npz"),
        synth_npz(vec![("entries.npy".to_string(), entry_blob)]),
    )
    .unwrap();
}

// ── ArchConfig defaults ───────────────────────────────────────────────

#[test]
fn default_arch_config_matches_apollo11() {
    let cfg = ArchConfig::default();
    assert_eq!(cfg.retrieval_layer, 29);
    assert_eq!(cfg.query_head, 4);
    assert_eq!(cfg.injection_layer, 30);
    assert_eq!(cfg.inject_coefficient, 10.0);
}

// ── Load — happy path & high-level errors ─────────────────────────────

#[test]
fn load_missing_directory_errors() {
    let r = ApolloStore::load(Path::new("/tmp/apollo-does-not-exist-xyz"));
    assert!(matches!(r.unwrap_err(), StoreLoadError::Io { .. }));
}

#[test]
fn load_minimal_synthetic_store_succeeds() {
    let dir = TempDir::new().unwrap();
    let entries = vec![
        VecInjectEntry {
            token_id: 42,
            coefficient: 1.0,
            window_id: 0,
            position_in_window: 0,
            fact_id: 0,
        },
        VecInjectEntry {
            token_id: 99,
            coefficient: 2.5,
            window_id: 1,
            position_in_window: 3,
            fact_id: 1,
        },
    ];
    write_minimal_store(dir.path(), 2, 4, 5, &entries);
    let store = ApolloStore::load(dir.path()).expect("load");
    assert_eq!(store.boundaries.len(), 2);
    assert_eq!(store.boundaries[0].len(), 4);
    assert_eq!(store.window_tokens.len(), 2);
    assert_eq!(store.window_tokens[0].len(), 5);
    assert_eq!(store.entries.len(), 2);
    assert_eq!(store.entries[0].token_id, 42);
    assert_eq!(store.entries[1].window_id, 1);
    assert_eq!(store.manifest.num_entries, 2);
    assert_eq!(store.manifest.num_windows, 2);
}

#[test]
fn load_with_optional_boundary_residual() {
    let dir = TempDir::new().unwrap();
    write_minimal_store(dir.path(), 1, 4, 3, &[]);
    // Add the optional boundary_residual.npy.
    let res_blob = synth_f32_3d_npy(&[1.0, 2.0, 3.0, 4.0], &[1, 1, 4]);
    std::fs::write(dir.path().join("boundary_residual.npy"), res_blob).unwrap();
    let store = ApolloStore::load(dir.path()).unwrap();
    assert_eq!(store.boundary_residual, Some(vec![1.0, 2.0, 3.0, 4.0]));
}

#[test]
fn load_manifest_mismatch_on_boundary_count() {
    let dir = TempDir::new().unwrap();
    write_minimal_store(dir.path(), 2, 4, 3, &[]);
    // Tamper: rewrite manifest claiming 3 windows but only 2 boundaries on disk.
    let manifest_path = dir.path().join("manifest.json");
    let mut m: StoreManifest =
        serde_json::from_slice(&std::fs::read(&manifest_path).unwrap()).unwrap();
    m.num_windows = 3;
    std::fs::write(&manifest_path, serde_json::to_vec(&m).unwrap()).unwrap();
    // Loader will fail on missing boundaries/window_002.npy → Io
    let err = ApolloStore::load(dir.path()).unwrap_err();
    assert!(matches!(err, StoreLoadError::Io { .. }));
}

#[test]
fn load_manifest_mismatch_on_entry_count() {
    let dir = TempDir::new().unwrap();
    let entries = vec![VecInjectEntry {
        token_id: 1,
        coefficient: 1.0,
        window_id: 0,
        position_in_window: 0,
        fact_id: 0,
    }];
    write_minimal_store(dir.path(), 1, 4, 3, &entries);
    let manifest_path = dir.path().join("manifest.json");
    let mut m: StoreManifest =
        serde_json::from_slice(&std::fs::read(&manifest_path).unwrap()).unwrap();
    m.num_entries = 99; // disagrees with on-disk entries.npy (1)
    std::fs::write(&manifest_path, serde_json::to_vec(&m).unwrap()).unwrap();
    let err = ApolloStore::load(dir.path()).unwrap_err();
    match err {
        StoreLoadError::ManifestMismatch(msg) => assert!(msg.contains("num_entries")),
        other => panic!("expected ManifestMismatch, got {other:?}"),
    }
}

/// Drive the npy error-wrap path in `load_boundaries` (lines 181-184):
/// write a malformed window_000.npy so `npy::read_f32_1d` fails.
#[test]
fn load_boundaries_malformed_npy_returns_npy_error() {
    let dir = TempDir::new().unwrap();
    write_minimal_store(dir.path(), 1, 4, 3, &[]);
    // Overwrite the boundary file with garbage bytes.
    let bdir = dir.path().join("boundaries");
    std::fs::write(bdir.join("window_000.npy"), b"NOT_AN_NPY_BLOB").unwrap();
    let err = ApolloStore::load(dir.path()).unwrap_err();
    assert!(
        matches!(err, StoreLoadError::Npy { .. }),
        "expected Npy error, got {err:?}"
    );
}

/// Drive the `load_boundary_residual` npy error path (lines 193-196):
/// optional file — when present and malformed, the wrap fires; when
/// absent, `.ok()` in the caller swallows it (already covered).
#[test]
fn load_boundary_residual_malformed_npy_returns_none_via_ok() {
    let dir = TempDir::new().unwrap();
    write_minimal_store(dir.path(), 1, 4, 3, &[]);
    // Plant a malformed boundary_residual.npy.
    std::fs::write(dir.path().join("boundary_residual.npy"), b"bogus").unwrap();
    // Loader's `.ok()` on load_boundary_residual swallows the error;
    // the store loads successfully with `boundary_residual = None`.
    let store = ApolloStore::load(dir.path()).expect("store loads despite malformed residual");
    assert!(store.boundary_residual.is_none());
}

/// Drive the npz error-wrap path in `load_window_tokens`
/// (lines 206-209): write a non-zip file as window_token_lists.npz.
#[test]
fn load_window_tokens_malformed_npz_returns_archive_error() {
    let dir = TempDir::new().unwrap();
    write_minimal_store(dir.path(), 1, 4, 3, &[]);
    std::fs::write(dir.path().join("window_token_lists.npz"), b"NOT_A_ZIP").unwrap();
    let err = ApolloStore::load(dir.path()).unwrap_err();
    assert!(
        matches!(err, StoreLoadError::Archive { .. }),
        "expected Archive error on malformed npz, got {err:?}"
    );
}

/// A gapped window archive (member ids 0, 1, 3) must be rejected: dense
/// packing after the numeric sort would place window 3's tokens at
/// index 2, misaligning every id-keyed lookup (boundaries, entries).
/// Before the contiguity check this loaded "successfully".
#[test]
fn load_window_tokens_gapped_archive_errors() {
    let dir = TempDir::new().unwrap();
    write_minimal_store(dir.path(), 3, 4, 3, &[]);
    // Replace the archive with members 0, 1, 3 (id 2 missing).
    let members: Vec<(String, Vec<u8>)> = [0u32, 1, 3]
        .iter()
        .map(|w| {
            (
                format!("{w}.npy"),
                synth_u32_npy(&[*w * 1000, *w * 1000 + 1, *w * 1000 + 2]),
            )
        })
        .collect();
    std::fs::write(
        dir.path().join("window_token_lists.npz"),
        synth_npz(members),
    )
    .unwrap();
    let err = ApolloStore::load(dir.path()).unwrap_err();
    match err {
        StoreLoadError::WindowArchive(msg) => {
            assert!(
                msg.contains("expected id 2") && msg.contains("found 3"),
                "unexpected message: {msg}"
            );
        }
        other => panic!("expected WindowArchive, got {other:?}"),
    }
}

/// A bare "0" alongside "0.npy" reads as the same window id. Which one is meant is
/// not clear, so it is refused rather than guessed at.
#[test]
fn load_window_tokens_bare_numeric_name_is_ambiguous() {
    let dir = TempDir::new().unwrap();
    write_minimal_store(dir.path(), 2, 4, 3, &[]);
    let members = vec![
        ("0.npy".to_string(), synth_u32_npy(&[1, 2, 3])),
        ("0".to_string(), synth_u32_npy(&[4, 5, 6])),
    ];
    std::fs::write(
        dir.path().join("window_token_lists.npz"),
        synth_npz(members),
    )
    .unwrap();
    let err = ApolloStore::load(dir.path()).unwrap_err();
    match err {
        StoreLoadError::Archive {
            source: ArchiveError::AmbiguousMemberName(name),
            ..
        } => assert_eq!(name, "0"),
        other => panic!("expected AmbiguousMemberName, got {other:?}"),
    }
}

/// The manifest's window count must match the archive: a contiguous but
/// short archive (window 1 of 2 missing) previously loaded fine and
/// out-of-range window ids surfaced only as routing misses.
#[test]
fn load_window_tokens_count_mismatch_vs_manifest_errors() {
    let dir = TempDir::new().unwrap();
    write_minimal_store(dir.path(), 2, 4, 3, &[]);
    // Rewrite the archive with only window 0 (contiguous, wrong count).
    let members = vec![("0.npy".to_string(), synth_u32_npy(&[1, 2, 3]))];
    std::fs::write(
        dir.path().join("window_token_lists.npz"),
        synth_npz(members),
    )
    .unwrap();
    let err = ApolloStore::load(dir.path()).unwrap_err();
    match err {
        StoreLoadError::ManifestMismatch(msg) => {
            assert!(
                msg.contains("num_windows=2") && msg.contains("1 window token lists"),
                "unexpected message: {msg}"
            );
        }
        other => panic!("expected ManifestMismatch, got {other:?}"),
    }
}

#[test]
fn load_invalid_manifest_json_returns_json_error() {
    let dir = TempDir::new().unwrap();
    std::fs::write(dir.path().join("manifest.json"), b"not a json").unwrap();
    let err = ApolloStore::load(dir.path()).unwrap_err();
    assert!(matches!(err, StoreLoadError::Json(_)));
}

// ── In-memory accessors ───────────────────────────────────────────────

#[test]
fn total_bytes_sums_components() {
    let store = ApolloStore {
        manifest: dummy_manifest(2),
        boundaries: vec![vec![0.0; 4], vec![0.0; 4]],
        boundary_residual: Some(vec![0.0; 4]),
        window_tokens: vec![vec![0u32; 3], vec![0u32; 3]],
        entries: vec![
            VecInjectEntry {
                token_id: 0,
                coefficient: 0.0,
                window_id: 0,
                position_in_window: 0,
                fact_id: 0,
            };
            5
        ],
    };
    let entry_size = std::mem::size_of::<VecInjectEntry>();
    let expected = 4 * 4 * 2 + 4 * 4 + 3 * 4 * 2 + 5 * entry_size;
    assert_eq!(store.total_bytes(), expected);
}

#[test]
fn hidden_size_is_first_boundary_len() {
    let store = ApolloStore {
        manifest: dummy_manifest(1),
        boundaries: vec![vec![0.0; 16]],
        boundary_residual: None,
        window_tokens: vec![vec![]],
        entries: vec![],
    };
    assert_eq!(store.hidden_size(), 16);

    let empty = ApolloStore {
        manifest: dummy_manifest(0),
        boundaries: vec![],
        boundary_residual: None,
        window_tokens: vec![],
        entries: vec![],
    };
    assert_eq!(empty.hidden_size(), 0);
}

fn dummy_manifest(num_windows: usize) -> StoreManifest {
    StoreManifest {
        version: 1,
        num_entries: 0,
        num_windows,
        num_tokens: 0,
        entries_per_window: 0,
        crystal_layer: 29,
        window_size: 0,
        arch_config: ArchConfig::default(),
        has_residuals: false,
    }
}

// ── parse_structured_entries_npy direct tests ─────────────────────────

#[test]
fn structured_npy_zero_rows() {
    let blob = synth_structured_entries_npy(&[]);
    let parsed = parse_structured_entries_npy(&blob).unwrap();
    assert!(parsed.is_empty());
}

#[test]
fn structured_npy_three_rows_roundtrip() {
    let rows = vec![
        VecInjectEntry {
            token_id: 1,
            coefficient: 0.5,
            window_id: 10,
            position_in_window: 7,
            fact_id: 3,
        },
        VecInjectEntry {
            token_id: 2,
            coefficient: -1.5,
            window_id: 11,
            position_in_window: 0,
            fact_id: 4,
        },
        VecInjectEntry {
            token_id: u32::MAX,
            coefficient: 12345.0,
            window_id: u16::MAX,
            position_in_window: u16::MAX,
            fact_id: u16::MAX,
        },
    ];
    let blob = synth_structured_entries_npy(&rows);
    let parsed = parse_structured_entries_npy(&blob).unwrap();
    assert_eq!(parsed.len(), 3);
    assert_eq!(parsed[0].token_id, 1);
    assert_eq!(parsed[0].coefficient, 0.5);
    assert_eq!(parsed[1].coefficient, -1.5);
    assert_eq!(parsed[2].token_id, u32::MAX);
    assert_eq!(parsed[2].fact_id, u16::MAX);
}

#[test]
fn structured_npy_missing_field_errors() {
    // Build a structured npy whose descr is missing 'fact_id'.
    let descr = "[('token_id', '<u4'), ('coefficient', '<f4'), \
                 ('window_id', '<u2'), ('position_in_window', '<u2')]";
    let header = format!("{{'descr': {descr}, 'fortran_order': False, 'shape': (1,), }}");
    let mut padded = header.into_bytes();
    let total = 10 + padded.len();
    let pad_to = (total + 63) & !63;
    while 10 + padded.len() + 1 < pad_to {
        padded.push(b' ');
    }
    padded.push(b'\n');
    let mut blob = Vec::new();
    blob.extend_from_slice(b"\x93NUMPY");
    blob.push(1);
    blob.push(0);
    blob.extend_from_slice(&(padded.len() as u16).to_le_bytes());
    blob.extend_from_slice(&padded);
    // 12 bytes: u32 + f32 + u16 + u16
    blob.extend_from_slice(&[0u8; 12]);
    let err = parse_structured_entries_npy(&blob).unwrap_err();
    assert!(err.to_string().contains("fact_id"));
}

/// Build a structured .npy blob with an arbitrary descr string and raw
/// row bytes — for exercising the exact-descr validation.
fn synth_structured_with_descr(descr: &str, rows: usize, row_bytes: &[u8]) -> Vec<u8> {
    let header = format!("{{'descr': {descr}, 'fortran_order': False, 'shape': ({rows},), }}");
    let mut padded = header.into_bytes();
    let total = 10 + padded.len();
    let pad_to = (total + 63) & !63;
    while 10 + padded.len() + 1 < pad_to {
        padded.push(b' ');
    }
    padded.push(b'\n');
    let mut blob = Vec::new();
    blob.extend_from_slice(b"\x93NUMPY");
    blob.push(1);
    blob.push(0);
    blob.extend_from_slice(&(padded.len() as u16).to_le_bytes());
    blob.extend_from_slice(&padded);
    blob.extend_from_slice(row_bytes);
    blob
}

/// A reordered descr passes a substring-presence check but decodes into
/// garbage at the decoder's fixed offsets — it must be rejected.
/// (Fail-before: the old check only tested `descr.contains(field)`.)
#[test]
fn structured_npy_reordered_descr_errors() {
    // coefficient and token_id swapped; binary layout matches the
    // reordered descr (f32 then u32), still 14 bytes/row.
    let descr = "[('coefficient', '<f4'), ('token_id', '<u4'), \
                 ('window_id', '<u2'), ('position_in_window', '<u2'), \
                 ('fact_id', '<u2')]";
    let mut row = Vec::new();
    row.extend_from_slice(&1.5f32.to_le_bytes());
    row.extend_from_slice(&7u32.to_le_bytes());
    row.extend_from_slice(&0u16.to_le_bytes());
    row.extend_from_slice(&0u16.to_le_bytes());
    row.extend_from_slice(&0u16.to_le_bytes());
    let blob = synth_structured_with_descr(descr, 1, &row);
    let err = parse_structured_entries_npy(&blob).unwrap_err();
    assert!(
        err.to_string().contains("structured descr mismatch"),
        "unexpected error: {err}"
    );
    assert!(
        err.to_string()
            .contains(r#"("coefficient", "<f4"), ("token_id", "<u4")"#),
        "error should report the offending field order, got: {err}"
    );
}

/// Same fields and order but a retyped dtype (i4 vs u4) also decodes
/// wrongly and must be rejected.
#[test]
fn structured_npy_retyped_field_errors() {
    let descr = "[('token_id', '<i4'), ('coefficient', '<f4'), \
                 ('window_id', '<u2'), ('position_in_window', '<u2'), \
                 ('fact_id', '<u2')]";
    let blob = synth_structured_with_descr(descr, 1, &[0u8; 14]);
    let err = parse_structured_entries_npy(&blob).unwrap_err();
    assert!(
        err.to_string().contains("structured descr mismatch") && err.to_string().contains("<i4"),
        "unexpected error: {err}"
    );
}

#[test]
fn structured_npy_2d_shape_errors() {
    let descr = "[('token_id', '<u4'), ('coefficient', '<f4'), \
                 ('window_id', '<u2'), ('position_in_window', '<u2'), \
                 ('fact_id', '<u2')]";
    let header = format!("{{'descr': {descr}, 'fortran_order': False, 'shape': (1, 1), }}");
    let mut padded = header.into_bytes();
    let total = 10 + padded.len();
    let pad_to = (total + 63) & !63;
    while 10 + padded.len() + 1 < pad_to {
        padded.push(b' ');
    }
    padded.push(b'\n');
    let mut blob = Vec::new();
    blob.extend_from_slice(b"\x93NUMPY");
    blob.push(1);
    blob.push(0);
    blob.extend_from_slice(&(padded.len() as u16).to_le_bytes());
    blob.extend_from_slice(&padded);
    blob.extend_from_slice(&[0u8; 14]);
    let err = parse_structured_entries_npy(&blob).unwrap_err();
    assert!(err.to_string().contains("expected 1D"));
}

#[test]
fn structured_npy_data_size_mismatch_errors() {
    // Build a blob declaring 2 rows but provide only 1 row of data.
    let descr = "[('token_id', '<u4'), ('coefficient', '<f4'), \
                 ('window_id', '<u2'), ('position_in_window', '<u2'), \
                 ('fact_id', '<u2')]";
    let header = format!("{{'descr': {descr}, 'fortran_order': False, 'shape': (2,), }}");
    let mut padded = header.into_bytes();
    let total = 10 + padded.len();
    let pad_to = (total + 63) & !63;
    while 10 + padded.len() + 1 < pad_to {
        padded.push(b' ');
    }
    padded.push(b'\n');
    let mut blob = Vec::new();
    blob.extend_from_slice(b"\x93NUMPY");
    blob.push(1);
    blob.push(0);
    blob.extend_from_slice(&(padded.len() as u16).to_le_bytes());
    blob.extend_from_slice(&padded);
    blob.extend_from_slice(&[0u8; 14]); // only 1 row × 14 bytes
    let err = parse_structured_entries_npy(&blob).unwrap_err();
    assert!(err.to_string().contains("data size"));
}

#[test]
fn structured_npy_bad_magic_errors() {
    let blob = b"not a valid npy".to_vec();
    let err = parse_structured_entries_npy(&blob).unwrap_err();
    assert!(!err.to_string().is_empty());
}

mod archive_errors;
mod file_reading;
#[cfg(target_pointer_width = "64")]
mod raw_archive;
// `shape` and `num_windows` values past 2^32 do not parse on a 32-bit target,
// so these cases would be refused earlier there, with a different variant.
#[cfg(target_pointer_width = "64")]
mod hostile_archive_sizes;
#[cfg(target_pointer_width = "64")]
mod hostile_manifest;
#[cfg(target_pointer_width = "64")]
mod hostile_npy_shapes;
#[cfg(target_pointer_width = "64")]
mod npz_format;
